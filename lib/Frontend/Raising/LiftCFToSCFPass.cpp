// Recover structured control flow inside every callable region in place.
//
// An imported `llvm.func` is the sole callable and ABI owner of its LLVM
// function, so CFG recovery runs on that callable's own region instead of on
// a copy in another dialect. `loom-llvm-cf-to-cf` first converts the supported
// LLVM branch terminators to their exact `cf` counterparts; this pass then
// invokes the upstream region-level `mlir::transformCFGToSCF` utility.
//
// The adapter below reuses the upstream `cf`-to-`scf` implementation and only
// respells what an LLVM callable spells differently:
//
//   * return -- `llvm.return` is return-like, so the upstream exit combiner
//     already merges the callable's returns without a Loom-specific rule.
//   * undef -- structuring creates paths on which a value is never defined.
//     Upstream materializes `ub.poison`, which is deferred undefined behavior
//     and therefore stronger than what LLVM states here; an imported callable
//     gets `llvm.mlir.undef` of the same type instead.
//   * unreachable -- upstream builds a `func.return` of poison results, which
//     an LLVM callable cannot host. `llvm.unreachable` states the same fact
//     without inventing a result value.
//
// Imported loop annotations are hints Part 1 requires to stay associated with
// their loop. A well-formed annotation arrives on a branch that closes a cycle;
// this pass resolves it to the loop header it repeats to and hands it to the
// structured loop that ends up owning that cycle. An annotation on a branch
// with no backedge is orphan metadata under LLVM's loop contract. It is removed
// only from a detached clone that successfully structures, and is never guessed
// onto an unrelated loop. A cycle entered on more than one edge is
// headed by an entry multiplexer upstream creates, and every edge to its entry
// blocks is redirected through that multiplexer, so the cycle's annotated
// original header is the dispatch destination that has the multiplexer as its
// unique predecessor. The carry looks only for that destination: a nested
// cycle's annotation still has its own back edge as another predecessor, so it
// stays in the map for the nested loop's own recovery. More than one such
// destination is an association the pass cannot prove, and an annotation that
// reaches no recovered loop at all would be silently lost; in both cases the
// region keeps its original control rather than an approximated association.
//
// Structuring is a per-region decision between exactly two mechanical
// dispositions, and neither of them fails the module. A region is structured
// when every Loom-owned question about it has an exact answer; otherwise it is
// preserved with its complete original semantics. Unstructured `cf` control is
// ordinary legal S0, so preservation costs nothing an unselected or
// InstructionCore-owned region needs. Rejection belongs to the boundary that
// actually narrows the surface: a candidate that selects a
// `loom.spatial_region` must present structured control there, and cannot do so
// from a preserved CFG.
//
// A region is preserved when:
//
//   * a reachable branch carries weights -- `scf.if` and `scf.index_switch`
//     state no branch probability, so lifting would drop imported profile data;
//   * a reachable terminator with successors is not exactly cf.br, cf.cond_br,
//     or cf.switch -- the transformation erases a one-successor terminator it
//     does not recognize and splices its successor away, which would silently
//     restate a one-target llvm.indirectbr as an unconditional branch;
//   * a `cf.switch` selector or case value does not fit the structured
//     switch's index and 64-bit case carriers;
//   * an imported callable holds a value whose type LLVM cannot spell, since
//     the adapter would otherwise have to state an undefined value of that type
//     as the stronger `ub.poison`; or
//   * a loop annotation's owning loop is not exactly identifiable.
//
// Deciding this over all callable regions before any of them is touched also
// avoids relying on the transformation's own unknown-terminator check, which
// runs per region, after earlier regions may already be structured.
//
// The structuring traversal then works on detached clones of the callable ops:
// it erases the clone region's own unreachable blocks, which is the one cleanup
// upstream's documented structural preconditions require of this surface, and
// structures the clone region without descending into nested regions. Each
// clone is taken from the then-current original and published back into its
// original callable op the moment it is complete. The callable walk is
// post-order, so a nested callable is structured and published before its
// enclosing callable: the ancestor is therefore cloned from an original that
// already holds the structured descendant, and its clone carries that structure
// through. A deferred publication would instead clone the ancestor from a
// snapshot taken before the descendant published, so publishing that stale
// ancestor clone would overwrite the descendant's structured body with the
// unstructured copy the clone still held. Upstream's documented "unspecified IR
// on interface failure" unwinds inside the clone, and an annotation the
// completed clone could not place leaves that region preserved, so a clone that
// declines is dropped without publishing. Publishing cannot fail: it preserves
// the region's owning callable op and carries already-structured descendant
// bodies through ancestor clones, leaving each imported callable in llvm.func
// form as the sole ABI envelope of its body.

#include "Frontend/Raising/Passes.h"

#include "CallableRegions.h"
#include "PreservedHints.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/CFGToSCF.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <iterator>
#include <memory>
#include <optional>

namespace {

// Loop header block of each annotated cycle mapped to the imported annotation
// that describes it. Keys are blocks of the original IR; the structuring
// traversal remaps them to each clone's own blocks before the adapter runs.
using LoopAnnotations = ::llvm::DenseMap<::mlir::Block *, ::mlir::Attribute>;
using OrphanLoopHintBlocks = ::llvm::SmallVector<::mlir::Block *, 2>;

// Mechanical disposition of one callable region.
enum class Disposition {
  // Recover structured control from the region's cf branch structure.
  Structure,
  // Retain the region and its complete semantics exactly as they stand.
  Preserve,
};

// Upstream leaves a region it will not restructure untouched, so it needs no
// acceptance decision and its hints stay where they are.
bool isStructured(::mlir::Region &region) { return region.hasOneBlock(); }

// True when `terminator` states an imported branch probability. No structured
// operation owns one, so the region that holds it is preserved rather than
// lifted. LLVM's own weighted terminators are read too: `loom-llvm-cf-to-cf`
// leaves a weighted switch in LLVM form for exactly this reason.
bool statesBranchWeights(::mlir::Operation *terminator) {
  if (auto condBranch = ::mlir::dyn_cast<::mlir::cf::CondBranchOp>(terminator))
    return condBranch.getBranchWeightsAttr() != nullptr;
  if (auto condBranch = ::mlir::dyn_cast<::mlir::LLVM::CondBrOp>(terminator))
    return condBranch.getBranchWeightsAttr() != nullptr;
  if (auto switchOp = ::mlir::dyn_cast<::mlir::LLVM::SwitchOp>(terminator))
    return switchOp.getBranchWeightsAttr() != nullptr;
  return false;
}

// Upstream structures a `cf.switch` into `scf.index_switch`: it converts the
// selector with `arith.index_castui` to `index` and reads each case through
// 64-bit storage. A selector wider than the target's `index`, or a case value
// wider than that storage, would silently drop the high bits and send some
// runtime values to the wrong destination.
bool structuredSwitchCarrierHolds(::mlir::cf::SwitchOp op,
                                  uint64_t indexBitwidth) {
  if (op.getFlag().getType().getIntOrFloatBitWidth() > indexBitwidth)
    return false;
  if (::std::optional<::mlir::DenseIntElementsAttr> cases = op.getCaseValues())
    for (const ::llvm::APInt &value : *cases)
      if (!value.isIntN(64))
        return false;
  return true;
}

// Every value the structuring may have to replace with an undefined placeholder
// is a block argument or an operation result of the region, so an imported
// callable can be structured only when LLVM can spell all of them.
bool holdsOnlyLLVMSpellableValues(::mlir::Block &block) {
  for (::mlir::BlockArgument argument : block.getArguments())
    if (!::mlir::LLVM::isCompatibleType(argument.getType()))
      return false;
  for (::mlir::Operation &op : block)
    for (::mlir::Value result : op.getResults())
      if (!::mlir::LLVM::isCompatibleType(result.getType()))
        return false;
  return true;
}

// Decide the disposition of `region`, resolving every loop annotation to the
// single loop header it repeats to. Modifies nothing, and hands `annotations`
// the resolved headers only when the region will actually be structured.
Disposition decideRegion(::mlir::Region &region, LoopAnnotations &annotations,
                         OrphanLoopHintBlocks &orphanHints) {
  if (isStructured(region))
    return Disposition::Preserve;

  bool importedLLVMCallable =
      ::mlir::isa<::mlir::LLVM::LLVMFuncOp>(region.getParentOp());
  ::mlir::DominanceInfo dominance(region.getParentOp());
  uint64_t indexBitwidth =
      ::mlir::DataLayout::closest(region.getParentOp())
          .getTypeSizeInBits(::mlir::IndexType::get(region.getContext()))
          .getFixedValue();

  LoopAnnotations resolved;
  for (::mlir::Block &block : region) {
    // An unreachable block states no executed control decision and the
    // structuring traversal erases the clone's, so it carries no hint that
    // structuring could lose and no value it could leave undefined.
    if (!dominance.isReachableFromEntry(&block))
      continue;

    if (importedLLVMCallable && !holdsOnlyLLVMSpellableValues(block))
      return Disposition::Preserve;

    ::mlir::Operation *terminator = block.getTerminator();

    if (statesBranchWeights(terminator))
      return Disposition::Preserve;

    if (terminator->getNumSuccessors() != 0 &&
        !::mlir::isa<::mlir::cf::BranchOp, ::mlir::cf::CondBranchOp,
                     ::mlir::cf::SwitchOp>(terminator))
      return Disposition::Preserve;

    if (auto switchOp = ::mlir::dyn_cast<::mlir::cf::SwitchOp>(terminator))
      if (!structuredSwitchCarrierHolds(switchOp, indexBitwidth))
        return Disposition::Preserve;

    ::mlir::Attribute annotation =
        terminator->getAttr(loom::raising::loopAnnotationName);
    if (!annotation)
      continue;

    // The annotated edge repeats a loop, so its successor dominates it. More
    // than one such successor leaves the described loop ambiguous.
    ::mlir::Block *header = nullptr;
    bool ambiguous = false;
    for (::mlir::Block *successor : block.getSuccessors()) {
      if (!dominance.dominates(successor, &block) || successor == header)
        continue;
      if (header) {
        ambiguous = true;
        break;
      }
      header = successor;
    }
    if (ambiguous)
      return Disposition::Preserve;
    if (!header) {
      orphanHints.push_back(&block);
      continue;
    }

    auto [entry, inserted] = resolved.try_emplace(header, annotation);
    if (!inserted && entry->second != annotation)
      return Disposition::Preserve;
  }

  annotations.insert(resolved.begin(), resolved.end());
  return Disposition::Structure;
}

class CallableStructuring : public ::mlir::ControlFlowToSCFTransformation {
public:
  CallableStructuring(bool importedLLVMCallable, LoopAnnotations &annotations)
      : importedLLVMCallable(importedLLVMCallable), annotations(annotations) {}

  // Structuring can leave a value undefined on a created path. An imported
  // callable states that with LLVM's own undef, which is exactly as weak as
  // the value LLVM had there; `ub.poison` would deepen it into deferred
  // undefined behavior. Acceptance proved every value type of an imported
  // callable region is LLVM-spellable, so this never falls back there. A
  // genuinely native callable keeps the upstream spelling, which is its own
  // semantics.
  ::mlir::Value getUndefValue(::mlir::Location loc, ::mlir::OpBuilder &builder,
                              ::mlir::Type type) override {
    if (importedLLVMCallable) {
      assert(::mlir::LLVM::isCompatibleType(type) &&
             "accepted imported callable holds only LLVM-spellable values");
      return ::mlir::LLVM::UndefOp::create(builder, loc, type);
    }
    return ControlFlowToSCFTransformation::getUndefValue(loc, builder, type);
  }

  // A loop without an exit edge never reaches its continuation. Inside an
  // imported LLVM callable that continuation is spelled `llvm.unreachable`,
  // which needs no result value and therefore cannot weaken a signature that
  // returns one.
  ::mlir::FailureOr<::mlir::Operation *>
  createUnreachableTerminator(::mlir::Location loc, ::mlir::OpBuilder &builder,
                              ::mlir::Region &region) override {
    if (importedLLVMCallable)
      return ::mlir::LLVM::UnreachableOp::create(builder, loc).getOperation();
    return ControlFlowToSCFTransformation::createUnreachableTerminator(
        loc, builder, region);
  }

  // The created loop owns the cycle entered at the loop body's entry block,
  // so it is the operation that inherits that cycle's imported annotation.
  // When the cycle was entered on more than one edge, the body entry is the
  // entry multiplexer upstream created, and every edge to the cycle's entry
  // blocks was redirected through it, so the cycle's annotated original
  // header is a dispatch destination of the multiplexer that has the
  // multiplexer as its unique predecessor. An annotation nested deeper in the
  // body still has its own back edge as a second predecessor, so it stays in
  // the map for the nested loop's own recovery. A carried annotation is
  // consumed so it can never attach to a second loop.
  //
  // More than one such destination is an association this pass cannot prove.
  // Upstream does not unwind a failed do-while creation (its blocks are
  // already removed from their region), so this entry point never fails: it
  // records the first unprovable association and carries nothing, and the
  // completed clone is then dropped in favour of the preserved original.
  ::mlir::FailureOr<::mlir::Operation *> createStructuredDoWhileLoopOp(
      ::mlir::OpBuilder &builder, ::mlir::Operation *replacedOp,
      ::mlir::ValueRange loopValuesInit, ::mlir::Value condition,
      ::mlir::ValueRange loopValuesNextIter,
      ::mlir::Region &&loopBody) override {
    ::mlir::Attribute annotation;
    auto frontEntry = annotations.find(&loopBody.front());
    if (frontEntry != annotations.end()) {
      annotation = frontEntry->second;
      annotations.erase(frontEntry);
    } else {
      auto candidate = annotations.end();
      bool ambiguous = false;
      ::mlir::Block *front = &loopBody.front();
      for (::mlir::Block *dispatch : front->getSuccessors()) {
        if (dispatch->getUniquePredecessor() != front)
          continue;
        auto entry = annotations.find(dispatch);
        if (entry == annotations.end())
          continue;
        if (candidate != annotations.end() && candidate != entry) {
          ambiguous = true;
          break;
        }
        candidate = entry;
      }
      if (ambiguous) {
        if (!firstUnprovenAssociation)
          firstUnprovenAssociation = replacedOp->getLoc();
      } else if (candidate != annotations.end()) {
        annotation = candidate->second;
        annotations.erase(candidate);
      }
    }
    ::mlir::FailureOr<::mlir::Operation *> loop =
        ControlFlowToSCFTransformation::createStructuredDoWhileLoopOp(
            builder, replacedOp, loopValuesInit, condition, loopValuesNextIter,
            std::move(loopBody));
    if (failed(loop))
      return ::mlir::failure();
    loom::raising::carryLoopAnnotation(annotation, *loop);
    return loop;
  }

  // Location of the first loop annotation whose owning loop could not be
  // proven, if any was met.
  const ::std::optional<::mlir::Location> &firstUnprovenAssociationLoc() const {
    return firstUnprovenAssociation;
  }

private:
  bool importedLLVMCallable;
  LoopAnnotations &annotations;
  ::std::optional<::mlir::Location> firstUnprovenAssociation;
};

// The pinned cf.switch printer accepts a numbered multi-result value in case
// successor operands but its parser rejects one in the default operands. Such
// values appear only after CFG-to-SCF replaces an entry multiplexer argument,
// so repair the completed clone rather than perturbing the transformation.
void makeResidualSwitchesRoundTripSafe(::mlir::Region &region,
                                       ::mlir::IRRewriter &rewriter) {
  ::llvm::SmallVector<::mlir::cf::SwitchOp, 2> switches;
  region.walk([&](::mlir::cf::SwitchOp op) {
    if (::llvm::any_of(op.getDefaultOperands(), [](::mlir::Value value) {
          auto result = ::mlir::dyn_cast<::mlir::OpResult>(value);
          return result && result.getOwner()->getNumResults() > 1;
        }))
      switches.push_back(op);
  });

  for (::mlir::cf::SwitchOp op : switches) {
    ::mlir::SuccessorRange cases = op.getCaseDestinations();
    rewriter.setInsertionPoint(op);
    if (cases.empty()) {
      rewriter.replaceOpWithNewOp<::mlir::cf::BranchOp>(
          op, op.getDefaultDestination(), op.getDefaultOperands());
      continue;
    }
    if (cases.size() == 1) {
      const ::llvm::APInt &caseValue =
          *op.getCaseValuesAttr().getValues<::llvm::APInt>().begin();
      ::mlir::Value constant = ::mlir::arith::ConstantOp::create(
          rewriter, op.getLoc(),
          ::mlir::IntegerAttr::get(op.getFlag().getType(), caseValue));
      ::mlir::Value selected = ::mlir::arith::CmpIOp::create(
          rewriter, op.getLoc(), ::mlir::arith::CmpIPredicate::eq, op.getFlag(),
          constant);
      ::mlir::cf::CondBranchOp::create(
          rewriter, op.getLoc(), selected, cases.front(), op.getCaseOperands(0),
          op.getDefaultDestination(), op.getDefaultOperands());
      rewriter.eraseOp(op);
      continue;
    }

    ::mlir::Block *dispatchBlock = op->getBlock();
    ::mlir::Block *defaultTrampoline = rewriter.createBlock(
        dispatchBlock->getParent(), ::std::next(dispatchBlock->getIterator()));
    ::mlir::cf::BranchOp::create(rewriter, op.getLoc(),
                                 op.getDefaultDestination(),
                                 op.getDefaultOperands());
    rewriter.setInsertionPoint(op);
    ::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>, 4>
        caseArgumentStorage;
    for (::mlir::OperandRange operands : op.getCaseOperands())
      caseArgumentStorage.emplace_back(operands.begin(), operands.end());
    ::llvm::SmallVector<::mlir::ValueRange, 4> caseArguments;
    for (const auto &operands : caseArgumentStorage)
      caseArguments.emplace_back(operands);
    ::mlir::cf::SwitchOp::create(rewriter, op.getLoc(), op.getFlag(),
                                 defaultTrampoline, ::mlir::ValueRange{},
                                 op.getCaseValuesAttr(), cases, caseArguments);
    rewriter.eraseOp(op);
  }
}

struct LiftCFToSCFPass
    : public ::mlir::PassWrapper<LiftCFToSCFPass, ::mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftCFToSCFPass)

  ::llvm::StringRef getArgument() const final { return "loom-lift-cf-to-scf"; }
  ::llvm::StringRef getDescription() const final {
    return "Structure the cf-shaped control flow of each callable region with "
           "the upstream CFG-to-SCF transformation, leaving an imported "
           "llvm.func as the callable and ABI owner of its body and moving "
           "each imported loop annotation to the loop that owns its cycle. A "
           "region that cannot be structured exactly is preserved as it "
           "stands.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::cf::ControlFlowDialect,
                    ::mlir::LLVM::LLVMDialect, ::mlir::scf::SCFDialect,
                    ::mlir::ub::UBDialect>();
  }

  void runOnOperation() final {
    // One plan per callable region chosen for structuring, holding only that
    // region's own resolved loop annotations. Annotation headers are blocks of
    // the region the plan owns, so a plan is consulted only while that region's
    // original blocks are still live and never after the region is published.
    struct RegionPlan {
      ::mlir::Region *region;
      LoopAnnotations annotations;
      OrphanLoopHintBlocks orphanHints;
    };
    // The callable walk is post-order, so a nested callable is planned before
    // its enclosing callable. Structuring in that order is what makes nested
    // publication correct: an enclosing callable is cloned from the
    // then-current original, which already holds the structured descendant, so
    // publishing the ancestor cannot overwrite the descendant's structure.
    ::llvm::SmallVector<RegionPlan, 4> plans;
    (void)loom::raising::forEachCallableRegion(
        getOperation(), [&](::mlir::Region &region) {
          LoopAnnotations regionAnnotations;
          OrphanLoopHintBlocks orphanHints;
          if (decideRegion(region, regionAnnotations, orphanHints) ==
              Disposition::Structure)
            plans.push_back({&region, std::move(regionAnnotations),
                             std::move(orphanHints)});
          return ::mlir::success();
        });

    ::mlir::IRRewriter rewriter(&getContext());
    bool structuredAny = false;
    for (RegionPlan &plan : plans) {
      ::mlir::Region *region = plan.region;
      ::mlir::Operation *originalCallable = region->getParentOp();
      ::mlir::IRMapping mapping;
      ::mlir::OwningOpRef<::mlir::Operation *> clone(
          originalCallable->clone(mapping));
      ::mlir::Region &cloneRegion = clone->getRegion(region->getRegionNumber());

      // The adapter resolves this region's annotations on the clone's own
      // blocks. Only this plan's headers are read, so a descendant already
      // published in an earlier iteration -- whose original blocks no longer
      // exist -- is never reached here.
      LoopAnnotations cloneAnnotations;
      for (auto &[header, annotation] : plan.annotations) {
        ::mlir::Block *cloneHeader = mapping.lookupOrNull(header);
        assert(cloneHeader && "cloned region holds every original block");
        cloneAnnotations[cloneHeader] = annotation;
      }
      for (::mlir::Block *orphanBlock : plan.orphanHints) {
        ::mlir::Block *cloneBlock = mapping.lookupOrNull(orphanBlock);
        assert(cloneBlock && "cloned region holds every original block");
        cloneBlock->getTerminator()->removeAttr(
            loom::raising::loopAnnotationName);
      }

      // Upstream rejects a region holding a block no edge reaches. Such a
      // block runs never, so erasing it is the one cleanup this surface
      // needs and it decides nothing. Nested regions are left alone.
      (void)::mlir::eraseUnreachableBlocks(rewriter, cloneRegion,
                                           /*recurse=*/false);

      // The transformation moves, creates and erases blocks, so the
      // dominance information it invalidates in place belongs to exactly
      // the region being structured.
      ::mlir::DominanceInfo dominance(clone.get());
      CallableStructuring structuring(
          ::mlir::isa<::mlir::LLVM::LLVMFuncOp>(clone.get()), cloneAnnotations);
      if (failed(
              ::mlir::transformCFGToSCF(cloneRegion, structuring, dominance)))
        continue;

      makeResidualSwitchesRoundTripSafe(cloneRegion, rewriter);

      // Two facts only the completed clone can state: an entry multiplexer
      // that dispatches to more than one annotated loop header leaves the
      // owning loop of each annotation unprovable, and a leftover entry
      // means an annotation reached no recovered loop. Either way the
      // clone is dropped and the original region keeps its annotation
      // exactly where it was imported.
      if (structuring.firstUnprovenAssociationLoc() ||
          !cloneAnnotations.empty())
        continue;

      // Publish immediately. The clone was taken from the current original, so
      // a descendant already published in an earlier iteration is captured
      // structured. Replacing an ancestor body may replace that descendant op
      // with its clone, but cannot restore the stale unstructured body.
      // takeBody preserves this region's owning callable op and leaves every
      // imported callable in llvm.func form as its body's sole ABI envelope.
      region->takeBody(cloneRegion);
      structuredAny = true;
    }

    if (!structuredAny)
      markAllAnalysesPreserved();
  }
};

} // namespace

namespace loom {
namespace raising {

std::unique_ptr<::mlir::Pass> createLiftCFToSCFPass() {
  return std::make_unique<LiftCFToSCFPass>();
}

void registerLiftCFToSCFPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LiftCFToSCFPass>();
    return true;
  }();
  (void)once;
}

} // namespace raising
} // namespace loom

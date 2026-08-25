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
// Imported loop annotations must stay associated with their loop. A
// well-formed annotation arrives on a branch that closes a cycle;
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
// A completely admissible callable is structured as one region. If an exact
// local obstacle prevents that, maximal single-entry, single-continuation CFG
// regions around the obstacle are considered independently. Each local region
// is moved into a temporary scf.execute_region in a detached callable clone,
// transformed with the same upstream utility, and immediately inlined. The
// temporary operation is only an implementation boundary: it is never
// published. An unprovable local region stays as `cf`, while independent local
// regions in the same callable can still be recovered.
//
// A local region is excluded when:
//
//   * a reachable branch carries weights -- `scf.if` and `scf.index_switch`
//     state no branch probability, so lifting would drop imported profile data;
//   * a reachable terminator with successors is not exactly cf.br, cf.cond_br,
//     or cf.switch -- the transformation erases a one-successor terminator it
//     does not recognize and splices its successor away, which would silently
//     restate a one-target llvm.indirectbr as an unconditional branch;
//   * a `cf.switch` selector or case value does not fit the structured
//     switch's index and 64-bit case carriers;
//   * a block owns llvm.blocktag, whose parent block identity may be observed
//     by a module-level llvm.blockaddress independently of SSA uses;
//   * an imported callable holds a value whose type LLVM cannot spell, since
//     the adapter would otherwise have to state an undefined value of that type
//     as the stronger `ub.poison`; or
//   * a loop annotation's owning loop is not exactly identifiable.
//
// Every structuring traversal works on a detached clone of the callable op.
// Unreachable components with no externally visible block identity are erased
// from that clone, which is the one cleanup upstream's documented structural
// preconditions require. An unreachable component containing llvm.blocktag is
// instead retained because a module-level llvm.blockaddress can observe that
// identity without an SSA or CFG use. The whole-callable path declines such a
// clone. A local region may still structure when the retained component does
// not enter its extraction boundary. Each clone is taken from the then-current
// original and published back into its original callable op only after the
// complete attempted rewrite succeeds. The walk is
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
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/CFGToSCF.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <iterator>
#include <memory>
#include <optional>

namespace {

struct LoopHints final {
  ::mlir::Attribute annotation;
  ::mlir::Attribute candidate;

  friend bool operator==(const LoopHints &lhs, const LoopHints &rhs) {
    return lhs.annotation == rhs.annotation && lhs.candidate == rhs.candidate;
  }
};

// Loop header block of each hinted cycle mapped to the imported hints that
// describe it. Keys are blocks of the original IR; the structuring traversal
// remaps them to each clone's own blocks before the adapter runs.
using LoopHintMap = ::llvm::DenseMap<::mlir::Block *, LoopHints>;
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

bool acceptsBlockForStructuring(::mlir::Block &block, bool importedLLVMCallable,
                                uint64_t indexBitwidth) {
  if (importedLLVMCallable && !holdsOnlyLLVMSpellableValues(block))
    return false;
  if (!block.getOps<::mlir::LLVM::BlockTagOp>().empty())
    return false;

  ::mlir::Operation *terminator = block.getTerminator();
  if (statesBranchWeights(terminator))
    return false;
  if (terminator->getNumSuccessors() != 0 &&
      !::mlir::isa<::mlir::cf::BranchOp, ::mlir::cf::CondBranchOp,
                   ::mlir::cf::SwitchOp>(terminator))
    return false;
  if (auto switchOp = ::mlir::dyn_cast<::mlir::cf::SwitchOp>(terminator))
    if (!structuredSwitchCarrierHolds(switchOp, indexBitwidth))
      return false;
  return true;
}

// Decide the disposition of `region`, resolving every loop annotation to the
// single loop header it repeats to. Modifies nothing, and hands `annotations`
// the resolved headers only when the region will actually be structured.
Disposition decideRegion(::mlir::Region &region, bool importedLLVMCallable,
                         uint64_t indexBitwidth, LoopHintMap &annotations,
                         OrphanLoopHintBlocks &orphanHints) {
  if (isStructured(region))
    return Disposition::Preserve;

  ::mlir::DominanceInfo dominance(region.getParentOp());

  LoopHintMap resolved;
  for (::mlir::Block &block : region) {
    // An unreachable block states no executed control decision and the
    // structuring traversal erases the clone's, so it carries no hint that
    // structuring could lose and no value it could leave undefined.
    if (!dominance.isReachableFromEntry(&block))
      continue;

    if (!acceptsBlockForStructuring(block, importedLLVMCallable, indexBitwidth))
      return Disposition::Preserve;

    ::mlir::Operation *terminator = block.getTerminator();
    ::mlir::Attribute annotation =
        terminator->getAttr(loom::raising::loopAnnotationName);
    ::mlir::Attribute candidate =
        terminator->getAttr(loom::raising::candidateLoopHintName);
    if (!annotation && !candidate)
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

    LoopHints hints{annotation, candidate};
    auto [entry, inserted] = resolved.try_emplace(header, hints);
    if (!inserted && !(entry->second == hints))
      return Disposition::Preserve;
  }

  annotations.insert(resolved.begin(), resolved.end());
  return Disposition::Structure;
}

struct LocalRegionPlan final {
  ::mlir::Block *entry = nullptr;
  ::mlir::Block *continuation = nullptr;
  uint64_t indexBitwidth = 0;
  ::llvm::SmallVector<::mlir::Block *, 8> blocks;
};

using DeclinedLocalRegions =
    ::llvm::SmallVector<::std::pair<::mlir::Block *, ::mlir::Block *>, 4>;

// Remove only unreachable CFG components whose block identity cannot be
// observed outside the callable. llvm.blockaddress references a blocktag by
// symbol and integer tag rather than an SSA use, so preserving the complete
// weakly connected unreachable component is necessary to retain both that
// identity and the component's internal SSA/branch closure.
bool eraseUnanchoredUnreachableBlocks(::mlir::Operation *callable,
                                      ::mlir::Region &region,
                                      ::mlir::IRRewriter &rewriter) {
  ::mlir::DominanceInfo dominance(callable);
  ::llvm::SmallVector<::mlir::Block *, 8> unreachable;
  ::llvm::SmallVector<::mlir::Block *, 4> worklist;
  ::llvm::DenseSet<::mlir::Block *> retained;
  for (::mlir::Block &block : region) {
    if (dominance.isReachableFromEntry(&block))
      continue;
    unreachable.push_back(&block);
    bool hasBlockIdentity = false;
    (void)block.walk([&](::mlir::LLVM::BlockTagOp) {
      hasBlockIdentity = true;
      return ::mlir::WalkResult::interrupt();
    });
    if (hasBlockIdentity && retained.insert(&block).second)
      worklist.push_back(&block);
  }

  while (!worklist.empty()) {
    ::mlir::Block *block = worklist.pop_back_val();
    auto retain = [&](::mlir::Block *neighbor) {
      if (!dominance.isReachableFromEntry(neighbor) &&
          retained.insert(neighbor).second)
        worklist.push_back(neighbor);
    };
    for (::mlir::Block *predecessor : block->getPredecessors())
      retain(predecessor);
    for (::mlir::Block *successor : block->getSuccessors())
      retain(successor);
  }

  ::llvm::SmallVector<::mlir::Block *, 8> erased;
  for (::mlir::Block *block : unreachable)
    if (!retained.contains(block))
      erased.push_back(block);
  for (::mlir::Block *block : erased)
    block->dropAllDefinedValueUses();
  for (::mlir::Block *block : erased)
    rewriter.eraseBlock(block);
  return retained.empty();
}

bool isDeclinedLocalRegion(const DeclinedLocalRegions &declined,
                           ::mlir::Block *entry, ::mlir::Block *continuation) {
  return ::llvm::is_contained(declined, ::std::make_pair(entry, continuation));
}

// A local region retains exactly one externally visible entry block and one
// continuation. Blocks are collected by forward reachability from the entry,
// stopping at the continuation, and then validated in both directions. This
// makes the extraction boundary a fact of the existing CFG rather than a
// guessed structured shape.
::std::optional<LocalRegionPlan> analyzeLocalRegion(
    ::mlir::Region &region, ::mlir::Block *entry, ::mlir::Block *continuation,
    bool importedLLVMCallable, ::mlir::DominanceInfo &dominance,
    ::mlir::PostDominanceInfo &postDominance, uint64_t indexBitwidth) {
  if (entry == continuation ||
      !postDominance.postDominates(continuation, entry))
    return ::std::nullopt;

  ::llvm::DenseSet<::mlir::Block *> members;
  ::llvm::SmallVector<::mlir::Block *, 8> worklist{entry};
  bool reachesContinuation = false;
  while (!worklist.empty()) {
    ::mlir::Block *block = worklist.pop_back_val();
    if (block == continuation) {
      reachesContinuation = true;
      continue;
    }
    if (!members.insert(block).second)
      continue;
    for (::mlir::Block *successor : block->getSuccessors())
      worklist.push_back(successor);
  }
  if (!reachesContinuation)
    return ::std::nullopt;

  bool changesCFGShape = members.size() > 1;
  for (::mlir::Block *block : members) {
    if (!dominance.isReachableFromEntry(block) ||
        !dominance.dominates(entry, block) ||
        !postDominance.postDominates(continuation, block) ||
        !acceptsBlockForStructuring(*block, importedLLVMCallable,
                                    indexBitwidth))
      return ::std::nullopt;

    ::mlir::Operation *terminator = block->getTerminator();
    if (terminator->getNumSuccessors() == 0)
      return ::std::nullopt;
    changesCFGShape |= terminator->getNumSuccessors() > 1;

    for (auto predecessor = block->pred_begin();
         predecessor != block->pred_end(); ++predecessor) {
      if (members.contains(*predecessor))
        continue;
      if (!dominance.isReachableFromEntry(*predecessor))
        continue;
      if (block != entry || dominance.dominates(entry, *predecessor))
        return ::std::nullopt;
    }

    for (auto [successorIndex, successor] :
         ::llvm::enumerate(block->getSuccessors())) {
      if (members.contains(successor))
        continue;
      if (successor != continuation)
        return ::std::nullopt;
      auto branch = ::mlir::dyn_cast<::mlir::BranchOpInterface>(terminator);
      if (!branch || branch.getSuccessorOperands(successorIndex)
                             .getProducedOperandCount() != 0)
        return ::std::nullopt;
    }

    ::mlir::Attribute annotation =
        terminator->getAttr(loom::raising::loopAnnotationName);
    ::mlir::Attribute candidate =
        terminator->getAttr(loom::raising::candidateLoopHintName);
    if (!annotation && !candidate)
      continue;

    ::mlir::Block *header = nullptr;
    for (::mlir::Block *successor : block->getSuccessors()) {
      if (!dominance.dominates(successor, block) || successor == header)
        continue;
      if (header)
        return ::std::nullopt;
      header = successor;
    }
    if (header && !members.contains(header))
      return ::std::nullopt;
  }
  if (!changesCFGShape)
    return ::std::nullopt;

  LocalRegionPlan plan;
  plan.entry = entry;
  plan.continuation = continuation;
  plan.indexBitwidth = indexBitwidth;
  for (::mlir::Block &block : region)
    if (members.contains(&block))
      plan.blocks.push_back(&block);
  return plan;
}

::std::optional<LocalRegionPlan>
findMaximalLocalRegion(::mlir::Region &region, bool importedLLVMCallable,
                       const DeclinedLocalRegions &declined) {
  if (region.hasOneBlock())
    return ::std::nullopt;

  ::mlir::Operation *callable = region.getParentOp();
  ::mlir::DominanceInfo dominance(callable);
  ::mlir::PostDominanceInfo postDominance(callable);
  uint64_t indexBitwidth =
      ::mlir::DataLayout::closest(callable)
          .getTypeSizeInBits(::mlir::IndexType::get(region.getContext()))
          .getFixedValue();

  ::std::optional<LocalRegionPlan> maximal;
  for (::mlir::Block &entry : region) {
    if (!dominance.isReachableFromEntry(&entry))
      continue;
    auto *postDomNode = postDominance.getNode(&entry);
    auto *continuationNode = postDomNode ? postDomNode->getIDom() : nullptr;
    ::mlir::Block *continuation =
        continuationNode ? continuationNode->getBlock() : nullptr;
    if (!continuation || isDeclinedLocalRegion(declined, &entry, continuation))
      continue;
    auto candidate =
        analyzeLocalRegion(region, &entry, continuation, importedLLVMCallable,
                           dominance, postDominance, indexBitwidth);
    if (!candidate)
      continue;
    if (!maximal || candidate->blocks.size() > maximal->blocks.size())
      maximal = ::std::move(candidate);
  }
  return maximal;
}

class CallableStructuring : public ::mlir::ControlFlowToSCFTransformation {
public:
  CallableStructuring(bool importedLLVMCallable, LoopHintMap &annotations)
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
    LoopHints hints;
    auto frontEntry = annotations.find(&loopBody.front());
    if (frontEntry != annotations.end()) {
      hints = frontEntry->second;
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
        hints = candidate->second;
        annotations.erase(candidate);
      }
    }
    ::mlir::FailureOr<::mlir::Operation *> loop =
        ControlFlowToSCFTransformation::createStructuredDoWhileLoopOp(
            builder, replacedOp, loopValuesInit, condition, loopValuesNextIter,
            std::move(loopBody));
    if (failed(loop))
      return ::mlir::failure();
    loom::raising::carryLoopAnnotation(hints.annotation, *loop);
    loom::raising::carryCandidateLoopHint(hints.candidate, *loop);
    return loop;
  }

  // Location of the first loop annotation whose owning loop could not be
  // proven, if any was met.
  const ::std::optional<::mlir::Location> &firstUnprovenAssociationLoc() const {
    return firstUnprovenAssociation;
  }

private:
  bool importedLLVMCallable;
  LoopHintMap &annotations;
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

LocalRegionPlan remapLocalRegionPlan(const LocalRegionPlan &plan,
                                     ::mlir::IRMapping &mapping) {
  LocalRegionPlan remapped;
  remapped.entry = mapping.lookupOrNull(plan.entry);
  remapped.continuation = mapping.lookupOrNull(plan.continuation);
  remapped.indexBitwidth = plan.indexBitwidth;
  assert(remapped.entry && remapped.continuation &&
         "cloned callable contains local region boundary blocks");
  for (::mlir::Block *block : plan.blocks) {
    ::mlir::Block *cloneBlock = mapping.lookupOrNull(block);
    assert(cloneBlock && "cloned callable contains every local region block");
    remapped.blocks.push_back(cloneBlock);
  }
  return remapped;
}

bool useIsInsideLocalRegion(::mlir::OpOperand &use,
                            ::mlir::Region &callableRegion,
                            const ::llvm::DenseSet<::mlir::Block *> &members) {
  ::mlir::Block *ownerBlock = use.getOwner()->getBlock();
  ::mlir::Block *callableBlock =
      callableRegion.findAncestorBlockInRegion(*ownerBlock);
  return callableBlock && members.contains(callableBlock);
}

::std::optional<::llvm::SmallVector<::mlir::Value, 8>>
collectLocalLiveOuts(::mlir::Region &callableRegion,
                     const LocalRegionPlan &plan,
                     const ::llvm::DenseSet<::mlir::Block *> &members,
                     ::mlir::DominanceInfo &dominance) {
  ::llvm::SmallVector<::mlir::Value, 8> liveOuts;
  auto consider = [&](::mlir::Value value) {
    if (::llvm::any_of(value.getUses(), [&](::mlir::OpOperand &use) {
          return !useIsInsideLocalRegion(use, callableRegion, members);
        }))
      liveOuts.push_back(value);
  };

  bool recurrentEntry = ::llvm::any_of(plan.entry->getPredecessors(),
                                       [&](::mlir::Block *predecessor) {
                                         return members.contains(predecessor);
                                       });
  for (::mlir::Block *block : plan.blocks) {
    if (block != plan.entry || recurrentEntry)
      for (::mlir::BlockArgument argument : block->getArguments())
        consider(argument);
    for (::mlir::Operation &operation : *block)
      for (::mlir::Value result : operation.getResults())
        consider(result);
  }

  for (::mlir::Value liveOut : liveOuts)
    for (::mlir::Block *block : plan.blocks) {
      ::mlir::Operation *terminator = block->getTerminator();
      if (::llvm::none_of(block->getSuccessors(),
                          [&](::mlir::Block *successor) {
                            return successor == plan.continuation;
                          }))
        continue;
      if (!dominance.dominates(liveOut, terminator))
        return ::std::nullopt;
    }
  return liveOuts;
}

bool structureLocalRegion(::mlir::Operation *clonedCallable,
                          ::mlir::Region &callableRegion,
                          const LocalRegionPlan &plan,
                          bool importedLLVMCallable,
                          ::mlir::IRRewriter &rewriter) {
  ::llvm::DenseSet<::mlir::Block *> members(plan.blocks.begin(),
                                            plan.blocks.end());
  ::mlir::DominanceInfo originalDominance(clonedCallable);
  for (::mlir::Block *block : plan.blocks)
    for (::mlir::Block *predecessor : block->getPredecessors())
      if (!members.contains(predecessor) &&
          !originalDominance.isReachableFromEntry(predecessor))
        return false;
  auto liveOuts =
      collectLocalLiveOuts(callableRegion, plan, members, originalDominance);
  if (!liveOuts)
    return false;

  ::llvm::SmallVector<::mlir::Type, 8> resultTypes;
  for (::mlir::BlockArgument argument : plan.continuation->getArguments())
    resultTypes.push_back(argument.getType());
  for (::mlir::Value liveOut : *liveOuts)
    resultTypes.push_back(liveOut.getType());

  ::mlir::Location location = plan.entry->getTerminator()->getLoc();
  rewriter.setInsertionPointToStart(plan.entry);
  auto execute = ::mlir::scf::ExecuteRegionOp::create(
      rewriter, location, resultTypes, /*no_inline=*/true);
  ::mlir::Region &localRegion = execute.getRegion();
  ::mlir::Block *gateway = new ::mlir::Block;
  ::mlir::Block *innerEntry = new ::mlir::Block;
  localRegion.push_back(gateway);
  localRegion.push_back(innerEntry);
  for (::mlir::BlockArgument argument : plan.entry->getArguments())
    innerEntry->addArgument(argument.getType(), argument.getLoc());

  innerEntry->getOperations().splice(
      innerEntry->end(), plan.entry->getOperations(),
      ::std::next(execute->getIterator()), plan.entry->end());
  for (::mlir::Block *block : plan.blocks) {
    if (block == plan.entry)
      continue;
    localRegion.getBlocks().splice(
        localRegion.end(), callableRegion.getBlocks(), block->getIterator());
  }

  for (auto [outerArgument, innerArgument] : ::llvm::zip_equal(
           plan.entry->getArguments(), innerEntry->getArguments())) {
    outerArgument.replaceUsesWithIf(innerArgument, [&](::mlir::OpOperand &use) {
      return execute->isAncestor(use.getOwner());
    });
  }

  ::llvm::SmallVector<::mlir::Value, 8> yieldedLiveOuts;
  yieldedLiveOuts.reserve(liveOuts->size());
  for (::mlir::Value liveOut : *liveOuts) {
    auto argument = ::mlir::dyn_cast<::mlir::BlockArgument>(liveOut);
    if (argument && argument.getOwner() == plan.entry)
      liveOut = innerEntry->getArgument(argument.getArgNumber());
    yieldedLiveOuts.push_back(liveOut);
  }

  for (::mlir::Block &block : localRegion) {
    if (&block == gateway || block.empty())
      continue;
    ::mlir::Operation *terminator = block.getTerminator();
    for (unsigned index = 0; index != terminator->getNumSuccessors(); ++index)
      if (terminator->getSuccessor(index) == plan.entry)
        terminator->setSuccessor(innerEntry, index);
  }

  ::mlir::Block *exitTrampoline = new ::mlir::Block;
  localRegion.push_back(exitTrampoline);
  ::llvm::SmallVector<::mlir::Location, 8> resultLocations(resultTypes.size(),
                                                           location);
  exitTrampoline->addArguments(resultTypes, resultLocations);

  bool redirectedExit = false;
  for (::mlir::Block &block : localRegion) {
    if (&block == gateway || &block == exitTrampoline || block.empty())
      continue;
    ::mlir::Operation *terminator = block.getTerminator();
    auto branch = ::mlir::dyn_cast<::mlir::BranchOpInterface>(terminator);
    if (!branch)
      return false;
    for (unsigned index = 0; index != terminator->getNumSuccessors(); ++index) {
      if (terminator->getSuccessor(index) != plan.continuation)
        continue;
      ::mlir::SuccessorOperands successorOperands =
          branch.getSuccessorOperands(index);
      if (successorOperands.getProducedOperandCount() != 0)
        return false;
      successorOperands.append(yieldedLiveOuts);
      terminator->setSuccessor(exitTrampoline, index);
      redirectedExit = true;
    }
  }
  if (!redirectedExit)
    return false;

  rewriter.setInsertionPointToEnd(gateway);
  ::mlir::cf::BranchOp::create(rewriter, location, innerEntry,
                               plan.entry->getArguments());
  rewriter.setInsertionPointToEnd(exitTrampoline);
  ::mlir::scf::YieldOp::create(rewriter, location,
                               exitTrampoline->getArguments());

  rewriter.setInsertionPointToEnd(plan.entry);
  ::mlir::cf::BranchOp::create(
      rewriter, location, plan.continuation,
      execute.getResults().take_front(plan.continuation->getNumArguments()));
  for (auto [liveOut, replacement] : ::llvm::zip_equal(
           *liveOuts, execute.getResults().drop_front(
                          plan.continuation->getNumArguments()))) {
    liveOut.replaceUsesWithIf(replacement, [&](::mlir::OpOperand &use) {
      return !execute->isAncestor(use.getOwner());
    });
  }

  LoopHintMap annotations;
  OrphanLoopHintBlocks orphanHints;
  if (decideRegion(localRegion, importedLLVMCallable, plan.indexBitwidth,
                   annotations, orphanHints) != Disposition::Structure)
    return false;
  for (::mlir::Block *orphanBlock : orphanHints) {
    orphanBlock->getTerminator()->removeAttr(loom::raising::loopAnnotationName);
    orphanBlock->getTerminator()->removeAttr(
        loom::raising::candidateLoopHintName);
  }

  ::mlir::DominanceInfo dominance(clonedCallable);
  CallableStructuring structuring(importedLLVMCallable, annotations);
  if (::mlir::failed(
          ::mlir::transformCFGToSCF(localRegion, structuring, dominance)))
    return false;
  makeResidualSwitchesRoundTripSafe(localRegion, rewriter);
  if (structuring.firstUnprovenAssociationLoc() || !annotations.empty() ||
      !localRegion.hasOneBlock())
    return false;

  auto yield = ::mlir::dyn_cast<::mlir::scf::YieldOp>(
      localRegion.front().getTerminator());
  if (!yield || yield.getNumOperands() != execute.getNumResults())
    return false;
  ::llvm::SmallVector<::mlir::Value, 8> yielded(yield.getOperands());
  rewriter.eraseOp(yield);
  rewriter.inlineBlockBefore(&localRegion.front(), execute.getOperation());
  rewriter.replaceOp(execute, yielded);
  return true;
}

bool structureCallableLocally(::mlir::Region &region, bool importedLLVMCallable,
                              ::mlir::IRRewriter &rewriter) {
  ::mlir::Operation *callable = region.getParentOp();
  DeclinedLocalRegions declined;
  bool changed = false;
  while (auto localPlan =
             findMaximalLocalRegion(region, importedLLVMCallable, declined)) {
    const size_t originalBlockCount = region.getBlocks().size();
    const size_t originalBranchCount =
        ::llvm::count_if(region, [](::mlir::Block &block) {
          return block.getTerminator()->getNumSuccessors() > 1;
        });
    ::mlir::IRMapping mapping;
    ::mlir::OwningOpRef<::mlir::Operation *> clone(callable->clone(mapping));
    ::mlir::Region &cloneRegion = clone->getRegion(region.getRegionNumber());
    (void)eraseUnanchoredUnreachableBlocks(clone.get(), cloneRegion, rewriter);
    LocalRegionPlan clonePlan = remapLocalRegionPlan(*localPlan, mapping);
    if (!structureLocalRegion(clone.get(), cloneRegion, clonePlan,
                              importedLLVMCallable, rewriter)) {
      declined.emplace_back(localPlan->entry, localPlan->continuation);
      continue;
    }

    const size_t cloneBlockCount = cloneRegion.getBlocks().size();
    const size_t cloneBranchCount =
        ::llvm::count_if(cloneRegion, [](::mlir::Block &block) {
          return block.getTerminator()->getNumSuccessors() > 1;
        });
    if (cloneBlockCount > originalBlockCount ||
        (cloneBlockCount == originalBlockCount &&
         cloneBranchCount >= originalBranchCount)) {
      declined.emplace_back(localPlan->entry, localPlan->continuation);
      continue;
    }

    region.takeBody(cloneRegion);
    changed = true;
    declined.clear();
  }
  return changed;
}

bool structureCallableWhole(::mlir::Region &region, bool importedLLVMCallable,
                            const LoopHintMap &annotations,
                            const OrphanLoopHintBlocks &orphanHints,
                            ::mlir::IRRewriter &rewriter) {
  ::mlir::Operation *callable = region.getParentOp();
  ::mlir::IRMapping mapping;
  ::mlir::OwningOpRef<::mlir::Operation *> clone(callable->clone(mapping));
  ::mlir::Region &cloneRegion = clone->getRegion(region.getRegionNumber());

  LoopHintMap cloneAnnotations;
  for (const auto &[header, hints] : annotations) {
    ::mlir::Block *cloneHeader = mapping.lookupOrNull(header);
    assert(cloneHeader && "cloned region holds every original block");
    cloneAnnotations[cloneHeader] = hints;
  }
  for (::mlir::Block *orphanBlock : orphanHints) {
    ::mlir::Block *cloneBlock = mapping.lookupOrNull(orphanBlock);
    assert(cloneBlock && "cloned region holds every original block");
    cloneBlock->getTerminator()->removeAttr(loom::raising::loopAnnotationName);
    cloneBlock->getTerminator()->removeAttr(
        loom::raising::candidateLoopHintName);
  }

  if (!eraseUnanchoredUnreachableBlocks(clone.get(), cloneRegion, rewriter))
    return false;
  ::mlir::DominanceInfo dominance(clone.get());
  CallableStructuring structuring(importedLLVMCallable, cloneAnnotations);
  if (::mlir::failed(
          ::mlir::transformCFGToSCF(cloneRegion, structuring, dominance)))
    return false;
  makeResidualSwitchesRoundTripSafe(cloneRegion, rewriter);
  if (structuring.firstUnprovenAssociationLoc() || !cloneAnnotations.empty())
    return false;

  region.takeBody(cloneRegion);
  return true;
}

struct LiftCFToSCFPass
    : public ::mlir::PassWrapper<LiftCFToSCFPass, ::mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftCFToSCFPass)

  ::llvm::StringRef getArgument() const final { return "loom-lift-cf-to-scf"; }
  ::llvm::StringRef getDescription() const final {
    return "Structure each maximal exactly provable cf-shaped region with "
           "the upstream CFG-to-SCF transformation, leaving an imported "
           "llvm.func as the callable and ABI owner of its body, retaining "
           "weighted or unsupported local control, and moving each imported "
           "loop annotation to the loop that owns its cycle.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::cf::ControlFlowDialect,
                    ::mlir::LLVM::LLVMDialect, ::mlir::scf::SCFDialect,
                    ::mlir::ub::UBDialect>();
  }

  void runOnOperation() final {
    // One plan per callable region. A completely admissible region carries its
    // resolved annotations for the whole-region fast path; every other region
    // is revisited for local SESE extraction when its turn is reached.
    struct RegionPlan {
      ::mlir::Region *region;
      bool structureWhole;
      LoopHintMap annotations;
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
          LoopHintMap regionAnnotations;
          OrphanLoopHintBlocks orphanHints;
          bool importedLLVMCallable =
              ::mlir::isa<::mlir::LLVM::LLVMFuncOp>(region.getParentOp());
          uint64_t indexBitwidth =
              ::mlir::DataLayout::closest(region.getParentOp())
                  .getTypeSizeInBits(
                      ::mlir::IndexType::get(region.getContext()))
                  .getFixedValue();
          bool structureWhole =
              decideRegion(region, importedLLVMCallable, indexBitwidth,
                           regionAnnotations,
                           orphanHints) == Disposition::Structure;
          plans.push_back({&region, structureWhole,
                           std::move(regionAnnotations),
                           std::move(orphanHints)});
          return ::mlir::success();
        });

    ::mlir::IRRewriter rewriter(&getContext());
    bool structuredAny = false;
    for (RegionPlan &plan : plans) {
      ::mlir::Region *region = plan.region;
      ::mlir::Operation *originalCallable = region->getParentOp();
      bool importedLLVMCallable =
          ::mlir::isa<::mlir::LLVM::LLVMFuncOp>(originalCallable);

      if (plan.structureWhole &&
          structureCallableWhole(*region, importedLLVMCallable,
                                 plan.annotations, plan.orphanHints,
                                 rewriter)) {
        structuredAny = true;
        continue;
      }

      // A whole-region attempt can discover a loop-hint association that was
      // unprovable only after upstream formed its entry multiplexer. Retry on
      // independently closed local regions so that one such loop cannot hide
      // unrelated exact structure in the same callable.
      structuredAny |=
          structureCallableLocally(*region, importedLLVMCallable, rewriter);
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

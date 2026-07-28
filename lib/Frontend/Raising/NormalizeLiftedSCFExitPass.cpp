// Normalize only the loop-exit scaffold emitted by CFG-to-SCF structuring.
//
// The lift encodes an exit with a yield-only scf.if that selects either
// continuation values or undefined-value placeholders, followed by an i32
// discriminator and an i32 shouldRepeat flag. The placeholder is
// llvm.mlir.undef inside an imported LLVM callable and ub.poison inside a
// native one. The shouldRepeat flag is truncated to i1 for scf.condition.
// The scaffold collapses back into the direct arith.cmpi condition it
// selects, which is the exact canonical form of the lifted exit.
//
// Generic SCF canonicalization is not safe here: it can combine nested lazy
// scf.if conditions into an eager arith.andi. This pass instead recognizes the
// complete lift-owned scaffold and rewrites it directly. A live result may be
// redirected only to its exact dominating exit value or to another recurrence
// lane carrying that value. A source poison, undef, or freeze therefore keeps
// its own meaning.

#include "Frontend/Raising/Passes.h"

#include "CallableRegions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace {

::mlir::scf::YieldOp getOnlyYield(::mlir::Region &region) {
  if (!region.hasOneBlock() || !::llvm::hasSingleElement(region.front()))
    return {};
  return ::mlir::dyn_cast<::mlir::scf::YieldOp>(region.front().front());
}

bool matchZeroOrOne(::mlir::Value value, bool &one) {
  ::llvm::APInt constant;
  if (!::mlir::matchPattern(value, ::mlir::m_ConstantInt(&constant)))
    return false;
  if (constant.isZero()) {
    one = false;
    return true;
  }
  if (constant.isOne()) {
    one = true;
    return true;
  }
  return false;
}

bool normalizeLiftedExit(::mlir::scf::ConditionOp condition,
                         ::mlir::IRRewriter &rewriter) {
  auto loop = ::mlir::dyn_cast<::mlir::scf::WhileOp>(condition->getParentOp());
  if (!loop || !loop.getBefore().hasOneBlock() ||
      condition->getBlock() != &loop.getBefore().front() ||
      loop.getInits().size() != loop->getNumResults() ||
      condition.getArgs().size() != loop->getNumResults())
    return false;

  // The lift scaffold's after region is the identity continuation emitted by
  // createStructuredDoWhileLoopOp(): its only op is an scf.yield forwarding
  // every after-block argument in positional order. Any other shape means the
  // scaffold is not lift-owned, so leave it untouched.
  ::mlir::scf::YieldOp afterYield = getOnlyYield(loop.getAfter());
  if (!afterYield)
    return false;
  if (!::llvm::equal(afterYield.getResults(),
                     loop.getAfter().front().getArguments()))
    return false;

  auto trunc =
      condition.getCondition().getDefiningOp<::mlir::arith::TruncIOp>();
  if (!trunc || !trunc.getType().isInteger(1) ||
      !trunc.getIn().getType().isInteger(32) ||
      trunc->getBlock() != condition->getBlock() ||
      trunc->getNextNode() != condition.getOperation() || !trunc->hasOneUse() ||
      trunc.getOverflowFlags() != ::mlir::arith::IntegerOverflowFlags::none)
    return false;

  auto flagResult = ::llvm::dyn_cast<::mlir::OpResult>(trunc.getIn());
  if (!flagResult || !flagResult.hasOneUse() ||
      flagResult.use_begin()->getOwner() != trunc.getOperation())
    return false;

  auto branch = ::mlir::dyn_cast<::mlir::scf::IfOp>(flagResult.getOwner());
  if (!branch || branch->getBlock() != condition->getBlock() ||
      branch->getNextNode() != trunc.getOperation() ||
      branch->getNumResults() < 2 ||
      branch->getNumResults() > condition.getArgs().size() + 2)
    return false;

  unsigned loopValueCount = condition.getArgs().size();
  unsigned controlledValueCount = branch->getNumResults() - 2;
  unsigned discriminatorIndex = controlledValueCount;
  unsigned shouldRepeatIndex = controlledValueCount + 1;
  if (flagResult.getResultNumber() != shouldRepeatIndex)
    return false;

  ::mlir::scf::YieldOp thenYield = getOnlyYield(branch.getThenRegion());
  ::mlir::scf::YieldOp elseYield = getOnlyYield(branch.getElseRegion());
  if (!thenYield || !elseYield ||
      thenYield.getResults().size() != branch->getNumResults() ||
      elseYield.getResults().size() != branch->getNumResults())
    return false;

  bool shouldRepeatThen;
  bool shouldRepeatElse;
  if (!matchZeroOrOne(thenYield.getResults()[shouldRepeatIndex],
                      shouldRepeatThen) ||
      !matchZeroOrOne(elseYield.getResults()[shouldRepeatIndex],
                      shouldRepeatElse) ||
      shouldRepeatThen == shouldRepeatElse)
    return false;

  ::mlir::OpResult discriminator = branch->getResult(discriminatorIndex);
  bool discriminatorThen;
  bool discriminatorElse;
  if (!discriminator.use_empty() || !discriminator.getType().isInteger(32) ||
      !matchZeroOrOne(thenYield.getResults()[discriminatorIndex],
                      discriminatorThen) ||
      !matchZeroOrOne(elseYield.getResults()[discriminatorIndex],
                      discriminatorElse) ||
      discriminatorThen == shouldRepeatThen ||
      discriminatorElse == shouldRepeatElse)
    return false;

  // Branch-controlled values occupy a prefix of the if results and appear in
  // the condition arguments in the same relative order. A condition argument
  // may instead be the same direct value on both edges. This is how the pinned
  // lift publishes a live accumulator without routing it through the branch.
  ::llvm::SmallVector<int, 4> controlledResultByLane(loopValueCount, -1);
  unsigned nextControlledResult = 0;
  for (unsigned index = 0; index < loopValueCount; ++index) {
    auto result =
        ::llvm::dyn_cast<::mlir::OpResult>(condition.getArgs()[index]);
    if (!result || result.getOwner() != branch.getOperation())
      continue;
    if (result.getResultNumber() != nextControlledResult ||
        result.getResultNumber() >= controlledValueCount ||
        !result.hasOneUse() ||
        result.use_begin()->getOwner() != condition.getOperation())
      return false;
    controlledResultByLane[index] = static_cast<int>(nextControlledResult++);
  }
  if (nextControlledResult != controlledValueCount)
    return false;

  ::llvm::SmallVector<::mlir::Value, 4> continuationArgs;
  ::llvm::SmallVector<::mlir::Value, 4> exitArgs;
  continuationArgs.reserve(loopValueCount);
  exitArgs.reserve(loopValueCount);
  for (unsigned index = 0; index < loopValueCount; ++index) {
    int controlled = controlledResultByLane[index];
    if (controlled < 0) {
      continuationArgs.push_back(condition.getArgs()[index]);
      exitArgs.push_back(condition.getArgs()[index]);
      continue;
    }
    unsigned resultIndex = static_cast<unsigned>(controlled);
    continuationArgs.push_back(shouldRepeatThen
                                   ? thenYield.getResults()[resultIndex]
                                   : elseYield.getResults()[resultIndex]);
    exitArgs.push_back(shouldRepeatThen ? elseYield.getResults()[resultIndex]
                                        : thenYield.getResults()[resultIndex]);
  }

  const auto exceptionalPlaceholder = [](::mlir::Value value) {
    return ::mlir::isa_and_present<::mlir::ub::PoisonOp, ::mlir::LLVM::UndefOp>(
        value.getDefiningOp());
  };

  struct ResultProjection {
    unsigned source;
    ::mlir::Value replacement;
    std::optional<unsigned> targetLane;
    ::llvm::SmallVector<::mlir::OpOperand *, 2> uses;
  };
  ::llvm::SmallVector<ResultProjection, 2> projections;
  ::llvm::SmallVector<::mlir::Operation *, 8> placeholders;
  ::mlir::DominanceInfo dominance(loop->getParentOp());
  const auto rememberPlaceholder = [&](::mlir::Value value) {
    ::mlir::Operation *operation = value.getDefiningOp();
    if (operation && !::llvm::is_contained(placeholders, operation))
      placeholders.push_back(operation);
  };

  for (unsigned index = 0; index < loopValueCount; ++index) {
    ::mlir::Value result = loop.getResult(index);
    if (result.use_empty()) {
      if (controlledResultByLane[index] >= 0 &&
          !exceptionalPlaceholder(exitArgs[index]))
        return false;
      if (controlledResultByLane[index] >= 0)
        rememberPlaceholder(exitArgs[index]);
      continue;
    }

    if (dominance.properlyDominates(exitArgs[index], loop)) {
      ResultProjection projection{index, exitArgs[index], std::nullopt, {}};
      for (::mlir::OpOperand &use : result.getUses())
        projection.uses.push_back(&use);
      projections.push_back(std::move(projection));
      continue;
    }

    if (exceptionalPlaceholder(exitArgs[index]))
      return false;
    auto target = ::llvm::find(continuationArgs, exitArgs[index]);
    if (target == continuationArgs.end())
      return false;
    unsigned targetIndex =
        static_cast<unsigned>(target - continuationArgs.begin());

    ResultProjection projection{
        index, loop.getResult(targetIndex), targetIndex, {}};
    for (::mlir::OpOperand &use : result.getUses())
      projection.uses.push_back(&use);

    if (targetIndex != index) {
      if (!loop.getBeforeArguments()[index].use_empty() ||
          !exceptionalPlaceholder(loop.getInits()[index]))
        return false;
      rememberPlaceholder(loop.getInits()[index]);
      if (exceptionalPlaceholder(continuationArgs[index]))
        rememberPlaceholder(continuationArgs[index]);
    }
    projections.push_back(std::move(projection));
  }

  for (const ResultProjection &projection : projections) {
    if (!projection.targetLane || projection.source == *projection.targetLane)
      continue;
    loop->setOperand(projection.source,
                     loop.getInits()[*projection.targetLane]);
    continuationArgs[projection.source] =
        continuationArgs[*projection.targetLane];
  }

  rewriter.setInsertionPoint(condition);
  ::mlir::Value selector = branch.getCondition();
  ::mlir::Operation *obsoleteComparison = nullptr;
  if (!shouldRepeatThen) {
    if (auto comparison =
            branch.getCondition().getDefiningOp<::mlir::arith::CmpIOp>()) {
      selector = ::mlir::arith::CmpIOp::create(
          rewriter, comparison.getLoc(),
          ::mlir::arith::invertPredicate(comparison.getPredicate()),
          comparison.getLhs(), comparison.getRhs());
      if (comparison->hasOneUse())
        obsoleteComparison = comparison;
    } else {
      ::mlir::Value one = ::mlir::arith::ConstantOp::create(
          rewriter, branch.getLoc(), rewriter.getBoolAttr(true));
      selector = ::mlir::arith::XOrIOp::create(rewriter, branch.getLoc(),
                                               branch.getCondition(), one);
    }
  }

  rewriter.replaceOpWithNewOp<::mlir::scf::ConditionOp>(condition, selector,
                                                        continuationArgs);
  rewriter.eraseOp(trunc);
  rewriter.eraseOp(branch);
  if (obsoleteComparison)
    rewriter.eraseOp(obsoleteComparison);
  for (const ResultProjection &projection : projections)
    for (::mlir::OpOperand *use : projection.uses)
      use->set(projection.replacement);
  for (::mlir::Operation *placeholder : placeholders)
    if (placeholder->use_empty())
      rewriter.eraseOp(placeholder);
  return true;
}

struct NormalizeLiftedSCFExitPass
    : public ::mlir::PassWrapper<NormalizeLiftedSCFExitPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeLiftedSCFExitPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-normalize-lifted-scf-exit";
  }
  ::llvm::StringRef getDescription() const final {
    return "Normalize the exact poison-safe loop-exit scaffold emitted by "
           "CFG-to-SCF structuring";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::scf::SCFDialect,
                    ::mlir::ub::UBDialect>();
  }

  void runOnOperation() final {
    ::llvm::SmallVector<::mlir::scf::ConditionOp> conditions;
    (void)loom::raising::forEachCallableRegion(
        getOperation(), [&](::mlir::Region &region) {
          (void)loom::raising::forEachOwnedOperation(
              region, [&](::mlir::Operation *op) {
                if (auto condition =
                        ::mlir::dyn_cast<::mlir::scf::ConditionOp>(op))
                  conditions.push_back(condition);
                return ::mlir::WalkResult::advance();
              });
          return ::mlir::success();
        });

    ::mlir::IRRewriter rewriter(&getContext());
    for (::mlir::scf::ConditionOp condition : conditions)
      normalizeLiftedExit(condition, rewriter);
  }
};

} // namespace

namespace loom {
namespace raising {

std::unique_ptr<::mlir::Pass> createNormalizeLiftedSCFExitPass() {
  return std::make_unique<NormalizeLiftedSCFExitPass>();
}

void registerNormalizeLiftedSCFExitPass() {
  static bool once = []() {
    ::mlir::PassRegistration<NormalizeLiftedSCFExitPass>();
    return true;
  }();
  (void)once;
}

} // namespace raising
} // namespace loom

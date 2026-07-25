// Normalize only the loop-exit scaffold emitted by CFG-to-SCF structuring.
//
// The lift encodes an exit with a yield-only scf.if that selects either
// continuation values or undefined-value placeholders, followed by an i32
// discriminator and an i32 shouldRepeat flag. The placeholder is
// llvm.mlir.undef inside an imported LLVM callable and ub.poison inside a
// native one. The shouldRepeat flag is truncated to i1 for scf.condition.
// Counted-loop uplift needs that condition to be an arith.cmpi directly.
//
// Generic SCF canonicalization is not safe here: it can combine nested lazy
// scf.if conditions into an eager arith.andi. This pass instead recognizes the
// complete lift-owned scaffold and rewrites it directly. If any while result is
// live, the scaffold is left intact so its exit-edge value remains observable.
// Only that unobservable placeholder is removed, so a source poison, undef or
// freeze keeps its own meaning.

#include "Frontend/Raising/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

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

  // Replacing an exit-edge placeholder is exact only when the corresponding
  // loop result is unobservable. Keep the whole scaffold if any result is live.
  if (::llvm::any_of(loop.getResults(),
                     [](::mlir::Value result) { return !result.use_empty(); }))
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
      branch->getNumResults() != condition.getArgs().size() + 2)
    return false;

  unsigned loopValueCount = condition.getArgs().size();
  unsigned discriminatorIndex = loopValueCount;
  unsigned shouldRepeatIndex = loopValueCount + 1;
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

  // The loop values occupy the fixed prefix and feed the condition in the
  // same order. No arbitrary permutation is part of the pinned lift shape.
  for (unsigned index = 0; index < loopValueCount; ++index) {
    ::mlir::OpResult result = branch->getResult(index);
    if (condition.getArgs()[index] != result || !result.hasOneUse() ||
        result.use_begin()->getOwner() != condition.getOperation())
      return false;
  }

  auto comparison =
      branch.getCondition().getDefiningOp<::mlir::arith::CmpIOp>();
  if (!comparison || comparison->getBlock() != condition->getBlock() ||
      comparison->getNextNode() != branch.getOperation() ||
      !comparison->hasOneUse())
    return false;

  ::llvm::SmallVector<::mlir::Value, 4> continuationArgs;
  continuationArgs.reserve(loopValueCount);
  for (unsigned index = 0; index < loopValueCount; ++index) {
    ::mlir::Value continuation = shouldRepeatThen
                                     ? thenYield.getResults()[index]
                                     : elseYield.getResults()[index];
    ::mlir::Value exit = shouldRepeatThen ? elseYield.getResults()[index]
                                          : thenYield.getResults()[index];
    if (!::mlir::isa_and_present<::mlir::ub::PoisonOp, ::mlir::LLVM::UndefOp>(
            exit.getDefiningOp()))
      return false;
    continuationArgs.push_back(continuation);
  }

  rewriter.setInsertionPoint(condition);
  ::mlir::Value selector = branch.getCondition();
  if (!shouldRepeatThen)
    selector = ::mlir::arith::CmpIOp::create(
        rewriter, comparison.getLoc(),
        ::mlir::arith::invertPredicate(comparison.getPredicate()),
        comparison.getLhs(), comparison.getRhs());

  rewriter.replaceOpWithNewOp<::mlir::scf::ConditionOp>(condition, selector,
                                                        continuationArgs);
  rewriter.eraseOp(trunc);
  rewriter.eraseOp(branch);
  if (!shouldRepeatThen)
    rewriter.eraseOp(comparison);
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
    getOperation().walk([&](::mlir::scf::ConditionOp condition) {
      conditions.push_back(condition);
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

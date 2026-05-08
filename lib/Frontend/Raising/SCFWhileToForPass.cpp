// Uplift counted scf.while loops produced by --lift-cf-to-scf into
// scf.for. We run two flavours of pattern:
//
//   1. The upstream `scf::populateUpliftWhileToForPatterns` patterns,
//      which handle while loops whose `before` block contains only an
//      arith.cmpi slt/sgt and forward all carried values directly.
//
//   2. A Loom-specific pattern for the do-while shape that
//      --lift-cf-to-scf emits when the original LLVM IR had a
//      counted-reduction loop with the increment placed inside the
//      latch. That shape looks like:
//
//          scf.while (%iv = %lb, %carry = %init)
//                  : (i64, T) -> (i64, T) {
//             ... body using %iv, %carry ...
//             %iv_next = arith.addi %iv, %step
//             %cond = arith.cmpi ne, %iv_next, %ub
//             scf.condition(%cond) %iv_next, %carry_next : i64, T
//          } do {
//          ^bb0(%iv_next: i64, %carry_next: T):
//             scf.yield %iv_next, %carry_next : i64, T
//          }
//
//      We rewrite this into an scf.for whose induction variable spans
//      [%lb, %ub) with the same %step, and whose iter_args carry the
//      remaining loop-carried values. The induction-variable use inside
//      the body is rewritten to the scf.for IV (i.e. the value before
//      the bump).

#include "Frontend/Raising/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

namespace {

// Helper: given an scf.while loop, identify whether one of the carried
// values is a counted induction var that is bumped by an arith.addi
// inside the `before` block, and whose loop bound is checked via
// arith.cmpi <ne|slt|sgt|ult|ugt> against a loop-invariant value. On
// match returns the index of the iv in the before-block argument list,
// the addi op that bumps it, the cmpi op, the upper bound, and the
// step. The cmpi must be the only user of the addi, and the addi must
// be the only user of the iv apart from the addi/cmpi chain.
struct CountedInfo {
  unsigned ivIndex = 0;
  ::mlir::arith::AddIOp addOp;
  ::mlir::arith::CmpIOp cmpOp;
  ::mlir::Value step;
  ::mlir::Value upperBound;
};

bool isLoopInvariant(::mlir::Value v, ::mlir::scf::WhileOp loop) {
  if (auto blockArg = ::mlir::dyn_cast<::mlir::BlockArgument>(v)) {
    return !loop->isAncestor(blockArg.getOwner()->getParentOp());
  }
  ::mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  return !loop->isAncestor(def);
}

::mlir::FailureOr<CountedInfo>
matchCountedDoWhile(::mlir::scf::WhileOp loop) {
  ::mlir::Block *beforeBody = loop.getBeforeBody();
  ::mlir::Block *afterBody = loop.getAfterBody();
  ::mlir::scf::ConditionOp condOp = loop.getConditionOp();
  ::mlir::scf::YieldOp yieldOp = loop.getYieldOp();

  // The condition must be the result of an arith.cmpi.
  auto cmpOp = condOp.getCondition().getDefiningOp<::mlir::arith::CmpIOp>();
  if (!cmpOp)
    return ::mlir::failure();
  if (cmpOp->getParentRegion() != &loop.getBefore())
    return ::mlir::failure();

  // Predicate accepted: ne/slt/sgt/ult/ugt -- in all those cases the
  // induction variable is the bumped one and we exit when it reaches
  // the bound.
  using Pred = ::mlir::arith::CmpIPredicate;
  Pred pred = cmpOp.getPredicate();
  if (pred != Pred::ne && pred != Pred::slt && pred != Pred::sgt &&
      pred != Pred::ult && pred != Pred::ugt && pred != Pred::sle &&
      pred != Pred::ule && pred != Pred::sge && pred != Pred::uge &&
      pred != Pred::eq) {
    return ::mlir::failure();
  }
  // For the `eq` predicate we are testing for the exit value, but
  // `scf.condition` keeps iterating while its condition is *true*, so
  // `eq` would terminate exactly once which is not a loop.
  if (pred == Pred::eq)
    return ::mlir::failure();

  // The bumped iv comes from an arith.addi of a before-block argument
  // and a loop-invariant step. Try both lhs and rhs of the cmpi.
  ::mlir::Value lhs = cmpOp.getLhs();
  ::mlir::Value rhs = cmpOp.getRhs();
  ::mlir::arith::AddIOp addOp;
  ::mlir::Value upperBound;
  for (bool flip : {false, true}) {
    ::mlir::Value bumped = flip ? rhs : lhs;
    ::mlir::Value bound = flip ? lhs : rhs;
    auto candAdd = bumped.getDefiningOp<::mlir::arith::AddIOp>();
    if (!candAdd)
      continue;
    if (!isLoopInvariant(bound, loop))
      continue;
    addOp = candAdd;
    upperBound = bound;
    break;
  }
  if (!addOp || !upperBound)
    return ::mlir::failure();

  // One of the addi operands must be a before-block arg (the iv), the
  // other must be loop-invariant (the step).
  ::mlir::BlockArgument ivArg;
  ::mlir::Value step;
  for (bool flip : {false, true}) {
    ::mlir::Value cand = flip ? addOp.getRhs() : addOp.getLhs();
    ::mlir::Value other = flip ? addOp.getLhs() : addOp.getRhs();
    auto barg = ::mlir::dyn_cast<::mlir::BlockArgument>(cand);
    if (!barg || barg.getOwner() != beforeBody)
      continue;
    if (!isLoopInvariant(other, loop))
      continue;
    ivArg = barg;
    step = other;
    break;
  }
  if (!ivArg || !step)
    return ::mlir::failure();

  // The bumped iv must be passed back through the condition op at the
  // matching position, and the same iv arg must be the after-block's
  // matching arg, and the after-block must yield it back unchanged.
  unsigned ivIdx = ivArg.getArgNumber();
  if (condOp.getArgs().size() <= ivIdx)
    return ::mlir::failure();
  if (condOp.getArgs()[ivIdx] != addOp.getResult())
    return ::mlir::failure();
  if (afterBody->getNumArguments() <= ivIdx)
    return ::mlir::failure();
  ::mlir::BlockArgument afterIv = afterBody->getArgument(ivIdx);
  if (yieldOp.getResults().size() <= ivIdx)
    return ::mlir::failure();
  if (yieldOp.getResults()[ivIdx] != afterIv)
    return ::mlir::failure();

  // The addi result must only feed the cmpi and the condition op's iv
  // slot. (The iv argument itself may be used freely throughout the
  // body -- it is the loop's induction variable.)
  unsigned addUsers = 0;
  for (::mlir::Operation *user : addOp->getUsers()) {
    if (user == cmpOp.getOperation() || user == condOp.getOperation())
      ++addUsers;
    else
      return ::mlir::failure();
  }
  if (addUsers == 0)
    return ::mlir::failure();

  CountedInfo info;
  info.ivIndex = ivIdx;
  info.addOp = addOp;
  info.cmpOp = cmpOp;
  info.step = step;
  info.upperBound = upperBound;
  return info;
}

// Pattern: do-while-counted -> scf.for. Acts as a fallback for the
// upstream uplift pattern, which only handles slt/sgt before-body-only
// cases.
struct UpliftDoWhileToFor
    : public ::mlir::OpRewritePattern<::mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::scf::WhileOp loop,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto matched = matchCountedDoWhile(loop);
    if (failed(matched))
      return ::mlir::failure();
    CountedInfo info = *matched;

    ::mlir::Location loc = loop.getLoc();
    ::mlir::Value lb = loop.getInits()[info.ivIndex];
    ::mlir::Value ub = info.upperBound;
    ::mlir::Value step = info.step;
    ::mlir::Type ivType = lb.getType();
    if (ivType != ub.getType() || ivType != step.getType())
      return ::mlir::failure();

    // Build the iter_args list: the loop's inits with the iv slot
    // removed, kept in the original order.
    ::llvm::SmallVector<::mlir::Value, 4> iterArgs;
    iterArgs.reserve(loop.getInits().size() - 1);
    for (auto pair : ::llvm::enumerate(loop.getInits())) {
      if (pair.index() == info.ivIndex)
        continue;
      iterArgs.push_back(pair.value());
    }

    // Empty builder -- we will inline the original body manually.
    auto emptyBuilder = [](::mlir::OpBuilder &, ::mlir::Location,
                           ::mlir::Value, ::mlir::ValueRange) {};
    auto forOp = ::mlir::scf::ForOp::create(rewriter, loc, lb, ub, step,
                                            iterArgs, emptyBuilder);
    ::mlir::Block *forBody = forOp.getBody();

    // Erase the trivial yield ForOp::create added; we will append our
    // own when the body has been inlined.
    if (!forBody->empty())
      rewriter.eraseOp(forBody->getTerminator());

    // Map the original before-block arguments to the for-body
    // arguments. For-body arguments are: induction var, iter_arg0,
    // iter_arg1, ... -- they are in the same order as iterArgs.
    ::mlir::IRMapping mapping;
    ::mlir::Block *origBefore = loop.getBeforeBody();
    unsigned forIterIdx = 1; // skip iv
    for (auto pair : ::llvm::enumerate(origBefore->getArguments())) {
      if (pair.index() == info.ivIndex) {
        mapping.map(pair.value(), forBody->getArgument(0));
      } else {
        mapping.map(pair.value(), forBody->getArgument(forIterIdx++));
      }
    }

    rewriter.setInsertionPointToEnd(forBody);
    for (::mlir::Operation &op : origBefore->without_terminator()) {
      // Skip the addi+cmpi pair -- those are absorbed by scf.for. The
      // matchCountedDoWhile invariant guarantees the addi only feeds
      // the cmpi/condition and is not used elsewhere.
      if (&op == info.addOp.getOperation() ||
          &op == info.cmpOp.getOperation())
        continue;
      rewriter.clone(op, mapping);
    }

    // Build the scf.yield: condition op's args minus the bumped iv,
    // with each value remapped through the body clone.
    ::llvm::SmallVector<::mlir::Value, 4> yieldArgs;
    yieldArgs.reserve(loop.getInits().size() - 1);
    auto condArgs = loop.getConditionOp().getArgs();
    for (auto pair : ::llvm::enumerate(condArgs)) {
      if (pair.index() == info.ivIndex)
        continue;
      ::mlir::Value v = pair.value();
      if (auto mapped = mapping.lookupOrNull(v))
        v = mapped;
      yieldArgs.push_back(v);
    }
    ::mlir::scf::YieldOp::create(rewriter, loc, yieldArgs);

    // Replace the original scf.while results with the scf.for results
    // plus a synthesized final iv value (we approximate it as the
    // upper bound, which is the conventional scf.for exit value).
    ::llvm::SmallVector<::mlir::Value, 4> whileReplacements;
    whileReplacements.reserve(loop->getNumResults());
    auto forResults = forOp.getResults();
    unsigned forResIdx = 0;
    for (unsigned i = 0; i < loop->getNumResults(); ++i) {
      if (i == info.ivIndex) {
        whileReplacements.push_back(ub);
      } else {
        whileReplacements.push_back(forResults[forResIdx++]);
      }
    }
    rewriter.replaceOp(loop, whileReplacements);
    return ::mlir::success();
  }
};

struct SCFWhileToForPass
    : public ::mlir::PassWrapper<SCFWhileToForPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCFWhileToForPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-scf-while-to-for";
  }
  ::llvm::StringRef getDescription() const final {
    return "Uplift counted scf.while loops into scf.for, supporting both "
           "the upstream UpliftWhileToFor and the do-while-counted shape "
           "produced by --lift-cf-to-scf on raised LLVM IR.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::MLIRContext *ctx = &getContext();

    ::mlir::RewritePatternSet patterns(ctx);
    ::mlir::scf::populateUpliftWhileToForPatterns(patterns);
    patterns.add<UpliftDoWhileToFor>(ctx);

    if (failed(::mlir::applyPatternsGreedily(module, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace loom {
namespace raising {

std::unique_ptr<::mlir::Pass> createSCFWhileToForPass() {
  return std::make_unique<SCFWhileToForPass>();
}

void registerSCFWhileToForPass() {
  static bool once = []() {
    ::mlir::PassRegistration<SCFWhileToForPass>();
    return true;
  }();
  (void)once;
}

} // namespace raising
} // namespace loom

// Uplift a counted scf.while loop into scf.for, using the upstream
// `scf::upliftWhileToForLoop` utility.
//
// The upstream utility recognizes the pre-tested counted-loop shape -- a
// `before` block whose only comparison is an arith.cmpi slt/sgt against a
// loop-invariant bound, with the induction bump in the `after` block. With
// a proven positive step that shape has the standard scf.for trip count,
// so its termination is structurally equivalent to scf.for and needs no
// loop-semantics analysis.
//
// The utility is unsafe in one respect: it reconstructs the loop's exit
// induction value from the trip count rather than forwarding the exact value
// the failed scf.condition carried. That reconstruction can disagree with the
// observed loop result. We therefore gate the utility on the loop having no
// external users: when no result escapes the loop, the reconstructed exit
// value is unobservable and the rewrite is exact.
//
// It accepts a second thing we must gate: a loop-invariant induction step
// of unproven sign. For %lb < %ub and step zero the source while does not
// terminate while the generated scf.for violates its semantic contract, and
// a negative or runtime-unknown step makes the two trip counts disagree.
// The rewrite is therefore offered only for a statically proven positive
// constant step; `hasProvenPositiveConstantStep` extracts just enough of
// the accepted shape -- the yielded induction add and its step operand --
// to prove it, and every other structural check stays with the utility.
//
// The post-tested (do-while) shape that CFG-to-SCF structuring emits for a
// counted loop -- increment and comparison in the `before` block, condition
// on the bumped induction value -- is NOT uplifted. Its body runs at least
// once even when %lb >= %ub, it can fail to terminate, and the failed
// condition forwards an induction value that overshoots the bound unless the
// step lands on it exactly. Proving trip-count and exit-value equivalence
// for that shape needs a loop-semantics analysis this mechanical pass does
// not own, so those loops are left as legal scf.while.

#include "Frontend/Raising/Passes.h"

#include "CallableRegions.h"
#include "ExactRewrite.h"
#include "PreservedHints.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace {

using loom::raising::carryLoopAnnotation;
using loom::raising::loopAnnotationName;

// Prove the induction step of the pre-tested counted shape is a positive
// constant. The before block's condition comparison must test a before
// argument (the induction candidate); the after argument at the position
// that candidate occupies among the scf.condition operands is its
// after-side value; and the after yield at the candidate's own argument
// index must add a constant to that value. Uniqueness and every other
// structural check remain the utility's; anything unprovable fails closed
// and is never offered to it.
bool hasProvenPositiveConstantStep(::mlir::scf::WhileOp loop) {
  ::mlir::Block *beforeBody = loop.getBeforeBody();
  ::mlir::scf::ConditionOp condition = loop.getConditionOp();
  auto cmp = condition.getCondition().getDefiningOp<::mlir::arith::CmpIOp>();
  if (!cmp || cmp->getParentRegion() != &loop.getBefore())
    return false;
  ::mlir::BlockArgument ivArg;
  for (::mlir::Value operand : {cmp.getLhs(), cmp.getRhs()}) {
    auto arg = ::mlir::dyn_cast<::mlir::BlockArgument>(operand);
    if (arg && arg.getOwner() == beforeBody) {
      ivArg = arg;
      break;
    }
  }
  if (!ivArg)
    return false;
  ::mlir::BlockArgument afterIv;
  for (auto [index, forwarded] : ::llvm::enumerate(condition.getArgs())) {
    if (forwarded == ivArg) {
      afterIv = loop.getAfterBody()->getArgument(index);
      break;
    }
  }
  if (!afterIv)
    return false;
  unsigned ivIdx = ivArg.getArgNumber();
  ::mlir::scf::YieldOp yield = loop.getYieldOp();
  if (ivIdx >= yield.getResults().size())
    return false;
  auto add = yield.getResults()[ivIdx].getDefiningOp<::mlir::arith::AddIOp>();
  if (!add)
    return false;
  ::mlir::Value step;
  if (add.getLhs() == afterIv) {
    step = add.getRhs();
  } else if (add.getRhs() == afterIv) {
    step = add.getLhs();
  } else {
    return false;
  }
  auto constant = step.getDefiningOp<::mlir::arith::ConstantOp>();
  if (!constant)
    return false;
  auto intValue = ::mlir::dyn_cast<::mlir::IntegerAttr>(constant.getValue());
  return intValue && intValue.getValue().isStrictlyPositive();
}

// Uplift a pre-tested counted while loop with the upstream utility, keeping
// the loop annotation on the loop the utility creates. The utility
// reconstructs the exit induction value instead of forwarding the exact
// failed-condition value, so the rewrite is only offered when the loop has
// no external users -- i.e. no result whose reconstructed value could be
// observed. It also accepts a step of unproven sign, so the rewrite is
// further gated on a statically proven positive constant induction step.
struct UpliftCountedWhileToFor
    : public ::mlir::OpRewritePattern<::mlir::scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::scf::WhileOp loop,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (!loop->use_empty())
      return ::mlir::failure();
    if (!hasProvenPositiveConstantStep(loop))
      return ::mlir::failure();
    ::mlir::Attribute annotation = loop->getAttr(loopAnnotationName);
    ::mlir::FailureOr<::mlir::scf::ForOp> uplifted =
        ::mlir::scf::upliftWhileToForLoop(rewriter, loop);
    if (failed(uplifted))
      return ::mlir::failure();
    carryLoopAnnotation(annotation, *uplifted);
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
    return "Uplift counted scf.while loops into scf.for when the upstream "
           "utility proves a structurally equivalent trip count, the "
           "induction step is a proven positive constant, and no loop "
           "result is observable.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::MLIRContext *ctx = &getContext();

    ::mlir::RewritePatternSet patterns(ctx);
    patterns.add<UpliftCountedWhileToFor>(ctx);
    ::mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    (void)loom::raising::forEachCallableRegion(
        module, [&](::mlir::Region &region) {
          loom::raising::applyExactPatternsOnce(region, frozen);
          return ::mlir::success();
        });
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

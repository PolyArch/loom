#include "Frontend/Raising/Passes.h"
#include "Frontend/Analysis/CallableRegions.h"
#include "Frontend/Lowering/LoopIndependence.h"
#include "ExactRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace {

// Materialise an Index-typed value from `v`. If `v` is already index,
// return it unchanged; otherwise insert an arith.index_cast.
::mlir::Value toIndex(::mlir::OpBuilder &builder, ::mlir::Location loc,
                      ::mlir::Value v) {
  if (::mlir::isa<::mlir::IndexType>(v.getType()))
    return v;
  return ::mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getIndexType(), v);
}

// Rewrite a parallel scf.for into scf.forall.
struct ForToForall : public ::mlir::OpRewritePattern<::mlir::scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::scf::ForOp loop,
                  ::mlir::PatternRewriter &rewriter) const override {
    // No iter_args: pure parallel-shape only.
    if (!loop.getInitArgs().empty())
      return ::mlir::failure();

    // Bounds and step must be loop-invariant.
    ::mlir::Value lb = loop.getLowerBound();
    ::mlir::Value ub = loop.getUpperBound();
    ::mlir::Value step = loop.getStep();

    // The bound/step types must be representable as Index. arith
    // requires lhs/rhs share types; since for already enforces this,
    // checking lb is enough.
    ::mlir::Type ivType = lb.getType();
    if (!::mlir::isa<::mlir::IndexType, ::mlir::IntegerType>(ivType))
      return ::mlir::failure();
    // The typed proof owner covers both ordinary affine-style bodies and the
    // closed strip-mined point-domain spelling. Rewriting is mechanical once
    // that owner establishes independence on this exact clone.
    if (loom::lowering::proveIndependentIterations(loop) !=
        loom::lowering::ParallelDependenceResult::ProvenIndependent)
      return ::mlir::failure();

    ::mlir::Location loc = loop.getLoc();
    ::mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(loop);

    // Cast bounds/step to index for scf.forall.
    ::mlir::Value lbIdx = toIndex(rewriter, loc, lb);
    ::mlir::Value ubIdx = toIndex(rewriter, loc, ub);
    ::mlir::Value stepIdx = toIndex(rewriter, loc, step);

    ::llvm::SmallVector<::mlir::OpFoldResult, 1> lbs{lbIdx};
    ::llvm::SmallVector<::mlir::OpFoldResult, 1> ubs{ubIdx};
    ::llvm::SmallVector<::mlir::OpFoldResult, 1> steps{stepIdx};

    auto forallOp = ::mlir::scf::ForallOp::create(
        rewriter, loc, lbs, ubs, steps, /*outputs=*/::mlir::ValueRange{},
        /*mapping=*/std::nullopt);

    // The block builder used by ForallOp::create only sets up a body
    // block with the iv argument; we need to migrate the original
    // body into it. forallOp.getBody() has rank-many iv arguments
    // followed by the shared-out arguments (none here).
    ::mlir::Block *forallBody = forallOp.getBody();
    ::mlir::Value forallIv = forallOp.getInductionVar(0);

    ::mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
    rewriter.setInsertionPointToStart(forallBody);
    ::mlir::Value ivAsOriginal = forallIv;
    if (forallIv.getType() != ivType) {
      ivAsOriginal =
          ::mlir::arith::IndexCastOp::create(rewriter, loc, ivType, forallIv);
    }

    // Map the original loop's iv to the recovered integer iv. Other
    // captures (values defined outside the loop) flow through unchanged
    // since the body is moved, not cloned.
    ::mlir::Block *origBody = loop.getBody();
    ::mlir::IRMapping mapping;
    mapping.map(loop.getInductionVar(), ivAsOriginal);

    // Clone every op except the trailing scf.yield (which had no
    // operands since iter_args is empty); the in_parallel terminator
    // already exists in the new body.
    for (::mlir::Operation &op : origBody->without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // Remove the original scf.for; it had no results.
    rewriter.eraseOp(loop);
    return ::mlir::success();
  }
};

struct SCFForToForallPass
    : public ::mlir::PassWrapper<SCFForToForallPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SCFForToForallPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-scf-for-to-forall";
  }
  ::llvm::StringRef getDescription() const final {
    return "Lift trivially parallel scf.for loops (no iter_args, "
           "syntactic affine-style stores, disjoint read / write bases) "
           "into scf.forall so downstream lowerings can see parallel "
           "intent natively.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::MLIRContext *ctx = &getContext();

    ::mlir::RewritePatternSet patterns(ctx);
    patterns.add<ForToForall>(ctx);
    ::mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    (void)loom::frontend::analysis::forEachCallableRegion(
        module, [&](::mlir::Region &region) {
          loom::raising::applyExactPatternsOnce(region, frozen);
          return ::mlir::success();
        });
  }
};

} // namespace

namespace loom {
namespace raising {

::mlir::LogicalResult
materializeIndependentLoopAsForall(::mlir::scf::ForOp loop) {
  ::mlir::PatternRewriter rewriter(loop.getContext());
  return ForToForall(loop.getContext()).matchAndRewrite(loop, rewriter);
}

std::unique_ptr<::mlir::Pass> createSCFForToForallPass() {
  return std::make_unique<SCFForToForallPass>();
}

void registerSCFForToForallPass() {
  static bool once = []() {
    ::mlir::PassRegistration<SCFForToForallPass>();
    return true;
  }();
  (void)once;
}

} // namespace raising
} // namespace loom

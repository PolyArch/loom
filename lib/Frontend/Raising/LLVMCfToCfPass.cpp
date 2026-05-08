// Convert llvm.br / llvm.cond_br terminators into cf.br / cf.cond_br so
// the upstream --lift-cf-to-scf pass (which only operates on the cf
// dialect, see ControlFlowToSCF.cpp:isa<cf::CondBranchOp, cf::SwitchOp>)
// can subsequently lift the body of each function into structured SCF
// form. llvm.return is intentionally preserved -- it is the
// function-body terminator and the func-to-func pass replaces it after
// the function shape itself has been raised.

#include "Frontend/Raising/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

struct LLVMBrToCfBr : public ::mlir::OpRewritePattern<::mlir::LLVM::BrOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::BrOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::cf::BranchOp>(op, op.getDest(),
                                                     op.getDestOperands());
    return ::mlir::success();
  }
};

struct LLVMCondBrToCfCondBr
    : public ::mlir::OpRewritePattern<::mlir::LLVM::CondBrOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::CondBrOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::cf::CondBranchOp>(
        op, op.getCondition(), op.getTrueDest(), op.getTrueDestOperands(),
        op.getFalseDest(), op.getFalseDestOperands());
    return ::mlir::success();
  }
};

struct LLVMCfToCfPass
    : public ::mlir::PassWrapper<LLVMCfToCfPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMCfToCfPass)

  ::llvm::StringRef getArgument() const final { return "loom-llvm-cf-to-cf"; }
  ::llvm::StringRef getDescription() const final {
    return "Rewrite llvm.br / llvm.cond_br inside llvm.func bodies into "
           "cf.br / cf.cond_br so --lift-cf-to-scf can structure the CFG.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::cf::ControlFlowDialect,
                    ::mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::MLIRContext *ctx = &getContext();

    ::mlir::RewritePatternSet patterns(ctx);
    patterns.add<LLVMBrToCfBr, LLVMCondBrToCfCondBr>(ctx);
    ::mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    // Apply patterns inside every llvm.func body. Apply per-function so we
    // do not accidentally mutate ops outside an llvm.func (the upstream
    // patterns rewrite only the matched op, but limiting to function bodies
    // keeps the pass intentful).
    auto walkResult = module.walk([&](::mlir::LLVM::LLVMFuncOp funcOp) {
      if (funcOp.getBody().empty())
        return ::mlir::WalkResult::advance();
      if (failed(::mlir::applyPatternsGreedily(funcOp.getBody(), frozen)))
        return ::mlir::WalkResult::interrupt();
      return ::mlir::WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace

namespace loom {
namespace raising {

std::unique_ptr<::mlir::Pass> createLLVMCfToCfPass() {
  return std::make_unique<LLVMCfToCfPass>();
}

} // namespace raising
} // namespace loom

namespace loom {
namespace raising {

namespace {
struct LLVMCfToCfRegistration {
  LLVMCfToCfRegistration() {
    ::mlir::PassRegistration<LLVMCfToCfPass>();
  }
};
} // namespace

// Hook used by registerRaisingPasses() in Pipeline.cpp.
void registerLLVMCfToCfPass() {
  static LLVMCfToCfRegistration once;
  (void)once;
}

} // namespace raising
} // namespace loom

// Convert llvm.br / llvm.cond_br / llvm.switch terminators into the
// matching cf.br / cf.cond_br / cf.switch ops so the upstream
// --lift-cf-to-scf pass (which only operates on the cf dialect, see
// ControlFlowToSCF.cpp) can subsequently lift the body of each function
// into structured SCF form. llvm.return is intentionally preserved --
// it is the function-body terminator and the func-to-func pass replaces
// it BEFORE this pass runs (see Pipeline.cpp; func-to-func now runs
// first, then this pass nested under func::FuncOp).
//
// This pass runs as a function-level pass strictly under func.func.
// Aggregate-signature llvm.func ops are skipped by func-to-func and
// stay fully llvm-shaped; this pass never touches them, which keeps
// the multi-dialect raising contract clean (raised callers may still
// llvm.call into unraised aggregate callees).

#include "Frontend/Raising/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

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

// Convert llvm.switch into cf.switch. The two ops carry the same
// payload (a flag value, a default destination + operands, and a list
// of (caseValue, caseDest, caseOperands) triples). The case values are
// stored as a DenseIntElementsAttr in both ops; we copy them through
// directly.
struct LLVMSwitchToCfSwitch
    : public ::mlir::OpRewritePattern<::mlir::LLVM::SwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::SwitchOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    ::mlir::Block *defaultDest = op.getDefaultDestination();
    ::mlir::ValueRange defaultOperands = op.getDefaultOperands();
    ::mlir::DenseIntElementsAttr caseValuesAttr =
        op.getCaseValuesAttr();

    ::llvm::SmallVector<::mlir::Block *, 4> caseDests;
    for (::mlir::Block *dest : op.getCaseDestinations())
      caseDests.push_back(dest);

    ::llvm::SmallVector<::mlir::ValueRange, 4> caseOperands;
    for (auto operands : op.getCaseOperands())
      caseOperands.push_back(operands);

    rewriter.replaceOpWithNewOp<::mlir::cf::SwitchOp>(
        op, op.getValue(), defaultDest, defaultOperands, caseValuesAttr,
        caseDests, caseOperands);
    return ::mlir::success();
  }
};

struct LLVMCfToCfPass
    : public ::mlir::PassWrapper<
          LLVMCfToCfPass,
          ::mlir::OperationPass<::mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMCfToCfPass)

  ::llvm::StringRef getArgument() const final { return "loom-llvm-cf-to-cf"; }
  ::llvm::StringRef getDescription() const final {
    return "Rewrite llvm.br / llvm.cond_br / llvm.switch inside func.func "
           "bodies into cf.br / cf.cond_br / cf.switch so --lift-cf-to-scf "
           "can structure the CFG. Skipped (aggregate-signature) llvm.func "
           "ops are left untouched.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::cf::ControlFlowDialect,
                    ::mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() final {
    ::mlir::func::FuncOp funcOp = getOperation();
    if (funcOp.isExternal())
      return;
    ::mlir::MLIRContext *ctx = &getContext();
    ::mlir::RewritePatternSet patterns(ctx);
    patterns.add<LLVMBrToCfBr, LLVMCondBrToCfCondBr,
                 LLVMSwitchToCfSwitch>(ctx);
    if (failed(::mlir::applyPatternsGreedily(funcOp.getBody(),
                                             std::move(patterns))))
      signalPassFailure();
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

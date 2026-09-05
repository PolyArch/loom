// Convert llvm.br / llvm.cond_br / llvm.switch terminators into the
// matching cf.br / cf.cond_br / cf.switch ops so the upstream region-level
// CFG-to-SCF transformation (which recognizes cf branch structure, see
// ControlFlowToSCF.cpp) can subsequently structure the callable body.
//
// llvm.return is intentionally preserved: it is the return of an imported
// LLVM callable, which stays the sole owner of its ABI envelope. The
// transformation treats it as an ordinary return-like exit.
//
// A branch also carries imported hints. Branch weights move to cf.cond_br's
// own branch_weights attribute, and a loop annotation stays on the replacing
// branch under the preserved-hint carrier until the structured loop that owns
// the cycle can take it. cf.switch has no weight carrier, so a weighted
// llvm.switch keeps its LLVM form and its weights: preserving the operation
// that owns the hint is the exact disposition, while respelling it would
// silently drop imported profile data and rejecting it would make an
// unselected weighted branch fail ordinary module compilation.
//
// The rewrite is scoped to callable regions so that constant regions such as
// an llvm.mlir.global initializer, which must stay expressible as an LLVM
// constant, are never rewritten. Each declared pattern is offered once per
// operation and nothing else runs: no folding, no constant CSE, no dead-code
// or unreachable-block removal. An unreachable block stays until the
// structuring pass removes it explicitly.

#include "Frontend/Raising/Passes.h"

#include "Frontend/Analysis/CallableRegions.h"
#include "ExactRewrite.h"
#include "PreservedHints.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace {

using loom::raising::carryCandidateLoopHint;
using loom::raising::carryLoopAnnotation;

void carryHints(::mlir::Operation *target, ::mlir::Attribute annotation,
                ::mlir::Attribute candidate) {
  carryLoopAnnotation(annotation, target);
  carryCandidateLoopHint(candidate, target);
}

struct LLVMBrToCfBr : public ::mlir::OpRewritePattern<::mlir::LLVM::BrOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::BrOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    ::mlir::Attribute annotation = op.getLoopAnnotationAttr();
    ::mlir::Attribute candidate =
        op->getAttr(loom::raising::candidateLoopHintName);
    auto replacement = rewriter.replaceOpWithNewOp<::mlir::cf::BranchOp>(
        op, op.getDest(), op.getDestOperands());
    carryHints(replacement, annotation, candidate);
    return ::mlir::success();
  }
};

struct LLVMCondBrToCfCondBr
    : public ::mlir::OpRewritePattern<::mlir::LLVM::CondBrOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::CondBrOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    ::mlir::Attribute annotation = op.getLoopAnnotationAttr();
    ::mlir::Attribute candidate =
        op->getAttr(loom::raising::candidateLoopHintName);
    ::mlir::DenseI32ArrayAttr weights = op.getBranchWeightsAttr();
    auto replacement = rewriter.replaceOpWithNewOp<::mlir::cf::CondBranchOp>(
        op, op.getCondition(), op.getTrueDest(), op.getTrueDestOperands(),
        op.getFalseDest(), op.getFalseDestOperands());
    if (weights)
      replacement.setBranchWeightsAttr(weights);
    carryHints(replacement, annotation, candidate);
    return ::mlir::success();
  }
};

// Convert llvm.switch into cf.switch. The two ops carry the same payload: a
// flag value, a default destination and its operands, and a list of case
// value, destination and operand triples. Both store the case values as one
// DenseIntElementsAttr, so it is copied through directly.
struct LLVMSwitchToCfSwitch
    : public ::mlir::OpRewritePattern<::mlir::LLVM::SwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::LLVM::SwitchOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // cf.switch has no branch-weight carrier, so respelling a weighted switch
    // would lose the imported profile. The switch keeps its LLVM form instead.
    if (op.getBranchWeightsAttr())
      return ::mlir::failure();

    ::mlir::Block *defaultDest = op.getDefaultDestination();
    ::mlir::ValueRange defaultOperands = op.getDefaultOperands();
    ::mlir::DenseIntElementsAttr caseValuesAttr = op.getCaseValuesAttr();

    ::llvm::SmallVector<::mlir::Block *, 4> caseDests;
    for (::mlir::Block *dest : op.getCaseDestinations())
      caseDests.push_back(dest);

    ::llvm::SmallVector<::mlir::ValueRange, 4> caseOperands;
    for (auto operands : op.getCaseOperands())
      caseOperands.push_back(operands);

    // llvm.switch has no loop-annotation property, so a hint on it can only
    // have arrived through the preserved-hint carrier; move it along with the
    // branch it describes.
    ::mlir::Attribute annotation =
        op->getAttr(loom::raising::loopAnnotationName);
    ::mlir::Attribute candidate =
        op->getAttr(loom::raising::candidateLoopHintName);
    auto replacement = rewriter.replaceOpWithNewOp<::mlir::cf::SwitchOp>(
        op, op.getValue(), defaultDest, defaultOperands, caseValuesAttr,
        caseDests, caseOperands);
    carryHints(replacement, annotation, candidate);
    return ::mlir::success();
  }
};

struct LLVMCfToCfPass
    : public ::mlir::PassWrapper<LLVMCfToCfPass, ::mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMCfToCfPass)

  ::llvm::StringRef getArgument() const final { return "loom-llvm-cf-to-cf"; }
  ::llvm::StringRef getDescription() const final {
    return "Rewrite llvm.br / llvm.cond_br / llvm.switch inside callable "
           "regions into cf.br / cf.cond_br / cf.switch, carrying imported "
           "branch weights and loop annotations, so the CFG-to-SCF "
           "transformation can structure them.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry
        .insert<::mlir::cf::ControlFlowDialect, ::mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() final {
    ::mlir::MLIRContext *ctx = &getContext();
    ::mlir::RewritePatternSet patterns(ctx);
    patterns.add<LLVMBrToCfBr, LLVMCondBrToCfCondBr, LLVMSwitchToCfSwitch>(ctx);
    ::mlir::FrozenRewritePatternSet frozen(std::move(patterns));

    (void)loom::frontend::analysis::forEachCallableRegion(
        getOperation(), [&](::mlir::Region &region) {
          loom::raising::applyExactPatternsOnce(region, frozen);
          return ::mlir::success();
        });
  }
};

} // namespace

namespace loom {
namespace raising {

std::unique_ptr<::mlir::Pass> createLLVMCfToCfPass() {
  return std::make_unique<LLVMCfToCfPass>();
}

void registerLLVMCfToCfPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LLVMCfToCfPass>();
    return true;
  }();
  (void)once;
}

} // namespace raising
} // namespace loom

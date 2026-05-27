// Raise llvm.func declarations / definitions whose signatures consist
// only of MLIR-native types (builtin integers, floats, index, !llvm.ptr)
// into matching func.func ops. Calls to those functions inside other
// (also raised) llvm.func bodies are rewritten to func.call.
//
// Design notes:
//   * Variadic functions are skipped (func.func has no variadic form).
//   * Functions whose signature contains LLVM aggregates (struct, array,
//     non-builtin vector) are skipped; they remain as llvm.func.
//   * Functions returning !llvm.void are mapped to a no-result func.func.
//   * llvm.return inside a raised function body becomes func.return on
//     the corresponding block.
//   * llvm.call to a raised callee becomes func.call (only for direct
//     calls; indirect llvm.call ops are left alone).
//
// Contract on skipped callees / mixed islands:
//   If a callee is SKIPPED (e.g. it has an aggregate signature), the
//   raised callers KEEP their existing `llvm.call @callee(...)` op
//   pointing at that unraised `llvm.func @callee`. This is allowed MLIR
//   -- a `func.func` body may host `llvm.call` ops as long as the
//   referenced symbol still resolves to an `llvm.func`. Callers should
//   expect this multi-dialect island shape; do not attempt to "fix" it
//   by re-lowering raised callers, because doing so loses parallel /
//   structured control-flow information already recovered by the
//   subsequent --lift-cf-to-scf and arith-to-arith passes.
//
// Pipeline ordering:
//   This pass runs FIRST in the raising pipeline (see Pipeline.cpp).
//   The cf-to-cf and arith-to-arith passes that follow are nested under
//   func.func, so SKIPPED `llvm.func` ops keep their bodies in pristine
//   LLVM form (no half-rewritten cf.br + llvm.return mixed shape).

#include "Frontend/Raising/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace {

// Build-up helpers ----------------------------------------------------

bool isRaiseFriendlyType(::mlir::Type t) {
  if (::mlir::isa<::mlir::IntegerType, ::mlir::IndexType, ::mlir::FloatType>(t))
    return true;
  if (::mlir::isa<::mlir::LLVM::LLVMPointerType>(t))
    return true;
  return false;
}

bool functionSignatureIsRaiseFriendly(::mlir::LLVM::LLVMFuncOp funcOp) {
  if (funcOp.isVarArg())
    return false;
  for (::mlir::Type t : funcOp.getArgumentTypes()) {
    if (!isRaiseFriendlyType(t))
      return false;
  }
  ::mlir::ArrayRef<::mlir::Type> resultTypes = funcOp.getResultTypes();
  for (::mlir::Type t : resultTypes) {
    if (!isRaiseFriendlyType(t))
      return false;
  }
  return true;
}

// Build a builtin FunctionType from an LLVMFuncOp's signature. Caller
// has already checked functionSignatureIsRaiseFriendly.
::mlir::FunctionType buildBuiltinFunctionType(::mlir::LLVM::LLVMFuncOp funcOp) {
  return ::mlir::FunctionType::get(funcOp.getContext(),
                                   funcOp.getArgumentTypes(),
                                   funcOp.getResultTypes());
}

// Rewrite each llvm.return op inside `region` into func.return. The
// LLVM return has VariadicOperands but its accessor getArg() returns a
// nullable TypedValue, so we must use the underlying operand range to
// avoid passing a null Value into func.return's ValueRange constructor.
void rewriteReturns(::mlir::Region &region) {
  ::llvm::SmallVector<::mlir::LLVM::ReturnOp, 4> returns;
  region.walk([&](::mlir::LLVM::ReturnOp op) { returns.push_back(op); });
  for (::mlir::LLVM::ReturnOp op : returns) {
    ::mlir::OpBuilder b(op);
    ::mlir::func::ReturnOp::create(b, op.getLoc(), op.getOperands());
    op.erase();
  }
}

// Rewrite direct llvm.call to a raised callee into func.call.
void rewriteCalls(::mlir::Region &region,
                  const ::llvm::DenseMap<::llvm::StringRef, ::mlir::func::FuncOp>
                      &raised) {
  ::llvm::SmallVector<::mlir::LLVM::CallOp, 4> calls;
  region.walk([&](::mlir::LLVM::CallOp op) { calls.push_back(op); });
  for (::mlir::LLVM::CallOp op : calls) {
    auto callee = op.getCallee();
    if (!callee.has_value())
      continue;
    auto it = raised.find(*callee);
    if (it == raised.end())
      continue;

    ::mlir::func::FuncOp target = it->second;
    auto fnType = target.getFunctionType();
    auto argOperands = op.getArgOperands();
    if (argOperands.size() != fnType.getNumInputs())
      continue;
    bool typesMismatch = false;
    for (auto pair : ::llvm::zip(argOperands, fnType.getInputs())) {
      if (std::get<0>(pair).getType() != std::get<1>(pair)) {
        typesMismatch = true;
        break;
      }
    }
    if (typesMismatch)
      continue;
    if (op->getNumResults() != fnType.getNumResults())
      continue;
    if (op->getNumResults() == 1 &&
        op->getResult(0).getType() != fnType.getResult(0))
      continue;

    ::mlir::OpBuilder b(op);
    auto newCall = ::mlir::func::CallOp::create(
        b, op.getLoc(), target, argOperands);
    if (op->getNumResults() == 1)
      op->getResult(0).replaceAllUsesWith(newCall.getResult(0));
    op.erase();
  }
}

struct LLVMFuncToFuncPass
    : public ::mlir::PassWrapper<LLVMFuncToFuncPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMFuncToFuncPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-llvm-func-to-func";
  }
  ::llvm::StringRef getDescription() const final {
    return "Raise llvm.func ops with builtin / pointer signatures into "
           "func.func; rewrite llvm.return and direct llvm.call.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::func::FuncDialect, ::mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();

    // Phase 1: collect raise-friendly llvm.func ops.
    ::llvm::SmallVector<::mlir::LLVM::LLVMFuncOp, 8> candidates;
    module.walk([&](::mlir::LLVM::LLVMFuncOp funcOp) {
      if (!functionSignatureIsRaiseFriendly(funcOp))
        return;
      // Skip declarations -- there is nothing to move and the original
      // llvm.func is needed by remaining llvm.call sites that do not get
      // rewritten.
      if (funcOp.getBody().empty())
        return;
      candidates.push_back(funcOp);
    });

    // Phase 2: build the func.func shells with empty bodies, registered
    // by name so the call rewriter can find them.
    ::llvm::DenseMap<::llvm::StringRef, ::mlir::func::FuncOp> raised;
    raised.reserve(candidates.size());
    ::llvm::SmallVector<std::pair<::mlir::LLVM::LLVMFuncOp,
                                  ::mlir::func::FuncOp>,
                        8>
        pairs;
    pairs.reserve(candidates.size());

    ::mlir::OpBuilder builder(module.getBodyRegion());
    for (::mlir::LLVM::LLVMFuncOp funcOp : candidates) {
      auto fnType = buildBuiltinFunctionType(funcOp);
      builder.setInsertionPoint(funcOp);
      auto newFunc = ::mlir::func::FuncOp::create(
          builder, funcOp.getLoc(), funcOp.getSymName(), fnType);
      // Mark internal-linkage functions as private so the symbol does
      // not become an exported ABI symbol after the raise.
      if (funcOp.getLinkage() == ::mlir::LLVM::Linkage::Internal ||
          funcOp.getLinkage() == ::mlir::LLVM::Linkage::Private) {
        newFunc.setPrivate();
      }
      raised.insert({funcOp.getSymName(), newFunc});
      pairs.emplace_back(funcOp, newFunc);
    }

    // Phase 3: move the body region from each llvm.func into its
    // matching func.func, then rewrite the terminator + direct calls.
    for (auto &kv : pairs) {
      ::mlir::LLVM::LLVMFuncOp src = kv.first;
      ::mlir::func::FuncOp dst = kv.second;
      dst.getBody().takeBody(src.getBody());
      rewriteReturns(dst.getBody());
    }

    // Phase 4: rewrite calls inside every raised function body now that
    // every shell exists. Calls that point at unraised callees stay as
    // llvm.call -- the spec allows the multi-dialect output.
    for (auto &kv : pairs) {
      rewriteCalls(kv.second.getBody(), raised);
    }

    // Phase 5: erase the original llvm.func ops.
    for (auto &kv : pairs)
      kv.first.erase();
  }
};

} // namespace

namespace loom {
namespace raising {

std::unique_ptr<::mlir::Pass> createLLVMFuncToFuncPass() {
  return std::make_unique<LLVMFuncToFuncPass>();
}

void registerLLVMFuncToFuncPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LLVMFuncToFuncPass>();
    return true;
  }();
  (void)once;
}

} // namespace raising
} // namespace loom

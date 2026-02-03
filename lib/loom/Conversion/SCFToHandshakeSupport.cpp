//===- SCFToHandshakeSupport.cpp - Shared helpers ----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/SCFToHandshakeImpl.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

namespace loom {
namespace detail {
namespace {

static bool hasLoomAnnotation(mlir::Operation *op, llvm::StringRef prefix) {
  if (!op)
    return false;
  auto attr = op->getAttrOfType<mlir::ArrayAttr>("loom.annotations");
  if (!attr)
    return false;
  for (mlir::Attribute entry : attr) {
    auto str = mlir::dyn_cast<mlir::StringAttr>(entry);
    if (!str)
      continue;
    if (str.getValue().starts_with(prefix))
      return true;
  }
  return false;
}

} // namespace

bool isAccelFunc(mlir::func::FuncOp func) {
  return func && hasLoomAnnotation(func, "loom.accel");
}

mlir::Value getMemrefRoot(mlir::Value memref) {
  if (!memref)
    return memref;
  while (true) {
    if (auto castOp = memref.getDefiningOp<mlir::memref::CastOp>()) {
      memref = castOp.getSource();
      continue;
    }
    if (auto subviewOp = memref.getDefiningOp<mlir::memref::SubViewOp>()) {
      memref = subviewOp.getSource();
      continue;
    }
    if (auto viewOp = memref.getDefiningOp<mlir::memref::ViewOp>()) {
      memref = viewOp.getSource();
      continue;
    }
    if (auto reinterpretOp =
            memref.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
      memref = reinterpretOp.getSource();
      continue;
    }
    if (auto collapseOp =
            memref.getDefiningOp<mlir::memref::CollapseShapeOp>()) {
      memref = collapseOp.getSrc();
      continue;
    }
    if (auto expandOp = memref.getDefiningOp<mlir::memref::ExpandShapeOp>()) {
      memref = expandOp.getSrc();
      continue;
    }
    break;
  }
  return memref;
}

mlir::LogicalResult inlineCallsInAccel(mlir::func::FuncOp func,
                                       mlir::SymbolTable &symbols) {
  bool changed = true;
  while (changed) {
    changed = false;
    mlir::WalkResult result = func.walk([&](mlir::func::CallOp call) {
      auto callee = symbols.lookup<mlir::func::FuncOp>(call.getCallee());
      if (!callee || callee.isExternal()) {
        call.emitError("cannot inline call inside accel function");
        return mlir::WalkResult::interrupt();
      }
      if (callee == func) {
        call.emitError("recursive call inside accel function");
        return mlir::WalkResult::interrupt();
      }
      if (!callee.getBody().hasOneBlock()) {
        call.emitError("callee must have single-block body for inlining");
        return mlir::WalkResult::interrupt();
      }
      mlir::Block &calleeBlock = callee.getBody().front();
      auto *terminator = calleeBlock.getTerminator();
      auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(terminator);
      if (!ret) {
        call.emitError("callee missing func.return for inlining");
        return mlir::WalkResult::interrupt();
      }
      if (ret.getNumOperands() != call.getNumResults()) {
        call.emitError("callee return arity mismatch for inlining");
        return mlir::WalkResult::interrupt();
      }

      mlir::IRMapping mapping;
      for (auto [arg, operand] :
           llvm::zip(calleeBlock.getArguments(), call.getOperands())) {
        mapping.map(arg, operand);
      }

      mlir::OpBuilder builder(call);
      for (mlir::Operation &nested : calleeBlock.without_terminator())
        builder.clone(nested, mapping);

      for (unsigned i = 0, e = call.getNumResults(); i < e; ++i) {
        mlir::Value mapped = mapping.lookupOrDefault(ret.getOperand(i));
        call.getResult(i).replaceAllUsesWith(mapped);
      }
      call.erase();
      changed = true;
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted())
      return mlir::failure();
  }
  mlir::WalkResult remaining =
      func.walk([&](mlir::func::CallOp) { return mlir::WalkResult::interrupt(); });
  if (remaining.wasInterrupted()) {
    func.emitError("remaining call inside accel function after inlining");
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace detail
} // namespace loom

//===-- SCFToHandshakeSupport.cpp - Shared lowering helpers -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements shared helper functions for the SCF-to-Handshake
// lowering, including detection of accelerator-marked functions, memref root
// tracing through casts and views, and call inlining within accelerator
// functions.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/SCFToHandshakeImpl.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <limits>

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

MemTargetHint getMemTargetHint(mlir::Operation *op) {
  if (!op)
    return MemTargetHint::None;
  auto attr = op->getAttrOfType<mlir::ArrayAttr>("loom.annotations");
  if (!attr)
    return MemTargetHint::None;
  for (mlir::Attribute entry : attr) {
    auto str = mlir::dyn_cast<mlir::StringAttr>(entry);
    if (!str)
      continue;
    llvm::StringRef value = str.getValue();
    if (!value.starts_with("loom.target="))
      continue;
    llvm::StringRef target = value.drop_front(sizeof("loom.target=") - 1);
    if (target == "rom")
      return MemTargetHint::Rom;
    if (target == "extmemory")
      return MemTargetHint::Extmemory;
  }
  return MemTargetHint::None;
}

std::optional<int64_t> getStaticMemrefByteSize(mlir::MemRefType type) {
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  int64_t elementBytes = 0;
  mlir::Type elementType = type.getElementType();
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
    elementBytes = (intType.getWidth() + 7) / 8;
  } else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType)) {
    elementBytes = (floatType.getWidth() + 7) / 8;
  } else {
    return std::nullopt;
  }
  int64_t count = 1;
  for (int64_t dim : type.getShape()) {
    if (dim <= 0)
      return std::nullopt;
    if (count > (std::numeric_limits<int64_t>::max() / dim))
      return std::nullopt;
    count *= dim;
  }
  if (elementBytes <= 0)
    return std::nullopt;
  if (count > (std::numeric_limits<int64_t>::max() / elementBytes))
    return std::nullopt;
  return count * elementBytes;
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

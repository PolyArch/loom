#ifndef LOOM_LIB_FRONTEND_RAISING_CALLABLEREGIONS_H
#define LOOM_LIB_FRONTEND_RAISING_CALLABLEREGIONS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace loom {
namespace raising {

// True when `op` is a callable whose own region is the subject of a separate
// region-level raising decision. An S0 program contains exactly two callable
// kinds: an imported llvm.func and a genuinely standard-MLIR-native func.func.
// It is deliberately not every FunctionOpInterface operation, because a later
// ownership carrier such as a Dataflow definition is not an input to
// mechanical raising and its region is not something these passes can claim to
// structure exactly.
//
// Both kinds are callables, but only one is an imported LLVM ABI authority. A
// native func.func is a callable region raised like any other and is never an
// ABI envelope: a floating-point environment is read only when the nearest
// callable is an llvm.func, and an llvm.func is never copied into another
// dialect to obtain a pass wrapper. That leaves llvm.func the sole imported
// LLVM callable and ABI owner of its body.
inline bool isCallableOp(::mlir::Operation *op) {
  return ::mlir::isa<::mlir::LLVM::LLVMFuncOp, ::mlir::func::FuncOp>(op);
}

// Return the nearest callable that owns `op`, or null when `op` is outside
// every callable region. A nested callable cuts off ownership inherited from
// any callable above it.
inline ::mlir::Operation *getNearestCallableOp(::mlir::Operation *op) {
  for (::mlir::Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isCallableOp(parent))
      return parent;
  }
  return nullptr;
}

// Run `transform` on every non-empty region of every callable reachable from
// `root`, visiting nested callables before their ancestors and stopping at the
// first failure.
//
// Callable regions are the sole subject of mechanical raising. An imported
// llvm.func owns its body and its complete ABI envelope, so raising rewrites
// that body where it stands instead of copying the function into another
// dialect to obtain a pass wrapper. A region outside a callable, such as an
// llvm.mlir.global initializer, carries no recoverable control flow and must
// stay expressible as an LLVM constant, so it is never rewritten.
inline ::mlir::LogicalResult forEachCallableRegion(
    ::mlir::Operation *root,
    ::llvm::function_ref<::mlir::LogicalResult(::mlir::Region &)> transform) {
  ::mlir::WalkResult walked =
      root->walk<::mlir::WalkOrder::PostOrder>([&](::mlir::Operation *op) {
        if (!isCallableOp(op))
          return ::mlir::WalkResult::advance();
        for (::mlir::Region &region : op->getRegions()) {
          if (region.empty())
            continue;
          if (failed(transform(region)))
            return ::mlir::WalkResult::interrupt();
        }
        return ::mlir::WalkResult::advance();
      });
  return walked.wasInterrupted() ? ::mlir::failure() : ::mlir::success();
}

// Offer `visit` to every operation `region` owns, recursing into nested
// regions that belong to the same callable -- scf.for, scf.if, a graph
// region -- but stopping at any nested callable, whose own body this region
// must not claim to own.
//
// This is the operation-level half of callable ownership: a callable processes
// exactly the operations its nearest enclosing callable owns, and a nested
// callable's body is left to that callable's own region-level walk. Crossing
// into a nested callable here would visit its operations twice -- once
// descended into from the enclosing region and once from the callable's own
// walk -- so the nested callable is pruned instead. Pruning happens in
// pre-order: in a post-order walk a callable's body is visited before the
// callable itself, so the skip would arrive one descent too late.
inline ::mlir::WalkResult forEachOwnedOperation(
    ::mlir::Region &region,
    ::llvm::function_ref<::mlir::WalkResult(::mlir::Operation *)> visit) {
  return region.walk<::mlir::WalkOrder::PreOrder>(
      [&](::mlir::Operation *op) -> ::mlir::WalkResult {
        if (isCallableOp(op))
          return ::mlir::WalkResult::skip();
        return visit(op);
      });
}

} // namespace raising
} // namespace loom

#endif // LOOM_LIB_FRONTEND_RAISING_CALLABLEREGIONS_H

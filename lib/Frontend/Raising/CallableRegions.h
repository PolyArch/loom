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

// Run `transform` on every non-empty region of every callable reachable from
// `root`, stopping at the first failure.
//
// Callable regions are the sole subject of mechanical raising. An imported
// llvm.func owns its body and its complete ABI envelope, so raising rewrites
// that body where it stands instead of copying the function into another
// dialect to obtain a pass wrapper. A region outside a callable, such as an
// llvm.mlir.global initializer, carries no recoverable control flow and must
// stay expressible as an LLVM constant, so it is never rewritten.
//
// The visited set is exactly the two callable kinds an S0 program contains:
// an imported llvm.func and a genuinely standard-MLIR-native func.func. It is
// deliberately not every FunctionOpInterface operation, because a later
// ownership carrier such as a Dataflow definition is not an input to
// mechanical raising and its region is not something these passes can claim
// to structure exactly.
inline ::mlir::LogicalResult forEachCallableRegion(
    ::mlir::Operation *root,
    ::llvm::function_ref<::mlir::LogicalResult(::mlir::Region &)> transform) {
  ::mlir::WalkResult walked = root->walk([&](::mlir::Operation *op) {
    if (!::mlir::isa<::mlir::LLVM::LLVMFuncOp, ::mlir::func::FuncOp>(op))
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

} // namespace raising
} // namespace loom

#endif // LOOM_LIB_FRONTEND_RAISING_CALLABLEREGIONS_H

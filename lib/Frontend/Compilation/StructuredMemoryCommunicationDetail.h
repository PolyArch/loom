#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATIONDETAIL_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATIONDETAIL_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"

#include "llvm/Support/Error.h"

namespace loom::frontend::detail {

bool canPromoteOrderedBufferToChannel(mlir::memref::AllocOp allocation);
bool canPromoteOrderedBufferToChannel(mlir::LLVM::AllocaOp allocation);

llvm::Error
promoteOrderedBufferToChannel(mlir::memref::AllocOp allocation,
                              mlir::Operation *&trackedSpatialRegion);
llvm::Error
promoteOrderedBufferToChannel(mlir::LLVM::AllocaOp allocation,
                              mlir::Operation *&trackedSpatialRegion);

} // namespace loom::frontend::detail

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATIONDETAIL_H

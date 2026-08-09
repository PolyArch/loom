#ifndef LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATIONDETAIL_H
#define LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATIONDETAIL_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"

#include "llvm/Support/Error.h"

namespace loom::frontend::detail {

bool areKnownDistinctMemoryRoots(mlir::Value lhs, mlir::Value rhs);

bool canPromoteSpscBufferToChannel(mlir::memref::AllocOp allocation);

llvm::Error promoteSpscBufferToChannel(mlir::memref::AllocOp allocation);

} // namespace loom::frontend::detail

#endif // LOOM_FRONTEND_COMPILATION_STRUCTUREDMEMORYCOMMUNICATIONDETAIL_H

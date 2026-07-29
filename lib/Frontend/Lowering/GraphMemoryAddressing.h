#ifndef LOOM_FRONTEND_LOWERING_GRAPHMEMORYADDRESSING_H
#define LOOM_FRONTEND_LOWERING_GRAPHMEMORYADDRESSING_H

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace loom::lowering {

struct LinearByteTerm {
  mlir::Value index;
  std::int64_t byteStride = 1;
};

struct LinearElementTerm {
  mlir::Value index;
  std::int64_t scale = 1;
  unsigned exactSignedDivideShift = 0;
};

struct ResolvedLinearMemoryAddress {
  mlir::Value root;
  llvm::SmallVector<LinearByteTerm, 4> terms;
  llvm::SmallVector<LinearElementTerm, 4> elementTerms;
  mlir::Type indexType;
  std::int64_t byteBias = 0;
  std::int64_t elementBias = 0;
  unsigned byteToElementShift = 0;
  std::uint64_t elementAllocByteCount = 0;
  std::uint64_t accessByteCount = 0;
  llvm::SmallVector<mlir::Operation *, 4> gepsLeafToRoot;
};

std::optional<ResolvedLinearMemoryAddress>
resolveLinearMemoryAddress(mlir::Value pointer, mlir::Type accessType,
                           unsigned canonicalIndexBits);

std::optional<ResolvedLinearMemoryAddress>
resolveLinearMemoryAddress(mlir::Value pointer, dataflow::GraphOp graph,
                           mlir::Type accessType, unsigned canonicalIndexBits);

} // namespace loom::lowering

#endif // LOOM_FRONTEND_LOWERING_GRAPHMEMORYADDRESSING_H

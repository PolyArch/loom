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

struct ResolvedLinearGepAddress {
  mlir::Value root;
  llvm::SmallVector<LinearByteTerm, 4> terms;
  mlir::Type indexType;
  std::int64_t byteBias = 0;
  unsigned byteToElementShift = 0;
  llvm::SmallVector<mlir::Operation *, 4> gepsLeafToRoot;
};

std::optional<ResolvedLinearGepAddress>
resolveLinearGepAddress(mlir::LLVM::GEPOp leafGep, dataflow::GraphOp graph,
                        mlir::Type elementType,
                        unsigned canonicalIndexBits);

} // namespace loom::lowering

#endif // LOOM_FRONTEND_LOWERING_GRAPHMEMORYADDRESSING_H

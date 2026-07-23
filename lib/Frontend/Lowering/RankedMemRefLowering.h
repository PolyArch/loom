#ifndef LOOM_FRONTEND_LOWERING_RANKED_MEMREF_LOWERING_H
#define LOOM_FRONTEND_LOWERING_RANKED_MEMREF_LOWERING_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

namespace loom {
namespace lowering {
namespace detail {

::mlir::LogicalResult checkRankedMemRefAccess(::mlir::Operation *access,
                                              ::mlir::MemRefType type,
                                              ::mlir::ValueRange indices,
                                              unsigned indexBits);

::mlir::LogicalResult checkRankedMemRefCopy(::mlir::memref::CopyOp copy,
                                            unsigned indexBits);

::mlir::Value buildRowMajorLinearIndex(::mlir::OpBuilder &builder,
                                       ::mlir::Location loc,
                                       ::mlir::MemRefType type,
                                       ::mlir::ValueRange indices,
                                       ::mlir::Value execution);

} // namespace detail
} // namespace lowering
} // namespace loom

#endif // LOOM_FRONTEND_LOWERING_RANKED_MEMREF_LOWERING_H

#ifndef LOOM_FRONTEND_ANALYSIS_DENSEPARALLELMEMORYPROJECTION_H
#define LOOM_FRONTEND_ANALYSIS_DENSEPARALLELMEMORYPROJECTION_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
class Operation;
}

namespace loom::frontend::analysis {

/// Proves that one store address is an exact dense row-major projection of
/// every coordinate in the supplied rectangular domain. Memref stores must
/// use the coordinates directly. LLVM stores must use an inbounds,
/// same-element-type GEP chain whose dimension term is the coordinate times
/// every inner upper bound. Multiplication must carry a no-wrap contract.
/// Zero-based coordinates may pass through exact canonical-index round trips;
/// a narrowed dynamic bound must be proven representable before it can match
/// its wider source value.
bool hasExactDenseCoordinateStoreProjection(
    mlir::Operation *store, llvm::ArrayRef<mlir::Value> coordinates,
    llvm::ArrayRef<mlir::OpFoldResult> upperBounds, unsigned indexWidth);

} // namespace loom::frontend::analysis

#endif // LOOM_FRONTEND_ANALYSIS_DENSEPARALLELMEMORYPROJECTION_H

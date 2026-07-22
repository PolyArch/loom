#ifndef LOOM_COMMON_VECTORWIDTH_H
#define LOOM_COMMON_VECTORWIDTH_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Error.h"

namespace loom {

// Bit width of one fixed vector value: the checked product of every dimension
// and `elementBitWidth`. The flattened lane order is row-major, so the last
// axis varies fastest and flattened lane zero owns the least significant bit
// slice. Scalable vectors, rank-zero vectors, zero-width elements, and products
// beyond the unsigned range have no such representation.
llvm::Expected<unsigned> getFixedVectorBitWidth(mlir::VectorType vector,
                                                unsigned elementBitWidth);

} // namespace loom

#endif // LOOM_COMMON_VECTORWIDTH_H

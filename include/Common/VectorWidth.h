#ifndef LOOM_COMMON_VECTORWIDTH_H
#define LOOM_COMMON_VECTORWIDTH_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom {

// Bit width of one fixed vector value: the checked product of every dimension
// and `elementBitWidth`. The flattened lane order is row-major, so the last
// axis varies fastest and flattened lane zero owns the least significant bit
// slice. This is the exact width, so a consumer whose own representation is
// narrower checks and narrows it at its own boundary. Scalable vectors,
// rank-zero vectors, zero-width elements, and products beyond an exact 64-bit
// count have no such representation.
llvm::Expected<std::uint64_t>
getFixedVectorBitWidth(mlir::VectorType vector, std::uint64_t elementBitWidth);

} // namespace loom

#endif // LOOM_COMMON_VECTORWIDTH_H

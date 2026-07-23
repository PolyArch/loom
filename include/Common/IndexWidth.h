#ifndef LOOM_COMMON_INDEXWIDTH_H
#define LOOM_COMMON_INDEXWIDTH_H

#include "mlir/IR/Operation.h"
#include "llvm/Support/Error.h"

namespace loom {

// Bit width that maps the MLIR `index` type onto a fabric.bits<N> port.
// Defaults to 32. Overridden once per process by the LOOM_INDEX_WIDTH
// environment variable when it is a positive integer this type can represent.
// An override it cannot represent is reported by `getIndexBitWidth` instead of
// narrowed here, so it never becomes a different legal width.
unsigned getIndexWidth();

// The canonical `index` bit width in effect at `op`. An explicit entry in the
// closest enclosing data layout owns the width; otherwise the configured width
// owns it. This is the only resolution of that fact, so IR semantics,
// lowering, and simulation cannot disagree about it. A width without a fixed
// positive representation is a checked error, reported at its own value,
// rather than a silently truncated width or a later type mismatch.
llvm::Expected<unsigned> getIndexBitWidth(mlir::Operation *op);

} // namespace loom

#endif // LOOM_COMMON_INDEXWIDTH_H

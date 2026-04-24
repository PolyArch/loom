#ifndef LOOM_COMMON_INDEXWIDTH_H
#define LOOM_COMMON_INDEXWIDTH_H

namespace loom {

// Bit width that maps the MLIR `index` type onto a fabric.bits<N> port.
// Defaults to 32. Overridden once per process by the LOOM_INDEX_WIDTH
// environment variable when it is a positive integer.
unsigned getIndexWidth();

} // namespace loom

#endif // LOOM_COMMON_INDEXWIDTH_H

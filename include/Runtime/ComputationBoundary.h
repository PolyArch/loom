#ifndef LOOM_RUNTIME_COMPUTATIONBOUNDARY_H
#define LOOM_RUNTIME_COMPUTATIONBOUNDARY_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>

namespace loom::runtime {

/// The source-declared computation interval. `Runtime/Computation.h` defines
/// the two weak markers a source program calls around the computation it
/// wants measured; initialization, warmup, and result checking stay outside
/// them. Every owner that has to recognize those markers resolves the symbol
/// through this domain instead of repeating the spelling: the deployment
/// image builder replaces them with System observation events, and the native
/// oracle measures the same interval so the analytic model and System QoR
/// describe one quantity.
enum class ComputationBoundary : std::uint8_t {
  Begin = 0,
  End = 1,
};

inline llvm::StringRef computationBoundarySymbol(ComputationBoundary boundary) {
  switch (boundary) {
  case ComputationBoundary::Begin:
    return "loom_computation_begin";
  case ComputationBoundary::End:
    return "loom_computation_end";
  }
  llvm_unreachable("closed computation boundary domain");
}

} // namespace loom::runtime

#endif

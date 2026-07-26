#ifndef LOOM_ADG_MEMORYLIBRARY_H
#define LOOM_ADG_MEMORYLIBRARY_H

#include "ADG/Builder.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom::adg {

/// Exact temporal parameters of the catalog local-memory recipe. Absence from
/// HybridF32LocalMemoryParameters selects a Spatial operation engine.
struct TemporalMemoryParameters final {
  std::uint32_t tagWidth = 0;
  std::uint64_t residentContextCount = 0;
};

/// Parameters of the catalog's local 128-bit load/store memory. The recipe
/// admits scalar f32 and contiguous vector<4xf32> plain accesses and exposes
/// no manager or subordinate memory capability endpoints.
struct HybridF32LocalMemoryParameters final {
  std::uint64_t capacityBytes = 0;
  std::optional<TemporalMemoryParameters> temporal;
};

/// Builds one ordinary MemorySpec through the canonical Fabric memory owners.
/// The returned spec is consumed by SpatialCoreBuilder::addMemory; this helper
/// owns no persistent schema or identity of its own.
llvm::Expected<MemorySpec>
makeHybridF32LocalMemory(HybridF32LocalMemoryParameters parameters);

} // namespace loom::adg

#endif // LOOM_ADG_MEMORYLIBRARY_H

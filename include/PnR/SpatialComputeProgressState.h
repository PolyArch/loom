#ifndef LOOM_PNR_SPATIALCOMPUTEPROGRESSSTATE_H
#define LOOM_PNR_SPATIALCOMPUTEPROGRESSSTATE_H

#include "Mapping/Artifact/MappingProgressProjection.h"
#include "PnR/PnrIndex.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace loom::pnr {

/// An immutable, removable projection of selected compute uses and physical
/// result release conditions. Transactions retain the prior value for rollback.
struct SpatialComputeProgressState final {
  std::vector<std::uint64_t> selectionKey;
  ::loom::mapping::MappingProgressClosure closure;
  ::loom::mapping::MappingProgressObjectiveProjection objective;
  std::vector<PnrIndex> witnessRealizations;

  std::size_t retainedStorageBytes() const;
};

using SpatialComputeProgressStateHandle =
    std::shared_ptr<const SpatialComputeProgressState>;

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALCOMPUTEPROGRESSSTATE_H

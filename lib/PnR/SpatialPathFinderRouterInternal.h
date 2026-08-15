#ifndef LOOM_LIB_PNR_SPATIALPATHFINDERROUTERINTERNAL_H
#define LOOM_LIB_PNR_SPATIALPATHFINDERROUTERINTERNAL_H

#include "PnR/SpatialPathFinderRouter.h"

#include "llvm/Support/JSON.h"

#include <optional>

namespace loom::pnr::detail {

std::optional<PnrIndex>
resourceStateForCapacity(const FrozenSpatialResourceIndex &resources,
                         PnrIndex capacity);

std::optional<PnrIndex>
resourceOwnerForState(const FrozenSpatialResourceIndex &resources,
                      PnrIndex state);

llvm::json::Object encodeLogicalNetDetail(
    const SpatialCandidateState &candidate, PnrIndex logicalNet);

llvm::json::Array
encodeSelectedOrdinalRanges(llvm::ArrayRef<std::uint8_t> selected);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALPATHFINDERROUTERINTERNAL_H

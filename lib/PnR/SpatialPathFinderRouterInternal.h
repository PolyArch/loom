#ifndef LOOM_LIB_PNR_SPATIALPATHFINDERROUTERINTERNAL_H
#define LOOM_LIB_PNR_SPATIALPATHFINDERROUTERINTERNAL_H

#include "PnR/SpatialPathFinderRouter.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"

#include <optional>

namespace loom::pnr::detail {

llvm::Error pathFinderError(const llvm::Twine &message);

std::string errorMessage(const llvm::ErrorInfoBase &error);

llvm::Error classifyIterationFailure(llvm::Error failure, bool &completed);

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

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

/// One selected RouteTree traversal offered as repair freedom for a fresh
/// handshake witness, including its active switch contention component.
/// Omitting it is a trial and need not remove the witnessed cycle.
struct HandshakeCycleRouteTraversal final {
  PnrIndex logicalNet = 0;
  PnrIndex traversal = 0;
};

/// Returns contributors among the supplied RouteTrees. An empty net list
/// selects every routed logical net in the frozen problem.
llvm::Expected<std::vector<HandshakeCycleRouteTraversal>>
selectedHandshakeCycleRouteTraversals(
    const SpatialCandidateState &candidate,
    const SpatialTagAssignmentSummary &tagSummary,
    llvm::ArrayRef<PnrIndex> frozenWitness,
    llvm::ArrayRef<PnrIndex> logicalNets = {});

/// Exact tag segments whose Temporal-switch demands include one selected
/// repair traversal. The result is diagnostic evidence for an invocation-local
/// candidate; it does not encode a no-good.
llvm::Expected<std::vector<SpatialHandshakeCycleTagSelection>>
selectedHandshakeCycleTagSelections(
    const SpatialCandidateState &candidate,
    const SpatialTagAssignmentSummary &tagSummary,
    llvm::ArrayRef<HandshakeCycleRouteTraversal> contributors);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALPATHFINDERROUTERINTERNAL_H

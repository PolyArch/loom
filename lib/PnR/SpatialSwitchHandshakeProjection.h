#ifndef LOOM_LIB_PNR_SPATIALSWITCHHANDSHAKEPROJECTION_H
#define LOOM_LIB_PNR_SPATIALSWITCHHANDSHAKEPROJECTION_H

#include "PnR/RouteTreeState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <vector>

namespace loom::pnr::detail {

struct SpatialTagAssignmentStateStorage;

/// Returns whether the frozen routing graph contains a Fabric-owned Temporal
/// switch match domain. This allocation-free guard does not infer activation
/// membership and therefore cannot hide a missing frozen activation.
bool hasSpatialTemporalSwitchHandshakeDomain(
    const FrozenSpatialPnrProblem &problem);

/// Rebuilds the exact Temporal switch handshake fragment set selected by the
/// current routes. Rows beyond Fabric resident capacity remain visible to the
/// independent TagResidentCapacityOveruse owner but have no physical
/// activation. Every resident `(row, input)` activation must exist in the
/// frozen handshake inventory; missing owner data is a candidate error.
llvm::Expected<std::vector<PnrIndex>>
deriveSpatialTemporalSwitchHandshakeFragments(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routes,
    llvm::ArrayRef<llvm::ArrayRef<std::optional<llvm::APInt>>> tagValues);

/// Projects the same exact fragment set from the Tag-assignment owner's
/// current route-demand cache. The by-domain form initializes a candidate
/// cache; the single-domain form updates only a transaction's affected domain.
llvm::Expected<std::vector<std::vector<PnrIndex>>>
deriveSpatialTemporalSwitchHandshakeFragmentsByDomain(
    const FrozenSpatialPnrProblem &problem,
    const SpatialTagAssignmentStateStorage &assignments);
llvm::Expected<std::vector<PnrIndex>>
deriveSpatialTemporalSwitchHandshakeDomainFragments(
    const FrozenSpatialPnrProblem &problem, PnrIndex domain,
    const SpatialTagAssignmentStateStorage &assignments);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALSWITCHHANDSHAKEPROJECTION_H

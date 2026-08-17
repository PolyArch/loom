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

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALSWITCHHANDSHAKEPROJECTION_H

#ifndef LOOM_PNR_SPATIALMAPPINGSELECTIONPROJECTION_H
#define LOOM_PNR_SPATIALMAPPINGSELECTIONPROJECTION_H

#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/RouteTreeState.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace loom::pnr {

class SpatialCandidateState;

/// Compares every independently selectable persistent Spatial Mapping fact
/// against one candidate under an explicit route/tag projection. The supplied
/// RouteTrees and values may be transaction-local; no committed route, tag
/// cache, candidate identity, digest, or search-local ordinal is consulted in
/// their place.
///
/// `false` means at least one well-formed selection differs. A malformed or
/// non-resolvable Mapping/candidate projection is a typed error, so callers
/// cannot mistake an incomplete comparison for a broken equality literal.
llvm::Expected<bool> spatialMappingSelectionEqualsCandidate(
    const ::loom::mapping::SpatialMappingView &mapping,
    const SpatialCandidateState &candidate,
    llvm::ArrayRef<const RouteTreeState *> provisionalRoutes,
    llvm::ArrayRef<llvm::ArrayRef<std::optional<llvm::APInt>>>
        provisionalTagValues);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALMAPPINGSELECTIONPROJECTION_H

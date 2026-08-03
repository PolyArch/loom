#ifndef LOOM_PNR_SPATIALMAPPINGMATERIALIZER_H
#define LOOM_PNR_SPATIALMAPPINGMATERIALIZER_H

#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/SpatialCandidateState.h"

#include "llvm/Support/Error.h"

namespace loom::pnr {

/// Projects one closed dense Spatial candidate into the persistent Mapping
/// wire and runs the independent Mapping finalizer before publication. The
/// sealed upstream views are invocation-local read-through values; exact
/// Artifact references remain the persistent authority.
llvm::Expected<::loom::mapping::FinalizedSpatialMapping>
finalizeSpatialMappingCandidate(
    const SpatialCandidateState &candidate,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactStore &store);

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALMAPPINGMATERIALIZER_H

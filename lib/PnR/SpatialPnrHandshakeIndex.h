#ifndef LOOM_LIB_PNR_SPATIALPNRHANDSHAKEINDEX_H
#define LOOM_LIB_PNR_SPATIALPNRHANDSHAKEINDEX_H

#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr::detail {

llvm::Expected<FrozenSpatialHandshakeIndex> buildFrozenSpatialHandshakeIndex(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialRoutingGraph &routing);

llvm::Error verifyFrozenSpatialHandshakeIndex(
    const FrozenSpatialHandshakeIndex &handshake,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialRoutingGraph &routing);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALPNRHANDSHAKEINDEX_H

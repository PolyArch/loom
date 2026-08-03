#ifndef LOOM_LIB_PNR_SPATIALPNRCAPACITYINDEX_H
#define LOOM_LIB_PNR_SPATIALPNRCAPACITYINDEX_H

#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr::detail {

llvm::Expected<FrozenSpatialCapacityIndex> buildFrozenSpatialCapacityIndex(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialMemoryIndex &memory,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialRoutingGraph &routing,
    const FrozenSpatialHandshakeIndex &handshake);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALPNRCAPACITYINDEX_H

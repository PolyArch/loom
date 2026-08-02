#ifndef LOOM_LIB_PNR_SPATIALPNRPORTINDEX_H
#define LOOM_LIB_PNR_SPATIALPNRPORTINDEX_H

#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr::detail {

llvm::Expected<FrozenSpatialPortIndex> buildFrozenSpatialPortIndex(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    FrozenSpatialRealizationIndex &realizations,
    FrozenSpatialTransferIndex &transfers,
    const FrozenSpatialRoutingGraph &routing);

llvm::Error
verifyFrozenSpatialPortIndex(const FrozenSpatialRealizationIndex &realizations,
                             const FrozenSpatialTransferIndex &transfers,
                             const FrozenSpatialPortIndex &ports,
                             const FrozenSpatialRoutingGraph &routing);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALPNRPORTINDEX_H

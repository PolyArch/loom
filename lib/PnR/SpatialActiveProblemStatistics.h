#ifndef LOOM_LIB_PNR_SPATIALACTIVEPROBLEMSTATISTICS_H
#define LOOM_LIB_PNR_SPATIALACTIVEPROBLEMSTATISTICS_H

#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr {

SpatialActiveProblemStatistics buildSpatialActiveProblemStatistics(
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialMemoryIndex &memory,
    const FrozenSpatialTransferIndex &transfers,
    const FrozenSpatialLocalTransferIndex &localTransfers,
    const FrozenSpatialPortIndex &ports,
    const FrozenSpatialCapacityIndex &capacity,
    const FrozenSpatialActiveRoutingDomain &activeRouting,
    const FrozenSpatialHandshakeIndex &handshake,
    std::uint64_t constructionNanoseconds);

} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALACTIVEPROBLEMSTATISTICS_H

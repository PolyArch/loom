#ifndef LOOM_LIB_PNR_SPATIALACTIVEROUTINGDOMAIN_H
#define LOOM_LIB_PNR_SPATIALACTIVEROUTINGDOMAIN_H

#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr {

llvm::Expected<FrozenSpatialActiveRoutingDomain>
buildFrozenSpatialActiveRoutingDomain(
    const FrozenSpatialTransferIndex &transfers,
    const FrozenSpatialLocalTransferIndex &localTransfers,
    const FrozenSpatialPortIndex &ports,
    const FrozenSpatialRoutingGraph &routing);

} // namespace loom::pnr

#endif // LOOM_LIB_PNR_SPATIALACTIVEROUTINGDOMAIN_H

#ifndef LOOM_LIB_PNR_SPATIALPNRRESOURCEINDEX_H
#define LOOM_LIB_PNR_SPATIALPNRRESOURCEINDEX_H

#include "Fabric/Identity/FabricRefImport.h"
#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr::detail {

llvm::Expected<FrozenSpatialResourceIndex> buildFrozenSpatialResourceIndex(
    const ::loom::fabric::FabricArtifactView &fabric);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALPNRRESOURCEINDEX_H

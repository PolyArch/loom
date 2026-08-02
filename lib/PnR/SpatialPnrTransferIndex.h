#ifndef LOOM_LIB_PNR_SPATIALPNRTRANSFERINDEX_H
#define LOOM_LIB_PNR_SPATIALPNRTRANSFERINDEX_H

#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr::detail {

llvm::Expected<FrozenSpatialTransferIndex> buildFrozenSpatialTransferIndex(
    const ::loom::mapping::TechMappingView &techMapping);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALPNRTRANSFERINDEX_H

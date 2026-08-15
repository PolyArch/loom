#ifndef LOOM_LIB_PNR_SPATIALLOCALTRANSFERINDEX_H
#define LOOM_LIB_PNR_SPATIALLOCALTRANSFERINDEX_H

#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr {
struct SpatialComputeBindingSelection;
}

namespace loom::pnr::detail {

llvm::Expected<FrozenSpatialLocalTransferIndex>
buildFrozenSpatialLocalTransferIndex(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialTransferIndex &transfers,
    const FrozenSpatialRoutingGraph &routing);

llvm::Expected<std::vector<PnrIndex>>
derivePreferredSpatialLocalTransferSelections(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<SpatialComputeBindingSelection> computeBindings);

llvm::Expected<std::optional<PnrIndex>>
findPreferredAvailableSpatialLocalTransfer(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<SpatialComputeBindingSelection> computeBindings,
    llvm::ArrayRef<PnrIndex> selections, PnrIndex logicalNet);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALLOCALTRANSFERINDEX_H

#ifndef LOOM_LIB_PNR_SPATIALLOCALTRANSFERINDEX_H
#define LOOM_LIB_PNR_SPATIALLOCALTRANSFERINDEX_H

#include "PnR/SpatialAction.h"
#include "PnR/SpatialPnrProblem.h"

#include <vector>

namespace loom::pnr {
struct SpatialComputeBindingSelection;
}

namespace loom::pnr::detail {

class SpatialComputePlacementHandshakeSelections;

/// Freezes the register-FIFO local-transfer domain. An alternative admitted by
/// the physical-demand projection is retained only when the Fabric handshake
/// owner leaves its own implied selection acyclic; an alternative that closes
/// a combinational cycle by itself is counted and never becomes an option.
llvm::Expected<FrozenSpatialLocalTransferIndex>
buildFrozenSpatialLocalTransferIndex(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricHandshakeContext &handshakeContext,
    const SpatialComputePlacementHandshakeSelections &placementSelections,
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

/// Enumerates the adoptions of `logicalNet` in canonical order: options
/// resident under the current placements first, then options reachable by
/// relocating exactly one endpoint through the first relation-legal compute
/// choice of `legalComputeChoices` on the required placement. Options whose
/// register FIFO another selected net owns are excluded, and one free FIFO
/// stands for every option that pairs the same writer and reader under the
/// same placements. A net with route constraints, an active pairing, or an
/// empty domain enumerates nothing.
llvm::Error enumerateSpatialLocalTransferAdoptions(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<SpatialComputeBindingSelection> computeBindings,
    llvm::ArrayRef<PnrIndex> selections, PnrIndex logicalNet,
    llvm::ArrayRef<SpatialRealizationBindingAction> legalComputeChoices,
    std::vector<SpatialLocalTransferAdoption> &adoptions);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALLOCALTRANSFERINDEX_H

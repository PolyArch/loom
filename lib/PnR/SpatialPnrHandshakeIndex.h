#ifndef LOOM_LIB_PNR_SPATIALPNRHANDSHAKEINDEX_H
#define LOOM_LIB_PNR_SPATIALPNRHANDSHAKEINDEX_H

#include "PnR/SpatialPnrProblem.h"

namespace loom::pnr::detail {

/// The exact Fabric FU handshake selection of every frozen compute placement:
/// the placement's FU occurrence, its realization's capability template, and
/// the TechMapping actor correspondence. The frozen local-transfer domain and
/// the frozen handshake index resolve one selection per placement from this
/// owner, so a register-FIFO alternative is classified against exactly the
/// placement fragments a candidate selecting it activates.
class SpatialComputePlacementHandshakeSelections final {
public:
  static llvm::Expected<SpatialComputePlacementHandshakeSelections>
  build(const ::dataflow::CanonicalDataflowProgramView &dataflow,
        const ::loom::mapping::TechMappingView &techMapping,
        const ::loom::fabric::FabricArtifactView &fabric,
        const FrozenSpatialRealizationIndex &realizations);

  llvm::ArrayRef<::loom::fabric::FabricFuHandshakeSelection>
  placements() const {
    return placements_;
  }

  /// Whether one register-FIFO transfer alternative already closes a
  /// combinational handshake cycle through the fragments it alone implies:
  /// the producer and consumer placement selections, the write and read
  /// traversals, and the exact writer/reader pairing. Such an alternative can
  /// never belong to an acyclic candidate, so the frozen local-transfer
  /// domain does not admit it.
  llvm::Expected<bool> registerFifoTransferClosed(
      const ::loom::fabric::FabricHandshakeContext &context,
      PnrIndex producerPlacement, PnrIndex consumerPlacement,
      const ::loom::fabric::FabricPeRegisterFifoHandshakeSelection &pairing)
      const;

private:
  std::vector<::loom::fabric::FabricFuHandshakeSelection> placements_;
};

llvm::Expected<FrozenSpatialHandshakeIndex> buildFrozenSpatialHandshakeIndex(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricHandshakeContext &handshakeContext,
    const SpatialComputePlacementHandshakeSelections &placementSelections,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialLocalTransferIndex &localTransfers,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialRoutingGraph &routing,
    const FrozenSpatialActiveRoutingDomain &activeRouting);

llvm::Error verifyFrozenSpatialHandshakeIndex(
    const FrozenSpatialHandshakeIndex &handshake,
    const FrozenSpatialRealizationIndex &realizations,
    const FrozenSpatialLocalTransferIndex &localTransfers,
    const FrozenSpatialResourceIndex &resources,
    const FrozenSpatialRoutingGraph &routing);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALPNRHANDSHAKEINDEX_H

#ifndef LOOM_LIB_MAPPING_ARTIFACT_CONFIGUREDHARDWAREPROJECTIONINTERNAL_H
#define LOOM_LIB_MAPPING_ARTIFACT_CONFIGUREDHARDWAREPROJECTIONINTERNAL_H

#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"

#include "llvm/ADT/APInt.h"

#include <optional>

namespace loom::mapping::detail {

struct ConfiguredHardwareProjectionViewAccess final {
  static ConfiguredHardwareProjectionView
  create(std::vector<ConfiguredHardwareFieldValueView> fields) {
    return ConfiguredHardwareProjectionView(std::move(fields));
  }
};

/// One mechanically resolved register-FIFO Mapping selection shared by
/// configured-hardware projection and selected-handshake verification.
struct SpatialPeLocalTransferUse final {
  std::uint64_t producerRealization = 0;
  std::uint64_t consumerRealization = 0;
  ::loom::fabric::FabricFuOccurrencePortRef producerPort;
  ::loom::fabric::FabricFuOccurrencePortRef consumerPort;
  ::loom::fabric::FabricPeOccurrenceRef pe;
  ::loom::fabric::FabricOrdinal registerFifo = 0;
  ::loom::fabric::FabricPhysicalTraversalRef writeTraversal;
  ::loom::fabric::FabricPhysicalTraversalRef readTraversal;
  llvm::APInt tag = llvm::APInt(1, 0);
};

llvm::Expected<std::vector<SpatialPeLocalTransferUse>>
deriveSpatialPeLocalTransferUses(
    const ::loom::fabric::FabricArtifactView &fabric,
    const TechMappingView &techMapping,
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> transfers);

llvm::Expected<ConfiguredHardwareProjectionView>
canonicalizeConfiguredHardwareProjection(
    std::vector<ConfiguredHardwareFieldValueView> fields);

llvm::Expected<::loom::fabric::FabricConfigurationSlotRef>
resolveConfiguredHardwareSlot(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricSemanticConfigFieldRef &field,
    std::optional<::loom::fabric::InstructionContextRef> instructionContext =
        std::nullopt);

llvm::Expected<llvm::APInt> resolveConfiguredHardwarePhysicalTag(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> segments,
    std::uint64_t routeOrdinal, std::uint64_t nodeOrdinal);

llvm::Expected<std::vector<ConfiguredHardwareFieldValueView>>
deriveConfiguredPeFields(
    const ::loom::fabric::FabricArtifactView &fabric,
    const TechMappingView &techMapping,
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView>
        operandQueueMatchGroups);

llvm::Expected<std::vector<ConfiguredHardwareFieldValueView>>
deriveConfiguredBoundaryFields(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments);

llvm::Expected<ConfiguredHardwareProjectionView>
deriveConfiguredHardwareProjection(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialMemoryEngineBindingView> memoryEngines,
    llvm::ArrayRef<SpatialMemoryBindingView> memoryBindings,
    llvm::ArrayRef<SpatialRegisterFifoTransferView> registerFifoTransfers,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments,
    llvm::ArrayRef<SpatialPeOperandQueueMatchGroupView>
        operandQueueMatchGroups);

llvm::Expected<std::vector<ConfiguredHardwareFieldValueView>>
deriveConfiguredMemoryFields(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialMemoryEngineBindingView> memoryEngines,
    llvm::ArrayRef<SpatialMemoryBindingView> memoryBindings,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_CONFIGUREDHARDWAREPROJECTIONINTERNAL_H

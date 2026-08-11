#ifndef LOOM_LIB_MAPPING_ARTIFACT_CONFIGUREDHARDWAREPROJECTIONINTERNAL_H
#define LOOM_LIB_MAPPING_ARTIFACT_CONFIGUREDHARDWAREPROJECTIONINTERNAL_H

#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/APInt.h"

#include <optional>

namespace loom::mapping::detail {

struct ConfiguredHardwareProjectionViewAccess final {
  static ConfiguredHardwareProjectionView
  create(std::vector<ConfiguredHardwareFieldValueView> fields) {
    return ConfiguredHardwareProjectionView(std::move(fields));
  }
};

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
    llvm::ArrayRef<SpatialComputeBindingView> bindings,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments);

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
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments);

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

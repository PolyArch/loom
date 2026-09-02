#ifndef LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGMEMORYIMPORT_H
#define LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGMEMORYIMPORT_H

#include "Fabric/Identity/FabricMemoryServiceHandshake.h"
#include "Mapping/Artifact/MappingArtifact.h"

namespace loom::mapping::detail {

using SpatialMemoryResourceOwnerRef =
    std::variant<SpatialMemoryEngineResourceOwnerRef,
                 SpatialMemoryBindingResourceOwnerRef>;

struct SpatialMemoryResourceUseRequirement final {
  SpatialMemoryResourceOwnerRef owner;
  SpatialActivityEventRef trigger;
  std::vector<::loom::fabric::FabricUsePatternRef> admissiblePatterns;
};

struct ImportedSpatialMemoryView final {
  std::vector<SpatialMemoryEngineBindingView> engineBindings;
  std::vector<SpatialMemoryBindingView> memoryBindings;
  std::vector<SpatialMemoryResourceUseRequirement> requiredResourceUses;
};

llvm::Expected<ImportedSpatialMemoryView> importSpatialMemoryView(
    ::mapping::SpatialOp root,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric);

struct SpatialMemoryHandshakeSelections final {
  std::vector<::loom::fabric::FabricMemoryHandshakeSelection> operations;
  ::loom::fabric::FabricMemoryServiceHandshakeSelection services;
};

llvm::Expected<SpatialMemoryHandshakeSelections>
deriveSpatialMemoryHandshakeSelections(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialMemoryEngineBindingView> engines,
    llvm::ArrayRef<SpatialMemoryBindingView> bindings,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses);

} // namespace loom::mapping::detail

#endif // LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGMEMORYIMPORT_H

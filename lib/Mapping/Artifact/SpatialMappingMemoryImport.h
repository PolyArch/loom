#ifndef LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGMEMORYIMPORT_H
#define LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGMEMORYIMPORT_H

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

} // namespace loom::mapping::detail

#endif // LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGMEMORYIMPORT_H

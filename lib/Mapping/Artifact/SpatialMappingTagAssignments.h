#ifndef LOOM_LIB_MAPPING_ARTIFACT_SPATIALMAPPINGTAGASSIGNMENTS_H
#define LOOM_LIB_MAPPING_ARTIFACT_SPATIALMAPPINGTAGASSIGNMENTS_H

#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <string>
#include <vector>

namespace loom::mapping::detail {

struct RequiredPhysicalTagUse final {
  SpatialResourceOwnerRef owner;
  SpatialActivityEventRef trigger;
  ::loom::fabric::FabricPhysicalTagAssignmentPointView assignmentPoint;
  std::vector<::loom::fabric::FabricOrdinal> matchDomains;
  std::uint64_t routeTreeOrdinal = 0;
  std::uint64_t segmentOrdinal = 0;
  std::vector<std::uint64_t> nodeOrdinals;
};

llvm::Expected<std::string>
physicalTagUseKey(const SpatialResourceOwnerRef &owner,
                  const SpatialActivityEventRef &trigger,
                  const ::loom::fabric::FabricUsePatternRef &pattern,
                  const ArtifactIdentity &dataflowIdentity);

llvm::Expected<std::map<std::string, RequiredPhysicalTagUse>>
deriveRequiredPhysicalTagUses(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_SPATIALMAPPINGTAGASSIGNMENTS_H

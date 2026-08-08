#ifndef LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGCAPACITYVERIFICATION_H
#define LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGCAPACITYVERIFICATION_H

#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::mapping::detail {

struct SpatialCapacityOveruseWitness final {
  ::loom::fabric::FabricInventoryOwnerRef owner;
  ::fabric::StateKey state;
  ::fabric::CapacityDimensionKey dimension;
  std::uint64_t usage = 0;
  std::uint64_t capacity = 0;
  std::vector<std::uint8_t> canonicalOccupancyKey;
};

struct SpatialCapacityOveruseProjection final {
  std::uint64_t total = 0;
  std::optional<SpatialCapacityOveruseWitness> firstWitness;
};

llvm::Expected<std::string> deriveSpatialCapacityActivationKey(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactIdentity &dataflowIdentity,
    const SpatialResourceUseView &resourceUse);

llvm::Expected<SpatialCapacityOveruseProjection> deriveSpatialCapacityOveruse(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactIdentity &dataflowIdentity,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<std::vector<::loom::fabric::FabricPhysicalTraversalRef>>
        routeTraversals);

} // namespace loom::mapping::detail

#endif // LOOM_MAPPING_ARTIFACT_SPATIALMAPPINGCAPACITYVERIFICATION_H

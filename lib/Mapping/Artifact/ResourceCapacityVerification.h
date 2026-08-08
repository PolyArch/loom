#ifndef LOOM_MAPPING_ARTIFACT_RESOURCECAPACITYVERIFICATION_H
#define LOOM_MAPPING_ARTIFACT_RESOURCECAPACITYVERIFICATION_H

#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::mapping::detail {

/// One physical namespace for capacity accounting. `qualifier` is empty for a
/// standalone Fabric root and occurrence-qualified by the System projection
/// for an imported Module. The referenced Fabric remains the ResourceContract
/// authority.
struct ResourceCapacityNamespaceView final {
  const ::loom::fabric::FabricArtifactView *fabric = nullptr;
  std::vector<std::uint8_t> qualifier;
};

struct ResourceCapacityUseProjection final {
  std::size_t namespaceOrdinal = 0;
  ::loom::fabric::FabricUsePatternRef pattern;
  std::string activationKey;
};

struct ResourceCapacityRouteProjection final {
  std::size_t namespaceOrdinal = 0;
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversals;
};

struct ResourceCapacityOveruseWitness final {
  std::size_t namespaceOrdinal = 0;
  ::loom::fabric::FabricInventoryOwnerRef owner;
  ::fabric::StateKey state;
  ::fabric::CapacityDimensionKey dimension;
  std::uint64_t usage = 0;
  std::uint64_t capacity = 0;
  std::vector<std::uint8_t> canonicalOccupancyKey;
};

struct ResourceCapacityOveruseProjection final {
  std::uint64_t total = 0;
  std::optional<ResourceCapacityOveruseWitness> firstWitness;
};

llvm::Expected<ResourceCapacityOveruseProjection> deriveResourceCapacityOveruse(
    llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
    llvm::ArrayRef<ResourceCapacityUseProjection> resourceUses,
    llvm::ArrayRef<ResourceCapacityRouteProjection> routeTraversals);

} // namespace loom::mapping::detail

#endif // LOOM_MAPPING_ARTIFACT_RESOURCECAPACITYVERIFICATION_H

#ifndef LOOM_MAPPING_ARTIFACT_RESOURCECAPACITYVERIFICATION_H
#define LOOM_MAPPING_ARTIFACT_RESOURCECAPACITYVERIFICATION_H

#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
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
  std::uint32_t payloadWidthBits = 0;
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

struct ResourceCapacityPatternSource final {
  std::size_t namespaceOrdinal = 0;
  ::loom::fabric::FabricUsePatternRef pattern;
};

struct ResourceCapacityTraversalSource final {
  std::size_t namespaceOrdinal = 0;
  ::loom::fabric::FabricPhysicalTraversalRef traversal;
};

struct FrozenResourceCapacityCell final {
  std::size_t namespaceOrdinal = 0;
  ::loom::fabric::FabricInventoryOwnerRef owner;
  ::fabric::StateKey state;
  ::fabric::CapacityDimensionKey dimension;
  std::uint64_t capacity = 0;
  std::uint64_t initialOccupancy = 0;
  std::string canonicalKey;
};

struct FrozenResourceCapacityClaim final {
  std::size_t cell = 0;
  std::uint64_t amount = 0;
};

struct FrozenResourceCapacityPattern final {
  std::size_t namespaceOrdinal = 0;
  ::loom::fabric::FabricUsePatternRef reference;
  std::uint64_t beginRank = 0;
  std::uint64_t endRank = 0;
  std::vector<FrozenResourceCapacityClaim> claims;
  MappingResourceProgressUse progressUse;
  ::fabric::UsePatternTiming timing;
};

struct FrozenResourceCapacityRouteClaim final {
  std::string canonicalKey;
  std::size_t cell = 0;
  std::uint64_t amount = 0;
};

struct FrozenResourceCapacityTraversal final {
  std::size_t namespaceOrdinal = 0;
  ::loom::fabric::FabricPhysicalTraversalRef reference;
  std::vector<FrozenResourceCapacityRouteClaim> claims;
  std::vector<MappingResourceProgressUse> progressUses;
  ::loom::fabric::FabricPhysicalTraversalTimingView timing;
};

class FrozenResourceCapacityIndex final {
public:
  llvm::ArrayRef<FrozenResourceCapacityCell> cells() const { return cells_; }
  llvm::ArrayRef<FrozenResourceCapacityPattern> patterns() const {
    return patterns_;
  }
  llvm::ArrayRef<FrozenResourceCapacityTraversal> traversals() const {
    return traversals_;
  }

  llvm::Expected<std::size_t>
  patternOrdinal(std::size_t namespaceOrdinal,
                 const ::loom::fabric::FabricUsePatternRef &pattern) const;
  llvm::Expected<std::size_t> traversalOrdinal(
      std::size_t namespaceOrdinal,
      const ::loom::fabric::FabricPhysicalTraversalRef &traversal) const;

private:
  std::vector<FrozenResourceCapacityCell> cells_;
  std::vector<FrozenResourceCapacityPattern> patterns_;
  std::vector<FrozenResourceCapacityTraversal> traversals_;
  std::map<std::string, std::size_t> patternOrdinals_;
  std::map<std::string, std::size_t> traversalOrdinals_;

  friend llvm::Expected<FrozenResourceCapacityIndex>
      freezeResourceCapacityIndex(
          llvm::ArrayRef<ResourceCapacityNamespaceView>,
          llvm::ArrayRef<ResourceCapacityPatternSource>,
          llvm::ArrayRef<ResourceCapacityTraversalSource>);
};

struct FrozenResourceCapacityUseSelection final {
  std::size_t patternOrdinal = 0;
  std::string activationKey;
};

struct FrozenResourceCapacityRouteSelection final {
  std::vector<std::size_t> traversalOrdinals;
  std::uint32_t payloadWidthBits = 0;
};

/// Candidate-specific intrinsic physical timing derived from the same exact
/// ResourceUse and traversal selections as capacity and progress. Latency and
/// initiation interval are cycle quantities from Fabric. Transport demand is
/// logical payload bits multiplied by the traversal's minimum initiation
/// interval. Dynamic stalls and measured implementation delay remain outside
/// this projection.
struct ResourcePhysicalTimingProjection final {
  std::uint64_t releaseLatencyCycles = 0;
  std::uint64_t minimumInitiationIntervalCycles = 1;
  std::uint64_t transportBitCycleDemand = 0;
};

/// One removable physical-demand projection shared by System PnR and strict
/// SystemMapping verification. Capacity and progress are derived from the
/// same selected UsePattern and traversal ordinals.
struct ResourcePhysicalDemandProjection final {
  ResourceCapacityOveruseProjection capacity;
  std::vector<std::uint64_t> baselineOccupancy;
  std::vector<MappingResourceProgressUse> progressUses;
  ResourcePhysicalTimingProjection timing;
};

llvm::Expected<ResourcePhysicalDemandProjection> deriveResourcePhysicalDemand(
    const FrozenResourceCapacityIndex &index,
    llvm::ArrayRef<FrozenResourceCapacityUseSelection> resourceUses,
    llvm::ArrayRef<FrozenResourceCapacityRouteSelection> routeTraversals);

/// Derives one demand projection from canonically ordered route segments
/// without materializing a concatenated route-selection vector.
llvm::Expected<ResourcePhysicalDemandProjection> deriveResourcePhysicalDemand(
    const FrozenResourceCapacityIndex &index,
    llvm::ArrayRef<FrozenResourceCapacityUseSelection> resourceUses,
    llvm::ArrayRef<llvm::ArrayRef<FrozenResourceCapacityRouteSelection>>
        routeSegments);

llvm::Expected<ResourcePhysicalDemandProjection> deriveResourcePhysicalDemand(
    llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
    llvm::ArrayRef<ResourceCapacityUseProjection> resourceUses,
    llvm::ArrayRef<ResourceCapacityRouteProjection> routeTraversals);

std::vector<std::uint8_t>
rootResourceCapacityQualifier(const ::loom::fabric::FabricArtifactView &fabric);
std::vector<std::uint8_t> occurrenceResourceCapacityQualifier(
    const ::loom::fabric::FabricArtifactView &system,
    ::loom::fabric::SpatialCoreOccurrenceRef spatialCore);

llvm::Expected<FrozenResourceCapacityIndex> freezeResourceCapacityIndex(
    llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
    llvm::ArrayRef<ResourceCapacityPatternSource> patterns,
    llvm::ArrayRef<ResourceCapacityTraversalSource> traversals);

llvm::Expected<ResourceCapacityOveruseProjection> deriveResourceCapacityOveruse(
    const FrozenResourceCapacityIndex &index,
    llvm::ArrayRef<FrozenResourceCapacityUseSelection> resourceUses,
    llvm::ArrayRef<FrozenResourceCapacityRouteSelection> routeTraversals);

/// Derives the exact reset occupancy after statically selected route claims.
/// Route-local requester groups are deduplicated exactly as in capacity
/// verification. The returned vector is indexed by `index.cells()`.
llvm::Expected<std::vector<std::uint64_t>>
deriveResourceCapacityBaselineOccupancy(
    const FrozenResourceCapacityIndex &index,
    llvm::ArrayRef<FrozenResourceCapacityRouteSelection> routeTraversals);

llvm::Expected<ResourceCapacityOveruseProjection> deriveResourceCapacityOveruse(
    llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
    llvm::ArrayRef<ResourceCapacityUseProjection> resourceUses,
    llvm::ArrayRef<ResourceCapacityRouteProjection> routeTraversals);

} // namespace loom::mapping::detail

#endif // LOOM_MAPPING_ARTIFACT_RESOURCECAPACITYVERIFICATION_H

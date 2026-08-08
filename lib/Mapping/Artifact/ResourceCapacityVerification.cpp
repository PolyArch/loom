#include "ResourceCapacityVerification.h"

#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "resource_capacity_verification_invalid: " + message);
}

void appendU32(std::string &bytes, std::uint32_t value) {
  for (unsigned shift = 24;; shift -= 8) {
    bytes.push_back(static_cast<char>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendU64(std::string &bytes, std::uint64_t value) {
  for (unsigned shift = 56;; shift -= 8) {
    bytes.push_back(static_cast<char>(value >> shift));
    if (shift == 0)
      break;
  }
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendSized(std::vector<std::uint8_t> &bytes,
                 llvm::ArrayRef<std::uint8_t> value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value.size() >> shift));
  bytes.insert(bytes.end(), value.begin(), value.end());
}

template <typename Ref> std::string refKey(const Ref &reference) {
  const auto bytes = ::loom::fabric::canonicalFabricBytes(reference);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename Ref>
std::string indexedRefKey(std::size_t namespaceOrdinal, const Ref &reference) {
  std::string result;
  appendU64(result, namespaceOrdinal);
  const std::string local = refKey(reference);
  appendU64(result, local.size());
  result.append(local);
  return result;
}

llvm::Expected<const ResourceCapacityNamespaceView &>
resolveNamespace(llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
                 std::size_t ordinal) {
  if (ordinal >= namespaces.size() || !namespaces[ordinal].fabric)
    return invalid("capacity input has an absent physical namespace");
  return namespaces[ordinal];
}

std::string
physicalOwnerKey(const ResourceCapacityNamespaceView &space,
                 const ::loom::fabric::FabricInventoryOwnerRef &owner) {
  std::string result;
  appendU64(result, space.qualifier.size());
  result.append(reinterpret_cast<const char *>(space.qualifier.data()),
                space.qualifier.size());
  const std::string local = refKey(owner);
  appendU64(result, local.size());
  result.append(local);
  return result;
}

std::string dimensionKey(const ResourceCapacityNamespaceView &space,
                         const ::loom::fabric::FabricInventoryOwnerRef &owner,
                         ::fabric::StateKey state,
                         ::fabric::CapacityDimensionKey dimension) {
  std::string result = physicalOwnerKey(space, owner);
  appendU32(result, state.ordinal());
  appendU32(result, dimension.ordinal());
  return result;
}

std::string activationGroupKey(
    const ResourceCapacityNamespaceView &space,
    const ::loom::fabric::FabricTraversalActivationGroupView &group) {
  std::string result = physicalOwnerKey(space, group.owner);
  appendU32(result, static_cast<std::uint32_t>(group.kind));
  appendU64(result, group.ordinal);
  return result;
}

std::string
routeClaimKey(const ResourceCapacityNamespaceView &space,
              const ::loom::fabric::FabricTraversalActivationGroupView &group,
              const ::loom::fabric::FabricInventoryOwnerRef &owner,
              ::fabric::StateKey state,
              ::fabric::CapacityDimensionKey dimension) {
  const std::string activation = activationGroupKey(space, group);
  const std::string capacity = dimensionKey(space, owner, state, dimension);
  std::string result;
  appendU64(result, activation.size());
  result.append(activation);
  appendU64(result, capacity.size());
  result.append(capacity);
  return result;
}

llvm::Error checkedAdd(std::uint64_t amount, std::uint64_t &value,
                       llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - value)
    return invalid(subject + " overflows u64");
  value += amount;
  return llvm::Error::success();
}

struct ResolvedPattern final {
  const ResourceCapacityNamespaceView *space = nullptr;
  ::loom::fabric::FabricInventoryOwnerRef owner;
  const ::fabric::ResourceContract *contract = nullptr;
  ::fabric::UsePattern pattern;
};

llvm::Expected<ResolvedPattern>
resolvePattern(llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
               const ResourceCapacityUseProjection &use) {
  auto space = resolveNamespace(namespaces, use.namespaceOrdinal);
  if (!space)
    return space.takeError();
  const auto owner = use.pattern.owner.catalog();
  const ::fabric::ResourceContract *contract =
      space->fabric->resourceContract(owner);
  if (!contract || use.pattern.ordinal >= contract->usePatternCount())
    return invalid("ResourceUse does not resolve a Fabric pattern");
  const ::fabric::UsePattern pattern =
      contract->usePattern(::fabric::UsePatternKey(use.pattern.ordinal));
  return ResolvedPattern{&*space, owner, contract, pattern};
}

struct CapacityCell final {
  std::size_t namespaceOrdinal = 0;
  ::loom::fabric::FabricInventoryOwnerRef owner;
  ::fabric::StateKey state;
  ::fabric::CapacityDimensionKey dimension;
  std::uint64_t capacity = 0;
  std::uint64_t initial = 0;
};

class CapacityCatalog final {
public:
  explicit CapacityCatalog(
      llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces)
      : namespaces_(namespaces) {}

  llvm::Expected<std::uint32_t>
  get(std::size_t namespaceOrdinal,
      const ::loom::fabric::FabricInventoryOwnerRef &owner,
      const ::fabric::ResourceContract &contract,
      const ::fabric::Claim &claim) {
    auto space = resolveNamespace(namespaces_, namespaceOrdinal);
    if (!space)
      return space.takeError();
    if (claim.state.ordinal() >= contract.stateCount())
      return invalid("claim state is outside its Fabric contract");
    const auto dimensions = contract.capacityDimensions(claim.state);
    if (claim.dimension.ordinal() >= dimensions.size())
      return invalid("claim dimension is outside its Fabric state");
    const std::string key =
        dimensionKey(*space, owner, claim.state, claim.dimension);
    auto found = ordinals_.find(key);
    if (found != ordinals_.end())
      return found->second;
    if (cells_.size() >= std::numeric_limits<std::uint32_t>::max())
      return invalid("capacity dimension inventory exceeds u32");
    const auto &dimension = dimensions[claim.dimension.ordinal()];
    const std::uint32_t ordinal = static_cast<std::uint32_t>(cells_.size());
    ordinals_.try_emplace(key, ordinal);
    cells_.push_back({namespaceOrdinal, owner, claim.state, claim.dimension,
                      dimension.capacity.value(),
                      dimension.initialOccupancy.value()});
    return ordinal;
  }

  llvm::ArrayRef<CapacityCell> cells() const { return cells_; }

private:
  llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces_;
  llvm::StringMap<std::uint32_t> ordinals_;
  std::vector<CapacityCell> cells_;
};

std::vector<std::uint8_t> unsignedBytes(llvm::StringRef bytes) {
  std::vector<std::uint8_t> result;
  result.reserve(bytes.size());
  for (char byte : bytes)
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

struct PeakUsage final {
  std::uint64_t usage = 0;
  std::string occupancyKey;
};

void updatePeak(PeakUsage &peak, std::uint64_t usage,
                llvm::StringRef occupancyKey) {
  if (usage > peak.usage ||
      (usage == peak.usage && occupancyKey < peak.occupancyKey)) {
    peak.usage = usage;
    peak.occupancyKey = occupancyKey.str();
  }
}

} // namespace

std::vector<std::uint8_t> rootResourceCapacityQualifier(
    const ::loom::fabric::FabricArtifactView &fabric) {
  std::vector<std::uint8_t> result;
  appendU32(result, 0);
  appendSized(result, fabric.identity().bytes());
  return result;
}

std::vector<std::uint8_t> occurrenceResourceCapacityQualifier(
    const ::loom::fabric::FabricArtifactView &system,
    ::loom::fabric::SpatialCoreOccurrenceRef spatialCore) {
  std::vector<std::uint8_t> result;
  appendU32(result, 1);
  appendSized(result, system.identity().bytes());
  appendSized(result, ::loom::fabric::canonicalFabricBytes(spatialCore));
  return result;
}

llvm::Expected<std::size_t> FrozenResourceCapacityIndex::patternOrdinal(
    std::size_t namespaceOrdinal,
    const ::loom::fabric::FabricUsePatternRef &pattern) const {
  const auto found =
      patternOrdinals_.find(indexedRefKey(namespaceOrdinal, pattern));
  if (found == patternOrdinals_.end())
    return invalid("ResourceUse pattern is absent from the frozen index");
  return found->second;
}

llvm::Expected<std::size_t> FrozenResourceCapacityIndex::traversalOrdinal(
    std::size_t namespaceOrdinal,
    const ::loom::fabric::FabricPhysicalTraversalRef &traversal) const {
  const auto found =
      traversalOrdinals_.find(indexedRefKey(namespaceOrdinal, traversal));
  if (found == traversalOrdinals_.end())
    return invalid("route traversal is absent from the frozen index");
  return found->second;
}

llvm::Expected<FrozenResourceCapacityIndex> freezeResourceCapacityIndex(
    llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
    llvm::ArrayRef<ResourceCapacityPatternSource> patternSources,
    llvm::ArrayRef<ResourceCapacityTraversalSource> traversalSources) {
  std::vector<ResourceCapacityPatternSource> patterns(patternSources.begin(),
                                                      patternSources.end());
  llvm::sort(patterns, [](const auto &lhs, const auto &rhs) {
    return indexedRefKey(lhs.namespaceOrdinal, lhs.pattern) <
           indexedRefKey(rhs.namespaceOrdinal, rhs.pattern);
  });
  patterns.erase(std::unique(patterns.begin(), patterns.end(),
                             [](const auto &lhs, const auto &rhs) {
                               return lhs.namespaceOrdinal ==
                                          rhs.namespaceOrdinal &&
                                      lhs.pattern == rhs.pattern;
                             }),
                 patterns.end());

  std::vector<ResourceCapacityTraversalSource> traversals(
      traversalSources.begin(), traversalSources.end());
  llvm::sort(traversals, [](const auto &lhs, const auto &rhs) {
    return indexedRefKey(lhs.namespaceOrdinal, lhs.traversal) <
           indexedRefKey(rhs.namespaceOrdinal, rhs.traversal);
  });
  traversals.erase(std::unique(traversals.begin(), traversals.end(),
                               [](const auto &lhs, const auto &rhs) {
                                 return lhs.namespaceOrdinal ==
                                            rhs.namespaceOrdinal &&
                                        lhs.traversal == rhs.traversal;
                               }),
                   traversals.end());

  FrozenResourceCapacityIndex result;
  CapacityCatalog dimensions(namespaces);
  for (const ResourceCapacityPatternSource &source : patterns) {
    const ResourceCapacityUseProjection selected{
        source.namespaceOrdinal, source.pattern, {}};
    auto resolved = resolvePattern(namespaces, selected);
    if (!resolved)
      return resolved.takeError();
    const auto ranks =
        resolved->contract->eventOrder(resolved->pattern.timingAndProgress);
    if (resolved->pattern.acquire.ordinal() >= ranks.size() ||
        resolved->pattern.release.ordinal() >= ranks.size())
      return invalid("Fabric pattern has an incomplete timing relation");
    const std::uint64_t begin = ranks[resolved->pattern.acquire.ordinal()];
    const std::uint64_t release = ranks[resolved->pattern.release.ordinal()];
    if (begin == std::numeric_limits<std::uint64_t>::max())
      return invalid("Fabric pattern timing rank exceeds u64");
    FrozenResourceCapacityPattern frozen{source.namespaceOrdinal,
                                         source.pattern,
                                         begin,
                                         release > begin ? release : begin + 1,
                                         {}};
    for (const ::fabric::Claim &claim : resolved->pattern.claims) {
      auto cell = dimensions.get(source.namespaceOrdinal, resolved->owner,
                                 *resolved->contract, claim);
      if (!cell)
        return cell.takeError();
      frozen.claims.push_back({*cell, claim.amount.value()});
    }
    const std::size_t ordinal = result.patterns_.size();
    if (!result.patternOrdinals_
             .emplace(indexedRefKey(source.namespaceOrdinal, source.pattern),
                      ordinal)
             .second)
      return invalid("frozen ResourceUse pattern inventory is not unique");
    result.patterns_.push_back(std::move(frozen));
  }

  for (const ResourceCapacityTraversalSource &source : traversals) {
    auto space = resolveNamespace(namespaces, source.namespaceOrdinal);
    if (!space)
      return space.takeError();
    const auto found = llvm::find_if(
        space->fabric->physicalTraversals(), [&](const auto &candidate) {
          return candidate.reference == source.traversal;
        });
    if (found == space->fabric->physicalTraversals().end())
      return invalid("route traversal is absent from its Fabric namespace");
    FrozenResourceCapacityTraversal frozen{
        source.namespaceOrdinal, source.traversal, {}};
    for (const auto &use : found->impliedUses) {
      const ResourceCapacityUseProjection selected{
          source.namespaceOrdinal, use.pattern, {}};
      auto resolved = resolvePattern(namespaces, selected);
      if (!resolved)
        return resolved.takeError();
      for (const ::fabric::Claim &claim : resolved->pattern.claims) {
        auto cell = dimensions.get(source.namespaceOrdinal, resolved->owner,
                                   *resolved->contract, claim);
        if (!cell)
          return cell.takeError();
        frozen.claims.push_back(
            {routeClaimKey(*space, use.activationGroup, resolved->owner,
                           claim.state, claim.dimension),
             *cell, claim.amount.value()});
      }
    }
    const std::size_t ordinal = result.traversals_.size();
    if (!result.traversalOrdinals_
             .emplace(indexedRefKey(source.namespaceOrdinal, source.traversal),
                      ordinal)
             .second)
      return invalid("frozen route traversal inventory is not unique");
    result.traversals_.push_back(std::move(frozen));
  }

  result.cells_.reserve(dimensions.cells().size());
  for (const CapacityCell &cell : dimensions.cells()) {
    auto space = resolveNamespace(namespaces, cell.namespaceOrdinal);
    if (!space)
      return space.takeError();
    result.cells_.push_back(
        {cell.namespaceOrdinal, cell.owner, cell.state, cell.dimension,
         cell.capacity, cell.initial,
         dimensionKey(*space, cell.owner, cell.state, cell.dimension)});
  }
  return result;
}

llvm::Expected<ResourceCapacityOveruseProjection> deriveResourceCapacityOveruse(
    const FrozenResourceCapacityIndex &index,
    llvm::ArrayRef<FrozenResourceCapacityUseSelection> resourceUses,
    llvm::ArrayRef<FrozenResourceCapacityRouteSelection> routeTraversals) {
  struct SelectedClaim final {
    std::size_t cell = 0;
    std::uint64_t amount = 0;
  };
  std::vector<std::uint64_t> usage;
  usage.reserve(index.cells().size());
  for (const FrozenResourceCapacityCell &cell : index.cells())
    usage.push_back(cell.initialOccupancy);
  for (const FrozenResourceCapacityRouteSelection &route : routeTraversals) {
    std::map<std::string, SelectedClaim> selectedClaims;
    for (std::size_t traversalOrdinal : route.traversalOrdinals) {
      if (traversalOrdinal >= index.traversals().size())
        return invalid("route selection names a foreign traversal");
      for (const FrozenResourceCapacityRouteClaim &claim :
           index.traversals()[traversalOrdinal].claims) {
        auto [position, inserted] = selectedClaims.try_emplace(
            claim.canonicalKey, SelectedClaim{claim.cell, claim.amount});
        if (!inserted && (position->second.cell != claim.cell ||
                          position->second.amount != claim.amount))
          return invalid(
              "one route claim key has inconsistent capacity demand");
      }
    }
    for (const auto &[key, claim] : selectedClaims) {
      (void)key;
      if (claim.cell >= usage.size())
        return invalid("route claim names a foreign capacity cell");
      if (llvm::Error error = checkedAdd(claim.amount, usage[claim.cell],
                                         "route capacity usage"))
        return std::move(error);
    }
  }

  std::vector<PeakUsage> peaks;
  peaks.reserve(index.cells().size());
  for (const auto &[ordinal, cell] : llvm::enumerate(index.cells())) {
    std::string witnessKey;
    appendU32(witnessKey, 1);
    appendU64(witnessKey, cell.canonicalKey.size());
    witnessKey.append(cell.canonicalKey);
    peaks.push_back({usage[ordinal], std::move(witnessKey)});
  }

  std::map<std::string, std::vector<std::size_t>> usesByActivation;
  for (const FrozenResourceCapacityUseSelection &use : resourceUses) {
    if (use.patternOrdinal >= index.patterns().size())
      return invalid("ResourceUse selection names a foreign pattern");
    usesByActivation[use.activationKey].push_back(use.patternOrdinal);
  }
  struct FrozenBoundaryChange final {
    std::size_t cell = 0;
    std::uint64_t rank = 0;
    std::uint64_t added = 0;
    std::uint64_t removed = 0;
  };
  for (const auto &[activation, selectedPatterns] : usesByActivation) {
    std::vector<FrozenBoundaryChange> changes;
    for (std::size_t patternOrdinal : selectedPatterns) {
      const FrozenResourceCapacityPattern &pattern =
          index.patterns()[patternOrdinal];
      for (const FrozenResourceCapacityClaim &claim : pattern.claims) {
        if (claim.cell >= usage.size())
          return invalid("ResourceUse claim names a foreign capacity cell");
        changes.push_back({claim.cell, pattern.beginRank, claim.amount, 0});
        changes.push_back({claim.cell, pattern.endRank, 0, claim.amount});
      }
    }
    llvm::sort(changes, [](const auto &lhs, const auto &rhs) {
      return std::tie(lhs.cell, lhs.rank) < std::tie(rhs.cell, rhs.rank);
    });
    for (std::size_t begin = 0; begin < changes.size();) {
      const std::size_t cell = changes[begin].cell;
      const std::uint64_t residentUsage = usage[cell];
      std::size_t cursor = begin;
      while (cursor < changes.size() && changes[cursor].cell == cell) {
        const std::uint64_t rank = changes[cursor].rank;
        std::uint64_t added = 0;
        std::uint64_t removed = 0;
        while (cursor < changes.size() && changes[cursor].cell == cell &&
               changes[cursor].rank == rank) {
          if (llvm::Error error =
                  checkedAdd(changes[cursor].added, added, "capacity addition"))
            return std::move(error);
          if (llvm::Error error = checkedAdd(changes[cursor].removed, removed,
                                             "capacity removal"))
            return std::move(error);
          ++cursor;
        }
        if (removed > usage[cell])
          return invalid("capacity removal exceeds active usage");
        usage[cell] -= removed;
        if (llvm::Error error =
                checkedAdd(added, usage[cell], "capacity usage"))
          return std::move(error);
        std::string witnessKey;
        appendU32(witnessKey, 0);
        appendU64(witnessKey, activation.size());
        witnessKey.append(activation);
        appendU64(witnessKey, index.cells()[cell].canonicalKey.size());
        witnessKey.append(index.cells()[cell].canonicalKey);
        appendU64(witnessKey, rank);
        updatePeak(peaks[cell], usage[cell], witnessKey);
      }
      if (usage[cell] != residentUsage)
        return invalid("timed envelope does not release every claim");
      begin = cursor;
    }
  }

  ResourceCapacityOveruseProjection result;
  for (const auto &[ordinal, cell] : llvm::enumerate(index.cells())) {
    const std::uint64_t overuse = peaks[ordinal].usage > cell.capacity
                                      ? peaks[ordinal].usage - cell.capacity
                                      : 0;
    if (llvm::Error error = checkedAdd(overuse, result.total, "total overuse"))
      return std::move(error);
    if (overuse == 0)
      continue;
    ResourceCapacityOveruseWitness witness{
        cell.namespaceOrdinal,
        cell.owner,
        cell.state,
        cell.dimension,
        peaks[ordinal].usage,
        cell.capacity,
        unsignedBytes(peaks[ordinal].occupancyKey)};
    if (!result.firstWitness || witness.canonicalOccupancyKey <
                                    result.firstWitness->canonicalOccupancyKey)
      result.firstWitness = std::move(witness);
  }
  return result;
}

llvm::Expected<ResourceCapacityOveruseProjection> deriveResourceCapacityOveruse(
    llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
    llvm::ArrayRef<ResourceCapacityUseProjection> resourceUses,
    llvm::ArrayRef<ResourceCapacityRouteProjection> routeTraversals) {
  std::vector<ResourceCapacityPatternSource> patterns;
  patterns.reserve(resourceUses.size());
  for (const ResourceCapacityUseProjection &use : resourceUses)
    patterns.push_back({use.namespaceOrdinal, use.pattern});
  std::vector<ResourceCapacityTraversalSource> traversals;
  for (const ResourceCapacityRouteProjection &route : routeTraversals)
    for (const auto &traversal : route.traversals)
      traversals.push_back({route.namespaceOrdinal, traversal});
  auto index = freezeResourceCapacityIndex(namespaces, patterns, traversals);
  if (!index)
    return index.takeError();
  std::vector<FrozenResourceCapacityUseSelection> selectedUses;
  selectedUses.reserve(resourceUses.size());
  for (const ResourceCapacityUseProjection &use : resourceUses) {
    auto pattern = index->patternOrdinal(use.namespaceOrdinal, use.pattern);
    if (!pattern)
      return pattern.takeError();
    selectedUses.push_back({*pattern, use.activationKey});
  }
  std::vector<FrozenResourceCapacityRouteSelection> selectedRoutes;
  selectedRoutes.reserve(routeTraversals.size());
  for (const ResourceCapacityRouteProjection &route : routeTraversals) {
    FrozenResourceCapacityRouteSelection selected;
    selected.traversalOrdinals.reserve(route.traversals.size());
    for (const auto &traversal : route.traversals) {
      auto ordinal = index->traversalOrdinal(route.namespaceOrdinal, traversal);
      if (!ordinal)
        return ordinal.takeError();
      selected.traversalOrdinals.push_back(*ordinal);
    }
    selectedRoutes.push_back(std::move(selected));
  }
  return deriveResourceCapacityOveruse(*index, selectedUses, selectedRoutes);
}

} // namespace loom::mapping::detail

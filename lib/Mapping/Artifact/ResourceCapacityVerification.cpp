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

template <typename Ref> std::string refKey(const Ref &reference) {
  const auto bytes = ::loom::fabric::canonicalFabricBytes(reference);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
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
  std::uint64_t usage = 0;
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
                      dimension.initialOccupancy.value(),
                      dimension.initialOccupancy.value()});
    return ordinal;
  }

  CapacityCell &operator[](std::uint32_t ordinal) { return cells_[ordinal]; }
  llvm::ArrayRef<CapacityCell> cells() const { return cells_; }

private:
  llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces_;
  llvm::StringMap<std::uint32_t> ordinals_;
  std::vector<CapacityCell> cells_;
};

struct BoundaryChange final {
  std::uint32_t dimension = 0;
  std::uint64_t rank = 0;
  std::uint64_t added = 0;
  std::uint64_t removed = 0;
};

std::vector<std::uint8_t> unsignedBytes(llvm::StringRef bytes) {
  std::vector<std::uint8_t> result;
  result.reserve(bytes.size());
  for (char byte : bytes)
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

void selectFirstWitness(ResourceCapacityOveruseProjection &projection,
                        const CapacityCell &cell, std::uint64_t usage,
                        llvm::StringRef occupancyKey) {
  if (usage <= cell.capacity)
    return;
  ResourceCapacityOveruseWitness witness{cell.namespaceOrdinal,
                                         cell.owner,
                                         cell.state,
                                         cell.dimension,
                                         usage,
                                         cell.capacity,
                                         unsignedBytes(occupancyKey)};
  if (!projection.firstWitness ||
      witness.canonicalOccupancyKey <
          projection.firstWitness->canonicalOccupancyKey)
    projection.firstWitness = std::move(witness);
}

struct PeakUsage final {
  std::uint64_t usage = 0;
  std::string occupancyKey;
};

llvm::Error
initializePeaks(llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
                const CapacityCatalog &dimensions,
                std::vector<PeakUsage> &peaks) {
  for (std::size_t ordinal = peaks.size(); ordinal < dimensions.cells().size();
       ++ordinal) {
    const CapacityCell &cell = dimensions.cells()[ordinal];
    auto space = resolveNamespace(namespaces, cell.namespaceOrdinal);
    if (!space)
      return space.takeError();
    std::string witnessKey;
    appendU32(witnessKey, 1);
    const std::string cellKey =
        dimensionKey(*space, cell.owner, cell.state, cell.dimension);
    appendU64(witnessKey, cellKey.size());
    witnessKey.append(cellKey);
    peaks.push_back({cell.usage, std::move(witnessKey)});
  }
  return llvm::Error::success();
}

void updatePeak(PeakUsage &peak, std::uint64_t usage,
                llvm::StringRef occupancyKey) {
  if (usage > peak.usage ||
      (usage == peak.usage && occupancyKey < peak.occupancyKey)) {
    peak.usage = usage;
    peak.occupancyKey = occupancyKey.str();
  }
}

llvm::Error
accumulateTimedPeaks(llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
                     llvm::ArrayRef<ResourceCapacityUseProjection> uses,
                     llvm::StringRef activation, CapacityCatalog &dimensions,
                     std::vector<PeakUsage> &peaks) {
  std::vector<BoundaryChange> changes;
  for (const auto &use : uses) {
    auto resolved = resolvePattern(namespaces, use);
    if (!resolved)
      return resolved.takeError();
    const auto ranks =
        resolved->contract->eventOrder(resolved->pattern.timingAndProgress);
    if (resolved->pattern.acquire.ordinal() >= ranks.size() ||
        resolved->pattern.release.ordinal() >= ranks.size())
      return invalid("Fabric pattern has an incomplete timing relation");
    const std::uint64_t begin = ranks[resolved->pattern.acquire.ordinal()];
    const std::uint64_t release = ranks[resolved->pattern.release.ordinal()];
    const std::uint64_t end = release > begin ? release : begin + 1;
    for (const ::fabric::Claim &claim : resolved->pattern.claims) {
      auto dimension = dimensions.get(use.namespaceOrdinal, resolved->owner,
                                      *resolved->contract, claim);
      if (!dimension)
        return dimension.takeError();
      changes.push_back({*dimension, begin, claim.amount.value(), 0});
      changes.push_back({*dimension, end, 0, claim.amount.value()});
    }
  }
  if (llvm::Error error = initializePeaks(namespaces, dimensions, peaks))
    return error;

  llvm::sort(changes, [](const BoundaryChange &lhs, const BoundaryChange &rhs) {
    return std::tie(lhs.dimension, lhs.rank) <
           std::tie(rhs.dimension, rhs.rank);
  });
  for (std::size_t begin = 0; begin < changes.size();) {
    const std::uint32_t dimension = changes[begin].dimension;
    CapacityCell &cell = dimensions[dimension];
    const std::uint64_t residentUsage = cell.usage;
    std::size_t cursor = begin;
    while (cursor < changes.size() && changes[cursor].dimension == dimension) {
      const std::uint64_t rank = changes[cursor].rank;
      std::uint64_t added = 0;
      std::uint64_t removed = 0;
      while (cursor < changes.size() &&
             changes[cursor].dimension == dimension &&
             changes[cursor].rank == rank) {
        if (llvm::Error error =
                checkedAdd(changes[cursor].added, added, "capacity addition"))
          return error;
        if (llvm::Error error = checkedAdd(changes[cursor].removed, removed,
                                           "capacity removal"))
          return error;
        ++cursor;
      }
      if (removed > cell.usage)
        return invalid("capacity removal exceeds active usage");
      cell.usage -= removed;
      if (llvm::Error error = checkedAdd(added, cell.usage, "capacity usage"))
        return error;
      std::string witnessKey;
      appendU32(witnessKey, 0);
      appendU64(witnessKey, activation.size());
      witnessKey.append(activation.data(), activation.size());
      auto space = resolveNamespace(namespaces, cell.namespaceOrdinal);
      if (!space)
        return space.takeError();
      const std::string cellKey =
          dimensionKey(*space, cell.owner, cell.state, cell.dimension);
      appendU64(witnessKey, cellKey.size());
      witnessKey.append(cellKey);
      appendU64(witnessKey, rank);
      updatePeak(peaks[dimension], cell.usage, witnessKey);
    }
    if (cell.usage != residentUsage)
      return invalid("timed envelope does not release every claim");
    begin = cursor;
  }
  return llvm::Error::success();
}

llvm::Error accumulateRouteUsage(
    llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
    llvm::ArrayRef<ResourceCapacityRouteProjection> routeTraversals,
    CapacityCatalog &dimensions) {
  std::map<std::string, const ::loom::fabric::FabricPhysicalTraversalView *>
      traversalByRef;
  for (const auto &[namespaceOrdinal, space] : llvm::enumerate(namespaces)) {
    if (!space.fabric)
      return invalid("capacity input has an absent physical namespace");
    for (const auto &traversal : space.fabric->physicalTraversals()) {
      std::string key;
      appendU64(key, namespaceOrdinal);
      const std::string local = refKey(traversal.reference);
      appendU64(key, local.size());
      key.append(local);
      if (!traversalByRef.emplace(std::move(key), &traversal).second)
        return invalid("Fabric traversal projection contains a duplicate");
    }
  }

  for (const auto &route : routeTraversals) {
    auto space = resolveNamespace(namespaces, route.namespaceOrdinal);
    if (!space)
      return space.takeError();
    struct SelectedClaim final {
      std::uint32_t dimension = 0;
      std::uint64_t amount = 0;
    };
    std::map<std::string, SelectedClaim> selectedClaims;
    for (const auto &reference : route.traversals) {
      std::string traversalKey;
      appendU64(traversalKey, route.namespaceOrdinal);
      const std::string local = refKey(reference);
      appendU64(traversalKey, local.size());
      traversalKey.append(local);
      auto found = traversalByRef.find(traversalKey);
      if (found == traversalByRef.end())
        return invalid("RouteTree names an absent Fabric traversal");
      for (const auto &use : found->second->impliedUses) {
        const ResourceCapacityUseProjection selected{
            route.namespaceOrdinal, use.pattern, {}};
        auto resolved = resolvePattern(namespaces, selected);
        if (!resolved)
          return resolved.takeError();
        for (const ::fabric::Claim &claim : resolved->pattern.claims) {
          auto dimension =
              dimensions.get(route.namespaceOrdinal, resolved->owner,
                             *resolved->contract, claim);
          if (!dimension)
            return dimension.takeError();
          const std::string key =
              routeClaimKey(*space, use.activationGroup, resolved->owner,
                            claim.state, claim.dimension);
          auto [position, inserted] = selectedClaims.try_emplace(
              key, SelectedClaim{*dimension, claim.amount.value()});
          if (!inserted && (position->second.dimension != *dimension ||
                            position->second.amount != claim.amount.value()))
            return invalid(
                "one route claim key has inconsistent capacity demand");
        }
      }
    }
    for (const auto &[key, claim] : selectedClaims) {
      (void)key;
      if (llvm::Error error =
              checkedAdd(claim.amount, dimensions[claim.dimension].usage,
                         "route capacity usage"))
        return error;
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<ResourceCapacityOveruseProjection> deriveResourceCapacityOveruse(
    llvm::ArrayRef<ResourceCapacityNamespaceView> namespaces,
    llvm::ArrayRef<ResourceCapacityUseProjection> resourceUses,
    llvm::ArrayRef<ResourceCapacityRouteProjection> routeTraversals) {
  std::map<std::string, std::vector<ResourceCapacityUseProjection>>
      usesByActivation;
  for (const auto &use : resourceUses)
    usesByActivation[use.activationKey].push_back(use);

  CapacityCatalog dimensions(namespaces);
  if (llvm::Error error =
          accumulateRouteUsage(namespaces, routeTraversals, dimensions))
    return std::move(error);
  std::vector<PeakUsage> peaks;
  if (llvm::Error error = initializePeaks(namespaces, dimensions, peaks))
    return std::move(error);
  for (const auto &[activation, uses] : usesByActivation) {
    if (llvm::Error error = accumulateTimedPeaks(namespaces, uses, activation,
                                                 dimensions, peaks))
      return std::move(error);
  }

  ResourceCapacityOveruseProjection result;
  for (const auto &[ordinal, cell] : llvm::enumerate(dimensions.cells())) {
    const std::uint64_t usage = peaks[ordinal].usage;
    const std::uint64_t overuse =
        usage > cell.capacity ? usage - cell.capacity : 0;
    if (llvm::Error error = checkedAdd(overuse, result.total, "total overuse"))
      return std::move(error);
    selectFirstWitness(result, cell, usage, peaks[ordinal].occupancyKey);
  }
  return result;
}

} // namespace loom::mapping::detail

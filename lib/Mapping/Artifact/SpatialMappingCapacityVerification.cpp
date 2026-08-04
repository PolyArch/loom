#include "SpatialMappingCapacityVerification.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/IR/UsePatternValue.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "spatial_capacity_verification_invalid: " + message);
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

std::string dimensionKey(const ::loom::fabric::FabricInventoryOwnerRef &owner,
                         ::fabric::StateKey state,
                         ::fabric::CapacityDimensionKey dimension) {
  std::string result = refKey(owner);
  appendU32(result, state.ordinal());
  appendU32(result, dimension.ordinal());
  return result;
}

std::string activationGroupKey(
    const ::loom::fabric::FabricTraversalActivationGroupView &group) {
  std::string result = refKey(group.owner);
  appendU32(result, static_cast<std::uint32_t>(group.kind));
  appendU64(result, group.ordinal);
  return result;
}

std::string
routeClaimKey(const ::loom::fabric::FabricTraversalActivationGroupView &group,
              const ::loom::fabric::FabricInventoryOwnerRef &owner,
              ::fabric::StateKey state,
              ::fabric::CapacityDimensionKey dimension) {
  const std::string activation = activationGroupKey(group);
  const std::string capacity = dimensionKey(owner, state, dimension);
  std::string result;
  appendU64(result, activation.size());
  result.append(activation);
  appendU64(result, capacity.size());
  result.append(capacity);
  return result;
}

llvm::Expected<std::string>
resourceOwnerKey(const ArtifactIdentity &dataflowIdentity,
                 const SpatialResourceOwnerRef &owner) {
  std::string result;
  return std::visit(
      [&](const auto &typed) -> llvm::Expected<std::string> {
        using Owner = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<Owner, SpatialComputeResourceOwnerRef>) {
          appendU32(result, 0);
          appendU64(result, typed.realization);
        } else if constexpr (std::is_same_v<
                                 Owner, SpatialMemoryEngineResourceOwnerRef>) {
          appendU32(result, 1);
          appendU64(result, typed.realization);
        } else if constexpr (std::is_same_v<
                                 Owner, SpatialMemoryBindingResourceOwnerRef>) {
          appendU32(result, 2);
          appendU64(result, typed.binding);
        } else {
          appendU32(result, 3);
          auto logicalNet = ::dataflow::encodeDataflowReference(
              dataflowIdentity, typed.logicalNet);
          if (!logicalNet)
            return logicalNet.takeError();
          appendU64(result, logicalNet->size());
          result.append(reinterpret_cast<const char *>(logicalNet->data()),
                        logicalNet->size());
          appendU64(result, typed.nodeOrdinal);
        }
        return result;
      },
      owner);
}

llvm::Expected<std::string>
activationKey(const ::loom::fabric::FabricArtifactView &fabric,
              const ArtifactIdentity &dataflowIdentity,
              const SpatialResourceUseView &use) {
  auto result = resourceOwnerKey(dataflowIdentity, use.owner);
  if (!result)
    return result.takeError();
  auto event = encodeSpatialActivityEventKey(dataflowIdentity,
                                             use.activation.trigger.event);
  if (!event)
    return event.takeError();
  appendU64(*result, event->size());
  result->append(reinterpret_cast<const char *>(event->data()), event->size());

  const auto owner = use.useSite.owner.catalog();
  const ::fabric::ResourceContract *contract = fabric.resourceContract(owner);
  if (!contract || use.useSite.ordinal >= contract->usePatternCount())
    return invalid("ResourceUse does not resolve a Fabric pattern");
  const ::fabric::UsePattern pattern =
      contract->usePattern(::fabric::UsePatternKey(use.useSite.ordinal));
  if (pattern.parameters.size() != use.parameters.size())
    return invalid("ResourceUse parameter count disagrees with its pattern");
  appendU64(*result, use.parameters.size());
  for (std::size_t ordinal = 0; ordinal < use.parameters.size(); ++ordinal) {
    auto encoded = ::fabric::encodeUsePatternValue(pattern.parameters[ordinal],
                                                   use.parameters[ordinal]);
    if (!encoded)
      return encoded.takeError();
    appendU64(*result, encoded->size());
    result->append(reinterpret_cast<const char *>(encoded->data()),
                   encoded->size());
  }
  return result;
}

llvm::Error checkedAdd(std::uint64_t amount, std::uint64_t &value,
                       llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - value)
    return invalid(subject + " overflows u64");
  value += amount;
  return llvm::Error::success();
}

struct CapacityCell final {
  ::loom::fabric::FabricInventoryOwnerRef owner;
  ::fabric::StateKey state;
  ::fabric::CapacityDimensionKey dimension;
  std::uint64_t capacity = 0;
  std::uint64_t initial = 0;
  std::uint64_t usage = 0;
};

class CapacityCatalog final {
public:
  llvm::Expected<std::uint32_t>
  get(const ::loom::fabric::FabricInventoryOwnerRef &owner,
      const ::fabric::ResourceContract &contract,
      const ::fabric::Claim &claim) {
    if (claim.state.ordinal() >= contract.stateCount())
      return invalid("claim state is outside its Fabric contract");
    const auto dimensions = contract.capacityDimensions(claim.state);
    if (claim.dimension.ordinal() >= dimensions.size())
      return invalid("claim dimension is outside its Fabric state");
    const std::string key = dimensionKey(owner, claim.state, claim.dimension);
    auto found = ordinals_.find(key);
    if (found != ordinals_.end())
      return found->second;
    if (cells_.size() >= std::numeric_limits<std::uint32_t>::max())
      return invalid("capacity dimension inventory exceeds u32");
    const auto &dimension = dimensions[claim.dimension.ordinal()];
    const std::uint32_t ordinal = static_cast<std::uint32_t>(cells_.size());
    ordinals_.try_emplace(key, ordinal);
    cells_.push_back({owner, claim.state, claim.dimension,
                      dimension.capacity.value(),
                      dimension.initialOccupancy.value(),
                      dimension.initialOccupancy.value()});
    return ordinal;
  }

  CapacityCell &operator[](std::uint32_t ordinal) { return cells_[ordinal]; }
  llvm::ArrayRef<CapacityCell> cells() const { return cells_; }

private:
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

void selectFirstWitness(SpatialCapacityOveruseProjection &projection,
                        const CapacityCell &cell, std::uint64_t usage,
                        llvm::StringRef occupancyKey) {
  if (usage <= cell.capacity)
    return;
  SpatialCapacityOveruseWitness witness{
      cell.owner, cell.state,    cell.dimension,
      usage,      cell.capacity, unsignedBytes(occupancyKey)};
  if (!projection.firstWitness ||
      witness.canonicalOccupancyKey <
          projection.firstWitness->canonicalOccupancyKey)
    projection.firstWitness = std::move(witness);
}

llvm::Error mergeProjection(SpatialCapacityOveruseProjection source,
                            SpatialCapacityOveruseProjection &target,
                            llvm::StringRef subject) {
  if (llvm::Error error = checkedAdd(source.total, target.total, subject))
    return error;
  if (source.firstWitness &&
      (!target.firstWitness || source.firstWitness->canonicalOccupancyKey <
                                   target.firstWitness->canonicalOccupancyKey))
    target.firstWitness = std::move(source.firstWitness);
  return llvm::Error::success();
}

llvm::Expected<SpatialCapacityOveruseProjection> timedEnvelopeOveruse(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<::loom::fabric::FabricUsePatternRef> patterns,
    llvm::StringRef activation) {
  CapacityCatalog dimensions;
  std::vector<BoundaryChange> changes;
  for (const auto &reference : patterns) {
    const auto owner = reference.owner.catalog();
    const ::fabric::ResourceContract *contract = fabric.resourceContract(owner);
    if (!contract || reference.ordinal >= contract->usePatternCount())
      return invalid("ResourceUse does not resolve a Fabric pattern");
    const ::fabric::UsePattern pattern =
        contract->usePattern(::fabric::UsePatternKey(reference.ordinal));
    const auto ranks = contract->eventOrder(pattern.timingAndProgress);
    if (pattern.acquire.ordinal() >= ranks.size() ||
        pattern.release.ordinal() >= ranks.size())
      return invalid("Fabric pattern has an incomplete timing relation");
    const std::uint64_t begin = ranks[pattern.acquire.ordinal()];
    const std::uint64_t release = ranks[pattern.release.ordinal()];
    const std::uint64_t end = release > begin ? release : begin + 1;
    for (const ::fabric::Claim &claim : pattern.claims) {
      auto dimension = dimensions.get(owner, *contract, claim);
      if (!dimension)
        return dimension.takeError();
      changes.push_back({*dimension, begin, claim.amount.value(), 0});
      changes.push_back({*dimension, end, 0, claim.amount.value()});
    }
  }

  std::sort(changes.begin(), changes.end(),
            [](const BoundaryChange &lhs, const BoundaryChange &rhs) {
              return std::tie(lhs.dimension, lhs.rank) <
                     std::tie(rhs.dimension, rhs.rank);
            });
  SpatialCapacityOveruseProjection result;
  for (std::size_t begin = 0; begin < changes.size();) {
    const std::uint32_t dimension = changes[begin].dimension;
    CapacityCell &cell = dimensions[dimension];
    std::uint64_t maximum = 0;
    std::uint64_t maximumUsage = cell.usage;
    std::uint64_t maximumRank = 0;
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
          return std::move(error);
        if (llvm::Error error = checkedAdd(changes[cursor].removed, removed,
                                           "capacity removal"))
          return std::move(error);
        ++cursor;
      }
      if (removed > cell.usage)
        return invalid("capacity removal exceeds active usage");
      cell.usage -= removed;
      if (llvm::Error error = checkedAdd(added, cell.usage, "capacity usage"))
        return std::move(error);
      const std::uint64_t overuse =
          cell.usage > cell.capacity ? cell.usage - cell.capacity : 0;
      if (overuse > maximum) {
        maximum = overuse;
        maximumUsage = cell.usage;
        maximumRank = rank;
      }
    }
    if (cell.usage != cell.initial)
      return invalid("timed envelope does not release every claim");
    if (llvm::Error error =
            checkedAdd(maximum, result.total, "capacity overuse"))
      return std::move(error);
    if (maximum != 0) {
      std::string witnessKey;
      appendU32(witnessKey, 0);
      appendU64(witnessKey, activation.size());
      witnessKey.append(activation.data(), activation.size());
      const std::string cellKey =
          dimensionKey(cell.owner, cell.state, cell.dimension);
      appendU64(witnessKey, cellKey.size());
      witnessKey.append(cellKey);
      appendU64(witnessKey, maximumRank);
      selectFirstWitness(result, cell, maximumUsage, witnessKey);
    }
    begin = cursor;
  }
  return result;
}

llvm::Expected<SpatialCapacityOveruseProjection> routeOveruse(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<std::vector<::loom::fabric::FabricPhysicalTraversalRef>>
        routeTraversals) {
  llvm::StringMap<const ::loom::fabric::FabricPhysicalTraversalView *>
      traversalByRef;
  for (const auto &traversal : fabric.physicalTraversals())
    if (!traversalByRef.try_emplace(refKey(traversal.reference), &traversal)
             .second)
      return invalid("Fabric traversal projection contains a duplicate");

  CapacityCatalog dimensions;
  for (const auto &traversals : routeTraversals) {
    struct SelectedClaim final {
      std::uint32_t dimension = 0;
      std::uint64_t amount = 0;
    };
    std::map<std::string, SelectedClaim> selectedClaims;
    for (const auto &reference : traversals) {
      auto found = traversalByRef.find(refKey(reference));
      if (found == traversalByRef.end())
        return invalid("RouteTree names an absent Fabric traversal");
      for (const auto &use : found->second->impliedUses) {
        const auto owner = use.pattern.owner.catalog();
        const ::fabric::ResourceContract *contract =
            fabric.resourceContract(owner);
        if (!contract || use.pattern.ordinal >= contract->usePatternCount())
          return invalid("route use does not resolve a Fabric pattern");
        const ::fabric::UsePattern pattern =
            contract->usePattern(::fabric::UsePatternKey(use.pattern.ordinal));
        for (const ::fabric::Claim &claim : pattern.claims) {
          auto dimension = dimensions.get(owner, *contract, claim);
          if (!dimension)
            return dimension.takeError();
          const std::string key = routeClaimKey(use.activationGroup, owner,
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
        return std::move(error);
    }
  }

  SpatialCapacityOveruseProjection result;
  for (const CapacityCell &cell : dimensions.cells()) {
    const std::uint64_t overuse =
        cell.usage > cell.capacity ? cell.usage - cell.capacity : 0;
    if (llvm::Error error = checkedAdd(overuse, result.total, "route overuse"))
      return std::move(error);
    if (overuse != 0) {
      std::string witnessKey;
      appendU32(witnessKey, 1);
      const std::string cellKey =
          dimensionKey(cell.owner, cell.state, cell.dimension);
      appendU64(witnessKey, cellKey.size());
      witnessKey.append(cellKey);
      selectFirstWitness(result, cell, cell.usage, witnessKey);
    }
  }
  return result;
}

} // namespace

llvm::Expected<SpatialCapacityOveruseProjection> deriveSpatialCapacityOveruse(
    const ::loom::fabric::FabricArtifactView &fabric,
    const ArtifactIdentity &dataflowIdentity,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<std::vector<::loom::fabric::FabricPhysicalTraversalRef>>
        routeTraversals) {
  std::map<std::string, std::vector<::loom::fabric::FabricUsePatternRef>>
      patternsByActivation;
  for (const SpatialResourceUseView &use : resourceUses) {
    auto key = activationKey(fabric, dataflowIdentity, use);
    if (!key)
      return key.takeError();
    patternsByActivation[*key].push_back(use.useSite);
  }

  SpatialCapacityOveruseProjection result;
  for (const auto &[activation, patterns] : patternsByActivation) {
    auto overuse = timedEnvelopeOveruse(fabric, patterns, activation);
    if (!overuse)
      return overuse.takeError();
    if (llvm::Error error =
            mergeProjection(std::move(*overuse), result, "atomic overuse"))
      return std::move(error);
  }
  auto route = routeOveruse(fabric, routeTraversals);
  if (!route)
    return route.takeError();
  if (llvm::Error error =
          mergeProjection(std::move(*route), result, "total overuse"))
    return std::move(error);
  return result;
}

} // namespace loom::mapping::detail

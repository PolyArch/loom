#include "PnR/EndpointRoutingTopology.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "PnR/RoutingNegotiation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenEndpointRoutingTopology";
constexpr PnrCapacityContext endpointContext{
    frozenArtifact, "routing_endpoints", "endpoint", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext traversalContext{frozenArtifact,
                                              "routing_traversals", "traversal",
                                              PnrCapacityMeasure::Index};
constexpr PnrCapacityContext traversalEndpointContext{
    frozenArtifact, "routing_traversals", "endpoint",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext replicationGroupContext{
    frozenArtifact, "replication_groups", "group", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext arcContext{frozenArtifact, "routing_arcs", "arc",
                                        PnrCapacityMeasure::Index};
constexpr PnrCapacityContext capacityCellContext{
    frozenArtifact, "capacity", "cell", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext capacityActivationContext{
    frozenArtifact, "capacity", "activation", PnrCapacityMeasure::Index};
constexpr PnrCapacityContext capacityClaimContext{
    frozenArtifact, "capacity", "claim", PnrCapacityMeasure::Offset};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "endpoint_routing_topology_invalid: " +
                                     message);
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename Ref> std::string refKey(const Ref &reference) {
  return byteKey(canonicalFabricBytes(reference));
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU32Be(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (unsigned shift = 24; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

std::string capacityCellKey(const FabricInventoryOwnerRef &owner,
                            ::fabric::StateKey state,
                            ::fabric::CapacityDimensionKey dimension) {
  std::vector<std::uint8_t> bytes = canonicalFabricBytes(owner);
  appendU32Be(bytes, state.ordinal());
  appendU32Be(bytes, dimension.ordinal());
  return byteKey(bytes);
}

std::string requesterKey(const FabricTraversalRequesterGroupView &requester) {
  std::vector<std::uint8_t> bytes = canonicalFabricBytes(requester.owner);
  appendU32Be(bytes, static_cast<std::uint32_t>(requester.kind));
  appendU64Be(bytes, requester.ordinal);
  return byteKey(bytes);
}

std::string switchReplicationKey(const FabricSwitchTraversalPayload &payload) {
  std::vector<std::uint8_t> bytes;
  const auto owner = canonicalFabricBytes(payload.owner);
  bytes.reserve(9 + owner.size());
  bytes.push_back(0);
  appendU64Be(bytes, payload.input);
  bytes.insert(bytes.end(), owner.begin(), owner.end());
  return byteKey(bytes);
}

std::string
transferPatternReplicationKey(const FabricTransferPatternLegPayload &payload) {
  std::vector<std::uint8_t> bytes{1};
  const auto owner = canonicalFabricBytes(payload.owner);
  bytes.insert(bytes.end(), owner.begin(), owner.end());
  return byteKey(bytes);
}

std::uint32_t tagCapacity(const ::fabric::DataPathType &path) {
  return path.kind == ::fabric::DataPathKind::BitsTag ? path.tagWidthBits : 0;
}

llvm::Expected<PnrIndex> checked(PnrCapacityContext context,
                                 std::size_t value) {
  return checkedPnrIndex(context, static_cast<std::uint64_t>(value));
}

} // namespace

llvm::Expected<FrozenEndpointRoutingTopology>
loom::pnr::freezeEndpointRoutingTopology(const FabricArtifactView &fabric) {
  FrozenEndpointRoutingTopology result;
  const auto endpointRefs = fabric.transportEndpoints();
  const auto traversalViews = fabric.physicalTraversals();
  if (llvm::Error error =
          preflightPnrIndexCapacity(endpointContext, endpointRefs.size()))
    return std::move(error);
  if (llvm::Error error =
          preflightPnrIndexCapacity(traversalContext, traversalViews.size()))
    return std::move(error);

  result.endpoints_.reserve(endpointRefs.size());
  for (auto [ordinal, reference] : llvm::enumerate(endpointRefs)) {
    const auto direction = fabric.transportEndpointDirection(reference);
    const auto dataPath = fabric.transportEndpointDataPath(reference);
    if (!direction || !dataPath)
      return invalid("a canonical endpoint has no typed projection");
    auto index = checked(endpointContext, ordinal);
    if (!index)
      return index.takeError();
    if (!result.endpointOrdinals_.try_emplace(refKey(reference), *index)
             .second)
      return invalid("the canonical endpoint inventory has a duplicate");
    result.endpoints_.push_back({reference, *direction, *dataPath});
  }

  for (auto [ordinal, traversal] : llvm::enumerate(traversalViews)) {
    auto index = checked(traversalContext, ordinal);
    if (!index)
      return index.takeError();
    if (!result.traversalOrdinals_
             .try_emplace(refKey(traversal.reference), *index)
             .second)
      return invalid("the canonical traversal inventory has a duplicate");
  }

  struct ArcDraft final {
    PnrIndex source = 0;
    EndpointRoutingArc arc;
  };
  std::vector<ArcDraft> arcDrafts;
  std::size_t arcDraftCapacity = 0;
  bool reserveArcDrafts = true;
  for (const auto &traversal : traversalViews) {
    if (traversal.destinations.size() != 0 &&
        traversal.sources.size() > std::numeric_limits<std::size_t>::max() /
                                       traversal.destinations.size()) {
      reserveArcDrafts = false;
      break;
    }
    const std::size_t arcProduct =
        traversal.sources.size() * traversal.destinations.size();
    if (arcProduct >
        std::numeric_limits<std::size_t>::max() - arcDraftCapacity) {
      reserveArcDrafts = false;
      break;
    }
    arcDraftCapacity += arcProduct;
    if constexpr (sizeof(PnrIndex) < sizeof(std::size_t))
      if (arcDraftCapacity >
          static_cast<std::size_t>(std::numeric_limits<PnrIndex>::max()))
        reserveArcDrafts = false;
  }
  if (reserveArcDrafts)
    arcDrafts.reserve(arcDraftCapacity);
  llvm::StringMap<PnrIndex> replicationGroups;
  llvm::StringMap<PnrIndex> capacityCells;
  llvm::StringMap<PnrIndex> capacityActivations;
  result.traversals_.reserve(traversalViews.size());
  result.traversalReplicationGroups_.reserve(traversalViews.size());
  for (auto [ordinal, traversal] : llvm::enumerate(traversalViews)) {
    auto traversalIndex = checked(traversalContext, ordinal);
    if (!traversalIndex)
      return traversalIndex.takeError();
    auto sourceOffset =
        checked(traversalEndpointContext, result.traversalEndpoints_.size());
    if (!sourceOffset)
      return sourceOffset.takeError();
    std::vector<PnrIndex> sources;
    sources.reserve(traversal.sources.size());
    for (const FabricTransportEndpointRef &source : traversal.sources) {
      const auto found = result.endpointOrdinal(source);
      if (!found)
        return invalid(
            "a traversal source is absent from the endpoint inventory");
      sources.push_back(*found);
      result.traversalEndpoints_.push_back(*found);
    }
    auto sourceCount = checked(traversalEndpointContext, sources.size());
    if (!sourceCount)
      return sourceCount.takeError();
    auto destinationOffset =
        checked(traversalEndpointContext, result.traversalEndpoints_.size());
    if (!destinationOffset)
      return destinationOffset.takeError();
    std::vector<PnrIndex> destinations;
    destinations.reserve(traversal.destinations.size());
    for (const FabricTransportEndpointRef &destination :
         traversal.destinations) {
      const auto found = result.endpointOrdinal(destination);
      if (!found)
        return invalid(
            "a traversal destination is absent from the endpoint inventory");
      destinations.push_back(*found);
      result.traversalEndpoints_.push_back(*found);
    }
    auto destinationCount =
        checked(traversalEndpointContext, destinations.size());
    if (!destinationCount)
      return destinationCount.takeError();

    PnrIndex replicationGroup = getInvalidPnrIndex();
    std::string replicationKey;
    const auto *switchPayload =
        std::get_if<FabricSwitchTraversalPayload>(&traversal.reference.payload);
    const auto *transferPatternPayload =
        std::get_if<FabricTransferPatternLegPayload>(
            &traversal.reference.payload);
    if (switchPayload)
      replicationKey = switchReplicationKey(*switchPayload);
    else if (transferPatternPayload)
      replicationKey = transferPatternReplicationKey(*transferPatternPayload);
    if (!replicationKey.empty()) {
      auto found = replicationGroups.find(replicationKey);
      if (found == replicationGroups.end()) {
        auto group = checked(replicationGroupContext, replicationGroups.size());
        if (!group)
          return group.takeError();
        found = replicationGroups.try_emplace(replicationKey, *group).first;
      }
      replicationGroup = found->second;
    }
    auto capacityClaimOffset =
        checked(capacityClaimContext, result.capacityClaims_.size());
    if (!capacityClaimOffset)
      return capacityClaimOffset.takeError();
    std::map<std::pair<PnrIndex, PnrIndex>, std::uint64_t> traversalClaims;
    for (const FabricTraversalUseView &use : traversal.impliedUses) {
      const FabricInventoryOwnerRef owner = use.pattern.owner.catalog();
      const ::fabric::ResourceContract *contract =
          fabric.resourceContract(owner);
      if (!contract || use.pattern.ordinal >= contract->usePatternCount())
        return invalid("a traversal use does not resolve its Fabric pattern");
      const ::fabric::UsePattern pattern =
          contract->usePattern(::fabric::UsePatternKey(use.pattern.ordinal));
      if (use.requesterGroup.kind ==
          FabricTraversalRequesterGroupKind::SwitchRequester)
        if (!switchPayload ||
            use.requesterGroup.owner !=
                FabricInventoryOwnerRef::of(switchPayload->owner) ||
            use.requesterGroup.ordinal != pattern.requester.ordinal())
          return invalid("a switch requester group disagrees with its pattern");
      if (use.occupancyKind == FabricTraversalUseOccupancyKind::RuntimeService)
        continue;
      const std::string requester = requesterKey(use.requesterGroup);
      auto activationPosition = capacityActivations.find(requester);
      if (activationPosition == capacityActivations.end()) {
        auto index =
            checked(capacityActivationContext, capacityActivations.size());
        if (!index)
          return index.takeError();
        activationPosition =
            capacityActivations.try_emplace(requester, *index).first;
      }
      for (const ::fabric::Claim &claim : pattern.claims) {
        if (claim.state.ordinal() >= contract->stateCount())
          return invalid("a traversal claim has an invalid Fabric state");
        const auto dimensions = contract->capacityDimensions(claim.state);
        if (claim.dimension.ordinal() >= dimensions.size())
          return invalid("a traversal claim has an invalid capacity dimension");
        const auto &dimension = dimensions[claim.dimension.ordinal()];
        const std::string cell =
            capacityCellKey(owner, claim.state, claim.dimension);
        auto cellPosition = capacityCells.find(cell);
        if (cellPosition == capacityCells.end()) {
          auto index = checked(capacityCellContext, capacityCells.size());
          if (!index)
            return index.takeError();
          cellPosition = capacityCells.try_emplace(cell, *index).first;
          result.capacityCells_.push_back({owner, claim.state, claim.dimension,
                                           dimension.capacity.value(),
                                           dimension.initialOccupancy.value()});
        } else {
          const auto &existing = result.capacityCells_[cellPosition->second];
          if (existing.capacity != dimension.capacity.value() ||
              existing.initialOccupancy != dimension.initialOccupancy.value())
            return invalid("one capacity cell has inconsistent Fabric values");
        }
        auto [position, inserted] = traversalClaims.try_emplace(
            std::make_pair(activationPosition->second, cellPosition->second),
            claim.amount.value());
        if (!inserted && position->second != claim.amount.value())
          return invalid("one traversal activation has inconsistent claims");
      }
    }
    for (const auto &[key, amount] : traversalClaims) {
      const auto &cell = result.capacityCells_[key.second];
      auto qCost = normalizedRouteClaimCost(amount, cell.capacity);
      if (!qCost)
        return qCost.takeError();
      result.capacityClaims_.push_back({key.second, key.first, amount, *qCost});
    }
    auto capacityClaimCount =
        checked(capacityClaimContext, traversalClaims.size());
    if (!capacityClaimCount)
      return capacityClaimCount.takeError();
    result.traversalReplicationGroups_.push_back(replicationGroup);
    result.traversals_.push_back(
        {traversal.reference, *sourceOffset, *sourceCount, *destinationOffset,
         *destinationCount, *capacityClaimOffset, *capacityClaimCount,
         traversal.timing.architecturalLatencyCycles});

    auto arcProduct = checkedPnrIndexMultiply(arcContext, sources.size(),
                                              destinations.size());
    if (!arcProduct)
      return arcProduct.takeError();
    auto arcEnd = checkedPnrIndexAdd(arcContext, arcDrafts.size(), *arcProduct);
    if (!arcEnd)
      return arcEnd.takeError();
    for (PnrIndex source : sources) {
      const auto &sourcePath = result.endpoints_[source].dataPath;
      for (PnrIndex destination : destinations) {
        const auto &destinationPath = result.endpoints_[destination].dataPath;
        arcDrafts.push_back({source,
                             {destination, *traversalIndex,
                              std::min(sourcePath.payloadWidthBits,
                                       destinationPath.payloadWidthBits),
                              std::min(tagCapacity(sourcePath),
                                       tagCapacity(destinationPath))}});
      }
    }
  }

  llvm::sort(arcDrafts, [](const ArcDraft &lhs, const ArcDraft &rhs) {
    return std::tie(lhs.source, lhs.arc.target, lhs.arc.traversal) <
           std::tie(rhs.source, rhs.arc.target, rhs.arc.traversal);
  });
  result.adjacencyOffsets_.reserve(result.endpoints_.size() + 1);
  std::size_t cursor = 0;
  for (std::size_t source = 0; source < result.endpoints_.size(); ++source) {
    auto offset = checked(arcContext, result.arcs_.size());
    if (!offset)
      return offset.takeError();
    result.adjacencyOffsets_.push_back(*offset);
    while (cursor < arcDrafts.size() && arcDrafts[cursor].source == source) {
      result.arcSources_.push_back(arcDrafts[cursor].source);
      result.arcs_.push_back(arcDrafts[cursor++].arc);
    }
  }
  auto arcEnd = checked(arcContext, result.arcs_.size());
  if (!arcEnd)
    return arcEnd.takeError();
  result.adjacencyOffsets_.push_back(*arcEnd);
  if (cursor != arcDrafts.size())
    return invalid("routing CSR retained an out-of-range source");

  result.reverseAdjacencyOffsets_.assign(result.endpoints_.size() + 1, 0);
  for (const EndpointRoutingArc &arc : result.arcs_) {
    auto count = checkedPnrIndexAdd(
        arcContext, result.reverseAdjacencyOffsets_[arc.target + 1], 1);
    if (!count)
      return count.takeError();
    result.reverseAdjacencyOffsets_[arc.target + 1] = *count;
  }
  for (std::size_t endpoint = 1;
       endpoint < result.reverseAdjacencyOffsets_.size(); ++endpoint) {
    auto prefix = checkedPnrIndexAdd(
        arcContext, result.reverseAdjacencyOffsets_[endpoint - 1],
        result.reverseAdjacencyOffsets_[endpoint]);
    if (!prefix)
      return prefix.takeError();
    result.reverseAdjacencyOffsets_[endpoint] = *prefix;
  }
  result.reverseArcOrdinals_.resize(result.arcs_.size());
  std::vector<PnrIndex> reverseCursors = result.reverseAdjacencyOffsets_;
  for (auto [ordinal, arc] : llvm::enumerate(result.arcs_)) {
    auto arcIndex = checked(arcContext, ordinal);
    if (!arcIndex)
      return arcIndex.takeError();
    result.reverseArcOrdinals_[reverseCursors[arc.target]++] = *arcIndex;
  }
  return result;
}

std::optional<PnrIndex> FrozenEndpointRoutingTopology::endpointOrdinal(
    const ::loom::fabric::FabricTransportEndpointRef &reference) const {
  const auto found = endpointOrdinals_.find(refKey(reference));
  if (found == endpointOrdinals_.end())
    return std::nullopt;
  return found->second;
}

std::optional<PnrIndex> FrozenEndpointRoutingTopology::traversalOrdinal(
    const ::loom::fabric::FabricPhysicalTraversalRef &reference) const {
  const auto found = traversalOrdinals_.find(refKey(reference));
  if (found == traversalOrdinals_.end())
    return std::nullopt;
  return found->second;
}

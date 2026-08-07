#include "PnR/EndpointRoutingTopology.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
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

bool matchesSwitchRequester(
    const FabricTraversalActivationGroupView &activation,
    const FabricSwitchTraversalPayload &payload) {
  return activation.kind ==
             FabricTraversalActivationGroupKind::SwitchRequester &&
         activation.owner == FabricInventoryOwnerRef::of(payload.owner) &&
         activation.ordinal == payload.input;
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

  llvm::StringMap<PnrIndex> endpointOrdinals;
  result.endpoints_.reserve(endpointRefs.size());
  for (auto [ordinal, reference] : llvm::enumerate(endpointRefs)) {
    const auto direction = fabric.transportEndpointDirection(reference);
    const auto dataPath = fabric.transportEndpointDataPath(reference);
    if (!direction || !dataPath)
      return invalid("a canonical endpoint has no typed projection");
    auto index = checked(endpointContext, ordinal);
    if (!index)
      return index.takeError();
    if (!endpointOrdinals.try_emplace(refKey(reference), *index).second)
      return invalid("the canonical endpoint inventory has a duplicate");
    result.endpoints_.push_back({reference, *direction, *dataPath});
  }

  struct ArcDraft final {
    PnrIndex source = 0;
    EndpointRoutingArc arc;
  };
  std::vector<ArcDraft> arcDrafts;
  llvm::StringMap<PnrIndex> replicationGroups;
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
      auto found = endpointOrdinals.find(refKey(source));
      if (found == endpointOrdinals.end())
        return invalid(
            "a traversal source is absent from the endpoint inventory");
      sources.push_back(found->second);
      result.traversalEndpoints_.push_back(found->second);
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
      auto found = endpointOrdinals.find(refKey(destination));
      if (found == endpointOrdinals.end())
        return invalid(
            "a traversal destination is absent from the endpoint inventory");
      destinations.push_back(found->second);
      result.traversalEndpoints_.push_back(found->second);
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
    for (const FabricTraversalUseView &use : traversal.impliedUses)
      if (use.activationGroup.kind ==
          FabricTraversalActivationGroupKind::SwitchRequester)
        if (!switchPayload ||
            !matchesSwitchRequester(use.activationGroup, *switchPayload))
          return invalid(
              "a switch requester activation disagrees with its traversal");
    result.traversalReplicationGroups_.push_back(replicationGroup);
    result.traversals_.push_back({traversal.reference, *sourceOffset,
                                  *sourceCount, *destinationOffset,
                                  *destinationCount});

    auto arcProduct = checkedPnrIndexMultiply(arcContext, sources.size(),
                                              destinations.size());
    if (!arcProduct)
      return arcProduct.takeError();
    auto arcEnd = checkedPnrIndexAdd(arcContext, arcDrafts.size(), *arcProduct);
    if (!arcEnd)
      return arcEnd.takeError();
    arcDrafts.reserve(*arcEnd);
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

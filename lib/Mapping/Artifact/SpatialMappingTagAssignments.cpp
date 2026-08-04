#include "SpatialMappingTagAssignments.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "mapping_artifact_invalid: Physical Tag projection: " + message);
}

llvm::Expected<std::optional<std::uint32_t>>
physicalTagWidth(const ::loom::fabric::FabricArtifactView &fabric,
                 const ::loom::fabric::FabricTransportEndpointRef &endpoint) {
  auto path = fabric.transportEndpointDataPath(endpoint);
  if (!path)
    return invalid("RouteTree endpoint has no Fabric data-path projection");
  if (path->kind != ::fabric::DataPathKind::BitsTag)
    return std::optional<std::uint32_t>();
  return std::optional<std::uint32_t>(path->tagWidthBits);
}

llvm::Expected<SpatialResourceOwnerRef>
routeSourceOwner(const TechMappingView &techMapping,
                 const SpatialRouteTreeView &route) {
  if (std::holds_alternative<::dataflow::GraphIngressTokenRef>(
          route.logicalNet))
    return SpatialResourceOwnerRef(
        SpatialRouteNodeResourceOwnerRef{route.logicalNet, 0});

  const auto &producer =
      std::get<::dataflow::ActorTokenResultRef>(route.logicalNet);
  std::optional<SpatialResourceOwnerRef> result;
  for (const auto &realization : techMapping.computeRealizations())
    if (llvm::any_of(realization.actors, [&](const auto &actor) {
          return actor.actor == producer.actor;
        })) {
      if (result)
        return invalid("route producer belongs to multiple realizations");
      result = SpatialComputeResourceOwnerRef{realization.entityId};
    }
  for (const auto &realization : techMapping.memoryRealizations())
    if (llvm::any_of(realization.actors, [&](const auto &actor) {
          return actor.actor == producer.actor;
        })) {
      if (result)
        return invalid("route producer belongs to multiple realizations");
      result = SpatialMemoryEngineResourceOwnerRef{realization.entityId};
    }
  if (!result)
    return invalid("route producer has no realization owner");
  return std::move(*result);
}

} // namespace

llvm::Expected<std::string>
physicalTagUseKey(const SpatialResourceOwnerRef &owner,
                  const SpatialActivityEventRef &trigger,
                  const ::loom::fabric::FabricUsePatternRef &pattern,
                  const ArtifactIdentity &dataflowIdentity) {
  std::string result;
  const auto appendU64 = [&](std::uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte)
      result.push_back(static_cast<char>(value >> (8 * (7 - byte))));
  };
  const auto appendFramed = [&](llvm::ArrayRef<std::uint8_t> bytes) {
    appendU64(bytes.size());
    result.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  };
  llvm::Error ownerError = llvm::Error::success();
  std::visit(
      [&](const auto &selected) {
        using Owner = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<Owner, SpatialComputeResourceOwnerRef>) {
          result.push_back(0);
          appendU64(selected.realization);
        } else if constexpr (std::is_same_v<
                                 Owner, SpatialMemoryEngineResourceOwnerRef>) {
          result.push_back(1);
          appendU64(selected.realization);
        } else if constexpr (std::is_same_v<
                                 Owner, SpatialMemoryBindingResourceOwnerRef>) {
          result.push_back(2);
          appendU64(selected.binding);
        } else {
          result.push_back(3);
          auto logicalNet = ::dataflow::encodeDataflowReference(
              dataflowIdentity, selected.logicalNet);
          if (!logicalNet) {
            ownerError = logicalNet.takeError();
            return;
          }
          appendFramed(*logicalNet);
          appendU64(selected.nodeOrdinal);
        }
      },
      owner);
  if (ownerError)
    return std::move(ownerError);
  auto encodedEvent = encodeSpatialActivityEventKey(dataflowIdentity, trigger);
  if (!encodedEvent)
    return encodedEvent.takeError();
  appendFramed(*encodedEvent);
  appendFramed(::loom::fabric::canonicalFabricBytes(pattern));
  return result;
}

llvm::Expected<std::map<std::string, RequiredPhysicalTagUse>>
deriveRequiredPhysicalTagUses(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes) {
  std::map<std::string, RequiredPhysicalTagUse> result;
  for (auto indexedRoute : llvm::enumerate(routes)) {
    const std::uint64_t routeTreeOrdinal = indexedRoute.index();
    const SpatialRouteTreeView &route = indexedRoute.value();
    if (route.nodes.empty() || route.nodes.front().ordinal != 0 ||
        route.nodes.front().parentOrdinal ||
        route.nodes.front().endpoint != route.rootEndpoint)
      return invalid("RouteTree has no canonical root node");

    std::vector<RequiredPhysicalTagUse> segments;
    std::vector<std::optional<std::size_t>> nodeSegments(route.nodes.size());
    const auto appendSegment =
        [&](const ::loom::fabric::FabricTransportEndpointRef &endpoint,
            ::loom::fabric::FabricPhysicalTagAssignmentPointKind expectedKind,
            SpatialResourceOwnerRef owner,
            std::uint32_t width) -> llvm::Expected<std::size_t> {
      auto point = fabric.physicalTagAssignmentPoint(endpoint);
      if (!point || point->kind != expectedKind || point->tagWidthBits != width)
        return invalid(
            "continuity origin has no exact Fabric assignment point");
      segments.push_back(
          RequiredPhysicalTagUse{std::move(owner),
                                 SpatialActivityEventRef(route.logicalNet),
                                 *point,
                                 {},
                                 routeTreeOrdinal,
                                 segments.size(),
                                 {}});
      return segments.size() - 1;
    };
    const auto addMatchDomain = [&](std::size_t segment,
                                    const SpatialRouteNodeView &node) {
      auto domain = fabric.transportEndpointTagMatchDomain(node.endpoint);
      if (domain)
        segments[segment].matchDomains.push_back(*domain);
    };

    auto rootWidth = physicalTagWidth(fabric, route.rootEndpoint);
    if (!rootWidth)
      return rootWidth.takeError();
    if (*rootWidth) {
      auto owner = routeSourceOwner(techMapping, route);
      if (!owner)
        return owner.takeError();
      const auto expectedKind =
          std::holds_alternative<::dataflow::GraphIngressTokenRef>(
              route.logicalNet)
              ? ::loom::fabric::FabricPhysicalTagAssignmentPointKind::Ingress
              : ::loom::fabric::FabricPhysicalTagAssignmentPointKind::Writer;
      auto segment = appendSegment(route.rootEndpoint, expectedKind,
                                   std::move(*owner), **rootWidth);
      if (!segment)
        return segment.takeError();
      nodeSegments[0] = *segment;
      addMatchDomain(*segment, route.nodes.front());
    }

    for (std::size_t ordinal = 1; ordinal < route.nodes.size(); ++ordinal) {
      const SpatialRouteNodeView &node = route.nodes[ordinal];
      if (node.ordinal != ordinal || !node.parentOrdinal ||
          *node.parentOrdinal >= ordinal || !node.incomingTraversal)
        return invalid("RouteTree node is not in canonical preorder");
      const SpatialRouteNodeView &parent = route.nodes[*node.parentOrdinal];
      auto sourceWidth = physicalTagWidth(fabric, parent.endpoint);
      if (!sourceWidth)
        return sourceWidth.takeError();
      auto destinationWidth = physicalTagWidth(fabric, node.endpoint);
      if (!destinationWidth)
        return destinationWidth.takeError();

      std::optional<::loom::fabric::FabricBoundaryTagContinuityPointView>
          boundaryPoint;
      if (const auto *boundary =
              std::get_if<::loom::fabric::FabricBoundaryTraversalPayload>(
                  &node.incomingTraversal->payload))
        boundaryPoint = fabric.boundaryTagContinuityPoint(boundary->owner);

      std::optional<std::size_t> childSegment;
      if (!boundaryPoint) {
        if (sourceWidth->has_value() != destinationWidth->has_value() ||
            (*sourceWidth && **sourceWidth != **destinationWidth))
          return invalid("non-boundary traversal changes tag shape");
        childSegment = nodeSegments[*node.parentOrdinal];
      } else {
        using Kind = ::loom::fabric::FabricBoundaryTagContinuityKind;
        switch (boundaryPoint->kind) {
        case Kind::TokenWriter:
        case Kind::ConfigurableWriter: {
          if (*sourceWidth || !*destinationWidth ||
              nodeSegments[*node.parentOrdinal] ||
              boundaryPoint->inputTagWidthBits != 0 ||
              boundaryPoint->outputTagWidthBits != **destinationWidth)
            return invalid("tag writer has inconsistent endpoints");
          auto segment = appendSegment(
              node.endpoint,
              ::loom::fabric::FabricPhysicalTagAssignmentPointKind::Writer,
              SpatialRouteNodeResourceOwnerRef{route.logicalNet, ordinal},
              **destinationWidth);
          if (!segment)
            return segment.takeError();
          childSegment = *segment;
          break;
        }
        case Kind::Rewriter: {
          if (!*sourceWidth || !*destinationWidth ||
              !nodeSegments[*node.parentOrdinal] ||
              boundaryPoint->inputTagWidthBits != **sourceWidth ||
              boundaryPoint->outputTagWidthBits != **destinationWidth)
            return invalid("tag rewriter has inconsistent endpoints");
          auto segment = appendSegment(
              node.endpoint,
              ::loom::fabric::FabricPhysicalTagAssignmentPointKind::Writer,
              SpatialRouteNodeResourceOwnerRef{route.logicalNet, ordinal},
              **destinationWidth);
          if (!segment)
            return segment.takeError();
          childSegment = *segment;
          break;
        }
        case Kind::Remover:
          if (!*sourceWidth || *destinationWidth ||
              !nodeSegments[*node.parentOrdinal] ||
              boundaryPoint->inputTagWidthBits != **sourceWidth ||
              boundaryPoint->outputTagWidthBits != 0)
            return invalid("tag remover has inconsistent endpoints");
          break;
        }
      }
      if (destinationWidth->has_value() != childSegment.has_value())
        return invalid("tagged RouteTree node has no continuity segment");
      nodeSegments[ordinal] = childSegment;
      if (childSegment)
        addMatchDomain(*childSegment, node);
    }

    for (auto [nodeOrdinal, segment] : llvm::enumerate(nodeSegments))
      if (segment)
        segments[*segment].nodeOrdinals.push_back(nodeOrdinal);

    for (RequiredPhysicalTagUse &use : segments) {
      llvm::sort(use.matchDomains);
      use.matchDomains.erase(
          std::unique(use.matchDomains.begin(), use.matchDomains.end()),
          use.matchDomains.end());
      auto key =
          physicalTagUseKey(use.owner, use.trigger, use.assignmentPoint.pattern,
                            dataflow.identity());
      if (!key)
        return key.takeError();
      if (!result.emplace(std::move(*key), std::move(use)).second)
        return invalid("continuity origins derive a duplicate ResourceUse");
    }
  }
  return result;
}

} // namespace loom::mapping::detail

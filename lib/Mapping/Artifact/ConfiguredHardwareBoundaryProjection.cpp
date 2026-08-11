#include "ConfiguredHardwareProjectionInternal.h"

#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

bool selectsBoundary(
    llvm::ArrayRef<::loom::fabric::FabricPhysicalTraversalRef> traversals,
    ::loom::fabric::FabricBoundaryOccurrenceRef boundary) {
  return llvm::any_of(traversals, [&](const auto &traversal) {
    const auto *payload =
        std::get_if<::loom::fabric::FabricBoundaryTraversalPayload>(
            &traversal.payload);
    return payload && payload->owner == boundary;
  });
}

} // namespace

llvm::Expected<std::vector<ConfiguredHardwareFieldValueView>>
deriveConfiguredBoundaryFields(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments) {
  std::vector<::loom::fabric::FabricPhysicalTraversalRef> traversals;
  for (const SpatialRouteTreeView &route : routes) {
    if (route.localTraversal)
      traversals.push_back(*route.localTraversal);
    for (const SpatialRouteNodeView &node : route.nodes)
      if (node.incomingTraversal)
        traversals.push_back(*node.incomingTraversal);
    for (const SpatialRouteSinkView &sink : route.sinks)
      if (sink.localTraversal)
        traversals.push_back(*sink.localTraversal);
  }

  std::vector<ConfiguredHardwareFieldValueView> result;
  for (const auto boundary : fabric.boundaryOccurrences()) {
    if (!selectsBoundary(traversals, boundary))
      continue;
    const auto point = fabric.boundaryTagContinuityPoint(boundary);
    if (!point)
      return invalid("selected boundary has no continuity shape");

    ::loom::fabric::FabricBoundaryConfiguration configuration;
    bool foundContextualSelection = false;
    for (const auto &[routeOrdinal, route] : llvm::enumerate(routes)) {
      for (const SpatialRouteNodeView &node : route.nodes) {
        if (!node.incomingTraversal)
          continue;
        const auto *payload =
            std::get_if<::loom::fabric::FabricBoundaryTraversalPayload>(
                &node.incomingTraversal->payload);
        if (!payload || payload->owner != boundary)
          continue;
        foundContextualSelection = true;

        using Kind = ::loom::fabric::FabricBoundaryTagContinuityKind;
        if (point->kind == Kind::ConfigurableWriter) {
          auto tag = resolveConfiguredHardwarePhysicalTag(
              fabric, routes, resourceUses, physicalTagSegments, routeOrdinal,
              node.ordinal);
          if (!tag)
            return tag.takeError();
          if (configuration.configuredTag &&
              *configuration.configuredTag != *tag)
            return invalid("one configurable boundary selects multiple tags");
          configuration.configuredTag = std::move(*tag);
        } else if (point->kind == Kind::Rewriter) {
          if (!node.parentOrdinal)
            return invalid("tag rewrite boundary has no route parent");
          auto inputTag = resolveConfiguredHardwarePhysicalTag(
              fabric, routes, resourceUses, physicalTagSegments, routeOrdinal,
              *node.parentOrdinal);
          if (!inputTag)
            return inputTag.takeError();
          auto outputTag = resolveConfiguredHardwarePhysicalTag(
              fabric, routes, resourceUses, physicalTagSegments, routeOrdinal,
              node.ordinal);
          if (!outputTag)
            return outputTag.takeError();
          configuration.tagRewrites.push_back(
              {std::move(*inputTag), std::move(*outputTag)});
        }
      }
    }

    using Kind = ::loom::fabric::FabricBoundaryTagContinuityKind;
    if ((point->kind == Kind::ConfigurableWriter ||
         point->kind == Kind::Rewriter) &&
        !foundContextualSelection)
      return invalid("selected configurable boundary has no RouteTree node");

    const ::loom::fabric::FabricSemanticConfigFieldRef field{
        ::loom::fabric::FabricConfigurationOwnerRef(
            ::loom::fabric::FabricInventoryOwnerRef::of(boundary)),
        0};
    auto slot = resolveConfiguredHardwareSlot(fabric, field);
    if (!slot)
      return slot.takeError();
    auto value = ::loom::fabric::encodeFabricBoundaryConfiguration(
        fabric, field, std::move(configuration));
    if (!value)
      return value.takeError();
    result.push_back({std::move(*slot), std::move(*value)});
  }
  return result;
}

} // namespace loom::mapping::detail

#include "ConfiguredHardwareProjectionInternal.h"

#include "Fabric/Identity/FabricRefImport.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

} // namespace

llvm::Expected<llvm::APInt> resolveConfiguredHardwarePhysicalTag(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> segments,
    std::uint64_t routeOrdinal, std::uint64_t nodeOrdinal) {
  if (routeOrdinal >= routes.size() ||
      nodeOrdinal >= routes[routeOrdinal].nodes.size())
    return invalid("Physical Tag query is outside its RouteTree");
  const auto endpoint = routes[routeOrdinal].nodes[nodeOrdinal].endpoint;
  auto path = fabric.transportEndpointDataPath(endpoint);
  if (!path)
    return invalid("Physical Tag endpoint has no data-path projection");
  if (path->tagWidthBits == 0)
    return llvm::APInt(1, 0);

  const SpatialPhysicalTagSegmentView *selected = nullptr;
  for (const SpatialPhysicalTagSegmentView &segment : segments) {
    if (segment.routeTreeOrdinal != routeOrdinal ||
        !llvm::is_contained(segment.nodeOrdinals, nodeOrdinal))
      continue;
    if (selected)
      return invalid("RouteTree node belongs to multiple Physical Tag "
                     "segments");
    selected = &segment;
  }
  if (!selected || selected->resourceUseOrdinal >= resourceUses.size())
    return invalid("tagged RouteTree node has no Physical Tag assignment");
  const auto &assignments =
      resourceUses[selected->resourceUseOrdinal].sharingAssignments;
  if (assignments.size() != 1)
    return invalid("Physical Tag assignment has the wrong value shape");
  const auto *tag =
      std::get_if<::fabric::PhysicalTagPatternValue>(&assignments.front());
  if (!tag || tag->value.getBitWidth() != path->tagWidthBits)
    return invalid("Physical Tag value has the wrong width");
  return tag->value;
}

} // namespace loom::mapping::detail

llvm::Expected<llvm::APInt> loom::mapping::resolveSpatialPhysicalTag(
    const SpatialMappingView &mapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    std::uint64_t routeTreeOrdinal, std::uint64_t nodeOrdinal) {
  return detail::resolveConfiguredHardwarePhysicalTag(
      fabric, mapping.routeTrees(), mapping.resourceUses(),
      mapping.physicalTagSegments(), routeTreeOrdinal, nodeOrdinal);
}

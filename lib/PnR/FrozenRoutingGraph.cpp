#include "PnR/FrozenRoutingGraph.h"

#include "Mapping/FabricOccurrenceIndex.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom::mapping;
using namespace loom::mapping::detail;
using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral frozenArtifact = "FrozenRoutingGraph";
constexpr PnrCapacityContext endpointCountContext{
    frozenArtifact, "routing_endpoints", "transport_endpoints",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext endpointIndexContext{
    frozenArtifact, "routing_endpoints", "transport_endpoints",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext resourceCountContext{
    frozenArtifact, "transport_resources", "transport_resources",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext resourceIndexContext{
    frozenArtifact, "transport_resources", "transport_resources",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext occurrenceCountContext{
    frozenArtifact, "routing_endpoints", "compute_occurrences",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext occurrenceIndexContext{
    frozenArtifact, "routing_endpoints", "compute_occurrences",
    PnrCapacityMeasure::Index};
constexpr PnrCapacityContext resourceEndpointCountContext{
    frozenArtifact, "resource_endpoint_vertices", "transport_endpoints",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext resourceEndpointOffsetContext{
    frozenArtifact, "transport_resources", "resource_endpoint_vertices",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext computeEndpointCountContext{
    frozenArtifact, "compute_endpoint_vertices", "compute_endpoints",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext adjacencyCountContext{
    frozenArtifact, "adjacency_offsets", "routing_endpoints",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext adjacencyOffsetContext{
    frozenArtifact, "adjacency_offsets", "routing_arcs",
    PnrCapacityMeasure::Offset};
constexpr PnrCapacityContext arcCountContext{
    frozenArtifact, "routing_arcs", "routing_arcs", PnrCapacityMeasure::Count};

llvm::Error freezeError(std::string message) {
  return llvm::make_error<llvm::StringError>(
      std::move(message), std::make_error_code(std::errc::invalid_argument));
}

std::uint64_t sizeValue(std::size_t size) {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  return static_cast<std::uint64_t>(size);
}

llvm::Error preflight(PnrCapacityContext context, std::size_t size) {
  return preflightPnrIndexCapacity(context, sizeValue(size));
}

llvm::Expected<PnrIndex> checked(PnrCapacityContext context,
                                 std::size_t value) {
  return checkedPnrIndex(context, sizeValue(value));
}

FrozenRoutingEndpointOwnerKind
freezeOwnerKind(ValidatedRoutingEndpointOwnerKind kind) {
  switch (kind) {
  case ValidatedRoutingEndpointOwnerKind::ComputeOccurrence:
    return FrozenRoutingEndpointOwnerKind::ComputeOccurrence;
  case ValidatedRoutingEndpointOwnerKind::TransportResource:
    return FrozenRoutingEndpointOwnerKind::TransportResource;
  }
  llvm_unreachable("invalid validated routing endpoint owner kind");
}

FrozenRoutingArcKind freezeArcKind(ValidatedRoutingArcKind kind) {
  switch (kind) {
  case ValidatedRoutingArcKind::PointToPoint:
    return FrozenRoutingArcKind::PointToPoint;
  case ValidatedRoutingArcKind::Traversal:
    return FrozenRoutingArcKind::Traversal;
  }
  llvm_unreachable("invalid validated routing arc kind");
}

} // namespace

llvm::Error loom::pnr::detail::preflightFrozenRoutingGraphCapacity(
    std::uint64_t endpointCount, std::uint64_t resourceCount,
    std::uint64_t computeEndpointCount, std::uint64_t arcCount) {
  if (llvm::Error error =
          preflightPnrIndexCapacity(endpointCountContext, endpointCount))
    return error;
  if (llvm::Error error =
          preflightPnrIndexCapacity(resourceCountContext, resourceCount))
    return error;
  if (llvm::Error error = preflightPnrIndexCapacity(computeEndpointCountContext,
                                                    computeEndpointCount))
    return error;
  if (llvm::Error error = preflightPnrIndexCapacity(arcCountContext, arcCount))
    return error;
  auto adjacencyCount =
      checkedPnrIndexAdd(adjacencyCountContext, endpointCount, 1);
  if (!adjacencyCount)
    return adjacencyCount.takeError();
  return llvm::Error::success();
}

llvm::Expected<FrozenRoutingGraph>
loom::pnr::freezeRoutingGraph(const FabricHardwareView &fabric,
                              const ValidatedTechMapping &mapping) {
  const ValidatedFabricProjection &fabricProjection =
      ValidatedTechMappingAccess::fabricProjection(mapping);
  if (fabricProjection.identity != fabric.identity)
    return freezeError("cannot freeze routing graph: validated Fabric "
                       "projection identity does not match the input");
  const ValidatedFabricRoutingProjection &projection = fabricProjection.routing;
  const std::size_t routableComputeEndpointCount = static_cast<std::size_t>(
      std::count_if(fabricProjection.computeEndpoints.begin(),
                    fabricProjection.computeEndpoints.end(),
                    [](const ValidatedComputeEndpoint &endpoint) {
                      return endpoint.kind != PortKind::Memory;
                    }));

  if (llvm::Error error = detail::preflightFrozenRoutingGraphCapacity(
          sizeValue(projection.endpoints.size()),
          sizeValue(projection.resources.size()),
          sizeValue(routableComputeEndpointCount),
          sizeValue(projection.arcs.size())))
    return std::move(error);
  if (llvm::Error error = preflight(occurrenceCountContext,
                                    fabricProjection.computeOccurrences.size()))
    return std::move(error);
  if (llvm::Error error = preflight(resourceEndpointCountContext,
                                    projection.resourceEndpoints.size()))
    return std::move(error);

  std::vector<FrozenTransportResource> resources;
  std::vector<PnrIndex> resourceEndpoints;
  resources.reserve(projection.resources.size());
  resourceEndpoints.reserve(projection.resourceEndpoints.size());
  for (const ValidatedTransportResource &resource : projection.resources) {
    auto endpointOffset =
        checked(resourceEndpointOffsetContext, resourceEndpoints.size());
    if (!endpointOffset)
      return endpointOffset.takeError();
    auto endpointCount =
        checked(resourceEndpointCountContext, resource.endpointCount);
    if (!endpointCount)
      return endpointCount.takeError();
    auto endpointEnd =
        checkedPnrIndexAdd(resourceEndpointOffsetContext,
                           resource.endpointOffset, resource.endpointCount);
    if (!endpointEnd)
      return endpointEnd.takeError();
    for (std::size_t endpointIndex = resource.endpointOffset;
         endpointIndex < *endpointEnd; ++endpointIndex) {
      auto endpoint = checked(endpointIndexContext,
                              projection.resourceEndpoints[endpointIndex]);
      if (!endpoint)
        return endpoint.takeError();
      resourceEndpoints.push_back(*endpoint);
    }
    resources.push_back({resource.id, resource.kind, resource.boundaryDirection,
                         *endpointOffset, *endpointCount});
  }

  std::vector<FrozenRoutingEndpoint> endpoints;
  endpoints.reserve(projection.endpoints.size());
  for (const ValidatedRoutingEndpoint &endpoint : projection.endpoints) {
    const PnrCapacityContext ownerContext =
        endpoint.ownerKind ==
                ValidatedRoutingEndpointOwnerKind::ComputeOccurrence
            ? occurrenceIndexContext
            : resourceIndexContext;
    auto owner = checked(ownerContext, endpoint.owner);
    if (!owner)
      return owner.takeError();
    endpoints.push_back({endpoint.id, freezeOwnerKind(endpoint.ownerKind),
                         *owner, endpoint.direction, endpoint.portKind,
                         endpoint.transportKind, endpoint.payloadCapacityBits,
                         endpoint.tagCapacityBits});
  }

  std::vector<PnrIndex> computeEndpoints;
  computeEndpoints.reserve(routableComputeEndpointCount);
  for (const ValidatedComputeEndpoint &computeEndpoint :
       fabricProjection.computeEndpoints) {
    if (computeEndpoint.kind == PortKind::Memory)
      continue;
    const std::optional<std::size_t> endpoint =
        findRoutingEndpoint(projection, computeEndpoint.id);
    if (!endpoint)
      return freezeError("cannot freeze routing graph: compute endpoint is "
                         "missing from the validated routing projection");
    auto frozenEndpoint = checked(endpointIndexContext, *endpoint);
    if (!frozenEndpoint)
      return frozenEndpoint.takeError();
    computeEndpoints.push_back(*frozenEndpoint);
  }

  std::vector<PnrIndex> adjacencyOffsets;
  std::vector<FrozenRoutingArc> arcs;
  adjacencyOffsets.reserve(projection.endpoints.size() + 1);
  arcs.reserve(projection.arcs.size());
  std::size_t arcCursor = 0;
  for (std::size_t endpoint = 0; endpoint < projection.endpoints.size();
       ++endpoint) {
    auto offset = checked(adjacencyOffsetContext, arcs.size());
    if (!offset)
      return offset.takeError();
    adjacencyOffsets.push_back(*offset);
    while (arcCursor < projection.arcs.size() &&
           projection.arcs[arcCursor].source == endpoint) {
      const ValidatedRoutingArc &arc = projection.arcs[arcCursor++];
      auto target = checked(endpointIndexContext, arc.target);
      if (!target)
        return target.takeError();
      std::optional<PnrIndex> resource;
      if (arc.resource) {
        auto frozenResource = checked(resourceIndexContext, *arc.resource);
        if (!frozenResource)
          return frozenResource.takeError();
        resource = *frozenResource;
      }
      arcs.push_back({*target, freezeArcKind(arc.kind), resource,
                      arc.payloadCapacityBits, arc.tagCapacityBits});
    }
  }
  auto finalOffset = checked(adjacencyOffsetContext, arcs.size());
  if (!finalOffset)
    return finalOffset.takeError();
  adjacencyOffsets.push_back(*finalOffset);
  if (arcCursor != projection.arcs.size())
    return freezeError(
        "cannot freeze routing graph: validated arc order is inconsistent");

  return FrozenRoutingGraph(std::move(resources), std::move(resourceEndpoints),
                            std::move(endpoints), std::move(computeEndpoints),
                            std::move(adjacencyOffsets), std::move(arcs));
}

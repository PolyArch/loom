#include "FabricRoutingIndex.h"

#include "Fabric/IR/BoundaryDataPath.h"
#include "FabricOccurrenceIndex.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

using namespace loom::mapping;
using namespace loom::mapping::detail;

namespace {

struct ResolvedArc {
  std::size_t source;
  std::size_t target;
};

struct ResolvedTraversal {
  std::size_t resource;
  std::size_t source;
  std::size_t target;
};

bool validDirection(PortDirection direction) {
  return direction == PortDirection::Input ||
         direction == PortDirection::Output;
}

bool validRoutingPortKind(PortKind kind) {
  return kind == PortKind::Value || kind == PortKind::Stream;
}

bool validResourceKind(TransportResourceKind kind) {
  return kind == TransportResourceKind::Switch ||
         kind == TransportResourceKind::Fifo ||
         kind == TransportResourceKind::Boundary;
}

llvm::Error invalidTransport(const llvm::Twine &message) {
  return mappingError(MappingErrorCode::InvalidPortConnection, message);
}

fabric::DataPathType dataPathType(const ValidatedRoutingEndpoint &endpoint) {
  return {endpoint.transportKind, endpoint.payloadCapacityBits,
          endpoint.tagCapacityBits};
}

llvm::Expected<std::size_t>
resolveEndpoint(const TransportEndpointRef &reference,
                const FabricHardwareView &fabric, const EntityKinds &kinds,
                const std::map<std::uint64_t, std::size_t> &endpoints) {
  if (reference.artifact != fabric.identity)
    return mappingError(MappingErrorCode::ForeignEntityReference,
                        "reference names a foreign artifact");
  const auto kind = kinds.find(reference.entity.value());
  if (kind == kinds.end())
    return mappingError(MappingErrorCode::UnresolvedEntityId,
                        "reference names an unresolved entity ID");
  if (kind->second != EntityKind::ComputeEndpoint &&
      kind->second != EntityKind::TransportEndpoint)
    return mappingError(MappingErrorCode::WrongEntityKind,
                        "reference names an entity of the wrong kind");
  const auto endpoint = endpoints.find(reference.entity.value());
  if (endpoint == endpoints.end())
    return invalidTransport(
        "reference names an endpoint outside the compute-only transport "
        "domain");
  return endpoint->second;
}

llvm::Expected<std::size_t>
resolveResource(const TransportResourceRef &reference,
                const FabricHardwareView &fabric, const EntityKinds &kinds,
                const std::map<std::uint64_t, std::size_t> &resources) {
  if (reference.artifact != fabric.identity)
    return mappingError(MappingErrorCode::ForeignEntityReference,
                        "reference names a foreign artifact");
  const auto kind = kinds.find(reference.entity.value());
  if (kind == kinds.end())
    return mappingError(MappingErrorCode::UnresolvedEntityId,
                        "reference names an unresolved entity ID");
  if (kind->second != EntityKind::TransportResource)
    return mappingError(MappingErrorCode::WrongEntityKind,
                        "reference names an entity of the wrong kind");
  return resources.at(reference.entity.value());
}

} // namespace

llvm::Expected<ValidatedFabricRoutingProjection>
loom::mapping::detail::buildValidatedFabricRoutingProjection(
    const FabricHardwareView &fabric, EntityKinds &kinds,
    const ValidatedFabricProjection &computeProjection) {
  if (computeProjection.identity != fabric.identity)
    return mappingError(MappingErrorCode::InternalError,
                        "validated compute projection identity mismatch");
  std::vector<const TransportResourceDescriptor *> resources;
  resources.reserve(fabric.transportResources.size());
  for (const TransportResourceDescriptor &resource : fabric.transportResources)
    resources.push_back(&resource);
  std::sort(resources.begin(), resources.end(),
            [](const TransportResourceDescriptor *lhs,
               const TransportResourceDescriptor *rhs) {
              return lhs->id.value() < rhs->id.value();
            });

  std::map<std::uint64_t, std::size_t> resourcesById;
  for (std::size_t resourceIndex = 0; resourceIndex < resources.size();
       ++resourceIndex) {
    const TransportResourceDescriptor &resource = *resources[resourceIndex];
    if (llvm::Error error = addEntity(kinds, resource.id.value(),
                                      EntityKind::TransportResource))
      return std::move(error);
    if (!validResourceKind(resource.kind))
      return invalidTransport("transport resource has an invalid kind");
    if (resource.kind == TransportResourceKind::Boundary) {
      if (!resource.boundaryDirection)
        return invalidTransport("boundary resource has a missing direction");
    } else if (resource.boundaryDirection) {
      return invalidTransport(
          "non-boundary transport resource declares a boundary direction");
    }
    resourcesById.emplace(resource.id.value(), resourceIndex);
  }

  ValidatedFabricRoutingProjection projection;
  projection.endpoints.reserve(computeProjection.computeEndpoints.size());
  for (std::size_t occurrenceIndex = 0;
       occurrenceIndex < computeProjection.peOccurrences.size();
       ++occurrenceIndex) {
    const ValidatedPeOccurrence &occurrence =
        computeProjection.peOccurrences[occurrenceIndex];
    for (std::size_t endpointIndex = occurrence.endpointOffset;
         endpointIndex < occurrence.endpointOffset + occurrence.endpointCount;
         ++endpointIndex) {
      const ValidatedComputeEndpoint &endpoint =
          computeProjection.computeEndpoints[endpointIndex];
      if (llvm::Error error = requireLocalKind(kinds, endpoint.id.value(),
                                               EntityKind::ComputeEndpoint))
        return std::move(error);
      if (endpoint.kind == PortKind::Memory)
        continue;
      if (!validRoutingPortKind(endpoint.kind) ||
          !fabric::DataPathType{endpoint.transportKind,
                                endpoint.payloadCapacityBits,
                                endpoint.tagCapacityBits}
               .isWellFormed())
        return invalidTransport(
            "compute endpoint has an invalid transport signature");
      projection.endpoints.push_back(
          {endpoint.id, ValidatedRoutingEndpointOwnerKind::ComputeOccurrence,
           occurrenceIndex, endpoint.direction, endpoint.kind,
           endpoint.transportKind, endpoint.payloadCapacityBits,
           endpoint.tagCapacityBits});
    }
  }

  for (std::size_t resourceIndex = 0; resourceIndex < resources.size();
       ++resourceIndex) {
    for (const TransportEndpointDescriptor &endpoint :
         resources[resourceIndex]->endpoints) {
      if (llvm::Error error = addEntity(kinds, endpoint.id.value(),
                                        EntityKind::TransportEndpoint))
        return std::move(error);
      if (!validDirection(endpoint.direction) ||
          !validRoutingPortKind(endpoint.kind) ||
          !fabric::DataPathType{endpoint.transportKind,
                                endpoint.payloadCapacityBits,
                                endpoint.tagCapacityBits}
               .isWellFormed())
        return invalidTransport(
            "transport resource endpoint has an invalid signature");
      projection.endpoints.push_back(
          {endpoint.id, ValidatedRoutingEndpointOwnerKind::TransportResource,
           resourceIndex, endpoint.direction, endpoint.kind,
           endpoint.transportKind, endpoint.payloadCapacityBits,
           endpoint.tagCapacityBits});
    }
  }

  std::sort(projection.endpoints.begin(), projection.endpoints.end(),
            [](const ValidatedRoutingEndpoint &lhs,
               const ValidatedRoutingEndpoint &rhs) {
              return lhs.id.value() < rhs.id.value();
            });

  std::map<std::uint64_t, std::size_t> endpointsById;
  std::vector<std::vector<std::size_t>> resourceEndpointIndices(
      resources.size());
  for (std::size_t endpointIndex = 0;
       endpointIndex < projection.endpoints.size(); ++endpointIndex) {
    const ValidatedRoutingEndpoint &endpoint =
        projection.endpoints[endpointIndex];
    endpointsById.emplace(endpoint.id.value(), endpointIndex);
    if (endpoint.ownerKind ==
        ValidatedRoutingEndpointOwnerKind::TransportResource)
      resourceEndpointIndices[endpoint.owner].push_back(endpointIndex);
  }

  projection.resources.reserve(resources.size());
  for (std::size_t resourceIndex = 0; resourceIndex < resources.size();
       ++resourceIndex) {
    const TransportResourceDescriptor &resource = *resources[resourceIndex];
    const std::vector<std::size_t> &endpointIndices =
        resourceEndpointIndices[resourceIndex];
    const std::size_t endpointOffset = projection.resourceEndpoints.size();
    projection.resourceEndpoints.insert(projection.resourceEndpoints.end(),
                                        endpointIndices.begin(),
                                        endpointIndices.end());
    std::size_t inputCount = 0;
    std::size_t outputCount = 0;
    for (std::size_t endpointIndex : endpointIndices) {
      if (projection.endpoints[endpointIndex].direction == PortDirection::Input)
        ++inputCount;
      else
        ++outputCount;
    }
    if ((resource.kind == TransportResourceKind::Switch &&
         (inputCount == 0 || outputCount == 0)) ||
        ((resource.kind == TransportResourceKind::Fifo ||
          resource.kind == TransportResourceKind::Boundary) &&
         (inputCount != 1 || outputCount != 1)))
      return invalidTransport(
          "transport resource has an invalid endpoint structure");
    projection.resources.push_back({resource.id, resource.kind,
                                    resource.boundaryDirection, endpointOffset,
                                    endpointIndices.size()});
  }

  std::vector<ResolvedArc> pointArcs;
  pointArcs.reserve(fabric.transportArcs.size());
  std::vector<bool> hasPointSuccessor(projection.endpoints.size());
  std::vector<bool> hasPointPredecessor(projection.endpoints.size());
  for (const TransportArcDescriptor &arc : fabric.transportArcs) {
    auto source = resolveEndpoint(arc.source, fabric, kinds, endpointsById);
    if (!source)
      return source.takeError();
    auto target = resolveEndpoint(arc.target, fabric, kinds, endpointsById);
    if (!target)
      return target.takeError();
    const ValidatedRoutingEndpoint &sourceEndpoint =
        projection.endpoints[*source];
    const ValidatedRoutingEndpoint &targetEndpoint =
        projection.endpoints[*target];
    if (sourceEndpoint.direction != PortDirection::Output ||
        targetEndpoint.direction != PortDirection::Input ||
        sourceEndpoint.portKind != targetEndpoint.portKind ||
        sourceEndpoint.transportKind != targetEndpoint.transportKind ||
        hasPointSuccessor[*source] || hasPointPredecessor[*target])
      return invalidTransport("transport arc is not point-to-point legal");
    hasPointSuccessor[*source] = true;
    hasPointPredecessor[*target] = true;
    pointArcs.push_back({*source, *target});
  }
  std::vector<ResolvedTraversal> traversals;
  traversals.reserve(fabric.transportTraversals.size());
  std::vector<std::size_t> traversalCounts(resources.size());
  for (const TransportTraversalDescriptor &traversal :
       fabric.transportTraversals) {
    auto resource =
        resolveResource(traversal.resource, fabric, kinds, resourcesById);
    if (!resource)
      return resource.takeError();
    auto source =
        resolveEndpoint(traversal.source, fabric, kinds, endpointsById);
    if (!source)
      return source.takeError();
    auto target =
        resolveEndpoint(traversal.target, fabric, kinds, endpointsById);
    if (!target)
      return target.takeError();
    const ValidatedRoutingEndpoint &sourceEndpoint =
        projection.endpoints[*source];
    const ValidatedRoutingEndpoint &targetEndpoint =
        projection.endpoints[*target];
    if (sourceEndpoint.ownerKind !=
            ValidatedRoutingEndpointOwnerKind::TransportResource ||
        targetEndpoint.ownerKind !=
            ValidatedRoutingEndpointOwnerKind::TransportResource ||
        sourceEndpoint.owner != *resource ||
        targetEndpoint.owner != *resource ||
        sourceEndpoint.direction != PortDirection::Input ||
        targetEndpoint.direction != PortDirection::Output ||
        sourceEndpoint.portKind != targetEndpoint.portKind)
      return invalidTransport(
          "transport traversal does not belong to its resource");
    const TransportResourceDescriptor &resourceDescriptor =
        *resources[*resource];
    if (resourceDescriptor.kind == TransportResourceKind::Boundary) {
      if (fabric::checkBoundaryDataPath(*resourceDescriptor.boundaryDirection,
                                        dataPathType(sourceEndpoint),
                                        dataPathType(targetEndpoint)) !=
          fabric::BoundaryDataPathError::None)
        return invalidTransport(
            "boundary traversal does not match its declared direction");
    } else if (sourceEndpoint.transportKind != targetEndpoint.transportKind) {
      return invalidTransport(
          "ordinary transport traversal changes native transport kind");
    }
    ++traversalCounts[*resource];
    traversals.push_back({*resource, *source, *target});
  }
  std::sort(traversals.begin(), traversals.end(),
            [](const ResolvedTraversal &lhs, const ResolvedTraversal &rhs) {
              return std::tie(lhs.resource, lhs.source, lhs.target) <
                     std::tie(rhs.resource, rhs.source, rhs.target);
            });
  if (std::adjacent_find(
          traversals.begin(), traversals.end(),
          [](const ResolvedTraversal &lhs, const ResolvedTraversal &rhs) {
            return lhs.resource == rhs.resource && lhs.source == rhs.source &&
                   lhs.target == rhs.target;
          }) != traversals.end())
    return invalidTransport("transport resource repeats a traversal");
  for (std::size_t resourceIndex = 0; resourceIndex < resources.size();
       ++resourceIndex) {
    const TransportResourceKind kind = resources[resourceIndex]->kind;
    if ((kind == TransportResourceKind::Fifo ||
         kind == TransportResourceKind::Boundary) &&
        traversalCounts[resourceIndex] != 1)
      return invalidTransport(
          "FIFO or boundary resource must declare one traversal");
  }

  projection.arcs.reserve(pointArcs.size() + traversals.size());
  for (const ResolvedArc &arc : pointArcs) {
    const ValidatedRoutingEndpoint &source = projection.endpoints[arc.source];
    const ValidatedRoutingEndpoint &target = projection.endpoints[arc.target];
    projection.arcs.push_back(
        {arc.source, arc.target, ValidatedRoutingArcKind::PointToPoint,
         std::nullopt,
         std::min(source.payloadCapacityBits, target.payloadCapacityBits),
         std::min(source.tagCapacityBits, target.tagCapacityBits)});
  }
  for (const ResolvedTraversal &traversal : traversals) {
    const ValidatedRoutingEndpoint &source =
        projection.endpoints[traversal.source];
    const ValidatedRoutingEndpoint &target =
        projection.endpoints[traversal.target];
    projection.arcs.push_back(
        {traversal.source, traversal.target, ValidatedRoutingArcKind::Traversal,
         traversal.resource,
         std::min(source.payloadCapacityBits, target.payloadCapacityBits),
         std::min(source.tagCapacityBits, target.tagCapacityBits)});
  }
  std::sort(projection.arcs.begin(), projection.arcs.end(),
            [](const ValidatedRoutingArc &lhs, const ValidatedRoutingArc &rhs) {
              return std::make_tuple(lhs.source, lhs.target, lhs.kind,
                                     lhs.resource.value_or(0)) <
                     std::make_tuple(rhs.source, rhs.target, rhs.kind,
                                     rhs.resource.value_or(0));
            });
  return projection;
}

std::optional<std::size_t> loom::mapping::detail::findRoutingEndpoint(
    const ValidatedFabricRoutingProjection &projection,
    TransportEndpointId endpoint) {
  const auto found = std::lower_bound(
      projection.endpoints.begin(), projection.endpoints.end(), endpoint,
      [](const ValidatedRoutingEndpoint &candidate,
         TransportEndpointId expected) {
        return candidate.id.value() < expected.value();
      });
  if (found == projection.endpoints.end() || found->id != endpoint)
    return std::nullopt;
  return static_cast<std::size_t>(found - projection.endpoints.begin());
}

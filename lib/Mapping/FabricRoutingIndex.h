#ifndef LOOM_LIB_MAPPING_FABRICROUTINGINDEX_H
#define LOOM_LIB_MAPPING_FABRICROUTINGINDEX_H

#include "VerifierInternal.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace loom::mapping::detail {

struct ValidatedFabricProjection;

enum class ValidatedRoutingEndpointOwnerKind {
  ComputeOccurrence,
  MemoryOccurrence,
  TransportResource,
};

enum class ValidatedRoutingArcKind {
  PointToPoint,
  Traversal,
};

struct ValidatedTransportResource {
  TransportResourceId id;
  TransportResourceKind kind;
  std::optional<fabric::BoundaryDirection> boundaryDirection;
  std::size_t endpointOffset;
  std::size_t endpointCount;
};

struct ValidatedRoutingEndpoint {
  TransportEndpointId id;
  ValidatedRoutingEndpointOwnerKind ownerKind;
  std::size_t owner;
  PortDirection direction;
  PortKind portKind;
  fabric::DataPathKind transportKind;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
};

struct ValidatedRoutingArc {
  std::size_t source;
  std::size_t target;
  ValidatedRoutingArcKind kind;
  std::optional<std::size_t> resource;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
};

struct ValidatedFabricRoutingProjection {
  std::vector<ValidatedTransportResource> resources;
  std::vector<std::size_t> resourceEndpoints;
  std::vector<ValidatedRoutingEndpoint> endpoints;
  std::vector<ValidatedRoutingArc> arcs;
};

llvm::Expected<ValidatedFabricRoutingProjection>
buildValidatedFabricRoutingProjection(
    const FabricHardwareView &fabric, EntityKinds &kinds,
    const ValidatedFabricProjection &computeProjection);

std::optional<std::size_t>
findRoutingEndpoint(const ValidatedFabricRoutingProjection &projection,
                    TransportEndpointId endpoint);

} // namespace loom::mapping::detail

#endif // LOOM_LIB_MAPPING_FABRICROUTINGINDEX_H

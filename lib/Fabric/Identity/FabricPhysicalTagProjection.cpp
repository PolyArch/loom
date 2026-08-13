#include "FabricPhysicalTagProjection.h"

#include "Fabric/Identity/FabricRefBytes.h"

namespace loom::fabric::detail {

std::optional<FabricPhysicalTagMatchDomainView>
projectPhysicalTagMatchDomain(const FabricArtifactView &view,
                              const FabricTransportEndpointRef &endpoint) {
  if (view.transportEndpointDirection(endpoint) != FabricPortDirection::Input)
    return std::nullopt;
  const auto path = view.transportEndpointDataPath(endpoint);
  if (!path || path->kind != ::fabric::DataPathKind::BitsTag)
    return std::nullopt;

  switch (endpoint.owner.kind()) {
  case FabricTransportEndpointOwnerKind::FabricPeOccurrence: {
    const auto owner = std::get<FabricPeOccurrenceRef>(endpoint.owner.payload);
    if (view.peSchedule(owner) != ::fabric::Schedule::Temporal)
      return std::nullopt;
    return FabricPhysicalTagMatchDomainView{
        FabricPhysicalTagMatchDomainKind::TemporalPeIngress,
        FabricInventoryOwnerRef::of(owner), endpoint, path->tagWidthBits,
        std::nullopt};
  }
  case FabricTransportEndpointOwnerKind::FabricMemoryOccurrence: {
    const auto owner =
        std::get<FabricMemoryOccurrenceRef>(endpoint.owner.payload);
    if (view.memorySchedule(owner) != ::fabric::Schedule::Temporal)
      return std::nullopt;
    return FabricPhysicalTagMatchDomainView{
        FabricPhysicalTagMatchDomainKind::TemporalMemoryIngress,
        FabricInventoryOwnerRef::of(owner), endpoint, path->tagWidthBits,
        std::nullopt};
  }
  case FabricTransportEndpointOwnerKind::FabricSwitchOccurrence: {
    const auto owner =
        std::get<FabricSwitchOccurrenceRef>(endpoint.owner.payload);
    return FabricPhysicalTagMatchDomainView{
        FabricPhysicalTagMatchDomainKind::TemporalSwitchTable,
        FabricInventoryOwnerRef::of(owner), std::nullopt, path->tagWidthBits,
        view.switchRouteTableSize(owner)};
  }
  case FabricTransportEndpointOwnerKind::FabricBoundaryOccurrence: {
    const auto owner =
        std::get<FabricBoundaryOccurrenceRef>(endpoint.owner.payload);
    const auto point = view.boundaryTagContinuityPoint(owner);
    if (!point || point->kind != FabricBoundaryTagContinuityKind::Rewriter)
      return std::nullopt;
    return FabricPhysicalTagMatchDomainView{
        FabricPhysicalTagMatchDomainKind::BoundaryLookup,
        FabricInventoryOwnerRef::of(owner), std::nullopt, path->tagWidthBits,
        view.boundaryLookupTableSize(owner)};
  }
  case FabricTransportEndpointOwnerKind::SpatialCoreOccurrence:
  case FabricTransportEndpointOwnerKind::FabricFuOccurrence:
  case FabricTransportEndpointOwnerKind::FabricFifoOccurrence:
  case FabricTransportEndpointOwnerKind::SystemServiceEndpoint:
  case FabricTransportEndpointOwnerKind::SystemTransportResource:
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<FabricPhysicalTagAssignmentPointKind>
classifyPhysicalTagAssignmentPoint(const FabricArtifactView &view,
                                   const FabricTransportEndpointRef &endpoint) {
  const auto path = view.transportEndpointDataPath(endpoint);
  const auto direction = view.transportEndpointDirection(endpoint);
  if (!path || path->kind != ::fabric::DataPathKind::BitsTag || !direction)
    return std::nullopt;

  if (*direction == FabricPortDirection::Input) {
    switch (endpoint.owner.kind()) {
    case FabricTransportEndpointOwnerKind::FabricPeOccurrence:
      if (view.peSchedule(std::get<FabricPeOccurrenceRef>(
              endpoint.owner.payload)) != ::fabric::Schedule::Temporal)
        return std::nullopt;
      break;
    case FabricTransportEndpointOwnerKind::FabricMemoryOccurrence:
      if (view.memorySchedule(std::get<FabricMemoryOccurrenceRef>(
              endpoint.owner.payload)) != ::fabric::Schedule::Temporal)
        return std::nullopt;
      break;
    case FabricTransportEndpointOwnerKind::FabricSwitchOccurrence:
    case FabricTransportEndpointOwnerKind::FabricFifoOccurrence:
    case FabricTransportEndpointOwnerKind::FabricBoundaryOccurrence:
      break;
    default:
      return std::nullopt;
    }
    return FabricPhysicalTagAssignmentPointKind::Ingress;
  }

  switch (endpoint.owner.kind()) {
  case FabricTransportEndpointOwnerKind::FabricPeOccurrence:
    if (view.peSchedule(std::get<FabricPeOccurrenceRef>(
            endpoint.owner.payload)) == ::fabric::Schedule::Temporal)
      return FabricPhysicalTagAssignmentPointKind::Writer;
    return std::nullopt;
  case FabricTransportEndpointOwnerKind::FabricMemoryOccurrence:
    if (view.memorySchedule(std::get<FabricMemoryOccurrenceRef>(
            endpoint.owner.payload)) == ::fabric::Schedule::Temporal)
      return FabricPhysicalTagAssignmentPointKind::Writer;
    return std::nullopt;
  case FabricTransportEndpointOwnerKind::FabricBoundaryOccurrence: {
    const auto point = view.boundaryTagContinuityPoint(
        std::get<FabricBoundaryOccurrenceRef>(endpoint.owner.payload));
    if (point && point->kind != FabricBoundaryTagContinuityKind::Remover)
      return FabricPhysicalTagAssignmentPointKind::Writer;
    return std::nullopt;
  }
  default:
    return std::nullopt;
  }
}

std::vector<std::uint8_t>
ownerWideTagMatchDomainKey(const FabricPhysicalTagMatchDomainView &domain) {
  std::vector<std::uint8_t> result;
  result.push_back(static_cast<std::uint8_t>(domain.kind));
  const auto owner = canonicalFabricBytes(domain.owner);
  result.insert(result.end(), owner.begin(), owner.end());
  return result;
}

} // namespace loom::fabric::detail

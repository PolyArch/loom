#include "FabricTraversalProjection.h"

#include "Fabric/Artifact/FabricSystemRootView.h"

#include <cstdint>
#include <vector>

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

llvm::Expected<FabricTransportEndpointRef>
directionalEndpoint(const FabricArtifactView &view,
                    const FabricTransportEndpointOwnerRef &owner,
                    FabricPortDirection direction, FabricOrdinal ordinal) {
  FabricOrdinal matched = 0;
  const std::uint64_t count = view.transportEndpointCount(owner);
  for (FabricOrdinal candidate = 0; candidate < count; ++candidate) {
    FabricTransportEndpointRef endpoint{owner, candidate};
    if (view.transportEndpointDirection(endpoint) != direction)
      continue;
    if (matched++ == ordinal)
      return endpoint;
  }
  return invalid("physical traversal endpoint ordinal is out of range");
}

std::vector<FabricTransportEndpointRef>
directionalEndpoints(const FabricArtifactView &view,
                     const FabricTransportEndpointOwnerRef &owner,
                     FabricPortDirection direction) {
  std::vector<FabricTransportEndpointRef> result;
  const std::uint64_t count = view.transportEndpointCount(owner);
  for (FabricOrdinal ordinal = 0; ordinal < count; ++ordinal) {
    FabricTransportEndpointRef endpoint{owner, ordinal};
    if (view.transportEndpointDirection(endpoint) == direction)
      result.push_back(endpoint);
  }
  return result;
}

} // namespace

llvm::Expected<FabricPhysicalTraversalView>
projectFabricTraversal(const FabricArtifactView &view,
                       const FabricPhysicalTraversalRef &reference) {
  FabricPhysicalTraversalView result;
  result.reference = reference;
  switch (reference.kind()) {
  case FabricPhysicalTraversalKind::PointConnection: {
    const auto &payload =
        std::get<FabricPointConnectionPayload>(reference.payload);
    result.sources.push_back(payload.source);
    result.destinations.push_back(payload.destination);
    break;
  }
  case FabricPhysicalTraversalKind::PeSelectorTraversal: {
    const auto &payload = std::get<FabricPeSelectorPayload>(reference.payload);
    result.sources.push_back(payload.source);
    result.destinations.push_back(payload.destination);
    break;
  }
  case FabricPhysicalTraversalKind::PeRegisterFifoTraversal:
    return invalid(
        "PE register-FIFO traversal has no owner endpoint projection");
  case FabricPhysicalTraversalKind::SwitchTraversal: {
    const auto &payload =
        std::get<FabricSwitchTraversalPayload>(reference.payload);
    const auto owner = FabricTransportEndpointOwnerRef::of(payload.owner);
    auto source = directionalEndpoint(view, owner, FabricPortDirection::Input,
                                      payload.input);
    if (!source)
      return source.takeError();
    auto destination = directionalEndpoint(
        view, owner, FabricPortDirection::Output, payload.output);
    if (!destination)
      return destination.takeError();
    result.sources.push_back(*source);
    result.destinations.push_back(*destination);
    break;
  }
  case FabricPhysicalTraversalKind::FifoTraversal: {
    const auto &payload =
        std::get<FabricFifoTraversalPayload>(reference.payload);
    const auto owner = FabricTransportEndpointOwnerRef::of(payload.owner);
    auto source =
        directionalEndpoint(view, owner, FabricPortDirection::Input, 0);
    if (!source)
      return source.takeError();
    auto destination =
        directionalEndpoint(view, owner, FabricPortDirection::Output, 0);
    if (!destination)
      return destination.takeError();
    result.sources.push_back(*source);
    result.destinations.push_back(*destination);
    break;
  }
  case FabricPhysicalTraversalKind::BoundaryTraversal: {
    const auto &payload =
        std::get<FabricBoundaryTraversalPayload>(reference.payload);
    const auto owner = FabricTransportEndpointOwnerRef::of(payload.owner);
    result.sources =
        directionalEndpoints(view, owner, FabricPortDirection::Input);
    auto destination = directionalEndpoint(
        view, owner, FabricPortDirection::Output, payload.output);
    if (!destination)
      return destination.takeError();
    result.destinations.push_back(*destination);
    break;
  }
  case FabricPhysicalTraversalKind::SystemTransferPatternLeg: {
    const auto &payload =
        std::get<FabricTransferPatternLegPayload>(reference.payload);
    auto system = requireSystemRoot(view);
    if (!system)
      return system.takeError();
    const SystemTransferPatternRecord *pattern =
        system->transferPattern(payload.owner);
    if (!pattern || payload.egress >= pattern->egresses().size())
      return invalid(
          "system transfer-pattern traversal has no endpoint relation");
    result.sources.push_back(pattern->ingress());
    result.destinations.push_back(pattern->egresses()[payload.egress]);
    break;
  }
  }
  if (result.sources.empty() || result.destinations.empty())
    return invalid("physical traversal has an empty endpoint relation");
  return result;
}

} // namespace loom::fabric::detail

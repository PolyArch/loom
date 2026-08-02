#include "FabricTraversalProjection.h"

#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FifoResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
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

FabricResourceStateRef resourceState(const FabricInventoryOwnerRef &owner,
                                     FabricOrdinal ordinal) {
  return FabricResourceStateRef{FabricResourceStateOwnerRef(owner), ordinal};
}

llvm::Error appendPatternStates(const FabricArtifactView &view,
                                const FabricUsePatternRef &pattern,
                                std::vector<FabricResourceStateRef> &states) {
  const FabricInventoryOwnerRef &owner = pattern.owner.catalog();
  const ::fabric::ResourceContract *contract = view.resourceContract(owner);
  if (!contract || pattern.ordinal >= contract->usePatternCount())
    return invalid("physical traversal selects an invalid use pattern");
  for (const ::fabric::Claim &claim :
       contract->usePattern(::fabric::UsePatternKey(pattern.ordinal)).claims)
    states.push_back(resourceState(owner, claim.state.ordinal()));
  return llvm::Error::success();
}

void canonicalizeStates(std::vector<FabricResourceStateRef> &states) {
  llvm::sort(states, [](const FabricResourceStateRef &lhs,
                        const FabricResourceStateRef &rhs) {
    return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
  });
  states.erase(std::unique(states.begin(), states.end()), states.end());
}

void appendPhysicalResourceOwner(const FabricArtifactView &view,
                                 FabricInventoryOwnerRef owner,
                                 std::vector<FabricInventoryOwnerRef> &owners) {
  if (view.resourceContract(owner))
    owners.push_back(std::move(owner));
}

} // namespace

llvm::Expected<std::vector<FabricInventoryOwnerRef>>
projectModuleResourceOwners(const FabricArtifactView &view) {
  std::vector<FabricInventoryOwnerRef> owners;
  if (view.rootKind() != FabricRootKind::Module)
    return owners;
  for (FabricPeOccurrenceRef pe : view.peOccurrences())
    appendPhysicalResourceOwner(view, FabricInventoryOwnerRef::of(pe), owners);
  for (FabricFuOccurrenceRef fu : view.fuOccurrences()) {
    const FabricInventoryOwnerRef fuOwner = FabricInventoryOwnerRef::of(fu);
    appendPhysicalResourceOwner(view, fuOwner, owners);
    const std::uint64_t nodeCount =
        view.inventorySize(fuOwner, FabricInventoryKind::FuNode);
    for (FabricOrdinal ordinal = 0; ordinal < nodeCount; ++ordinal) {
      const std::optional<FabricFuNodeKind> kind =
          view.fuNodeKind(fuOwner, ordinal);
      if (!kind)
        return invalid("FU occurrence has an invalid node inventory");
      appendPhysicalResourceOwner(
          view,
          FabricInventoryOwnerRef::of(
              FabricFuOccurrenceNodeRef{*kind, fu, ordinal}),
          owners);
    }
  }
  for (FabricMemoryOccurrenceRef memory : view.memoryOccurrences()) {
    appendPhysicalResourceOwner(view, FabricInventoryOwnerRef::of(memory),
                                owners);
    for (FabricMemoryOperationPortRef port : view.memoryOperationPorts(memory))
      appendPhysicalResourceOwner(view, FabricInventoryOwnerRef::of(port),
                                  owners);
    if (view.declaresLocalMemoryService(memory))
      appendPhysicalResourceOwner(
          view,
          FabricInventoryOwnerRef::of(FabricMemoryServiceRef::local(memory)),
          owners);
  }
  for (FabricSwitchOccurrenceRef resource : view.switchOccurrences())
    appendPhysicalResourceOwner(view, FabricInventoryOwnerRef::of(resource),
                                owners);
  for (FabricFifoOccurrenceRef resource : view.fifoOccurrences())
    appendPhysicalResourceOwner(view, FabricInventoryOwnerRef::of(resource),
                                owners);
  for (FabricBoundaryOccurrenceRef resource : view.boundaryOccurrences())
    appendPhysicalResourceOwner(view, FabricInventoryOwnerRef::of(resource),
                                owners);

  llvm::sort(owners, [](const FabricInventoryOwnerRef &lhs,
                        const FabricInventoryOwnerRef &rhs) {
    return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
  });
  if (std::adjacent_find(owners.begin(), owners.end()) != owners.end())
    return invalid("physical resource-owner inventory contains a duplicate");
  return owners;
}

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
  case FabricPhysicalTraversalKind::PeRegisterFifoTraversal: {
    const auto &payload =
        std::get<FabricPeRegisterFifoPayload>(reference.payload);
    const FabricInventoryOwnerRef owner =
        FabricInventoryOwnerRef::of(payload.owner);
    const ::fabric::ResourceContract *contract = view.resourceContract(owner);
    const std::uint64_t registerFifoCount =
        view.inventorySize(owner, FabricInventoryKind::RegisterFifo);
    if (!contract ||
        registerFifoCount > std::numeric_limits<std::uint32_t>::max())
      return invalid("PE register-FIFO resource contract is unavailable");
    auto state = ::fabric::resolveTemporalPeRegisterFifoState(
        *contract, static_cast<std::uint32_t>(registerFifoCount),
        static_cast<std::uint32_t>(payload.registerFifo));
    if (!state)
      return state.takeError();
    result.resourceStates.push_back(resourceState(owner, state->ordinal()));
    break;
  }
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
    const FabricInventoryOwnerRef resourceOwner =
        FabricInventoryOwnerRef::of(payload.owner);
    if (const ::fabric::ResourceContract *contract =
            view.resourceContract(resourceOwner)) {
      const std::uint64_t inputCount =
          view.inventorySize(resourceOwner, FabricInventoryKind::SwitchInput);
      const std::uint64_t outputState = inputCount + payload.output;
      if (payload.input >= contract->stateCount() ||
          outputState >= contract->stateCount())
        return invalid("switch traversal resource state is out of range");
      result.resourceStates.push_back(
          resourceState(resourceOwner, payload.input));
      result.resourceStates.push_back(
          resourceState(resourceOwner, outputState));
    }
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
    const FabricInventoryOwnerRef resourceOwner =
        FabricInventoryOwnerRef::of(payload.owner);
    const ::fabric::ResourceContract *contract =
        view.resourceContract(resourceOwner);
    const FabricOrdinal state = static_cast<FabricOrdinal>(
        payload.mode == FabricFifoTraversalMode::Buffered
            ? ::fabric::FifoResourceState::BufferedQueue
            : ::fabric::FifoResourceState::BypassTransfer);
    if (!contract || state >= contract->stateCount())
      return invalid("FIFO traversal resource state is out of range");
    result.resourceStates.push_back(resourceState(resourceOwner, state));
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
    if (llvm::Error error = appendPatternStates(view, pattern->usePattern(),
                                                result.resourceStates))
      return std::move(error);
    break;
  }
  }
  if (reference.kind() !=
          FabricPhysicalTraversalKind::PeRegisterFifoTraversal &&
      (result.sources.empty() || result.destinations.empty()))
    return invalid("physical traversal has an empty endpoint relation");
  canonicalizeStates(result.resourceStates);
  return result;
}

} // namespace loom::fabric::detail

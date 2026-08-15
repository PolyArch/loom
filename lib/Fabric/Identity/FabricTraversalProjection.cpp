#include "FabricTraversalProjection.h"

#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/BoundaryTransfer.h"
#include "Fabric/IR/FifoResourceContract.h"
#include "Fabric/IR/SwitchResourceContract.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
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

FabricTraversalRequesterGroupView
patternRequester(const FabricUsePatternRef &pattern) {
  return {FabricTraversalRequesterGroupKind::UsePattern,
          pattern.owner.catalog(), pattern.ordinal};
}

FabricTraversalRequesterGroupView
switchRequester(FabricSwitchOccurrenceRef owner, FabricOrdinal requester) {
  return {FabricTraversalRequesterGroupKind::SwitchRequester,
          FabricInventoryOwnerRef::of(owner), requester};
}

llvm::Error
appendTimingPattern(const FabricArtifactView &view,
                    const FabricUsePatternRef &pattern,
                    FabricPhysicalTraversalView &traversal) {
  const FabricInventoryOwnerRef &owner = pattern.owner.catalog();
  const ::fabric::ResourceContract *contract = view.resourceContract(owner);
  if (!contract || pattern.ordinal >= contract->usePatternCount())
    return invalid("physical traversal selects an invalid timing pattern");
  const ::fabric::UsePatternTiming timing = contract->usePatternTiming(
      ::fabric::UsePatternKey(pattern.ordinal));
  traversal.timing.releaseLatencyCycles =
      std::max(traversal.timing.releaseLatencyCycles,
               timing.releaseLatencyCycles);
  traversal.timing.minimumInitiationIntervalCycles =
      std::max(traversal.timing.minimumInitiationIntervalCycles,
               timing.minimumInitiationIntervalCycles);
  return llvm::Error::success();
}

llvm::Error
appendImpliedUse(const FabricArtifactView &view,
                 const FabricUsePatternRef &pattern,
                 FabricTraversalRequesterGroupView requester,
                 FabricPhysicalTraversalView &traversal,
                 FabricTraversalUseOccupancyKind occupancyKind =
                     FabricTraversalUseOccupancyKind::MappingResident) {
  if (llvm::Error error =
          appendPatternStates(view, pattern, traversal.resourceStates))
    return error;
  if (llvm::Error error = appendTimingPattern(view, pattern, traversal))
    return error;
  traversal.impliedUses.push_back(
      {pattern, std::move(requester), occupancyKind});
  return llvm::Error::success();
}

void canonicalizeStates(std::vector<FabricResourceStateRef> &states) {
  llvm::sort(states, [](const FabricResourceStateRef &lhs,
                        const FabricResourceStateRef &rhs) {
    return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
  });
  states.erase(std::unique(states.begin(), states.end()), states.end());
}

template <typename Ref>
llvm::Error
appendModulePhysicalOwner(const Ref &reference,
                          std::vector<FabricModulePhysicalOwnerRef> &owners) {
  auto owner = FabricModulePhysicalOwnerRef::create(reference);
  if (!owner)
    return owner.takeError();
  owners.push_back(std::move(*owner));
  return llvm::Error::success();
}

FabricInventoryOwnerRef
inventoryOwner(const FabricModulePhysicalOwnerRef &owner) {
  return std::visit(
      [](const auto &value) -> FabricInventoryOwnerRef {
        using Type = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Type, LocalMemoryServiceRef>)
          return FabricInventoryOwnerRef::of(value.underlying());
        else
          return FabricInventoryOwnerRef::of(value);
      },
      owner.payload());
}

llvm::Expected<std::vector<FabricModulePhysicalOwnerRef>>
collectModulePhysicalOwners(const FabricArtifactView &view) {
  std::vector<FabricModulePhysicalOwnerRef> owners;
  if (view.rootKind() != FabricRootKind::Module)
    return owners;

  for (FabricPeOccurrenceRef pe : view.peOccurrences()) {
    if (llvm::Error error = appendModulePhysicalOwner(pe, owners))
      return std::move(error);
    for (FabricOrdinal ordinal = 0; ordinal < view.peResidentContextCount(pe);
         ++ordinal)
      if (llvm::Error error = appendModulePhysicalOwner(
              InstructionContextRef{pe, ordinal}, owners))
        return std::move(error);
  }
  for (FabricFuOccurrenceRef fu : view.fuOccurrences()) {
    if (llvm::Error error = appendModulePhysicalOwner(fu, owners))
      return std::move(error);
    const FabricInventoryOwnerRef fuOwner = FabricInventoryOwnerRef::of(fu);
    const std::uint64_t nodeCount =
        view.inventorySize(fuOwner, FabricInventoryKind::FuNode);
    for (FabricOrdinal ordinal = 0; ordinal < nodeCount; ++ordinal) {
      const std::optional<FabricFuNodeKind> kind =
          view.fuNodeKind(fuOwner, ordinal);
      if (!kind)
        return invalid("FU occurrence has an invalid node inventory");
      if (llvm::Error error = appendModulePhysicalOwner(
              FabricFuOccurrenceNodeRef{*kind, fu, ordinal}, owners))
        return std::move(error);
    }
  }
  for (FabricMemoryOccurrenceRef memory : view.memoryOccurrences()) {
    if (llvm::Error error = appendModulePhysicalOwner(memory, owners))
      return std::move(error);
    for (FabricMemoryOperationPortRef port : view.memoryOperationPorts(memory))
      if (llvm::Error error = appendModulePhysicalOwner(port, owners))
        return std::move(error);
    if (view.declaresLocalMemoryService(memory))
      if (llvm::Error error = appendModulePhysicalOwner(
              LocalMemoryServiceRef(FabricMemoryServiceRef::local(memory)),
              owners))
        return std::move(error);
  }
  for (FabricSwitchOccurrenceRef resource : view.switchOccurrences())
    if (llvm::Error error = appendModulePhysicalOwner(resource, owners))
      return std::move(error);
  for (FabricFifoOccurrenceRef resource : view.fifoOccurrences())
    if (llvm::Error error = appendModulePhysicalOwner(resource, owners))
      return std::move(error);
  for (FabricBoundaryOccurrenceRef resource : view.boundaryOccurrences())
    if (llvm::Error error = appendModulePhysicalOwner(resource, owners))
      return std::move(error);

  llvm::sort(owners, [](const FabricModulePhysicalOwnerRef &lhs,
                        const FabricModulePhysicalOwnerRef &rhs) {
    return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
  });
  if (std::adjacent_find(owners.begin(), owners.end()) != owners.end())
    return invalid("Module physical-owner inventory contains a duplicate");
  return owners;
}

} // namespace

llvm::Expected<std::vector<FabricInventoryOwnerRef>>
projectModuleResourceOwners(const FabricArtifactView &view) {
  std::vector<FabricInventoryOwnerRef> owners;
  auto physicalOwners = collectModulePhysicalOwners(view);
  if (!physicalOwners)
    return physicalOwners.takeError();
  for (const FabricModulePhysicalOwnerRef &physicalOwner : *physicalOwners) {
    FabricInventoryOwnerRef owner = inventoryOwner(physicalOwner);
    if (view.resourceContract(owner))
      owners.push_back(std::move(owner));
  }

  llvm::sort(owners, [](const FabricInventoryOwnerRef &lhs,
                        const FabricInventoryOwnerRef &rhs) {
    return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
  });
  if (std::adjacent_find(owners.begin(), owners.end()) != owners.end())
    return invalid("physical resource-owner inventory contains a duplicate");
  return owners;
}

llvm::Expected<std::vector<FabricModuleDomainMemberRef>>
projectModuleDomainMembers(const FabricArtifactView &view) {
  if (view.rootKind() != FabricRootKind::Module)
    return invalid("Module domain-member projection requires a Module root");
  const std::optional<FabricModuleTemplateRef> module =
      view.moduleRootTemplate();
  if (!module)
    return invalid("Module root has no unique canonical template");

  std::vector<FabricModuleDomainMemberRef> members;
  for (FabricPortDirection direction :
       {FabricPortDirection::Input, FabricPortDirection::Output})
    for (FabricOrdinal ordinal = 0;
         ordinal < view.moduleBoundaryEndpointCount(*module, direction);
         ++ordinal)
      members.push_back(FabricModuleDomainMemberRef::of(
          FabricModuleBoundaryEndpointRef{*module, direction, ordinal}));

  auto physicalOwners = collectModulePhysicalOwners(view);
  if (!physicalOwners)
    return physicalOwners.takeError();
  for (const FabricModulePhysicalOwnerRef &owner : *physicalOwners)
    members.push_back(FabricModuleDomainMemberRef::of(owner));

  llvm::sort(members, [](const FabricModuleDomainMemberRef &lhs,
                         const FabricModuleDomainMemberRef &rhs) {
    return canonicalFabricBytes(lhs) < canonicalFabricBytes(rhs);
  });
  if (std::adjacent_find(members.begin(), members.end()) != members.end())
    return invalid("Module domain-member inventory contains a duplicate");
  return members;
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
    auto pattern = ::fabric::resolveTemporalPeRegisterFifoPattern(
        *contract, static_cast<std::uint32_t>(registerFifoCount),
        static_cast<std::uint32_t>(payload.registerFifo),
        payload.role == FabricRegisterFifoPathRole::Write);
    if (!pattern)
      return pattern.takeError();
    const FabricUsePatternRef patternRef{FabricUsePatternOwnerRef(owner),
                                         pattern->ordinal()};
    if (llvm::Error error = appendImpliedUse(
            view, patternRef, patternRequester(patternRef), result))
      return std::move(error);
    if (payload.role == FabricRegisterFifoPathRole::Write)
      result.timing.architecturalLatencyCycles = 1;
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
      if (inputCount > std::numeric_limits<std::uint32_t>::max() ||
          payload.input > std::numeric_limits<std::uint32_t>::max() ||
          payload.output > std::numeric_limits<std::uint32_t>::max())
        return invalid("switch traversal domain exceeds u32");
      const std::uint64_t outputState = inputCount + payload.output;
      if (payload.input >= contract->stateCount() ||
          outputState >= contract->stateCount())
        return invalid("switch traversal resource state is out of range");
      result.resourceStates.push_back(
          resourceState(resourceOwner, payload.input));
      result.resourceStates.push_back(
          resourceState(resourceOwner, outputState));
      auto pattern = ::fabric::resolveSwitchTraversalPattern(
          *contract, static_cast<std::uint32_t>(inputCount),
          static_cast<std::uint32_t>(payload.input),
          static_cast<std::uint32_t>(payload.output));
      if (!pattern)
        return pattern.takeError();
      const FabricUsePatternRef patternRef{
          FabricUsePatternOwnerRef(resourceOwner), pattern->ordinal()};
      const FabricOrdinal requester =
          contract->usePattern(*pattern).requester.ordinal();
      const auto occupancyKind =
          view.switchSchedule(payload.owner) == ::fabric::Schedule::Temporal
              ? FabricTraversalUseOccupancyKind::RuntimeService
              : FabricTraversalUseOccupancyKind::MappingResident;
      if (llvm::Error error = appendImpliedUse(
              view, patternRef, switchRequester(payload.owner, requester),
              result, occupancyKind))
        return std::move(error);
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
    if (payload.mode == FabricFifoTraversalMode::Bypass) {
      const auto pattern =
          ::fabric::fifoUsePattern(::fabric::FifoUsePattern::BypassTransfer);
      if (pattern.ordinal() >= contract->usePatternCount())
        return invalid("FIFO bypass traversal has no use pattern");
      const FabricUsePatternRef patternRef{
          FabricUsePatternOwnerRef(resourceOwner), pattern.ordinal()};
      if (llvm::Error error = appendImpliedUse(
              view, patternRef, patternRequester(patternRef), result))
        return std::move(error);
    } else {
      result.timing.architecturalLatencyCycles = 1;
      for (::fabric::FifoUsePattern selected :
           {::fabric::FifoUsePattern::Enqueue,
            ::fabric::FifoUsePattern::Dequeue}) {
        const auto pattern = ::fabric::fifoUsePattern(selected);
        if (pattern.ordinal() >= contract->usePatternCount())
          return invalid("FIFO buffered traversal has no timing pattern");
        const FabricUsePatternRef patternRef{
            FabricUsePatternOwnerRef(resourceOwner), pattern.ordinal()};
        if (llvm::Error error = appendTimingPattern(view, patternRef, result))
          return std::move(error);
      }
    }
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
    const FabricInventoryOwnerRef resourceOwner =
        FabricInventoryOwnerRef::of(payload.owner);
    const ::fabric::ResourceContract *contract =
        view.resourceContract(resourceOwner);
    if (!contract || ::fabric::boundaryTransferPattern.ordinal() >=
                         contract->usePatternCount())
      return invalid("boundary traversal has no atomic use pattern");
    const FabricUsePatternRef patternRef{
        FabricUsePatternOwnerRef(resourceOwner),
        ::fabric::boundaryTransferPattern.ordinal()};
    if (llvm::Error error = appendImpliedUse(
            view, patternRef, patternRequester(patternRef), result))
      return std::move(error);
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
    if (llvm::Error error =
            appendImpliedUse(view, pattern->usePattern(),
                             patternRequester(pattern->usePattern()), result))
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

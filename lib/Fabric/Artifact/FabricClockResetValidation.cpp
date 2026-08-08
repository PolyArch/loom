#include "Fabric/Artifact/FabricClockResetValidation.h"

#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom::fabric {
namespace {

using OwnerKey = std::vector<std::uint8_t>;
using ClockMembership = std::map<OwnerKey, ClockDomainRef>;
using ModuleSlotPair =
    std::pair<FabricModuleDomainSlotRef, FabricModuleDomainSlotRef>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

OwnerKey ownerKey(const FabricInventoryOwnerRef &owner) {
  return canonicalFabricBytes(owner);
}

std::optional<ClockDomainRef>
ordinaryClock(const ClockMembership &membership,
              const FabricInventoryOwnerRef &owner) {
  auto found = membership.find(ownerKey(owner));
  return found == membership.end()
             ? std::nullopt
             : std::optional<ClockDomainRef>(found->second);
}

llvm::Expected<std::optional<ClockDomainRef>>
effectiveClock(const FabricSystemRootView &system,
               const ClockMembership &membership,
               const FabricTransportEndpointRef &endpoint) {
  const FabricArtifactView &artifact = system.artifact();
  auto direction = artifact.transportEndpointDirection(endpoint);
  if (!direction)
    return invalid("clock validation received an unknown transport endpoint");

  if (endpoint.owner.kind() ==
      FabricTransportEndpointOwnerKind::SystemTransportResource) {
    const SystemTransportResourceRef resource =
        std::get<SystemTransportResourceRef>(endpoint.owner.payload);
    if (const ClockCrossingContractRecord *crossing =
            system.clockCrossing(resource))
      return std::optional<ClockDomainRef>(*direction ==
                                                   FabricPortDirection::Input
                                               ? crossing->sourceClock()
                                               : crossing->destinationClock());
  }
  return ordinaryClock(membership, projectFabricInventoryOwner(endpoint.owner));
}

llvm::Expected<std::optional<ClockDomainRef>>
effectiveClock(const FabricSystemRootView &system,
               const ClockMembership &membership,
               const FabricMemoryEndpointRef &endpoint) {
  const FabricArtifactView &artifact = system.artifact();
  if (!artifact.memoryEndpointRole(endpoint))
    return invalid("clock validation received an unknown memory endpoint");
  return ordinaryClock(membership, projectFabricInventoryOwner(endpoint.owner));
}

template <typename Owner>
llvm::Expected<FabricModuleDomainMemberRef>
modulePhysicalMember(const Owner &owner) {
  auto physical = FabricModulePhysicalOwnerRef::create(owner);
  if (!physical)
    return physical.takeError();
  return FabricModuleDomainMemberRef::of(*physical);
}

llvm::Expected<FabricModuleDomainMemberRef>
moduleMember(const FabricTransportEndpointOwnerRef &owner) {
  return std::visit(
      [](const auto &value) -> llvm::Expected<FabricModuleDomainMemberRef> {
        using Owner = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Owner, FabricPeOccurrenceRef> ||
                      std::is_same_v<Owner, FabricFuOccurrenceRef> ||
                      std::is_same_v<Owner, FabricMemoryOccurrenceRef> ||
                      std::is_same_v<Owner, FabricSwitchOccurrenceRef> ||
                      std::is_same_v<Owner, FabricFifoOccurrenceRef> ||
                      std::is_same_v<Owner, FabricBoundaryOccurrenceRef>) {
          return modulePhysicalMember(value);
        }
        return invalid("Module connection names a non-Module owner");
      },
      owner.payload);
}

llvm::Expected<FabricModuleDomainMemberRef>
moduleMember(const FabricMemoryEndpointOwnerRef &owner) {
  return std::visit(
      [](const auto &value) -> llvm::Expected<FabricModuleDomainMemberRef> {
        using Owner = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Owner, FabricMemoryOccurrenceRef>)
          return modulePhysicalMember(value);
        return invalid("Module memory attachment names a non-Module owner");
      },
      owner.payload);
}

llvm::Expected<FabricModuleDomainMemberRef>
moduleMember(FabricMemoryOccurrenceRef memory,
             const ::fabric::MemoryDispatchTarget &target) {
  if (std::holds_alternative<::fabric::LocalMemoryDispatchTarget>(target))
    return modulePhysicalMember(
        LocalMemoryServiceRef(FabricMemoryServiceRef::local(memory)));
  return modulePhysicalMember(memory);
}

llvm::Expected<FabricModuleDomainMemberRef>
moduleMember(const FabricArtifactView &artifact,
             FabricFuOccurrenceRef occurrence,
             const FabricFuCapabilityTemplateEndpointRef &endpoint) {
  const std::optional<FabricFuTemplateRef> definition =
      artifact.fuTemplateOf(occurrence);
  if (!definition)
    return invalid("Module FU occurrence has no canonical template");
  return std::visit(
      [&](const auto &value) -> llvm::Expected<FabricModuleDomainMemberRef> {
        using Endpoint = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Endpoint, FabricFuTemplatePortRef>) {
          if (value.fu != *definition)
            return invalid("Module FU edge names a foreign template port");
          return modulePhysicalMember(occurrence);
        } else {
          if (value.node.fu != *definition)
            return invalid("Module FU edge names a foreign template node");
          auto node =
              deriveFabricFuOccurrenceNode(artifact, value.node, occurrence);
          if (!node)
            return node.takeError();
          return modulePhysicalMember(*node);
        }
      },
      endpoint.payload);
}

llvm::Expected<ModuleSlotPair>
moduleSlots(const FabricModuleRootView &module,
            const FabricModuleDomainMemberRef &member) {
  std::optional<FabricModuleDomainSlotRef> clock;
  std::optional<FabricModuleDomainSlotRef> reset;
  for (const ModuleDomainAssignment &assignment : module.domainAssignments()) {
    if (assignment.member != member)
      continue;
    std::optional<FabricModuleDomainSlotRef> *selected = nullptr;
    switch (assignment.slot.kind) {
    case FabricClockResetKind::Clock:
      selected = &clock;
      break;
    case FabricClockResetKind::Reset:
      selected = &reset;
      break;
    }
    if (!selected || *selected)
      return invalid("Module member has duplicate or unknown domain rows");
    *selected = assignment.slot;
  }
  if (!clock || !reset)
    return invalid("Module member has no complete Clock and Reset assignment");
  return ModuleSlotPair{*clock, *reset};
}

llvm::Error
requireSameModuleSlots(const FabricModuleRootView &module,
                       const FabricModuleDomainMemberRef &source,
                       const FabricModuleDomainMemberRef &destination) {
  auto sourceSlots = moduleSlots(module, source);
  if (!sourceSlots)
    return sourceSlots.takeError();
  auto destinationSlots = moduleSlots(module, destination);
  if (!destinationSlots)
    return destinationSlots.takeError();
  if (*sourceSlots != *destinationSlots)
    return invalid(
        "ordinary Module connection crosses symbolic Clock or Reset slots");
  return llvm::Error::success();
}

} // namespace

llvm::Error validateModuleClockReset(const FabricModuleRootView &module) {
  const FabricArtifactView &artifact = module.artifact();
  for (const FabricPointConnectionPayload &connection :
       artifact.pointConnections()) {
    auto source = moduleMember(connection.source.owner);
    if (!source)
      return source.takeError();
    auto destination = moduleMember(connection.destination.owner);
    if (!destination)
      return destination.takeError();
    if (llvm::Error error =
            requireSameModuleSlots(module, *source, *destination))
      return error;
  }
  for (const FabricModuleBoundaryTransportAttachmentView &attachment :
       artifact.moduleBoundaryTransportAttachments()) {
    auto internal = moduleMember(attachment.endpoint.owner);
    if (!internal)
      return internal.takeError();
    if (llvm::Error error = requireSameModuleSlots(
            module, FabricModuleDomainMemberRef::of(attachment.boundary),
            *internal))
      return error;
  }
  for (const FabricModuleBoundaryMemoryAttachmentView &attachment :
       artifact.moduleBoundaryMemoryAttachments()) {
    auto internal = moduleMember(attachment.endpoint.owner);
    if (!internal)
      return internal.takeError();
    if (llvm::Error error = requireSameModuleSlots(
            module, FabricModuleDomainMemberRef::of(attachment.boundary),
            *internal))
      return error;
  }
  for (const FabricMemoryServiceConnectionPayload &connection :
       artifact.memoryServiceConnections()) {
    auto requester = moduleMember(connection.source.owner);
    if (!requester)
      return requester.takeError();
    auto provider = moduleMember(connection.destination.owner);
    if (!provider)
      return provider.takeError();
    if (llvm::Error error =
            requireSameModuleSlots(module, *requester, *provider))
      return error;
  }
  for (FabricMemoryOccurrenceRef memory : artifact.memoryOccurrences()) {
    const ::fabric::MemoryConnectivityContractRecord *connectivity =
        artifact.memoryConnectivity(memory);
    if (!connectivity)
      return invalid("Module memory occurrence has no connectivity contract");
    llvm::ArrayRef<FabricMemoryOperationPortRef> operationPorts =
        artifact.memoryOperationPorts(memory);
    if (connectivity->operationPorts().size() != operationPorts.size())
      return invalid("Module memory dispatch inventory is inconsistent");
    for (auto [ordinal, dispatch] :
         llvm::enumerate(connectivity->operationPorts())) {
      auto source = modulePhysicalMember(operationPorts[ordinal]);
      if (!source)
        return source.takeError();
      for (llvm::ArrayRef<::fabric::MemoryDispatchTarget> targets :
           dispatch.capabilityTargetDomains)
        for (const ::fabric::MemoryDispatchTarget &target : targets) {
          auto destination = moduleMember(memory, target);
          if (!destination)
            return destination.takeError();
          if (llvm::Error error =
                  requireSameModuleSlots(module, *source, *destination))
            return error;
        }
    }
    auto subordinateSource = modulePhysicalMember(memory);
    if (!subordinateSource)
      return subordinateSource.takeError();
    for (const ::fabric::MemorySubordinateDispatchDeclaration &dispatch :
         connectivity->subordinateEndpoints())
      for (const ::fabric::MemoryDispatchTarget &target :
           dispatch.targetDomain) {
        auto destination = moduleMember(memory, target);
        if (!destination)
          return destination.takeError();
        if (llvm::Error error = requireSameModuleSlots(
                module, *subordinateSource, *destination))
          return error;
      }
    const auto endpointOwners = [&](FabricOrdinal endpoint)
        -> llvm::Expected<llvm::SmallVector<FabricModuleDomainMemberRef, 2>> {
      llvm::SmallVector<FabricModuleDomainMemberRef, 2> owners;
      for (FabricMemoryOperationPortRef port : operationPorts) {
        const MemoryOperationPortView *record =
            artifact.memoryOperationPort(port);
        if (!record)
          return invalid("Module memory operation port cannot be resolved");
        if (!llvm::is_contained(record->endpointInventory(), endpoint))
          continue;
        auto owner = modulePhysicalMember(port);
        if (!owner)
          return owner.takeError();
        owners.push_back(std::move(*owner));
      }
      if (owners.empty())
        return invalid("Module memory token endpoint has no operation port");
      return owners;
    };
    for (const ::fabric::MemoryInternalConnectionDeclaration &connection :
         connectivity->internalConnections()) {
      auto sources = endpointOwners(connection.sourceEndpointOrdinal);
      if (!sources)
        return sources.takeError();
      auto destinations = endpointOwners(connection.sinkEndpointOrdinal);
      if (!destinations)
        return destinations.takeError();
      for (const FabricModuleDomainMemberRef &source : *sources)
        for (const FabricModuleDomainMemberRef &destination : *destinations)
          if (llvm::Error error =
                  requireSameModuleSlots(module, source, destination))
            return error;
    }
  }
  for (const FabricPhysicalTraversalView &traversal :
       artifact.physicalTraversals()) {
    const auto *selector =
        std::get_if<FabricPeSelectorPayload>(&traversal.reference.payload);
    if (!selector)
      continue;
    auto source = moduleMember(selector->source.owner);
    if (!source)
      return source.takeError();
    auto destination = moduleMember(selector->destination.owner);
    if (!destination)
      return destination.takeError();
    if (llvm::Error error =
            requireSameModuleSlots(module, *source, *destination))
      return error;
  }
  for (FabricFuOccurrenceRef occurrence : artifact.fuOccurrences()) {
    const std::optional<FabricFuTemplateRef> definition =
        artifact.fuTemplateOf(occurrence);
    if (!definition)
      return invalid("Module FU occurrence has no canonical template");
    for (const FabricFuCapabilityTemplateRecord &capability :
         artifact.fuCapabilityTemplates(*definition))
      for (const FabricFuCapabilityTemplateEdge &edge :
           capability.activeEdges) {
        auto source = moduleMember(artifact, occurrence, edge.source);
        if (!source)
          return source.takeError();
        auto destination = moduleMember(artifact, occurrence, edge.destination);
        if (!destination)
          return destination.takeError();
        if (llvm::Error error =
                requireSameModuleSlots(module, *source, *destination))
          return error;
      }
  }
  for (const FabricModuleBoundaryTransportPassthroughView &passthrough :
       artifact.moduleBoundaryTransportPassthroughs())
    if (llvm::Error error = requireSameModuleSlots(
            module, FabricModuleDomainMemberRef::of(passthrough.input),
            FabricModuleDomainMemberRef::of(passthrough.output)))
      return error;
  return llvm::Error::success();
}

llvm::Expected<ValidatedClockResetView>
validateClockReset(FabricSystemRootView system) {
  const FabricArtifactView &artifact = system.artifact();
  ClockMembership clockMembership;

  for (HardwareDomainRef domain : system.hardwareDomains()) {
    if (llvm::Error error = validateFabricRef(artifact, domain))
      return std::move(error);
    const HardwareDomainContractRecord *record =
        system.hardwareDomainContract(domain);
    if (!record || artifact.hardwareDomainKind(domain) != record->kind())
      return invalid("hardware-domain view is incomplete or inconsistent");
    if (record->kind() != FabricHardwareDomainKind::Clock)
      continue;
    const ClockDomainRef clock(domain);
    for (const FabricInventoryOwnerRef &member : record->members()) {
      if (llvm::Error error = validateFabricRef(artifact, member))
        return std::move(error);
      if (!clockMembership.emplace(ownerKey(member), clock).second)
        return invalid("one Fabric owner belongs to multiple Clock domains");
    }
  }

  for (HardwareDomainRef domain : system.hardwareDomains()) {
    const HardwareDomainContractRecord *record =
        system.hardwareDomainContract(domain);
    if (!record)
      return invalid("hardware-domain view is incomplete");
    if (const auto *reset =
            std::get_if<ResetDomainContractRecord>(&record->contract())) {
      if (!reset->synchronousTo())
        continue;
      if (llvm::Error error =
              validateFabricRef(artifact, *reset->synchronousTo()))
        return std::move(error);
      for (const FabricInventoryOwnerRef &member : record->members()) {
        std::optional<ClockDomainRef> memberClock =
            ordinaryClock(clockMembership, member);
        if (!memberClock || *memberClock != *reset->synchronousTo())
          return invalid("synchronous reset member is not in its declared "
                         "Clock domain");
      }
      continue;
    }
    if (const auto *consistency =
            std::get_if<::fabric::MemoryConsistencyContract>(
                &record->contract()))
      if (const auto *bounded = std::get_if<::fabric::BoundedCompletion>(
              &consistency->progress()))
        if (llvm::Error error =
                validateFabricRef(artifact, bounded->progressClock))
          return std::move(error);
  }

  for (SystemTransportResourceRef resource : system.transportResources()) {
    const ClockCrossingContractRecord *crossing =
        system.clockCrossing(resource);
    if (!crossing)
      continue;
    if (ordinaryClock(clockMembership, FabricInventoryOwnerRef::of(resource)))
      return invalid(
          "clock-crossing resource has ordinary Clock-domain membership");
    if (llvm::Error error =
            validateFabricRef(artifact, crossing->sourceClock()))
      return std::move(error);
    if (llvm::Error error =
            validateFabricRef(artifact, crossing->destinationClock()))
      return std::move(error);
  }

  for (const FabricPointConnectionPayload &connection :
       artifact.pointConnections()) {
    auto source = effectiveClock(system, clockMembership, connection.source);
    if (!source)
      return source.takeError();
    auto destination =
        effectiveClock(system, clockMembership, connection.destination);
    if (!destination)
      return destination.takeError();
    if (*source != *destination)
      return invalid("point connection crosses Clock domains without an "
                     "explicit crossing resource");
  }

  for (const FabricMemoryServiceConnectionPayload &connection :
       artifact.memoryServiceConnections()) {
    auto source = effectiveClock(system, clockMembership, connection.source);
    if (!source)
      return source.takeError();
    auto destination =
        effectiveClock(system, clockMembership, connection.destination);
    if (!destination)
      return destination.takeError();
    if (*source != *destination)
      return invalid("memory-service connection crosses Clock domains");
  }

  return ValidatedClockResetView(std::move(system));
}

} // namespace loom::fabric

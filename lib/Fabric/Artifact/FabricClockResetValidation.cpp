#include "Fabric/Artifact/FabricClockResetValidation.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace loom::fabric {
namespace {

using OwnerKey = std::vector<std::uint8_t>;
using ClockMembership = std::map<OwnerKey, ClockDomainRef>;

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

} // namespace

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

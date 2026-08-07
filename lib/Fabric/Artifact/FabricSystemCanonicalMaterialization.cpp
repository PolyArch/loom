#include "FabricSystemCanonicalLabeling.h"

#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/SystemServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

struct CanonicalEntity {
  FabricEntityKind kind;
  FabricEntityId id;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

DenseI8ArrayAttr denseBytes(MLIRContext *context,
                            llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return DenseI8ArrayAttr::get(context, signedBytes);
}

class SystemReferenceRemapper {
public:
  static llvm::Expected<SystemReferenceRemapper>
  create(::fabric::SystemOp root,
         const FabricSystemCanonicalLabeling &labeling) {
    SystemReferenceRemapper remapper(labeling.sourceDependencyToCanonical);
    for (const FabricSystemEntityCarrier &carrier : labeling.carriers) {
      if (!carrier.op)
        return invalid("System canonical entity has no operation carrier");
      auto authored =
          carrier.op->getAttrOfType<::fabric::EntityIdAttr>("entity_id");
      if (!authored)
        continue;
      if (!remapper.entities_
               .try_emplace(authored.getId(),
                            CanonicalEntity{carrier.kind, carrier.id})
               .second)
        return invalid("System materialization repeats a provisional EntityId");
    }

    for (Operation &operation : root.getBody().front()) {
      auto pattern = dyn_cast<::fabric::SystemTransferPatternOp>(&operation);
      if (!pattern)
        continue;
      auto record = decodeSystemTransferPatternRecord(
          unsignedBytes(pattern.getContractAttr()));
      if (!record)
        return record.takeError();
      auto ordinal = labeling.transferPatternOrdinalByOperation.find(
          pattern.getOperation());
      if (ordinal == labeling.transferPatternOrdinalByOperation.end())
        return invalid("System transfer pattern has no canonical ordinal");
      auto resource = remapper.entity(record->pattern().resource);
      if (!resource)
        return resource.takeError();
      const std::vector<std::uint8_t> oldKey =
          canonicalFabricBytes(record->pattern());
      const FabricTransferPatternRef replacement{*resource, ordinal->second};
      if (!remapper.patterns_.emplace(oldKey, replacement).second)
        return invalid("System materialization repeats a transfer-pattern ref");
    }
    return remapper;
  }

  template <FabricEntityKind Kind>
  llvm::Expected<FabricTypedEntityRef<Kind>>
  entity(const FabricTypedEntityRef<Kind> &reference) const {
    auto found = entities_.find(reference.id());
    if (found == entities_.end())
      return invalid("System reference names an unknown provisional entity");
    if (found->second.kind != Kind)
      return invalid(
          "System reference names the wrong provisional entity kind");
    return FabricTypedEntityRef<Kind>(found->second.id);
  }

  llvm::Expected<SpatialCoreOccurrenceRef>
  remap(const SpatialCoreOccurrenceRef &reference) const {
    auto core = entity(reference.core);
    if (!core)
      return core.takeError();
    return SpatialCoreOccurrenceRef{*core};
  }

  llvm::Expected<InstructionCoreContextRef>
  remap(const InstructionCoreContextRef &reference) const {
    auto core = entity(reference.core);
    if (!core)
      return core.takeError();
    return InstructionCoreContextRef{*core};
  }

  llvm::Expected<InstructionContextRef>
  remap(const InstructionContextRef &reference) const {
    auto pe = entity(reference.pe);
    if (!pe)
      return pe.takeError();
    return InstructionContextRef{*pe, reference.ordinal};
  }

  llvm::Expected<FabricFuTemplateNodeRef>
  remap(const FabricFuTemplateNodeRef &reference) const {
    auto fu = entity(reference.fu);
    if (!fu)
      return fu.takeError();
    return FabricFuTemplateNodeRef{reference.node, *fu, reference.ordinal};
  }

  llvm::Expected<FabricFuOccurrenceNodeRef>
  remap(const FabricFuOccurrenceNodeRef &reference) const {
    auto fu = entity(reference.fu);
    if (!fu)
      return fu.takeError();
    return FabricFuOccurrenceNodeRef{reference.node, *fu, reference.ordinal};
  }

  llvm::Expected<FabricMemoryOperationPortRef>
  remap(const FabricMemoryOperationPortRef &reference) const {
    auto memory = entity(reference.memory);
    if (!memory)
      return memory.takeError();
    return FabricMemoryOperationPortRef{*memory, reference.ordinal};
  }

  llvm::Expected<FabricTransferPatternRef>
  remap(const FabricTransferPatternRef &reference) const {
    auto found = patterns_.find(canonicalFabricBytes(reference));
    if (found == patterns_.end())
      return invalid("System reference names an unknown transfer pattern");
    return found->second;
  }

  template <FabricEntityKind Kind>
  llvm::Expected<FabricTypedEntityRef<Kind>>
  remap(const FabricTypedEntityRef<Kind> &reference) const {
    return entity(reference);
  }

  llvm::Expected<FabricMemoryServiceRef>
  remap(const FabricMemoryServiceRef &reference) const {
    return std::visit(
        [&](const auto &payload) -> llvm::Expected<FabricMemoryServiceRef> {
          auto mapped = remap(payload);
          if (!mapped)
            return mapped.takeError();
          using T = std::decay_t<decltype(payload)>;
          if constexpr (std::is_same_v<T, FabricMemoryOccurrenceRef>)
            return FabricMemoryServiceRef::local(*mapped);
          else
            return FabricMemoryServiceRef::system(*mapped);
        },
        reference.payload);
  }

  llvm::Expected<FabricTransportEndpointOwnerRef>
  remap(const FabricTransportEndpointOwnerRef &reference) const {
    return std::visit(
        [&](const auto &payload)
            -> llvm::Expected<FabricTransportEndpointOwnerRef> {
          auto mapped = remap(payload);
          if (!mapped)
            return mapped.takeError();
          return FabricTransportEndpointOwnerRef::of(*mapped);
        },
        reference.payload);
  }

  llvm::Expected<FabricMemoryEndpointOwnerRef>
  remap(const FabricMemoryEndpointOwnerRef &reference) const {
    return std::visit(
        [&](const auto &payload)
            -> llvm::Expected<FabricMemoryEndpointOwnerRef> {
          auto mapped = remap(payload);
          if (!mapped)
            return mapped.takeError();
          return FabricMemoryEndpointOwnerRef::of(*mapped);
        },
        reference.payload);
  }

  llvm::Expected<FabricInventoryOwnerRef>
  remap(const FabricInventoryOwnerRef &reference) const {
    return std::visit(
        [&](const auto &payload) -> llvm::Expected<FabricInventoryOwnerRef> {
          auto mapped = remap(payload);
          if (!mapped)
            return mapped.takeError();
          return FabricInventoryOwnerRef::of(*mapped);
        },
        reference.payload);
  }

  llvm::Expected<FabricTransportEndpointRef>
  remap(const FabricTransportEndpointRef &reference) const {
    auto owner = remap(reference.owner);
    if (!owner)
      return owner.takeError();
    return FabricTransportEndpointRef{*owner, reference.ordinal};
  }

  llvm::Expected<FabricMemoryEndpointRef>
  remap(const FabricMemoryEndpointRef &reference) const {
    auto owner = remap(reference.owner);
    if (!owner)
      return owner.takeError();
    return FabricMemoryEndpointRef{*owner, reference.ordinal};
  }

  llvm::Expected<FabricUsePatternRef>
  remap(const FabricUsePatternRef &reference) const {
    auto owner = remap(reference.owner.catalog());
    if (!owner)
      return owner.takeError();
    return FabricUsePatternRef{FabricUsePatternOwnerRef(*owner),
                               reference.ordinal};
  }

  llvm::Expected<FabricMemoryServiceRegionRef>
  remap(const FabricMemoryServiceRegionRef &reference) const {
    auto service = remap(reference.service);
    if (!service)
      return service.takeError();
    return FabricMemoryServiceRegionRef{*service, reference.ordinal};
  }

  llvm::Expected<ClockDomainRef> remap(const ClockDomainRef &reference) const {
    auto domain = entity(reference.underlying());
    if (!domain)
      return domain.takeError();
    return ClockDomainRef(*domain);
  }

  llvm::Expected<MemoryConsistencyDomainRef>
  remap(const MemoryConsistencyDomainRef &reference) const {
    auto domain = entity(reference.underlying());
    if (!domain)
      return domain.takeError();
    return MemoryConsistencyDomainRef(*domain);
  }

  llvm::Expected<SubordinateEndpointRef>
  remap(const SubordinateEndpointRef &reference) const {
    auto endpoint = remap(reference.underlying());
    if (!endpoint)
      return endpoint.takeError();
    return SubordinateEndpointRef(*endpoint);
  }

  llvm::Expected<FabricImportedModuleTargetRef>
  remap(const FabricImportedModuleTargetRef &reference) const {
    auto dependency = dependencyOrdinal(reference.dependencyOrdinal);
    if (!dependency)
      return dependency.takeError();
    return FabricImportedModuleTargetRef{*dependency, reference.target};
  }

  llvm::Expected<FabricImportedModuleBoundaryEndpointRef>
  remap(const FabricImportedModuleBoundaryEndpointRef &reference) const {
    auto dependency = dependencyOrdinal(reference.dependencyOrdinal);
    if (!dependency)
      return dependency.takeError();
    return FabricImportedModuleBoundaryEndpointRef{*dependency,
                                                   reference.target};
  }

private:
  explicit SystemReferenceRemapper(
      llvm::ArrayRef<std::uint64_t> sourceDependencyToCanonical)
      : sourceDependencyToCanonical_(sourceDependencyToCanonical.begin(),
                                     sourceDependencyToCanonical.end()) {}

  llvm::Expected<std::uint64_t>
  dependencyOrdinal(std::uint64_t sourceOrdinal) const {
    if (sourceOrdinal >= sourceDependencyToCanonical_.size())
      return invalid("System field references a dependency outside its table");
    return sourceDependencyToCanonical_[sourceOrdinal];
  }

  llvm::DenseMap<FabricEntityId, CanonicalEntity> entities_;
  std::map<std::vector<std::uint8_t>, FabricTransferPatternRef> patterns_;
  std::vector<std::uint64_t> sourceDependencyToCanonical_;
};

llvm::Expected<ServiceProgress>
remapProgress(const ServiceProgress &progress,
              const SystemReferenceRemapper &remapper) {
  if (std::holds_alternative<::fabric::FairEventual>(progress))
    return ServiceProgress(std::in_place_type<::fabric::FairEventual>);
  const auto &bounded = std::get<::fabric::BoundedCompletion>(progress);
  auto clock = remapper.remap(bounded.progressClock);
  if (!clock)
    return clock.takeError();
  return ServiceProgress(
      std::in_place_type<::fabric::BoundedCompletion>,
      ::fabric::BoundedCompletion{*clock, bounded.maxIssueToRetireTicks});
}

llvm::Expected<CanonicalServiceCapabilityRecord>
remapCapability(const CanonicalServiceCapabilityRecord &capability,
                const SystemReferenceRemapper &remapper) {
  std::optional<CanonicalServiceCapabilityDomain> domain;
  if (const auto *message =
          std::get_if<MessageTransferCapabilityDomain>(&capability.domain())) {
    auto mapped =
        MessageTransferCapabilityDomain::fromCanonical(message->payloadTypes());
    if (!mapped)
      return mapped.takeError();
    domain.emplace(std::in_place_type<MessageTransferCapabilityDomain>,
                   std::move(*mapped));
  } else if (const auto *addressed =
                 std::get_if<AddressedMemoryCapabilityDomain>(
                     &capability.domain())) {
    std::optional<MemoryConsistencyDomainRef> consistency;
    if (addressed->consistencyDomain()) {
      auto mapped = remapper.remap(*addressed->consistencyDomain());
      if (!mapped)
        return mapped.takeError();
      consistency = *mapped;
    }
    auto mapped = AddressedMemoryCapabilityDomain::create(
        addressed->actorContracts(), addressed->accesses(),
        addressed->addressBytes(), addressed->serviceBeatWidthBits(),
        std::move(consistency));
    if (!mapped)
      return mapped.takeError();
    domain.emplace(std::in_place_type<AddressedMemoryCapabilityDomain>,
                   std::move(*mapped));
  } else {
    const auto &fence = std::get<FenceCapabilityDomain>(capability.domain());
    auto consistency = remapper.remap(fence.consistencyDomain());
    if (!consistency)
      return consistency.takeError();
    auto mapped =
        FenceCapabilityDomain::create(fence.actorContracts(), *consistency);
    if (!mapped)
      return mapped.takeError();
    domain.emplace(std::in_place_type<FenceCapabilityDomain>,
                   std::move(*mapped));
  }

  auto rateClock = remapper.remap(capability.rate().rateClock());
  if (!rateClock)
    return rateClock.takeError();
  auto progress = remapProgress(capability.rate().progress(), remapper);
  if (!progress)
    return progress.takeError();
  auto rate = ServiceRateContractRecord::create(
      *rateClock, capability.rate().operationsPerWindow(),
      capability.rate().windowTicks(), capability.rate().maxOutstanding(),
      std::move(*progress));
  if (!rate)
    return rate.takeError();
  return CanonicalServiceCapabilityRecord::create(
      capability.kind(), capability.role(), std::move(*domain),
      std::move(*rate));
}

llvm::Error remapAccCore(::fabric::SystemAccCoreOp core,
                         const SystemReferenceRemapper &remapper) {
  auto target = decodeFabricImportedModuleTargetRef(
      unsignedBytes(core.getSpatialCoreAttr()));
  if (!target)
    return target.takeError();
  auto mapped = remapper.remap(*target);
  if (!mapped)
    return mapped.takeError();
  core.setSpatialCoreAttr(denseBytes(
      core.getContext(), encodeFabricImportedModuleTargetRef(*mapped)));
  return llvm::Error::success();
}

llvm::Error remapMemoryService(::fabric::SystemMemoryServiceOp service,
                               const SystemReferenceRemapper &remapper) {
  auto record = ::fabric::decodeMemoryServiceContractRecord(
      unsignedBytes(service.getServiceContractAttr().getRecord()),
      service.getContext(), ::fabric::MemoryServiceOwnerKind::System);
  if (!record)
    return record.takeError();
  ::fabric::MemoryServiceContractDeclaration declaration{
      std::vector<::fabric::MemoryServiceRegionDeclaration>(
          record->regions().begin(), record->regions().end()),
      record->resourceContract(),
      {}};
  for (const ::fabric::MemoryServiceCapabilityDeclaration &capability :
       record->capabilities()) {
    ::fabric::MemoryServiceConsistencyBinding consistency =
        capability.consistencyBinding;
    if (const auto *domain =
            std::get_if<MemoryConsistencyDomainRef>(&consistency)) {
      auto mapped = remapper.remap(*domain);
      if (!mapped)
        return mapped.takeError();
      consistency = *mapped;
    }
    declaration.capabilities.push_back(
        {capability.actorContractDomain, capability.accessDomain,
         capability.serviceRegionOrdinals, capability.serviceBeatWidthBits,
         capability.admissibleUsePatterns, std::move(consistency)});
  }
  auto mapped = ::fabric::MemoryServiceContractRecord::create(
      service.getContext(), ::fabric::MemoryServiceOwnerKind::System,
      std::move(declaration));
  if (!mapped)
    return mapped.takeError();
  auto bytes = ::fabric::encodeMemoryServiceContractRecord(*mapped);
  if (!bytes)
    return bytes.takeError();
  service.setServiceContractAttr(::fabric::MemoryServiceContractAttr::get(
      service.getContext(), denseBytes(service.getContext(), *bytes)));
  return llvm::Error::success();
}

llvm::Error remapServiceEndpoint(::fabric::SystemServiceEndpointOp endpoint,
                                 const SystemReferenceRemapper &remapper) {
  auto owner = decodeSystemServiceEndpointOwnerRef(
      unsignedBytes(endpoint.getOwnerAttr()));
  if (!owner)
    return owner.takeError();
  auto mappedOwner = remapper.remap(owner->owner());
  if (!mappedOwner)
    return mappedOwner.takeError();
  auto ownerRef = SystemServiceEndpointOwnerRef::create(*mappedOwner);
  if (!ownerRef)
    return ownerRef.takeError();

  auto capabilities = decodeCanonicalServiceCapabilitySet(
      unsignedBytes(endpoint.getCapabilitiesAttr()), endpoint.getContext());
  if (!capabilities)
    return capabilities.takeError();
  std::vector<CanonicalServiceCapabilityRecord> mappedCapabilities;
  for (const CanonicalServiceCapabilityRecord &capability :
       capabilities->capabilities()) {
    auto mapped = remapCapability(capability, remapper);
    if (!mapped)
      return mapped.takeError();
    mappedCapabilities.push_back(std::move(*mapped));
  }
  auto capabilitySet =
      CanonicalServiceCapabilitySet::create(std::move(mappedCapabilities));
  if (!capabilitySet)
    return capabilitySet.takeError();
  auto capabilityBytes = encodeCanonicalServiceCapabilitySet(*capabilitySet);
  if (!capabilityBytes)
    return capabilityBytes.takeError();
  endpoint.setOwnerAttr(denseBytes(
      endpoint.getContext(), encodeSystemServiceEndpointOwnerRef(*ownerRef)));
  endpoint.setCapabilitiesAttr(
      denseBytes(endpoint.getContext(), *capabilityBytes));
  return llvm::Error::success();
}

llvm::Error remapServiceTransform(::fabric::SystemServiceTransformOp transform,
                                  const SystemReferenceRemapper &remapper) {
  auto record = decodeSystemServiceTransformRecord(
      unsignedBytes(transform.getContractAttr()));
  if (!record)
    return record.takeError();
  std::vector<FabricMemoryEndpointRef> inputs;
  std::vector<FabricMemoryEndpointRef> outputs;
  for (const FabricMemoryEndpointRef &input : record->inputs()) {
    auto mapped = remapper.remap(input);
    if (!mapped)
      return mapped.takeError();
    inputs.push_back(*mapped);
  }
  for (const FabricMemoryEndpointRef &output : record->outputs()) {
    auto mapped = remapper.remap(output);
    if (!mapped)
      return mapped.takeError();
    outputs.push_back(*mapped);
  }
  ServiceTransformContract contract = record->contract();
  if (auto *coherent = std::get_if<CoherentMemoryTransform>(&contract)) {
    auto domain = remapper.remap(coherent->consistencyDomain);
    if (!domain)
      return domain.takeError();
    coherent->consistencyDomain = *domain;
    for (CoherentMemoryRegionCorrespondence &region : coherent->regions) {
      auto input = remapper.remap(region.input);
      if (!input)
        return input.takeError();
      auto output = remapper.remap(region.output);
      if (!output)
        return output.takeError();
      region = {*input, *output};
    }
  }
  auto mapped = SystemServiceTransformRecord::create(
      std::move(inputs), std::move(outputs), std::move(contract));
  if (!mapped)
    return mapped.takeError();
  auto bytes = encodeSystemServiceTransformRecord(*mapped);
  if (!bytes)
    return bytes.takeError();
  transform.setContractAttr(denseBytes(transform.getContext(), *bytes));
  return llvm::Error::success();
}

llvm::Expected<::fabric::MemoryConsistencyContract>
remapConsistencyContract(const ::fabric::MemoryConsistencyContract &contract,
                         const SystemReferenceRemapper &remapper) {
  std::vector<::fabric::MemoryConsistencyParticipant> participants;
  for (const ::fabric::MemoryConsistencyParticipant &participant :
       contract.participants()) {
    if (const auto *service =
            std::get_if<FabricMemoryServiceRef>(&participant.payload)) {
      auto mapped = remapper.remap(*service);
      if (!mapped)
        return mapped.takeError();
      participants.push_back(
          ::fabric::MemoryConsistencyParticipant::service(*mapped));
    } else {
      auto mapped =
          remapper.remap(std::get<SubordinateEndpointRef>(participant.payload));
      if (!mapped)
        return mapped.takeError();
      participants.push_back(
          ::fabric::MemoryConsistencyParticipant::provider(*mapped));
    }
  }
  ::fabric::MemoryConsistencyProgress progress = contract.progress();
  if (auto *bounded = std::get_if<::fabric::BoundedCompletion>(&progress)) {
    auto clock = remapper.remap(bounded->progressClock);
    if (!clock)
      return clock.takeError();
    bounded->progressClock = *clock;
  }
  return ::fabric::MemoryConsistencyContract::create(
      {std::move(participants), contract.releaseVisibilityPoint(),
       std::move(progress), contract.resourceContract()});
}

llvm::Error remapHardwareDomain(::fabric::SystemHardwareDomainOp domain,
                                const SystemReferenceRemapper &remapper) {
  auto record = decodeHardwareDomainContractRecord(
      unsignedBytes(domain.getContractAttr()));
  if (!record)
    return record.takeError();
  std::vector<FabricInventoryOwnerRef> members;
  for (const FabricInventoryOwnerRef &member : record->members()) {
    auto mapped = remapper.remap(member);
    if (!mapped)
      return mapped.takeError();
    members.push_back(*mapped);
  }
  HardwareDomainContract contract = record->contract();
  if (auto *reset = std::get_if<ResetDomainContractRecord>(&contract)) {
    std::optional<ClockDomainRef> clock;
    if (reset->synchronousTo()) {
      auto mapped = remapper.remap(*reset->synchronousTo());
      if (!mapped)
        return mapped.takeError();
      clock = *mapped;
    }
    auto mapped = ResetDomainContractRecord::create(
        reset->polarity(), reset->assertion(), reset->deassertion(),
        reset->initialState(), std::move(clock), reset->releaseLatencyCycles());
    if (!mapped)
      return mapped.takeError();
    contract = std::move(*mapped);
  } else if (const auto *consistency =
                 std::get_if<::fabric::MemoryConsistencyContract>(&contract)) {
    auto mapped = remapConsistencyContract(*consistency, remapper);
    if (!mapped)
      return mapped.takeError();
    contract = std::move(*mapped);
  }
  auto mapped = HardwareDomainContractRecord::create(std::move(members),
                                                     std::move(contract));
  if (!mapped)
    return mapped.takeError();
  auto bytes = encodeHardwareDomainContractRecord(*mapped);
  if (!bytes)
    return bytes.takeError();
  domain.setContractAttr(denseBytes(domain.getContext(), *bytes));
  return llvm::Error::success();
}

llvm::Error remapTransportResource(::fabric::SystemTransportResourceOp resource,
                                   const SystemReferenceRemapper &remapper) {
  DenseI8ArrayAttr crossing = resource.getClockCrossingAttr();
  if (!crossing)
    return llvm::Error::success();
  auto record = decodeClockCrossingContractRecord(unsignedBytes(crossing));
  if (!record)
    return record.takeError();
  auto pattern = remapper.remap(record->transferPattern());
  if (!pattern)
    return pattern.takeError();
  auto source = remapper.remap(record->sourceClock());
  if (!source)
    return source.takeError();
  auto destination = remapper.remap(record->destinationClock());
  if (!destination)
    return destination.takeError();
  auto mapped = ClockCrossingContractRecord::createAsyncFifo(
      *pattern, *source, *destination, record->depth(),
      record->synchronizerStages());
  if (!mapped)
    return mapped.takeError();
  auto bytes = encodeClockCrossingContractRecord(*mapped);
  if (!bytes)
    return bytes.takeError();
  resource.setClockCrossingAttr(denseBytes(resource.getContext(), *bytes));
  return llvm::Error::success();
}

llvm::Error remapTransferPattern(::fabric::SystemTransferPatternOp pattern,
                                 const SystemReferenceRemapper &remapper) {
  auto record = decodeSystemTransferPatternRecord(
      unsignedBytes(pattern.getContractAttr()));
  if (!record)
    return record.takeError();
  auto patternRef = remapper.remap(record->pattern());
  if (!patternRef)
    return patternRef.takeError();
  auto ingress = remapper.remap(record->ingress());
  if (!ingress)
    return ingress.takeError();
  std::vector<FabricTransportEndpointRef> egresses;
  for (const FabricTransportEndpointRef &egress : record->egresses()) {
    auto mapped = remapper.remap(egress);
    if (!mapped)
      return mapped.takeError();
    egresses.push_back(*mapped);
  }
  auto use = remapper.remap(record->usePattern());
  if (!use)
    return use.takeError();
  auto mapped = SystemTransferPatternRecord::create(*patternRef, *ingress,
                                                    std::move(egresses), *use);
  if (!mapped)
    return mapped.takeError();
  pattern.setContractAttr(denseBytes(
      pattern.getContext(), encodeSystemTransferPatternRecord(*mapped)));
  return llvm::Error::success();
}

llvm::Error remapConnection(::fabric::SystemConnectionOp connection,
                            const SystemReferenceRemapper &remapper) {
  if (connection.getMemoryServiceAttr()) {
    auto source = decodeFabricRef<FabricMemoryEndpointRef>(
        unsignedBytes(connection.getSourceAttr()));
    if (!source)
      return source.takeError();
    auto destination = decodeFabricRef<FabricMemoryEndpointRef>(
        unsignedBytes(connection.getDestinationAttr()));
    if (!destination)
      return destination.takeError();
    auto mappedSource = remapper.remap(*source);
    if (!mappedSource)
      return mappedSource.takeError();
    auto mappedDestination = remapper.remap(*destination);
    if (!mappedDestination)
      return mappedDestination.takeError();
    connection.setSourceAttr(denseBytes(connection.getContext(),
                                        canonicalFabricBytes(*mappedSource)));
    connection.setDestinationAttr(denseBytes(
        connection.getContext(), canonicalFabricBytes(*mappedDestination)));
    return llvm::Error::success();
  }
  auto source = decodeFabricRef<FabricTransportEndpointRef>(
      unsignedBytes(connection.getSourceAttr()));
  if (!source)
    return source.takeError();
  auto destination = decodeFabricRef<FabricTransportEndpointRef>(
      unsignedBytes(connection.getDestinationAttr()));
  if (!destination)
    return destination.takeError();
  auto mappedSource = remapper.remap(*source);
  if (!mappedSource)
    return mappedSource.takeError();
  auto mappedDestination = remapper.remap(*destination);
  if (!mappedDestination)
    return mappedDestination.takeError();
  connection.setSourceAttr(
      denseBytes(connection.getContext(), canonicalFabricBytes(*mappedSource)));
  connection.setDestinationAttr(denseBytes(
      connection.getContext(), canonicalFabricBytes(*mappedDestination)));
  return llvm::Error::success();
}

llvm::Error
remapSpatialAttachment(::fabric::SystemSpatialAttachmentOp attachment,
                       const SystemReferenceRemapper &remapper) {
  auto module = decodeFabricImportedModuleBoundaryEndpointRef(
      unsignedBytes(attachment.getModuleEndpointAttr()));
  if (!module)
    return module.takeError();
  auto mappedModule = remapper.remap(*module);
  if (!mappedModule)
    return mappedModule.takeError();
  auto spatial = decodeFabricSpatialAttachmentEndpointRef(
      unsignedBytes(attachment.getSpatialEndpointAttr()));
  if (!spatial)
    return spatial.takeError();
  llvm::Expected<FabricSpatialAttachmentEndpointRef> mappedSpatial = [&]() {
    if (const FabricTransportEndpointRef *transport = spatial->transport()) {
      auto mapped = remapper.remap(*transport);
      if (!mapped)
        return llvm::Expected<FabricSpatialAttachmentEndpointRef>(
            mapped.takeError());
      return FabricSpatialAttachmentEndpointRef::create(*mapped);
    }
    auto mapped = remapper.remap(*spatial->memory());
    if (!mapped)
      return llvm::Expected<FabricSpatialAttachmentEndpointRef>(
          mapped.takeError());
    return FabricSpatialAttachmentEndpointRef::create(*mapped);
  }();
  if (!mappedSpatial)
    return mappedSpatial.takeError();
  attachment.setModuleEndpointAttr(
      denseBytes(attachment.getContext(),
                 encodeFabricImportedModuleBoundaryEndpointRef(*mappedModule)));
  attachment.setSpatialEndpointAttr(
      denseBytes(attachment.getContext(),
                 encodeFabricSpatialAttachmentEndpointRef(*mappedSpatial)));
  if (DenseI8ArrayAttr serviceAttribute = attachment.getServiceEndpointAttr()) {
    auto service = decodeFabricRef<SystemServiceEndpointRef>(
        unsignedBytes(serviceAttribute));
    if (!service)
      return service.takeError();
    auto mappedService = remapper.remap(*service);
    if (!mappedService)
      return mappedService.takeError();
    attachment.setServiceEndpointAttr(denseBytes(
        attachment.getContext(), canonicalFabricBytes(*mappedService)));
  }
  return llvm::Error::success();
}

llvm::Error remapServiceLegCarrierAttachment(
    ::fabric::SystemServiceLegCarrierAttachmentOp attachment,
    const SystemReferenceRemapper &remapper) {
  auto record = decodeServiceLegCarrierAttachmentRecord(
      unsignedBytes(attachment.getRecordAttr()));
  if (!record)
    return record.takeError();
  auto endpoint = remapper.remap(record->endpoint());
  if (!endpoint)
    return endpoint.takeError();
  std::vector<FabricTransportEndpointRef> carriers;
  carriers.reserve(record->carriers().size());
  for (const FabricTransportEndpointRef &carrier : record->carriers()) {
    auto mapped = remapper.remap(carrier);
    if (!mapped)
      return mapped.takeError();
    carriers.push_back(*mapped);
  }
  auto mapped = ServiceLegCarrierAttachmentRecord::create(
      *endpoint, record->kind(), record->legOrdinal(), std::move(carriers));
  if (!mapped)
    return mapped.takeError();
  auto bytes = encodeServiceLegCarrierAttachmentRecord(*mapped);
  if (!bytes)
    return bytes.takeError();
  attachment.setRecordAttr(denseBytes(attachment.getContext(), *bytes));
  return llvm::Error::success();
}

llvm::Error remapOperation(Operation *operation,
                           const SystemReferenceRemapper &remapper) {
  if (auto core = dyn_cast<::fabric::SystemAccCoreOp>(operation))
    return remapAccCore(core, remapper);
  if (auto service = dyn_cast<::fabric::SystemMemoryServiceOp>(operation))
    return remapMemoryService(service, remapper);
  if (auto endpoint = dyn_cast<::fabric::SystemServiceEndpointOp>(operation))
    return remapServiceEndpoint(endpoint, remapper);
  if (auto transform = dyn_cast<::fabric::SystemServiceTransformOp>(operation))
    return remapServiceTransform(transform, remapper);
  if (auto domain = dyn_cast<::fabric::SystemHardwareDomainOp>(operation))
    return remapHardwareDomain(domain, remapper);
  if (auto resource = dyn_cast<::fabric::SystemTransportResourceOp>(operation))
    return remapTransportResource(resource, remapper);
  if (auto pattern = dyn_cast<::fabric::SystemTransferPatternOp>(operation))
    return remapTransferPattern(pattern, remapper);
  if (auto connection = dyn_cast<::fabric::SystemConnectionOp>(operation))
    return remapConnection(connection, remapper);
  if (auto attachment =
          dyn_cast<::fabric::SystemSpatialAttachmentOp>(operation))
    return remapSpatialAttachment(attachment, remapper);
  if (auto attachment =
          dyn_cast<::fabric::SystemServiceLegCarrierAttachmentOp>(operation))
    return remapServiceLegCarrierAttachment(attachment, remapper);
  if (isa<::fabric::SystemHostCoreOp, ::fabric::SystemExternalBoundaryOp>(
          operation))
    return llvm::Error::success();
  return invalid("System materialization encountered an unknown child op");
}

} // namespace

llvm::Error materializeFabricSystemCanonicalForm(
    ::fabric::SystemOp root, const FabricSystemCanonicalLabeling &labeling) {
  auto remapper = SystemReferenceRemapper::create(root, labeling);
  if (!remapper)
    return remapper.takeError();
  for (Operation &operation : root.getBody().front())
    if (llvm::Error error = remapOperation(&operation, *remapper))
      return error;

  for (const FabricSystemEntityCarrier &carrier : labeling.carriers)
    carrier.op->setAttr("entity_id", ::fabric::EntityIdAttr::get(
                                         carrier.op->getContext(), carrier.id));

  Block &block = root.getBody().front();
  for (Operation *operation : labeling.canonicalOperationOrder)
    operation->moveBefore(&block, block.end());
  return llvm::Error::success();
}

} // namespace loom::fabric::detail

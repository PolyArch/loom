#include "Mapping/Artifact/SystemServiceBindingProjection.h"

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>
#include <variant>

namespace loom::mapping {
namespace {

using ::dataflow::semantics::CanonicalMemoryAccessView;
using ::dataflow::semantics::ServiceKind;
using ::loom::fabric::AddressedMemoryCapabilityDomain;
using ::loom::fabric::CanonicalServiceCapabilityRecord;
using ::loom::fabric::CanonicalServiceEndpointPlane;
using ::loom::fabric::CanonicalServiceEndpointRole;
using ::loom::fabric::FenceCapabilityDomain;
using ::loom::fabric::MessageTransferCapabilityDomain;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_service_binding_projection_invalid: " +
                                     message);
}

template <typename Ref> void canonicalizeFabricRefs(std::vector<Ref> &values) {
  llvm::sort(values, [](const Ref &left, const Ref &right) {
    return ::loom::fabric::canonicalFabricBytes(left) <
           ::loom::fabric::canonicalFabricBytes(right);
  });
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

bool ownsMessageEndpoint(
    const ::loom::fabric::SystemServiceEndpointOwnerRef &candidate,
    const SystemMessageExecutionOwner &owner) {
  return std::visit(
      [&](const auto expected) {
        const auto *actual = std::get_if<std::decay_t<decltype(expected)>>(
            &candidate.owner().payload);
        return actual && *actual == expected;
      },
      owner);
}

llvm::Expected<const CanonicalServiceCapabilityRecord *> messageCapability(
    const ::loom::fabric::CanonicalServiceCapabilitySet &capabilities) {
  const CanonicalServiceCapabilityRecord *result = nullptr;
  for (const auto &capability : capabilities.capabilities()) {
    if (capability.kind() != ServiceKind::MessageTransfer)
      continue;
    if (result)
      return invalid("one service endpoint repeats MessageTransfer");
    result = &capability;
  }
  return result;
}

bool roleOwnsMessageTerminal(
    CanonicalServiceEndpointRole role,
    ::dataflow::semantics::ServiceLegDirection direction, bool source) {
  const bool terminalIsInitiator =
      source == (direction ==
                 ::dataflow::semantics::ServiceLegDirection::InitiatorToServer);
  return terminalIsInitiator ==
         (role == CanonicalServiceEndpointRole::Initiate);
}

llvm::Expected<bool>
messagePayloadCompatible(const CanonicalServiceCapabilityRecord &capability,
                         mlir::Type payload,
                         const ::loom::PointerLayout *pointerLayout) {
  const auto *domain =
      std::get_if<MessageTransferCapabilityDomain>(&capability.domain());
  if (!domain)
    return false;
  return domain->admits(payload, pointerLayout);
}

struct ResolvedOperationMember final {
  ServiceKind kind;
  ::dataflow::ContextualActorRef contextualActor;
  ::dataflow::CanonicalActorSchemaProjection actor;
  std::optional<CanonicalMemoryAccessView> access;
};

llvm::Expected<ResolvedOperationMember>
resolveOperationMember(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                       const ::dataflow::ServiceMemberRef &member) {
  std::optional<::dataflow::ContextualActorRef> contextual;
  const bool addressed =
      std::holds_alternative<::dataflow::AddressedMemoryActorMemberRef>(member);
  if (const auto *memory =
          std::get_if<::dataflow::AddressedMemoryActorMemberRef>(&member))
    contextual = memory->actor;
  else if (const auto *fence =
               std::get_if<::dataflow::FenceActorMemberRef>(&member))
    contextual = fence->actor;
  else
    return invalid("operation projection received a non-operation member");
  auto actorView = dataflow.resolve(contextual->actor);
  if (!actorView)
    return actorView.takeError();
  if (llvm::Error error = dataflow.validate(*contextual))
    return std::move(error);
  auto issue = deriveSpatialMemoryIssueEvent(dataflow, contextual->actor);
  if (!issue)
    return issue.takeError();
  auto actor =
      ::dataflow::projectRegisteredActorSchemaProjection(actorView->op);
  if (!actor)
    return actor.takeError();
  auto kind = ::dataflow::semantics::getMemoryServiceKind(actor->schema);
  if (!kind)
    return kind.takeError();
  std::optional<CanonicalMemoryAccessView> access;
  if (addressed) {
    auto resolved =
        ::dataflow::semantics::getCanonicalMemoryAccessView(actorView->op);
    if (!resolved)
      return resolved.takeError();
    access.emplace(std::move(*resolved));
  }
  return ResolvedOperationMember{*kind, *contextual, std::move(*actor),
                                 std::move(access)};
}

llvm::Expected<bool>
capabilityMatches(const CanonicalServiceCapabilityRecord &capability,
                  const ResolvedOperationMember &member) {
  if (capability.kind() != member.kind)
    return false;
  if (const auto *addressed =
          std::get_if<AddressedMemoryCapabilityDomain>(&capability.domain()))
    return member.access &&
           addressed->actorContracts().contains(member.actor) &&
           addressed->accesses().contains(*member.access);
  const auto *fence = std::get_if<FenceCapabilityDomain>(&capability.domain());
  return fence && !member.access &&
         fence->actorContracts().contains(member.actor);
}

llvm::Expected<const CanonicalServiceCapabilityRecord *> matchingCapability(
    const ::loom::fabric::CanonicalServiceCapabilitySet &capabilities,
    const ResolvedOperationMember &member) {
  const CanonicalServiceCapabilityRecord *result = nullptr;
  for (const auto &capability : capabilities.capabilities()) {
    if (capability.kind() != member.kind)
      continue;
    if (result)
      return invalid("one service endpoint repeats an operation service kind");
    result = &capability;
  }
  if (!result)
    return nullptr;
  auto matches = capabilityMatches(*result, member);
  if (!matches)
    return matches.takeError();
  return *matches ? result : nullptr;
}

llvm::Expected<const ::loom::fabric::ClockDomainContractRecord *>
clockContract(const ::loom::fabric::FabricSystemRootView &fabric,
              ::loom::fabric::ClockDomainRef clock) {
  const auto *domain = fabric.hardwareDomainContract(clock.underlying());
  const auto *contract =
      domain ? std::get_if<::loom::fabric::ClockDomainContractRecord>(
                   &domain->contract())
             : nullptr;
  if (!contract)
    return invalid("service progress clock does not resolve");
  return contract;
}

llvm::Expected<::loom::fabric::ClockDomainRef>
ownerClock(const ::loom::fabric::FabricSystemRootView &fabric,
           const ::loom::fabric::FabricInventoryOwnerRef &owner) {
  std::optional<::loom::fabric::ClockDomainRef> result;
  for (const auto domain : fabric.hardwareDomains()) {
    const auto *record = fabric.hardwareDomainContract(domain);
    if (!record ||
        !std::holds_alternative<::loom::fabric::ClockDomainContractRecord>(
            record->contract()) ||
        !llvm::is_contained(record->members(), owner))
      continue;
    if (result)
      return invalid("service issuer belongs to multiple clock domains");
    result = ::loom::fabric::ClockDomainRef(domain);
  }
  if (!result)
    return invalid("service issuer has no clock domain");
  return *result;
}

llvm::Expected<std::uint64_t>
convertCompletionTicks(std::uint64_t ticks, std::uint64_t progressPeriodFs,
                       std::uint64_t issuerPeriodFs) {
  auto scaled =
      llvm::checkedMulUnsigned<std::uint64_t>(ticks, progressPeriodFs);
  if (!scaled || *scaled > std::numeric_limits<std::uint64_t>::max() -
                               (issuerPeriodFs - 1))
    return invalid("service completion conversion exceeds u64");
  return (*scaled + issuerPeriodFs - 1) / issuerPeriodFs;
}

llvm::Expected<bool>
memoryCapabilitySupports(const ::fabric::MemoryServiceContractRecord &service,
                         const ResolvedOperationMember &member,
                         ::loom::fabric::FabricOrdinal regionOrdinal) {
  if (!member.access)
    return false;
  auto matches = service.matchingCapabilities(member.actor, member.access);
  if (!matches)
    return matches.takeError();
  for (std::uint64_t ordinal : *matches) {
    if (ordinal >= service.capabilities().size())
      return invalid("memory service returned an invalid capability ordinal");
    if (llvm::is_contained(
            service.capabilities()[ordinal].serviceRegionOrdinals,
            regionOrdinal))
      return true;
  }
  return false;
}

void canonicalizePairs(std::vector<SystemBoundMemoryEndpointPairView> &pairs) {
  llvm::sort(pairs, [](const auto &left, const auto &right) {
    const auto leftSystem =
        ::loom::fabric::canonicalFabricBytes(left.systemEndpoint);
    const auto rightSystem =
        ::loom::fabric::canonicalFabricBytes(right.systemEndpoint);
    if (leftSystem != rightSystem)
      return leftSystem < rightSystem;
    return ::loom::fabric::canonicalFabricBytes(left.occurrenceEndpoint) <
           ::loom::fabric::canonicalFabricBytes(right.occurrenceEndpoint);
  });
  pairs.erase(
      std::unique(pairs.begin(), pairs.end(),
                  [](const auto &left, const auto &right) {
                    return left.systemEndpoint == right.systemEndpoint &&
                           left.occurrenceEndpoint == right.occurrenceEndpoint;
                  }),
      pairs.end());
}

llvm::Error appendManagerPairs(
    const ::loom::fabric::FabricSystemRootView &fabric,
    std::uint64_t moduleDependencyOrdinal,
    ::loom::fabric::FabricMemoryEndpointRef moduleEndpoint,
    std::optional<::loom::fabric::AccCoreOccurrenceRef> exactAccCore,
    std::vector<SystemBoundMemoryEndpointPairView> &pairs) {
  if (moduleDependencyOrdinal >= fabric.artifact().importedModules().size())
    return invalid("SpatialMapping Module dependency ordinal is invalid");
  const auto &module =
      fabric.artifact().importedModules()[moduleDependencyOrdinal];
  for (const auto &moduleAttachment :
       module.moduleBoundaryMemoryAttachments()) {
    if (moduleAttachment.endpoint != moduleEndpoint)
      continue;
    for (const auto &attachment : fabric.spatialAttachments()) {
      if (attachment.moduleEndpoint.dependencyOrdinal !=
              moduleDependencyOrdinal ||
          attachment.moduleEndpoint.target != moduleAttachment.boundary)
        continue;
      const auto *occurrence = attachment.spatialEndpoint.memory();
      if (!occurrence || !attachment.serviceEndpoint)
        return invalid("memory Module boundary has an incomplete System "
                       "spatial attachment");
      const auto *spatialCore =
          std::get_if<::loom::fabric::SpatialCoreOccurrenceRef>(
              &occurrence->owner.payload);
      if (!spatialCore)
        return invalid("memory spatial attachment is not occurrence-owned");
      if (exactAccCore && spatialCore->core != *exactAccCore)
        continue;
      pairs.push_back({*attachment.serviceEndpoint, *occurrence});
    }
  }
  return llvm::Error::success();
}

bool sameInterval(const SpatialMemoryIntervalView &left,
                  const SpatialMemoryIntervalView &right) {
  if (left.index() != right.index())
    return false;
  if (std::holds_alternative<SpatialMemoryWholeIntervalView>(left))
    return true;
  const auto &leftRange = std::get<SpatialMemoryByteRangeView>(left);
  const auto &rightRange = std::get<SpatialMemoryByteRangeView>(right);
  return leftRange.offsetBytes == rightRange.offsetBytes &&
         leftRange.sizeBytes == rightRange.sizeBytes;
}

} // namespace

llvm::Expected<SystemSpatialMemoryBindingProjection>
projectSystemSpatialMemoryBinding(
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SpatialMappingView &mapping, std::uint64_t moduleDependencyOrdinal,
    const ServicePlanSelectionAnchor &anchor,
    std::optional<::loom::fabric::AccCoreOccurrenceRef> exactAccCore) {
  if (moduleDependencyOrdinal >= fabric.artifact().importedModules().size() ||
      fabric.artifact().importedModules()[moduleDependencyOrdinal].identity() !=
          mapping.fabricIdentity())
    return invalid("SpatialMapping does not match its System Module import");
  if (exactAccCore) {
    auto target = fabric.spatialCoreTarget(*exactAccCore);
    if (!target || target->dependencyOrdinal != moduleDependencyOrdinal)
      return invalid("execution AccCore does not target the SpatialMapping "
                     "Module occurrence");
  }

  SystemSpatialMemoryBindingProjection result;
  const auto append =
      [&](::loom::fabric::FabricMemoryEndpointRef endpoint) -> llvm::Error {
    return appendManagerPairs(fabric, moduleDependencyOrdinal, endpoint,
                              exactAccCore, result.endpointPairs);
  };

  if (const auto *member =
          std::get_if<ServiceMemberPlanSelectionAnchor>(&anchor)) {
    if (const auto *addressed =
            std::get_if<::dataflow::AddressedMemoryActorMemberRef>(
                &member->member)) {
      for (const auto &engine : mapping.memoryEngineBindings())
        for (const auto &operation : engine.operations) {
          const auto *candidate =
              std::get_if<SpatialAddressedMemoryOperationView>(&operation);
          if (!candidate || candidate->actor != addressed->actor.actor)
            continue;
          for (const auto &use : candidate->uses) {
            if (use.launch != addressed->actor.launch)
              continue;
            const auto *manager =
                std::get_if<::loom::fabric::ManagerEndpointRef>(&use.dispatch);
            if (!manager)
              continue;
            const auto binding =
                llvm::find_if(mapping.memoryBindings(), [&](const auto &entry) {
                  return entry.entityId == use.binding;
                });
            if (binding == mapping.memoryBindings().end())
              return invalid("addressed service member names an absent "
                             "Spatial memory binding");
            if (!std::holds_alternative<SpatialMemoryBoundaryProxyView>(
                    binding->target))
              return invalid("System addressed service does not use a "
                             "boundary proxy");
            if (result.interval &&
                !sameInterval(*result.interval, binding->interval))
              return invalid("one addressed service member selects multiple "
                             "logical intervals");
            result.interval = binding->interval;
            if (llvm::Error error = append(manager->underlying()))
              return std::move(error);
          }
        }
    } else if (const auto *fence = std::get_if<::dataflow::FenceActorMemberRef>(
                   &member->member)) {
      for (const auto &engine : mapping.memoryEngineBindings())
        for (const auto &operation : engine.operations) {
          const auto *candidate =
              std::get_if<SpatialFenceMemoryOperationView>(&operation);
          if (!candidate || candidate->actor != fence->actor.actor)
            continue;
          for (const auto &use : candidate->uses) {
            if (use.launch != fence->actor.launch)
              continue;
            const auto *manager =
                std::get_if<::loom::fabric::ManagerEndpointRef>(
                    &use.consistency);
            if (manager)
              if (llvm::Error error = append(manager->underlying()))
                return std::move(error);
          }
        }
    } else {
      return invalid("memory binding projection received a message member");
    }
  } else {
    const auto exposure =
        std::get<MemoryExposurePlanSelectionAnchor>(anchor).exposure;
    for (const auto &binding : mapping.memoryBindings())
      for (const auto &candidate : binding.exposures) {
        if (candidate.exposure != exposure)
          continue;
        const auto *manager = std::get_if<::loom::fabric::ManagerEndpointRef>(
            &candidate.dispatch);
        if (!manager)
          continue;
        if (!std::holds_alternative<SpatialMemoryBoundaryProxyView>(
                binding.target))
          return invalid("System memory exposure does not use a boundary "
                         "proxy");
        if (result.interval &&
            (!sameInterval(*result.interval, binding.interval) ||
             *result.exposureTerminal != candidate.terminal))
          return invalid("one memory exposure selects multiple boundary "
                         "providers");
        result.interval = binding.interval;
        result.exposureTerminal = candidate.terminal;
        if (llvm::Error error = append(manager->underlying()))
          return std::move(error);
      }
  }
  canonicalizePairs(result.endpointPairs);
  return result;
}

llvm::Expected<ServiceKind> resolveSystemOperationServiceKind(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::dataflow::ServiceMemberRef &member) {
  auto resolved = resolveOperationMember(dataflow, member);
  if (!resolved)
    return resolved.takeError();
  return resolved->kind;
}

llvm::Expected<std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>
projectSystemOperationTargetRegions(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::SystemServiceEndpointRef endpoint,
    const ::dataflow::ServiceMemberRef &member) {
  auto resolved = resolveOperationMember(dataflow, member);
  if (!resolved)
    return resolved.takeError();
  if (!resolved->access)
    return invalid("region projection received a fence member");
  const auto *capabilities = fabric.serviceEndpointCapabilities(endpoint);
  if (!capabilities ||
      capabilities->plane() != CanonicalServiceEndpointPlane::Memory)
    return invalid("bound System endpoint has no memory capability set");
  auto endpointCapability = matchingCapability(*capabilities, *resolved);
  if (!endpointCapability)
    return endpointCapability.takeError();
  std::vector<::loom::fabric::FabricMemoryServiceRegionRef> result;
  if (!*endpointCapability)
    return result;
  auto plans =
      ::loom::fabric::projectFabricMemoryServiceTargetPlans(fabric, endpoint);
  if (!plans)
    return plans.takeError();
  for (const auto &plan : *plans)
    for (const auto &branch : plan.branches) {
      const auto *systemService =
          std::get_if<::loom::fabric::SystemMemoryServiceRef>(
              &branch.region.service.payload);
      if (!systemService)
        return invalid("System target closure names a non-System service");
      const auto *service = fabric.memoryService(*systemService);
      if (!service || branch.region.ordinal >= service->regions().size())
        return invalid("System target closure names an invalid service region");
      auto supports =
          memoryCapabilitySupports(*service, *resolved, branch.region.ordinal);
      if (!supports)
        return supports.takeError();
      if (*supports)
        result.push_back(branch.region);
    }
  canonicalizeFabricRefs(result);
  return result;
}

llvm::Expected<std::vector<::loom::fabric::MemoryConsistencyDomainRef>>
projectSystemFenceTargetDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::SystemServiceEndpointRef endpoint,
    const ::dataflow::ServiceMemberRef &member) {
  auto resolved = resolveOperationMember(dataflow, member);
  if (!resolved)
    return resolved.takeError();
  if (resolved->access)
    return invalid("consistency projection received an addressed member");
  const auto *capabilities = fabric.serviceEndpointCapabilities(endpoint);
  if (!capabilities ||
      capabilities->plane() != CanonicalServiceEndpointPlane::Memory)
    return invalid("bound System endpoint has no memory capability set");
  auto endpointCapability = matchingCapability(*capabilities, *resolved);
  if (!endpointCapability)
    return endpointCapability.takeError();
  if (!*endpointCapability)
    return std::vector<::loom::fabric::MemoryConsistencyDomainRef>{};
  const auto *fence =
      std::get_if<FenceCapabilityDomain>(&(*endpointCapability)->domain());
  if (!fence)
    return invalid("matching fence capability has a non-fence domain");
  return std::vector<::loom::fabric::MemoryConsistencyDomainRef>{
      fence->consistencyDomain()};
}

llvm::Expected<SystemOperationCompletionProjection>
projectSystemOperationCompletion(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::SystemServiceEndpointRef endpoint,
    ::loom::fabric::AccCoreOccurrenceRef accCore,
    const ::dataflow::ServiceMemberRef &member) {
  auto resolved = resolveOperationMember(dataflow, member);
  if (!resolved)
    return resolved.takeError();
  const auto *capabilities = fabric.serviceEndpointCapabilities(endpoint);
  if (!capabilities ||
      capabilities->plane() != CanonicalServiceEndpointPlane::Memory)
    return invalid("bound System endpoint has no memory capability set");
  auto capability = matchingCapability(*capabilities, *resolved);
  if (!capability)
    return capability.takeError();
  if (!*capability)
    return SystemOperationCompletionProjection{};
  const auto *bounded = std::get_if<::fabric::BoundedCompletion>(
      &(*capability)->rate().progress());
  if (!bounded)
    return SystemOperationCompletionProjection{true, std::nullopt};

  auto progressClock = clockContract(fabric, bounded->progressClock);
  if (!progressClock)
    return progressClock.takeError();
  auto issuerClock = ownerClock(
      fabric, ::loom::fabric::FabricInventoryOwnerRef::of(
                  ::loom::fabric::SpatialCoreOccurrenceRef{accCore}));
  if (!issuerClock)
    return issuerClock.takeError();
  auto issuerClockContract = clockContract(fabric, *issuerClock);
  if (!issuerClockContract)
    return issuerClockContract.takeError();
  auto cycles = convertCompletionTicks(bounded->maxIssueToRetireTicks,
                                       (*progressClock)->periodFs(),
                                       (*issuerClockContract)->periodFs());
  if (!cycles)
    return cycles.takeError();
  if (*cycles == 0)
    return invalid("bounded service completion converted to zero cycles");
  return SystemOperationCompletionProjection{true, *cycles};
}

llvm::Expected<std::vector<::loom::fabric::FabricMemoryServiceTargetPlan>>
projectSystemMemoryTargetPlans(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    ::loom::fabric::SystemServiceEndpointRef endpoint,
    ::dataflow::LogicalMemoryRootOrViewRef logicalMemory,
    const SpatialMemoryIntervalView &interval) {
  std::optional<::loom::fabric::FabricMemoryServiceSourceInterval> source;
  if (const auto *range = std::get_if<SpatialMemoryByteRangeView>(&interval)) {
    source = ::loom::fabric::FabricMemoryServiceSourceInterval{
        range->offsetBytes, range->sizeBytes};
  } else {
    auto extent = dataflow.staticMemoryByteExtent(logicalMemory);
    if (!extent)
      return extent.takeError();
    if (!*extent)
      return ::loom::fabric::projectFabricMemoryServiceTargetPlans(fabric,
                                                                   endpoint);
    source = ::loom::fabric::FabricMemoryServiceSourceInterval{0, **extent};
  }
  return ::loom::fabric::projectFabricMemoryServiceTargetPlans(fabric, endpoint,
                                                               *source);
}

llvm::Expected<std::vector<SystemMemoryUsePatternDomainView>>
projectSystemMemoryUsePatternDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ::dataflow::ServiceMemberRef &member,
    llvm::ArrayRef<::loom::fabric::FabricMemoryServiceTargetPlan> plans) {
  auto resolved = resolveOperationMember(dataflow, member);
  if (!resolved)
    return resolved.takeError();
  std::vector<SystemMemoryUsePatternDomainView> result;
  if (!resolved->access)
    return result;
  for (const auto &plan : plans)
    for (const auto &branch : plan.branches) {
      if (llvm::any_of(result, [&](const auto &entry) {
            return entry.region == branch.region;
          }))
        continue;
      const auto *systemService =
          std::get_if<::loom::fabric::SystemMemoryServiceRef>(
              &branch.region.service.payload);
      if (!systemService)
        return invalid("System target plan names a non-System service");
      const auto *service = fabric.memoryService(*systemService);
      if (!service || branch.region.ordinal >= service->regions().size())
        return invalid("System target plan names an invalid service region");
      auto matching =
          service->matchingCapabilities(resolved->actor, resolved->access);
      if (!matching)
        return matching.takeError();
      std::vector<::loom::fabric::FabricUsePatternRef> patterns;
      for (std::uint64_t capabilityOrdinal : *matching) {
        if (capabilityOrdinal >= service->capabilities().size())
          return invalid("memory service returned an invalid capability");
        const auto &capability = service->capabilities()[capabilityOrdinal];
        if (!llvm::is_contained(capability.serviceRegionOrdinals,
                                branch.region.ordinal))
          continue;
        const auto owner = ::loom::fabric::FabricUsePatternOwnerRef(
            ::loom::fabric::FabricInventoryOwnerRef::of(branch.region.service));
        for (const ::fabric::UsePatternKey pattern :
             capability.admissibleUsePatterns)
          patterns.push_back({owner, static_cast<::loom::fabric::FabricOrdinal>(
                                         pattern.ordinal())});
      }
      canonicalizeFabricRefs(patterns);
      if (patterns.empty())
        return invalid("selected memory service region has no admissible use "
                       "pattern");
      result.push_back({branch.region, std::move(patterns)});
    }
  llvm::sort(result, [](const auto &left, const auto &right) {
    return ::loom::fabric::canonicalFabricBytes(left.region) <
           ::loom::fabric::canonicalFabricBytes(right.region);
  });
  return result;
}

llvm::Expected<std::vector<SystemMessageTerminalEndpointDomainView>>
projectSystemMessageTerminalEndpointDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const SystemTransferTerminalKey &terminal, mlir::Type payload,
    const SystemMessageExecutionOwner &owner) {
  const bool source =
      std::holds_alternative<SystemTransferSourceTerminalKey>(terminal);
  const CanonicalServiceLegKey &leg =
      source ? std::get<SystemTransferSourceTerminalKey>(terminal).leg
             : std::get<SystemTransferSinkTerminalKey>(terminal).leg;
  const auto *producer =
      std::get_if<TransferObligationFamilyKey>(&leg.obligation);
  if (!producer)
    return invalid("message terminal belongs to a non-transfer obligation");
  auto resolved = dataflow.resolve(*producer);
  if (!resolved)
    return resolved.takeError();
  auto expectedPayload = ::dataflow::encodeCanonicalType(resolved->payloadType);
  auto actualPayload = ::dataflow::encodeCanonicalType(payload);
  if (!expectedPayload)
    return expectedPayload.takeError();
  if (!actualPayload)
    return actualPayload.takeError();
  if (expectedPayload->bytes() != actualPayload->bytes())
    return invalid("message payload disagrees with its producer terminal");
  if (!source) {
    const auto sinkOrdinal =
        std::get<SystemTransferSinkTerminalKey>(terminal).sinkOrdinal;
    std::size_t sinkCount = 0;
    if (llvm::Error error =
            dataflow.pairedSinks(*producer, [&](const auto &) { ++sinkCount; }))
      return std::move(error);
    if (sinkOrdinal >= sinkCount)
      return invalid("message sink ordinal is out of range");
  }

  auto direction = ::dataflow::semantics::getCanonicalServiceLegDirection(
      ServiceKind::MessageTransfer, leg.ordinal);
  if (!direction)
    return direction.takeError();
  std::optional<::loom::PointerLayout> pointerLayout;
  if (auto pointer = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(payload)) {
    auto resolved = dataflow.pointerLayout(pointer.getAddressSpace());
    if (!resolved)
      return resolved.takeError();
    pointerLayout = *resolved;
  }
  std::vector<SystemMessageTerminalEndpointDomainView> result;
  for (const auto endpoint : fabric.artifact().systemServiceEndpoints()) {
    const auto *endpointOwner = fabric.serviceEndpointOwner(endpoint);
    if (!endpointOwner || !ownsMessageEndpoint(*endpointOwner, owner))
      continue;
    const auto *capabilities = fabric.serviceEndpointCapabilities(endpoint);
    if (!capabilities ||
        capabilities->plane() != CanonicalServiceEndpointPlane::Transport)
      continue;
    auto capability = messageCapability(*capabilities);
    if (!capability)
      return capability.takeError();
    if (!*capability ||
        !roleOwnsMessageTerminal((*capability)->role(), *direction, source))
      continue;
    const ::loom::fabric::FabricTransportEndpointRef bound{
        ::loom::fabric::FabricTransportEndpointOwnerRef::of(endpoint), 0};
    const auto expectedDirection =
        source ? ::loom::fabric::FabricPortDirection::Output
               : ::loom::fabric::FabricPortDirection::Input;
    if (fabric.artifact().transportEndpointDirection(bound) !=
        expectedDirection)
      return invalid("message service endpoint has the wrong direction");
    auto compatible = messagePayloadCompatible(
        **capability, payload, pointerLayout ? &*pointerLayout : nullptr);
    if (!compatible)
      return compatible.takeError();
    result.push_back({bound, *compatible});
  }
  llvm::sort(result, [](const auto &left, const auto &right) {
    return ::loom::fabric::canonicalFabricBytes(left.endpoint) <
           ::loom::fabric::canonicalFabricBytes(right.endpoint);
  });
  if (std::adjacent_find(result.begin(), result.end(),
                         [](const auto &left, const auto &right) {
                           return left.endpoint == right.endpoint;
                         }) != result.end())
    return invalid("message terminal projection contains a duplicate endpoint");
  return result;
}

} // namespace loom::mapping

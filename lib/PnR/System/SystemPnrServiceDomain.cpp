#include "SystemPnrSearchDomainInternal.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/Artifact/FabricMemoryServiceClosure.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr::detail {
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
                                 "system_pnr_search_domain_invalid: " +
                                     message);
}

std::string keyString(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

template <typename Ref> void canonicalizeFabricRefs(std::vector<Ref> &values) {
  llvm::sort(values, [](const Ref &left, const Ref &right) {
    return ::loom::fabric::canonicalFabricBytes(left) <
           ::loom::fabric::canonicalFabricBytes(right);
  });
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

struct ResolvedServiceMember final {
  ServiceKind kind;
  std::optional<mlir::Type> messagePayload;
  std::optional<::dataflow::ContextualActorRef> contextualActor;
  std::optional<::dataflow::CanonicalActorSchemaProjection> actor;
  std::optional<CanonicalMemoryAccessView> access;
};

struct BoundMemoryEndpointPair final {
  ::loom::fabric::SystemServiceEndpointRef systemEndpoint;
  ::loom::fabric::FabricMemoryEndpointRef occurrenceEndpoint;
};

llvm::Expected<std::map<std::string, mlir::Type>>
collectMessagePayloads(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                       llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots) {
  std::map<std::string, mlir::Type> result;
  for (const ::dataflow::RootThreadLaunchRef root : roots) {
    if (llvm::Error error = dataflow.forEachProducerTerminal(
            root,
            [&](const ::dataflow::CanonicalProducerTerminalView &view)
                -> llvm::Error {
              auto bytes = ::dataflow::encodeDataflowReference(
                  dataflow.identity(), view.terminal);
              if (!bytes)
                return bytes.takeError();
              auto [position, inserted] =
                  result.emplace(keyString(*bytes), view.payloadType);
              if (!inserted) {
                auto existing =
                    ::dataflow::encodeCanonicalType(position->second);
                if (!existing)
                  return existing.takeError();
                auto incoming =
                    ::dataflow::encodeCanonicalType(view.payloadType);
                if (!incoming)
                  return incoming.takeError();
                if (existing->bytes() != incoming->bytes())
                  return invalid(
                      "one producer terminal has conflicting payload types");
              }
              return llvm::Error::success();
            }))
      return std::move(error);
  }
  return result;
}

llvm::Expected<ResolvedServiceMember> resolveMember(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::SystemServiceObligationProjection &obligation,
    const ::dataflow::ServiceMemberRef &member,
    const std::map<std::string, mlir::Type> &messagePayloads) {
  if (std::holds_alternative<::dataflow::MessageTransferMemberRef>(member)) {
    const auto *producer =
        std::get_if<::dataflow::CanonicalProducerTerminalRef>(&obligation.key);
    if (!producer)
      return invalid("message member belongs to an operation obligation");
    auto bytes =
        ::dataflow::encodeDataflowReference(dataflow.identity(), *producer);
    if (!bytes)
      return bytes.takeError();
    const auto found = messagePayloads.find(keyString(*bytes));
    if (found == messagePayloads.end())
      return invalid("message obligation has no Dataflow-owned payload type");
    return ResolvedServiceMember{ServiceKind::MessageTransfer, found->second,
                                 std::nullopt, std::nullopt, std::nullopt};
  }

  std::optional<::dataflow::ContextualActorRef> contextual;
  if (const auto *addressed =
          std::get_if<::dataflow::AddressedMemoryActorMemberRef>(&member))
    contextual = addressed->actor;
  else if (const auto *fence =
               std::get_if<::dataflow::FenceActorMemberRef>(&member))
    contextual = fence->actor;
  else
    return invalid("unknown canonical service-member variant");
  auto actorView = dataflow.resolve(contextual->actor);
  if (!actorView)
    return actorView.takeError();
  if (llvm::Error error = dataflow.validate(*contextual))
    return std::move(error);
  auto actor =
      ::dataflow::projectRegisteredActorSchemaProjection(actorView->op);
  if (!actor)
    return actor.takeError();
  auto kind = ::dataflow::semantics::getMemoryServiceKind(actor->schema);
  if (!kind)
    return kind.takeError();
  std::optional<CanonicalMemoryAccessView> access;
  if (std::holds_alternative<::dataflow::AddressedMemoryActorMemberRef>(
          member)) {
    auto resolved =
        ::dataflow::semantics::getCanonicalMemoryAccessView(actorView->op);
    if (!resolved)
      return resolved.takeError();
    access.emplace(std::move(*resolved));
  }
  return ResolvedServiceMember{*kind, std::nullopt, *contextual,
                               std::move(*actor), std::move(access)};
}

llvm::Expected<bool>
capabilityMatches(const CanonicalServiceCapabilityRecord &capability,
                  const ResolvedServiceMember &member) {
  if (capability.kind() != member.kind)
    return false;
  if (const auto *message =
          std::get_if<MessageTransferCapabilityDomain>(&capability.domain())) {
    if (!member.messagePayload || !*member.messagePayload)
      return false;
    auto wanted = ::dataflow::encodeCanonicalType(*member.messagePayload);
    if (!wanted)
      return wanted.takeError();
    for (mlir::Type candidate : message->payloadTypes()) {
      auto encoded = ::dataflow::encodeCanonicalType(candidate);
      if (!encoded)
        return encoded.takeError();
      if (encoded->bytes() == wanted->bytes())
        return true;
    }
    return false;
  }
  if (!member.actor)
    return false;
  if (const auto *addressed =
          std::get_if<AddressedMemoryCapabilityDomain>(&capability.domain()))
    return member.access &&
           addressed->actorContracts().contains(*member.actor) &&
           addressed->accesses().contains(*member.access);
  const auto *fence = std::get_if<FenceCapabilityDomain>(&capability.domain());
  return fence && !member.access &&
         fence->actorContracts().contains(*member.actor);
}

llvm::Expected<const CanonicalServiceCapabilityRecord *> capabilityForKind(
    const ::loom::fabric::CanonicalServiceCapabilitySet &capabilities,
    ServiceKind kind) {
  const CanonicalServiceCapabilityRecord *result = nullptr;
  for (const CanonicalServiceCapabilityRecord &capability :
       capabilities.capabilities()) {
    if (capability.kind() != kind)
      continue;
    if (result)
      return invalid("one service endpoint repeats a service kind");
    result = &capability;
  }
  return result;
}

llvm::Expected<const CanonicalServiceCapabilityRecord *> matchingCapability(
    const ::loom::fabric::CanonicalServiceCapabilitySet &capabilities,
    const ResolvedServiceMember &member) {
  auto capability = capabilityForKind(capabilities, member.kind);
  if (!capability)
    return capability.takeError();
  if (!*capability)
    return nullptr;
  auto matches = capabilityMatches(**capability, member);
  if (!matches)
    return matches.takeError();
  return *matches ? *capability : nullptr;
}

bool roleOwnsTerminal(CanonicalServiceEndpointRole role,
                      ::dataflow::semantics::ServiceLegDirection direction,
                      bool source) {
  const bool terminalIsInitiator =
      source == (direction ==
                 ::dataflow::semantics::ServiceLegDirection::InitiatorToServer);
  return terminalIsInitiator ==
         (role == CanonicalServiceEndpointRole::Initiate);
}

llvm::Expected<bool>
memoryCapabilitySupports(const ::fabric::MemoryServiceContractRecord &service,
                         const ResolvedServiceMember &member,
                         ::loom::fabric::FabricOrdinal regionOrdinal) {
  if (!member.actor || !member.access)
    return false;
  auto matches = service.matchingCapabilities(*member.actor, member.access);
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

llvm::Expected<std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>
compatibleServiceRegions(const ::loom::fabric::FabricSystemRootView &fabric,
                         ::loom::fabric::SystemServiceEndpointRef endpoint,
                         const std::optional<ResolvedServiceMember> &member) {
  std::vector<::loom::fabric::FabricMemoryServiceRegionRef> result;
  auto plans =
      ::loom::fabric::projectFabricMemoryServiceTargetPlans(fabric, endpoint);
  if (!plans)
    return plans.takeError();
  for (const ::loom::fabric::FabricMemoryServiceTargetPlan &plan : *plans) {
    for (const ::loom::fabric::FabricMemoryServiceTargetBranch &branch :
         plan.branches) {
      const auto *systemRef =
          std::get_if<::loom::fabric::SystemMemoryServiceRef>(
              &branch.region.service.payload);
      if (!systemRef)
        return invalid("System target closure names a non-System service");
      const auto *service = fabric.memoryService(*systemRef);
      if (!service || branch.region.ordinal >= service->regions().size())
        return invalid("System target closure names an invalid service region");
      if (member) {
        auto supports =
            memoryCapabilitySupports(*service, *member, branch.region.ordinal);
        if (!supports)
          return supports.takeError();
        if (!*supports)
          continue;
      }
      result.push_back(branch.region);
    }
  }
  canonicalizeFabricRefs(result);
  return result;
}

void canonicalizePairs(std::vector<BoundMemoryEndpointPair> &pairs) {
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

llvm::Error
appendManagerPairs(const ::loom::fabric::FabricSystemRootView &fabric,
                   const SpatialCatalogEntry &entry,
                   ::dataflow::RootedGraphLaunchRef launch,
                   ::loom::fabric::FabricMemoryEndpointRef moduleEndpoint,
                   const SystemFrozenConstraintIndex &constraints,
                   std::vector<BoundMemoryEndpointPair> &pairs) {
  if (entry.moduleDependencyOrdinal >=
      fabric.artifact().importedModules().size())
    return invalid("SpatialMapping Module dependency ordinal is invalid");
  const auto &module =
      fabric.artifact().importedModules()[entry.moduleDependencyOrdinal];
  for (const auto &moduleAttachment :
       module.moduleBoundaryMemoryAttachments()) {
    if (moduleAttachment.endpoint != moduleEndpoint)
      continue;
    for (const auto &attachment : fabric.spatialAttachments()) {
      if (attachment.moduleEndpoint.dependencyOrdinal !=
              entry.moduleDependencyOrdinal ||
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
      if (!systemConstraintAllows(
              constraints,
              ::mapping::SystemConstraintProjection::
                  GraphSelectedSpatialMapping,
              ::loom::mapping::SystemConstraintSubject{launch},
              entry.reference) ||
          !systemConstraintAllows(
              constraints,
              ::mapping::SystemConstraintProjection::GraphTargetSpatialCore,
              ::loom::mapping::SystemConstraintSubject{launch}, *spatialCore) ||
          !systemConstraintAllows(
              constraints,
              ::mapping::SystemConstraintProjection::ThreadTargetAccCore,
              ::loom::mapping::SystemConstraintSubject{launch.rootThreadLaunch},
              spatialCore->core))
        continue;
      pairs.push_back({*attachment.serviceEndpoint, *occurrence});
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<BoundMemoryEndpointPair>>
boundPairsForMember(const ::loom::fabric::FabricSystemRootView &fabric,
                    const ResolvedServiceMember &member,
                    llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog,
                    const SystemFrozenConstraintIndex &constraints) {
  std::vector<BoundMemoryEndpointPair> pairs;
  if (!member.contextualActor)
    return pairs;
  for (const SpatialCatalogEntry &entry : spatialCatalog) {
    for (const auto &engine : entry.mapping.view().memoryEngineBindings()) {
      for (const auto &operation : engine.operations) {
        if (const auto *addressed = std::get_if<
                ::loom::mapping::SpatialAddressedMemoryOperationView>(
                &operation)) {
          if (!member.access ||
              addressed->actor != member.contextualActor->actor)
            continue;
          for (const auto &use : addressed->uses) {
            if (use.launch != member.contextualActor->launch)
              continue;
            const auto *manager =
                std::get_if<::loom::fabric::ManagerEndpointRef>(&use.dispatch);
            if (manager)
              if (llvm::Error error = appendManagerPairs(
                      fabric, entry, member.contextualActor->launch,
                      manager->underlying(), constraints, pairs))
                return std::move(error);
          }
        } else {
          const auto &fence =
              std::get<::loom::mapping::SpatialFenceMemoryOperationView>(
                  operation);
          if (member.access || fence.actor != member.contextualActor->actor)
            continue;
          for (const auto &use : fence.uses) {
            if (use.launch != member.contextualActor->launch)
              continue;
            const auto *manager =
                std::get_if<::loom::fabric::ManagerEndpointRef>(
                    &use.consistency);
            if (manager)
              if (llvm::Error error = appendManagerPairs(
                      fabric, entry, member.contextualActor->launch,
                      manager->underlying(), constraints, pairs))
                return std::move(error);
          }
        }
      }
    }
  }
  canonicalizePairs(pairs);
  return pairs;
}

llvm::Expected<std::vector<BoundMemoryEndpointPair>>
boundPairsForExposure(const ::loom::fabric::FabricSystemRootView &fabric,
                      ::dataflow::MemoryExposureRef exposure,
                      llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog,
                      const SystemFrozenConstraintIndex &constraints) {
  std::vector<BoundMemoryEndpointPair> pairs;
  for (const SpatialCatalogEntry &entry : spatialCatalog)
    for (const auto &binding : entry.mapping.view().memoryBindings())
      for (const auto &entryExposure : binding.exposures) {
        if (entryExposure.exposure != exposure)
          continue;
        const auto *manager = std::get_if<::loom::fabric::ManagerEndpointRef>(
            &entryExposure.dispatch);
        if (manager)
          if (llvm::Error error =
                  appendManagerPairs(fabric, entry, exposure.launch,
                                     manager->underlying(), constraints, pairs))
            return std::move(error);
      }
  canonicalizePairs(pairs);
  return pairs;
}

std::vector<::loom::fabric::SystemServiceEndpointRef>
systemEndpoints(llvm::ArrayRef<BoundMemoryEndpointPair> pairs) {
  std::vector<::loom::fabric::SystemServiceEndpointRef> result;
  result.reserve(pairs.size());
  for (const auto &pair : pairs)
    result.push_back(pair.systemEndpoint);
  canonicalizeFabricRefs(result);
  return result;
}

llvm::Expected<std::vector<::loom::fabric::FabricTransportEndpointRef>>
carriersForMemoryTerminal(const ::loom::fabric::FabricSystemRootView &fabric,
                          ::loom::fabric::FabricMemoryEndpointRef endpoint,
                          ServiceKind kind,
                          ::dataflow::StructuralOrdinal legOrdinal, bool source,
                          bool admitted) {
  std::vector<::loom::fabric::FabricTransportEndpointRef> result;
  if (!admitted)
    return result;
  const auto attachment = llvm::find_if(
      fabric.serviceLegCarrierAttachments(), [&](const auto &candidate) {
        return candidate.endpoint() == endpoint && candidate.kind() == kind &&
               candidate.legOrdinal() == legOrdinal;
      });
  if (attachment == fabric.serviceLegCarrierAttachments().end())
    return result;
  const auto expectedDirection =
      source ? ::loom::fabric::FabricPortDirection::Output
             : ::loom::fabric::FabricPortDirection::Input;
  for (const auto carrier : attachment->carriers()) {
    if (fabric.artifact().transportEndpointDirection(carrier) !=
        expectedDirection)
      return invalid("memory service carrier has the wrong direction");
    result.push_back(carrier);
  }
  return result;
}

llvm::Expected<SystemBoundTerminalEndpoint> selectMemoryTerminalEndpoint(
    const ::loom::fabric::FabricSystemRootView &fabric,
    const BoundMemoryEndpointPair &pair,
    ::dataflow::semantics::ServiceLegDirection direction, bool source) {
  const ::loom::fabric::FabricMemoryEndpointRef systemEndpoint{
      ::loom::fabric::FabricMemoryEndpointOwnerRef::of(pair.systemEndpoint), 0};
  const auto occurrenceRole =
      fabric.artifact().memoryEndpointRole(pair.occurrenceEndpoint);
  const auto systemRole = fabric.artifact().memoryEndpointRole(systemEndpoint);
  if (!occurrenceRole || !systemRole || *occurrenceRole == *systemRole)
    return invalid("System memory attachment does not form a complementary "
                   "endpoint pair");
  const bool terminalIsManager =
      source == (direction ==
                 ::dataflow::semantics::ServiceLegDirection::InitiatorToServer);
  const auto selected =
      (*occurrenceRole == ::loom::fabric::FabricMemoryEndpointRole::Manager) ==
              terminalIsManager
          ? pair.occurrenceEndpoint
          : systemEndpoint;
  return SystemBoundTerminalEndpoint{
      SystemMemoryOrFenceTerminalEndpoint{selected}};
}

bool sameBoundEndpoint(const SystemBoundTerminalEndpoint &left,
                       const SystemBoundTerminalEndpoint &right) {
  if (left.index() != right.index())
    return false;
  if (const auto *message = std::get_if<SystemMessageTerminalEndpoint>(&left))
    return message->endpoint ==
           std::get<SystemMessageTerminalEndpoint>(right).endpoint;
  return std::get<SystemMemoryOrFenceTerminalEndpoint>(left).endpoint ==
         std::get<SystemMemoryOrFenceTerminalEndpoint>(right).endpoint;
}

bool sameTargetSubject(const SystemServiceTargetSubject &left,
                       const SystemServiceTargetSubject &right) {
  if (left.index() != right.index())
    return false;
  if (const auto *member = std::get_if<SystemServiceMemberTargetSubject>(&left))
    return member->member ==
           std::get<SystemServiceMemberTargetSubject>(right).member;
  return std::get<SystemMemoryExposureTargetSubject>(left).exposure ==
         std::get<SystemMemoryExposureTargetSubject>(right).exposure;
}

llvm::Error appendTerminalCompatibility(
    SystemSearchServiceDomain &domain,
    ::loom::mapping::SystemTransferTerminalKey terminal,
    SystemBoundTerminalEndpoint bound,
    std::vector<::loom::fabric::FabricTransportEndpointRef> carriers) {
  const auto existing =
      llvm::find_if(domain.transferTerminalCompatibility, [&](const auto &row) {
        return row.terminal == terminal &&
               sameBoundEndpoint(row.boundEndpoint, bound);
      });
  if (existing != domain.transferTerminalCompatibility.end()) {
    if (existing->compatibleTransportEndpoints != carriers)
      return invalid("one exact bound terminal has inconsistent carriers");
    return llvm::Error::success();
  }
  domain.transferTerminalCompatibility.push_back(
      {std::move(terminal), std::move(bound), std::move(carriers)});
  return llvm::Error::success();
}

llvm::Error appendMemoryRows(
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ::loom::mapping::SystemServiceObligationProjection &obligation,
    const ::dataflow::ServiceMemberRef &memberRef,
    const ResolvedServiceMember &member,
    llvm::ArrayRef<BoundMemoryEndpointPair> pairs,
    SystemSearchServiceDomain &domain) {
  const SystemServiceTargetSubject subject{
      SystemServiceMemberTargetSubject{memberRef}};
  for (const auto endpoint : systemEndpoints(pairs)) {
    const auto *capabilities = fabric.serviceEndpointCapabilities(endpoint);
    if (!capabilities ||
        capabilities->plane() != CanonicalServiceEndpointPlane::Memory)
      return invalid("bound System memory endpoint has no memory capability "
                     "set");
    auto capability = matchingCapability(*capabilities, member);
    if (!capability)
      return capability.takeError();
    if (member.access) {
      std::vector<::loom::fabric::FabricMemoryServiceRegionRef> regions;
      if (*capability) {
        auto compatible = compatibleServiceRegions(fabric, endpoint, member);
        if (!compatible)
          return compatible.takeError();
        regions = std::move(*compatible);
      }
      domain.targetCompatibility.push_back(
          {subject, endpoint, std::move(regions)});
    } else {
      std::vector<::loom::fabric::MemoryConsistencyDomainRef> consistency;
      if (*capability) {
        const auto *fence =
            std::get_if<FenceCapabilityDomain>(&(*capability)->domain());
        if (!fence)
          return invalid("matching fence capability has a non-fence domain");
        consistency.push_back(fence->consistencyDomain());
      }
      domain.targetCompatibility.push_back(
          {subject, endpoint, std::move(consistency)});
    }
  }

  for (const auto &leg : obligation.legs) {
    if (leg.member != memberRef)
      continue;
    auto direction = ::dataflow::semantics::getCanonicalServiceLegDirection(
        member.kind, leg.ordinal);
    if (!direction)
      return direction.takeError();
    for (const BoundMemoryEndpointPair &pair : pairs) {
      const auto *capabilities =
          fabric.serviceEndpointCapabilities(pair.systemEndpoint);
      if (!capabilities)
        return invalid("bound System endpoint has no capability set");
      auto capability = matchingCapability(*capabilities, member);
      if (!capability)
        return capability.takeError();
      for (const bool source : {true, false}) {
        auto bound =
            selectMemoryTerminalEndpoint(fabric, pair, *direction, source);
        if (!bound)
          return bound.takeError();
        const auto selected =
            std::get<SystemMemoryOrFenceTerminalEndpoint>(*bound).endpoint;
        auto carriers = carriersForMemoryTerminal(fabric, selected, member.kind,
                                                  leg.ordinal, source,
                                                  *capability != nullptr);
        if (!carriers)
          return carriers.takeError();
        ::loom::mapping::SystemTransferTerminalKey terminal =
            source
                ? ::loom::mapping::SystemTransferTerminalKey(
                      ::loom::mapping::SystemTransferSourceTerminalKey{leg})
                : ::loom::mapping::SystemTransferTerminalKey(
                      ::loom::mapping::SystemTransferSinkTerminalKey{leg, 0});
        if (llvm::Error error = appendTerminalCompatibility(
                domain, std::move(terminal), std::move(*bound),
                std::move(*carriers)))
          return error;
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error
appendExposureRows(const ::loom::fabric::FabricSystemRootView &fabric,
                   ::dataflow::MemoryExposureRef exposure,
                   llvm::ArrayRef<BoundMemoryEndpointPair> pairs,
                   SystemSearchServiceDomain &domain) {
  const SystemServiceTargetSubject subject{
      SystemMemoryExposureTargetSubject{exposure}};
  for (const auto endpoint : systemEndpoints(pairs)) {
    const auto *capabilities = fabric.serviceEndpointCapabilities(endpoint);
    if (!capabilities ||
        capabilities->plane() != CanonicalServiceEndpointPlane::Memory)
      return invalid("bound System memory endpoint has no memory capability "
                     "set");
    auto regions = compatibleServiceRegions(
        fabric, endpoint, std::optional<ResolvedServiceMember>());
    if (!regions)
      return regions.takeError();
    domain.targetCompatibility.push_back(
        {subject, endpoint, std::move(*regions)});
  }
  return llvm::Error::success();
}

llvm::Error appendMessageRows(
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ::loom::mapping::SystemServiceObligationProjection &obligation,
    const ResolvedServiceMember &member,
    llvm::ArrayRef<SystemSearchBindingDomain> bindings,
    SystemSearchServiceDomain &domain) {
  const auto compatibleCores = [&](::dataflow::RootThreadLaunchRef root)
      -> llvm::Expected<std::vector<::loom::fabric::AccCoreOccurrenceRef>> {
    std::vector<::loom::fabric::AccCoreOccurrenceRef> result;
    for (const SystemSearchBindingDomain &binding : bindings) {
      const auto *boundRoot =
          std::get_if<::dataflow::RootThreadLaunchRef>(&binding.key);
      if (!boundRoot || *boundRoot != root)
        continue;
      for (const SystemSearchAtom &atom : binding.atoms) {
        const auto *thread =
            std::get_if<SystemThreadBindingDomain>(&atom.domain);
        if (!thread)
          return invalid("root thread binding has a non-thread atom domain");
        result.insert(result.end(), thread->compatibleAccCores.begin(),
                      thread->compatibleAccCores.end());
      }
    }
    canonicalizeFabricRefs(result);
    if (result.empty())
      return invalid("message terminal has no legal AccCore owner domain");
    return result;
  };

  struct AllowedOwnerDomain final {
    bool host = false;
    std::vector<::loom::fabric::AccCoreOccurrenceRef> accCores;
  };
  const auto rootBoundaryOwner =
      [&](const ::dataflow::RootThreadBoundaryTransferRef &transfer,
          bool source) -> llvm::Expected<AllowedOwnerDomain> {
    const bool completion =
        std::holds_alternative<::dataflow::RootThreadCompletionTransferRef>(
            transfer);
    const bool host = completion != source;
    if (host)
      return AllowedOwnerDomain{true, {}};
    const ::dataflow::RootThreadLaunchRef root =
        std::visit([](const auto &value) { return value.launch; }, transfer);
    auto cores = compatibleCores(root);
    if (!cores)
      return cores.takeError();
    return AllowedOwnerDomain{false, std::move(*cores)};
  };
  const auto producerOwner =
      [&](const ::dataflow::CanonicalProducerTerminalRef &producer)
      -> llvm::Expected<AllowedOwnerDomain> {
    if (const auto *root =
            std::get_if<::dataflow::RootThreadBoundarySourceRef>(&producer))
      return rootBoundaryOwner(root->transfer, true);
    if (const auto *graph =
            std::get_if<::dataflow::GraphLaunchBoundarySourceRef>(&producer)) {
      const auto root = std::visit(
          [](const auto &value) { return value.launch.rootThreadLaunch; },
          graph->transfer);
      auto cores = compatibleCores(root);
      if (!cores)
        return cores.takeError();
      return AllowedOwnerDomain{false, std::move(*cores)};
    }
    const auto &channel =
        std::get<::dataflow::ChannelProducerTerminalRef>(producer).producer;
    const auto root = std::visit(
        [](const auto &value) {
          if constexpr (std::is_same_v<
                            std::decay_t<decltype(value)>,
                            ::dataflow::GraphStreamOutputProducerRef>)
            return value.launch.rootThreadLaunch;
          else
            return value.launch;
        },
        channel);
    auto cores = compatibleCores(root);
    if (!cores)
      return cores.takeError();
    return AllowedOwnerDomain{false, std::move(*cores)};
  };
  const auto sinkOwner = [&](const ::dataflow::CanonicalSinkTerminalRef &sink)
      -> llvm::Expected<AllowedOwnerDomain> {
    if (const auto *root =
            std::get_if<::dataflow::RootThreadBoundarySinkRef>(&sink))
      return rootBoundaryOwner(root->transfer, false);
    if (const auto *graph =
            std::get_if<::dataflow::GraphLaunchBoundarySinkRef>(&sink)) {
      const auto root = std::visit(
          [](const auto &value) { return value.launch.rootThreadLaunch; },
          graph->transfer);
      auto cores = compatibleCores(root);
      if (!cores)
        return cores.takeError();
      return AllowedOwnerDomain{false, std::move(*cores)};
    }
    const auto &channel =
        std::get<::dataflow::ChannelConsumerTerminalRef>(sink).consumer;
    const auto root = std::visit(
        [](const auto &value) {
          if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                       ::dataflow::GraphStreamInputConsumerRef>)
            return value.launch.rootThreadLaunch;
          else
            return value.launch;
        },
        channel);
    auto cores = compatibleCores(root);
    if (!cores)
      return cores.takeError();
    return AllowedOwnerDomain{false, std::move(*cores)};
  };
  const auto endpointAllowed =
      [&](::loom::fabric::SystemServiceEndpointRef endpoint,
          const AllowedOwnerDomain &allowed) -> llvm::Expected<bool> {
    const auto *owner = fabric.serviceEndpointOwner(endpoint);
    if (!owner)
      return invalid("message service endpoint has no owner");
    if (const auto *host = std::get_if<::loom::fabric::HostCoreOccurrenceRef>(
            &owner->owner().payload))
      return allowed.host &&
             llvm::is_contained(fabric.artifact().hostCoreOccurrences(), *host);
    if (const auto *core = std::get_if<::loom::fabric::AccCoreOccurrenceRef>(
            &owner->owner().payload))
      return !allowed.host && llvm::is_contained(allowed.accCores, *core);
    return false;
  };

  const auto *producer =
      std::get_if<::loom::mapping::TransferObligationFamilyKey>(
          &obligation.key);
  if (!producer)
    return invalid("message obligation has a non-transfer key");
  for (const auto &leg : obligation.legs) {
    auto direction = ::dataflow::semantics::getCanonicalServiceLegDirection(
        member.kind, leg.ordinal);
    if (!direction)
      return direction.takeError();
    for (const bool source : {true, false}) {
      const std::size_t sinkCount = source ? 1 : obligation.sinks.size();
      for (std::size_t sink = 0; sink < sinkCount; ++sink) {
        auto allowed = source ? producerOwner(*producer)
                              : sinkOwner(obligation.sinks[sink]);
        if (!allowed)
          return allowed.takeError();
        for (const auto endpoint : fabric.artifact().systemServiceEndpoints()) {
          auto admitted = endpointAllowed(endpoint, *allowed);
          if (!admitted)
            return admitted.takeError();
          if (!*admitted)
            continue;
          const auto *capabilities =
              fabric.serviceEndpointCapabilities(endpoint);
          if (!capabilities ||
              capabilities->plane() != CanonicalServiceEndpointPlane::Transport)
            continue;
          auto capability = capabilityForKind(*capabilities, member.kind);
          if (!capability)
            return capability.takeError();
          if (!*capability ||
              !roleOwnsTerminal((*capability)->role(), *direction, source))
            continue;
          const ::loom::fabric::FabricTransportEndpointRef bound{
              ::loom::fabric::FabricTransportEndpointOwnerRef::of(endpoint), 0};
          const auto expectedDirection =
              source ? ::loom::fabric::FabricPortDirection::Output
                     : ::loom::fabric::FabricPortDirection::Input;
          if (fabric.artifact().transportEndpointDirection(bound) !=
              expectedDirection)
            return invalid("message service endpoint has the wrong direction");
          auto compatible = capabilityMatches(**capability, member);
          if (!compatible)
            return compatible.takeError();
          std::vector<::loom::fabric::FabricTransportEndpointRef> targets;
          if (*compatible)
            targets.push_back(bound);
          ::loom::mapping::SystemTransferTerminalKey terminal =
              source
                  ? ::loom::mapping::SystemTransferTerminalKey(
                        ::loom::mapping::SystemTransferSourceTerminalKey{leg})
                  : ::loom::mapping::SystemTransferTerminalKey(
                        ::loom::mapping::SystemTransferSinkTerminalKey{
                            leg,
                            static_cast<::dataflow::StructuralOrdinal>(sink)});
          if (llvm::Error error = appendTerminalCompatibility(
                  domain, std::move(terminal),
                  SystemBoundTerminalEndpoint{
                      SystemMessageTerminalEndpoint{bound}},
                  std::move(targets)))
            return error;
        }
      }
    }
  }
  return llvm::Error::success();
}

void applyServiceRestrictions(SystemSearchServiceDomain &domain,
                              const SystemFrozenConstraintIndex &constraints) {
  if (const auto *operation =
          std::get_if<::loom::mapping::OperationServiceObligationFamilyKey>(
              &domain.key)) {
    const ::loom::mapping::SystemConstraintSubject subject{*operation};
    for (SystemSearchServiceTargetCompatibility &row :
         domain.targetCompatibility) {
      auto *regions = std::get_if<
          std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>(
          &row.compatibleTargets);
      if (regions)
        applySystemConstraintRestriction(
            *regions, constraints,
            ::mapping::SystemConstraintProjection::ServiceTargetRegion,
            subject);
    }
  }
  for (SystemSearchTransferTerminalCompatibility &row :
       domain.transferTerminalCompatibility)
    applySystemConstraintRestriction(
        row.compatibleTransportEndpoints, constraints,
        ::mapping::SystemConstraintProjection::TransferTerminalAttachment,
        ::loom::mapping::SystemConstraintSubject{row.terminal});
}

} // namespace

llvm::Expected<std::vector<SystemSearchServiceDomain>>
projectSystemServiceDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    llvm::ArrayRef<SystemSearchBindingDomain> bindings,
    llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog,
    const SystemFrozenConstraintIndex &constraints, bool flatGraphSearch) {
  auto obligations =
      ::loom::mapping::projectSystemServiceObligations(dataflow, roots);
  if (!obligations)
    return obligations.takeError();
  const bool hasOperationService =
      llvm::any_of(*obligations, [](const auto &obligation) {
        return std::holds_alternative<
            ::loom::mapping::OperationServiceObligationFamilyKey>(
            obligation.key);
      });
  if (flatGraphSearch && hasOperationService)
    return llvm::make_error<UnsupportedSystemPnrSearchDomain>(
        UnsupportedSystemPnrSearchDomainReason::
            FlatOperationServiceDomainProjectionUnavailable,
        "flat operation-service compatibility projection is not implemented "
        "by the System PnR search-domain projector");
  auto messagePayloads = collectMessagePayloads(dataflow, roots);
  if (!messagePayloads)
    return messagePayloads.takeError();

  std::vector<SystemSearchServiceDomain> result;
  result.reserve(obligations->size());
  for (const auto &obligation : *obligations) {
    SystemSearchServiceDomain domain{obligation.key, {}, {}};
    std::vector<ResolvedServiceMember> members;
    members.reserve(obligation.members.size());
    for (const auto &memberRef : obligation.members) {
      auto member =
          resolveMember(dataflow, obligation, memberRef, *messagePayloads);
      if (!member)
        return member.takeError();
      members.push_back(std::move(*member));
    }

    if (std::holds_alternative<::loom::mapping::TransferObligationFamilyKey>(
            obligation.key)) {
      if (members.size() != 1 ||
          members.front().kind != ServiceKind::MessageTransfer)
        return invalid("message obligation does not have one message member");
      if (llvm::Error error = appendMessageRows(
              fabric, obligation, members.front(), bindings, domain))
        return std::move(error);
    } else {
      for (auto [memberRef, member] :
           llvm::zip_equal(obligation.members, members)) {
        auto pairs =
            boundPairsForMember(fabric, member, spatialCatalog, constraints);
        if (!pairs)
          return pairs.takeError();
        if (llvm::Error error = appendMemoryRows(fabric, obligation, memberRef,
                                                 member, *pairs, domain))
          return std::move(error);
      }
      for (const auto &exposure : obligation.exposures) {
        auto pairs = boundPairsForExposure(fabric, exposure, spatialCatalog,
                                           constraints);
        if (!pairs)
          return pairs.takeError();
        if (llvm::Error error =
                appendExposureRows(fabric, exposure, *pairs, domain))
          return std::move(error);
      }
    }
    applyServiceRestrictions(domain, constraints);
    result.push_back(std::move(domain));
  }
  return result;
}

llvm::Expected<std::vector<FrozenSystemMemoryServiceBinding>>
projectSystemMemoryServiceBindings(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog,
    const SystemFrozenConstraintIndex &constraints) {
  auto obligations =
      ::loom::mapping::projectSystemServiceObligations(dataflow, roots);
  if (!obligations)
    return obligations.takeError();
  auto messagePayloads = collectMessagePayloads(dataflow, roots);
  if (!messagePayloads)
    return messagePayloads.takeError();

  struct BindingMetadata final {
    std::optional<::loom::mapping::SpatialMemoryIntervalView> interval;
    std::optional<::loom::fabric::SubordinateEndpointRef> exposureTerminal;
  };
  const auto memberMetadata = [&](const SpatialCatalogEntry &entry,
                                  const ResolvedServiceMember &member)
      -> llvm::Expected<BindingMetadata> {
    if (!member.access)
      return BindingMetadata{};
    if (!member.contextualActor)
      return invalid("addressed service member has no contextual actor");
    const ::loom::mapping::SpatialMemoryBindingView *selected = nullptr;
    for (const auto &engine : entry.mapping.view().memoryEngineBindings())
      for (const auto &operation : engine.operations) {
        const auto *addressed =
            std::get_if<::loom::mapping::SpatialAddressedMemoryOperationView>(
                &operation);
        if (!addressed || addressed->actor != member.contextualActor->actor)
          continue;
        for (const auto &use : addressed->uses) {
          if (use.launch != member.contextualActor->launch)
            continue;
          const auto binding =
              llvm::find_if(entry.mapping.view().memoryBindings(),
                            [&](const auto &candidate) {
                              return candidate.entityId == use.binding;
                            });
          if (binding == entry.mapping.view().memoryBindings().end())
            return invalid("addressed service member names an absent binding");
          if (!std::holds_alternative<
                  ::loom::mapping::SpatialMemoryBoundaryProxyView>(
                  binding->target))
            return invalid(
                "System service member does not use a boundary proxy");
          if (selected && selected->entityId != binding->entityId)
            return invalid("one addressed service member selects multiple "
                           "logical intervals");
          selected = &*binding;
        }
      }
    if (!selected)
      return invalid("addressed service member has no boundary binding");
    return BindingMetadata{selected->interval, std::nullopt};
  };
  const auto exposureMetadata = [&](const SpatialCatalogEntry &entry,
                                    ::dataflow::MemoryExposureRef exposure)
      -> llvm::Expected<BindingMetadata> {
    const ::loom::mapping::SpatialMemoryBindingView *selected = nullptr;
    std::optional<::loom::fabric::SubordinateEndpointRef> terminal;
    for (const auto &binding : entry.mapping.view().memoryBindings())
      for (const auto &candidate : binding.exposures) {
        if (candidate.exposure != exposure)
          continue;
        if (!std::holds_alternative<
                ::loom::mapping::SpatialMemoryBoundaryProxyView>(
                binding.target))
          return invalid(
              "System memory exposure does not use a boundary proxy");
        if (selected && (selected->entityId != binding.entityId ||
                         *terminal != candidate.terminal))
          return invalid("one memory exposure selects multiple boundary "
                         "providers");
        selected = &binding;
        terminal = candidate.terminal;
      }
    if (!selected || !terminal)
      return invalid("memory exposure has no boundary provider");
    return BindingMetadata{selected->interval, *terminal};
  };

  std::vector<FrozenSystemMemoryServiceBinding> result;
  const auto append =
      [&](const auto &obligation, const SystemServiceTargetSubject &subject,
          const SpatialCatalogEntry &entry, const BindingMetadata &metadata,
          llvm::ArrayRef<BoundMemoryEndpointPair> pairs) -> llvm::Error {
    const auto *operation =
        std::get_if<::loom::mapping::OperationServiceObligationFamilyKey>(
            &obligation.key);
    const auto *logicalMemory =
        operation
            ? std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(operation)
            : nullptr;
    std::optional<::loom::fabric::FabricMemoryServiceSourceInterval>
        sourceInterval;
    if (logicalMemory) {
      if (!metadata.interval)
        return invalid("addressed service binding has no logical interval");
      if (const auto *range =
              std::get_if<::loom::mapping::SpatialMemoryByteRangeView>(
                  &*metadata.interval)) {
        sourceInterval = ::loom::fabric::FabricMemoryServiceSourceInterval{
            range->offsetBytes, range->sizeBytes};
      } else {
        auto extent = dataflow.staticMemoryByteExtent(*logicalMemory);
        if (!extent)
          return extent.takeError();
        if (*extent)
          sourceInterval =
              ::loom::fabric::FabricMemoryServiceSourceInterval{0, **extent};
      }
    }
    for (const BoundMemoryEndpointPair &pair : pairs) {
      const auto *spatialCore =
          std::get_if<::loom::fabric::SpatialCoreOccurrenceRef>(
              &pair.occurrenceEndpoint.owner.payload);
      if (!spatialCore)
        return invalid("memory service binding is not occurrence-qualified");
      llvm::Expected<std::vector<::loom::fabric::FabricMemoryServiceTargetPlan>>
          targetPlans =
              logicalMemory
                  ? sourceInterval
                        ? ::loom::fabric::projectFabricMemoryServiceTargetPlans(
                              fabric, pair.systemEndpoint, *sourceInterval)
                        : llvm::Expected<std::vector<
                              ::loom::fabric::FabricMemoryServiceTargetPlan>>(
                              std::vector<::loom::fabric::
                                              FabricMemoryServiceTargetPlan>{})
                  : ::loom::fabric::projectFabricMemoryServiceTargetPlans(
                        fabric, pair.systemEndpoint);
      if (!targetPlans)
        return targetPlans.takeError();
      result.push_back({obligation.key, subject, entry.reference,
                        spatialCore->core, pair.systemEndpoint,
                        pair.occurrenceEndpoint, std::move(*targetPlans),
                        metadata.interval, metadata.exposureTerminal});
    }
    return llvm::Error::success();
  };

  for (const auto &obligation : *obligations) {
    if (!std::holds_alternative<
            ::loom::mapping::OperationServiceObligationFamilyKey>(
            obligation.key))
      continue;
    for (const auto &memberRef : obligation.members) {
      auto member =
          resolveMember(dataflow, obligation, memberRef, *messagePayloads);
      if (!member)
        return member.takeError();
      const SystemServiceTargetSubject subject{
          SystemServiceMemberTargetSubject{memberRef}};
      for (const SpatialCatalogEntry &entry : spatialCatalog) {
        auto pairs = boundPairsForMember(
            fabric, *member, llvm::ArrayRef<SpatialCatalogEntry>(&entry, 1),
            constraints);
        if (!pairs)
          return pairs.takeError();
        if (pairs->empty())
          continue;
        auto metadata = memberMetadata(entry, *member);
        if (!metadata)
          return metadata.takeError();
        if (llvm::Error error =
                append(obligation, subject, entry, *metadata, *pairs))
          return std::move(error);
      }
    }
    for (const auto &exposure : obligation.exposures) {
      const SystemServiceTargetSubject subject{
          SystemMemoryExposureTargetSubject{exposure}};
      for (const SpatialCatalogEntry &entry : spatialCatalog) {
        auto pairs = boundPairsForExposure(
            fabric, exposure, llvm::ArrayRef<SpatialCatalogEntry>(&entry, 1),
            constraints);
        if (!pairs)
          return pairs.takeError();
        if (pairs->empty())
          continue;
        auto metadata = exposureMetadata(entry, exposure);
        if (!metadata)
          return metadata.takeError();
        if (llvm::Error error =
                append(obligation, subject, entry, *metadata, *pairs))
          return std::move(error);
      }
    }
  }

  return result;
}

llvm::Error validateSystemServiceDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots,
    llvm::ArrayRef<SystemSearchBindingDomain> bindings,
    llvm::ArrayRef<SystemSearchServiceDomain> services,
    const SystemFrozenConstraintIndex &constraints,
    llvm::ArrayRef<ArtifactRootReference> constraintSpatialMappings,
    const ArtifactStore &store) {
  bool flatGraphSearch = false;
  std::vector<ArtifactRootReference> spatialMappings(
      constraintSpatialMappings.begin(), constraintSpatialMappings.end());
  for (const SystemSearchBindingDomain &binding : bindings) {
    if (!std::holds_alternative<::dataflow::RootedGraphLaunchRef>(binding.key))
      continue;
    for (const SystemSearchAtom &atom : binding.atoms) {
      if (const auto *hierarchical =
              std::get_if<SystemHierarchicalGraphBindingDomain>(&atom.domain)) {
        spatialMappings.insert(spatialMappings.end(),
                               hierarchical->compatibleSpatialMappings.begin(),
                               hierarchical->compatibleSpatialMappings.end());
      } else {
        flatGraphSearch = true;
      }
    }
  }
  auto catalog = importSpatialCatalog(spatialMappings, dataflow, fabric, store);
  if (!catalog)
    return catalog.takeError();
  auto expected =
      projectSystemServiceDomains(dataflow, fabric, roots, bindings, *catalog,
                                  constraints, flatGraphSearch);
  if (!expected)
    return expected.takeError();
  if (services.size() != expected->size())
    return invalid("service-obligation closure differs from Dataflow");

  for (const SystemSearchServiceDomain &service : services) {
    const auto expectedService =
        llvm::find_if(*expected, [&](const auto &candidate) {
          return candidate.key == service.key;
        });
    if (expectedService == expected->end())
      return invalid("service-obligation closure differs from Dataflow");
    if (service.targetCompatibility.size() !=
        expectedService->targetCompatibility.size())
      return invalid("target-compatibility row closure differs from legal "
                     "bindings");
    for (const SystemSearchServiceTargetCompatibility &row :
         service.targetCompatibility) {
      const auto expectedRow = llvm::find_if(
          expectedService->targetCompatibility, [&](const auto &candidate) {
            return sameTargetSubject(candidate.subject, row.subject) &&
                   candidate.boundEndpoint == row.boundEndpoint;
          });
      if (expectedRow == expectedService->targetCompatibility.end())
        return invalid("target-compatibility row closure differs from legal "
                       "bindings");
      if (!(row.compatibleTargets == expectedRow->compatibleTargets))
        return invalid("target-compatibility domain is not exact");
    }

    if (service.transferTerminalCompatibility.size() !=
        expectedService->transferTerminalCompatibility.size())
      return invalid("transfer-terminal compatibility row closure differs "
                     "from legal bindings");
    for (const SystemSearchTransferTerminalCompatibility &row :
         service.transferTerminalCompatibility) {
      const auto expectedRow =
          llvm::find_if(expectedService->transferTerminalCompatibility,
                        [&](const auto &candidate) {
                          return candidate.terminal == row.terminal &&
                                 sameBoundEndpoint(candidate.boundEndpoint,
                                                   row.boundEndpoint);
                        });
      if (expectedRow == expectedService->transferTerminalCompatibility.end())
        return invalid("transfer-terminal compatibility row closure differs "
                       "from legal bindings");
      if (row.compatibleTransportEndpoints !=
          expectedRow->compatibleTransportEndpoints)
        return invalid("transfer-terminal compatibility domain is not exact");
    }
  }
  return llvm::Error::success();
}

} // namespace loom::pnr::detail

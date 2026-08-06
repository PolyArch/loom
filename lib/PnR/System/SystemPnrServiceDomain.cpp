#include "SystemPnrSearchDomainInternal.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
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
  std::optional<::dataflow::CanonicalActorSchemaProjection> actor;
  std::optional<CanonicalMemoryAccessView> access;
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
                                 std::nullopt, std::nullopt};
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
  } else if (!std::holds_alternative<::dataflow::FenceActorMemberRef>(member)) {
    return invalid("unknown canonical service-member variant");
  }
  return ResolvedServiceMember{*kind, std::nullopt, std::move(*actor),
                               std::move(access)};
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

llvm::Expected<const CanonicalServiceCapabilityRecord *> matchingCapability(
    const ::loom::fabric::CanonicalServiceCapabilitySet &capabilities,
    const ResolvedServiceMember &member) {
  const CanonicalServiceCapabilityRecord *result = nullptr;
  for (const CanonicalServiceCapabilityRecord &capability :
       capabilities.capabilities()) {
    auto matches = capabilityMatches(capability, member);
    if (!matches)
      return matches.takeError();
    if (!*matches)
      continue;
    if (result)
      return invalid("one service endpoint repeats a matching service kind");
    result = &capability;
  }
  return result;
}

bool roleOwnsTerminal(CanonicalServiceEndpointRole role,
                      ::dataflow::semantics::ServiceLegDirection direction,
                      bool source) {
  const bool initiatorIsSource =
      direction ==
      ::dataflow::semantics::ServiceLegDirection::InitiatorToServer;
  const bool endpointIsInitiator =
      role == CanonicalServiceEndpointRole::Initiate;
  return source ? endpointIsInitiator == initiatorIsSource
                : endpointIsInitiator != initiatorIsSource;
}

llvm::Expected<std::vector<::loom::fabric::FabricTransportEndpointRef>>
terminalEndpoints(const ::loom::fabric::FabricSystemRootView &fabric,
                  const ResolvedServiceMember &member,
                  ::dataflow::StructuralOrdinal legOrdinal, bool source) {
  auto direction = ::dataflow::semantics::getCanonicalServiceLegDirection(
      member.kind, legOrdinal);
  if (!direction)
    return direction.takeError();

  std::vector<::loom::fabric::FabricTransportEndpointRef> result;
  for (const ::loom::fabric::SystemServiceEndpointRef endpoint :
       fabric.artifact().systemServiceEndpoints()) {
    const auto *capabilities = fabric.serviceEndpointCapabilities(endpoint);
    if (!capabilities)
      return invalid("System service endpoint has no capability set");
    auto capability = matchingCapability(*capabilities, member);
    if (!capability)
      return capability.takeError();
    if (!*capability ||
        !roleOwnsTerminal((*capability)->role(), *direction, source))
      continue;

    const auto expectedPortDirection =
        source ? ::loom::fabric::FabricPortDirection::Output
               : ::loom::fabric::FabricPortDirection::Input;
    if (capabilities->plane() == CanonicalServiceEndpointPlane::Transport) {
      if (member.kind != ServiceKind::MessageTransfer)
        return invalid("memory service uses a transport-plane capability");
      const ::loom::fabric::FabricTransportEndpointRef carrier{
          ::loom::fabric::FabricTransportEndpointOwnerRef::of(endpoint), 0};
      if (fabric.artifact().transportEndpointDirection(carrier) !=
          expectedPortDirection)
        return invalid("message service carrier has the wrong direction");
      result.push_back(carrier);
      continue;
    }
    if (member.kind == ServiceKind::MessageTransfer)
      return invalid("message service uses a memory-plane capability");
    const ::loom::fabric::FabricMemoryEndpointRef memoryEndpoint{
        ::loom::fabric::FabricMemoryEndpointOwnerRef::of(endpoint), 0};
    const auto attachment = llvm::find_if(
        fabric.serviceLegCarrierAttachments(), [&](const auto &candidate) {
          return candidate.endpoint() == memoryEndpoint &&
                 candidate.kind() == member.kind &&
                 candidate.legOrdinal() == legOrdinal;
        });
    if (attachment == fabric.serviceLegCarrierAttachments().end())
      return invalid("matching memory service leg has no carrier attachment");
    for (const ::loom::fabric::FabricTransportEndpointRef &carrier :
         attachment->carriers()) {
      if (fabric.artifact().transportEndpointDirection(carrier) !=
          expectedPortDirection)
        return invalid("memory service carrier has the wrong direction");
      result.push_back(carrier);
    }
  }
  canonicalizeFabricRefs(result);
  return result;
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
                         llvm::ArrayRef<ResolvedServiceMember> members) {
  std::vector<::loom::fabric::FabricMemoryServiceRegionRef> result;
  if (members.empty() ||
      llvm::any_of(members, [](const auto &member) { return !member.access; }))
    return result;

  for (const ::loom::fabric::SystemServiceEndpointRef endpoint :
       fabric.artifact().systemServiceEndpoints()) {
    const auto *capabilities = fabric.serviceEndpointCapabilities(endpoint);
    const auto *owner = fabric.serviceEndpointOwner(endpoint);
    if (!capabilities || !owner ||
        capabilities->plane() != CanonicalServiceEndpointPlane::Memory ||
        capabilities->role() != CanonicalServiceEndpointRole::Serve)
      continue;
    bool endpointSupportsAll = true;
    for (const ResolvedServiceMember &member : members) {
      auto capability = matchingCapability(*capabilities, member);
      if (!capability)
        return capability.takeError();
      endpointSupportsAll &= *capability != nullptr;
    }
    if (!endpointSupportsAll)
      continue;

    const auto *serviceRef =
        std::get_if<::loom::fabric::FabricMemoryServiceRef>(
            &owner->owner().payload);
    if (!serviceRef)
      continue;
    const auto *systemRef = std::get_if<::loom::fabric::SystemMemoryServiceRef>(
        &serviceRef->payload);
    if (!systemRef)
      continue;
    const auto *service = fabric.memoryService(*systemRef);
    if (!service)
      return invalid("service endpoint names an absent System memory service");
    for (::loom::fabric::FabricOrdinal region = 0;
         region < service->regions().size(); ++region) {
      bool supportsAll = true;
      for (const ResolvedServiceMember &member : members) {
        auto supports = memoryCapabilitySupports(*service, member, region);
        if (!supports)
          return supports.takeError();
        supportsAll &= *supports;
      }
      if (supportsAll)
        result.push_back(
            {::loom::fabric::FabricMemoryServiceRef::system(*systemRef),
             region});
    }
  }
  canonicalizeFabricRefs(result);
  return result;
}

} // namespace

llvm::Expected<std::vector<SystemSearchServiceDomain>>
projectSystemServiceDomains(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> roots) {
  auto obligations =
      ::loom::mapping::projectSystemServiceObligations(dataflow, roots);
  if (!obligations)
    return obligations.takeError();
  if (!fabric.artifact().systemServiceTransforms().empty() &&
      llvm::any_of(*obligations, [](const auto &obligation) {
        return std::holds_alternative<
            ::loom::mapping::OperationServiceObligationFamilyKey>(
            obligation.key);
      }))
    return llvm::make_error<UnsupportedSystemPnrSearchDomain>(
        UnsupportedSystemPnrSearchDomainReason::
            ServiceTransformProjectionUnavailable,
        "service-transform closure is not implemented by the System PnR "
        "search-domain projector");
  auto messagePayloads = collectMessagePayloads(dataflow, roots);
  if (!messagePayloads)
    return messagePayloads.takeError();

  std::vector<SystemSearchServiceDomain> result;
  result.reserve(obligations->size());
  for (const ::loom::mapping::SystemServiceObligationProjection &obligation :
       *obligations) {
    std::vector<ResolvedServiceMember> members;
    members.reserve(obligation.members.size());
    for (const ::dataflow::ServiceMemberRef &member : obligation.members) {
      auto resolved =
          resolveMember(dataflow, obligation, member, *messagePayloads);
      if (!resolved)
        return resolved.takeError();
      members.push_back(std::move(*resolved));
    }

    SystemSearchServiceDomain domain{obligation.key, std::nullopt, {}};
    if (std::holds_alternative<
            ::loom::mapping::OperationServiceObligationFamilyKey>(
            obligation.key)) {
      auto regions = compatibleServiceRegions(fabric, members);
      if (!regions)
        return regions.takeError();
      domain.compatibleServiceRegions = std::move(*regions);
    }

    std::size_t expectedLegCount = 0;
    for (const ResolvedServiceMember &member : members)
      expectedLegCount +=
          ::dataflow::semantics::getCanonicalServiceLegCount(member.kind);
    if (obligation.legs.size() != expectedLegCount)
      return invalid("canonical service obligation has incomplete leg closure");
    for (const ::loom::mapping::CanonicalServiceLegKey &leg : obligation.legs) {
      const auto member = llvm::find(obligation.members, leg.member);
      if (member == obligation.members.end())
        return invalid("canonical service leg names a foreign member");
      const std::size_t memberIndex =
          static_cast<std::size_t>(member - obligation.members.begin());
      auto sources =
          terminalEndpoints(fabric, members[memberIndex], leg.ordinal,
                            /*source=*/true);
      if (!sources)
        return sources.takeError();
      domain.transferTerminals.push_back(
          {SystemTransferSourceTerminalKey{leg}, std::move(*sources)});

      const std::size_t sinkCount =
          members[memberIndex].kind == ServiceKind::MessageTransfer
              ? obligation.sinks.size()
              : 1;
      for (std::size_t sink = 0; sink < sinkCount; ++sink) {
        auto sinks =
            terminalEndpoints(fabric, members[memberIndex], leg.ordinal,
                              /*source=*/false);
        if (!sinks)
          return sinks.takeError();
        domain.transferTerminals.push_back(
            {SystemTransferSinkTerminalKey{
                 leg, static_cast<::dataflow::StructuralOrdinal>(sink)},
             std::move(*sinks)});
      }
    }
    result.push_back(std::move(domain));
  }
  return result;
}

} // namespace loom::pnr::detail

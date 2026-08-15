#include "SystemPnrSearchDomainInternal.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/Artifact/FabricMemoryServiceClosure.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/SystemServiceBindingProjection.h"

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

using ::dataflow::semantics::ServiceKind;
using ::loom::fabric::CanonicalServiceEndpointPlane;

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
  bool addressed = false;
};

struct BoundMemoryEndpointPair final {
  ::loom::fabric::SystemServiceEndpointRef systemEndpoint;
  ::loom::fabric::FabricMemoryEndpointRef occurrenceEndpoint;
};

llvm::Expected<ResolvedServiceMember> resolveMember(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::SystemServiceObligationProjection &obligation,
    const ::dataflow::ServiceMemberRef &member) {
  if (std::holds_alternative<::dataflow::MessageTransferMemberRef>(member)) {
    const auto *producer =
        std::get_if<::dataflow::CanonicalProducerTerminalRef>(&obligation.key);
    if (!producer)
      return invalid("message member belongs to an operation obligation");
    auto resolved = dataflow.resolve(*producer);
    if (!resolved)
      return resolved.takeError();
    return ResolvedServiceMember{ServiceKind::MessageTransfer,
                                 resolved->payloadType, std::nullopt, false};
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
  auto kind =
      ::loom::mapping::resolveSystemOperationServiceKind(dataflow, member);
  if (!kind)
    return kind.takeError();
  return ResolvedServiceMember{
      *kind, std::nullopt, *contextual,
      std::holds_alternative<::dataflow::AddressedMemoryActorMemberRef>(
          member)};
}

llvm::Expected<std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>
compatibleServiceRegions(const ::loom::fabric::FabricSystemRootView &fabric,
                         ::loom::fabric::SystemServiceEndpointRef endpoint) {
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

llvm::Expected<std::vector<BoundMemoryEndpointPair>> admittedBindingPairs(
    const SpatialCatalogEntry &entry, ::dataflow::RootedGraphLaunchRef launch,
    const ::loom::mapping::SystemSpatialMemoryBindingProjection &projection,
    const SystemFrozenConstraintIndex &constraints) {
  std::vector<BoundMemoryEndpointPair> result;
  for (const auto &pair : projection.endpointPairs) {
    const auto *spatialCore =
        std::get_if<::loom::fabric::SpatialCoreOccurrenceRef>(
            &pair.occurrenceEndpoint.owner.payload);
    if (!spatialCore)
      return invalid("memory spatial attachment is not occurrence-owned");
    if (!systemConstraintAllows(
            constraints,
            ::mapping::SystemConstraintProjection::GraphSelectedSpatialMapping,
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
    result.push_back({pair.systemEndpoint, pair.occurrenceEndpoint});
  }
  canonicalizePairs(result);
  return result;
}

llvm::Expected<std::vector<BoundMemoryEndpointPair>>
boundPairsForMember(const ::loom::fabric::FabricSystemRootView &fabric,
                    const ResolvedServiceMember &member,
                    llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog,
                    const SystemFrozenConstraintIndex &constraints) {
  std::vector<BoundMemoryEndpointPair> pairs;
  if (!member.contextualActor)
    return pairs;
  ::dataflow::ServiceMemberRef memberRef =
      member.addressed
          ? ::dataflow::ServiceMemberRef(
                ::dataflow::AddressedMemoryActorMemberRef{
                    *member.contextualActor})
          : ::dataflow::ServiceMemberRef(
                ::dataflow::FenceActorMemberRef{*member.contextualActor});
  const ::loom::mapping::ServicePlanSelectionAnchor anchor =
      ::loom::mapping::ServiceMemberPlanSelectionAnchor{memberRef};
  for (const SpatialCatalogEntry &entry : spatialCatalog) {
    auto projection = ::loom::mapping::projectSystemSpatialMemoryBinding(
        fabric, entry.mapping->view(), entry.moduleDependencyOrdinal, anchor);
    if (!projection)
      return projection.takeError();
    auto admitted = admittedBindingPairs(entry, member.contextualActor->launch,
                                         *projection, constraints);
    if (!admitted)
      return admitted.takeError();
    pairs.insert(pairs.end(), admitted->begin(), admitted->end());
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
  const ::loom::mapping::ServicePlanSelectionAnchor anchor =
      ::loom::mapping::MemoryExposurePlanSelectionAnchor{exposure};
  for (const SpatialCatalogEntry &entry : spatialCatalog) {
    auto projection = ::loom::mapping::projectSystemSpatialMemoryBinding(
        fabric, entry.mapping->view(), entry.moduleDependencyOrdinal, anchor);
    if (!projection)
      return projection.takeError();
    auto admitted =
        admittedBindingPairs(entry, exposure.launch, *projection, constraints);
    if (!admitted)
      return admitted.takeError();
    pairs.insert(pairs.end(), admitted->begin(), admitted->end());
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
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ::loom::mapping::SystemServiceObligationProjection &obligation,
    const ::dataflow::ServiceMemberRef &memberRef,
    const ResolvedServiceMember &member,
    llvm::ArrayRef<BoundMemoryEndpointPair> pairs,
    SystemSearchServiceDomain &domain) {
  const SystemServiceTargetSubject subject{
      SystemServiceMemberTargetSubject{memberRef}};
  std::map<std::string, bool> admittedEndpoints;
  for (const auto endpoint : systemEndpoints(pairs)) {
    if (member.addressed) {
      auto regions = ::loom::mapping::projectSystemOperationTargetRegions(
          dataflow, fabric, endpoint, memberRef);
      if (!regions)
        return regions.takeError();
      admittedEndpoints.emplace(
          keyString(::loom::fabric::canonicalFabricBytes(endpoint)),
          !regions->empty());
      domain.targetCompatibility.push_back(
          {subject, endpoint, std::move(*regions)});
    } else {
      auto consistency = ::loom::mapping::projectSystemFenceTargetDomains(
          dataflow, fabric, endpoint, memberRef);
      if (!consistency)
        return consistency.takeError();
      admittedEndpoints.emplace(
          keyString(::loom::fabric::canonicalFabricBytes(endpoint)),
          !consistency->empty());
      domain.targetCompatibility.push_back(
          {subject, endpoint, std::move(*consistency)});
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
      const auto admitted = admittedEndpoints.find(
          keyString(::loom::fabric::canonicalFabricBytes(pair.systemEndpoint)));
      if (admitted == admittedEndpoints.end())
        return invalid("bound System endpoint lost its target domain");
      for (const bool source : {true, false}) {
        auto bound =
            selectMemoryTerminalEndpoint(fabric, pair, *direction, source);
        if (!bound)
          return bound.takeError();
        const auto selected =
            std::get<SystemMemoryOrFenceTerminalEndpoint>(*bound).endpoint;
        auto carriers =
            carriersForMemoryTerminal(fabric, selected, member.kind,
                                      leg.ordinal, source, admitted->second);
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
    auto regions = compatibleServiceRegions(fabric, endpoint);
    if (!regions)
      return regions.takeError();
    domain.targetCompatibility.push_back(
        {subject, endpoint, std::move(*regions)});
  }
  return llvm::Error::success();
}

llvm::Error appendMessageRows(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
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

  using Owner = ::loom::mapping::SystemMessageExecutionOwner;
  using Owners = std::vector<Owner>;
  const auto rootBoundaryOwner =
      [&](const ::dataflow::RootThreadBoundaryTransferRef &transfer,
          bool source) -> llvm::Expected<Owners> {
    const bool completion =
        std::holds_alternative<::dataflow::RootThreadCompletionTransferRef>(
            transfer);
    const bool host = completion != source;
    if (host) {
      if (fabric.artifact().hostCoreOccurrences().size() != 1)
        return invalid("message runtime terminal has no unique HostCore owner");
      return Owners{Owner{fabric.artifact().hostCoreOccurrences().front()}};
    }
    const ::dataflow::RootThreadLaunchRef root =
        std::visit([](const auto &value) { return value.launch; }, transfer);
    auto cores = compatibleCores(root);
    if (!cores)
      return cores.takeError();
    Owners result;
    for (const auto core : *cores)
      result.emplace_back(core);
    return result;
  };
  const auto producerOwner =
      [&](const ::dataflow::CanonicalProducerTerminalRef &producer)
      -> llvm::Expected<Owners> {
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
      Owners result;
      for (const auto core : *cores)
        result.emplace_back(core);
      return result;
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
    Owners result;
    for (const auto core : *cores)
      result.emplace_back(core);
    return result;
  };
  const auto sinkOwner = [&](const ::dataflow::CanonicalSinkTerminalRef &sink)
      -> llvm::Expected<Owners> {
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
      Owners result;
      for (const auto core : *cores)
        result.emplace_back(core);
      return result;
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
    Owners result;
    for (const auto core : *cores)
      result.emplace_back(core);
    return result;
  };

  const auto *producer =
      std::get_if<::loom::mapping::TransferObligationFamilyKey>(
          &obligation.key);
  if (!producer)
    return invalid("message obligation has a non-transfer key");
  if (!member.messagePayload || !*member.messagePayload)
    return invalid("message obligation has no Dataflow payload type");
  for (const auto &leg : obligation.legs) {
    for (const bool source : {true, false}) {
      const std::size_t sinkCount = source ? 1 : obligation.sinks.size();
      for (std::size_t sink = 0; sink < sinkCount; ++sink) {
        auto allowed = source ? producerOwner(*producer)
                              : sinkOwner(obligation.sinks[sink]);
        if (!allowed)
          return allowed.takeError();
        const ::loom::mapping::SystemTransferTerminalKey terminal =
            source ? ::loom::mapping::SystemTransferTerminalKey(
                         ::loom::mapping::SystemTransferSourceTerminalKey{leg})
                   : ::loom::mapping::SystemTransferTerminalKey(
                         ::loom::mapping::SystemTransferSinkTerminalKey{
                             leg,
                             static_cast<::dataflow::StructuralOrdinal>(sink)});
        for (const Owner &owner : *allowed) {
          auto rows =
              ::loom::mapping::projectSystemMessageTerminalEndpointDomains(
                  dataflow, fabric, terminal, *member.messagePayload, owner);
          if (!rows)
            return rows.takeError();
          for (const auto &row : *rows) {
            std::vector<::loom::fabric::FabricTransportEndpointRef> targets;
            if (row.payloadCompatible)
              targets.push_back(row.endpoint);
            if (llvm::Error error = appendTerminalCompatibility(
                    domain, terminal,
                    SystemBoundTerminalEndpoint{
                        SystemMessageTerminalEndpoint{row.endpoint}},
                    std::move(targets)))
              return error;
          }
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
    const SystemFrozenConstraintIndex &constraints) {
  auto obligations =
      ::loom::mapping::projectSystemServiceObligations(dataflow, roots);
  if (!obligations)
    return obligations.takeError();
  std::vector<SystemSearchServiceDomain> result;
  result.reserve(obligations->size());
  for (const auto &obligation : *obligations) {
    SystemSearchServiceDomain domain{obligation.key, {}, {}};
    std::vector<ResolvedServiceMember> members;
    members.reserve(obligation.members.size());
    for (const auto &memberRef : obligation.members) {
      auto member = resolveMember(dataflow, obligation, memberRef);
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
              dataflow, fabric, obligation, members.front(), bindings, domain))
        return std::move(error);
    } else {
      for (auto [memberRef, member] :
           llvm::zip_equal(obligation.members, members)) {
        auto pairs =
            boundPairsForMember(fabric, member, spatialCatalog, constraints);
        if (!pairs)
          return pairs.takeError();
        if (llvm::Error error =
                appendMemoryRows(dataflow, fabric, obligation, memberRef,
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
  struct BindingMetadata final {
    std::optional<::loom::mapping::SpatialMemoryIntervalView> interval;
    std::optional<::loom::fabric::SubordinateEndpointRef> exposureTerminal;
  };

  std::vector<FrozenSystemMemoryServiceBinding> result;
  const auto append =
      [&](const auto &obligation, const SystemServiceTargetSubject &subject,
          const ResolvedServiceMember *member, const SpatialCatalogEntry &entry,
          const BindingMetadata &metadata,
          llvm::ArrayRef<BoundMemoryEndpointPair> pairs) -> llvm::Error {
    const auto *operation =
        std::get_if<::loom::mapping::OperationServiceObligationFamilyKey>(
            &obligation.key);
    const auto *logicalMemory =
        operation
            ? std::get_if<::dataflow::LogicalMemoryRootOrViewRef>(operation)
            : nullptr;
    for (const BoundMemoryEndpointPair &pair : pairs) {
      const auto *spatialCore =
          std::get_if<::loom::fabric::SpatialCoreOccurrenceRef>(
              &pair.occurrenceEndpoint.owner.payload);
      if (!spatialCore)
        return invalid("memory service binding is not occurrence-qualified");
      llvm::Expected<std::vector<::loom::fabric::FabricMemoryServiceTargetPlan>>
          targetPlans =
              logicalMemory
                  ? metadata.interval
                        ? ::loom::mapping::projectSystemMemoryTargetPlans(
                              dataflow, fabric, pair.systemEndpoint,
                              *logicalMemory, *metadata.interval)
                        : llvm::Expected<std::vector<
                              ::loom::fabric::FabricMemoryServiceTargetPlan>>(
                              invalid("addressed service binding has "
                                      "no logical interval"))
                  : ::loom::fabric::projectFabricMemoryServiceTargetPlans(
                        fabric, pair.systemEndpoint);
      if (!targetPlans)
        return targetPlans.takeError();
      std::vector<FrozenSystemMemoryServiceBinding::UsePatternDomain> patterns;
      if (member && member->addressed) {
        const auto *memberSubject =
            std::get_if<SystemServiceMemberTargetSubject>(&subject);
        if (!memberSubject)
          return invalid("addressed service binding has no member subject");
        auto projected = ::loom::mapping::projectSystemMemoryUsePatternDomains(
            dataflow, fabric, memberSubject->member, *targetPlans);
        if (!projected)
          return projected.takeError();
        patterns.reserve(projected->size());
        for (auto &domain : *projected)
          patterns.push_back({domain.region, std::move(domain.patterns)});
      }
      result.push_back({obligation.key, subject, entry.reference,
                        spatialCore->core, pair.systemEndpoint,
                        pair.occurrenceEndpoint, std::move(*targetPlans),
                        std::move(patterns), metadata.interval,
                        metadata.exposureTerminal});
    }
    return llvm::Error::success();
  };

  for (const auto &obligation : *obligations) {
    if (!std::holds_alternative<
            ::loom::mapping::OperationServiceObligationFamilyKey>(
            obligation.key))
      continue;
    for (const auto &memberRef : obligation.members) {
      auto member = resolveMember(dataflow, obligation, memberRef);
      if (!member)
        return member.takeError();
      const SystemServiceTargetSubject subject{
          SystemServiceMemberTargetSubject{memberRef}};
      for (const SpatialCatalogEntry &entry : spatialCatalog) {
        const ::loom::mapping::ServicePlanSelectionAnchor anchor =
            ::loom::mapping::ServiceMemberPlanSelectionAnchor{memberRef};
        auto projection = ::loom::mapping::projectSystemSpatialMemoryBinding(
            fabric, entry.mapping->view(), entry.moduleDependencyOrdinal,
            anchor);
        if (!projection)
          return projection.takeError();
        if (!member->contextualActor)
          return invalid("operation service member has no contextual actor");
        auto pairs = admittedBindingPairs(
            entry, member->contextualActor->launch, *projection, constraints);
        if (!pairs)
          return pairs.takeError();
        if (pairs->empty())
          continue;
        const BindingMetadata metadata{projection->interval,
                                       projection->exposureTerminal};
        if (llvm::Error error =
                append(obligation, subject, &*member, entry, metadata, *pairs))
          return std::move(error);
      }
    }
    for (const auto &exposure : obligation.exposures) {
      const SystemServiceTargetSubject subject{
          SystemMemoryExposureTargetSubject{exposure}};
      for (const SpatialCatalogEntry &entry : spatialCatalog) {
        const ::loom::mapping::ServicePlanSelectionAnchor anchor =
            ::loom::mapping::MemoryExposurePlanSelectionAnchor{exposure};
        auto projection = ::loom::mapping::projectSystemSpatialMemoryBinding(
            fabric, entry.mapping->view(), entry.moduleDependencyOrdinal,
            anchor);
        if (!projection)
          return projection.takeError();
        auto pairs = admittedBindingPairs(entry, exposure.launch, *projection,
                                          constraints);
        if (!pairs)
          return pairs.takeError();
        if (pairs->empty())
          continue;
        const BindingMetadata metadata{projection->interval,
                                       projection->exposureTerminal};
        if (llvm::Error error =
                append(obligation, subject, nullptr, entry, metadata, *pairs))
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
    llvm::ArrayRef<SpatialCatalogEntry> spatialCatalog) {
  auto expected = projectSystemServiceDomains(dataflow, fabric, roots, bindings,
                                              spatialCatalog, constraints);
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

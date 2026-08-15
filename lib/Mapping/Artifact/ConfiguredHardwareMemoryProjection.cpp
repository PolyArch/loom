#include "ConfiguredHardwareProjectionInternal.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Identity/FabricMemoryConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::mapping::detail {
namespace {

using Role = ::dataflow::semantics::ServiceValueRole;

constexpr std::uint32_t roleCount =
    static_cast<std::uint32_t>(Role::Completion) + 1;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

const TechMemoryRealizationView *findRealization(const TechMappingView &mapping,
                                                 std::uint64_t entity) {
  const TechMemoryRealizationView *result = nullptr;
  for (const TechMemoryRealizationView &candidate :
       mapping.memoryRealizations()) {
    if (candidate.entityId != entity)
      continue;
    if (result)
      return nullptr;
    result = &candidate;
  }
  return result;
}

const TechMemoryActorView *findActor(const TechMemoryRealizationView &mapping,
                                     ::dataflow::ActorRef actor) {
  const TechMemoryActorView *result = nullptr;
  for (const TechMemoryActorView &candidate : mapping.actors) {
    if (candidate.actor != actor)
      continue;
    if (result)
      return nullptr;
    result = &candidate;
  }
  return result;
}

const SpatialMemoryBindingView *
findBinding(llvm::ArrayRef<SpatialMemoryBindingView> bindings,
            std::uint64_t entity) {
  const SpatialMemoryBindingView *result = nullptr;
  for (const SpatialMemoryBindingView &candidate : bindings) {
    if (candidate.entityId != entity)
      continue;
    if (result)
      return nullptr;
    result = &candidate;
  }
  return result;
}

llvm::Expected<std::vector<::loom::fabric::ManagerEndpointRef>>
managerInventory(const ::loom::fabric::FabricArtifactView &fabric,
                 ::loom::fabric::FabricMemoryOccurrenceRef memory) {
  std::vector<::loom::fabric::ManagerEndpointRef> result;
  const auto owner = ::loom::fabric::FabricMemoryEndpointOwnerRef::of(memory);
  for (::loom::fabric::FabricOrdinal ordinal = 0;
       ordinal != fabric.memoryEndpointCount(owner); ++ordinal) {
    const ::loom::fabric::FabricMemoryEndpointRef endpoint{owner, ordinal};
    auto role = fabric.memoryEndpointRole(endpoint);
    if (!role)
      return invalid("memory endpoint has no role");
    if (*role == ::loom::fabric::FabricMemoryEndpointRole::Manager)
      result.emplace_back(endpoint);
  }
  return result;
}

llvm::Expected<std::vector<::loom::fabric::SubordinateEndpointRef>>
subordinateInventory(const ::loom::fabric::FabricArtifactView &fabric,
                     ::loom::fabric::FabricMemoryOccurrenceRef memory) {
  std::vector<::loom::fabric::SubordinateEndpointRef> result;
  const auto owner = ::loom::fabric::FabricMemoryEndpointOwnerRef::of(memory);
  for (::loom::fabric::FabricOrdinal ordinal = 0;
       ordinal != fabric.memoryEndpointCount(owner); ++ordinal) {
    const ::loom::fabric::FabricMemoryEndpointRef endpoint{owner, ordinal};
    auto role = fabric.memoryEndpointRole(endpoint);
    if (!role)
      return invalid("memory endpoint has no role");
    if (*role == ::loom::fabric::FabricMemoryEndpointRole::Subordinate)
      result.emplace_back(endpoint);
  }
  return result;
}

llvm::Expected<::fabric::MemoryDispatchTarget>
dispatchTarget(const ::loom::fabric::FabricArtifactView &fabric,
               ::loom::fabric::FabricMemoryOccurrenceRef memory,
               const SpatialMemoryDispatchTargetView &target) {
  if (std::holds_alternative<::loom::fabric::LocalMemoryServiceRef>(target))
    return ::fabric::MemoryDispatchTarget(
        std::in_place_type<::fabric::LocalMemoryDispatchTarget>);
  auto managers = managerInventory(fabric, memory);
  if (!managers)
    return managers.takeError();
  const auto &selected = std::get<::loom::fabric::ManagerEndpointRef>(target);
  const auto found = llvm::find(*managers, selected);
  if (found == managers->end())
    return invalid("memory dispatch selects a foreign manager endpoint");
  return ::fabric::MemoryDispatchTarget(
      std::in_place_type<::fabric::ManagerMemoryDispatchTarget>,
      ::fabric::ManagerMemoryDispatchTarget{
          static_cast<std::uint64_t>(std::distance(managers->begin(), found))});
}

llvm::Expected<::fabric::MemoryDispatchTarget>
consistencyTarget(const ::loom::fabric::FabricArtifactView &fabric,
                  ::loom::fabric::FabricMemoryOccurrenceRef memory,
                  const SpatialMemoryConsistencyTargetView &target) {
  if (const auto *manager =
          std::get_if<::loom::fabric::ManagerEndpointRef>(&target))
    return dispatchTarget(
        fabric, memory,
        SpatialMemoryDispatchTargetView(
            std::in_place_type<::loom::fabric::ManagerEndpointRef>, *manager));
  if (!fabric.declaresLocalMemoryService(memory))
    return invalid("local consistency target has no local memory service");
  return ::fabric::MemoryDispatchTarget(
      std::in_place_type<::fabric::LocalMemoryDispatchTarget>);
}

std::uint64_t logicalIntervalBegin(const SpatialMemoryBindingView &binding) {
  if (const auto *range =
          std::get_if<SpatialMemoryByteRangeView>(&binding.interval))
    return range->offsetBytes;
  return 0;
}

llvm::Expected<std::uint64_t>
logicalIntervalSize(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                    const SpatialMemoryBindingView &binding) {
  if (const auto *range =
          std::get_if<SpatialMemoryByteRangeView>(&binding.interval))
    return range->sizeBytes;
  auto extent = dataflow.staticMemoryByteExtent(binding.logicalMemory);
  if (!extent)
    return extent.takeError();
  if (!*extent || **extent == 0)
    return invalid("provider range requires a finite logical memory extent");
  return **extent;
}

std::uint64_t localBaseOffset(const SpatialMemoryBindingView &binding) {
  const auto *local =
      std::get_if<SpatialMemoryLocalRegionView>(&binding.target);
  if (!local)
    return 0;
  return local->physicalOffsetBytes - logicalIntervalBegin(binding);
}

llvm::Expected<std::pair<::fabric::MemoryDispatchTarget, std::uint64_t>>
operationUseProjection(const ::loom::fabric::FabricArtifactView &fabric,
                       ::loom::fabric::FabricMemoryOccurrenceRef memory,
                       llvm::ArrayRef<SpatialMemoryBindingView> bindings,
                       const SpatialAddressedMemoryUseView &use,
                       bool rootRelative) {
  auto target = dispatchTarget(fabric, memory, use.dispatch);
  if (!target)
    return target.takeError();
  const SpatialMemoryBindingView *binding = findBinding(bindings, use.binding);
  if (!binding)
    return invalid("memory operation use has no unique binding");
  return std::make_pair(std::move(*target),
                        rootRelative ? localBaseOffset(*binding) : 0);
}

bool sameTarget(const ::fabric::MemoryDispatchTarget &lhs,
                const ::fabric::MemoryDispatchTarget &rhs) {
  return lhs == rhs;
}

llvm::Expected<std::pair<::fabric::MemoryDispatchTarget, std::uint64_t>>
deriveRowTarget(const ::loom::fabric::FabricArtifactView &fabric,
                ::loom::fabric::FabricMemoryOccurrenceRef memory,
                llvm::ArrayRef<SpatialMemoryBindingView> bindings,
                const SpatialMemoryOperationView &operation,
                bool rootRelative) {
  std::optional<std::pair<::fabric::MemoryDispatchTarget, std::uint64_t>>
      selected;
  llvm::Error error = llvm::Error::success();
  std::visit(
      [&](const auto &typed) {
        using Operation = std::decay_t<decltype(typed)>;
        for (const auto &use : typed.uses) {
          llvm::Expected<
              std::pair<::fabric::MemoryDispatchTarget, std::uint64_t>>
              projected = [&]()
              -> llvm::Expected<
                  std::pair<::fabric::MemoryDispatchTarget, std::uint64_t>> {
            if constexpr (std::is_same_v<Operation,
                                         SpatialAddressedMemoryOperationView>) {
              return operationUseProjection(fabric, memory, bindings, use,
                                            rootRelative);
            } else {
              auto target = consistencyTarget(fabric, memory, use.consistency);
              if (!target)
                return target.takeError();
              return std::make_pair(std::move(*target), 0);
            }
          }();
          if (!projected) {
            error = projected.takeError();
            return;
          }
          if (selected && (!sameTarget(selected->first, projected->first) ||
                           selected->second != projected->second)) {
            error = invalid(
                "one memory operation placement requires different rooted-use "
                "service targets or base addresses");
            return;
          }
          selected = std::move(*projected);
        }
      },
      operation);
  if (error)
    return std::move(error);
  if (!selected)
    return invalid("memory operation has no rooted use");
  return std::move(*selected);
}

llvm::Expected<const SpatialRouteTreeView *>
findProducerRoute(llvm::ArrayRef<SpatialRouteTreeView> routes,
                  const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
                  std::uint64_t &ordinal) {
  const SpatialRouteTreeView *result = nullptr;
  for (auto [candidateOrdinal, candidate] : llvm::enumerate(routes)) {
    if (candidate.logicalNet != producer)
      continue;
    if (result)
      return invalid("logical producer has multiple RouteTrees");
    result = &candidate;
    ordinal = candidateOrdinal;
  }
  return result;
}

llvm::Expected<std::pair<const SpatialRouteTreeView *, std::uint64_t>>
findConsumerRoute(llvm::ArrayRef<SpatialRouteTreeView> routes,
                  const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer,
                  std::uint64_t &routeOrdinal) {
  std::optional<std::pair<const SpatialRouteTreeView *, std::uint64_t>> result;
  for (auto [candidateOrdinal, route] : llvm::enumerate(routes)) {
    for (const SpatialRouteSinkView &sink : route.sinks) {
      if (sink.sink != consumer)
        continue;
      if (result)
        return invalid("logical consumer appears in multiple RouteTrees");
      result = std::make_pair(&route, sink.nodeOrdinal);
      routeOrdinal = candidateOrdinal;
    }
  }
  if (!result)
    return invalid("external memory operand has no RouteTree");
  return *result;
}

llvm::Expected<::loom::fabric::FabricMemoryExternalRoleSource>
externalInputSource(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> segments,
    const ::dataflow::CanonicalGraphConsumerEndpointRef &consumer,
    ::loom::fabric::FabricTransportEndpointRef expectedEndpoint) {
  std::uint64_t routeOrdinal = 0;
  auto selected = findConsumerRoute(routes, consumer, routeOrdinal);
  if (!selected)
    return selected.takeError();
  if (selected->first->nodes[selected->second].endpoint != expectedEndpoint)
    return invalid("memory input route selects the wrong occurrence endpoint");
  auto tag = resolveConfiguredHardwarePhysicalTag(
      fabric, routes, resourceUses, segments, routeOrdinal, selected->second);
  if (!tag)
    return tag.takeError();
  return ::loom::fabric::FabricMemoryExternalRoleSource{
      expectedEndpoint.ordinal, std::move(*tag)};
}

llvm::Expected<::loom::fabric::FabricMemoryExternalRoleSource>
externalOutputDestination(
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> segments,
    const ::dataflow::CanonicalGraphProducerEndpointRef &producer,
    ::loom::fabric::FabricTransportEndpointRef expectedEndpoint) {
  std::uint64_t routeOrdinal = 0;
  auto route = findProducerRoute(routes, producer, routeOrdinal);
  if (!route)
    return route.takeError();
  if (!*route)
    return invalid("externally exposed memory result has no RouteTree");
  if ((*route)->rootEndpoint != expectedEndpoint)
    return invalid("memory output route selects the wrong occurrence endpoint");
  auto tag = resolveConfiguredHardwarePhysicalTag(fabric, routes, resourceUses,
                                                  segments, routeOrdinal, 0);
  if (!tag)
    return tag.takeError();
  return ::loom::fabric::FabricMemoryExternalRoleSource{
      expectedEndpoint.ordinal, std::move(*tag)};
}

llvm::Expected<::loom::fabric::FabricOrdinal>
selectedUsePattern(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                   const ::loom::fabric::FabricArtifactView &fabric,
                   llvm::ArrayRef<SpatialResourceUseView> resourceUses,
                   std::uint64_t realization, ::dataflow::ActorRef actor,
                   ::loom::fabric::FabricMemoryOperationPortRef port) {
  auto trigger = deriveSpatialMemoryIssueEvent(dataflow, actor);
  if (!trigger)
    return trigger.takeError();
  const ::loom::fabric::FabricUsePatternOwnerRef expectedOwner(
      ::loom::fabric::FabricInventoryOwnerRef::of(port));
  std::optional<::loom::fabric::FabricOrdinal> selected;
  for (const SpatialResourceUseView &use : resourceUses) {
    const auto *owner =
        std::get_if<SpatialMemoryEngineResourceOwnerRef>(&use.owner);
    if (!owner || owner->realization != realization ||
        !(use.activation.trigger.event == SpatialActivityEventRef(*trigger)) ||
        use.useSite.owner != expectedOwner)
      continue;
    if (selected)
      return invalid("memory operation has multiple selected UsePatterns");
    selected = use.useSite.ordinal;
  }
  if (!selected)
    return invalid("memory operation has no selected UsePattern");
  const auto *record = fabric.memoryOperationPort(port);
  if (!record || *selected >= record->resourceContract().usePatternCount())
    return invalid("memory operation UsePattern is out of range");
  return *selected;
}

struct RoleProjection final {
  std::vector<std::optional<::loom::fabric::FabricMemoryRoleSource>> sources;
  std::vector<std::optional<::loom::fabric::FabricMemoryRoleDestination>>
      destinations;
};

llvm::Expected<RoleProjection>
deriveRoles(const ::dataflow::CanonicalDataflowProgramView &dataflow,
            const ::loom::fabric::FabricArtifactView &fabric,
            const TechMemoryActorView &actor,
            const SpatialMemoryActorRoleDemandView &demand,
            llvm::ArrayRef<SpatialRouteTreeView> routes,
            llvm::ArrayRef<SpatialResourceUseView> resourceUses,
            llvm::ArrayRef<SpatialPhysicalTagSegmentView> tagSegments,
            ::loom::fabric::FabricMemoryOccurrenceRef memory) {
  auto resolved = dataflow.resolve(actor.actor);
  if (!resolved)
    return resolved.takeError();
  auto service =
      ::dataflow::semantics::CanonicalService::forActor(resolved->op);
  if (!service)
    return service.takeError();
  if (actor.operandPorts.size() != service->arguments().size() ||
      actor.resultPorts.size() != service->results().size())
    return invalid("Tech memory role map has the wrong service shape");
  if (demand.actor != actor.actor || demand.occurrence != memory ||
      demand.sources.size() != roleCount ||
      demand.destinations.size() != roleCount)
    return invalid("memory physical role demand has the wrong owner or shape");

  RoleProjection result;
  result.sources.resize(roleCount);
  result.destinations.resize(roleCount);
  const auto endpointOwner =
      ::loom::fabric::FabricTransportEndpointOwnerRef::of(memory);
  for (auto [ordinal, argument] : llvm::enumerate(service->arguments())) {
    auto operand = service->argumentValue(resolved->op, ordinal);
    if (!operand)
      return operand.takeError();
    const ::dataflow::CanonicalGraphConsumerEndpointRef consumer(
        ::dataflow::ActorTokenOperandRef{
            actor.actor, static_cast<::dataflow::StructuralOrdinal>(
                             (*operand)->getOperandNumber())});
    const std::size_t role = static_cast<std::size_t>(argument.role);
    if (role >= demand.sources.size() || !demand.sources[role])
      return invalid("memory input role has no physical demand");
    if (const auto *internal =
            std::get_if<::loom::fabric::FabricMemoryHandshakeInternalRoleSource>(
                &*demand.sources[role])) {
      result.sources[static_cast<unsigned>(argument.role)] =
          ::loom::fabric::FabricMemoryInternalRoleSource{
              internal->connection};
      continue;
    }
    const auto &externalDemand =
        std::get<::loom::fabric::FabricMemoryHandshakeExternalRoleSource>(
            *demand.sources[role]);
    auto projectedExternal = externalInputSource(
        fabric, routes, resourceUses, tagSegments, consumer,
        ::loom::fabric::FabricTransportEndpointRef{
            endpointOwner, externalDemand.endpoint});
    if (!projectedExternal)
      return projectedExternal.takeError();
    result.sources[static_cast<unsigned>(argument.role)] =
        std::move(*projectedExternal);
  }

  for (auto [ordinal, output] : llvm::enumerate(service->results())) {
    auto value = service->resultValue(resolved->op, ordinal);
    if (!value)
      return value.takeError();
    const ::dataflow::CanonicalGraphProducerEndpointRef producer(
        ::dataflow::ActorTokenResultRef{
            actor.actor, static_cast<::dataflow::StructuralOrdinal>(
                             value->getResultNumber())});
    const std::size_t role = static_cast<std::size_t>(output.role);
    if (role >= demand.destinations.size() || !demand.destinations[role])
      return invalid("memory output role has no physical demand");
    ::loom::fabric::FabricMemoryRoleDestination destination;
    destination.internalConnections =
        demand.destinations[role]->internalConnections;
    std::uint64_t routeOrdinal = 0;
    auto producerRoute = findProducerRoute(routes, producer, routeOrdinal);
    if (!producerRoute)
      return producerRoute.takeError();
    if (demand.destinations[role]->externalEndpoint) {
      if (!*producerRoute)
        return invalid("external memory result has no residual route");
      auto external = externalOutputDestination(
          fabric, routes, resourceUses, tagSegments, producer,
          ::loom::fabric::FabricTransportEndpointRef{
              endpointOwner,
              *demand.destinations[role]->externalEndpoint});
      if (!external)
        return external.takeError();
      destination.external = std::move(*external);
    } else if (*producerRoute) {
      return invalid("internal-only memory result has an external route");
    }
    if (!destination.external && destination.internalConnections.empty())
      return invalid("memory result has no external or internal destination");
    result.destinations[static_cast<unsigned>(output.role)] =
        std::move(destination);
  }
  return result;
}

struct MemoryConfigurationBuilder final {
  ::loom::fabric::FabricMemoryConfigurationSchemaView schema;
  ::loom::fabric::FabricMemoryActive active;
  bool selected = false;
};

llvm::Expected<MemoryConfigurationBuilder>
makeBuilder(const ::loom::fabric::FabricArtifactView &fabric,
            ::loom::fabric::FabricMemoryOccurrenceRef memory) {
  auto schema = fabric.memoryConfigurationSchema(memory);
  if (!schema)
    return schema.takeError();
  ::loom::fabric::FabricMemoryActive active;
  active.operationRows.resize(schema->layout().operationRows.size());
  active.providerDecodeRows.resize(schema->layout().providerRows.size());
  for (auto [ordinal, rows] : llvm::enumerate(schema->layout().providerRows))
    active.providerDecodeRows[ordinal].resize(rows.size());
  return MemoryConfigurationBuilder{std::move(*schema), std::move(active),
                                    false};
}

llvm::Expected<std::uint64_t>
operationRowOrdinal(const SpatialMemoryOperationPlacementView &placement) {
  if (const auto *port =
          std::get_if<::loom::fabric::FabricMemoryOperationPortRef>(&placement))
    return port->ordinal;
  return std::get<::loom::fabric::FabricMemoryOperationContextRef>(placement)
      .ordinal;
}

::loom::fabric::FabricMemoryOperationPortRef
operationPort(const SpatialMemoryOperationPlacementView &placement) {
  if (const auto *port =
          std::get_if<::loom::fabric::FabricMemoryOperationPortRef>(&placement))
    return *port;
  return std::get<::loom::fabric::FabricMemoryOperationContextRef>(placement)
      .port;
}

llvm::Error addEngineConfiguration(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialMemoryBindingView> bindings,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> tagSegments,
    const SpatialMemoryEngineBindingView &engine,
    MemoryConfigurationBuilder &builder) {
  const TechMemoryRealizationView *realization =
      findRealization(techMapping, engine.realization);
  if (!realization)
    return invalid("memory configuration has no unique Tech realization");
  auto roleDemands = deriveSpatialMemoryActorRoleDemands(
      dataflow, techMapping, fabric, *realization, engine.occurrence);
  if (!roleDemands)
    return roleDemands.takeError();
  for (const SpatialMemoryOperationView &operation : engine.operations) {
    const auto actorRef = std::visit(
        [](const auto &selected) { return selected.actor; }, operation);
    const auto &placement = std::visit(
        [](const auto &selected)
            -> const SpatialMemoryOperationPlacementView & {
          return selected.placement;
        },
        operation);
    const TechMemoryActorView *actor = findActor(*realization, actorRef);
    if (!actor)
      return invalid("memory configuration has no unique Tech actor");
    const auto demand = llvm::find_if(
        *roleDemands, [&](const SpatialMemoryActorRoleDemandView &candidate) {
          return candidate.actor == actorRef;
        });
    if (demand == roleDemands->end())
      return invalid("memory configuration has no physical role demand");
    auto resolved = dataflow.resolve(actorRef);
    if (!resolved)
      return resolved.takeError();
    auto actorProjection =
        ::dataflow::projectRegisteredActorSchemaProjection(resolved->op);
    if (!actorProjection)
      return actorProjection.takeError();
    std::optional<::dataflow::semantics::CanonicalMemoryAccessView> access;
    if (actorProjection->schema !=
        ::dataflow::OperationSchemaId::DataflowFence) {
      auto projected =
          ::dataflow::semantics::getCanonicalMemoryAccessView(resolved->op);
      if (!projected)
        return projected.takeError();
      access.emplace(std::move(*projected));
    }
    const auto port = operationPort(placement);
    auto pattern = selectedUsePattern(dataflow, fabric, resourceUses,
                                      engine.realization, actorRef, port);
    if (!pattern)
      return pattern.takeError();
    auto roles = deriveRoles(dataflow, fabric, *actor, *demand, routes,
                             resourceUses, tagSegments, engine.occurrence);
    if (!roles)
      return roles.takeError();
    auto target = deriveRowTarget(
        fabric, engine.occurrence, bindings, operation,
        !access || access->addressForm() ==
                       ::dataflow::semantics::MemoryAddressForm::RootRelative);
    if (!target)
      return target.takeError();
    auto row = builder.schema.projectOperationRow(
        port.ordinal, actor->capability.ordinal, *pattern, *actorProjection,
        access, target->second, std::move(roles->sources),
        std::move(roles->destinations), std::move(target->first));
    if (!row)
      return row.takeError();
    auto rowOrdinal = operationRowOrdinal(placement);
    if (!rowOrdinal)
      return rowOrdinal.takeError();
    if (*rowOrdinal >= builder.active.operationRows.size())
      return invalid("memory operation placement exceeds its row table");
    auto &selected = builder.active.operationRows[*rowOrdinal];
    if (selected)
      return invalid("one memory row is selected by multiple operations");
    selected = std::move(*row);
    builder.selected = true;
  }
  return llvm::Error::success();
}

llvm::Expected<::loom::fabric::FabricMemoryProviderMatch>
prefixMatch(std::uint64_t base, std::uint64_t size) {
  if (!llvm::isPowerOf2_64(size) || base % size != 0)
    return invalid(
        "provider Prefix cannot exactly represent its binding range");
  const unsigned suffix = llvm::Log2_64(size);
  return ::loom::fabric::FabricMemoryPrefixMatch{
      base, static_cast<std::uint8_t>(64 - suffix)};
}

llvm::Expected<::loom::fabric::FabricMemoryProviderDecodeRow>
providerRow(const ::dataflow::CanonicalDataflowProgramView &dataflow,
            const ::loom::fabric::FabricArtifactView &fabric,
            ::loom::fabric::FabricMemoryOccurrenceRef memory,
            const SpatialMemoryBindingView &binding,
            const SpatialExposureEntryView &exposure,
            const ::fabric::MemorySubordinateDispatchDeclaration &declaration) {
  ::loom::fabric::FabricMemoryProviderDecodeRow result;
  const std::uint64_t begin = logicalIntervalBegin(binding);
  std::optional<std::uint64_t> size;
  for (::fabric::MemoryProviderMatchField field : declaration.matchFields) {
    if (field == ::fabric::MemoryProviderMatchField::AddressSpace ||
        field == ::fabric::MemoryProviderMatchField::Context)
      return invalid(
          "provider AddressSpace or Context match has no canonical Mapping "
          "projection");
    if (!size) {
      auto projected = logicalIntervalSize(dataflow, binding);
      if (!projected)
        return projected.takeError();
      size = *projected;
    }
    if (field == ::fabric::MemoryProviderMatchField::Range) {
      result.matches.emplace_back(
          ::loom::fabric::FabricMemoryRangeMatch{begin, *size});
    } else {
      auto prefix = prefixMatch(begin, *size);
      if (!prefix)
        return prefix.takeError();
      result.matches.push_back(std::move(*prefix));
    }
  }
  auto target = dispatchTarget(fabric, memory, exposure.dispatch);
  if (!target)
    return target.takeError();
  result.serviceTarget = std::move(*target);
  const std::uint64_t offset = localBaseOffset(binding);
  if (declaration.addressTransform ==
      ::fabric::MemoryProviderAddressTransform::ConstantBaseOffset) {
    result.baseOffsetBytes = offset;
  } else if (offset != 0) {
    return invalid("provider binding requires an undeclared base transform");
  }
  return result;
}

llvm::Error addProviderConfigurations(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialMemoryBindingView> bindings,
    ::loom::fabric::FabricMemoryOccurrenceRef memory,
    MemoryConfigurationBuilder &builder) {
  auto subordinates = subordinateInventory(fabric, memory);
  if (!subordinates)
    return subordinates.takeError();
  const auto *connectivity = fabric.memoryConnectivity(memory);
  if (!connectivity ||
      connectivity->subordinateEndpoints().size() != subordinates->size())
    return invalid("memory subordinate inventories disagree");

  for (auto [endpointOrdinal, terminal] : llvm::enumerate(*subordinates)) {
    struct BindingExposure final {
      const SpatialMemoryBindingView *binding = nullptr;
      const SpatialExposureEntryView *exposure = nullptr;
    };
    std::map<std::uint64_t, BindingExposure> selectedBindings;
    for (const SpatialMemoryBindingView &binding : bindings) {
      for (const SpatialExposureEntryView &exposure : binding.exposures) {
        if (exposure.terminal != terminal)
          continue;
        auto [entry, inserted] = selectedBindings.emplace(
            binding.entityId, BindingExposure{&binding, &exposure});
        if (!inserted && entry->second.exposure->dispatch != exposure.dispatch)
          return invalid("one provider binding has conflicting dispatch rows");
      }
    }
    const auto &declaration =
        connectivity->subordinateEndpoints()[endpointOrdinal];
    if (selectedBindings.size() >
        builder.active.providerDecodeRows[endpointOrdinal].size())
      return invalid("provider decode exceeds its bounded row capacity");
    if (selectedBindings.size() > 1 && declaration.matchFields.empty())
      return invalid("multi-binding provider has no match field");
    std::size_t rowOrdinal = 0;
    for (const auto &[entity, selection] : selectedBindings) {
      (void)entity;
      auto row = providerRow(dataflow, fabric, memory, *selection.binding,
                             *selection.exposure, declaration);
      if (!row)
        return row.takeError();
      builder.active.providerDecodeRows[endpointOrdinal][rowOrdinal++] =
          std::move(*row);
      builder.selected = true;
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<ConfiguredHardwareFieldValueView>>
deriveConfiguredMemoryFields(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialMemoryEngineBindingView> memoryEngines,
    llvm::ArrayRef<SpatialMemoryBindingView> memoryBindings,
    llvm::ArrayRef<SpatialRouteTreeView> routes,
    llvm::ArrayRef<SpatialResourceUseView> resourceUses,
    llvm::ArrayRef<SpatialPhysicalTagSegmentView> physicalTagSegments) {
  std::map<std::string, std::pair<::loom::fabric::FabricMemoryOccurrenceRef,
                                  MemoryConfigurationBuilder>>
      builders;
  for (const auto memory : fabric.memoryOccurrences()) {
    auto builder = makeBuilder(fabric, memory);
    if (!builder)
      return builder.takeError();
    builders.emplace(byteKey(::loom::fabric::canonicalFabricBytes(memory)),
                     std::make_pair(memory, std::move(*builder)));
  }

  for (const SpatialMemoryEngineBindingView &engine : memoryEngines) {
    auto found = builders.find(
        byteKey(::loom::fabric::canonicalFabricBytes(engine.occurrence)));
    if (found == builders.end())
      return invalid("memory engine binding names an absent occurrence");
    if (llvm::Error error = addEngineConfiguration(
            dataflow, techMapping, fabric, memoryBindings, routes, resourceUses,
            physicalTagSegments, engine, found->second.second))
      return std::move(error);
  }
  for (auto &[key, entry] : builders) {
    (void)key;
    if (llvm::Error error = addProviderConfigurations(
            dataflow, fabric, memoryBindings, entry.first, entry.second))
      return std::move(error);
  }

  std::vector<ConfiguredHardwareFieldValueView> fields;
  for (auto &[key, entry] : builders) {
    (void)key;
    MemoryConfigurationBuilder &builder = entry.second;
    if (!builder.selected)
      continue;
    auto value = builder.schema.encode(
        ::loom::fabric::FabricMemoryConfigurationValue{builder.active});
    if (!value)
      return value.takeError();
    auto residencies = fabric.configurationResidencies(builder.schema.field());
    if (!residencies)
      return residencies.takeError();
    const ::loom::fabric::FabricConfigurationResidency staticResidency =
        ::loom::fabric::FabricStaticConfigurationResidency{};
    if (residencies->size() != 1 ||
        !llvm::is_contained(*residencies, staticResidency))
      return invalid("memory configuration field is not uniquely static");
    fields.push_back({::loom::fabric::FabricConfigurationSlotRef{
                          builder.schema.field(), staticResidency},
                      std::move(*value)});
  }
  return fields;
}

} // namespace loom::mapping::detail

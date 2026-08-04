#include "CGRAMemoryPlan.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <map>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

enum class MemoryActionOwnerKind : std::uint8_t { Engine, Binding };

struct MemoryActionKey final {
  MemoryActionOwnerKind kind = MemoryActionOwnerKind::Engine;
  std::uint64_t owner = 0;
  std::vector<std::uint8_t> event;

  friend bool operator<(const MemoryActionKey &lhs,
                        const MemoryActionKey &rhs) {
    return std::tie(lhs.kind, lhs.owner, lhs.event) <
           std::tie(rhs.kind, rhs.owner, rhs.event);
  }
};

struct MemoryActorProjection final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  ::dataflow::semantics::CanonicalService service;
  std::optional<::dataflow::semantics::CanonicalMemoryAccessView> access;
  ::loom::mapping::SpatialActorTransitionEventRef trigger;
  ::dataflow::GraphRef graph;
};

llvm::Expected<MemoryActorProjection>
projectMemoryActor(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                   ::dataflow::ActorRef actor) {
  auto resolved = dataflow.resolve(actor);
  if (!resolved)
    return resolved.takeError();
  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(resolved->op);
  if (!projection)
    return projection.takeError();
  auto service =
      ::dataflow::semantics::CanonicalService::forActor(resolved->op);
  if (!service)
    return service.takeError();
  std::optional<::dataflow::semantics::CanonicalMemoryAccessView> access;
  if (service->kind() != ::dataflow::semantics::ServiceKind::MemoryFence) {
    auto projected =
        ::dataflow::semantics::getCanonicalMemoryAccessView(resolved->op);
    if (!projected)
      return projected.takeError();
    access.emplace(std::move(*projected));
  }
  auto trigger =
      ::loom::mapping::deriveSpatialMemoryIssueEvent(dataflow, actor);
  if (!trigger)
    return trigger.takeError();
  return MemoryActorProjection{std::move(*projection), std::move(*service),
                               std::move(access), std::move(*trigger),
                               resolved->graph};
}

::loom::fabric::FabricMemoryOperationPortRef operationPort(
    const ::loom::mapping::SpatialMemoryOperationPlacementView &placement) {
  return std::visit(
      [](const auto &selected) {
        using Placement = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<
                          Placement,
                          ::loom::fabric::FabricMemoryOperationPortRef>)
          return selected;
        else
          return selected.port;
      },
      placement);
}

llvm::Expected<std::vector<std::uint8_t>>
eventKey(const ::dataflow::CanonicalDataflowProgramView &dataflow,
         const ::loom::mapping::SpatialActorTransitionEventRef &event) {
  return ::loom::mapping::encodeSpatialActivityEventKey(
      dataflow.identity(), ::loom::mapping::SpatialActivityEventRef(event));
}

llvm::Expected<std::map<MemoryActionKey, std::uint64_t>> indexMemoryActions(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::SpatialMappingView &spatial,
    llvm::ArrayRef<CgraPhysicalUseClientKind> physicalUseClients) {
  if (physicalUseClients.size() != spatial.resourceUses().size())
    return invalid("CGRA memory physical-use client index is incomplete");
  std::map<MemoryActionKey, std::uint64_t> result;
  for (auto [ordinal, use] : llvm::enumerate(spatial.resourceUses())) {
    if (physicalUseClients[ordinal] !=
        CgraPhysicalUseClientKind::MemoryTransition)
      continue;
    const auto *trigger =
        std::get_if<::loom::mapping::SpatialActorTransitionEventRef>(
            &use.activation.trigger.event);
    if (!trigger)
      return invalid("CGRA memory ResourceUse has no actor transition");
    auto encoded = eventKey(dataflow, *trigger);
    if (!encoded)
      return encoded.takeError();

    MemoryActionKey key;
    if (const auto *engine =
            std::get_if<::loom::mapping::SpatialMemoryEngineResourceOwnerRef>(
                &use.owner)) {
      key = {MemoryActionOwnerKind::Engine, engine->realization,
             std::move(*encoded)};
    } else if (const auto *binding = std::get_if<
                   ::loom::mapping::SpatialMemoryBindingResourceOwnerRef>(
                   &use.owner)) {
      key = {MemoryActionOwnerKind::Binding, binding->binding,
             std::move(*encoded)};
    } else {
      return invalid("CGRA memory ResourceUse has an invalid owner");
    }
    if (!result.try_emplace(std::move(key), ordinal).second)
      return invalid("CGRA memory ResourceUse has a duplicate owner event");
  }
  return result;
}

llvm::Expected<std::uint64_t>
requireAction(const std::map<MemoryActionKey, std::uint64_t> &actions,
              MemoryActionOwnerKind kind, std::uint64_t owner,
              llvm::ArrayRef<std::uint8_t> event) {
  MemoryActionKey key{kind, owner,
                      std::vector<std::uint8_t>(event.begin(), event.end())};
  auto found = actions.find(key);
  if (found == actions.end())
    return invalid("CGRA memory execution has no selected physical action");
  return found->second;
}

llvm::Error
appendTransactionPlan(const ::fabric::MemoryPortTransactionPlan &transaction,
                      CgraMemoryPlan &result, CgraMemoryActorPlan &actor) {
  if (transaction.transactions().size() >
          std::numeric_limits<std::uint32_t>::max() ||
      transaction.assembly().results().size() >
          std::numeric_limits<std::uint32_t>::max())
    return invalid("CGRA memory transaction projection exceeds u32");
  actor.childTransactionOffset = result.childTransactions.size();
  actor.childTransactionCount =
      static_cast<std::uint32_t>(transaction.transactions().size());
  for (const auto &child : transaction.transactions())
    result.childTransactions.push_back(
        {child.activation().kind(), child.activation().lane(),
         child.projection().kind(), child.projection().lane()});
  actor.resultAssemblyOffset = result.resultAssemblies.size();
  actor.resultAssemblyCount =
      static_cast<std::uint32_t>(transaction.assembly().results().size());
  for (const auto &assembly : transaction.assembly().results())
    result.resultAssemblies.push_back({assembly.role(), assembly.strategy(),
                                       assembly.laneCount(),
                                       assembly.inactiveValue()});
  return llvm::Error::success();
}

} // namespace

llvm::Expected<CgraMemoryPlan> freezeCgraMemoryPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::mapping::TechMappingView &tech,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::mapping::SpatialMappingView &spatial,
    llvm::ArrayRef<CgraPhysicalUseClientKind> physicalUseClients) {
  auto actions = indexMemoryActions(dataflow, spatial, physicalUseClients);
  if (!actions)
    return actions.takeError();

  llvm::DenseMap<std::uint64_t,
                 const ::loom::mapping::TechMemoryRealizationView *>
      realizations;
  realizations.reserve(tech.memoryRealizations().size());
  for (const auto &realization : tech.memoryRealizations())
    if (!realizations.try_emplace(realization.entityId, &realization).second)
      return invalid("CGRA memory plan found duplicate Tech realizations");

  llvm::DenseMap<std::uint64_t,
                 const ::loom::mapping::SpatialMemoryBindingView *>
      bindings;
  bindings.reserve(spatial.memoryBindings().size());
  CgraMemoryPlan result;
  result.bindings.reserve(spatial.memoryBindings().size());
  for (const auto &binding : spatial.memoryBindings()) {
    if (!bindings.try_emplace(binding.entityId, &binding).second)
      return invalid("CGRA memory plan found duplicate memory bindings");
    result.bindings.push_back({binding.entityId, binding.logicalMemory,
                               binding.interval, binding.target});
  }

  llvm::DenseSet<std::uint64_t> consumedActions;
  for (const auto &engine : spatial.memoryEngineBindings()) {
    auto realization = realizations.find(engine.realization);
    if (realization == realizations.end())
      return invalid("CGRA memory plan found an unknown Tech realization");
    if (fabric.memoryEngineTemplateOf(engine.occurrence) !=
        realization->second->engine)
      return invalid("CGRA memory occurrence selects the wrong definition");

    for (const auto &operation : engine.operations) {
      const auto actorRef = std::visit(
          [](const auto &selected) { return selected.actor; }, operation);
      const auto &placement = std::visit(
          [](const auto &selected)
              -> const ::loom::mapping::SpatialMemoryOperationPlacementView & {
            return selected.placement;
          },
          operation);
      auto techActor = llvm::find_if(
          realization->second->actors,
          [&](const auto &candidate) { return candidate.actor == actorRef; });
      if (techActor == realization->second->actors.end())
        return invalid("CGRA memory actor is absent from Tech realization");
      auto projection = projectMemoryActor(dataflow, actorRef);
      if (!projection)
        return projection.takeError();
      auto encodedEvent = eventKey(dataflow, projection->trigger);
      if (!encodedEvent)
        return encodedEvent.takeError();
      auto operationAction =
          requireAction(*actions, MemoryActionOwnerKind::Engine,
                        engine.realization, *encodedEvent);
      if (!operationAction)
        return operationAction.takeError();
      consumedActions.insert(*operationAction);

      const auto port = operationPort(placement);
      if (port.memory != engine.occurrence ||
          techActor->operationPort.ordinal != port.ordinal ||
          techActor->capability.port != techActor->operationPort)
        return invalid("CGRA memory placement disagrees with Tech selection");
      const auto *portRecord = fabric.memoryOperationPort(port);
      const ::loom::fabric::FabricMemoryCapabilityAlternativeRef capability{
          port, techActor->capability.ordinal};
      const auto *capabilityRecord =
          fabric.memoryCapabilityAlternative(capability);
      if (!portRecord || !capabilityRecord)
        return invalid("CGRA memory selection has no physical capability");
      const auto &selectedUse = spatial.resourceUses()[*operationAction];
      if (!llvm::is_contained(
              capabilityRecord->admissibleUsePatterns,
              ::fabric::UsePatternKey(selectedUse.useSite.ordinal)))
        return invalid("CGRA memory action selects an inadmissible pattern");

      std::vector<::fabric::MemoryPortTransactionProjection> projections;
      projections.reserve(portRecord->operationPatterns().size());
      for (const auto &pattern : portRecord->operationPatterns())
        projections.push_back(pattern.transactionProjection);
      auto resource = ::fabric::MemoryOperationPortResourceView::create(
          port, portRecord->resourceContract(), projections);
      if (!resource)
        return resource.takeError();
      auto pattern = resource->operationPattern(selectedUse.useSite);
      if (!pattern)
        return pattern.takeError();
      auto transaction = ::fabric::deriveMemoryPortTransactionPlan(
          *pattern, projection->actor, projection->service, projection->access);
      if (!transaction)
        return transaction.takeError();

      CgraMemoryActorPlan actor{actorRef,
                                projection->graph,
                                engine.occurrence,
                                placement,
                                capability,
                                *operationAction,
                                result.rootedUses.size(),
                                0,
                                0,
                                0,
                                0,
                                0};
      const auto appendAddressedUse =
          [&](const ::loom::mapping::SpatialAddressedMemoryUseView &use)
          -> llvm::Error {
        if (!bindings.contains(use.binding))
          return invalid("CGRA memory use references an unknown binding");
        CgraMemoryRootedUsePlan rooted{
            use.launch, use.binding,
            std::visit(
                [](const auto &target) -> CgraMemoryServiceTarget {
                  return CgraMemoryServiceTarget(target);
                },
                use.dispatch),
            std::nullopt};
        if (std::holds_alternative<::loom::fabric::LocalMemoryServiceRef>(
                use.dispatch)) {
          auto serviceAction =
              requireAction(*actions, MemoryActionOwnerKind::Binding,
                            use.binding, *encodedEvent);
          if (!serviceAction)
            return serviceAction.takeError();
          rooted.localServicePhysicalUseOrdinal = *serviceAction;
          consumedActions.insert(*serviceAction);
        }
        result.rootedUses.push_back(std::move(rooted));
        return llvm::Error::success();
      };
      if (const auto *addressed =
              std::get_if<::loom::mapping::SpatialAddressedMemoryOperationView>(
                  &operation)) {
        for (const auto &use : addressed->uses)
          if (llvm::Error error = appendAddressedUse(use))
            return std::move(error);
      } else {
        const auto &fence =
            std::get<::loom::mapping::SpatialFenceMemoryOperationView>(
                operation);
        for (const auto &use : fence.uses)
          result.rootedUses.push_back(
              {use.launch, std::nullopt,
               std::visit(
                   [](const auto &target) -> CgraMemoryServiceTarget {
                     return CgraMemoryServiceTarget(target);
                   },
                   use.consistency),
               std::nullopt});
      }
      const std::uint64_t rootedCount =
          result.rootedUses.size() - actor.rootedUseOffset;
      if (rootedCount == 0 ||
          rootedCount > std::numeric_limits<std::uint32_t>::max())
        return invalid("CGRA memory actor has an invalid rooted-use count");
      actor.rootedUseCount = static_cast<std::uint32_t>(rootedCount);
      if (llvm::Error error =
              appendTransactionPlan(*transaction, result, actor))
        return std::move(error);
      result.actors.push_back(std::move(actor));
    }
  }

  if (consumedActions.size() != actions->size())
    return invalid("CGRA memory ResourceUse has no selected operation or use");
  return result;
}

} // namespace loom::sim::detail

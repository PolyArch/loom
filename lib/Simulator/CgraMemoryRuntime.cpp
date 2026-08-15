#include "CgraMemoryRuntime.h"

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/Identity/FabricMemoryInternalConnection.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <algorithm>
#include <limits>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom::sim::detail {
namespace {

constexpr std::uint64_t invalidBinding =
    std::numeric_limits<std::uint64_t>::max();
constexpr std::size_t memoryRoleCount =
    static_cast<std::size_t>(
        ::dataflow::semantics::ServiceValueRole::Completion) +
    1;

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

llvm::Error unsupported(llvm::Twine message) {
  return llvm::createStringError(std::make_error_code(std::errc::not_supported),
                                 message);
}

bool isPermissionEvent(const CgraPhysicalUseTiming &timing,
                       CgraPhysicalLifecycleKind kind) {
  return timing.commitRank ? kind == CgraPhysicalLifecycleKind::Committed
                           : kind == CgraPhysicalLifecycleKind::Granted;
}

llvm::Expected<std::uint32_t>
activeChildCount(llvm::ArrayRef<CgraMemoryChildTransactionPlan> children,
                 const llvm::APInt &activeLanes) {
  std::uint64_t count = 0;
  for (const CgraMemoryChildTransactionPlan &child : children) {
    bool active = false;
    switch (child.activation) {
    case ::fabric::MemoryChildActivationKind::Always:
      active = true;
      break;
    case ::fabric::MemoryChildActivationKind::ParentMaskAny:
      active = !activeLanes.isZero();
      break;
    case ::fabric::MemoryChildActivationKind::ParentMaskLane:
      if (!child.activationLane ||
          *child.activationLane >= activeLanes.getBitWidth())
        return invalid("CGRA memory child activation lane is out of range");
      active = activeLanes[*child.activationLane];
      break;
    }
    if (active)
      ++count;
  }
  if (count > std::numeric_limits<std::uint32_t>::max())
    return invalid("CGRA active memory child count exceeds u32");
  return static_cast<std::uint32_t>(count);
}

llvm::Error validateChild(const CgraMemoryChildTransactionPlan &child) {
  if ((child.activation ==
       ::fabric::MemoryChildActivationKind::ParentMaskLane) !=
      child.activationLane.has_value())
    return invalid("CGRA memory child activation lane is malformed");
  if ((child.projection == ::fabric::MemoryChildProjectionKind::ElementLane) !=
      child.projectionLane.has_value())
    return invalid("CGRA memory child projection lane is malformed");
  return llvm::Error::success();
}

llvm::Error executionFailure(const SimulatorState &state,
                             llvm::StringRef fallback) {
  const llvm::StringRef diagnostic =
      state.diagnostics.empty() ? fallback : state.diagnostics.back();
  if (state.failure == RunFailure::UnsupportedCapability)
    return unsupported(diagnostic);
  return invalid(diagnostic);
}

} // namespace

llvm::Expected<CgraMemoryRuntime> CgraMemoryRuntime::create(
    const CgraFrozenExecutionPlan &plan,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef launch, ::dataflow::GraphRef graph,
    const PreparedGraphExecution &execution, SimulatorState &state,
    CgraPhysicalActionRuntime &physical) {
  if (state.execution != &execution)
    return invalid("CGRA memory state does not use the prepared graph");
  auto launchedGraph = dataflow.resolve(launch);
  if (!launchedGraph)
    return launchedGraph.takeError();
  if (*launchedGraph != graph)
    return invalid("CGRA memory launch selects another graph");
  if (plan.physicalUseTimings.size() != plan.resources.selectedUses.size() ||
      plan.physicalUseClients.size() != plan.physicalUseTimings.size())
    return invalid("CGRA memory physical action coverage is incomplete");

  llvm::DenseMap<mlir::Operation *, std::uint64_t> semanticOrdinals;
  semanticOrdinals.reserve(execution.actorPlans.size());
  for (auto [ordinal, actor] : llvm::enumerate(execution.actorPlans))
    if (!semanticOrdinals.try_emplace(actor.operation, ordinal).second)
      return invalid("prepared graph contains a duplicate actor operation");

  std::vector<ActorBinding> bindings;
  std::vector<std::uint64_t> bindingBySemanticActor(execution.actorPlans.size(),
                                                    invalidBinding);
  std::vector<bool> internalProducerValidated(
      plan.memory.internalConnections.size(), false);
  std::vector<bool> internalConsumerValidated(
      plan.memory.internalConnections.size(), false);
  std::vector<::loom::fabric::FabricMemoryInternalConnectionUse> closureUses;
  for (std::size_t ordinal = 0;
       ordinal != plan.memory.internalConnections.size(); ++ordinal) {
    const CgraMemoryInternalConnectionPlan &connection =
        plan.memory.internalConnections[ordinal];
    for (std::size_t previous = 0; previous != ordinal; ++previous) {
      const CgraMemoryInternalConnectionPlan &other =
          plan.memory.internalConnections[previous];
      if (connection.occurrence != other.occurrence ||
          connection.connection != other.connection)
        continue;
      if (connection.producer == other.producer &&
          connection.consumer == other.consumer)
        return invalid("CGRA memory internal connection repeats a consumer");
    }
  }
  for (const CgraMemoryActorPlan &actor : plan.memory.actors) {
    if (actor.graph != graph)
      continue;
    auto resolved = dataflow.resolve(actor.actor);
    if (!resolved)
      return resolved.takeError();
    if (resolved->graph != graph)
      return invalid("CGRA memory actor belongs to another graph");
    auto semanticPosition = semanticOrdinals.find(resolved->op);
    if (semanticPosition == semanticOrdinals.end())
      return invalid("CGRA memory actor is absent from graph execution");
    const ActorExecutionPlan &semantic =
        execution.actorPlans[semanticPosition->second];
    if (!semantic.memory)
      return unsupported(
          "CGRA memory execution currently requires load/store semantics");
    if (bindingBySemanticActor[semanticPosition->second] != invalidBinding)
      return invalid("CGRA memory actor has duplicate runtime bindings");
    if (actor.operationPhysicalUseOrdinal >= plan.physicalUseClients.size() ||
        plan.physicalUseClients[actor.operationPhysicalUseOrdinal] !=
            CgraPhysicalUseClientKind::MemoryTransition)
      return invalid("CGRA memory operation action is not selected");
    if (actor.rootedUseOffset > plan.memory.rootedUses.size() ||
        actor.rootedUseCount >
            plan.memory.rootedUses.size() - actor.rootedUseOffset)
      return invalid("CGRA memory rooted-use slice is malformed");
    if (actor.childTransactionOffset > plan.memory.childTransactions.size() ||
        actor.childTransactionCount >
            plan.memory.childTransactions.size() - actor.childTransactionOffset)
      return invalid("CGRA memory child-transaction slice is malformed");
    if (actor.resultAssemblyOffset > plan.memory.resultAssemblies.size() ||
        actor.resultAssemblyCount >
            plan.memory.resultAssemblies.size() - actor.resultAssemblyOffset)
      return invalid("CGRA memory result-assembly slice is malformed");
    for (const CgraMemoryChildTransactionPlan &child :
         llvm::ArrayRef(plan.memory.childTransactions)
             .slice(actor.childTransactionOffset, actor.childTransactionCount))
      if (llvm::Error error = validateChild(child))
        return std::move(error);

    const CgraMemoryRootedUsePlan *selectedUse = nullptr;
    for (const CgraMemoryRootedUsePlan &use :
         llvm::ArrayRef(plan.memory.rootedUses)
             .slice(actor.rootedUseOffset, actor.rootedUseCount)) {
      if (use.launch != launch)
        continue;
      if (selectedUse)
        return invalid("CGRA memory actor repeats one rooted use");
      selectedUse = &use;
    }
    if (!selectedUse)
      return invalid("CGRA memory actor has no exact rooted use");
    if (!std::holds_alternative<::loom::fabric::LocalMemoryServiceRef>(
            selectedUse->target))
      return unsupported(
          "CGRA memory service target has no registered execution provider");
    if (!selectedUse->localServicePhysicalUseOrdinal ||
        *selectedUse->localServicePhysicalUseOrdinal >=
            plan.physicalUseClients.size() ||
        plan.physicalUseClients[*selectedUse->localServicePhysicalUseOrdinal] !=
            CgraPhysicalUseClientKind::MemoryTransition)
      return invalid("CGRA local memory service action is not selected");

    auto service =
        ::dataflow::semantics::CanonicalService::forActor(resolved->op);
    if (!service)
      return service.takeError();
    if (actor.roleSources.size() != memoryRoleCount ||
        actor.roleDestinations.size() != memoryRoleCount)
      return invalid("CGRA memory activation role domain is incomplete");
    std::vector<bool> activeSources(memoryRoleCount, false);
    std::vector<bool> activeDestinations(memoryRoleCount, false);
    for (auto [argumentOrdinal, argument] :
         llvm::enumerate(service->arguments())) {
      auto value = service->argumentValue(resolved->op, argumentOrdinal);
      if (!value)
        return value.takeError();
      const ::dataflow::ActorTokenOperandRef consumer{
          actor.actor, static_cast<::dataflow::StructuralOrdinal>(
                           (*value)->getOperandNumber())};
      llvm::SmallVector<std::uint64_t, 2> connections;
      for (auto [connectionOrdinal, connection] :
           llvm::enumerate(plan.memory.internalConnections))
        if (connection.occurrence == actor.occurrence &&
            connection.consumer == consumer)
          connections.push_back(connectionOrdinal);
      const std::size_t role = static_cast<std::size_t>(argument.role);
      if (role >= actor.roleSources.size() || !actor.roleSources[role])
        return invalid("CGRA memory input role activation is incomplete");
      if (activeSources[role])
        return invalid("CGRA memory activation repeats an input role");
      activeSources[role] = true;
      const auto *internal = std::get_if<
          ::loom::fabric::FabricMemoryHandshakeInternalRoleSource>(
          &*actor.roleSources[role]);
      if (internal) {
        if (connections.size() != 1 ||
            plan.memory.internalConnections[connections.front()].connection !=
                internal->connection)
          return invalid("CGRA memory input role selects another internal "
                         "connection");
        internalConsumerValidated[connections.front()] = true;
        closureUses.push_back(
            {actor.occurrence, internal->connection,
             ::loom::fabric::FabricMemoryInternalConnectionUseKind::Consumer});
      } else if (!connections.empty()) {
        return invalid("CGRA memory input role externalizes an internal "
                       "connection");
      }
    }
    const auto serviceResults = service->results();
    if (serviceResults.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA memory result count exceeds u32");
    std::vector<ResultBinding> results;
    results.reserve(serviceResults.size());
    std::uint64_t assemblyLocal = 0;
    for (auto [resultOrdinal, result] : llvm::enumerate(serviceResults)) {
      auto value = service->resultValue(resolved->op, resultOrdinal);
      if (!value)
        return value.takeError();
      const ::dataflow::ActorTokenResultRef producer{
          actor.actor, static_cast<::dataflow::StructuralOrdinal>(
                           value->getResultNumber())};
      llvm::SmallVector<std::uint64_t, 2> connectionRows;
      std::vector<::loom::fabric::FabricOrdinal> connectionOrdinals;
      for (auto [connectionOrdinal, connection] :
           llvm::enumerate(plan.memory.internalConnections)) {
        if (connection.occurrence != actor.occurrence ||
            connection.producer != producer)
          continue;
        connectionRows.push_back(connectionOrdinal);
        connectionOrdinals.push_back(connection.connection);
      }
      llvm::sort(connectionOrdinals);
      connectionOrdinals.erase(
          std::unique(connectionOrdinals.begin(), connectionOrdinals.end()),
          connectionOrdinals.end());
      const std::size_t role = static_cast<std::size_t>(result.role);
      if (role >= actor.roleDestinations.size() ||
          !actor.roleDestinations[role] ||
          actor.roleDestinations[role]->internalConnections !=
              connectionOrdinals)
        return invalid("CGRA memory output role selects another internal "
                       "connection set");
      for (::loom::fabric::FabricOrdinal connection : connectionOrdinals)
        closureUses.push_back(
            {actor.occurrence, connection,
             ::loom::fabric::FabricMemoryInternalConnectionUseKind::Producer});
      if (activeDestinations[role])
        return invalid("CGRA memory activation repeats an output role");
      activeDestinations[role] = true;
      for (std::uint64_t connectionOrdinal : connectionRows)
        internalProducerValidated[connectionOrdinal] = true;
      std::optional<std::uint64_t> assemblyOrdinal;
      if (result.role != ::dataflow::semantics::ServiceValueRole::Completion) {
        if (assemblyLocal == actor.resultAssemblyCount)
          return invalid("CGRA memory result assembly is incomplete");
        assemblyOrdinal = actor.resultAssemblyOffset + assemblyLocal++;
        if (plan.memory.resultAssemblies[*assemblyOrdinal].role != result.role)
          return invalid("CGRA memory result assembly has the wrong role");
      }
      results.push_back(
          {result.role, value->getResultNumber(), assemblyOrdinal});
    }
    if (assemblyLocal != actor.resultAssemblyCount)
      return invalid("CGRA memory result assembly has unused rows");
    for (std::size_t role = 0; role != memoryRoleCount; ++role) {
      if (!activeSources[role] && actor.roleSources[role])
        return invalid("CGRA memory activation has an inactive input role");
      if (!activeDestinations[role] && actor.roleDestinations[role])
        return invalid("CGRA memory activation has an inactive output role");
    }

    const std::uint64_t bindingOrdinal = bindings.size();
    bindingBySemanticActor[semanticPosition->second] = bindingOrdinal;
    bindings.push_back({semanticPosition->second, &semantic, &actor,
                        selectedUse, std::move(results), 0, false, 0});
  }

  for (auto [ordinal, connection] :
       llvm::enumerate(plan.memory.internalConnections)) {
    auto producer = dataflow.resolve(connection.producer.actor);
    if (!producer)
      return producer.takeError();
    auto consumer = dataflow.resolve(connection.consumer.actor);
    if (!consumer)
      return consumer.takeError();
    if (producer->graph != consumer->graph)
      return invalid("CGRA memory internal connection spans graphs");
    if (producer->graph == graph &&
        (!internalProducerValidated[ordinal] ||
         !internalConsumerValidated[ordinal]))
      return invalid("CGRA memory internal connection activation is open");
  }
  switch (
      ::loom::fabric::deriveFabricMemoryInternalConnectionClosure(closureUses)) {
  case ::loom::fabric::FabricMemoryInternalConnectionClosure::Closed:
    break;
  case ::loom::fabric::FabricMemoryInternalConnectionClosure::Open:
    return invalid("CGRA memory internal connection activation is open");
  case ::loom::fabric::FabricMemoryInternalConnectionClosure::
      MultipleProducers:
    return invalid("CGRA memory internal connection has multiple producers");
  }

  for (auto [ordinal, actor] : llvm::enumerate(execution.actorPlans))
    if (actor.isPlainMemory() &&
        bindingBySemanticActor[ordinal] == invalidBinding)
      return invalid("CGRA memory execution lacks a selected actor binding");

  return CgraMemoryRuntime(plan, state, std::move(bindings),
                           std::move(bindingBySemanticActor), physical);
}

bool CgraMemoryRuntime::ownsActor(std::uint64_t semanticActorOrdinal) const {
  return semanticActorOrdinal < bindingBySemanticActor_.size() &&
         bindingBySemanticActor_[semanticActorOrdinal] != invalidBinding;
}

llvm::Expected<std::uint64_t>
CgraMemoryRuntime::allocateFiring(std::uint64_t bindingOrdinal,
                                  ReadyPlainMemoryAction ready,
                                  std::optional<Token> storeData) {
  ActorBinding &binding = bindings_[bindingOrdinal];
  if (binding.nextOccurrenceOrdinal ==
      std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(
        std::errc::value_too_large,
        "CGRA memory actor occurrence ordinal overflows u64");
  auto childCount =
      activeChildCount(llvm::ArrayRef(plan_->memory.childTransactions)
                           .slice(binding.physical->childTransactionOffset,
                                  binding.physical->childTransactionCount),
                       ready.activeLanes);
  if (!childCount)
    return childCount.takeError();
  std::uint64_t slot = 0;
  if (freeFiringSlots_.empty()) {
    slot = firings_.size();
    firings_.emplace_back();
  } else {
    slot = freeFiringSlots_.back();
    freeFiringSlots_.pop_back();
  }
  Firing &firing = firings_[slot];
  firing = Firing{};
  firing.active = true;
  firing.bindingOrdinal = bindingOrdinal;
  firing.actorOccurrenceOrdinal = binding.nextOccurrenceOrdinal++;
  firing.ready.emplace(std::move(ready));
  firing.storeData = std::move(storeData);
  firing.activeChildCount = *childCount;
  binding.retirementPending = true;
  binding.activeOccurrenceOrdinal = firing.actorOccurrenceOrdinal;
  ++activeActorCount_;
  state_->plainMemoryCandidates.reset(binding.semanticActorOrdinal);
  return slot;
}

llvm::Expected<CgraPhysicalLifecycleEvent> CgraMemoryRuntime::requestAction(
    std::uint64_t firingSlot, std::uint64_t actionOrdinal,
    std::uint64_t localActionOrdinal, bool operation,
    const SpatialEventCoordinate &coordinate) {
  if (actionOrdinal >= nextActionOccurrence_.size())
    return invalid("CGRA memory request names an unknown physical action");
  if (nextActionOccurrence_[actionOrdinal] ==
      std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(
        std::errc::value_too_large,
        "CGRA memory action occurrence ordinal overflows u64");
  const std::uint64_t occurrence = nextActionOccurrence_[actionOrdinal];
  auto requested = physical_->request(actionOrdinal, occurrence, coordinate);
  if (!requested)
    return requested.takeError();
  ++nextActionOccurrence_[actionOrdinal];
  if (!actionToFiring_
           .try_emplace({actionOrdinal, occurrence},
                        ActionIndex{firingSlot, localActionOrdinal, operation})
           .second)
    return invalid("CGRA memory action occurrence is duplicated");
  return std::move(*requested);
}

llvm::Error
CgraMemoryRuntime::scheduleReady(SpatialEventCoordinate coordinate) {
  for (const ActorBinding &binding : bindings_)
    if (binding.retirementPending)
      state_->plainMemoryCandidates.reset(binding.semanticActorOrdinal);
  if (!state_->plainMemoryCandidates.any())
    return llvm::Error::success();
  if (!admitReadyPlainMemoryActions(*state_))
    return executionFailure(*state_, "CGRA memory action admission failed");

  for (std::uint64_t bindingOrdinal = 0; bindingOrdinal != bindings_.size();
       ++bindingOrdinal) {
    ActorBinding &binding = bindings_[bindingOrdinal];
    if (binding.retirementPending)
      continue;
    auto admitted =
        state_->admittedPlainMemoryActions.find(binding.semantic->operation);
    if (admitted == state_->admittedPlainMemoryActions.end())
      continue;
    std::optional<Token> storeData;
    if (binding.semantic->memory->dataOperandOrdinal) {
      state_->currentActorPlan = binding.semantic;
      llvm::scope_exit resetPlan([&] { state_->currentActorPlan = nullptr; });
      storeData = peekInputToken(*state_,
                                 *binding.semantic->memory->dataOperandOrdinal);
    }
    ReadyPlainMemoryAction ready = std::move(admitted->second);
    state_->admittedPlainMemoryActions.erase(admitted);
    auto firingSlot =
        allocateFiring(bindingOrdinal, std::move(ready), std::move(storeData));
    if (!firingSlot)
      return firingSlot.takeError();
    auto requested = requestAction(
        *firingSlot, binding.physical->operationPhysicalUseOrdinal,
        /*localActionOrdinal=*/0, /*operation=*/true, coordinate);
    if (!requested)
      return requested.takeError();
    requestedEvents_.schedule(
        {{requested->coordinate, requested->actionOrdinal,
          requested->occurrenceOrdinal, requested->ownerEventOrdinal},
         0});
  }
  if (!state_->admittedPlainMemoryActions.empty())
    return invalid("CGRA memory admission retained an unmapped actor");
  return llvm::Error::success();
}

llvm::Error CgraMemoryRuntime::start(SpatialEventCoordinate coordinate) {
  if (started_)
    return invalid("CGRA memory runtime was already started");
  started_ = true;
  return scheduleReady(std::move(coordinate));
}

llvm::Error CgraMemoryRuntime::acceptReadyCandidates(
    SpatialEventCoordinate coordinate,
    const llvm::SmallBitVector &semanticCandidates) {
  if (!started_)
    return invalid("CGRA memory runtime has not started");
  if (semanticCandidates.size() != bindingBySemanticActor_.size())
    return invalid("CGRA memory candidate domain disagrees with the graph");
  for (int ordinal = semanticCandidates.find_first(); ordinal >= 0;
       ordinal = semanticCandidates.find_next(ordinal))
    if (ownsActor(ordinal) &&
        !bindings_[bindingBySemanticActor_[ordinal]].retirementPending)
      state_->plainMemoryCandidates.set(ordinal);
  return scheduleReady(std::move(coordinate));
}

llvm::Error
CgraMemoryRuntime::commitIssue(std::uint64_t firingSlot,
                               const SpatialEventCoordinate &coordinate,
                               CgraMemoryLifecycleFrame &frame) {
  Firing &firing = firings_[firingSlot];
  if (!firing.active || firing.issueCommitted || !firing.ready)
    return invalid("CGRA memory issue names an invalid firing");
  ActorBinding &binding = bindings_[firing.bindingOrdinal];
  state_->currentActorPlan = binding.semantic;
  llvm::scope_exit resetPlan([&] { state_->currentActorPlan = nullptr; });
  consumePlainMemoryIssueInputs(*firing.ready, *binding.semantic->memory,
                                *state_);
  firing.issueCommitted = true;
  frame.actorEvents.push_back(
      {CgraActorLifecycleKind::Committed, binding.semanticActorOrdinal,
       firing.actorOccurrenceOrdinal, 0,
       static_cast<std::uint32_t>(binding.results.size()), coordinate});

  const std::uint64_t serviceAction =
      *binding.rootedUse->localServicePhysicalUseOrdinal;
  for (std::uint32_t child = 0; child != firing.activeChildCount; ++child) {
    auto requested = requestAction(firingSlot, serviceAction,
                                   /*localActionOrdinal=*/1 + child,
                                   /*operation=*/false, coordinate);
    if (!requested)
      return requested.takeError();
    frame.physicalEvents.push_back(std::move(*requested));
  }
  if (firing.activeChildCount == 0)
    return linearize(firingSlot, frame);
  return llvm::Error::success();
}

llvm::Expected<CgraPhysicalTraceBinding>
CgraMemoryRuntime::physicalTraceBinding(
    const CgraPhysicalLifecycleEvent &event) const {
  if (event.actionOrdinal >= plan_->physicalUseClients.size() ||
      plan_->physicalUseClients[event.actionOrdinal] !=
          CgraPhysicalUseClientKind::MemoryTransition)
    return invalid("CGRA trace memory action has another client");
  auto indexed =
      actionToFiring_.find({event.actionOrdinal, event.occurrenceOrdinal});
  if (indexed == actionToFiring_.end())
    return invalid("CGRA trace memory action has no active firing");
  const ActionIndex index = indexed->second;
  if (index.firingSlot >= firings_.size() || !firings_[index.firingSlot].active)
    return invalid("CGRA trace memory action names an inactive firing");
  const Firing &firing = firings_[index.firingSlot];
  const ActorBinding &binding = bindings_[firing.bindingOrdinal];
  auto target = projectPhysicalUseTarget(*plan_, event.actionOrdinal);
  if (!target)
    return target.takeError();
  return CgraPhysicalTraceBinding{
      PhysicalActionOccurrenceRef{
          TransitionPhysicalActionParent{ActorTransitionOccurrenceRef{
              GraphInvocationOccurrenceRef{0}, binding.physical->actor,
              firing.actorOccurrenceOrdinal}},
          index.localActionOrdinal},
      std::move(*target)};
}

llvm::Error CgraMemoryRuntime::linearize(std::uint64_t firingSlot,
                                         CgraMemoryLifecycleFrame &frame) {
  Firing &firing = firings_[firingSlot];
  if (!firing.active || !firing.issueCommitted || firing.linearized ||
      !firing.ready)
    return invalid("CGRA memory linearization names an invalid firing");
  ActorBinding &binding = bindings_[firing.bindingOrdinal];
  state_->currentActorPlan = binding.semantic;
  llvm::scope_exit resetPlan([&] { state_->currentActorPlan = nullptr; });

  std::optional<DataflowMemoryRead> read;
  std::optional<DataflowMemoryWrite> write;
  if (binding.semantic->memory->dataOperandOrdinal) {
    if (!firing.storeData)
      return invalid("CGRA store firing lost its data token");
    write = preparePlainMemoryWrite(*firing.storeData, *firing.ready,
                                    *binding.semantic->memory, *state_);
    if (!write)
      return executionFailure(*state_, "CGRA memory write preparation failed");
  } else {
    read = preparePlainMemoryRead(*firing.ready, *binding.semantic->memory,
                                  *state_);
    if (!read)
      return executionFailure(*state_, "CGRA memory read preparation failed");
  }
  auto publication = linearizePlainMemoryAction(*firing.ready, *state_);
  if (!publication)
    return executionFailure(*state_, "CGRA memory ordering failed");
  if (write)
    commitDataflowMemoryWrite(firing.ready->view, *write);

  if (state_->actorEmissionCapture)
    return invalid("CGRA memory linearization found a nested emission capture");
  llvm::SmallVector<ActorResultEmission, 4> emissions;
  state_->actorEmissionCapture = &emissions;
  llvm::scope_exit resetCapture(
      [&] { state_->actorEmissionCapture = nullptr; });
  for (const ResultBinding &result : binding.results) {
    if (result.role == ::dataflow::semantics::ServiceValueRole::Completion) {
      emitResultTokenWithMemoryOrder(*state_, result.resultOrdinal, noneToken(),
                                     *publication);
      continue;
    }
    if (result.role != ::dataflow::semantics::ServiceValueRole::Data || !read ||
        !result.assemblyOrdinal)
      return unsupported(
          "CGRA memory result assembly has no semantic provider");
    const CgraMemoryResultAssemblyPlan &assembly =
        plan_->memory.resultAssemblies[*result.assemblyOrdinal];
    switch (assembly.strategy) {
    case ::fabric::MemoryResultAssemblyStrategy::PassThroughParent:
    case ::fabric::MemoryResultAssemblyStrategy::
        ParentResponseOrZeroOnEmptyMask:
    case ::fabric::MemoryResultAssemblyStrategy::RowMajorLaneValues:
      emitResultTokenWithMemoryOrder(*state_, result.resultOrdinal, read->data,
                                     MemoryOrderFrontierId());
      break;
    }
  }
  if (emissions.size() != binding.results.size())
    return invalid("CGRA memory result emission is incomplete");
  for (ActorResultEmission &emission : emissions)
    frame.actorEmissions.push_back(
        {binding.semanticActorOrdinal, firing.actorOccurrenceOrdinal, 0,
         emission.resultOrdinal, std::move(emission.token)});
  recordEvent(*state_, binding.semantic->projection.schema);
  if (!firing.ready->activeLanes.isZero()) {
    frame.memoryLinearizations.push_back(MemoryLinearizedTraceEvent{
        MemoryActionOccurrenceRef{
            ActorTransitionOccurrenceRef{GraphInvocationOccurrenceRef{0},
                                         binding.physical->actor,
                                         firing.actorOccurrenceOrdinal},
            ActorWideMemoryActionRef{}},
        std::nullopt, std::nullopt, std::nullopt});
  }
  firing.linearized = true;
  return llvm::Error::success();
}

void CgraMemoryRuntime::maybeComplete(std::uint64_t firingSlot,
                                      CgraMemoryLifecycleFrame &frame) {
  Firing &firing = firings_[firingSlot];
  if (!firing.active || !firing.linearized || !firing.operationRetired ||
      firing.retiredChildCount != firing.activeChildCount)
    return;
  const ActorBinding &binding = bindings_[firing.bindingOrdinal];
  frame.physicalCompletions.push_back(
      {binding.semanticActorOrdinal, firing.actorOccurrenceOrdinal, 0});
  releaseFiring(firingSlot);
}

void CgraMemoryRuntime::releaseFiring(std::uint64_t firingSlot) {
  Firing &firing = firings_[firingSlot];
  firing.ready.reset();
  firing.storeData.reset();
  firing.active = false;
  freeFiringSlots_.push_back(firingSlot);
}

llvm::Error
CgraMemoryRuntime::processPhysicalEvent(const CgraPhysicalLifecycleEvent &event,
                                        CgraMemoryLifecycleFrame &frame) {
  if (event.actionOrdinal >= plan_->physicalUseClients.size() ||
      event.actionOrdinal >= plan_->physicalUseTimings.size())
    return invalid("CGRA memory lifecycle names an unknown action");
  if (plan_->physicalUseClients[event.actionOrdinal] !=
      CgraPhysicalUseClientKind::MemoryTransition)
    return llvm::Error::success();
  auto indexed =
      actionToFiring_.find({event.actionOrdinal, event.occurrenceOrdinal});
  if (indexed == actionToFiring_.end())
    return invalid("CGRA memory lifecycle has no active firing");
  const ActionIndex index = indexed->second;
  if (index.firingSlot >= firings_.size() || !firings_[index.firingSlot].active)
    return invalid("CGRA memory lifecycle names an inactive firing");
  Firing &firing = firings_[index.firingSlot];

  if (event.kind == CgraPhysicalLifecycleKind::Requested)
    return invalid("CGRA physical runtime repeated a memory request");
  if (isPermissionEvent(plan_->physicalUseTimings[event.actionOrdinal],
                        event.kind)) {
    if (index.operation) {
      if (firing.operationPermitted)
        return invalid("CGRA memory operation was permitted twice");
      firing.operationPermitted = true;
      if (llvm::Error error =
              commitIssue(index.firingSlot, event.coordinate, frame))
        return error;
    } else {
      if (++firing.permittedChildCount > firing.activeChildCount)
        return invalid("CGRA memory child permit count exceeds its domain");
      if (firing.permittedChildCount == firing.activeChildCount)
        if (llvm::Error error = linearize(index.firingSlot, frame))
          return error;
    }
  }
  if (event.kind == CgraPhysicalLifecycleKind::Retired) {
    if (index.operation) {
      if (firing.operationRetired)
        return invalid("CGRA memory operation retired twice");
      firing.operationRetired = true;
    } else if (++firing.retiredChildCount > firing.activeChildCount) {
      return invalid("CGRA memory child retire count exceeds its domain");
    }
    actionToFiring_.erase(indexed);
  }
  maybeComplete(index.firingSlot, frame);
  return llvm::Error::success();
}

llvm::Expected<CgraMemoryLifecycleFrame>
CgraMemoryRuntime::acceptPhysicalEvents(
    const CgraPhysicalLifecycleFrame &physicalFrame) {
  if (!started_)
    return invalid("CGRA memory runtime has not started");
  CgraMemoryLifecycleFrame result{physicalFrame.coordinate, {}, {}, {}, {}, {}};
  for (const CgraPhysicalLifecycleEvent &event : physicalFrame.events) {
    if (compareSpatialEventCoordinates(event.coordinate,
                                       physicalFrame.coordinate) != 0)
      return invalid("CGRA physical frame contains another coordinate");
    if (llvm::Error error = processPhysicalEvent(event, result))
      return std::move(error);
  }
  return result;
}

llvm::Error CgraMemoryRuntime::retireActor(std::uint64_t semanticActorOrdinal,
                                           std::uint64_t occurrenceOrdinal,
                                           SpatialEventCoordinate coordinate) {
  if (!started_)
    return invalid("CGRA memory runtime has not started");
  if (!ownsActor(semanticActorOrdinal))
    return invalid("CGRA memory retirement names an unknown semantic actor");
  ActorBinding &binding =
      bindings_[bindingBySemanticActor_[semanticActorOrdinal]];
  if (!binding.retirementPending ||
      binding.activeOccurrenceOrdinal != occurrenceOrdinal)
    return invalid("CGRA memory retirement disagrees with its active firing");
  binding.retirementPending = false;
  if (activeActorCount_ == 0)
    return invalid("CGRA active memory actor count underflow");
  --activeActorCount_;
  state_->plainMemoryCandidates.set(semanticActorOrdinal);
  auto next = nextSpatialDelta(coordinate);
  if (!next)
    return next.takeError();
  return scheduleReady(std::move(*next));
}

llvm::Expected<std::optional<CgraMemoryLifecycleFrame>>
CgraMemoryRuntime::advance() {
  if (!started_)
    return invalid("CGRA memory runtime has not started");
  auto requested = requestedEvents_.popNextFrameView();
  if (!requested)
    return requested.takeError();
  if (!*requested)
    return std::optional<CgraMemoryLifecycleFrame>{};
  CgraMemoryLifecycleFrame frame{(**requested).coordinate, {}, {}, {}, {}, {}};
  for (const CgraScheduledEvent &event : (**requested).events)
    frame.physicalEvents.push_back(
        {CgraPhysicalLifecycleKind::Requested,
         event.order.structuralActionOrdinal, event.order.occurrenceOrdinal,
         event.order.ownerEventOrdinal, event.order.coordinate});
  return std::optional<CgraMemoryLifecycleFrame>(std::move(frame));
}

} // namespace loom::sim::detail

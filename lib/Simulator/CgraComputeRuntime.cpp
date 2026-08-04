#include "CgraComputeRuntime.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

void selectEarlier(std::optional<SpatialEventCoordinate> candidate,
                   std::optional<SpatialEventCoordinate> &selected) {
  if (candidate &&
      (!selected || compareSpatialEventCoordinates(*candidate, *selected) < 0))
    selected = std::move(candidate);
}

bool isAt(const std::optional<SpatialEventCoordinate> &candidate,
          const SpatialEventCoordinate &coordinate) {
  return candidate &&
         compareSpatialEventCoordinates(*candidate, coordinate) == 0;
}

} // namespace

CgraComputeRuntime::CgraComputeRuntime(
    const CgraFrozenExecutionPlan &plan, SimulatorState &state,
    std::vector<ActorBinding> bindings,
    std::vector<std::uint64_t> transitionByCase,
    std::vector<std::uint64_t> bindingBySemanticActor,
    std::vector<std::uint64_t> bindingByActorPlan,
    CgraPhysicalActionRuntime &physical)
    : plan_(&plan), state_(&state), bindings_(std::move(bindings)),
      transitionByCase_(std::move(transitionByCase)),
      bindingBySemanticActor_(std::move(bindingBySemanticActor)),
      bindingByActorPlan_(std::move(bindingByActorPlan)), physical_(&physical),
      readyCandidates_(bindings_.size(), false),
      nextActionOccurrence_(plan.physicalUseTimings.size(), 0) {}

llvm::Expected<CgraComputeRuntime> CgraComputeRuntime::create(
    const CgraFrozenExecutionPlan &plan,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::GraphRef graph, const PreparedGraphExecution &execution,
    SimulatorState &state, CgraPhysicalActionRuntime &physical) {
  if (state.execution != &execution)
    return invalid("CGRA compute state does not use the prepared graph");
  if (plan.physicalUseTimings.size() != plan.resources.selectedUses.size())
    return invalid("CGRA compute physical timing coverage is incomplete");
  if (plan.physicalUseClients.size() != plan.physicalUseTimings.size())
    return invalid("CGRA physical-use client coverage is incomplete");
  llvm::DenseMap<mlir::Operation *, std::uint64_t> semanticOrdinals;
  semanticOrdinals.reserve(execution.actorPlans.size());
  for (auto [ordinal, actor] : llvm::enumerate(execution.actorPlans))
    if (!semanticOrdinals.try_emplace(actor.operation, ordinal).second)
      return invalid("prepared graph contains a duplicate actor operation");

  std::vector<ActorBinding> bindings;
  std::vector<std::uint64_t> transitionByCase;
  std::vector<std::uint64_t> bindingBySemanticActor(
      execution.actorPlans.size(), std::numeric_limits<std::uint64_t>::max());
  std::vector<std::uint64_t> bindingByActorPlan(
      plan.computeActors.size(), std::numeric_limits<std::uint64_t>::max());
  for (auto [actorPlanOrdinal, actor] : llvm::enumerate(plan.computeActors)) {
    if (actor.graph != graph)
      continue;
    auto resolved = dataflow.resolve(actor.actor);
    if (!resolved)
      return resolved.takeError();
    if (resolved->graph != graph)
      return invalid("CGRA compute actor belongs to another mapped graph");
    auto semantic = semanticOrdinals.find(resolved->op);
    if (semantic == semanticOrdinals.end())
      return invalid("CGRA compute actor is absent from graph execution");
    const ActorExecutionPlan &semanticPlan =
        execution.actorPlans[semantic->second];
    if (semanticPlan.transitionProbe == ActorTransitionProbeKind::Unavailable)
      return llvm::createStringError(
          std::errc::not_supported,
          "CGRA compute actor has no typed transition probe");
    if (actor.transitionOffset > plan.computeTransitions.size() ||
        actor.transitionCount >
            plan.computeTransitions.size() - actor.transitionOffset)
      return invalid("CGRA compute actor transition slice is out of range");
    if (actor.transitionCount != semanticPlan.handshakeCases.size())
      return invalid("CGRA compute transition domain disagrees with schema");

    const std::uint64_t caseOffset = transitionByCase.size();
    transitionByCase.resize(caseOffset + actor.transitionCount,
                            std::numeric_limits<std::uint64_t>::max());
    for (std::uint32_t local = 0; local != actor.transitionCount; ++local) {
      const std::uint64_t transitionOrdinal = actor.transitionOffset + local;
      const CgraComputeTransitionPlan &transition =
          plan.computeTransitions[transitionOrdinal];
      if (transition.caseOrdinal >= actor.transitionCount ||
          transitionByCase[caseOffset + transition.caseOrdinal] !=
              std::numeric_limits<std::uint64_t>::max())
        return invalid("CGRA compute transition cases are not a dense domain");
      if (transition.physicalUseCount == 0 ||
          transition.physicalUseOffset >
              plan.actorTransitionPhysicalUses.size() ||
          transition.physicalUseCount >
              plan.actorTransitionPhysicalUses.size() -
                  transition.physicalUseOffset)
        return invalid("CGRA compute physical-use slice is malformed");
      for (std::uint64_t action :
           llvm::ArrayRef(plan.actorTransitionPhysicalUses)
               .slice(transition.physicalUseOffset,
                      transition.physicalUseCount))
        if (action >= plan.physicalUseTimings.size())
          return invalid("CGRA compute transition names an unknown action");
        else if (plan.physicalUseClients[action] !=
                 CgraPhysicalUseClientKind::ComputeTransition)
          return invalid("CGRA compute transition names another client action");
      transitionByCase[caseOffset + transition.caseOrdinal] = transitionOrdinal;
    }
    const std::uint64_t bindingOrdinal = bindings.size();
    if (bindingBySemanticActor[semantic->second] !=
            std::numeric_limits<std::uint64_t>::max() ||
        bindingByActorPlan[actorPlanOrdinal] !=
            std::numeric_limits<std::uint64_t>::max())
      return invalid("CGRA compute actor has duplicate runtime bindings");
    bindingBySemanticActor[semantic->second] = bindingOrdinal;
    bindingByActorPlan[actorPlanOrdinal] = bindingOrdinal;
    bindings.push_back(ActorBinding{
        static_cast<std::uint64_t>(actorPlanOrdinal), &semanticPlan, caseOffset,
        actor.transitionCount, 0, false, false, 0});
  }
  if (bindings.empty())
    return invalid("CGRA compute graph has no selected compute actor");
  return CgraComputeRuntime(plan, state, std::move(bindings),
                            std::move(transitionByCase),
                            std::move(bindingBySemanticActor),
                            std::move(bindingByActorPlan), physical);
}

llvm::Expected<std::uint64_t> CgraComputeRuntime::allocateFiring(
    std::uint64_t bindingOrdinal, const CgraComputeTransitionPlan &transition) {
  ActorBinding &binding = bindings_[bindingOrdinal];
  if (binding.nextOccurrenceOrdinal ==
      std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(
        std::errc::value_too_large,
        "CGRA actor transition occurrence ordinal overflows u64");
  if (activeActorCount_ == std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(std::errc::value_too_large,
                                   "CGRA active actor count exceeds u64");
  std::uint64_t slot = 0;
  if (freeFiringSlots_.empty()) {
    slot = firings_.size();
    firings_.emplace_back();
  } else {
    slot = freeFiringSlots_.back();
    freeFiringSlots_.pop_back();
  }
  Firing &firing = firings_[slot];
  firing.active = true;
  firing.bindingOrdinal = bindingOrdinal;
  firing.actorOccurrenceOrdinal = binding.nextOccurrenceOrdinal++;
  firing.transitionCaseOrdinal = transition.caseOrdinal;
  firing.commitScheduled = false;
  firing.committed = false;
  firing.actionCount = transition.physicalUseCount;
  firing.permittedCount = 0;
  firing.retiredCount = 0;
  binding.commitPending = true;
  binding.retirementPending = true;
  binding.activeOccurrenceOrdinal = firing.actorOccurrenceOrdinal;
  ++activeActorCount_;
  return slot;
}

llvm::Error
CgraComputeRuntime::scheduleReady(SpatialEventCoordinate coordinate) {
  for (int candidate = readyCandidates_.find_first(); candidate >= 0;
       candidate = readyCandidates_.find_next(candidate)) {
    ActorBinding &binding = bindings_[candidate];
    if (binding.retirementPending)
      continue;
    readyCandidates_.reset(candidate);
    auto selected = probeActorTransition(*binding.semantic, *state_);
    if (!selected)
      return selected.takeError();
    if (!*selected)
      continue;
    if (**selected >= binding.transitionCount)
      return invalid("CGRA probe selected an unknown transition case");
    const std::uint64_t transitionOrdinal =
        transitionByCase_[binding.transitionIndexOffset + **selected];
    const CgraComputeTransitionPlan &transition =
        plan_->computeTransitions[transitionOrdinal];
    auto firingSlot = allocateFiring(candidate, transition);
    if (!firingSlot)
      return firingSlot.takeError();

    llvm::ArrayRef<std::uint64_t> actions(plan_->actorTransitionPhysicalUses);
    actions = actions.slice(transition.physicalUseOffset,
                            transition.physicalUseCount);
    for (std::uint64_t actionOrdinal : actions) {
      if (nextActionOccurrence_[actionOrdinal] ==
          std::numeric_limits<std::uint64_t>::max())
        return llvm::createStringError(
            std::errc::value_too_large,
            "CGRA physical action occurrence ordinal overflows u64");
      const std::uint64_t occurrence = nextActionOccurrence_[actionOrdinal]++;
      auto requested =
          physical_->request(actionOrdinal, occurrence, coordinate);
      if (!requested)
        return requested.takeError();
      const auto key = std::make_pair(actionOrdinal, occurrence);
      if (!actionToFiring_.try_emplace(key, FiringActionIndex{*firingSlot})
               .second)
        return invalid("CGRA physical action occurrence is duplicated");
      requestedEvents_.schedule(CgraScheduledEvent{
          {requested->coordinate, requested->actionOrdinal,
           requested->occurrenceOrdinal, requested->ownerEventOrdinal},
          0});
    }
  }
  return llvm::Error::success();
}

llvm::Error CgraComputeRuntime::start(SpatialEventCoordinate coordinate) {
  if (started_)
    return invalid("CGRA compute runtime was already started");
  started_ = true;
  readyCandidates_.set();
  return scheduleReady(std::move(coordinate));
}

llvm::Error CgraComputeRuntime::acceptReadyCandidates(
    SpatialEventCoordinate coordinate,
    const llvm::SmallBitVector &semanticCandidates) {
  if (!started_)
    return invalid("CGRA compute runtime has not started");
  if (semanticCandidates.size() != bindingBySemanticActor_.size())
    return invalid(
        "CGRA ready-candidate domain disagrees with graph execution");
  for (int semantic = semanticCandidates.find_first(); semantic >= 0;
       semantic = semanticCandidates.find_next(semantic)) {
    const std::uint64_t binding = bindingBySemanticActor_[semantic];
    if (binding != std::numeric_limits<std::uint64_t>::max())
      readyCandidates_.set(binding);
  }
  return scheduleReady(std::move(coordinate));
}

llvm::Error CgraComputeRuntime::maybeScheduleCommit(
    std::uint64_t firingSlot, const SpatialEventCoordinate &coordinate) {
  Firing &firing = firings_[firingSlot];
  if (!firing.active || firing.commitScheduled ||
      firing.permittedCount != firing.actionCount)
    return llvm::Error::success();
  auto commitCoordinate = nextSpatialDelta(coordinate);
  if (!commitCoordinate)
    return commitCoordinate.takeError();
  const ActorBinding &binding = bindings_[firing.bindingOrdinal];
  actorCommitEvents_.schedule(CgraScheduledEvent{
      {*commitCoordinate, binding.actorPlanOrdinal,
       firing.actorOccurrenceOrdinal, firing.transitionCaseOrdinal},
      firingSlot});
  firing.commitScheduled = true;
  return llvm::Error::success();
}

llvm::Error CgraComputeRuntime::processPhysicalEvent(
    const CgraPhysicalLifecycleEvent &event, CgraComputeLifecycleFrame &frame,
    llvm::SmallVectorImpl<std::uint64_t> &affectedFirings) {
  if (event.kind == CgraPhysicalLifecycleKind::Requested)
    return invalid("CGRA physical runtime repeated a request event");
  if (event.actionOrdinal >= plan_->physicalUseClients.size() ||
      event.actionOrdinal >= plan_->physicalUseTimings.size())
    return invalid("CGRA physical lifecycle names an unknown action");
  if (plan_->physicalUseClients[event.actionOrdinal] !=
      CgraPhysicalUseClientKind::ComputeTransition) {
    return llvm::Error::success();
  }
  auto indexed =
      actionToFiring_.find({event.actionOrdinal, event.occurrenceOrdinal});
  if (indexed == actionToFiring_.end())
    return invalid("CGRA physical lifecycle has no actor transition owner");
  const FiringActionIndex index = indexed->second;
  if (index.firingSlot >= firings_.size() || !firings_[index.firingSlot].active)
    return invalid("CGRA physical lifecycle names an inactive firing");
  Firing &firing = firings_[index.firingSlot];
  const bool requiresOwnerCommit =
      plan_->physicalUseTimings[event.actionOrdinal].commitRank.has_value();
  switch (event.kind) {
  case CgraPhysicalLifecycleKind::Requested:
    llvm_unreachable("request lifecycle rejected above");
  case CgraPhysicalLifecycleKind::Granted:
    if (!requiresOwnerCommit) {
      if (++firing.permittedCount > firing.actionCount)
        return invalid("CGRA physical permit count exceeds its action count");
    }
    break;
  case CgraPhysicalLifecycleKind::Committed:
    if (!requiresOwnerCommit || ++firing.permittedCount > firing.actionCount)
      return invalid("CGRA physical owner transition is inconsistent");
    break;
  case CgraPhysicalLifecycleKind::Retired:
    if (++firing.retiredCount > firing.actionCount)
      return invalid("CGRA physical retire count exceeds its action count");
    actionToFiring_.erase(indexed);
    break;
  }
  affectedFirings.push_back(index.firingSlot);
  frame.physicalEvents.push_back(event);
  return llvm::Error::success();
}

llvm::Error
CgraComputeRuntime::processActorCommit(std::uint64_t firingSlot,
                                       CgraComputeLifecycleFrame &frame) {
  if (firingSlot >= firings_.size() || !firings_[firingSlot].active)
    return invalid("CGRA actor commit names an inactive firing");
  Firing &firing = firings_[firingSlot];
  ActorBinding &binding = bindings_[firing.bindingOrdinal];
  auto selected = probeActorTransition(*binding.semantic, *state_);
  if (!selected)
    return selected.takeError();
  if (!*selected || **selected != firing.transitionCaseOrdinal)
    return invalid("CGRA actor readiness changed after physical reservation");
  if (state_->actorEmissionCapture)
    return invalid("CGRA actor commit found a nested emission capture");
  llvm::SmallVector<ActorResultEmission, 4> emissions;
  state_->actorEmissionCapture = &emissions;
  const ActorTransitionCommitOutcome outcome =
      commitActorTransition(*binding.semantic, *state_);
  state_->actorEmissionCapture = nullptr;
  if (outcome == ActorTransitionCommitOutcome::NotReady)
    return invalid("CGRA actor became blocked after physical reservation");
  if (outcome == ActorTransitionCommitOutcome::Failed)
    return invalid("CGRA actor provider failed after physical reservation");
  for (ActorResultEmission &emission : emissions)
    frame.actorEmissions.push_back(
        {binding.actorPlanOrdinal, firing.actorOccurrenceOrdinal,
         firing.transitionCaseOrdinal, emission.resultOrdinal,
         std::move(emission.token)});
  firing.committed = true;
  binding.commitPending = false;
  frame.actorEvents.push_back({CgraComputeActorLifecycleKind::Committed,
                               binding.actorPlanOrdinal,
                               firing.actorOccurrenceOrdinal,
                               firing.transitionCaseOrdinal, frame.coordinate});
  return llvm::Error::success();
}

void CgraComputeRuntime::releaseFiring(std::uint64_t firingSlot) {
  Firing &firing = firings_[firingSlot];
  firing.active = false;
  freeFiringSlots_.push_back(firingSlot);
}

llvm::Error CgraComputeRuntime::retireActor(std::uint64_t actorPlanOrdinal,
                                            std::uint64_t occurrenceOrdinal,
                                            SpatialEventCoordinate coordinate) {
  if (!started_)
    return invalid("CGRA compute runtime has not started");
  if (actorPlanOrdinal >= bindingByActorPlan_.size())
    return invalid("CGRA actor retirement names an unknown actor plan");
  const std::uint64_t bindingOrdinal = bindingByActorPlan_[actorPlanOrdinal];
  if (bindingOrdinal == std::numeric_limits<std::uint64_t>::max())
    return invalid("CGRA actor retirement names another graph");
  ActorBinding &binding = bindings_[bindingOrdinal];
  if (!binding.retirementPending || binding.commitPending ||
      binding.activeOccurrenceOrdinal != occurrenceOrdinal)
    return invalid("CGRA actor retirement disagrees with its active firing");
  binding.retirementPending = false;
  if (activeActorCount_ == 0)
    return invalid("CGRA active actor count underflow");
  --activeActorCount_;
  readyCandidates_.set(bindingOrdinal);
  auto next = nextSpatialDelta(coordinate);
  if (!next)
    return next.takeError();
  return scheduleReady(std::move(*next));
}

void CgraComputeRuntime::maybeComplete(std::uint64_t firingSlot,
                                       CgraComputeLifecycleFrame &frame) {
  Firing &firing = firings_[firingSlot];
  if (!firing.active || !firing.committed ||
      firing.retiredCount != firing.actionCount)
    return;
  const ActorBinding &binding = bindings_[firing.bindingOrdinal];
  frame.physicalCompletions.push_back({binding.actorPlanOrdinal,
                                       firing.actorOccurrenceOrdinal,
                                       firing.transitionCaseOrdinal});
  releaseFiring(firingSlot);
}

llvm::Expected<std::optional<CgraComputeLifecycleFrame>>
CgraComputeRuntime::advance() {
  if (!started_)
    return invalid("CGRA compute runtime has not started");
  std::optional<SpatialEventCoordinate> coordinate;
  selectEarlier(requestedEvents_.nextCoordinate(), coordinate);
  selectEarlier(actorCommitEvents_.nextCoordinate(), coordinate);
  if (!coordinate)
    return std::optional<CgraComputeLifecycleFrame>{};

  CgraComputeLifecycleFrame frame{*coordinate, {}, {}, {}, {}};
  if (isAt(requestedEvents_.nextCoordinate(), *coordinate)) {
    auto requested = requestedEvents_.popNextFrame();
    if (!requested)
      return requested.takeError();
    for (const CgraScheduledEvent &event : (**requested).events)
      frame.physicalEvents.push_back(
          {CgraPhysicalLifecycleKind::Requested,
           event.order.structuralActionOrdinal, event.order.occurrenceOrdinal,
           event.order.ownerEventOrdinal, event.order.coordinate});
  }

  llvm::SmallVector<std::uint64_t, 4> committedFirings;
  if (isAt(actorCommitEvents_.nextCoordinate(), *coordinate)) {
    auto commits = actorCommitEvents_.popNextFrame();
    if (!commits)
      return commits.takeError();
    for (const CgraScheduledEvent &event : (**commits).events) {
      if (llvm::Error error = processActorCommit(event.payload, frame))
        return std::move(error);
      committedFirings.push_back(event.payload);
    }
  }

  for (std::uint64_t firing : committedFirings)
    if (firing < firings_.size() && firings_[firing].active)
      maybeComplete(firing, frame);

  llvm::sort(frame.physicalEvents, [](const CgraPhysicalLifecycleEvent &lhs,
                                      const CgraPhysicalLifecycleEvent &rhs) {
    return std::tie(lhs.actionOrdinal, lhs.occurrenceOrdinal, lhs.kind,
                    lhs.ownerEventOrdinal) <
           std::tie(rhs.actionOrdinal, rhs.occurrenceOrdinal, rhs.kind,
                    rhs.ownerEventOrdinal);
  });
  llvm::sort(frame.actorEvents, [](const CgraComputeActorLifecycleEvent &lhs,
                                   const CgraComputeActorLifecycleEvent &rhs) {
    return std::tie(lhs.actorPlanOrdinal, lhs.occurrenceOrdinal,
                    lhs.transitionCaseOrdinal) <
           std::tie(rhs.actorPlanOrdinal, rhs.occurrenceOrdinal,
                    rhs.transitionCaseOrdinal);
  });
  llvm::sort(frame.actorEmissions, [](const CgraComputeActorEmission &lhs,
                                      const CgraComputeActorEmission &rhs) {
    return std::tie(lhs.actorPlanOrdinal, lhs.occurrenceOrdinal,
                    lhs.transitionCaseOrdinal, lhs.resultOrdinal) <
           std::tie(rhs.actorPlanOrdinal, rhs.occurrenceOrdinal,
                    rhs.transitionCaseOrdinal, rhs.resultOrdinal);
  });
  llvm::sort(frame.physicalCompletions,
             [](const CgraTransitionPhysicalCompletion &lhs,
                const CgraTransitionPhysicalCompletion &rhs) {
               return std::tie(lhs.actorPlanOrdinal, lhs.occurrenceOrdinal,
                               lhs.transitionCaseOrdinal) <
                      std::tie(rhs.actorPlanOrdinal, rhs.occurrenceOrdinal,
                               rhs.transitionCaseOrdinal);
             });
  return std::optional<CgraComputeLifecycleFrame>(std::move(frame));
}

llvm::Expected<CgraComputeLifecycleFrame>
CgraComputeRuntime::acceptPhysicalEvents(
    const CgraPhysicalLifecycleFrame &physicalFrame) {
  if (!started_)
    return invalid("CGRA compute runtime has not started");
  CgraComputeLifecycleFrame frame{physicalFrame.coordinate, {}, {}, {}, {}};
  llvm::SmallVector<std::uint64_t, 8> affectedFirings;
  for (const CgraPhysicalLifecycleEvent &event : physicalFrame.events) {
    if (compareSpatialEventCoordinates(event.coordinate,
                                       physicalFrame.coordinate) != 0)
      return invalid("CGRA physical frame contains another coordinate");
    if (llvm::Error error = processPhysicalEvent(event, frame, affectedFirings))
      return std::move(error);
  }
  llvm::sort(affectedFirings);
  affectedFirings.erase(
      std::unique(affectedFirings.begin(), affectedFirings.end()),
      affectedFirings.end());
  for (std::uint64_t firing : affectedFirings) {
    if (!firings_[firing].active)
      continue;
    if (llvm::Error error =
            maybeScheduleCommit(firing, physicalFrame.coordinate))
      return std::move(error);
    maybeComplete(firing, frame);
  }
  return frame;
}

std::optional<SpatialEventCoordinate>
CgraComputeRuntime::nextCoordinate() const {
  std::optional<SpatialEventCoordinate> coordinate;
  selectEarlier(requestedEvents_.nextCoordinate(), coordinate);
  selectEarlier(actorCommitEvents_.nextCoordinate(), coordinate);
  return coordinate;
}

bool CgraComputeRuntime::hasPendingEvents() const {
  return !requestedEvents_.empty() || !actorCommitEvents_.empty();
}

} // namespace loom::sim::detail

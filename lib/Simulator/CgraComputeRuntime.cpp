#include "CgraComputeRuntime.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <system_error>
#include <tuple>
#include <utility>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

llvm::Expected<SpatialEventCoordinate>
nextDelta(const SpatialEventCoordinate &coordinate) {
  if (coordinate.delta == std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(std::errc::value_too_large,
                                   "CGRA delta cycle overflows u64");
  return SpatialEventCoordinate{coordinate.referenceCycle,
                                coordinate.delta + 1};
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
    CgraPhysicalActionRuntime physical)
    : plan_(&plan), state_(&state), bindings_(std::move(bindings)),
      transitionByCase_(std::move(transitionByCase)),
      physical_(std::move(physical)), readyCandidates_(bindings_.size(), true),
      nextActionOccurrence_(plan.physicalUseTimings.size(), 0) {}

llvm::Expected<CgraComputeRuntime> CgraComputeRuntime::create(
    const CgraFrozenExecutionPlan &plan,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::GraphRef graph, const PreparedGraphExecution &execution,
    SimulatorState &state) {
  if (state.execution != &execution)
    return invalid("CGRA compute state does not use the prepared graph");
  if (plan.physicalUseTimings.size() != plan.resources.selectedUses.size())
    return invalid("CGRA compute physical timing coverage is incomplete");
  auto physical = CgraPhysicalActionRuntime::create(plan.resources,
                                                    plan.physicalUseTimings);
  if (!physical)
    return physical.takeError();

  llvm::DenseMap<mlir::Operation *, std::uint64_t> semanticOrdinals;
  semanticOrdinals.reserve(execution.actorPlans.size());
  for (auto [ordinal, actor] : llvm::enumerate(execution.actorPlans))
    if (!semanticOrdinals.try_emplace(actor.operation, ordinal).second)
      return invalid("prepared graph contains a duplicate actor operation");

  std::vector<ActorBinding> bindings;
  std::vector<std::uint64_t> transitionByCase;
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
      transitionByCase[caseOffset + transition.caseOrdinal] = transitionOrdinal;
    }
    bindings.push_back(ActorBinding{
        static_cast<std::uint64_t>(actorPlanOrdinal), &semanticPlan, caseOffset,
        actor.transitionCount, 0, false});
  }
  if (bindings.empty())
    return invalid("CGRA compute graph has no selected compute actor");
  return CgraComputeRuntime(plan, state, std::move(bindings),
                            std::move(transitionByCase), std::move(*physical));
}

llvm::Expected<std::uint64_t> CgraComputeRuntime::allocateFiring(
    std::uint64_t bindingOrdinal, const CgraComputeTransitionPlan &transition) {
  ActorBinding &binding = bindings_[bindingOrdinal];
  if (binding.nextOccurrenceOrdinal ==
      std::numeric_limits<std::uint64_t>::max())
    return llvm::createStringError(
        std::errc::value_too_large,
        "CGRA actor transition occurrence ordinal overflows u64");
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
  return slot;
}

llvm::Error
CgraComputeRuntime::scheduleReady(SpatialEventCoordinate coordinate) {
  for (int candidate = readyCandidates_.find_first(); candidate >= 0;
       candidate = readyCandidates_.find_next(candidate)) {
    readyCandidates_.reset(candidate);
    ActorBinding &binding = bindings_[candidate];
    if (binding.commitPending)
      return invalid("CGRA compute actor has two uncommitted transitions");
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
      auto requested = physical_.request(actionOrdinal, occurrence, coordinate);
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

llvm::Expected<std::optional<CgraComputeLifecycleFrame>>
CgraComputeRuntime::start(SpatialEventCoordinate coordinate) {
  if (started_)
    return invalid("CGRA compute runtime was already started");
  started_ = true;
  if (llvm::Error error = scheduleReady(std::move(coordinate)))
    return std::move(error);
  return advance();
}

llvm::Error CgraComputeRuntime::maybeScheduleCommit(
    std::uint64_t firingSlot, const SpatialEventCoordinate &coordinate) {
  Firing &firing = firings_[firingSlot];
  if (!firing.active || firing.commitScheduled ||
      firing.permittedCount != firing.actionCount)
    return llvm::Error::success();
  auto commitCoordinate = nextDelta(coordinate);
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
  const ActorTransitionCommitOutcome outcome =
      commitActorTransition(*binding.semantic, *state_);
  if (outcome == ActorTransitionCommitOutcome::NotReady)
    return invalid("CGRA actor became blocked after physical reservation");
  if (outcome == ActorTransitionCommitOutcome::Failed)
    return invalid("CGRA actor provider failed after physical reservation");
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
  selectEarlier(physical_.nextCoordinate(), coordinate);
  selectEarlier(actorCommitEvents_.nextCoordinate(), coordinate);
  if (!coordinate)
    return std::optional<CgraComputeLifecycleFrame>{};

  CgraComputeLifecycleFrame frame{*coordinate, {}, {}, {}};
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

  llvm::SmallVector<std::uint64_t, 8> affectedFirings;
  while (isAt(physical_.nextCoordinate(), *coordinate)) {
    auto physicalFrame = physical_.advance();
    if (!physicalFrame)
      return physicalFrame.takeError();
    if (!*physicalFrame)
      return invalid("CGRA physical calendar lost its next frame");
    for (const CgraPhysicalLifecycleEvent &event : (**physicalFrame).events)
      if (llvm::Error error =
              processPhysicalEvent(event, frame, affectedFirings))
        return std::move(error);
  }
  llvm::sort(affectedFirings);
  affectedFirings.erase(
      std::unique(affectedFirings.begin(), affectedFirings.end()),
      affectedFirings.end());
  for (std::uint64_t firing : affectedFirings)
    if (firings_[firing].active)
      if (llvm::Error error = maybeScheduleCommit(firing, *coordinate))
        return std::move(error);

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

  for (std::uint64_t firing : affectedFirings)
    if (firing < firings_.size() && firings_[firing].active)
      maybeComplete(firing, frame);
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

bool CgraComputeRuntime::hasPendingEvents() const {
  return !requestedEvents_.empty() || !actorCommitEvents_.empty() ||
         physical_.hasPendingActions();
}

} // namespace loom::sim::detail

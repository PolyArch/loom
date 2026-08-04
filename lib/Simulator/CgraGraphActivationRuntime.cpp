#include "CgraGraphActivationRuntime.h"

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

using FiringKey = std::pair<std::uint64_t, std::uint64_t>;

} // namespace

llvm::Expected<CgraGraphActivationRuntime> CgraGraphActivationRuntime::create(
    const CgraFrozenExecutionPlan &plan,
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    ::dataflow::RootedGraphLaunchRef launch, ::dataflow::GraphRef graph,
    const PreparedGraphExecution &execution, SimulatorState &state) {
  auto physicalRuntime = CgraPhysicalActionRuntime::create(
      plan.resources, plan.physicalUseTimings);
  if (!physicalRuntime)
    return physicalRuntime.takeError();
  auto physical =
      std::make_unique<CgraPhysicalActionRuntime>(std::move(*physicalRuntime));
  auto computeRuntime = CgraComputeRuntime::create(plan, dataflow, graph,
                                                   execution, state, *physical);
  if (!computeRuntime)
    return computeRuntime.takeError();
  auto memoryRuntime = CgraMemoryRuntime::create(plan, dataflow, launch, graph,
                                                 execution, state, *physical);
  if (!memoryRuntime)
    return memoryRuntime.takeError();
  auto transportRuntime = CgraTransportRuntime::create(
      plan, dataflow, graph, execution, state, *physical);
  if (!transportRuntime)
    return transportRuntime.takeError();
  return CgraGraphActivationRuntime(
      state, std::move(physical),
      std::make_unique<CgraComputeRuntime>(std::move(*computeRuntime)),
      std::make_unique<CgraMemoryRuntime>(std::move(*memoryRuntime)),
      std::make_unique<CgraTransportRuntime>(std::move(*transportRuntime)));
}

llvm::Error CgraGraphActivationRuntime::start(
    SpatialEventCoordinate coordinate,
    llvm::MutableArrayRef<GraphIngressEmission> ingress) {
  if (started_)
    return invalid("CGRA graph activation was already started");
  started_ = true;
  if (llvm::Error error = compute_->start(coordinate))
    return error;
  if (llvm::Error error = memory_->start(coordinate))
    return error;
  state_->nextActorCandidates.reset();
  return transport_->acceptGraphIngressEmissions(coordinate, ingress);
}

std::optional<SpatialEventCoordinate>
CgraGraphActivationRuntime::nextCoordinate() const {
  std::optional<SpatialEventCoordinate> coordinate;
  selectEarlier(compute_->nextCoordinate(), coordinate);
  selectEarlier(memory_->nextCoordinate(), coordinate);
  selectEarlier(transport_->nextCoordinate(), coordinate);
  selectEarlier(physical_->nextCoordinate(), coordinate);
  return coordinate;
}

bool CgraGraphActivationRuntime::hasPendingEvents() const {
  return compute_->hasPendingEvents() || compute_->hasActiveActors() ||
         memory_->hasPendingEvents() || memory_->hasActiveActors() ||
         transport_->hasPendingEvents() || transport_->hasBlockedTransfers() ||
         physical_->hasPendingActions() || !firingByOccurrence_.empty();
}

llvm::Expected<std::uint64_t> CgraGraphActivationRuntime::addCommittedFiring(
    const CgraActorLifecycleEvent &event) {
  const FiringKey key{event.semanticActorOrdinal, event.occurrenceOrdinal};
  if (firingByOccurrence_.contains(key))
    return invalid("CGRA actor firing committed twice");
  std::uint64_t slot = 0;
  if (freeFiringSlots_.empty()) {
    slot = firings_.size();
    firings_.emplace_back();
  } else {
    slot = freeFiringSlots_.back();
    freeFiringSlots_.pop_back();
  }
  firings_[slot] = ActorFiring{true,
                               event.semanticActorOrdinal,
                               event.occurrenceOrdinal,
                               event.transitionCaseOrdinal,
                               event.expectedTransferCount,
                               0,
                               false};
  firingByOccurrence_.try_emplace(key, slot);
  return slot;
}

void CgraGraphActivationRuntime::releaseFiring(std::uint64_t firingSlot) {
  ActorFiring &firing = firings_[firingSlot];
  firingByOccurrence_.erase(
      FiringKey{firing.semanticActorOrdinal, firing.occurrenceOrdinal});
  firing.active = false;
  freeFiringSlots_.push_back(firingSlot);
}

llvm::Error CgraGraphActivationRuntime::maybeRetire(
    std::uint64_t firingSlot, const SpatialEventCoordinate &coordinate,
    CgraGraphActivationFrame &result) {
  if (firingSlot >= firings_.size() || !firings_[firingSlot].active)
    return invalid("CGRA actor retirement names an inactive firing");
  ActorFiring &firing = firings_[firingSlot];
  if (!firing.physicalComplete ||
      firing.completedTransfers != firing.expectedTransfers)
    return llvm::Error::success();
  result.actorEvents.push_back(
      {CgraActorLifecycleKind::Retired, firing.semanticActorOrdinal,
       firing.occurrenceOrdinal, firing.transitionCaseOrdinal, 0, coordinate});
  const bool computeOwner = compute_->ownsActor(firing.semanticActorOrdinal);
  const bool memoryOwner = memory_->ownsActor(firing.semanticActorOrdinal);
  if (computeOwner == memoryOwner)
    return invalid("CGRA actor retirement has no unique runtime owner");
  if (computeOwner) {
    if (llvm::Error error = compute_->retireActor(
            firing.semanticActorOrdinal, firing.occurrenceOrdinal, coordinate))
      return error;
  } else if (llvm::Error error =
                 memory_->retireActor(firing.semanticActorOrdinal,
                                      firing.occurrenceOrdinal, coordinate)) {
    return error;
  }
  releaseFiring(firingSlot);
  return llvm::Error::success();
}

llvm::Error CgraGraphActivationRuntime::markPhysicalCompletion(
    const CgraActorPhysicalCompletion &completion,
    const SpatialEventCoordinate &coordinate,
    CgraGraphActivationFrame &result) {
  auto found = firingByOccurrence_.find(
      {completion.semanticActorOrdinal, completion.occurrenceOrdinal});
  if (found == firingByOccurrence_.end())
    return invalid("CGRA physical completion has no committed actor firing");
  ActorFiring &firing = firings_[found->second];
  if (firing.transitionCaseOrdinal != completion.transitionCaseOrdinal ||
      firing.physicalComplete)
    return invalid("CGRA physical completion disagrees with its actor firing");
  firing.physicalComplete = true;
  return maybeRetire(found->second, coordinate, result);
}

llvm::Error CgraGraphActivationRuntime::consumeTransportCompletions(
    llvm::ArrayRef<CgraTransportCompletion> completions,
    const SpatialEventCoordinate &coordinate,
    CgraGraphActivationFrame &result) {
  for (const CgraTransportCompletion &completion : completions) {
    auto found = firingByOccurrence_.find(
        {completion.semanticActorOrdinal, completion.occurrenceOrdinal});
    if (found == firingByOccurrence_.end())
      return invalid("CGRA transfer completion has no committed actor firing");
    ActorFiring &firing = firings_[found->second];
    if (firing.completedTransfers == firing.expectedTransfers)
      return invalid("CGRA actor firing completed too many transfers");
    ++firing.completedTransfers;
    if (llvm::Error error = maybeRetire(found->second, coordinate, result))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error CgraGraphActivationRuntime::consumeComputeFrame(
    CgraComputeLifecycleFrame frame, CgraGraphActivationFrame &result) {
  result.physicalEvents.insert(result.physicalEvents.end(),
                               frame.physicalEvents.begin(),
                               frame.physicalEvents.end());

  llvm::DenseMap<FiringKey, std::uint32_t> emissionsByFiring;
  for (const CgraActorEmission &emission : frame.actorEmissions) {
    const FiringKey key{emission.semanticActorOrdinal,
                        emission.occurrenceOrdinal};
    std::uint32_t &count = emissionsByFiring[key];
    if (count == std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA actor emission count exceeds u32");
    ++count;
  }
  for (const CgraActorLifecycleEvent &event : frame.actorEvents) {
    if (event.kind != CgraActorLifecycleKind::Committed)
      return invalid("CGRA compute runtime emitted actor retirement");
    const FiringKey key{event.semanticActorOrdinal, event.occurrenceOrdinal};
    if (emissionsByFiring.lookup(key) != event.expectedTransferCount)
      return invalid("CGRA compute emission count disagrees with actor commit");
    auto added = addCommittedFiring(event);
    if (!added)
      return added.takeError();
    result.actorEvents.push_back(event);
  }
  for (const auto &[key, count] : emissionsByFiring) {
    (void)count;
    if (!firingByOccurrence_.contains(key))
      return invalid("CGRA actor emission has no same-frame commit");
  }
  if (!frame.actorEmissions.empty())
    if (llvm::Error error = transport_->acceptActorEmissions(
            frame.coordinate, frame.actorEmissions))
      return error;
  if (!frame.actorEvents.empty())
    if (llvm::Error error = transport_->retryBlocked(frame.coordinate))
      return error;
  for (const CgraActorPhysicalCompletion &completion :
       frame.physicalCompletions)
    if (llvm::Error error =
            markPhysicalCompletion(completion, frame.coordinate, result))
      return error;
  return llvm::Error::success();
}

llvm::Error CgraGraphActivationRuntime::consumeMemoryFrame(
    CgraMemoryLifecycleFrame frame, CgraGraphActivationFrame &result) {
  result.physicalEvents.insert(result.physicalEvents.end(),
                               frame.physicalEvents.begin(),
                               frame.physicalEvents.end());
  for (const CgraActorLifecycleEvent &event : frame.actorEvents) {
    if (event.kind != CgraActorLifecycleKind::Committed)
      return invalid("CGRA memory runtime emitted actor retirement");
    auto added = addCommittedFiring(event);
    if (!added)
      return added.takeError();
    result.actorEvents.push_back(event);
  }
  for (const CgraActorEmission &emission : frame.actorEmissions)
    if (!firingByOccurrence_.contains(
            {emission.semanticActorOrdinal, emission.occurrenceOrdinal}))
      return invalid("CGRA memory emission has no committed actor firing");
  if (!frame.actorEmissions.empty())
    if (llvm::Error error = transport_->acceptActorEmissions(
            frame.coordinate, frame.actorEmissions))
      return error;
  if (!frame.actorEvents.empty())
    if (llvm::Error error = transport_->retryBlocked(frame.coordinate))
      return error;
  for (const CgraActorPhysicalCompletion &completion :
       frame.physicalCompletions)
    if (llvm::Error error =
            markPhysicalCompletion(completion, frame.coordinate, result))
      return error;
  return llvm::Error::success();
}

llvm::Error CgraGraphActivationRuntime::consumeTransportFrame(
    CgraTransportFrame frame, CgraGraphActivationFrame &result) {
  result.physicalEvents.insert(result.physicalEvents.end(),
                               frame.physicalEvents.begin(),
                               frame.physicalEvents.end());
  result.publications.insert(
      result.publications.end(),
      std::make_move_iterator(frame.publications.begin()),
      std::make_move_iterator(frame.publications.end()));
  return consumeTransportCompletions(frame.completions, frame.coordinate,
                                     result);
}

llvm::Error CgraGraphActivationRuntime::schedulePublishedCandidates(
    const SpatialEventCoordinate &coordinate) {
  if (!state_->nextActorCandidates.any())
    return llvm::Error::success();
  auto next = nextSpatialDelta(coordinate);
  if (!next)
    return next.takeError();
  llvm::SmallBitVector candidates = state_->nextActorCandidates;
  state_->nextActorCandidates.reset();
  if (llvm::Error error = compute_->acceptReadyCandidates(*next, candidates))
    return error;
  return memory_->acceptReadyCandidates(std::move(*next), candidates);
}

llvm::Expected<std::optional<CgraGraphActivationFrame>>
CgraGraphActivationRuntime::advance() {
  if (!started_)
    return invalid("CGRA graph activation has not started");
  const std::optional<SpatialEventCoordinate> coordinate = nextCoordinate();
  if (!coordinate)
    return std::optional<CgraGraphActivationFrame>{};
  CgraGraphActivationFrame result{*coordinate, {}, {}, {}};

  while (true) {
    bool progressed = false;
    if (isAt(compute_->nextCoordinate(), *coordinate)) {
      auto frame = compute_->advance();
      if (!frame)
        return frame.takeError();
      if (!*frame)
        return invalid("CGRA compute calendar lost its next frame");
      if (llvm::Error error = consumeComputeFrame(std::move(**frame), result))
        return std::move(error);
      progressed = true;
    }
    if (isAt(memory_->nextCoordinate(), *coordinate)) {
      auto frame = memory_->advance();
      if (!frame)
        return frame.takeError();
      if (!*frame)
        return invalid("CGRA memory calendar lost its next frame");
      if (llvm::Error error = consumeMemoryFrame(std::move(**frame), result))
        return std::move(error);
      progressed = true;
    }
    if (isAt(transport_->nextCoordinate(), *coordinate)) {
      auto frame = transport_->advance();
      if (!frame)
        return frame.takeError();
      if (!*frame)
        return invalid("CGRA transport calendar lost its next frame");
      if (llvm::Error error = consumeTransportFrame(std::move(**frame), result))
        return std::move(error);
      progressed = true;
    }
    if (isAt(physical_->nextCoordinate(), *coordinate)) {
      auto physicalFrame = physical_->advance();
      if (!physicalFrame)
        return physicalFrame.takeError();
      if (!*physicalFrame)
        return invalid("CGRA physical calendar lost its next frame");
      result.physicalEvents.insert(result.physicalEvents.end(),
                                   (**physicalFrame).events.begin(),
                                   (**physicalFrame).events.end());
      auto computeFrame = compute_->acceptPhysicalEvents(**physicalFrame);
      if (!computeFrame)
        return computeFrame.takeError();
      computeFrame->physicalEvents.clear();
      if (llvm::Error error =
              consumeComputeFrame(std::move(*computeFrame), result))
        return std::move(error);
      auto memoryFrame = memory_->acceptPhysicalEvents(**physicalFrame);
      if (!memoryFrame)
        return memoryFrame.takeError();
      if (llvm::Error error =
              consumeMemoryFrame(std::move(*memoryFrame), result))
        return std::move(error);
      auto completions = transport_->acceptPhysicalEvents(**physicalFrame);
      if (!completions)
        return completions.takeError();
      if (llvm::Error error = consumeTransportCompletions(
              *completions, (*physicalFrame)->coordinate, result))
        return std::move(error);
      progressed = true;
    }
    if (!progressed || (!isAt(compute_->nextCoordinate(), *coordinate) &&
                        !isAt(memory_->nextCoordinate(), *coordinate) &&
                        !isAt(transport_->nextCoordinate(), *coordinate) &&
                        !isAt(physical_->nextCoordinate(), *coordinate)))
      break;
  }

  if (llvm::Error error = schedulePublishedCandidates(*coordinate))
    return std::move(error);
  llvm::sort(result.physicalEvents, [](const CgraPhysicalLifecycleEvent &lhs,
                                       const CgraPhysicalLifecycleEvent &rhs) {
    return std::tie(lhs.actionOrdinal, lhs.occurrenceOrdinal, lhs.kind,
                    lhs.ownerEventOrdinal) <
           std::tie(rhs.actionOrdinal, rhs.occurrenceOrdinal, rhs.kind,
                    rhs.ownerEventOrdinal);
  });
  llvm::sort(result.actorEvents, [](const CgraActorLifecycleEvent &lhs,
                                    const CgraActorLifecycleEvent &rhs) {
    return std::tie(lhs.semanticActorOrdinal, lhs.occurrenceOrdinal, lhs.kind,
                    lhs.transitionCaseOrdinal) <
           std::tie(rhs.semanticActorOrdinal, rhs.occurrenceOrdinal, rhs.kind,
                    rhs.transitionCaseOrdinal);
  });
  return std::optional<CgraGraphActivationFrame>(std::move(result));
}

} // namespace loom::sim::detail

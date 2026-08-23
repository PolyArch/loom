#include "Simulator/CGRASimulator.h"

#include "CGRAPreparedExecutionInternal.h"
#include "CgraGraphActivationRuntime.h"
#include "SimulationWireInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::sim {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

llvm::Expected<SpatialEventCoordinate> launchCoordinate() {
  auto cycle = evaluation::ExactRatio::get(0, 1);
  if (!cycle)
    return cycle.takeError();
  return SpatialEventCoordinate{std::move(*cycle), 0};
}

std::optional<std::uint64_t>
integralReferenceCycleDistance(const SpatialEventCoordinate &from,
                               const SpatialEventCoordinate &to) {
  if (compareSpatialEventCoordinates(to, from) < 0)
    return std::nullopt;
  using u128 = unsigned __int128;
  const u128 fromValue = static_cast<u128>(from.referenceCycle.numerator()) *
                         to.referenceCycle.denominator();
  const u128 toValue = static_cast<u128>(to.referenceCycle.numerator()) *
                       from.referenceCycle.denominator();
  const u128 commonDenominator = static_cast<u128>(
      from.referenceCycle.denominator()) *
      to.referenceCycle.denominator();
  const u128 difference = toValue - fromValue;
  if (commonDenominator == 0 || difference % commonDenominator != 0 ||
      difference / commonDenominator >
          std::numeric_limits<std::uint64_t>::max())
    return std::nullopt;
  return static_cast<std::uint64_t>(difference / commonDenominator);
}

std::vector<std::uint64_t> findTransferWaitCycle(
    llvm::ArrayRef<CgraClosedWaitSetDiagnostic::Transfer> transfers) {
  const std::uint64_t absent = std::numeric_limits<std::uint64_t>::max();
  std::vector<std::vector<std::uint64_t>> edges(transfers.size());
  for (std::uint64_t waiting = 0; waiting != transfers.size(); ++waiting) {
    const auto &transfer = transfers[waiting];
    if (!transfer.blocked || transfer.blockingActorOrdinal == absent)
      continue;
    for (std::uint64_t blocking = 0; blocking != transfers.size(); ++blocking)
      if (transfers[blocking].producerActorOrdinal ==
          transfer.blockingActorOrdinal)
        edges[waiting].push_back(blocking);
    llvm::sort(edges[waiting]);
    edges[waiting].erase(
        std::unique(edges[waiting].begin(), edges[waiting].end()),
        edges[waiting].end());
  }

  std::vector<std::uint8_t> state(edges.size(), 0);
  std::vector<std::uint64_t> stack;
  std::vector<std::uint64_t> stackPosition(edges.size(), absent);
  std::vector<std::uint64_t> cycle;
  std::function<bool(std::uint64_t)> visit = [&](std::uint64_t node) {
    state[node] = 1;
    stackPosition[node] = stack.size();
    stack.push_back(node);
    for (std::uint64_t sink : edges[node]) {
      if (state[sink] == 0) {
        if (visit(sink))
          return true;
        continue;
      }
      if (state[sink] != 1)
        continue;
      cycle.assign(stack.begin() + stackPosition[sink], stack.end());
      cycle.push_back(sink);
      return true;
    }
    stack.pop_back();
    stackPosition[node] = absent;
    state[node] = 2;
    return false;
  };
  for (std::uint64_t node = 0; node != edges.size(); ++node)
    if (state[node] == 0 && visit(node))
      break;
  return cycle;
}

struct ActorWaitCase final {
  std::vector<std::uint64_t> internalProducers;
};

struct ActorWaitState final {
  std::vector<std::uint64_t> outputBackpressure;
  std::vector<ActorWaitCase> missingInputCases;
  bool usesOutputBackpressure = false;
  bool eligible = false;
};

llvm::Expected<std::vector<CgraClosedWaitSetDiagnostic::ActorWaitCycleEdge>>
deriveActorWaitCycle(
    const detail::PreparedGraphExecution &execution,
    const detail::SimulatorState &state,
    const CgraClosedWaitSetDiagnostic &closedWait) {
  const std::uint64_t absent = std::numeric_limits<std::uint64_t>::max();
  const std::size_t actorCount = execution.actorPlans.size();
  llvm::DenseMap<mlir::Operation *, std::uint64_t> actorByOperation;
  actorByOperation.reserve(actorCount);
  for (const auto [ordinal, actor] : llvm::enumerate(execution.actorPlans))
    actorByOperation.try_emplace(actor.operation, ordinal);

  std::vector<bool> active(actorCount, false);
  for (const auto &firing : closedWait.actorFirings)
    if (firing.semanticActorOrdinal < active.size())
      active[firing.semanticActorOrdinal] = true;

  std::vector<ActorWaitState> waits(actorCount);
  for (const auto &transfer : closedWait.transfers) {
    if (!transfer.blocked || transfer.producerActorOrdinal >= actorCount ||
        transfer.blockingActorOrdinal >= actorCount)
      continue;
    waits[transfer.producerActorOrdinal].outputBackpressure.push_back(
        transfer.blockingActorOrdinal);
  }
  for (ActorWaitState &wait : waits) {
    llvm::sort(wait.outputBackpressure);
    wait.outputBackpressure.erase(
        std::unique(wait.outputBackpressure.begin(),
                    wait.outputBackpressure.end()),
        wait.outputBackpressure.end());
  }

  for (std::size_t ordinal = 0; ordinal != actorCount; ++ordinal) {
    ActorWaitState &wait = waits[ordinal];
    if (active[ordinal] || !wait.outputBackpressure.empty()) {
      wait.usesOutputBackpressure = true;
      wait.eligible = !wait.outputBackpressure.empty();
      continue;
    }
    const auto &actor = execution.actorPlans[ordinal];
    // This is a diagnostic proof attempt over an already halted execution.
    // An unavailable semantic probe must leave the narrower actor-cycle proof
    // unknown; it must not change the simulator outcome.
    if (actor.transitionProbe == detail::ActorTransitionProbeKind::Unavailable)
      continue;
    auto selected = detail::probeActorTransition(actor, state);
    if (!selected) {
      llvm::consumeError(selected.takeError());
      continue;
    }
    if (*selected)
      continue;

    wait.missingInputCases.reserve(actor.handshakeCases.size());
    bool allCasesHaveInternalWait = !actor.handshakeCases.empty();
    for (const auto &handshake : actor.handshakeCases) {
      ActorWaitCase blockedCase;
      bool hasUnownedMissingInput = false;
      for (std::uint32_t input : handshake.consumedInputs) {
        if (input >= actor.inputChannelCount)
          return invalid("CGRA handshake case names an unknown actor input");
        const std::uint64_t channel = actor.firstInputChannel + input;
        if (channel >= state.channelSlots.size())
          return invalid("CGRA actor input channel is outside runtime state");
        if (!state.channelSlots[channel].ready.empty())
          continue;
        mlir::Value value = actor.operation->getOperand(input);
        mlir::Operation *producer = value.getDefiningOp();
        const auto found = producer ? actorByOperation.find(producer)
                                    : actorByOperation.end();
        if (found == actorByOperation.end()) {
          hasUnownedMissingInput = true;
          continue;
        }
        blockedCase.internalProducers.push_back(found->second);
      }
      llvm::sort(blockedCase.internalProducers);
      blockedCase.internalProducers.erase(
          std::unique(blockedCase.internalProducers.begin(),
                      blockedCase.internalProducers.end()),
          blockedCase.internalProducers.end());
      if (hasUnownedMissingInput || blockedCase.internalProducers.empty())
        allCasesHaveInternalWait = false;
      wait.missingInputCases.push_back(std::move(blockedCase));
    }
    wait.eligible = allCasesHaveInternalWait;
  }

  std::vector<bool> closed(actorCount, false);
  for (std::size_t actor = 0; actor != actorCount; ++actor)
    closed[actor] = waits[actor].eligible;
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t actor = 0; actor != actorCount; ++actor) {
      if (!closed[actor])
        continue;
      const ActorWaitState &wait = waits[actor];
      bool internallyBlocked = false;
      if (wait.usesOutputBackpressure) {
        internallyBlocked = llvm::any_of(
            wait.outputBackpressure,
            [&](std::uint64_t target) { return closed[target]; });
      } else {
        internallyBlocked = llvm::all_of(
            wait.missingInputCases, [&](const ActorWaitCase &blockedCase) {
              return llvm::any_of(blockedCase.internalProducers,
                                  [&](std::uint64_t producer) {
                                    return closed[producer];
                                  });
            });
      }
      if (!internallyBlocked) {
        closed[actor] = false;
        changed = true;
      }
    }
  }

  using Edge =
      std::pair<std::uint64_t, CgraClosedWaitSetDiagnostic::ActorWaitKind>;
  std::vector<std::vector<Edge>> edges(actorCount);
  for (std::size_t actor = 0; actor != actorCount; ++actor) {
    if (!closed[actor])
      continue;
    const ActorWaitState &wait = waits[actor];
    if (wait.usesOutputBackpressure) {
      for (std::uint64_t target : wait.outputBackpressure)
        if (closed[target])
          edges[actor].push_back(
              {target, CgraClosedWaitSetDiagnostic::ActorWaitKind::
                           OutputBackpressure});
    } else {
      for (const ActorWaitCase &blockedCase : wait.missingInputCases)
        for (std::uint64_t producer : blockedCase.internalProducers)
          if (closed[producer])
            edges[actor].push_back(
                {producer,
                 CgraClosedWaitSetDiagnostic::ActorWaitKind::MissingInput});
    }
    llvm::sort(edges[actor], [](const Edge &lhs, const Edge &rhs) {
      return std::tie(lhs.first, lhs.second) <
             std::tie(rhs.first, rhs.second);
    });
    edges[actor].erase(std::unique(edges[actor].begin(), edges[actor].end()),
                       edges[actor].end());
  }

  std::vector<std::uint8_t> visitState(actorCount, 0);
  std::vector<std::uint64_t> stack;
  std::vector<std::uint64_t> stackPosition(actorCount, absent);
  std::vector<std::uint64_t> cycle;
  std::function<bool(std::uint64_t)> visit = [&](std::uint64_t actor) {
    visitState[actor] = 1;
    stackPosition[actor] = stack.size();
    stack.push_back(actor);
    for (const Edge &edge : edges[actor]) {
      const std::uint64_t target = edge.first;
      if (visitState[target] == 0) {
        if (visit(target))
          return true;
        continue;
      }
      if (visitState[target] != 1)
        continue;
      cycle.assign(stack.begin() + stackPosition[target], stack.end());
      cycle.push_back(target);
      return true;
    }
    stack.pop_back();
    stackPosition[actor] = absent;
    visitState[actor] = 2;
    return false;
  };
  for (std::uint64_t actor = 0; actor != actorCount; ++actor)
    if (closed[actor] && visitState[actor] == 0 && visit(actor))
      break;

  std::vector<CgraClosedWaitSetDiagnostic::ActorWaitCycleEdge> result;
  for (std::size_t index = 1; index < cycle.size(); ++index) {
    const std::uint64_t waiting = cycle[index - 1];
    const std::uint64_t blocking = cycle[index];
    const auto selected = llvm::find_if(edges[waiting], [&](const Edge &edge) {
      return edge.first == blocking;
    });
    if (selected == edges[waiting].end())
      return invalid("CGRA actor wait cycle lost its dependency edge");
    result.push_back({waiting, blocking, selected->second});
  }
  return result;
}

} // namespace

struct CgraExecutionSession::Impl final {
  const PreparedCgraExecution::Impl *prepared = nullptr;
  const CanonicalSimulationWorkload *workload = nullptr;
  const CanonicalSimulationRuntimeInput *runtimeInput = nullptr;
  const detail::PreparedCgraGraph *graphExecution = nullptr;
  detail::ResolvedLaunchContext context;
  detail::SimulatorState dynamicState;
  std::optional<detail::CgraGraphActivationRuntime> runtime;
  SpatialExecutionSessionState lifecycle =
      SpatialExecutionSessionState::Runnable;
  CgraSimulationCounters counters;
  std::optional<SpatialEventCoordinate> graphRetirement;
  std::optional<SpatialEventCoordinate> lastCoordinate;
  std::optional<CgraClosedWaitSetDiagnostic> closedWait;
  std::optional<SpatialDiagnosticTrace> trace;
  std::map<std::pair<std::uint64_t, std::uint64_t>, SpatialEventCoordinate>
      physicalRequestCoordinates;
  std::map<std::pair<std::uint64_t, std::uint64_t>, SpatialEventCoordinate>
      physicalGrantCoordinates;
  bool resultTaken = false;

  Impl(const PreparedCgraExecution::Impl &prepared,
       const CanonicalSimulationWorkload &workload,
       const CanonicalSimulationRuntimeInput &runtimeInput,
       const detail::PreparedCgraGraph &graphExecution,
       detail::ResolvedLaunchContext context,
       std::optional<TraceCaptureLevel> traceLevel)
      : prepared(&prepared), workload(&workload), runtimeInput(&runtimeInput),
        graphExecution(&graphExecution), context(std::move(context)) {
    if (traceLevel)
      trace.emplace(SpatialDiagnosticTrace{*traceLevel, {}});
  }

  llvm::Expected<ActorTransitionOccurrenceRef>
  transitionOccurrence(const detail::CgraActorLifecycleEvent &event) const {
    if (event.semanticActorOrdinal >= graphExecution->actors.size())
      return invalid("CGRA trace actor ordinal is out of range");
    return ActorTransitionOccurrenceRef{
        GraphInvocationOccurrenceRef{0},
        graphExecution->actors[event.semanticActorOrdinal],
        event.occurrenceOrdinal};
  }

  llvm::Expected<TokenOccurrenceRef>
  tokenOccurrence(const detail::CgraTokenPublication &publication) const {
    if (const auto *ingress = std::get_if<::dataflow::GraphIngressTokenRef>(
            &publication.producer)) {
      if (publication.occurrenceOrdinal != publication.producerSequenceOrdinal)
        return invalid("CGRA graph-ingress trace sequence is not dense");
      return TokenOccurrenceRef{GraphIngressTokenOccurrenceRef{
          GraphInvocationOccurrenceRef{0}, *ingress,
          publication.producerSequenceOrdinal}};
    }
    const auto &result =
        std::get<::dataflow::ActorTokenResultRef>(publication.producer);
    return TokenOccurrenceRef{ActorResultTokenOccurrenceRef{
        ActorTransitionOccurrenceRef{GraphInvocationOccurrenceRef{0},
                                     result.actor,
                                     publication.occurrenceOrdinal},
        result.ordinal, publication.producerSequenceOrdinal}};
  }

  llvm::Error captureFrame(const detail::CgraGraphActivationFrame &frame) {
    if (!trace)
      return llvm::Error::success();
    SpatialTraceFrame projected{frame.coordinate, {}};
    projected.events.reserve(
        frame.actorEvents.size() + frame.publications.size() +
        frame.memoryLinearizations.size() + frame.physicalTraceEvents.size());
    for (const detail::CgraActorLifecycleEvent &event : frame.actorEvents) {
      auto transition = transitionOccurrence(event);
      if (!transition)
        return transition.takeError();
      if (event.kind == detail::CgraActorLifecycleKind::Committed)
        projected.events.push_back(ActorCommittedTraceEvent{*transition});
      else
        projected.events.push_back(ActorRetiredTraceEvent{*transition});
    }
    if (trace->level >= TraceCaptureLevel::Semantic) {
      projected.events.insert(projected.events.end(),
                              frame.memoryLinearizations.begin(),
                              frame.memoryLinearizations.end());
      for (const detail::CgraTokenPublication &publication :
           frame.publications) {
        auto occurrence = tokenOccurrence(publication);
        if (!occurrence)
          return occurrence.takeError();
        auto type = prepared->dataflowView.tokenType(publication.producer);
        if (!type)
          return type.takeError();
        auto value = detail::canonicalValueSequenceFromTokens(
            llvm::ArrayRef(publication.token), *type,
            context.graphOp.getOperation());
        if (!value)
          return value.takeError();
        projected.events.push_back(TokenPublishedTraceEvent{
            std::move(*occurrence), std::move(*value)});
      }
    }
    if (trace->level >= TraceCaptureLevel::Microarchitecture)
      projected.events.insert(projected.events.end(),
                              frame.physicalTraceEvents.begin(),
                              frame.physicalTraceEvents.end());
    if (projected.events.empty())
      return llvm::Error::success();
    return appendSpatialTraceFrame(*trace, std::move(projected));
  }

  llvm::Error observeGraphRetirement(const SpatialEventCoordinate &coordinate) {
    if (graphRetirement ||
        !detail::graphCompletionReady(graphExecution->execution, dynamicState))
      return llvm::Error::success();
    graphRetirement = coordinate;
    return llvm::Error::success();
  }

  llvm::Error settleQuiescence() {
    if (runtime->nextCoordinate())
      return invalid("CGRA session quiesced with a scheduled event");
    if (dynamicState.failure != detail::RunFailure::None ||
        !dynamicState.diagnostics.empty()) {
      lifecycle = SpatialExecutionSessionState::Failed;
      return llvm::createStringError(
          std::errc::state_not_recoverable,
          "CGRA execution ended with a semantic provider failure");
    }
    if (graphRetirement && !runtime->hasPendingEvents()) {
      if (llvm::Error error = detail::validateGraphRetirementBoundary(
              context.graphOp, graphExecution->execution, dynamicState)) {
        lifecycle = SpatialExecutionSessionState::Failed;
        return error;
      }
      if (detail::hasPendingVectorGroups(dynamicState)) {
        lifecycle = SpatialExecutionSessionState::Failed;
        return invalid("CGRA execution retired with incomplete vector state");
      }
      lifecycle = SpatialExecutionSessionState::Retired;
      return llvm::Error::success();
    }

    lifecycle = SpatialExecutionSessionState::Halted;
    closedWait.emplace();
    closedWait->pendingActorFirings = runtime->pendingActorFiringCount();
    closedWait->pendingTransfers = runtime->pendingTransferCount();
    closedWait->pendingPhysicalActions = runtime->pendingPhysicalActionCount();
    closedWait->graphRetirementVisible = graphRetirement.has_value();
    closedWait->ownerReferences = CgraExecutionOwnerReferences{
        {::dataflow::canonicalDataflowSchema.identity.str(),
         ::dataflow::canonicalDataflowSchema.version,
         prepared->dataflow.identity()},
        prepared->fabric.reference(), prepared->tech.reference(),
        prepared->spatial.reference()};
    const auto &operandProgress = runtime->operandQueueProgress();
    closedWait->operandQueueGroupCount = operandProgress.groupCount;
    closedWait->operandQueuePotentiallyBlockingGroupCount =
        operandProgress.potentiallyBlockingGroupCount;
    closedWait->operandQueueSharedIngressPressure =
        operandProgress.sharedIngressPressure;
    closedWait->operandQueueDistinctIngressCount =
        operandProgress.distinctIngressCount;
    closedWait->operandQueuePairingKeyCount =
        operandProgress.pairingKeyCount;
    closedWait->operandQueueProgressStatus =
        static_cast<std::uint8_t>(operandProgress.status);
    closedWait->operandQueueProgressSupport =
        static_cast<std::uint8_t>(operandProgress.support);
    closedWait->operandQueueProjectionDigest =
        operandProgress.projectionDigest;
    for (const auto &head : runtime->pendingOperandQueueHeadDiagnostics())
      closedWait->operandQueueHeads.push_back(
          {head.queue,
           head.fu,
           head.allocationUnit,
           head.capacity,
           head.occupancy,
           head.reservations,
           head.headBindingOrdinal,
           head.headOccurrenceOrdinal,
           head.headProducerSequenceOrdinal,
           head.headTag,
           head.exactHead,
           head.consumers});
    for (const auto &firing : runtime->pendingActorFiringDiagnostics())
      closedWait->actorFirings.push_back(
          {firing.semanticActorOrdinal, firing.occurrenceOrdinal,
           firing.transitionCaseOrdinal, firing.expectedTransfers,
           firing.completedTransfers, firing.physicalComplete,
           firing.causalReleaseSatisfied});
    for (const auto &transfer : runtime->pendingTransferDiagnostics()) {
      std::vector<CgraClosedWaitSetDiagnostic::Transfer::OperandQueueWait>
          operandQueueWaits;
      operandQueueWaits.reserve(transfer.operandQueueWaits.size());
      for (const auto &wait : transfer.operandQueueWaits)
        operandQueueWaits.push_back(
            {wait.queue, wait.fu, wait.ingress, wait.tag, wait.allocationUnit,
             wait.occupancy, wait.reservations, wait.capacity});
      closedWait->transfers.push_back(
          {transfer.bindingOrdinal,
           transfer.occurrenceOrdinal,
           transfer.producerActorOrdinal,
           transfer.producerResultOrdinal,
           transfer.blocked,
           transfer.arrivalScheduled,
           transfer.publicationReady,
           transfer.published,
           transfer.consumedRequested,
           transfer.operandCapacityReserved,
           transfer.operandCapacityBlocked,
           transfer.producedPermitted,
           transfer.producedRetired,
           transfer.traversalPermitted,
           transfer.traversalRetired,
           transfer.traversalTerminalsPermitted,
           transfer.consumedPermitted,
           transfer.consumedRetired,
           transfer.readySinkCount,
           transfer.publishedSinkCount,
           transfer.sinkCount,
           transfer.publicationCount,
           transfer.requestedPublicationCount,
           transfer.publishedPublicationCount,
           transfer.unpublishedActorOrdinals,
           transfer.unpublishedInputOrdinals,
           transfer.unpublishedReadyTokenCounts,
           transfer.blockingTraversalNodeOrdinal,
           transfer.blockingStorageOrdinal,
           transfer.blockingFifoOccurrence,
           transfer.blockingStorageOccupancy,
           transfer.blockingStorageReservations,
           transfer.blockingStorageCapacity,
           transfer.blockingTraversalState,
           transfer.blockingDownstreamStorageCount,
           transfer.blockingUnbufferedSinkCount,
           transfer.blockingDownstreamStorageOrdinal,
           transfer.blockingDownstreamStorageOccupancy,
           transfer.blockingDownstreamStorageReservations,
           transfer.blockingDownstreamStorageCapacity,
           transfer.blockingDownstreamStorageReserved,
           transfer.blockingActorOrdinal,
           transfer.blockingReadyTokenCount,
           transfer.blockingQueueOccupancy,
           transfer.blockingQueueReservations,
           transfer.blockingQueueCapacity,
           std::move(operandQueueWaits)});
    }
    for (const auto &action : runtime->pendingPhysicalActionDiagnostics())
      closedWait->physicalActions.push_back(
          {action.action.actionOrdinal, action.action.occurrenceOrdinal,
           static_cast<std::uint8_t>(action.client), action.semanticActorOrdinal,
           action.action.granted,
           action.action.hasCommit, action.action.requiresCausalRelease,
           action.action.intrinsicReleaseReached,
           action.action.causalReleaseReached});
    const std::vector<std::uint64_t> transferCycle =
        findTransferWaitCycle(closedWait->transfers);
    for (std::size_t edge = 1; edge < transferCycle.size(); ++edge) {
      const auto &waiting = closedWait->transfers[transferCycle[edge - 1]];
      const auto &blocking = closedWait->transfers[transferCycle[edge]];
      closedWait->transferWaitCycle.push_back(
          {waiting.bindingOrdinal, waiting.occurrenceOrdinal,
           waiting.blockingActorOrdinal, blocking.bindingOrdinal,
           blocking.occurrenceOrdinal});
    }
    auto actorCycle = deriveActorWaitCycle(graphExecution->execution,
                                           dynamicState, *closedWait);
    if (!actorCycle)
      return actorCycle.takeError();
    closedWait->actorWaitCycle = std::move(*actorCycle);
    return llvm::Error::success();
  }
};

CgraExecutionSession::CgraExecutionSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CgraExecutionSession::CgraExecutionSession(CgraExecutionSession &&) noexcept =
    default;
CgraExecutionSession &
CgraExecutionSession::operator=(CgraExecutionSession &&) noexcept = default;
CgraExecutionSession::~CgraExecutionSession() = default;

SpatialExecutionSessionState CgraExecutionSession::state() const {
  return impl_ ? impl_->lifecycle : SpatialExecutionSessionState::Failed;
}

const CgraSimulationCounters &CgraExecutionSession::counters() const {
  static const CgraSimulationCounters empty;
  return impl_ ? impl_->counters : empty;
}

const std::optional<CgraClosedWaitSetDiagnostic> &
CgraExecutionSession::closedWaitSet() const {
  static const std::optional<CgraClosedWaitSetDiagnostic> empty;
  return impl_ ? impl_->closedWait : empty;
}

const std::optional<SpatialDiagnosticTrace> &
CgraExecutionSession::diagnosticTrace() const {
  static const std::optional<SpatialDiagnosticTrace> empty;
  return impl_ ? impl_->trace : empty;
}

llvm::Expected<SpatialExecutionSessionState> CgraExecutionSession::advance(
    std::uint64_t maxEventFrames,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline) {
  if (!impl_)
    return invalid("CGRA execution session is empty");
  if (impl_->resultTaken)
    return invalid("CGRA execution result was already taken");
  if (impl_->lifecycle != SpatialExecutionSessionState::Runnable)
    return impl_->lifecycle;
  if (maxEventFrames == 0)
    return invalid("CGRA execution advance requires a positive frame budget");

  std::uint64_t advanced = 0;
  while (advanced != maxEventFrames) {
    if (executionDeadline &&
        std::chrono::steady_clock::now() >= *executionDeadline) {
      impl_->lifecycle = SpatialExecutionSessionState::StoppedByLimit;
      return impl_->lifecycle;
    }

    auto frame = impl_->runtime->advance();
    if (!frame) {
      impl_->lifecycle = SpatialExecutionSessionState::Failed;
      return frame.takeError();
    }
    if (!*frame) {
      if (llvm::Error error = impl_->settleQuiescence())
        return std::move(error);
      return impl_->lifecycle;
    }

    impl_->lastCoordinate = (**frame).coordinate;
    ++impl_->counters.eventFrameCount;
    ++advanced;
    impl_->counters.maximumReferenceCycleNumerator = std::max(
        impl_->counters.maximumReferenceCycleNumerator,
        (**frame).coordinate.referenceCycle.numerator());
    impl_->counters.maximumEventDelta = std::max(
        impl_->counters.maximumEventDelta, (**frame).coordinate.delta);
    impl_->counters.emptyEventFrameCount +=
        (**frame).physicalEvents.empty() && (**frame).actorEvents.empty() &&
        (**frame).publications.empty() &&
        (**frame).memoryLinearizations.empty();
    impl_->counters.computeSourceFrameCount += ((**frame).sourceMask & 1) != 0;
    impl_->counters.memorySourceFrameCount += ((**frame).sourceMask & 2) != 0;
    impl_->counters.transportSourceFrameCount +=
        ((**frame).sourceMask & 4) != 0;
    impl_->counters.physicalSourceFrameCount += ((**frame).sourceMask & 8) != 0;
    for (const detail::CgraActorLifecycleEvent &event : (**frame).actorEvents) {
      if (event.kind == detail::CgraActorLifecycleKind::Committed)
        ++impl_->counters.actorCommitCount;
      else
        ++impl_->counters.actorRetirementCount;
    }
    impl_->counters.tokenPublicationCount += (**frame).publications.size();
    impl_->counters.memoryLinearizationCount +=
        (**frame).memoryLinearizations.size();
    for (const detail::CgraPhysicalLifecycleEvent &event :
         (**frame).physicalEvents) {
      const auto key = std::make_pair(event.actionOrdinal,
                                      event.occurrenceOrdinal);
      switch (event.kind) {
      case detail::CgraPhysicalLifecycleKind::Requested:
        ++impl_->counters.physicalRequestCount;
        if (!impl_->physicalRequestCoordinates.emplace(key, event.coordinate)
                 .second) {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid("CGRA physical request was observed twice");
        }
        break;
      case detail::CgraPhysicalLifecycleKind::Granted:
        ++impl_->counters.physicalGrantCount;
        if (auto request = impl_->physicalRequestCoordinates.find(key);
            request == impl_->physicalRequestCoordinates.end()) {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid("CGRA physical grant has no request observation");
        } else {
          auto wait = integralReferenceCycleDistance(request->second,
                                                     event.coordinate);
          if (!wait) {
            ++impl_->counters.nonIntegralTimingObservationCount;
          } else if (*wait == 0) {
            ++impl_->counters.physicalGrantSameCycleCount;
          } else {
            ++impl_->counters.physicalGrantDelayedCount;
            impl_->counters.physicalGrantWaitCycleSum += *wait;
            impl_->counters.physicalGrantWaitCycleMax = std::max(
                impl_->counters.physicalGrantWaitCycleMax, *wait);
          }
        }
        impl_->physicalGrantCoordinates.insert_or_assign(key, event.coordinate);
        break;
      case detail::CgraPhysicalLifecycleKind::Committed:
        break;
      case detail::CgraPhysicalLifecycleKind::Retired:
        ++impl_->counters.physicalRetirementCount;
        if (auto request = impl_->physicalRequestCoordinates.find(key);
            request == impl_->physicalRequestCoordinates.end()) {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid("CGRA physical retirement has no request observation");
        } else if (auto lifetime = integralReferenceCycleDistance(
                       request->second, event.coordinate)) {
          impl_->counters.physicalActionLifetimeCycleSum += *lifetime;
          impl_->counters.physicalActionLifetimeCycleMax = std::max(
              impl_->counters.physicalActionLifetimeCycleMax, *lifetime);
        } else {
          ++impl_->counters.nonIntegralTimingObservationCount;
        }
        if (auto grant = impl_->physicalGrantCoordinates.find(key);
            grant != impl_->physicalGrantCoordinates.end()) {
          if (auto active = integralReferenceCycleDistance(
                  grant->second, event.coordinate)) {
            impl_->counters.physicalGrantedLifetimeCycleSum += *active;
            impl_->counters.physicalGrantedLifetimeCycleMax = std::max(
                impl_->counters.physicalGrantedLifetimeCycleMax, *active);
          } else {
            ++impl_->counters.nonIntegralTimingObservationCount;
          }
        } else {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid(
              "CGRA physical retirement has no grant observation");
        }
        impl_->physicalRequestCoordinates.erase(key);
        impl_->physicalGrantCoordinates.erase(key);
        break;
      }
    }
    if (llvm::Error error = impl_->captureFrame(**frame)) {
      impl_->lifecycle = SpatialExecutionSessionState::Failed;
      return std::move(error);
    }
    if (llvm::Error error =
            impl_->observeGraphRetirement((**frame).coordinate)) {
      impl_->lifecycle = SpatialExecutionSessionState::Failed;
      return std::move(error);
    }
    if (!impl_->runtime->hasPendingEvents()) {
      if (llvm::Error error = impl_->settleQuiescence())
        return std::move(error);
      return impl_->lifecycle;
    }
  }
  return impl_->lifecycle;
}

llvm::Expected<RetiredCgraSimulation>
CgraExecutionSession::takeRetiredSimulation() {
  if (!impl_)
    return invalid("CGRA execution session is empty");
  if (impl_->resultTaken)
    return invalid("CGRA execution result was already taken");
  if (impl_->lifecycle != SpatialExecutionSessionState::Retired ||
      !impl_->graphRetirement || !impl_->lastCoordinate)
    return llvm::createStringError(
        std::errc::state_not_recoverable,
        "CGRA execution session has not retired successfully");

  auto observations = detail::projectRetiredFunctionalObservations(
      impl_->context.graphOp, impl_->dynamicState, *impl_->workload,
      *impl_->runtimeInput, impl_->context, impl_->prepared->dataflowView);
  if (!observations)
    return observations.takeError();
  impl_->resultTaken = true;
  auto launch = launchCoordinate();
  if (!launch)
    return launch.takeError();
  return RetiredCgraSimulation{
      std::move(*observations),
      SpatialProgressObservations{std::move(*launch), impl_->graphRetirement,
                                  *impl_->lastCoordinate},
      impl_->counters};
}

llvm::Expected<CgraExecutionSession>
startCgraExecutionSession(const PreparedCgraExecution &prepared,
                          const CanonicalSimulationWorkload &workload,
                          const CanonicalSimulationRuntimeInput &runtimeInput,
                          std::optional<TraceCaptureLevel> traceLevel,
                          CgraExternalMemoryProvider *externalMemoryProvider) {
  if (!prepared.impl_)
    return invalid("prepared CGRA execution is empty");
  const SpatialSimulationWorkload *spatial = workload.spatial();
  if (!spatial)
    return invalid("CGRA execution session requires a Spatial workload");
  auto graph = admitCgraSpatialSimulation(prepared, workload, runtimeInput);
  if (!graph)
    return graph.takeError();
  auto graphFound = llvm::find_if(
      prepared.impl_->graphs, [&](const detail::PreparedCgraGraph &candidate) {
        return candidate.graph == *graph;
      });
  const detail::PreparedCgraGraph *graphExecution =
      graphFound == prepared.impl_->graphs.end() ? nullptr : &*graphFound;
  if (!graphExecution)
    return invalid("CGRA session has no prepared semantic graph");
  auto context = detail::resolveLaunchContext(prepared.impl_->dataflowView,
                                              spatial->launchRef);
  if (!context)
    return context.takeError();
  if (std::optional<std::string> reason =
          detail::unsupportedTypedDfgInput(workload, runtimeInput, *context))
    return llvm::createStringError(std::errc::not_supported, "%s",
                                   reason->c_str());

  auto impl = std::make_unique<CgraExecutionSession::Impl>(
      *prepared.impl_, workload, runtimeInput, *graphExecution,
      std::move(*context), traceLevel);

  llvm::SmallVector<detail::GraphIngressEmission, 4> ingress;
  impl->dynamicState.graphIngressCapture = &ingress;
  llvm::scope_exit clearCapture(
      [&] { impl->dynamicState.graphIngressCapture = nullptr; });
  if (llvm::Error error = detail::initializeTypedGraphExecutionState(
          impl->dynamicState, graphExecution->execution, impl->context.graphOp,
          workload, runtimeInput, impl->context))
    return std::move(error);
  clearCapture.release();
  impl->dynamicState.graphIngressCapture = nullptr;

  auto runtime = detail::CgraGraphActivationRuntime::create(
      prepared.impl_->executionPlan, prepared.impl_->dataflowView,
      spatial->launchRef, *graph, graphExecution->execution, impl->dynamicState,
      traceLevel == TraceCaptureLevel::Microarchitecture,
      externalMemoryProvider);
  if (!runtime)
    return runtime.takeError();
  impl->runtime.emplace(std::move(*runtime));
  auto launch = launchCoordinate();
  if (!launch)
    return launch.takeError();
  if (llvm::Error error = impl->runtime->start(*launch, ingress))
    return std::move(error);
  if (llvm::Error error = impl->observeGraphRetirement(*launch))
    return std::move(error);
  return CgraExecutionSession(std::move(impl));
}

llvm::Expected<CgraSimulationOutcome> simulateCgraWorkload(
    const PreparedCgraExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventFrames,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline,
    CgraExternalMemoryProvider *externalMemoryProvider) {
  if (maxEventFrames == 0)
    return invalid("CGRA simulation requires a positive event-frame limit");
  auto session = startCgraExecutionSession(
      prepared, workload, runtimeInput, std::nullopt, externalMemoryProvider);
  if (!session)
    return session.takeError();
  auto advanced = session->advance(maxEventFrames, executionDeadline);
  if (!advanced)
    return advanced.takeError();
  SpatialExecutionSessionState state = *advanced;
  if (state == SpatialExecutionSessionState::Runnable) {
    session->impl_->lifecycle = SpatialExecutionSessionState::StoppedByLimit;
    state = session->impl_->lifecycle;
  }

  CgraSimulationOutcome result;
  result.state = state;
  result.counters = session->counters();
  result.closedWaitSet = session->closedWaitSet();
  if (state == SpatialExecutionSessionState::Retired) {
    auto retired = session->takeRetiredSimulation();
    if (!retired)
      return retired.takeError();
    result.retired = std::move(*retired);
  }
  return result;
}

} // namespace loom::sim

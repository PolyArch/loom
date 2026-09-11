#include "Simulator/CGRASimulator.h"
#include "CgraFabricActivityRuntime.h"
#include "Simulator/CGRA/EventQueue.h"

#include "CGRAPreparedExecutionInternal.h"
#include "CgraClosedWaitProjection.h"
#include "CgraGraphActivationRuntime.h"
#include "SimulationWireInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::sim {

char CgraExecutionUnsupported::ID = 0;

void CgraExecutionUnsupported::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code CgraExecutionUnsupported::convertToErrorCode() const {
  return std::make_error_code(std::errc::not_supported);
}

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


} // namespace

struct PreparedCgraWorkloadExecution::Impl final {
  std::shared_ptr<const PreparedCgraExecution::Impl> prepared;
  const detail::PreparedCgraGraph *graphExecution = nullptr;
  detail::ResolvedLaunchContext context;
  ArtifactIdentity workload;
  ArtifactIdentity runtimeInput;

  Impl(std::shared_ptr<const PreparedCgraExecution::Impl> prepared,
       const detail::PreparedCgraGraph &graphExecution,
       detail::ResolvedLaunchContext context, ArtifactIdentity workload,
       ArtifactIdentity runtimeInput)
      : prepared(std::move(prepared)), graphExecution(&graphExecution),
        context(std::move(context)), workload(workload),
        runtimeInput(runtimeInput) {}
};

PreparedCgraWorkloadExecution::PreparedCgraWorkloadExecution(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedCgraWorkloadExecution::PreparedCgraWorkloadExecution(
    PreparedCgraWorkloadExecution &&) noexcept = default;
PreparedCgraWorkloadExecution &PreparedCgraWorkloadExecution::operator=(
    PreparedCgraWorkloadExecution &&) noexcept = default;
PreparedCgraWorkloadExecution::~PreparedCgraWorkloadExecution() = default;

struct CgraExecutionSession::Impl final {
  struct PhysicalActionObservation final {
    SpatialEventCoordinate requested;
    std::optional<SpatialEventCoordinate> granted;
  };

  std::shared_ptr<const PreparedCgraExecution::Impl> preparedOwner;
  const PreparedCgraExecution::Impl *prepared = nullptr;
  const CanonicalSimulationWorkload *workload = nullptr;
  const CanonicalSimulationRuntimeInput *runtimeInput = nullptr;
  const detail::PreparedCgraGraph *graphExecution = nullptr;
  detail::ResolvedLaunchContext context;
  /// Committed and retired firings per semantic actor ordinal. The scalar
  /// counters below total the same events; this table keeps their actor
  /// attribution so occupancy can be measured per actor kind.
  std::vector<ActorTransitionCounts> actorTransitions;
  detail::SimulatorState dynamicState;
  detail::ExternalStreamInputState streamInputs;
  std::optional<CanonicalSimulationRuntimeInput> capturedRuntimeInput;
  std::unique_ptr<detail::CgraFabricActivityRuntime> activity;
  std::optional<detail::CgraGraphActivationRuntime> runtime;
  SpatialExecutionSessionState lifecycle =
      SpatialExecutionSessionState::Runnable;
  CgraSimulationCounters counters;
  std::optional<SpatialEventCoordinate> graphRetirement;
  std::optional<SpatialEventCoordinate> lastCoordinate;
  std::optional<CgraClosedWaitSetDiagnostic> closedWait;
  std::optional<SpatialDiagnosticTrace> trace;
  llvm::DenseMap<std::pair<std::uint64_t, std::uint64_t>,
                 PhysicalActionObservation>
      physicalActionObservations;
  bool resultTaken = false;

  Impl(std::shared_ptr<const PreparedCgraExecution::Impl> prepared,
       const CanonicalSimulationWorkload &workload,
       const CanonicalSimulationRuntimeInput &runtimeInput,
       const detail::PreparedCgraGraph &graphExecution,
       detail::ResolvedLaunchContext context,
       std::optional<TraceCaptureLevel> traceLevel)
      : preparedOwner(std::move(prepared)), prepared(preparedOwner.get()),
        workload(&workload), runtimeInput(&runtimeInput),
        graphExecution(&graphExecution), context(std::move(context)),
        actorTransitions(graphExecution.actors.size()) {
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
        auto type = prepared->dataflow.view().tokenType(publication.producer);
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
    if (activity)
      return activity->close(ActivityWindow::LaunchToGraphRetirement,
                             coordinate);
    return llvm::Error::success();
  }

  llvm::Expected<bool>
  requestStreamInput(const SpatialEventCoordinate &coordinate) {
    if (graphRetirement || streamInputs.ordinals.empty() ||
        dynamicState.failure != detail::RunFailure::None ||
        !dynamicState.diagnostics.empty())
      return false;
    auto requested = streamInputs.request(
        dynamicState, context, coordinate,
        [&](detail::ChannelOrdinal channel,
            std::uint64_t occurrence) -> llvm::Expected<bool> {
          if (runtime->channelArrivalCount(channel) != occurrence)
            return false;
          auto argument = mlir::cast<mlir::BlockArgument>(
              graphExecution->execution.channels[channel].operand->get());
          return runtime->canAcceptGraphIngress(argument.getArgNumber());
        });
    if (!requested) {
      lifecycle = SpatialExecutionSessionState::Failed;
      return requested.takeError();
    }
    if (*requested)
      lifecycle = SpatialExecutionSessionState::WaitingForExternalStreamInput;
    return *requested;
  }

  /// The actor-transition table over the whole launch-to-terminal window. It
  /// is total over the rooted graph's actor inventory, so an actor that never
  /// fired is reported with zero counts rather than omitted.
  std::optional<ActivitySummary> actorTransitionSummary() const {
    if (actorTransitions.empty())
      return std::nullopt;
    std::vector<ActorTransitionEntry> transitions;
    transitions.reserve(actorTransitions.size());
    for (std::size_t ordinal = 0; ordinal != actorTransitions.size(); ++ordinal)
      transitions.push_back(
          {graphExecution->actors[ordinal], actorTransitions[ordinal]});
    llvm::sort(transitions, [](const ActorTransitionEntry &lhs,
                               const ActorTransitionEntry &rhs) {
      return lhs.actor.entity.value() < rhs.actor.entity.value();
    });
    return ActivitySummary{ActivityWindow::LaunchToTerminal,
                           ActivityCoverage::Complete,
                           ActorTransitionsActivity{std::move(transitions)}};
  }

  llvm::Expected<std::vector<ActivitySummary>>
  finishActivity(const SpatialEventCoordinate &terminal) {
    std::vector<ActivitySummary> summaries;
    // The actor table sorts before a Fabric summary of the same window.
    if (auto actors = actorTransitionSummary())
      summaries.push_back(std::move(*actors));
    if (!activity)
      return summaries;
    if (llvm::Error error =
            activity->close(ActivityWindow::LaunchToTerminal, terminal))
      return std::move(error);
    llvm::append_range(summaries, activity->takeSummaries());
    llvm::sort(summaries, [](const ActivitySummary &lhs,
                             const ActivitySummary &rhs) {
      return std::make_pair(lhs.window, lhs.payload.index()) <
             std::make_pair(rhs.window, rhs.payload.index());
    });
    return summaries;
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
      if (!streamInputs.ordinals.empty()) {
        auto captured =
            streamInputs.capture(dynamicState, *workload, *runtimeInput,
                                 context, prepared->dataflow.view());
        if (!captured) {
          lifecycle = SpatialExecutionSessionState::Failed;
          return captured.takeError();
        }
        capturedRuntimeInput.emplace(std::move(*captured));
        runtimeInput = &*capturedRuntimeInput;
      }
      runtime->emitActorTimingStatistics();
      lifecycle = SpatialExecutionSessionState::Retired;
      return llvm::Error::success();
    }

    lifecycle = SpatialExecutionSessionState::Halted;
    auto projected = detail::projectCgraClosedWaitSet(
        graphExecution->execution, dynamicState, *runtime,
        CgraExecutionOwnerReferences{
            {::dataflow::canonicalDataflowSchema.identity.str(),
             ::dataflow::canonicalDataflowSchema.version,
             prepared->dataflow.identity()},
            prepared->fabric.reference(), prepared->tech.reference(),
            prepared->spatial.reference()},
        graphRetirement.has_value());
    if (!projected)
      return projected.takeError();
    closedWait.emplace(std::move(*projected));
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

  if (!impl_->streamInputs.ordinals.empty()) {
    auto coordinate = launchCoordinate();
    if (!coordinate)
      return coordinate.takeError();
    if (impl_->lastCoordinate)
      *coordinate = *impl_->lastCoordinate;
    auto requested = impl_->requestStreamInput(*coordinate);
    if (!requested)
      return requested.takeError();
    if (*requested)
      return impl_->lifecycle;
  }

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
      if (impl_->runtime->waitingForExternalMemory()) {
        impl_->lifecycle =
            SpatialExecutionSessionState::WaitingForExternalMemory;
        return impl_->lifecycle;
      }
      if (llvm::Error error = impl_->settleQuiescence())
        return std::move(error);
      return impl_->lifecycle;
    }

    impl_->lastCoordinate = (**frame).coordinate;
    ++impl_->counters.eventFrameCount;
    ++advanced;
    impl_->counters.maximumReferenceCycleNumerator =
        std::max(impl_->counters.maximumReferenceCycleNumerator,
                 (**frame).coordinate.referenceCycle.numerator());
    impl_->counters.maximumEventDelta =
        std::max(impl_->counters.maximumEventDelta, (**frame).coordinate.delta);
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
      if (event.semanticActorOrdinal >= impl_->actorTransitions.size())
        return invalid("CGRA actor lifecycle ordinal is out of range");
      ActorTransitionCounts &counts =
          impl_->actorTransitions[event.semanticActorOrdinal];
      if (event.kind == detail::CgraActorLifecycleKind::Committed) {
        ++impl_->counters.actorCommitCount;
        ++counts.committedFirings;
      } else {
        ++impl_->counters.actorRetirementCount;
        ++counts.retiredFirings;
      }
    }
    impl_->counters.tokenPublicationCount += (**frame).publications.size();
    impl_->counters.memoryLinearizationCount +=
        (**frame).memoryLinearizations.size();
    for (const detail::CgraPhysicalLifecycleEvent &event :
         (**frame).physicalEvents) {
      const auto key =
          std::make_pair(event.actionOrdinal, event.occurrenceOrdinal);
      switch (event.kind) {
      case detail::CgraPhysicalLifecycleKind::Requested:
        ++impl_->counters.physicalRequestCount;
        if (!impl_->physicalActionObservations
                 .try_emplace(key,
                              Impl::PhysicalActionObservation{event.coordinate,
                                                              std::nullopt})
                 .second) {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid("CGRA physical request was observed twice");
        }
        break;
      case detail::CgraPhysicalLifecycleKind::Granted:
        ++impl_->counters.physicalGrantCount;
        if (auto observation = impl_->physicalActionObservations.find(key);
            observation == impl_->physicalActionObservations.end()) {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid("CGRA physical grant has no request observation");
        } else {
          auto wait = integralSpatialReferenceCycleDistance(
              observation->second.requested, event.coordinate);
          if (!wait) {
            ++impl_->counters.nonIntegralTimingObservationCount;
          } else if (*wait == 0) {
            ++impl_->counters.physicalGrantSameCycleCount;
          } else {
            ++impl_->counters.physicalGrantDelayedCount;
            impl_->counters.physicalGrantWaitCycleSum += *wait;
            impl_->counters.physicalGrantWaitCycleMax =
                std::max(impl_->counters.physicalGrantWaitCycleMax, *wait);
          }
          observation->second.granted = event.coordinate;
        }
        break;
      case detail::CgraPhysicalLifecycleKind::Committed:
        break;
      case detail::CgraPhysicalLifecycleKind::Retired: {
        ++impl_->counters.physicalRetirementCount;
        auto observation = impl_->physicalActionObservations.find(key);
        if (observation == impl_->physicalActionObservations.end()) {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid("CGRA physical retirement has no request observation");
        }
        if (auto lifetime = integralSpatialReferenceCycleDistance(
                observation->second.requested, event.coordinate)) {
          impl_->counters.physicalActionLifetimeCycleSum += *lifetime;
          impl_->counters.physicalActionLifetimeCycleMax = std::max(
              impl_->counters.physicalActionLifetimeCycleMax, *lifetime);
        } else {
          ++impl_->counters.nonIntegralTimingObservationCount;
        }
        if (!observation->second.granted) {
          impl_->lifecycle = SpatialExecutionSessionState::Failed;
          return invalid("CGRA physical retirement has no grant observation");
        }
        if (auto active = integralSpatialReferenceCycleDistance(
                *observation->second.granted, event.coordinate)) {
          impl_->counters.physicalGrantedLifetimeCycleSum += *active;
          impl_->counters.physicalGrantedLifetimeCycleMax = std::max(
              impl_->counters.physicalGrantedLifetimeCycleMax, *active);
        } else {
          ++impl_->counters.nonIntegralTimingObservationCount;
        }
        impl_->physicalActionObservations.erase(observation);
        break;
      }
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
    // A live receive is exposed at its first causal actor-demand boundary,
    // even while unrelated actors still have future calendar work.
    auto requested = impl_->requestStreamInput((**frame).coordinate);
    if (!requested)
      return requested.takeError();
    if (*requested)
      return impl_->lifecycle;
    if (!impl_->runtime->hasPendingEvents()) {
      if (llvm::Error error = impl_->settleQuiescence())
        return std::move(error);
      return impl_->lifecycle;
    }
  }
  return impl_->lifecycle;
}

llvm::Error CgraExecutionSession::completeExternalMemory(
    CgraExternalMemoryRequestId request, CgraExternalMemoryResponse response) {
  if (!impl_)
    return invalid("CGRA execution session is empty");
  // Requests are pipelined, so a response may arrive while the session is
  // runnable or awaiting another external event. Only a finished session has
  // no external memory left to answer.
  const bool live =
      impl_->lifecycle == SpatialExecutionSessionState::Runnable ||
      impl_->lifecycle ==
          SpatialExecutionSessionState::WaitingForExternalMemory ||
      impl_->lifecycle ==
          SpatialExecutionSessionState::WaitingForExternalStreamInput;
  if (impl_->resultTaken || !live) {
    impl_->lifecycle = SpatialExecutionSessionState::Failed;
    return invalid("CGRA execution session has no pending external memory");
  }
  if (llvm::Error error = impl_->runtime->completeExternalMemory(
          std::move(request), std::move(response))) {
    impl_->lifecycle = SpatialExecutionSessionState::Failed;
    return error;
  }
  if (impl_->lifecycle ==
          SpatialExecutionSessionState::WaitingForExternalMemory &&
      !impl_->runtime->waitingForExternalMemory())
    impl_->lifecycle = SpatialExecutionSessionState::Runnable;
  return llvm::Error::success();
}

const std::optional<SpatialStreamInputRequest> &
CgraExecutionSession::pendingStreamInput() const {
  static const std::optional<SpatialStreamInputRequest> empty;
  return impl_ ? impl_->streamInputs.pending : empty;
}

llvm::Error CgraExecutionSession::completeStreamInput(
    const SpatialStreamInputRequest &request,
    const CanonicalValueSequence &value) {
  if (!impl_ || impl_->resultTaken ||
      impl_->lifecycle != SpatialExecutionSessionState::WaitingForExternalStreamInput)
    return invalid("CGRA session has no pending stream input");
  const SpatialEventCoordinate readyCoordinate = request.readyCoordinate;
  llvm::SmallVector<detail::GraphIngressEmission, 1> ingress;
  impl_->dynamicState.graphIngressCapture = &ingress;
  llvm::scope_exit clearCapture(
      [&] { impl_->dynamicState.graphIngressCapture = nullptr; });
  if (llvm::Error error = impl_->streamInputs.complete(
          impl_->dynamicState, impl_->context, *impl_->runtimeInput,
          request, value)) {
    impl_->lifecycle = SpatialExecutionSessionState::Failed;
    return error;
  }
  auto coordinate = nextSpatialDelta(readyCoordinate);
  if (!coordinate) {
    impl_->lifecycle = SpatialExecutionSessionState::Failed;
    return coordinate.takeError();
  }
  if (llvm::Error error =
          impl_->runtime->appendGraphIngress(*coordinate, ingress)) {
    impl_->lifecycle = SpatialExecutionSessionState::Failed;
    return error;
  }
  impl_->lifecycle = SpatialExecutionSessionState::Runnable;
  return llvm::Error::success();
}

const CanonicalSimulationRuntimeInput *
CgraExecutionSession::retiredRuntimeInput() const {
  return impl_ && impl_->lifecycle == SpatialExecutionSessionState::Retired
             ? impl_->runtimeInput
             : nullptr;
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
      *impl_->runtimeInput, impl_->context, impl_->prepared->dataflow.view());
  if (!observations)
    return observations.takeError();
  auto launch = launchCoordinate();
  if (!launch)
    return launch.takeError();
  auto activity = impl_->finishActivity(*impl_->lastCoordinate);
  if (!activity)
    return activity.takeError();
  impl_->resultTaken = true;
  return RetiredCgraSimulation{
      std::move(*observations),
      SpatialProgressObservations{std::move(*launch), impl_->graphRetirement,
                                  *impl_->lastCoordinate},
      impl_->counters, std::move(*activity)};
}

llvm::Expected<HaltedCgraSimulation>
CgraExecutionSession::takeHaltedSimulation() {
  if (!impl_)
    return invalid("CGRA execution session is empty");
  if (impl_->resultTaken)
    return invalid("CGRA execution result was already taken");
  if (impl_->lifecycle != SpatialExecutionSessionState::Halted ||
      !impl_->closedWait)
    return llvm::createStringError(
        std::errc::state_not_recoverable,
        "CGRA execution session has no proven Halted result");

  auto observations = detail::projectHaltedFunctionalObservations(
      impl_->context.graphOp, impl_->dynamicState, *impl_->workload,
      *impl_->runtimeInput, impl_->context, impl_->prepared->dataflow.view());
  if (!observations)
    return observations.takeError();
  auto launch = launchCoordinate();
  if (!launch)
    return launch.takeError();
  const SpatialEventCoordinate terminal =
      impl_->lastCoordinate ? *impl_->lastCoordinate : *launch;
  auto activity = impl_->finishActivity(terminal);
  if (!activity)
    return activity.takeError();
  impl_->resultTaken = true;
  return HaltedCgraSimulation{
      std::move(*observations),
      SpatialProgressObservations{std::move(*launch), impl_->graphRetirement,
                                  terminal},
      impl_->counters, std::move(*activity)};
}

llvm::Expected<PreparedCgraWorkloadExecution> prepareCgraWorkloadExecution(
    const PreparedCgraExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput) {
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
  auto context = detail::resolveLaunchContext(prepared.impl_->dataflow.view(),
                                              spatial->launchRef);
  if (!context)
    return context.takeError();
  if (std::optional<std::string> reason =
          detail::unsupportedTypedDfgInput(workload, runtimeInput, *context))
    return llvm::createStringError(std::errc::not_supported, "%s",
                                   reason->c_str());

  return PreparedCgraWorkloadExecution(
      std::make_unique<PreparedCgraWorkloadExecution::Impl>(
          prepared.impl_, *graphExecution, std::move(*context),
          workload.identity(), runtimeInput.identity()));
}

llvm::Expected<CgraExecutionSession>
startCgraExecutionSession(const PreparedCgraExecution &prepared,
                          const CanonicalSimulationWorkload &workload,
                          const CanonicalSimulationRuntimeInput &runtimeInput,
                          std::optional<TraceCaptureLevel> traceLevel,
                          CgraExternalMemoryProvider *externalMemoryProvider,
                          llvm::ArrayRef<std::uint64_t> liveStreamInputs,
                          std::optional<ActivityWindow> fabricActivityWindow) {
  auto preparedWorkload =
      prepareCgraWorkloadExecution(prepared, workload, runtimeInput);
  if (!preparedWorkload)
    return preparedWorkload.takeError();
  return startCgraExecutionSession(*preparedWorkload, workload, runtimeInput,
                                   traceLevel, externalMemoryProvider,
                                   liveStreamInputs, fabricActivityWindow);
}

llvm::Expected<CgraExecutionSession>
startCgraExecutionSession(const PreparedCgraWorkloadExecution &prepared,
                          const CanonicalSimulationWorkload &workload,
                          const CanonicalSimulationRuntimeInput &runtimeInput,
                          std::optional<TraceCaptureLevel> traceLevel,
                          CgraExternalMemoryProvider *externalMemoryProvider,
                          llvm::ArrayRef<std::uint64_t> liveStreamInputs,
                          std::optional<ActivityWindow> fabricActivityWindow) {
  if (!prepared.impl_ || !prepared.impl_->prepared ||
      !prepared.impl_->graphExecution)
    return invalid("prepared CGRA workload execution is empty");
  if (prepared.impl_->workload != workload.identity() ||
      prepared.impl_->runtimeInput != runtimeInput.identity())
    return invalid("prepared CGRA workload execution has foreign inputs");
  const SpatialSimulationWorkload *spatial = workload.spatial();
  if (!spatial)
    return invalid("prepared CGRA workload execution is not Spatial");
  const detail::PreparedCgraGraph &graphExecution =
      *prepared.impl_->graphExecution;

  auto impl = std::make_unique<CgraExecutionSession::Impl>(
      prepared.impl_->prepared, workload, runtimeInput, graphExecution,
      prepared.impl_->context, traceLevel);

  if (llvm::Error error =
          impl->streamInputs.initialize(runtimeInput, liveStreamInputs))
    return std::move(error);
  llvm::SmallVector<detail::GraphIngressEmission, 4> ingress;
  impl->dynamicState.graphIngressCapture = &ingress;
  llvm::scope_exit clearCapture(
      [&] { impl->dynamicState.graphIngressCapture = nullptr; });
  if (llvm::Error error = detail::initializeTypedGraphExecutionState(
          impl->dynamicState, graphExecution.execution, impl->context.graphOp,
          workload, runtimeInput, impl->context))
    return std::move(error);
  clearCapture.release();
  impl->dynamicState.graphIngressCapture = nullptr;

  auto launch = launchCoordinate();
  if (!launch)
    return launch.takeError();
  if (fabricActivityWindow) {
    auto activity = detail::CgraFabricActivityRuntime::create(
        impl->prepared->fabric.view(), impl->prepared->executionPlan,
        *fabricActivityWindow, *launch);
    if (!activity)
      return activity.takeError();
    impl->activity = std::move(*activity);
  }
  auto runtime = detail::CgraGraphActivationRuntime::create(
      prepared.impl_->prepared->executionPlan,
      prepared.impl_->prepared->dataflow.view(), spatial->launchRef,
      graphExecution.graph, graphExecution.execution, graphExecution.transport,
      impl->dynamicState, traceLevel == TraceCaptureLevel::Microarchitecture,
      externalMemoryProvider, impl->activity.get());
  if (!runtime)
    return runtime.takeError();
  impl->runtime.emplace(std::move(*runtime));
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
    CgraExternalMemoryProvider *externalMemoryProvider,
    std::optional<ActivityWindow> fabricActivityWindow) {
  if (maxEventFrames == 0)
    return invalid("CGRA simulation requires a positive event-frame limit");
  auto preparedWorkload =
      prepareCgraWorkloadExecution(prepared, workload, runtimeInput);
  if (!preparedWorkload)
    return preparedWorkload.takeError();
  return simulateCgraWorkload(*preparedWorkload, workload, runtimeInput,
                              maxEventFrames, executionDeadline,
                              externalMemoryProvider, fabricActivityWindow);
}

llvm::Expected<CgraSimulationOutcome> simulateCgraWorkload(
    const PreparedCgraWorkloadExecution &prepared,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    std::uint64_t maxEventFrames,
    std::optional<std::chrono::steady_clock::time_point> executionDeadline,
    CgraExternalMemoryProvider *externalMemoryProvider,
    std::optional<ActivityWindow> fabricActivityWindow) {
  if (maxEventFrames == 0)
    return invalid("CGRA simulation requires a positive event-frame limit");
  auto session = startCgraExecutionSession(prepared, workload, runtimeInput,
                                           std::nullopt, externalMemoryProvider,
                                           {}, fabricActivityWindow);
  if (!session)
    return session.takeError();
  auto advanced = session->advance(maxEventFrames, executionDeadline);
  if (!advanced)
    return advanced.takeError();
  SpatialExecutionSessionState state = *advanced;
  if (state == SpatialExecutionSessionState::WaitingForExternalMemory)
    return invalid("deferred CGRA memory requires an execution session");
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
  } else if (state == SpatialExecutionSessionState::Halted) {
    auto halted = session->takeHaltedSimulation();
    if (!halted)
      return halted.takeError();
    result.halted = std::move(*halted);
  }
  return result;
}

} // namespace loom::sim

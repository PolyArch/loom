#include "Simulator/CGRASimulator.h"

#include "CGRAPreparedExecutionInternal.h"
#include "CgraGraphActivationRuntime.h"
#include "SimulationWireInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <system_error>
#include <utility>

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
    if (llvm::Error error = detail::validateGraphRetirementBoundary(
            context.graphOp, graphExecution->execution, dynamicState))
      return error;
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
      if (detail::hasPendingVectorGroups(dynamicState)) {
        lifecycle = SpatialExecutionSessionState::Failed;
        return invalid("CGRA execution retired with incomplete vector state");
      }
      lifecycle = SpatialExecutionSessionState::Retired;
      return llvm::Error::success();
    }

    lifecycle = SpatialExecutionSessionState::Halted;
    closedWait = CgraClosedWaitSetDiagnostic{
        runtime->pendingActorFiringCount(), runtime->pendingTransferCount(),
        runtime->pendingPhysicalActionCount(), graphRetirement.has_value()};
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
      switch (event.kind) {
      case detail::CgraPhysicalLifecycleKind::Requested:
        ++impl_->counters.physicalRequestCount;
        break;
      case detail::CgraPhysicalLifecycleKind::Granted:
        ++impl_->counters.physicalGrantCount;
        break;
      case detail::CgraPhysicalLifecycleKind::Committed:
        break;
      case detail::CgraPhysicalLifecycleKind::Retired:
        ++impl_->counters.physicalRetirementCount;
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
                          std::optional<TraceCaptureLevel> traceLevel) {
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
      traceLevel == TraceCaptureLevel::Microarchitecture);
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
    std::optional<std::chrono::steady_clock::time_point> executionDeadline) {
  if (maxEventFrames == 0)
    return invalid("CGRA simulation requires a positive event-frame limit");
  auto session = startCgraExecutionSession(prepared, workload, runtimeInput);
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

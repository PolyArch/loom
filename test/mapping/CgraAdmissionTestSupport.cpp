#include "CgraAdmissionTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/Models/SimulationComparison.h"
#include "Simulator/CGRAAdmission.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "CGRA admission test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

bool rejected(llvm::Expected<loom::sim::PreparedCgraExecution> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

struct TraceEventPoint final {
  loom::sim::SpatialEventCoordinate coordinate;
  std::vector<std::uint8_t> key;
};

std::vector<TraceEventPoint>
projectTrace(const loom::sim::SpatialDiagnosticTrace &trace,
             loom::sim::TraceCaptureLevel level) {
  std::vector<TraceEventPoint> result;
  for (const loom::sim::SpatialTraceFrame &frame : trace.frames)
    for (const loom::sim::SpatialTraceEvent &event : frame.events)
      if (loom::sim::minimumTraceCaptureLevel(event) <= level)
        result.push_back(
            {frame.coordinate,
             take(loom::sim::canonicalSpatialTraceEventKey(event))});
  return result;
}

bool sameTraceProjection(llvm::ArrayRef<TraceEventPoint> lhs,
                         llvm::ArrayRef<TraceEventPoint> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (loom::sim::compareSpatialEventCoordinates(left.coordinate,
                                                  right.coordinate) != 0 ||
        left.key != right.key)
      return false;
  return true;
}

bool sameTransition(const loom::sim::ActorTransitionOccurrenceRef &lhs,
                    const loom::sim::ActorTransitionOccurrenceRef &rhs) {
  return lhs.invocation.invocationOrdinal == rhs.invocation.invocationOrdinal &&
         lhs.actor == rhs.actor &&
         lhs.transitionOrdinal == rhs.transitionOrdinal;
}

bool sameAction(const loom::sim::PhysicalActionOccurrenceRef &lhs,
                const loom::sim::PhysicalActionOccurrenceRef &rhs) {
  const auto *left =
      std::get_if<loom::sim::TransitionPhysicalActionParent>(&lhs.parent);
  const auto *right =
      std::get_if<loom::sim::TransitionPhysicalActionParent>(&rhs.parent);
  return left && right && sameTransition(left->transition, right->transition) &&
         lhs.localActionOrdinal == rhs.localActionOrdinal;
}

} // namespace

void loom::test::exerciseCgraAdmission(
    const ArtifactRootReference &dataflowReference,
    const ArtifactRootReference &fabricReference,
    const ArtifactRootReference &spatialMappingReference,
    const ArtifactRootReference &foreignFabricReference,
    const ArtifactStore &store, const BlobStore &blobs, bool expectPhysicalTags,
    bool expectCausalComputeRelease) {
  auto dataflow =
      take(::dataflow::importCanonicalDataflow(dataflowReference, store));
  auto view = take(dataflow.view());
  if (view.rootThreadLaunches().size() != 1 ||
      view.staticGraphLaunches().size() != 1)
    fail("fixture does not have one rooted graph launch");
  const ::dataflow::RootedGraphLaunchRef launch{
      view.rootThreadLaunches().front().ref,
      view.staticGraphLaunches().front().ref};

  sim::SpatialSimulationWorkload workloadDraft{launch};
  workloadDraft.valueInputPlan = {sim::RuntimeValueInput{}};
  workloadDraft.observableContract.valueResults = {0};
  auto workload = take(sim::finalizeSimulationWorkload(workloadDraft, view));
  sim::SpatialSimulationRuntimeInputDraft runtimeDraft{workload.identity()};
  runtimeDraft.runtimeValues = {
      {0, {1, {sim::SemanticLane::defined(llvm::APInt(32, 7))}}}};
  auto runtime =
      take(sim::finalizeSimulationRuntimeInput(runtimeDraft, workload, view));
  const ArtifactRootReference workloadReference =
      take(sim::publishSimulationWorkload(workload, store));
  const ArtifactRootReference runtimeReference =
      take(sim::publishSimulationRuntimeInput(runtime, store));

  auto prepared = take(sim::prepareCgraExecution(
      dataflowReference, fabricReference, spatialMappingReference, store));
  const sim::CgraExecutionPlanSummary summary = prepared.summary();
  if (summary.mappedGraphCount != 1 || summary.computeActorCount != 1 ||
      summary.actorTransitionCount == 0 ||
      summary.computeTransitionPhysicalUseCount == 0 ||
      summary.physicalUseCount == 0 || summary.resourceOwnerCount == 0 ||
      summary.claimCount == 0 || summary.routeTreeCount == 0 ||
      summary.routeNodeCount == 0 || summary.routeSinkCount == 0 ||
      summary.selectedTraversalCount == 0 ||
      (expectPhysicalTags && (summary.physicalTagSegmentCount == 0 ||
                              summary.taggedRouteNodeCount == 0)))
    fail("CGRA preparation did not freeze selected compute/transport facts");
  if (summary.computeTransitionPhysicalUseCount +
          summary.memoryTransitionPhysicalUseCount +
          summary.producedPhysicalUseCount + summary.consumedPhysicalUseCount +
          summary.traversalPhysicalUseCount !=
      summary.physicalUseCount)
    fail("CGRA preparation did not partition every selected ResourceUse");
  const auto graph =
      take(sim::admitCgraSpatialSimulation(prepared, workload, runtime));
  if (graph != view.graphs().front().ref)
    fail("CGRA admission resolved a different graph");

  auto session = take(sim::startCgraExecutionSession(
      prepared, workload, runtime, sim::TraceCaptureLevel::Semantic));
  while (session.state() == sim::SpatialExecutionSessionState::Runnable)
    take(session.advance(/*maxEventFrames=*/1));
  if (session.state() != sim::SpatialExecutionSessionState::Retired)
    fail("CGRA session did not retire the mapped graph");
  auto retired = take(session.takeRetiredSimulation());
  if (retired.counters.actorCommitCount == 0 ||
      retired.counters.actorRetirementCount == 0 ||
      retired.counters.tokenPublicationCount == 0 ||
      retired.counters.physicalRequestCount == 0 ||
      retired.counters.physicalGrantCount == 0 ||
      retired.counters.physicalRetirementCount == 0)
    fail("CGRA session bypassed selected execution lifecycles");
  if (retired.observations.valueResults.size() != 1)
    fail("CGRA session projected the wrong value-result count");
  const auto *published = std::get_if<sim::PublishedValueResult>(
      &retired.observations.valueResults.front());
  if (!published || published->value.tokenCount != 1 ||
      published->value.lanes.size() != 1 ||
      published->value.lanes.front().state != sim::SemanticState::Defined ||
      published->value.lanes.front().bits != llvm::APInt(32, 7))
    fail("CGRA session projected the wrong functional value");
  const auto &trace = session.diagnosticTrace();
  if (!trace || trace->level != sim::TraceCaptureLevel::Semantic ||
      trace->frames.empty())
    fail("CGRA session did not retain the requested semantic trace");
  std::uint64_t committed = 0;
  std::uint64_t retiredActors = 0;
  std::uint64_t publishedTokens = 0;
  for (const sim::SpatialTraceFrame &frame : trace->frames)
    for (const sim::SpatialTraceEvent &event : frame.events) {
      committed += std::holds_alternative<sim::ActorCommittedTraceEvent>(event);
      retiredActors +=
          std::holds_alternative<sim::ActorRetiredTraceEvent>(event);
      publishedTokens +=
          std::holds_alternative<sim::TokenPublishedTraceEvent>(event);
      if (sim::minimumTraceCaptureLevel(event) >
          sim::TraceCaptureLevel::Semantic)
        fail("semantic trace retained a microarchitecture event");
    }
  if (committed != retired.counters.actorCommitCount ||
      retiredActors != retired.counters.actorRetirementCount ||
      publishedTokens != retired.counters.tokenPublicationCount)
    fail("CGRA semantic trace is not lifecycle-complete");

  auto microarchitectureSession = take(sim::startCgraExecutionSession(
      prepared, workload, runtime, sim::TraceCaptureLevel::Microarchitecture));
  while (microarchitectureSession.state() ==
         sim::SpatialExecutionSessionState::Runnable)
    take(microarchitectureSession.advance(/*maxEventFrames=*/1));
  if (microarchitectureSession.state() !=
      sim::SpatialExecutionSessionState::Retired)
    fail("CGRA microarchitecture trace execution did not retire");
  auto microarchitectureRetired =
      take(microarchitectureSession.takeRetiredSimulation());
  const auto &microarchitectureTrace =
      microarchitectureSession.diagnosticTrace();
  if (!microarchitectureTrace || microarchitectureTrace->level !=
                                     sim::TraceCaptureLevel::Microarchitecture)
    fail("CGRA session did not retain the requested microarchitecture trace");
  std::uint64_t requested = 0;
  std::uint64_t granted = 0;
  std::uint64_t retiredPhysical = 0;
  for (const sim::SpatialTraceFrame &frame : microarchitectureTrace->frames)
    for (const sim::SpatialTraceEvent &event : frame.events) {
      requested +=
          std::holds_alternative<sim::PhysicalRequestedTraceEvent>(event);
      granted += std::holds_alternative<sim::PhysicalGrantedTraceEvent>(event);
      retiredPhysical +=
          std::holds_alternative<sim::PhysicalRetiredTraceEvent>(event);
    }
  if (requested != microarchitectureRetired.counters.physicalRequestCount ||
      granted != microarchitectureRetired.counters.physicalGrantCount ||
      retiredPhysical !=
          microarchitectureRetired.counters.physicalRetirementCount)
    fail("CGRA microarchitecture trace is not physical-lifecycle complete");
  if (expectCausalComputeRelease) {
    std::vector<sim::PhysicalActionOccurrenceRef> causalActions;
    std::vector<sim::ActorTransitionOccurrenceRef> publishedTransitions;
    bool observedRetirement = false;
    for (const sim::SpatialTraceFrame &frame : microarchitectureTrace->frames) {
      for (const sim::SpatialTraceEvent &event : frame.events) {
        const auto *published =
            std::get_if<sim::TokenPublishedTraceEvent>(&event);
        if (!published)
          continue;
        const auto *result =
            std::get_if<sim::ActorResultTokenOccurrenceRef>(&published->token);
        if (result)
          publishedTransitions.push_back(result->transition);
      }
      for (const sim::SpatialTraceEvent &event : frame.events) {
        if (const auto *physical =
                std::get_if<sim::PhysicalRequestedTraceEvent>(&event)) {
          const auto *use =
              std::get_if<sim::PhysicalUseTarget>(&physical->target);
          if (use && use->usePattern.owner.catalog().kind() ==
                         fabric::FabricInventoryOwnerKind::FuOccurrenceNode)
            causalActions.push_back(physical->action);
          continue;
        }
        const auto *retired =
            std::get_if<sim::PhysicalRetiredTraceEvent>(&event);
        if (!retired || llvm::none_of(causalActions, [&](const auto &action) {
              return sameAction(action, retired->action);
            }))
          continue;
        const auto *parent = std::get_if<sim::TransitionPhysicalActionParent>(
            &retired->action.parent);
        if (!parent ||
            llvm::none_of(publishedTransitions, [&](const auto &published) {
              return sameTransition(published, parent->transition);
            }))
          fail("one-cycle physical use retired before token publication");
        observedRetirement = true;
      }
    }
    if (causalActions.empty() || !observedRetirement)
      fail("CGRA trace lost its causally released one-cycle physical use");
  }
  if (!sameTraceProjection(
          projectTrace(*trace, sim::TraceCaptureLevel::Semantic),
          projectTrace(*microarchitectureTrace,
                       sim::TraceCaptureLevel::Semantic)))
    fail("CGRA microarchitecture trace does not include the semantic trace");

  auto firingSession = take(sim::startCgraExecutionSession(
      prepared, workload, runtime, sim::TraceCaptureLevel::Firing));
  while (firingSession.state() == sim::SpatialExecutionSessionState::Runnable)
    take(firingSession.advance(/*maxEventFrames=*/1));
  if (firingSession.state() != sim::SpatialExecutionSessionState::Retired)
    fail("CGRA firing trace execution did not retire");
  (void)take(firingSession.takeRetiredSimulation());
  const auto &firingTrace = firingSession.diagnosticTrace();
  if (!firingTrace ||
      !sameTraceProjection(
          projectTrace(*firingTrace, sim::TraceCaptureLevel::Firing),
          projectTrace(*trace, sim::TraceCaptureLevel::Firing)))
    fail("CGRA semantic trace does not include the firing trace");

  auto limited = take(sim::simulateCgraWorkload(prepared, workload, runtime,
                                                /*maxEventFrames=*/1));
  if (limited.state != sim::SpatialExecutionSessionState::StoppedByLimit)
    fail("CGRA event budget did not produce StoppedByLimit");

  auto preparedDfg = take(evaluation::models::prepareDfgSimulationEvaluation(
      dataflowReference, workloadReference, runtimeReference,
      defaultResolvedConfig(), store));
  auto dfgEvidence = take(evaluation::models::evaluateDfgSimulation(
      preparedDfg, {128, std::nullopt}, store, blobs));
  if (dfgEvidence.outcomeKind() != evaluation::EvidenceOutcomeKind::Completed ||
      dfgEvidence.outputBindings().size() != 1 ||
      dfgEvidence.outputBindings().front().artifacts.size() != 1)
    fail("DFG comparison reference did not produce one execution");

  auto preparedCgra = take(evaluation::models::prepareCgraSimulationEvaluation(
      dataflowReference, fabricReference, spatialMappingReference,
      workloadReference, runtimeReference, defaultResolvedConfig(), store));
  auto cgraEvidence = take(evaluation::models::evaluateCgraSimulation(
      preparedCgra, {128, std::nullopt}, store, blobs));
  if (cgraEvidence.outcomeKind() !=
          evaluation::EvidenceOutcomeKind::Completed ||
      cgraEvidence.outputBindings().size() != 1 ||
      cgraEvidence.outputBindings().front().artifacts.size() != 1)
    fail("CGRA Evaluation did not produce one execution");

  auto comparison =
      take(evaluation::models::prepareSimulationComparisonEvaluation(
          dfgEvidence.outputBindings().front().artifacts.front(),
          preparedDfg.resolution,
          cgraEvidence.outputBindings().front().artifacts.front(),
          preparedCgra.resolution, defaultResolvedConfig(), store));
  auto comparisonEvidence = take(
      evaluation::models::evaluateSimulationComparison(comparison, store,
                                                       blobs));
  const auto *completed =
      std::get_if<evaluation::CompletedEvidence>(&comparisonEvidence.outcome());
  if (!completed || completed->findingResults.size() != 1 ||
      !std::holds_alternative<evaluation::AbsentFinding>(
          completed->findingResults.front().result))
    fail("exact DFG/CGRA observations did not compare equal");

  const ArtifactRootReference cgraExecutionReference =
      cgraEvidence.outputBindings().front().artifacts.front();
  auto importedCgra = take(sim::importSimulationExecution(
      cgraExecutionReference, preparedCgra.resolution, store));
  sim::SpatialFunctionalObservations changed =
      importedCgra.functionalObservations();
  auto *changedValue =
      std::get_if<sim::PublishedValueResult>(&changed.valueResults.front());
  if (!changedValue)
    fail("CGRA comparison fixture has no published result");
  changedValue->value.lanes.front().bits = llvm::APInt(32, 8);
  sim::SpatialSimulationExecution mismatched{
      importedCgra.request(),
      sim::RetiredExecution{},
      std::move(changed),
      importedCgra.progressObservations(),
      {}};
  auto finalizedMismatch = take(sim::finalizeSimulationExecution(
      mismatched, preparedCgra.resolution, store));
  const ArtifactRootReference mismatchReference =
      take(sim::publishSimulationExecution(finalizedMismatch, store));
  auto mismatchComparison =
      take(evaluation::models::prepareSimulationComparisonEvaluation(
          dfgEvidence.outputBindings().front().artifacts.front(),
          preparedDfg.resolution, mismatchReference, preparedCgra.resolution,
          defaultResolvedConfig(), store));
  auto mismatchEvidence = take(evaluation::models::evaluateSimulationComparison(
      mismatchComparison, store, blobs));
  const auto *mismatchCompleted =
      std::get_if<evaluation::CompletedEvidence>(&mismatchEvidence.outcome());
  if (!mismatchCompleted || mismatchCompleted->findingResults.size() != 1 ||
      !std::holds_alternative<evaluation::PresentFinding>(
          mismatchCompleted->findingResults.front().result))
    fail("deterministic DFG/CGRA mismatch was not reported");

  if (!rejected(sim::prepareCgraExecution(dataflowReference,
                                          foreignFabricReference,
                                          spatialMappingReference, store)))
    fail("CGRA preparation accepted a foreign Fabric");
}

void loom::test::exerciseCgraMemoryAdmission(
    const ArtifactRootReference &dataflowReference,
    const ArtifactRootReference &fabricReference,
    const ArtifactRootReference &spatialMappingReference,
    const ArtifactStore &store) {
  auto prepared = take(sim::prepareCgraExecution(
      dataflowReference, fabricReference, spatialMappingReference, store));
  const sim::CgraExecutionPlanSummary summary = prepared.summary();
  if (summary.mappedGraphCount != 1 || summary.memoryActorCount != 1 ||
      summary.memoryRootedUseCount != 2 ||
      summary.memoryChildTransactionCount == 0 ||
      summary.memoryResultAssemblyCount == 0 ||
      summary.memoryTransitionPhysicalUseCount != 2)
    fail("CGRA preparation did not freeze selected memory execution facts");
}

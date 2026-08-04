#include "CgraAdmissionTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
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

} // namespace

void loom::test::exerciseCgraAdmission(
    const ArtifactRootReference &dataflowReference,
    const ArtifactRootReference &fabricReference,
    const ArtifactRootReference &spatialMappingReference,
    const ArtifactRootReference &foreignFabricReference,
    const ArtifactStore &store, bool expectPhysicalTags) {
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

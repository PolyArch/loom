#include "CgraAdmissionTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Simulator/CGRAAdmission.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <utility>

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

  if (!rejected(sim::prepareCgraExecution(dataflowReference,
                                          foreignFabricReference,
                                          spatialMappingReference, store)))
    fail("CGRA preparation accepted a foreign Fabric");
}

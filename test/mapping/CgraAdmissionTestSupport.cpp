#include "CgraAdmissionTestSupport.h"

#include "CGRAExecutionPlan.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/Models/CgraClosedWait.h"
#include "Evaluation/Models/DfgSimulation.h"
#include "Evaluation/Models/SimulationComparison.h"
#include "Evaluation/StandardFindings.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/IR/MappingSchema.h"
#include "Simulator/CGRAAdmission.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/CgraClosedWaitCertificate.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <map>
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

template <typename T> bool rejected(llvm::Expected<T> value) {
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

void verifySameInputDistinctTemporalSwitchRows(
    const loom::sim::detail::CgraFrozenExecutionPlan &plan,
    const loom::fabric::FabricArtifactView &fabric) {
  struct Activation final {
    loom::fabric::FabricSwitchOccurrenceRef owner;
    loom::fabric::FabricOrdinal input = 0;
    llvm::APInt tag = llvm::APInt(1, 0);
    std::uint64_t instance = 0;
  };
  std::vector<Activation> activations;
  for (const auto &route : plan.transport.routes) {
    if (route.nodeOffset > plan.transport.routeNodes.size() ||
        route.nodeCount > plan.transport.routeNodes.size() - route.nodeOffset)
      fail("CGRA route-node slice is malformed");
    for (const auto &node :
         llvm::ArrayRef(plan.transport.routeNodes)
             .slice(route.nodeOffset, route.nodeCount)) {
      if (node.incomingTraversalOrdinal ==
          loom::sim::detail::invalidCgraTransportOrdinal)
        continue;
      if (node.incomingTraversalOrdinal >= plan.transport.traversals.size())
        fail("CGRA route node names an absent traversal");
      const auto &traversal =
          plan.transport.traversals[node.incomingTraversalOrdinal].reference;
      const auto *sw =
          std::get_if<loom::fabric::FabricSwitchTraversalPayload>(
              &traversal.payload);
      if (!sw || fabric.switchSchedule(sw->owner) != ::fabric::Schedule::Temporal)
        continue;
      if (node.physicalTagOrdinal >= plan.transport.physicalTags.size() ||
          node.impliedUseOffset > plan.transport.traversalUses.size() ||
          node.impliedUseCount == 0 ||
          node.impliedUseCount >
              plan.transport.traversalUses.size() - node.impliedUseOffset)
        fail("Temporal switch route node has an incomplete activation");
      const auto uses = llvm::ArrayRef(plan.transport.traversalUses)
                            .slice(node.impliedUseOffset,
                                   node.impliedUseCount);
      const std::uint64_t instance = uses.front().activationInstanceOrdinal;
      if (instance == loom::sim::detail::invalidCgraTransportOrdinal ||
          llvm::any_of(uses, [&](const auto &use) {
            return use.activationInstanceOrdinal != instance;
          }))
        fail("Temporal switch row/input did not form one atomic activation");
      activations.push_back({sw->owner, sw->input,
                             plan.transport
                                 .physicalTags[node.physicalTagOrdinal]
                                 .value,
                             instance});
    }
  }

  bool observedDistinctRows = false;
  for (std::size_t lhs = 0; lhs != activations.size(); ++lhs)
    for (std::size_t rhs = lhs + 1; rhs != activations.size(); ++rhs) {
      if (activations[lhs].owner != activations[rhs].owner ||
          activations[lhs].input != activations[rhs].input)
        continue;
      if (activations[lhs].tag == activations[rhs].tag) {
        if (activations[lhs].instance != activations[rhs].instance)
          fail("one Temporal switch row/input gained multiple activations");
        continue;
      }
      observedDistinctRows = true;
      if (activations[lhs].instance == activations[rhs].instance)
        fail("different Temporal switch rows shared one activation");
    }
  if (!observedDistinctRows)
    fail("fixture did not expose two tag rows at one Temporal switch input");
}

void verifySpatialSwitchActivationIdentity(
    const loom::sim::detail::CgraFrozenExecutionPlan &plan,
    const loom::fabric::FabricArtifactView &fabric) {
  using OwnerKey = std::vector<std::uint8_t>;
  std::map<OwnerKey, std::map<loom::fabric::FabricOrdinal, std::uint64_t>>
      instances;
  for (const auto &traversal : plan.transport.traversals) {
    const auto *sw =
        std::get_if<loom::fabric::FabricSwitchTraversalPayload>(
            &traversal.reference.payload);
    if (!sw || fabric.switchSchedule(sw->owner) != ::fabric::Schedule::Spatial)
      continue;
    if (traversal.impliedUseOffset > plan.transport.traversalUses.size() ||
        traversal.impliedUseCount == 0 ||
        traversal.impliedUseCount >
            plan.transport.traversalUses.size() - traversal.impliedUseOffset)
      fail("Spatial switch traversal has an incomplete activation");
    const auto uses = llvm::ArrayRef(plan.transport.traversalUses)
                          .slice(traversal.impliedUseOffset,
                                 traversal.impliedUseCount);
    const std::uint64_t instance = uses.front().activationInstanceOrdinal;
    if (instance == loom::sim::detail::invalidCgraTransportOrdinal ||
        llvm::any_of(uses, [&](const auto &use) {
          return use.activationInstanceOrdinal != instance;
        }))
      fail("one Spatial switch traversal gained multiple activations");
    auto [position, inserted] =
        instances[loom::fabric::canonicalFabricBytes(sw->owner)].try_emplace(
            sw->input, instance);
    if (!inserted && position->second != instance)
      fail("one Spatial switch input gained multiple activations");
  }

  for (const auto &[owner, inputs] : instances) {
    (void)owner;
    for (auto lhs = inputs.begin(); lhs != inputs.end(); ++lhs)
      for (auto rhs = std::next(lhs); rhs != inputs.end(); ++rhs) {
        if (lhs->second == rhs->second)
          fail("different Spatial switch inputs shared one activation");
      }
  }
}

} // namespace

void loom::test::exerciseCgraAdmission(
    const ArtifactRootReference &dataflowReference,
    const ArtifactRootReference &fabricReference,
    const ArtifactRootReference &spatialMappingReference,
    const ArtifactRootReference &foreignFabricReference,
    const ArtifactStore &store, const BlobStore &blobs, bool expectPhysicalTags,
    bool expectCausalComputeRelease,
    bool expectSameInputDistinctSwitchRows) {
  auto dataflow =
      take(::dataflow::importCanonicalDataflow(dataflowReference, store));
  auto view = take(dataflow.view());
  if (view.rootThreadLaunches().size() != 1 ||
      view.staticGraphLaunches().size() != 1)
    fail("fixture does not have one rooted graph launch");
  const ::dataflow::RootedGraphLaunchRef launch{
      view.rootThreadLaunches().front().ref,
      view.staticGraphLaunches().front().ref};

  auto fabric =
      take(::loom::fabric::importEntireFabricRoot(fabricReference, store));
  auto spatial = take(
      ::loom::mapping::importSpatialMapping(spatialMappingReference, store));
  const ArtifactRootReference techReference{
      ::loom::mapping::mappingArtifactSchema.identity.str(),
      ::loom::mapping::mappingArtifactSchema.version,
      spatial.view().techMappingIdentity()};
  auto tech = take(::loom::mapping::importTechMapping(techReference, store));
  auto plan = take(sim::detail::freezeCgraExecutionPlan(
      view, tech.view(), fabric.view(), spatial.view()));
  verifySpatialSwitchActivationIdentity(plan, fabric.view());
  if (expectSameInputDistinctSwitchRows)
    verifySameInputDistinctTemporalSwitchRows(plan, fabric.view());

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
  auto preparedWorkload =
      take(sim::prepareCgraWorkloadExecution(prepared, workload, runtime));

  auto session = take(sim::startCgraExecutionSession(
      preparedWorkload, workload, runtime, sim::TraceCaptureLevel::Semantic));
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

  auto detachedPreparedWorkload = take([&] {
    auto owner = take(sim::prepareCgraExecution(
        dataflowReference, fabricReference, spatialMappingReference, store));
    return sim::prepareCgraWorkloadExecution(owner, workload, runtime);
  }());
  auto limited = take(sim::simulateCgraWorkload(
      detachedPreparedWorkload, workload, runtime, /*maxEventFrames=*/1));
  if (limited.state != sim::SpatialExecutionSessionState::StoppedByLimit)
    fail("CGRA event budget did not produce StoppedByLimit");
  sim::SpatialSimulationRuntimeInputDraft foreignRuntimeDraft{
      workload.identity()};
  foreignRuntimeDraft.runtimeValues = {
      {0, {1, {sim::SemanticLane::defined(llvm::APInt(32, 8))}}}};
  auto foreignRuntime = take(sim::finalizeSimulationRuntimeInput(
      foreignRuntimeDraft, workload, view));
  if (!rejected(sim::startCgraExecutionSession(preparedWorkload, workload,
                                               foreignRuntime)))
    fail("prepared CGRA workload execution accepted a foreign runtime input");

  auto preparedDfg = take(evaluation::models::prepareDfgSimulationEvaluation(
      dataflowReference, workloadReference, runtimeReference,
      defaultResolvedConfig(), store, blobs));
  auto dfgEvidence = take(evaluation::models::evaluateDfgSimulation(
      preparedDfg, {128, std::nullopt}, store, blobs));
  if (dfgEvidence.outcomeKind() != evaluation::EvidenceOutcomeKind::Completed ||
      dfgEvidence.outputBindings().size() != 1 ||
      dfgEvidence.outputBindings().front().artifacts.size() != 1)
    fail("DFG comparison reference did not produce one execution");

  auto preparedCgra = take(evaluation::models::prepareCgraSimulationEvaluation(
      dataflowReference, fabricReference, spatialMappingReference,
      workloadReference, runtimeReference, defaultResolvedConfig(), store,
      blobs));
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
          preparedCgra.resolution, defaultResolvedConfig(), store, blobs));
  auto comparisonEvidence =
      take(evaluation::models::evaluateSimulationComparison(comparison, store,
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
      cgraExecutionReference, preparedCgra.resolution, store, blobs));
  const auto *cgraCompleted =
      std::get_if<evaluation::CompletedEvidence>(&cgraEvidence.outcome());
  if (!cgraCompleted)
    fail("retired CGRA Evidence has no completed payload");
  const auto presentClosedWait = [] {
    return evaluation::FindingResult{
        evaluation::PresentFinding{{evaluation::FindingOccurrence::get(
            sim::TerminalWitnessRef{evaluation::ModelOutputSlotRef(0), 0})}}};
  };

  const ArtifactRootReference retiredEvidenceReference =
      take(evaluation::publishEvaluationEvidence(cgraEvidence, store));
  if (!rejected(evaluation::models::importVerifiedCgraClosedWaitEvidence(
          retiredEvidenceReference, store, blobs)))
    fail("retired CGRA Evidence was imported as a closed wait");

  auto nonCompleted = take(evaluation::EvaluationEvidence::get(
      preparedCgra.request, {{evaluation::ModelOutputSlotRef(0), {}}},
      evaluation::ExecutionFailedEvidence{
          evaluation::OutcomeReason::AdapterFailure},
      preparedCgra.resolution, store, blobs));
  const ArtifactRootReference nonCompletedReference =
      take(evaluation::publishEvaluationEvidence(nonCompleted, store));
  if (!rejected(evaluation::models::importVerifiedCgraClosedWaitEvidence(
          nonCompletedReference, store, blobs)))
    fail("non-Completed Evidence was imported as a closed wait");

  if (!rejected(evaluation::EvaluationEvidence::get(
          preparedCgra.request,
          {{evaluation::ModelOutputSlotRef(0), {cgraExecutionReference}}},
          evaluation::CompletedEvidence{cgraCompleted->metricResults, {}},
          preparedCgra.resolution, store, blobs)))
    fail("Completed CGRA Evidence omitted its mandatory finding result");
  if (!rejected(evaluation::EvaluationEvidence::get(
          preparedCgra.request,
          {{evaluation::ModelOutputSlotRef(0), {cgraExecutionReference}}},
          evaluation::CompletedEvidence{
              cgraCompleted->metricResults,
              {evaluation::FindingResult{evaluation::AbsentFinding{}},
               evaluation::FindingResult{evaluation::AbsentFinding{}}}},
          preparedCgra.resolution, store, blobs)))
    fail("Completed CGRA Evidence repeated its finding result");
  auto duplicatePresent = presentClosedWait();
  auto &duplicateOccurrences =
      std::get<evaluation::PresentFinding>(duplicatePresent.result).occurrences;
  duplicateOccurrences.push_back(evaluation::FindingOccurrence::get(
      sim::TerminalWitnessRef{evaluation::ModelOutputSlotRef(0), 0}));
  if (!rejected(evaluation::EvaluationEvidence::get(
          preparedCgra.request,
          {{evaluation::ModelOutputSlotRef(0), {cgraExecutionReference}}},
          evaluation::CompletedEvidence{cgraCompleted->metricResults,
                                        {std::move(duplicatePresent)}},
          preparedCgra.resolution, store, blobs)))
    fail("terminal CGRA finding accepted non-unique Present occurrences");

  // A structurally closed witness with the right Request owners is still not
  // runtime provenance. Forge one beside the real retired execution and prove
  // that the Evidence-only carrier rejects it by deterministic replay.
  if (spatial.view().routeTrees().empty())
    fail("CGRA provenance fixture has no logical route");
  sim::CgraClosedWaitSetDiagnostic forgedWait;
  forgedWait.ownerReferences = take(prepared.ownerReferences());
  sim::CgraClosedWaitSetDiagnostic::Transfer forgedTransfer;
  forgedTransfer.bindingOrdinal = 0;
  forgedTransfer.occurrenceOrdinal = 0;
  forgedTransfer.producer = spatial.view().routeTrees().front().logicalNet;
  forgedTransfer.blockingStorageOrdinal = 0;
  forgedWait.transfers.push_back(std::move(forgedTransfer));
  using Wait = sim::CgraClosedWaitSetDiagnostic;
  const Wait::WaitOwnerKey actor{Wait::WaitActorFiringKey{0, 0}};
  const Wait::WaitOwnerKey storage{Wait::WaitStorageQueueKey{
      Wait::WaitStorageDomain::TraversalStorage, 0,
      Wait::WaitQueueClass::global()}};
  Wait::WaitEdge actorWait;
  actorWait.from = actor;
  actorWait.to = storage;
  actorWait.kind = Wait::WaitEdgeKind::ActorOutputBackpressure;
  actorWait.bindingOrdinal = 0;
  actorWait.occurrenceOrdinal = 0;
  actorWait.storageOrdinal = 0;
  Wait::WaitEdge storageWait = actorWait;
  storageWait.from = storage;
  storageWait.to = actor;
  storageWait.kind = Wait::WaitEdgeKind::StorageConsumer;
  forgedWait.waitCertificate = {std::move(actorWait), std::move(storageWait)};
  auto forgedCertificate =
      take(sim::buildCgraClosedWaitCertificate(forgedWait));
  const auto encodedCertificate =
      take(sim::encodeCgraClosedWaitCertificate(forgedCertificate));
  std::vector<std::uint8_t> tamperedCertificateBytes = encodedCertificate;
  tamperedCertificateBytes.push_back(0);
  if (!rejected(sim::decodeCgraClosedWaitCertificate(
          tamperedCertificateBytes)))
    fail("closed-wait codec accepted tampered trailing certificate bytes");
  auto alteredCertificate = forgedCertificate;
  ++alteredCertificate.edges.front().storageCapacity;
  const auto originalDigest =
      take(sim::digestCgraClosedWaitCertificate(forgedCertificate));
  const auto alteredDigest =
      take(sim::digestCgraClosedWaitCertificate(alteredCertificate));
  if (originalDigest == alteredDigest)
    fail("closed-wait digest omitted a tampered certificate field");

  sim::SpatialSimulationExecution wrongFindingKind{
      importedCgra.request(),
      sim::HaltedExecution{
          evaluation::standard_findings::FunctionalMismatch,
          evaluation::OwnerValue::get(std::uint64_t{0})},
      importedCgra.spatialFunctionalObservations(),
      importedCgra.spatialProgressObservations(),
      {}};
  if (!rejected(sim::finalizeSimulationExecution(
          wrongFindingKind, preparedCgra.resolution, store, blobs)))
    fail("Halted CGRA execution accepted a foreign finding kind");
  sim::SpatialSimulationExecution wrongWitnessOwner{
      importedCgra.request(),
      sim::HaltedExecution{evaluation::models::CgraClosedWait,
                           evaluation::OwnerValue::get(std::uint64_t{0})},
      importedCgra.spatialFunctionalObservations(),
      importedCgra.spatialProgressObservations(),
      {}};
  if (!rejected(sim::finalizeSimulationExecution(
          wrongWitnessOwner, preparedCgra.resolution, store, blobs)))
    fail("Halted CGRA execution accepted a foreign witness owner type");

  sim::SpatialSimulationExecution forgedExecution{
      importedCgra.request(),
      sim::HaltedExecution{
          evaluation::models::CgraClosedWait,
          evaluation::OwnerValue::get(std::move(forgedCertificate))},
      importedCgra.spatialFunctionalObservations(),
      importedCgra.spatialProgressObservations(),
      {}};
  auto finalizedForged = take(sim::finalizeSimulationExecution(
      forgedExecution, preparedCgra.resolution, store, blobs));
  const ArtifactRootReference forgedExecutionReference =
      take(sim::publishSimulationExecution(finalizedForged, store));
  auto forgedEvidence = take(evaluation::EvaluationEvidence::get(
      preparedCgra.request,
      {{evaluation::ModelOutputSlotRef(0), {forgedExecutionReference}}},
      evaluation::CompletedEvidence{cgraCompleted->metricResults,
                                    {presentClosedWait()}},
      preparedCgra.resolution, store, blobs));
  const ArtifactRootReference forgedEvidenceReference =
      take(evaluation::publishEvaluationEvidence(forgedEvidence, store));
  auto forgedImport =
      evaluation::models::importVerifiedCgraClosedWaitEvidence(
          forgedEvidenceReference, store, blobs);
  if (forgedImport)
    fail("deterministic replay accepted a forged closed-wait certificate");
  const std::string forgedMessage = llvm::toString(forgedImport.takeError());
  if (!llvm::StringRef(forgedMessage)
           .contains("differs from deterministic replay"))
    fail("forged closed-wait Evidence failed for the wrong reason: " +
         forgedMessage);

  if (!rejected(evaluation::EvaluationEvidence::get(
          preparedCgra.request,
          {{evaluation::ModelOutputSlotRef(0),
            {dfgEvidence.outputBindings().front().artifacts.front()}}},
          evaluation::CompletedEvidence{cgraCompleted->metricResults,
                                        {presentClosedWait()}},
          preparedCgra.resolution, store, blobs)))
    fail("CGRA Evidence accepted a foreign execution output");

  const ArtifactRootReference foreignRuntimeReference =
      take(sim::publishSimulationRuntimeInput(foreignRuntime, store));
  auto foreignPreparedCgra =
      take(evaluation::models::prepareCgraSimulationEvaluation(
          dataflowReference, fabricReference, spatialMappingReference,
          workloadReference, foreignRuntimeReference, defaultResolvedConfig(),
          store, blobs));
  if (!rejected(evaluation::EvaluationEvidence::get(
          foreignPreparedCgra.request,
          {{evaluation::ModelOutputSlotRef(0), {forgedExecutionReference}}},
          evaluation::CompletedEvidence{cgraCompleted->metricResults,
                                        {presentClosedWait()}},
          foreignPreparedCgra.resolution, store, blobs)))
    fail("CGRA Evidence accepted an execution from a foreign Request");

  sim::SpatialFunctionalObservations changed =
      importedCgra.spatialFunctionalObservations();
  auto *changedValue =
      std::get_if<sim::PublishedValueResult>(&changed.valueResults.front());
  if (!changedValue)
    fail("CGRA comparison fixture has no published result");
  changedValue->value.lanes.front().bits = llvm::APInt(32, 8);
  sim::SpatialSimulationExecution mismatched{
      importedCgra.request(),
      sim::RetiredExecution{},
      std::move(changed),
      importedCgra.spatialProgressObservations(),
      {}};
  auto finalizedMismatch = take(sim::finalizeSimulationExecution(
      mismatched, preparedCgra.resolution, store, blobs));
  const ArtifactRootReference mismatchReference =
      take(sim::publishSimulationExecution(finalizedMismatch, store));
  auto mismatchComparison =
      take(evaluation::models::prepareSimulationComparisonEvaluation(
          dfgEvidence.outputBindings().front().artifacts.front(),
          preparedDfg.resolution, mismatchReference, preparedCgra.resolution,
          defaultResolvedConfig(), store, blobs));
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

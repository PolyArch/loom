#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/DataflowEvaluationAcquisition.h"
#include "DSE/InvocationManifest.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/Plan.h"
#include "DSE/Promotion.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "DSE/SpatialMappingEvaluationAcquisition.h"
#include "DSE/SpatialMappingFeedbackCandidateGenerator.h"
#include "DSE/StructuredOwnership.h"
#include "DSE/StructuredOwnershipInvocation.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/Transforms/DataflowRewrite.h"
#include "Evaluation/Models/CanonicalDataflowFunctional.h"
#include "Evaluation/Models/CgraSimulation.h"
#include "Evaluation/StandardFindings.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Inspection/SpatialMappingInspection.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialCanonicalSeed.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/SpatialPnrProblem.h"
#include "RootCompleteSpatialFeedbackTestSupport.h"
#include "RootCompleteSpatialPnrTestSupport.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::test::buildAlternateDataflow;
using loom::test::buildAlternativeTechSpatialCore;
using loom::test::buildDataflow;
using loom::test::buildFeedbackSpatialConfig;
using loom::test::buildSingleCandidateSpatialConfig;
using loom::test::buildSingleCandidateSpatialResolvedConfig;
using loom::test::buildSpatialConfig;
using loom::test::buildSpatialResolvedConfig;
using loom::test::buildVectorDataflow;
using loom::test::makeContext;

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "root-complete Spatial PnR generator anchor failed: "
               << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

const loom::dse::CompletedDsePlanExecution &
requireClosedPlan(const loom::dse::DsePlanExecutionOutcome &outcome,
                  std::uint64_t nodeOrdinal) {
  if (const auto *completed =
          std::get_if<loom::dse::CompletedDsePlanExecution>(&outcome))
    return *completed;
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteDsePlanExecution>(&outcome);
  const auto *reason =
      incomplete ? std::get_if<loom::dse::CandidateGeneratorIncompleteReason>(
                       &incomplete->reason())
                 : nullptr;
  if (!incomplete || incomplete->nodeOrdinal() != nodeOrdinal || !reason ||
      *reason !=
          loom::dse::CandidateGeneratorIncompleteReason::SemanticLimitReached ||
      incomplete->executionStopped())
    fail("DSE plan did not close or retain its semantic-limit prefix");
  return incomplete->availableExecution();
}

struct ImmediateStopSource final {
  static bool query(const void *) { return true; }

  loom::ExecutionControlView control() const { return {this, query}; }
};

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-root-complete-spatial-pnr", path_);
    if (error)
      fail("cannot create ArtifactStore directory: " + error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

void requireSpatialWorkSummary(
    llvm::ArrayRef<loom::dse::CandidateGeneratorWorkUnitSummary> summary,
    bool expectConsumedWork) {
  if (summary.size() != loom::dse::pnrCandidateGeneratorWorkUnits.size())
    fail("Spatial PnR work summary does not cover the owner catalog");
  bool consumedAny = false;
  for (std::size_t ordinal = 0; ordinal != summary.size(); ++ordinal) {
    if (summary[ordinal].unit.ordinal() != ordinal ||
        summary[ordinal].planned != summary[ordinal].consumed)
      fail("Spatial PnR work summary is not dense and exact");
    consumedAny |= summary[ordinal].consumed != 0;
  }
  if (consumedAny != expectConsumedWork)
    fail("Spatial PnR work summary changed empty/nonempty accounting");
  if (expectConsumedWork && summary[0].consumed == 0)
    fail("Spatial PnR omitted a required executed search domain");
}

const std::vector<loom::ArtifactRootReference> &
techMappingOutputs(const loom::dse::CandidateGeneratorProviderResult &outcome) {
  const std::vector<loom::dse::CandidateGeneratorOutputBinding> *bindings =
      nullptr;
  if (const auto *completed =
          std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
              &outcome.outcome)) {
    bindings = &completed->outputBindings;
  } else if (const auto *incomplete =
                 std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
                     &outcome.outcome);
             incomplete && incomplete->reason ==
                               loom::dse::CandidateGeneratorIncompleteReason::
                                   SemanticLimitReached) {
    bindings = &incomplete->retainedOutputBindings;
  } else {
    std::string diagnostic;
    llvm::raw_string_ostream stream(diagnostic);
    if (incomplete)
      stream << " outcome="
             << loom::dse::toString(
                    loom::dse::DsePlanIncompleteReason{incomplete->reason})
             << " retained_outputs="
             << incomplete->retainedOutputBindings.size();
    else
      stream << " outcome=unknown";
    for (const auto &[ordinal, work] : llvm::enumerate(outcome.workSummary))
      stream << " work[" << ordinal << "]={planned=" << work.planned
             << ",consumed=" << work.consumed << '}';
    fail("root-complete TechMapping fixture did not publish a usable prefix:" +
         diagnostic);
  }
  if (bindings->size() != 1)
    fail("root-complete TechMapping fixture published the wrong output shape");
  return bindings->front().artifacts;
}

loom::ArtifactRootReference
generateTechMapping(const loom::ArtifactRootReference &dataflow,
                    const loom::ArtifactRootReference &fabric,
                    loom::ArtifactStore &store, const loom::BlobStore &blobs) {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
          {dataflow}, fabric));
  auto binding =
      take(loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto &outputs = techMappingOutputs(outcome);
  if (outputs.size() != 1)
    fail("root-complete TechMapping fixture did not publish one candidate");
  return outputs.front();
}

std::vector<loom::ArtifactRootReference>
generateTechMappingSet(const loom::ArtifactRootReference &dataflow,
                       const loom::ArtifactRootReference &fabric,
                       loom::ArtifactStore &store,
                       const loom::BlobStore &blobs) {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 4;
  auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
          {dataflow}, fabric));
  auto binding =
      take(loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto &outputs = techMappingOutputs(outcome);
  if (outputs.empty())
    fail("TechMapping fixture did not publish a candidate");
  return outputs;
}

loom::ArtifactRootReference
normalizedTimingProfile(const loom::ArtifactRootReference &fabricReference,
                        loom::ArtifactStore &store) {
  auto fabric =
      take(loom::fabric::importEntireFabricRoot(fabricReference, store));
  auto profile =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          fabric.view()));
  return take(loom::fabric::publishFabricPhysicalTimingProfile(profile, store));
}

struct Fixture final {
  dataflow::CanonicalDataflowArtifact dataflow;
  loom::ArtifactRootReference dataflowReference;
  loom::fabric::FinalizedFabricRoot fabric;
  loom::ArtifactRootReference physicalTimingProfile;
  loom::ArtifactRootReference techMappingReference;
};

Fixture buildFixture(mlir::MLIRContext &context, loom::ArtifactStore &store,
                     const loom::BlobStore &blobs) {
  auto dataflow = loom::test::buildRootCompleteSpatialDataflow(context);
  auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflow, store));
  auto fabric = loom::test::buildSpatialCore(store);
  auto physicalTiming = normalizedTimingProfile(fabric.reference(), store);
  auto techMappingReference =
      generateTechMapping(dataflowReference, fabric.reference(), store, blobs);
  return {std::move(dataflow), std::move(dataflowReference), std::move(fabric),
          std::move(physicalTiming), std::move(techMappingReference)};
}

loom::ArtifactRootReference
generateSpatialMapping(const loom::ArtifactRootReference &techMapping,
                       const loom::ArtifactRootReference &fabric,
                       loom::ArtifactStore &store,
                       const loom::BlobStore &blobs) {
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          {techMapping}, fabric, normalizedTimingProfile(fabric, store)));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          buildSingleCandidateSpatialConfig()));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 1)
    fail("SpatialMapping fixture did not publish one candidate");
  return completed->outputBindings.front().artifacts.front();
}

std::vector<loom::ArtifactRootReference> generateSpatialMappingSet(
    llvm::ArrayRef<loom::ArtifactRootReference> techMappings,
    const loom::ArtifactRootReference &fabric, loom::ArtifactStore &store,
    const loom::BlobStore &blobs) {
  loom::ResolvedConfig resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.initializer.seedAttemptCount = 4;
  resolved.dse.spatialPnr.search.routing.negotiationIterationLimit = 8;
  resolved.dse.spatialPnr.search.routing.negotiation =
      loom::ResolvedPathFinderPolicy{
          loom::ResolvedPathFinderPriceKernel::Additive, 1, {3, 2}, 1};
  auto config = take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          techMappings, fabric, normalizedTimingProfile(fabric, store)));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &outcome.outcome);
  if (!completed)
    fail("SpatialMapping fixture did not complete one output binding");
  if (completed->outputBindings.size() != 1)
    fail("SpatialMapping fixture completed with the wrong output width");
  if (completed->outputBindings.front().artifacts.size() < 2)
    fail("SpatialMapping fixture published " +
         llvm::Twine(completed->outputBindings.front().artifacts.size()) +
         " distinct candidates instead of two");
  return completed->outputBindings.front().artifacts;
}

struct PublishedSpatialInputs final {
  loom::ArtifactRootReference workload;
  loom::ArtifactRootReference runtimeInput;
};

struct GeneratedSpatialFeedbackFixture final {
  loom::ArtifactRootReference mapping;
  loom::mapping::FinalizedSpatialMappingConstraintSet constraints;
};

GeneratedSpatialFeedbackFixture generateSpatialFeedbackFixture(
    const loom::ArtifactRootReference &dataflowReference,
    const loom::ArtifactRootReference &techMappingReference,
    const loom::fabric::FinalizedFabricRoot &fabric,
    loom::ArtifactStore &store) {
  auto dataflow =
      take(dataflow::importCanonicalDataflow(dataflowReference, store));
  auto dataflowView = take(dataflow.view());
  auto tech =
      take(loom::mapping::importTechMapping(techMappingReference, store));
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflowView, tech.view(), fabric.view(), store));
  auto config = buildFeedbackSpatialConfig();
  auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          fabric.view()));
  auto problem = take(loom::pnr::freezeSpatialPnrProblem(
      dataflowView, tech.view(), fabric.view(), physicalTiming, config,
      constraints.view()));
  auto outcome = loom::pnr::generateSpatialMappings(
      {dataflowView, tech.view(), fabric.view(), physicalTiming, config,
       constraints.view(), store});
  if (const auto *generated =
          std::get_if<loom::pnr::GeneratedSpatialMappings>(&outcome)) {
    for (const auto &reference : generated->candidates) {
      auto mapping =
          take(loom::mapping::importSpatialMapping(reference, store));
      auto claims = take(loom::pnr::projectSpatialMappingTraversalClaims(
          *problem, mapping.view()));
      if (claims.total != 0)
        return {reference, std::move(constraints)};
    }
    fail("feedback fixture produced no Mapping with a selected traversal "
         "claim");
  }
  if (const auto *incomplete =
          std::get_if<loom::pnr::IncompleteSpatialPnrGeneration>(&outcome))
    fail("feedback fixture Mapping is incomplete: " + incomplete->diagnostic);
  if (const auto *infeasible =
          std::get_if<loom::pnr::ProvenInfeasibleSpatialMapping>(&outcome))
    fail("feedback fixture Mapping is infeasible: " + infeasible->diagnostic);
  if (const auto *unsupported =
          std::get_if<loom::pnr::UnsupportedSpatialPnrGeneration>(&outcome))
    fail("feedback fixture Mapping is unsupported: " + unsupported->diagnostic);
  if (const auto *invalid =
          std::get_if<loom::pnr::InvalidSpatialPnrGeneration>(&outcome))
    fail("feedback fixture Mapping is invalid: " + invalid->diagnostic);
  fail("feedback fixture Mapping failed internally: " +
       std::get<loom::pnr::InternalSpatialPnrGeneration>(outcome).diagnostic);
}

PublishedSpatialInputs
publishSpatialInputs(const dataflow::CanonicalDataflowArtifact &dataflow,
                     loom::ArtifactStore &store) {
  const auto view = take(dataflow.view());
  const dataflow::RootedGraphLaunchRef launch{
      view.rootThreadLaunches().front().ref,
      view.staticGraphLaunches().front().ref};
  loom::sim::SpatialSimulationWorkload workloadDraft{launch};
  workloadDraft.valueInputPlan = {loom::sim::RuntimeValueInput{}};
  workloadDraft.observableContract.valueResults = {0};
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, view));
  loom::sim::SpatialSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  runtimeDraft.runtimeValues = {
      {0, {1, {loom::sim::SemanticLane::defined(llvm::APInt(32, 7))}}}};
  auto runtime = take(
      loom::sim::finalizeSimulationRuntimeInput(runtimeDraft, workload, view));
  return {take(loom::sim::publishSimulationWorkload(workload, store)),
          take(loom::sim::publishSimulationRuntimeInput(runtime, store))};
}

PublishedSpatialInputs
publishVectorSpatialInputs(const dataflow::CanonicalDataflowArtifact &dataflow,
                           loom::ArtifactStore &store,
                           unsigned laneWidth = 32) {
  const auto view = take(dataflow.view());
  const dataflow::RootedGraphLaunchRef launch{
      view.rootThreadLaunches().front().ref,
      view.staticGraphLaunches().front().ref};
  loom::sim::SpatialSimulationWorkload workloadDraft{launch};
  workloadDraft.valueInputPlan = {loom::sim::RuntimeValueInput{}};
  workloadDraft.observableContract.valueResults = {0};
  auto workload =
      take(loom::sim::finalizeSimulationWorkload(workloadDraft, view));
  loom::sim::SpatialSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  runtimeDraft.runtimeValues = {
      {0,
       {1,
        {loom::sim::SemanticLane::defined(llvm::APInt(laneWidth, 1)),
         loom::sim::SemanticLane::defined(llvm::APInt(laneWidth, 2)),
         loom::sim::SemanticLane::defined(llvm::APInt(laneWidth, 3)),
         loom::sim::SemanticLane::defined(llvm::APInt(laneWidth, 4))}}}};
  auto runtime = take(
      loom::sim::finalizeSimulationRuntimeInput(runtimeDraft, workload, view));
  return {take(loom::sim::publishSimulationWorkload(workload, store)),
          take(loom::sim::publishSimulationRuntimeInput(runtime, store))};
}

void emptyConstraintOwnerPublishesExactArtifact() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store, blobs);
  auto dataflow = take(fixture.dataflow.view());
  auto tech = take(
      loom::mapping::importTechMapping(fixture.techMappingReference, store));

  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), fixture.fabric.view(), store));
  if (constraints.view().dataflowIdentity() != dataflow.identity() ||
      constraints.view().techMappingIdentity() != tech.view().identity() ||
      constraints.view().fabricIdentity() != fixture.fabric.view().identity() ||
      !constraints.view().clauses().empty())
    fail("empty constraint owner lost its exact D/T/F binding");

  auto imported = take(loom::mapping::importSpatialMappingConstraintSet(
      constraints.reference(), store));
  if (imported.reference() != constraints.reference() ||
      imported.canonicalBytes().bytes() !=
          constraints.canonicalBytes().bytes() ||
      !imported.view().clauses().empty())
    fail("strict empty constraint import changed canonical semantics");
}

void rootCompleteAdapterPublishesPhysicalMapping() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store, blobs);
  requireSuccess(
      loom::dse::registerRootCompleteTechMappingCandidateGenerator());
  requireSuccess(loom::dse::registerRootCompleteSpatialPnrCandidateGenerator());
  loom::ResolvedConfig resolved = buildSingleCandidateSpatialResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto spatialConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  resolved.dse.planNodes = {
      loom::dse::GeneratePlanNodeDefinition{
          loom::dse::rootCompleteTechMappingCandidateGeneratorDescriptor()
              .reference(),
          {loom::dse::ExactPlanArtifacts{{fixture.dataflowReference}},
           loom::dse::ExactPlanArtifacts{{fixture.fabric.reference()}}},
          techConfig.canonicalViewBytes().vec(),
          techConfig.digest()},
      loom::dse::GeneratePlanNodeDefinition{
          loom::dse::rootCompleteSpatialPnrCandidateGeneratorDescriptor()
              .reference(),
          {loom::dse::PlanOutputRef{0, 0},
           loom::dse::ExactPlanArtifacts{{fixture.fabric.reference()}},
           loom::dse::ExactPlanArtifacts{{fixture.physicalTimingProfile}}},
          spatialConfig.canonicalViewBytes().vec(),
          spatialConfig.digest()},
  };
  auto view = take(loom::dse::projectResolvedDseConfigView(resolved));
  auto outcome = take(loom::dse::executeDsePlan(view, store, blobs));
  const auto &completed = requireClosedPlan(outcome, 0);
  if (completed.generateInvocations().size() != 2 ||
      completed.resolve(loom::dse::PlanOutputRef{0, 0}).size() != 1 ||
      completed.resolve(loom::dse::PlanOutputRef{1, 0}).size() != 1)
    fail("root-complete Mapping plan did not publish T then SpatialMapping");
  const auto &spatialInvocation = completed.generateInvocations().back();
  if (spatialInvocation.lineageEdges.size() != 1)
    fail("root-complete Spatial invocation lost mechanical lineage");
  const auto &edge = spatialInvocation.lineageEdges.front();
  if (edge.kind !=
          loom::dse::CandidateGeneratorLineageEdgeKind::MechanicalDerivation ||
      !edge.parents.empty() || !edge.ownerPayload.empty())
    fail("root-complete Spatial adapter published non-mechanical lineage");

  auto dataflow = take(fixture.dataflow.view());
  auto tech = take(loom::mapping::importTechMapping(
      completed.resolve(loom::dse::PlanOutputRef{0, 0}).front(), store));
  auto spatial = take(loom::mapping::importSpatialMapping(
      completed.resolve(loom::dse::PlanOutputRef{1, 0}).front(), store));
  if (spatial.view().dataflowIdentity() != dataflow.identity() ||
      spatial.view().fabricIdentity() != fixture.fabric.view().identity() ||
      spatial.view().computeBindings().empty() ||
      spatial.view().routeTrees().empty() ||
      spatial.view().resourceUses().empty())
    fail("root-complete Spatial adapter published an empty or foreign Mapping");
  auto inspection = take(loom::mapping::inspectSpatialMapping(
      dataflow, tech.view(), fixture.fabric.view(), spatial.view()));
  if (inspection.summary.selectedActorCount == 0 ||
      inspection.summary.routeTreeCount == 0 ||
      inspection.summary.resourceUseCount == 0)
    fail("Spatial Mapping inspection found no physical work");

  auto wrongProfileInputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          {completed.resolve(loom::dse::PlanOutputRef{1, 0}).front()},
          fixture.fabric.reference(), fixture.physicalTimingProfile));
  auto spatialBinding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          spatialConfig));
  auto wrongProfile = loom::dse::invokeCandidateGenerator(
      wrongProfileInputs, spatialBinding, store, blobs);
  if (wrongProfile)
    fail("SpatialMapping was accepted in the TechMapping input slot");
  const std::string wrongProfileMessage =
      llvm::toString(wrongProfile.takeError());
  if (!llvm::StringRef(wrongProfileMessage).contains("TechMapping"))
    fail("wrong Mapping profile rejection lost its owner diagnostic");

  auto repeated = take(loom::dse::executeDsePlan(view, store, blobs));
  const auto &repeatedCompleted = requireClosedPlan(repeated, 0);
  if (!std::holds_alternative<loom::dse::CompletedDsePlanExecution>(repeated))
    fail("canonical singleton Mapping plan did not complete");
  if (repeatedCompleted.resolve(loom::dse::PlanOutputRef{0, 0}) !=
          completed.resolve(loom::dse::PlanOutputRef{0, 0}) ||
      repeatedCompleted.resolve(loom::dse::PlanOutputRef{1, 0}) !=
          completed.resolve(loom::dse::PlanOutputRef{1, 0}))
    fail("root-complete Mapping plan is not deterministic");

  const auto firstWork = completed.generateWorkSummaries();
  const auto repeatedWork = repeatedCompleted.generateWorkSummaries();
  if (firstWork.size() != repeatedWork.size())
    fail("deterministic replay changed production work summary width");
  for (std::size_t invocation = 0; invocation != firstWork.size(); ++invocation)
    if (firstWork[invocation].planNodeOrdinal !=
            repeatedWork[invocation].planNodeOrdinal ||
        firstWork[invocation].units != repeatedWork[invocation].units)
      fail("deterministic replay changed production provider work");

  const std::vector<loom::dse::GenerateInvocationWorkSummary> expectedWork(
      repeatedWork.begin(), repeatedWork.end());
  const loom::ArtifactRootReference selectedMapping =
      repeatedCompleted.resolve(loom::dse::PlanOutputRef{1, 0}).front();
  const loom::ArtifactIdentity storedConfig =
      take(store.put(loom::ResolvedConfig::artifactSchema,
                     loom::canonicalResolvedConfigBytes(resolved)));
  if (storedConfig != loom::resolvedConfigIdentity(resolved))
    fail("Manifest anchor changed the exact ResolvedConfig identity");
  auto manifestRecords =
      loom::dse::takeDsePlanGenerateInvocationRecords(std::move(repeated));
  auto closure = take(loom::dse::DseRunClosure::get(
      take(loom::dse::DseProducerSemanticBuildIdentity::get(
          "loom.test.root_complete_spatial_pnr.v1")),
      {fixture.dataflowReference, fixture.fabric.reference()}, resolved, {},
      store));
  auto manifest = take(loom::dse::InvocationManifest::get(
      std::move(closure), 0, std::nullopt, resolved, manifestRecords,
      loom::dse::InvocationCompletedSelection{{selectedMapping}, {}}, store,
      blobs));
  auto reorderedClosure = take(loom::dse::DseRunClosure::get(
      take(loom::dse::DseProducerSemanticBuildIdentity::get(
          "loom.test.root_complete_spatial_pnr.v1")),
      {fixture.fabric.reference(), fixture.dataflowReference}, resolved, {},
      store));
  auto reorderedManifest = take(loom::dse::InvocationManifest::get(
      std::move(reorderedClosure), 0, std::nullopt, resolved, manifestRecords,
      loom::dse::InvocationCompletedSelection{{selectedMapping}, {}}, store,
      blobs));
  if (reorderedManifest.canonicalBytes() != manifest.canonicalBytes())
    fail("semantic-input authoring order changed production Manifest bytes");
  auto adopted = take(loom::dse::adoptInvocationManifest(
      manifest.canonicalBytes(), resolved, store, blobs));
  if (adopted.generateRecords().size() != expectedWork.size())
    fail("Manifest dropped a production Generate work summary");
  for (std::size_t invocation = 0; invocation != expectedWork.size();
       ++invocation) {
    const auto &actual = adopted.generateRecords()[invocation].workSummary;
    const auto &expected = expectedWork[invocation];
    if (actual.planNodeOrdinal != expected.planNodeOrdinal ||
        actual.units != expected.units)
      fail("Manifest changed a production provider work summary");
    for (const auto &unit : expected.units)
      if (unit.planned != unit.consumed)
        fail("completed production provider left a planned logical work slot "
             "unconsumed");
  }
  auto workTotals = [](const loom::dse::InvocationManifest &value) {
    std::pair<std::uint64_t, std::uint64_t> totals{0, 0};
    for (const auto &record : value.generateRecords())
      for (const auto &unit : record.workSummary.units) {
        totals.first += unit.planned;
        totals.second += unit.consumed;
      }
    return totals;
  };
  if (workTotals(manifest) != workTotals(adopted) ||
      workTotals(manifest) != workTotals(reorderedManifest))
    fail("production work totals changed across import or input reordering");
}

void descriptorAndEmptySetAreClosed() {
  if (llvm::Error error =
          loom::dse::registerRootCompleteSpatialPnrCandidateGenerator())
    fail(llvm::toString(std::move(error)));
  const auto &descriptor =
      loom::dse::rootCompleteSpatialPnrCandidateGeneratorDescriptor();
  auto config = buildSpatialConfig();
  if (descriptor.kind !=
          loom::dse::rootCompleteSpatialPnrCandidateGeneratorKind ||
      descriptor.determinism !=
          loom::dse::CandidateGeneratorDeterminism::Deterministic ||
      descriptor.inputSlots.size() != 3 || descriptor.outputSlots.size() != 1 ||
      descriptor.workUnits.size() != 9 ||
      descriptor.inputSlots[0].semanticRole != "tech_mapping" ||
      descriptor.inputSlots[0].cardinality !=
          loom::dse::PlanValueCardinality::FiniteSet ||
      descriptor.inputSlots[1].semanticRole != "fabric" ||
      descriptor.inputSlots[1].cardinality !=
          loom::dse::PlanValueCardinality::ExactlyOne ||
      descriptor.inputSlots[2].semanticRole != "physical_timing_profile" ||
      descriptor.inputSlots[2].schema !=
          &loom::fabric::fabricPhysicalTimingProfileArtifactSchema ||
      descriptor.inputSlots[2].cardinality !=
          loom::dse::PlanValueCardinality::ExactlyOne ||
      descriptor.resolvedConfigView.schemaDescriptorBytes !=
          config.schemaDescriptorBytes())
    fail("root-complete Spatial descriptor is not closed over exact T/F/P");
  if (descriptor.workUnits.size() !=
      loom::dse::pnrCandidateGeneratorWorkUnits.size())
    fail("root-complete Spatial descriptor copied the PnR work-unit catalog");
  for (std::size_t ordinal = 0; ordinal != descriptor.workUnits.size();
       ++ordinal) {
    const auto &actual = descriptor.workUnits[ordinal];
    const auto &owner = loom::dse::pnrCandidateGeneratorWorkUnits[ordinal];
    if (!(actual.unit == owner.unit) || actual.spelling != owner.spelling)
      fail("root-complete Spatial descriptor diverged from PnR work units");
  }

  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  auto fabric = loom::test::buildSpatialCore(store);
  const auto physicalTiming =
      normalizedTimingProfile(fabric.reference(), store);
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          {}, fabric.reference(), physicalTiming));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      !completed->outputBindings.front().artifacts.empty() ||
      !completed->lineageEdges.empty())
    fail("empty TechMapping set did not propagate as completed empty");
  requireSpatialWorkSummary(outcome.workSummary, false);
}

void finiteSetTraversesEveryCanonicalTechMapping() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store, blobs);
  auto alternateDataflow =
      loom::test::buildAlternateRootCompleteSpatialDataflow(context);
  auto alternateDataflowReference =
      take(dataflow::publishCanonicalDataflow(alternateDataflow, store));
  auto alternateTechMapping = generateTechMapping(
      alternateDataflowReference, fixture.fabric.reference(), store, blobs);

  std::array<loom::ArtifactRootReference, 2> techMappings = {
      fixture.techMappingReference, alternateTechMapping};
  if (loom::artifactRootReferenceLess(techMappings[1], techMappings[0]))
    std::swap(techMappings[0], techMappings[1]);
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          techMappings, fixture.fabric.reference(),
          fixture.physicalTimingProfile));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          buildSingleCandidateSpatialConfig()));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 2 ||
      completed->lineageEdges.size() != 2)
    fail("finite TechMapping set did not produce one Spatial set");
  requireSpatialWorkSummary(outcome.workSummary, true);

  bool foundFirst = false;
  bool foundSecond = false;
  for (const auto &reference : completed->outputBindings.front().artifacts) {
    auto spatial = take(loom::mapping::importSpatialMapping(reference, store));
    if (spatial.view().techMappingIdentity() == techMappings[0].artifact)
      foundFirst = true;
    else if (spatial.view().techMappingIdentity() == techMappings[1].artifact)
      foundSecond = true;
    else
      fail("finite traversal published a SpatialMapping for a foreign T");
  }
  if (!foundFirst || !foundSecond)
    fail("finite traversal skipped a canonical TechMapping input");

  const std::array<loom::dse::CandidateGeneratorOutputDemand, 1> outputDemands =
      {{{loom::dse::CandidateGeneratorOutputSlotRef(0), 1}}};
  const loom::dse::CandidateGeneratorInvocationView limitedInvocation(
      {}, outputDemands);
  auto limitedOutcome = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, store, blobs, limitedInvocation));
  const auto *limitedCompleted =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &limitedOutcome.outcome);
  if (!limitedCompleted || limitedCompleted->outputBindings.size() != 1 ||
      limitedCompleted->outputBindings.front().artifacts.size() != 1 ||
      limitedCompleted->lineageEdges.size() != 1 ||
      limitedOutcome.workSummary != outcome.workSummary)
    fail("root-complete output demand truncated or reclassified exhaustive "
         "TechMapping work");
}

void firstVerifiedAvoidsSpeculativeRouteRanking() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto dataflow = loom::test::buildRootCompleteSpatialDataflow(context);
  const auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflow, store));
  auto fabric = loom::test::buildAlternativeTechSpatialCore(store);
  const auto physicalTiming =
      normalizedTimingProfile(fabric.reference(), store);
  const auto techMappings = generateTechMappingSet(
      dataflowReference, fabric.reference(), store, blobs);
  if (techMappings.size() < 2)
    fail("fixture did not expose alternative TechMappings for one graph");

  loom::ResolvedConfig resolved = buildSingleCandidateSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  const auto config =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          techMappings, fabric.reference(), physicalTiming));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &outcome.outcome);
  if (!incomplete ||
      incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::SemanticLimitReached ||
      incomplete->retainedOutputBindings.size() != 1 ||
      incomplete->retainedOutputBindings.front().artifacts.size() != 1 ||
      outcome.workSummary.size() !=
          loom::dse::pnrCandidateGeneratorWorkUnits.size() ||
      outcome.workSummary.front().planned != 1 ||
      outcome.workSummary.front().consumed != 1)
    fail("first-verified root adapter performed speculative route ranking");
  (void)take(loom::mapping::importSpatialMapping(
      incomplete->retainedOutputBindings.front().artifacts.front(), store));
}

void candidateWorkerCountPreservesFormalResult() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store, blobs);
  auto dataflow = take(fixture.dataflow.view());
  auto tech = take(
      loom::mapping::importTechMapping(fixture.techMappingReference, store));
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), fixture.fabric.view(), store));
  loom::ResolvedConfig resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.initializer.seedAttemptCount = 4;
  auto config = take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          fixture.fabric.view()));
  const auto run = [&](std::uint32_t workerCount,
                       loom::ExecutionResourceBudget executionBudget,
                       std::optional<std::uint64_t> maximumPublications =
                           std::nullopt) {
    loom::pnr::SpatialPnrGenerationInputs inputs{
        dataflow,       tech.view(), fixture.fabric.view(),
        physicalTiming, config,      constraints.view(),
        store,          workerCount};
    inputs.executionBudget = executionBudget;
    inputs.maximumCandidatePublications = maximumPublications;
    return loom::pnr::generateSpatialMappings(inputs);
  };
  const auto single = run(4, {1, std::nullopt});
  const auto memoryConstrained = run(4, {4, 1});
  const auto parallel = run(4, {3, 0});
  const auto publicationBounded = run(4, {3, 0}, 1);
  if (single.index() != memoryConstrained.index() ||
      single.index() != parallel.index() ||
      single.index() != publicationBounded.index())
    fail("candidate worker count changed the Spatial PnR outcome kind");
  const auto *singleGenerated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&single);
  const auto *memoryConstrainedGenerated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&memoryConstrained);
  const auto *parallelGenerated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&parallel);
  const auto *publicationBoundedGenerated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&publicationBounded);
  if (!singleGenerated || !memoryConstrainedGenerated || !parallelGenerated ||
      !publicationBoundedGenerated)
    fail("worker-invariance fixture did not produce Spatial Mappings");
  if (singleGenerated->termination != memoryConstrainedGenerated->termination ||
      singleGenerated->termination != parallelGenerated->termination ||
      !(singleGenerated->accounting ==
        memoryConstrainedGenerated->accounting) ||
      !(singleGenerated->accounting == parallelGenerated->accounting) ||
      singleGenerated->candidates != memoryConstrainedGenerated->candidates ||
      singleGenerated->candidates != parallelGenerated->candidates)
    fail("candidate worker count changed formal Spatial PnR output or work");
  loom::pnr::SpatialPnrGenerationAccounting publicationWork =
      publicationBoundedGenerated->accounting;
  publicationWork.finalizedRestarts =
      parallelGenerated->accounting.finalizedRestarts;
  publicationWork.publicationSlots =
      parallelGenerated->accounting.publicationSlots;
  if (publicationBoundedGenerated->termination !=
          parallelGenerated->termination ||
      publicationBoundedGenerated->accounting.seedAttemptSlots != 4 ||
      publicationBoundedGenerated->accounting.publicationSlots != 1 ||
      publicationBoundedGenerated->accounting.finalizedRestarts != 1 ||
      publicationBoundedGenerated->candidates.size() > 1 ||
      !(publicationWork == parallelGenerated->accounting))
    fail("publication demand serialized, truncated, or reclassified exhaustive "
         "Spatial work");
  for (std::size_t index = 0; index != singleGenerated->candidates.size();
       ++index) {
    auto singleMapping = take(loom::mapping::importSpatialMapping(
        singleGenerated->candidates[index], store));
    auto parallelMapping = take(loom::mapping::importSpatialMapping(
        parallelGenerated->candidates[index], store));
    if (singleMapping.canonicalBytes().bytes() !=
        parallelMapping.canonicalBytes().bytes())
      fail("candidate worker count changed canonical Spatial Mapping bytes");
  }
  llvm::outs() << "spatial_worker_budget constrained_workers=1"
               << " parallel_workers=3 exhaustive_restart_slots="
               << publicationBoundedGenerated->accounting.seedAttemptSlots
               << " bounded_publications="
               << publicationBoundedGenerated->accounting.publicationSlots
               << '\n';
}

void canonicalSeedHandoffPreservesFormalResult() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store, blobs);
  auto dataflow = take(fixture.dataflow.view());
  auto tech = take(
      loom::mapping::importTechMapping(fixture.techMappingReference, store));
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), fixture.fabric.view(), store));
  loom::ResolvedConfig resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.initializer.seedAttemptCount = 4;
  auto config = take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          fixture.fabric.view()));
  auto problem = take(loom::pnr::freezeSpatialPnrProblem(
      dataflow, tech.view(), fixture.fabric.view(), physicalTiming, config,
      constraints.view()));

  const auto makeHandoff = [&]() {
    loom::pnr::SpatialPathFinderSeedWorkSummary work;
    auto seed = take(loom::pnr::createPathFinderSpatialSeed(problem, 0, work));
    auto handoff =
        std::make_shared<loom::pnr::SpatialPathFinderSeedHandoff>();
    handoff->attemptOrdinal = 0;
    handoff->problemCacheKey = problem->cacheKey();
    handoff->workSummary = work;
    handoff->seed = std::move(seed);
    return handoff;
  };
  const auto run = [&](loom::ExecutionResourceBudget budget,
                       loom::pnr::SpatialPathFinderSeedHandoffHandle handoff) {
    loom::pnr::SpatialPnrGenerationInputs inputs{
        dataflow,       tech.view(), fixture.fabric.view(), physicalTiming,
        config,         constraints.view(), store, 4};
    inputs.preparedActiveProblem = problem;
    inputs.executionBudget = budget;
    inputs.preparedCanonicalSeed = std::move(handoff);
    return loom::pnr::generateSpatialMappings(inputs);
  };
  auto malformed = makeHandoff();
  malformed->seed->attemptOrdinal = 1;
  const auto malformedOutcome = run({3, 0}, std::move(malformed));
  const auto *malformedInvalid =
      std::get_if<loom::pnr::InvalidSpatialPnrGeneration>(&malformedOutcome);
  if (!malformedInvalid ||
      malformedInvalid->reason !=
          loom::pnr::InvalidSpatialPnrGenerationReason::FrozenInput)
    fail("malformed canonical seed handoff was not rejected as FrozenInput");
  auto failedHandoff = makeHandoff();
  failedHandoff->seed.reset();
  failedHandoff->failure = llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "synthetic transferred seed failure");
  const auto failedOutcome = run({3, 0}, std::move(failedHandoff));
  const auto *failedInternal =
      std::get_if<loom::pnr::InternalSpatialPnrGeneration>(&failedOutcome);
  if (!failedInternal)
    fail("transferred seed failure changed its typed internal classification");
  requireSuccess(loom::pnr::verifySpatialPnrWorkAccounting(
      failedInternal->accounting, /*requireClosedWork=*/false));
  if (failedInternal->accounting.plannedSeedAttemptSlots !=
          failedInternal->accounting.seedAttemptSlots ||
      failedInternal->accounting.plannedInitializerAssignmentAttempts !=
          failedInternal->accounting.initializerAssignmentAttempts ||
      failedInternal->accounting.plannedEndpointExpansionSlots !=
          failedInternal->accounting.endpointExpansionSlots ||
      failedInternal->accounting.plannedNegotiationIterationSlots !=
          failedInternal->accounting.negotiationIterationSlots)
    fail("transferred seed failure changed its owner-completed work summary");
  const auto cold = run({3, 0}, nullptr);
  const auto warm = run({3, 0}, makeHandoff());
  const auto coldMemory = run({4, 1}, nullptr);
  const auto warmMemory = run({4, 1}, makeHandoff());
  const auto compare = [&](const auto &lhs, const auto &rhs,
                           llvm::StringRef label) {
    if (lhs.index() != rhs.index())
      fail(llvm::Twine(label) + " changed the formal outcome kind");
    const auto *lhsGenerated =
        std::get_if<loom::pnr::GeneratedSpatialMappings>(&lhs);
    const auto *rhsGenerated =
        std::get_if<loom::pnr::GeneratedSpatialMappings>(&rhs);
    if (!lhsGenerated || !rhsGenerated ||
        lhsGenerated->termination != rhsGenerated->termination ||
        !(lhsGenerated->accounting == rhsGenerated->accounting) ||
        lhsGenerated->candidates != rhsGenerated->candidates)
      fail(llvm::Twine(label) +
           " changed termination, work accounting, or candidates");
  };
  compare(cold, warm, "canonical seed handoff");
  compare(coldMemory, warmMemory, "memory-calibrated canonical seed handoff");
  llvm::outs() << "spatial_seed_handoff reused_ordinal=0"
               << " memory_calibration=verified"
               << " seed_slots="
               << std::get<loom::pnr::GeneratedSpatialMappings>(warm)
                      .accounting.seedAttemptSlots
               << '\n';
}

void firstVerifiedCandidateRetainsTypedPrefix() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store, blobs);
  auto dataflow = take(fixture.dataflow.view());
  auto tech = take(
      loom::mapping::importTechMapping(fixture.techMappingReference, store));
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), fixture.fabric.view(), store));
  auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          fixture.fabric.view()));

  const auto exhaustive = loom::pnr::generateSpatialMappings(
      {dataflow, tech.view(), fixture.fabric.view(), physicalTiming,
       buildSpatialConfig(), constraints.view(), store, 1});
  loom::ResolvedConfig boundedResolved = buildSpatialResolvedConfig();
  boundedResolved.dse.spatialPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  const auto boundedConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(boundedResolved));
  const auto bounded = loom::pnr::generateSpatialMappings(
      {dataflow, tech.view(), fixture.fabric.view(), physicalTiming,
       boundedConfig, constraints.view(), store, 2});

  const auto *exhaustiveGenerated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&exhaustive);
  const auto *boundedGenerated =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&bounded);
  if (!exhaustiveGenerated || !boundedGenerated ||
      exhaustiveGenerated->termination !=
          loom::pnr::PnrGenerationTermination::FixedAttemptsCompleted ||
      exhaustiveGenerated->accounting.plannedSeedAttemptSlots != 2 ||
      exhaustiveGenerated->accounting.seedAttemptSlots != 2 ||
      boundedGenerated->termination !=
          loom::pnr::PnrGenerationTermination::SemanticLimitReached ||
      boundedGenerated->candidates.size() != 1 ||
      boundedGenerated->accounting.plannedSeedAttemptSlots != 1 ||
      boundedGenerated->accounting.seedAttemptSlots != 1 ||
      boundedGenerated->accounting.calibrationProposalSlots != 0 ||
      boundedGenerated->accounting.annealingBaseProposalSlots != 0 ||
      boundedGenerated->accounting.annealingMovableProposalSlots != 0 ||
      boundedGenerated->accounting.finalClosureAttempts == 0 ||
      boundedGenerated->accounting.plannedFinalClosureAttempts !=
          boundedGenerated->accounting.finalClosureAttempts ||
      boundedGenerated->accounting.plannedInitializerAssignmentAttempts !=
          boundedGenerated->accounting.initializerAssignmentAttempts ||
      boundedGenerated->accounting.plannedEndpointExpansionSlots !=
          boundedGenerated->accounting.endpointExpansionSlots ||
      boundedGenerated->accounting.plannedNegotiationIterationSlots !=
          boundedGenerated->accounting.negotiationIterationSlots ||
      boundedGenerated->accounting.finalizedRestarts != 1 ||
      boundedGenerated->accounting.publicationSlots != 1 ||
      exhaustiveGenerated->accounting.calibrationProposalSlots == 0)
    fail("first-verified Spatial search lost its verified bounded prefix");
  (void)take(loom::mapping::importSpatialMapping(
      boundedGenerated->candidates.front(), store));
}

void interruptionReturnsTypedSpatialSnapshot() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  requireSuccess(
      llvm::errorCodeToError(llvm::sys::fs::create_directories(blobPath)));
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store, blobs);
  auto dataflow = take(fixture.dataflow.view());
  auto tech = take(
      loom::mapping::importTechMapping(fixture.techMappingReference, store));
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), fixture.fabric.view(), store));
  auto physicalTiming =
      take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
          fixture.fabric.view()));
  const ImmediateStopSource stop;
  const auto outcome = loom::pnr::generateSpatialMappings(
      {dataflow, tech.view(), fixture.fabric.view(), physicalTiming,
       buildSpatialConfig(), constraints.view(), store, 1, stop.control()});
  const auto *interrupted =
      std::get_if<loom::pnr::InterruptedSpatialPnrGeneration>(&outcome);
  if (!interrupted ||
      interrupted->snapshot.stage !=
          loom::pnr::SpatialPnrInterruptionStage::InputAdmission ||
      interrupted->accounting.plannedSeedAttemptSlots != 0 ||
      interrupted->accounting.seedAttemptSlots != 0 ||
      interrupted->accounting.plannedInitializerAssignmentAttempts != 0 ||
      interrupted->accounting.initializerAssignmentAttempts != 0 ||
      interrupted->snapshot.frontier.seedAttemptSlots != 0 ||
      interrupted->snapshot.bestSelectedRank ||
      interrupted->snapshot.closureResidual.violationValues ||
      interrupted->snapshot.closureResidual.retainedCandidates != 0)
    fail("Spatial interruption did not preserve its typed empty frontier");

  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          {fixture.techMappingReference}, fixture.fabric.reference(),
          fixture.physicalTimingProfile));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          buildSpatialConfig()));
  const auto provider = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, store, blobs, stop.control()));
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &provider.outcome);
  if (!incomplete ||
      incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::CancelledOrTimeout ||
      llvm::any_of(provider.workSummary, [](const auto &unit) {
        return unit.planned != 0 || unit.consumed != 0;
      }))
    fail("Spatial interruption did not map to a cancelled provider outcome");
}

void unavailableNegotiationFailsClosedAtConfigProjection() {
  loom::ResolvedConfig resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.routing.negotiation =
      loom::ResolvedDualSubgradientPolicy{
          loom::ResolvedDualDirectionKernel::ProjectedSigned,
          std::nullopt,
          {loom::ResolvedDualStepScheduleKind::Constant, 1, 0, 0, 0}};
  auto config = loom::pnr::projectResolvedSpatialPnrConfigView(resolved);
  if (config)
    fail("unavailable routing kernel passed configuration projection");
  const std::string diagnostic = llvm::toString(config.takeError());
  if (!llvm::StringRef(diagnostic).contains("pnr_config_unsupported"))
    fail("unavailable routing kernel lost its fail-closed diagnostic");
}

void initializerSemanticLimitIsTypedIncomplete() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store, blobs);
  loom::ResolvedConfig resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.initializer.seedAttemptCount = 1;
  resolved.dse.spatialPnr.search.initializer.assignmentAttemptLimitPerSeed = 1;
  auto config = take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          {fixture.techMappingReference}, fixture.fabric.reference(),
          fixture.physicalTimingProfile));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &outcome.outcome);
  if (!incomplete)
    fail("one initializer assignment unexpectedly completed Spatial PnR");
  if (incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::SemanticLimitReached ||
      incomplete->retainedOutputBindings.size() != 1 ||
      !incomplete->retainedOutputBindings.front().artifacts.empty() ||
      !incomplete->lineageEdges.empty() ||
      outcome.workSummary.size() !=
          loom::dse::pnrCandidateGeneratorWorkUnits.size() ||
      outcome.workSummary[0].planned != 1 ||
      outcome.workSummary[0].consumed != 1 ||
      outcome.workSummary[1].planned != 1 ||
      outcome.workSummary[1].consumed != 1 ||
      llvm::any_of(outcome.workSummary, [](const auto &unit) {
        return unit.consumed > unit.planned;
      }))
    fail("initializer semantic limit did not remain typed Incomplete");
}

void foreignFabricIsRejectedBeforeSearch() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store, blobs);
  auto foreignFabric = loom::test::buildSpatialCore(store, 64);
  const auto foreignTiming =
      normalizedTimingProfile(foreignFabric.reference(), store);
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          {fixture.techMappingReference}, foreignFabric.reference(),
          foreignTiming));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          buildSpatialConfig()));
  auto outcome =
      loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs);
  if (outcome)
    fail("root-complete Spatial adapter accepted a foreign Fabric");
  const std::string message = llvm::toString(outcome.takeError());
  if (!llvm::StringRef(message).contains("foreign Fabric"))
    fail("foreign Fabric rejection lost its exact-owner diagnostic");
}

void spatialMappingPromotionExecutesExactCgraCase() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store, blobs);
  const loom::ArtifactRootReference spatialMapping = generateSpatialMapping(
      fixture.techMappingReference, fixture.fabric.reference(), store, blobs);
  const PublishedSpatialInputs simulationInputs =
      publishSpatialInputs(fixture.dataflow, store);
  const loom::ArtifactRootReference &workloadReference =
      simulationInputs.workload;
  const loom::ArtifactRootReference &runtimeReference =
      simulationInputs.runtimeInput;

  auto obligation =
      take(loom::dse::prepareCgraSimulationEvidenceObligationTemplate(
          fixture.dataflowReference, fixture.fabric.reference(), spatialMapping,
          workloadReference, runtimeReference, loom::defaultResolvedConfig(),
          store, blobs));
  const std::array<loom::dse::EvidenceObligationTemplateRef, 1> obligationRefs =
      {loom::dse::EvidenceObligationTemplateRef(0)};
  auto acquisitionConfig =
      take(loom::dse::projectResolvedEvidenceObligationSetConfigView(
          obligationRefs));
  auto acquisitionBinding = take(
      loom::dse::resolveSpatialMappingEvaluationPromotionAcquisitionBinding(
          acquisitionConfig));
  auto alternate =
      loom::test::buildAlternateRootCompleteSpatialDataflow(context);
  const loom::ArtifactRootReference alternateReference =
      take(dataflow::publishCanonicalDataflow(alternate, store));
  std::array<loom::ArtifactRootReference, 2> dataflows = {
      fixture.dataflowReference, alternateReference};
  llvm::sort(dataflows, loom::artifactRootReferenceLess);
  auto inputs = take(loom::dse::bindSpatialMappingEvaluationPromotionInputs(
      {spatialMapping}, dataflows, fixture.fabric.reference(),
      workloadReference, runtimeReference));
  auto outcome = take(loom::dse::invokePromotionAcquisition(
      inputs, acquisitionBinding, {obligation},
      {{spatialMapping}, obligationRefs}, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedPromotionAcquisition>(&outcome);
  if (!completed || completed->evidence.size() != 1)
    fail("SpatialMapping promotion did not acquire one CGRA Evidence");
  const loom::dse::PromotionEvidence &evidence = completed->evidence.front();
  const auto programs = evidence.request.subjectBindings().subjects(
      loom::evaluation::models::cgraSimulationProgramRole());
  const auto mappings = evidence.request.subjectBindings().subjects(
      loom::evaluation::models::cgraSimulationSpatialMappingRole());
  if (programs.size() != 1 || programs.front() != fixture.dataflowReference ||
      mappings.size() != 1 || mappings.front() != spatialMapping ||
      evidence.evidence.outcomeKind() !=
          loom::evaluation::EvidenceOutcomeKind::Completed ||
      evidence.evidence.outputBindings().size() != 1 ||
      evidence.evidence.outputBindings().front().artifacts.size() != 1)
    fail("SpatialMapping promotion lost exact CGRA execution evidence");

  auto wrongInputs =
      take(loom::dse::bindSpatialMappingEvaluationPromotionInputs(
          {spatialMapping}, {alternateReference}, fixture.fabric.reference(),
          workloadReference, runtimeReference));
  auto wrong = loom::dse::invokePromotionAcquisition(
      wrongInputs, acquisitionBinding, {obligation},
      {{spatialMapping}, obligationRefs}, store, blobs);
  if (wrong)
    fail("SpatialMapping promotion accepted a foreign Dataflow owner set");
  const std::string message = llvm::toString(wrong.takeError());
  if (!llvm::StringRef(message).contains("Dataflow owner"))
    fail("foreign Dataflow owner rejection lost its typed diagnostic");

  loom::ResolvedConfig resolved = buildSingleCandidateSpatialResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto spatialConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  resolved.dse.modelAuthorizations = {
      {loom::evaluation::models::cgraSimulationModelDescriptorRef()}};
  resolved.dse.evidenceObligationTemplates = {obligation};
  resolved.dse.qualityGatePolicies = {
      take(loom::dse::QualityGatePolicy::get({}))};
  resolved.dse.planNodes = {
      loom::dse::GeneratePlanNodeDefinition{
          loom::dse::rootCompleteTechMappingCandidateGeneratorDescriptor()
              .reference(),
          {loom::dse::ExactPlanArtifacts{{fixture.dataflowReference}},
           loom::dse::ExactPlanArtifacts{{fixture.fabric.reference()}}},
          techConfig.canonicalViewBytes().vec(),
          techConfig.digest()},
      loom::dse::GeneratePlanNodeDefinition{
          loom::dse::rootCompleteSpatialPnrCandidateGeneratorDescriptor()
              .reference(),
          {loom::dse::PlanOutputRef{0, 0},
           loom::dse::ExactPlanArtifacts{{fixture.fabric.reference()}},
           loom::dse::ExactPlanArtifacts{{fixture.physicalTimingProfile}}},
          spatialConfig.canonicalViewBytes().vec(),
          spatialConfig.digest()},
      loom::dse::PromotePlanNodeDefinition{
          loom::dse::spatialMappingEvaluationPromotionAcquisitionDescriptor()
              .reference(),
          {loom::dse::PlanOutputRef{1, 0},
           loom::dse::ExactPlanArtifacts{
               std::vector<loom::ArtifactRootReference>(dataflows.begin(),
                                                        dataflows.end())},
           loom::dse::ExactPlanArtifacts{{fixture.fabric.reference()}},
           loom::dse::ExactPlanArtifacts{{workloadReference}},
           loom::dse::ExactPlanArtifacts{{runtimeReference}}},
          acquisitionConfig.canonicalViewBytes().vec(),
          acquisitionConfig.digest(),
          loom::dse::QualityGatePolicyRef(0),
          loom::dse::AllPassingSelection{},
          loom::dse::PromotePurpose::CandidateSelection},
  };
  auto planView = take(loom::dse::projectResolvedDseConfigView(resolved));
  auto planOutcome = take(loom::dse::executeDsePlan(planView, store, blobs));
  const auto &planExecution = requireClosedPlan(planOutcome, 0);
  if (planExecution.generateInvocations().size() != 2 ||
      planExecution.resolve({1, 0}).size() != 1 ||
      planExecution.resolve({2, 0}) != planExecution.resolve({1, 0}) ||
      planExecution.resolve({2, 1}).size() != 1)
    fail("central Mapping plan did not close M to CGRA Evidence");
}

void spatialMappingPromotionKeepsEveryCandidateLineage() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto dataflow = loom::test::buildRootCompleteSpatialDataflow(context);
  const loom::ArtifactRootReference dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflow, store));
  auto fabric = loom::test::buildSpatialCore(store);
  const std::vector<loom::ArtifactRootReference> techMappings =
      generateTechMappingSet(dataflowReference, fabric.reference(), store,
                             blobs);
  std::vector<loom::ArtifactRootReference> mappings =
      generateSpatialMappingSet(techMappings, fabric.reference(), store, blobs);
  mappings.erase(mappings.begin() + 2, mappings.end());
  const PublishedSpatialInputs simulationInputs =
      publishSpatialInputs(dataflow, store);

  auto obligation =
      take(loom::dse::prepareCgraSimulationEvidenceObligationTemplate(
          dataflowReference, fabric.reference(), mappings.front(),
          simulationInputs.workload, simulationInputs.runtimeInput,
          loom::defaultResolvedConfig(), store, blobs));
  const std::array<loom::dse::EvidenceObligationTemplateRef, 1> obligations = {
      loom::dse::EvidenceObligationTemplateRef(0)};
  auto acquisitionConfig = take(
      loom::dse::projectResolvedEvidenceObligationSetConfigView(obligations));
  auto binding = take(
      loom::dse::resolveSpatialMappingEvaluationPromotionAcquisitionBinding(
          acquisitionConfig));
  auto inputs = take(loom::dse::bindSpatialMappingEvaluationPromotionInputs(
      mappings, {dataflowReference}, fabric.reference(),
      simulationInputs.workload, simulationInputs.runtimeInput));
  auto outcome = take(loom::dse::invokePromotionAcquisition(
      inputs, binding, {obligation}, {mappings, obligations}, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedPromotionAcquisition>(&outcome);
  if (!completed || completed->evidence.size() != mappings.size())
    fail("multi-candidate Spatial promotion lost CGRA Evidence");
  std::vector<loom::ArtifactRootReference> observedMappings;
  for (const loom::dse::PromotionEvidence &record : completed->evidence) {
    const auto programs = record.request.subjectBindings().subjects(
        loom::evaluation::models::cgraSimulationProgramRole());
    const auto selectedMappings = record.request.subjectBindings().subjects(
        loom::evaluation::models::cgraSimulationSpatialMappingRole());
    if (programs.size() != 1 || programs.front() != dataflowReference)
      fail("multi-candidate Spatial promotion selected a foreign Dataflow");
    if (selectedMappings.size() != 1)
      fail("multi-candidate Spatial promotion lost its Mapping subject");
    observedMappings.push_back(selectedMappings.front());
  }
  llvm::sort(observedMappings, loom::artifactRootReferenceLess);
  if (observedMappings != mappings)
    fail("multi-candidate Spatial promotion reused another candidate");
}

void spatialMappingFeedbackPublishesNarrowImmutableDataflow() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto dataflow = loom::test::buildVectorRootCompleteSpatialDataflow(context);
  const loom::ArtifactRootReference dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflow, store));
  auto fabric = loom::test::buildLineageSpatialCore(store);
  const loom::ArtifactRootReference techMapping =
      generateTechMapping(dataflowReference, fabric.reference(), store, blobs);
  auto spatial = generateSpatialFeedbackFixture(dataflowReference, techMapping,
                                                fabric, store);
  const loom::ArtifactRootReference &spatialMapping = spatial.mapping;
  auto dataflowView = take(dataflow.view());
  auto tech = take(loom::mapping::importTechMapping(techMapping, store));
  auto mapping =
      take(loom::mapping::importSpatialMapping(spatialMapping, store));
  auto frozen = take(loom::pnr::freezeSpatialPnrProblem(
      dataflowView, tech.view(), fabric.view(), buildFeedbackSpatialConfig(),
      spatial.constraints.view()));
  const auto &routing = frozen->routing();
  bool foundSpatialBroadcast = false;
  for (std::size_t first = 0; first < routing.traversals().size(); ++first) {
    const auto &lhs = routing.traversals()[first];
    const auto *lhsSwitch =
        std::get_if<loom::fabric::FabricSwitchTraversalPayload>(
            &lhs.reference.payload);
    if (!lhsSwitch || lhs.routeClaimCount == 0)
      continue;
    for (std::size_t second = first + 1; second < routing.traversals().size();
         ++second) {
      const auto &rhs = routing.traversals()[second];
      const auto *rhsSwitch =
          std::get_if<loom::fabric::FabricSwitchTraversalPayload>(
              &rhs.reference.payload);
      if (!rhsSwitch || rhs.routeClaimCount == 0 ||
          lhsSwitch->owner != rhsSwitch->owner ||
          lhsSwitch->input != rhsSwitch->input ||
          lhsSwitch->output == rhsSwitch->output)
        continue;
      const auto groups = routing.traversalReplicationGroups();
      if (groups[first] == loom::pnr::getInvalidPnrIndex() ||
          groups[first] != groups[second])
        fail("spatial switch broadcast lost its physical replication group");
      const auto lhsClaims = routing.traversalClaimKeys().slice(
          lhs.routeClaimOffset, lhs.routeClaimCount);
      const auto rhsClaims = routing.traversalClaimKeys().slice(
          rhs.routeClaimOffset, rhs.routeClaimCount);
      std::size_t sharedClaimCount = 0;
      for (loom::pnr::PnrIndex claim : lhsClaims)
        sharedClaimCount += llvm::is_contained(rhsClaims, claim);
      if (sharedClaimCount != 1)
        fail("spatial switch broadcast did not share exactly one ingress "
             "claim");
      foundSpatialBroadcast = true;
    }
  }
  if (!foundSpatialBroadcast)
    fail("vector feedback fixture has no spatial switch broadcast anchor");
  auto traversalClaims = take(
      loom::pnr::projectSpatialMappingTraversalClaims(*frozen, mapping.view()));
  if (traversalClaims.total == 0)
    fail("vector feedback fixture has no selected traversal claim");
  const PublishedSpatialInputs simulationInputs =
      publishVectorSpatialInputs(dataflow, store);
  auto preparedCgra =
      take(loom::evaluation::models::prepareCgraSimulationEvaluation(
          dataflowReference, fabric.reference(), spatialMapping,
          simulationInputs.workload, simulationInputs.runtimeInput,
          loom::defaultResolvedConfig(), store, blobs));

  auto obligation =
      take(loom::dse::prepareCgraSimulationEvidenceObligationTemplate(
          dataflowReference, fabric.reference(), spatialMapping,
          simulationInputs.workload, simulationInputs.runtimeInput,
          loom::defaultResolvedConfig(), store, blobs));
  const std::array<loom::dse::EvidenceObligationTemplateRef, 1> obligations = {
      loom::dse::EvidenceObligationTemplateRef(0)};
  auto acquisitionConfig = take(
      loom::dse::projectResolvedEvidenceObligationSetConfigView(obligations));
  loom::ResolvedConfig planConfig = buildSpatialResolvedConfig();
  planConfig.dse.techMapping.candidatePublicationLimit = 1;
  auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(planConfig));
  auto feedbackConfig = buildFeedbackSpatialConfig();
  requireSuccess(loom::dse::registerSpatialMappingFeedbackCandidateGenerator());
  planConfig.dse.modelAuthorizations = {
      {loom::evaluation::models::cgraSimulationModelDescriptorRef()}};
  planConfig.dse.evidenceObligationTemplates = {obligation};
  planConfig.dse.qualityGatePolicies = {
      take(loom::dse::QualityGatePolicy::get({}))};
  planConfig.dse.planNodes = {
      loom::dse::PromotePlanNodeDefinition{
          loom::dse::spatialMappingEvaluationPromotionAcquisitionDescriptor()
              .reference(),
          {loom::dse::ExactPlanArtifacts{{spatialMapping}},
           loom::dse::ExactPlanArtifacts{{dataflowReference}},
           loom::dse::ExactPlanArtifacts{{fabric.reference()}},
           loom::dse::ExactPlanArtifacts{{simulationInputs.workload}},
           loom::dse::ExactPlanArtifacts{{simulationInputs.runtimeInput}}},
          acquisitionConfig.canonicalViewBytes().vec(),
          acquisitionConfig.digest(),
          loom::dse::QualityGatePolicyRef(0),
          loom::dse::AllPassingSelection{},
          loom::dse::PromotePurpose::CandidateSelection},
      loom::dse::GeneratePlanNodeDefinition{
          loom::dse::spatialMappingFeedbackCandidateGeneratorDescriptor()
              .reference(),
          {loom::dse::ExactPlanArtifacts{{dataflowReference}},
           loom::dse::PlanOutputRef{0, 0},
           loom::dse::ExactPlanArtifacts{{spatial.constraints.reference()}},
           loom::dse::PlanOutputRef{0, 1},
           loom::dse::ExactPlanArtifacts{{simulationInputs.workload}},
           loom::dse::ExactPlanArtifacts{{simulationInputs.runtimeInput}}},
          feedbackConfig.canonicalViewBytes().vec(),
          feedbackConfig.digest()},
      loom::dse::GeneratePlanNodeDefinition{
          loom::dse::rootCompleteTechMappingCandidateGeneratorDescriptor()
              .reference(),
          {loom::dse::PlanOutputRef{1, 0},
           loom::dse::ExactPlanArtifacts{{fabric.reference()}}},
          techConfig.canonicalViewBytes().vec(),
          techConfig.digest()},
  };
  auto planView = take(loom::dse::projectResolvedDseConfigView(planConfig));
  auto planOutcome = take(loom::dse::executeDsePlan(planView, store, blobs));
  const auto *completed = &requireClosedPlan(planOutcome, 2);
  if (completed->generateInvocations().size() != 2)
    fail("central Mapping feedback plan changed its Generate invocation count");
  if (completed->resolve({0, 0}) !=
      llvm::ArrayRef<loom::ArtifactRootReference>{spatialMapping})
    fail("central Mapping feedback plan lost the promoted Mapping");
  if (completed->resolve({0, 1}).size() != 1)
    fail("central Mapping feedback plan lost promoted Evidence");
  if (completed->resolve({1, 0}).size() != 1)
    fail("central Mapping feedback plan lost the immutable Dataflow child");
  if (completed->resolve({2, 0}).size() != 1)
    fail("central Mapping feedback plan lost downstream TechMapping");
  const auto &childReference = completed->resolve({1, 0}).front();
  if (childReference == dataflowReference)
    fail("Mapping feedback returned its unchanged parent");
  auto child = take(dataflow::importCanonicalDataflow(childReference, store));
  unsigned narrowAdds = 0;
  child.module().walk([&](mlir::arith::AddIOp add) {
    const auto type = llvm::dyn_cast<mlir::VectorType>(add.getType());
    narrowAdds += type && type.getShape() == llvm::ArrayRef<std::int64_t>{2};
  });
  if (narrowAdds != 2)
    fail("Mapping feedback did not choose the canonical narrower vector form");
  auto childTech = take(loom::mapping::importTechMapping(
      completed->resolve({2, 0}).front(), store));
  if (childTech.view().dataflowIdentity() != childReference.artifact)
    fail("feedback child did not reach exact downstream TechMapping");
  const auto &feedbackInvocation = completed->generateInvocations().front();
  if (feedbackInvocation.planNodeOrdinal != 1 ||
      feedbackInvocation.lineageEdges.size() != 1)
    fail("central feedback Generate lost its exact invocation lineage");
  const auto &lineage = feedbackInvocation.lineageEdges.front();
  auto decision =
      take(dataflow::adoptDataflowRewriteDecision(lineage.ownerPayload));
  const auto *chunk =
      std::get_if<dataflow::ElementwiseVectorChunkRewrite>(&decision);
  if (lineage.kind !=
          loom::dse::CandidateGeneratorLineageEdgeKind::CandidateDecision ||
      lineage.parents !=
          std::vector<loom::ArtifactRootReference>{dataflowReference} ||
      !chunk || chunk->leadingBlocksPerChunk != 2)
    fail("Mapping feedback lost its typed Dataflow decision lineage");

  auto unsupportedEvidence = take(loom::evaluation::EvaluationEvidence::get(
      preparedCgra.request, {{loom::evaluation::ModelOutputSlotRef(0), {}}},
      loom::evaluation::UnsupportedEvidence{
          loom::evaluation::OutcomeReason::RuntimeCapabilityUnavailable},
      preparedCgra.resolution, store, blobs));
  const loom::ArtifactRootReference unsupportedReference = take(
      loom::evaluation::publishEvaluationEvidence(unsupportedEvidence, store));
  auto feedbackBinding =
      take(loom::dse::resolveSpatialMappingFeedbackCandidateGeneratorBinding(
          feedbackConfig));
  auto unsupportedInputs =
      take(loom::dse::bindSpatialMappingFeedbackCandidateGeneratorInputs(
          {dataflowReference}, {spatialMapping},
          spatial.constraints.reference(), {unsupportedReference},
          simulationInputs.workload, simulationInputs.runtimeInput));
  auto unsupportedOutcome = take(loom::dse::invokeCandidateGenerator(
      unsupportedInputs, feedbackBinding, store, blobs));
  const auto *proofNotEstablished =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &unsupportedOutcome.outcome);
  if (!proofNotEstablished ||
      proofNotEstablished->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::ProofNotEstablished ||
      proofNotEstablished->retainedOutputBindings.size() != 1 ||
      !proofNotEstablished->retainedOutputBindings.front().artifacts.empty() ||
      !proofNotEstablished->lineageEdges.empty())
    fail("non-Completed Mapping Evidence did not remain proof-incomplete");

  auto impersonatingSubjects =
      take(loom::evaluation::EvaluationSubjectBindings::get(
          {{loom::evaluation::models::cgraSimulationProgramRole(),
            {dataflowReference}},
           {loom::evaluation::models::cgraSimulationHardwareRole(),
            {fabric.reference()}},
           {loom::evaluation::models::cgraSimulationSpatialMappingRole(),
            {techMapping}}}));
  auto impersonatingBinding =
      take(loom::evaluation::ResolvedModelBinding::project(
          loom::evaluation::models::cgraSimulationModelDescriptorRef(), {},
          loom::defaultResolvedConfig()));
  auto impersonatingRequest = take(loom::evaluation::EvaluationRequest::get(
      std::move(impersonatingSubjects), simulationInputs.workload,
      simulationInputs.runtimeInput, preparedCgra.request.baseConditions(),
      preparedCgra.request.metricRequests(),
      preparedCgra.request.findingRequests(),
      std::move(impersonatingBinding), 0, preparedCgra.resolution, store,
      blobs));
  take(loom::evaluation::publishEvaluationRequest(impersonatingRequest, store));
  auto impersonatingEvidence = take(loom::evaluation::EvaluationEvidence::get(
      impersonatingRequest, {{loom::evaluation::ModelOutputSlotRef(0), {}}},
      loom::evaluation::UnsupportedEvidence{
          loom::evaluation::OutcomeReason::RuntimeCapabilityUnavailable},
      preparedCgra.resolution, store, blobs));
  const loom::ArtifactRootReference impersonatingReference =
      take(loom::evaluation::publishEvaluationEvidence(impersonatingEvidence,
                                                       store));
  auto impersonatingInputs =
      take(loom::dse::bindSpatialMappingFeedbackCandidateGeneratorInputs(
          {dataflowReference}, {spatialMapping},
          spatial.constraints.reference(), {impersonatingReference},
          simulationInputs.workload, simulationInputs.runtimeInput));
  auto impersonatingOutcome = loom::dse::invokeCandidateGenerator(
      impersonatingInputs, feedbackBinding, store, blobs);
  if (impersonatingOutcome)
    fail("TechMapping Evidence impersonated a SpatialMapping subject");
  const std::string impersonatingMessage =
      llvm::toString(impersonatingOutcome.takeError());
  if (!llvm::StringRef(impersonatingMessage)
           .contains("Evidence subjects differ"))
    fail("subject impersonation lost its exact rejection");

  loom::ArtifactIdentity::Storage missingBytes{};
  missingBytes.fill(0xff);
  const loom::ArtifactRootReference missingEvidence{
      loom::evaluation::EvaluationEvidence::artifactSchema.identity.str(),
      loom::evaluation::EvaluationEvidence::artifactSchema.version,
      take(loom::ArtifactIdentity::fromBytes(missingBytes))};
  auto directMissing = loom::evaluation::importEvaluationEvidence(
      missingEvidence, preparedCgra.resolution, store, blobs);
  if (directMissing)
    fail("missing Evidence unexpectedly exists in the ArtifactStore");
  const std::string directMissingMessage =
      llvm::toString(directMissing.takeError());
  const loom::ArtifactRootReference completedReference =
      completed->resolve({0, 1}).front();
  auto requireExactMissingError =
      [&](llvm::ArrayRef<loom::ArtifactRootReference> evidence) {
        auto missingInputs =
            take(loom::dse::bindSpatialMappingFeedbackCandidateGeneratorInputs(
                {dataflowReference}, {spatialMapping},
                spatial.constraints.reference(), evidence,
                simulationInputs.workload, simulationInputs.runtimeInput));
        auto missingOutcome = loom::dse::invokeCandidateGenerator(
            missingInputs, feedbackBinding, store, blobs);
        if (missingOutcome)
          fail("missing Mapping Evidence was treated as an absent record");
        if (llvm::toString(missingOutcome.takeError()) != directMissingMessage)
          fail("Mapping feedback did not propagate the exact Evidence import "
               "error");
      };
  const std::array<loom::ArtifactRootReference, 2> validThenMissing = {
      completedReference, missingEvidence};
  const std::array<loom::ArtifactRootReference, 2> missingThenValid = {
      missingEvidence, completedReference};
  requireExactMissingError(validThenMissing);
  requireExactMissingError(missingThenValid);

  auto extraInputs =
      take(loom::dse::bindSpatialMappingFeedbackCandidateGeneratorInputs(
          {dataflowReference}, {spatialMapping},
          spatial.constraints.reference(),
          {completedReference, unsupportedReference}, simulationInputs.workload,
          simulationInputs.runtimeInput));
  auto extraOutcome = loom::dse::invokeCandidateGenerator(
      extraInputs, feedbackBinding, store, blobs);
  if (extraOutcome)
    fail("Mapping feedback ignored an unmatched extra Evidence record");
  if (!llvm::StringRef(llvm::toString(extraOutcome.takeError()))
           .contains("unmatched record"))
    fail("unmatched Evidence lost its exact rejection");

  auto alternate =
      loom::test::buildAlternateRootCompleteSpatialDataflow(context);
  const loom::ArtifactRootReference alternateReference =
      take(dataflow::publishCanonicalDataflow(alternate, store));
  auto ambiguous =
      loom::dse::bindSpatialMappingFeedbackCandidateGeneratorInputs(
          {dataflowReference, alternateReference}, {spatialMapping},
          spatial.constraints.reference(), completed->resolve({0, 1}),
          simulationInputs.workload, simulationInputs.runtimeInput);
  if (ambiguous)
    fail("one Mapping constraint owner accepted multiple Dataflow roots");
  llvm::consumeError(ambiguous.takeError());

  auto repeated = take(loom::dse::executeDsePlan(planView, store, blobs));
  const auto &repeatedExecution = requireClosedPlan(repeated, 2);
  if (repeatedExecution.resolve({0, 0}) != completed->resolve({0, 0}) ||
      repeatedExecution.resolve({0, 1}) != completed->resolve({0, 1}) ||
      repeatedExecution.resolve({1, 0}) != completed->resolve({1, 0}) ||
      repeatedExecution.resolve({2, 0}) != completed->resolve({2, 0}))
    fail("finite Mapping feedback plan is not deterministic");
}

void spatialMappingFeedbackReplaysAgainstItsSourceWorkload() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();
  auto source = loom::test::buildWideVectorStructuredSource(context);
  const loom::ArtifactRootReference sourceReference =
      take(loom::frontend::publishStructuredProgram(source, store));
  loom::test::PublishedStructuredSimulationInputs sourceInputs =
      loom::test::publishWideVectorStructuredInputs(source, store);
  auto fabric = loom::test::buildFeedbackPruningSpatialCore(store);

  loom::dse::StructuredOwnershipInvocation invocation(
      source, source, sourceInputs.workload, sourceInputs.runtimeInput, fabric,
      loom::defaultResolvedConfig(), {}, 1,
      {100000, 1000000, 256ULL * 1024ULL * 1024ULL});
  loom::dse::StructuredOwnershipInvocationScope invocationScope(invocation);
  loom::dse::StructuredOwnershipGenerationOptions ownership;
  ownership.protocolCallableRoots = {
      loom::test::findStructuredCallable(source, "kernel")};
  auto generated = take(loom::dse::generateStructuredOwnershipCandidates(
      source, source, sourceInputs.workload, sourceInputs.runtimeInput, fabric,
      ownership, store));

  std::optional<loom::ArtifactRootReference> structuredParent;
  std::optional<loom::ArtifactRootReference> dataflowReference;
  for (const loom::ArtifactRootReference &candidate :
       generated.candidates.candidates()) {
    if (candidate == sourceReference)
      continue;
    auto structured =
        take(loom::frontend::importStructuredProgram(candidate, store));
    bool hasVectorAdd = false;
    structured.module().walk([&](loom::SpatialRegionOp spatial) {
      spatial.walk([&](mlir::arith::AddIOp add) {
        auto type = llvm::dyn_cast<mlir::VectorType>(add.getType());
        hasVectorAdd |= type &&
                        type.getShape() == llvm::ArrayRef<std::int64_t>{4} &&
                        type.getElementType().isInteger(64);
      });
    });
    if (!hasVectorAdd)
      continue;
    requireSuccess(loom::dse::detail::StructuredOwnershipInvocationAccess::
                       primeFunctionalReplay(invocation, candidate, store));
    structuredParent = candidate;
    dataflowReference =
        take(invocation.prepareDataflowGeneration(candidate, store));
    break;
  }
  if (!structuredParent || !dataflowReference)
    fail("source-backed feedback fixture produced no vector Dataflow root");

  auto dataflow =
      take(dataflow::importCanonicalDataflow(*dataflowReference, store));
  const loom::ArtifactRootReference techMapping =
      generateTechMapping(*dataflowReference, fabric.reference(), store, blobs);
  auto spatial = generateSpatialFeedbackFixture(*dataflowReference, techMapping,
                                                fabric, store);
  const PublishedSpatialInputs spatialInputs =
      publishVectorSpatialInputs(dataflow, store, 64);
  auto cgraObligation =
      take(loom::dse::prepareCgraSimulationEvidenceObligationTemplate(
          *dataflowReference, fabric.reference(), spatial.mapping,
          spatialInputs.workload, spatialInputs.runtimeInput,
          loom::defaultResolvedConfig(), store, blobs));
  const std::array<loom::dse::EvidenceObligationTemplateRef, 1> cgraRefs = {
      loom::dse::EvidenceObligationTemplateRef(0)};
  auto cgraConfig =
      take(loom::dse::projectResolvedEvidenceObligationSetConfigView(cgraRefs));
  auto cgraBinding = take(
      loom::dse::resolveSpatialMappingEvaluationPromotionAcquisitionBinding(
          cgraConfig));
  auto cgraInputs = take(loom::dse::bindSpatialMappingEvaluationPromotionInputs(
      {spatial.mapping}, {*dataflowReference}, fabric.reference(),
      spatialInputs.workload, spatialInputs.runtimeInput));
  auto cgraOutcome = take(loom::dse::invokePromotionAcquisition(
      cgraInputs, cgraBinding, {cgraObligation}, {{spatial.mapping}, cgraRefs},
      store, blobs));
  const auto *cgraCompleted =
      std::get_if<loom::dse::CompletedPromotionAcquisition>(&cgraOutcome);
  if (!cgraCompleted || cgraCompleted->evidence.size() != 1)
    fail("source-backed feedback fixture produced no CGRA Evidence");
  const loom::ArtifactRootReference cgraEvidence =
      take(loom::evaluation::publishEvaluationEvidence(
          cgraCompleted->evidence.front().evidence, store));

  requireSuccess(loom::dse::registerSpatialMappingFeedbackCandidateGenerator());
  const auto feedbackConfig = buildFeedbackSpatialConfig();
  loom::ResolvedConfig feedbackPlanConfig = buildSpatialResolvedConfig();
  feedbackPlanConfig.dse.planNodes = {loom::dse::GeneratePlanNodeDefinition{
      loom::dse::spatialMappingFeedbackCandidateGeneratorDescriptor()
          .reference(),
      {loom::dse::ExactPlanArtifacts{{*dataflowReference}},
       loom::dse::ExactPlanArtifacts{{spatial.mapping}},
       loom::dse::ExactPlanArtifacts{{spatial.constraints.reference()}},
       loom::dse::ExactPlanArtifacts{{cgraEvidence}},
       loom::dse::ExactPlanArtifacts{{spatialInputs.workload}},
       loom::dse::ExactPlanArtifacts{{spatialInputs.runtimeInput}}},
      feedbackConfig.canonicalViewBytes().vec(),
      feedbackConfig.digest()}};
  auto feedbackPlanView =
      take(loom::dse::projectResolvedDseConfigView(feedbackPlanConfig));
  auto feedbackPlanOutcome =
      take(loom::dse::executeDsePlan(feedbackPlanView, store, blobs));
  const auto *feedbackCompleted =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&feedbackPlanOutcome);
  if (!feedbackCompleted || feedbackCompleted->resolve({0, 0}).size() != 1 ||
      feedbackCompleted->generateInvocations().size() != 1 ||
      feedbackCompleted->generateWorkSummaries().size() != 1)
    fail("source-backed Mapping feedback produced no immutable child");
  const loom::ArtifactRootReference child =
      feedbackCompleted->resolve({0, 0}).front();
  const auto &feedbackInvocation =
      feedbackCompleted->generateInvocations().front();
  const auto &feedbackWork = feedbackCompleted->generateWorkSummaries().front();
  if (feedbackInvocation.lineageEdges.size() != 1 ||
      feedbackWork.units.size() != 1 ||
      feedbackWork.units.front().planned !=
          feedbackWork.units.front().consumed ||
      feedbackWork.units.front().consumed <=
          feedbackInvocation.lineageEdges.size())
    fail("Mapping feedback did not preserve the capability-pruned before/after "
         "candidate domain");
  const std::uint64_t attemptedFeedbackDecisions =
      feedbackWork.units.front().consumed;

  auto functionalObligation = take(
      loom::dse::prepareCanonicalDataflowFunctionalEvidenceObligationTemplate(
          child, *structuredParent, sourceInputs.workloadReference,
          sourceInputs.runtimeInputReference, loom::defaultResolvedConfig(),
          store, blobs));
  const std::array<loom::dse::EvidenceObligationTemplateRef, 1> functionalRefs =
      {loom::dse::EvidenceObligationTemplateRef(0)};
  auto functionalConfig =
      take(loom::dse::projectResolvedEvidenceObligationSetConfigView(
          functionalRefs));
  auto functionalBinding =
      take(loom::dse::resolveDataflowEvaluationPromotionAcquisitionBinding(
          functionalConfig));
  auto functionalInputs = take(loom::dse::bindDataflowEvaluationPromotionInputs(
      {child}, *structuredParent, fabric.reference(),
      sourceInputs.workloadReference, sourceInputs.runtimeInputReference));
  auto functionalOutcome = take(loom::dse::invokePromotionAcquisition(
      functionalInputs, functionalBinding, {functionalObligation},
      {{child}, functionalRefs}, store, blobs));
  const auto *functionalCompleted =
      std::get_if<loom::dse::CompletedPromotionAcquisition>(&functionalOutcome);
  if (!functionalCompleted || functionalCompleted->evidence.size() != 1)
    fail("Mapping feedback child received no functional Evidence");

  if (functionalObligation.findingRequests().size() != 1 ||
      functionalObligation.findingRequests().front().query.kind !=
          loom::evaluation::standard_findings::FunctionalMismatch)
    fail("functional Evidence obligation lost its exact mismatch query");
  auto candidates = take(
      loom::dse::CandidateSet::get(dataflow::canonicalDataflowSchema, {child}));
  auto functionalGate = take(loom::dse::QualityGatePolicy::get(
      {{{loom::dse::FindingGate{0, loom::evaluation::FindingRequestOrdinal(0),
                                loom::dse::RequiredFindingState::Absent}}}}));
  auto promotion = take(loom::dse::promoteCandidates(
      candidates,
      loom::evaluation::models::canonicalDataflowFunctionalCandidateRole(),
      functionalCompleted->evidence, functionalGate,
      loom::dse::AllPassingSelection{}, nullptr, store));
  const auto *promoted = std::get_if<loom::dse::CompletedSelection>(&promotion);
  if (!promoted ||
      promoted->selected != std::vector<loom::ArtifactRootReference>{child} ||
      promoted->satisfiedEvidence.size() != 1)
    fail("functional Evidence did not promote the immutable feedback child");

  const loom::ArtifactIdentity storedConfig =
      take(store.put(loom::ResolvedConfig::artifactSchema,
                     loom::canonicalResolvedConfigBytes(feedbackPlanConfig)));
  if (storedConfig != loom::resolvedConfigIdentity(feedbackPlanConfig))
    fail("feedback Manifest changed the ResolvedConfig identity");
  auto manifestRecords = loom::dse::takeDsePlanGenerateInvocationRecords(
      std::move(feedbackPlanOutcome));
  auto closure = take(loom::dse::DseRunClosure::get(
      take(loom::dse::DseProducerSemanticBuildIdentity::get(
          "loom.test.mapping_feedback_workflow.v1")),
      {*dataflowReference, fabric.reference(), spatial.mapping,
       spatial.constraints.reference(), spatialInputs.workload,
       spatialInputs.runtimeInput},
      feedbackPlanConfig, {cgraEvidence}, store));
  auto manifest = take(loom::dse::InvocationManifest::get(
      std::move(closure), 0, std::nullopt, feedbackPlanConfig, manifestRecords,
      loom::dse::InvocationCompletedSelection{promoted->selected,
                                              promoted->satisfiedEvidence},
      store, blobs));
  auto adoptedManifest = take(loom::dse::adoptInvocationManifest(
      manifest.canonicalBytes(), feedbackPlanConfig, store, blobs));
  if (adoptedManifest.generateRecords().size() != 1 ||
      adoptedManifest.generateRecords().front().workSummary.units.size() != 1 ||
      adoptedManifest.generateRecords()
              .front()
              .workSummary.units.front()
              .consumed != attemptedFeedbackDecisions ||
      adoptedManifest.generateRecords()
              .front()
              .invocation.lineageEdges.size() != 1)
    fail("feedback Manifest lost its capability admission facts or lineage");

  auto selected = take(invocation.materializeSelectedDataflowCandidate(
      *structuredParent, child, store));
  if (selected.dataflowRewriteDerivations.size() != 1 ||
      !selected.functionalReplay ||
      selected.functionalReplay->status !=
          loom::sim::SourceBackedDfgValidationStatus::Equivalent ||
      selected.functionalReplay->dynamicActivations == 0)
    fail("Mapping feedback child lost source-backed replay or typed lineage");

  llvm::outs() << "feedback_workflow before=" << attemptedFeedbackDecisions
               << " after=1 child=" << llvm::toHex(child.artifact.bytes(), true)
               << '\n';
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected exactly one test case name");

  using TestFunction = void (*)();
  TestFunction test =
      llvm::StringSwitch<TestFunction>(argv[1])
          .Case("empty-constraint-owner",
                emptyConstraintOwnerPublishesExactArtifact)
          .Case("root-complete-adapter",
                rootCompleteAdapterPublishesPhysicalMapping)
          .Case("descriptor-and-empty-set", descriptorAndEmptySetAreClosed)
          .Case("finite-set-traversal",
                finiteSetTraversesEveryCanonicalTechMapping)
          .Case("first-verified-lazy-ranking",
                firstVerifiedAvoidsSpeculativeRouteRanking)
          .Case("worker-invariance", candidateWorkerCountPreservesFormalResult)
          .Case("canonical-seed-handoff",
                canonicalSeedHandoffPreservesFormalResult)
          .Case("first-verified-prefix",
                firstVerifiedCandidateRetainsTypedPrefix)
          .Case("typed-interruption", interruptionReturnsTypedSpatialSnapshot)
          .Case("unavailable-negotiation",
                unavailableNegotiationFailsClosedAtConfigProjection)
          .Case("initializer-semantic-limit",
                initializerSemanticLimitIsTypedIncomplete)
          .Case("foreign-fabric-rejection", foreignFabricIsRejectedBeforeSearch)
          .Case("spatial-mapping-promotion",
                spatialMappingPromotionExecutesExactCgraCase)
          .Case("spatial-mapping-lineage",
                spatialMappingPromotionKeepsEveryCandidateLineage)
          .Case("spatial-mapping-feedback",
                spatialMappingFeedbackPublishesNarrowImmutableDataflow)
          .Case("spatial-mapping-feedback-replay",
                spatialMappingFeedbackReplaysAgainstItsSourceWorkload)
          .Default(nullptr);
  if (!test)
    fail("unknown test case: " + llvm::Twine(argv[1]));
  test();
  return EXIT_SUCCESS;
}

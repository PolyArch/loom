#include "RootCompleteSpatialFeedbackTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/Plan.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "llvm/ADT/STLExtras.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/MappingObjective.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/SpatialPnrProblem.h"
#include "RootCompleteSpatialPnrTestSupport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <utility>

namespace loom::test {
namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "root-complete Spatial feedback fixture failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
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

} // namespace

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

RootCompleteSpatialPnrFixture
buildRootCompleteSpatialPnrFixture(mlir::MLIRContext &context, loom::ArtifactStore &store,
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


frontend::StructuredEntityRef
findStructuredCallable(const frontend::StructuredProgramCandidate &candidate,
                       llvm::StringRef name) {
  auto view = take(candidate.view());
  for (const frontend::StructuredEntity &entity :
       view.entities(frontend::StructuredEntityKind::Operation)) {
    auto function =
        llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(entity.operation);
    if (function && function.getSymName() == name)
      return entity.reference;
  }
  fail("callable is absent from the Structured Program: " + name);
}

frontend::StructuredProgramCandidate
buildWideVectorStructuredSource(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  llvm.func internal @kernel(%value: vector<4xi64>) -> vector<4xi64> {
    %sum = arith.addi %value, %value : vector<4xi64>
    llvm.return %sum : vector<4xi64>
  }
  llvm.func @main() -> i32 {
    %value = arith.constant dense<[1, 2, 3, 4]> : vector<4xi64>
    %result = llvm.call @kernel(%value)
        : (vector<4xi64>) -> vector<4xi64>
    %zero = arith.constant 0 : i32
    llvm.return %zero : i32
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse wide-vector Structured source fixture");
  if (llvm::InitializeNativeTarget() ||
      llvm::InitializeNativeTargetAsmPrinter())
    fail("cannot initialize the native target");
  auto target = take(llvm::orc::JITTargetMachineBuilder::detectHost());
  module->getOperation()->setAttr(
      "llvm.target_triple",
      mlir::StringAttr::get(&context, "riscv64-unknown-unknown-elf"));
  module->getOperation()->setAttr(
      "llvm.data_layout",
      mlir::StringAttr::get(&context,
                            take(target.getDefaultDataLayoutForTarget())
                                .getStringRepresentation()));
  return take(frontend::finalizeStructuredProgram(module.get()));
}

PublishedStructuredSimulationInputs publishWideVectorStructuredInputs(
    const frontend::StructuredProgramCandidate &source, ArtifactStore &store) {
  auto view = take(source.view());
  sim::StructuredProgramSimulationWorkload workloadDraft{
      findStructuredCallable(source, "main")};
  workloadDraft.observableContract.returnValue = true;
  auto workload = take(sim::finalizeSimulationWorkload(workloadDraft, view));
  sim::StructuredProgramSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  auto runtime =
      take(sim::finalizeSimulationRuntimeInput(runtimeDraft, workload, view));
  auto workloadReference =
      take(sim::publishSimulationWorkload(workload, store));
  auto runtimeInputReference =
      take(sim::publishSimulationRuntimeInput(runtime, store));
  return {std::move(workload), std::move(runtime), std::move(workloadReference),
          std::move(runtimeInputReference)};
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

GeneratedSpatialFeedbackFixture generateSpatialFeedbackFixture(
    const loom::ArtifactRootReference &dataflowReference,
    const loom::ArtifactRootReference &techMappingReference,
    const loom::fabric::FinalizedFabricRoot &fabric,
    loom::ArtifactStore &store) {
  auto dataflow =
      take(dataflow::importCanonicalDataflow(dataflowReference, store));
  const auto &dataflowView = dataflow.view();
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
  const auto &view = dataflow.view();
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
                           unsigned laneWidth) {
  const auto &view = dataflow.view();
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

} // namespace loom::test

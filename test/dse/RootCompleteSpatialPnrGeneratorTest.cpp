#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Inspection/SpatialMappingInspection.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <limits>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

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

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i32) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant 7 : i32
    %thread = dataflow.thread.launch @worker(%value)
        : (i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildAlternateDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i32) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant 8 : i32
    %thread = dataflow.thread.launch @worker(%value)
        : (i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse alternate Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

void addTokenSyncFu(loom::adg::PeBuilder &pe,
                    llvm::ArrayRef<loom::adg::PeValue> inputs,
                    const loom::adg::PortType &type,
                    std::uint32_t payloadWidth) {
  const std::vector<loom::adg::PortType> types(4, type);
  auto fu = take(pe.addFu(inputs, loom::adg::FuSpec{types, types}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    fuInputs.push_back(take(fu.input(ordinal)));
  auto operation = take(fu.addOperation(
      fuInputs, loom::adg::OperationCapabilitySpec{
                    ::fabric::ImplementationFamilyId::TokenSync,
                    ::fabric::RoutedTokenParams{payloadWidth, 4},
                    {::dataflow::OperationSchemaId::DataflowSync},
                    types,
                    ::fabric::oneCycleElasticOperationResourceContract()}));
  requireSuccess(fu.addCapabilityTemplate(
      loom::adg::FuCapabilityTemplateSpec{{operation}, {}}));
  std::vector<loom::adg::FuValue> outputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    outputs.push_back(take(operation.output(ordinal)));
  requireSuccess(fu.close(outputs));
}

loom::fabric::FinalizedFabricRoot
buildSpatialCore(loom::ArtifactStore &store, std::uint32_t payloadWidth = 128) {
  const loom::adg::PortType payloadType =
      take(loom::adg::PortType::bits(payloadWidth));
  const std::vector<loom::adg::PortType> types(4, payloadType);
  loom::adg::DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("sync", types, types));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe = take(
      spatial.addPe(spatialInputs, loom::adg::PeSpec::spatial(types, types)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  addTokenSyncFu(pe, peInputs, payloadType, payloadWidth);
  requireSuccess(pe.close());
  std::vector<loom::adg::SpatialValue> outputs;
  for (std::size_t ordinal = 0; ordinal != types.size(); ++ordinal)
    outputs.push_back(take(pe.output(ordinal)));
  requireSuccess(spatial.close(outputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("SpatialCore fixture did not publish exactly one Fabric root");
  return design.roots().front();
}

loom::ResolvedObjectiveCatalogs availableSpatialObjectiveCatalogs() {
  loom::ResolvedObjectiveCatalogs catalogs;
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  catalogs.dimensions = {
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::UnroutedObligation},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
      {loom::ResolvedMappingViolationObjectiveSource{
           loom::ResolvedPnrViolationKind::CapacityOveruse},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
      {loom::ResolvedMappingMeasureObjectiveSource{static_cast<std::uint32_t>(
           loom::pnr::MappingMeasureKind::TotalSelectedTraversalClaim)},
       loom::ResolvedObjectiveDirection::Minimize,
       loom::resolvedObjectiveInteger(0), loom::resolvedObjectiveInteger(1), 0,
       maximum},
  };
  catalogs.weightedLevels = {
      {{{0, 1}, {1, 1}, {2, 1}}},
  };
  catalogs.totalOrderings = {{{0}}};
  return catalogs;
}

loom::ResolvedConfig buildSpatialResolvedConfig() {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.objectiveCatalogs = availableSpatialObjectiveCatalogs();
  resolved.dse.spatialPnr.temporaryViolations.admitted = {
      loom::ResolvedPnrViolationKind::UnroutedObligation,
      loom::ResolvedPnrViolationKind::CapacityOveruse,
  };
  resolved.dse.spatialPnr.objectiveSelection = {0, 0, {}};
  auto &search = resolved.dse.spatialPnr.search;
  search.initializer.seedAttemptCount = 2;
  search.actionProposal = {0, 1, 0};
  search.annealing.calibrationProposalCount = 1;
  search.annealing.fallbackTemperature = 1;
  search.annealing.minimumTemperature = 1;
  search.annealing.coolingRatio = {1, 2};
  search.annealing.proposalsPerLevelBase = 1;
  search.annealing.proposalsPerMovableDecision = 0;
  search.exactRepair = {loom::ResolvedPnrExactRepairKind::Disabled, 0, 0};
  return resolved;
}

loom::pnr::ResolvedPnrConfigView buildSpatialConfig() {
  return take(loom::pnr::projectResolvedSpatialPnrConfigView(
      buildSpatialResolvedConfig()));
}

loom::ArtifactRootReference
generateTechMapping(const loom::ArtifactRootReference &dataflow,
                    const loom::ArtifactRootReference &fabric,
                    loom::ArtifactStore &store) {
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
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorInvocation>(&outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 1)
    fail("root-complete TechMapping fixture did not publish one candidate");
  return completed->outputBindings.front().artifacts.front();
}

struct Fixture final {
  dataflow::CanonicalDataflowArtifact dataflow;
  loom::ArtifactRootReference dataflowReference;
  loom::fabric::FinalizedFabricRoot fabric;
  loom::ArtifactRootReference techMappingReference;
};

Fixture buildFixture(mlir::MLIRContext &context, loom::ArtifactStore &store) {
  auto dataflow = buildDataflow(context);
  auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflow, store));
  auto fabric = buildSpatialCore(store);
  auto techMappingReference =
      generateTechMapping(dataflowReference, fabric.reference(), store);
  return {std::move(dataflow), std::move(dataflowReference), std::move(fabric),
          std::move(techMappingReference)};
}

void emptyConstraintOwnerPublishesExactArtifact() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store);
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
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store);
  requireSuccess(
      loom::dse::registerRootCompleteTechMappingCandidateGenerator());
  requireSuccess(loom::dse::registerRootCompleteSpatialPnrCandidateGenerator());
  loom::ResolvedConfig resolved = buildSpatialResolvedConfig();
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
           loom::dse::ExactPlanArtifacts{{fixture.fabric.reference()}}},
          spatialConfig.canonicalViewBytes().vec(),
          spatialConfig.digest()},
  };
  auto view = take(loom::dse::projectResolvedDseConfigView(resolved));
  auto outcome = take(loom::dse::executeDsePlan(view, store));
  const auto *completed =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&outcome);
  if (!completed || completed->generateInvocations().size() != 2 ||
      completed->resolve(loom::dse::PlanOutputRef{0, 0}).size() != 1 ||
      completed->resolve(loom::dse::PlanOutputRef{1, 0}).size() != 1)
    fail("root-complete Mapping plan did not publish T then SpatialMapping");
  const auto &spatialInvocation = completed->generateInvocations().back();
  if (spatialInvocation.lineageEdges.size() != 1)
    fail("root-complete Spatial invocation lost mechanical lineage");
  const auto &edge = spatialInvocation.lineageEdges.front();
  if (edge.kind !=
          loom::dse::CandidateGeneratorLineageEdgeKind::MechanicalDerivation ||
      !edge.parents.empty() || !edge.ownerPayload.empty())
    fail("root-complete Spatial adapter published non-mechanical lineage");

  auto dataflow = take(fixture.dataflow.view());
  auto tech = take(loom::mapping::importTechMapping(
      completed->resolve(loom::dse::PlanOutputRef{0, 0}).front(), store));
  auto spatial = take(loom::mapping::importSpatialMapping(
      completed->resolve(loom::dse::PlanOutputRef{1, 0}).front(), store));
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
          {completed->resolve(loom::dse::PlanOutputRef{1, 0}).front()},
          fixture.fabric.reference()));
  auto spatialBinding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          spatialConfig));
  auto wrongProfile = loom::dse::invokeCandidateGenerator(
      wrongProfileInputs, spatialBinding, store);
  if (wrongProfile)
    fail("SpatialMapping was accepted in the TechMapping input slot");
  const std::string wrongProfileMessage =
      llvm::toString(wrongProfile.takeError());
  if (!llvm::StringRef(wrongProfileMessage).contains("TechMapping"))
    fail("wrong Mapping profile rejection lost its owner diagnostic");

  auto repeated = take(loom::dse::executeDsePlan(view, store));
  const auto *repeatedCompleted =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&repeated);
  if (!repeatedCompleted ||
      repeatedCompleted->resolve(loom::dse::PlanOutputRef{0, 0}) !=
          completed->resolve(loom::dse::PlanOutputRef{0, 0}) ||
      repeatedCompleted->resolve(loom::dse::PlanOutputRef{1, 0}) !=
          completed->resolve(loom::dse::PlanOutputRef{1, 0}))
    fail("root-complete Mapping plan is not deterministic");
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
      descriptor.inputSlots.size() != 2 || descriptor.outputSlots.size() != 1 ||
      descriptor.workUnits.size() != 10 ||
      descriptor.inputSlots[0].semanticRole != "tech_mapping" ||
      descriptor.inputSlots[0].cardinality !=
          loom::dse::PlanValueCardinality::FiniteSet ||
      descriptor.inputSlots[1].semanticRole != "fabric" ||
      descriptor.inputSlots[1].cardinality !=
          loom::dse::PlanValueCardinality::ExactlyOne ||
      descriptor.resolvedConfigView.schemaDescriptorBytes !=
          config.schemaDescriptorBytes())
    fail("root-complete Spatial descriptor is not closed over exact T/F");
  if (descriptor.workUnits.size() !=
      loom::dse::spatialPnrCandidateGeneratorWorkUnits.size())
    fail("root-complete Spatial descriptor copied the PnR work-unit catalog");
  for (std::size_t ordinal = 0; ordinal != descriptor.workUnits.size();
       ++ordinal) {
    const auto &actual = descriptor.workUnits[ordinal];
    const auto &owner =
        loom::dse::spatialPnrCandidateGeneratorWorkUnits[ordinal];
    if (!(actual.unit == owner.unit) || actual.spelling != owner.spelling)
      fail("root-complete Spatial descriptor diverged from PnR work units");
  }

  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  auto fabric = buildSpatialCore(store);
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          {}, fabric.reference()));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorInvocation>(&outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      !completed->outputBindings.front().artifacts.empty() ||
      !completed->lineageEdges.empty())
    fail("empty TechMapping set did not propagate as completed empty");
}

void finiteSetTraversesEveryCanonicalTechMapping() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store);
  auto alternateDataflow = buildAlternateDataflow(context);
  auto alternateDataflowReference =
      take(dataflow::publishCanonicalDataflow(alternateDataflow, store));
  auto alternateTechMapping = generateTechMapping(
      alternateDataflowReference, fixture.fabric.reference(), store);

  std::array<loom::ArtifactRootReference, 2> techMappings = {
      fixture.techMappingReference, alternateTechMapping};
  if (loom::artifactRootReferenceLess(techMappings[1], techMappings[0]))
    std::swap(techMappings[0], techMappings[1]);
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          techMappings, fixture.fabric.reference()));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          buildSpatialConfig()));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorInvocation>(&outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 2 ||
      completed->lineageEdges.size() != 2)
    fail("finite TechMapping set did not produce one Spatial set");

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
}

void unavailableNegotiationIsTypedIncomplete() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store);
  loom::ResolvedConfig resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.routing.negotiation =
      loom::ResolvedDualSubgradientPolicy{
          loom::ResolvedDualDirectionKernel::ProjectedSigned,
          std::nullopt,
          {loom::ResolvedDualStepScheduleKind::Constant, 1, 0, 0, 0}};
  auto config = take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          {fixture.techMappingReference}, fixture.fabric.reference()));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store));
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorInvocation>(&outcome);
  if (!incomplete ||
      incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::Unsupported ||
      incomplete->retainedOutputBindings.size() != 1 ||
      !incomplete->retainedOutputBindings.front().artifacts.empty() ||
      !incomplete->lineageEdges.empty())
    fail("unavailable routing kernel did not remain typed Unsupported");
}

void initializerSemanticLimitIsTypedIncomplete() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store);
  loom::ResolvedConfig resolved = buildSpatialResolvedConfig();
  resolved.dse.spatialPnr.search.initializer.seedAttemptCount = 1;
  resolved.dse.spatialPnr.search.initializer.assignmentAttemptLimitPerSeed = 1;
  auto config = take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          {fixture.techMappingReference}, fixture.fabric.reference()));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          config));
  auto outcome =
      take(loom::dse::invokeCandidateGenerator(inputs, binding, store));
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorInvocation>(&outcome);
  if (!incomplete)
    fail("one initializer assignment unexpectedly completed Spatial PnR");
  if (incomplete->reason !=
          loom::dse::CandidateGeneratorIncompleteReason::SemanticLimitReached ||
      incomplete->retainedOutputBindings.size() != 1 ||
      !incomplete->retainedOutputBindings.front().artifacts.empty() ||
      !incomplete->lineageEdges.empty())
    fail("initializer semantic limit did not remain typed Incomplete");
}

void foreignFabricIsRejectedBeforeSearch() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  Fixture fixture = buildFixture(context, store);
  auto foreignFabric = buildSpatialCore(store, 64);
  auto inputs =
      take(loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
          {fixture.techMappingReference}, foreignFabric.reference()));
  auto binding =
      take(loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
          buildSpatialConfig()));
  auto outcome = loom::dse::invokeCandidateGenerator(inputs, binding, store);
  if (outcome)
    fail("root-complete Spatial adapter accepted a foreign Fabric");
  const std::string message = llvm::toString(outcome.takeError());
  if (!llvm::StringRef(message).contains("foreign Fabric"))
    fail("foreign Fabric rejection lost its exact-owner diagnostic");
}

} // namespace

int main() {
  emptyConstraintOwnerPublishesExactArtifact();
  rootCompleteAdapterPublishesPhysicalMapping();
  descriptorAndEmptySetAreClosed();
  finiteSetTraversesEveryCanonicalTechMapping();
  unavailableNegotiationIsTypedIncomplete();
  initializerSemanticLimitIsTypedIncomplete();
  foreignFabricIsRejectedBeforeSearch();
  return EXIT_SUCCESS;
}

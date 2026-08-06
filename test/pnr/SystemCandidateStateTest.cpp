#include "PnR/System/SystemCandidateState.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/ResourceContract.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/System/SystemPnrProblem.h"
#include "PnR/System/SystemPnrSearchDomain.h"

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
#include <optional>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System CandidateState anchor failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void requireFailureContains(llvm::Expected<T> value,
                            llvm::StringRef diagnostic) {
  if (value)
    fail("adverse CandidateState input unexpectedly succeeded");
  const std::string actual = llvm::toString(value.takeError());
  require(llvm::StringRef(actual).contains(diagnostic),
          "adverse diagnostic changed: " + actual);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-system-candidate-state", path_);
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
      %value: i32) ctrl (%ctrl: none) iv (%iv: index) {
    %first_result, %first_done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    %second_result, %second_done = dataflow.graph.launch @sync deps(%first_done)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %second_done : none
  }
  func.func private @host() {
    %value = arith.constant 7 : i32
    %extent = arith.constant 8 : index
    %first = dataflow.thread.launch @worker(%value) grid(%extent)
        : (i32) -> !dataflow.thread_token
    %second = dataflow.thread.launch @worker(%value) grid(%extent)
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

loom::ResolvedObjectiveCatalogs spatialObjectiveCatalogs() {
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
  catalogs.weightedLevels = {{{{0, 1}, {1, 1}, {2, 1}}}};
  catalogs.totalOrderings = {{{0}}};
  return catalogs;
}

loom::ResolvedConfig buildResolvedConfig() {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.objectiveCatalogs = spatialObjectiveCatalogs();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  resolved.dse.spatialPnr.temporaryViolations.admitted = {
      loom::ResolvedPnrViolationKind::UnroutedObligation,
      loom::ResolvedPnrViolationKind::CapacityOveruse};
  resolved.dse.spatialPnr.objectiveSelection = {0, 0, {}};
  auto &search = resolved.dse.spatialPnr.search;
  search.initializer.seedAttemptCount = 1;
  search.actionProposal = {0, 1, 0};
  search.annealing.calibrationProposalCount = 1;
  search.annealing.fallbackTemperature = 1;
  search.annealing.minimumTemperature = 1;
  search.annealing.coolingRatio = {1, 2};
  search.annealing.proposalsPerLevelBase = 1;
  search.annealing.proposalsPerMovableDecision = 0;
  search.exactRepair = {loom::ResolvedPnrExactRepairKind::Disabled, 0, 0};
  resolved.dse.systemPnr.temporaryViolations.admitted = {
      loom::ResolvedPnrViolationKind::UnroutedObligation,
      loom::ResolvedPnrViolationKind::CapacityOveruse};
  resolved.dse.systemPnr.objectiveSelection = {0, 0, {}};
  return resolved;
}

::fabric::ResourceContract exclusiveResourceContract() {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {
      {::fabric::StateKey(0),
       {{::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1),
         ::fabric::CapacityUnits(0)}}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContracts = {{::fabric::TimingContractKey(0), {0, 1}}};
  declaration.usePatterns = {
      {::fabric::UsePatternKey(0),
       ::fabric::RequesterKey(0),
       ::fabric::EligibilityKey(0),
       ::fabric::EventKey(0),
       ::fabric::EventKey(1),
       std::nullopt,
       ::fabric::TimingContractKey(0),
       {{::fabric::ClaimKey(0), ::fabric::StateKey(0),
         ::fabric::CapacityDimensionKey(0), ::fabric::CapacityUnits(1)}},
       {{{::fabric::ClaimKey(0)}}}}};
  return take(::fabric::ResourceContract::create(std::move(declaration)));
}

loom::fabric::InstructionCoreMicroarchitecturalRealization
inOrderMicroarchitecture() {
  loom::fabric::InstructionCoreCommonDeclaration common{
      1,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1},
       {loom::fabric::InstructionOperationClass::LoadStore, 1, 2, 1}},
      exclusiveResourceContract()};
  loom::fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1,
                                                             1, 1, 4, 2};
  return take(
      loom::fabric::InstructionCoreMicroarchitecturalRealization::createInOrder(
          std::move(common), pipeline));
}

loom::adg::FinalizedFabricDesign buildSpatialModule(loom::ArtifactStore &store,
                                                    bool addBoundaryBuffer) {
  loom::adg::DesignBuilder design(store);
  auto expansion = take(loom::adg::expandBuiltinSpatialCore(
      design, loom::adg::BuiltinTargetPreset::Small));
  if (addBoundaryBuffer) {
    const auto bits128 = take(loom::adg::PortType::bits(128));
    expansion.outputs.front() = take(expansion.spatialCore.addFifo(
        expansion.outputs.front(), loom::adg::FifoSpec{bits128, 2, true}));
  }
  if (llvm::Error error = expansion.spatialCore.close(expansion.outputs))
    fail(llvm::toString(std::move(error)));
  auto finalized = take(std::move(design).finalize());
  require(finalized.roots().size() == 1,
          "SpatialCore fixture did not publish one Module root");
  return finalized;
}

loom::adg::FinalizedFabricDesign buildHeterogeneousSystem(
    loom::ArtifactStore &store,
    const loom::fabric::FinalizedFabricRoot &primaryModule,
    const loom::fabric::FinalizedFabricRoot &alternateModule) {
  loom::adg::DesignBuilder design(store);
  auto system = take(loom::adg::expandBuiltinSystem(
      design, loom::adg::BuiltinTargetPreset::Small, primaryModule));
  auto imported = take(system.importSpatialCore(alternateModule));
  const auto architecture =
      take(loom::adg::getBuiltinInstructionCoreArchitecture());
  auto extraCore = take(
      system.addAccCore(architecture, inOrderMicroarchitecture(), imported));

  const auto bits128 = take(loom::adg::PortType::bits(128));
  const auto transportContract = exclusiveResourceContract();
  std::vector<loom::adg::HardwareDomainMember> domainMembers = {
      extraCore.instructionCoreDomainMember(),
      extraCore.spatialCoreDomainMember()};
  for (std::uint32_t gateway = 0; gateway != 2; ++gateway) {
    auto transport = take(
        system.addTransportResource({{bits128}, {bits128}, transportContract}));
    auto pattern = take(system.addTransferPattern(transport, 0, {0}, 0));
    if (llvm::Error error =
            system.connect(take(extraCore.spatialTransportOutput(gateway)),
                           take(transport.input(0))))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error =
            system.connect(take(transport.output(0)),
                           take(extraCore.spatialTransportInput(gateway))))
      fail(llvm::toString(std::move(error)));
    domainMembers.push_back(transport.domainMember());
    domainMembers.push_back(pattern.domainMember());
  }
  auto domain = take(system.createHardwareDomain());
  auto clock = take(loom::fabric::ClockDomainContractRecord::create(1'000, 0));
  if (llvm::Error error = domain.close(domainMembers, std::move(clock)))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = system.close())
    fail(llvm::toString(std::move(error)));
  auto finalized = take(std::move(design).finalize());
  require(finalized.roots().size() == 1,
          "heterogeneous fixture did not publish one System root");
  return finalized;
}

loom::ArtifactRootReference
generateSpatialMapping(const dataflow::CanonicalDataflowProgramView &dataflow,
                       const loom::fabric::FinalizedFabricRoot &module,
                       const loom::ResolvedConfig &resolved,
                       loom::ArtifactStore &store) {
  const auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  auto techOutcome = loom::mapping::generateTechMappings(
      {dataflow, covers, module.view(), techConfig, store});
  const auto *techCandidates =
      std::get_if<loom::mapping::GeneratedTechMappings>(&techOutcome);
  require(techCandidates && techCandidates->candidates.size() == 1,
          "TechMapping fixture did not produce one candidate");
  auto tech = take(loom::mapping::importTechMapping(
      techCandidates->candidates.front(), store));
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, tech.view(), module.view(), store));
  const auto spatialConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  auto spatialOutcome = loom::pnr::generateSpatialMappings(
      {dataflow, tech.view(), module.view(), spatialConfig, constraints.view(),
       store});
  const auto *spatialCandidates =
      std::get_if<loom::pnr::GeneratedSpatialMappings>(&spatialOutcome);
  require(spatialCandidates && spatialCandidates->candidates.size() == 1,
          "SpatialMapping fixture did not produce one candidate");
  return spatialCandidates->candidates.front();
}

} // namespace

int main() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  auto primaryDesign = buildSpatialModule(store, false);
  auto alternateDesign = buildSpatialModule(store, true);
  auto design = buildHeterogeneousSystem(store, primaryDesign.roots().front(),
                                         alternateDesign.roots().front());
  const auto &systemRoot = design.roots().front();
  auto system = take(loom::fabric::requireSystemRoot(systemRoot.view()));
  require(systemRoot.directDependencies().size() == 2,
          "heterogeneous System did not retain both SpatialCores");

  const loom::ResolvedConfig resolved = buildResolvedConfig();
  std::vector<loom::ArtifactRootReference> spatialMappings;
  for (const auto &dependency : systemRoot.directDependencies()) {
    auto module =
        take(loom::fabric::importEntireFabricRoot(dependency.root, store));
    spatialMappings.push_back(
        generateSpatialMapping(dataflow, module, resolved, store));
  }
  std::vector<dataflow::RootThreadLaunchRef> roots;
  for (const dataflow::CanonicalRootThreadLaunchView &root :
       dataflow.rootThreadLaunches())
    roots.push_back(root.ref);
  auto constraints =
      take(loom::mapping::finalizeEmptySystemMappingConstraintSet(
          dataflow, system, roots, store));
  auto partition = take(loom::pnr::projectWholeDomainPresburgerPartitionPlan(
      dataflow, constraints.view().rootThreadLaunches()));
  auto searchDomain = take(loom::pnr::projectSystemPnrSearchDomain(
      dataflow, system, constraints, partition, spatialMappings, store));
  const auto config =
      take(loom::pnr::projectResolvedSystemPnrConfigView(resolved));
  auto problem = take(loom::pnr::freezeSystemPnrProblem(
      dataflow, system, searchDomain, config, constraints, store));

  require(problem->threadDecisions().size() == 2 &&
              problem->graphDecisions().size() == 4,
          "frozen System problem merged execution atoms");
  require(problem->accCores().size() == 5 &&
              problem->spatialMappings().size() == 2 &&
              problem->targetClasses().size() == 2,
          "frozen System target catalogs are incomplete");

  auto first = take(loom::pnr::initializeCanonicalSystemCandidate(problem));
  auto second = take(loom::pnr::initializeCanonicalSystemCandidate(problem));
  require(first.state->threadChoices() == second.state->threadChoices() &&
              first.state->graphChoices() == second.state->graphChoices() &&
              first.assignmentAttempts == second.assignmentAttempts,
          "canonical System initializer is not deterministic");
  if (llvm::Error error = first.state->verify())
    fail(llvm::toString(std::move(error)));
  for (loom::pnr::PnrIndex decision = 0;
       decision != problem->graphDecisions().size(); ++decision) {
    const auto graphDomain = problem->graphChoiceCatalogOrdinals(decision);
    const auto selectedMapping =
        graphDomain[first.state->graphChoice(decision)];
    const auto threadDomain = problem->threadChoiceCatalogOrdinals(
        problem->graphDecisions()[decision].launch.rootThreadLaunch ==
                problem->threadDecisions().front().root
            ? 0
            : 1);
    const auto selectedCore = threadDomain[first.state->threadChoice(
        problem->graphDecisions()[decision].launch.rootThreadLaunch ==
                problem->threadDecisions().front().root
            ? 0
            : 1)];
    require(problem->spatialMappingTargetClass(selectedMapping) ==
                problem->accCoreTargetClass(selectedCore),
            "canonical initializer selected incompatible execution targets");
  }

  std::vector<loom::pnr::PnrIndex> threadChoices(
      problem->threadDecisions().size(), 0);
  std::vector<loom::pnr::PnrIndex> graphChoices(
      problem->graphDecisions().size(), 0);
  require(problem->threadChoiceCatalogOrdinals(0).size() > 1,
          "fixture needs two compatible AccCore choices");
  const auto initialThreadDomain = problem->threadChoiceCatalogOrdinals(0);
  loom::pnr::PnrIndex sameClassFirst = 0;
  loom::pnr::PnrIndex sameClassSecond = 0;
  loom::pnr::PnrIndex sharedClass = 0;
  bool foundSameClassAlternative = false;
  for (loom::pnr::PnrIndex firstChoice = 0;
       firstChoice != initialThreadDomain.size() && !foundSameClassAlternative;
       ++firstChoice)
    for (loom::pnr::PnrIndex secondChoice = firstChoice + 1;
         secondChoice != initialThreadDomain.size(); ++secondChoice)
      if (problem->accCoreTargetClass(initialThreadDomain[firstChoice]) ==
          problem->accCoreTargetClass(initialThreadDomain[secondChoice])) {
        sameClassFirst = firstChoice;
        sameClassSecond = secondChoice;
        sharedClass =
            problem->accCoreTargetClass(initialThreadDomain[firstChoice]);
        foundSameClassAlternative = true;
        break;
      }
  require(foundSameClassAlternative,
          "fixture needs two AccCores in one SpatialCore target class");

  for (loom::pnr::PnrIndex decision = 0;
       decision != problem->threadDecisions().size(); ++decision) {
    const auto domain = problem->threadChoiceCatalogOrdinals(decision);
    bool found = false;
    for (loom::pnr::PnrIndex choice = 0; choice != domain.size(); ++choice)
      if (problem->accCoreTargetClass(domain[choice]) == sharedClass) {
        threadChoices[decision] = choice;
        found = true;
        break;
      }
    require(found, "thread domain lost a compatible target class");
  }
  for (loom::pnr::PnrIndex decision = 0;
       decision != problem->graphDecisions().size(); ++decision) {
    const auto domain = problem->graphChoiceCatalogOrdinals(decision);
    bool found = false;
    for (loom::pnr::PnrIndex choice = 0; choice != domain.size(); ++choice)
      if (problem->spatialMappingTargetClass(domain[choice]) == sharedClass) {
        graphChoices[decision] = choice;
        found = true;
        break;
      }
    require(found, "graph domain lost a compatible target class");
  }
  threadChoices[0] = sameClassFirst;
  auto sameClassBase = take(loom::pnr::SystemCandidateState::create(
      problem, {threadChoices, graphChoices}));
  threadChoices[0] = sameClassSecond;
  auto alternate = take(loom::pnr::SystemCandidateState::create(
      problem, {threadChoices, graphChoices}));
  if (llvm::Error error = alternate->verify())
    fail(llvm::toString(std::move(error)));
  require(alternate->selectedAccCore(0) != sameClassBase->selectedAccCore(0),
          "explicit thread choice did not change the selected AccCore");

  const auto firstThreadDomain = problem->threadChoiceCatalogOrdinals(0);
  const auto firstGraphDomain = problem->graphChoiceCatalogOrdinals(0);
  bool foundMismatch = false;
  for (loom::pnr::PnrIndex threadChoice = 0;
       threadChoice != firstThreadDomain.size() && !foundMismatch;
       ++threadChoice)
    for (loom::pnr::PnrIndex graphChoice = 0;
         graphChoice != firstGraphDomain.size() && !foundMismatch;
         ++graphChoice)
      if (problem->accCoreTargetClass(firstThreadDomain[threadChoice]) !=
          problem->spatialMappingTargetClass(firstGraphDomain[graphChoice])) {
        threadChoices.assign(problem->threadDecisions().size(), threadChoice);
        graphChoices.assign(problem->graphDecisions().size(), graphChoice);
        requireFailureContains(loom::pnr::SystemCandidateState::create(
                                   problem, {threadChoices, graphChoices}),
                               "target classes are incompatible");
        foundMismatch = true;
      }
  require(foundMismatch,
          "heterogeneous fixture did not expose an incompatible target pair");

  threadChoices.assign(problem->threadDecisions().size(), 0);
  graphChoices.assign(problem->graphDecisions().size(), 0);
  threadChoices[0] = problem->threadChoiceCatalogOrdinals(0).size();
  requireFailureContains(loom::pnr::SystemCandidateState::create(
                             problem, {threadChoices, graphChoices}),
                         "thread choice is outside its H domain");
  threadChoices.pop_back();
  requireFailureContains(loom::pnr::SystemCandidateState::create(
                             problem, {threadChoices, graphChoices}),
                         "thread choice count does not match H");

  llvm::outs() << "System CandidateState anchors passed\n";
  return EXIT_SUCCESS;
}

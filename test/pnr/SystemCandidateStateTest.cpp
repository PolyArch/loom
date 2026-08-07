#include "PnR/System/SystemCandidateState.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/System/SystemMappingMaterializer.h"
#include "PnR/System/SystemPnrProblem.h"
#include "PnR/System/SystemPnrSearchDomain.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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

void requireVerificationFailureContains(mlir::Operation *operation,
                                        llvm::StringRef expected) {
  std::vector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      operation->getContext(), [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return mlir::success();
      });
  require(mlir::failed(mlir::verify(operation)),
          "adverse SystemMapping operation unexpectedly verified");
  require(llvm::any_of(diagnostics,
                       [&](const std::string &diagnostic) {
                         return llvm::StringRef(diagnostic).contains(expected);
                       }),
          "adverse SystemMapping diagnostic changed");
}

mlir::DenseI8ArrayAttr bytesAttr(mlir::MLIRContext *context,
                                 llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

std::vector<std::uint8_t> unsignedBytes(mlir::DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> result;
  result.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

::mapping::ArtifactRootReferenceAttr
rootReferenceAttr(mlir::MLIRContext *context,
                  const loom::ArtifactRootReference &reference) {
  return ::mapping::ArtifactRootReferenceAttr::get(
      context,
      bytesAttr(context, loom::encodeArtifactRootReference(reference)));
}

loom::CanonicalSemanticBytes rawSystemBytes(::mapping::SystemOp root) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  root.print(stream, mlir::OpPrintingFlags().enableDebugInfo(false));
  stream << '\n';
  stream.flush();
  return loom::CanonicalSemanticBytes(
      std::vector<std::uint8_t>(text.begin(), text.end()));
}

::mapping::SystemPresburgerCellAttr
withFirstCoordinateLowerBound(::mapping::SystemPresburgerCellAttr cell,
                              std::int64_t lowerBound) {
  require(cell.getDimensionCount() != 0,
          "Presburger test cell has no logical coordinate");
  llvm::SmallVector<mlir::Attribute> inequalities(
      cell.getInequalities().begin(), cell.getInequalities().end());
  std::vector<std::int64_t> row(
      static_cast<std::size_t>(cell.getDimensionCount()) +
          cell.getSymbolCount() + 1,
      0);
  row.front() = 1;
  row.back() = -lowerBound;
  inequalities.push_back(mlir::DenseI64ArrayAttr::get(cell.getContext(), row));
  return ::mapping::SystemPresburgerCellAttr::get(
      cell.getContext(), cell.getDimensionCount(), cell.getSymbolCount(),
      cell.getEqualities(),
      mlir::ArrayAttr::get(cell.getContext(), inequalities));
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
        values(%first_result) stream_inputs() memories() stream_outputs()
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
                                        expansion.outputs.front(),
                                        loom::adg::FifoSpec{bits128, 2, true}))
                                    .value();
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
    const loom::fabric::FinalizedFabricRoot &baselineSystem,
    const loom::fabric::FinalizedFabricRoot &primaryModule,
    const loom::fabric::FinalizedFabricRoot &alternateModule,
    mlir::MLIRContext &context) {
  auto baseline =
      take(loom::fabric::requireSystemRoot(baselineSystem.view()));
  require(!baseline.artifact().systemMemoryServices().empty() &&
              !baseline.artifact().systemServiceEndpoints().empty(),
          "builtin System has no memory service capability source");
  const auto *memoryContract = baseline.memoryService(
      baseline.artifact().systemMemoryServices().front());
  const auto *memoryCapabilities = baseline.serviceEndpointCapabilities(
      baseline.artifact().systemServiceEndpoints().front());
  require(memoryContract && memoryCapabilities,
          "builtin System memory service contract is incomplete");

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
  std::vector<loom::adg::SystemTransportEndpoint> requestCarriers;
  std::vector<loom::adg::SystemTransportEndpoint> responseCarriers;
  std::vector<loom::adg::SystemTransportEndpoint> occurrenceRequestCarriers;
  std::vector<loom::adg::SystemTransportEndpoint> occurrenceResponseCarriers;
  for (std::uint32_t gateway = 0; gateway != 2; ++gateway) {
    auto transport = take(
        system.addTransportResource({{bits128}, {bits128}, transportContract}));
    auto pattern = take(system.addTransferPattern(transport, 0, {0}, 0));
    auto input = take(transport.input(0));
    auto output = take(transport.output(0));
    requestCarriers.push_back(input);
    responseCarriers.push_back(output);
    auto occurrenceRequest = take(extraCore.spatialTransportOutput(gateway));
    auto occurrenceResponse = take(extraCore.spatialTransportInput(gateway));
    occurrenceRequestCarriers.push_back(occurrenceRequest);
    occurrenceResponseCarriers.push_back(occurrenceResponse);
    if (llvm::Error error = system.connect(occurrenceRequest, input))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error = system.connect(output, occurrenceResponse))
      fail(llvm::toString(std::move(error)));
    domainMembers.push_back(transport.domainMember());
    domainMembers.push_back(pattern.domainMember());
  }
  auto domain = take(system.createHardwareDomain());
  auto rate = take(system.createServiceRate(
      domain, 1, 1, 4,
      loom::fabric::ServiceProgress(
          std::in_place_type<::fabric::FairEventual>)));
  std::vector<loom::fabric::CanonicalServiceCapabilityRecord>
      localMemoryCapabilities;
  localMemoryCapabilities.reserve(memoryCapabilities->capabilities().size());
  for (const auto &capability : memoryCapabilities->capabilities())
    localMemoryCapabilities.push_back(
        take(loom::fabric::CanonicalServiceCapabilityRecord::create(
            capability.kind(), capability.role(), capability.domain(), rate)));
  auto memoryCapabilitySet =
      take(loom::fabric::CanonicalServiceCapabilitySet::create(
          std::move(localMemoryCapabilities)));
  auto memoryService = take(system.addMemoryService(*memoryContract));
  auto memoryEndpoint =
      take(system.addServiceEndpoint(memoryService, memoryCapabilitySet));
  auto spatialMemory = take(extraCore.spatialMemoryManager(0));
  if (llvm::Error error =
          system.attachSpatialMemory(spatialMemory, memoryEndpoint))
    fail(llvm::toString(std::move(error)));
  auto memoryEndpointRef = take(memoryEndpoint.memory());
  for (const auto &capability : memoryCapabilitySet.capabilities()) {
    const auto legCount =
        dataflow::semantics::getCanonicalServiceLegCount(capability.kind());
    for (dataflow::StructuralOrdinal leg = 0; leg != legCount; ++leg) {
      const auto direction = take(
          dataflow::semantics::getCanonicalServiceLegDirection(
              capability.kind(), leg));
      const bool endpointIsInitiator =
          capability.role() ==
          loom::fabric::CanonicalServiceEndpointRole::Initiate;
      const bool legSourceIsInitiator =
          direction ==
          dataflow::semantics::ServiceLegDirection::InitiatorToServer;
      const auto &carriers = endpointIsInitiator == legSourceIsInitiator
                                 ? responseCarriers
                                 : requestCarriers;
      if (llvm::Error error = system.attachServiceLegCarriers(
              memoryEndpointRef, capability.kind(), leg, carriers))
        fail(llvm::toString(std::move(error)));
      const auto &occurrenceCarriers =
          endpointIsInitiator == legSourceIsInitiator
              ? occurrenceResponseCarriers
              : occurrenceRequestCarriers;
      if (llvm::Error error = system.attachServiceLegCarriers(
              spatialMemory, capability.kind(), leg, occurrenceCarriers))
        fail(llvm::toString(std::move(error)));
    }
  }
  domainMembers.push_back(memoryService.domainMember());
  domainMembers.push_back(memoryEndpoint.domainMember());

  auto initiateCapability =
      take(loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MessageTransfer,
          loom::fabric::CanonicalServiceEndpointRole::Initiate,
          take(loom::fabric::MessageTransferCapabilityDomain::create(
              {mlir::NoneType::get(&context),
               mlir::IntegerType::get(&context, 32)})),
          rate));
  auto serveCapability =
      take(loom::fabric::CanonicalServiceCapabilityRecord::create(
          dataflow::semantics::ServiceKind::MessageTransfer,
          loom::fabric::CanonicalServiceEndpointRole::Serve,
          take(loom::fabric::MessageTransferCapabilityDomain::create(
              {mlir::NoneType::get(&context),
               mlir::IntegerType::get(&context, 32)})),
          rate));
  auto initiateSet = take(loom::fabric::CanonicalServiceCapabilitySet::create(
      {std::move(initiateCapability)}));
  auto serveSet = take(loom::fabric::CanonicalServiceCapabilitySet::create(
      {std::move(serveCapability)}));
  const auto bits32 = take(loom::adg::PortType::bits(32));
  auto messageSource =
      take(system.addServiceEndpoint(extraCore, initiateSet, bits32));
  auto messageSink =
      take(system.addServiceEndpoint(extraCore, serveSet, bits32));
  auto messageTransport = take(
      system.addTransportResource({{bits32}, {bits32}, transportContract}));
  auto messagePattern =
      take(system.addTransferPattern(messageTransport, 0, {0}, 0));
  if (llvm::Error error = system.connect(take(messageSource.transport()),
                                         take(messageTransport.input(0))))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = system.connect(take(messageTransport.output(0)),
                                         take(messageSink.transport())))
    fail(llvm::toString(std::move(error)));
  domainMembers.push_back(messageSource.domainMember());
  domainMembers.push_back(messageSink.domainMember());
  domainMembers.push_back(messageTransport.domainMember());
  domainMembers.push_back(messagePattern.domainMember());
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
  auto baselineDesign = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  require(baselineDesign.roots().size() == 1 &&
              baselineDesign.roots().front().directDependencies().size() == 1,
          "builtin System fixture did not publish one Module dependency");
  auto primaryModule = take(loom::fabric::importEntireFabricRoot(
      baselineDesign.roots().front().directDependencies().front().root, store));
  auto alternateDesign = buildSpatialModule(store, true);
  auto design = buildHeterogeneousSystem(
      store, baselineDesign.roots().front(), primaryModule,
      alternateDesign.roots().front(), context);
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
  require(!searchDomain.serviceObligations().empty(),
          "System route fixture has no service obligation");
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
  require(!problem->serviceLegs().empty(),
          "frozen System problem lost its service legs");

  auto first = take(loom::pnr::initializeCanonicalSystemCandidate(problem));
  auto second = take(loom::pnr::initializeCanonicalSystemCandidate(problem));
  require(first.state->threadChoices() == second.state->threadChoices() &&
              first.state->graphChoices() == second.state->graphChoices() &&
              first.assignmentAttempts == second.assignmentAttempts,
          "canonical System initializer is not deterministic");
  require(first.state->serviceRoutes().size() == problem->serviceLegs().size(),
          "canonical System initializer did not route every service leg");
  for (const loom::pnr::SystemServiceRouteSelection &route :
       first.state->serviceRoutes()) {
    require(route.nodeCount != 0 && route.sinkCount != 0,
            "canonical System route is empty");
    require(route.rootEndpoint != loom::pnr::getInvalidPnrIndex(),
            "canonical System route has no root endpoint");
  }
  if (llvm::Error error = first.state->verify())
    fail(llvm::toString(std::move(error)));

  std::vector<loom::pnr::SystemServiceRouteSelection> incompleteRoutes(
      first.state->serviceRoutes().begin(), first.state->serviceRoutes().end());
  incompleteRoutes.front().sinkCount = 0;
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem, {first.state->threadChoices(), first.state->graphChoices(),
                    incompleteRoutes, first.state->serviceRouteNodes(),
                    first.state->serviceRouteSinks()}),
      "service route does not cover every sink terminal");

  std::vector<loom::pnr::SystemServiceRouteSinkSelection> foreignSinks(
      first.state->serviceRouteSinks().begin(),
      first.state->serviceRouteSinks().end());
  foreignSinks.front().terminal =
      problem->serviceLegs()[first.state->serviceRoutes().front().leg]
          .sourceTerminal;
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem, {first.state->threadChoices(), first.state->graphChoices(),
                    first.state->serviceRoutes(),
                    first.state->serviceRouteNodes(), foreignSinks}),
      "service route sink is outside its exact H domain");

  auto withCanonicalRoutes =
      [&](llvm::ArrayRef<loom::pnr::PnrIndex> threadChoices,
          llvm::ArrayRef<loom::pnr::PnrIndex> graphChoices) {
        return loom::pnr::SystemCandidateInitialization{
            threadChoices, graphChoices, first.state->serviceRoutes(),
            first.state->serviceRouteNodes(), first.state->serviceRouteSinks()};
      };

  auto firstDraft =
      take(loom::pnr::materializeSystemCandidateDraft(*first.state, context));
  auto secondDraft =
      take(loom::pnr::materializeSystemCandidateDraft(*first.state, context));
  auto firstRoot = mlir::cast<::mapping::SystemOp>(firstDraft.get());
  std::size_t materializedRouteCount = 0;
  for (auto service :
       firstRoot.getBody().front().getOps<::mapping::ServiceRealizationOp>()) {
    require(llvm::hasSingleElement(
                service.getBody().front().getOps<::mapping::ServicePlanOp>()),
            "materialized service has more than one selected plan");
    auto plan =
        *service.getBody().front().getOps<::mapping::ServicePlanOp>().begin();
    require(plan.getPlanOrdinal() == 0,
            "materialized selected service plan has a nonzero ordinal");
    for (auto route :
         plan.getBody().front().getOps<::mapping::TransferLegRealizationOp>()) {
      const auto leg = take(loom::mapping::decodeCanonicalServiceLegKey(
          unsignedBytes(route.getLeg().getRecord()),
          problem->dataflowIdentity()));
      loom::pnr::PnrIndex selectedOrdinal = loom::pnr::getInvalidPnrIndex();
      for (const auto &[ordinal, selected] :
           llvm::enumerate(first.state->serviceRoutes()))
        if (problem->serviceLegs()[selected.leg].key == leg) {
          selectedOrdinal = static_cast<loom::pnr::PnrIndex>(ordinal);
          break;
        }
      require(selectedOrdinal != loom::pnr::getInvalidPnrIndex(),
              "materialized route has no selected Candidate route");
      const auto &selected = first.state->serviceRoutes()[selectedOrdinal];
      const auto expectedRoot = loom::fabric::canonicalFabricBytes(
          problem->routingTopology()
              .endpoints()[selected.rootEndpoint]
              .reference);
      require(unsignedBytes(route.getRootEndpoint().getRecord()) ==
                  std::vector<std::uint8_t>(expectedRoot.begin(),
                                            expectedRoot.end()),
              "materialized route changed its selected root endpoint");

      auto selectedNodes = first.state->serviceRouteNodes().slice(
          selected.nodeOffset, selected.nodeCount);
      auto materializedNodes =
          route.getBody().front().getOps<::mapping::SystemRouteNodeOp>();
      require(
          std::distance(materializedNodes.begin(), materializedNodes.end()) +
                  1 ==
              selectedNodes.size(),
          "materialized route changed its node count");
      for (const auto &[nodeOrdinal, node] :
           llvm::enumerate(materializedNodes)) {
        const auto &expected = selectedNodes[nodeOrdinal + 1];
        const auto expectedTraversal = loom::fabric::canonicalFabricBytes(
            problem->routingTopology()
                .traversals()[expected.incomingTraversal]
                .reference);
        require(node.getNodeOrdinal() == nodeOrdinal + 1 &&
                    node.getParentNodeOrdinal() == expected.parentNode &&
                    unsignedBytes(node.getIncomingTraversal().getRecord()) ==
                        std::vector<std::uint8_t>(expectedTraversal.begin(),
                                                  expectedTraversal.end()),
                "materialized route changed a selected traversal");
      }

      auto selectedSinks = first.state->serviceRouteSinks().slice(
          selected.sinkOffset, selected.sinkCount);
      auto materializedSinks =
          route.getBody().front().getOps<::mapping::SystemRouteSinkOp>();
      require(std::distance(materializedSinks.begin(),
                            materializedSinks.end()) == selectedSinks.size(),
              "materialized route changed its sink count");
      for (const auto &[sinkOrdinal, sink] :
           llvm::enumerate(materializedSinks)) {
        const auto &expected = selectedSinks[sinkOrdinal];
        const auto expectedTerminal =
            take(loom::mapping::encodeSystemTransferTerminalKey(
                problem->dataflowIdentity(),
                problem->serviceTerminals()[expected.terminal].key));
        require(unsignedBytes(sink.getTerminal().getRecord()) ==
                        expectedTerminal &&
                    sink.getNodeOrdinal() == expected.node,
                "materialized route changed a selected sink attachment");
      }
      ++materializedRouteCount;
    }
  }
  require(materializedRouteCount == first.state->serviceRoutes().size(),
          "materializer omitted a selected service route");
  auto firstBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(firstRoot));
  auto secondBytes = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(secondDraft.get())));
  require(firstBytes.bytes() == secondBytes.bytes(),
          "System execution materialization is not deterministic");

  mlir::OwningOpRef<mlir::Operation *> reordered(firstDraft->clone());
  auto reorderedRoot = mlir::cast<::mapping::SystemOp>(reordered.get());
  llvm::SmallVector<mlir::Attribute> reversedRoots(
      reorderedRoot.getRootThreadLaunches().begin(),
      reorderedRoot.getRootThreadLaunches().end());
  std::reverse(reversedRoots.begin(), reversedRoots.end());
  reorderedRoot.setRootThreadLaunchesAttr(
      mlir::ArrayAttr::get(&context, reversedRoots));
  auto reorderedBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(reorderedRoot));
  require(reorderedBytes.bytes() == firstBytes.bytes(),
          "System root authoring order changed canonical bytes");
  auto rawReorderedBytes = rawSystemBytes(reorderedRoot);
  require(rawReorderedBytes.bytes() != firstBytes.bytes(),
          "noncanonical System fixture accidentally matched canonical bytes");
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             rawReorderedBytes, dataflow, system, store),
                         "payload is not canonical");

  mlir::OwningOpRef<mlir::Operation *> missingThread(firstDraft->clone());
  auto missingRoot = mlir::cast<::mapping::SystemOp>(missingThread.get());
  auto missingBinding = *missingRoot.getBody()
                             .front()
                             .getOps<::mapping::ThreadExecutionBindingOp>()
                             .begin();
  missingBinding.erase();
  requireVerificationFailureContains(missingRoot,
                                     "exactly one ThreadExecutionBinding");

  mlir::OwningOpRef<mlir::Operation *> defaultOnly(firstDraft->clone());
  auto defaultRoot = mlir::cast<::mapping::SystemOp>(defaultOnly.get());
  auto defaultBinding = *defaultRoot.getBody()
                             .front()
                             .getOps<::mapping::ThreadExecutionBindingOp>()
                             .begin();
  auto defaultClause = *defaultBinding.getBody()
                            .front()
                            .getOps<::mapping::ThreadPresburgerClauseOp>()
                            .begin();
  defaultBinding->setAttr("default_target", defaultClause.getTarget());
  defaultClause.erase();
  auto defaultBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(defaultRoot));
  auto defaultExecution =
      take(loom::mapping::strictImportSystemExecutionBindings(
          defaultBytes, dataflow, system, store));
  require(defaultExecution.threadBindings().front().clauses.empty() &&
              defaultExecution.threadBindings().front().defaultTarget,
          "default-only whole-domain relation did not round trip");

  mlir::OwningOpRef<mlir::Operation *> graphDefaultOnly(firstDraft->clone());
  auto graphDefaultRoot =
      mlir::cast<::mapping::SystemOp>(graphDefaultOnly.get());
  auto graphDefaultBinding = *graphDefaultRoot.getBody()
                                  .front()
                                  .getOps<::mapping::GraphExecutionBindingOp>()
                                  .begin();
  auto graphDefaultClause = *graphDefaultBinding.getBody()
                                 .front()
                                 .getOps<::mapping::GraphPresburgerClauseOp>()
                                 .begin();
  graphDefaultBinding->setAttr("default_target",
                               graphDefaultClause.getTarget());
  graphDefaultClause.erase();
  auto graphDefaultBytes = take(
      loom::mapping::writeCanonicalSystemMappingAssembly(graphDefaultRoot));
  auto graphDefaultExecution =
      take(loom::mapping::strictImportSystemExecutionBindings(
          graphDefaultBytes, dataflow, system, store));
  require(graphDefaultExecution.graphBindings().front().clauses.empty() &&
              graphDefaultExecution.graphBindings().front().defaultTarget,
          "default-only graph relation did not round trip");

  mlir::OwningOpRef<mlir::Operation *> emptyThread(firstDraft->clone());
  auto emptyThreadBinding = *mlir::cast<::mapping::SystemOp>(emptyThread.get())
                                 .getBody()
                                 .front()
                                 .getOps<::mapping::ThreadExecutionBindingOp>()
                                 .begin();
  emptyThreadBinding.getBody().front().front().erase();
  requireVerificationFailureContains(emptyThreadBinding,
                                     "requires a clause or default target");

  mlir::OwningOpRef<mlir::Operation *> emptyGraph(firstDraft->clone());
  auto emptyGraphBinding = *mlir::cast<::mapping::SystemOp>(emptyGraph.get())
                                .getBody()
                                .front()
                                .getOps<::mapping::GraphExecutionBindingOp>()
                                .begin();
  emptyGraphBinding.getBody().front().front().erase();
  requireVerificationFailureContains(emptyGraphBinding,
                                     "requires a clause or default target");

  mlir::OwningOpRef<mlir::Operation *> domainGap(firstDraft->clone());
  auto gapBinding = *mlir::cast<::mapping::SystemOp>(domainGap.get())
                         .getBody()
                         .front()
                         .getOps<::mapping::ThreadExecutionBindingOp>()
                         .begin();
  auto gapClause = *gapBinding.getBody()
                        .front()
                        .getOps<::mapping::ThreadPresburgerClauseOp>()
                        .begin();
  auto wholeCell =
      mlir::cast<::mapping::SystemPresburgerCellAttr>(gapClause.getCells()[0]);
  auto partialCell = withFirstCoordinateLowerBound(wholeCell, 1);
  gapClause->setAttr("cells", mlir::ArrayAttr::get(&context, {partialCell}));
  auto gapBytes = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(domainGap.get())));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             gapBytes, dataflow, system, store),
                         "does not cover its Dataflow may-domain");

  mlir::OwningOpRef<mlir::Operation *> domainOverlap(firstDraft->clone());
  auto overlapBinding = *mlir::cast<::mapping::SystemOp>(domainOverlap.get())
                             .getBody()
                             .front()
                             .getOps<::mapping::ThreadExecutionBindingOp>()
                             .begin();
  auto overlapClause = *overlapBinding.getBody()
                            .front()
                            .getOps<::mapping::ThreadPresburgerClauseOp>()
                            .begin();
  llvm::SmallVector<mlir::Attribute> overlappingCells = {wholeCell,
                                                         partialCell};
  overlapClause->setAttr("cells",
                         mlir::ArrayAttr::get(&context, overlappingCells));
  auto overlapBytes = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(domainOverlap.get())));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             overlapBytes, dataflow, system, store),
                         "overlapping Presburger cells");

  mlir::OwningOpRef<mlir::Operation *> redundantDefault(firstDraft->clone());
  auto redundantBinding =
      *mlir::cast<::mapping::SystemOp>(redundantDefault.get())
           .getBody()
           .front()
           .getOps<::mapping::ThreadExecutionBindingOp>()
           .begin();
  auto redundantClause = *redundantBinding.getBody()
                              .front()
                              .getOps<::mapping::ThreadPresburgerClauseOp>()
                              .begin();
  redundantBinding->setAttr("default_target", redundantClause.getTarget());
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             rawSystemBytes(mlir::cast<::mapping::SystemOp>(
                                 redundantDefault.get())),
                             dataflow, system, store),
                         "default is forbidden for an empty complement");

  const auto selectedMapping = first.state->selectedSpatialMapping(0);
  const auto unselectedMapping = spatialMappings.front() == selectedMapping
                                     ? spatialMappings.back()
                                     : spatialMappings.front();
  mlir::OwningOpRef<mlir::Operation *> extraImport(firstDraft->clone());
  auto extraImportRoot = mlir::cast<::mapping::SystemOp>(extraImport.get());
  llvm::SmallVector<mlir::Attribute> imports(
      extraImportRoot.getSpatialMappingImports().begin(),
      extraImportRoot.getSpatialMappingImports().end());
  imports.push_back(rootReferenceAttr(&context, unselectedMapping));
  extraImportRoot.setSpatialMappingImportsAttr(
      mlir::ArrayAttr::get(&context, imports));
  auto extraImportBytes =
      take(loom::mapping::writeCanonicalSystemMappingAssembly(extraImportRoot));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             extraImportBytes, dataflow, system, store),
                         "not the exact selected B_graph range");

  mlir::OwningOpRef<mlir::Operation *> incompatible(firstDraft->clone());
  auto incompatibleRoot = mlir::cast<::mapping::SystemOp>(incompatible.get());
  llvm::SmallVector<mlir::Attribute> incompatibleImports(
      incompatibleRoot.getSpatialMappingImports().begin(),
      incompatibleRoot.getSpatialMappingImports().end());
  incompatibleImports.push_back(rootReferenceAttr(&context, unselectedMapping));
  incompatibleRoot.setSpatialMappingImportsAttr(
      mlir::ArrayAttr::get(&context, incompatibleImports));
  auto incompatibleBinding = *incompatibleRoot.getBody()
                                  .front()
                                  .getOps<::mapping::GraphExecutionBindingOp>()
                                  .begin();
  auto incompatibleClause = *incompatibleBinding.getBody()
                                 .front()
                                 .getOps<::mapping::GraphPresburgerClauseOp>()
                                 .begin();
  incompatibleClause->setAttr(
      "target", ::mapping::SpatialMappingImportRefAttr::get(&context, 1));
  auto incompatibleBytes = take(
      loom::mapping::writeCanonicalSystemMappingAssembly(incompatibleRoot));
  requireFailureContains(loom::mapping::strictImportSystemExecutionBindings(
                             incompatibleBytes, dataflow, system, store),
                         "graph and thread targets are incompatible");

  auto execution = take(loom::mapping::strictImportSystemExecutionBindings(
      firstBytes, dataflow, system, store));
  require(execution.rootThreadLaunches().size() == 2 &&
              execution.threadBindings().size() == 2 &&
              execution.graphBindings().size() == 4,
          "strict execution import lost factorized binding keys");
  require(execution.spatialMappingImports().size() == 1,
          "System import table is not the exact selected B_graph range");
  for (const auto &binding : execution.threadBindings())
    require(binding.clauses.size() == 1 && !binding.defaultTarget,
            "whole-domain thread binding was not canonicalized");
  for (const auto &binding : execution.graphBindings())
    require(binding.clauses.size() == 1 && !binding.defaultTarget &&
                binding.clauses.front().target ==
                    execution.spatialMappingImports().front(),
            "whole-domain graph binding did not resolve its exact import");
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
      problem, withCanonicalRoutes(threadChoices, graphChoices)));
  threadChoices[0] = sameClassSecond;
  auto alternate = take(loom::pnr::SystemCandidateState::create(
      problem, withCanonicalRoutes(threadChoices, graphChoices)));
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
        requireFailureContains(
            loom::pnr::SystemCandidateState::create(
                problem, withCanonicalRoutes(threadChoices, graphChoices)),
            "target classes are incompatible");
        foundMismatch = true;
      }
  require(foundMismatch,
          "heterogeneous fixture did not expose an incompatible target pair");

  threadChoices.assign(problem->threadDecisions().size(), 0);
  graphChoices.assign(problem->graphDecisions().size(), 0);
  threadChoices[0] = problem->threadChoiceCatalogOrdinals(0).size();
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem, withCanonicalRoutes(threadChoices, graphChoices)),
      "thread choice is outside its H domain");
  threadChoices.pop_back();
  requireFailureContains(
      loom::pnr::SystemCandidateState::create(
          problem, withCanonicalRoutes(threadChoices, graphChoices)),
      "thread choice count does not match H");

  llvm::outs() << "System CandidateState anchors passed\n";
  return EXIT_SUCCESS;
}

#include "SpatialMemoryMappingArtifactTestSupport.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "ADG/MemoryLibrary.h"
#include "CgraAdmissionTestSupport.h"
#include "SpatialCandidateSelectionTestSupport.h"
#include "TechMappingCandidateTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricMemoryConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialExactRepair.h"
#include "PnR/SpatialMappingMaterializer.h"
#include "PnR/SpatialPathFinderRouter.h"
#include "PnR/SpatialPnrProblem.h"
#include "PnR/SpatialRouteCostState.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::test {
namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial memory mapping artifact test: " << message << '\n';
  std::exit(1);
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

template <typename Attr, typename Ref>
Attr fabricReferenceAttr(mlir::MLIRContext *context, const Ref &reference) {
  const auto bytes = fabric::canonicalFabricBytes(reference);
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return Attr::get(context, mlir::DenseI8ArrayAttr::get(context, signedBytes));
}

template <typename T> bool rejected(llvm::Expected<T> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

bool rejected(llvm::Error error) {
  if (!error)
    return false;
  llvm::consumeError(std::move(error));
  return true;
}

template <typename Callable>
bool rejectedWithoutDiagnostic(mlir::MLIRContext &context,
                               Callable &&callable) {
  mlir::ScopedDiagnosticHandler capture(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  return rejected(std::forward<Callable>(callable)());
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-spatial-memory-mapping-artifact", path))
      fail("cannot create ArtifactStore directory: " + error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "cannot remove test directory: " << error.message()
                   << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string path_;
};

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::mapping::MappingDialect,
                  ::fabric::FabricDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

mlir::OwningOpRef<mlir::ModuleOp>
parseSpatial(mlir::MLIRContext &context, const CanonicalSemanticBytes &bytes) {
  std::string text = "module {\n";
  text.append(reinterpret_cast<const char *>(bytes.bytes().data()),
              bytes.bytes().size());
  text += "}\n";
  return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
}

// The memory-specific Dataflow, Fabric, transaction, and artifact fixture is
// kept together below because each layer is needed to validate the same rooted
// memory-service ownership contract.

dataflow::CanonicalDataflowArtifact
buildMemoryDataflow(mlir::MLIRContext &context, bool splitExposures = false) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @load(
      %start: none, %index: index, %memory: memref<4xi32>,
      %exported: memref<4xi32>) -> (i32, memref<4xi32>, memref<4xi32>)
      attributes {input_segments = array<i32: 1, 0, 2>,
                  result_segments = array<i32: 1, 0, 2>} {
    %value, %done = dataflow.load %memory[%index] %start : memref<4xi32>
    dataflow.graph.return values(%value : i32) streams()
        memories(%exported, %exported : memref<4xi32>, memref<4xi32>)
        complete(%done : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %index: index, %memory: memref<4xi32>,
      %exported: memref<4xi32>) ctrl (%ctrl: none) {
    %value, %exposed0, %exposed1, %done = dataflow.graph.launch @load deps(%ctrl)
        values(%index) stream_inputs() memories(%memory, %exported)
        stream_outputs()
        : (none, index, memref<4xi32>, memref<4xi32>)
          -> (i32, memref<4xi32>, memref<4xi32>, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host(%index: index, %memory: memref<4xi32>,
                          %exported: memref<4xi32>) {
    %token0 = dataflow.thread.launch @worker(%index, %memory, %exported)
        : (index, memref<4xi32>, memref<4xi32>) -> !dataflow.thread_token
    %token1 = dataflow.thread.launch @worker(%index, %memory, %exported)
        : (index, memref<4xi32>, memref<4xi32>) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse memory Dataflow fixture");
  if (splitExposures) {
    auto graph = *module->getOps<::dataflow::GraphOp>().begin();
    auto result =
        *graph.getBody().front().getOps<::dataflow::GraphReturnOp>().begin();
    result.getMemoriesMutable().slice(0, 1).assign(
        graph.getBody().front().getArgument(2));
  }
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

::fabric::UnsignedDomain singleton(std::uint64_t value) {
  return take(::fabric::UnsignedDomain::fromCanonical({{value, value}}));
}

loom::adg::MemorySpec makeStorageProvider(mlir::MLIRContext &context) {
  auto alignment = take(::fabric::AlignmentDomain::create(
      take(::fabric::UnsignedDomain::fromCanonical({{0, 3}}))));
  auto read = take(
      ::fabric::ClosedEnumDomain<::fabric::ReadSubwordSemantics>::fromCanonical(
          {::fabric::ReadSubwordSemantics::Exact}));
  auto write =
      take(::fabric::ClosedEnumDomain<::fabric::WriteSubwordSemantics>::
               fromCanonical({::fabric::WriteSubwordSemantics::NotApplicable}));
  auto address =
      take(::fabric::MemoryAddressDomain::rootRelative(singleton(64)));
  auto access = take(::fabric::MemoryAccessClass::create(
      ::dataflow::semantics::MemoryAccessForm::Element, singleton(32),
      singleton(1),
      {{::dataflow::semantics::MemoryMaskForm::Absent,
        ::fabric::InactiveLaneSemantics::NotApplicable}},
      std::move(alignment), std::move(read), std::move(write),
      std::move(address)));
  auto accesses = take(
      ::fabric::ParameterizedMemoryAccessDomain::create({std::move(access)}));
  ::fabric::MemoryActorContractClause plain =
      ::fabric::LoadStorePlainContractClause{{false}};
  auto actors = take(::fabric::MemoryActorContractDomain::create(
      ::dataflow::OperationSchemaId::DataflowLoad, {plain}));
  auto serviceRecord = take(::fabric::MemoryServiceContractRecord::create(
      &context, ::fabric::MemoryServiceOwnerKind::Local,
      {{{0, 4096, ::fabric::MemoryServiceRegionBehavior::Storage,
         std::nullopt}},
       ::fabric::oneCycleElasticOperationResourceContract(),
       {{std::move(actors),
         std::move(accesses),
         {0},
         32,
         {::fabric::UsePatternKey(0)},
         ::fabric::NoMemoryServiceConsistency{}}}}));
  auto service =
      take(loom::adg::LocalMemoryServiceSpec::create(4096, serviceRecord));
  ::fabric::MemoryConnectivityDeclaration connectivity;
  connectivity.subordinateEndpoints = {
      {1,
       {},
       ::fabric::MemoryProviderAddressTransform::None,
       {::fabric::MemoryDispatchTarget(
           std::in_place_type<::fabric::LocalMemoryDispatchTarget>)}}};
  auto connectivitySpec =
      take(loom::adg::MemoryConnectivitySpec::create(std::move(connectivity)));
  auto bits32 = take(loom::adg::PortType::bits(32));
  auto memory = take(loom::adg::PortType::memory({4}, bits32));
  return take(loom::adg::MemorySpec::create({}, {memory}, {}, {0}, std::nullopt,
                                            std::move(service),
                                            std::move(connectivitySpec)));
}

loom::fabric::FinalizedFabricRoot buildMemoryFabric(loom::ArtifactStore &store,
                                                    bool temporal) {
  loom::adg::LocalMemoryParameters parameters;
  parameters.capacityBytes = 4096;
  parameters.interface = {
      loom::adg::MemoryAccessDomainParameters{128, 128, 16, singleton(64)}, 128,
      128};
  parameters.managerEndpoint = true;
  if (temporal)
    parameters.temporal = loom::adg::TemporalMemoryParameters{4, 2};
  auto memory = take(loom::adg::makeGeneral64LocalMemory(parameters));
  const std::vector<loom::adg::PortType> inputs(memory.inputTypes().begin(),
                                                memory.inputTypes().end());
  mlir::MLIRContext storageContext(mlir::MLIRContext::Threading::DISABLED);
  auto storage = makeStorageProvider(storageContext);
  std::vector<loom::adg::PortType> outputs(memory.outputTypes().begin(),
                                           memory.outputTypes().end());
  outputs.insert(outputs.end(), storage.outputTypes().begin(),
                 storage.outputTypes().end());
  loom::adg::DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("memory", inputs, outputs));
  std::vector<loom::adg::SpatialValue> values;
  values.reserve(inputs.size());
  for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal)
    values.push_back(take(spatial.input(ordinal)));
  auto memoryOutputs = take(spatial.addMemory(values, memory));
  auto storageOutputs = take(spatial.addMemory({}, storage));
  std::vector<loom::adg::SpatialValue> combinedOutputs;
  combinedOutputs.reserve(memoryOutputs.values().size() +
                          storageOutputs.values().size());
  combinedOutputs.insert(combinedOutputs.end(), memoryOutputs.values().begin(),
                         memoryOutputs.values().end());
  combinedOutputs.insert(combinedOutputs.end(), storageOutputs.values().begin(),
                         storageOutputs.values().end());
  requireSuccess(spatial.close(combinedOutputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("memory SpatialCore did not publish exactly one root");
  return design.roots().front();
}

} // namespace
void completeMemorySpatialMappingRoundTrip(bool temporal, bool splitExposures) {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildMemoryDataflow(context, splitExposures);
  const auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildMemoryFabric(store, temporal);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto techConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  auto generated = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), techConfig, store});
  auto *candidates =
      std::get_if<loom::mapping::GeneratedTechMappings>(&generated);
  if (!candidates || candidates->candidates.size() != 1)
    fail("memory TechMapping fixture did not produce one candidate");
  const auto tech = take(
      loom::mapping::importTechMapping(candidates->candidates.front(), store));
  if (tech.view().memoryRealizations().size() != 1)
    fail("memory TechMapping fixture did not select one realization");

  const auto constraints = loom::test::buildSpatialMappingConstraints(
      context, dataflow, tech.view(), fabric.view(), store);
  const auto pnrConfig = take(loom::pnr::projectResolvedSpatialPnrConfigView(
      loom::test::buildSpatialPnrTestResolvedConfig()));
  auto problem = take(loom::pnr::freezeSpatialPnrProblem(
      dataflow, tech.view(), fabric.view(), pnrConfig, constraints.view()));
  if (splitExposures) {
    const auto unsupported =
        loom::pnr::unsupportedSpatialExactRepairDomain(*problem);
    if (!unsupported ||
        !llvm::StringRef(*unsupported).contains("exposure-provider"))
      fail("CpSat exact-repair preflight admitted unsupported exposure "
           "capacity");
    return;
  }
  auto candidate = take(loom::pnr::createCanonicalSpatialCandidate(problem));
  loom::test::exerciseSpatialMemoryActionDomain(problem, *candidate);
  loom::pnr::SpatialCandidateScratch candidateScratch;
  requireSuccess(candidateScratch.prepare(*problem));

  const auto &memoryIndex = problem->memory();
  if (memoryIndex.logicalBindings().size() != 2 ||
      memoryIndex.rootedUses().size() != 2 ||
      memoryIndex.exposures().size() != 4 ||
      memoryIndex.exposureProviders().size() != 1 ||
      memoryIndex.exposureOptions().size() != 1)
    fail("memory transaction fixture lost its bindings or rooted use");
  if (memoryIndex.serviceUseGroups().size() != 1 ||
      memoryIndex.serviceUseGroups().front().useCount != 2)
    fail("same-binding rooted uses were not factorized into one service use");
  const auto &memoryActor =
      problem->realizations()
          .memoryActors()[memoryIndex.rootedUses().front().actor]
          .actor;
  const auto issueEvent =
      take(loom::mapping::deriveSpatialMemoryIssueEvent(dataflow, memoryActor));
  if (issueEvent.actor != memoryActor || issueEvent.transition != 0)
    fail("Mapping owner derived the wrong memory issue event");
  const auto &capacity = problem->capacity();
  const auto planEnvelopes = capacity.memoryOperationPlanEnvelopes();
  if (planEnvelopes.size() !=
      problem->handshake().memoryOperationPlans().size())
    fail("memory operation plans lost their resource-time envelopes");
  const loom::pnr::PnrIndex selectedActor =
      memoryIndex.rootedUses().front().actor;
  const loom::pnr::PnrIndex selectedPlan =
      candidate->memoryOperationPlan(selectedActor);
  if (selectedPlan >= planEnvelopes.size() ||
      planEnvelopes[selectedPlan] >= capacity.resourceTimeEnvelopes().size())
    fail("selected memory operation plan has no resource-time envelope");
  const auto &planEnvelope =
      capacity.resourceTimeEnvelopes()[planEnvelopes[selectedPlan]];
  if (planEnvelope.event >= capacity.resourceEvents().size())
    fail("memory operation envelope has no resource event");
  const auto &planEvent = capacity.resourceEvents()[planEnvelope.event];
  const auto *planIssue =
      std::get_if<loom::mapping::SpatialActorTransitionEventRef>(
          &planEvent.reference);
  if (planEvent.ownerKind !=
          loom::pnr::FrozenSpatialResourceEventOwnerKind::MemoryRealization ||
      planEvent.owner != 0 || !planIssue || !(*planIssue == issueEvent) ||
      planEnvelope.useCount != 1 || planEnvelope.segmentCount == 0)
    fail("memory operation resource-time projection is incomplete");
  const auto originalLogicalBinding = candidate->logicalMemoryBinding(0);
  const auto &serviceGroup = memoryIndex.serviceUseGroups().front();
  const auto serviceUses = memoryIndex.serviceGroupUses().slice(
      serviceGroup.useOffset, serviceGroup.useCount);
  const auto originalDispatch = candidate->memoryUseDispatch(serviceUses[0]);
  for (loom::pnr::PnrIndex use : serviceUses)
    if (candidate->memoryUseDispatch(use) != originalDispatch)
      fail("same-binding rooted uses selected different service dispatches");
  const auto groupEnvelopeOffsets =
      capacity.memoryServiceGroupEnvelopeOffsets();
  if (groupEnvelopeOffsets.size() != memoryIndex.serviceUseGroups().size() + 1)
    fail("memory service groups lost their envelope offsets");
  const auto groupEnvelopes = capacity.memoryServicePatternEnvelopes().slice(
      groupEnvelopeOffsets.front(),
      groupEnvelopeOffsets.back() - groupEnvelopeOffsets.front());
  if (groupEnvelopes.size() != 1 ||
      groupEnvelopes.front().pattern !=
          capacity.memoryDispatchOptionPatterns()[originalDispatch] ||
      groupEnvelopes.front().envelope >=
          capacity.resourceTimeEnvelopes().size())
    fail("memory service group lost its distinct UsePattern envelope");
  const auto &serviceEnvelope =
      capacity.resourceTimeEnvelopes()[groupEnvelopes.front().envelope];
  const auto &serviceEvent = capacity.resourceEvents()[serviceEnvelope.event];
  const auto *serviceIssue =
      std::get_if<loom::mapping::SpatialActorTransitionEventRef>(
          &serviceEvent.reference);
  if (serviceEvent.ownerKind != loom::pnr::FrozenSpatialResourceEventOwnerKind::
                                    LogicalMemoryBinding ||
      serviceEvent.owner != serviceGroup.logicalBinding || !serviceIssue ||
      !(*serviceIssue == issueEvent) || serviceEnvelope.useCount != 1 ||
      serviceEnvelope.segmentCount == 0)
    fail("memory service resource-time projection is incomplete");
  const loom::pnr::PnrIndex planEnvelopeOrdinal = planEnvelopes[selectedPlan];
  const loom::pnr::PnrIndex serviceEnvelopeOrdinal =
      groupEnvelopes.front().envelope;
  if (candidate->resourceTimeEnvelopeRefcount(planEnvelopeOrdinal) != 1 ||
      candidate->resourceTimeEnvelopeRefcount(serviceEnvelopeOrdinal) != 1 ||
      !candidate->resourceTimeEnvelopeActive(planEnvelopeOrdinal) ||
      !candidate->resourceTimeEnvelopeActive(serviceEnvelopeOrdinal))
    fail("initial candidate lost a selected resource-time envelope");
  const loom::pnr::PnrIndex initialActiveEnvelopeCount =
      candidate->activeResourceTimeEnvelopeCount();
  if (initialActiveEnvelopeCount < 2)
    fail("initial candidate has too few active resource-time envelopes");
  std::optional<loom::pnr::PnrIndex> boundaryTarget;
  for (auto [ordinal, target] : llvm::enumerate(memoryIndex.bindingTargets()))
    if (std::holds_alternative<loom::pnr::FrozenSpatialMemoryBoundaryProxy>(
            target.target))
      boundaryTarget = static_cast<loom::pnr::PnrIndex>(ordinal);
  if (!boundaryTarget)
    fail("memory transaction fixture has no BoundaryProxy target");

  const auto &rootedUse = memoryIndex.rootedUses().front();
  const auto selectedDispatchPlacement = candidate->memoryBinding(0).placement;
  const auto dispatchDomain =
      llvm::find_if(memoryIndex.dispatchDomains(), [&](const auto &domain) {
        return domain.placement == selectedDispatchPlacement &&
               domain.actor == rootedUse.actor;
      });
  if (dispatchDomain == memoryIndex.dispatchDomains().end())
    fail("memory transaction fixture has no selected dispatch domain");
  std::optional<loom::pnr::PnrIndex> managerDispatch;
  for (loom::pnr::PnrIndex option = dispatchDomain->optionOffset;
       option != dispatchDomain->optionOffset + dispatchDomain->optionCount;
       ++option)
    if (std::holds_alternative<loom::fabric::ManagerEndpointRef>(
            memoryIndex.dispatchOptions()[option].target))
      managerDispatch = option;
  if (!managerDispatch)
    fail("memory transaction fixture has no manager dispatch");

  {
    auto move = take(candidate->beginMove(candidateScratch));
    requireSuccess(move.setLogicalMemoryBinding(0, *boundaryTarget, 0));
    auto closed = move.close();
    if (closed)
      fail("unpaired BoundaryProxy binding passed transaction validation");
    llvm::consumeError(closed.takeError());
    move.rollback();
  }
  if (candidate->logicalMemoryBinding(0).target !=
          originalLogicalBinding.target ||
      llvm::any_of(serviceUses,
                   [&](loom::pnr::PnrIndex use) {
                     return candidate->memoryUseDispatch(use) !=
                            originalDispatch;
                   }) ||
      candidate->resourceTimeEnvelopeRefcount(serviceEnvelopeOrdinal) != 1 ||
      candidate->activeResourceTimeEnvelopeCount() !=
          initialActiveEnvelopeCount)
    fail("failed memory transaction did not roll back atomically");

  {
    auto move = take(candidate->beginMove(candidateScratch));
    requireSuccess(move.setLogicalMemoryBinding(0, *boundaryTarget, 0));
    for (loom::pnr::PnrIndex use : serviceUses)
      requireSuccess(move.setMemoryUseDispatch(use, *managerDispatch));
    if (!take(move.close()))
      fail("paired BoundaryProxy move closes a selected handshake cycle");
    requireSuccess(move.commit());
  }
  if (candidate->resourceTimeEnvelopeRefcount(planEnvelopeOrdinal) != 1 ||
      candidate->resourceTimeEnvelopeRefcount(serviceEnvelopeOrdinal) != 0 ||
      !candidate->resourceTimeEnvelopeActive(planEnvelopeOrdinal) ||
      candidate->resourceTimeEnvelopeActive(serviceEnvelopeOrdinal) ||
      candidate->activeResourceTimeEnvelopeCount() + 1 !=
          initialActiveEnvelopeCount)
    fail("BoundaryProxy move retained a local-service envelope");
  requireSuccess(candidate->verify());
  {
    auto move = take(candidate->beginMove(candidateScratch));
    requireSuccess(move.setLogicalMemoryBinding(
        0, originalLogicalBinding.target,
        originalLogicalBinding.physicalOffsetBytes));
    for (loom::pnr::PnrIndex use : serviceUses)
      requireSuccess(move.setMemoryUseDispatch(use, originalDispatch));
    if (!take(move.close()))
      fail("restored local memory move closes a selected handshake cycle");
    move.rollback();
  }
  if (candidate->logicalMemoryBinding(0).target != *boundaryTarget ||
      llvm::any_of(serviceUses,
                   [&](loom::pnr::PnrIndex use) {
                     return candidate->memoryUseDispatch(use) !=
                            *managerDispatch;
                   }) ||
      candidate->resourceTimeEnvelopeRefcount(serviceEnvelopeOrdinal) != 0 ||
      candidate->activeResourceTimeEnvelopeCount() + 1 !=
          initialActiveEnvelopeCount)
    fail("memory transaction rollback did not preserve committed state");
  {
    auto move = take(candidate->beginMove(candidateScratch));
    requireSuccess(move.setLogicalMemoryBinding(
        0, originalLogicalBinding.target,
        originalLogicalBinding.physicalOffsetBytes));
    for (loom::pnr::PnrIndex use : serviceUses)
      requireSuccess(move.setMemoryUseDispatch(use, originalDispatch));
    if (!take(move.close()))
      fail("restored local memory move closes a selected handshake cycle");
    requireSuccess(move.commit());
  }
  if (candidate->resourceTimeEnvelopeRefcount(serviceEnvelopeOrdinal) != 1 ||
      !candidate->resourceTimeEnvelopeActive(serviceEnvelopeOrdinal) ||
      candidate->activeResourceTimeEnvelopeCount() !=
          initialActiveEnvelopeCount)
    fail("restored local memory move lost its resource-time envelope");
  {
    auto move = take(candidate->beginMove(candidateScratch));
    requireSuccess(
        loom::test::selectReachableGraphBoundaries(*candidate, move));
    if (!take(move.close()))
      fail("reachable memory boundaries close a selected handshake cycle");
    requireSuccess(move.commit());
  }
  auto costs = take(loom::pnr::SpatialRouteCostState::create(*candidate));
  loom::pnr::SpatialPathFinderRouterScratch router;
  requireSuccess(router.prepare(*problem));
  bool closed = false;
  const auto &memoryRealization =
      problem->realizations().memoryRealizations().front();
  const auto selectedPlacement = candidate->memoryBinding(0).placement;
  const auto domainOffset =
      problem->handshake().memoryPlacementDomainOffsets()[selectedPlacement];
  const auto &domain =
      problem->handshake().memoryOperationDomains()[domainOffset];
  for (loom::pnr::PnrIndex plan = domain.planOffset;
       plan != domain.planOffset + domain.planCount; ++plan) {
    const auto &planRecord = problem->handshake().memoryOperationPlans()[plan];
    if (planRecord.temporalResident != temporal)
      continue;
    auto move = take(candidate->beginMove(candidateScratch));
    requireSuccess(
        move.setMemoryOperationPlan(memoryRealization.actorOffset, plan));
    if (!take(move.close())) {
      move.rollback();
      continue;
    }
    requireSuccess(move.commit());
    auto routed = router.routeToClosure(
        *candidate, candidateScratch, costs,
        {pnrConfig.policy().search.routing.endpointExpansionLimit,
         pnrConfig.policy().search.routing.negotiationIterationLimit,
         pnrConfig.policy().search.routing.noProgressIterationLimit,
         pnrConfig.policy().search.routing.noProgressTrendWindow},
        {});
    if (routed) {
      closed = true;
      break;
    }
    llvm::consumeError(routed.takeError());
  }
  if (!closed)
    fail("memory SpatialMapping fixture has no closed operation plan");
  requireSuccess(candidate->verify());
  auto finalized = take(loom::pnr::finalizeSpatialMappingCandidate(
      *candidate, dataflow, tech.view(), fabric.view(), constraints.view(),
      store));
  auto imported =
      take(loom::mapping::importSpatialMapping(finalized.reference(), store));
  if (imported.view().memoryEngineBindings().size() != 1 ||
      imported.view().memoryBindings().size() != 2)
    fail("strict SpatialMapping round trip lost memory bindings");
  std::size_t exposureCount = 0;
  for (const auto &binding : imported.view().memoryBindings())
    exposureCount += binding.exposures.size();
  if (exposureCount != 4)
    fail("strict SpatialMapping round trip lost the memory exposure");
  const auto &engine = imported.view().memoryEngineBindings().front();
  if (engine.operations.size() != 1 ||
      !std::holds_alternative<
          loom::mapping::SpatialAddressedMemoryOperationView>(
          engine.operations.front()) ||
      std::get<loom::mapping::SpatialAddressedMemoryOperationView>(
          engine.operations.front())
              .uses.size() != 2)
    fail("strict SpatialMapping round trip lost the rooted memory use");
  const auto &serviceHandshake =
      imported.view().memoryServiceHandshakeSelection();
  if (serviceHandshake.operations.size() != engine.operations.size() ||
      serviceHandshake.providers.empty())
    fail("strict SpatialMapping round trip lost memory-service handshake "
         "selection");
  auto handshakeContext =
      take(loom::fabric::buildFabricHandshakeContext(fabric.view()));
  requireSuccess(loom::fabric::verifySelectedMemoryServiceHandshakeAcyclic(
      fabric.view(), imported.view().handshakeSelection(), serviceHandshake,
      handshakeContext));
  auto incompleteServiceHandshake = serviceHandshake;
  incompleteServiceHandshake.operations.pop_back();
  if (!rejected(loom::fabric::verifySelectedMemoryServiceHandshakeAcyclic(
          fabric.view(), imported.view().handshakeSelection(),
          incompleteServiceHandshake, handshakeContext)))
    fail("memory-service handshake verification accepted an incomplete exact "
         "selection");
  const auto memorySchema =
      take(fabric.view().memoryConfigurationSchema(engine.occurrence));
  const loom::mapping::ConfiguredHardwareFieldValueView *memoryField = nullptr;
  for (const auto &field : imported.view().configuredHardware().fields())
    if (field.slot.field == memorySchema.field()) {
      if (memoryField)
        fail("configured hardware duplicated its memory field");
      memoryField = &field;
    }
  if (!memoryField)
    fail("configured hardware omitted its memory field");
  const auto memoryConfiguration =
      take(memorySchema.decode(memoryField->value.bytes()));
  const auto *active =
      std::get_if<loom::fabric::FabricMemoryActive>(&memoryConfiguration);
  if (!active)
    fail("mapped memory configuration became Disabled");
  const std::size_t activeRows = llvm::count_if(
      active->operationRows, [](const auto &row) { return row.has_value(); });
  if (activeRows != 1)
    fail("mapped memory configuration has the wrong active row count");
  const auto &operation =
      std::get<loom::mapping::SpatialAddressedMemoryOperationView>(
          engine.operations.front());
  std::size_t operationPortUseCount = 0;
  std::size_t localServiceUseCount = 0;
  for (const auto &use : imported.view().resourceUses()) {
    if (!use.sharingAssignments.empty())
      continue;
    if (!use.activation.release.empty())
      fail("memory ResourceUse gained a causal release");
    operationPortUseCount += std::holds_alternative<
        loom::mapping::SpatialMemoryEngineResourceOwnerRef>(use.owner);
    localServiceUseCount += std::holds_alternative<
        loom::mapping::SpatialMemoryBindingResourceOwnerRef>(use.owner);
  }
  if (operationPortUseCount != 1 || localServiceUseCount != 1)
    fail("strict SpatialMapping round trip lost a memory ResourceUse");
  if (temporal) {
    // Resident contexts of one operation port are interchangeable, so the
    // ordinal is derived from the canonical order of the actors that resolve
    // to the port. This fixture places one actor there, which takes the first.
    const auto *residentContext =
        std::get_if<loom::fabric::FabricMemoryOperationContextRef>(
            &operation.placement);
    if (!residentContext || residentContext->ordinal != 0)
      fail("Temporal memory placement lost its derived resident context");
    if (residentContext->ordinal >= active->operationRows.size() ||
        !active->operationRows[residentContext->ordinal])
      fail("Temporal memory configuration selected the wrong resident row");

    auto noncanonicalRow = parseSpatial(context, finalized.canonicalBytes());
    if (!noncanonicalRow)
      fail("cannot reparse Temporal memory row fixture");
    auto noncanonicalRoot =
        *noncanonicalRow->getOps<::mapping::SpatialOp>().begin();
    std::optional<::mapping::AddressedMemoryOperationOp> operationToMutate;
    noncanonicalRoot.walk([&](::mapping::AddressedMemoryOperationOp entry) {
      operationToMutate = entry;
    });
    if (!operationToMutate)
      fail("Temporal memory fixture has no addressed operation");
    const loom::fabric::FabricMemoryOperationContextRef laterContext{
        residentContext->port, residentContext->ordinal + 1};
    (*operationToMutate)
        ->setAttr(
            "placement",
            fabricReferenceAttr<::mapping::FabricMemoryOperationContextRefAttr>(
                &context, laterContext));
    if (!rejected(loom::mapping::verifySpatialMappingBase(
            noncanonicalRoot, dataflow, tech.view(), fabric.view())))
      fail("SpatialMapping accepted a noncanonical Temporal memory row");
  } else if (!std::holds_alternative<
                 loom::fabric::FabricMemoryOperationPortRef>(
                 operation.placement)) {
    fail("Spatial memory placement gained a resident context");
  } else {
    const auto port = std::get<loom::fabric::FabricMemoryOperationPortRef>(
        operation.placement);
    if (port.ordinal >= active->operationRows.size() ||
        !active->operationRows[port.ordinal])
      fail("Spatial memory configuration selected the wrong operation row");
  }
  loom::test::exerciseCgraMemoryAdmission(dataflowReference, fabric.reference(),
                                          finalized.reference(), store);
  auto missingUse = parseSpatial(context, finalized.canonicalBytes());
  if (!missingUse)
    fail("cannot reparse memory ResourceUse fixture");
  auto missingUseRoot = *missingUse->getOps<::mapping::SpatialOp>().begin();
  auto resourceUses =
      missingUseRoot.getBody().front().getOps<::mapping::ResourceUseOp>();
  if (resourceUses.empty())
    fail("memory SpatialMapping fixture has no ResourceUse to remove");
  (*resourceUses.begin()).erase();
  if (!rejected(loom::mapping::verifySpatialMappingBase(
          missingUseRoot, dataflow, tech.view(), fabric.view())))
    fail("SpatialMapping finalized without a required memory ResourceUse");

  auto missingExposure = parseSpatial(context, finalized.canonicalBytes());
  if (!missingExposure)
    fail("cannot reparse memory exposure fixture");
  auto missingExposureRoot =
      *missingExposure->getOps<::mapping::SpatialOp>().begin();
  std::optional<::mapping::ExposureEntryOp> exposureToErase;
  missingExposureRoot.walk(
      [&](::mapping::ExposureEntryOp exposure) { exposureToErase = exposure; });
  if (!exposureToErase)
    fail("memory SpatialMapping fixture has no ExposureEntry to remove");
  exposureToErase->erase();
  if (!rejected(loom::mapping::verifySpatialMappingBase(
          missingExposureRoot, dataflow, tech.view(), fabric.view())))
    fail("SpatialMapping finalized without a required memory exposure");

  std::optional<loom::fabric::FabricMemoryEndpointRef> managerEndpoint;
  for (loom::fabric::FabricMemoryOccurrenceRef memory :
       fabric.view().memoryOccurrences()) {
    const auto owner = loom::fabric::FabricMemoryEndpointOwnerRef::of(memory);
    for (std::uint64_t ordinal = 0;
         ordinal < fabric.view().memoryEndpointCount(owner); ++ordinal) {
      const loom::fabric::FabricMemoryEndpointRef endpoint{owner, ordinal};
      if (fabric.view().memoryEndpointRole(endpoint) ==
          loom::fabric::FabricMemoryEndpointRole::Manager)
        managerEndpoint = endpoint;
    }
  }
  if (!managerEndpoint)
    fail("memory SpatialMapping fixture has no manager endpoint");
  auto wrongTerminal = parseSpatial(context, finalized.canonicalBytes());
  if (!wrongTerminal)
    fail("cannot reparse memory exposure terminal fixture");
  auto wrongTerminalRoot =
      *wrongTerminal->getOps<::mapping::SpatialOp>().begin();
  std::optional<::mapping::ExposureEntryOp> exposureToMutate;
  wrongTerminalRoot.walk([&](::mapping::ExposureEntryOp exposure) {
    exposureToMutate = exposure;
  });
  if (!exposureToMutate)
    fail("memory SpatialMapping fixture has no ExposureEntry to mutate");
  (*exposureToMutate)
      ->setAttr("terminal",
                fabricReferenceAttr<::mapping::SubordinateEndpointRefAttr>(
                    &context,
                    loom::fabric::SubordinateEndpointRef(*managerEndpoint)));
  if (!rejected(loom::mapping::verifySpatialMappingBase(
          wrongTerminalRoot, dataflow, tech.view(), fabric.view())))
    fail("SpatialMapping accepted a manager as an exposure terminal");

  if (!temporal) {
    auto overlap = parseSpatial(context, finalized.canonicalBytes());
    if (!overlap)
      fail("cannot reparse memory SpatialMapping fixture");
    auto root = *overlap->getOps<::mapping::SpatialOp>().begin();
    auto records = root.getBody().front().getOps<::mapping::MemoryBindingOp>();
    auto original = *records.begin();
    mlir::OpBuilder builder(&context);
    builder.setInsertionPoint(original);
    auto first = ::mapping::MemoryBindingOp::create(
        builder, original.getLoc(), UINT64_C(0), original.getLogicalMemory(),
        ::mapping::MemoryByteRangeAttr::get(&context, 0, 8),
        original.getTarget());
    first.getBody().push_back(new mlir::Block());
    auto second = ::mapping::MemoryBindingOp::create(
        builder, original.getLoc(), UINT64_C(1), original.getLogicalMemory(),
        ::mapping::MemoryByteRangeAttr::get(&context, 8, 8),
        original.getTarget());
    second.getBody().push_back(new mlir::Block());
    original.erase();
    if (!rejectedWithoutDiagnostic(context, [&] {
          return loom::mapping::verifySpatialMappingBase(
              root, dataflow, tech.view(), fabric.view());
        }))
      fail("SpatialMapping accepted overlapping local physical intervals");
  }
}
} // namespace loom::test

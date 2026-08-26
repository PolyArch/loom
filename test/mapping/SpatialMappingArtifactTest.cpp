#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "ADG/MemoryLibrary.h"
#include "CgraAdmissionTestSupport.h"
#include "SpatialCandidateSelectionTestSupport.h"
#include "TemporalMappingFabricTestSupport.h"
#include "TemporalPeTagDomainTestSupport.h"
#include "TechMappingCandidateTestSupport.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/MappingCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/UsePatternValue.h"
#include "Fabric/Identity/FabricMemoryConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"
#include "ConfiguredHardwareProjectionInternal.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Inspection/SpatialMappingInspection.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialCanonicalSeed.h"
#include "PnR/SpatialExactRepair.h"
#include "PnR/SpatialGlobalRoutingClosure.h"
#include "PnR/SpatialMappingMaterializer.h"
#include "PnR/SpatialPathFinderRouter.h"
#include "PnR/SpatialPnrGenerator.h"
#include "PnR/SpatialPnrProblem.h"
#include "PnR/SpatialRouteCostState.h"
#include "PnR/SpatialTagAssignment.h"
#include "PnR/SpatialTagContinuity.h"
#include "SpatialOperandPairingPressure.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial mapping artifact test: " << message << '\n';
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
  const auto bytes = loom::fabric::canonicalFabricBytes(reference);
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
            "loom-spatial-mapping-artifact", path))
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

std::size_t storedObjectCount(llvm::StringRef root) {
  std::size_t count = 0;
  std::error_code error;
  llvm::sys::fs::recursive_directory_iterator iterator(root, error), end;
  if (error)
    fail("cannot inspect ArtifactStore: " + error.message());
  while (iterator != end) {
    count += llvm::sys::fs::is_regular_file(iterator->path());
    iterator.increment(error);
    if (error)
      fail("cannot inspect ArtifactStore: " + error.message());
  }
  return count;
}

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::mapping::MappingDialect,
                  ::fabric::FabricDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
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

dataflow::CanonicalDataflowArtifact buildMemoryDataflow(
    mlir::MLIRContext &context, bool splitExposures = false) {
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
    auto result = *graph.getBody().front().getOps<::dataflow::GraphReturnOp>().begin();
    result.getMemoriesMutable().slice(0, 1).assign(graph.getBody().front().getArgument(2));
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

enum class ComputeContractKind { OneCycleElastic, LoopStream, Transparent };

const ::fabric::ResourceContract &
computeContract(ComputeContractKind contractKind) {
  switch (contractKind) {
  case ComputeContractKind::OneCycleElastic:
    return ::fabric::oneCycleElasticOperationResourceContract();
  case ComputeContractKind::LoopStream:
    return ::fabric::loopStreamOperationResourceContract();
  case ComputeContractKind::Transparent:
    return ::fabric::loopCarryOperationResourceContract();
  }
  fail("unknown compute contract kind");
}

loom::fabric::FinalizedFabricRoot buildFabric(
    loom::ArtifactStore &store,
    ComputeContractKind contractKind = ComputeContractKind::OneCycleElastic) {
  using loom::adg::DesignBuilder;
  using loom::adg::PeSpec;
  using loom::adg::PortType;

  const PortType bits128 = take(PortType::bits(128));
  const std::vector<PortType> types(4, bits128);
  DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("sync", types, types));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe = take(spatial.addPe(spatialInputs, PeSpec::spatial(types, types)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  loom::test::addTokenSyncFu(pe, peInputs, bits128,
                             computeContract(contractKind));
  requireSuccess(pe.close());
  std::vector<loom::adg::SpatialValue> outputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    outputs.push_back(take(pe.output(ordinal)));
  requireSuccess(spatial.close(outputs));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("Fabric fixture did not publish exactly one root");
  return design.roots().front();
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

loom::fabric::FinalizedFabricRoot
buildTemporalFabric(loom::ArtifactStore &store) {
  auto design = loom::test::buildTemporalCapacityFabric(store);
  if (design.roots().size() != 1)
    fail("Temporal Fabric fixture did not publish exactly one root");
  return design.roots().front();
}

loom::fabric::FinalizedFabricRoot
buildTemporalSwitchPackingFabric(loom::ArtifactStore &store,
                                 std::uint64_t residentRows = 2) {
  auto design =
      loom::test::buildTemporalSwitchPackingFabric(store, residentRows);
  if (design.roots().size() != 1)
    fail("Temporal switch fixture did not publish exactly one root");
  return design.roots().front();
}

void temporalPeTagMatchDomainsAreIngressLocal() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  const auto fabric = loom::test::buildBoundaryTemporalFabric(store);
  requireSuccess(loom::test::verifyTemporalPeIngressTagDomains(fabric));
}

mlir::OwningOpRef<mlir::ModuleOp>
parseSpatial(mlir::MLIRContext &context,
             const loom::CanonicalSemanticBytes &bytes) {
  std::string text = "module {\n";
  text.append(reinterpret_cast<const char *>(bytes.bytes().data()),
              bytes.bytes().size());
  text += "}\n";
  return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
}

void selectLegalTemporalBinding(loom::pnr::SpatialCandidateState &candidate,
                                loom::pnr::SpatialCandidateScratch &scratch,
                                bool requireLocalTagInterference,
                                bool requireDistinctBoundaryEndpoints) {
  const auto &problem = candidate.problem();
  const auto realizations = problem.realizations().computeRealizations();
  if (realizations.size() != 1)
    fail("Temporal SpatialMapping fixture does not have one realization");
  const auto &realization = realizations.front();
  std::optional<loom::pnr::SpatialComputeBindingSelection> legal;
  for (loom::pnr::PnrIndex placement = realization.placementOffset;
       placement != realization.placementOffset + realization.placementCount;
       ++placement) {
    const auto &record = problem.realizations().computePlacements()[placement];
    for (loom::pnr::PnrIndex context = record.contextOffset;
         context != record.contextOffset + record.contextCount; ++context)
      if (problem.capacity().computeInstructionContextOveruse()[context] == 0) {
        legal = loom::pnr::SpatialComputeBindingSelection{placement, context};
        break;
      }
    if (legal)
      break;
  }
  if (!legal)
    fail("Temporal SpatialMapping fixture has no legal compute binding");
  auto move = take(candidate.beginMove(scratch));
  requireSuccess(
      move.setComputeBinding(0, legal->placement, legal->instructionContext));
  std::vector<loom::pnr::PnrIndex> selectedAttachments(
      problem.ports().portDemands().size());
  std::vector<loom::pnr::PnrIndex> selectedEndpoints;
  for (auto [demandOrdinal, demand] :
       llvm::enumerate(problem.ports().portDemands())) {
    const auto &domain =
        problem.ports()
            .placementDomains()[demand.placementDomainOffset +
                                legal->placement - realization.placementOffset];
    selectedAttachments[demandOrdinal] = domain.attachmentOptionOffset;
    if (requireLocalTagInterference)
      continue;
    const auto options = problem.ports().attachmentOptions();
    const auto available = options.slice(domain.attachmentOptionOffset,
                                         domain.attachmentOptionCount);
    const auto found = llvm::find_if(available, [&](const auto &option) {
      return !llvm::is_contained(selectedEndpoints, option.endpoint);
    });
    if (found == available.end())
      fail("Temporal route fixture has no distinct PE ingress assignment");
    selectedAttachments[demandOrdinal] =
        domain.attachmentOptionOffset +
        static_cast<loom::pnr::PnrIndex>(found - available.begin());
    selectedEndpoints.push_back(found->endpoint);
  }
  if (requireLocalTagInterference) {
    bool foundSharedIngress = false;
    const auto demands = problem.ports().portDemands();
    const auto domains = problem.ports().placementDomains();
    const auto options = problem.ports().attachmentOptions();
    for (std::size_t lhs = 0; !foundSharedIngress && lhs < demands.size();
         ++lhs) {
      const auto &lhsDomain =
          domains[demands[lhs].placementDomainOffset + legal->placement -
                  realization.placementOffset];
      for (std::size_t rhs = lhs + 1;
           !foundSharedIngress && rhs < demands.size(); ++rhs) {
        if (demands[lhs].logicalNet == demands[rhs].logicalNet)
          continue;
        const auto &rhsDomain =
            domains[demands[rhs].placementDomainOffset + legal->placement -
                    realization.placementOffset];
        for (loom::pnr::PnrIndex lhsOption = lhsDomain.attachmentOptionOffset;
             !foundSharedIngress &&
             lhsOption != lhsDomain.attachmentOptionOffset +
                              lhsDomain.attachmentOptionCount;
             ++lhsOption)
          for (loom::pnr::PnrIndex rhsOption = rhsDomain.attachmentOptionOffset;
               rhsOption != rhsDomain.attachmentOptionOffset +
                                rhsDomain.attachmentOptionCount;
               ++rhsOption)
            if (options[lhsOption].endpoint == options[rhsOption].endpoint) {
              selectedAttachments[lhs] = lhsOption;
              selectedAttachments[rhs] = rhsOption;
              foundSharedIngress = true;
              break;
            }
      }
    }
    if (!foundSharedIngress)
      fail("Temporal route fixture has no shared ingress attachment");
  }
  for (auto [demandOrdinal, option] : llvm::enumerate(selectedAttachments))
    requireSuccess(move.setPortAttachment(
        static_cast<loom::pnr::PnrIndex>(demandOrdinal), option));
  requireSuccess(loom::test::selectReachableGraphBoundaries(
      candidate, move, selectedAttachments, requireDistinctBoundaryEndpoints));
  if (!take(move.close()))
    fail("legal Temporal binding closes a selected handshake cycle");
  requireSuccess(move.commit());
}

void operandPairingPressureIsIncremental(
    loom::pnr::SpatialCandidateState &candidate,
    loom::pnr::SpatialCandidateScratch &scratch) {
  const auto &problem = candidate.problem();
  const auto groups = problem.ports().operandPairingGroups();
  if (groups.empty())
    fail("Temporal workflow has no Dataflow-owned operand pairing group");
  const auto options = problem.ports().attachmentOptions();
  std::optional<loom::pnr::PnrIndex> changedDemand;
  std::optional<loom::pnr::PnrIndex> sharedOption;
  for (loom::pnr::PnrIndex group = 0; group < groups.size() && !sharedOption;
       ++group) {
    const auto members = problem.ports().operandPairingGroupMembers(group);
    for (std::size_t lhs = 0; lhs < members.size() && !sharedOption; ++lhs) {
      const auto &lhsOption = options[candidate.portAttachment(members[lhs])];
      for (std::size_t rhs = lhs + 1; rhs < members.size() && !sharedOption;
           ++rhs) {
        const loom::pnr::PnrIndex demand = members[rhs];
        const auto &record = problem.ports().portDemands()[demand];
        const auto &binding = candidate.computeBinding(record.realization);
        for (const auto &domain : problem.ports().placementDomains().slice(
                 record.placementDomainOffset, record.placementDomainCount)) {
          if (domain.placement != binding.placement)
            continue;
          for (loom::pnr::PnrIndex option = domain.attachmentOptionOffset;
               option !=
               domain.attachmentOptionOffset + domain.attachmentOptionCount;
               ++option) {
            if (option != candidate.portAttachment(demand) &&
                options[option].endpoint == lhsOption.endpoint &&
                options[option].progressBoundary ==
                    loom::mapping::SpatialDurableProgressBoundaryKind::
                        TemporalPeOperandQueue) {
              changedDemand = demand;
              sharedOption = option;
              break;
            }
          }
        }
      }
    }
  }
  if (!changedDemand || !sharedOption)
    fail("Temporal workflow has no same-ingress analytic alternative");

  const std::uint64_t baseline = candidate.sharedOperandIngressPressure();
  {
    auto move = take(candidate.beginMove(scratch));
    requireSuccess(move.setPortAttachment(*changedDemand, *sharedOption));
    if (candidate.sharedOperandIngressPressure() <= baseline)
      fail("same-ingress ordered inputs did not increase pairing pressure");
    const auto objectivePressure = take(loom::pnr::spatialMappingMeasureValue(
        candidate,
        loom::pnr::MappingMeasureKind::SharedOperandIngressPressure));
    if (objectivePressure != candidate.sharedOperandIngressPressure())
      fail("central objective lost incremental operand pairing pressure");
    std::vector<loom::pnr::PnrIndex> registerFifoTransfers(
        problem.transfers().logicalNets().size());
    for (loom::pnr::PnrIndex logicalNet = 0;
         logicalNet < registerFifoTransfers.size(); ++logicalNet)
      registerFifoTransfers[logicalNet] =
          candidate.registerFifoTransfer(logicalNet);
    const auto coldPressure =
        take(loom::pnr::detail::measureSpatialOperandIngressPressure(
            problem, candidate.portAttachmentSelections(),
            registerFifoTransfers));
    if (coldPressure != candidate.sharedOperandIngressPressure())
      fail("incremental operand pairing pressure differs from cold replay");
    move.rollback();
  }
  if (candidate.sharedOperandIngressPressure() != baseline)
    fail("operand pairing rollback did not restore the candidate measure");
  requireSuccess(candidate.verify());
}

enum class SwitchResidencyExpectation : std::uint8_t {
  Fits,
  ExceedsCapacity,
};

void completeCandidateRoundTrip(
    bool temporal, bool boundaryWrapped = false, bool forceTagConflict = false,
    ComputeContractKind contractKind = ComputeContractKind::OneCycleElastic,
    bool switchPackingFabric = false, bool requireSeparatedSwitchRows = false,
    SwitchResidencyExpectation switchResidency =
        SwitchResidencyExpectation::Fits) {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  llvm::SmallString<128> blobPath(directory.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  const loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildDataflow(context);
  const auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const std::uint64_t switchResidentRows =
      switchResidency == SwitchResidencyExpectation::ExceedsCapacity ? 1 : 2;
  const auto fabric =
      boundaryWrapped ? loom::test::buildBoundaryTemporalFabric(store)
      : switchPackingFabric
          ? buildTemporalSwitchPackingFabric(store, switchResidentRows)
      : temporal ? buildTemporalFabric(store)
                 : buildFabric(store, contractKind);

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
    fail("TechMapping fixture did not produce one candidate");
  const auto tech = take(
      loom::mapping::importTechMapping(candidates->candidates.front(), store));
  const auto constraints = loom::test::buildSpatialMappingConstraints(
      context, dataflow, tech.view(), fabric.view(), store, forceTagConflict);
  if (boundaryWrapped && !forceTagConflict)
    loom::test::exerciseSpatialTagConstraintRelations(
        context, dataflow, tech.view(), fabric.view(), store);
  if (!temporal && !boundaryWrapped && !forceTagConflict)
    loom::test::exerciseSpatialAttachmentConstraintRelations(
        context, dataflow, tech.view(), fabric.view(), store);
  if (contractKind == ComputeContractKind::OneCycleElastic && !temporal &&
      !boundaryWrapped && !forceTagConflict) {
    loom::ResolvedConfig generatorResolved =
        loom::test::buildSpatialPnrTestResolvedConfig();
    auto &search = generatorResolved.dse.spatialPnr.search;
    search.initializer.seedAttemptCount = 2;
    search.actionProposal = {0, 1, 0};
    search.annealing.calibrationProposalCount = 1;
    search.annealing.fallbackTemperature = 1;
    search.annealing.minimumTemperature = 1;
    search.annealing.coolingRatio = {1, 2};
    search.annealing.proposalsPerLevelBase = 1;
    search.annealing.proposalsPerMovableDecision = 0;
    search.exactRepair = {loom::ResolvedPnrExactRepairKind::Disabled, 0, 0};
    const auto generatorConfig =
        take(loom::pnr::projectResolvedSpatialPnrConfigView(generatorResolved));
    const auto physicalTiming =
        take(loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
            fabric.view()));
    const auto physicalTimingReference = take(
        loom::fabric::publishFabricPhysicalTimingProfile(physicalTiming,
                                                         store));
    const auto typedGeneratorInputs =
        take(loom::dse::bindSpatialPnrCandidateGeneratorInputs(
            dataflowReference, candidates->candidates.front(),
            fabric.reference(), physicalTimingReference,
            constraints.reference()));
    const auto generatorBinding = take(
        loom::dse::resolveSpatialPnrCandidateGeneratorBinding(generatorConfig));
    const loom::pnr::SpatialPnrGenerationInputs generatorInputs{
        dataflow, tech.view(), fabric.view(), physicalTiming, generatorConfig,
        constraints.view(), store};
    auto generatedSpatial = loom::dse::invokeSpatialPnrCandidateGenerator(
        typedGeneratorInputs, generatorBinding, store);
    const auto *generated =
        std::get_if<loom::pnr::GeneratedSpatialMappings>(&generatedSpatial);
    if (!generated || generated->candidates.size() != 2 ||
        generated->accounting.seedAttemptSlots != 2 ||
        generated->accounting.preparedSeeds != 2 ||
        generated->accounting.finalizedRestarts != 2 ||
        generated->accounting.publicationSlots != 2)
      fail("Spatial PnR generator refilled or lost a fixed restart slot");
    if (!std::is_sorted(generated->candidates.begin(),
                        generated->candidates.end(),
                        loom::artifactRootReferenceLess))
      fail("Spatial PnR generator did not return a canonical candidate set");
    auto genericSpatial = take(loom::dse::invokeCandidateGenerator(
        typedGeneratorInputs, generatorBinding, store, blobs));
    const auto *genericCompleted =
        std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
            &genericSpatial.outcome);
    if (!genericCompleted || genericCompleted->outputBindings.size() != 1 ||
        genericCompleted->outputBindings.front().artifacts !=
            generated->candidates)
      fail("typed Generate provider diverged from its Spatial PnR owner");
    if (genericCompleted->lineageEdges.size() != generated->candidates.size())
      fail("typed Generate provider lost Spatial Mapping lineage");
    for (const loom::dse::CandidateGeneratorLineageEdge &edge :
         genericCompleted->lineageEdges)
      if (edge.kind != loom::dse::CandidateGeneratorLineageEdgeKind::
                           MechanicalDerivation ||
          edge.outputSlot != loom::dse::CandidateGeneratorOutputSlotRef(0) ||
          !std::binary_search(generated->candidates.begin(),
                              generated->candidates.end(), edge.output,
                              loom::artifactRootReferenceLess) ||
          !edge.parents.empty() || !edge.ownerPayload.empty())
        fail("Spatial Mapping lineage is not a mechanical derivation");
    auto repeatedSpatial = loom::dse::invokeSpatialPnrCandidateGenerator(
        typedGeneratorInputs, generatorBinding, store);
    const auto *repeated =
        std::get_if<loom::pnr::GeneratedSpatialMappings>(&repeatedSpatial);
    if (!repeated || repeated->candidates != generated->candidates ||
        !(repeated->accounting == generated->accounting))
      fail("Spatial PnR generator is not deterministic for exact inputs");
    auto directSpatial = loom::pnr::generateSpatialMappings(generatorInputs);
    const auto *direct =
        std::get_if<loom::pnr::GeneratedSpatialMappings>(&directSpatial);
    if (!direct || direct->candidates != generated->candidates ||
        !(direct->accounting == generated->accounting))
      fail("central Spatial generator diverged from its PnR owner API");
    auto generatedView = take(loom::mapping::importSpatialMapping(
        generated->candidates.front(), store));
    if (generatedView.view().computeBindings().empty() ||
        generatedView.view().routeTrees().empty() ||
        generatedView.view().resourceUses().empty())
      fail("Spatial PnR generator published an empty Mapping");
    const auto inspection = take(loom::mapping::inspectSpatialMapping(
        dataflow, tech.view(), fabric.view(), generatedView.view()));
    if (inspection.summary.computeRealizationCount == 0 ||
        inspection.summary.selectedActorCount == 0 ||
        inspection.summary.routeTreeCount == 0 ||
        inspection.summary.routeNodeCount == 0 ||
        inspection.summary.routeSinkCount == 0 ||
        inspection.summary.resourceUseCount == 0 ||
        inspection.computeOccupancy.empty() || inspection.routes.empty())
      fail("Spatial Mapping inspection projected an empty physical mapping");
    const auto foreignFabric = buildTemporalFabric(store);
    loom::test::exerciseCgraAdmission(
        dataflowReference, fabric.reference(), generated->candidates.front(),
        foreignFabric.reference(), store, blobs, false, true);
    auto wrongInspection = loom::mapping::inspectSpatialMapping(
        dataflow, tech.view(), foreignFabric.view(), generatedView.view());
    if (wrongInspection)
      fail("Spatial Mapping inspection accepted a foreign Fabric");
    llvm::consumeError(wrongInspection.takeError());
    requireSuccess(loom::mapping::admitSpatialMappingConstraints(
        dataflow, tech.view(), fabric.view(), constraints.view(),
        generatedView.view()));

    const auto placementRejectingConstraints =
        loom::test::buildSpatialMappingConstraints(
            context, dataflow, tech.view(), fabric.view(), store,
            /*restrictTagsToZero=*/false, /*rejectComputePlacement=*/true);
    llvm::Error rejection = loom::mapping::admitSpatialMappingConstraints(
        dataflow, tech.view(), fabric.view(),
        placementRejectingConstraints.view(), generatedView.view());
    bool rejectedByConstraintSet = false;
    llvm::handleAllErrors(
        std::move(rejection),
        [&](const loom::mapping::SpatialMappingConstraintRejection &) {
          rejectedByConstraintSet = true;
        },
        [&](const llvm::ErrorInfoBase &error) { fail(error.message()); });
    if (!rejectedByConstraintSet)
      fail("Spatial constraint admission accepted a forbidden compute "
           "placement");

    generatorResolved.dse.systemPnr.search.routing.negotiation =
        loom::ResolvedDualSubgradientPolicy{
            loom::ResolvedDualDirectionKernel::ProjectedSigned,
            std::nullopt,
            {loom::ResolvedDualStepScheduleKind::Constant, 1, 0, 0, 0}};
    const auto wrongDomainConfig =
        take(loom::pnr::projectResolvedSystemPnrConfigView(generatorResolved));
    const loom::pnr::SpatialPnrGenerationInputs wrongDomainInputs{
        dataflow,          tech.view(),        fabric.view(), physicalTiming,
        wrongDomainConfig, constraints.view(), store};
    const auto wrongDomainOutcome =
        loom::pnr::generateSpatialMappings(wrongDomainInputs);
    const auto *invalid = std::get_if<loom::pnr::InvalidSpatialPnrGeneration>(
        &wrongDomainOutcome);
    if (!invalid ||
        invalid->reason !=
            loom::pnr::InvalidSpatialPnrGenerationReason::FrozenInput ||
        invalid->accounting.seedAttemptSlots != 0 ||
        invalid->accounting.publicationSlots != 0)
      fail("Spatial PnR generator checked provider support before exact input "
           "coupling");
  }
  const auto pnrConfig = take(loom::pnr::projectResolvedSpatialPnrConfigView(
      loom::test::buildSpatialPnrTestResolvedConfig()));
  auto problem = take(loom::pnr::freezeSpatialPnrProblem(
      dataflow, tech.view(), fabric.view(), pnrConfig, constraints.view()));
  if (problem->routing().tagContinuity().traversalPointOrdinals().size() !=
      problem->routing().traversals().size())
    fail("Spatial freeze omitted its traversal-dense tag-continuity index");
  loom::pnr::SpatialCandidateStateHandle candidate;
  if (temporal) {
    candidate = take(loom::pnr::createCanonicalSpatialCandidate(problem));
    for (loom::pnr::PnrIndex net = 0;
         net < problem->transfers().logicalNets().size(); ++net) {
      const auto unrouted = take(
          loom::pnr::deriveSpatialTagContinuity(candidate->routeTree(net)));
      if (!unrouted.segments().empty() || !unrouted.segmentDomains().empty() ||
          !unrouted.domainSegments().empty() ||
          !llvm::equal(unrouted.segmentDomainOffsets(),
                       std::array<loom::pnr::PnrIndex, 1>{0}) ||
          unrouted.domainSegmentOffsets().size() !=
              problem->routing().tagContinuity().matchDomains().size() + 1 ||
          llvm::any_of(unrouted.domainSegmentOffsets(),
                       [](loom::pnr::PnrIndex value) { return value != 0; }))
        fail("unrouted tag continuity did not produce a complete empty CSR");
    }
    loom::pnr::SpatialCandidateScratch candidateScratch;
    requireSuccess(candidateScratch.prepare(*problem));
    selectLegalTemporalBinding(*candidate, candidateScratch,
                               boundaryWrapped || forceTagConflict ||
                                   requireSeparatedSwitchRows,
                               switchPackingFabric &&
                                   !requireSeparatedSwitchRows);
    if (!boundaryWrapped && !forceTagConflict && !switchPackingFabric)
      operandPairingPressureIsIncremental(*candidate, candidateScratch);
    if (forceTagConflict ||
        switchResidency == SwitchResidencyExpectation::ExceedsCapacity) {
      auto costs = take(loom::pnr::SpatialRouteCostState::create(*candidate));
      loom::pnr::SpatialNetRouterScratch router;
      requireSuccess(router.prepare(*problem));
      std::vector<loom::pnr::PnrIndex> logicalNets(
          problem->transfers().logicalNets().size());
      std::iota(logicalNets.begin(), logicalNets.end(), 0);
      requireSuccess(router.beginConstraintSweep(logicalNets));
      auto move = take(candidate->beginMove(candidateScratch));
      for (loom::pnr::PnrIndex logicalNet : logicalNets) {
        requireSuccess(costs.selectLogicalNet(logicalNet));
        take(router.routeWholeNet(
            move, *candidate, costs, logicalNet,
            pnrConfig.policy().search.routing.endpointExpansionLimit));
        requireSuccess(costs.acceptSelectedLogicalNet());
        requireSuccess(router.finishConstraintNet(logicalNet));
      }
      if (!take(move.close()))
        fail("tag-pressure fixture closed a selected handshake cycle");
      requireSuccess(move.commit());
      requireSuccess(costs.resetFromCandidate());
      if (!costs.hasTagPressureViolation())
        fail("PathFinder cost state ignored committed tag pressure");
      if (switchResidency == SwitchResidencyExpectation::ExceedsCapacity) {
        if (candidate->tagResidentCapacityOveruse() == 0)
          fail("Temporal switch fixture did not exceed resident capacity");
        requireSuccess(candidate->verify());
        return;
      }
      if (candidate->tagConflictCount() == 0)
        fail("conflicting tag fixture did not construct a collision");
      const std::vector<loom::pnr::RouteCost> baselineTagCosts(
          costs.currentArcCosts().begin(), costs.currentArcCosts().end());
      requireSuccess(costs.advancePathFinderIteration());
      if (!llvm::any_of(
              llvm::zip_equal(baselineTagCosts, costs.currentArcCosts()),
              [](const auto &entry) {
                return std::get<1>(entry) > std::get<0>(entry);
              }))
        fail("PathFinder did not raise cost around a conflicting tag domain");
      requireSuccess(costs.resetFromCandidate());
      if (!llvm::equal(baselineTagCosts, costs.currentArcCosts()))
        fail("PathFinder reset retained tag-domain history pressure");
    }
  } else {
    loom::pnr::SpatialPathFinderSeedWorkSummary firstWork;
    loom::pnr::SpatialPathFinderSeedWorkSummary secondWork;
    auto first = take(
        loom::pnr::createCanonicalPathFinderSpatialSeed(problem, firstWork));
    auto second = take(
        loom::pnr::createCanonicalPathFinderSpatialSeed(problem, secondWork));
    if (firstWork.negotiationIterations != secondWork.negotiationIterations ||
        firstWork.endpointExpansions != secondWork.endpointExpansions ||
        first.candidate->unroutedObligationCount() != 0 ||
        second.candidate->unroutedObligationCount() != 0)
      fail("canonical Spatial routing seed is not closed and deterministic");
    for (loom::pnr::PnrIndex net = 0;
         net < problem->transfers().logicalNets().size(); ++net) {
      const auto &firstTree = first.candidate->routeTree(net);
      const auto &secondTree = second.candidate->routeTree(net);
      if (!firstTree.isRouted() || !secondTree.isRouted() ||
          firstTree.sourceEndpoint() != secondTree.sourceEndpoint() ||
          !llvm::equal(firstTree.nodeStorage(), secondTree.nodeStorage()))
        fail("canonical Spatial routing seed changed its RouteTree");
      for (loom::pnr::PnrIndex sink = 0;
           sink < problem->transfers().logicalNets()[net].sinkCount; ++sink)
        if (firstTree.sinkEndpoint(sink) != secondTree.sinkEndpoint(sink))
          fail("canonical Spatial routing seed changed a sink attachment");
    }
    requireSuccess(second.candidate->verify());
    candidate = std::move(first.candidate);
  }
  loom::pnr::SpatialGlobalRoutingClosureScratch globalRoutingClosure;
  std::optional<loom::pnr::HandshakeActiveDemandStatistics>
      handshakeAfterClosure;
  std::optional<loom::pnr::SpatialProgressStatistics> progressAfterClosure;
  if (forceTagConflict) {
    const std::uint64_t selectedTraversalClaim =
        candidate->totalSelectedTraversalClaim();
    const std::uint64_t unroutedObligations =
        candidate->unroutedObligationCount();
    const std::uint64_t tagConflicts = candidate->tagConflictCount();
    llvm::Error rejectedClosure = globalRoutingClosure.run(*candidate);
    if (!rejectedClosure)
      fail("global routing closure accepted conflicting Physical Tags");
    bool observedBoundedFailure = false;
    llvm::handleAllErrors(
        std::move(rejectedClosure),
        [&](const loom::pnr::SpatialActionTransitionFailure &failure) {
          observedBoundedFailure =
              failure.kind() ==
              loom::pnr::SpatialActionTransitionFailureKind::WorkLimit;
        });
    if (!observedBoundedFailure)
      fail("global routing closure lost its bounded tag-conflict failure");
    if (candidate->totalSelectedTraversalClaim() != selectedTraversalClaim ||
        candidate->unroutedObligationCount() != unroutedObligations ||
        candidate->tagConflictCount() != tagConflicts)
      fail("rejected global routing closure changed the candidate");
  } else {
    const auto handshakeBeforeClosure =
        candidate->handshake().materializationStatistics();
    const auto progressBeforeClosure = candidate->progress().statistics();
    requireSuccess(globalRoutingClosure.run(*candidate));
    const std::size_t retainedClosureBytes =
        globalRoutingClosure.retainedStorageBytes();
    requireSuccess(globalRoutingClosure.run(*candidate));
    const std::size_t repeatedClosureBytes =
        globalRoutingClosure.retainedStorageBytes();
    if (repeatedClosureBytes > retainedClosureBytes)
      fail("warmed global routing closure grew worker-local storage from " +
           std::to_string(retainedClosureBytes) + " to " +
           std::to_string(repeatedClosureBytes));
    handshakeAfterClosure =
        candidate->handshake().materializationStatistics();
    progressAfterClosure = candidate->progress().statistics();
    if (handshakeAfterClosure->cachedVerificationCount <=
            handshakeBeforeClosure.cachedVerificationCount ||
        progressAfterClosure->cachedVerificationCount <=
            progressBeforeClosure.cachedVerificationCount)
      fail("global routing closure did not verify incremental candidate state");
    if (handshakeAfterClosure->coldVerificationConstructionCount !=
            handshakeBeforeClosure.coldVerificationConstructionCount ||
        progressAfterClosure->coldVerificationCount !=
            progressBeforeClosure.coldVerificationCount ||
        progressAfterClosure->coldProgressScanCount !=
            progressBeforeClosure.coldProgressScanCount)
      fail("global routing closure invoked a publication verifier");
  }
  requireSuccess(candidate->verify());
  if (handshakeAfterClosure &&
      (candidate->handshake()
               .materializationStatistics()
               .coldVerificationConstructionCount !=
           handshakeAfterClosure->coldVerificationConstructionCount + 1 ||
       candidate->progress().statistics().coldVerificationCount !=
           progressAfterClosure->coldVerificationCount + 1 ||
       candidate->progress().statistics().coldProgressScanCount !=
           progressAfterClosure->coldProgressScanCount + 1))
    fail("final candidate verification omitted an independent reconstruction");

  std::vector<const loom::pnr::RouteTreeState *> selectedRoutes;
  selectedRoutes.reserve(problem->transfers().logicalNets().size());
  for (loom::pnr::PnrIndex net = 0;
       net < problem->transfers().logicalNets().size(); ++net)
    selectedRoutes.push_back(&candidate->routeTree(net));
  const auto tagAssignments =
      take(loom::pnr::deriveCanonicalSpatialTagAssignments(*problem,
                                                           selectedRoutes));
  const auto repeatedTagAssignments =
      take(loom::pnr::deriveCanonicalSpatialTagAssignments(*problem,
                                                           selectedRoutes));
  const std::uint64_t expectedTagConflicts = tagAssignments.conflictCount();
  if (tagAssignments.segments() != repeatedTagAssignments.segments() ||
      tagAssignments.values() != repeatedTagAssignments.values() ||
      tagAssignments.segmentDomains() !=
          repeatedTagAssignments.segmentDomains() ||
      tagAssignments.unassignedCount() != 0 ||
      (forceTagConflict ? expectedTagConflicts == 0
                        : expectedTagConflicts != 0))
    fail("canonical Physical Tag assignment is incomplete or unstable");
  if (candidate->tagUnassignedCount() != 0 ||
      candidate->tagConflictCount() != expectedTagConflicts)
    fail("Spatial Candidate lost its Physical Tag assignment violations");
  if (take(loom::pnr::spatialMappingViolationValue(
          *candidate, loom::ResolvedPnrViolationKind::TagUnassigned)) != 0 ||
      take(loom::pnr::spatialMappingViolationValue(
          *candidate, loom::ResolvedPnrViolationKind::TagConflict)) !=
          expectedTagConflicts)
    fail("Spatial objective did not project Candidate-owned Tag violations");
  for (auto [segment, value] :
       llvm::zip_equal(tagAssignments.segments(), tagAssignments.values())) {
    if (!value)
      fail("routed Physical Tag segment has no canonical value");
    const auto encoded =
        take(fabric::encodePhysicalTagValue(segment.tagWidthBits, *value));
    if (take(fabric::decodePhysicalTagValue(segment.tagWidthBits, encoded)) !=
        value->zextOrTrunc(segment.tagWidthBits))
      fail("candidate Physical Tag disagrees with the Fabric owner codec");
  }

  bool observedSharedLocalDomain = false;
  bool observedPackedSwitchDomain = false;
  bool observedSeparatedSwitchDomain = false;
  std::size_t maximumLocalDomainOccupancy = 0;
  const auto matchDomains = problem->routing().tagContinuity().matchDomains();
  for (loom::pnr::PnrIndex domain = 0; domain < matchDomains.size(); ++domain) {
    const auto members = tagAssignments.domainSegments(domain);
    maximumLocalDomainOccupancy =
        std::max(maximumLocalDomainOccupancy, members.size());
    observedSharedLocalDomain |= members.size() > 1;
    std::size_t distinctValues = 0;
    for (std::size_t position = 0; position != members.size(); ++position) {
      const loom::pnr::PnrIndex member = members[position];
      if (llvm::none_of(members.take_front(position),
                        [&](loom::pnr::PnrIndex prior) {
                          return tagAssignments.values()[prior] ==
                                 tagAssignments.values()[member];
                        }))
        ++distinctValues;
    }
    const bool temporalSwitch =
        matchDomains[domain].kind ==
        loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable;
    if (temporalSwitch) {
      if (candidate->tagDomainResidentCount(domain) != distinctValues ||
          candidate->tagDomainResidentCapacityOveruse(domain) != 0)
        fail("Temporal switch residency did not count packed tag rows");
      observedPackedSwitchDomain |= members.size() > distinctValues;
      observedSeparatedSwitchDomain |= distinctValues > 1;
      continue;
    }
    for (auto [position, lhs] : llvm::enumerate(members))
      for (loom::pnr::PnrIndex rhs : llvm::drop_begin(members, position + 1))
        if (!forceTagConflict &&
            tagAssignments.values()[lhs] == tagAssignments.values()[rhs])
          fail("one local Physical Tag match domain contains a collision");
  }
  if (switchPackingFabric && !forceTagConflict &&
      !requireSeparatedSwitchRows && !observedPackedSwitchDomain)
    fail("Temporal route fixture did not pack compatible switch segments");
  if (requireSeparatedSwitchRows && !observedSeparatedSwitchDomain)
    fail("Temporal route fixture did not select distinct resident rows");
  if (switchPackingFabric && forceTagConflict) {
    const bool observedSwitchConflict = llvm::any_of(
        llvm::seq<loom::pnr::PnrIndex>(0, matchDomains.size()),
        [&](loom::pnr::PnrIndex domain) {
          return matchDomains[domain].kind ==
                     loom::fabric::FabricPhysicalTagMatchDomainKind::
                         TemporalSwitchTable &&
                 candidate->tagDomainConflictCount(domain) != 0;
        });
    if (!observedSwitchConflict)
      fail("incompatible Temporal switch signatures did not conflict");
  }
  if (boundaryWrapped && !observedSharedLocalDomain)
    fail(
        ("Temporal route fixture did not exercise local tag interference: " +
         llvm::Twine(tagAssignments.segments().size()) + " segments across " +
         llvm::Twine(problem->routing().tagContinuity().matchDomains().size()) +
         " domains, maximum local occupancy " +
         llvm::Twine(maximumLocalDomainOccupancy))
            .str());

  bool observedDisjointDomainReuse = false;
  const auto segmentDomainOffsets = tagAssignments.segmentDomainOffsets();
  const auto segmentDomains = tagAssignments.segmentDomains();
  for (loom::pnr::PnrIndex lhs = 0; lhs < tagAssignments.segments().size();
       ++lhs)
    for (loom::pnr::PnrIndex rhs = lhs + 1;
         rhs < tagAssignments.segments().size(); ++rhs) {
      if (tagAssignments.values()[lhs] != tagAssignments.values()[rhs])
        continue;
      const auto lhsDomains = segmentDomains.slice(
          segmentDomainOffsets[lhs],
          segmentDomainOffsets[lhs + 1] - segmentDomainOffsets[lhs]);
      const auto rhsDomains = segmentDomains.slice(
          segmentDomainOffsets[rhs],
          segmentDomainOffsets[rhs + 1] - segmentDomainOffsets[rhs]);
      observedDisjointDomainReuse |=
          llvm::none_of(lhsDomains, [&](loom::pnr::PnrIndex domain) {
            return llvm::is_contained(rhsDomains, domain);
          });
    }
  if (boundaryWrapped && !observedDisjointDomainReuse)
    fail("Physical Tags were made globally unique across disjoint domains");

  if (boundaryWrapped) {
    const std::vector<loom::pnr::SpatialTagContinuitySegment> originalSegments(
        candidate->tagSegments(0).begin(), candidate->tagSegments(0).end());
    const std::vector<std::optional<llvm::APInt>> originalValues(
        candidate->tagValues(0).begin(), candidate->tagValues(0).end());
    if (originalSegments.empty() ||
        originalSegments.size() != originalValues.size())
      fail("routed Spatial Candidate has no Physical Tag decisions");

    loom::pnr::SpatialCandidateScratch tagScratch;
    requireSuccess(tagScratch.prepare(*problem));
    auto tagCosts = take(loom::pnr::SpatialRouteCostState::create(*candidate));
    const std::vector<loom::pnr::RouteCost> originalTagCosts(
        tagCosts.currentArcCosts().begin(), tagCosts.currentArcCosts().end());
    const bool originalTagPressure = tagCosts.hasTagPressureViolation();
    auto move = take(candidate->beginMove(tagScratch));
    requireSuccess(move.ripUpWholeRoute(0));
    const auto provisional = take(move.projectCurrentRoutes());
    if (candidate->unroutedObligationCount() != 0 ||
        provisional.unroutedObligationCount !=
            problem->transfers().logicalNets()[0].sinkCount)
      fail("active Spatial move did not project its provisional RouteTree");
    const bool closed = take(move.close());
    if (closed != provisional.selectedHandshakeAcyclic)
      fail("provisional tag-aware handshake projection disagrees with move "
           "closure");
    const auto projectionStatistics =
        tagScratch.handshakeProjectionStatistics();
    if (projectionStatistics.projectionCount != 1 ||
        projectionStatistics.peakActiveNodeCount == 0 ||
        projectionStatistics.peakActiveArcCount == 0)
      fail("provisional handshake projection omitted its worker-local "
           "construction statistics");
    if (!closed)
      fail("Physical Tag rollback fixture closed a handshake cycle");
    const auto tagDelta = take(move.summarizeCurrentTagAssignmentDelta());
    requireSuccess(tagCosts.synchronizeTagProjection(tagDelta));
    if (!tagCosts.hasActiveTagProjectionDelta())
      fail("Physical Tag route-cost delta did not retain its inverse");
    if (!candidate->tagSegments(0).empty() || !candidate->tagValues(0).empty())
      fail("route rip-up retained stale Physical Tag decisions");
    move.rollback();
    requireSuccess(tagCosts.rollbackTagProjectionDelta());
    if (tagCosts.hasActiveTagProjectionDelta() ||
        tagCosts.hasTagPressureViolation() != originalTagPressure ||
        !llvm::equal(tagCosts.currentArcCosts(), originalTagCosts))
      fail("Physical Tag route-cost rollback changed its exact projection");
    if (!llvm::equal(candidate->tagSegments(0), originalSegments) ||
        !llvm::equal(candidate->tagValues(0), originalValues) ||
        candidate->tagUnassignedCount() != 0 ||
        candidate->tagConflictCount() != expectedTagConflicts)
      fail("Spatial move rollback changed Physical Tag decisions");
    requireSuccess(candidate->verify());
  }

  if (boundaryWrapped) {
    bool observedBoundaryOrigin = false;
    bool observedRouteSourceOrigin = false;
    bool observedUntaggedNode = false;
    bool observedRemoverStop = false;
    bool observedTagMatchDomain = false;
    for (loom::pnr::PnrIndex net = 0;
         net < problem->transfers().logicalNets().size(); ++net) {
      const auto projection = take(
          loom::pnr::deriveSpatialTagContinuity(candidate->routeTree(net)));
      const auto repeated = take(
          loom::pnr::deriveSpatialTagContinuity(candidate->routeTree(net)));
      if (!llvm::equal(projection.segments(), repeated.segments()) ||
          !llvm::equal(projection.nodeSegments(), repeated.nodeSegments()) ||
          !llvm::equal(projection.segmentDomainOffsets(),
                       repeated.segmentDomainOffsets()) ||
          !llvm::equal(projection.segmentDomains(),
                       repeated.segmentDomains()) ||
          !llvm::equal(projection.domainSegmentOffsets(),
                       repeated.domainSegmentOffsets()) ||
          !llvm::equal(projection.domainSegments(), repeated.domainSegments()))
        fail("tag-continuity projection is not deterministic");
      const auto &tree = candidate->routeTree(net);
      if (projection.nodeSegments().size() != tree.nodeStorage().size())
        fail("tag-continuity projection is not RouteTree-slot dense");
      for (const auto &segment : projection.segments()) {
        observedBoundaryOrigin |=
            segment.originKind ==
            loom::pnr::SpatialTagContinuityOriginKind::BoundaryPoint;
        observedRouteSourceOrigin |=
            segment.originKind ==
            loom::pnr::SpatialTagContinuityOriginKind::RouteSource;
      }
      for (auto [slot, node] : llvm::enumerate(tree.nodeStorage())) {
        if (!node.isActive())
          continue;
        const auto &dataPath =
            problem->routing().routingEndpoints()[node.endpoint].dataPath;
        const bool tagged = dataPath.kind == ::fabric::DataPathKind::BitsTag;
        if (!tagged)
          observedUntaggedNode = true;
        if ((projection.nodeSegments()[slot] !=
             loom::pnr::getInvalidPnrIndex()) != tagged)
          fail("tag-continuity segment disagrees with route endpoint kind");
        if (node.parentArc == loom::pnr::getInvalidPnrIndex())
          continue;
        const auto &arc = problem->routing().routingArcs()[node.parentArc];
        const auto point = problem->routing()
                               .tagContinuity()
                               .traversalPointOrdinals()[arc.traversal];
        if (point != loom::pnr::getInvalidPnrIndex() &&
            problem->routing().tagContinuity().points()[point].kind ==
                loom::fabric::FabricBoundaryTagContinuityKind::Remover &&
            projection.nodeSegments()[slot] == loom::pnr::getInvalidPnrIndex())
          observedRemoverStop = true;
      }

      const auto segmentDomainOffsets = projection.segmentDomainOffsets();
      const auto segmentDomains = projection.segmentDomains();
      const auto domainSegmentOffsets = projection.domainSegmentOffsets();
      const auto domainSegments = projection.domainSegments();
      const auto matchDomains =
          problem->routing().tagContinuity().matchDomains();
      if (segmentDomainOffsets.size() != projection.segments().size() + 1 ||
          domainSegmentOffsets.size() != matchDomains.size() + 1 ||
          segmentDomainOffsets.back() != segmentDomains.size() ||
          domainSegmentOffsets.back() != domainSegments.size())
        fail("tag match-domain incidence has malformed CSR bounds");
      for (loom::pnr::PnrIndex segment = 0;
           segment < projection.segments().size(); ++segment) {
        for (loom::pnr::PnrIndex incidence = segmentDomainOffsets[segment];
             incidence < segmentDomainOffsets[segment + 1]; ++incidence) {
          const auto domain = segmentDomains[incidence];
          if (domain >= matchDomains.size() ||
              matchDomains[domain].tagWidthBits !=
                  projection.segments()[segment].tagWidthBits)
            fail("tag segment intersects an incompatible match domain");
          const auto reverse = domainSegments.slice(
              domainSegmentOffsets[domain],
              domainSegmentOffsets[domain + 1] - domainSegmentOffsets[domain]);
          if (!llvm::is_contained(reverse, segment))
            fail("tag segment/domain reverse incidence is incomplete");
          observedTagMatchDomain = true;
        }
      }
    }
    if (!observedBoundaryOrigin || !observedRouteSourceOrigin ||
        !observedUntaggedNode || !observedRemoverStop ||
        !observedTagMatchDomain)
      fail("boundary Temporal route did not exercise start and stop semantics");
  }

  if (forceTagConflict)
    return;

  const auto placementRejectingConstraints =
      loom::test::buildSpatialMappingConstraints(
          context, dataflow, tech.view(), fabric.view(), store,
          /*restrictTagsToZero=*/false, /*rejectComputePlacement=*/true);
  const std::size_t storedBeforeRejection = storedObjectCount(directory.path());
  auto rejectedFinalization = loom::pnr::finalizeSpatialMappingCandidate(
      *candidate, dataflow, tech.view(), fabric.view(),
      placementRejectingConstraints.view(), store);
  if (rejectedFinalization)
    fail("constraint-rejected Spatial candidate was published");
  bool rejectedByConstraintSet = false;
  llvm::handleAllErrors(
      rejectedFinalization.takeError(),
      [&](const loom::mapping::SpatialMappingConstraintRejection &) {
        rejectedByConstraintSet = true;
      },
      [&](const llvm::ErrorInfoBase &error) { fail(error.message()); });
  if (!rejectedByConstraintSet)
    fail("Spatial candidate finalization lost typed constraint rejection");
  if (storedObjectCount(directory.path()) != storedBeforeRejection)
    fail("constraint-rejected Spatial candidate reached ArtifactStore");

  auto finalized = take(loom::pnr::finalizeSpatialMappingCandidate(
      *candidate, dataflow, tech.view(), fabric.view(), constraints.view(),
      store));
  auto imported =
      take(loom::mapping::importSpatialMapping(finalized.reference(), store));
  if (imported.reference() != finalized.reference() ||
      imported.view().computeBindings().size() != 1 ||
      imported.view().routeTrees().empty() ||
      imported.view().resourceUses().empty())
    fail("strict SpatialMapping round trip lost selected closure");
  bool observedCompleteResultTuple = false;
  bool sawComputeTransition = false;
  bool allComputeIntrinsic = true;
  for (const auto &use : imported.view().resourceUses()) {
    if (!std::holds_alternative<loom::mapping::SpatialComputeResourceOwnerRef>(
            use.owner) ||
        !std::holds_alternative<loom::mapping::SpatialActorTransitionEventRef>(
            use.activation.trigger.event))
      continue;
    sawComputeTransition = true;
    allComputeIntrinsic &= use.activation.release.empty();
    if (use.activation.release.size() != 2)
      continue;
    const auto &transition =
        std::get<loom::mapping::SpatialActorTransitionEventRef>(
            use.activation.trigger.event);
    bool firstResult = false;
    bool secondResult = false;
    for (const auto &point : use.activation.release) {
      const auto *producer =
          std::get_if<dataflow::CanonicalGraphProducerEndpointRef>(
              &point.event);
      if (!producer)
        continue;
      const auto *result = std::get_if<dataflow::ActorTokenResultRef>(producer);
      if (!result || result->actor != transition.actor)
        continue;
      firstResult |= result->ordinal == 0;
      secondResult |= result->ordinal == 1;
    }
    observedCompleteResultTuple |= firstResult && secondResult;
  }
  const bool requiresActiveResultHandoff =
      contractKind != ComputeContractKind::Transparent;
  if (requiresActiveResultHandoff && !observedCompleteResultTuple)
    fail("Spatial compute use lost its complete active-result release tuple");
  if (!requiresActiveResultHandoff &&
      (!sawComputeTransition || !allComputeIntrinsic))
    fail("same-cycle compute use gained a causal result release");
  if (requiresActiveResultHandoff) {
    const auto requireRejectedReleaseMutation = [&](bool removeMember) {
      auto mutated = parseSpatial(context, finalized.canonicalBytes());
      if (!mutated)
        fail("cannot reparse active-result release fixture");
      auto mutatedRoot = *mutated->getOps<::mapping::SpatialOp>().begin();
      std::optional<::mapping::ResourceUseOp> computeUse;
      for (auto use :
           mutatedRoot.getBody().front().getOps<::mapping::ResourceUseOp>()) {
        auto activation =
            mlir::dyn_cast<::mapping::SpatialRelativeActivationAttr>(
                use.getActivation());
        if (activation && activation.getRelease().size() == 2) {
          computeUse = use;
          break;
        }
      }
      if (!computeUse)
        fail("SpatialMapping fixture has no complete result release tuple");
      auto activation = mlir::cast<::mapping::SpatialRelativeActivationAttr>(
          computeUse->getActivation());
      auto release = llvm::to_vector(activation.getRelease());
      if (removeMember) {
        release.pop_back();
      } else {
        release.insert(release.begin(), activation.getTrigger());
      }
      computeUse->setActivationAttr(
          ::mapping::SpatialRelativeActivationAttr::get(
              &context, activation.getTrigger(),
              mlir::ArrayAttr::get(&context, release)));
      if (!rejected(loom::mapping::verifySpatialMappingBase(
              mutatedRoot, dataflow, tech.view(), fabric.view())))
        fail(removeMember
                 ? "SpatialMapping accepted an incomplete result release tuple"
                 : "SpatialMapping accepted an extra result release member");
    };
    requireRejectedReleaseMutation(true);
    requireRejectedReleaseMutation(false);
  }
  auto coldTraversalClaims =
      take(loom::pnr::projectSpatialMappingTraversalClaims(*problem,
                                                           imported.view()));
  if (coldTraversalClaims.total != candidate->totalSelectedTraversalClaim())
    fail("cold Spatial Mapping measure disagrees with Candidate state");

  if (temporal) {
    const auto operandQueueGroups =
        take(loom::mapping::deriveSpatialPeOperandQueueMatchGroups(
            tech.view(), fabric.view(), imported.view().computeBindings(),
            imported.view().routeTrees(), imported.view().resourceUses(),
            imported.view().physicalTagSegments()));
    if (operandQueueGroups.empty() ||
        llvm::any_of(operandQueueGroups, [](const auto &group) {
          return group.matches.empty() || group.tag.getBitWidth() != 4;
        }))
      fail("Temporal SpatialMapping lost its operand queue match groups");
    const auto operandProgress =
        take(loom::mapping::deriveSpatialPeOperandProgressFeedback(
            dataflow, tech.view(), operandQueueGroups));
    if (operandProgress.pairingKeyCount == 0 ||
        operandProgress.pairingKeyCount < operandProgress.distinctPairingKeyCount ||
        operandProgress.distinctIngressCount == 0)
      fail("Temporal SpatialMapping lost its qualified pairing projection");
    if (operandProgress.sharedIngressPressure !=
        candidate->sharedOperandIngressPressure())
      fail("persistent SpatialMapping changed operand ingress pressure");
    bool observedEnqueue = false;
    bool observedTransition = false;
    for (const auto &use : imported.view().resourceUses()) {
      const bool enqueue =
          std::holds_alternative<dataflow::CanonicalGraphConsumerEndpointRef>(
              use.activation.trigger.event);
      observedEnqueue |= enqueue;
      if (enqueue && !use.activation.release.empty())
        fail("Temporal queue use gained a causal result release");
      observedTransition |=
          std::holds_alternative<loom::mapping::SpatialActorTransitionEventRef>(
              use.activation.trigger.event);
    }
    if (!observedEnqueue || !observedTransition)
      fail("Temporal SpatialMapping round trip lost queue or operation uses");
    const auto switchRows =
        take(loom::mapping::deriveSpatialTemporalSwitchPackedRows(
            fabric.view(), imported.view().routeTrees(),
            imported.view().resourceUses(),
            imported.view().physicalTagSegments()));
    std::vector<loom::fabric::FabricSwitchOccurrenceRef> configuredSwitches;
    for (const auto &row : switchRows)
      if (!llvm::is_contained(configuredSwitches, row.occurrence))
        configuredSwitches.push_back(row.occurrence);
    for (const auto occurrence : configuredSwitches) {
      std::vector<loom::fabric::FabricTemporalSwitchRouteEntry> entries;
      for (const auto &row : switchRows)
        if (row.occurrence == occurrence)
          entries.push_back({row.tag, row.traversals});
      const loom::fabric::FabricSemanticConfigFieldRef field{
          loom::fabric::FabricConfigurationOwnerRef(
              loom::fabric::FabricInventoryOwnerRef::of(occurrence)),
          0};
      const auto slot = take(
          loom::mapping::detail::resolveConfiguredHardwareSlot(fabric.view(),
                                                               field));
      const auto expected = take(loom::fabric::encodeTemporalSwitchConfiguration(
          fabric.view(), field, entries));
      const auto configured = llvm::find_if(
          imported.view().configuredHardware().fields(),
          [&](const auto &candidate) { return candidate.slot == slot; });
      if (configured == imported.view().configuredHardware().fields().end() ||
          !configured->value.bytes().equals(expected.bytes()))
        fail("configured hardware diverged from Fabric switch rows");
    }
    if (switchPackingFabric && !requireSeparatedSwitchRows &&
        llvm::none_of(switchRows, [](const auto &row) {
          return row.signatures.size() > 1;
        }))
      fail("strict SpatialMapping did not preserve a packed switch row");
    if (requireSeparatedSwitchRows && switchRows.size() < 2)
      fail("strict SpatialMapping merged distinct switch rows");
    if (switchPackingFabric)
      loom::test::exerciseCgraAdmission(
          dataflowReference, fabric.reference(), finalized.reference(),
          buildTemporalFabric(store).reference(), store, blobs, true, false,
          requireSeparatedSwitchRows);
  }

  if (boundaryWrapped) {
    std::size_t expectedAssignments = 0;
    for (loom::pnr::PnrIndex net = 0;
         net < problem->transfers().logicalNets().size(); ++net)
      expectedAssignments += candidate->tagSegments(net).size();
    loom::test::exerciseCgraAdmission(
        dataflowReference, fabric.reference(), finalized.reference(),
        buildTemporalFabric(store).reference(), store, blobs, true, false);
    std::size_t observedAssignments = 0;
    for (const auto &use : imported.view().resourceUses()) {
      if (!use.sharingAssignments.empty() && !use.activation.release.empty())
        fail("Physical Tag ResourceUse gained a causal release");
      for (const auto &value : use.sharingAssignments) {
        const auto *tag = std::get_if<fabric::PhysicalTagPatternValue>(&value);
        if (!tag || tag->value.getBitWidth() != 4)
          fail("SpatialMapping did not adopt an exact Physical Tag value");
        ++observedAssignments;
      }
    }
    if (observedAssignments != expectedAssignments)
      fail("SpatialMapping did not persist every continuity origin exactly "
           "once");

    auto missingTag = parseSpatial(context, finalized.canonicalBytes());
    if (!missingTag)
      fail("cannot reparse Physical Tag ResourceUse fixture");
    auto missingTagRoot = *missingTag->getOps<::mapping::SpatialOp>().begin();
    std::optional<::mapping::ResourceUseOp> tagUse;
    for (auto use :
         missingTagRoot.getBody().front().getOps<::mapping::ResourceUseOp>())
      if (!use.getSharingAssignments().empty()) {
        tagUse = use;
        break;
      }
    if (!tagUse)
      fail("SpatialMapping fixture has no Physical Tag ResourceUse");
    tagUse->erase();
    if (!rejected(loom::mapping::verifySpatialMappingBase(
            missingTagRoot, dataflow, tech.view(), fabric.view())))
      fail("SpatialMapping finalized without a required Physical Tag "
           "assignment");

    auto malformedTag = parseSpatial(context, finalized.canonicalBytes());
    if (!malformedTag)
      fail("cannot reparse malformed Physical Tag fixture");
    auto malformedTagRoot =
        *malformedTag->getOps<::mapping::SpatialOp>().begin();
    std::optional<::mapping::ResourceUseOp> malformedUse;
    for (auto use :
         malformedTagRoot.getBody().front().getOps<::mapping::ResourceUseOp>())
      if (!use.getSharingAssignments().empty()) {
        malformedUse = use;
        break;
      }
    if (!malformedUse)
      fail("SpatialMapping fixture has no Physical Tag value to corrupt");
    const std::array<std::int8_t, 1> noncanonical = {
        static_cast<std::int8_t>(0xf0)};
    auto malformedValue = ::mapping::OwnerTypedValueAttr::get(
        &context, mlir::DenseI8ArrayAttr::get(&context, noncanonical));
    malformedUse->setSharingAssignmentsAttr(
        mlir::ArrayAttr::get(&context, {malformedValue}));
    if (!rejected(loom::mapping::verifySpatialMappingBase(
            malformedTagRoot, dataflow, tech.view(), fabric.view())))
      fail("SpatialMapping accepted a noncanonical Physical Tag value");

    auto collidingTags = parseSpatial(context, finalized.canonicalBytes());
    if (!collidingTags)
      fail("cannot reparse colliding Physical Tag fixture");
    auto collidingTagRoot =
        *collidingTags->getOps<::mapping::SpatialOp>().begin();
    const std::array<std::int8_t, 1> zeroTag = {0};
    auto zeroValue = ::mapping::OwnerTypedValueAttr::get(
        &context, mlir::DenseI8ArrayAttr::get(&context, zeroTag));
    std::size_t rewrittenAssignments = 0;
    for (auto use :
         collidingTagRoot.getBody().front().getOps<::mapping::ResourceUseOp>())
      if (!use.getSharingAssignments().empty()) {
        use.setSharingAssignmentsAttr(
            mlir::ArrayAttr::get(&context, {zeroValue}));
        ++rewrittenAssignments;
      }
    if (rewrittenAssignments < 2 ||
        !rejected(loom::mapping::verifySpatialMappingBase(
            collidingTagRoot, dataflow, tech.view(), fabric.view())))
      fail("SpatialMapping accepted colliding local Physical Tags");
  }

  auto missingRoute = parseSpatial(context, finalized.canonicalBytes());
  if (!missingRoute)
    fail("cannot parse finalized SpatialMapping fixture");
  auto root = *missingRoute->getOps<::mapping::SpatialOp>().begin();
  auto routes = root.getBody().front().getOps<::mapping::RouteTreeOp>();
  (*routes.begin()).erase();
  if (!rejectedWithoutDiagnostic(context, [&] {
        return loom::mapping::verifySpatialMappingBase(
            root, dataflow, tech.view(), fabric.view());
      }))
    fail("SpatialMapping finalized without a required RouteTree");

  auto missingUse = parseSpatial(context, finalized.canonicalBytes());
  if (!missingUse)
    fail("cannot reparse finalized SpatialMapping fixture");
  auto useRoot = *missingUse->getOps<::mapping::SpatialOp>().begin();
  auto uses = useRoot.getBody().front().getOps<::mapping::ResourceUseOp>();
  (*uses.begin()).erase();
  if (!rejected(loom::mapping::verifySpatialMappingBase(
          useRoot, dataflow, tech.view(), fabric.view())))
    fail("SpatialMapping finalized without a required ResourceUse");
}

void completeMemoryCandidateRoundTrip(bool temporal, bool splitExposures = false) {
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
    const auto unsupported = loom::pnr::unsupportedSpatialExactRepairDomain(*problem);
    if (!unsupported || !llvm::StringRef(*unsupported).contains("exposure-provider"))
      fail("CpSat exact-repair preflight admitted unsupported exposure capacity");
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
} // namespace

int main() {
  temporalPeTagMatchDomainsAreIngressLocal();
  completeCandidateRoundTrip(false);
  completeCandidateRoundTrip(false, false, false,
                             ComputeContractKind::LoopStream);
  completeCandidateRoundTrip(false, false, false,
                             ComputeContractKind::Transparent);
  completeCandidateRoundTrip(true);
  completeCandidateRoundTrip(true, false, false,
                             ComputeContractKind::OneCycleElastic, true);
  completeCandidateRoundTrip(true, false, false,
                             ComputeContractKind::OneCycleElastic, true, true);
  completeCandidateRoundTrip(true, false, false,
                             ComputeContractKind::OneCycleElastic, true, true,
                             SwitchResidencyExpectation::ExceedsCapacity);
  completeCandidateRoundTrip(true, false, true,
                             ComputeContractKind::OneCycleElastic, true);
  completeCandidateRoundTrip(true, true);
  completeCandidateRoundTrip(true, true, true);
  completeMemoryCandidateRoundTrip(false);
  completeMemoryCandidateRoundTrip(true);
  completeMemoryCandidateRoundTrip(false, true);
  llvm::outs() << "spatial mapping artifact tests passed\n";
  return 0;
}

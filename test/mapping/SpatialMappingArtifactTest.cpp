#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "ADG/MemoryLibrary.h"
#include "TechMappingCandidateTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryServiceContract.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialCandidateInitializer.h"
#include "PnR/SpatialCanonicalSeed.h"
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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iterator>
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

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::mapping::MappingDialect,
                  ::fabric::FabricDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

loom::fabric::FinalizedFabricRoot
buildTagBoundaryFabric(mlir::MLIRContext &context,
                       const loom::ArtifactStore &store) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  fabric.module @tag_boundaries(
      %dynamic_data: !fabric.bits<32>,
      %dynamic_tag: !fabric.bits<4>,
      %configured_data: !fabric.bits<16>,
      %rewrite_input: !fabric.bits_tag<8, 3>,
      %remove_input: !fabric.bits_tag<64, 5>)
      -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<16, 6>,
          !fabric.bits_tag<8, 7>, !fabric.bits<64>, !fabric.bits<5>) {
    %dynamic = fabric.boundary [s2t] %dynamic_data, %dynamic_tag
        : (!fabric.bits<32>, !fabric.bits<4>)
       -> !fabric.bits_tag<32, 4>
    %queued = fabric.fifo %dynamic [max_depth = 2, bypassable = false]
        : !fabric.bits_tag<32, 4>
    %configured = fabric.boundary [s2t] %configured_data
        : !fabric.bits<16> -> !fabric.bits_tag<16, 6>
    %rewritten = fabric.boundary [t2t] %rewrite_input
        {hw_params = [{lut_size = 5 : i32}]}
        : !fabric.bits_tag<8, 3> -> !fabric.bits_tag<8, 7>
    %removed:2 = fabric.boundary [t2s] %remove_input
        : !fabric.bits_tag<64, 5> -> (!fabric.bits<64>, !fabric.bits<5>)
    fabric.yield %queued, %configured, %rewritten, %removed#0, %removed#1
        : !fabric.bits_tag<32, 4>, !fabric.bits_tag<16, 6>,
          !fabric.bits_tag<8, 7>, !fabric.bits<64>, !fabric.bits<5>
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse tag-continuity Fabric fixture");
  auto roots = module->getOps<::fabric::ModuleOp>();
  if (std::distance(roots.begin(), roots.end()) != 1)
    fail("tag-continuity Fabric fixture does not have one root");
  return take(loom::fabric::finalizeFabricRoot(*roots.begin(), store));
}

void frozenTagContinuityIndexIsOwnerNormalized() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  const auto fabric = buildTagBoundaryFabric(context, store);
  const auto index =
      take(loom::pnr::freezeSpatialTagContinuityIndex(fabric.view()));

  const auto points = index.points();
  const auto boundaries = fabric.view().boundaryOccurrences();
  if (points.size() != 4 || points.size() != boundaries.size())
    fail("frozen tag-continuity index lost a boundary point");
  for (auto [ordinal, point] : llvm::enumerate(points))
    if (point.reference != boundaries[ordinal])
      fail("frozen tag-continuity index changed canonical boundary order");

  const auto traversals = fabric.view().physicalTraversals();
  const auto traversalPoints = index.traversalPointOrdinals();
  if (traversalPoints.size() != traversals.size())
    fail("frozen tag-continuity index is not traversal-dense");
  std::vector<std::uint32_t> pointUseCount(points.size(), 0);
  bool observedNonBoundary = false;
  for (auto [ordinal, traversal] : llvm::enumerate(traversals)) {
    const auto point = traversalPoints[ordinal];
    if (traversal.reference.kind() !=
        loom::fabric::FabricPhysicalTraversalKind::BoundaryTraversal) {
      observedNonBoundary = true;
      if (point != loom::pnr::getInvalidPnrIndex())
        fail("non-boundary traversal acquired a tag-continuity point");
      continue;
    }
    if (point >= points.size())
      fail("boundary traversal has no tag-continuity point");
    const auto &owner = std::get<loom::fabric::FabricBoundaryTraversalPayload>(
                            traversal.reference.payload)
                            .owner;
    if (points[point].reference != owner)
      fail("boundary traversal was assigned to a foreign continuity point");
    ++pointUseCount[point];
  }
  if (!observedNonBoundary)
    fail("tag-continuity fixture has no ordinary physical traversal");

  bool observedSplitRemover = false;
  for (auto [ordinal, point] : llvm::enumerate(points))
    if (point.kind == loom::fabric::FabricBoundaryTagContinuityKind::Remover) {
      observedSplitRemover = pointUseCount[ordinal] == 2;
      if (point.inputTagWidthBits != 5 || point.outputTagWidthBits != 0)
        fail("frozen remover changed its exact tag widths");
    }
  if (!observedSplitRemover)
    fail("split remover did not share one tag-continuity point");
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
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildMemoryDataflow(mlir::MLIRContext &context) {
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
  auto access = take(::fabric::MemoryAccessClass::create(
      ::dataflow::semantics::MemoryAccessForm::Element, singleton(32),
      singleton(1),
      {{::dataflow::semantics::MemoryMaskForm::Absent,
        ::fabric::InactiveLaneSemantics::NotApplicable}},
      std::move(alignment), std::move(read), std::move(write)));
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

void addTokenSyncFu(loom::adg::PeBuilder &pe,
                    llvm::ArrayRef<loom::adg::PeValue> inputs,
                    const loom::adg::PortType &type) {
  using loom::adg::FuCapabilityTemplateSpec;
  using loom::adg::FuSpec;
  using loom::adg::OperationCapabilitySpec;

  const std::vector<loom::adg::PortType> types(4, type);
  auto fu = take(pe.addFu(inputs, FuSpec{types, types}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    fuInputs.push_back(take(fu.input(ordinal)));
  auto operation = take(fu.addOperation(
      fuInputs, OperationCapabilitySpec{
                    ::fabric::ImplementationFamilyId::TokenSync,
                    ::fabric::RoutedTokenParams{128, 4},
                    {::dataflow::OperationSchemaId::DataflowSync},
                    types,
                    ::fabric::oneCycleElasticOperationResourceContract()}));
  requireSuccess(
      fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}));
  std::vector<loom::adg::FuValue> outputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    outputs.push_back(take(operation.output(ordinal)));
  requireSuccess(fu.close(outputs));
}

loom::fabric::FinalizedFabricRoot buildFabric(loom::ArtifactStore &store) {
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
  addTokenSyncFu(pe, peInputs, bits128);
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
  memoryOutputs.insert(memoryOutputs.end(), storageOutputs.begin(),
                       storageOutputs.end());
  requireSuccess(spatial.close(memoryOutputs));
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

std::string byteList(llvm::ArrayRef<std::uint8_t> bytes) {
  std::string text = "[";
  for (auto [ordinal, byte] : llvm::enumerate(bytes)) {
    if (ordinal)
      text += ", ";
    text += std::to_string(static_cast<std::int8_t>(byte));
  }
  return text + "]";
}

std::string identityAttr(const loom::ArtifactIdentity &identity) {
  return "#mapping.artifact_identity<" + byteList(identity.bytes()) + ">";
}

loom::mapping::FinalizedSpatialMappingConstraintSet
buildConstraints(mlir::MLIRContext &context,
                 const dataflow::CanonicalDataflowProgramView &dataflow,
                 const loom::mapping::TechMappingView &tech,
                 const loom::fabric::FabricArtifactView &fabric,
                 const loom::ArtifactStore &store) {
  const std::string text = "module {\n  mapping.constraints.spatial dataflow(" +
                           identityAttr(dataflow.identity()) +
                           ") tech_mapping(" + identityAttr(tech.identity()) +
                           ") fabric(" + identityAttr(fabric.identity()) +
                           ") {\n  }\n}\n";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    fail("cannot parse empty MappingConstraintSet fixture");
  auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
  return take(loom::mapping::finalizeSpatialMappingConstraintSet(
      *roots.begin(), dataflow, tech, fabric, store));
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

void selectReachableGraphBoundaries(loom::pnr::SpatialCandidateState &candidate,
                                    loom::pnr::SpatialMoveTransaction &move) {
  const auto &problem = candidate.problem();
  const auto reachable = [&](loom::pnr::PnrIndex source,
                             loom::pnr::PnrIndex destination) {
    const auto &routing = problem.routing();
    std::vector<std::uint8_t> visited(routing.routingEndpoints().size(), 0);
    std::vector<loom::pnr::PnrIndex> worklist{source};
    visited[source] = 1;
    for (std::size_t cursor = 0; cursor < worklist.size(); ++cursor) {
      const auto current = worklist[cursor];
      if (current == destination)
        return true;
      const auto offsets = routing.adjacencyOffsets();
      for (loom::pnr::PnrIndex arc = offsets[current];
           arc != offsets[current + 1]; ++arc) {
        const auto next = routing.routingArcs()[arc].target;
        if (!visited[next]) {
          visited[next] = 1;
          worklist.push_back(next);
        }
      }
    }
    return false;
  };
  const auto selectedEndpoint =
      [&](loom::pnr::FrozenSpatialTerminalBinding binding,
          std::optional<std::pair<loom::pnr::PnrIndex, loom::pnr::PnrIndex>>
              override) {
        loom::pnr::PnrIndex option = 0;
        if (binding.kind ==
            loom::pnr::FrozenSpatialTerminalBindingKind::PortDemand) {
          option = candidate.portAttachment(binding.index);
        } else if (override && override->first == binding.index) {
          option = override->second;
        } else {
          option = candidate.graphBoundaryAttachment(binding.index);
        }
        return problem.ports().attachmentOptions()[option].endpoint;
      };
  for (auto [boundaryOrdinal, boundary] :
       llvm::enumerate(problem.ports().graphBoundaries())) {
    const auto netOrdinal = boundary.logicalNet;
    const auto &net = problem.transfers().logicalNets()[netOrdinal];
    bool selected = false;
    for (loom::pnr::PnrIndex option = boundary.attachmentOptionOffset;
         option !=
         boundary.attachmentOptionOffset + boundary.attachmentOptionCount;
         ++option) {
      const auto override = std::make_pair(
          static_cast<loom::pnr::PnrIndex>(boundaryOrdinal), option);
      const auto source = selectedEndpoint(
          problem.transfers().logicalNetSourceBindings()[netOrdinal], override);
      bool connects = true;
      for (const auto sink : problem.transfers().logicalNetSinkBindings().slice(
               net.sinkOffset, net.sinkCount))
        connects &= reachable(source, selectedEndpoint(sink, override));
      if (!connects)
        continue;
      requireSuccess(move.setGraphBoundaryAttachment(
          static_cast<loom::pnr::PnrIndex>(boundaryOrdinal), option));
      selected = true;
      break;
    }
    if (!selected)
      fail("graph boundary has no reachable attachment");
  }
}

void selectLegalTemporalBinding(loom::pnr::SpatialCandidateState &candidate,
                                loom::pnr::SpatialCandidateScratch &scratch) {
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
  if (candidate.computeBinding(0).placement == legal->placement &&
      candidate.computeBinding(0).instructionContext ==
          legal->instructionContext)
    return;

  auto move = take(candidate.beginMove(scratch));
  requireSuccess(
      move.setComputeBinding(0, legal->placement, legal->instructionContext));
  for (auto [demandOrdinal, demand] :
       llvm::enumerate(problem.ports().portDemands())) {
    const auto &domain =
        problem.ports()
            .placementDomains()[demand.placementDomainOffset +
                                legal->placement - realization.placementOffset];
    requireSuccess(
        move.setPortAttachment(static_cast<loom::pnr::PnrIndex>(demandOrdinal),
                               domain.attachmentOptionOffset));
  }
  selectReachableGraphBoundaries(candidate, move);
  if (!take(move.close()))
    fail("legal Temporal binding closes a selected handshake cycle");
  requireSuccess(move.commit());
}

void completeCandidateRoundTrip(bool temporal) {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric =
      temporal ? buildTemporalFabric(store) : buildFabric(store);

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
  const auto constraints =
      buildConstraints(context, dataflow, tech.view(), fabric.view(), store);
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
    loom::pnr::SpatialCandidateScratch candidateScratch;
    requireSuccess(candidateScratch.prepare(*problem));
    selectLegalTemporalBinding(*candidate, candidateScratch);
    auto costs = take(loom::pnr::SpatialRouteCostState::create(*candidate));
    loom::pnr::SpatialPathFinderRouterScratch router;
    requireSuccess(router.prepare(*problem));
    take(router.routeToClosure(
        *candidate, candidateScratch, costs,
        {pnrConfig.policy().search.routing.endpointExpansionLimit,
         pnrConfig.policy().search.routing.negotiationIterationLimit},
        {}));
  } else {
    auto first = take(loom::pnr::createCanonicalPathFinderSpatialSeed(problem));
    auto second =
        take(loom::pnr::createCanonicalPathFinderSpatialSeed(problem));
    if (first.routing.completedIterations !=
            second.routing.completedIterations ||
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
  requireSuccess(candidate->verify());

  auto finalized = take(loom::pnr::finalizeSpatialMappingCandidate(
      *candidate, dataflow, tech.view(), fabric.view(), store));
  auto imported =
      take(loom::mapping::importSpatialMapping(finalized.reference(), store));
  if (imported.reference() != finalized.reference() ||
      imported.view().computeBindings().size() != 1 ||
      imported.view().routeTrees().empty() ||
      imported.view().resourceUses().empty())
    fail("strict SpatialMapping round trip lost selected closure");

  if (temporal) {
    bool observedEnqueue = false;
    bool observedTransition = false;
    for (const auto &use : imported.view().resourceUses()) {
      observedEnqueue |=
          std::holds_alternative<dataflow::CanonicalGraphConsumerEndpointRef>(
              use.activation.trigger.event);
      observedTransition |=
          std::holds_alternative<loom::mapping::SpatialActorTransitionEventRef>(
              use.activation.trigger.event);
    }
    if (!observedEnqueue || !observedTransition)
      fail("Temporal SpatialMapping round trip lost queue or operation uses");
  }

  auto missingRoute = parseSpatial(context, finalized.canonicalBytes());
  if (!missingRoute)
    fail("cannot parse finalized SpatialMapping fixture");
  auto root = *missingRoute->getOps<::mapping::SpatialOp>().begin();
  auto routes = root.getBody().front().getOps<::mapping::RouteTreeOp>();
  (*routes.begin()).erase();
  if (!rejected(loom::mapping::finalizeSpatialMapping(root, store)))
    fail("SpatialMapping finalized without a required RouteTree");

  auto missingUse = parseSpatial(context, finalized.canonicalBytes());
  if (!missingUse)
    fail("cannot reparse finalized SpatialMapping fixture");
  auto useRoot = *missingUse->getOps<::mapping::SpatialOp>().begin();
  auto uses = useRoot.getBody().front().getOps<::mapping::ResourceUseOp>();
  (*uses.begin()).erase();
  if (!rejected(loom::mapping::finalizeSpatialMapping(useRoot, store)))
    fail("SpatialMapping finalized without a required ResourceUse");
}

void completeMemoryCandidateRoundTrip(bool temporal) {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildMemoryDataflow(context);
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

  const auto constraints =
      buildConstraints(context, dataflow, tech.view(), fabric.view(), store);
  const auto pnrConfig = take(loom::pnr::projectResolvedSpatialPnrConfigView(
      loom::test::buildSpatialPnrTestResolvedConfig()));
  auto problem = take(loom::pnr::freezeSpatialPnrProblem(
      dataflow, tech.view(), fabric.view(), pnrConfig, constraints.view()));
  auto candidate = take(loom::pnr::createCanonicalSpatialCandidate(problem));
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
    selectReachableGraphBoundaries(*candidate, move);
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
    if (temporal && planRecord.residentContext != 1)
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
         pnrConfig.policy().search.routing.negotiationIterationLimit},
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
      *candidate, dataflow, tech.view(), fabric.view(), store));
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
  const auto &operation =
      std::get<loom::mapping::SpatialAddressedMemoryOperationView>(
          engine.operations.front());
  bool hasOperationPortUse = false;
  bool hasLocalServiceUse = false;
  for (const auto &use : imported.view().resourceUses()) {
    hasOperationPortUse |= std::holds_alternative<
        loom::mapping::SpatialMemoryEngineResourceOwnerRef>(use.owner);
    hasLocalServiceUse |= std::holds_alternative<
        loom::mapping::SpatialMemoryBindingResourceOwnerRef>(use.owner);
  }
  if (imported.view().resourceUses().size() != 2 || !hasOperationPortUse ||
      !hasLocalServiceUse)
    fail("strict SpatialMapping round trip lost a memory ResourceUse");
  if (temporal) {
    const auto *context =
        std::get_if<loom::fabric::FabricMemoryOperationContextRef>(
            &operation.placement);
    if (!context || context->ordinal != 1)
      fail("Temporal memory placement lost its selected resident context");
  } else if (!std::holds_alternative<
                 loom::fabric::FabricMemoryOperationPortRef>(
                 operation.placement)) {
    fail("Spatial memory placement gained a resident context");
  }

  auto missingUse = parseSpatial(context, finalized.canonicalBytes());
  if (!missingUse)
    fail("cannot reparse memory ResourceUse fixture");
  auto missingUseRoot = *missingUse->getOps<::mapping::SpatialOp>().begin();
  auto resourceUses =
      missingUseRoot.getBody().front().getOps<::mapping::ResourceUseOp>();
  if (resourceUses.empty())
    fail("memory SpatialMapping fixture has no ResourceUse to remove");
  (*resourceUses.begin()).erase();
  if (!rejected(loom::mapping::finalizeSpatialMapping(missingUseRoot, store)))
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
  if (!rejected(
          loom::mapping::finalizeSpatialMapping(missingExposureRoot, store)))
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
  if (!rejected(
          loom::mapping::finalizeSpatialMapping(wrongTerminalRoot, store)))
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
    if (!rejected(loom::mapping::finalizeSpatialMapping(root, store)))
      fail("SpatialMapping accepted overlapping local physical intervals");
  }
}

} // namespace

int main() {
  frozenTagContinuityIndexIsOwnerNormalized();
  completeCandidateRoundTrip(false);
  completeCandidateRoundTrip(true);
  completeMemoryCandidateRoundTrip(false);
  completeMemoryCandidateRoundTrip(true);
  llvm::outs() << "spatial mapping artifact tests passed\n";
  return 0;
}

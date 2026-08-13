#include "ADG/Builder.h"
#include "ADG/MemoryLibrary.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"
#include "PnR/PnrConfig.h"
#include "PnR/SpatialPnrGenerator.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "tech mapping joint legality test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-tech-mapping-joint-legality", path))
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
  registry.insert<::dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::memref::MemRefDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact
buildTwoLoadDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @two_loads(
      %start: none, %first_index: index, %second_index: index,
      %memory: memref<4xi32>) -> (i32, i32)
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 2, 0, 0>} {
    %first, %first_done =
        dataflow.load %memory[%first_index] %start : memref<4xi32>
    %second, %second_done =
        dataflow.load %memory[%second_index] %first_done : memref<4xi32>
    dataflow.graph.return values(%first, %second : i32, i32) streams()
        memories() complete(%second_done : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse two-load Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildBroadcastDataflow(mlir::MLIRContext &context, bool sameProducer) {
  const llvm::StringRef text = sameProducer ? R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @same_producer(
      %start: none, %lhs: i32, %rhs: i32) -> (i32, i32)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %first = arith.addi %lhs, %rhs : i32
    %second = arith.addi %lhs, %rhs : i32
    %retired:3 = dataflow.sync %start, %first, %second
        : (none, i32, i32) -> (none, i32, i32)
    dataflow.graph.return values(%retired#1, %retired#2 : i32, i32) streams()
        memories() complete(%retired#0 : none)
  }
}
)mlir"
                                            : R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @distinct_producers(
      %start: none, %first_lhs: i32, %second_lhs: i32, %rhs: i32)
      -> (i32, i32)
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %first = arith.addi %first_lhs, %rhs : i32
    %second = arith.addi %second_lhs, %rhs : i32
    %retired:3 = dataflow.sync %start, %first, %second
        : (none, i32, i32) -> (none, i32, i32)
    dataflow.graph.return values(%retired#1, %retired#2 : i32, i32) streams()
        memories() complete(%retired#0 : none)
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    fail("cannot parse FU-broadcast Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

loom::fabric::FinalizedFabricRoot buildMemoryFabric(loom::ArtifactStore &store,
                                                    bool temporal) {
  loom::adg::LocalMemoryParameters parameters;
  parameters.capacityBytes = 4096;
  parameters.interface = {
      loom::adg::MemoryAccessDomainParameters{
          128, 128, 16,
          take(fabric::UnsignedDomain::fromCanonical({{64, 64}}))},
      128, 128};
  if (temporal)
    parameters.temporal = loom::adg::TemporalMemoryParameters{4, 1};
  auto memory = take(loom::adg::makeGeneral64LocalMemory(parameters));
  const std::vector<loom::adg::PortType> inputs(memory.inputTypes().begin(),
                                                memory.inputTypes().end());
  const std::vector<loom::adg::PortType> outputs(memory.outputTypes().begin(),
                                                 memory.outputTypes().end());
  loom::adg::DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore(
      temporal ? "temporal-memory" : "spatial-memory", inputs, outputs));
  std::vector<loom::adg::SpatialValue> values;
  for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal)
    values.push_back(take(spatial.input(ordinal)));
  auto memoryOutputs = take(spatial.addMemory(values, memory));
  if (llvm::Error error = spatial.close(memoryOutputs.values()))
    fail(llvm::toString(std::move(error)));
  auto design = take(std::move(builder).finalize());
  return design.roots().front();
}

loom::adg::OperationCapabilitySpec
integerAddCapability(const loom::adg::PortType &bits32) {
  return loom::adg::OperationCapabilitySpec{
      ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
      ::fabric::ScalarIntegerParams{
          ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I32})},
      {::dataflow::OperationSchemaId::ArithAddI},
      {bits32},
      ::fabric::oneCycleElasticOperationResourceContract()};
}

void addTokenSyncFu(loom::adg::PeBuilder &pe,
                    llvm::ArrayRef<loom::adg::PeValue> inputs,
                    const loom::adg::PortType &bits128) {
  const std::vector<loom::adg::PortType> types(4, bits128);
  auto fu = take(pe.addFu(inputs, loom::adg::FuSpec{types, types}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    fuInputs.push_back(take(fu.input(ordinal)));
  auto operation = take(fu.addOperation(
      fuInputs, loom::adg::OperationCapabilitySpec{
                    ::fabric::ImplementationFamilyId::TokenSync,
                    ::fabric::RoutedTokenParams{128, 4},
                    {::dataflow::OperationSchemaId::DataflowSync},
                    types,
                    ::fabric::oneCycleElasticOperationResourceContract()}));
  if (llvm::Error error = fu.addCapabilityTemplate(
          loom::adg::FuCapabilityTemplateSpec{{operation}, {}}))
    fail(llvm::toString(std::move(error)));
  std::vector<loom::adg::FuValue> outputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    outputs.push_back(take(operation.output(ordinal)));
  if (llvm::Error error = fu.close(outputs))
    fail(llvm::toString(std::move(error)));
}

loom::fabric::FinalizedFabricRoot
buildBroadcastFabric(loom::ArtifactStore &store) {
  using loom::adg::DesignBuilder;
  using loom::adg::FuCapabilityTemplateSpec;
  using loom::adg::FuSpec;
  using loom::adg::PeSpec;
  using loom::adg::PortType;

  const PortType bits32 = take(PortType::bits(32));
  const PortType bits128 = take(PortType::bits(128));
  const std::vector<PortType> inputs(4, bits128);
  const std::vector<PortType> outputs(4, bits128);
  DesignBuilder builder(store);
  auto spatial =
      take(builder.createSpatialCore("broadcast-fu", inputs, outputs));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe =
      take(spatial.addPe(spatialInputs, PeSpec::spatial(inputs, outputs)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  const std::vector<PortType> fuInputs(2, bits32);
  const std::vector<PortType> fuOutputs(2, bits128);
  auto fu =
      take(pe.addFu({peInputs[0], peInputs[1]}, FuSpec{fuInputs, fuOutputs}));
  const auto first = take(fu.addOperation(
      {take(fu.input(0)), take(fu.input(1))}, integerAddCapability(bits32)));
  const auto second = take(fu.addOperation(
      {take(fu.input(0)), take(fu.input(1))}, integerAddCapability(bits32)));
  if (llvm::Error error = fu.addCapabilityTemplate(
          FuCapabilityTemplateSpec{{first, second}, {}}))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          fu.close({take(first.output(0)), take(second.output(0))}))
    fail(llvm::toString(std::move(error)));
  addTokenSyncFu(pe, peInputs, bits128);
  if (llvm::Error error = pe.close())
    fail(llvm::toString(std::move(error)));
  std::vector<loom::adg::SpatialValue> spatialOutputs;
  for (std::size_t ordinal = 0; ordinal < outputs.size(); ++ordinal)
    spatialOutputs.push_back(take(pe.output(ordinal)));
  if (llvm::Error error = spatial.close(spatialOutputs))
    fail(llvm::toString(std::move(error)));
  auto design = take(std::move(builder).finalize());
  return design.roots().front();
}

loom::mapping::TechMappingGenerationOutcome
generate(dataflow::CanonicalDataflowArtifact &artifact,
         const loom::fabric::FinalizedFabricRoot &fabric,
         loom::ArtifactStore &store) {
  take(dataflow::publishCanonicalDataflow(artifact, store));
  const auto dataflow = take(artifact.view());
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 64;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  return loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), config, store});
}

bool hasRealizationWithActorCount(
    const loom::mapping::TechMappingGenerationOutcome &outcome,
    const loom::ArtifactStore &store, std::size_t actorCount, bool memory) {
  const auto *generated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&outcome);
  if (!generated)
    return false;
  for (const auto &reference : generated->candidates) {
    const auto candidate =
        take(loom::mapping::importTechMapping(reference, store));
    if (memory && llvm::any_of(candidate.view().memoryRealizations(),
                               [&](const auto &row) {
                                 return row.actors.size() == actorCount;
                               }))
      return true;
    if (!memory && llvm::any_of(candidate.view().computeRealizations(),
                                [&](const auto &row) {
                                  return row.actors.size() == actorCount;
                                }))
      return true;
  }
  return false;
}

void memoryScheduleOwnsJointAdmission() {
  for (bool temporal : {true, false}) {
    TemporaryDirectory directory;
    loom::ArtifactStore store(directory.path());
    mlir::MLIRContext context = makeContext();
    auto dataflow = buildTwoLoadDataflow(context);
    const auto fabric = buildMemoryFabric(store, temporal);
    const auto outcome = generate(dataflow, fabric, store);
    if (hasRealizationWithActorCount(outcome, store, 2, true))
      fail(temporal ? "Temporal K=1 admitted two memory actors"
                    : "Spatial memory port admitted two software operations");
  }
}

void spatialMemoryRowsAreOccurrenceUnique() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildTwoLoadDataflow(context);
  const auto fabric = buildMemoryFabric(store, false);
  const auto outcome = generate(dataflowArtifact, fabric, store);
  const auto *generated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&outcome);
  if (!generated)
    fail("Spatial memory fixture produced no TechMapping candidates");

  std::optional<loom::mapping::FinalizedTechMapping> selected;
  for (const auto &reference : generated->candidates) {
    auto candidate = take(loom::mapping::importTechMapping(reference, store));
    const auto realizations = candidate.view().memoryRealizations();
    if (realizations.size() != 2 ||
        !llvm::all_of(realizations,
                      [](const auto &row) { return row.actors.size() == 1; }))
      continue;
    selected.emplace(std::move(candidate));
    break;
  }
  if (!selected)
    fail("Spatial memory fixture has no two-row TechMapping candidate");

  const auto dataflow = take(dataflowArtifact.view());
  auto constraints =
      take(loom::mapping::finalizeEmptySpatialMappingConstraintSet(
          dataflow, selected->view(), fabric.view(), store));
  const auto config = take(loom::pnr::projectResolvedSpatialPnrConfigView(
      loom::defaultResolvedConfig()));
  auto spatial = loom::pnr::generateSpatialMappings(
      {dataflow, selected->view(), fabric.view(), config, constraints.view(),
       store, 1});
  const auto *infeasible =
      std::get_if<loom::pnr::ProvenInfeasibleSpatialMapping>(&spatial);
  if (!infeasible)
    fail("two static memory rows shared one physical configuration row");
  if (infeasible->accounting.endpointExpansionSlots != 0 ||
      infeasible->accounting.negotiationIterationSlots != 0)
    fail("static memory row conflict reached transport search");
}

void broadcastPreservesProducerIdentity() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  const auto fabric = buildBroadcastFabric(store);

  mlir::MLIRContext sameContext = makeContext();
  auto same = buildBroadcastDataflow(sameContext, true);
  const auto sameOutcome = generate(same, fabric, store);
  if (!hasRealizationWithActorCount(sameOutcome, store, 2, false))
    fail("same-producer FU broadcast was rejected");

  mlir::MLIRContext distinctContext = makeContext();
  auto distinct = buildBroadcastDataflow(distinctContext, false);
  const auto distinctOutcome = generate(distinct, fabric, store);
  if (hasRealizationWithActorCount(distinctOutcome, store, 2, false))
    fail("distinct Dataflow producers reused one FU boundary input");
}

} // namespace

int main() {
  memoryScheduleOwnsJointAdmission();
  spatialMemoryRowsAreOccurrenceUnique();
  broadcastPreservesProducerIdentity();
  llvm::outs() << "tech mapping joint legality tests passed\n";
  return 0;
}

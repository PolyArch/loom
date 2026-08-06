#include "ADG/Builder.h"
#include "ADG/Builtin.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/MemoryActorContractDomain.h"
#include "Fabric/IR/MemoryCapabilityDomains.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryOperationPort.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "Mapping/Tech/TechMappingGenerator.h"

#include "SpatialMemoryConstraintTestSupport.h"

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

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "tech mapping generator test: " << message << '\n';
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
            "loom-tech-mapping-generator", path))
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
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

using CandidateKey = std::vector<std::vector<std::uint8_t>>;

CandidateKey
candidateKey(const loom::ArtifactRootReference &reference,
             const dataflow::CanonicalDataflowProgramView &dataflow,
             const loom::ArtifactStore &store) {
  const auto candidate =
      take(loom::mapping::importTechMapping(reference, store));
  CandidateKey key;
  for (const auto &row : candidate.view().computeRealizations())
    key.push_back(take(
        loom::mapping::canonicalTechMatchRowKey(row, dataflow.identity())));
  for (const auto &row : candidate.view().memoryRealizations())
    key.push_back(take(
        loom::mapping::canonicalTechMatchRowKey(row, dataflow.identity())));
  llvm::sort(key);
  return key;
}

dataflow::CanonicalDataflowArtifact
buildMixedDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @load_then_sync(
      %start: none, %index: index, %memory: memref<4xi32>) -> i32
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value, %done = dataflow.load %memory[%index] %start : memref<4xi32>
    %synced:2 = dataflow.sync %done, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%synced#1 : i32) streams() memories()
        complete(%synced#0 : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse mixed Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildSingleSyncDataflow(mlir::MLIRContext &context) {
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
    fail("cannot parse single-sync Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildSyncWithDeadResultDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync_dead_result(%start: none, %value: i32) -> ()
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values() streams() memories()
        complete(%result#0 : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse dead-result sync Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildIntegerAddChainDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @add_chain(
      %start: none, %a: i32, %b: i32, %c: i32, %d: i32) -> i32
      attributes {input_segments = array<i32: 4, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %ab = arith.addi %a, %b : i32
    %abc = arith.addi %ab, %c : i32
    %abcd = arith.addi %abc, %d : i32
    %result:2 = dataflow.sync %start, %abcd
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse integer-add-chain Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildInterleavedSyncDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync_pair(
      %start: none, %a: i32, %b: i32, %c: i32) -> (i32, i32, i32)
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 3, 0, 0>} {
    %narrow = dataflow.sync %start : (none) -> none
    %wide:4 = dataflow.sync %narrow, %a, %b, %c
        : (none, i32, i32, i32) -> (none, i32, i32, i32)
    dataflow.graph.return values(%wide#1, %wide#2, %wide#3
        : i32, i32, i32) streams() memories() complete(%wide#0 : none)
  }

  dataflow.graph private @sync_single(%start: none, %value: i32) -> i32
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
    fail("cannot parse interleaved-sync Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildIndependentIntegerAddsDataflow(mlir::MLIRContext &context,
                                    std::size_t actorCount) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module attributes {dlti.dl_spec = "
            "#dlti.dl_spec<#dlti.dl_entry<index, 64>>} {\n"
            "  dataflow.graph private @independent_adds(\n"
            "      %start: none, %lhs: i32, %rhs: i32) -> (";
  for (std::size_t actor = 0; actor < actorCount; ++actor) {
    if (actor != 0)
      stream << ", ";
    stream << "i32";
  }
  stream << ") attributes {input_segments = array<i32: 2, 0, 0>, "
            "result_segments = array<i32: "
         << actorCount << ", 0, 0>} {\n";
  for (std::size_t actor = 0; actor < actorCount; ++actor)
    stream << "    %sum" << actor << " = arith.addi %lhs, %rhs : i32\n";
  for (std::size_t actor = 0; actor < actorCount; ++actor)
    stream << "    %retire" << actor << ":2 = dataflow.sync %"
           << (actor == 0 ? std::string("start")
                          : "retire" + std::to_string(actor - 1) + "#0")
           << ", %sum" << actor << " : (none, i32) -> (none, i32)\n";
  stream << "    dataflow.graph.return values(";
  for (std::size_t actor = 0; actor < actorCount; ++actor) {
    if (actor != 0)
      stream << ", ";
    stream << "%retire" << actor << "#1";
  }
  stream << " : ";
  for (std::size_t actor = 0; actor < actorCount; ++actor) {
    if (actor != 0)
      stream << ", ";
    stream << "i32";
  }
  stream << ") streams() memories() complete(%retire" << actorCount - 1
         << "#0 : none)\n  }\n}\n";
  stream.flush();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module)
    fail("cannot parse independent-integer-add Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildMemoryChainDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @load_then_store(
      %start: none, %load_index: index, %store_index: index,
      %load_memory: memref<4xi32>, %store_memory: memref<4xi32>) -> ()
      attributes {input_segments = array<i32: 2, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %value, %load_done =
        dataflow.load %load_memory[%load_index] %start : memref<4xi32>
    %store_done = dataflow.store %store_memory[%store_index] %value %load_done
        : memref<4xi32>
    dataflow.graph.return values() streams() memories()
        complete(%store_done : none)
  }
  dataflow.thread private @memory_worker
      domain(#dataflow.thread_domain<dense>)(
          %load_index: index, %store_index: index,
          %load_memory: memref<4xi32>, %store_memory: memref<4xi32>)
      ctrl (%ctrl: none) {
    %done = dataflow.graph.launch @load_then_store deps(%ctrl)
        values(%load_index, %store_index) stream_inputs()
        memories(%load_memory, %store_memory)
        stream_outputs()
        : (none, index, index, memref<4xi32>, memref<4xi32>) -> none
    dataflow.thread.yield %done : none
  }
  func.func private @memory_host(
      %load_index: index, %store_index: index,
      %load_memory: memref<4xi32>, %store_memory: memref<4xi32>) {
    %token = dataflow.thread.launch @memory_worker(
        %load_index, %store_index, %load_memory, %store_memory)
        : (index, index, memref<4xi32>, memref<4xi32>)
          -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse memory-chain Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildUnsupportedDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @wide_add(
      %start: none, %lhs: i128, %rhs: i128) -> i128
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = arith.addi %lhs, %rhs : i128
    %result:2 = dataflow.sync %start, %sum
        : (none, i128) -> (none, i128)
    dataflow.graph.return values(%result#1 : i128) streams() memories()
        complete(%result#0 : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse unsupported Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildUnsupportedMemoryDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @wide_load(
      %start: none, %index: index, %memory: memref<4xi64>) -> i64
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value, %done = dataflow.load %memory[%index] %start : memref<4xi64>
    dataflow.graph.return values(%value : i64) streams() memories()
        complete(%done : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse unsupported memory Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildSupportedAndUnsupportedDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @supported(
      %start: none, %lhs: i32, %rhs: i32) -> i32
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = arith.addi %lhs, %rhs : i32
    %result:2 = dataflow.sync %start, %sum
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }

  dataflow.graph private @unsupported(
      %start: none, %lhs: i128, %rhs: i128) -> i128
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %sum = arith.addi %lhs, %rhs : i128
    %result:2 = dataflow.sync %start, %sum
        : (none, i128) -> (none, i128)
    dataflow.graph.return values(%result#1 : i128) streams() memories()
        complete(%result#0 : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse mixed-feasibility Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

loom::fabric::FinalizedFabricRoot buildSmallFabric(loom::ArtifactStore &store) {
  loom::adg::DesignBuilder builder(store);
  auto expansion = take(loom::adg::expandBuiltinSpatialCore(
      builder, loom::adg::BuiltinTargetPreset::Small));
  if (llvm::Error error = expansion.spatialCore.close(expansion.outputs))
    fail(llvm::toString(std::move(error)));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("builtin SpatialCore did not publish exactly one Fabric root");
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
  using loom::adg::FuCapabilityTemplateSpec;
  using loom::adg::FuSpec;
  using loom::adg::OperationCapabilitySpec;

  const std::vector<loom::adg::PortType> types(4, bits128);
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
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(llvm::toString(std::move(error)));
  std::vector<loom::adg::FuValue> outputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    outputs.push_back(take(operation.output(ordinal)));
  if (llvm::Error error = fu.close(outputs))
    fail(llvm::toString(std::move(error)));
}

loom::fabric::FinalizedFabricRoot
buildSerialIntegerAddFabric(loom::ArtifactStore &store) {
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
      take(builder.createSpatialCore("serial-integer-add", inputs, outputs));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe =
      take(spatial.addPe(spatialInputs, PeSpec::spatial(inputs, outputs)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  const std::vector<PortType> addInputs(4, bits32);
  const std::vector<PortType> addOutputs{bits128};
  auto fu = take(pe.addFu(peInputs, FuSpec{addInputs, addOutputs}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal < addInputs.size(); ++ordinal)
    fuInputs.push_back(take(fu.input(ordinal)));

  const auto capability = integerAddCapability(bits32);
  auto first = take(fu.addOperation({fuInputs[0], fuInputs[1]}, capability));
  auto second =
      take(fu.addOperation({take(first.output(0)), fuInputs[2]}, capability));
  auto third =
      take(fu.addOperation({take(second.output(0)), fuInputs[3]}, capability));
  if (llvm::Error error = fu.addCapabilityTemplate(
          FuCapabilityTemplateSpec{{first, second, third}, {}}))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({take(third.output(0))}))
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
  if (design.roots().size() != 1)
    fail("serial integer-add Fabric did not publish exactly one root");
  return design.roots().front();
}

loom::fabric::FinalizedFabricRoot
buildTokenSyncFabric(loom::ArtifactStore &store) {
  using loom::adg::DesignBuilder;
  using loom::adg::PeSpec;
  using loom::adg::PortType;

  const PortType bits128 = take(PortType::bits(128));
  const std::vector<PortType> types(4, bits128);
  DesignBuilder builder(store);
  auto spatial = take(builder.createSpatialCore("token-sync", types, types));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    spatialInputs.push_back(take(spatial.input(ordinal)));
  auto pe = take(spatial.addPe(spatialInputs, PeSpec::spatial(types, types)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    peInputs.push_back(take(pe.input(ordinal)));
  addTokenSyncFu(pe, peInputs, bits128);
  if (llvm::Error error = pe.close())
    fail(llvm::toString(std::move(error)));
  std::vector<loom::adg::SpatialValue> spatialOutputs;
  for (std::size_t ordinal = 0; ordinal < types.size(); ++ordinal)
    spatialOutputs.push_back(take(pe.output(ordinal)));
  if (llvm::Error error = spatial.close(spatialOutputs))
    fail(llvm::toString(std::move(error)));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("token-sync Fabric did not publish exactly one root");
  return design.roots().front();
}

fabric::UnsignedDomain singletonDomain(std::uint64_t value) {
  return take(fabric::UnsignedDomain::fromCanonical({{value, value}}));
}

fabric::ResourceContract memoryPortResourceContract() {
  fabric::ResourceContractDeclaration declaration;
  declaration.states = {fabric::ResourceStateDeclaration{
      fabric::StateKey(0),
      {{fabric::CapacityDimensionKey(0), fabric::CapacityUnits(1),
        fabric::CapacityUnits(0)}}}};
  declaration.requesters = {fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.timingContracts = {{fabric::TimingContractKey(0), {0, 1}}};
  declaration.usePatterns = {
      {fabric::UsePatternKey(0),
       fabric::RequesterKey(0),
       fabric::EligibilityKey(0),
       fabric::EventKey(0),
       fabric::EventKey(1),
       std::nullopt,
       fabric::TimingContractKey(0),
       {{fabric::ClaimKey(0), fabric::StateKey(0),
         fabric::CapacityDimensionKey(0), fabric::CapacityUnits(1)}},
       {{{fabric::ClaimKey(0)}}}}};
  return take(fabric::ResourceContract::create(std::move(declaration)));
}

fabric::MemoryOperationPortDeclaration memoryPort(bool reads) {
  auto alignment = take(fabric::AlignmentDomain::create(
      take(fabric::UnsignedDomain::fromCanonical({{0, 63}}))));
  auto read = take(
      fabric::ClosedEnumDomain<fabric::ReadSubwordSemantics>::fromCanonical(
          {reads ? fabric::ReadSubwordSemantics::ZeroExtend
                 : fabric::ReadSubwordSemantics::NotApplicable}));
  auto write = take(
      fabric::ClosedEnumDomain<fabric::WriteSubwordSemantics>::fromCanonical(
          {reads ? fabric::WriteSubwordSemantics::NotApplicable
                 : fabric::WriteSubwordSemantics::ByteEnable}));
  auto address = take(fabric::MemoryAddressDomain::rootRelative(
      singletonDomain(64)));
  auto access = take(fabric::MemoryAccessClass::create(
      dataflow::semantics::MemoryAccessForm::Element, singletonDomain(32),
      singletonDomain(1),
      {{dataflow::semantics::MemoryMaskForm::Absent,
        fabric::InactiveLaneSemantics::NotApplicable},
       {dataflow::semantics::MemoryMaskForm::Dynamic,
        reads ? fabric::InactiveLaneSemantics::SuppressAndZeroFill
              : fabric::InactiveLaneSemantics::Suppress}},
      std::move(alignment), std::move(read), std::move(write),
      std::move(address)));
  auto accessDomain = take(
      fabric::ParameterizedMemoryAccessDomain::create({std::move(access)}));
  fabric::MemoryActorContractClause plain =
      fabric::LoadStorePlainContractClause{{false}};
  auto actorDomain = take(fabric::MemoryActorContractDomain::create(
      reads ? dataflow::OperationSchemaId::DataflowLoad
            : dataflow::OperationSchemaId::DataflowStore,
      {plain}));

  fabric::MemoryCapabilityAlternativeRecord alternative{
      std::move(actorDomain),
      reads
          ? std::vector<
                fabric::MemoryRoleEndpointBindingRecord>{{dataflow::semantics::
                                                              ServiceValueRole::
                                                                  Address,
                                                          0},
                                                         {dataflow::semantics::
                                                              ServiceValueRole::
                                                                  Data,
                                                          7},
                                                         {dataflow::semantics::
                                                              ServiceValueRole::
                                                                  Mask,
                                                          1},
                                                         {dataflow::semantics::
                                                              ServiceValueRole::
                                                                  Control,
                                                          2},
                                                         {dataflow::semantics::
                                                              ServiceValueRole::
                                                                  Completion,
                                                          8}}
          : std::vector<
                fabric::MemoryRoleEndpointBindingRecord>{{dataflow::semantics::
                                                              ServiceValueRole::
                                                                  Address,
                                                          3},
                                                         {dataflow::semantics::
                                                              ServiceValueRole::
                                                                  Data,
                                                          4},
                                                         {dataflow::semantics::
                                                              ServiceValueRole::
                                                                  Mask,
                                                          5},
                                                         {dataflow::semantics::
                                                              ServiceValueRole::
                                                                  Control,
                                                          6},
                                                         {dataflow::semantics::
                                                              ServiceValueRole::
                                                                  Completion,
                                                          9}},
      std::move(accessDomain),
      {fabric::UsePatternKey(0)}};
  return {reads ? std::vector<std::uint64_t>{0, 1, 2, 7, 8}
                : std::vector<std::uint64_t>{3, 4, 5, 6, 9},
          memoryPortResourceContract(),
          {{fabric::MemoryPortTransactionProjection::Direct}},
          {std::move(alternative)}};
}

loom::fabric::FinalizedFabricRoot
buildInternalMemoryEdgeFabric(loom::ArtifactStore &store) {
  using loom::adg::MemoryConnectivitySpec;
  using loom::adg::MemoryEngineSpec;
  using loom::adg::MemorySpec;
  using loom::adg::PortType;

  const auto bits8 = take(PortType::bits(8));
  const auto manager =
      take(PortType::memory({PortType::kDynamicExtent}, bits8));
  std::vector<PortType> inputs{manager};
  for (std::uint32_t width : {64u, 4u, 0u, 64u, 128u, 4u, 0u})
    inputs.push_back(take(PortType::bits(width)));
  std::vector<PortType> outputs;
  for (std::uint32_t width : {128u, 0u, 0u})
    outputs.push_back(take(PortType::bits(width)));

  fabric::MemoryDispatchTarget managerTarget(
      std::in_place_type<fabric::ManagerMemoryDispatchTarget>,
      fabric::ManagerMemoryDispatchTarget{0});
  fabric::MemoryConnectivityDeclaration connectivity;
  fabric::MemoryOperationPortDispatchDeclaration readDispatch;
  readDispatch.capabilityTargetDomains = {{managerTarget}};
  fabric::MemoryOperationPortDispatchDeclaration writeDispatch;
  writeDispatch.capabilityTargetDomains = {{managerTarget}};
  connectivity.operationPorts = {std::move(readDispatch),
                                 std::move(writeDispatch)};
  connectivity.internalConnections = {{8, 6}};
  auto spec = take(MemorySpec::create(
      inputs, outputs, {0}, {},
      MemoryEngineSpec::spatial({memoryPort(true), memoryPort(false)}),
      std::nullopt,
      take(MemoryConnectivitySpec::create(std::move(connectivity)))));

  loom::adg::DesignBuilder builder(store);
  auto spatial =
      take(builder.createSpatialCore("memory-internal-edge", inputs, outputs));
  std::vector<loom::adg::SpatialValue> inputValues;
  for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal)
    inputValues.push_back(take(spatial.input(ordinal)));
  auto memoryOutputs = take(spatial.addMemory(inputValues, spec));
  if (llvm::Error error = spatial.close(memoryOutputs.values()))
    fail(llvm::toString(std::move(error)));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("internal-edge Fabric did not publish exactly one root");
  return design.roots().front();
}

void serialComputeTemplateAcceptsExactActorChain() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildIntegerAddChainDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildSerialIntegerAddFabric(store);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  const auto outcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), config, store});
  const auto *generated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&outcome);
  if (!generated || generated->candidates.size() != 1)
    fail("exact serial compute topology was pruned before realization");

  const auto candidate = take(
      loom::mapping::importTechMapping(generated->candidates.front(), store));
  if (!llvm::any_of(candidate.view().computeRealizations(),
                    [](const auto &realization) {
                      return realization.actors.size() == 3;
                    }))
    fail("serial compute realization did not retain its three actors");
}

void matchRowLimitPreservesGlobalActorOrder() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildInterleavedSyncDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildTokenSyncFabric(store);

  std::vector<std::uint64_t> pairActors;
  std::optional<std::uint64_t> singleActor;
  std::optional<dataflow::GraphRef> singleGraph;
  for (const auto &graph : dataflow.graphs()) {
    std::vector<std::uint64_t> actors;
    for (const auto &actor : dataflow.actors())
      if (actor.graph == graph.ref)
        actors.push_back(actor.ref.entity.value());
    if (actors.size() == 2)
      pairActors = std::move(actors);
    else if (actors.size() == 1) {
      singleActor = actors.front();
      singleGraph = graph.ref;
    }
  }
  llvm::sort(pairActors);
  if (pairActors.size() != 2 || !singleActor || !singleGraph ||
      !(pairActors.front() < *singleActor && *singleActor < pairActors.back()))
    fail("cross-graph ordering fixture does not interleave canonical actors: " +
         (pairActors.empty() ? std::string("none")
                             : std::to_string(pairActors.front())) +
         "," +
         (pairActors.size() < 2 ? std::string("none")
                                : std::to_string(pairActors.back())) +
         " versus " +
         (singleActor ? std::to_string(*singleActor) : std::string("none")));

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.matchRowAttemptLimit = 400;
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  std::vector<dataflow::GraphRef> covers;
  for (const auto &graph : dataflow.graphs())
    covers.push_back(graph.ref);
  const auto outcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), config, store});
  const auto *generated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&outcome);
  if (!generated || generated->candidates.size() != 1)
    fail("match-row prefix was enumerated by graph instead of actor key");

  const std::array<dataflow::GraphRef, 1> singleCover = {*singleGraph};
  const auto singleOutcome = loom::mapping::generateTechMappings(
      {dataflow, singleCover, fabric.view(), config, store});
  const auto *singleGenerated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&singleOutcome);
  if (!singleGenerated || singleGenerated->candidates.size() != 1)
    fail("single covered graph did not produce an exact TechMapping");
  const auto singleCandidate = take(loom::mapping::importTechMapping(
      singleGenerated->candidates.front(), store));
  for (const auto &net : singleCandidate.view().residualLogicalNets()) {
    dataflow::GraphRef producerGraph = [&]() {
      if (const auto *result =
              std::get_if<dataflow::ActorTokenResultRef>(&net.producer))
        return take(dataflow.resolve(result->actor)).graph;
      return std::visit([](const auto &ingress) { return ingress.graph; },
                        std::get<dataflow::GraphIngressTokenRef>(net.producer));
    }();
    if (producerGraph != *singleGraph)
      fail("residual logical net escaped the TechMapping graph cover");
  }
}

void serialTopologyPrunesIndependentActorScale() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildIndependentIntegerAddsDataflow(context, 1000);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildSerialIntegerAddFabric(store);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.matchRowAttemptLimit = 1000;
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  const auto start = std::chrono::steady_clock::now();
  const auto outcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), config, store});
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const auto *incomplete =
      std::get_if<loom::mapping::IncompleteTechMappingGeneration>(&outcome);
  if (!incomplete || incomplete->accounting.matchRowAttempts != 1000)
    fail("independent-actor scale fixture did not stop at its semantic limit");
  if (elapsed >= std::chrono::seconds(10))
    fail("independent actors caused factorial serial-topology enumeration");
}

void unrelatedBuiltinOperationsDoNotConsumeSeedBudget() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildIndependentIntegerAddsDataflow(context, 64);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildSmallFabric(store);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.matchRowAttemptLimit = 4096;
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  const auto outcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), config, store});
  const auto *generated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&outcome);
  if (!generated || generated->candidates.size() != 1 ||
      generated->accounting.matchRowAttempts >= 4096) {
    const auto accounting = std::visit(
        [](const auto &result) { return result.accounting; }, outcome);
    fail("schema-incompatible builtin operations consumed the seed budget: " +
         std::to_string(accounting.matchRowAttempts) + " attempts");
  }
  const auto candidate = take(
      loom::mapping::importTechMapping(generated->candidates.front(), store));
  std::size_t coveredActors = 0;
  for (const auto &realization : candidate.view().computeRealizations())
    coveredActors += realization.actors.size();
  if (coveredActors != dataflow.actors().size())
    fail("schema-indexed match rows did not cover every selected actor");
  if (!llvm::any_of(candidate.view().residualLogicalNets(),
                    [](const auto &net) { return net.sinks.size() == 64; }))
    fail("multicast producer lost residual sink obligations");
}

void forcedComputeAndMemoryRowsPublishDeterministically() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildMixedDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  if (dataflow.graphs().size() != 1 || dataflow.actors().size() != 2)
    fail("mixed fixture did not retain exactly two actors");

  const auto fabric = buildSmallFabric(store);
  if (fabric.view().fuTemplates().empty() ||
      fabric.view().memoryEngineTemplates().empty())
    fail("sealed Fabric omitted a semantic template inventory");

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  const loom::mapping::TechMappingGenerationInputs inputs{
      dataflow, covers, fabric.view(), config, store};

  const auto first = loom::mapping::generateTechMappings(inputs);
  const auto *generated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&first);
  if (!generated) {
    const auto accounting = std::visit(
        [](const auto &outcome) { return outcome.accounting; }, first);
    fail("generator returned a non-Generated outcome after " +
         std::to_string(accounting.matchRowAttempts) + " row attempts and " +
         std::to_string(accounting.partialCoverExpansions) +
         " cover expansions");
  }
  if (generated->candidates.size() != 1)
    fail("generator published the wrong candidate count");
  auto imported = take(
      loom::mapping::importTechMapping(generated->candidates.front(), store));
  if (imported.view().computeRealizations().size() != 1 ||
      imported.view().memoryRealizations().size() != 1)
    fail("published candidate did not cover compute and memory actors");
  if (imported.view().computeRealizations().front().actors.size() != 1 ||
      imported.view().memoryRealizations().front().actors.size() != 1)
    fail("realization rows do not cover each actor exactly once");
  if (generated->accounting.matchRowAttempts == 0 ||
      generated->accounting.partialCoverExpansions == 0 ||
      generated->accounting.publicationSlots != 1)
    fail("generator did not report semantic work accounting");

  const auto second = loom::mapping::generateTechMappings(inputs);
  const auto *repeated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&second);
  if (!repeated || repeated->candidates != generated->candidates ||
      !(repeated->accounting == generated->accounting) ||
      repeated->termination != generated->termination)
    fail("identical invocation changed its canonical finite prefix");
}

void multiActorMemoryRowsCompeteWithSingletonCover() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildMemoryChainDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildSmallFabric(store);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 8;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  const loom::mapping::TechMappingGenerationInputs inputs{
      dataflow, covers, fabric.view(), config, store};

  const auto outcome = loom::mapping::generateTechMappings(inputs);
  const auto *generated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&outcome);
  if (!generated)
    fail("memory-chain generation did not produce a candidate");
  if (generated->termination !=
      loom::mapping::TechMappingGenerationTermination::SearchExhausted)
    fail("memory-chain ordering fixture did not exhaust its candidate set");

  std::vector<CandidateKey> candidateKeys;
  for (const auto &reference : generated->candidates)
    candidateKeys.push_back(candidateKey(reference, dataflow, store));
  if (!llvm::is_sorted(candidateKeys))
    fail("generated candidates do not follow canonical row-key order");

  loom::ResolvedConfig prefixResolved = loom::defaultResolvedConfig();
  prefixResolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto prefixConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(prefixResolved));
  const auto prefixOutcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), prefixConfig, store});
  const auto *prefix =
      std::get_if<loom::mapping::GeneratedTechMappings>(&prefixOutcome);
  if (!prefix || prefix->candidates.size() != 1 || candidateKeys.empty() ||
      candidateKey(prefix->candidates.front(), dataflow, store) !=
          candidateKeys.front())
    fail("candidate publication limit did not preserve the canonical prefix");

  bool foundGrouped = false;
  for (const auto &reference : generated->candidates) {
    auto candidate = take(loom::mapping::importTechMapping(reference, store));
    if (candidate.view().memoryRealizations().size() != 1)
      continue;
    const auto &realization = candidate.view().memoryRealizations().front();
    if (realization.actors.size() == 2) {
      loom::test::exerciseSpatialMemoryOperationPortRelations(
          context, dataflow, candidate.view(), fabric.view(), store);
      foundGrouped = true;
      break;
    }
  }
  if (!foundGrouped)
    fail("memory actor grouping was absent from the exact-cover domain");
}

void internalMemoryEdgeIsAnExplicitCandidateChoice() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildMemoryChainDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildInternalMemoryEdgeFabric(store);

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 8;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  const auto outcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), config, store});
  const auto *generated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&outcome);
  if (!generated)
    fail("internal-edge Fabric did not produce a TechMapping candidate");

  bool foundExternal = false;
  bool foundInternal = false;
  std::optional<std::size_t> externalNetCount;
  std::optional<std::size_t> internalNetCount;
  for (const auto &reference : generated->candidates) {
    auto candidate = take(loom::mapping::importTechMapping(reference, store));
    if (candidate.view().memoryRealizations().size() != 1)
      continue;
    const auto &realization = candidate.view().memoryRealizations().front();
    if (realization.actors.size() != 2)
      continue;
    if (realization.internalEdges.empty()) {
      foundExternal = true;
      externalNetCount = candidate.view().residualLogicalNets().size();
    }
    if (realization.internalEdges.size() == 1) {
      foundInternal = true;
      internalNetCount = candidate.view().residualLogicalNets().size();
    }
  }
  if (!foundExternal || !foundInternal)
    fail("memory internal edge was not retained as an explicit row choice");
  if (!externalNetCount || !internalNetCount ||
      *internalNetCount + 1 != *externalNetCount)
    fail("memory internal edge did not remove exactly one residual net");
}

void semanticLimitsDoNotBecomeInfeasibilityProofs() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildMixedDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildSmallFabric(store);
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};

  const auto run = [&](loom::ResolvedConfig resolved) {
    const auto config =
        take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
    return loom::mapping::generateTechMappings(
        {dataflow, covers, fabric.view(), config, store});
  };

  loom::ResolvedConfig rowLimited = loom::defaultResolvedConfig();
  rowLimited.dse.techMapping.matchRowAttemptLimit = 1;
  const auto rowOutcome = run(rowLimited);
  const auto *rowIncomplete =
      std::get_if<loom::mapping::IncompleteTechMappingGeneration>(&rowOutcome);
  if (!rowIncomplete || rowIncomplete->accounting.matchRowAttempts != 1)
    fail("match-row limit was not reported as incomplete work");

  loom::ResolvedConfig coverLimited = loom::defaultResolvedConfig();
  coverLimited.dse.techMapping.partialCoverExpansionLimit = 1;
  const auto coverOutcome = run(coverLimited);
  const auto *coverIncomplete =
      std::get_if<loom::mapping::IncompleteTechMappingGeneration>(
          &coverOutcome);
  if (!coverIncomplete ||
      coverIncomplete->accounting.partialCoverExpansions != 1)
    fail("cover-expansion limit was not reported as incomplete work");

  loom::ResolvedConfig publicationLimited = loom::defaultResolvedConfig();
  publicationLimited.dse.techMapping.candidatePublicationLimit = 1;
  const auto publicationOutcome = run(publicationLimited);
  const auto *generated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&publicationOutcome);
  if (!generated || generated->accounting.publicationSlots != 1 ||
      generated->termination !=
          loom::mapping::TechMappingGenerationTermination::SemanticLimitReached)
    fail("publication limit did not preserve its generated finite prefix");
}

void completedCoverSurvivesExpansionLimit() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildSingleSyncDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildSmallFabric(store);
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};

  loom::ResolvedConfig exhaustive = loom::defaultResolvedConfig();
  exhaustive.dse.techMapping.candidatePublicationLimit = 32;
  const auto exhaustiveConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(exhaustive));
  const auto exhaustiveOutcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), exhaustiveConfig, store});
  const auto *all =
      std::get_if<loom::mapping::GeneratedTechMappings>(&exhaustiveOutcome);
  if (!all || all->candidates.size() < 2)
    fail("single-sync expansion fixture has fewer than two exact covers");

  loom::ResolvedConfig limited = loom::defaultResolvedConfig();
  limited.dse.techMapping.partialCoverExpansionLimit = 1;
  limited.dse.techMapping.candidatePublicationLimit = 32;
  const auto limitedConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(limited));
  const auto limitedOutcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), limitedConfig, store});
  const auto *prefix =
      std::get_if<loom::mapping::GeneratedTechMappings>(&limitedOutcome);
  if (!prefix || prefix->candidates.size() != 1 ||
      prefix->termination != loom::mapping::TechMappingGenerationTermination::
                                 SemanticLimitReached ||
      prefix->accounting.partialCoverExpansions != 1 ||
      candidateKey(prefix->candidates.front(), dataflow, store) !=
          candidateKey(all->candidates.front(), dataflow, store))
    fail("completed canonical cover was discarded at the expansion limit");
}

void deadResultDerivesPhysicalDiscardWithoutSoftwareBoundary() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildSyncWithDeadResultDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildTokenSyncFabric(store);
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(resolved));
  const auto outcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), config, store});
  const auto *generated =
      std::get_if<loom::mapping::GeneratedTechMappings>(&outcome);
  if (!generated || generated->candidates.size() != 1)
    fail("dead sync result did not derive a valid discard realization");

  const auto candidate = take(
      loom::mapping::importTechMapping(generated->candidates.front(), store));
  const auto realizations = candidate.view().computeRealizations();
  if (realizations.size() != 1 || realizations.front().actors.size() != 1 ||
      realizations.front().boundaries.size() != 3)
    fail("dead result created a persistent software boundary");
  if (llvm::any_of(realizations.front().boundaries, [](const auto &boundary) {
        return boundary.direction ==
                   loom::fabric::FabricPortDirection::Output &&
               boundary.portOrdinal == 1;
      }))
    fail("dead result was persisted as an exposed software boundary");
}

void exhaustiveUnsupportedActorProvesInfeasible() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildUnsupportedDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildSmallFabric(store);
  loom::ResolvedConfig exhaustive = loom::defaultResolvedConfig();
  exhaustive.dse.techMapping.candidatePublicationLimit = 65536;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(exhaustive));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  const auto outcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), config, store});
  if (!std::holds_alternative<loom::mapping::ProvenInfeasibleTechMapping>(
          outcome))
    fail("exhaustive unsupported actor did not prove infeasibility");

  loom::ResolvedConfig limited = loom::defaultResolvedConfig();
  limited.dse.techMapping.matchRowAttemptLimit = 1;
  const auto limitedConfig =
      take(loom::mapping::projectResolvedTechMappingConfigView(limited));
  const auto limitedOutcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), limitedConfig, store});
  const auto *incomplete =
      std::get_if<loom::mapping::IncompleteTechMappingGeneration>(
          &limitedOutcome);
  if (!incomplete || incomplete->accounting.matchRowAttempts != 1)
    fail("a rejected prospective seed bypassed match-row accounting");
}

void nonmatchingMemoryCapabilityDoesNotEnterSeedDomain() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildUnsupportedMemoryDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildInternalMemoryEdgeFabric(store);

  loom::ResolvedConfig limited = loom::defaultResolvedConfig();
  limited.dse.techMapping.matchRowAttemptLimit = 1;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(limited));
  const std::array<dataflow::GraphRef, 1> covers = {
      dataflow.graphs().front().ref};
  const auto outcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), config, store});
  const auto *infeasible =
      std::get_if<loom::mapping::ProvenInfeasibleTechMapping>(&outcome);
  if (!infeasible || infeasible->accounting.matchRowAttempts != 0)
    fail("a nonmatching memory capability entered the match-row seed domain");
}

void publicationLimitDoesNotTruncateComponentProofs() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildSupportedAndUnsupportedDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildSmallFabric(store);

  loom::ResolvedConfig limited = loom::defaultResolvedConfig();
  limited.dse.techMapping.candidatePublicationLimit = 1;
  const auto config =
      take(loom::mapping::projectResolvedTechMappingConfigView(limited));
  std::vector<dataflow::GraphRef> covers;
  for (const auto &graph : dataflow.graphs())
    covers.push_back(graph.ref);
  const auto outcome = loom::mapping::generateTechMappings(
      {dataflow, covers, fabric.view(), config, store});
  const auto *infeasible =
      std::get_if<loom::mapping::ProvenInfeasibleTechMapping>(&outcome);
  if (!infeasible || infeasible->accounting.publicationSlots != 0)
    fail("candidate publication limit truncated component exhaustion");
}

void malformedGraphCoversReturnTypedInvalidOutcomes() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();
  auto dataflowArtifact = buildMixedDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflow = take(dataflowArtifact.view());
  const auto fabric = buildSmallFabric(store);
  const auto config = take(loom::mapping::projectResolvedTechMappingConfigView(
      loom::defaultResolvedConfig()));

  const auto empty = loom::mapping::generateTechMappings(
      {dataflow, {}, fabric.view(), config, store});
  const auto *emptyInvalid =
      std::get_if<loom::mapping::InvalidTechMappingGeneration>(&empty);
  if (!emptyInvalid ||
      emptyInvalid->reason !=
          loom::mapping::InvalidTechMappingGenerationReason::EmptyGraphCover)
    fail("empty graph cover did not return typed Invalid");

  const std::array<dataflow::GraphRef, 1> foreign = {dataflow::GraphRef{
      fabric.view().identity(), dataflow.graphs().front().ref.entity}};
  const auto foreignOutcome = loom::mapping::generateTechMappings(
      {dataflow, foreign, fabric.view(), config, store});
  const auto *foreignInvalid =
      std::get_if<loom::mapping::InvalidTechMappingGeneration>(&foreignOutcome);
  if (!foreignInvalid || foreignInvalid->reason !=
                             loom::mapping::InvalidTechMappingGenerationReason::
                                 ForeignGraphReference)
    fail("foreign graph cover did not return typed Invalid");

  const std::array<dataflow::GraphRef, 2> duplicate = {
      dataflow.graphs().front().ref, dataflow.graphs().front().ref};
  const auto duplicateOutcome = loom::mapping::generateTechMappings(
      {dataflow, duplicate, fabric.view(), config, store});
  const auto *duplicateInvalid =
      std::get_if<loom::mapping::InvalidTechMappingGeneration>(
          &duplicateOutcome);
  if (!duplicateInvalid ||
      duplicateInvalid->reason !=
          loom::mapping::InvalidTechMappingGenerationReason::
              NonCanonicalGraphCover)
    fail("duplicate graph cover did not return typed Invalid");
}

} // namespace

int main() {
  serialComputeTemplateAcceptsExactActorChain();
  matchRowLimitPreservesGlobalActorOrder();
  serialTopologyPrunesIndependentActorScale();
  unrelatedBuiltinOperationsDoNotConsumeSeedBudget();
  forcedComputeAndMemoryRowsPublishDeterministically();
  multiActorMemoryRowsCompeteWithSingletonCover();
  internalMemoryEdgeIsAnExplicitCandidateChoice();
  semanticLimitsDoNotBecomeInfeasibilityProofs();
  completedCoverSurvivesExpansionLimit();
  deadResultDerivesPhysicalDiscardWithoutSoftwareBoundary();
  exhaustiveUnsupportedActorProvesInfeasible();
  nonmatchingMemoryCapabilityDoesNotEnterSeedDomain();
  publicationLimitDoesNotTruncateComponentProofs();
  malformedGraphCoversReturnTypedInvalidOutcomes();
  llvm::outs() << "tech mapping generator tests passed\n";
  return 0;
}

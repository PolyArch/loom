#include "MappedRtlSimulationTestSupport.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "ADG/FuLibrary.h"
#include "ADG/MemoryLibrary.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/ExecutionControl.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "DSE/SpatialRuntimeFeedback.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Evaluation/Models/MappedRtlSimulation.h"
#include "Evaluation/ProductionRegistry.h"
#include "Fabric/Artifact/InterconnectImplementation.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FifoResourceContract.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/RTL/PortableProviders.h"
#include "Hardware/RTL/SpatialCoreImplementation.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "Simulator/SimulationArtifacts.h"

#include "ConfigurationABITestSupport.h"
#include "RootCompleteSpatialPnrTestSupport.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::eda::test {
namespace {

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    deployment::test::fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    deployment::test::fail(test, llvm::toString(std::move(error)));
}

class MappedSpatialHardwareFixtureObservation final {
public:
  MappedSpatialHardwareFixtureObservation(
      MappedSpatialHardwareFixtureObserver observer,
      MappedSpatialHardwareFixtureOperation operation)
      : observer_(observer), operation_(operation) {
    if (observer_)
      observer_(operation_, MappedSpatialHardwareFixtureBoundary::Begin);
  }

  ~MappedSpatialHardwareFixtureObservation() {
    if (observer_)
      observer_(operation_, MappedSpatialHardwareFixtureBoundary::End);
  }

private:
  MappedSpatialHardwareFixtureObserver observer_;
  MappedSpatialHardwareFixtureOperation operation_;
};

template <typename Build>
auto observeMappedSpatialHardwareFixtureOperation(
    MappedSpatialHardwareFixtureObserver observer,
    MappedSpatialHardwareFixtureOperation operation, Build &&build) {
  MappedSpatialHardwareFixtureObservation observation(observer, operation);
  return std::forward<Build>(build)();
}

void writePackedField(std::vector<std::uint8_t> &bytes, std::uint32_t bitOffset,
                      std::uint32_t bitCount, std::uint64_t value) {
  for (std::uint32_t bit = 0; bit != bitCount; ++bit)
    if (((value >> bit) & 1U) != 0)
      bytes[(bitOffset + bit) / 8] |=
          static_cast<std::uint8_t>(1U << ((bitOffset + bit) % 8));
}

std::vector<std::uint8_t>
directInactiveValue(llvm::StringRef test,
                    const fabric::FabricArtifactView &module,
                    const fabric::FabricSemanticConfigFieldRef &field,
                    const fabric::FabricSemanticFieldRelation &relation,
                    mlir::MLIRContext &context) {
  const std::uint64_t bitCount = *relation.directEncodedBitCount();
  std::vector<std::uint8_t> candidate((bitCount + 7) / 8, 0);
  if (llvm::Error error = relation.validateSemanticValue(candidate))
    llvm::consumeError(std::move(error));
  else
    return candidate;

  if (field.owner.catalog().kind() !=
      fabric::FabricInventoryOwnerKind::FuOccurrenceNode)
    deployment::test::fail(
        test, "non-operation Direct field has no zero inactive carrier");
  const auto operation = std::get<fabric::FabricFuOccurrenceNodeRef>(
      field.owner.catalog().payload);
  const auto *capability = module.resolvedFabricOpCapability(operation);
  deployment::test::require(test, capability != nullptr,
                            "Direct operation field has no capability");
  const auto operationRelation =
      take(test, capability->resolveSemanticFieldRelation(context));

  const auto accepts = [&](std::vector<std::uint8_t> value)
      -> std::optional<std::vector<std::uint8_t>> {
    if (llvm::Error error = relation.validateSemanticValue(value)) {
      llvm::consumeError(std::move(error));
      return std::nullopt;
    }
    return value;
  };
  const auto tryElementWidths =
      [&](auto makeCandidate) -> std::optional<std::vector<std::uint8_t>> {
    for (::fabric::IntegerWidth width : ::fabric::integerWidthDomain)
      if (auto value = makeCandidate(::fabric::getBitWidth(width)); value)
        return value;
    for (::fabric::FloatFormat format : ::fabric::floatFormatDomain)
      if (auto value = makeCandidate(::fabric::getBitWidth(format)); value)
        return value;
    return std::nullopt;
  };

  if (const auto *layout =
          operationRelation.fixedVectorSliceAlignMergeLayout()) {
    for (::dataflow::OperationSchemaId schema :
         capability->enabledOperationSchemas) {
      if (schema != ::dataflow::OperationSchemaId::VectorExtract &&
          schema != ::dataflow::OperationSchemaId::VectorInsert)
        continue;
      auto value = tryElementWidths([&](std::uint32_t width) {
        std::vector<std::uint8_t> encoded((bitCount + 7) / 8, 0);
        if (layout->encodesMode)
          writePackedField(
              encoded, layout->modeBitOffset, 1,
              schema == ::dataflow::OperationSchemaId::VectorInsert ? 1 : 0);
        writePackedField(encoded, layout->sliceWidthBitOffset,
                         layout->sliceWidthBitCount, width - 1);
        return accepts(std::move(encoded));
      });
      if (value)
        return std::move(*value);
    }
  }

  if (const auto *layout = operationRelation.fixedVectorShuffleLayout()) {
    auto value = tryElementWidths([&](std::uint32_t width) {
      std::vector<std::uint8_t> encoded((bitCount + 7) / 8, 0);
      writePackedField(encoded, layout->blockWidthBitOffset,
                       layout->blockWidthBitCount, width - 1);
      return accepts(std::move(encoded));
    });
    if (value)
      return std::move(*value);
  }

  deployment::test::fail(test,
                         "Direct Fabric relation has no inactive witness");
}

mlir::MLIRContext makeDataflowContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::mapping::MappingDialect,
                  mlir::arith::ArithDialect, ::fabric::FabricDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(llvm::StringRef test,
                                                  mlir::MLIRContext &context) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @mapped_rtl(
      %start: none, %value: i32, %increment: i32, %address: index,
      %store_value: i32, %stream0: i32, %stream1: i32,
      %memory: memref<4xi32>) -> (i32, i32, i32)
      attributes {input_segments = array<i32: 4, 2, 1>,
                  result_segments = array<i32: 1, 2, 0>} {
    %select = dataflow.constant %start {const_value = false} : i1
    %stream0_lane:2 = dataflow.demux %select, %stream0
        : (i1, i32) -> (i32, i32)
    %stream1_lane:2 = dataflow.demux %select, %stream1
        : (i1, i32) -> (i32, i32)
    %published_value:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    %sum = arith.addi %stream1_lane#0, %increment : i32
    %stream1_sum = arith.addi %sum, %increment : i32
    %scheduled = arith.addi %value, %increment : i32
    %scheduled_done:2 = dataflow.sync %published_value#0, %scheduled
        : (none, i32) -> (none, i32)
    %published_stream0 = dataflow.sync %stream0_lane#0 : (i32) -> i32
    %published_stream1 = dataflow.sync %stream1_sum : (i32) -> i32
    %retired:3 = dataflow.sync %scheduled_done#0, %published_stream0,
        %published_stream1 : (none, i32, i32) -> (none, i32, i32)
    %stored = dataflow.store %memory[%address] %store_value
        %retired#0 : memref<4xi32>
    dataflow.graph.return values(%published_value#1 : i32)
        streams(%retired#1, %retired#2 : i32, i32)
        memories() complete(%stored : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %stream0: !dataflow.channel<i32>, %stream1: !dataflow.channel<i32>,
      %output0: !dataflow.channel<i32>, %output1: !dataflow.channel<i32>,
      %memory: memref<4xi32>, %value: i32, %increment: i32,
      %address: index, %store_value: i32) ctrl (%ctrl: none) {
    %stream0_value = arith.constant 33 : i32
    %stream1_value = arith.constant 49 : i32
    dataflow.channel.send %stream0, %stream0_value
        : !dataflow.channel<i32>
    dataflow.channel.send %stream1, %stream1_value
        : !dataflow.channel<i32>
    %result, %done = dataflow.graph.launch @mapped_rtl deps(%ctrl)
        values(%value, %increment, %address, %store_value)
        stream_inputs(%stream0 source_map affine_map<() -> ()>,
                      %stream1 source_map affine_map<() -> ()>)
        memories(%memory) stream_outputs(%output0, %output1)
        : (none, i32, i32, index, i32,
           !dataflow.channel<i32>,
           !dataflow.channel<i32>, memref<4xi32>, !dataflow.channel<i32>,
           !dataflow.channel<i32>) -> (i32, none)
    %observed0 = dataflow.channel.receive %output0
        : !dataflow.channel<i32>
    %observed1 = dataflow.channel.receive %output1
        : !dataflow.channel<i32>
    dataflow.thread.yield %done : none
  }
  func.func private @host(
      %stream0: !dataflow.channel<i32>, %stream1: !dataflow.channel<i32>,
      %output0: !dataflow.channel<i32>, %output1: !dataflow.channel<i32>,
      %memory: memref<4xi32>) {
    %value = arith.constant 7 : i32
    %increment = arith.constant 5 : i32
    %address = arith.constant 0 : index
    %store_value = arith.constant 1144201745 : i32
    %thread = dataflow.thread.launch @worker(
        %stream0, %stream1, %output0, %output1, %memory, %value, %increment,
        %address, %store_value)
        : (!dataflow.channel<i32>, !dataflow.channel<i32>,
           !dataflow.channel<i32>, !dataflow.channel<i32>, memref<4xi32>,
           i32, i32, index, i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  deployment::test::require(test, static_cast<bool>(source),
                            "cannot parse Dataflow fixture");
  return take(test, dataflow::finalizeCanonicalDataflow(*source));
}

fabric::FinalizedFabricRoot
buildBuiltinSpatialCore(llvm::StringRef test, ArtifactStore &artifacts,
                        const adg::BuiltinTargetScale &scale) {
  adg::DesignBuilder builder(artifacts);
  auto expansion = take(test, adg::expandBuiltinSpatialCore(builder, scale));
  requireSuccess(test, expansion.spatialCore.close(expansion.outputs));
  auto design = take(test, std::move(builder).finalize());
  deployment::test::require(test, design.roots().size() == 1,
                            "builtin fixture did not publish one Module");
  return design.roots().front();
}

fabric::FinalizedFabricRoot
buildSpatialCore(llvm::StringRef test, ArtifactStore &artifacts,
                 MappedRtlFixtureTopology topology,
                 std::size_t spatialMemoryOccurrenceCount) {
  if (topology == MappedRtlFixtureTopology::Minimal)
    return loom::test::buildSpatialCore(artifacts);
  if (topology == MappedRtlFixtureTopology::BuiltinCoverage)
    return buildBuiltinSpatialCore(test, artifacts,
                                   adg::builtinLargeTarget.scale);
  deployment::test::require(test, spatialMemoryOccurrenceCount > 0,
                            "portable fixture requires a memory occurrence");
  deployment::test::require(
      test,
      spatialMemoryOccurrenceCount <=
          std::numeric_limits<std::uint32_t>::max() - 3,
      "portable fixture memory count exceeds its mesh coordinate domain");

  constexpr std::uint32_t payloadWidth = 128;
  constexpr std::uint32_t tagWidth = 2;
  constexpr std::size_t boundaryBankWidth = 4;
  constexpr std::size_t directNetworkInputCount = 16;
  const adg::PortType payload = take(test, adg::PortType::bits(payloadWidth));
  const adg::PortType tag = take(test, adg::PortType::bits(tagWidth));
  const adg::PortType taggedPayload =
      take(test, adg::PortType::taggedBits(payloadWidth, tagWidth));
  const std::vector<adg::PortType> syncPorts(3, payload);
  const std::vector<adg::PortType> demuxInputPorts(2, payload);
  const std::vector<adg::PortType> demuxOutputPorts(4, payload);
  const std::vector<adg::PortType> constantPorts(1, payload);
  const std::vector<adg::PortType> temporalMeshInputPorts(2, payload);
  const std::vector<adg::PortType> temporalForkOutputs(2, payload);
  const std::vector<adg::PortType> temporalInjectionPorts(3, payload);
  const std::vector<adg::PortType> loopInputPorts(5, payload);
  const std::vector<adg::PortType> loopOutputPorts(3, payload);
  constexpr std::size_t firstMemoryInputCount = 4;
  constexpr std::size_t firstMemoryOutputCount = 2;

  auto indexWidths =
      take(test, ::fabric::UnsignedDomain::fromCanonical({{32, 32}, {64, 64}}));
  adg::ManagerMemoryParameters memoryParameters;
  memoryParameters.interface = {
      adg::MemoryAccessDomainParameters{128, std::nullopt, 4,
                                        std::move(indexWidths)},
      64, 128};
  adg::MemorySpec memory =
      take(test, adg::makeHybrid32ManagerMemory(std::move(memoryParameters)));

  std::vector<adg::PortType> moduleInputs(directNetworkInputCount, payload);
  moduleInputs.push_back(payload);
  for (std::size_t ordinal = 0; ordinal != spatialMemoryOccurrenceCount;
       ++ordinal)
    moduleInputs.push_back(memory.inputTypes().front());
  std::vector<adg::PortType> moduleOutputs(4 * boundaryBankWidth, payload);
  moduleOutputs.insert(moduleOutputs.end(), 2, tag);

  adg::DesignBuilder builder(artifacts);
  auto spatial =
      take(test, builder.createSpatialCore("mapped-rtl-heterogeneous",
                                           std::move(moduleInputs),
                                           std::move(moduleOutputs)));
  const std::vector<adg::PortType> boundaryBank(boundaryBankWidth, payload);
  const std::uint32_t rightBoundaryX =
      static_cast<std::uint32_t>(3 + spatialMemoryOccurrenceCount);
  std::vector<adg::MeshCellAttachmentSpec> meshAttachments{
      {0, 0, boundaryBank, boundaryBank},
      {0, 3, boundaryBank, boundaryBank},
      {rightBoundaryX, 0, boundaryBank, boundaryBank},
      {rightBoundaryX, 3, boundaryBank, boundaryBank},
      {rightBoundaryX, 1, temporalMeshInputPorts, temporalInjectionPorts},
      {1, 0, syncPorts, syncPorts},
      {1, 1, syncPorts, syncPorts},
      {1, 2, syncPorts, syncPorts},
      {1, 3, syncPorts, syncPorts},
      {2, 2, syncPorts, syncPorts},
      {2, 0, demuxInputPorts, demuxOutputPorts},
      {2, 3, demuxInputPorts, demuxOutputPorts},
      {2, 1, constantPorts, constantPorts},
      {3, 1, demuxInputPorts, constantPorts},
      {3, 0, std::vector<adg::PortType>(firstMemoryInputCount, payload),
       std::vector<adg::PortType>(firstMemoryOutputCount, payload)},
      {3, 3,
       std::vector<adg::PortType>(
           memory.inputTypes().size() - 1 - firstMemoryInputCount, payload),
       std::vector<adg::PortType>(
           memory.outputTypes().size() - firstMemoryOutputCount, payload)},
      {3, 2, constantPorts, constantPorts},
      {0, 1, constantPorts, constantPorts},
      {0, 2, loopInputPorts, loopOutputPorts}};
  for (std::size_t ordinal = 1; ordinal != spatialMemoryOccurrenceCount;
       ++ordinal) {
    const std::uint32_t x = static_cast<std::uint32_t>(3 + ordinal);
    meshAttachments.push_back(
        {x, 0, std::vector<adg::PortType>(firstMemoryInputCount, payload),
         std::vector<adg::PortType>(firstMemoryOutputCount, payload)});
    meshAttachments.push_back(
        {x, 3,
         std::vector<adg::PortType>(
             memory.inputTypes().size() - 1 - firstMemoryInputCount, payload),
         std::vector<adg::PortType>(
             memory.outputTypes().size() - firstMemoryOutputCount, payload)});
  }
  auto network = take(test, spatial.addMeshSwitchNetwork(take(
                                test, adg::MeshSwitchNetworkSpec::spatial(
                                          rightBoundaryX + 1, 4, 2, payload, 1,
                                          ::fabric::FifoQueueDiscipline::
                                              StrictFifo,
                                          std::move(meshAttachments)))));
  std::array<adg::MeshCellAttachment, 5> boundaryAttachments = {
      take(test, network.attachment(0)), take(test, network.attachment(1)),
      take(test, network.attachment(2)), take(test, network.attachment(3)),
      take(test, network.attachment(4))};
  for (std::size_t bank = 0; bank != 4; ++bank) {
    std::vector<adg::SpatialValue> inputs;
    inputs.reserve(boundaryBankWidth);
    for (std::size_t ordinal = 0; ordinal != boundaryBankWidth; ++ordinal)
      inputs.push_back(
          take(test, spatial.input(bank * boundaryBankWidth + ordinal)));
    requireSuccess(test, boundaryAttachments[bank].connectOutputs(inputs));
  }

  const auto addSyncFu = [&](adg::PeBuilder &pe) {
    std::vector<adg::PeValue> peInputs;
    peInputs.reserve(syncPorts.size());
    for (std::size_t ordinal = 0; ordinal != syncPorts.size(); ++ordinal)
      peInputs.push_back(take(test, pe.input(ordinal)));
    auto fu = take(test, pe.addFu(peInputs, adg::FuSpec{syncPorts, syncPorts}));
    std::vector<adg::FuValue> fuInputs;
    fuInputs.reserve(syncPorts.size());
    for (std::size_t ordinal = 0; ordinal != syncPorts.size(); ++ordinal)
      fuInputs.push_back(take(test, fu.input(ordinal)));
    auto operation = take(
        test,
        fu.addOperation(
            fuInputs,
            adg::OperationCapabilitySpec{
                ::fabric::ImplementationFamilyId::TokenSync,
                ::fabric::RoutedTokenParams{
                    payloadWidth, static_cast<std::uint32_t>(syncPorts.size())},
                {::dataflow::OperationSchemaId::DataflowSync},
                syncPorts,
                ::fabric::oneCycleElasticOperationResourceContract()}));
    requireSuccess(test, fu.addCapabilityTemplate(
                             adg::FuCapabilityTemplateSpec{{operation}, {}}));
    std::vector<adg::FuValue> outputs;
    outputs.reserve(syncPorts.size());
    for (std::size_t ordinal = 0; ordinal != syncPorts.size(); ++ordinal)
      outputs.push_back(take(test, operation.output(ordinal)));
    requireSuccess(test, fu.close(outputs));
  };

  for (std::size_t peOrdinal = 0; peOrdinal != 5; ++peOrdinal) {
    auto attachment = take(test, network.attachment(5 + peOrdinal));
    auto pe =
        take(test, spatial.addPe(attachment.inputs(),
                                 adg::PeSpec::spatial(syncPorts, syncPorts)));
    addSyncFu(pe);
    requireSuccess(test, pe.close());
    std::vector<adg::SpatialValue> outputs;
    outputs.reserve(syncPorts.size());
    for (std::size_t ordinal = 0; ordinal != syncPorts.size(); ++ordinal)
      outputs.push_back(take(test, pe.output(ordinal)));
    requireSuccess(test, attachment.connectOutputs(outputs));
  }

  for (std::size_t peOrdinal = 0; peOrdinal != 2; ++peOrdinal) {
    auto demuxAttachment = take(test, network.attachment(10 + peOrdinal));
    auto demuxPe = take(
        test,
        spatial.addPe(demuxAttachment.inputs(),
                      adg::PeSpec::spatial(demuxInputPorts, demuxOutputPorts)));
    std::vector<adg::PeValue> demuxPeInputs{take(test, demuxPe.input(0)),
                                            take(test, demuxPe.input(1))};
    auto demuxFu =
        take(test, demuxPe.addFu(demuxPeInputs, adg::FuSpec{demuxInputPorts,
                                                            demuxOutputPorts}));
    auto demux = take(
        test, demuxFu.addOperation(
                  {take(test, demuxFu.input(0)), take(test, demuxFu.input(1))},
                  adg::OperationCapabilitySpec{
                      ::fabric::ImplementationFamilyId::TokenDemux,
                      ::fabric::RoutedTokenParams{
                          payloadWidth,
                          static_cast<std::uint32_t>(demuxOutputPorts.size())},
                      {::dataflow::OperationSchemaId::DataflowDemux},
                      demuxOutputPorts,
                      ::fabric::oneCycleElasticOperationResourceContract()}));
    requireSuccess(test, demuxFu.addCapabilityTemplate(
                             adg::FuCapabilityTemplateSpec{{demux}, {}}));
    std::vector<adg::FuValue> demuxOutputs;
    std::vector<adg::SpatialValue> demuxPeOutputs;
    for (std::size_t ordinal = 0; ordinal != demuxOutputPorts.size(); ++ordinal)
      demuxOutputs.push_back(take(test, demux.output(ordinal)));
    requireSuccess(test, demuxFu.close(demuxOutputs));
    requireSuccess(test, demuxPe.close());
    for (std::size_t ordinal = 0; ordinal != demuxOutputPorts.size(); ++ordinal)
      demuxPeOutputs.push_back(take(test, demuxPe.output(ordinal)));
    requireSuccess(test, demuxAttachment.connectOutputs(demuxPeOutputs));
  }

  const auto addConstantPe = [&](std::size_t attachmentOrdinal) {
    auto attachment = take(test, network.attachment(attachmentOrdinal));
    auto pe =
        take(test,
             spatial.addPe(attachment.inputs(),
                           adg::PeSpec::spatial(constantPorts, constantPorts)));
    auto fu = take(test, pe.addFu({take(test, pe.input(0))},
                                  adg::FuSpec{constantPorts, constantPorts}));
    auto operation = take(
        test, fu.addOperation(
                  {take(test, fu.input(0))},
                  adg::OperationCapabilitySpec{
                      ::fabric::ImplementationFamilyId::TokenConstant,
                      ::fabric::PayloadCapacityParams{payloadWidth},
                      {::dataflow::OperationSchemaId::DataflowConstant},
                      constantPorts,
                      ::fabric::oneCycleElasticOperationResourceContract()}));
    requireSuccess(test, fu.addCapabilityTemplate(
                             adg::FuCapabilityTemplateSpec{{operation}, {}}));
    requireSuccess(test, fu.close({take(test, operation.output(0))}));
    requireSuccess(test, pe.close());
    requireSuccess(test, attachment.connectOutputs({take(test, pe.output(0))}));
  };
  addConstantPe(12);
  addConstantPe(16);
  addConstantPe(17);

  const auto addLoopPe = [&](std::size_t attachmentOrdinal) {
    auto attachment = take(test, network.attachment(attachmentOrdinal));
    auto pe = take(test, spatial.addPe(attachment.inputs(),
                                       adg::PeSpec::spatial(loopInputPorts,
                                                            loopOutputPorts)));
    std::vector<adg::PeValue> inputs;
    inputs.reserve(loopInputPorts.size());
    for (std::size_t ordinal = 0; ordinal != loopInputPorts.size(); ++ordinal)
      inputs.push_back(take(test, pe.input(ordinal)));
    requireSuccess(
        test, adg::addLoopControlFu(pe, llvm::ArrayRef(inputs).take_front(4),
                                    ::dataflow::StreamStepKind::Add,
                                    ::dataflow::StreamStepKind::Sub));
    requireSuccess(test, pe.close());
    std::vector<adg::SpatialValue> outputs;
    outputs.reserve(loopOutputPorts.size());
    for (std::size_t ordinal = 0; ordinal != loopOutputPorts.size(); ++ordinal)
      outputs.push_back(take(test, pe.output(ordinal)));
    requireSuccess(test, attachment.connectOutputs(outputs));
  };
  addLoopPe(18);

  auto spatialAddAttachment = take(test, network.attachment(13));
  auto spatialAddPe =
      take(test,
           spatial.addPe(spatialAddAttachment.inputs(),
                         adg::PeSpec::spatial(demuxInputPorts, constantPorts)));
  std::vector<adg::PeValue> spatialAddInputs{take(test, spatialAddPe.input(0)),
                                             take(test, spatialAddPe.input(1))};
  auto spatialAddFu = take(
      test, spatialAddPe.addFu(spatialAddInputs,
                               adg::FuSpec{demuxInputPorts, constantPorts}));
  auto spatialAdd =
      take(test,
           spatialAddFu.addOperation(
               {take(test, spatialAddFu.input(0)),
                take(test, spatialAddFu.input(1))},
               adg::OperationCapabilitySpec{
                   ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                   ::fabric::ScalarIntegerParams{::fabric::IntegerWidthSet::get(
                       {::fabric::IntegerWidth::I32})},
                   {::dataflow::OperationSchemaId::ArithAddI},
                   constantPorts,
                   ::fabric::oneCycleElasticOperationResourceContract()}));
  requireSuccess(test, spatialAddFu.addCapabilityTemplate(
                           adg::FuCapabilityTemplateSpec{{spatialAdd}, {}}));
  requireSuccess(test, spatialAddFu.close({take(test, spatialAdd.output(0))}));
  requireSuccess(test, spatialAddPe.close());
  requireSuccess(test, spatialAddAttachment.connectOutputs(
                           {take(test, spatialAddPe.output(0))}));

  const std::size_t temporalPayloadInput = directNetworkInputCount;
  auto temporalInputFifo = take(
      test, spatial.addFifo(take(test, spatial.input(temporalPayloadInput)),
                            adg::FifoSpec{payload, 2, true, std::nullopt}));
  auto temporalInputFork =
      take(test, spatial.addSwitch({temporalInputFifo.value()},
                                   adg::SwitchSpec::spatial(constantPorts,
                                                            temporalForkOutputs,
                                                            {{0}, {0}})));
  std::vector<adg::SpatialValue> temporalInputs;
  temporalInputs.reserve(temporalMeshInputPorts.size());
  for (std::size_t ordinal = 0; ordinal != temporalMeshInputPorts.size();
       ++ordinal) {
    auto spatialToTemporal = take(
        test, spatial.addBoundary({boundaryAttachments[4].inputs()[ordinal]},
                                  adg::BoundarySpec::s2tWithConfiguredTag(
                                      payload, taggedPayload)));
    temporalInputs.push_back(spatialToTemporal.front());
  }
  const std::vector<adg::PortType> temporalInputTypes(2, payload);
  const std::vector<adg::PortType> temporalOutputTypes(2, taggedPayload);
  auto temporalPe = take(
      test,
      spatial.addPe(temporalInputs,
                    adg::PeSpec::temporal(
                        temporalInputTypes, temporalOutputTypes,
                        adg::TemporalPeParameters{
                            4, adg::FuConfigurationMode::PerInstruction,
                            ::fabric::OperandBufferMode::PerInstruction, 4,
                            adg::TemporalRegisterFifoParameters{2, 4, 2}})));
  std::vector<adg::PeValue> addInputs{take(test, temporalPe.input(0)),
                                      take(test, temporalPe.input(1))};
  auto addFu =
      take(test, temporalPe.addFu(addInputs,
                                  adg::FuSpec{demuxInputPorts, {payload}}));
  auto add =
      take(test,
           addFu.addOperation(
               {take(test, addFu.input(0)), take(test, addFu.input(1))},
               adg::OperationCapabilitySpec{
                   ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                   ::fabric::ScalarIntegerParams{::fabric::IntegerWidthSet::get(
                       {::fabric::IntegerWidth::I32})},
                   {::dataflow::OperationSchemaId::ArithAddI},
                   {payload},
                   ::fabric::oneCycleElasticOperationResourceContract()}));
  requireSuccess(test, addFu.addCapabilityTemplate(
                           adg::FuCapabilityTemplateSpec{{add}, {}}));
  requireSuccess(test, addFu.close({take(test, add.output(0))}));
  requireSuccess(test, temporalPe.close());
  std::vector<adg::SpatialValue> temporalPayloadOutputs;
  std::vector<adg::SpatialValue> temporalTagOutputs;
  for (std::size_t ordinal = 0; ordinal != temporalOutputTypes.size();
       ++ordinal) {
    auto temporalToSpatial =
        take(test, spatial.addBoundary(
                       {take(test, temporalPe.output(ordinal))},
                       adg::BoundarySpec::t2s(taggedPayload, {payload, tag})));
    if (ordinal == 0) {
      auto temporalOutputFifo =
          take(test, spatial.addFifo(temporalToSpatial[0],
                                     adg::FifoSpec{payload, 2, false,
                                                   std::nullopt}));
      temporalPayloadOutputs.push_back(temporalOutputFifo.value());
    } else {
      temporalPayloadOutputs.push_back(temporalToSpatial[0]);
    }
    temporalTagOutputs.push_back(temporalToSpatial[1]);
  }
  requireSuccess(test, boundaryAttachments[4].connectOutputs(
                           {temporalPayloadOutputs[0],
                            temporalPayloadOutputs[1], temporalInputFork[1]}));

  for (std::size_t ordinal = 0; ordinal != spatialMemoryOccurrenceCount;
       ++ordinal) {
    const std::size_t firstAttachmentOrdinal =
        ordinal == 0 ? 14 : 19 + (ordinal - 1) * 2;
    auto firstMemoryAttachment =
        take(test, network.attachment(firstAttachmentOrdinal));
    auto secondMemoryAttachment =
        take(test, network.attachment(firstAttachmentOrdinal + 1));
    std::vector<adg::SpatialValue> memoryInputs;
    memoryInputs.reserve(memory.inputTypes().size());
    memoryInputs.push_back(
        take(test, spatial.input(temporalPayloadInput + 1 + ordinal)));
    memoryInputs.insert(memoryInputs.end(),
                        firstMemoryAttachment.inputs().begin(),
                        firstMemoryAttachment.inputs().end());
    memoryInputs.insert(memoryInputs.end(),
                        secondMemoryAttachment.inputs().begin(),
                        secondMemoryAttachment.inputs().end());
    auto memoryOutputs = take(test, spatial.addMemory(memoryInputs, memory));
    std::vector<adg::SpatialValue> routedMemoryOutputs;
    routedMemoryOutputs.reserve(memoryOutputs.values().size());
    for (const adg::SpatialValue output : memoryOutputs.values()) {
      auto fifo =
          take(test, spatial.addFifo(
                         output,
                         adg::FifoSpec{payload, 2, true, std::nullopt}));
      routedMemoryOutputs.push_back(fifo.value());
    }
    requireSuccess(test, firstMemoryAttachment.connectOutputs(
                             llvm::ArrayRef(routedMemoryOutputs)
                                 .take_front(firstMemoryOutputCount)));
    requireSuccess(test, secondMemoryAttachment.connectOutputs(
                             llvm::ArrayRef(routedMemoryOutputs)
                                 .drop_front(firstMemoryOutputCount)));
  }

  std::vector<adg::SpatialValue> outputs;
  for (std::size_t bank = 0; bank != 4; ++bank)
    outputs.insert(outputs.end(), boundaryAttachments[bank].inputs().begin(),
                   boundaryAttachments[bank].inputs().end());
  outputs.insert(outputs.end(), temporalTagOutputs.begin(),
                 temporalTagOutputs.end());
  requireSuccess(test, spatial.close(outputs));
  auto design = take(test, std::move(builder).finalize());
  deployment::test::require(test, design.roots().size() == 1,
                            "heterogeneous fixture did not publish one Module");
  fabric::FinalizedFabricRoot module = design.roots().front();
  bool hasSpatialPe = false;
  bool hasTemporalPe = false;
  for (const auto pe : module.view().peOccurrences()) {
    hasSpatialPe |= module.view().peSchedule(pe) == ::fabric::Schedule::Spatial;
    hasTemporalPe |=
        module.view().peSchedule(pe) == ::fabric::Schedule::Temporal;
  }
  deployment::test::require(
      test,
      hasSpatialPe && hasTemporalPe &&
          !module.view().switchOccurrences().empty() &&
          !module.view().memoryOccurrences().empty() &&
          !module.view().fifoOccurrences().empty() &&
          !module.view().boundaryOccurrences().empty(),
      "portable fixture omits a required heterogeneous hierarchy component");
  return module;
}

fabric::FinalizedFabricRoot
buildSystem(llvm::StringRef test, const fabric::FinalizedFabricRoot &module,
            llvm::ArrayRef<mlir::Type> messagePayloads,
            ArtifactStore &artifacts,
            deployment::test::MappedSpatialSystemSpec spec) {
  return deployment::test::buildMappedSpatialSystem(
      test, module, messagePayloads, artifacts, spec);
}

pnr::ResolvedPnrConfigView spatialConfig(llvm::StringRef test) {
  ResolvedConfig config = defaultResolvedConfig();
  config.dse.spatialPnr.temporaryViolations.admitted = {
      ResolvedPnrViolationKind::UnroutedObligation,
      ResolvedPnrViolationKind::CapacityOveruse};
  const auto &selection = config.dse.spatialPnr.objectiveSelection;
  config.dse.spatialPnr.objectiveSelection.selectedSearchEnergy =
      config.dse.objectiveCatalogs
          .totalOrderings[selection.selectedTotalOrdering]
          .weightedLevels.front();
  auto &search = config.dse.spatialPnr.search;
  search.initializer.seedAttemptCount = 1;
  search.actionProposal = {1, 0, 0};
  search.annealing.calibrationProposalCount = 1;
  search.annealing.fallbackTemperature = 1;
  search.annealing.minimumTemperature = 1;
  search.annealing.coolingRatio = {1, 2};
  search.annealing.proposalsPerLevelBase = 1;
  search.annealing.proposalsPerMovableDecision = 0;
  return take(test, pnr::projectResolvedSpatialPnrConfigView(config));
}

ArtifactRootReference generateTechMapping(llvm::StringRef test,
                                          const ArtifactRootReference &dataflow,
                                          const ArtifactRootReference &fabric,
                                          ArtifactStore &artifacts,
                                          const BlobStore &blobs) {
  ResolvedConfig config = defaultResolvedConfig();
  config.dse.techMapping.candidatePublicationLimit = 1;
  auto view = take(test, mapping::projectResolvedTechMappingConfigView(config));
  auto inputs =
      take(test, dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
                     {dataflow}, fabric));
  auto binding = take(
      test, dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(view));
  auto outcome = take(
      test, dse::invokeCandidateGenerator(inputs, binding, artifacts, blobs));
  const std::vector<dse::CandidateGeneratorOutputBinding> *outputs = nullptr;
  const auto *completed =
      std::get_if<dse::CompletedCandidateGeneratorResult>(&outcome.outcome);
  if (completed)
    outputs = &completed->outputBindings;
  if (const auto *incomplete =
          std::get_if<dse::IncompleteCandidateGeneratorResult>(
              &outcome.outcome);
      incomplete &&
      incomplete->reason ==
          dse::CandidateGeneratorIncompleteReason::SemanticLimitReached)
    outputs = &incomplete->retainedOutputBindings;
  deployment::test::require(
      test,
      outputs && outputs->size() == 1 && outputs->front().artifacts.size() == 1,
      "TechMapping fixture did not publish one candidate");
  return outputs->front().artifacts.front();
}

llvm::Expected<dse::CandidateGeneratorProviderResult>
invokeRootCompleteSpatialPnr(
    llvm::ArrayRef<ArtifactRootReference> techMappings,
    const ArtifactRootReference &fabric,
    const pnr::ResolvedPnrConfigView &spatialPnrConfig,
    const ExecutionControlView &executionControl, ArtifactStore &artifacts,
    const BlobStore &blobs) {
  auto fabricRoot = fabric::importEntireFabricRoot(fabric, artifacts);
  if (!fabricRoot)
    return fabricRoot.takeError();
  auto physicalTiming = fabric::projectNormalizedFabricPhysicalTimingProfile(
      fabricRoot->view());
  if (!physicalTiming)
    return physicalTiming.takeError();
  auto physicalTimingReference =
      fabric::publishFabricPhysicalTimingProfile(*physicalTiming, artifacts);
  if (!physicalTimingReference)
    return physicalTimingReference.takeError();
  auto inputs = dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
      techMappings, fabric, *physicalTimingReference);
  if (!inputs)
    return inputs.takeError();
  auto binding = dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
      spatialPnrConfig);
  if (!binding)
    return binding.takeError();
  return dse::invokeCandidateGenerator(*inputs, *binding, artifacts, blobs,
                                       executionControl);
}

ArtifactRootReference
generateSpatialMapping(llvm::StringRef test, mlir::MLIRContext &context,
                       const ArtifactRootReference &techMapping,
                       const ArtifactRootReference &fabric,
                       const pnr::ResolvedPnrConfigView &spatialPnrConfig,
                       const ExecutionControlView &executionControl,
                       ArtifactStore &artifacts, const BlobStore &blobs,
                       MappedRtlRouteCoverage routeCoverage) {
  auto outcome = take(test, invokeRootCompleteSpatialPnr(
                                {techMapping}, fabric, spatialPnrConfig,
                                executionControl, artifacts, blobs));
  const auto *completed =
      std::get_if<dse::CompletedCandidateGeneratorResult>(&outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.empty()) {
    std::string diagnostic =
        completed ? "completed_without_candidate" : "incomplete";
    if (const auto *incomplete =
            std::get_if<dse::IncompleteCandidateGeneratorResult>(
                &outcome.outcome))
      diagnostic +=
          " reason=" +
          std::to_string(static_cast<std::uint32_t>(incomplete->reason));
    for (const dse::CandidateGeneratorWorkUnitSummary &summary :
         outcome.workSummary)
      diagnostic += " work[" + std::to_string(summary.unit.ordinal()) +
                    "]=" + std::to_string(summary.consumed) + "/" +
                    std::to_string(summary.planned);
    deployment::test::fail(
        test,
        "SpatialMapping fixture did not publish a candidate: " + diagnostic);
  }
  const ArtifactRootReference initial =
      completed->outputBindings.front().artifacts.front();
  auto initialMapping =
      take(test, mapping::importSpatialMapping(initial, artifacts));
  if (routeCoverage == MappedRtlRouteCoverage::AnyLegal)
    return initial;
  auto fabricRoot =
      take(test, fabric::importEntireFabricRoot(fabric, artifacts));
  auto physicalTiming = take(
      test,
      fabric::projectNormalizedFabricPhysicalTimingProfile(fabricRoot.view()));
  const auto physicalTimingReference =
      take(test, fabric::publishFabricPhysicalTimingProfile(physicalTiming,
                                                            artifacts));
  const auto containsBypass = [](const mapping::SpatialMappingView &mapping) {
    const auto isBypass = [](const auto &traversal) {
      if (!traversal)
        return false;
      const auto *fifo =
          std::get_if<fabric::FabricFifoTraversalPayload>(&traversal->payload);
      return fifo && fifo->mode == fabric::FabricFifoTraversalMode::Bypass;
    };
    for (const mapping::SpatialRouteTreeView &route : mapping.routeTrees()) {
      if (isBypass(route.localTraversal))
        return true;
      for (const mapping::SpatialRouteNodeView &node : route.nodes)
        if (isBypass(node.incomingTraversal))
          return true;
      for (const mapping::SpatialRouteSinkView &sink : route.sinks)
        if (isBypass(sink.localTraversal))
          return true;
    }
    return false;
  };
  if (containsBypass(initialMapping.view()))
    return initial;

  auto tech = take(test, mapping::importTechMapping(techMapping, artifacts));
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      tech.view().dataflowIdentity()};
  auto dataflowArtifact = take(
      test, dataflow::importCanonicalDataflow(dataflowReference, artifacts));
  auto dataflowView = take(test, dataflowArtifact.view());

  const auto byteList = [](llvm::ArrayRef<std::uint8_t> bytes) {
    std::string text = "[";
    for (const auto [ordinal, byte] : llvm::enumerate(bytes)) {
      if (ordinal)
        text += ", ";
      text += std::to_string(static_cast<std::int8_t>(byte));
    }
    return text + "]";
  };
  const auto identityAttr = [&](const ArtifactIdentity &identity) {
    return "#mapping.artifact_identity<" + byteList(identity.bytes()) + ">";
  };
  const auto producerAttr =
      [&](const dataflow::CanonicalGraphProducerEndpointRef &producer) {
        auto encoded = take(test, dataflow::encodeDataflowReference(
                                      dataflowView.identity(), producer));
        return "#mapping.graph_producer_endpoint_ref<" + byteList(encoded) +
               ">";
      };
  const auto consumerAttr =
      [&](const dataflow::CanonicalGraphConsumerEndpointRef &consumer) {
        auto encoded = take(test, dataflow::encodeDataflowReference(
                                      dataflowView.identity(), consumer));
        return "#mapping.graph_consumer_endpoint_ref<" + byteList(encoded) +
               ">";
      };
  const auto endpointAttr =
      [&](const fabric::FabricTransportEndpointRef &endpoint) {
        return "#mapping.fabric_transport_endpoint_ref<" +
               byteList(fabric::canonicalFabricBytes(endpoint)) + ">";
      };
  const auto traversalAttr =
      [&](const fabric::FabricPhysicalTraversalRef &traversal) {
        return "#mapping.fabric_physical_traversal_ref<" +
               byteList(fabric::canonicalFabricBytes(traversal)) + ">";
      };
  const auto isAdmitted = [&](const fabric::FabricPhysicalTraversalRef &value) {
    return llvm::is_contained(fabricRoot.view().admittedTraversals(), value);
  };

  const auto attachmentRestriction =
      [&](llvm::StringRef terminal,
          const fabric::FabricTransportEndpointRef &endpoint) {
        return "    mapping.constraint.domain_restriction "
               "projection(spatial_transfer_attachment) subject(" +
               terminal.str() + ") admissible_domain([" +
               endpointAttr(endpoint) + "])\n";
      };
  for (const mapping::SpatialRouteTreeView &route :
       initialMapping.view().routeTrees()) {
    std::vector<fabric::FabricPhysicalTraversalRef> selected;
    const auto append = [&](const auto &traversal) {
      if (traversal)
        selected.push_back(*traversal);
    };
    append(route.localTraversal);
    for (const mapping::SpatialRouteNodeView &node : route.nodes)
      append(node.incomingTraversal);
    for (const mapping::SpatialRouteSinkView &sink : route.sinks)
      append(sink.localTraversal);

    for (const fabric::FabricPhysicalTraversalRef &candidate : selected) {
      const auto *fifo =
          std::get_if<fabric::FabricFifoTraversalPayload>(&candidate.payload);
      if (!fifo || fifo->mode != fabric::FabricFifoTraversalMode::Buffered)
        continue;
      const fabric::FabricPhysicalTraversalRef bypass =
          fabric::FabricPhysicalTraversalRef::fifoTraversal(
              fifo->owner, fabric::FabricFifoTraversalMode::Bypass);
      if (!isAdmitted(bypass))
        continue;
      std::vector<fabric::FabricPhysicalTraversalRef> domain;
      domain.reserve(selected.size());
      for (const fabric::FabricPhysicalTraversalRef &traversal : selected) {
        const auto *selectedFifo =
            std::get_if<fabric::FabricFifoTraversalPayload>(&traversal.payload);
        domain.push_back(selectedFifo && selectedFifo->owner == fifo->owner
                             ? bypass
                             : traversal);
      }
      llvm::sort(domain, [](const auto &lhs, const auto &rhs) {
        return fabric::canonicalFabricBytes(lhs) <
               fabric::canonicalFabricBytes(rhs);
      });
      domain.erase(std::unique(domain.begin(), domain.end()), domain.end());
      std::string domainText = "[";
      for (const auto [ordinal, traversal] : llvm::enumerate(domain)) {
        if (ordinal)
          domainText += ", ";
        domainText += traversalAttr(traversal);
      }
      domainText += "]";
      std::string attachmentClauses = attachmentRestriction(
          "#mapping.spatial_transfer_terminal<producer = " +
              producerAttr(route.logicalNet) + ">",
          route.rootEndpoint);
      for (const mapping::SpatialRouteSinkView &sink : route.sinks) {
        const auto node = llvm::find_if(
            route.nodes, [&](const mapping::SpatialRouteNodeView &candidate) {
              return candidate.ordinal == sink.nodeOrdinal;
            });
        deployment::test::require(
            test, node != route.nodes.end(),
            "initial Mapping sink names an absent route node");
        attachmentClauses += attachmentRestriction(
            "#mapping.spatial_transfer_terminal<producer = " +
                producerAttr(route.logicalNet) +
                ", consumer = " + consumerAttr(sink.sink) + ">",
            node->endpoint);
      }
      const std::string source =
          "module {\n  mapping.constraints.spatial dataflow(" +
          identityAttr(dataflowView.identity()) + ") tech_mapping(" +
          identityAttr(tech.view().identity()) + ") fabric(" +
          identityAttr(fabricRoot.view().identity()) + ") {\n" +
          attachmentClauses +
          "    mapping.constraint.domain_restriction "
          "projection(net_selected_physical_traversals) subject(" +
          producerAttr(route.logicalNet) + ") admissible_domain(" + domainText +
          ")\n  }\n}\n";
      auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
      deployment::test::require(test, static_cast<bool>(module),
                                "cannot parse bypass MappingConstraintSet");
      auto roots = module->getOps<::mapping::ConstraintsSpatialOp>();
      auto constraints =
          take(test, mapping::finalizeSpatialMappingConstraintSet(
                         *roots.begin(), dataflowView, tech.view(),
                         fabricRoot.view(), artifacts));
      auto constrainedInputs =
          take(test, dse::bindSpatialPnrCandidateGeneratorInputs(
                         dataflowReference, techMapping, fabric,
                         physicalTimingReference, constraints.reference()));
      auto constrainedBinding = take(
          test,
          dse::resolveSpatialPnrCandidateGeneratorBinding(spatialPnrConfig));
      auto constrainedOutcome = take(
          test,
          dse::invokeCandidateGenerator(constrainedInputs, constrainedBinding,
                                        artifacts, blobs, executionControl));
      const auto *constrained =
          std::get_if<dse::CompletedCandidateGeneratorResult>(
              &constrainedOutcome.outcome);
      if (!constrained || constrained->outputBindings.size() != 1)
        continue;
      for (const ArtifactRootReference &reference :
           constrained->outputBindings.front().artifacts) {
        auto mapping = mapping::importSpatialMapping(reference, artifacts);
        if (!mapping) {
          llvm::consumeError(mapping.takeError());
          continue;
        }
        if (containsBypass(mapping->view()))
          return reference;
      }
    }
  }
  deployment::test::fail(
      test, "real Spatial PnR could not realize a legal bypass FIFO route");
}

std::pair<ArtifactRootReference, ArtifactRootReference>
publishSpatialInputs(llvm::StringRef test,
                     const dataflow::CanonicalDataflowArtifact &dataflow,
                     ArtifactStore &artifacts) {
  const auto view = take(test, dataflow.view());
  const dataflow::RootedGraphLaunchRef launch{
      view.rootThreadLaunches().front().ref,
      view.staticGraphLaunches().front().ref};
  sim::SpatialSimulationWorkload workloadDraft{launch};
  workloadDraft.valueInputPlan = {
      sim::RuntimeValueInput{}, sim::RuntimeValueInput{},
      sim::RuntimeValueInput{}, sim::RuntimeValueInput{}};
  workloadDraft.observableContract.valueResults = {0};
  workloadDraft.observableContract.streamOutputs = {0, 1};
  deployment::test::require(test, view.logicalMemoryRoots().size() == 1,
                            "mapped fixture has no unique memory root");
  const dataflow::LogicalMemoryRootRef memoryRoot =
      view.logicalMemoryRoots().front().ref;
  workloadDraft.observableContract.memories = {
      {dataflow::LogicalMemoryRootOrViewRef{memoryRoot},
       sim::MemoryObservationForm::FullState}};
  auto workload =
      take(test, sim::finalizeSimulationWorkload(workloadDraft, view));
  sim::SpatialSimulationRuntimeInputDraft runtimeDraft{workload.identity()};
  runtimeDraft.runtimeValues = {
      {0, {1, {sim::SemanticLane::defined(llvm::APInt(32, 7))}}},
      {1, {1, {sim::SemanticLane::defined(llvm::APInt(32, 5))}}},
      {2, {1, {sim::SemanticLane::defined(llvm::APInt(64, 0))}}},
      {3, {1, {sim::SemanticLane::defined(llvm::APInt(32, 0x44332211))}}}};
  sim::CanonicalStreamSequence stream0;
  stream0.values = {1, {sim::SemanticLane::defined(llvm::APInt(32, 0x21))}};
  stream0.termination = sim::StreamTermination::ClosedAfterLast;
  sim::CanonicalStreamSequence stream1;
  stream1.values = {1, {sim::SemanticLane::defined(llvm::APInt(32, 0x31))}};
  stream1.termination = sim::StreamTermination::ClosedAfterLast;
  runtimeDraft.runtimeStreams = {std::move(stream0), std::move(stream1)};
  runtimeDraft.memoryObjects = {
      sim::RuntimeMemoryObject(std::vector<sim::SemanticMemoryByte>(
          16, sim::SemanticMemoryByte{sim::SemanticState::Defined, 0}))};
  runtimeDraft.memoryRootBindings = {{memoryRoot, 0, 0}};
  auto runtime = take(
      test, sim::finalizeSimulationRuntimeInput(runtimeDraft, workload, view));
  return {take(test, sim::publishSimulationWorkload(workload, artifacts)),
          take(test, sim::publishSimulationRuntimeInput(runtime, artifacts))};
}

std::vector<hardware::FinalizedHardwareImplementation>
buildImplementation(llvm::StringRef test,
                    const fabric::FinalizedFabricRoot &module,
                    const fabric::FinalizedFabricRoot &system,
                    mlir::MLIRContext &relationContext,
                    ArtifactStore &artifacts, BlobStore &blobs) {
  const auto inventoryOwner = [](const auto &owner) {
    return std::visit(
        [](const auto &value) -> fabric::FabricInventoryOwnerRef {
          using Type = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<Type, fabric::LocalMemoryServiceRef>)
            return fabric::FabricInventoryOwnerRef::of(value.underlying());
          else
            return fabric::FabricInventoryOwnerRef::of(value);
        },
        owner.payload());
  };
  std::vector<hardware::test::ConfigurationFieldEncodingOverride> overrides;
  for (const fabric::FabricModuleDomainMemberRef &member :
       module.view().moduleDomainMembers()) {
    if (member.kind() != fabric::FabricModuleDomainMemberKind::Internal)
      continue;
    const auto &localOwner =
        std::get<fabric::FabricModulePhysicalOwnerRef>(member.payload);
    const fabric::FabricInventoryOwnerRef owner = inventoryOwner(localOwner);
    const std::uint64_t fieldCount = module.view().inventorySize(
        owner, fabric::FabricInventoryKind::SemanticConfigField);
    for (fabric::FabricOrdinal ordinal = 0; ordinal < fieldCount; ++ordinal) {
      const fabric::FabricSemanticConfigFieldRef localField{
          fabric::FabricConfigurationOwnerRef(owner), ordinal};
      auto relation = take(test, module.view().semanticFieldRelation(
                                     localField, relationContext));
      if (relation.kind() != fabric::FabricSemanticFieldRelationKind::Direct)
        continue;
      const auto width = relation.directEncodedBitCount();
      deployment::test::require(
          test, width.has_value(),
          "direct Fabric relation has no exact carrier width");
      std::vector<std::uint8_t> inactive = directInactiveValue(
          test, module.view(), localField, relation, relationContext);
      for (const fabric::AccCoreOccurrenceRef core :
           system.view().accCoreOccurrences()) {
        auto target = take(
            test, fabric::FabricModulePhysicalTargetRef::create(localField));
        auto physical =
            take(test, fabric::FabricPhysicalConfigurationFieldRef::create(
                           fabric::SpatialCoreInternalOccurrenceRef{
                               fabric::SpatialCoreOccurrenceRef{core},
                               std::move(target)}));
        overrides.push_back({std::move(physical),
                             hardware::DirectBitsEncoding{*width}, inactive});
      }
    }
  }
  auto abiDraft = take(test, hardware::test::makeCompleteConfigurationABIDraft(
                                 system, overrides));
  auto abi = take(
      test, hardware::finalizeConfigurationABI(std::move(abiDraft), artifacts));
  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  hardware::rtl::FabricOperationProviderRegistry providers;
  requireSuccess(test,
                 hardware::rtl::registerPortableOperationProviders(providers));
  hardware::ExternalImplementationContractCatalog contracts;
  const auto accCores = system.view().accCoreOccurrences();
  deployment::test::require(test, !accCores.empty(),
                            "mapped fixture System has no SpatialCore");
  std::vector<hardware::FinalizedHardwareImplementation> implementations;
  implementations.reserve(accCores.size());
  for (fabric::AccCoreOccurrenceRef accCore : accCores)
    implementations.push_back(take(
        test, hardware::rtl::finalizePortableSpatialCoreHardwareImplementation(
                  context, abi, fabric::SpatialCoreOccurrenceRef{accCore},
                  std::nullopt, providers, contracts, artifacts, blobs)));
  return implementations;
}

evaluation::CaseArtifactResolution
buildResolution(llvm::StringRef test, const ArtifactRootReference &dataflow,
                const fabric::FinalizedFabricRoot &module,
                const fabric::FinalizedFabricRoot &system,
                const ArtifactRootReference &techMapping,
                const ArtifactRootReference &spatialMapping,
                const hardware::FinalizedHardwareImplementation &implementation,
                const deployment::FinalizedDeployment &deployment,
                const ArtifactRootReference &workload,
                const ArtifactRootReference &runtimeInput) {
  return take(
      test, evaluation::CaseArtifactResolution::get(
                {{dataflow, {}},
                 {module.reference(), {}},
                 {system.reference(), {module.reference()}},
                 {techMapping, {dataflow, module.reference()}},
                 {spatialMapping, {dataflow, module.reference(), techMapping}},
                 {implementation.reference(), {system.reference()}},
                 {deployment.reference(),
                  {dataflow, system.reference(), spatialMapping,
                   implementation.reference()}},
                 {workload, {dataflow}},
                 {runtimeInput, {dataflow, workload}}}));
}

MappedSpatialMappingFixture
mapSpatialModule(llvm::StringRef test, const ArtifactRootReference &dataflow,
                 fabric::FinalizedFabricRoot module, mlir::MLIRContext &context,
                 const pnr::ResolvedPnrConfigView &spatialPnrConfig,
                 const ExecutionControlView &executionControl,
                 ArtifactStore &artifacts, BlobStore &blobs,
                 MappedRtlRouteCoverage routeCoverage) {
  const ArtifactRootReference techMapping =
      generateTechMapping(test, dataflow, module.reference(), artifacts, blobs);
  const ArtifactRootReference spatialMappingReference = generateSpatialMapping(
      test, context, techMapping, module.reference(), spatialPnrConfig,
      executionControl, artifacts, blobs, routeCoverage);
  auto spatialMapping = take(
      test, mapping::importSpatialMapping(spatialMappingReference, artifacts));
  return {std::move(module), techMapping, std::move(spatialMapping)};
}

} // namespace

MappedSpatialMappingFixture buildMappedSpatialMappingFixture(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    mlir::MLIRContext &context, ArtifactStore &artifacts, BlobStore &blobs,
    MappedRtlFixtureTopology topology, MappedRtlRouteCoverage routeCoverage,
    std::size_t spatialMemoryOccurrenceCount) {
  const ArtifactRootReference dataflowReference =
      take(test, dataflow::publishCanonicalDataflow(dataflow, artifacts));
  auto module =
      buildSpatialCore(test, artifacts, topology, spatialMemoryOccurrenceCount);
  const pnr::ResolvedPnrConfigView config = spatialConfig(test);
  const ExecutionControlView executionControl;
  return mapSpatialModule(test, dataflowReference, std::move(module), context,
                          config, executionControl, artifacts, blobs,
                          routeCoverage);
}

MappedSpatialMappingFixture buildMappedBuiltinSpatialMappingFixture(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    const adg::BuiltinTargetScale &scale, mlir::MLIRContext &context,
    const pnr::ResolvedPnrConfigView &spatialPnrConfig,
    const ExecutionControlView &executionControl, ArtifactStore &artifacts,
    BlobStore &blobs, MappedRtlRouteCoverage routeCoverage) {
  const ArtifactRootReference dataflowReference =
      take(test, dataflow::publishCanonicalDataflow(dataflow, artifacts));
  auto module = buildBuiltinSpatialCore(test, artifacts, scale);
  return mapSpatialModule(test, dataflowReference, std::move(module), context,
                          spatialPnrConfig, executionControl, artifacts, blobs,
                          routeCoverage);
}

llvm::Expected<MappedBuiltinSpatialPnrInvocation>
invokeMappedBuiltinSpatialPnrFixture(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    const adg::BuiltinTargetScale &scale,
    const mapping::ResolvedTechMappingConfigView &techMappingConfig,
    const pnr::ResolvedPnrConfigView &spatialPnrConfig,
    const ExecutionControlView &executionControl, ArtifactStore &artifacts,
    BlobStore &blobs) {
  const ArtifactRootReference dataflowReference =
      take(test, dataflow::publishCanonicalDataflow(dataflow, artifacts));
  auto module = buildBuiltinSpatialCore(test, artifacts, scale);
  auto techInputs = dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
      {dataflowReference}, module.reference());
  if (!techInputs)
    return techInputs.takeError();
  auto techBinding =
      dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
          techMappingConfig);
  if (!techBinding)
    return techBinding.takeError();
  auto techResult = dse::invokeCandidateGenerator(
      *techInputs, *techBinding, artifacts, blobs, executionControl);
  if (!techResult)
    return techResult.takeError();
  const std::vector<dse::CandidateGeneratorOutputBinding> *techOutputs = nullptr;
  if (const auto *completed =
          std::get_if<dse::CompletedCandidateGeneratorResult>(
              &techResult->outcome))
    techOutputs = &completed->outputBindings;
  else
    techOutputs =
        &std::get<dse::IncompleteCandidateGeneratorResult>(techResult->outcome)
             .retainedOutputBindings;
  if (techOutputs->size() != 1)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "mapped builtin TechMapping provider changed its output shape");
  if (techOutputs->front().artifacts.empty())
    return MappedBuiltinSpatialPnrInvocation{
        std::move(module), std::move(*techResult), std::nullopt};
  auto spatialResult = invokeRootCompleteSpatialPnr(
      techOutputs->front().artifacts, module.reference(),
      spatialPnrConfig, executionControl, artifacts, blobs);
  if (!spatialResult)
    return spatialResult.takeError();
  return MappedBuiltinSpatialPnrInvocation{std::move(module),
                                           std::move(*techResult),
                                           std::move(*spatialResult)};
}

llvm::Expected<MappedSpatialMappingRepairFixture>
rerouteMappedSpatialMappingFixture(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    const MappedSpatialMappingFixture &parent,
    const dse::SpatialTransportRepairAlternative &alternative,
    const pnr::ResolvedPnrConfigView &spatialPnrConfig,
    const ExecutionControlView &executionControl, ArtifactStore &artifacts,
    BlobStore &blobs) {
  auto dataflowView = take(test, dataflow.view());
  auto tech =
      take(test, mapping::importTechMapping(parent.techMapping, artifacts));
  std::vector<fabric::FabricPhysicalTraversalRef> domain;
  domain.reserve(parent.module.view().admittedTraversals().size());
  for (const auto &traversal : parent.module.view().admittedTraversals())
    if (traversal != alternative.forbiddenTraversal)
      domain.push_back(traversal);
  deployment::test::require(
      test,
      domain.size() + 1 == parent.module.view().admittedTraversals().size(),
      "transport repair did not exclude one admitted traversal");
  auto constraints =
      take(test, mapping::finalizeSpatialNetTraversalDomainConstraintSet(
                     dataflowView, tech.view(), parent.module.view(),
                     alternative.producer, domain, artifacts));
  auto physicalTiming =
      take(test, fabric::projectNormalizedFabricPhysicalTimingProfile(
                     parent.module.view()));
  const ArtifactRootReference physicalTimingReference =
      take(test, fabric::publishFabricPhysicalTimingProfile(physicalTiming,
                                                            artifacts));
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version, dataflowView.identity()};
  auto inputs =
      take(test,
           dse::bindSpatialPnrCandidateGeneratorInputs(
               dataflowReference, parent.techMapping, parent.module.reference(),
               physicalTimingReference, constraints.reference()));
  auto binding = take(
      test, dse::resolveSpatialPnrCandidateGeneratorBinding(spatialPnrConfig));
  auto outcome =
      take(test, dse::invokeCandidateGenerator(inputs, binding, artifacts,
                                               blobs, executionControl));
  const auto *completed =
      std::get_if<dse::CompletedCandidateGeneratorResult>(&outcome.outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 1)
    return MappedSpatialMappingRepairFixture{std::nullopt,
                                             constraints.reference(),
                                             std::move(outcome)};
  auto child =
      take(test,
           mapping::importSpatialMapping(
               completed->outputBindings.front().artifacts.front(), artifacts));
  requireSuccess(test, mapping::admitSpatialMappingConstraints(
                           dataflowView, tech.view(), parent.module.view(),
                           constraints.view(), child.view()));
  deployment::test::require(
      test, child.reference() != parent.spatialMapping.reference(),
      "transport repair reproduced its excluded parent Mapping");
  return MappedSpatialMappingRepairFixture{std::move(child),
                                           constraints.reference(),
                                           std::move(outcome)};
}

fabric::FinalizedFabricRoot
buildMappedBuiltinSystemFixture(llvm::StringRef test,
                                const fabric::FinalizedFabricRoot &module,
                                ArtifactStore &artifacts) {
  return buildMappedBuiltinSystemFixture(test, adg::builtinLargeTarget.scale,
                                         module, artifacts);
}

fabric::FinalizedFabricRoot buildMappedBuiltinSystemFixture(
    llvm::StringRef test, const adg::BuiltinTargetScale &scale,
    const fabric::FinalizedFabricRoot &module, ArtifactStore &artifacts) {
  adg::DesignBuilder builder(artifacts);
  auto system = take(test, adg::expandBuiltinSystem(builder, scale, module));
  requireSuccess(test, system.close());
  auto design = take(test, std::move(builder).finalize());
  deployment::test::require(test, design.roots().size() == 1,
                            "builtin fixture did not publish one System");
  return design.roots().front();
}

MappedSpatialHardwareFixture buildMappedSpatialHardwareFixture(
    llvm::StringRef test, const dataflow::CanonicalDataflowArtifact &dataflow,
    mlir::MLIRContext &context, ArtifactStore &artifacts, BlobStore &blobs,
    deployment::test::MappedSpatialSystemSpec systemSpec,
    MappedRtlFixtureTopology topology, MappedRtlRouteCoverage routeCoverage,
    MappedSystemInterconnect interconnectKind,
    MappedSpatialHardwareFixtureObserver observer,
    std::size_t spatialMemoryOccurrenceCount) {
  const ArtifactRootReference dataflowReference =
      observeMappedSpatialHardwareFixtureOperation(
          observer, MappedSpatialHardwareFixtureOperation::DataflowPublication,
          [&] {
            return take(
                test, dataflow::publishCanonicalDataflow(dataflow, artifacts));
          });
  auto module = observeMappedSpatialHardwareFixtureOperation(
      observer,
      MappedSpatialHardwareFixtureOperation::
          FabricModuleConstructionAndFinalization,
      [&] {
        return buildSpatialCore(test, artifacts, topology,
                                spatialMemoryOccurrenceCount);
      });
  const ArtifactRootReference techMapping =
      observeMappedSpatialHardwareFixtureOperation(
          observer, MappedSpatialHardwareFixtureOperation::TechMapping, [&] {
            return generateTechMapping(test, dataflowReference,
                                       module.reference(), artifacts, blobs);
          });
  auto spatialMapping = observeMappedSpatialHardwareFixtureOperation(
      observer, MappedSpatialHardwareFixtureOperation::SpatialPnr, [&] {
        const pnr::ResolvedPnrConfigView config = spatialConfig(test);
        const ExecutionControlView executionControl;
        const ArtifactRootReference reference = generateSpatialMapping(
            test, context, techMapping, module.reference(), config,
            executionControl, artifacts, blobs, routeCoverage);
        return take(test, mapping::importSpatialMapping(reference, artifacts));
      });
  const std::array<mlir::Type, 3> messagePayloads{
      mlir::NoneType::get(&context), mlir::IntegerType::get(&context, 32),
      mlir::IndexType::get(&context)};
  auto systemAndInterconnect = observeMappedSpatialHardwareFixtureOperation(
      observer,
      MappedSpatialHardwareFixtureOperation::
          SystemFabricAndInterconnectConstruction,
      [&] {
        auto system =
            buildSystem(test, module, messagePayloads, artifacts, systemSpec);
        std::optional<ArtifactRootReference> interconnect;
        if (interconnectKind == MappedSystemInterconnect::Gem5EventTransport)
          interconnect =
              take(test, fabric::finalizeGem5EventInterconnectImplementation(
                             system.reference(), artifacts))
                  .reference();
        return std::pair(std::move(system), std::move(interconnect));
      });
  auto system = std::move(systemAndInterconnect.first);
  auto interconnect = std::move(systemAndInterconnect.second);
  auto implementations = observeMappedSpatialHardwareFixtureOperation(
      observer,
      MappedSpatialHardwareFixtureOperation::
          ConfigurationAbiAndHardwareImplementationGeneration,
      [&] {
        return buildImplementation(test, module, system, context, artifacts,
                                   blobs);
      });
  return {std::move(module), techMapping, std::move(spatialMapping),
          std::move(system), std::move(interconnect),
          std::move(implementations)};
}

MappedRtlRequestFixture
buildMappedRtlRequestFixture(llvm::StringRef test,
                             llvm::StringRef stableSimulatorBuildIdentity,
                             ArtifactStore &artifacts, BlobStore &blobs,
                             const deployment::test::TemporaryTree &tree,
                             MappedRtlFixtureTopology topology) {
  requireSuccess(test, evaluation::models::registerMappedRtlSimulationModel());
  mlir::MLIRContext dataflowContext = makeDataflowContext();
  auto dataflow = buildDataflow(test, dataflowContext);
  auto hardware = buildMappedSpatialHardwareFixture(
      test, dataflow, dataflowContext, artifacts, blobs,
      deployment::test::MappedSpatialSystemSpec{
          2, false, topology != MappedRtlFixtureTopology::Minimal},
      topology);
  const ArtifactRootReference dataflowReference =
      take(test, dataflow::publishCanonicalDataflow(dataflow, artifacts));
  const ArtifactRootReference spatialMapping =
      hardware.spatialMapping.reference();
  auto deployment = deployment::test::buildMappedSpatialDeployment(
      test, dataflow, hardware.system, hardware.spatialMapping,
      hardware.implementations, artifacts, blobs, tree);
  deployment::test::require(
      test, deployment.deployment().hardwareBindings().size() == 1,
      "mapped RTL request did not select one SpatialCore implementation");
  auto implementation = take(
      test, hardware::importHardwareImplementation(deployment.deployment()
                                                       .hardwareBindings()
                                                       .front()
                                                       .hardwareImplementation,
                                                   artifacts, blobs));
  const auto [workload, runtimeInput] =
      publishSpatialInputs(test, dataflow, artifacts);
  auto resolution =
      buildResolution(test, dataflowReference, hardware.module, hardware.system,
                      hardware.techMapping, spatialMapping, implementation,
                      deployment, workload, runtimeInput);

  auto subjects = take(
      test,
      evaluation::EvaluationSubjectBindings::get(
          {{evaluation::models::mappedRtlHardwareImplementationSubjectRole(),
            {implementation.reference()}},
           {evaluation::models::mappedRtlDeploymentSubjectRole(),
            {deployment.reference()}}}));
  auto evaluationCase =
      take(test, evaluation::EvaluationCase::get(
                     evaluation::mappedRtlSimulationCaseSignatureRef(),
                     std::move(subjects), workload, runtimeInput, {},
                     resolution, artifacts, blobs));
  auto cycleCount = take(
      test, evaluation::MetricRequest::get(
                {evaluation::MetricKind::CycleCount,
                 evaluation::EvaluationScope{evaluation::ScopeFormRef(0), {}}},
                {}, evaluationCase, resolution, artifacts));
  ResolvedConfig config = defaultResolvedConfig();
  config.evaluation.mappedRtlSimulator =
      evaluation::models::MappedRtlSimulatorBinding{
          stableSimulatorBuildIdentity.str()};
  auto model =
      take(test, evaluation::ResolvedModelBinding::project(
                     evaluation::models::mappedRtlSimulatorModelDescriptorRef(),
                     {}, config));
  auto request =
      take(test, evaluation::EvaluationRequest::get(
                     evaluationCase, {cycleCount}, {}, std::move(model), 0,
                     resolution, artifacts, blobs));
  const ArtifactRootReference published =
      take(test, evaluation::publishEvaluationRequest(request, artifacts));
  deployment::test::require(
      test, published == evaluation::evaluationRequestReference(request),
      "request publication changed its identity");
  return {std::move(request),
          std::move(resolution),
          std::move(implementation),
          std::move(deployment),
          workload,
          runtimeInput,
          hardware.module.reference(),
          hardware.techMapping,
          spatialMapping};
}

} // namespace loom::eda::test

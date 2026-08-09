#include "ADG/Builder.h"
#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FixedVectorParallelizeSerialize.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/OperationResourceContract.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FabricFuOccurrenceNodeRef;
using loom::fabric::FinalizedFabricRoot;
using namespace loom::hardware;
using namespace loom::hardware::rtl;

enum class AdapterKind { Parallelize, Serialize };
enum class ContractKind { OrderedCardinality, LegacyOneCycle };

struct FabricFixture final {
  AdapterKind kind;
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

struct AdapterMode final {
  unsigned elementWidth = 0;
  unsigned laneCount = 0;
};

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted malformed fixed-vector cardinality input");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted malformed fixed-vector cardinality input");
  expectError(test, value.takeError(), expected);
}

::fabric::ImplementationFamilyId familyId(AdapterKind kind) {
  return kind == AdapterKind::Parallelize
             ? ::fabric::ImplementationFamilyId::FixedVectorParallelize
             : ::fabric::ImplementationFamilyId::FixedVectorSerialize;
}

::dataflow::OperationSchemaId schemaId(AdapterKind kind) {
  return kind == AdapterKind::Parallelize
             ? ::dataflow::OperationSchemaId::DataflowParallelize
             : ::dataflow::OperationSchemaId::DataflowSerialize;
}

llvm::StringRef moduleName(AdapterKind kind) {
  return kind == AdapterKind::Parallelize ? "fixed_vector_parallelize"
                                          : "fixed_vector_serialize";
}

mlir::MLIRContext &fabricContext() {
  static mlir::MLIRContext *context = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *context;
}

std::unique_ptr<mlir::MLIRContext> makeCirctContext() {
  mlir::DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                  circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
  context->loadAllAvailableDialects();
  return context;
}

FabricFixture
makeFabric(llvm::StringRef test, ArtifactStore &store, AdapterKind kind,
           ContractKind contractKind = ContractKind::OrderedCardinality,
           std::uint32_t maximumPayloadBits = 32) {
  using loom::adg::DesignBuilder;
  using loom::adg::FuCapabilityTemplateSpec;
  using loom::adg::FuSpec;
  using loom::adg::OperationCapabilitySpec;
  using loom::adg::PeSpec;
  using loom::adg::PortType;

  const PortType value = take(test, PortType::bits(32));
  const std::vector<PortType> inputs = kind == AdapterKind::Parallelize
                                           ? std::vector<PortType>(2, value)
                                           : std::vector<PortType>(3, value);
  const std::vector<PortType> outputs = kind == AdapterKind::Parallelize
                                            ? std::vector<PortType>(3, value)
                                            : std::vector<PortType>(2, value);
  const std::vector<PortType> boundaryInputs(inputs.size(), value);
  const std::vector<PortType> boundaryOutputs(outputs.size(), value);
  const ::fabric::FixedVectorAdapterParams parameters{
      ::fabric::IntegerWidthSet::get(
          {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16}),
      ::fabric::FloatFormatSet{}, maximumPayloadBits};
  const std::uint32_t maximumLanes =
      take(test, ::fabric::maximumFixedVectorAdapterLaneCount(parameters));
  ::fabric::ResourceContract contract =
      contractKind == ContractKind::OrderedCardinality
          ? take(test,
                 ::fabric::createOrderedCardinalityOperationResourceContract(
                     schemaId(kind), maximumLanes))
          : ::fabric::oneCycleElasticOperationResourceContract();

  DesignBuilder builder(store);
  auto spatial =
      take(test, builder.createSpatialCore(moduleName(kind), boundaryInputs,
                                           boundaryOutputs));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != boundaryInputs.size(); ++ordinal)
    spatialInputs.push_back(take(test, spatial.input(ordinal)));
  auto pe = take(
      test, spatial.addPe(spatialInputs,
                          PeSpec::spatial(boundaryInputs, boundaryOutputs)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != boundaryInputs.size(); ++ordinal)
    peInputs.push_back(take(test, pe.input(ordinal)));
  auto fu = take(test, pe.addFu(peInputs, FuSpec{inputs, outputs}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal != inputs.size(); ++ordinal)
    fuInputs.push_back(take(test, fu.input(ordinal)));
  auto operation =
      take(test, fu.addOperation(fuInputs,
                                 OperationCapabilitySpec{familyId(kind),
                                                         parameters,
                                                         {schemaId(kind)},
                                                         outputs,
                                                         std::move(contract)}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  std::vector<loom::adg::FuValue> operationOutputs;
  for (std::size_t ordinal = 0; ordinal != outputs.size(); ++ordinal)
    operationOutputs.push_back(take(test, operation.output(ordinal)));
  if (llvm::Error error = fu.close(operationOutputs))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  std::vector<loom::adg::SpatialValue> spatialOutputs;
  for (std::size_t ordinal = 0; ordinal != boundaryOutputs.size(); ++ordinal)
    spatialOutputs.push_back(take(test, pe.output(ordinal)));
  if (llvm::Error error = spatial.close(spatialOutputs))
    fail(test, llvm::toString(std::move(error)));
  auto design = take(test, std::move(builder).finalize());
  require(test, design.roots().size() == 1,
          "adapter fixture did not publish one Fabric root");
  FinalizedFabricRoot fabric = design.roots().front();

  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &capability :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (capability.implementationFamily != familyId(kind))
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), capability.occurrence, fuOccurrence));
      FinalizedFabricRoot system =
          take(test, loom::hardware::test::makeSingleSpatialCoreSystem(fabric,
                                                                       store));
      auto systemView =
          take(test, loom::fabric::requireSystemRoot(system.view()));
      auto operations =
          take(test, enumerateFabricPhysicalOperations(systemView));
      const auto physical = llvm::find_if(operations, [&](const auto &entry) {
        return entry.localOccurrence == occurrence;
      });
      require(test, physical != operations.end(),
              "System has no physical adapter occurrence");
      return {kind, std::move(fabric), occurrence, std::move(system),
              physical->physicalOccurrence};
    }
  }
  fail(test, "fixture has no fixed-vector cardinality capability");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

AdapterMode modeOf(llvm::StringRef test, AdapterKind kind,
                   const ::dataflow::CanonicalActorSchemaProjection &actor) {
  require(test, actor.schema == schemaId(kind),
          "adapter relation returned another schema");
  mlir::Type vectorType = kind == AdapterKind::Parallelize
                              ? actor.type.getResult(0)
                              : actor.type.getInput(0);
  auto vector = mlir::dyn_cast<mlir::VectorType>(vectorType);
  require(test, vector && vector.getRank() == 1 && vector.getDimSize(0) > 0,
          "adapter relation returned a non-rank-one vector");
  return {vector.getElementTypeBitWidth(),
          static_cast<unsigned>(vector.getDimSize(0))};
}

std::uint8_t modeCode(const AdapterMode &mode) {
  return mode.elementWidth == 8 ? static_cast<std::uint8_t>(mode.laneCount)
                                : static_cast<std::uint8_t>(4 + mode.laneCount);
}

FinalizedConfigurationABI makeConfigurationAbi(llvm::StringRef test,
                                               const ArtifactStore &store,
                                               const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  require(test, resolved.configurationFieldSchema.size() == 1,
          "configured adapter does not have one semantic field");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              !relation.finiteBehaviorDomain().empty(),
          "adapter relation is not a sealed finite domain");
  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  unsigned inactiveLanes = 0;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "configured adapter behavior has no semantic value");
    AdapterMode mode = modeOf(test, fixture.kind, point.representativeActor);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (mode.elementWidth == 8 && mode.laneCount > inactiveLanes) {
      inactive = semantic;
      inactiveLanes = mode.laneCount;
    }
    entries.push_back({std::move(semantic), {modeCode(mode)}});
  }
  require(test, !inactive.empty(), "adapter relation has no integer mode");
  auto field =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride override{
      field,
      SemanticFieldEncoding{FiniteCodebookEncoding{3, std::move(entries)}},
      std::move(inactive)};
  return take(
      test,
      finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         fixture.system, {std::move(override)})),
          store));
}

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fixture,
                             const ConfigurationABI &abi,
                             bool malformedContinuation = false) {
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports = take(
      test, deriveFabricOperationLeafPorts(builder, fixture.physicalOccurrence,
                                           capability(test, fixture), abi));
  if (malformedContinuation) {
    auto continuation = llvm::find_if(ports, [](const auto &port) {
      return port.getName() == "continuation_current";
    });
    require(test, continuation != ports.end(),
            "adapter leaf has no continuation input");
    continuation->type = builder.getIntegerType(2);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(moduleName(fixture.kind)), ports);
  return {std::move(module), leaf};
}

FabricOperationProviderRegistry makeRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableFixedVectorParallelizeSerializeProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

std::string moduleText(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream output(text);
  module.print(output);
  return text;
}

llvm::Expected<FabricOperationProviderOutput>
trySpecialize(SkeletonFixture &skeleton, const FabricFixture &fixture,
              const FinalizedConfigurationABI &abi,
              FabricOperationProviderRegistry &registry) {
  ExternalImplementationContractCatalog externalContracts;
  return specializeFabricOperationLeaves(
      *skeleton.module, abi, {{skeleton.leaf, fixture.physicalOccurrence}},
      {{fixture.physicalOccurrence,
        BackendRecipeKey::PortableSystemVerilog,
        {}}},
      registry, externalContracts);
}

std::string specialize(llvm::StringRef test, SkeletonFixture skeleton,
                       const FabricFixture &fixture,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry = makeRegistry(test);
  ExternalImplementationContractCatalog externalContracts;
  auto conformance =
      take(test, loom::hardware::test::specializeAndExportPortableProvider(
                     {std::move(skeleton.module),
                      {{skeleton.leaf, fixture.physicalOccurrence}}},
                     abi, registry, externalContracts));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "portable adapter emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

void requireOnce(llvm::StringRef test,
                 const dataflow::semantics::ActorResultProductionGroup &group,
                 llvm::ArrayRef<std::uint32_t> results) {
  require(
      test,
      std::holds_alternative<dataflow::semantics::ActorResultProductionOnce>(
          group.repeat) &&
          llvm::ArrayRef<std::uint32_t>(group.activeResults) == results,
      "ordered production group is not the required Once tuple");
}

void canonicalCasesAndContracts() {
  const llvm::StringRef test = __func__;
  using namespace dataflow::semantics;
  auto parallel =
      take(test, projectActorHandshakeCases(
                     ::dataflow::OperationSchemaId::DataflowParallelize, 2, 3));
  require(test,
          parallel.size() == 4 && parallel[0].productionGroups.empty() &&
              parallel[1].productionGroups.size() == 1 &&
              parallel[2].productionGroups.size() == 1 &&
              parallel[3].productionGroups.size() == 2,
          "parallelize production groups are not 0/1/1/2");
  requireOnce(test, parallel[1].productionGroups[0], {0, 1, 2});
  requireOnce(test, parallel[2].productionGroups[0], {2});
  requireOnce(test, parallel[3].productionGroups[0], {0, 1, 2});
  requireOnce(test, parallel[3].productionGroups[1], {2});

  auto serial =
      take(test, projectActorHandshakeCases(
                     ::dataflow::OperationSchemaId::DataflowSerialize, 3, 2));
  require(test,
          serial.size() == 2 && serial[0].productionGroups.size() == 1 &&
              serial[1].productionGroups.size() == 1,
          "serialize production groups are not active/close");
  const auto *repeated =
      std::get_if<ActorResultProductionForEachDefinedOneLane>(
          &serial[0].productionGroups[0].repeat);
  const std::array<std::uint32_t, 2> serializedResults{0, 1};
  require(test,
          repeated &&
              repeated->maskInputOrdinal ==
                  static_cast<std::uint32_t>(SerializeInput::Mask) &&
              serial[0].productionGroups[0].activeResults ==
                  llvm::ArrayRef<std::uint32_t>(serializedResults),
          "serialize active group lost defined-one lane production");
  requireOnce(test, serial[1].productionGroups[0], {1});

  for (AdapterKind kind : {AdapterKind::Parallelize, AdapterKind::Serialize}) {
    auto contract =
        take(test, ::fabric::createOrderedCardinalityOperationResourceContract(
                       schemaId(kind), 4));
    auto exact =
        take(test, ::fabric::isOrderedCardinalityOperationResourceContract(
                       contract, schemaId(kind), 4));
    require(test, exact, "adapter contract did not recognize its exact M=4");
    const std::array<std::uint32_t, 4> expectedParallel{0, 1, 1, 2};
    if (kind == AdapterKind::Parallelize)
      for (std::uint32_t ordinal = 0; ordinal != 4; ++ordinal)
        require(test,
                contract.usePattern(::fabric::UsePatternKey(ordinal))
                        .internalTransactionCount == expectedParallel[ordinal],
                "parallelize contract lost 0/1/1/2 transaction slots");
    else
      require(test,
              contract.usePattern(::fabric::UsePatternKey(0))
                          .internalTransactionCount == 4 &&
                  contract.usePattern(::fabric::UsePatternKey(1))
                          .internalTransactionCount == 1,
              "serialize contract lost M/1 transaction slots");
  }
}

std::string parallelizeTestbench() {
  return R"sv(module testbench;
  logic clock = 0;
  logic reset = 1;
  always #5 clock = ~clock;
  logic [31:0] data;
  logic [31:0] phase;
  logic valid_data, valid_phase;
  logic ready_vector, ready_mask, ready_phase;
  logic [35:0] state = 0;
  logic continuation = 0;
  logic ready_data_in, ready_phase_in;
  logic [31:0] vector_out;
  logic [31:0] mask_out;
  logic [31:0] phase_out;
  logic valid_vector, valid_mask, valid_phase_out;
  logic [35:0] state_next;
  logic state_write, final_production;
  logic capture;

  fixed_vector_parallelize dut(
    .data_input_0(data), .data_input_1(phase),
    .valid_input_0(valid_data), .valid_input_1(valid_phase),
    .ready_output_0(ready_vector), .ready_output_1(ready_mask),
    .ready_output_2(ready_phase), .state_current(state),
    .continuation_current(continuation), .config_0(3'd4),
    .ready_input_0(ready_data_in), .ready_input_1(ready_phase_in),
    .data_output_0(vector_out), .data_output_1(mask_out),
    .data_output_2(phase_out), .valid_output_0(valid_vector),
    .valid_output_1(valid_mask), .valid_output_2(valid_phase_out),
    .final_production(final_production), .state_next(state_next),
    .state_write(state_write));

  assign capture = (valid_vector && ready_vector) ||
                   (valid_mask && ready_mask) ||
                   (valid_phase_out && ready_phase);
  always @(posedge clock) begin
    if (reset) begin
      state <= 0;
      continuation <= 0;
    end else begin
      if (state_write)
        state <= state_next;
      if (capture)
        continuation <= !final_production;
    end
  end

  task automatic pulse_item(input [7:0] item);
    begin
      data = {24'h0, item}; phase = 32'h1;
      valid_data = 1; valid_phase = 1;
      ready_vector = 1; ready_mask = 1; ready_phase = 1;
      #1;
      if (!ready_data_in || !ready_phase_in || !state_write)
        $fatal(1, "parallelize did not accept an accumulating item");
      @(posedge clock); #1;
      valid_data = 0; valid_phase = 0;
    end
  endtask

  initial begin
    data = 0; phase = 0; valid_data = 0; valid_phase = 0;
    ready_vector = 1; ready_mask = 1; ready_phase = 1;
    repeat (2) @(posedge clock);
    @(negedge clock); reset = 0;
    pulse_item(8'h11); pulse_item(8'h22); pulse_item(8'h33);
    data = 32'h44; phase = 32'h1; valid_data = 1; valid_phase = 1;
    ready_vector = 1; ready_mask = 0; ready_phase = 1; #1;
    if (valid_vector || !valid_mask || valid_phase_out || state_write ||
        ready_data_in || vector_out !== 32'h44332211 || mask_out !== 32'hf ||
        phase_out !== 32'h1 || !final_production)
      $fatal(1, "parallelize full group did not hold atomically");
    ready_mask = 1; #1;
    if (!valid_vector || !valid_mask || !valid_phase_out || !state_write)
      $fatal(1, "parallelize full group did not release");
    @(posedge clock); #1; valid_data = 0; valid_phase = 0;
    if (state !== 0 || continuation)
      $fatal(1, "parallelize full group did not retire");

    pulse_item(8'haa); pulse_item(8'hbb);
    phase = 32'h0; valid_phase = 1; valid_data = 0;
    ready_vector = 1; ready_mask = 0; ready_phase = 1; #1;
    if (valid_vector || !valid_mask || valid_phase_out || final_production ||
        vector_out !== 32'h0000bbaa || mask_out !== 32'h3 ||
        phase_out !== 32'h1 ||
        state_write || ready_phase_in)
      $fatal(1, "parallelize partial payload did not hold as non-final");
    ready_mask = 1; #1;
    if (!state_write || !valid_vector || !valid_mask || !valid_phase_out)
      $fatal(1, "parallelize partial payload did not release");
    @(posedge clock); #1;
    data = 32'hcc; phase = 32'h1; valid_data = 1; valid_phase = 1;
    ready_phase = 0; #1;
    if (valid_vector || valid_mask || !valid_phase_out ||
        phase_out !== 32'h0 ||
        !final_production || state_write || ready_data_in || ready_phase_in)
      $fatal(1, "parallelize terminal phase did not block replacement");
    ready_phase = 1; #1;
    if (!state_write)
      $fatal(1, "parallelize terminal phase did not release");
    @(posedge clock); #1;
    if (continuation || !ready_data_in || !ready_phase_in || !state_write)
      $fatal(1, "parallelize final release did not admit replacement");
    @(posedge clock); #1; valid_data = 0; valid_phase = 0;
    if (state !== {4'h1, 32'hcc})
      $fatal(1, "parallelize replacement did not commit after release");
    reset = 1; @(posedge clock); #1; reset = 0;
    if (state !== 0 || continuation)
      $fatal(1, "parallelize replacement reset did not clear state");

    phase = 32'h0; valid_phase = 1; ready_phase = 1; #1;
    if (!ready_phase_in || !valid_phase_out || phase_out !== 32'h0 ||
        !final_production)
      $fatal(1, "parallelize empty close changed its one-group case");
    @(posedge clock); #1; valid_phase = 0;

    pulse_item(8'h5a);
    phase = 32'h0; valid_phase = 1; ready_vector = 1; ready_mask = 1;
    ready_phase = 1; #1; @(posedge clock); #1;
    valid_phase = 0; ready_phase = 0;
    if (!continuation)
      $fatal(1, "parallelize did not enter close continuation");
    reset = 1; @(posedge clock); #1; reset = 0;
    if (state !== 0 || continuation)
      $fatal(1, "parallelize reset did not clear buffered continuation");
    $finish;
  end
endmodule
)sv";
}

std::string serializeTestbench() {
  return R"sv(module testbench;
  logic clock = 0;
  logic reset = 1;
  always #5 clock = ~clock;
  logic [31:0] vector_in;
  logic [31:0] mask_in;
  logic [31:0] phase_in;
  logic [2:0] mode = 3'd4;
  logic valid_vector_in, valid_mask_in, valid_phase_in;
  logic ready_data, ready_phase;
  logic [35:0] state = 0;
  logic continuation = 0;
  logic ready_vector_in, ready_mask_in, ready_phase_in;
  logic [31:0] data_out;
  logic [31:0] phase_out;
  logic valid_data, valid_phase;
  logic [35:0] state_next;
  logic state_write, final_production;
  logic capture;

  fixed_vector_serialize dut(
    .data_input_0(vector_in), .data_input_1(mask_in),
    .data_input_2(phase_in), .valid_input_0(valid_vector_in),
    .valid_input_1(valid_mask_in), .valid_input_2(valid_phase_in),
    .ready_output_0(ready_data), .ready_output_1(ready_phase),
    .state_current(state), .continuation_current(continuation),
    .config_0(mode), .ready_input_0(ready_vector_in),
    .ready_input_1(ready_mask_in), .ready_input_2(ready_phase_in),
    .data_output_0(data_out), .data_output_1(phase_out),
    .valid_output_0(valid_data), .valid_output_1(valid_phase),
    .final_production(final_production), .state_next(state_next),
    .state_write(state_write));

  assign capture = (valid_data && ready_data) ||
                   (valid_phase && ready_phase);
  always @(posedge clock) begin
    if (reset) begin
      state <= 0;
      continuation <= 0;
    end else begin
      if (state_write)
        state <= state_next;
      if (capture)
        continuation <= !final_production;
    end
  end

  initial begin
    vector_in = 0; mask_in = 0; phase_in = 0;
    valid_vector_in = 0; valid_mask_in = 0; valid_phase_in = 0;
    ready_data = 1; ready_phase = 1;
    repeat (2) @(posedge clock); @(negedge clock); reset = 0;

    vector_in = 32'h44332211; mask_in = 32'ha; phase_in = 32'h1;
    valid_vector_in = 1; valid_mask_in = 1; valid_phase_in = 1;
    ready_data = 1; ready_phase = 0; #1;
    if (valid_data || !valid_phase || data_out !== 32'h22 ||
        phase_out !== 32'h1 ||
        final_production || state_write || ready_vector_in || ready_mask_in ||
        ready_phase_in)
      $fatal(1, "serialize first sparse lane did not hold atomically");
    ready_phase = 1; #1;
    if (!valid_data || !valid_phase || !state_write || !ready_vector_in ||
        !ready_mask_in || !ready_phase_in)
      $fatal(1, "serialize first sparse lane did not release");
    @(posedge clock); #1;
    vector_in = 32'hdeadbeef; mask_in = 0; phase_in = 32'h1;
    valid_vector_in = 1; valid_mask_in = 1; valid_phase_in = 1;
    ready_data = 0; ready_phase = 1; #1;
    if (!valid_data || valid_phase || data_out !== 32'h44 ||
        phase_out !== 32'h1 ||
        !final_production || state_write || ready_vector_in || ready_mask_in ||
        ready_phase_in)
      $fatal(1, "serialize final sparse lane did not block replacement");
    ready_data = 1; #1;
    if (!state_write || !valid_data || !valid_phase)
      $fatal(1, "serialize final sparse lane did not release");
    @(posedge clock); #1;
    if (!ready_vector_in || !ready_mask_in || !ready_phase_in || !state_write ||
        valid_data || valid_phase || continuation)
      $fatal(1, "serialize final release did not commit zero-mask replacement");
    @(posedge clock); #1;
    valid_vector_in = 0; valid_mask_in = 0; valid_phase_in = 0;
    if (continuation)
      $fatal(1, "serialize all-zero mask manufactured a continuation");

    mode = 3'd6; vector_in = 32'hbbbbaaaa; mask_in = 32'hc;
    phase_in = 32'h1;
    valid_vector_in = 1; valid_mask_in = 1; valid_phase_in = 1;
    ready_data = 0; ready_phase = 0; #1;
    if (!ready_vector_in || !ready_mask_in || !ready_phase_in || !state_write ||
        valid_data || valid_phase || continuation)
      $fatal(1, "serialize observed i16 padding-mask lanes");
    @(posedge clock); #1;
    mask_in = 32'h1; ready_data = 1; ready_phase = 1; #1;
    if (!valid_data || !valid_phase || data_out !== 32'haaaa ||
        phase_out !== 32'h1 ||
        !final_production || !state_write)
      $fatal(1, "serialize changed the i16 lane payload or final group");
    @(posedge clock); #1;
    valid_vector_in = 0; valid_mask_in = 0; valid_phase_in = 0; mode = 3'd4;
    if (continuation || state !== 0)
      $fatal(1, "serialize i16 single-lane group did not retire");

    phase_in = 32'h0; valid_phase_in = 1; ready_phase = 0; #1;
    if (!valid_phase || phase_out !== 32'h0 || !final_production ||
        state_write ||
        ready_phase_in || ready_vector_in || ready_mask_in)
      $fatal(1, "serialize close did not hold its terminal phase");
    ready_phase = 1; #1;
    if (!ready_phase_in || !state_write)
      $fatal(1, "serialize close did not release");
    @(posedge clock); #1; valid_phase_in = 0;

    vector_in = 32'h88776655; mask_in = 32'h8; phase_in = 32'h1;
    valid_vector_in = 1; valid_mask_in = 1; valid_phase_in = 1;
    ready_data = 1; ready_phase = 1; #1; @(posedge clock); #1;
    valid_vector_in = 0; valid_mask_in = 0; valid_phase_in = 0;
    if (continuation)
      $fatal(1, "single-lane serialize did not identify final production");

    vector_in = 32'h44332211; mask_in = 32'ha; phase_in = 32'h1;
    valid_vector_in = 1; valid_mask_in = 1; valid_phase_in = 1; #1;
    @(posedge clock); #1;
    valid_vector_in = 0; valid_mask_in = 0; valid_phase_in = 0;
    ready_data = 0; ready_phase = 1;
    if (!continuation)
      $fatal(1, "serialize did not retain sparse continuation");
    reset = 1; @(posedge clock); #1; reset = 0;
    if (state !== 0 || continuation)
      $fatal(1, "serialize reset did not clear buffered continuation");
    $finish;
  end
endmodule
)sv";
}

std::string yosysScript(AdapterKind kind) {
  const llvm::StringRef name = moduleName(kind);
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "read_verilog -sv " << name << ".sv\n"
         << "hierarchy -check -top " << name << "\n"
         << "proc\nopt\ncheck -assert\n"
         << "select -assert-none t:$*ff* t:$*latch* t:$_*FF* "
            "t:$_*LATCH* t:$mem*\n"
         << "synth -noabc -top " << name << "\ncheck -assert\n"
         << "select -assert-none t:$*ff* t:$*latch* t:$_*FF* "
            "t:$_*LATCH* t:$mem*\nstat\n";
  return output.str();
}

void configuredProvidersAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricOperationProviderRegistry registry = makeRegistry(test);
  unsigned covered = 0;
  for (const auto &entry : registry.coverage())
    if (entry.implementationFamily ==
            ::fabric::ImplementationFamilyId::FixedVectorParallelize ||
        entry.implementationFamily ==
            ::fabric::ImplementationFamilyId::FixedVectorSerialize) {
      require(test,
              entry.recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
              "adapter registry has the wrong recipe coverage");
      ++covered;
    }
  require(test, covered == 2, "adapter registry did not cover both families");

  std::vector<loom::hardware::test::PortableProviderArtifact> artifacts;
  for (AdapterKind kind : {AdapterKind::Parallelize, AdapterKind::Serialize}) {
    FabricFixture fixture = makeFabric(test, store, kind);
    const auto &resolved = capability(test, fixture);
    const auto *parameters = std::get_if<::fabric::FixedVectorAdapterParams>(
        &resolved.parameterizedCapability);
    require(test,
            parameters &&
                take(test, ::fabric::maximumFixedVectorAdapterLaneCount(
                               *parameters)) == 4,
            "adapter did not derive sealed maximum lane count M=4");
    auto exact = take(
        test, ::fabric::isOrderedCardinalityOperationResourceContract(
                  resolved.resourceStateAndTimingContract, schemaId(kind), 4));
    require(test, exact, "adapter did not retain its exact ordered contract");
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
    auto stateLayout =
        take(test, deriveFabricOperationLeafStateLayout(resolved));
    require(
        test,
        stateLayout && stateLayout->encodedBitCount() == 36 &&
            stateLayout->find(
                FabricOperationLeafStateFieldKind::BufferedValue) &&
            stateLayout->find(FabricOperationLeafStateFieldKind::BufferedMask),
        "adapter state is not value-plus-mask");
    std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
    SkeletonFixture first =
        makeSkeleton(test, *firstContext, fixture, abi.abi());
    const auto ports = first.leaf.getPortList();
    require(test,
            llvm::any_of(ports,
                         [](const auto &port) {
                           return port.getName() == "continuation_current" &&
                                  port.isInput();
                         }) &&
                llvm::any_of(ports,
                             [](const auto &port) {
                               return port.getName() == "final_production" &&
                                      port.isOutput();
                             }),
            "ordered-cardinality leaf lost continuation/final ports");
    const std::string firstRtl =
        specialize(test, std::move(first), fixture, abi);
    std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
    const std::string secondRtl =
        specialize(test, makeSkeleton(test, *secondContext, fixture, abi.abi()),
                   fixture, abi);
    require(test, firstRtl == secondRtl,
            "identical adapter inputs produced nondeterministic RTL");
    artifacts.push_back({moduleName(kind).str() + ".sv", firstRtl});
  }
  artifacts.push_back({"parallelize_testbench.sv", parallelizeTestbench()});
  artifacts.push_back({"serialize_testbench.sv", serializeTestbench()});
  artifacts.push_back({"portable_fixed_vector_parallelize.ys",
                       yosysScript(AdapterKind::Parallelize)});
  artifacts.push_back({"portable_fixed_vector_serialize.ys",
                       yosysScript(AdapterKind::Serialize)});
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated", artifacts))
    fail(test, llvm::toString(std::move(error)));
}

void nonPowerOfTwoMaximumLaneCount(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  for (AdapterKind kind : {AdapterKind::Parallelize, AdapterKind::Serialize}) {
    FabricFixture fixture =
        makeFabric(test, store, kind, ContractKind::OrderedCardinality, 24);
    const auto &resolved = capability(test, fixture);
    const auto *parameters = std::get_if<::fabric::FixedVectorAdapterParams>(
        &resolved.parameterizedCapability);
    require(test,
            parameters &&
                take(test, ::fabric::maximumFixedVectorAdapterLaneCount(
                               *parameters)) == 3,
            "24-bit adapter did not derive M=3");
    require(test,
            take(test, ::fabric::isOrderedCardinalityOperationResourceContract(
                           resolved.resourceStateAndTimingContract,
                           schemaId(kind), 3)),
            "24-bit adapter does not own the exact M=3 contract");
    auto layout = take(test, deriveFabricOperationLeafStateLayout(resolved));
    require(test, layout && layout->encodedBitCount() == 27,
            "M=3 adapter does not have 24-bit value plus 3-bit mask state");
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    const std::string rtl = specialize(
        test, makeSkeleton(test, *context, fixture, abi.abi()), fixture, abi);
    require(test, !rtl.empty(), "M=3 adapter did not materialize portable RTL");
  }
}

void unsupportedAndMalformedInputsAreTransactional(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeRegistry(test);

  for (AdapterKind kind : {AdapterKind::Parallelize, AdapterKind::Serialize}) {
    FabricFixture legacy =
        makeFabric(test, store, kind, ContractKind::LegacyOneCycle);
    FinalizedConfigurationABI legacyAbi =
        makeConfigurationAbi(test, store, legacy);
    std::unique_ptr<mlir::MLIRContext> legacyContext = makeCirctContext();
    SkeletonFixture legacySkeleton =
        makeSkeleton(test, *legacyContext, legacy, legacyAbi.abi());
    const std::string legacyBefore = moduleText(*legacySkeleton.module);
    auto result = trySpecialize(legacySkeleton, legacy, legacyAbi, registry);
    require(test, !result, "legacy one-cycle adapter contract was accepted");
    bool classified = false;
    llvm::handleAllErrors(
        result.takeError(),
        [&](const FabricOperationProviderUnsupportedError &error) {
          classified =
              error.implementationFamily() == familyId(kind) &&
              error.recipe() == BackendRecipeKey::PortableSystemVerilog;
        },
        [&](const llvm::ErrorInfoBase &error) {
          fail(test, "legacy contract returned the wrong error class: " +
                         error.message());
        });
    require(test, classified, "legacy adapter contract lost typed Unsupported");
    require(test, moduleText(*legacySkeleton.module) == legacyBefore,
            "typed Unsupported mutated the caller skeleton");
  }

  FabricFixture valid = makeFabric(test, store, AdapterKind::Parallelize);
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
  SkeletonFixture malformed =
      makeSkeleton(test, *malformedContext, valid, validAbi.abi(), true);
  const std::string before = moduleText(*malformed.module);
  expectError(test, trySpecialize(malformed, valid, validAbi, registry),
              "leaf port");
  require(test, moduleText(*malformed.module) == before,
          "malformed adapter input partially mutated the skeleton");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  canonicalCasesAndContracts();
  const std::filesystem::path root(argv[1]);
  configuredProvidersAndArtifacts(root);
  nonPowerOfTwoMaximumLaneCount(root / "m3");
  unsupportedAndMalformedInputsAreTransactional(root / "invalid");
  return EXIT_SUCCESS;
}

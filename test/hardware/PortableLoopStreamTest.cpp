#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/LoopStream.h"
#include "PortableProviderTestSupport.h"

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricPeConfiguration.h"

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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FabricFuOccurrenceNodeRef;
using loom::fabric::FinalizedFabricRoot;
using namespace loom::hardware;
using namespace loom::hardware::rtl;

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

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted malformed portable loop stream input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectTypedUnsupported(llvm::StringRef test,
                            llvm::Expected<FabricOperationProviderOutput> value,
                            llvm::StringRef description) {
  require(test, !value, std::string("provider accepted ") + description.str());
  bool typedUnsupported = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        typedUnsupported =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::LoopStream &&
            error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, description.str() +
                       " returned the wrong error class: " + error.message());
      });
  require(test, typedUnsupported,
          description.str() + " lost its typed Unsupported classification");
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

enum class ResourceContractKind { LoopStream, OneCycleElastic };

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
  loom::fabric::SpatialCoreOccurrenceRef spatialCore;
};

FabricFixture makeFabric(
    llvm::StringRef test, ArtifactStore &store, llvm::StringRef name,
    ::fabric::IntegerWidthSet widths, ::fabric::IntegerPredicateSet predicates,
    ::dataflow::StreamStepKind stepKind = ::dataflow::StreamStepKind::Add,
    ResourceContractKind contractKind = ResourceContractKind::LoopStream) {
  using loom::adg::DesignBuilder;
  using loom::adg::FuCapabilityTemplateSpec;
  using loom::adg::FuSpec;
  using loom::adg::OperationCapabilitySpec;
  using loom::adg::PeSpec;
  using loom::adg::PortType;

  unsigned physicalWidth = 0;
  for (::fabric::IntegerWidth width : ::fabric::integerWidthDomain)
    if (widths.contains(width))
      physicalWidth = std::max(physicalWidth, ::fabric::getBitWidth(width));
  require(test, physicalWidth != 0, "fixture has no integer width");
  const PortType value = take(test, PortType::bits(physicalWidth));
  const PortType phase = take(test, PortType::bits(1));
  const std::vector<PortType> inputs{value, value, value};
  const std::vector<PortType> outputs{value, value};
  const std::vector<PortType> operationOutputs{value, phase};

  DesignBuilder builder(store);
  auto spatial = take(test, builder.createSpatialCore(name, inputs, outputs));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (unsigned ordinal = 0; ordinal != inputs.size(); ++ordinal)
    spatialInputs.push_back(take(test, spatial.input(ordinal)));
  auto pe = take(
      test, spatial.addPe(spatialInputs, PeSpec::spatial(inputs, outputs)));
  std::vector<loom::adg::PeValue> peInputs;
  for (unsigned ordinal = 0; ordinal != inputs.size(); ++ordinal)
    peInputs.push_back(take(test, pe.input(ordinal)));
  auto fu = take(test, pe.addFu(peInputs, FuSpec{inputs, outputs}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (unsigned ordinal = 0; ordinal != inputs.size(); ++ordinal)
    fuInputs.push_back(take(test, fu.input(ordinal)));
  const ::fabric::ResourceContract &contract =
      contractKind == ResourceContractKind::LoopStream
          ? ::fabric::loopStreamOperationResourceContract()
          : ::fabric::oneCycleElasticOperationResourceContract();
  auto operation = take(
      test, fu.addOperation(
                fuInputs,
                OperationCapabilitySpec{
                    ::fabric::ImplementationFamilyId::LoopStream,
                    ::fabric::LoopStreamParams{widths, stepKind, predicates},
                    {::dataflow::OperationSchemaId::DataflowStream},
                    operationOutputs,
                    contract}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close(
          {take(test, operation.output(0)), take(test, operation.output(1))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          spatial.close({take(test, pe.output(0)), take(test, pe.output(1))}))
    fail(test, llvm::toString(std::move(error)));
  auto design = take(test, std::move(builder).finalize());
  require(test, design.roots().size() == 1,
          "fixture did not publish exactly one Fabric root");
  FinalizedFabricRoot fabric = design.roots().front();

  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &capability :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (capability.implementationFamily !=
          ::fabric::ImplementationFamilyId::LoopStream)
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
              "System has no physical loop stream occurrence");
      require(test, systemView.artifact().accCoreOccurrences().size() == 1,
              "System does not have one SpatialCore");
      return FabricFixture{
          std::move(fabric), occurrence, std::move(system),
          physical->physicalOccurrence,
          loom::fabric::SpatialCoreOccurrenceRef{
              systemView.artifact().accCoreOccurrences().front()}};
    }
  }
  fail(test, "fixture has no loop stream capability");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

struct ConfiguredMode final {
  unsigned width = 0;
  mlir::arith::CmpIPredicate predicate = mlir::arith::CmpIPredicate::eq;
};

ConfiguredMode modeOf(llvm::StringRef test,
                      const ::dataflow::CanonicalActorSchemaProjection &actor) {
  require(test,
          actor.schema == ::dataflow::OperationSchemaId::DataflowStream &&
              actor.type.getNumInputs() == 3 && actor.type.getNumResults() == 2,
          "sealed relation returned a non-stream behavior witness");
  const auto *payload =
      std::get_if<::dataflow::StreamRecurrencePayload>(&actor.payload);
  require(test, payload != nullptr,
          "stream behavior witness has no recurrence payload");
  return {actor.type.getInput(0).getIntOrFloatBitWidth(), payload->predicate};
}

std::uint8_t physicalCode(llvm::StringRef test, const ConfiguredMode &mode) {
  if (mode.width == 8 && mode.predicate == mlir::arith::CmpIPredicate::slt)
    return 1;
  if (mode.width == 8 && mode.predicate == mlir::arith::CmpIPredicate::sgt)
    return 3;
  if (mode.width == 16 && mode.predicate == mlir::arith::CmpIPredicate::slt)
    return 4;
  if (mode.width == 16 && mode.predicate == mlir::arith::CmpIPredicate::sgt)
    return 6;
  fail(test, "sealed relation returned an unexpected stream mode");
}

FinalizedConfigurationABI makeConfiguredAbi(llvm::StringRef test,
                                            const ArtifactStore &store,
                                            const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  require(test, resolved.configurationFieldSchema.size() == 1,
          "configured stream does not expose one semantic field");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 4,
          "configured stream does not have four reachable modes");
  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "configured stream mode has no semantic carrier");
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    const ConfiguredMode mode = modeOf(test, point.representativeActor);
    if (mode.width == 8 && mode.predicate == mlir::arith::CmpIPredicate::slt)
      inactive = semantic;
    entries.push_back({std::move(semantic), {physicalCode(test, mode)}});
  }
  require(test, !inactive.empty(), "configured stream has no inactive mode");
  auto field =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride override{
      std::move(field), FiniteCodebookEncoding{3, std::move(entries)},
      std::move(inactive)};
  return take(
      test,
      finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         fixture.system, {std::move(override)})),
          store));
}

FinalizedConfigurationABI makeDefaultAbi(llvm::StringRef test,
                                         const ArtifactStore &store,
                                         const FabricFixture &fixture) {
  return take(
      test,
      finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         fixture.system)),
          store));
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

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

SkeletonFixture makeLeaf(llvm::StringRef test, mlir::MLIRContext &context,
                         const FabricFixture &fabric,
                         const ConfigurationABI &abi,
                         bool wrongStateWidth = false) {
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports = take(
      test, deriveFabricOperationLeafPorts(builder, fabric.physicalOccurrence,
                                           capability(test, fabric), abi));
  if (wrongStateWidth) {
    const auto state = llvm::find_if(ports, [](const auto &port) {
      return port.getName() == "state_current";
    });
    require(test, state != ports.end(), "loop stream leaf has no state port");
    state->type = builder.getIntegerType(
        mlir::cast<mlir::IntegerType>(state->type).getWidth() + 1);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("loop_stream_leaf"), ports);
  return {std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

std::string specializeLeaf(llvm::StringRef test, SkeletonFixture &skeleton,
                           const FabricFixture &fabric,
                           const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableLoopStreamProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, fabric.physicalOccurrence}}};
  auto conformance =
      take(test, loom::hardware::test::specializeAndExportPortableProvider(
                     std::move(module), abi, registry, externalContracts));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "portable loop stream emitted implementation metadata");
  return std::move(conformance.systemVerilog);
}

const loom::fabric::FabricTransportEndpointRef &
boundaryEndpoint(llvm::StringRef test,
                 const loom::fabric::FabricArtifactView &module,
                 loom::fabric::FabricPortDirection direction,
                 loom::fabric::FabricOrdinal ordinal) {
  const loom::fabric::FabricTransportEndpointRef *result = nullptr;
  for (const auto &attachment : module.moduleBoundaryTransportAttachments()) {
    if (attachment.boundary.direction != direction ||
        attachment.boundary.ordinal != ordinal)
      continue;
    require(test, result == nullptr,
            "module boundary endpoint has duplicate attachments");
    result = &attachment.endpoint;
  }
  if (!result)
    fail(test, "module boundary endpoint is unattached");
  return *result;
}

loom::fabric::FabricPhysicalConfigurationFieldRef qualifyConfigurationField(
    llvm::StringRef test, loom::fabric::SpatialCoreOccurrenceRef spatialCore,
    const loom::fabric::FabricSemanticConfigFieldRef &field) {
  auto target =
      take(test, loom::fabric::FabricModulePhysicalTargetRef::create(field));
  return take(test, loom::fabric::FabricPhysicalConfigurationFieldRef::create(
                        loom::fabric::SpatialCoreInternalOccurrenceRef{
                            spatialCore, std::move(target)}));
}

struct ConfigurationImage final {
  std::string portName;
  std::uint64_t bitCount = 0;
  std::vector<std::uint8_t> payload;
};

ConfigurationImage makeRouteConfiguration(llvm::StringRef test,
                                          const FabricFixture &fixture,
                                          const ConfigurationABI &abi) {
  require(test,
          fixture.fabric.view().peOccurrences().size() == 1 &&
              fixture.fabric.view().fuOccurrences().size() == 1,
          "route fixture changed its PE/FU shape");
  const auto pe = fixture.fabric.view().peOccurrences().front();
  const auto fu = fixture.fabric.view().fuOccurrences().front();
  auto schema =
      take(test, fixture.fabric.view().spatialPeConfigurationSchema(pe));
  const ProgrammingUnit *owner = nullptr;
  std::vector<SemanticConfigurationValue> values;
  for (const auto &descriptor : schema.fields()) {
    loom::fabric::FabricPeConfigurationValue value;
    if (descriptor.kind ==
        loom::fabric::FabricPeConfigurationFieldKind::Activation) {
      value = loom::fabric::FabricPeActive{fu};
    } else {
      require(test, descriptor.port.has_value(),
              "PE selector field has no FU port");
      value = loom::fabric::FabricPeRoute{boundaryEndpoint(
          test, fixture.fabric.view(), descriptor.port->direction,
          descriptor.port->ordinal)};
    }
    const auto physical = qualifyConfigurationField(test, fixture.spatialCore,
                                                    descriptor.reference);
    const ProgrammingUnit *fieldOwner = nullptr;
    for (const ProgrammingUnit &unit : abi.programmingUnits())
      for (const ConfigurationFieldEncoding &field : unit.fields)
        if (field.field == physical)
          fieldOwner = &unit;
    if (!fieldOwner)
      fail(test, "configuration field has no programming owner");
    if (owner)
      require(test, owner->id == fieldOwner->id,
              "route configuration spans programming units");
    else
      owner = fieldOwner;
    auto encoded = take(test, schema.encode(descriptor.reference, value));
    values.push_back(
        {physical, std::vector<std::uint8_t>(encoded.bytes().begin(),
                                             encoded.bytes().end())});
  }
  if (!owner)
    fail(test, "route configuration has no programming unit");
  return {"configuration_" + std::to_string(owner->id), owner->payloadBitCount,
          take(test, abi.encode(owner->id, values))};
}

std::string bitLiteral(llvm::ArrayRef<std::uint8_t> bytes,
                       std::uint64_t bitCount) {
  std::string result;
  result.reserve(static_cast<std::size_t>(bitCount));
  for (std::uint64_t bit = bitCount; bit > 0; --bit) {
    const std::uint64_t index = bit - 1;
    result.push_back(
        ((bytes[static_cast<std::size_t>(index / 8)] >> (index % 8)) & 1U) != 0
            ? '1'
            : '0');
  }
  return result;
}

void schemaCasesAndResourceContractAreAuthoritative() {
  const llvm::StringRef test = __func__;
  using namespace ::dataflow::semantics;
  const auto projected =
      take(test, projectActorHandshakeCases(
                     ::dataflow::OperationSchemaId::DataflowStream, 3, 2));
  require(test, projected.size() == 4,
          "dataflow.stream does not project four transition cases");
  constexpr std::array transitions{
      StreamCase::StartTrue, StreamCase::StartClose, StreamCase::ContinueTrue,
      StreamCase::ContinueClose};
  for (auto [index, transition] : llvm::enumerate(transitions)) {
    const StreamCaseDescriptor descriptor = streamCaseDescriptor(transition);
    std::vector<std::uint32_t> consumed;
    for (unsigned ordinal = 0; ordinal != 3; ++ordinal)
      if (selectsSemanticInput(descriptor.consumedInputs,
                               static_cast<StreamInput>(ordinal)))
        consumed.push_back(ordinal);
    std::vector<std::uint32_t> active;
    if (descriptor.ivSource != StreamOutputSource::None)
      active.push_back(0);
    if (descriptor.emitPhase)
      active.push_back(1);
    require(test,
            projected[index].ordinal == index &&
                llvm::ArrayRef<std::uint32_t>(projected[index].consumedInputs)
                    .equals(consumed) &&
                llvm::ArrayRef<std::uint32_t>(projected[index].activeResults)
                    .equals(active),
            "projected stream case diverged from its schema descriptor");
    require(test,
            take(test, ::fabric::resolveOperationUsePattern(
                           ::fabric::loopStreamOperationResourceContract(),
                           projected[index].ordinal)) ==
                ::fabric::loopControlUsePattern(transition),
            "stream case did not retain its exact resource use pattern");
  }

  const StreamSemanticConfig config{::dataflow::StreamStepKind::Add,
                                    mlir::arith::CmpIPredicate::slt, 8};
  const StreamActivation open{llvm::APInt(8, 0), llvm::APInt(8, 2),
                              llvm::APInt(8, 1)};
  const StreamActivation empty{llvm::APInt(8, 2), llvm::APInt(8, 2),
                               llvm::APInt(8, 1)};
  auto startTrue = take(test, evaluateStreamTransition({}, config, open));
  auto startClose = take(test, evaluateStreamTransition({}, config, empty));
  StreamSemanticState running{StreamMode::Running, llvm::APInt(8, 1),
                              llvm::APInt(8, 2), llvm::APInt(8, 1)};
  auto continueTrue =
      take(test, evaluateStreamTransition(running, config, std::nullopt));
  running.current = llvm::APInt(8, 2);
  auto continueClose =
      take(test, evaluateStreamTransition(running, config, std::nullopt));
  require(test,
          startTrue.emitIv && startTrue.iv == llvm::APInt(8, 0) &&
              startTrue.emitPhase && startTrue.phase &&
              startTrue.nextState.current == llvm::APInt(8, 1) &&
              !startClose.emitIv && startClose.emitPhase && !startClose.phase &&
              startClose.nextState.mode == StreamMode::Idle &&
              continueTrue.emitIv && continueTrue.iv == llvm::APInt(8, 1) &&
              continueTrue.nextState.current == llvm::APInt(8, 2) &&
              !continueClose.emitIv && continueClose.emitPhase &&
              !continueClose.phase &&
              continueClose.nextState.mode == StreamMode::Idle,
          "Dataflow stream evaluator changed one of its four transitions");
}

std::string leafTestbenchText() {
  return R"sv(
module leaf_testbench;
  logic [15:0] data_input_0;
  logic [15:0] data_input_1;
  logic [15:0] data_input_2;
  logic        valid_input_0;
  logic        valid_input_1;
  logic        valid_input_2;
  logic        ready_output_0;
  logic        ready_output_1;
  logic [48:0] state_current;
  logic [2:0]  config_0;
  logic        ready_input_0;
  logic        ready_input_1;
  logic        ready_input_2;
  logic [15:0] data_output_0;
  logic        data_output_1;
  logic        valid_output_0;
  logic        valid_output_1;
  logic [48:0] state_next;
  logic        state_write;

  loop_stream_leaf dut(.*);

  function automatic [48:0] pack_state(
      input logic mode, input logic [15:0] current,
      input logic [15:0] limit, input logic [15:0] step);
    pack_state = {step, limit, current, mode};
  endfunction

  task automatic check(input bit condition, input string message);
    if (!condition) $fatal(1, "%s", message);
  endtask

  initial begin
    data_input_0 = 16'hff01;
    data_input_1 = 16'hee06;
    data_input_2 = 16'hdd02;
    valid_input_0 = 0;
    valid_input_1 = 1;
    valid_input_2 = 1;
    ready_output_0 = 1;
    ready_output_1 = 1;
    state_current = '0;
    config_0 = 3'b000;
    #1;
    check(ready_input_0 && !ready_input_1 && !ready_input_2 &&
              !valid_output_0 && !valid_output_1 && !state_write,
          "partial activation was not atomic");

    valid_input_0 = 1;
    #1;
    check(ready_input_0 && ready_input_1 && ready_input_2 && state_write &&
              state_next == pack_state(1, 16'h0001, 16'h0006, 16'h0002) &&
              !valid_output_0 && !valid_output_1,
          "inactive fallback did not capture the 8-bit activation");

    state_current = state_next;
    valid_input_0 = 0;
    valid_input_1 = 0;
    valid_input_2 = 0;
    ready_output_0 = 0;
    ready_output_1 = 1;
    #1;
    check(valid_output_0 && !valid_output_1 && data_output_0 == 16'h0001 &&
              data_output_1 && !state_write && state_next == state_current,
          "true transition violated IV backpressure");

    ready_output_0 = 1;
    ready_output_1 = 0;
    #1;
    check(!valid_output_0 && valid_output_1 && !state_write &&
              state_next == state_current,
          "true transition violated phase backpressure");

    ready_output_1 = 1;
    #1;
    check(valid_output_0 && valid_output_1 && state_write &&
              state_next == pack_state(1, 16'h0003, 16'h0006, 16'h0002),
          "StartTrue did not publish and commit at t+1");

    state_current = state_next;
    #1;
    check(valid_output_0 && valid_output_1 && data_output_0 == 16'h0003 &&
              state_write &&
              state_next == pack_state(1, 16'h0005, 16'h0006, 16'h0002),
          "ContinueTrue was not bubble-free or fixed-add");
    state_current = state_next;
    #1;
    check(data_output_0 == 16'h0005 && state_write &&
              state_next == pack_state(1, 16'h0007, 16'h0006, 16'h0002),
          "fixed add recurrence changed its step");
    state_current = state_next;
    #1;
    check(!valid_output_0 && valid_output_1 && !data_output_1 && state_write &&
              state_next == '0,
          "ContinueClose did not publish false and terminate");

    state_current = '0;
    data_input_0 = 16'h0004;
    data_input_1 = 16'h0004;
    data_input_2 = 16'h0001;
    valid_input_0 = 1;
    valid_input_1 = 1;
    valid_input_2 = 1;
    config_0 = 3'b001;
    #1;
    check(state_write &&
              state_next == pack_state(1, 16'h0004, 16'h0004, 16'h0001) &&
              !valid_output_0 && !valid_output_1,
          "StartClose activation did not enter the registered slot");
    state_current = state_next;
    valid_input_0 = 0;
    valid_input_1 = 0;
    valid_input_2 = 0;
    #1;
    check(!valid_output_0 && valid_output_1 && !data_output_1 && state_write &&
              state_next == '0,
          "StartClose did not publish false at t+1");

    state_current = '0;
    data_input_0 = 16'h0100;
    data_input_1 = 16'h0102;
    data_input_2 = 16'h0001;
    valid_input_0 = 1;
    valid_input_1 = 1;
    valid_input_2 = 1;
    config_0 = 3'b100;
    #1;
    check(state_write &&
              state_next == pack_state(1, 16'h0100, 16'h0102, 16'h0001),
          "16-bit configuration mode truncated its activation");
    state_current = state_next;
    valid_input_0 = 0;
    valid_input_1 = 0;
    valid_input_2 = 0;
    #1;
    check(valid_output_0 && valid_output_1 && data_output_0 == 16'h0100 &&
              state_next == pack_state(1, 16'h0101, 16'h0102, 16'h0001),
          "16-bit continuation mode did not advance");

    state_current = '0;
    data_input_0 = 16'h0003;
    data_input_1 = 16'h0000;
    data_input_2 = 16'h00ff;
    valid_input_0 = 1;
    valid_input_1 = 1;
    valid_input_2 = 1;
    config_0 = 3'b011;
    #1;
    state_current = state_next;
    valid_input_0 = 0;
    valid_input_1 = 0;
    valid_input_2 = 0;
    #1;
    check(valid_output_0 && data_output_0 == 16'h0003 &&
              state_next == pack_state(1, 16'h0002, 16'h0000, 16'h00ff),
          "signed greater-than mode did not use its physical code");
    $finish;
  end
endmodule
)sv";
}

std::string systemTestbenchText(const ConfigurationImage &configuration) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << R"sv(
module testbench;
  logic       clock;
  logic       reset;
  logic [7:0] input_0_data;
  logic       input_0_valid;
  logic       input_0_ready;
  logic [7:0] input_1_data;
  logic       input_1_valid;
  logic       input_1_ready;
  logic [7:0] input_2_data;
  logic       input_2_valid;
  logic       input_2_ready;
  logic [7:0] output_0_data;
  logic       output_0_valid;
  logic       output_0_ready;
  logic [7:0] output_1_data;
  logic       output_1_valid;
  logic       output_1_ready;
)sv";
  output << "  logic [" << configuration.bitCount - 1 << ":0] "
         << configuration.portName << ";\n\n";
  output << R"sv(  loom_module dut(.*);

  always #5 clock = ~clock;

  task automatic check(input bit condition, input string message);
    if (!condition) $fatal(1, "%s", message);
  endtask

  initial begin
    clock = 0;
    reset = 1;
    input_0_data = 8'h00;
    input_1_data = 8'h03;
    input_2_data = 8'h01;
    input_0_valid = 1;
    input_1_valid = 1;
    input_2_valid = 1;
    output_0_ready = 0;
    output_1_ready = 1;
)sv";
  output << "    " << configuration.portName << " = " << configuration.bitCount
         << "'b" << bitLiteral(configuration.payload, configuration.bitCount)
         << ";\n";
  output << R"sv(    repeat (2) @(posedge clock);
    @(negedge clock);
    check(!input_0_ready && !input_1_ready && !input_2_ready &&
              !output_0_valid && !output_1_valid,
          "reset did not quiesce the stream boundary");
    reset = 0;
    #1;
    check(input_0_ready && input_1_ready && input_2_ready &&
              !output_0_valid && !output_1_valid,
          "Idle did not accept one complete activation");

    @(posedge clock);
    #1;
    check(output_0_valid && !output_1_valid && output_0_data == 8'h00 &&
              output_1_data == 8'h01 && !input_0_ready,
          "first IV was not published at t+1");
    repeat (2) begin
      @(posedge clock);
      #1;
      check(output_0_valid && !output_1_valid &&
                output_0_data == 8'h00,
            "stalled IV changed or retired");
    end

    @(negedge clock);
    output_0_ready = 1;
    #1;
    check(output_0_valid && output_1_valid && output_0_data == 8'h00 &&
              output_1_data == 8'h01,
          "atomic publication did not release the first tuple");
    @(posedge clock);
    #1;
    check(output_0_valid && output_1_valid && output_0_data == 8'h01 &&
              output_1_data == 8'h01,
          "running progress inserted a bubble");
    @(posedge clock);
    #1;
    check(output_0_valid && output_1_valid && output_0_data == 8'h02 &&
              output_1_data == 8'h01,
          "running recurrence did not reach the final IV");

    @(posedge clock);
    #1;
    output_1_ready = 0;
    #1;
    check(!output_0_valid && output_1_valid && output_1_data == 8'h00,
          "termination did not publish only the false phase");
    repeat (2) begin
      @(posedge clock);
      #1;
      check(!output_0_valid && output_1_valid && output_1_data == 8'h00,
            "stalled close changed or retired");
    end
    @(negedge clock);
    input_0_valid = 0;
    input_1_valid = 0;
    input_2_valid = 0;
    output_1_ready = 1;
    @(posedge clock);
    #1;
    check(!output_0_valid && !output_1_valid && !input_0_ready &&
              !input_1_ready && !input_2_ready,
          "close did not terminate the stream");

    @(negedge clock);
    input_0_data = 8'h01;
    input_1_data = 8'h05;
    input_2_data = 8'h01;
    input_0_valid = 1;
    input_1_valid = 1;
    input_2_valid = 1;
    #1;
    check(input_0_ready && input_1_ready && input_2_ready,
          "closed stream did not return to Idle");
    @(posedge clock);
    #1;
    check(output_0_valid && output_1_valid && output_0_data == 8'h01,
          "replacement activation was not published at t+1");

    @(negedge clock);
    reset = 1;
    #1;
    check(!input_0_ready && !input_1_ready && !input_2_ready &&
              !output_0_valid && !output_1_valid,
          "active reset did not quiesce the stream");
    reset = 0;
    input_0_valid = 0;
    input_1_valid = 1;
    input_2_valid = 1;
    #1;
    check(input_0_ready && !input_1_ready && !input_2_ready &&
              !output_0_valid && !output_1_valid,
          "reset did not restore Idle");
    $finish;
  end
endmodule
)sv";
  return text;
}

std::string yosysScriptText() {
  return R"ys(
read_verilog -sv loop_stream_system.sv
hierarchy -check -top loom_module
proc
opt
check -assert
select -assert-none t:$dlatch t:$_DLATCH_*
synth -top loom_module
check -assert
select -assert-none t:$dlatch t:$_DLATCH_*
stat
)ys";
}

void configuredLeafAndSystemArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  const auto widths = ::fabric::IntegerWidthSet::get(
      {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16});
  const auto predicates = ::fabric::IntegerPredicateSet::get(
      {mlir::arith::CmpIPredicate::slt, mlir::arith::CmpIPredicate::sgt});
  FabricFixture configured = makeFabric(
      test, store, "portable-loop-stream-configured", widths, predicates);
  FinalizedConfigurationABI configuredAbi =
      makeConfiguredAbi(test, store, configured);
  auto stateLayout = take(
      test, deriveFabricOperationLeafStateLayout(capability(test, configured)));
  require(test, stateLayout.has_value() && stateLayout->encodedBitCount() == 49,
          "configured stream state is not mode-current-limit-step");
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first =
      makeLeaf(test, *firstContext, configured, configuredAbi.abi());
  const auto ports = first.leaf.getPortList();
  const std::vector<llvm::StringRef> expected{
      "data_input_0",   "data_input_1",  "data_input_2",   "valid_input_0",
      "valid_input_1",  "valid_input_2", "ready_output_0", "ready_output_1",
      "state_current",  "config_0",      "ready_input_0",  "ready_input_1",
      "ready_input_2",  "data_output_0", "data_output_1",  "valid_output_0",
      "valid_output_1", "state_next",    "state_write"};
  require(test, ports.size() == expected.size(),
          "loop stream leaf has the wrong port count");
  for (auto [index, name] : llvm::enumerate(expected))
    require(test, ports[index].getName() == name,
            "loop stream leaf ports are not canonical");
  require(test,
          ports[8].type == mlir::IntegerType::get(firstContext.get(), 49) &&
              ports[9].type == mlir::IntegerType::get(firstContext.get(), 3),
          "loop stream leaf lost its state or ABI field width");
  const std::string firstRtl =
      specializeLeaf(test, first, configured, configuredAbi);
  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeLeaf(test, *secondContext, configured, configuredAbi.abi());
  const std::string secondRtl =
      specializeLeaf(test, second, configured, configuredAbi);
  require(test, firstRtl == secondRtl,
          "identical stream inputs produced different SystemVerilog");

  FabricFixture systemFixture = makeFabric(
      test, store, "portable-loop-stream-system",
      ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I8}),
      ::fabric::IntegerPredicateSet::get({mlir::arith::CmpIPredicate::slt}));
  FinalizedConfigurationABI systemAbi =
      makeDefaultAbi(test, store, systemFixture);
  std::unique_ptr<mlir::MLIRContext> systemContext = makeCirctContext();
  auto skeleton = take(
      test, buildModuleRootCirctSkeleton(
                *systemContext, systemFixture.spatialCore, systemAbi.abi()));
  require(test, skeleton.operationLeaves.size() == 1,
          "CommonSkeleton did not expose one stream leaf");
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableLoopStreamProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  auto conformance = take(
      test, loom::hardware::test::specializeAndExportPortableProvider(
                std::move(skeleton), systemAbi, registry, externalContracts));
  require(test,
          llvm::StringRef(conformance.systemVerilog)
                  .contains("operation_state_reg") &&
              !llvm::StringRef(conformance.systemVerilog)
                   .contains("result_data_reg") &&
              !llvm::StringRef(conformance.systemVerilog)
                   .contains("result_valid_reg"),
          "managed stream timing did not remain in the shared state boundary");
  const ConfigurationImage configuration =
      makeRouteConfiguration(test, systemFixture, systemAbi.abi());
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"loop_stream_leaf.sv", firstRtl},
           {"leaf_testbench.sv", leafTestbenchText()},
           {"loop_stream_system.sv", conformance.systemVerilog},
           {"testbench.sv", systemTestbenchText(configuration)},
           {"portable_loop_stream.ys", yosysScriptText()}}))
    fail(test, llvm::toString(std::move(error)));
}

void invalidInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  const auto widths =
      ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I8});
  const auto predicates =
      ::fabric::IntegerPredicateSet::get({mlir::arith::CmpIPredicate::slt});
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableLoopStreamProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;

  FabricFixture unsupported = makeFabric(
      test, store, "portable-loop-stream-unsupported", widths, predicates,
      ::dataflow::StreamStepKind::Add, ResourceContractKind::OneCycleElastic);
  FinalizedConfigurationABI unsupportedAbi =
      makeDefaultAbi(test, store, unsupported);
  SkeletonFixture unsupportedSkeleton =
      makeLeaf(test, *context, unsupported, unsupportedAbi.abi());
  const std::string beforeContract = moduleText(*unsupportedSkeleton.module);
  const std::vector<FabricOperationLeafAssociation> unsupportedAssociations = {
      {unsupportedSkeleton.leaf, unsupported.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {unsupported.physicalOccurrence,
       BackendRecipeKey::PortableSystemVerilog,
       {}}};
  expectTypedUnsupported(
      test,
      specializeFabricOperationLeaves(*unsupportedSkeleton.module,
                                      unsupportedAbi, unsupportedAssociations,
                                      recipes, registry, externalContracts),
      "unsupported stream resource contract");
  require(test, moduleText(*unsupportedSkeleton.module) == beforeContract,
          "unsupported contract partially mutated the caller module");

  FabricFixture valid =
      makeFabric(test, store, "portable-loop-stream-valid", widths, predicates);
  FinalizedConfigurationABI validAbi = makeDefaultAbi(test, store, valid);
  SkeletonFixture malformed =
      makeLeaf(test, *context, valid, validAbi.abi(), true);
  const std::string beforeLeaf = moduleText(*malformed.module);
  const std::vector<FabricOperationLeafAssociation> malformedAssociations = {
      {malformed.leaf, valid.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> validRecipes = {
      {valid.physicalOccurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectError(test,
              specializeFabricOperationLeaves(
                  *malformed.module, validAbi, malformedAssociations,
                  validRecipes, registry, externalContracts),
              "leaf port");
  require(test, moduleText(*malformed.module) == beforeLeaf,
          "malformed leaf partially mutated the caller module");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  schemaCasesAndResourceContractAreAuthoritative();
  configuredLeafAndSystemArtifacts(argv[1]);
  invalidInputsAreTransactional(std::filesystem::path(argv[1]) / "invalid");
  return 0;
}

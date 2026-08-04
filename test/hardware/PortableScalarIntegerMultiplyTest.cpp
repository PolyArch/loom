#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/Providers/ScalarIntegerMultiply.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    fail(test, "accepted invalid portable scalar integer multiply input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
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

enum class FabricFixtureKind {
  Multiply,
  MultiplyAllWidths,
  PointerMultiply,
  MultiplyWithAddSchema,
  Add,
};

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  unsigned width = 0;
  std::vector<unsigned> admittedWidths;
};

std::vector<unsigned> admittedWidths(unsigned width, FabricFixtureKind kind) {
  if (kind == FabricFixtureKind::MultiplyAllWidths)
    return {8, 16, 32, 64};
  return {width};
}

std::string fabricSource(unsigned width, FabricFixtureKind kind) {
  const std::string bits = std::to_string(width);
  const bool add = kind == FabricFixtureKind::Add;
  const bool addSchema =
      add || kind == FabricFixtureKind::MultiplyWithAddSchema;
  const llvm::StringRef operation = addSchema ? "arith.addi" : "arith.muli";
  const llvm::StringRef family =
      add ? "ScalarIntegerAddSub" : "ScalarIntegerMultiply";
  std::string parameters = "integer_widths = [";
  llvm::raw_string_ostream parameterStream(parameters);
  const std::vector<unsigned> widths = admittedWidths(width, kind);
  for (const auto [index, admittedWidth] : llvm::enumerate(widths)) {
    if (index != 0)
      parameterStream << ", ";
    parameterStream << admittedWidth << " : i32";
  }
  parameterStream << ']';
  if (kind == FabricFixtureKind::PointerMultiply)
    parameterStream << ", pointer_formats = [{address_space = 0 : i32, "
                       "representation_bits = 64 : i32, address_bits = 64 : "
                       "i32, kind = \"stable_integral\"}]";

  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module { fabric.module @scalar_integer_"
         << (add ? "add" : "multiply") << '_' << width << "(%a: !fabric.bits<"
         << width << ">, %b: !fabric.bits<" << width << ">) -> !fabric.bits<"
         << width << "> { %pe = fabric.pe [spatial] (%pa = %a : !fabric.bits<"
         << width << ">, %pb = %b : !fabric.bits<" << width
         << ">) -> !fabric.bits<" << width
         << "> { %fu = fabric.fu (%fa = %pa : !fabric.bits<" << width
         << ">, %fb = %pb : !fabric.bits<" << width << ">) -> !fabric.bits<"
         << width << "> { %value = fabric.op [@" << operation
         << "] (%fa, %fb) {implementation_family = "
         << "#fabric.implementation_family<" << family << ">, hw_params = {"
         << parameters << "}} : (!fabric.bits<" << width << ">, !fabric.bits<"
         << width << ">) -> !fabric.bits<" << width
         << "> fabric.yield %value : !fabric.bits<" << width
         << "> } } fabric.yield %pe : !fabric.bits<" << width << "> } }";
  return source;
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         unsigned width,
                         FabricFixtureKind kind = FabricFixtureKind::Multiply) {
  const std::string sourceText = fabricSource(width, kind);
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");

  const std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  const std::vector<std::int8_t> signedContract(contract.begin(),
                                                contract.end());
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedContract));
  });

  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no root");
  FinalizedFabricRoot fabric =
      take(test, loom::fabric::finalizeFabricRoot(root, store));

  const ::fabric::ImplementationFamilyId expectedFamily =
      kind == FabricFixtureKind::Add
          ? ::fabric::ImplementationFamilyId::ScalarIntegerAddSub
          : ::fabric::ImplementationFamilyId::ScalarIntegerMultiply;
  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &capability :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (capability.implementationFamily != expectedFamily)
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), capability.occurrence, fuOccurrence));
      return FabricFixture{std::move(fabric), occurrence, width,
                           admittedWidths(width, kind)};
    }
  }
  fail(test, "Fabric fixture has no expected operation occurrence");
}

void expectFabricFinalizationError(llvm::StringRef test,
                                   const ArtifactStore &store, unsigned width,
                                   FabricFixtureKind kind,
                                   llvm::StringRef expected) {
  const std::string sourceText = fabricSource(width, kind);
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, static_cast<bool>(source), "could not parse invalid fixture");
  const std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  const std::vector<std::int8_t> signedContract(contract.begin(),
                                                contract.end());
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedContract));
  });
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "invalid fixture has no root");
  auto finalized = loom::fabric::finalizeFabricRoot(root, store);
  require(test, !finalized, "invalid Fabric capability was finalized");
  const std::string message = llvm::toString(finalized.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectFabricParseError(llvm::StringRef test, unsigned width,
                            FabricFixtureKind kind, llvm::StringRef expected) {
  std::vector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      &fabricContext(), [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return mlir::success();
      });
  const std::string sourceText = fabricSource(width, kind);
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, !source, "invalid Fabric schema parsed successfully");
  require(test, !diagnostics.empty(),
          "invalid Fabric schema produced no diagnostic");
  require(test, llvm::StringRef(diagnostics.front()).contains(expected),
          diagnostics.front());
}

FinalizedConfigurationABI makeConfigurationAbi(llvm::StringRef test,
                                               const ArtifactStore &store,
                                               const FabricFixture &fixture) {
  const auto *capability =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  require(test, capability->configurationFieldSchema.empty(),
          "integer multiply created a runtime configuration field");
  return take(
      test, finalizeConfigurationABI(
                ConfigurationABIDraft{fixture.fabric.reference(), {}}, store));
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

enum class LeafMutation {
  None,
  WrongInputWidth,
  ExtraConfigurationPort,
};

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fabric,
                             const ConfigurationABI &abi,
                             LeafMutation mutation = LeafMutation::None) {
  const auto *capability =
      fabric.fabric.view().resolvedFabricOpCapability(fabric.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports =
      take(test, deriveFabricOperationLeafPorts(builder, *capability, abi));
  if (mutation == LeafMutation::WrongInputWidth) {
    require(test, ports.size() == 3, "multiply leaf did not have three ports");
    ports.front().type = builder.getIntegerType(
        fabric.width == 8 ? 16 : static_cast<unsigned>(fabric.width / 2));
  } else if (mutation == LeafMutation::ExtraConfigurationPort) {
    require(test, ports.size() == 3, "multiply leaf did not have three ports");
    ports.insert(ports.begin() + 2,
                 circt::hw::PortInfo{
                     {builder.getStringAttr("config_0"), builder.getI1Type(),
                      circt::hw::ModulePort::Direction::Input}});
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("scalar_integer_multiply_" +
                            std::to_string(fabric.width)),
      ports);
  return SkeletonFixture{std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

std::string specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableScalarIntegerMultiplyProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.occurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  FabricOperationProviderOutput output =
      take(test, specializeFabricOperationLeaves(
                     *skeleton.module, fabric.fabric, abi, associations,
                     recipes, registry, externalContracts));
  require(test,
          output.payloads.empty() && output.activityPoints.empty() &&
              output.externalImplementationBindings.empty(),
          "portable multiply emitted external implementation state");
  bool unresolved = false;
  skeleton.module->walk(
      [&](circt::hw::HWModuleGeneratedOp) { unresolved = true; });
  require(test, !unresolved,
          "portable multiply left an unresolved operation leaf");
  if (llvm::Error error = verifySpecializedCirctModule(*skeleton.module))
    fail(test, llvm::toString(std::move(error)));
  return take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
}

void checkCapability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *capability =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  require(test,
          capability->implementationFamily ==
              ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
          "multiply capability changed implementation family");
  const ::fabric::ImplementationFamilyDescriptor &descriptor =
      ::fabric::implementationFamily(capability->implementationFamily);
  require(test,
          !capability->enabledOperationSchemas.empty() &&
              llvm::all_of(capability->enabledOperationSchemas,
                           [&](::dataflow::OperationSchemaId schema) {
                             return llvm::is_contained(
                                 descriptor.admittedSchemas, schema);
                           }),
          "multiply capability escaped its generated family descriptor");
  const auto *parameters = std::get_if<::fabric::ScalarIntegerParams>(
      &capability->parameterizedCapability);
  require(test, parameters != nullptr,
          "multiply capability changed parameter schema");
  require(test,
          parameters->integerWidths.valid() &&
              parameters->integerWidths.size() ==
                  fixture.admittedWidths.size() &&
              parameters->pointerFormats.empty(),
          "multiply capability changed its scalar integer parameters");
  for (unsigned admittedWidth : fixture.admittedWidths) {
    const auto expectedWidth = llvm::find_if(
        ::fabric::integerWidthDomain, [&](::fabric::IntegerWidth width) {
          return ::fabric::getBitWidth(width) == admittedWidth;
        });
    require(test,
            expectedWidth != ::fabric::integerWidthDomain.end() &&
                parameters->integerWidths.contains(*expectedWidth),
            "multiply capability lost an admitted integer width");
  }
  require(test, capability->configurationFieldSchema.empty(),
          "overflow semantics became a runtime configuration field");

  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const auto &port : capability->physicalPorts)
    (port.reference.direction == loom::fabric::FabricPortDirection::Input
         ? inputs
         : outputs)
        .push_back(&port);
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  require(test,
          inputs.size() == 2 && outputs.size() == 1 &&
              inputs[0]->reference.ordinal == 0 &&
              inputs[1]->reference.ordinal == 1 &&
              outputs[0]->reference.ordinal == 0 &&
              inputs[0]->payloadWidthBits == fixture.width &&
              inputs[1]->payloadWidthBits == fixture.width &&
              outputs[0]->payloadWidthBits == fixture.width,
          "multiply capability changed its binary physical port shape");
  const std::vector<std::uint8_t> actual =
      take(test, ::fabric::encodeResourceContractRecord(
                     capability->resourceStateAndTimingContract));
  const std::vector<std::uint8_t> expected =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  require(test, actual == expected,
          "multiply capability changed its one-cycle elastic contract");
}

struct EmittedRtl final {
  std::string width8;
  std::string width64;
};

EmittedRtl validWidthsAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  EmittedRtl emitted;
  for (unsigned width : std::array<unsigned, 4>{8, 16, 32, 64}) {
    FabricFixture fabric = makeFabric(test, store, width);
    checkCapability(test, fabric);
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);

    std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
    SkeletonFixture first =
        makeSkeleton(test, *firstContext, fabric, abi.abi());
    const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
    require(
        test,
        ports.size() == 3 && ports.atInput(0).getName() == "data_input_0" &&
            ports.atInput(1).getName() == "data_input_1" &&
            ports.atOutput(0).getName() == "data_output_0" &&
            mlir::cast<mlir::IntegerType>(ports.atInput(0).type).getWidth() ==
                width &&
            mlir::cast<mlir::IntegerType>(ports.atInput(1).type).getWidth() ==
                width &&
            mlir::cast<mlir::IntegerType>(ports.atOutput(0).type).getWidth() ==
                width,
        "derived multiply leaf ports are not canonical");
    for (const auto &port : ports)
      require(test, !port.getName().starts_with("config_"),
              "multiply leaf retained a configuration port");
    const std::string firstRtl = specialize(test, first, fabric, abi);

    std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
    SkeletonFixture second =
        makeSkeleton(test, *secondContext, fabric, abi.abi());
    const std::string secondRtl = specialize(test, second, fabric, abi);
    require(test, firstRtl == secondRtl,
            "identical multiply inputs produced different SystemVerilog");
    const llvm::StringRef rtl(firstRtl);
    require(test, rtl.contains(" * ") && !rtl.contains("config_"),
            "portable multiply is not a direct configuration-free product");
    if (width == 8)
      emitted.width8 = firstRtl;
    if (width == 64)
      emitted.width64 = firstRtl;
  }
  return emitted;
}

void multiWidthHsgUsesOneDatapath(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric =
      makeFabric(test, store, 64, FabricFixtureKind::MultiplyAllWidths);
  checkCapability(test, fabric);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test, llvm::StringRef(rtl).count(" * ") == 1,
          "multi-width HSG did not share one multiply datapath");
  require(test, !llvm::StringRef(rtl).contains("config_"),
          "multi-width multiply HSG created a selector");
}

void writeToolInputs(const std::filesystem::path &root,
                     const EmittedRtl &emitted) {
  std::ofstream(root / "scalar_integer_multiply_8.sv") << emitted.width8;
  std::ofstream(root / "scalar_integer_multiply_64.sv") << emitted.width64;
  std::ofstream(root / "testbench.sv") << R"sv(
module testbench;
  logic [7:0] data8_input_0;
  logic [7:0] data8_input_1;
  logic [7:0] data8_output_0;
  logic [63:0] data64_input_0;
  logic [63:0] data64_input_1;
  logic [63:0] data64_output_0;

  scalar_integer_multiply_8 multiply8(
    .data_input_0(data8_input_0),
    .data_input_1(data8_input_1),
    .data_output_0(data8_output_0));
  scalar_integer_multiply_64 multiply64(
    .data_input_0(data64_input_0),
    .data_input_1(data64_input_1),
    .data_output_0(data64_output_0));

  initial begin
    data8_input_0 = 8'h00;
    data8_input_1 = 8'ha5;
    #1;
    if (data8_output_0 !== 8'h00) $fatal(1, "8-bit zero failed");

    data8_input_0 = 8'h01;
    #1;
    if (data8_output_0 !== 8'ha5) $fatal(1, "8-bit identity failed");

    data8_input_0 = 8'h13;
    data8_input_1 = 8'h07;
    #1;
    if (data8_output_0 !== 8'h85) $fatal(1, "8-bit ordinary product failed");

    data8_input_0 = 8'hff;
    data8_input_1 = 8'h02;
    #1;
    if (data8_output_0 !== 8'hfe) $fatal(1, "8-bit overflow failed");

    data64_input_0 = 64'h0000000000000000;
    data64_input_1 = 64'h0123456789abcdef;
    #1;
    if (data64_output_0 !== 64'h0000000000000000)
      $fatal(1, "64-bit zero failed");

    data64_input_0 = 64'h0000000000000001;
    #1;
    if (data64_output_0 !== 64'h0123456789abcdef)
      $fatal(1, "64-bit identity failed");

    data64_input_0 = 64'h0000000000001234;
    data64_input_1 = 64'h0000000000000101;
    #1;
    if (data64_output_0 !== 64'h0000000000124634)
      $fatal(1, "64-bit ordinary product failed");

    data64_input_0 = 64'hffffffffffffffff;
    data64_input_1 = 64'h0000000000000002;
    #1;
    if (data64_output_0 !== 64'hfffffffffffffffe)
      $fatal(1, "64-bit modular product failed");
    $finish;
  end
endmodule
)sv";
  std::ofstream(root / "synthesis_top.sv") << R"sv(
module scalar_integer_multiply_synthesis_top(
  input logic [7:0] input8_0,
  input logic [7:0] input8_1,
  input logic [63:0] input64_0,
  input logic [63:0] input64_1,
  output logic [7:0] output8,
  output logic [63:0] output64);
  scalar_integer_multiply_8 multiply8(
    .data_input_0(input8_0), .data_input_1(input8_1),
    .data_output_0(output8));
  scalar_integer_multiply_64 multiply64(
    .data_input_0(input64_0), .data_input_1(input64_1),
    .data_output_0(output64));
endmodule
)sv";
  std::ofstream(root / "portable_scalar_integer_multiply.ys") << R"ys(
read_verilog -sv scalar_integer_multiply_8.sv scalar_integer_multiply_64.sv synthesis_top.sv
hierarchy -check -top scalar_integer_multiply_synthesis_top
proc
opt
check
synth -top scalar_integer_multiply_synthesis_top
stat
)ys";
}

void generatedRegistryRemainsTheSemanticOwner() {
  const llvm::StringRef test = __func__;
  const ::fabric::ImplementationFamilyDescriptor &descriptor =
      ::fabric::implementationFamily(
          ::fabric::ImplementationFamilyId::ScalarIntegerMultiply);
  require(test,
          descriptor.familyId ==
                  ::fabric::ImplementationFamilyId::ScalarIntegerMultiply &&
              descriptor.capabilityParamsSchema ==
                  ::fabric::CapabilityParamsSchemaId::ScalarIntegerParams &&
              !descriptor.admittedSchemas.empty(),
          "generated multiply family descriptor changed");

  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableScalarIntegerMultiplyProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  const auto coverage = registry.coverage();
  const auto entry = llvm::find_if(coverage, [](const auto &candidate) {
    return candidate.implementationFamily ==
           ::fabric::ImplementationFamilyId::ScalarIntegerMultiply;
  });
  require(test,
          entry != coverage.end() &&
              entry->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "multiply provider registration changed its exact coverage");
}

void invalidInputsFailClosed(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableScalarIntegerMultiplyProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;

  expectFabricFinalizationError(
      test, store, 64, FabricFixtureKind::PointerMultiply,
      "GEP member and pointer_formats must be present or absent together");
  expectFabricParseError(test, 8, FabricFixtureKind::MultiplyWithAddSchema,
                         "not admitted by implementation family");

  FabricFixture valid = makeFabric(test, store, 8);
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  const std::vector<FabricOperationRecipeBinding> validRecipes = {
      {valid.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  for (LeafMutation mutation :
       {LeafMutation::WrongInputWidth, LeafMutation::ExtraConfigurationPort}) {
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton =
        makeSkeleton(test, *context, valid, validAbi.abi(), mutation);
    const std::string before = moduleText(*skeleton.module);
    const std::vector<FabricOperationLeafAssociation> associations = {
        {skeleton.leaf, valid.occurrence}};
    expectError(test,
                specializeFabricOperationLeaves(
                    *skeleton.module, valid.fabric, validAbi, associations,
                    validRecipes, registry, externalContracts),
                "leaf port");
    require(test, moduleText(*skeleton.module) == before,
            "invalid leaf shape partially mutated the caller module");
  }

  FabricFixture add = makeFabric(test, store, 8, FabricFixtureKind::Add);
  FinalizedConfigurationABI addAbi = makeConfigurationAbi(test, store, add);
  std::unique_ptr<mlir::MLIRContext> addContext = makeCirctContext();
  SkeletonFixture addSkeleton =
      makeSkeleton(test, *addContext, add, addAbi.abi());
  const std::string addBefore = moduleText(*addSkeleton.module);
  const std::vector<FabricOperationLeafAssociation> addAssociations = {
      {addSkeleton.leaf, add.occurrence}};
  const std::vector<FabricOperationRecipeBinding> addRecipes = {
      {add.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  auto wrongFamily = specializeFabricOperationLeaves(
      *addSkeleton.module, add.fabric, addAbi, addAssociations, addRecipes,
      registry, externalContracts);
  require(test, !wrongFamily, "multiply provider accepted a different family");
  bool unsupportedAdd = false;
  llvm::handleAllErrors(
      wrongFamily.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        unsupportedAdd =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::ScalarIntegerAddSub &&
            error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "wrong-family input returned the wrong error class: " +
                       error.message());
      });
  require(test, unsupportedAdd,
          "wrong-family input lost its typed Unsupported classification");
  require(test, moduleText(*addSkeleton.module) == addBefore,
          "wrong-family input partially mutated the caller module");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  generatedRegistryRemainsTheSemanticOwner();
  const EmittedRtl emitted = validWidthsAndDeterminism(root);
  multiWidthHsgUsesOneDatapath(root / "multi_width_hsg");
  writeToolInputs(root, emitted);
  invalidInputsFailClosed(root / "invalid");
  return 0;
}

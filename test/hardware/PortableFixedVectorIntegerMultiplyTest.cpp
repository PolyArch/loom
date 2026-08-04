#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/Providers/FixedVectorIntegerMultiply.h"

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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
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
    fail(test, "accepted invalid portable fixed-vector multiply input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectTypedUnsupported(llvm::StringRef test,
                            llvm::Expected<FabricOperationProviderOutput> value,
                            ::fabric::ImplementationFamilyId family,
                            BackendRecipeKey recipe,
                            llvm::StringRef description) {
  require(test, !value, std::string("provider accepted ") + description.str());
  bool typedUnsupported = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        typedUnsupported =
            error.implementationFamily() == family && error.recipe() == recipe;
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

enum class FabricFixtureKind {
  Configured,
  Singleton,
  UnsupportedContract,
  OtherFamily,
  WrongSchema,
};

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
};

std::string fabricSource(FabricFixtureKind kind) {
  const bool otherFamily = kind == FabricFixtureKind::OtherFamily;
  const bool wrongSchema = kind == FabricFixtureKind::WrongSchema;
  const bool singleton =
      kind == FabricFixtureKind::Singleton || otherFamily || wrongSchema;
  const unsigned physicalWidth = singleton ? 32 : 40;
  const unsigned maxPayloadBits = singleton ? 32 : 33;
  const llvm::StringRef operation =
      otherFamily || wrongSchema ? "arith.addi" : "arith.muli";
  const llvm::StringRef family =
      otherFamily ? "FixedVectorIntegerAddSub" : "FixedVectorIntegerMultiply";

  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module { fabric.module @fixed_vector_integer_operation"
         << "(%a: !fabric.bits<" << physicalWidth << ">, %b: !fabric.bits<"
         << physicalWidth << ">) -> !fabric.bits<" << physicalWidth
         << "> { %pe = fabric.pe [spatial]"
         << "(%pa = %a : !fabric.bits<" << physicalWidth
         << ">, %pb = %b : !fabric.bits<" << physicalWidth
         << ">) -> !fabric.bits<" << physicalWidth << "> { %fu = fabric.fu"
         << "(%fa = %pa : !fabric.bits<" << physicalWidth
         << ">, %fb = %pb : !fabric.bits<" << physicalWidth
         << ">) -> !fabric.bits<" << physicalWidth
         << "> { %value = fabric.op [@" << operation << "] (%fa, %fb)"
         << " {implementation_family = #fabric.implementation_family<" << family
         << ">, hw_params = {element_widths = [8 : i32";
  if (!singleton)
    stream << ", 16 : i32";
  stream << "], max_payload_bits = " << maxPayloadBits
         << " : i32}} : (!fabric.bits<" << physicalWidth << ">, !fabric.bits<"
         << physicalWidth << ">) -> !fabric.bits<" << physicalWidth
         << "> fabric.yield %value : !fabric.bits<" << physicalWidth
         << "> } } fabric.yield %pe : !fabric.bits<" << physicalWidth
         << "> } }";
  return source;
}

FabricFixture
makeFabric(llvm::StringRef test, const ArtifactStore &store,
           FabricFixtureKind kind = FabricFixtureKind::Configured) {
  const std::string sourceText = fabricSource(kind);
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");

  const ::fabric::ResourceContract &resourceContract =
      kind == FabricFixtureKind::UnsupportedContract
          ? ::fabric::loopCarryOperationResourceContract()
          : ::fabric::oneCycleElasticOperationResourceContract();
  const std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(resourceContract));
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
      kind == FabricFixtureKind::OtherFamily
          ? ::fabric::ImplementationFamilyId::FixedVectorIntegerAddSub
          : ::fabric::ImplementationFamilyId::FixedVectorIntegerMultiply;
  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &candidate :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (candidate.implementationFamily != expectedFamily)
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), candidate.occurrence, fuOccurrence));
      return FabricFixture{std::move(fabric), occurrence};
    }
  }
  fail(test, "Fabric fixture has no expected operation occurrence");
}

void expectFabricParseError(llvm::StringRef test, FabricFixtureKind kind,
                            llvm::StringRef expected) {
  std::vector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      &fabricContext(), [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return mlir::success();
      });
  const std::string sourceText = fabricSource(kind);
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, !source, "invalid Fabric capability parsed successfully");
  require(test, !diagnostics.empty(),
          "invalid Fabric capability produced no diagnostic");
  require(test, llvm::StringRef(diagnostics.front()).contains(expected),
          diagnostics.front());
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

dataflow::CanonicalActorSchemaProjection
multiplyActor(std::initializer_list<std::int64_t> shape, unsigned elementWidth,
              mlir::arith::IntegerOverflowFlags flags =
                  mlir::arith::IntegerOverflowFlags::none) {
  mlir::MLIRContext &context = fabricContext();
  mlir::Type element = mlir::IntegerType::get(&context, elementWidth);
  mlir::VectorType vector = mlir::VectorType::get(shape, element);
  mlir::FunctionType type =
      mlir::FunctionType::get(&context, {vector, vector}, {vector});
  return {dataflow::OperationSchemaId::ArithMulI, type,
          dataflow::IntegerOverflowPayload{flags}};
}

std::vector<std::uint8_t>
configurationValue(llvm::StringRef test,
                   const loom::fabric::ResolvedFabricOpCapabilityView &resolved,
                   const loom::fabric::FabricSemanticConfigFieldRef &field,
                   const dataflow::CanonicalActorSchemaProjection &actor) {
  const loom::CanonicalSemanticBytes encoded =
      take(test, resolved.encodeSemanticConfiguration(field, actor, 64));
  return std::vector<std::uint8_t>(encoded.bytes().begin(),
                                   encoded.bytes().end());
}

unsigned behaviorElementWidth(
    const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  return mlir::cast<mlir::IntegerType>(
             mlir::cast<mlir::VectorType>(
                 point.representativeActor.type.getInput(0))
                 .getElementType())
      .getWidth();
}

unsigned behaviorLaneCount(
    const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  return static_cast<unsigned>(
      mlir::cast<mlir::VectorType>(point.representativeActor.type.getInput(0))
          .getNumElements());
}

std::vector<FiniteCodebookEntry>
completeEntries(llvm::StringRef test,
                const loom::fabric::ResolvedFabricOpCapabilityView &resolved) {
  const auto domain =
      take(test, resolved.resolveFiniteBehaviorDomain(fabricContext()));
  require(test, domain.size() == 2,
          "Fabric did not project the exact vector multiply behavior domain");

  std::vector<FiniteCodebookEntry> entries;
  entries.reserve(domain.size());
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured Fabric behavior has no semantic value");
    const unsigned width = behaviorElementWidth(point);
    require(test,
            point.representativeActor.schema ==
                    dataflow::OperationSchemaId::ArithMulI &&
                (width == 8 || width == 16),
            "Fabric projected an unexpected fixed-vector multiply behavior");
    entries.push_back(
        {std::vector<std::uint8_t>(point.semanticConfiguration->bytes().begin(),
                                   point.semanticConfiguration->bytes().end()),
         {static_cast<std::uint8_t>(width == 8 ? 1 : 2)}});
  }
  return entries;
}

enum class ConfigurationAbiKind {
  Complete,
  MissingI8,
  ExtraSemanticValue,
};

FinalizedConfigurationABI makeConfigurationAbi(
    llvm::StringRef test, const ArtifactStore &store,
    const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, finalizeConfigurationABI(
                          ConfigurationABIDraft{fixture.fabric.reference(), {}},
                          store));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "vector multiply fixture has an unexpected field count");
  std::vector<FiniteCodebookEntry> entries = completeEntries(test, resolved);
  if (kind == ConfigurationAbiKind::MissingI8) {
    const auto missing = llvm::find_if(entries, [](const auto &entry) {
      return entry.physicalCode == std::vector<std::uint8_t>{1};
    });
    require(test, missing != entries.end(),
            "i8 multiply behavior is absent from the Fabric domain");
    missing->semanticValue = {0xfd};
  }
  if (kind == ConfigurationAbiKind::ExtraSemanticValue)
    entries.push_back({{0xfe}, {3}});
  const auto inactive = llvm::find_if(entries, [](const auto &entry) {
    return entry.physicalCode == std::vector<std::uint8_t>{2};
  });
  require(test, inactive != entries.end(),
          "i16 multiply behavior is absent from the Fabric domain");
  const std::vector<std::uint8_t> inactiveValue = inactive->semanticValue;
  ConfigurationFieldEncoding field{
      resolved.configurationFieldSchema.front(),
      FiniteCodebookEncoding{2, std::move(entries)},
      {{0, 0, 2}},
      inactiveValue};
  ProgrammingUnitDraft unit{{field.field.owner.catalog()}, 2, {field}};
  return take(test, finalizeConfigurationABI(
                        ConfigurationABIDraft{fixture.fabric.reference(),
                                              {std::move(unit)}},
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

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fabric,
                             const ConfigurationABI &abi,
                             llvm::StringRef symbol,
                             bool wrongConfigurationWidth = false) {
  const auto &resolved = capability(test, fabric);
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports =
      take(test, deriveFabricOperationLeafPorts(builder, resolved, abi));
  if (wrongConfigurationWidth) {
    require(test, ports.size() == 4,
            "configured vector multiply leaf did not have four ports");
    ports[2].type = builder.getI1Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(symbol), ports);
  return SkeletonFixture{std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

FabricOperationProviderRegistry makeProviderRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableFixedVectorIntegerMultiplyProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

std::string specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
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
          "portable vector multiply emitted external implementation state");
  return take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
}

void generatedRegistryAndProviderCoverage() {
  const llvm::StringRef test = __func__;
  const auto family =
      ::fabric::ImplementationFamilyId::FixedVectorIntegerMultiply;
  const ::fabric::ImplementationFamilyDescriptor &descriptor =
      ::fabric::implementationFamily(family);
  require(
      test,
      descriptor.familyId == family &&
          descriptor.capabilityParamsSchema ==
              ::fabric::CapabilityParamsSchemaId::FixedVectorIntegerParams &&
          descriptor.typedAdmissionProvider ==
              ::fabric::TypedAdmissionProviderId::
                  FixedVectorOrdinaryIntegerAdmission &&
          descriptor.admittedSchemas.size() == 1 &&
          descriptor.admittedSchemas.front() ==
              dataflow::OperationSchemaId::ArithMulI,
      "generated fixed-vector multiply descriptor changed");

  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  const auto entry = llvm::find_if(coverage, [&](const auto &candidate) {
    return candidate.implementationFamily == family;
  });
  require(test,
          coverage.size() == ::fabric::implementationFamilyCount() &&
              entry != coverage.end() &&
              entry->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "fixed-vector multiply provider coverage changed");
}

void configuredLaneBehaviorAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  const std::filesystem::path artifactRoot = root / "artifacts";
  std::filesystem::create_directories(artifactRoot);
  ArtifactStore store(artifactRoot.string());
  FabricFixture fabric = makeFabric(test, store);
  const auto &resolved = capability(test, fabric);
  require(
      test,
      resolved.implementationFamily ==
              ::fabric::ImplementationFamilyId::FixedVectorIntegerMultiply &&
          std::holds_alternative<::fabric::FixedVectorIntegerParams>(
              resolved.parameterizedCapability) &&
          resolved.configurationFieldSchema.size() == 1,
      "configured vector multiply capability changed its exact schema");
  const auto &parameters = std::get<::fabric::FixedVectorIntegerParams>(
      resolved.parameterizedCapability);
  require(test,
          parameters.maxPayloadBits == 33 &&
              parameters.elementWidths.size() == 2,
          "configured vector multiply parameters changed");

  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const auto &port : resolved.physicalPorts)
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
              inputs[0]->payloadWidthBits == 40 &&
              inputs[1]->payloadWidthBits == 40 &&
              outputs[0]->payloadWidthBits == 40,
          "configured vector multiply changed its physical port shape");

  const auto domain =
      take(test, resolved.resolveFiniteBehaviorDomain(fabricContext()));
  require(test,
          domain.size() == 2 && behaviorElementWidth(domain[0]) !=
                                    behaviorElementWidth(domain[1]),
          "Fabric did not preserve both configured element widths");
  for (const auto &point : domain) {
    const unsigned width = behaviorElementWidth(point);
    require(test, behaviorLaneCount(point) == (width == 8 ? 4 : 2),
            "Fabric projected the wrong maximal row-major lane count");
  }

  const auto field = resolved.configurationFieldSchema.front();
  require(
      test,
      configurationValue(test, resolved, field, multiplyActor({4}, 8)) ==
          configurationValue(
              test, resolved, field,
              multiplyActor({2, 2}, 8, mlir::arith::IntegerOverflowFlags::nsw)),
      "vector shape or overflow promise entered configuration bytes");
  require(test,
          configurationValue(test, resolved, field, multiplyActor({4}, 8)) !=
              configurationValue(test, resolved, field, multiplyActor({2}, 16)),
          "distinct element widths collapsed to one configuration value");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fabric, abi.abi(),
                                       "fixed_vector_integer_multiply");
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(
      test,
      ports.size() == 4 && ports.atInput(0).getName() == "data_input_0" &&
          ports.atInput(1).getName() == "data_input_1" &&
          ports.atInput(2).getName() == "config_0" &&
          ports.atOutput(0).getName() == "data_output_0" &&
          mlir::cast<mlir::IntegerType>(ports.atInput(0).type).getWidth() ==
              40 &&
          mlir::cast<mlir::IntegerType>(ports.atInput(2).type).getWidth() ==
              2 &&
          mlir::cast<mlir::IntegerType>(ports.atOutput(0).type).getWidth() ==
              40,
      "derived vector multiply leaf ports are not canonical");
  const std::string firstRtl = specialize(test, first, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second = makeSkeleton(test, *secondContext, fabric, abi.abi(),
                                        "fixed_vector_integer_multiply");
  const std::string secondRtl = specialize(test, second, fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical vector multiply inputs produced different RTL");
  require(
      test,
      llvm::StringRef(firstRtl).contains("config_0") &&
          llvm::StringRef(firstRtl).count(" * ") == 6 &&
          !llvm::StringRef(firstRtl).contains("data_input_0 * data_input_1"),
      "portable provider did not emit independent lane multipliers");

  std::ofstream(root / "fixed_vector_integer_multiply.sv") << firstRtl;
  std::ofstream(root / "testbench.sv") << R"sv(
module testbench;
  logic [39:0] data_input_0;
  logic [39:0] data_input_1;
  logic [1:0] config_0;
  logic [39:0] data_output_0;

  fixed_vector_integer_multiply dut(.*);

  initial begin
    data_input_0 = 40'ha580ff0302;
    data_input_1 = 40'h5a02020304;

    config_0 = 2'b01;
    #1;
    if (data_output_0 !== 40'h0000fe0908)
      $fatal(1, "i8 packed-wide multiplication corrupted independent lanes");

    config_0 = 2'b10;
    #1;
    if (data_output_0 !== 40'h00fffe1208)
      $fatal(1, "i16 lane-wise modulo multiplication failed");

    config_0 = 2'b00;
    #1;
    if (data_output_0 !== 40'h00fffe1208)
      $fatal(1, "unassigned code did not preserve inactive width behavior");
    $finish;
  end
endmodule
)sv";
  std::ofstream(root / "portable_fixed_vector_integer_multiply.ys") << R"ys(
read_verilog -sv fixed_vector_integer_multiply.sv
hierarchy -check -top fixed_vector_integer_multiply
proc
opt
check -assert
select -assert-count 6 t:$mul
synth -top fixed_vector_integer_multiply
check -assert
stat
)ys";
}

void singletonNeedsNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  const std::filesystem::path artifactRoot = root / "artifacts";
  std::filesystem::create_directories(artifactRoot);
  ArtifactStore store(artifactRoot.string());
  FabricFixture fabric = makeFabric(test, store, FabricFixtureKind::Singleton);
  const auto &resolved = capability(test, fabric);
  require(test, resolved.configurationFieldSchema.empty(),
          "singleton vector multiply retained a configuration field");
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi(),
                                          "fixed_vector_integer_multiply_i8");
  require(test, skeleton.leaf.getPortList().size() == 3,
          "singleton vector multiply retained a redundant selector");
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test,
          !llvm::StringRef(rtl).contains("config_") &&
              llvm::StringRef(rtl).count(" * ") == 4,
          "singleton vector multiply did not lower directly by lane");
}

void malformedAndUnsupportedInputsAreTransactional(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  const std::filesystem::path artifactRoot = root / "artifacts";
  std::filesystem::create_directories(artifactRoot);
  ArtifactStore store(artifactRoot.string());
  expectFabricParseError(test, FabricFixtureKind::WrongSchema,
                         "not admitted by implementation family");

  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  ExternalImplementationContractCatalog externalContracts;
  FabricFixture valid = makeFabric(test, store);
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  const std::vector<FabricOperationRecipeBinding> validRecipes = {
      {valid.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};

  std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
  SkeletonFixture malformed =
      makeSkeleton(test, *malformedContext, valid, validAbi.abi(),
                   "malformed_vector_multiply", true);
  const std::string malformedBefore = moduleText(*malformed.module);
  const std::vector<FabricOperationLeafAssociation> malformedAssociations = {
      {malformed.leaf, valid.occurrence}};
  expectError(test,
              specializeFabricOperationLeaves(*malformed.module, valid.fabric,
                                              validAbi, malformedAssociations,
                                              validRecipes, registry,
                                              externalContracts),
              "leaf port");
  require(test, moduleText(*malformed.module) == malformedBefore,
          "malformed vector leaf partially mutated the caller module");

  for (ConfigurationAbiKind kind : {ConfigurationAbiKind::MissingI8,
                                    ConfigurationAbiKind::ExtraSemanticValue}) {
    FinalizedConfigurationABI malformedAbi =
        makeConfigurationAbi(test, store, valid, kind);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(
        test, *context, valid, malformedAbi.abi(), "malformed_vector_codebook");
    const std::string before = moduleText(*skeleton.module);
    const std::vector<FabricOperationLeafAssociation> associations = {
        {skeleton.leaf, valid.occurrence}};
    expectError(test,
                specializeFabricOperationLeaves(
                    *skeleton.module, valid.fabric, malformedAbi, associations,
                    validRecipes, registry, externalContracts),
                kind == ConfigurationAbiKind::MissingI8
                    ? "admitted semantic value"
                    : "configuration domain");
    require(test, moduleText(*skeleton.module) == before,
            "malformed vector codebook partially mutated the caller module");
  }

  FabricFixture unsupported =
      makeFabric(test, store, FabricFixtureKind::UnsupportedContract);
  FinalizedConfigurationABI unsupportedAbi =
      makeConfigurationAbi(test, store, unsupported);
  std::unique_ptr<mlir::MLIRContext> unsupportedContext = makeCirctContext();
  SkeletonFixture unsupportedSkeleton =
      makeSkeleton(test, *unsupportedContext, unsupported, unsupportedAbi.abi(),
                   "unsupported_vector_contract");
  const std::string unsupportedBefore = moduleText(*unsupportedSkeleton.module);
  const std::vector<FabricOperationLeafAssociation> unsupportedAssociations = {
      {unsupportedSkeleton.leaf, unsupported.occurrence}};
  const std::vector<FabricOperationRecipeBinding> unsupportedRecipes = {
      {unsupported.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectTypedUnsupported(
      test,
      specializeFabricOperationLeaves(
          *unsupportedSkeleton.module, unsupported.fabric, unsupportedAbi,
          unsupportedAssociations, unsupportedRecipes, registry,
          externalContracts),
      ::fabric::ImplementationFamilyId::FixedVectorIntegerMultiply,
      BackendRecipeKey::PortableSystemVerilog,
      "unsupported vector multiply resource contract");
  require(test, moduleText(*unsupportedSkeleton.module) == unsupportedBefore,
          "unsupported vector contract partially mutated the caller module");

  constexpr std::array nativeRecipes = {
      BackendRecipeKey::SynopsysDesignWare,
      BackendRecipeKey::CadenceChipWare,
      BackendRecipeKey::AmdXilinx,
      BackendRecipeKey::IntelAltera,
  };
  for (BackendRecipeKey recipe : nativeRecipes) {
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(
        test, *context, valid, validAbi.abi(), "unsupported_native_recipe");
    const std::string before = moduleText(*skeleton.module);
    const std::vector<FabricOperationLeafAssociation> associations = {
        {skeleton.leaf, valid.occurrence}};
    const std::vector<FabricOperationRecipeBinding> recipes = {
        {valid.occurrence, recipe, {}}};
    expectTypedUnsupported(
        test,
        specializeFabricOperationLeaves(*skeleton.module, valid.fabric,
                                        validAbi, associations, recipes,
                                        registry, externalContracts),
        ::fabric::ImplementationFamilyId::FixedVectorIntegerMultiply, recipe,
        "backend-native vector multiply recipe");
    require(test, moduleText(*skeleton.module) == before,
            "unsupported backend recipe partially mutated the caller module");
  }

  FabricFixture other = makeFabric(test, store, FabricFixtureKind::OtherFamily);
  FinalizedConfigurationABI otherAbi = makeConfigurationAbi(test, store, other);
  std::unique_ptr<mlir::MLIRContext> otherContext = makeCirctContext();
  SkeletonFixture otherSkeleton = makeSkeleton(
      test, *otherContext, other, otherAbi.abi(), "unsupported_vector_add");
  const std::string otherBefore = moduleText(*otherSkeleton.module);
  const std::vector<FabricOperationLeafAssociation> otherAssociations = {
      {otherSkeleton.leaf, other.occurrence}};
  const std::vector<FabricOperationRecipeBinding> otherRecipes = {
      {other.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectTypedUnsupported(
      test,
      specializeFabricOperationLeaves(*otherSkeleton.module, other.fabric,
                                      otherAbi, otherAssociations, otherRecipes,
                                      registry, externalContracts),
      ::fabric::ImplementationFamilyId::FixedVectorIntegerAddSub,
      BackendRecipeKey::PortableSystemVerilog, "wrong-family capability");
  require(test, moduleText(*otherSkeleton.module) == otherBefore,
          "wrong-family capability partially mutated the caller module");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  generatedRegistryAndProviderCoverage();
  configuredLaneBehaviorAndDeterminism(root);
  singletonNeedsNoSelector(root / "singleton");
  malformedAndUnsupportedInputsAreTransactional(root / "invalid");
  return 0;
}

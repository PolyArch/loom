#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/MathErf.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/SpecialMathAccuracy.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Simulator/OperationSemantics.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
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

enum class FixtureKind {
  Configured,
  Singleton,
  Max2Ulp,
  Max4Ulp,
  UnsupportedF32,
  UnsupportedF64,
  UnsupportedBehavior,
  CorrectlyRounded,
  UnsupportedContract,
};

enum class AbiKind { Complete, MissingBehavior, ExtraBehavior, DirectEncoding };

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

struct FloatLayout final {
  unsigned exponentBits;
  unsigned fractionBits;

  std::uint16_t sign() const { return UINT16_C(1) << 15; }
  std::uint16_t exponentMask() const {
    return ((UINT16_C(1) << exponentBits) - 1) << fractionBits;
  }
  std::uint16_t fractionMask() const {
    return (UINT16_C(1) << fractionBits) - 1;
  }
  std::uint16_t infinity() const { return exponentMask(); }
  std::uint16_t one() const {
    return ((UINT16_C(1) << (exponentBits - 1)) - 1) << fractionBits;
  }
  std::uint16_t four() const { return one() + (UINT16_C(2) << fractionBits); }
  std::uint16_t quietBit() const { return UINT16_C(1) << (fractionBits - 1); }
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

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted malformed scalar erf input");
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

std::string fabricSource(FixtureKind kind,
                         llvm::StringRef schema = "math.erf") {
  const bool configured = kind == FixtureKind::Configured;
  const bool f32 = kind == FixtureKind::UnsupportedF32;
  const bool f64 = kind == FixtureKind::UnsupportedF64;
  const bool exact = kind == FixtureKind::CorrectlyRounded;
  const bool numberPreferred = kind == FixtureKind::UnsupportedBehavior;
  const unsigned width = f64 ? 64 : (f32 ? 32 : 21);
  const llvm::StringRef formats =
      configured ? R"mlir(["f16", "bf16"])mlir"
                 : (f64 ? R"mlir(["f64"])mlir"
                        : (f32 ? R"mlir(["f32"])mlir" : R"mlir(["f16"])mlir"));
  const llvm::StringRef accuracy =
      exact ? "CorrectlyRounded"
            : (kind == FixtureKind::Max2Ulp
                   ? "Max2Ulp"
                   : (kind == FixtureKind::Max4Ulp ? "Max4Ulp" : "Max1Ulp"));
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "module {\n"
         << "  fabric.module @scalar_math_erf(%input: !fabric.bits<" << width
         << ">) -> !fabric.bits<" << width << "> {\n"
         << "    %pe = fabric.pe [spatial]\n"
         << "        (%pe_input = %input : !fabric.bits<" << width
         << ">) -> !fabric.bits<" << width << "> {\n"
         << "      %fu = fabric.fu\n"
         << "          (%fu_input = %pe_input : !fabric.bits<" << width
         << ">) -> !fabric.bits<" << width << "> {\n"
         << "        %value = fabric.op [@" << schema << "] (%fu_input)\n"
         << "          {implementation_family = "
            "#fabric.implementation_family<ScalarMathErf>,\n"
         << "           hw_params = {float_formats = " << formats
         << ", behavior = {rounding_modes = [\"to_nearest_even\"], "
            "nan_behaviors = [\""
         << (numberPreferred ? "number_preferred" : "ieee")
         << "\"], subnormal_behaviors = [\"preserve\"], "
            "signed_zero_behaviors = [\"preserve\"], fastmath = \""
         << (exact ? "none" : "afn") << "\"}, accuracy_guarantee = \""
         << accuracy << "\"}}\n"
         << "          : (!fabric.bits<" << width << ">) -> !fabric.bits<"
         << width << ">\n"
         << "        fabric.yield %value : !fabric.bits<" << width << ">\n"
         << "      }\n"
         << "    }\n"
         << "    fabric.yield %pe : !fabric.bits<" << width << ">\n"
         << "  }\n"
         << "}\n";
  return output.str();
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         FixtureKind kind) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(fabricSource(kind),
                                                        &fabricContext());
  require(test, static_cast<bool>(source), "could not parse erf fixture");
  const ::fabric::ResourceContract &resourceContract =
      kind == FixtureKind::UnsupportedContract
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
  require(test, static_cast<bool>(root), "erf fixture has no Fabric root");
  FinalizedFabricRoot fabric =
      take(test, loom::fabric::finalizeFabricRoot(root, store));
  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &capability :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (capability.implementationFamily !=
          ::fabric::ImplementationFamilyId::ScalarMathErf)
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
              "system has no physical scalar erf occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no scalar erf occurrence");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "scalar erf capability did not resolve");
  return *result;
}

::fabric::FloatFormat
behaviorFormat(llvm::StringRef test,
               const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  mlir::Type type = point.representativeActor.type.getInput(0);
  if (mlir::isa<mlir::Float16Type>(type))
    return ::fabric::FloatFormat::F16;
  if (mlir::isa<mlir::BFloat16Type>(type))
    return ::fabric::FloatFormat::BF16;
  if (mlir::isa<mlir::Float32Type>(type))
    return ::fabric::FloatFormat::F32;
  fail(test, "Fabric projected an unexpected scalar erf format");
}

std::uint8_t physicalCode(::fabric::FloatFormat format) {
  return format == ::fabric::FloatFormat::F16 ? 1 : 2;
}

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "scalar erf capability has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 2,
          "configured scalar erf relation is not the exact format quotient");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "configured scalar erf behavior has no semantic value");
    const auto format = behaviorFormat(test, point);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (format == ::fabric::FloatFormat::F16)
      inactive = semantic;
    if (kind == AbiKind::MissingBehavior &&
        format == ::fabric::FloatFormat::BF16)
      semantic = {0xfd};
    entries.push_back({std::move(semantic), {physicalCode(format)}});
  }
  if (kind == AbiKind::ExtraBehavior)
    entries.push_back({{0xfe}, {3}});
  require(test, !inactive.empty(), "scalar erf ABI has no inactive format");
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  SemanticFieldEncoding encoding =
      kind == AbiKind::DirectEncoding
          ? SemanticFieldEncoding{DirectBitsEncoding{2}}
          : SemanticFieldEncoding{
                FiniteCodebookEncoding{2, std::move(entries)}};
  if (kind == AbiKind::DirectEncoding)
    inactive = {0};
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, std::move(encoding), std::move(inactive)};
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, {std::move(field)}));
}

FinalizedConfigurationABI
makeConfigurationAbi(llvm::StringRef test, const ArtifactStore &store,
                     const FabricFixture &fixture,
                     AbiKind kind = AbiKind::Complete) {
  return take(test, finalizeConfigurationABI(
                        makeConfigurationAbiDraft(test, fixture, kind), store));
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

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fabric,
                             const ConfigurationABI &abi,
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
      take(test, deriveFabricOperationLeafPorts(
                     builder, fabric.physicalOccurrence, resolved, abi));
  if (wrongConfigurationWidth) {
    const auto field = llvm::find_if(
        ports, [](const auto &port) { return port.getName() == "config_0"; });
    require(test, field != ports.end(),
            "configured scalar erf leaf has no selector");
    field->type = builder.getI1Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("scalar_math_erf"), ports);
  return SkeletonFixture{std::move(module), leaf};
}

FabricOperationProviderRegistry makeProviderRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableMathErfProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

bool hasPortableCoverage(const FabricOperationProviderRegistry &registry) {
  return llvm::any_of(registry.coverage(), [](const auto &entry) {
    return entry.implementationFamily ==
               ::fabric::ImplementationFamilyId::ScalarMathErf &&
           entry.recipes == std::vector<BackendRecipeKey>{
                                BackendRecipeKey::PortableSystemVerilog};
  });
}

std::string moduleText(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

llvm::Expected<FabricOperationProviderOutput> trySpecialize(
    SkeletonFixture &skeleton, const FabricFixture &fabric,
    const FinalizedConfigurationABI &abi,
    FabricOperationProviderRegistry &registry,
    BackendRecipeKey recipe = BackendRecipeKey::PortableSystemVerilog) {
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, recipe, {}}};
  return specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                         recipes, registry, externalContracts);
}

std::string specialize(llvm::StringRef test, SkeletonFixture skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
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
          "portable scalar erf provider emitted external state");
  return std::move(conformance.systemVerilog);
}

FloatLayout layout(::fabric::FloatFormat format) {
  return format == ::fabric::FloatFormat::F16 ? FloatLayout{5, 10}
                                              : FloatLayout{8, 7};
}

const llvm::fltSemantics &semantics(::fabric::FloatFormat format) {
  return format == ::fabric::FloatFormat::F16 ? llvm::APFloat::IEEEhalf()
                                              : llvm::APFloat::BFloat();
}

mlir::Type floatingType(::fabric::FloatFormat format) {
  return format == ::fabric::FloatFormat::F16
             ? mlir::Type(mlir::Float16Type::get(&fabricContext()))
             : mlir::Type(mlir::BFloat16Type::get(&fabricContext()));
}

std::vector<std::uint16_t> oracleTable(llvm::StringRef test,
                                       ::fabric::FloatFormat format) {
  mlir::Type type = floatingType(format);
  loom::sim::PrimitiveOperationDescriptor operation{
      ::dataflow::CanonicalActorSchemaProjection{
          ::dataflow::OperationSchemaId::MathErf,
          mlir::FunctionType::get(&fabricContext(), {type}, {type}),
          ::dataflow::SpecialMathPayload{
              mlir::arith::FastMathFlags::afn,
              loom::SpecialMathAccuracyTier::Max1Ulp}},
      16, 16};
  std::vector<std::uint16_t> result(UINT16_MAX + std::size_t{1});
  for (std::uint32_t bits = 0; bits <= UINT16_MAX; ++bits) {
    llvm::APFloat input(semantics(format), llvm::APInt(16, bits));
    loom::sim::PrimitiveValue evaluated = take(
        test, loom::sim::evaluatePrimitiveOperation(
                  operation, {loom::sim::PrimitiveValue::floating(input)}));
    require(test, evaluated.isDefined() && evaluated.bits->getBitWidth() == 16,
            "MPFR/APFloat erf oracle returned an invalid value");
    result[bits] = static_cast<std::uint16_t>(evaluated.bits->getZExtValue());
  }
  return result;
}

bool isNaN(std::uint16_t bits, const FloatLayout &shape) {
  return (bits & shape.exponentMask()) == shape.exponentMask() &&
         (bits & shape.fractionMask()) != 0;
}

void verifyOracleInvariants(llvm::StringRef test, ::fabric::FloatFormat format,
                            llvm::ArrayRef<std::uint16_t> oracle) {
  const FloatLayout shape = layout(format);
  require(test,
          oracle[0] == 0 && oracle[shape.sign()] == shape.sign() &&
              oracle[shape.infinity()] == shape.one() &&
              oracle[shape.sign() | shape.infinity()] ==
                  (shape.sign() | shape.one()),
          "erf oracle lost signed zero or infinite limits");
  require(test,
          oracle[1] == 1 && oracle[shape.sign() | 1] == (shape.sign() | 1),
          "erf oracle lost the minimum subnormal near zero");
  require(test,
          oracle[shape.four()] == shape.one() &&
              oracle[shape.sign() | shape.four()] ==
                  (shape.sign() | shape.one()),
          "erf oracle did not saturate at magnitude four");
  for (std::uint32_t bits = shape.four(); bits < shape.infinity(); ++bits)
    require(test, oracle[bits] == shape.one(),
            "positive erf saturation is incomplete");
  for (std::uint32_t bits = 0; bits < shape.sign(); ++bits) {
    if (isNaN(bits, shape)) {
      require(test, oracle[bits] == (bits | shape.quietBit()),
              "erf oracle did not quiet a NaN payload");
      continue;
    }
    require(test, oracle[bits | shape.sign()] == (oracle[bits] | shape.sign()),
            "erf oracle violated odd symmetry");
  }
}

std::string oracleHex(llvm::ArrayRef<std::uint16_t> oracle) {
  std::string text;
  llvm::raw_string_ostream output(text);
  for (std::uint16_t value : oracle) {
    llvm::SmallString<8> digits;
    llvm::APInt(16, value).toStringUnsigned(digits, 16);
    output << std::string(4 - digits.size(), '0') << digits << '\n';
  }
  return output.str();
}

std::string testbench() {
  return R"sv(module testbench;
  logic [20:0] operand;
  logic [1:0] mode_select;
  wire [20:0] result;
  logic [15:0] expected_f16 [0:65535];
  logic [15:0] expected_bf16 [0:65535];
  logic [15:0] positive_result;
  integer index;

  scalar_math_erf dut(
      .data_input_0(operand), .config_0(mode_select),
      .data_output_0(result));

  function automatic logic is_nan(
      input logic [15:0] value, input logic bfloat);
    begin
      if (bfloat)
        is_nan = (&value[14:7]) && (|value[6:0]);
      else
        is_nan = (&value[14:10]) && (|value[9:0]);
    end
  endfunction

  function automatic [16:0] ordered_key(input logic [15:0] value);
    begin
      ordered_key = value[15] ? {1'b0, ~value} : {1'b1, value};
    end
  endfunction

  task automatic check_value(
      input logic [1:0] mode,
      input logic [15:0] value,
      input logic [15:0] expected,
      input logic bfloat);
    logic [16:0] actual_key;
    logic [16:0] expected_key;
    logic [16:0] distance;
    begin
      mode_select = mode;
      operand = {5'b10101, value};
      #1;
      if (result[20:16] !== 5'b00000)
        $fatal(1, "scalar erf output padding is nonzero mode=%0d input=%h",
               mode, value);
      actual_key = ordered_key(result[15:0]);
      expected_key = ordered_key(expected);
      distance = actual_key > expected_key ? actual_key - expected_key
                                           : expected_key - actual_key;
      if (is_nan(expected, bfloat) ||
          ((!bfloat && (&expected[14:10])) ||
           (bfloat && (&expected[14:7]))) || expected[14:0] == 0) begin
        if (result[15:0] !== expected)
          $fatal(1, "scalar erf exceptional mismatch mode=%0d input=%h got=%h expected=%h",
                 mode, value, result[15:0], expected);
      end else if (distance > 1) begin
        $fatal(1, "scalar erf exceeds one ULP mode=%0d input=%h got=%h expected=%h distance=%0d",
               mode, value, result[15:0], expected, distance);
      end
    end
  endtask

  task automatic check_exact(
      input logic [1:0] mode,
      input logic [15:0] value,
      input logic [15:0] expected);
    begin
      mode_select = mode;
      operand = {5'b11111, value};
      #1;
      if (result !== {5'b00000, expected})
        $fatal(1, "scalar erf exact mismatch mode=%0d input=%h got=%h expected=%h",
               mode, value, result[15:0], expected);
    end
  endtask

  initial begin
    $readmemh("erf_f16_expected.hex", expected_f16);
    $readmemh("erf_bf16_expected.hex", expected_bf16);
    operand = '0;
    mode_select = '0;
    for (index = 0; index < 65536; index = index + 1)
      check_value(2'd1, index[15:0], expected_f16[index], 1'b0);
    for (index = 0; index < 65536; index = index + 1)
      check_value(2'd2, index[15:0], expected_bf16[index], 1'b1);

    for (index = 0; index < 16'h7c00; index = index + 1) begin
      mode_select = 2'd1;
      operand = {5'b01010, index[15:0]};
      #1;
      positive_result = result[15:0];
      operand = {5'b10101, 1'b1, index[14:0]};
      #1;
      if (result !== {5'b00000, 1'b1, positive_result[14:0]})
        $fatal(1, "scalar erf f16 odd symmetry mismatch input=%h", index[15:0]);
    end
    for (index = 0; index < 16'h7f80; index = index + 1) begin
      mode_select = 2'd2;
      operand = {5'b01010, index[15:0]};
      #1;
      positive_result = result[15:0];
      operand = {5'b10101, 1'b1, index[14:0]};
      #1;
      if (result !== {5'b00000, 1'b1, positive_result[14:0]})
        $fatal(1, "scalar erf bf16 odd symmetry mismatch input=%h", index[15:0]);
    end

    check_exact(2'd1, 16'h0000, 16'h0000);
    check_exact(2'd1, 16'h8000, 16'h8000);
    check_exact(2'd1, 16'h0001, 16'h0001);
    check_exact(2'd1, 16'h7c00, 16'h3c00);
    check_exact(2'd1, 16'hfc00, 16'hbc00);
    check_exact(2'd1, 16'h4400, 16'h3c00);
    check_exact(2'd1, 16'h7bff, 16'h3c00);
    check_exact(2'd1, 16'hfbff, 16'hbc00);
    check_exact(2'd1, 16'h7c15, 16'h7e15);
    check_exact(2'd2, 16'h0000, 16'h0000);
    check_exact(2'd2, 16'h8000, 16'h8000);
    check_exact(2'd2, 16'h0001, 16'h0001);
    check_exact(2'd2, 16'h7f80, 16'h3f80);
    check_exact(2'd2, 16'hff80, 16'hbf80);
    check_exact(2'd2, 16'h4080, 16'h3f80);
    check_exact(2'd2, 16'h7f7f, 16'h3f80);
    check_exact(2'd2, 16'hff7f, 16'hbf80);
    check_exact(2'd2, 16'h7f95, 16'h7fd5);
    $display("portable scalar erf PASS");
    $finish;
  end
endmodule
)sv";
}

std::string synthesisTop() {
  return R"sv(module scalar_math_erf_synthesis_top(
    input [20:0] operand,
    input [1:0] mode_select,
    output [20:0] result);
  scalar_math_erf dut(
      .data_input_0(operand), .config_0(mode_select),
      .data_output_0(result));
endmodule
)sv";
}

std::string yosysScript() {
  return R"ys(read_verilog -sv scalar_math_erf.sv synthesis_top.sv
hierarchy -check -top scalar_math_erf_synthesis_top
proc
memory_map
opt_clean
check -assert
select -assert-none t:$*ff* t:$*latch* t:$mem* t:$_*FF* t:$_*LATCH*
synth -noabc -top scalar_math_erf_synthesis_top
check -assert
select -assert-none t:$*ff* t:$*latch* t:$mem* t:$_*FF* t:$_*LATCH*
stat
)ys";
}

std::string emit(llvm::StringRef test, const ArtifactStore &store,
                 FixtureKind kind) {
  FabricFixture fabric = makeFabric(test, store, kind);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  return specialize(test, std::move(skeleton), fabric, abi);
}

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  require(test, hasPortableCoverage(registry),
          "portable ScalarMathErf provider is absent");

  FabricFixture fabric = makeFabric(test, store, FixtureKind::Configured);
  const auto &resolved = capability(test, fabric);
  require(test,
          resolved.implementationFamily ==
                  ::fabric::ImplementationFamilyId::ScalarMathErf &&
              resolved.enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{
                      ::dataflow::OperationSchemaId::MathErf} &&
              std::holds_alternative<::fabric::ScalarSpecialMathParams>(
                  resolved.parameterizedCapability),
          "scalar erf escaped its generated family descriptor");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 2,
          "scalar erf did not consume the sealed format relation");
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test,
            point.representativeActor.schema ==
                    ::dataflow::OperationSchemaId::MathErf &&
                point.semanticConfiguration.has_value() &&
                point.operandPorts == std::vector<std::uint64_t>{0} &&
                point.resultPorts == std::vector<std::uint64_t>{0},
            "scalar erf relation contains a malformed witness");
  }

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> portContext = makeCirctContext();
  SkeletonFixture ports = makeSkeleton(test, *portContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo portInfo(ports.leaf.getPortList());
  require(test,
          portInfo.size() == 3 &&
              portInfo.atInput(0).getName() == "data_input_0" &&
              portInfo.atInput(0).type ==
                  mlir::IntegerType::get(portContext.get(), 21) &&
              portInfo.atInput(1).getName() == "config_0" &&
              portInfo.atInput(1).type ==
                  mlir::IntegerType::get(portContext.get(), 2) &&
              portInfo.atOutput(0).getName() == "data_output_0" &&
              portInfo.atOutput(0).type ==
                  mlir::IntegerType::get(portContext.get(), 21),
          "scalar erf leaf ports do not follow ConfigurationABI 3.0");

  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  std::unique_ptr<mlir::MLIRContext> thirdContext = makeCirctContext();
  const std::string first = specialize(
      test, makeSkeleton(test, *firstContext, fabric, abi.abi()), fabric, abi);
  const std::string second = specialize(
      test, makeSkeleton(test, *secondContext, fabric, abi.abi()), fabric, abi);
  const std::string third = specialize(
      test, makeSkeleton(test, *thirdContext, fabric, abi.abi()), fabric, abi);
  require(test, first == second && second == third,
          "identical scalar erf inputs produced different SystemVerilog");
  const llvm::StringRef rtl(first);
  require(test,
          rtl.contains("function automatic") && rtl.contains("config_0") &&
              rtl.contains("loom_erf_e5_f10") &&
              rtl.contains("loom_erf_e8_f7") && !rtl.contains("shortreal") &&
              !rtl.contains(" real") && !rtl.contains("DPI") &&
              !rtl.contains("$erf") && !rtl.contains("$exp"),
          "scalar erf RTL is incomplete or not synthesizable bit logic");

  const auto f16Oracle = oracleTable(test, ::fabric::FloatFormat::F16);
  const auto bf16Oracle = oracleTable(test, ::fabric::FloatFormat::BF16);
  verifyOracleInvariants(test, ::fabric::FloatFormat::F16, f16Oracle);
  verifyOracleInvariants(test, ::fabric::FloatFormat::BF16, bf16Oracle);
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated",
          {{"scalar_math_erf.sv", first},
           {"testbench.sv", testbench()},
           {"synthesis_top.sv", synthesisTop()},
           {"portable_scalar_math_erf.ys", yosysScript()},
           {"erf_f16_expected.hex", oracleHex(f16Oracle)},
           {"erf_bf16_expected.hex", oracleHex(bf16Oracle)}}))
    fail(test, llvm::toString(std::move(error)));
}

void singletonHasNoConfigurationSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  const std::string rtl = emit(test, store, FixtureKind::Singleton);
  require(test,
          llvm::StringRef(rtl).contains("loom_erf_e5_f10") &&
              !llvm::StringRef(rtl).contains("config_0"),
          "singleton scalar erf relation retained a selector");
}

void weakerAccuracyGuaranteesReuseTheCircuit(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  const std::string oneUlp = emit(test, store, FixtureKind::Singleton);
  const std::string twoUlp = emit(test, store, FixtureKind::Max2Ulp);
  const std::string fourUlp = emit(test, store, FixtureKind::Max4Ulp);
  require(test, oneUlp == twoUlp && twoUlp == fourUlp,
          "accepted accuracy tier changed the scalar erf circuit");
}

void expectTypedUnsupported(
    llvm::StringRef test, llvm::Expected<FabricOperationProviderOutput> result,
    BackendRecipeKey recipe) {
  require(test, !result, "unsupported scalar erf combination was accepted");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() ==
                         ::fabric::ImplementationFamilyId::ScalarMathErf &&
                     error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "scalar erf returned the wrong unsupported error: " +
                       error.message());
      });
  require(test, classified,
          "scalar erf lost its typed Unsupported classification");
}

void malformedAndUnsupportedInputsAreTransactional(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  FabricFixture valid = makeFabric(test, store, FixtureKind::Configured);

  for (AbiKind kind : {AbiKind::MissingBehavior, AbiKind::ExtraBehavior,
                       AbiKind::DirectEncoding})
    expectError(
        test,
        finalizeConfigurationABI(makeConfigurationAbiDraft(test, valid, kind),
                                 store),
        kind == AbiKind::DirectEncoding ? "finite codebook" : "semantic");

  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
  SkeletonFixture malformed =
      makeSkeleton(test, *malformedContext, valid, validAbi.abi(), true);
  const std::string malformedBefore = moduleText(*malformed.module);
  expectError(test, trySpecialize(malformed, valid, validAbi, registry),
              "leaf port");
  require(test, moduleText(*malformed.module) == malformedBefore,
          "malformed scalar erf input partially mutated the skeleton");

  for (FixtureKind kind :
       {FixtureKind::UnsupportedF32, FixtureKind::UnsupportedF64,
        FixtureKind::UnsupportedBehavior, FixtureKind::CorrectlyRounded,
        FixtureKind::UnsupportedContract}) {
    FabricFixture unsupported = makeFabric(test, store, kind);
    FinalizedConfigurationABI unsupportedAbi =
        makeConfigurationAbi(test, store, unsupported);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton =
        makeSkeleton(test, *context, unsupported, unsupportedAbi.abi());
    const std::string before = moduleText(*skeleton.module);
    expectTypedUnsupported(
        test, trySpecialize(skeleton, unsupported, unsupportedAbi, registry),
        BackendRecipeKey::PortableSystemVerilog);
    require(test, moduleText(*skeleton.module) == before,
            "Unsupported scalar erf request mutated the skeleton");
  }

  for (BackendRecipeKey recipe :
       {BackendRecipeKey::SynopsysDesignWare, BackendRecipeKey::CadenceChipWare,
        BackendRecipeKey::AmdXilinx, BackendRecipeKey::IntelAltera}) {
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture native =
        makeSkeleton(test, *context, valid, validAbi.abi());
    const std::string before = moduleText(*native.module);
    expectTypedUnsupported(
        test, trySpecialize(native, valid, validAbi, registry, recipe), recipe);
    require(test, moduleText(*native.module) == before,
            "unsupported native scalar erf recipe mutated the skeleton");
  }

  mlir::ParserConfig parserConfig(&fabricContext(), false);
  auto foreign = mlir::parseSourceString<mlir::ModuleOp>(
      fabricSource(FixtureKind::Singleton, "math.sqrt"), parserConfig);
  require(test, static_cast<bool>(foreign),
          "could not parse foreign-schema scalar erf fixture");
  std::vector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      &fabricContext(), [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return mlir::success();
      });
  require(test, mlir::failed(mlir::verify(*foreign)),
          "foreign scalar erf schema passed verification");
  require(test,
          llvm::any_of(
              diagnostics,
              [](const std::string &diagnostic) {
                return llvm::StringRef(diagnostic).contains("not admitted");
              }),
          diagnostics.empty() ? "Fabric verifier produced no diagnostic"
                              : diagnostics.front());
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  configuredBehaviorAndArtifacts(root / "configured");
  singletonHasNoConfigurationSelector(root / "singleton");
  weakerAccuracyGuaranteesReuseTheCircuit(root / "accuracy");
  malformedAndUnsupportedInputsAreTransactional(root / "malformed");
  return EXIT_SUCCESS;
}

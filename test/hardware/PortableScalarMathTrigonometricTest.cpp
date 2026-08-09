#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/ScalarMathTrigonometric.h"
#include "PortableProviderTestSupport.h"

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
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mpfr.h>
#include <sstream>
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

enum class TrigFamily { Sin, Cos, Tan };
enum class FixtureKind {
  Configured,
  Singleton,
  ApproximateSingleton,
  UnsupportedFormat,
  UnsupportedBehavior,
  UnsupportedContract
};
enum class AbiKind { Complete, MissingBehavior, ExtraBehavior, DirectEncoding };

struct FabricFixture final {
  TrigFamily family;
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
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
    fail(test, "accepted malformed trigonometric input");
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

constexpr std::array kFamilies = {TrigFamily::Sin, TrigFamily::Cos,
                                  TrigFamily::Tan};

::fabric::ImplementationFamilyId familyId(TrigFamily family) {
  switch (family) {
  case TrigFamily::Sin:
    return ::fabric::ImplementationFamilyId::ScalarMathSin;
  case TrigFamily::Cos:
    return ::fabric::ImplementationFamilyId::ScalarMathCos;
  case TrigFamily::Tan:
    return ::fabric::ImplementationFamilyId::ScalarMathTan;
  }
  llvm_unreachable("unknown trigonometric family");
}

::dataflow::OperationSchemaId schemaId(TrigFamily family) {
  switch (family) {
  case TrigFamily::Sin:
    return ::dataflow::OperationSchemaId::MathSin;
  case TrigFamily::Cos:
    return ::dataflow::OperationSchemaId::MathCos;
  case TrigFamily::Tan:
    return ::dataflow::OperationSchemaId::MathTan;
  }
  llvm_unreachable("unknown trigonometric family");
}

llvm::StringRef familyKeyword(TrigFamily family) {
  switch (family) {
  case TrigFamily::Sin:
    return "ScalarMathSin";
  case TrigFamily::Cos:
    return "ScalarMathCos";
  case TrigFamily::Tan:
    return "ScalarMathTan";
  }
  llvm_unreachable("unknown trigonometric family");
}

llvm::StringRef schemaKeyword(TrigFamily family) {
  switch (family) {
  case TrigFamily::Sin:
    return "math.sin";
  case TrigFamily::Cos:
    return "math.cos";
  case TrigFamily::Tan:
    return "math.tan";
  }
  llvm_unreachable("unknown trigonometric family");
}

llvm::StringRef shortName(TrigFamily family) {
  switch (family) {
  case TrigFamily::Sin:
    return "sin";
  case TrigFamily::Cos:
    return "cos";
  case TrigFamily::Tan:
    return "tan";
  }
  llvm_unreachable("unknown trigonometric family");
}

std::string moduleName(TrigFamily family) {
  return "scalar_math_" + shortName(family).str();
}

std::string fabricSource(TrigFamily family, FixtureKind kind) {
  llvm::StringRef formats = R"mlir(["f16", "bf16"])mlir";
  llvm::StringRef nan = "ieee";
  llvm::StringRef signedZero = "preserve";
  llvm::StringRef fastMath = "none";
  llvm::StringRef accuracy = "CorrectlyRounded";
  if (kind == FixtureKind::Singleton ||
      kind == FixtureKind::ApproximateSingleton ||
      kind == FixtureKind::UnsupportedBehavior)
    formats = R"mlir(["f16"])mlir";
  if (kind == FixtureKind::UnsupportedFormat)
    formats = R"mlir(["f16", "bf16", "f32"])mlir";
  if (kind == FixtureKind::ApproximateSingleton) {
    fastMath = "afn";
    accuracy = "Max4Ulp";
  }
  if (kind == FixtureKind::UnsupportedBehavior) {
    nan = "number_preferred";
    signedZero = "ignore_sign";
  }

  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @" << moduleName(family)
         << "(%a: !fabric.bits<32>) -> !fabric.bits<32> { "
            "%pe = fabric.pe [spatial](%pa = %a : !fabric.bits<32>) -> "
            "!fabric.bits<32> { %fu = fabric.fu"
            "(%fa = %pa : !fabric.bits<32>) -> !fabric.bits<32> { "
            "%value = fabric.op [@"
         << schemaKeyword(family) << "] (%fa) {implementation_family = "
         << "#fabric.implementation_family<" << familyKeyword(family)
         << ">, hw_params = {float_formats = " << formats
         << ", behavior = {rounding_modes = [\"to_nearest_even\"], "
            "nan_behaviors = [\""
         << nan
         << "\"], subnormal_behaviors = [\"preserve\"], "
            "signed_zero_behaviors = [\""
         << signedZero << "\"], fastmath = \"" << fastMath
         << "\"}, accuracy_guarantee = \"" << accuracy
         << "\"}} : (!fabric.bits<32>) -> !fabric.bits<32> "
            "fabric.yield %value : !fabric.bits<32> } } "
            "fabric.yield %pe : !fabric.bits<32> } }";
  return source.str();
}

void attachContract(llvm::StringRef test, mlir::ModuleOp module,
                    FixtureKind kind) {
  const ::fabric::ResourceContract &contract =
      kind == FixtureKind::UnsupportedContract
          ? ::fabric::loopCarryOperationResourceContract()
          : ::fabric::oneCycleElasticOperationResourceContract();
  const std::vector<std::uint8_t> encoded =
      take(test, ::fabric::encodeResourceContractRecord(contract));
  const std::vector<std::int8_t> signedBytes(encoded.begin(), encoded.end());
  module.walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedBytes));
  });
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         TrigFamily family,
                         FixtureKind kind = FixtureKind::Configured) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      fabricSource(family, kind), &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  attachContract(test, *source, kind);
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no root");
  FinalizedFabricRoot fabric =
      take(test, loom::fabric::finalizeFabricRoot(root, store));

  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &capability :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (capability.implementationFamily != familyId(family))
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
              "System has no physical trigonometric occurrence");
      return FabricFixture{family, std::move(fabric), occurrence,
                           std::move(system), physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no trigonometric occurrence");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
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
  fail(test, "Fabric projected an unexpected floating format");
}

std::uint8_t physicalCode(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return 3;
  case ::fabric::FloatFormat::BF16:
    return 10;
  case ::fabric::FloatFormat::F32:
    return 5;
  case ::fabric::FloatFormat::F64:
    return 12;
  }
  llvm_unreachable("unknown floating format");
}

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "trigonometric fixture has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "trigonometric relation is not finite");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "configured behavior has no semantic value");
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
    entries.push_back({{0xfe}, {0xf}});
  require(test, !inactive.empty(), "trigonometric domain has no inactive mode");
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  SemanticFieldEncoding encoding =
      kind == AbiKind::DirectEncoding
          ? SemanticFieldEncoding{DirectBitsEncoding{4}}
          : SemanticFieldEncoding{
                FiniteCodebookEncoding{4, std::move(entries)}};
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
    auto field = llvm::find_if(
        ports, [](const auto &port) { return port.getName() == "config_0"; });
    require(test, field != ports.end(),
            "configured trigonometric leaf has no selector");
    field->type = builder.getI1Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(moduleName(fabric.family)), ports);
  return SkeletonFixture{std::move(module), leaf};
}

FabricOperationProviderRegistry makeProviderRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableScalarMathTrigonometricProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

std::string moduleText(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream output(text);
  module.print(output);
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
          "portable trigonometric provider emitted external state");
  return std::move(conformance.systemVerilog);
}

const llvm::fltSemantics &semantics(::fabric::FloatFormat format) {
  return format == ::fabric::FloatFormat::F16 ? llvm::APFloat::IEEEhalf()
                                              : llvm::APFloat::BFloat();
}

class MpfrValue final {
public:
  explicit MpfrValue(mpfr_prec_t precision) { mpfr_init2(value_, precision); }
  ~MpfrValue() { mpfr_clear(value_); }
  mpfr_ptr get() { return value_; }
  mpfr_srcptr get() const { return value_; }

private:
  mpfr_t value_;
};

class MpfrFormatEnvironment final {
public:
  explicit MpfrFormatEnvironment(const llvm::fltSemantics &semantics)
      : previousMinimum_(mpfr_get_emin()), previousMaximum_(mpfr_get_emax()),
        previousFlags_(mpfr_flags_save()) {
    const mpfr_exp_t precision = llvm::APFloat::semanticsPrecision(semantics);
    const mpfr_exp_t minimum =
        llvm::APFloat::semanticsMinExponent(semantics) - precision + 2;
    const mpfr_exp_t maximum =
        llvm::APFloat::semanticsMaxExponent(semantics) + 1;
    if (mpfr_set_emin(minimum) != 0 || mpfr_set_emax(maximum) != 0)
      fail("MpfrFormatEnvironment", "could not select target exponent range");
    mpfr_clear_flags();
  }

  ~MpfrFormatEnvironment() {
    (void)mpfr_set_emin(previousMinimum_);
    (void)mpfr_set_emax(previousMaximum_);
    mpfr_flags_restore(previousFlags_, MPFR_FLAGS_ALL);
  }

private:
  mpfr_exp_t previousMinimum_;
  mpfr_exp_t previousMaximum_;
  mpfr_flags_t previousFlags_;
};

using MpfrOperation = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);

MpfrOperation mpfrOperation(TrigFamily family) {
  switch (family) {
  case TrigFamily::Sin:
    return &mpfr_sin;
  case TrigFamily::Cos:
    return &mpfr_cos;
  case TrigFamily::Tan:
    return &mpfr_tan;
  }
  llvm_unreachable("unknown trigonometric family");
}

std::uint16_t quietNaN(::fabric::FloatFormat format) {
  return format == ::fabric::FloatFormat::F16 ? 0x7e00 : 0x7fc0;
}

std::uint16_t oracle(TrigFamily family, ::fabric::FloatFormat format,
                     std::uint16_t bits) {
  const llvm::fltSemantics &target = semantics(format);
  llvm::APFloat operand(target, llvm::APInt(16, bits));
  if (operand.isNaN())
    return static_cast<std::uint16_t>(
        operand.makeQuiet().bitcastToAPInt().getZExtValue());
  if (operand.isInfinity())
    return quietNaN(format);

  MpfrValue input(
      llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEdouble()));
  MpfrValue result(llvm::APFloat::semanticsPrecision(target));
  mpfr_set_d(input.get(), operand.convertToDouble(), MPFR_RNDN);
  const int ternary =
      mpfrOperation(family)(result.get(), input.get(), MPFR_RNDN);
  (void)mpfr_subnormalize(result.get(), ternary, MPFR_RNDN);
  llvm::APFloat rounded(mpfr_get_d(result.get(), MPFR_RNDN));
  bool losesInformation = false;
  (void)rounded.convert(target, llvm::RoundingMode::NearestTiesToEven,
                        &losesInformation);
  return static_cast<std::uint16_t>(rounded.bitcastToAPInt().getZExtValue());
}

std::string expectedHex(TrigFamily family, ::fabric::FloatFormat format) {
  MpfrFormatEnvironment environment(semantics(format));
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::uint32_t bits = 0; bits != 65536; ++bits)
    output << std::setw(4)
           << oracle(family, format, static_cast<std::uint16_t>(bits)) << '\n';
  return output.str();
}

std::string makeTestbench() {
  return R"sv(module testbench;
  logic [31:0] operand;
  logic [3:0] mode_select;
  logic [31:0] sin_result;
  logic [31:0] cos_result;
  logic [31:0] tan_result;
  logic [15:0] sin_f16 [0:65535];
  logic [15:0] cos_f16 [0:65535];
  logic [15:0] tan_f16 [0:65535];
  logic [15:0] sin_bf16 [0:65535];
  logic [15:0] cos_bf16 [0:65535];
  logic [15:0] tan_bf16 [0:65535];
  integer index;

  scalar_math_sin sin_dut(.data_input_0(operand), .config_0(mode_select),
                          .data_output_0(sin_result));
  scalar_math_cos cos_dut(.data_input_0(operand), .config_0(mode_select),
                          .data_output_0(cos_result));
  scalar_math_tan tan_dut(.data_input_0(operand), .config_0(mode_select),
                          .data_output_0(tan_result));

  task automatic check_results(
      input logic [15:0] expected_sin,
      input logic [15:0] expected_cos,
      input logic [15:0] expected_tan);
    begin
      #1;
      if (sin_result !== {16'd0, expected_sin} ||
          cos_result !== {16'd0, expected_cos} ||
          tan_result !== {16'd0, expected_tan})
        $fatal(1, "trig mismatch mode=%0d input=%h sin=%h/%h cos=%h/%h tan=%h/%h",
               mode_select, operand, sin_result, expected_sin, cos_result,
               expected_cos, tan_result, expected_tan);
    end
  endtask

  initial begin
    $readmemh("sin_f16.hex", sin_f16);
    $readmemh("cos_f16.hex", cos_f16);
    $readmemh("tan_f16.hex", tan_f16);
    $readmemh("sin_bf16.hex", sin_bf16);
    $readmemh("cos_bf16.hex", cos_bf16);
    $readmemh("tan_bf16.hex", tan_bf16);
    mode_select = 4'd3;
    for (index = 0; index < 65536; index = index + 1) begin
      operand = {16'ha5a5, index[15:0]};
      check_results(sin_f16[index], cos_f16[index], tan_f16[index]);
    end
    mode_select = 4'd10;
    for (index = 0; index < 65536; index = index + 1) begin
      operand = {16'h5a5a, index[15:0]};
      check_results(sin_bf16[index], cos_bf16[index], tan_bf16[index]);
    end
    mode_select = 4'd15;
    operand = 32'hffff3c00;
    check_results(sin_f16[16'h3c00], cos_f16[16'h3c00], tan_f16[16'h3c00]);
    $finish;
  end
endmodule
)sv";
}

std::string makeSynthesisTop() {
  return R"sv(module trig_synthesis_top(
    input logic [31:0] data_input_0,
    input logic [3:0] config_0,
    output logic [95:0] data_output);
  scalar_math_sin sin_dut(.data_input_0(data_input_0), .config_0(config_0),
                          .data_output_0(data_output[31:0]));
  scalar_math_cos cos_dut(.data_input_0(data_input_0), .config_0(config_0),
                          .data_output_0(data_output[63:32]));
  scalar_math_tan tan_dut(.data_input_0(data_input_0), .config_0(config_0),
                          .data_output_0(data_output[95:64]));
endmodule
)sv";
}

std::string makeYosysScript() {
  return R"ys(read_verilog -sv scalar_math_sin.sv scalar_math_cos.sv scalar_math_tan.sv synthesis_top.sv
hierarchy -check -top trig_synthesis_top
proc
opt
check -assert
select -assert-none t:$*ff* t:$*latch* t:$_*FF* t:$_*LATCH* t:$mem*
synth -noabc -top trig_synthesis_top
check -assert
select -assert-none t:$*ff* t:$*latch* t:$_*FF* t:$_*LATCH* t:$mem*
stat
)ys";
}

bool hasPortableCoverage(
    llvm::ArrayRef<FabricOperationProviderCoverage> coverage,
    ::fabric::ImplementationFamilyId family) {
  return llvm::any_of(coverage, [&](const auto &entry) {
    return entry.implementationFamily == family &&
           entry.recipes == std::vector<BackendRecipeKey>{
                                BackendRecipeKey::PortableSystemVerilog};
  });
}

llvm::Expected<FabricOperationProviderOutput>
dummyProvider(FabricOperationProviderRequest) {
  return FabricOperationProviderOutput{};
}

void registrationIsTransactional() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registry.add({familyId(TrigFamily::Cos),
                                        BackendRecipeKey::PortableSystemVerilog,
                                        {},
                                        dummyProvider}))
    fail(test, llvm::toString(std::move(error)));
  llvm::Error error =
      registerPortableScalarMathTrigonometricProviders(registry);
  require(test, static_cast<bool>(error),
          "package registration accepted a conflicting provider");
  llvm::consumeError(std::move(error));
  const auto coverage = registry.coverage();
  require(test,
          !hasPortableCoverage(coverage, familyId(TrigFamily::Sin)) &&
              !hasPortableCoverage(coverage, familyId(TrigFamily::Tan)),
          "failed package registration partially mutated the registry");
}

std::string emitFamily(llvm::StringRef test, const ArtifactStore &store,
                       TrigFamily family) {
  FabricFixture fabric = makeFabric(test, store, family);
  const auto &resolved = capability(test, fabric);
  require(test,
          resolved.implementationFamily == familyId(family) &&
              resolved.enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{schemaId(family)},
          "provider escaped its generated family descriptor");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 2,
          "provider did not consume its sealed two-format relation");
  for (const auto &point : relation.finiteBehaviorDomain())
    require(test,
            point.representativeActor.schema == schemaId(family) &&
                point.semanticConfiguration.has_value(),
            "sealed relation contains a malformed witness");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 3 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "config_0" &&
              ports.atInput(1).type ==
                  mlir::IntegerType::get(firstContext.get(), 4) &&
              ports.atOutput(0).getName() == "data_output_0",
          "trigonometric leaf ports do not follow ABI2 geometry");
  const std::string firstRtl = specialize(test, std::move(first), fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  const std::string secondRtl = specialize(
      test, makeSkeleton(test, *secondContext, fabric, abi.abi()), fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical trigonometric inputs produced different RTL");
  const llvm::StringRef rtl(firstRtl);
  const std::string prefix = "loom_trig_" + shortName(family).str();
  require(test,
          rtl.contains("config_0") && rtl.contains(prefix + "_e5_f10") &&
              rtl.contains(prefix + "_e8_f7") && rtl.contains("256'h") &&
              !rtl.contains("shortreal") && !rtl.contains(" real") &&
              !rtl.contains("DPI") && !rtl.contains("$sin") &&
              !rtl.contains("$cos") && !rtl.contains("$tan"),
          "trigonometric RTL is incomplete or unsynthesizable");
  return firstRtl;
}

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  for (TrigFamily family : kFamilies)
    require(test, hasPortableCoverage(coverage, familyId(family)),
            "trigonometric provider registration is incomplete");

  std::vector<loom::hardware::test::PortableProviderArtifact> artifacts;
  for (TrigFamily family : kFamilies)
    artifacts.push_back(
        {moduleName(family) + ".sv", emitFamily(test, store, family)});
  for (TrigFamily family : kFamilies) {
    artifacts.push_back({shortName(family).str() + "_f16.hex",
                         expectedHex(family, ::fabric::FloatFormat::F16)});
    artifacts.push_back({shortName(family).str() + "_bf16.hex",
                         expectedHex(family, ::fabric::FloatFormat::BF16)});
  }
  artifacts.push_back({"testbench.sv", makeTestbench()});
  artifacts.push_back({"synthesis_top.sv", makeSynthesisTop()});
  artifacts.push_back(
      {"portable_scalar_math_trigonometric.ys", makeYosysScript()});
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated", artifacts))
    fail(test, llvm::toString(std::move(error)));
}

void singletonAndAccuracyGuarantees(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  for (TrigFamily family : kFamilies) {
    for (FixtureKind kind :
         {FixtureKind::Singleton, FixtureKind::ApproximateSingleton}) {
      FabricFixture fabric = makeFabric(test, store, family, kind);
      const auto &resolved = capability(test, fabric);
      auto relation =
          take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
      require(
          test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::None &&
              relation.finiteBehaviorDomain().size() == 1 &&
              !relation.finiteBehaviorDomain().front().semanticConfiguration,
          "singleton trigonometric relation retained a selector");
      FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
      std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
      SkeletonFixture skeleton =
          makeSkeleton(test, *context, fabric, abi.abi());
      require(test, skeleton.leaf.getPortList().size() == 2,
              "singleton trigonometric leaf retained a selector port");
      const std::string rtl =
          specialize(test, std::move(skeleton), fabric, abi);
      require(test,
              llvm::StringRef(rtl).contains("_e5_f10") &&
                  !llvm::StringRef(rtl).contains("config_0"),
              "singleton trigonometric witness was not emitted");
    }
  }
}

void expectTypedUnsupported(
    llvm::StringRef test, llvm::Expected<FabricOperationProviderOutput> result,
    ::fabric::ImplementationFamilyId family, BackendRecipeKey recipe) {
  require(test, !result, "unsupported trigonometric capability specialized");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified =
            error.implementationFamily() == family && error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unsupported provider returned the wrong error: " +
                       error.message());
      });
  require(test, classified,
          "trigonometric provider lost typed Unsupported classification");
}

void malformedInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  for (TrigFamily family : kFamilies) {
    FabricFixture valid = makeFabric(test, store, family);
    FinalizedConfigurationABI validAbi =
        makeConfigurationAbi(test, store, valid);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture malformed =
        makeSkeleton(test, *context, valid, validAbi.abi(), true);
    const std::string before = moduleText(*malformed.module);
    expectError(test, trySpecialize(malformed, valid, validAbi, registry),
                "leaf port");
    require(test, moduleText(*malformed.module) == before,
            "malformed leaf partially mutated the skeleton");

    for (AbiKind kind : {AbiKind::MissingBehavior, AbiKind::ExtraBehavior,
                         AbiKind::DirectEncoding})
      expectError(
          test,
          finalizeConfigurationABI(makeConfigurationAbiDraft(test, valid, kind),
                                   store),
          kind == AbiKind::DirectEncoding ? "finite codebook" : "semantic");

    for (FixtureKind unsupportedKind :
         {FixtureKind::UnsupportedFormat, FixtureKind::UnsupportedBehavior,
          FixtureKind::UnsupportedContract}) {
      FabricFixture unsupported =
          makeFabric(test, store, family, unsupportedKind);
      FinalizedConfigurationABI unsupportedAbi =
          makeConfigurationAbi(test, store, unsupported);
      std::unique_ptr<mlir::MLIRContext> unsupportedContext =
          makeCirctContext();
      SkeletonFixture unsupportedSkeleton = makeSkeleton(
          test, *unsupportedContext, unsupported, unsupportedAbi.abi());
      const std::string unsupportedBefore =
          moduleText(*unsupportedSkeleton.module);
      expectTypedUnsupported(test,
                             trySpecialize(unsupportedSkeleton, unsupported,
                                           unsupportedAbi, registry),
                             familyId(family),
                             BackendRecipeKey::PortableSystemVerilog);
      require(test,
              moduleText(*unsupportedSkeleton.module) == unsupportedBefore,
              "Unsupported capability mutated the caller skeleton");
    }

    for (BackendRecipeKey recipe :
         {BackendRecipeKey::SynopsysDesignWare,
          BackendRecipeKey::CadenceChipWare, BackendRecipeKey::AmdXilinx,
          BackendRecipeKey::IntelAltera}) {
      std::unique_ptr<mlir::MLIRContext> nativeContext = makeCirctContext();
      SkeletonFixture native =
          makeSkeleton(test, *nativeContext, valid, validAbi.abi());
      const std::string nativeBefore = moduleText(*native.module);
      expectTypedUnsupported(
          test, trySpecialize(native, valid, validAbi, registry, recipe),
          familyId(family), recipe);
      require(test, moduleText(*native.module) == nativeBefore,
              "unsupported native recipe mutated the caller skeleton");
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  registrationIsTransactional();
  configuredBehaviorAndArtifacts(root);
  singletonAndAccuracyGuarantees(root / "singleton");
  malformedInputsAreTransactional(root / "malformed");
  return EXIT_SUCCESS;
}

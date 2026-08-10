#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/MathExponential.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/SpecialMathAccuracy.h"
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
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::SpecialMathAccuracyTier;
using loom::fabric::FabricFuOccurrenceNodeRef;
using loom::fabric::FinalizedFabricRoot;
using namespace loom::hardware;
using namespace loom::hardware::rtl;

enum class MathFamily { Exp, Exp2, ExpM1 };
enum class FixtureKind {
  Configured,
  SingletonF16,
  F32,
  F64,
  CorrectlyRounded,
  Max1Ulp,
  Max2Ulp,
  UnsupportedContract,
};

struct FabricFixture final {
  MathFamily family;
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
  int bias;
  int minimumExponent;
  int maximumExponent;

  unsigned width() const { return 1 + exponentBits + fractionBits; }
  std::uint16_t sign() const {
    return static_cast<std::uint16_t>(1U << (width() - 1));
  }
  std::uint16_t exponentMask() const {
    return static_cast<std::uint16_t>((1U << exponentBits) - 1);
  }
  std::uint16_t infinity() const {
    return static_cast<std::uint16_t>(exponentMask() << fractionBits);
  }
  std::uint16_t one() const {
    return static_cast<std::uint16_t>(bias << fractionBits);
  }
};

struct NumericVector final {
  MathFamily family;
  ::fabric::FloatFormat format;
  std::uint8_t physicalCode;
  std::uint16_t input;
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

::fabric::ImplementationFamilyId familyId(MathFamily family) {
  switch (family) {
  case MathFamily::Exp:
    return ::fabric::ImplementationFamilyId::ScalarMathExp;
  case MathFamily::Exp2:
    return ::fabric::ImplementationFamilyId::ScalarMathExp2;
  case MathFamily::ExpM1:
    return ::fabric::ImplementationFamilyId::ScalarMathExpM1;
  }
  llvm_unreachable("unknown exponential family");
}

::dataflow::OperationSchemaId schemaId(MathFamily family) {
  switch (family) {
  case MathFamily::Exp:
    return ::dataflow::OperationSchemaId::MathExp;
  case MathFamily::Exp2:
    return ::dataflow::OperationSchemaId::MathExp2;
  case MathFamily::ExpM1:
    return ::dataflow::OperationSchemaId::MathExpM1;
  }
  llvm_unreachable("unknown exponential family");
}

llvm::StringRef familyKeyword(MathFamily family) {
  switch (family) {
  case MathFamily::Exp:
    return "ScalarMathExp";
  case MathFamily::Exp2:
    return "ScalarMathExp2";
  case MathFamily::ExpM1:
    return "ScalarMathExpM1";
  }
  llvm_unreachable("unknown exponential family");
}

llvm::StringRef schemaKeyword(MathFamily family) {
  switch (family) {
  case MathFamily::Exp:
    return "math.exp";
  case MathFamily::Exp2:
    return "math.exp2";
  case MathFamily::ExpM1:
    return "math.expm1";
  }
  llvm_unreachable("unknown exponential family");
}

llvm::StringRef moduleName(MathFamily family) {
  switch (family) {
  case MathFamily::Exp:
    return "scalar_math_exp";
  case MathFamily::Exp2:
    return "scalar_math_exp2";
  case MathFamily::ExpM1:
    return "scalar_math_expm1";
  }
  llvm_unreachable("unknown exponential family");
}

unsigned familyCode(MathFamily family) { return static_cast<unsigned>(family); }

FloatLayout layout(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return {5, 10, 15, -14, 15};
  case ::fabric::FloatFormat::BF16:
    return {8, 7, 127, -126, 127};
  case ::fabric::FloatFormat::F32:
    return {8, 23, 127, -126, 127};
  case ::fabric::FloatFormat::F64:
    return {11, 52, 1023, -1022, 1023};
  }
  llvm_unreachable("unknown floating format");
}

const llvm::fltSemantics &semantics(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return llvm::APFloat::IEEEhalf();
  case ::fabric::FloatFormat::BF16:
    return llvm::APFloat::BFloat();
  case ::fabric::FloatFormat::F32:
    return llvm::APFloat::IEEEsingle();
  case ::fabric::FloatFormat::F64:
    return llvm::APFloat::IEEEdouble();
  }
  llvm_unreachable("unknown floating format");
}

std::uint8_t physicalCode(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return 2;
  case ::fabric::FloatFormat::BF16:
    return 5;
  default:
    llvm_unreachable("format has no portable exponential code");
  }
}

constexpr std::array<std::uint8_t, 6> inactivePhysicalCodes = {0, 1, 3,
                                                               4, 6, 7};

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

std::string fabricSource(MathFamily family, FixtureKind kind) {
  llvm::StringRef formats = R"mlir(["f16", "bf16"])mlir";
  llvm::StringRef accuracy = "Max4Ulp";
  llvm::StringRef fastMath = "afn";
  unsigned physicalWidth = 32;
  switch (kind) {
  case FixtureKind::Configured:
  case FixtureKind::UnsupportedContract:
    break;
  case FixtureKind::SingletonF16:
    formats = R"mlir(["f16"])mlir";
    break;
  case FixtureKind::F32:
    formats = R"mlir(["f32"])mlir";
    break;
  case FixtureKind::F64:
    formats = R"mlir(["f64"])mlir";
    physicalWidth = 64;
    break;
  case FixtureKind::CorrectlyRounded:
    formats = R"mlir(["f16"])mlir";
    accuracy = "CorrectlyRounded";
    fastMath = "none";
    break;
  case FixtureKind::Max1Ulp:
    formats = R"mlir(["f16"])mlir";
    accuracy = "Max1Ulp";
    break;
  case FixtureKind::Max2Ulp:
    formats = R"mlir(["f16"])mlir";
    accuracy = "Max2Ulp";
    break;
  }

  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @" << moduleName(family)
         << "(%a: !fabric.bits<" << physicalWidth << ">) -> !fabric.bits<"
         << physicalWidth << "> { %pe = fabric.pe [spatial]"
         << "(%pa = %a : !fabric.bits<" << physicalWidth
         << ">) -> !fabric.bits<" << physicalWidth
         << "> { %fu = fabric.fu(%fa = %pa : !fabric.bits<" << physicalWidth
         << ">) -> !fabric.bits<" << physicalWidth
         << "> { %value = fabric.op [@" << schemaKeyword(family)
         << "] (%fa) {implementation_family = "
         << "#fabric.implementation_family<" << familyKeyword(family)
         << ">, hw_params = {float_formats = " << formats
         << ", behavior = {rounding_modes = [\"to_nearest_even\"], "
            "nan_behaviors = [\"ieee\"], subnormal_behaviors = "
            "[\"preserve\"], signed_zero_behaviors = [\"preserve\"], "
            "fastmath = \""
         << fastMath << "\"}, accuracy_guarantee = \"" << accuracy
         << "\"}} : (!fabric.bits<" << physicalWidth << ">) -> !fabric.bits<"
         << physicalWidth << "> fabric.yield %value : !fabric.bits<"
         << physicalWidth << "> } } fabric.yield %pe : !fabric.bits<"
         << physicalWidth << "> } }";
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
                         MathFamily family,
                         FixtureKind kind = FixtureKind::Configured) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      fabricSource(family, kind), &fabricContext());
  require(test, static_cast<bool>(source),
          "could not parse exponential Fabric fixture");
  attachContract(test, *source, kind);

  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "exponential fixture has no root");
  FinalizedFabricRoot fabric =
      take(test, loom::fabric::finalizeFabricRoot(root, store));
  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &candidate :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (candidate.implementationFamily != familyId(family))
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), candidate.occurrence, fuOccurrence));
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
              "System has no exponential physical occurrence");
      return {family, std::move(fabric), occurrence, std::move(system),
              physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no exponential operation occurrence");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "exponential capability did not resolve");
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
  if (mlir::isa<mlir::Float64Type>(type))
    return ::fabric::FloatFormat::F64;
  fail(test, "Fabric projected an unknown exponential format");
}

void requireStoredSemanticProjection(
    llvm::StringRef test,
    const ::fabric::FabricOpSemanticFieldRelation &relation,
    const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  require(test, point.semanticConfiguration.has_value(),
          "configured exponential behavior has no semantic value");
  const loom::CanonicalSemanticBytes projected =
      take(test, relation.projectSemanticValue(
                     point.representativeActor, point.operandPorts,
                     point.resultPorts, point.resolvedIndexWidth));
  require(test, point.semanticConfiguration->bytes().equals(projected.bytes()),
          "exponential behavior disagrees with its sealed semantic relation");
}

ConfigurationABIDraft makeConfigurationAbiDraft(llvm::StringRef test,
                                                const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "exponential capability has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 2,
          "configured exponential relation is not the sealed two-format set");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "configured exponential behavior has no semantic value");
    const auto format = behaviorFormat(test, point);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (format == ::fabric::FloatFormat::F16)
      inactive = semantic;
    entries.push_back({std::move(semantic), {physicalCode(format)}});
  }
  require(test, !inactive.empty(),
          "exponential codebook has no inactive semantic value");
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, FiniteCodebookEncoding{3, std::move(entries)},
      std::move(inactive)};
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, {std::move(field)}));
}

FinalizedConfigurationABI makeConfigurationAbi(llvm::StringRef test,
                                               const ArtifactStore &store,
                                               const FabricFixture &fixture) {
  return take(test, finalizeConfigurationABI(
                        makeConfigurationAbiDraft(test, fixture), store));
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
            "configured exponential leaf has no selector");
    field->type = builder.getI1Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(moduleName(fabric.family)), ports);
  return {std::move(module), leaf};
}

FabricOperationProviderRegistry makeProviderRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableMathExponentialProviders(registry))
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
          "portable exponential provider emitted external state");
  return std::move(conformance.systemVerilog);
}

llvm::APFloat floating(::fabric::FloatFormat format, std::uint16_t bits) {
  return llvm::APFloat(semantics(format),
                       llvm::APInt(layout(format).width(), bits));
}

std::vector<NumericVector> numericVectors() {
  std::vector<NumericVector> result;
  result.reserve(3 * 2 * 65536 + 3 * inactivePhysicalCodes.size());
  for (MathFamily family :
       {MathFamily::Exp, MathFamily::Exp2, MathFamily::ExpM1}) {
    for (::fabric::FloatFormat format :
         {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16}) {
      for (std::uint32_t input = 0; input != UINT32_C(65536); ++input)
        result.push_back({family, format, physicalCode(format),
                          static_cast<std::uint16_t>(input)});
    }
    for (std::uint8_t code : inactivePhysicalCodes)
      result.push_back({family, ::fabric::FloatFormat::F16, code,
                        layout(::fabric::FloatFormat::F16).one()});
  }
  return result;
}

std::string makeVectorFile(llvm::ArrayRef<NumericVector> vectors) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const NumericVector &vector : vectors) {
    const std::uint64_t packed =
        (std::uint64_t(familyCode(vector.family)) << 35) |
        (std::uint64_t(vector.physicalCode) << 32) |
        (0xa5a50000ULL | vector.input);
    output << std::setw(10) << packed << '\n';
  }
  return output.str();
}

std::string makeTestbench(std::size_t vectorCount) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "module testbench;\n"
         << "  localparam integer VECTOR_COUNT = " << vectorCount << ";\n"
         << R"sv(  logic [36:0] vectors [0:VECTOR_COUNT-1];
  logic [1:0] family_select;
  logic [2:0] config_select;
  logic [31:0] operand;
  logic [31:0] exp_result;
  logic [31:0] exp2_result;
  logic [31:0] expm1_result;
  logic [31:0] selected_result;
  integer output_file;
  integer index;

  scalar_math_exp exp_dut(
      .data_input_0(operand), .config_0(config_select),
      .data_output_0(exp_result));
  scalar_math_exp2 exp2_dut(
      .data_input_0(operand), .config_0(config_select),
      .data_output_0(exp2_result));
  scalar_math_expm1 expm1_dut(
      .data_input_0(operand), .config_0(config_select),
      .data_output_0(expm1_result));

  always_comb begin
    case (family_select)
      2'd0: selected_result = exp_result;
      2'd1: selected_result = exp2_result;
      2'd2: selected_result = expm1_result;
      default: selected_result = 32'd0;
    endcase
  end

  initial begin
    $readmemh("vectors.hex", vectors);
    output_file = $fopen("results.hex", "w");
    if (output_file == 0)
      $fatal(1, "could not open results.hex");
    family_select = 0;
    config_select = 0;
    operand = 0;
    for (index = 0; index < VECTOR_COUNT; index = index + 1) begin
      {family_select, config_select, operand} = vectors[index];
      #1;
      $fdisplay(output_file, "%08x", selected_result);
    end
    $fclose(output_file);
    $finish;
  end
endmodule
)sv";
  return output.str();
}

std::string makeSynthesisTop() {
  return R"sv(module math_exponential_synthesis_top(
    input logic [31:0] operand,
    input logic [2:0] config_select,
    output logic [31:0] exp_result,
    output logic [31:0] exp2_result,
    output logic [31:0] expm1_result);
  scalar_math_exp exp_instance(
      .data_input_0(operand), .config_0(config_select),
      .data_output_0(exp_result));
  scalar_math_exp2 exp2_instance(
      .data_input_0(operand), .config_0(config_select),
      .data_output_0(exp2_result));
  scalar_math_expm1 expm1_instance(
      .data_input_0(operand), .config_0(config_select),
      .data_output_0(expm1_result));
endmodule
)sv";
}

std::string makeYosysScript() {
  return R"ys(read_verilog -sv scalar_math_exp.sv
read_verilog -sv scalar_math_exp2.sv
read_verilog -sv scalar_math_expm1.sv
read_verilog -sv synthesis_top.sv
hierarchy -check -top math_exponential_synthesis_top
proc
opt
check -assert
select -assert-none math_exponential_synthesis_top/t:$*ff* math_exponential_synthesis_top/t:$*latch* math_exponential_synthesis_top/t:$_*FF* math_exponential_synthesis_top/t:$_*LATCH* math_exponential_synthesis_top/t:$mem* math_exponential_synthesis_top/m:*
synth -noabc -top math_exponential_synthesis_top
check -assert
select -assert-none math_exponential_synthesis_top/t:$*ff* math_exponential_synthesis_top/t:$*latch* math_exponential_synthesis_top/t:$_*FF* math_exponential_synthesis_top/t:$_*LATCH* math_exponential_synthesis_top/t:$mem* math_exponential_synthesis_top/m:*
stat
)ys";
}

std::string emitFamily(llvm::StringRef test, const ArtifactStore &store,
                       MathFamily family) {
  FabricFixture fabric = makeFabric(test, store, family);
  const auto &resolved = capability(test, fabric);
  require(test,
          resolved.implementationFamily == familyId(family) &&
              resolved.enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{schemaId(family)},
          "exponential provider escaped its generated descriptor");
  const auto *parameters = std::get_if<::fabric::ScalarSpecialMathParams>(
      &resolved.parameterizedCapability);
  require(test,
          parameters != nullptr &&
              parameters->accuracyGuarantee == SpecialMathAccuracyTier::Max4Ulp,
          "exponential capability lost ScalarSpecialMathParams");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 2,
          "exponential provider did not consume the sealed format relation");
  for (const auto &point : relation.finiteBehaviorDomain()) {
    requireStoredSemanticProjection(test, relation, point);
    const auto *payload = std::get_if<::dataflow::SpecialMathPayload>(
        &point.representativeActor.payload);
    require(test,
            point.representativeActor.schema == schemaId(family) &&
                point.semanticConfiguration.has_value() && payload != nullptr &&
                payload->accuracy == SpecialMathAccuracyTier::Max4Ulp,
            "exponential relation contains a malformed witness");
  }

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture firstSkeleton =
      makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(firstSkeleton.leaf.getPortList());
  require(test,
          ports.size() == 3 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "config_0" &&
              ports.atInput(1).type ==
                  mlir::IntegerType::get(firstContext.get(), 3) &&
              ports.atOutput(0).getName() == "data_output_0",
          "exponential leaf ports do not follow ConfigurationABI 3.0");
  const std::string first =
      specialize(test, std::move(firstSkeleton), fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture secondSkeleton =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string second =
      specialize(test, std::move(secondSkeleton), fabric, abi);
  require(test, first == second,
          "identical exponential inputs produced different SystemVerilog");
  const llvm::StringRef rtl(first);
  require(test,
          rtl.contains("function automatic") && rtl.contains("config_0") &&
              rtl.contains("range_reduced") && rtl.contains("signed [39:0]") &&
              rtl.contains("signed [79:0]") && rtl.contains("24'h800000") &&
              rtl.contains("rounded_magnitude[0]") && !rtl.contains("[63:0]") &&
              !rtl.contains("[127:0]") && !rtl.contains("shortreal") &&
              !rtl.contains(" real") && !rtl.contains("DPI") &&
              !rtl.contains("$exp") && !rtl.contains("$pow") &&
              !rtl.contains("1'bx") && !rtl.contains("1'bz"),
          "exponential RTL is incomplete or not synthesizable bit logic");
  if (family == MathFamily::ExpM1)
    require(test, rtl.contains("near_zero") && rtl.contains("pack_fixed"),
            "expm1 lacks its cancellation-safe near-zero path");
  return first;
}

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  const std::string exp = emitFamily(test, store, MathFamily::Exp);
  const std::string exp2 = emitFamily(test, store, MathFamily::Exp2);
  const std::string expm1 = emitFamily(test, store, MathFamily::ExpM1);
  const std::vector<NumericVector> vectors = numericVectors();
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"scalar_math_exp.sv", exp},
           {"scalar_math_exp2.sv", exp2},
           {"scalar_math_expm1.sv", expm1},
           {"testbench.sv", makeTestbench(vectors.size())},
           {"vectors.hex", makeVectorFile(vectors)},
           {"synthesis_top.sv", makeSynthesisTop()},
           {"portable_math_exponential.ys", makeYosysScript()}}))
    fail(test, llvm::toString(std::move(error)));
  llvm::outs() << "generated " << vectors.size()
               << " exponential numerical vectors\n";
}

void expectTypedUnsupported(
    llvm::StringRef test, llvm::Expected<FabricOperationProviderOutput> result,
    ::fabric::ImplementationFamilyId family, BackendRecipeKey recipe) {
  require(test, !result, "unsupported exponential combination was accepted");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified =
            error.implementationFamily() == family && error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test,
             "unsupported exponential provider returned the wrong error: " +
                 error.message());
      });
  require(test, classified,
          "exponential provider lost its typed Unsupported classification");
}

llvm::Expected<FabricOperationProviderOutput>
dummyProvider(FabricOperationProviderRequest) {
  return FabricOperationProviderOutput{};
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

void registrationIsTransactional() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  require(test,
          llvm::count_if(
              coverage,
              [](const auto &entry) { return !entry.recipes.empty(); }) == 3,
          "exponential package registered a foreign family");
  for (MathFamily family :
       {MathFamily::Exp, MathFamily::Exp2, MathFamily::ExpM1})
    require(test, hasPortableCoverage(coverage, familyId(family)),
            "exponential package registration is incomplete");

  FabricOperationProviderRegistry conflicting;
  if (llvm::Error error =
          conflicting.add({::fabric::ImplementationFamilyId::ScalarMathExp2,
                           BackendRecipeKey::PortableSystemVerilog,
                           {},
                           dummyProvider}))
    fail(test, llvm::toString(std::move(error)));
  llvm::Error error = registerPortableMathExponentialProviders(conflicting);
  require(test, static_cast<bool>(error),
          "registration fixture did not report its conflict");
  llvm::consumeError(std::move(error));
  require(test,
          !hasPortableCoverage(conflicting.coverage(),
                               ::fabric::ImplementationFamilyId::ScalarMathExp),
          "failed package registration partially added exp");
}

void singletonAndUnsupportedContracts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  for (MathFamily family :
       {MathFamily::Exp, MathFamily::Exp2, MathFamily::ExpM1}) {
    FabricFixture singleton =
        makeFabric(test, store, family, FixtureKind::SingletonF16);
    const auto &resolved = capability(test, singleton);
    auto relation =
        take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
    require(test,
            relation.kind() ==
                    ::fabric::FabricOpSemanticFieldRelationKind::None &&
                relation.finiteBehaviorDomain().size() == 1 &&
                !relation.finiteBehaviorDomain().front().semanticConfiguration,
            "singleton exponential relation retained a selector authority");
    FinalizedConfigurationABI singletonAbi =
        makeConfigurationAbi(test, store, singleton);
    std::unique_ptr<mlir::MLIRContext> singletonContext = makeCirctContext();
    SkeletonFixture singletonSkeleton =
        makeSkeleton(test, *singletonContext, singleton, singletonAbi.abi());
    require(test, singletonSkeleton.leaf.getPortList().size() == 2,
            "singleton exponential leaf retained a selector port");
    const std::string rtl =
        specialize(test, std::move(singletonSkeleton), singleton, singletonAbi);
    require(test, !llvm::StringRef(rtl).contains("config_0"),
            "singleton exponential RTL retained a selector");

    for (FixtureKind kind :
         {FixtureKind::F32, FixtureKind::F64, FixtureKind::CorrectlyRounded,
          FixtureKind::Max1Ulp, FixtureKind::Max2Ulp,
          FixtureKind::UnsupportedContract}) {
      FabricFixture unsupported = makeFabric(test, store, family, kind);
      FinalizedConfigurationABI abi =
          makeConfigurationAbi(test, store, unsupported);
      std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
      SkeletonFixture skeleton =
          makeSkeleton(test, *context, unsupported, abi.abi());
      const std::string before = moduleText(*skeleton.module);
      expectTypedUnsupported(
          test, trySpecialize(skeleton, unsupported, abi, registry),
          familyId(family), BackendRecipeKey::PortableSystemVerilog);
      require(test, moduleText(*skeleton.module) == before,
              "Unsupported exponential request mutated the skeleton");
    }

    FabricFixture configured = makeFabric(test, store, family);
    FinalizedConfigurationABI configuredAbi =
        makeConfigurationAbi(test, store, configured);
    std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
    SkeletonFixture malformed = makeSkeleton(
        test, *malformedContext, configured, configuredAbi.abi(), true);
    const std::string before = moduleText(*malformed.module);
    auto malformedResult =
        trySpecialize(malformed, configured, configuredAbi, registry);
    require(test, !malformedResult, "malformed exponential leaf was accepted");
    llvm::consumeError(malformedResult.takeError());
    require(test, moduleText(*malformed.module) == before,
            "malformed exponential request mutated the skeleton");

    for (BackendRecipeKey recipe :
         {BackendRecipeKey::SynopsysDesignWare,
          BackendRecipeKey::CadenceChipWare, BackendRecipeKey::AmdXilinx,
          BackendRecipeKey::IntelAltera}) {
      std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
      SkeletonFixture native =
          makeSkeleton(test, *context, configured, configuredAbi.abi());
      const std::string nativeBefore = moduleText(*native.module);
      expectTypedUnsupported(
          test,
          trySpecialize(native, configured, configuredAbi, registry, recipe),
          familyId(family), recipe);
      require(test, moduleText(*native.module) == nativeBefore,
              "unsupported native recipe mutated the skeleton");
    }
  }
}

class MpfrOracle final {
public:
  MpfrOracle() {
    mpfr_init2(input_, 256);
    mpfr_init2(highPrecisionResult_, 256);
    mpfr_init2(targetResult_, 11);
    previousMinimumExponent_ = mpfr_get_emin();
    previousMaximumExponent_ = mpfr_get_emax();
    previousFlags_ = mpfr_flags_save();
  }

  ~MpfrOracle() {
    (void)mpfr_set_emin(previousMinimumExponent_);
    (void)mpfr_set_emax(previousMaximumExponent_);
    mpfr_flags_restore(previousFlags_, MPFR_FLAGS_ALL);
    mpfr_clear(input_);
    mpfr_clear(highPrecisionResult_);
    mpfr_clear(targetResult_);
  }

  MpfrOracle(const MpfrOracle &) = delete;
  MpfrOracle &operator=(const MpfrOracle &) = delete;

  llvm::APFloat evaluate(llvm::StringRef test, const NumericVector &vector) {
    llvm::APFloat operand = floating(vector.format, vector.input);
    if (operand.isNaN())
      return operand.makeQuiet();

    const llvm::fltSemantics &target = semantics(vector.format);
    require(test,
            mpfr_set_emin(previousMinimumExponent_) == 0 &&
                mpfr_set_emax(previousMaximumExponent_) == 0,
            "could not select the broad MPFR exponent range");
    mpfr_set_prec(input_, 256);
    mpfr_set_prec(highPrecisionResult_, 256);
    mpfr_clear_flags();
    mpfr_set_d(input_, operand.convertToDouble(), MPFR_RNDN);
    switch (vector.family) {
    case MathFamily::Exp:
      (void)mpfr_exp(highPrecisionResult_, input_, MPFR_RNDN);
      break;
    case MathFamily::Exp2:
      (void)mpfr_exp2(highPrecisionResult_, input_, MPFR_RNDN);
      break;
    case MathFamily::ExpM1:
      (void)mpfr_expm1(highPrecisionResult_, input_, MPFR_RNDN);
      break;
    }

    const mpfr_prec_t precision = llvm::APFloat::semanticsPrecision(target);
    mpfr_set_prec(targetResult_, precision);
    int ternary = mpfr_set(targetResult_, highPrecisionResult_, MPFR_RNDN);
    const mpfr_exp_t minimumExponent =
        llvm::APFloat::semanticsMinExponent(target) - precision + 2;
    const mpfr_exp_t maximumExponent =
        llvm::APFloat::semanticsMaxExponent(target) + 1;
    require(test,
            mpfr_set_emin(minimumExponent) == 0 &&
                mpfr_set_emax(maximumExponent) == 0,
            "could not select MPFR target exponent range");
    ternary = mpfr_check_range(targetResult_, ternary, MPFR_RNDN);
    (void)mpfr_subnormalize(targetResult_, ternary, MPFR_RNDN);

    llvm::APFloat rounded(mpfr_get_d(targetResult_, MPFR_RNDN));
    bool losesInformation = false;
    (void)rounded.convert(target, llvm::RoundingMode::NearestTiesToEven,
                          &losesInformation);
    return rounded;
  }

private:
  mpfr_t input_;
  mpfr_t highPrecisionResult_;
  mpfr_t targetResult_;
  mpfr_exp_t previousMinimumExponent_ = 0;
  mpfr_exp_t previousMaximumExponent_ = 0;
  mpfr_flags_t previousFlags_ = 0;
};

void verifyResults(const std::filesystem::path &path) {
  const llvm::StringRef test = __func__;
  std::ifstream input(path);
  require(test, static_cast<bool>(input), "could not open Verilator results");
  const std::vector<NumericVector> vectors = numericVectors();
  MpfrOracle oracle;
  std::uint64_t maximumUlp = 0;
  std::size_t exactCount = 0;
  std::size_t finiteCount = 0;

  for (const auto &[index, vector] : llvm::enumerate(vectors)) {
    std::string token;
    require(test, static_cast<bool>(input >> token),
            "Verilator result file is truncated");
    std::uint64_t fullResult = 0;
    try {
      fullResult = std::stoull(token, nullptr, 16);
    } catch (const std::exception &) {
      fail(test, "Verilator result is not hexadecimal");
    }
    require(test, (fullResult & 0xffff0000ULL) == 0,
            "exponential provider leaked high padding bits");
    const std::uint16_t candidateBits = static_cast<std::uint16_t>(fullResult);
    llvm::APFloat operand = floating(vector.format, vector.input);
    llvm::APFloat reference = oracle.evaluate(test, vector);
    const std::uint16_t referenceBits =
        static_cast<std::uint16_t>(reference.bitcastToAPInt().getZExtValue());
    const bool exact = operand.isNaN() || operand.isInfinity() ||
                       operand.isZero() || reference.isInfinity() ||
                       reference.isZero();
    if (exact) {
      require(test, candidateBits == referenceBits,
              ("exceptional exponential mismatch at vector " +
               std::to_string(index))
                  .c_str());
      ++exactCount;
      continue;
    }

    llvm::APFloat candidate = floating(vector.format, candidateBits);
    require(test, candidate.isFinite() && !candidate.isNaN(),
            "finite exponential reference produced a non-finite candidate");
    require(
        test,
        take(test, loom::specialMathAccuracyConforms(
                       SpecialMathAccuracyTier::Max4Ulp, reference, candidate)),
        ("exponential result exceeded Max4Ulp at vector " +
         std::to_string(index))
            .c_str());
    maximumUlp = std::max(maximumUlp, take(test, loom::specialMathUlpDistance(
                                                     reference, candidate)));
    ++finiteCount;
  }
  std::string trailing;
  require(test, !(input >> trailing),
          "Verilator result file has trailing records");
  llvm::outs() << "verified " << exactCount << " exact exceptional and "
               << finiteCount << " finite exponential results; maximum ULP "
               << maximumUlp << '\n';
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 3 && llvm::StringRef(argv[1]) == "--verify") {
    verifyResults(argv[2]);
    return EXIT_SUCCESS;
  }
  if (argc != 2)
    fail("main", "expected an output directory or --verify result-file");
  const std::filesystem::path root(argv[1]);
  configuredBehaviorAndArtifacts(root);
  registrationIsTransactional();
  singletonAndUnsupportedContracts(root / "contracts");
  return EXIT_SUCCESS;
}

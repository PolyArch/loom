#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/MathRoot.h"
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
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
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

enum class RootFamily { Sqrt, Rsqrt };
enum class FixtureKind { Configured, Singleton, UnsupportedContract };
enum class AbiKind { Complete, MissingBehavior, ExtraBehavior, DirectEncoding };

struct FabricFixture final {
  RootFamily family;
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

  unsigned width() const { return 1 + exponentBits + fractionBits; }
  std::uint64_t sign() const { return std::uint64_t{1} << (width() - 1); }
  std::uint64_t infinity() const {
    return ((std::uint64_t{1} << exponentBits) - 1) << fractionBits;
  }
  std::uint64_t one() const {
    return std::uint64_t((std::uint64_t{1} << (exponentBits - 1)) - 1)
           << fractionBits;
  }
};

struct RootVector final {
  ::fabric::FloatFormat format;
  std::uint64_t input;
  std::uint64_t expected;
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
void expectInvalid(llvm::StringRef test, llvm::Expected<T> value,
                   llvm::StringRef expected) {
  require(test, !value, "accepted malformed math root input");
  bool classified = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &) {
        fail(test, "malformed math root input became typed Unsupported");
      },
      [&](const llvm::ErrorInfoBase &error) {
        require(test, llvm::StringRef(error.message()).contains(expected),
                error.message());
        classified = true;
      });
  require(test, classified, "malformed math root input lost its error");
}

void expectTypedUnsupported(llvm::StringRef test,
                            llvm::Expected<FabricOperationProviderOutput> value,
                            ::fabric::ImplementationFamilyId family,
                            BackendRecipeKey recipe) {
  require(test, !value, "unsupported math root input was accepted");
  bool classified = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified =
            error.implementationFamily() == family && error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unsupported math root returned the wrong error class: " +
                       error.message());
      });
  require(test, classified, "math root lost typed Unsupported classification");
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

::fabric::ImplementationFamilyId familyId(RootFamily family) {
  return family == RootFamily::Sqrt
             ? ::fabric::ImplementationFamilyId::ScalarMathSqrt
             : ::fabric::ImplementationFamilyId::ScalarMathRsqrt;
}

::dataflow::OperationSchemaId schemaId(RootFamily family) {
  return family == RootFamily::Sqrt ? ::dataflow::OperationSchemaId::MathSqrt
                                    : ::dataflow::OperationSchemaId::MathRsqrt;
}

llvm::StringRef familyKeyword(RootFamily family) {
  return family == RootFamily::Sqrt ? "ScalarMathSqrt" : "ScalarMathRsqrt";
}

llvm::StringRef schemaSpelling(RootFamily family) {
  return family == RootFamily::Sqrt ? "math.sqrt" : "math.rsqrt";
}

llvm::StringRef moduleName(RootFamily family) {
  return family == RootFamily::Sqrt ? "scalar_math_sqrt" : "scalar_math_rsqrt";
}

std::string fabricSource(RootFamily family, FixtureKind kind,
                         loom::SpecialMathAccuracyTier guarantee =
                             loom::SpecialMathAccuracyTier::Max4Ulp) {
  const bool singleton = kind == FixtureKind::Singleton;
  const llvm::StringRef formats =
      singleton ? R"mlir(["f32"])mlir"
                : R"mlir(["f16", "bf16", "f32", "f64"])mlir";
  const bool approximate =
      guarantee != loom::SpecialMathAccuracyTier::CorrectlyRounded;
  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @" << moduleName(family)
         << "(%input: !fabric.bits<64>) -> !fabric.bits<64> { "
         << "%pe = fabric.pe [spatial]"
         << "(%pin = %input : !fabric.bits<64>) -> !fabric.bits<64> { "
         << "%fu = fabric.fu"
         << "(%fin = %pin : !fabric.bits<64>) -> !fabric.bits<64> { "
         << "%value = fabric.op [@" << schemaSpelling(family)
         << "] (%fin) {implementation_family = "
         << "#fabric.implementation_family<" << familyKeyword(family)
         << ">, hw_params = {float_formats = " << formats
         << ", behavior = {rounding_modes = [\"to_nearest_even\"], "
            "nan_behaviors = [\"ieee\"], subnormal_behaviors = "
            "[\"preserve\"], signed_zero_behaviors = [\"preserve\"], "
            "fastmath = \""
         << (approximate ? "afn" : "none") << "\"}, accuracy_guarantee = \""
         << loom::stringifySpecialMathAccuracyTier(guarantee)
         << "\"}} : (!fabric.bits<64>) -> !fabric.bits<64> "
         << "fabric.yield %value : !fabric.bits<64> } } "
         << "fabric.yield %pe : !fabric.bits<64> } }";
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
                         RootFamily family,
                         FixtureKind kind = FixtureKind::Configured,
                         loom::SpecialMathAccuracyTier guarantee =
                             loom::SpecialMathAccuracyTier::Max4Ulp) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      fabricSource(family, kind, guarantee), &fabricContext());
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
              "System has no physical math root occurrence");
      return FabricFixture{family, std::move(fabric), occurrence,
                           std::move(system), physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no math root occurrence");
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
  if (mlir::isa<mlir::Float64Type>(type))
    return ::fabric::FloatFormat::F64;
  fail(test, "Fabric projected an unsupported floating format");
}

std::uint8_t modeCode(::fabric::FloatFormat format) {
  return 1 + static_cast<std::uint8_t>(format);
}

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "math root fixture has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  const auto &domain = relation.finiteBehaviorDomain();
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              domain.size() == 4,
          "configured math root domain is not the exact format set");
  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured math root behavior has no semantic key");
    const ::fabric::FloatFormat format = behaviorFormat(test, point);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (format == ::fabric::FloatFormat::F32)
      inactive = semantic;
    if (kind == AbiKind::MissingBehavior &&
        format == ::fabric::FloatFormat::BF16)
      semantic = {0xfd};
    entries.push_back({std::move(semantic), {modeCode(format)}});
  }
  if (kind == AbiKind::ExtraBehavior)
    entries.push_back({{0xfe}, {0x07}});
  require(test, !inactive.empty(), "math root domain has no inactive mode");
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  SemanticFieldEncoding encoding =
      kind == AbiKind::DirectEncoding
          ? SemanticFieldEncoding{DirectBitsEncoding{3}}
          : SemanticFieldEncoding{
                FiniteCodebookEncoding{3, std::move(entries)}};
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
            "configured math root leaf has no selector");
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
  if (llvm::Error error = registerPortableMathRootProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
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
          "portable math root emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

FloatLayout layout(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return {5, 10};
  case ::fabric::FloatFormat::BF16:
    return {8, 7};
  case ::fabric::FloatFormat::F32:
    return {8, 23};
  case ::fabric::FloatFormat::F64:
    return {11, 52};
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

std::uint64_t widthMask(unsigned width) {
  return width == 64 ? UINT64_MAX : (std::uint64_t{1} << width) - 1;
}

std::uint64_t evaluate(llvm::StringRef test, RootFamily family,
                       ::fabric::FloatFormat format, std::uint64_t input) {
  mlir::Type type;
  switch (format) {
  case ::fabric::FloatFormat::F16:
    type = mlir::Float16Type::get(&fabricContext());
    break;
  case ::fabric::FloatFormat::BF16:
    type = mlir::BFloat16Type::get(&fabricContext());
    break;
  case ::fabric::FloatFormat::F32:
    type = mlir::Float32Type::get(&fabricContext());
    break;
  case ::fabric::FloatFormat::F64:
    type = mlir::Float64Type::get(&fabricContext());
    break;
  }
  ::dataflow::CanonicalActorSchemaProjection actor{
      schemaId(family),
      mlir::FunctionType::get(&fabricContext(), {type}, {type}),
      ::dataflow::SpecialMathPayload{mlir::arith::FastMathFlags::afn,
                                     loom::SpecialMathAccuracyTier::Max4Ulp}};
  const unsigned width = layout(format).width();
  const llvm::APFloat operand(semantics(format), llvm::APInt(width, input));
  loom::sim::PrimitiveOperationDescriptor descriptor{actor, width, width};
  loom::sim::PrimitiveValue result = take(
      test, loom::sim::evaluatePrimitiveOperation(
                descriptor, {loom::sim::PrimitiveValue::floating(operand)}));
  require(test, result.isDefined(),
          "independent math root oracle returned a non-defined value");
  return result.bits->getZExtValue();
}

std::uint64_t nextRandom(std::uint64_t &state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

std::vector<RootVector> vectors(RootFamily family) {
  std::vector<RootVector> result;
  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain) {
    const FloatLayout shape = layout(format);
    const unsigned width = shape.width();
    const std::uint64_t sign = shape.sign();
    const std::uint64_t infinity = shape.infinity();
    const std::uint64_t one = shape.one();
    const std::uint64_t two = one + (std::uint64_t{1} << shape.fractionBits);
    const std::uint64_t four = one + (std::uint64_t{2} << shape.fractionBits);
    const std::uint64_t minimumNormal = std::uint64_t{1} << shape.fractionBits;
    const std::uint64_t maximumSubnormal = minimumNormal - 1;
    const std::uint64_t quiet = std::uint64_t{1} << (shape.fractionBits - 1);
    const std::array curated = {
        std::uint64_t{0},
        sign,
        infinity,
        sign | infinity,
        infinity | quiet | 5,
        sign | infinity | 3,
        sign | one,
        one,
        two,
        four,
        std::uint64_t{1},
        maximumSubnormal,
        minimumNormal,
        infinity - 1,
    };
    for (std::uint64_t input : curated)
      result.push_back(
          {format, input, evaluate("vectors", family, format, input)});
    std::uint64_t state =
        0x9e3779b97f4a7c15ULL ^ width ^ static_cast<unsigned>(family);
    for (unsigned count = 0; count != 24; ++count) {
      std::uint64_t input = nextRandom(state) & (infinity - 1);
      if (input == 0)
        input = one;
      result.push_back(
          {format, input, evaluate("vectors", family, format, input)});
    }
  }
  return result;
}

std::string hexLiteral(const llvm::APInt &value) {
  llvm::SmallString<64> digits;
  value.toStringUnsigned(digits, 16);
  const unsigned count = (value.getBitWidth() + 3) / 4;
  std::string padded(count > digits.size() ? count - digits.size() : 0, '0');
  padded += digits.str();
  return std::to_string(value.getBitWidth()) + "'h" + padded;
}

std::uint64_t paddedInput(const RootVector &vector) {
  const unsigned width = layout(vector.format).width();
  if (width == 64)
    return vector.input;
  return vector.input | (0xa5a55a5a5a5aa5a5ULL & ~widthMask(width));
}

std::string makeTestbench(RootFamily family) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "module testbench;\n"
         << "  logic [63:0] input_value;\n"
         << "  logic [2:0] mode;\n"
         << "  logic [63:0] result;\n"
         << "  " << moduleName(family) << " dut(\n"
         << "      .data_input_0(input_value), .config_0(mode),\n"
         << "      .data_output_0(result));\n"
         << "  task automatic check(input logic [2:0] selected,\n"
         << "      input logic [63:0] input_bits,\n"
         << "      input logic [63:0] expected);\n"
         << "    begin\n"
         << "      mode = selected; input_value = input_bits; #1;\n"
         << "      if (result !== expected)\n"
         << "        $fatal(1, \"math root mismatch mode=%0d input=%h "
            "got=%h expected=%h\", selected, input_bits, result, expected);\n"
         << "    end\n"
         << "  endtask\n"
         << "  initial begin\n";
  for (const RootVector &vector : vectors(family))
    output << "    check(3'd" << unsigned(modeCode(vector.format)) << ", "
           << hexLiteral(llvm::APInt(64, paddedInput(vector))) << ", "
           << hexLiteral(llvm::APInt(64, vector.expected)) << ");\n";
  output << "    $finish;\n  end\nendmodule\n";
  return output.str();
}

std::string makeYosysScript(RootFamily family) {
  std::string script;
  llvm::raw_string_ostream output(script);
  output << "read_verilog -sv " << moduleName(family) << ".sv\n"
         << "hierarchy -check -top " << moduleName(family) << '\n'
         << "proc\nopt\ncheck -assert\n"
         << "select -assert-none " << moduleName(family) << "/t:$*ff* "
         << moduleName(family) << "/t:$*latch* " << moduleName(family)
         << "/t:$_*FF* " << moduleName(family) << "/t:$_*LATCH* "
         << moduleName(family) << "/t:$mem* " << moduleName(family) << "/m:*\n"
         << "synth -noabc -top " << moduleName(family) << '\n'
         << "check -assert\nstat\n";
  return output.str();
}

std::string emitFamily(llvm::StringRef test, const ArtifactStore &store,
                       RootFamily family) {
  FabricFixture fixture = makeFabric(test, store, family);
  const auto &resolved = capability(test, fixture);
  require(test,
          resolved.implementationFamily == familyId(family) &&
              resolved.enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{schemaId(family)},
          "math root escaped its generated family descriptor");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 4,
          "math root did not consume the exact sealed format relation");
  std::array<bool, 4> seen{};
  for (const auto &point : relation.finiteBehaviorDomain()) {
    const auto *payload = std::get_if<::dataflow::SpecialMathPayload>(
        &point.representativeActor.payload);
    const ::fabric::FloatFormat format = behaviorFormat(test, point);
    require(test,
            point.representativeActor.schema == schemaId(family) && payload &&
                payload->accuracy == loom::SpecialMathAccuracyTier::Max4Ulp &&
                point.operandPorts == std::vector<std::uint64_t>{0} &&
                point.resultPorts == std::vector<std::uint64_t>{0} &&
                point.semanticConfiguration.has_value(),
            "math root relation lost its sealed physical correspondence");
    seen[static_cast<std::size_t>(format)] = true;
  }
  require(test, llvm::all_of(seen, [](bool value) { return value; }),
          "math root relation omitted an admitted format");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fixture, abi.abi());
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 3 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "config_0" &&
              ports.atInput(1).type ==
                  mlir::IntegerType::get(firstContext.get(), 3) &&
              ports.atOutput(0).getName() == "data_output_0",
          "math root leaf ports do not follow ConfigurationABI 2.0");
  const std::string firstRtl = specialize(test, std::move(first), fixture, abi);
  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fixture, abi.abi());
  const std::string secondRtl =
      specialize(test, std::move(second), fixture, abi);
  require(test, firstRtl == secondRtl,
          "identical math root inputs produced different SystemVerilog");
  const llvm::StringRef rtl(firstRtl);
  require(test,
          rtl.contains("function automatic") &&
              rtl.contains(family == RootFamily::Sqrt
                               ? "loom_math_sqrt_e11_f52"
                               : "loom_math_rsqrt_e11_f52") &&
              rtl.contains("config_0") && !rtl.contains("$sqrt") &&
              !rtl.contains("shortreal") && !rtl.contains(" DPI") &&
              !rtl.contains(" real"),
          "math root RTL is incomplete or not portable synthesizable logic");
  return firstRtl;
}

void configuredDomainAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  for (RootFamily family : {RootFamily::Sqrt, RootFamily::Rsqrt}) {
    const auto entry = llvm::find_if(coverage, [&](const auto &candidate) {
      return candidate.implementationFamily == familyId(family);
    });
    require(test,
            entry != coverage.end() &&
                entry->recipes ==
                    std::vector<BackendRecipeKey>{
                        BackendRecipeKey::PortableSystemVerilog},
            "math root provider registration is incomplete");
  }
  const std::string sqrt = emitFamily(test, store, RootFamily::Sqrt);
  const std::string rsqrt = emitFamily(test, store, RootFamily::Rsqrt);
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated",
          {{"scalar_math_sqrt.sv", sqrt},
           {"scalar_math_rsqrt.sv", rsqrt},
           {"sqrt_testbench.sv", makeTestbench(RootFamily::Sqrt)},
           {"rsqrt_testbench.sv", makeTestbench(RootFamily::Rsqrt)},
           {"portable_scalar_math_sqrt.ys", makeYosysScript(RootFamily::Sqrt)},
           {"portable_scalar_math_rsqrt.ys",
            makeYosysScript(RootFamily::Rsqrt)}}))
    fail(test, llvm::toString(std::move(error)));
}

void accuracyTiersUseOneCorrectlyRoundedCircuit(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  for (RootFamily family : {RootFamily::Sqrt, RootFamily::Rsqrt}) {
    for (loom::SpecialMathAccuracyTier tier :
         loom::specialMathAccuracyTiers()) {
      FabricFixture fixture =
          makeFabric(test, store, family, FixtureKind::Singleton, tier);
      const auto &resolved = capability(test, fixture);
      const auto *parameters = std::get_if<::fabric::ScalarSpecialMathParams>(
          &resolved.parameterizedCapability);
      require(test, parameters && parameters->accuracyGuarantee == tier,
              "math root provider lost the Fabric accuracy guarantee");
      FinalizedConfigurationABI abi =
          makeConfigurationAbi(test, store, fixture);
      std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
      SkeletonFixture skeleton =
          makeSkeleton(test, *context, fixture, abi.abi());
      require(test, skeleton.leaf.getPortList().size() == 2,
              "singleton math root retained a selector port");
      const std::string rtl =
          specialize(test, std::move(skeleton), fixture, abi);
      require(test,
              llvm::StringRef(rtl).contains(family == RootFamily::Sqrt
                                                ? "loom_math_sqrt_e8_f23"
                                                : "loom_math_rsqrt_e8_f23") &&
                  !llvm::StringRef(rtl).contains("config_0"),
              "math root tier did not use the singleton root circuit");
    }
  }
}

void malformedInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  for (RootFamily family : {RootFamily::Sqrt, RootFamily::Rsqrt}) {
    FabricFixture valid = makeFabric(test, store, family);
    FinalizedConfigurationABI validAbi =
        makeConfigurationAbi(test, store, valid);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture malformed =
        makeSkeleton(test, *context, valid, validAbi.abi(), true);
    const std::string before = moduleText(*malformed.module);
    expectInvalid(test, trySpecialize(malformed, valid, validAbi, registry),
                  "leaf port");
    require(test, moduleText(*malformed.module) == before,
            "malformed math root input partially mutated the skeleton");

    for (AbiKind kind : {AbiKind::MissingBehavior, AbiKind::ExtraBehavior,
                         AbiKind::DirectEncoding})
      expectInvalid(
          test,
          finalizeConfigurationABI(makeConfigurationAbiDraft(test, valid, kind),
                                   store),
          kind == AbiKind::DirectEncoding ? "finite codebook" : "semantic");

    FabricFixture unsupported =
        makeFabric(test, store, family, FixtureKind::UnsupportedContract);
    FinalizedConfigurationABI unsupportedAbi =
        makeConfigurationAbi(test, store, unsupported);
    std::unique_ptr<mlir::MLIRContext> unsupportedContext = makeCirctContext();
    SkeletonFixture unsupportedSkeleton = makeSkeleton(
        test, *unsupportedContext, unsupported, unsupportedAbi.abi());
    const std::string unsupportedBefore =
        moduleText(*unsupportedSkeleton.module);
    expectTypedUnsupported(test,
                           trySpecialize(unsupportedSkeleton, unsupported,
                                         unsupportedAbi, registry),
                           familyId(family),
                           BackendRecipeKey::PortableSystemVerilog);
    require(test, moduleText(*unsupportedSkeleton.module) == unsupportedBefore,
            "Unsupported math root mutated the caller skeleton");

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
  configuredDomainAndArtifacts(root);
  accuracyTiersUseOneCorrectlyRoundedCircuit(root / "tiers");
  malformedInputsAreTransactional(root / "malformed");
  return EXIT_SUCCESS;
}

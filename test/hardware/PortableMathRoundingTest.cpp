#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/MathRounding.h"
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

enum class RoundingFamily { Floor, Ceil, Round, Trunc, RoundEven };
enum class FixtureKind { Configured, Singleton, UnsupportedContract };
enum class AbiKind { Complete, MissingBehavior, ExtraBehavior, DirectEncoding };

constexpr std::array roundingFamilies = {
    RoundingFamily::Floor, RoundingFamily::Ceil, RoundingFamily::Round,
    RoundingFamily::Trunc, RoundingFamily::RoundEven};

struct FabricFixture final {
  RoundingFamily family;
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
  std::uint64_t exponentMask() const {
    return (std::uint64_t{1} << exponentBits) - 1;
  }
  std::uint64_t infinity() const { return exponentMask() << fractionBits; }
  std::uint64_t one() const {
    return ((std::uint64_t{1} << (exponentBits - 1)) - 1) << fractionBits;
  }
};

struct Vector final {
  RoundingFamily family;
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
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted malformed math-rounding input");
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

::fabric::ImplementationFamilyId familyId(RoundingFamily family) {
  using Family = ::fabric::ImplementationFamilyId;
  switch (family) {
  case RoundingFamily::Floor:
    return Family::ScalarMathFloor;
  case RoundingFamily::Ceil:
    return Family::ScalarMathCeil;
  case RoundingFamily::Round:
    return Family::ScalarMathRound;
  case RoundingFamily::Trunc:
    return Family::ScalarMathTrunc;
  case RoundingFamily::RoundEven:
    return Family::ScalarMathRoundEven;
  }
  llvm_unreachable("unknown math-rounding family");
}

::dataflow::OperationSchemaId schemaId(RoundingFamily family) {
  using Schema = ::dataflow::OperationSchemaId;
  switch (family) {
  case RoundingFamily::Floor:
    return Schema::MathFloor;
  case RoundingFamily::Ceil:
    return Schema::MathCeil;
  case RoundingFamily::Round:
    return Schema::MathRound;
  case RoundingFamily::Trunc:
    return Schema::MathTrunc;
  case RoundingFamily::RoundEven:
    return Schema::MathRoundEven;
  }
  llvm_unreachable("unknown math-rounding family");
}

llvm::StringRef familyKeyword(RoundingFamily family) {
  switch (family) {
  case RoundingFamily::Floor:
    return "ScalarMathFloor";
  case RoundingFamily::Ceil:
    return "ScalarMathCeil";
  case RoundingFamily::Round:
    return "ScalarMathRound";
  case RoundingFamily::Trunc:
    return "ScalarMathTrunc";
  case RoundingFamily::RoundEven:
    return "ScalarMathRoundEven";
  }
  llvm_unreachable("unknown math-rounding family");
}

llvm::StringRef schemaKeyword(RoundingFamily family) {
  switch (family) {
  case RoundingFamily::Floor:
    return "math.floor";
  case RoundingFamily::Ceil:
    return "math.ceil";
  case RoundingFamily::Round:
    return "math.round";
  case RoundingFamily::Trunc:
    return "math.trunc";
  case RoundingFamily::RoundEven:
    return "math.roundeven";
  }
  llvm_unreachable("unknown math-rounding family");
}

llvm::StringRef moduleName(RoundingFamily family) {
  switch (family) {
  case RoundingFamily::Floor:
    return "scalar_math_floor";
  case RoundingFamily::Ceil:
    return "scalar_math_ceil";
  case RoundingFamily::Round:
    return "scalar_math_round";
  case RoundingFamily::Trunc:
    return "scalar_math_trunc";
  case RoundingFamily::RoundEven:
    return "scalar_math_roundeven";
  }
  llvm_unreachable("unknown math-rounding family");
}

std::string fabricSource(RoundingFamily family, FixtureKind kind,
                         loom::SpecialMathAccuracyTier guarantee =
                             loom::SpecialMathAccuracyTier::CorrectlyRounded) {
  const bool singleton = kind == FixtureKind::Singleton;
  const llvm::StringRef formats =
      singleton ? R"mlir(["f32"])mlir"
                : R"mlir(["f16", "bf16", "f32", "f64"])mlir";
  const bool approximate =
      guarantee != loom::SpecialMathAccuracyTier::CorrectlyRounded;
  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @" << moduleName(family)
         << "(%input: !fabric.bits<64>) -> !fabric.bits<64> {"
         << " %pe = fabric.pe [spatial]"
         << "(%pe_input = %input : !fabric.bits<64>) -> !fabric.bits<64> {"
         << " %fu = fabric.fu"
         << "(%fu_input = %pe_input : !fabric.bits<64>)"
         << " -> !fabric.bits<64> {"
         << " %value = fabric.op [@" << schemaKeyword(family)
         << "] (%fu_input) {implementation_family = "
         << "#fabric.implementation_family<" << familyKeyword(family)
         << ">, hw_params = {float_formats = " << formats
         << ", behavior = {rounding_modes = [\"to_nearest_even\"],"
         << " nan_behaviors = [\"ieee\"],"
         << " subnormal_behaviors = [\"preserve\"],"
         << " signed_zero_behaviors = [\"preserve\"], fastmath = \""
         << (approximate ? "afn" : "none") << "\"}, accuracy_guarantee = \""
         << loom::stringifySpecialMathAccuracyTier(guarantee)
         << "\"}} : (!fabric.bits<64>) -> !fabric.bits<64>"
         << " fabric.yield %value : !fabric.bits<64> } }"
         << " fabric.yield %pe : !fabric.bits<64> } }";
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
                         RoundingFamily family,
                         FixtureKind kind = FixtureKind::Configured,
                         loom::SpecialMathAccuracyTier guarantee =
                             loom::SpecialMathAccuracyTier::CorrectlyRounded) {
  const std::string sourceText = fabricSource(family, kind, guarantee);
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
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
              "System has no physical math-rounding occurrence");
      return FabricFixture{family, std::move(fabric), occurrence,
                           std::move(system), physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no math-rounding occurrence");
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
  const mlir::Type type = point.representativeActor.type.getInput(0);
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
          "math-rounding fixture has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "math-rounding relation is not finite");
  const auto &domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 4,
          "configured math-rounding domain is not the format set");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured math-rounding behavior has no semantic value");
    const auto format = behaviorFormat(test, point);
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
  require(test, !inactive.empty(),
          "math-rounding domain has no inactive behavior");
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
            "configured math-rounding leaf has no selector");
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
  if (llvm::Error error = registerPortableMathRoundingProviders(registry))
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
          "portable math rounding emitted external implementation state");
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

llvm::RoundingMode roundingMode(RoundingFamily family) {
  switch (family) {
  case RoundingFamily::Floor:
    return llvm::RoundingMode::TowardNegative;
  case RoundingFamily::Ceil:
    return llvm::RoundingMode::TowardPositive;
  case RoundingFamily::Round:
    return llvm::RoundingMode::NearestTiesToAway;
  case RoundingFamily::Trunc:
    return llvm::RoundingMode::TowardZero;
  case RoundingFamily::RoundEven:
    return llvm::RoundingMode::NearestTiesToEven;
  }
  llvm_unreachable("unknown math-rounding family");
}

std::uint64_t rounded(RoundingFamily family, ::fabric::FloatFormat format,
                      std::uint64_t input) {
  const unsigned width = layout(format).width();
  llvm::APFloat value(semantics(format), llvm::APInt(width, input));
  (void)value.roundToIntegral(roundingMode(family));
  return value.bitcastToAPInt().getZExtValue();
}

std::uint64_t widthMask(unsigned width) {
  return width == 64 ? UINT64_MAX : (std::uint64_t{1} << width) - 1;
}

std::uint64_t encoded(::fabric::FloatFormat format, llvm::StringRef value) {
  return llvm::APFloat(semantics(format), value)
      .bitcastToAPInt()
      .getZExtValue();
}

std::vector<std::uint64_t> inputs(::fabric::FloatFormat format) {
  const FloatLayout shape = layout(format);
  const std::uint64_t sign = shape.sign();
  const std::uint64_t one = shape.one();
  const std::uint64_t unit = std::uint64_t{1} << shape.fractionBits;
  const std::uint64_t half = one - unit;
  const std::uint64_t quarter = half - (unit >> 1);
  const std::uint64_t infinity = shape.infinity();
  const std::uint64_t quiet = std::uint64_t{1} << (shape.fractionBits - 1);
  return {0,
          sign,
          1,
          sign | 1,
          unit - 1,
          sign | (unit - 1),
          quarter,
          sign | quarter,
          half - 1,
          sign | (half - 1),
          half,
          sign | half,
          half + 1,
          sign | (half + 1),
          one,
          sign | one,
          one + (unit >> 2),
          sign | (one + (unit >> 2)),
          one + (unit >> 1),
          sign | (one + (unit >> 1)),
          one + (unit >> 1) + (unit >> 2),
          sign | (one + (unit >> 1) + (unit >> 2)),
          one + unit + (unit >> 1),
          sign | (one + unit + (unit >> 1)),
          one + 2 * unit + (unit >> 1),
          sign | (one + 2 * unit + (unit >> 1)),
          infinity - 1,
          sign | (infinity - 1),
          infinity,
          sign | infinity,
          infinity | quiet | 0x5,
          sign | infinity | quiet | 0x5,
          infinity | 0x3,
          sign | infinity | 0x3,
          encoded(format, "2.5"),
          encoded(format, "-2.5"),
          encoded(format, "3.5"),
          encoded(format, "-3.5")};
}

std::vector<Vector> vectors() {
  std::vector<Vector> result;
  for (RoundingFamily family : roundingFamilies)
    for (::fabric::FloatFormat format : ::fabric::floatFormatDomain)
      for (std::uint64_t input : inputs(format))
        result.push_back(
            {family, format, input, rounded(family, format, input)});
  return result;
}

std::string hexDigits(const llvm::APInt &value) {
  llvm::SmallString<64> digits;
  value.toStringUnsigned(digits, 16);
  const unsigned digitCount = (value.getBitWidth() + 3) / 4;
  std::string padded(
      digitCount > digits.size() ? digitCount - digits.size() : 0, '0');
  padded += digits.str();
  return padded;
}

std::uint64_t paddedInput(::fabric::FloatFormat format, std::uint64_t value) {
  const unsigned width = layout(format).width();
  if (width == 64)
    return value;
  return value | (0xa5a55a5a5a5aa5a5ULL & ~widthMask(width));
}

std::string makeVectorMemory() {
  std::string text;
  llvm::raw_string_ostream output(text);
  for (const Vector &vector : vectors()) {
    llvm::APInt entry(134, vector.expected);
    entry.insertBits(llvm::APInt(64, paddedInput(vector.format, vector.input)),
                     64);
    entry.insertBits(llvm::APInt(3, modeCode(vector.format)), 128);
    entry.insertBits(llvm::APInt(3, static_cast<unsigned>(vector.family)), 131);
    output << hexDigits(entry) << '\n';
  }
  return output.str();
}

std::string makeTestbench() {
  const std::size_t vectorCount = vectors().size();
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "module testbench;\n"
         << "  localparam integer VECTOR_COUNT = " << vectorCount << ";\n"
         << R"sv(  logic [2:0] mode;
  logic [63:0] value;
  logic [63:0] floor_result;
  logic [63:0] ceil_result;
  logic [63:0] round_result;
  logic [63:0] trunc_result;
  logic [63:0] roundeven_result;
  logic [133:0] test_vectors [0:VECTOR_COUNT-1];
  integer vector_index;

  scalar_math_floor floor_dut(
      .data_input_0(value), .config_0(mode), .data_output_0(floor_result));
  scalar_math_ceil ceil_dut(
      .data_input_0(value), .config_0(mode), .data_output_0(ceil_result));
  scalar_math_round round_dut(
      .data_input_0(value), .config_0(mode), .data_output_0(round_result));
  scalar_math_trunc trunc_dut(
      .data_input_0(value), .config_0(mode), .data_output_0(trunc_result));
  scalar_math_roundeven roundeven_dut(
      .data_input_0(value), .config_0(mode),
      .data_output_0(roundeven_result));

  task automatic check_value(
      input logic [2:0] family,
      input logic [2:0] selected_mode,
      input logic [63:0] input_value,
      input logic [63:0] expected);
    logic [63:0] actual;
    begin
      mode = selected_mode;
      value = input_value;
      #1;
      case (family)
        0: actual = floor_result;
        1: actual = ceil_result;
        2: actual = round_result;
        3: actual = trunc_result;
        4: actual = roundeven_result;
        default: $fatal(1, "unknown math-rounding family");
      endcase
      if (actual !== expected)
        $fatal(1, "math rounding mismatch family=%0d mode=%0d input=%h got=%h expected=%h",
               family, selected_mode, input_value, actual, expected);
    end
  endtask

  initial begin
    $readmemh("vectors.mem", test_vectors);
    for (vector_index = 0; vector_index < VECTOR_COUNT;
         vector_index = vector_index + 1)
      check_value(test_vectors[vector_index][133:131],
                  test_vectors[vector_index][130:128],
                  test_vectors[vector_index][127:64],
                  test_vectors[vector_index][63:0]);
    $finish;
  end
endmodule
)sv";
  return output.str();
}

std::string makeSynthesisTop() {
  return R"sv(module math_rounding_synthesis_top(
    input logic [2:0] mode,
    input logic [63:0] value,
    output logic [319:0] result);
  scalar_math_floor floor_dut(
      .data_input_0(value), .config_0(mode), .data_output_0(result[63:0]));
  scalar_math_ceil ceil_dut(
      .data_input_0(value), .config_0(mode), .data_output_0(result[127:64]));
  scalar_math_round round_dut(
      .data_input_0(value), .config_0(mode), .data_output_0(result[191:128]));
  scalar_math_trunc trunc_dut(
      .data_input_0(value), .config_0(mode), .data_output_0(result[255:192]));
  scalar_math_roundeven roundeven_dut(
      .data_input_0(value), .config_0(mode), .data_output_0(result[319:256]));
endmodule
)sv";
}

std::string makeYosysScript() {
  return R"ys(read_verilog -sv scalar_math_floor.sv scalar_math_ceil.sv scalar_math_round.sv scalar_math_trunc.sv scalar_math_roundeven.sv synthesis_top.sv
hierarchy -check -top math_rounding_synthesis_top
proc
opt
check -assert
select -assert-none t:$*ff* t:$*latch* t:$_*FF* t:$_*LATCH* t:$mem*
synth -noabc -top math_rounding_synthesis_top
check -assert
select -assert-none t:$*ff* t:$*latch* t:$_*FF* t:$_*LATCH* t:$mem*
stat
)ys";
}

std::string emitFamily(llvm::StringRef test, const ArtifactStore &store,
                       RoundingFamily family) {
  FabricFixture fabric = makeFabric(test, store, family);
  const auto &resolved = capability(test, fabric);
  require(
      test,
      resolved.implementationFamily == familyId(family) &&
          resolved.enabledOperationSchemas ==
              std::vector<::dataflow::OperationSchemaId>{schemaId(family)} &&
          std::holds_alternative<::fabric::ScalarSpecialMathParams>(
              resolved.parameterizedCapability),
      "math rounding escaped its generated family descriptor");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 4,
          "math rounding did not consume the exact sealed format relation");
  for (const auto &point : relation.finiteBehaviorDomain()) {
    const auto *payload = std::get_if<::dataflow::SpecialMathPayload>(
        &point.representativeActor.payload);
    require(test,
            point.representativeActor.schema == schemaId(family) && payload &&
                payload->accuracy ==
                    loom::SpecialMathAccuracyTier::CorrectlyRounded &&
                point.semanticConfiguration.has_value(),
            "math-rounding relation contains a malformed behavior witness");
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
          "math-rounding leaf ports do not follow ConfigurationABI 2.0");
  const std::string first =
      specialize(test, std::move(firstSkeleton), fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture secondSkeleton =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string second =
      specialize(test, std::move(secondSkeleton), fabric, abi);
  require(test, first == second,
          "identical math-rounding inputs produced different SystemVerilog");
  const llvm::StringRef rtl(first);
  require(test,
          rtl.contains("function automatic") && rtl.contains("config_0") &&
              rtl.contains("loom_math_rounding_e5_f10") &&
              rtl.contains("loom_math_rounding_e8_f7") &&
              rtl.contains("loom_math_rounding_e8_f23") &&
              rtl.contains("loom_math_rounding_e11_f52") &&
              !rtl.contains("shortreal") && !rtl.contains("real") &&
              !rtl.contains("DPI"),
          "math-rounding RTL is incomplete or not portable logic");
  return first;
}

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  for (RoundingFamily family : roundingFamilies) {
    const auto entry = llvm::find_if(coverage, [&](const auto &candidate) {
      return candidate.implementationFamily == familyId(family);
    });
    require(test,
            entry != coverage.end() &&
                entry->recipes ==
                    std::vector<BackendRecipeKey>{
                        BackendRecipeKey::PortableSystemVerilog},
            "math-rounding provider registration is incomplete");
  }

  std::vector<loom::hardware::test::PortableProviderArtifact> artifacts;
  for (RoundingFamily family : roundingFamilies)
    artifacts.push_back(
        {moduleName(family).str() + ".sv", emitFamily(test, store, family)});
  artifacts.push_back({"vectors.mem", makeVectorMemory()});
  artifacts.push_back({"testbench.sv", makeTestbench()});
  artifacts.push_back({"synthesis_top.sv", makeSynthesisTop()});
  artifacts.push_back({"portable_math_rounding.ys", makeYosysScript()});
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated", artifacts))
    fail(test, llvm::toString(std::move(error)));
}

void singletonAndAccuracyTiers(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  for (RoundingFamily family : roundingFamilies) {
    FabricFixture fabric =
        makeFabric(test, store, family, FixtureKind::Singleton);
    const auto &resolved = capability(test, fabric);
    auto relation =
        take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
    require(test,
            relation.kind() ==
                    ::fabric::FabricOpSemanticFieldRelationKind::None &&
                relation.finiteBehaviorDomain().size() == 1 &&
                !relation.finiteBehaviorDomain().front().semanticConfiguration,
            "singleton math rounding retained a configuration authority");
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
    require(test, skeleton.leaf.getPortList().size() == 2,
            "singleton math-rounding leaf retained a selector port");
    const std::string rtl = specialize(test, std::move(skeleton), fabric, abi);
    require(test,
            llvm::StringRef(rtl).contains("loom_math_rounding_e8_f23") &&
                !llvm::StringRef(rtl).contains("config_0"),
            "singleton math rounding did not emit only its sealed witness");
  }

  for (loom::SpecialMathAccuracyTier tier : loom::specialMathAccuracyTiers()) {
    FabricFixture fabric = makeFabric(test, store, RoundingFamily::RoundEven,
                                      FixtureKind::Singleton, tier);
    const auto &resolved = capability(test, fabric);
    auto relation =
        take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
    const auto *payload = std::get_if<::dataflow::SpecialMathPayload>(
        &relation.finiteBehaviorDomain().front().representativeActor.payload);
    require(test, payload && payload->accuracy == tier,
            "math rounding lost the admitted accuracy tier");
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
    const std::string rtl = specialize(test, std::move(skeleton), fabric, abi);
    require(test, llvm::StringRef(rtl).contains("loom_math_rounding_e8_f23"),
            "admitted accuracy tier did not materialize exact rounding RTL");
  }
}

void expectTypedUnsupported(
    llvm::StringRef test, llvm::Expected<FabricOperationProviderOutput> result,
    ::fabric::ImplementationFamilyId family, BackendRecipeKey recipe) {
  require(test, !result, "unsupported math-rounding combination was accepted");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified =
            error.implementationFamily() == family && error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test,
             "unsupported math rounding returned the wrong error class: " +
                 error.message());
      });
  require(test, classified,
          "math rounding lost its typed Unsupported classification");
}

void malformedInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  FabricFixture valid = makeFabric(test, store, RoundingFamily::Floor);
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture malformed =
      makeSkeleton(test, *context, valid, validAbi.abi(), true);
  const std::string before = moduleText(*malformed.module);
  expectError(test, trySpecialize(malformed, valid, validAbi, registry),
              "leaf port");
  require(test, moduleText(*malformed.module) == before,
          "malformed math-rounding input partially mutated the skeleton");

  for (AbiKind kind : {AbiKind::MissingBehavior, AbiKind::ExtraBehavior,
                       AbiKind::DirectEncoding})
    expectError(
        test,
        finalizeConfigurationABI(makeConfigurationAbiDraft(test, valid, kind),
                                 store),
        kind == AbiKind::DirectEncoding ? "finite codebook" : "semantic");

  for (RoundingFamily family : roundingFamilies) {
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
            "Unsupported math rounding mutated the caller skeleton");
  }

  for (BackendRecipeKey recipe :
       {BackendRecipeKey::SynopsysDesignWare, BackendRecipeKey::CadenceChipWare,
        BackendRecipeKey::AmdXilinx, BackendRecipeKey::IntelAltera}) {
    std::unique_ptr<mlir::MLIRContext> nativeContext = makeCirctContext();
    SkeletonFixture native =
        makeSkeleton(test, *nativeContext, valid, validAbi.abi());
    const std::string nativeBefore = moduleText(*native.module);
    expectTypedUnsupported(
        test, trySpecialize(native, valid, validAbi, registry, recipe),
        familyId(RoundingFamily::Floor), recipe);
    require(test, moduleText(*native.module) == nativeBefore,
            "unsupported native recipe mutated the caller skeleton");
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  configuredBehaviorAndArtifacts(root);
  singletonAndAccuracyTiers(root / "singleton");
  malformedInputsAreTransactional(root / "malformed");
  return EXIT_SUCCESS;
}

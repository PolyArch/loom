#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/MathLogarithm.h"
#include "PortableProviderTestSupport.h"
#include "Simulator/OperationSemantics.h"

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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
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

enum class MathFamily { Log, Log2, Log10, Log1p };
enum class FixtureKind {
  Configured,
  Singleton,
  UnsupportedContract,
  UnsupportedFormat,
  UnsupportedAccuracy,
};
enum class AbiKind { Complete, MissingBehavior, ExtraBehavior, DirectEncoding };
enum class ExpectedKind : unsigned { Bounded = 0, Exact = 1, QuietNaN = 2 };

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

  unsigned width() const { return 1 + exponentBits + fractionBits; }
  std::uint64_t sign() const { return std::uint64_t{1} << (width() - 1); }
  std::uint64_t exponentMask() const {
    return (std::uint64_t{1} << exponentBits) - 1;
  }
  std::uint64_t infinity() const { return exponentMask() << fractionBits; }
  std::uint64_t one() const {
    return std::uint64_t((std::uint64_t{1} << (exponentBits - 1)) - 1)
           << fractionBits;
  }
};

struct NumericVector final {
  MathFamily family;
  ::fabric::FloatFormat format;
  std::uint64_t input;
  std::uint64_t expected;
  ExpectedKind kind;
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
    fail(test, "accepted malformed logarithm input");
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

::fabric::ImplementationFamilyId familyId(MathFamily family) {
  using Id = ::fabric::ImplementationFamilyId;
  switch (family) {
  case MathFamily::Log:
    return Id::ScalarMathLog;
  case MathFamily::Log2:
    return Id::ScalarMathLog2;
  case MathFamily::Log10:
    return Id::ScalarMathLog10;
  case MathFamily::Log1p:
    return Id::ScalarMathLog1p;
  }
  llvm_unreachable("unknown logarithm family");
}

::dataflow::OperationSchemaId schemaId(MathFamily family) {
  using Id = ::dataflow::OperationSchemaId;
  switch (family) {
  case MathFamily::Log:
    return Id::MathLog;
  case MathFamily::Log2:
    return Id::MathLog2;
  case MathFamily::Log10:
    return Id::MathLog10;
  case MathFamily::Log1p:
    return Id::MathLog1p;
  }
  llvm_unreachable("unknown logarithm family");
}

llvm::StringRef familyKeyword(MathFamily family) {
  switch (family) {
  case MathFamily::Log:
    return "ScalarMathLog";
  case MathFamily::Log2:
    return "ScalarMathLog2";
  case MathFamily::Log10:
    return "ScalarMathLog10";
  case MathFamily::Log1p:
    return "ScalarMathLog1p";
  }
  llvm_unreachable("unknown logarithm family");
}

llvm::StringRef schemaKeyword(MathFamily family) {
  switch (family) {
  case MathFamily::Log:
    return "math.log";
  case MathFamily::Log2:
    return "math.log2";
  case MathFamily::Log10:
    return "math.log10";
  case MathFamily::Log1p:
    return "math.log1p";
  }
  llvm_unreachable("unknown logarithm family");
}

llvm::StringRef shortName(MathFamily family) {
  switch (family) {
  case MathFamily::Log:
    return "log";
  case MathFamily::Log2:
    return "log2";
  case MathFamily::Log10:
    return "log10";
  case MathFamily::Log1p:
    return "log1p";
  }
  llvm_unreachable("unknown logarithm family");
}

std::string moduleName(MathFamily family) {
  return "scalar_math_" + shortName(family).str();
}

std::string fabricSource(MathFamily family, FixtureKind kind) {
  const llvm::StringRef formats =
      kind == FixtureKind::Configured ||
              kind == FixtureKind::UnsupportedContract
          ? R"mlir(["f16", "bf16", "f32"])mlir"
      : kind == FixtureKind::UnsupportedFormat ? R"mlir(["f64"])mlir"
                                               : R"mlir(["f32"])mlir";
  const llvm::StringRef accuracy =
      kind == FixtureKind::UnsupportedAccuracy ? "Max2Ulp" : "Max4Ulp";
  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @" << moduleName(family)
         << "(%a: !fabric.bits<64>) -> !fabric.bits<64> { "
            "%pe = fabric.pe [spatial](%pa = %a : !fabric.bits<64>) -> "
            "!fabric.bits<64> { %fu = fabric.fu"
            "(%fa = %pa : !fabric.bits<64>) -> !fabric.bits<64> { "
            "%value = fabric.op [@"
         << schemaKeyword(family) << "] (%fa) {implementation_family = "
         << "#fabric.implementation_family<" << familyKeyword(family)
         << ">, hw_params = {float_formats = " << formats
         << ", behavior = {rounding_modes = [\"to_nearest_even\"], "
            "nan_behaviors = [\"ieee\"], subnormal_behaviors = "
            "[\"preserve\"], signed_zero_behaviors = [\"preserve\"], "
            "fastmath = \"afn\"}, accuracy_guarantee = \""
         << accuracy
         << "\"}} : (!fabric.bits<64>) -> !fabric.bits<64> "
            "fabric.yield %value : !fabric.bits<64> } } "
            "fabric.yield %pe : !fabric.bits<64> } }";
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
              "System has no physical logarithm occurrence");
      return FabricFixture{family, std::move(fabric), occurrence,
                           std::move(system), physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no logarithm operation occurrence");
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
  fail(test, "Fabric projected an unsupported logarithm format");
}

std::uint8_t physicalCode(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return 5;
  case ::fabric::FloatFormat::BF16:
    return 17;
  case ::fabric::FloatFormat::F32:
    return 42;
  case ::fabric::FloatFormat::F64:
    return 55;
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
          "logarithm fixture has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 3,
          "logarithm relation is not the sealed three-format domain");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "configured logarithm behavior has no semantic value");
    const auto format = behaviorFormat(test, point);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (format == ::fabric::FloatFormat::F32)
      inactive = semantic;
    if (kind == AbiKind::MissingBehavior &&
        format == ::fabric::FloatFormat::BF16)
      semantic = {0xfd};
    entries.push_back({std::move(semantic), {physicalCode(format)}});
  }
  if (kind == AbiKind::ExtraBehavior)
    entries.push_back({{0xfe}, {0x3f}});
  require(test, !inactive.empty(), "logarithm domain has no inactive behavior");
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  SemanticFieldEncoding encoding =
      kind == AbiKind::DirectEncoding
          ? SemanticFieldEncoding{DirectBitsEncoding{6}}
          : SemanticFieldEncoding{
                FiniteCodebookEncoding{6, std::move(entries)}};
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
            "configured logarithm leaf has no selector");
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
  if (llvm::Error error = registerPortableMathLogarithmProviders(registry))
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
          "portable logarithm provider emitted external state");
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

mlir::Type floatType(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return mlir::Float16Type::get(&fabricContext());
  case ::fabric::FloatFormat::BF16:
    return mlir::BFloat16Type::get(&fabricContext());
  case ::fabric::FloatFormat::F32:
    return mlir::Float32Type::get(&fabricContext());
  case ::fabric::FloatFormat::F64:
    return mlir::Float64Type::get(&fabricContext());
  }
  llvm_unreachable("unknown floating format");
}

std::uint64_t widthMask(unsigned width) {
  return width == 64 ? UINT64_MAX : (std::uint64_t{1} << width) - 1;
}

llvm::APFloat floating(::fabric::FloatFormat format, std::uint64_t bits) {
  return llvm::APFloat(semantics(format),
                       llvm::APInt(layout(format).width(), bits));
}

std::uint64_t oracle(llvm::StringRef test, MathFamily family,
                     ::fabric::FloatFormat format, std::uint64_t input) {
  mlir::Type type = floatType(format);
  loom::sim::PrimitiveOperationDescriptor descriptor{
      {schemaId(family),
       mlir::FunctionType::get(&fabricContext(), {type}, {type}),
       ::dataflow::SpecialMathPayload{mlir::arith::FastMathFlags::afn,
                                      loom::SpecialMathAccuracyTier::Max4Ulp}},
      layout(format).width(),
      layout(format).width()};
  auto result = take(test, loom::sim::evaluatePrimitiveOperation(
                               descriptor, {loom::sim::PrimitiveValue::floating(
                                               floating(format, input))}));
  require(test, result.isDefined(), "MPFR logarithm oracle was not defined");
  return result.bits->getZExtValue();
}

ExpectedKind classifyExpected(::fabric::FloatFormat format, std::uint64_t input,
                              std::uint64_t expected) {
  llvm::APFloat operand = floating(format, input);
  llvm::APFloat result = floating(format, expected);
  if (operand.isNaN())
    return ExpectedKind::Exact;
  if (result.isNaN())
    return ExpectedKind::QuietNaN;
  if (result.isZero() || result.isInfinity())
    return ExpectedKind::Exact;
  return ExpectedKind::Bounded;
}

void appendVector(llvm::StringRef test, std::vector<NumericVector> &vectors,
                  MathFamily family, ::fabric::FloatFormat format,
                  std::uint64_t input, bool exact = false) {
  input &= widthMask(layout(format).width());
  const std::uint64_t expected = oracle(test, family, format, input);
  ExpectedKind kind = classifyExpected(format, input, expected);
  if (exact)
    kind = ExpectedKind::Exact;
  vectors.push_back({family, format, input, expected, kind});
}

std::uint64_t nextRandom(std::uint64_t &state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

std::uint64_t sqrtTwo(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return 0x3da8;
  case ::fabric::FloatFormat::BF16:
    return 0x3fb5;
  case ::fabric::FloatFormat::F32:
    return 0x3fb504f3;
  case ::fabric::FloatFormat::F64:
    return 0x3ff6a09e667f3bcdULL;
  }
  llvm_unreachable("unknown floating format");
}

std::uint64_t ten(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return 0x4900;
  case ::fabric::FloatFormat::BF16:
    return 0x4120;
  case ::fabric::FloatFormat::F32:
    return 0x41200000;
  case ::fabric::FloatFormat::F64:
    return 0x4024000000000000ULL;
  }
  llvm_unreachable("unknown floating format");
}

std::vector<NumericVector> numericVectors(MathFamily family) {
  const llvm::StringRef test = "numericVectors";
  std::vector<NumericVector> vectors;
  for (::fabric::FloatFormat format :
       {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
        ::fabric::FloatFormat::F32}) {
    const FloatLayout shape = layout(format);
    const std::uint64_t sign = shape.sign();
    const std::uint64_t one = shape.one();
    const std::uint64_t two = one + (std::uint64_t{1} << shape.fractionBits);
    const std::uint64_t half = one - (std::uint64_t{1} << shape.fractionBits);
    const std::uint64_t infinity = shape.infinity();
    const std::uint64_t maximumFinite = infinity - 1;
    const std::uint64_t quiet = std::uint64_t{1} << (shape.fractionBits - 1);
    const std::uint64_t quietNaN = sign | infinity | quiet | 5;
    const std::uint64_t signalingNaN = infinity | 3;
    const std::uint64_t minimumNormal = std::uint64_t{1} << shape.fractionBits;
    const std::uint64_t maximumSubnormal = minimumNormal - 1;
    const std::uint64_t boundary = sqrtTwo(format);
    const std::array<std::uint64_t, 27> curated = {0,
                                                   sign,
                                                   1,
                                                   sign | 1,
                                                   maximumSubnormal,
                                                   sign | maximumSubnormal,
                                                   minimumNormal,
                                                   sign | minimumNormal,
                                                   half,
                                                   sign | half,
                                                   one - 1,
                                                   sign | (one - 1),
                                                   one,
                                                   sign | one,
                                                   one + 1,
                                                   sign | (one + 1),
                                                   two,
                                                   sign | two,
                                                   ten(format),
                                                   boundary - 1,
                                                   boundary,
                                                   boundary + 1,
                                                   maximumFinite,
                                                   sign | maximumFinite,
                                                   infinity,
                                                   sign | infinity,
                                                   quietNaN};
    for (std::uint64_t input : curated)
      appendVector(test, vectors, family, format, input,
                   input == one ||
                       (family == MathFamily::Log2 && input == two) ||
                       (family == MathFamily::Log10 && input == ten(format)));
    appendVector(test, vectors, family, format, signalingNaN);

    for (std::uint64_t encoded = 1; encoded < shape.exponentMask(); ++encoded) {
      const std::uint64_t power = encoded << shape.fractionBits;
      appendVector(test, vectors, family, format, power);
      appendVector(test, vectors, family, format, power - 1);
      appendVector(test, vectors, family, format, power + 1);
    }

    std::uint64_t state = 0x9e3779b97f4a7c15ULL ^
                          (static_cast<std::uint64_t>(familyId(family)) << 32) ^
                          shape.width();
    for (unsigned index = 0; index != 384; ++index)
      appendVector(test, vectors, family, format,
                   nextRandom(state) & widthMask(shape.width()));
  }
  return vectors;
}

void baseConversionIdentities() {
  const llvm::StringRef test = __func__;
  for (::fabric::FloatFormat format :
       {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
        ::fabric::FloatFormat::F32}) {
    const FloatLayout shape = layout(format);
    const std::uint64_t one = shape.one();
    const std::uint64_t two = one + (std::uint64_t{1} << shape.fractionBits);
    require(test, oracle(test, MathFamily::Log2, format, two) == one,
            "MPFR oracle lost log2(2) = 1");
    require(test, oracle(test, MathFamily::Log10, format, ten(format)) == one,
            "MPFR oracle lost log10(10) = 1");
    require(test,
            oracle(test, MathFamily::Log1p, format, one) ==
                oracle(test, MathFamily::Log, format, two),
            "MPFR oracle lost log1p(1) = log(2)");
  }
}

std::string hexLiteral(const llvm::APInt &value) {
  llvm::SmallString<64> digits;
  value.toStringUnsigned(digits, 16);
  const unsigned hexDigits = (value.getBitWidth() + 3) / 4;
  std::string padded(hexDigits > digits.size() ? hexDigits - digits.size() : 0,
                     '0');
  padded += digits.str();
  return std::to_string(value.getBitWidth()) + "'h" + padded;
}

std::uint64_t paddedInput(const NumericVector &vector) {
  const unsigned width = layout(vector.format).width();
  return vector.input | (0xa5a55a5a5a55aa5ULL & ~widthMask(width));
}

std::string makeTestbench(MathFamily family) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << "module testbench;\n"
         << R"sv(  logic [63:0] data_input;
  logic [5:0] mode_select;
  logic [63:0] result;

)sv"
         << "  " << moduleName(family) << " dut(\n"
         << "      .data_input_0(data_input), .config_0(mode_select),\n"
         << "      .data_output_0(result));\n\n"
         << R"sv(  function automatic [63:0] ordered_key(
      input logic [63:0] bits,
      input integer width);
    reg [63:0] sign;
    reg [63:0] magnitude;
    begin
      sign = 64'd1 << (width - 1);
      magnitude = bits & (sign - 1);
      ordered_key = (bits & sign) != 0 ? sign - magnitude : sign + magnitude;
    end
  endfunction

  task automatic check_result(
      input logic [5:0] mode,
      input integer width,
      input integer exponent_bits,
      input integer fraction_bits,
      input logic [63:0] operand,
      input logic [63:0] expected,
      input integer expected_kind);
    reg [63:0] mask;
    reg [63:0] actual;
    reg [63:0] exponent_mask;
    reg [63:0] fraction_mask;
    reg [63:0] actual_key;
    reg [63:0] expected_key;
    reg [63:0] distance;
    begin
      mask = (64'd1 << width) - 1;
      exponent_mask = ((64'd1 << exponent_bits) - 1) << fraction_bits;
      fraction_mask = (64'd1 << fraction_bits) - 1;
      mode_select = mode;
      data_input = operand;
      #1;
      actual = result & mask;
      if ((result & ~mask) !== 0)
        $fatal(1, "logarithm output padding is nonzero");
      if (expected_kind == 1) begin
        if (actual !== expected)
          $fatal(1, "exact logarithm mismatch width=%0d input=%h got=%h expected=%h",
                 width, operand, actual, expected);
      end else if (expected_kind == 2) begin
        if ((actual & exponent_mask) !== exponent_mask ||
            (actual & fraction_mask) == 0 ||
            (actual & (64'd1 << (fraction_bits - 1))) == 0)
          $fatal(1, "logarithm domain result is not a quiet NaN");
      end else begin
        actual_key = ordered_key(actual, width);
        expected_key = ordered_key(expected, width);
        distance = actual_key >= expected_key ? actual_key - expected_key
                                              : expected_key - actual_key;
        if (distance > 4)
          $fatal(1, "logarithm exceeds four ULP width=%0d input=%h got=%h expected=%h distance=%0d",
                 width, operand, actual, expected, distance);
      end
    end
  endtask

  initial begin
)sv";
  for (const NumericVector &vector : numericVectors(family)) {
    const FloatLayout shape = layout(vector.format);
    output << "    check_result(6'd" << unsigned(physicalCode(vector.format))
           << ", " << shape.width() << ", " << shape.exponentBits << ", "
           << shape.fractionBits << ", "
           << hexLiteral(llvm::APInt(64, paddedInput(vector))) << ", "
           << hexLiteral(llvm::APInt(64, vector.expected)) << ", "
           << static_cast<unsigned>(vector.kind) << ");\n";
  }
  output << R"sv(    $finish;
  end
endmodule
)sv";
  return output.str();
}

std::string makeYosysScript(MathFamily family) {
  const std::string top = moduleName(family);
  const std::string source = top + ".sv";
  std::string script;
  llvm::raw_string_ostream output(script);
  output << "read_verilog -sv " << source << '\n'
         << "hierarchy -check -top " << top << '\n'
         << "proc\nopt\ncheck -assert\n"
         << "select -assert-none " << top << "/t:$*ff* " << top
         << "/t:$*latch* " << top << "/t:$_*FF* " << top << "/t:$_*LATCH* "
         << top << "/t:$mem* " << top << "/m:*\n"
         << "synth -noabc -top " << top << '\n'
         << "check -assert\n"
         << "select -assert-none " << top << "/t:$*ff* " << top
         << "/t:$*latch* " << top << "/t:$_*FF* " << top << "/t:$_*LATCH* "
         << top << "/t:$mem* " << top << "/m:*\n"
         << "stat\n";
  return output.str();
}

std::string emitFamily(llvm::StringRef test, const ArtifactStore &store,
                       MathFamily family) {
  FabricFixture fabric = makeFabric(test, store, family);
  const auto &resolved = capability(test, fabric);
  require(test,
          resolved.implementationFamily == familyId(family) &&
              resolved.enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{schemaId(family)},
          "logarithm provider escaped its generated family descriptor");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 3,
          "logarithm provider did not consume its sealed format domain");
  for (const auto &point : relation.finiteBehaviorDomain()) {
    const auto *payload = std::get_if<::dataflow::SpecialMathPayload>(
        &point.representativeActor.payload);
    require(test,
            point.representativeActor.schema == schemaId(family) && payload &&
                payload->accuracy == loom::SpecialMathAccuracyTier::Max4Ulp &&
                point.semanticConfiguration.has_value(),
            "logarithm relation contains a malformed behavior witness");
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
                  mlir::IntegerType::get(firstContext.get(), 6) &&
              ports.atOutput(0).getName() == "data_output_0",
          "logarithm leaf ports do not follow ConfigurationABI 3.0");
  const std::string first =
      specialize(test, std::move(firstSkeleton), fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture secondSkeleton =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string second =
      specialize(test, std::move(secondSkeleton), fabric, abi);
  require(test, first == second,
          "identical logarithm inputs produced different SystemVerilog");
  const llvm::StringRef rtl(first);
  const std::string expectedFunction =
      "loom_math_" + shortName(family).str() + "_e8_f23";
  require(test,
          rtl.contains("function automatic") && rtl.contains("config_0") &&
              rtl.contains(expectedFunction) && !rtl.contains("shortreal") &&
              !rtl.contains(" real") && !rtl.contains("DPI") &&
              !rtl.contains("$ln") && !rtl.contains("$log10") &&
              !rtl.contains("$pow") && !rtl.contains("DW_") &&
              !rtl.contains("xpm_") && !rtl.contains("altera_"),
          "logarithm RTL is incomplete or violates the portable recipe");
  return first;
}

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  std::vector<loom::hardware::test::PortableProviderArtifact> artifacts;
  for (MathFamily family : {MathFamily::Log, MathFamily::Log2,
                            MathFamily::Log10, MathFamily::Log1p}) {
    const auto entry = llvm::find_if(coverage, [&](const auto &candidate) {
      return candidate.implementationFamily == familyId(family);
    });
    require(test,
            entry != coverage.end() &&
                entry->recipes ==
                    std::vector<BackendRecipeKey>{
                        BackendRecipeKey::PortableSystemVerilog},
            "logarithm provider registration is incomplete");
    const std::string name = moduleName(family);
    artifacts.push_back({name + ".sv", emitFamily(test, store, family)});
    artifacts.push_back(
        {shortName(family).str() + "_testbench.sv", makeTestbench(family)});
    artifacts.push_back({"portable_" + name + ".ys", makeYosysScript(family)});
  }
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated", artifacts))
    fail(test, llvm::toString(std::move(error)));
}

void singletonRelationsNeedNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  for (MathFamily family : {MathFamily::Log, MathFamily::Log2,
                            MathFamily::Log10, MathFamily::Log1p}) {
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
            "singleton logarithm relation retained a configuration authority");
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
    require(test, skeleton.leaf.getPortList().size() == 2,
            "singleton logarithm leaf retained a selector port");
    const std::string rtl = specialize(test, std::move(skeleton), fabric, abi);
    const std::string expectedFunction =
        "loom_math_" + shortName(family).str() + "_e8_f23";
    require(test,
            llvm::StringRef(rtl).contains(expectedFunction) &&
                !llvm::StringRef(rtl).contains("config_0"),
            "singleton logarithm provider did not emit its sealed witness");
  }
}

void expectTypedUnsupported(
    llvm::StringRef test, llvm::Expected<FabricOperationProviderOutput> result,
    ::fabric::ImplementationFamilyId family, BackendRecipeKey recipe) {
  require(test, !result, "unsupported logarithm combination was accepted");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified =
            error.implementationFamily() == family && error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unsupported logarithm provider returned the wrong error: " +
                       error.message());
      });
  require(test, classified,
          "logarithm provider lost its typed Unsupported classification");
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
           llvm::is_contained(entry.recipes,
                              BackendRecipeKey::PortableSystemVerilog);
  });
}

void registrationIsTransactional() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registry.add({::fabric::ImplementationFamilyId::ScalarMathLog1p,
                        BackendRecipeKey::PortableSystemVerilog,
                        {},
                        dummyProvider}))
    fail(test, llvm::toString(std::move(error)));
  llvm::Error error = registerPortableMathLogarithmProviders(registry);
  require(test, static_cast<bool>(error),
          "package registration accepted a conflicting log1p provider");
  llvm::consumeError(std::move(error));
  for (MathFamily family :
       {MathFamily::Log, MathFamily::Log2, MathFamily::Log10})
    require(test, !hasPortableCoverage(registry.coverage(), familyId(family)),
            "failed package registration partially added a provider");
}

void malformedInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  for (MathFamily family : {MathFamily::Log, MathFamily::Log2,
                            MathFamily::Log10, MathFamily::Log1p}) {
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
            "malformed logarithm input partially mutated the skeleton");

    for (AbiKind kind : {AbiKind::MissingBehavior, AbiKind::ExtraBehavior,
                         AbiKind::DirectEncoding})
      expectError(
          test,
          finalizeConfigurationABI(makeConfigurationAbiDraft(test, valid, kind),
                                   store),
          kind == AbiKind::DirectEncoding ? "finite codebook" : "semantic");

    for (FixtureKind kind :
         {FixtureKind::UnsupportedContract, FixtureKind::UnsupportedFormat,
          FixtureKind::UnsupportedAccuracy}) {
      FabricFixture unsupported = makeFabric(test, store, family, kind);
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
              "Unsupported logarithm provider mutated the caller skeleton");
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
  baseConversionIdentities();
  configuredBehaviorAndArtifacts(root);
  singletonRelationsNeedNoSelector(root / "singleton");
  malformedInputsAreTransactional(root / "malformed");
  return EXIT_SUCCESS;
}

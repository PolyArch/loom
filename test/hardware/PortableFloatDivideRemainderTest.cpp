#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FloatDivideRemainder.h"
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

enum class FloatFamily { Divide, Remainder };
enum class FixtureKind { Configured, Singleton, UnsupportedContract };
enum class AbiKind { Complete, MissingBehavior, ExtraBehavior, DirectEncoding };

struct FabricFixture final {
  FloatFamily family;
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
  FloatFamily family;
  ::fabric::FloatFormat format;
  mlir::arith::RoundingMode rounding;
  std::uint64_t lhs;
  std::uint64_t rhs;
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
    fail(test, "accepted malformed floating divide/remainder input");
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

::fabric::ImplementationFamilyId familyId(FloatFamily family) {
  return family == FloatFamily::Divide
             ? ::fabric::ImplementationFamilyId::ScalarFloatDivide
             : ::fabric::ImplementationFamilyId::ScalarFloatRemainder;
}

::dataflow::OperationSchemaId schemaId(FloatFamily family) {
  return family == FloatFamily::Divide
             ? ::dataflow::OperationSchemaId::ArithDivF
             : ::dataflow::OperationSchemaId::ArithRemF;
}

llvm::StringRef familyKeyword(FloatFamily family) {
  return family == FloatFamily::Divide ? "ScalarFloatDivide"
                                       : "ScalarFloatRemainder";
}

llvm::StringRef schemaKeyword(FloatFamily family) {
  return family == FloatFamily::Divide ? "arith.divf" : "arith.remf";
}

llvm::StringRef moduleName(FloatFamily family) {
  return family == FloatFamily::Divide ? "scalar_float_divide"
                                       : "scalar_float_remainder";
}

std::string fabricSource(FloatFamily family, FixtureKind kind,
                         llvm::StringRef subnormal = "preserve",
                         std::optional<llvm::StringRef> schema = std::nullopt,
                         bool orphanRounding = false) {
  const bool singleton = kind == FixtureKind::Singleton;
  const llvm::StringRef formats =
      singleton ? R"mlir(["f32"])mlir"
                : R"mlir(["f16", "bf16", "f32", "f64"])mlir";
  const llvm::StringRef roundings =
      family == FloatFamily::Divide && !singleton
          ? R"mlir(["to_nearest_even", "downward", "upward", "toward_zero", "to_nearest_away"])mlir"
      : orphanRounding ? R"mlir(["to_nearest_even", "upward"])mlir"
                       : R"mlir(["to_nearest_even"])mlir";

  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @" << moduleName(family)
         << "(%a: !fabric.bits<64>, %b: !fabric.bits<64>) -> "
            "!fabric.bits<64> { %pe = fabric.pe [spatial]"
            "(%pa = %a : !fabric.bits<64>, %pb = %b : !fabric.bits<64>) -> "
            "!fabric.bits<64> { %fu = fabric.fu"
            "(%fa = %pa : !fabric.bits<64>, %fb = %pb : !fabric.bits<64>) -> "
            "!fabric.bits<64> { %value = fabric.op [@"
         << schema.value_or(schemaKeyword(family))
         << "] (%fa, %fb) {implementation_family = "
         << "#fabric.implementation_family<" << familyKeyword(family)
         << ">, hw_params = {float_formats = " << formats
         << ", behavior = {rounding_modes = " << roundings
         << ", nan_behaviors = [\"ieee\"], subnormal_behaviors = [\""
         << subnormal
         << "\"], signed_zero_behaviors = [\"preserve\"], fastmath = "
            "\"none\"}}} : (!fabric.bits<64>, !fabric.bits<64>) -> "
            "!fabric.bits<64> fabric.yield %value : !fabric.bits<64> } } "
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
                         FloatFamily family,
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
              "System has no physical floating operation occurrence");
      return FabricFixture{family, std::move(fabric), occurrence,
                           std::move(system), physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no floating operation occurrence");
}

void expectFabricRejected(llvm::StringRef test, const ArtifactStore &store,
                          FloatFamily family, llvm::StringRef subnormal,
                          llvm::StringRef schema, bool orphanRounding,
                          llvm::StringRef expected) {
  const std::string sourceText = fabricSource(
      family, FixtureKind::Configured, subnormal, schema, orphanRounding);
  mlir::ParserConfig parserConfig(&fabricContext(), false);
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, parserConfig);
  require(test, static_cast<bool>(source),
          "negative Fabric fixture did not parse");
  std::vector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      &fabricContext(), [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return mlir::success();
      });
  if (mlir::failed(mlir::verify(*source))) {
    require(
        test,
        llvm::any_of(diagnostics,
                     [&](const std::string &diagnostic) {
                       return llvm::StringRef(diagnostic).contains(expected);
                     }),
        diagnostics.empty() ? "Fabric verifier produced no diagnostic"
                            : diagnostics.front());
    return;
  }
  attachContract(test, *source, FixtureKind::Configured);
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  expectError(test, loom::fabric::finalizeFabricRoot(root, store), expected);
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

mlir::arith::RoundingMode behaviorRounding(
    llvm::StringRef test,
    const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  const auto *payload = std::get_if<::dataflow::FloatingPointPayload>(
      &point.representativeActor.payload);
  require(test, payload != nullptr,
          "Fabric projected an operation without floating behavior");
  return payload->roundingMode.value_or(
      mlir::arith::RoundingMode::to_nearest_even);
}

std::uint8_t physicalCode(FloatFamily family, ::fabric::FloatFormat format,
                          mlir::arith::RoundingMode rounding) {
  const unsigned ordinal =
      family == FloatFamily::Divide
          ? 5 * static_cast<unsigned>(format) + static_cast<unsigned>(rounding)
          : static_cast<unsigned>(format);
  return static_cast<std::uint8_t>(
      family == FloatFamily::Divide ? 1 + 3 * ordinal : 2 + 11 * ordinal);
}

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "floating fixture has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "floating relation is not finite");
  const auto &domain = relation.finiteBehaviorDomain();
  const std::size_t expectedDomain =
      fixture.family == FloatFamily::Divide ? 20 : 4;
  require(test, domain.size() == expectedDomain,
          "floating relation has the wrong sealed domain size");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured floating behavior has no semantic value");
    const auto format = behaviorFormat(test, point);
    const auto rounding = behaviorRounding(test, point);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (format == ::fabric::FloatFormat::F32 &&
        rounding == mlir::arith::RoundingMode::to_nearest_even)
      inactive = semantic;
    if (kind == AbiKind::MissingBehavior &&
        format == ::fabric::FloatFormat::BF16 &&
        rounding == mlir::arith::RoundingMode::to_nearest_even)
      semantic = {0xfd};
    entries.push_back({std::move(semantic),
                       {physicalCode(fixture.family, format, rounding)}});
  }
  if (kind == AbiKind::ExtraBehavior)
    entries.push_back({{0xfe}, {0x3f}});
  require(test, !inactive.empty(), "floating domain has no inactive behavior");
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
            "configured floating leaf has no selector");
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
          registerPortableFloatDivideRemainderProviders(registry))
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
          "portable floating provider emitted external state");
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

llvm::RoundingMode llvmRounding(mlir::arith::RoundingMode rounding) {
  using Mode = mlir::arith::RoundingMode;
  switch (rounding) {
  case Mode::to_nearest_even:
    return llvm::RoundingMode::NearestTiesToEven;
  case Mode::downward:
    return llvm::RoundingMode::TowardNegative;
  case Mode::upward:
    return llvm::RoundingMode::TowardPositive;
  case Mode::toward_zero:
    return llvm::RoundingMode::TowardZero;
  case Mode::to_nearest_away:
    return llvm::RoundingMode::NearestTiesToAway;
  }
  llvm_unreachable("unknown floating rounding mode");
}

std::uint64_t widthMask(unsigned width) {
  return width == 64 ? UINT64_MAX : (std::uint64_t{1} << width) - 1;
}

llvm::APFloat floating(::fabric::FloatFormat format, std::uint64_t bits) {
  return llvm::APFloat(semantics(format),
                       llvm::APInt(layout(format).width(), bits));
}

std::uint64_t evaluate(FloatFamily family, ::fabric::FloatFormat format,
                       mlir::arith::RoundingMode rounding, std::uint64_t lhs,
                       std::uint64_t rhs) {
  llvm::APFloat result = floating(format, lhs);
  const llvm::APFloat right = floating(format, rhs);
  if (family == FloatFamily::Divide)
    (void)result.divide(right, llvmRounding(rounding));
  else
    (void)result.mod(right);
  return result.bitcastToAPInt().getZExtValue();
}

std::uint64_t nextRandom(std::uint64_t &state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>
roundingSensitiveInputs(llvm::StringRef test, ::fabric::FloatFormat format) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> result;
  const unsigned width = layout(format).width();
  const std::uint64_t mask = widthMask(width);
  std::uint64_t state = 0x9e3779b97f4a7c15ULL ^ width;
  for (unsigned attempt = 0; attempt < 200000 && result.size() < 8; ++attempt) {
    const std::uint64_t lhs = nextRandom(state) & mask;
    const std::uint64_t rhs = nextRandom(state) & mask;
    const llvm::APFloat left = floating(format, lhs);
    const llvm::APFloat right = floating(format, rhs);
    if (!left.isFinite() || !right.isFinite() || left.isZero() ||
        right.isZero())
      continue;
    const std::uint64_t nearest =
        evaluate(FloatFamily::Divide, format,
                 mlir::arith::RoundingMode::to_nearest_even, lhs, rhs);
    const std::uint64_t upward =
        evaluate(FloatFamily::Divide, format, mlir::arith::RoundingMode::upward,
                 lhs, rhs);
    const std::uint64_t downward =
        evaluate(FloatFamily::Divide, format,
                 mlir::arith::RoundingMode::downward, lhs, rhs);
    if (nearest == upward && nearest == downward)
      continue;
    result.emplace_back(lhs, rhs);
  }
  require(test, result.size() == 8,
          "could not find deterministic rounding-sensitive quotients");
  return result;
}

std::vector<NumericVector> numericVectors() {
  std::vector<NumericVector> result;
  constexpr std::array roundings = {mlir::arith::RoundingMode::to_nearest_even,
                                    mlir::arith::RoundingMode::downward,
                                    mlir::arith::RoundingMode::upward,
                                    mlir::arith::RoundingMode::toward_zero,
                                    mlir::arith::RoundingMode::to_nearest_away};
  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain) {
    const FloatLayout shape = layout(format);
    const std::uint64_t sign = shape.sign();
    const std::uint64_t one = shape.one();
    const std::uint64_t two = one + (std::uint64_t{1} << shape.fractionBits);
    const std::uint64_t half = one - (std::uint64_t{1} << shape.fractionBits);
    const std::uint64_t infinity = shape.infinity();
    const std::uint64_t maximumFinite = infinity - 1;
    const std::uint64_t quiet = std::uint64_t{1} << (shape.fractionBits - 1);
    const std::uint64_t quietNaN = sign | infinity | quiet | 0x5;
    const std::uint64_t signalingNaN = infinity | 0x3;
    const std::uint64_t minimumNormal = std::uint64_t{1} << shape.fractionBits;
    const std::uint64_t maximumSubnormal = minimumNormal - 1;
    const std::uint64_t fiveAndHalf =
        (std::uint64_t((std::uint64_t{1} << (shape.exponentBits - 1)) + 1)
         << shape.fractionBits) |
        (std::uint64_t{3} << (shape.fractionBits - 3));
    const auto sensitive = roundingSensitiveInputs("numericVectors", format);
    for (mlir::arith::RoundingMode rounding : roundings) {
      const std::array<std::pair<std::uint64_t, std::uint64_t>, 18> curated = {
          std::pair{one, two},
          std::pair{sign | one, two},
          std::pair{std::uint64_t{0}, sign | one},
          std::pair{sign, sign | one},
          std::pair{one, std::uint64_t{0}},
          std::pair{sign | one, std::uint64_t{0}},
          std::pair{std::uint64_t{0}, std::uint64_t{0}},
          std::pair{infinity, infinity},
          std::pair{infinity, sign | one},
          std::pair{one, sign | infinity},
          std::pair{quietNaN, one},
          std::pair{one, signalingNaN},
          std::pair{quietNaN, signalingNaN},
          std::pair{std::uint64_t{1}, two},
          std::pair{minimumNormal, maximumFinite},
          std::pair{minimumNormal, maximumSubnormal},
          std::pair{maximumFinite, minimumNormal},
          std::pair{maximumFinite, one + 1}};
      for (const auto &[lhs, rhs] : curated)
        result.push_back(
            {FloatFamily::Divide, format, rounding, lhs, rhs,
             evaluate(FloatFamily::Divide, format, rounding, lhs, rhs)});
      for (const auto &[lhs, rhs] : sensitive)
        result.push_back(
            {FloatFamily::Divide, format, rounding, lhs, rhs,
             evaluate(FloatFamily::Divide, format, rounding, lhs, rhs)});
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> remainderInputs = {
        {fiveAndHalf, two},
        {sign | fiveAndHalf, two},
        {fiveAndHalf, sign | two},
        {sign | two, two},
        {std::uint64_t{0}, two},
        {sign, two},
        {one, infinity},
        {infinity, one},
        {one, std::uint64_t{0}},
        {quietNaN, one},
        {one, signalingNaN},
        {quietNaN, signalingNaN},
        {std::uint64_t{1}, minimumNormal},
        {minimumNormal, std::uint64_t{1}},
        {maximumFinite, maximumSubnormal},
        {maximumFinite, std::uint64_t{3}},
        {one + 1, std::uint64_t{2}},
        {std::uint64_t{3}, std::uint64_t{2}},
        {half, maximumFinite},
        {sign | maximumFinite, minimumNormal}};
    std::uint64_t state = 0xd1b54a32d192ed03ULL ^ shape.width();
    while (remainderInputs.size() < 34) {
      const std::uint64_t lhs = nextRandom(state) & widthMask(shape.width());
      const std::uint64_t rhs = nextRandom(state) & widthMask(shape.width());
      const llvm::APFloat left = floating(format, lhs);
      const llvm::APFloat right = floating(format, rhs);
      if (!left.isFinite() || !right.isFinite() || right.isZero())
        continue;
      remainderInputs.emplace_back(lhs, rhs);
    }
    for (const auto &[lhs, rhs] : remainderInputs)
      result.push_back(
          {FloatFamily::Remainder, format,
           mlir::arith::RoundingMode::to_nearest_even, lhs, rhs,
           evaluate(FloatFamily::Remainder, format,
                    mlir::arith::RoundingMode::to_nearest_even, lhs, rhs)});
  }
  return result;
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

std::uint64_t paddedInput(const NumericVector &vector, std::uint64_t value) {
  const unsigned width = layout(vector.format).width();
  if (width == 64)
    return value;
  return value | (0xa5a55a5a5a55aa5ULL & ~widthMask(width));
}

std::string makeTestbench(FloatFamily family) {
  std::string text;
  llvm::raw_string_ostream output(text);
  const llvm::StringRef familyName =
      family == FloatFamily::Divide ? "divide" : "remainder";
  output << "module testbench;\n"
         << R"sv(  logic [63:0] lhs;
  logic [63:0] rhs;
  logic [5:0] mode_select;
  logic [63:0] result;

)sv"
         << "  scalar_float_" << familyName << " dut(\n"
         << "      .data_input_0(lhs), .data_input_1(rhs), "
            ".config_0(mode_select),\n"
         << "      .data_output_0(result));\n\n"
         << R"sv(  task automatic check_result(
      input logic [5:0] mode,
      input logic [63:0] left,
      input logic [63:0] right,
      input logic [63:0] expected);
    begin
      mode_select = mode;
      lhs = left;
      rhs = right;
      #1;
      if (result !== expected)
)sv"
         << "        $fatal(1, \"" << familyName
         << " mismatch mode=%0d lhs=%h rhs=%h got=%h expected=%h\",\n"
         << "               mode, left, right, result, expected);\n"
         << R"sv(    end
  endtask

  initial begin
)sv";

  for (const NumericVector &vector : numericVectors()) {
    if (vector.family != family)
      continue;
    output << "    check_result(6'd"
           << unsigned(physicalCode(family, vector.format, vector.rounding))
           << ", "
           << hexLiteral(llvm::APInt(64, paddedInput(vector, vector.lhs)))
           << ", "
           << hexLiteral(llvm::APInt(64, paddedInput(vector, vector.rhs)))
           << ", " << hexLiteral(llvm::APInt(64, vector.expected)) << ");\n";
  }

  if (family == FloatFamily::Divide)
    output
        << R"sv(    check_result(6'd0, 64'hffffffff40c00000, 64'hffffffff40000000,
                 64'h0000000040400000);
)sv";
  else
    output
        << R"sv(    check_result(6'd0, 64'hffffffff40b00000, 64'hffffffff40000000,
                 64'h000000003fc00000);
)sv";
  output << R"sv(
    $finish;
  end
endmodule
)sv";
  return output.str();
}

std::string makeYosysScript(llvm::StringRef top, llvm::StringRef source) {
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
                       FloatFamily family) {
  FabricFixture fabric = makeFabric(test, store, family);
  const auto &resolved = capability(test, fabric);
  require(test,
          resolved.implementationFamily == familyId(family) &&
              resolved.enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{schemaId(family)},
          "floating provider escaped its generated family descriptor");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  const std::size_t expectedDomain = family == FloatFamily::Divide ? 20 : 4;
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == expectedDomain,
          "floating provider did not consume its sealed behavior domain");
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test,
            point.representativeActor.schema == schemaId(family) &&
                point.semanticConfiguration.has_value(),
            "floating relation contains a malformed behavior witness");
    if (family == FloatFamily::Remainder)
      require(test,
              behaviorRounding(test, point) ==
                  mlir::arith::RoundingMode::to_nearest_even,
              "floating remainder acquired a rounding selector");
  }

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture firstSkeleton =
      makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(firstSkeleton.leaf.getPortList());
  require(test,
          ports.size() == 4 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atInput(2).getName() == "config_0" &&
              ports.atInput(2).type ==
                  mlir::IntegerType::get(firstContext.get(), 6) &&
              ports.atOutput(0).getName() == "data_output_0",
          "floating leaf ports do not follow ConfigurationABI 3.0");
  const std::string first =
      specialize(test, std::move(firstSkeleton), fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture secondSkeleton =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string second =
      specialize(test, std::move(secondSkeleton), fabric, abi);
  require(test, first == second,
          "identical floating inputs produced different SystemVerilog");
  const llvm::StringRef rtl(first);
  const llvm::StringRef core = family == FloatFamily::Divide
                                   ? "loom_float_divide_e11_f52_core"
                                   : "loom_float_remainder_core";
  require(test,
          rtl.contains("function automatic") && rtl.contains("config_0") &&
              rtl.contains(core) && !rtl.contains("shortreal") &&
              !rtl.contains(" real") && !rtl.contains("DPI"),
          "floating RTL is incomplete or not synthesizable bit logic");
  if (family == FloatFamily::Divide) {
    for (llvm::StringRef divideCore :
         {"loom_float_divide_e5_f10_core(", "loom_float_divide_e8_f7_core(",
          "loom_float_divide_e8_f23_core(", "loom_float_divide_e11_f52_core("})
      require(test, rtl.count(divideCore) == 2,
              "floating divide evaluates a format core more than once");
  } else {
    require(test,
            rtl.count("loom_float_remainder_core(") == 2 &&
                rtl.contains("for (index = 0; index < 11;") &&
                !rtl.contains("for (index = 0; index < 2046;") &&
                rtl.contains("multiply_index = multiply_index - 4") &&
                !rtl.contains("multiply_index = multiply_index - 1") &&
                !rtl.contains('%'),
            "floating remainder uses an unbounded or divider-based scale");
  }
  return first;
}

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  for (FloatFamily family : {FloatFamily::Divide, FloatFamily::Remainder}) {
    const auto entry = llvm::find_if(coverage, [&](const auto &candidate) {
      return candidate.implementationFamily == familyId(family);
    });
    require(test,
            entry != coverage.end() &&
                entry->recipes ==
                    std::vector<BackendRecipeKey>{
                        BackendRecipeKey::PortableSystemVerilog},
            "floating provider registration is incomplete");
  }

  const std::string divide = emitFamily(test, store, FloatFamily::Divide);
  const std::string remainder = emitFamily(test, store, FloatFamily::Remainder);
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated",
          {{"scalar_float_divide.sv", divide},
           {"scalar_float_remainder.sv", remainder},
           {"divide_testbench.sv", makeTestbench(FloatFamily::Divide)},
           {"remainder_testbench.sv", makeTestbench(FloatFamily::Remainder)},
           {"portable_scalar_float_divide.ys",
            makeYosysScript("scalar_float_divide", "scalar_float_divide.sv")},
           {"portable_scalar_float_remainder.ys",
            makeYosysScript("scalar_float_remainder",
                            "scalar_float_remainder.sv")}}))
    fail(test, llvm::toString(std::move(error)));
}

void singletonRelationsNeedNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  for (FloatFamily family : {FloatFamily::Divide, FloatFamily::Remainder}) {
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
            "singleton floating relation retained a configuration authority");
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
    require(test, skeleton.leaf.getPortList().size() == 3,
            "singleton floating leaf retained a selector port");
    const std::string rtl = specialize(test, std::move(skeleton), fabric, abi);
    const llvm::StringRef expected = family == FloatFamily::Divide
                                         ? "loom_float_divide_e8_f23_core"
                                         : "loom_float_remainder_core";
    require(test,
            llvm::StringRef(rtl).contains(expected) &&
                !llvm::StringRef(rtl).contains("config_0"),
            "singleton floating provider did not emit its sealed witness");
  }
}

void expectTypedUnsupported(
    llvm::StringRef test, llvm::Expected<FabricOperationProviderOutput> result,
    ::fabric::ImplementationFamilyId family, BackendRecipeKey recipe) {
  require(test, !result, "unsupported floating combination was accepted");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified =
            error.implementationFamily() == family && error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unsupported floating provider returned the wrong error: " +
                       error.message());
      });
  require(test, classified,
          "floating provider lost its typed Unsupported classification");
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
          registry.add({::fabric::ImplementationFamilyId::ScalarFloatRemainder,
                        BackendRecipeKey::PortableSystemVerilog,
                        {},
                        dummyProvider}))
    fail(test, llvm::toString(std::move(error)));
  require(
      test,
      !hasPortableCoverage(registry.coverage(),
                           ::fabric::ImplementationFamilyId::ScalarFloatDivide),
      "registration fixture unexpectedly contains divide");
  llvm::Error error = registerPortableFloatDivideRemainderProviders(registry);
  require(test, static_cast<bool>(error),
          "package registration accepted a conflicting remainder provider");
  llvm::consumeError(std::move(error));
  require(
      test,
      !hasPortableCoverage(registry.coverage(),
                           ::fabric::ImplementationFamilyId::ScalarFloatDivide),
      "failed package registration partially added divide");
}

void malformedInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  for (FloatFamily family : {FloatFamily::Divide, FloatFamily::Remainder}) {
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
            "malformed floating input partially mutated the skeleton");

    for (AbiKind kind : {AbiKind::MissingBehavior, AbiKind::ExtraBehavior,
                         AbiKind::DirectEncoding})
      expectError(
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
            "Unsupported floating provider mutated the caller skeleton");

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

    const llvm::StringRef foreign =
        family == FloatFamily::Divide ? "arith.remf" : "arith.divf";
    expectFabricRejected(test, store, family, "flush_to_zero",
                         schemaKeyword(family), false, "subnormal");
    expectFabricRejected(test, store, family, "preserve", foreign, false,
                         "not admitted");
    if (family == FloatFamily::Remainder)
      expectFabricRejected(test, store, family, "preserve",
                           schemaKeyword(family), true, "rounding");
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  registrationIsTransactional();
  configuredBehaviorAndArtifacts(root);
  singletonRelationsNeedNoSelector(root / "singleton");
  malformedInputsAreTransactional(root / "malformed");
  return EXIT_SUCCESS;
}

#include "ConfigurationABI3TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FloatMultiply.h"
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

enum class MultiplyFamily { Scalar, FixedVector };

enum class FixtureKind { Configured, Singleton, UnsupportedContract };

enum class AbiKind {
  Complete,
  MissingBehavior,
  ExtraBehavior,
  DirectEncoding,
};

struct FabricFixture final {
  MultiplyFamily family;
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

struct ScalarVector final {
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
    fail(test, "accepted malformed float multiply input");
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

::fabric::ImplementationFamilyId familyId(MultiplyFamily family) {
  return family == MultiplyFamily::Scalar
             ? ::fabric::ImplementationFamilyId::ScalarFloatMultiply
             : ::fabric::ImplementationFamilyId::FixedVectorFloatMultiply;
}

llvm::StringRef familyKeyword(MultiplyFamily family) {
  return family == MultiplyFamily::Scalar ? "ScalarFloatMultiply"
                                          : "FixedVectorFloatMultiply";
}

llvm::StringRef moduleName(MultiplyFamily family) {
  return family == MultiplyFamily::Scalar ? "scalar_float_multiply"
                                          : "fixed_vector_float_multiply";
}

std::string fabricSource(MultiplyFamily family, FixtureKind kind,
                         llvm::StringRef subnormal = "preserve",
                         llvm::StringRef schema = "arith.mulf") {
  const bool singleton = kind == FixtureKind::Singleton;
  const unsigned portWidth = family == MultiplyFamily::Scalar ? 64 : 80;
  const llvm::StringRef formats =
      singleton ? R"mlir(["f32"])mlir"
                : R"mlir(["f16", "bf16", "f32", "f64"])mlir";
  const llvm::StringRef rounding =
      singleton
          ? R"mlir(["to_nearest_even"])mlir"
          : R"mlir(["to_nearest_even", "downward", "upward", "toward_zero", "to_nearest_away"])mlir";
  const llvm::StringRef formatField =
      family == MultiplyFamily::Scalar ? "float_formats" : "element_formats";

  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @" << moduleName(family)
         << "(%a: !fabric.bits<" << portWidth << ">, %b: !fabric.bits<"
         << portWidth << ">) -> !fabric.bits<" << portWidth
         << "> { %pe = fabric.pe [spatial]"
         << "(%pa = %a : !fabric.bits<" << portWidth
         << ">, %pb = %b : !fabric.bits<" << portWidth << ">) -> !fabric.bits<"
         << portWidth << "> { %fu = fabric.fu"
         << "(%fa = %pa : !fabric.bits<" << portWidth
         << ">, %fb = %pb : !fabric.bits<" << portWidth << ">) -> !fabric.bits<"
         << portWidth << "> { %value = fabric.op [@" << schema
         << "] (%fa, %fb) {implementation_family = "
         << "#fabric.implementation_family<" << familyKeyword(family)
         << ">, hw_params = {" << formatField << " = " << formats
         << ", behavior = {rounding_modes = " << rounding
         << ", nan_behaviors = [\"ieee\"], subnormal_behaviors = [\""
         << subnormal
         << "\"], signed_zero_behaviors = [\"preserve\"], fastmath = "
            "\"none\"}";
  if (family == MultiplyFamily::FixedVector)
    source << ", max_payload_bits = 80 : i32";
  source << "}} : (!fabric.bits<" << portWidth << ">, !fabric.bits<"
         << portWidth << ">) -> !fabric.bits<" << portWidth
         << "> fabric.yield %value : !fabric.bits<" << portWidth
         << "> } } fabric.yield %pe : !fabric.bits<" << portWidth << "> } }";
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
                         MultiplyFamily family,
                         FixtureKind kind = FixtureKind::Configured) {
  const std::string sourceText = fabricSource(family, kind);
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
              "System has no physical float multiply occurrence");
      return FabricFixture{family, std::move(fabric), occurrence,
                           std::move(system), physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no float multiply occurrence");
}

void expectFabricRejected(llvm::StringRef test, const ArtifactStore &store,
                          MultiplyFamily family, llvm::StringRef subnormal,
                          llvm::StringRef schema, llvm::StringRef expected) {
  const std::string sourceText =
      fabricSource(family, FixtureKind::Configured, subnormal, schema);
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
  if (auto vector = llvm::dyn_cast<mlir::VectorType>(type))
    type = vector.getElementType();
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
          "Fabric projected a multiply without floating behavior");
  return payload->roundingMode.value_or(
      mlir::arith::RoundingMode::to_nearest_even);
}

unsigned behaviorLaneCount(
    const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  if (auto vector = llvm::dyn_cast<mlir::VectorType>(
          point.representativeActor.type.getInput(0)))
    return static_cast<unsigned>(vector.getNumElements());
  return 1;
}

std::uint8_t modeCode(::fabric::FloatFormat format,
                      mlir::arith::RoundingMode rounding) {
  return 1 + 5 * static_cast<std::uint8_t>(format) +
         static_cast<std::uint8_t>(rounding);
}

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "float multiply fixture has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "float multiply relation is not finite");
  const auto &domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 20,
          "configured float multiply domain is not format by rounding");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured multiply behavior has no semantic value");
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
        rounding == mlir::arith::RoundingMode::downward)
      semantic = {0xfd};
    entries.push_back({std::move(semantic), {modeCode(format, rounding)}});
  }
  if (kind == AbiKind::ExtraBehavior)
    entries.push_back({{0xfe}, {0x1f}});
  require(test, !inactive.empty(), "multiply domain has no inactive behavior");
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  SemanticFieldEncoding encoding =
      kind == AbiKind::DirectEncoding
          ? SemanticFieldEncoding{DirectBitsEncoding{5}}
          : SemanticFieldEncoding{
                FiniteCodebookEncoding{5, std::move(entries)}};
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
            "configured multiply leaf has no selector");
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
  if (llvm::Error error = registerPortableFloatMultiplyProviders(registry))
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
          "portable multiply emitted external implementation state");
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
  const unsigned width = layout(format).width();
  return llvm::APFloat(semantics(format), llvm::APInt(width, bits));
}

std::uint64_t multiply(::fabric::FloatFormat format,
                       mlir::arith::RoundingMode rounding, std::uint64_t lhs,
                       std::uint64_t rhs) {
  llvm::APFloat result = floating(format, lhs);
  const llvm::APFloat right = floating(format, rhs);
  (void)result.multiply(right, llvmRounding(rounding));
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
  std::uint64_t state = 0xd1b54a32d192ed03ULL ^ width;
  for (unsigned attempt = 0; attempt < 200000 && result.size() < 8; ++attempt) {
    const std::uint64_t lhs = nextRandom(state) & mask;
    const std::uint64_t rhs = nextRandom(state) & mask;
    const llvm::APFloat left = floating(format, lhs);
    const llvm::APFloat right = floating(format, rhs);
    if (!left.isFinite() || !right.isFinite() || left.isZero() ||
        right.isZero())
      continue;
    const std::uint64_t nearest =
        multiply(format, mlir::arith::RoundingMode::to_nearest_even, lhs, rhs);
    const std::uint64_t upward =
        multiply(format, mlir::arith::RoundingMode::upward, lhs, rhs);
    const std::uint64_t downward =
        multiply(format, mlir::arith::RoundingMode::downward, lhs, rhs);
    if (nearest == upward && nearest == downward)
      continue;
    result.emplace_back(lhs, rhs);
  }
  require(test, result.size() == 8,
          "could not find deterministic rounding-sensitive products");
  return result;
}

std::vector<ScalarVector> scalarVectors() {
  std::vector<ScalarVector> result;
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
    const std::uint64_t quietNaN = infinity | quiet | 0x5;
    const std::uint64_t signalingNaN = infinity | 0x3;
    const std::uint64_t minimumNormal = std::uint64_t{1} << shape.fractionBits;
    const auto sensitive = roundingSensitiveInputs("scalarVectors", format);
    for (mlir::arith::RoundingMode rounding : roundings) {
      std::vector<std::pair<std::uint64_t, std::uint64_t>> curated = {
          std::pair{one, two},
          std::pair{sign | one, two},
          std::pair{std::uint64_t{0}, sign | one},
          std::pair{sign, sign | one},
          std::pair{std::uint64_t{0}, infinity},
          std::pair{infinity, sign | one},
          std::pair{quietNaN, one},
          std::pair{one, signalingNaN},
          std::pair{quietNaN, signalingNaN},
          std::pair{std::uint64_t{1}, one},
          std::pair{minimumNormal, half},
          std::pair{minimumNormal, one - 1},
          std::pair{maximumFinite, one},
          std::pair{maximumFinite, one + 1},
          std::pair{maximumFinite, two},
          std::pair{std::uint64_t{1}, half},
      };
      if (format == ::fabric::FloatFormat::F16) {
        curated.emplace_back(0x55b5, 0x619b);
        curated.emplace_back(0x470e, 0x7089);
      }
      for (const auto &[lhs, rhs] : curated)
        result.push_back(
            {format, rounding, lhs, rhs, multiply(format, rounding, lhs, rhs)});
      for (const auto &[lhs, rhs] : sensitive)
        result.push_back(
            {format, rounding, lhs, rhs, multiply(format, rounding, lhs, rhs)});
    }
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

std::uint64_t paddedScalarInput(const ScalarVector &vector,
                                std::uint64_t value) {
  const unsigned width = layout(vector.format).width();
  if (width == 64)
    return value;
  return value | (0xa5a55a5a5a5aa5a5ULL & ~widthMask(width));
}

std::string makeTestbench() {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << R"sv(`ifdef TEST_SCALAR
module elastic_scalar_float_multiply(
    input logic clock,
    input logic reset,
    input logic input_valid,
    output logic input_ready,
    input logic output_ready,
    output logic output_valid,
    input logic [4:0] mode,
    input logic [63:0] lhs,
    input logic [63:0] rhs,
    output logic [63:0] result);
  logic [63:0] next_result;
  scalar_float_multiply operation(
      .data_input_0(lhs), .data_input_1(rhs), .config_0(mode),
      .data_output_0(next_result));
  assign input_ready = !output_valid || output_ready;
  always_ff @(posedge clock) begin
    if (reset) begin
      output_valid <= 1'b0;
      result <= 64'd0;
    end else if (input_ready) begin
      output_valid <= input_valid;
      if (input_valid) result <= next_result;
    end
  end
endmodule
`endif

module testbench;
  logic [63:0] scalar_lhs;
  logic [63:0] scalar_rhs;
  logic [4:0] scalar_mode;
  logic [63:0] scalar_result;
  logic [79:0] vector_lhs;
  logic [79:0] vector_rhs;
  logic [4:0] vector_mode;
  logic [79:0] vector_result;
  logic clock;
  logic reset;
  logic input_valid;
  logic input_ready;
  logic output_ready;
  logic output_valid;
  logic [4:0] elastic_mode;
  logic [63:0] elastic_lhs;
  logic [63:0] elastic_rhs;
  logic [63:0] elastic_result;

`ifdef TEST_SCALAR
  scalar_float_multiply scalar_dut(
      .data_input_0(scalar_lhs), .data_input_1(scalar_rhs),
      .config_0(scalar_mode), .data_output_0(scalar_result));
`else
  fixed_vector_float_multiply vector_dut(
      .data_input_0(vector_lhs), .data_input_1(vector_rhs),
      .config_0(vector_mode), .data_output_0(vector_result));
`endif

`ifdef TEST_SCALAR
  elastic_scalar_float_multiply elastic_dut(
      .clock(clock), .reset(reset), .input_valid(input_valid),
      .input_ready(input_ready), .output_ready(output_ready),
      .output_valid(output_valid), .mode(elastic_mode), .lhs(elastic_lhs),
      .rhs(elastic_rhs), .result(elastic_result));

  initial clock = 0;
  always #5 clock = !clock;

  task automatic check_scalar(
      input logic [4:0] mode,
      input logic [63:0] lhs,
      input logic [63:0] rhs,
      input logic [63:0] expected);
    begin
      scalar_mode = mode;
      scalar_lhs = lhs;
      scalar_rhs = rhs;
      #1;
      if (scalar_result !== expected)
        $fatal(1, "scalar multiply mismatch mode=%0d lhs=%h rhs=%h got=%h expected=%h",
               mode, lhs, rhs, scalar_result, expected);
    end
  endtask

`else
  task automatic check_vector(
      input logic [4:0] mode,
      input logic [79:0] lhs,
      input logic [79:0] rhs,
      input logic [79:0] expected);
    begin
      vector_mode = mode;
      vector_lhs = lhs;
      vector_rhs = rhs;
      #1;
      if (vector_result !== expected)
        $fatal(1, "vector lane independence failed mode=%0d got=%h expected=%h",
               mode, vector_result, expected);
    end
  endtask
`endif

  initial begin
)sv";

  output << "`ifdef TEST_SCALAR\n";
  for (const ScalarVector &vector : scalarVectors()) {
    output << "    check_scalar(5'd"
           << unsigned(modeCode(vector.format, vector.rounding)) << ", "
           << hexLiteral(llvm::APInt(64, paddedScalarInput(vector, vector.lhs)))
           << ", "
           << hexLiteral(llvm::APInt(64, paddedScalarInput(vector, vector.rhs)))
           << ", " << hexLiteral(llvm::APInt(64, vector.expected)) << ");\n";
  }

  output << "`else\n";
  constexpr std::array roundings = {mlir::arith::RoundingMode::to_nearest_even,
                                    mlir::arith::RoundingMode::downward,
                                    mlir::arith::RoundingMode::upward,
                                    mlir::arith::RoundingMode::toward_zero,
                                    mlir::arith::RoundingMode::to_nearest_away};
  for (::fabric::FloatFormat format : ::fabric::floatFormatDomain) {
    const FloatLayout shape = layout(format);
    const unsigned lanes = 80 / shape.width();
    const std::uint64_t one = shape.one();
    const std::uint64_t two = one + (std::uint64_t{1} << shape.fractionBits);
    const std::uint64_t infinity = shape.infinity();
    const std::uint64_t quietNaN =
        infinity | (std::uint64_t{1} << (shape.fractionBits - 1)) | 0x5;
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 5> laneInputs = {
        std::pair{one, two}, std::pair{quietNaN, one},
        std::pair{std::uint64_t{0}, infinity}, std::pair{std::uint64_t{1}, one},
        std::pair{shape.sign() | one, two}};
    for (mlir::arith::RoundingMode rounding : roundings) {
      llvm::APInt lhs(80, 0);
      llvm::APInt rhs(80, 0);
      llvm::APInt expected(80, 0);
      for (unsigned lane = 0; lane < lanes; ++lane) {
        const auto [left, right] = laneInputs[lane % laneInputs.size()];
        const unsigned offset = lane * shape.width();
        lhs |= llvm::APInt(80, left).shl(offset);
        rhs |= llvm::APInt(80, right).shl(offset);
        expected |= llvm::APInt(80, multiply(format, rounding, left, right))
                        .shl(offset);
      }
      const unsigned activeBits = lanes * shape.width();
      if (activeBits < 80) {
        const llvm::APInt padding =
            llvm::APInt::getHighBitsSet(80, 80 - activeBits);
        lhs |= padding;
        rhs |= padding;
      }
      output << "    check_vector(5'd" << unsigned(modeCode(format, rounding))
             << ", " << hexLiteral(lhs) << ", " << hexLiteral(rhs) << ", "
             << hexLiteral(expected) << ");\n";
    }
  }
  output << "`endif\n";

  const std::uint8_t inactive = modeCode(
      ::fabric::FloatFormat::F32, mlir::arith::RoundingMode::to_nearest_even);
  (void)inactive;
  output << R"sv(
`ifdef TEST_SCALAR
    check_scalar(5'd0, 64'h000000003fc00000, 64'h0000000040000000,
                 64'h0000000040400000);

    reset = 1'b1;
    input_valid = 1'b0;
    output_ready = 1'b0;
    elastic_mode = 5'd11;
    elastic_lhs = 64'h000000003fc00000;
    elastic_rhs = 64'h0000000040000000;
    repeat (2) @(posedge clock);
    #1;
    if (output_valid) $fatal(1, "reset did not clear the elastic result slot");

    reset = 1'b0;
    input_valid = 1'b1;
    @(posedge clock);
    #1;
    if (!output_valid || elastic_result !== 64'h0000000040400000)
      $fatal(1, "elastic multiply did not publish after one cycle");

    elastic_lhs = 64'h0000000040000000;
    elastic_rhs = 64'h0000000040800000;
    #1;
    if (input_ready || !output_valid ||
        elastic_result !== 64'h0000000040400000)
      $fatal(1, "backpressure did not hold the published product stable");

    output_ready = 1'b1;
    @(posedge clock);
    #1;
    if (!output_valid || elastic_result !== 64'h0000000041000000)
      $fatal(1, "release-before-acquire replacement lost the next product");

    input_valid = 1'b0;
    output_ready = 1'b0;
    #1;
    if (input_ready || elastic_result !== 64'h0000000041000000)
      $fatal(1, "stall changed a retained replacement product");

    reset = 1'b1;
    @(posedge clock);
    #1;
    if (output_valid) $fatal(1, "reset did not discard stalled occupancy");
`endif
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
                       MultiplyFamily family) {
  FabricFixture fabric = makeFabric(test, store, family);
  const auto &resolved = capability(test, fabric);
  require(test,
          resolved.implementationFamily == familyId(family) &&
              resolved.enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{
                      ::dataflow::OperationSchemaId::ArithMulF},
          "multiply escaped its generated family descriptor");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 20,
          "multiply did not consume the exact sealed behavior product");
  std::array<unsigned, 4> formatCounts{};
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test,
            point.representativeActor.schema ==
                    ::dataflow::OperationSchemaId::ArithMulF &&
                point.semanticConfiguration.has_value(),
            "multiply relation contains a malformed behavior witness");
    const auto format = behaviorFormat(test, point);
    ++formatCounts[static_cast<std::size_t>(format)];
    const unsigned expectedLanes =
        family == MultiplyFamily::Scalar ? 1 : 80 / layout(format).width();
    require(test, behaviorLaneCount(point) == expectedLanes,
            "multiply relation changed scalar or vector geometry");
  }
  require(test,
          llvm::all_of(formatCounts, [](unsigned count) { return count == 5; }),
          "multiply relation omitted an admitted rounding mode");

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
                  mlir::IntegerType::get(firstContext.get(), 5) &&
              ports.atOutput(0).getName() == "data_output_0",
          "multiply leaf ports do not follow ConfigurationABI 3.0");
  const std::string first =
      specialize(test, std::move(firstSkeleton), fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture secondSkeleton =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string second =
      specialize(test, std::move(secondSkeleton), fabric, abi);
  require(test, first == second,
          "identical multiply inputs produced different SystemVerilog");
  const llvm::StringRef rtl(first);
  require(test,
          rtl.contains("function automatic") && rtl.contains("config_0") &&
              rtl.contains("loom_float_multiply_e5_f10_rne") &&
              rtl.contains("loom_float_multiply_e8_f7_rdn") &&
              rtl.contains("loom_float_multiply_e8_f23_rup") &&
              rtl.contains("loom_float_multiply_e11_f52_rtz") &&
              rtl.contains("loom_float_multiply_e11_f52_rna") &&
              !rtl.contains("shortreal") && !rtl.contains("real") &&
              !rtl.contains("DPI"),
          "multiply RTL is incomplete or not portable synthesizable logic");
  return first;
}

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  for (MultiplyFamily family :
       {MultiplyFamily::Scalar, MultiplyFamily::FixedVector}) {
    const auto entry = llvm::find_if(coverage, [&](const auto &candidate) {
      return candidate.implementationFamily == familyId(family);
    });
    require(test,
            entry != coverage.end() &&
                entry->recipes ==
                    std::vector<BackendRecipeKey>{
                        BackendRecipeKey::PortableSystemVerilog},
            "float multiply provider registration is incomplete");
  }

  const std::string scalar = emitFamily(test, store, MultiplyFamily::Scalar);
  const std::string vector =
      emitFamily(test, store, MultiplyFamily::FixedVector);
  const std::string testbench = makeTestbench();
  const std::string scalarYosys =
      makeYosysScript("scalar_float_multiply", "scalar_float_multiply.sv");
  const std::string vectorYosys = makeYosysScript(
      "fixed_vector_float_multiply", "fixed_vector_float_multiply.sv");
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated",
          {{"scalar_float_multiply.sv", scalar},
           {"fixed_vector_float_multiply.sv", vector},
           {"testbench.sv", testbench},
           {"portable_scalar_float_multiply.ys", scalarYosys},
           {"portable_fixed_vector_float_multiply.ys", vectorYosys}}))
    fail(test, llvm::toString(std::move(error)));
}

void singletonRelationsNeedNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  for (MultiplyFamily family :
       {MultiplyFamily::Scalar, MultiplyFamily::FixedVector}) {
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
            "singleton multiply retained a configuration authority");
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
    require(test, skeleton.leaf.getPortList().size() == 3,
            "singleton multiply leaf retained a selector port");
    const std::string rtl = specialize(test, std::move(skeleton), fabric, abi);
    require(test,
            llvm::StringRef(rtl).contains("loom_float_multiply_e8_f23_rne") &&
                !llvm::StringRef(rtl).contains("config_0"),
            "singleton multiply did not emit only its sealed witness");
  }
}

void expectTypedUnsupported(
    llvm::StringRef test, llvm::Expected<FabricOperationProviderOutput> result,
    ::fabric::ImplementationFamilyId family, BackendRecipeKey recipe) {
  require(test, !result, "unsupported multiply combination was accepted");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified =
            error.implementationFamily() == family && error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unsupported multiply returned the wrong error class: " +
                       error.message());
      });
  require(test, classified,
          "multiply lost its typed Unsupported classification");
}

void malformedInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  for (MultiplyFamily family :
       {MultiplyFamily::Scalar, MultiplyFamily::FixedVector}) {
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
            "malformed multiply input partially mutated the skeleton");

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
            "Unsupported multiply mutated the caller skeleton");

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

    expectFabricRejected(test, store, family, "flush_to_zero", "arith.mulf",
                         "subnormal");
    expectFabricRejected(test, store, family, "preserve", "arith.addf",
                         "not admitted");
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  configuredBehaviorAndArtifacts(root);
  singletonRelationsNeedNoSelector(root / "singleton");
  malformedInputsAreTransactional(root / "malformed");
  return EXIT_SUCCESS;
}

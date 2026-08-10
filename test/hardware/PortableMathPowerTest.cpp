#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/MathPower.h"
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
using loom::SpecialMathAccuracyTier;
using loom::fabric::FabricFuOccurrenceNodeRef;
using loom::fabric::FinalizedFabricRoot;
using namespace loom::hardware;
using namespace loom::hardware::rtl;

enum class FixtureKind {
  Configured,
  Singleton,
  UnsupportedContract,
  UnsupportedBehavior,
  UnsupportedF32,
  UnsupportedF64,
  CorrectlyRounded,
  Max1Ulp,
  Max2Ulp,
  MissingPhysicalInput,
  InsufficientPhysicalWidth,
};

enum class AbiKind { Complete, MissingBehavior, ExtraBehavior, DirectEncoding };

enum class ExpectedKind { Exact, NaN, Max4Ulp };

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

struct PowVector final {
  ::fabric::FloatFormat format;
  std::uint64_t base;
  std::uint64_t exponent;
  std::uint64_t reference;
  std::uint64_t lowerBound;
  std::uint64_t upperBound;
  ExpectedKind expectedKind;
};

struct AccuracyBounds final {
  std::uint64_t lower;
  std::uint64_t upper;
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
  require(test, !value, "accepted malformed math power input");
  bool classified = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &) {
        fail(test, "malformed math power input became typed Unsupported");
      },
      [&](const llvm::ErrorInfoBase &error) {
        require(test, llvm::StringRef(error.message()).contains(expected),
                error.message());
        classified = true;
      });
  require(test, classified, "malformed math power input lost its error");
}

void expectTypedUnsupported(llvm::StringRef test,
                            llvm::Expected<FabricOperationProviderOutput> value,
                            BackendRecipeKey recipe) {
  require(test, !value, "unsupported math power input was accepted");
  bool classified = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() ==
                         ::fabric::ImplementationFamilyId::ScalarMathPow &&
                     error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unsupported math power returned the wrong error class: " +
                       error.message());
      });
  require(test, classified, "math power lost typed Unsupported classification");
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

std::string fabricSource(FixtureKind kind) {
  llvm::StringRef formats = R"mlir(["f16", "bf16"])mlir";
  llvm::StringRef accuracy = "Max4Ulp";
  llvm::StringRef fastMath = "afn";
  llvm::StringRef signedZeroBehaviors = R"mlir(["preserve"])mlir";
  unsigned physicalWidth = 32;
  bool missingPhysicalInput = false;
  switch (kind) {
  case FixtureKind::Configured:
  case FixtureKind::UnsupportedContract:
    break;
  case FixtureKind::UnsupportedBehavior:
    formats = R"mlir(["f16"])mlir";
    signedZeroBehaviors = R"mlir(["ignore_sign"])mlir";
    break;
  case FixtureKind::Singleton:
    formats = R"mlir(["f16"])mlir";
    break;
  case FixtureKind::UnsupportedF32:
    formats = R"mlir(["f32"])mlir";
    break;
  case FixtureKind::UnsupportedF64:
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
  case FixtureKind::MissingPhysicalInput:
    formats = R"mlir(["f16"])mlir";
    missingPhysicalInput = true;
    break;
  case FixtureKind::InsufficientPhysicalWidth:
    formats = R"mlir(["f16"])mlir";
    physicalWidth = 8;
    break;
  }

  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @scalar_math_pow"
         << "(%base: !fabric.bits<" << physicalWidth << ">";
  if (!missingPhysicalInput)
    source << ", %exponent: !fabric.bits<" << physicalWidth << ">";
  source << ") -> !fabric.bits<" << physicalWidth << "> { "
         << "%pe = fabric.pe [spatial]"
         << "(%pbase = %base : !fabric.bits<" << physicalWidth << ">";
  if (!missingPhysicalInput)
    source << ", %pexponent = %exponent : !fabric.bits<" << physicalWidth
           << ">";
  source << ") -> !fabric.bits<" << physicalWidth << "> { "
         << "%fu = fabric.fu"
         << "(%fbase = %pbase : !fabric.bits<" << physicalWidth << ">";
  if (!missingPhysicalInput)
    source << ", %fexponent = %pexponent : !fabric.bits<" << physicalWidth
           << ">";
  source << ") -> !fabric.bits<" << physicalWidth << "> { "
         << "%value = fabric.op [@math.powf] (%fbase";
  if (!missingPhysicalInput)
    source << ", %fexponent";
  source << ") "
         << "{implementation_family = "
         << "#fabric.implementation_family<ScalarMathPow>, hw_params = "
         << "{float_formats = " << formats
         << ", behavior = {rounding_modes = [\"to_nearest_even\"], "
            "nan_behaviors = [\"ieee\"], subnormal_behaviors = "
            "[\"preserve\"], signed_zero_behaviors = "
         << signedZeroBehaviors << ", fastmath = \"" << fastMath
         << "\"}, accuracy_guarantee = \"" << accuracy
         << "\"}} : (!fabric.bits<" << physicalWidth << ">";
  if (!missingPhysicalInput)
    source << ", !fabric.bits<" << physicalWidth << ">";
  source << ") -> !fabric.bits<" << physicalWidth
         << "> fabric.yield %value : !fabric.bits<" << physicalWidth
         << "> } } fabric.yield %pe : !fabric.bits<" << physicalWidth
         << "> } }";
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

void expectFabricRejected(llvm::StringRef test, const ArtifactStore &store,
                          FixtureKind kind, llvm::StringRef expected) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(fabricSource(kind),
                                                        &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  attachContract(test, *source, kind);
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no root");
  expectInvalid(test, loom::fabric::finalizeFabricRoot(root, store), expected);
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         FixtureKind kind = FixtureKind::Configured) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(fabricSource(kind),
                                                        &fabricContext());
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
      if (capability.implementationFamily !=
          ::fabric::ImplementationFamilyId::ScalarMathPow)
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
              "System has no physical math power occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no math power occurrence");
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
  return 1 + 2 * static_cast<std::uint8_t>(format);
}

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "math power fixture has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  const auto &domain = relation.finiteBehaviorDomain();
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              domain.size() == 2,
          "configured math power domain is not the exact format set");
  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured math power behavior has no semantic key");
    const ::fabric::FloatFormat format = behaviorFormat(test, point);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (format == ::fabric::FloatFormat::F16)
      inactive = semantic;
    if (kind == AbiKind::MissingBehavior &&
        format == ::fabric::FloatFormat::BF16)
      semantic = {0xfd};
    entries.push_back({std::move(semantic), {modeCode(format)}});
  }
  if (kind == AbiKind::ExtraBehavior)
    entries.push_back({{0xfe}, {0x07}});
  require(test, !inactive.empty(), "math power domain has no inactive mode");
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
            "configured math power leaf has no selector");
    field->type = builder.getI1Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("scalar_math_pow"), ports);
  return SkeletonFixture{std::move(module), leaf};
}

FabricOperationProviderRegistry makeProviderRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableMathPowerProvider(registry))
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
          "portable math power emitted external implementation state");
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

std::uint64_t evaluate(llvm::StringRef test, ::fabric::FloatFormat format,
                       std::uint64_t base, std::uint64_t exponent) {
  mlir::Type type = floatType(format);
  ::dataflow::CanonicalActorSchemaProjection actor{
      ::dataflow::OperationSchemaId::MathPowF,
      mlir::FunctionType::get(&fabricContext(), {type, type}, {type}),
      ::dataflow::SpecialMathPayload{mlir::arith::FastMathFlags::afn,
                                     SpecialMathAccuracyTier::Max4Ulp}};
  const unsigned width = layout(format).width();
  loom::sim::PrimitiveOperationDescriptor descriptor{actor, width, width};
  loom::sim::PrimitiveValue result = take(
      test,
      loom::sim::evaluatePrimitiveOperation(
          descriptor,
          {loom::sim::PrimitiveValue::floating(floating(format, base)),
           loom::sim::PrimitiveValue::floating(floating(format, exponent))}));
  require(test, result.isDefined(),
          "independent MPFR oracle returned a non-defined value");
  return result.bits->getZExtValue();
}

std::uint64_t nextRandom(std::uint64_t &state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

ExpectedKind expectedKind(::fabric::FloatFormat format, std::uint64_t base,
                          std::uint64_t exponent, std::uint64_t reference) {
  const FloatLayout shape = layout(format);
  const llvm::APFloat result = floating(format, reference);
  if (result.isNaN())
    return ExpectedKind::NaN;
  const std::uint64_t magnitude = reference & (shape.sign() - 1);
  const bool exactBoundary = result.isInfinity() || result.isZero() ||
                             exponent == 0 || exponent == shape.sign() ||
                             exponent == shape.one() || base == shape.one() ||
                             base == (shape.sign() | shape.one());
  if (exactBoundary || magnitude == shape.one())
    return ExpectedKind::Exact;
  return ExpectedKind::Max4Ulp;
}

std::uint64_t orderedKey(::fabric::FloatFormat format, std::uint64_t bits) {
  const std::uint64_t sign = layout(format).sign();
  const std::uint64_t magnitude = bits & (sign - 1);
  return (bits & sign) != 0 ? sign - magnitude : sign + magnitude;
}

std::uint64_t bitsFromOrderedKey(::fabric::FloatFormat format,
                                 std::uint64_t key) {
  const std::uint64_t sign = layout(format).sign();
  return key < sign ? sign | (sign - key) : key - sign;
}

bool canonicallyConforms(llvm::StringRef test, ::fabric::FloatFormat format,
                         const llvm::APFloat &reference, std::uint64_t key) {
  const llvm::APFloat candidate =
      floating(format, bitsFromOrderedKey(format, key));
  auto conforms = loom::specialMathAccuracyConforms(
      SpecialMathAccuracyTier::Max4Ulp, reference, candidate);
  if (!conforms) {
    const std::string message = llvm::toString(conforms.takeError());
    require(test, !candidate.isFinite(), message);
    return false;
  }
  return *conforms;
}

AccuracyBounds acceptedBounds(llvm::StringRef test,
                              ::fabric::FloatFormat format,
                              std::uint64_t referenceBits) {
  const llvm::APFloat reference = floating(format, referenceBits);
  require(test, reference.isFinite() && !reference.isNaN(),
          "finite accuracy bounds require a finite reference");
  std::uint64_t lower = orderedKey(format, referenceBits);
  std::uint64_t upper = lower;
  while (lower != 0 && canonicallyConforms(test, format, reference, lower - 1))
    --lower;
  const std::uint64_t maximumKey = widthMask(layout(format).width());
  while (upper != maximumKey &&
         canonicallyConforms(test, format, reference, upper + 1))
    ++upper;
  return {bitsFromOrderedKey(format, lower), bitsFromOrderedKey(format, upper)};
}

PowVector makeVector(llvm::StringRef test, ::fabric::FloatFormat format,
                     std::uint64_t base, std::uint64_t exponent) {
  const std::uint64_t reference = evaluate(test, format, base, exponent);
  const ExpectedKind kind = expectedKind(format, base, exponent, reference);
  AccuracyBounds bounds{reference, reference};
  if (kind == ExpectedKind::Max4Ulp)
    bounds = acceptedBounds(test, format, reference);
  return {format, base, exponent, reference, bounds.lower, bounds.upper, kind};
}

std::vector<PowVector> vectors() {
  std::vector<PowVector> result;
  for (::fabric::FloatFormat format :
       {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16}) {
    const FloatLayout shape = layout(format);
    const std::uint64_t sign = shape.sign();
    const std::uint64_t one = shape.one();
    const std::uint64_t half = one - (std::uint64_t{1} << shape.fractionBits);
    const std::uint64_t two = one + (std::uint64_t{1} << shape.fractionBits);
    const std::uint64_t three = one + (std::uint64_t{1} << shape.fractionBits) +
                                (std::uint64_t{1} << (shape.fractionBits - 1));
    const std::uint64_t four = one + (std::uint64_t{2} << shape.fractionBits);
    const std::uint64_t infinity = shape.infinity();
    const std::uint64_t quiet = std::uint64_t{1} << (shape.fractionBits - 1);
    const std::uint64_t quietNaN = sign | infinity | quiet | 5;
    const std::uint64_t signalingNaN = infinity | 3;
    const std::uint64_t minimumNormal = std::uint64_t{1} << shape.fractionBits;
    const std::uint64_t maximumSubnormal = minimumNormal - 1;
    const std::uint64_t maximumFinite = infinity - 1;
    const std::uint64_t quarter =
        one - (std::uint64_t{2} << shape.fractionBits);
    const std::uint64_t eight = one + (std::uint64_t{3} << shape.fractionBits);
    const std::array curated = {
        std::pair{two, three},
        std::pair{three, two},
        std::pair{two, sign | three},
        std::pair{sign | two, three},
        std::pair{sign | two, four},
        std::pair{sign | two, sign | three},
        std::pair{sign | two, half},
        std::pair{four, half},
        std::pair{two, sign | half},
        std::pair{eight, one - (std::uint64_t{2} << shape.fractionBits)},
        std::pair{two, one - (std::uint64_t{1} << (shape.fractionBits - 1))},
        std::pair{std::uint64_t{0}, three},
        std::pair{sign, three},
        std::pair{sign, four},
        std::pair{std::uint64_t{0}, sign | three},
        std::pair{sign, sign | three},
        std::pair{sign, sign | four},
        std::pair{two, std::uint64_t{0}},
        std::pair{two, sign},
        std::pair{quietNaN, std::uint64_t{0}},
        std::pair{one, quietNaN},
        std::pair{quietNaN, two},
        std::pair{two, signalingNaN},
        std::pair{infinity, two},
        std::pair{infinity, sign | two},
        std::pair{sign | infinity, three},
        std::pair{half, infinity},
        std::pair{two, sign | infinity},
        std::pair{sign | one, infinity},
        std::pair{minimumNormal, two},
        std::pair{maximumSubnormal, half},
        std::pair{std::uint64_t{1}, two},
        std::pair{maximumFinite, two},
        std::pair{maximumFinite, one},
        std::pair{maximumFinite, sign | one},
        std::pair{minimumNormal, sign | one},
        std::pair{maximumSubnormal, one},
        std::pair{sign | maximumSubnormal, one},
        std::pair{two, one},
        std::pair{two, sign | one},
        std::pair{half, sign | one},
        std::pair{one + 1, maximumFinite},
        std::pair{one - 1, maximumFinite},
        std::pair{quarter, half},
    };
    for (const auto &[base, exponent] : curated)
      result.push_back(makeVector("vectors", format, base, exponent));

    std::uint64_t state = 0x9e3779b97f4a7c15ULL ^ shape.width();
    for (unsigned count = 0; count != 128; ++count) {
      std::uint64_t base = nextRandom(state) % infinity;
      if (base == 0)
        base = one;
      if ((count & 3U) == 0)
        base |= sign;
      const int integerExponent = static_cast<int>(nextRandom(state) % 13) - 6;
      llvm::APFloat exponentValue(semantics(format));
      require("vectors",
              take("vectors", exponentValue.convertFromString(
                                  std::to_string(integerExponent),
                                  llvm::RoundingMode::NearestTiesToEven)) ==
                  llvm::APFloat::opOK,
              "could not encode an exact integer exponent");
      std::uint64_t exponent = exponentValue.bitcastToAPInt().getZExtValue();
      if ((count & 1U) != 0)
        exponent = nextRandom(state) % infinity;
      result.push_back(makeVector("vectors", format, base, exponent));
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

std::uint64_t paddedInput(const PowVector &vector, std::uint64_t value) {
  const unsigned width = layout(vector.format).width();
  return value | (0xa5a55a5aULL & ~widthMask(width));
}

std::string makeTestbench() {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << R"sv(module testbench;
  logic [31:0] base;
  logic [31:0] exponent;
  logic [2:0] mode;
  logic [31:0] result;

  scalar_math_pow dut(
      .data_input_0(base), .data_input_1(exponent), .config_0(mode),
      .data_output_0(result));

  function automatic [16:0] ordered_key(input logic [15:0] value);
    begin
      if (value[15])
        ordered_key = 17'h10000 - {1'b0, value[14:0]};
      else
        ordered_key = 17'h10000 + {1'b0, value[14:0]};
    end
  endfunction

  task automatic check(
      input logic [2:0] selected,
      input logic [31:0] base_bits,
      input logic [31:0] exponent_bits,
      input logic [15:0] reference,
      input logic [15:0] lower_bound,
      input logic [15:0] upper_bound,
      input integer expected_kind);
    logic [16:0] actual_key;
    logic [16:0] lower_key;
    logic [16:0] upper_key;
    begin
      mode = selected;
      base = base_bits;
      exponent = exponent_bits;
      #1;
      if (result[31:16] !== 16'd0)
        $fatal(1, "math power did not clear output padding mode=%0d", selected);
      if (expected_kind == 1) begin
        if ((selected == 3'd1 && result[14:10] != 5'h1f) ||
            (selected == 3'd3 && result[14:7] != 8'hff))
          $fatal(1, "math power NaN exponent mismatch mode=%0d got=%h", selected, result);
        if ((selected == 3'd1 && (result[9:0] == 0 || !result[9])) ||
            (selected == 3'd3 && (result[6:0] == 0 || !result[6])))
          $fatal(1, "math power did not produce a quiet NaN mode=%0d got=%h", selected, result);
      end else if (expected_kind == 0) begin
        if (result[15:0] !== reference)
          $fatal(1, "math power exact mismatch mode=%0d base=%h exponent=%h got=%h expected=%h",
                 selected, base_bits, exponent_bits, result, reference);
      end else begin
        if ((selected == 3'd1 && result[14:10] == 5'h1f) ||
            (selected == 3'd3 && result[14:7] == 8'hff))
          $fatal(1, "math power finite reference produced non-finite result mode=%0d base=%h exponent=%h got=%h reference=%h",
                 selected, base_bits, exponent_bits, result, reference);
        actual_key = ordered_key(result[15:0]);
        lower_key = ordered_key(lower_bound);
        upper_key = ordered_key(upper_bound);
        if (actual_key < lower_key || actual_key > upper_key)
          $fatal(1, "math power exceeded canonical Max4Ulp bounds mode=%0d base=%h exponent=%h got=%h reference=%h lower=%h upper=%h",
                 selected, base_bits, exponent_bits, result, reference,
                 lower_bound, upper_bound);
      end
    end
  endtask

  initial begin
)sv";
  for (const PowVector &vector : vectors())
    output << "    check(3'd" << unsigned(modeCode(vector.format)) << ", "
           << hexLiteral(llvm::APInt(32, paddedInput(vector, vector.base)))
           << ", "
           << hexLiteral(llvm::APInt(32, paddedInput(vector, vector.exponent)))
           << ", " << hexLiteral(llvm::APInt(16, vector.reference)) << ", "
           << hexLiteral(llvm::APInt(16, vector.lowerBound)) << ", "
           << hexLiteral(llvm::APInt(16, vector.upperBound)) << ", "
           << static_cast<unsigned>(vector.expectedKind) << ");\n";
  output << "    $finish;\n  end\nendmodule\n";
  return output.str();
}

std::string makeYosysScript() {
  return R"ys(read_verilog -sv scalar_math_pow.sv
hierarchy -check -top scalar_math_pow
proc
opt
check -assert
select -assert-none scalar_math_pow/t:$*ff* scalar_math_pow/t:$*latch* scalar_math_pow/t:$_*FF* scalar_math_pow/t:$_*LATCH* scalar_math_pow/t:$mem* scalar_math_pow/m:*
synth -noabc -top scalar_math_pow
check -assert
select -assert-none scalar_math_pow/t:$*ff* scalar_math_pow/t:$*latch* scalar_math_pow/t:$_*FF* scalar_math_pow/t:$_*LATCH* scalar_math_pow/t:$mem* scalar_math_pow/m:*
stat
)ys";
}

std::string emitProvider(llvm::StringRef test, const ArtifactStore &store) {
  FabricFixture fixture = makeFabric(test, store);
  const auto &resolved = capability(test, fixture);
  require(test,
          resolved.implementationFamily ==
                  ::fabric::ImplementationFamilyId::ScalarMathPow &&
              resolved.enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{
                      ::dataflow::OperationSchemaId::MathPowF},
          "math power escaped its generated family descriptor");
  const auto *parameters = std::get_if<::fabric::ScalarSpecialMathParams>(
      &resolved.parameterizedCapability);
  require(test,
          parameters &&
              parameters->accuracyGuarantee == SpecialMathAccuracyTier::Max4Ulp,
          "math power lost the Fabric accuracy guarantee");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 2,
          "math power did not consume the exact sealed format relation");
  std::array<bool, 2> seen{};
  for (const auto &point : relation.finiteBehaviorDomain()) {
    const auto *payload = std::get_if<::dataflow::SpecialMathPayload>(
        &point.representativeActor.payload);
    const ::fabric::FloatFormat format = behaviorFormat(test, point);
    require(test,
            point.representativeActor.schema ==
                    ::dataflow::OperationSchemaId::MathPowF &&
                point.representativeActor.type.getNumInputs() == 2 && payload &&
                payload->accuracy == SpecialMathAccuracyTier::Max4Ulp &&
                point.operandPorts == std::vector<std::uint64_t>({0, 1}) &&
                point.resultPorts == std::vector<std::uint64_t>({0}) &&
                point.semanticConfiguration.has_value(),
            "math power relation lost ordered physical correspondence");
    require(test,
            format == ::fabric::FloatFormat::F16 ||
                format == ::fabric::FloatFormat::BF16,
            "math power relation contains an unsupported format");
    seen[static_cast<std::size_t>(format)] = true;
  }
  require(test, seen[0] && seen[1],
          "math power relation omitted an admitted format");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fixture, abi.abi());
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 4 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atInput(2).getName() == "config_0" &&
              ports.atInput(2).type ==
                  mlir::IntegerType::get(firstContext.get(), 3) &&
              ports.atOutput(0).getName() == "data_output_0",
          "math power leaf ports do not follow ConfigurationABI 3.0");
  const std::string firstRtl = specialize(test, std::move(first), fixture, abi);
  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fixture, abi.abi());
  const std::string secondRtl =
      specialize(test, std::move(second), fixture, abi);
  require(test, firstRtl == secondRtl,
          "identical math power inputs produced different SystemVerilog");
  const llvm::StringRef rtl(firstRtl);
  require(test,
          rtl.contains("function automatic") && rtl.contains("config_0") &&
              rtl.contains("loom_math_pow_e5_f10_max4") &&
              rtl.contains("loom_math_pow_e8_f7_max4") &&
              !rtl.contains("$pow") && !rtl.contains("$ln") &&
              !rtl.contains("$exp") && !rtl.contains("shortreal") &&
              !rtl.contains(" DPI") && !rtl.contains(" real") &&
              !rtl.contains("**"),
          "math power RTL is incomplete or not portable synthesizable logic");
  return firstRtl;
}

void configuredDomainAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  const auto entry = llvm::find_if(coverage, [](const auto &item) {
    return item.implementationFamily ==
           ::fabric::ImplementationFamilyId::ScalarMathPow;
  });
  require(test,
          entry != coverage.end() &&
              entry->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "math power provider registration is incomplete");
  const std::string rtl = emitProvider(test, store);
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "generated",
          {{"scalar_math_pow.sv", rtl},
           {"testbench.sv", makeTestbench()},
           {"portable_scalar_math_pow.ys", makeYosysScript()}}))
    fail(test, llvm::toString(std::move(error)));
}

void singletonHasNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fixture = makeFabric(test, store, FixtureKind::Singleton);
  const auto &resolved = capability(test, fixture);
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::None &&
              relation.finiteBehaviorDomain().size() == 1 &&
              !relation.finiteBehaviorDomain().front().semanticConfiguration,
          "singleton math power relation retained a selector authority");
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fixture, abi.abi());
  require(test, skeleton.leaf.getPortList().size() == 3,
          "singleton math power leaf retained a selector port");
  const std::string rtl = specialize(test, std::move(skeleton), fixture, abi);
  require(test,
          llvm::StringRef(rtl).contains("loom_math_pow_e5_f10_max4") &&
              !llvm::StringRef(rtl).contains("config_0"),
          "singleton math power RTL did not use its sealed witness");
}

void unsupportedAndMalformedAreTransactional(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  const std::array unsupportedCases = {
      std::pair{FixtureKind::UnsupportedContract, "unsupported contract"},
      std::pair{FixtureKind::UnsupportedBehavior, "unsupported behavior"},
      std::pair{FixtureKind::UnsupportedF32, "unsupported f32"},
      std::pair{FixtureKind::UnsupportedF64, "unsupported f64"},
      std::pair{FixtureKind::CorrectlyRounded, "correctly rounded"},
      std::pair{FixtureKind::Max1Ulp, "Max1Ulp"},
      std::pair{FixtureKind::Max2Ulp, "Max2Ulp"},
  };
  for (const auto &[kind, caseName] : unsupportedCases) {
    FabricFixture unsupported = makeFabric(test, store, kind);
    FinalizedConfigurationABI abi =
        makeConfigurationAbi(test, store, unsupported);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton =
        makeSkeleton(test, *context, unsupported, abi.abi());
    const std::string before = moduleText(*skeleton.module);
    expectTypedUnsupported(caseName,
                           trySpecialize(skeleton, unsupported, abi, registry),
                           BackendRecipeKey::PortableSystemVerilog);
    require(test, moduleText(*skeleton.module) == before,
            "Unsupported math power request mutated the skeleton");
  }

  expectFabricRejected(test, store, FixtureKind::MissingPhysicalInput,
                       "physical role inventory");
  expectFabricRejected(test, store, FixtureKind::InsufficientPhysicalWidth,
                       "physically reachable");

  FabricFixture valid = makeFabric(test, store);
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
  SkeletonFixture malformed =
      makeSkeleton(test, *malformedContext, valid, validAbi.abi(), true);
  const std::string malformedBefore = moduleText(*malformed.module);
  expectInvalid(test, trySpecialize(malformed, valid, validAbi, registry),
                "leaf port");
  require(test, moduleText(*malformed.module) == malformedBefore,
          "malformed math power input partially mutated the skeleton");

  for (AbiKind kind : {AbiKind::MissingBehavior, AbiKind::ExtraBehavior,
                       AbiKind::DirectEncoding})
    expectInvalid(
        test,
        finalizeConfigurationABI(makeConfigurationAbiDraft(test, valid, kind),
                                 store),
        kind == AbiKind::DirectEncoding ? "finite codebook" : "semantic");

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
            "unsupported native recipe mutated the skeleton");
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  configuredDomainAndArtifacts(root);
  singletonHasNoSelector(root / "singleton");
  unsupportedAndMalformedAreTransactional(root / "malformed");
  return EXIT_SUCCESS;
}

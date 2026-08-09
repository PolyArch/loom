#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/ScalarMathHyperbolic.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
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

enum class HyperbolicFamily { Sinh, Cosh, Tanh };
enum class FixtureKind {
  Configured,
  Singleton,
  UnsupportedContract,
  UnsupportedFormat,
  UnsupportedAccuracy,
  UnsupportedRounding,
  UnsupportedNaN,
  UnsupportedSignedZero,
  UnsupportedFastMath,
  InvalidSubnormal,
};
enum class AbiKind { Complete, MissingMode, ExtraMode, DirectEncoding };

struct FloatLayout final {
  unsigned exponentBits;
  unsigned fractionBits;

  unsigned width() const { return 1 + exponentBits + fractionBits; }
  std::uint32_t sign() const { return std::uint32_t{1} << (width() - 1); }
  std::uint32_t exponentMask() const {
    return (std::uint32_t{1} << exponentBits) - 1;
  }
  std::uint32_t infinity() const { return exponentMask() << fractionBits; }
  std::uint32_t one() const {
    return ((std::uint32_t{1} << (exponentBits - 1)) - 1) << fractionBits;
  }
};

struct FabricFixture final {
  HyperbolicFamily family;
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
  std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef>
      physicalOccurrences;
};

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

struct Mode final {
  ::fabric::FloatFormat format;
  ::dataflow::CanonicalActorSchemaProjection actor;
  std::uint8_t code;
};

struct EmittedProvider final {
  std::string rtl;
  std::vector<Mode> modes;
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
    fail(test, "accepted malformed portable hyperbolic input");
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

::fabric::ImplementationFamilyId familyId(HyperbolicFamily family) {
  switch (family) {
  case HyperbolicFamily::Sinh:
    return ::fabric::ImplementationFamilyId::ScalarMathSinh;
  case HyperbolicFamily::Cosh:
    return ::fabric::ImplementationFamilyId::ScalarMathCosh;
  case HyperbolicFamily::Tanh:
    return ::fabric::ImplementationFamilyId::ScalarMathTanh;
  }
  llvm_unreachable("unknown hyperbolic family");
}

llvm::StringRef familyKeyword(HyperbolicFamily family) {
  switch (family) {
  case HyperbolicFamily::Sinh:
    return "ScalarMathSinh";
  case HyperbolicFamily::Cosh:
    return "ScalarMathCosh";
  case HyperbolicFamily::Tanh:
    return "ScalarMathTanh";
  }
  llvm_unreachable("unknown hyperbolic family");
}

llvm::StringRef schemaKeyword(HyperbolicFamily family) {
  switch (family) {
  case HyperbolicFamily::Sinh:
    return "math.sinh";
  case HyperbolicFamily::Cosh:
    return "math.cosh";
  case HyperbolicFamily::Tanh:
    return "math.tanh";
  }
  llvm_unreachable("unknown hyperbolic family");
}

llvm::StringRef moduleName(HyperbolicFamily family) {
  switch (family) {
  case HyperbolicFamily::Sinh:
    return "scalar_math_sinh";
  case HyperbolicFamily::Cosh:
    return "scalar_math_cosh";
  case HyperbolicFamily::Tanh:
    return "scalar_math_tanh";
  }
  llvm_unreachable("unknown hyperbolic family");
}

std::string fabricSource(HyperbolicFamily family, FixtureKind kind) {
  const bool singleton = kind == FixtureKind::Singleton ||
                         kind == FixtureKind::UnsupportedAccuracy;
  const bool unsupportedFormat = kind == FixtureKind::UnsupportedFormat;
  const unsigned width = unsupportedFormat ? 64 : singleton ? 32 : 32;
  const llvm::StringRef formats = unsupportedFormat ? R"mlir(["f64"])mlir"
                                  : singleton
                                      ? R"mlir(["f32"])mlir"
                                      : R"mlir(["f16", "bf16", "f32"])mlir";
  const llvm::StringRef accuracy =
      kind == FixtureKind::UnsupportedAccuracy ? "Max2Ulp" : "Max4Ulp";
  const llvm::StringRef rounding =
      kind == FixtureKind::UnsupportedRounding ? "downward" : "to_nearest_even";
  const llvm::StringRef nan =
      kind == FixtureKind::UnsupportedNaN ? "number_preferred" : "ieee";
  const llvm::StringRef subnormal =
      kind == FixtureKind::InvalidSubnormal ? "flush_to_zero" : "preserve";
  const llvm::StringRef signedZero =
      kind == FixtureKind::UnsupportedSignedZero ? "ignore_sign" : "preserve";
  const llvm::StringRef fastmath =
      kind == FixtureKind::UnsupportedFastMath ? "fast" : "afn";

  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @" << moduleName(family)
         << "(%input: !fabric.bits<" << width << ">) -> !fabric.bits<" << width
         << "> { %pe = fabric.pe [spatial]"
         << "(%pe_input = %input : !fabric.bits<" << width
         << ">) -> !fabric.bits<" << width << "> { %fu = fabric.fu"
         << "(%fu_input = %pe_input : !fabric.bits<" << width
         << ">) -> !fabric.bits<" << width << "> { %value = fabric.op [@"
         << schemaKeyword(family) << "] (%fu_input) {implementation_family = "
         << "#fabric.implementation_family<" << familyKeyword(family)
         << ">, hw_params = {float_formats = " << formats
         << ", behavior = {rounding_modes = [\"" << rounding
         << "\"], nan_behaviors = [\"" << nan
         << "\"], subnormal_behaviors = [\"" << subnormal
         << "\"], signed_zero_behaviors = [\"" << signedZero
         << "\"], fastmath = \"" << fastmath << "\"}, accuracy_guarantee = \""
         << accuracy << "\"}} : (!fabric.bits<" << width
         << ">) -> !fabric.bits<" << width
         << "> fabric.yield %value : !fabric.bits<" << width
         << "> } } fabric.yield %pe : !fabric.bits<" << width << "> } }";
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
  const std::vector<std::int8_t> bytes(encoded.begin(), encoded.end());
  module.walk([&](::fabric::OpOp operation) {
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       mlir::DenseI8ArrayAttr::get(&fabricContext(), bytes));
  });
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         HyperbolicFamily family,
                         FixtureKind kind = FixtureKind::Configured,
                         std::uint64_t spatialCoreCount = 1) {
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
          take(test, loom::hardware::test::makeSpatialCoreSystem(
                         fabric, store, spatialCoreCount));
      auto systemView =
          take(test, loom::fabric::requireSystemRoot(system.view()));
      const auto operations =
          take(test, enumerateFabricPhysicalOperations(systemView));
      std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef>
          physicalOccurrences;
      for (const auto &entry : operations)
        if (entry.localOccurrence == occurrence)
          physicalOccurrences.push_back(entry.physicalOccurrence);
      require(test, physicalOccurrences.size() == spatialCoreCount,
              "System has the wrong physical hyperbolic occurrence count");
      return FabricFixture{family,
                           std::move(fabric),
                           occurrence,
                           std::move(system),
                           physicalOccurrences.front(),
                           std::move(physicalOccurrences)};
    }
  }
  fail(test, "Fabric fixture has no hyperbolic occurrence");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

::fabric::FloatFormat formatOf(llvm::StringRef test, mlir::Type type) {
  if (mlir::isa<mlir::Float16Type>(type))
    return ::fabric::FloatFormat::F16;
  if (mlir::isa<mlir::BFloat16Type>(type))
    return ::fabric::FloatFormat::BF16;
  if (mlir::isa<mlir::Float32Type>(type))
    return ::fabric::FloatFormat::F32;
  if (mlir::isa<mlir::Float64Type>(type))
    return ::fabric::FloatFormat::F64;
  fail(test, "behavior does not use a known floating format");
}

std::uint8_t formatCode(::fabric::FloatFormat format,
                        std::size_t occurrenceIndex = 0) {
  constexpr std::array<std::array<std::uint8_t, 3>, 2> codebooks = {
      {{{1, 2, 3}}, {{3, 1, 2}}}};
  const std::size_t codebook = occurrenceIndex % codebooks.size();
  switch (format) {
  case ::fabric::FloatFormat::F16:
    return codebooks[codebook][0];
  case ::fabric::FloatFormat::BF16:
    return codebooks[codebook][1];
  case ::fabric::FloatFormat::F32:
    return codebooks[codebook][2];
  case ::fabric::FloatFormat::F64:
    return 0;
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
          "hyperbolic fixture has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(
      test,
      relation.kind() == ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
          relation.finiteBehaviorDomain().size() == 3,
      "configured hyperbolic relation is not the sealed three-format domain");
  require(test,
          kind == AbiKind::Complete || fixture.physicalOccurrences.size() == 1,
          "malformed ABI fixture requires one physical occurrence");

  std::vector<loom::hardware::test::ConfigurationFieldEncodingOverride>
      overrides;
  for (const auto &[occurrenceIndex, physicalOccurrence] :
       llvm::enumerate(fixture.physicalOccurrences)) {
    std::vector<FiniteCodebookEntry> entries;
    std::vector<std::uint8_t> inactive;
    for (const auto &point : relation.finiteBehaviorDomain()) {
      require(test, point.semanticConfiguration.has_value(),
              "configured hyperbolic mode has no semantic value");
      const ::fabric::FloatFormat format =
          formatOf(test, point.representativeActor.type.getInput(0));
      std::vector<std::uint8_t> semantic(
          point.semanticConfiguration->bytes().begin(),
          point.semanticConfiguration->bytes().end());
      const ::fabric::FloatFormat inactiveFormat =
          occurrenceIndex == 0 ? ::fabric::FloatFormat::F32
                               : ::fabric::FloatFormat::F16;
      if (format == inactiveFormat)
        inactive = semantic;
      if (kind == AbiKind::MissingMode && format == ::fabric::FloatFormat::BF16)
        continue;
      entries.push_back(
          {std::move(semantic), {formatCode(format, occurrenceIndex)}});
    }
    if (kind == AbiKind::ExtraMode)
      entries.push_back({{0xfe}, {0}});
    require(test, !inactive.empty(),
            "hyperbolic relation has no inactive mode");
    const auto field =
        take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                       physicalOccurrence,
                       resolved.configurationFieldSchema.front().ordinal));
    SemanticFieldEncoding encoding =
        kind == AbiKind::DirectEncoding
            ? SemanticFieldEncoding{DirectBitsEncoding{2}}
            : SemanticFieldEncoding{
                  FiniteCodebookEncoding{2, std::move(entries)}};
    if (kind == AbiKind::DirectEncoding)
      inactive = {0};
    overrides.push_back({field, std::move(encoding), std::move(inactive)});
  }
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, std::move(overrides)));
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

SkeletonFixture makeSkeletonForOccurrence(
    llvm::StringRef test, mlir::MLIRContext &context,
    const FabricFixture &fixture, const ConfigurationABI &abi,
    const loom::fabric::FabricPhysicalOccurrenceOwnerRef &physicalOccurrence,
    llvm::StringRef name, bool wrongConfigurationWidth = false) {
  const auto &resolved = capability(test, fixture);
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports =
      take(test, deriveFabricOperationLeafPorts(builder, physicalOccurrence,
                                                resolved, abi));
  if (wrongConfigurationWidth) {
    require(test, ports.size() == 3,
            "configured hyperbolic leaf does not have three ports");
    ports[1].type = builder.getI1Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(name), ports);
  return SkeletonFixture{std::move(module), leaf};
}

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fixture,
                             const ConfigurationABI &abi,
                             bool wrongConfigurationWidth = false) {
  return makeSkeletonForOccurrence(
      test, context, fixture, abi, fixture.physicalOccurrence,
      moduleName(fixture.family), wrongConfigurationWidth);
}

FabricOperationProviderRegistry makeRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableScalarMathHyperbolicProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

llvm::Expected<loom::hardware::test::PortableProviderConformance>
trySpecializeOccurrence(
    SkeletonFixture skeleton, const FinalizedConfigurationABI &configurationAbi,
    const FabricOperationProviderRegistry &registry,
    const loom::fabric::FabricPhysicalOccurrenceOwnerRef &physicalOccurrence) {
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, physicalOccurrence}}};
  return loom::hardware::test::specializeAndExportPortableProvider(
      std::move(module), configurationAbi, registry, externalContracts);
}

llvm::Expected<loom::hardware::test::PortableProviderConformance>
trySpecialize(SkeletonFixture skeleton, const FabricFixture &fixture,
              const FinalizedConfigurationABI &configurationAbi,
              const FabricOperationProviderRegistry &registry) {
  return trySpecializeOccurrence(std::move(skeleton), configurationAbi,
                                 registry, fixture.physicalOccurrence);
}

std::vector<Mode> modes(llvm::StringRef test, const FabricFixture &fixture,
                        const FinalizedConfigurationABI &abi) {
  const auto &resolved = capability(test, fixture);
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  std::vector<Mode> result;
  if (resolved.configurationFieldSchema.empty()) {
    require(test,
            relation.kind() ==
                    ::fabric::FabricOpSemanticFieldRelationKind::None &&
                relation.finiteBehaviorDomain().size() == 1 &&
                !relation.finiteBehaviorDomain().front().semanticConfiguration,
            "singleton hyperbolic relation is not fieldless");
    const auto &actor =
        relation.finiteBehaviorDomain().front().representativeActor;
    result.push_back({formatOf(test, actor.type.getInput(0)), actor, 0});
    return result;
  }
  const auto *field = abi.abi().findOperationField(
      fixture.physicalOccurrence,
      resolved.configurationFieldSchema.front().ordinal);
  require(test, field != nullptr, "configured ABI field is absent");
  const auto *codebook =
      std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
  require(test, codebook != nullptr, "configured ABI is not a finite codebook");
  for (const auto &point : relation.finiteBehaviorDomain()) {
    const auto entry =
        llvm::find_if(codebook->entries, [&](const auto &candidate) {
          return llvm::ArrayRef<std::uint8_t>(candidate.semanticValue)
              .equals(point.semanticConfiguration->bytes());
        });
    require(test,
            entry != codebook->entries.end() && entry->physicalCode.size() == 1,
            "configured mode has no one-byte physical code");
    const auto &actor = point.representativeActor;
    result.push_back({formatOf(test, actor.type.getInput(0)), actor,
                      entry->physicalCode.front()});
  }
  return result;
}

EmittedProvider emit(llvm::StringRef test, const ArtifactStore &store,
                     HyperbolicFamily family, FixtureKind kind) {
  FabricFixture fixture = makeFabric(test, store, family, kind);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fixture, abi.abi());
  auto conformance = take(test, trySpecialize(std::move(skeleton), fixture, abi,
                                              makeRegistry(test)));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "self-contained hyperbolic provider emitted implementation metadata");
  return EmittedProvider{std::move(conformance.systemVerilog),
                         modes(test, fixture, abi)};
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

std::uint32_t bitsFromDouble(::fabric::FloatFormat format, double input) {
  llvm::APFloat value(input);
  bool losesInformation = false;
  (void)value.convert(semantics(format), llvm::RoundingMode::NearestTiesToEven,
                      &losesInformation);
  return static_cast<std::uint32_t>(value.bitcastToAPInt().getZExtValue());
}

void appendUnique(std::vector<std::uint32_t> &values, std::uint32_t value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
}

std::vector<std::uint32_t> inputs(::fabric::FloatFormat format) {
  const FloatLayout shape = layout(format);
  const std::uint32_t sign = shape.sign();
  const std::uint32_t infinity = shape.infinity();
  const std::uint32_t fractionMask =
      (std::uint32_t{1} << shape.fractionBits) - 1;
  const std::uint32_t quiet = std::uint32_t{1} << (shape.fractionBits - 1);
  std::vector<std::uint32_t> values;
  for (std::uint32_t bits :
       {std::uint32_t{0}, sign, std::uint32_t{1}, sign | 1, fractionMask,
        sign | fractionMask, std::uint32_t{1} << shape.fractionBits,
        sign | (std::uint32_t{1} << shape.fractionBits), infinity,
        sign | infinity, infinity | quiet | 5, sign | infinity | 3,
        infinity | 3, infinity - 1, sign | (infinity - 1)})
    appendUnique(values, bits);

  constexpr std::array<double, 24> points = {0x1p-14, 0x1p-13,
                                             0x1p-12, 0.001,
                                             0.01,    0.125,
                                             0.5,     0.6931471805599453,
                                             1.0,     1.3862943611198906,
                                             2.0,     4.0,
                                             5.0,     8.0,
                                             8.5,     9.0,
                                             10.0,    11.0,
                                             16.0,    80.0,
                                             88.0,    89.0,
                                             90.0,    100.0};
  for (double point : points) {
    const std::uint32_t positive = bitsFromDouble(format, point);
    if ((positive & infinity) == infinity)
      continue;
    for (std::uint32_t neighbor : {positive - (positive != 0), positive,
                                   positive + (positive + 1 < infinity)}) {
      appendUnique(values, neighbor);
      appendUnique(values, neighbor | sign);
    }
  }

  std::uint32_t state = 0x6d2b79f5U;
  for (unsigned index = 0; index != 48; ++index) {
    state = state * 1664525U + 1013904223U;
    const unsigned exponent = 1 + (state % (shape.exponentMask() - 1));
    const std::uint32_t fraction = (state >> 8) & fractionMask;
    const std::uint32_t value = (exponent << shape.fractionBits) | fraction;
    appendUnique(values, value);
    appendUnique(values, value | sign);
  }
  return values;
}

const Mode &findMode(llvm::StringRef test, const EmittedProvider &provider,
                     ::fabric::FloatFormat format) {
  const auto mode = llvm::find_if(provider.modes, [&](const Mode &candidate) {
    return candidate.format == format;
  });
  require(test, mode != provider.modes.end(), "provider omitted a format mode");
  return *mode;
}

std::uint32_t evaluate(llvm::StringRef test, const Mode &mode,
                       std::uint32_t input) {
  const unsigned width = layout(mode.format).width();
  llvm::APFloat operand(semantics(mode.format), llvm::APInt(width, input));
  loom::sim::PrimitiveOperationDescriptor descriptor{mode.actor, width, width};
  loom::sim::PrimitiveValue result = take(
      test, loom::sim::evaluatePrimitiveOperation(
                descriptor, {loom::sim::PrimitiveValue::floating(operand)}));
  require(test, result.isDefined(),
          "MPFR/APFloat hyperbolic oracle returned a non-defined value");
  return static_cast<std::uint32_t>(result.bits->getZExtValue());
}

template <typename Predicate>
std::uint32_t firstPositiveInputWhere(llvm::StringRef test, const Mode &mode,
                                      Predicate predicate) {
  const std::uint32_t maximumFinite = layout(mode.format).infinity() - 1;
  require(test, predicate(evaluate(test, mode, maximumFinite)),
          "oracle transition predicate does not hold at maximum finite input");
  std::uint32_t lower = 0;
  std::uint32_t upper = maximumFinite;
  while (lower < upper) {
    const std::uint32_t middle = lower + (upper - lower) / 2;
    if (predicate(evaluate(test, mode, middle)))
      upper = middle;
    else
      lower = middle + 1;
  }
  return lower;
}

void appendTransitionNeighborhood(std::vector<std::uint32_t> &values,
                                  std::uint32_t transition,
                                  std::uint32_t maximumFinite) {
  if (transition != 0)
    appendUnique(values, transition - 1);
  appendUnique(values, transition);
  if (transition != maximumFinite)
    appendUnique(values, transition + 1);
}

void appendOracleTransitions(llvm::StringRef test,
                             std::vector<std::uint32_t> &values,
                             const Mode &sinhMode, const Mode &coshMode,
                             const Mode &tanhMode) {
  const FloatLayout shape = layout(sinhMode.format);
  const std::uint32_t maximumFinite = shape.infinity() - 1;
  for (const Mode *mode : {&sinhMode, &coshMode}) {
    const std::uint32_t firstOverflow =
        firstPositiveInputWhere(test, *mode, [&](std::uint32_t result) {
          return result == shape.infinity();
        });
    appendTransitionNeighborhood(values, firstOverflow, maximumFinite);
  }
  const std::uint32_t firstSaturation =
      firstPositiveInputWhere(test, tanhMode, [&](std::uint32_t result) {
        return result == shape.one();
      });
  appendTransitionNeighborhood(values, firstSaturation, maximumFinite);
}

bool requiresExactCheck(::fabric::FloatFormat format, std::uint32_t input,
                        std::uint32_t expected) {
  const FloatLayout shape = layout(format);
  const std::uint32_t exponent =
      (input >> shape.fractionBits) & shape.exponentMask();
  const std::uint32_t expectedExponent =
      (expected >> shape.fractionBits) & shape.exponentMask();
  const std::uint32_t expectedFraction =
      expected & ((std::uint32_t{1} << shape.fractionBits) - 1);
  return exponent == 0 || exponent == shape.exponentMask() ||
         expectedExponent == shape.exponentMask() ||
         (expectedExponent == 0 && expectedFraction == 0);
}

std::string testbench(const std::array<EmittedProvider, 3> &providers) {
  std::string text;
  llvm::raw_string_ostream output(text);
  output << R"sv(module testbench;
  logic [31:0] data_input;
  logic [1:0] configuration;
  wire [31:0] sinh_output;
  wire [31:0] cosh_output;
  wire [31:0] tanh_output;

  scalar_math_sinh sinh_dut(
      .data_input_0(data_input), .config_0(configuration),
      .data_output_0(sinh_output));
  scalar_math_cosh cosh_dut(
      .data_input_0(data_input), .config_0(configuration),
      .data_output_0(cosh_output));
  scalar_math_tanh tanh_dut(
      .data_input_0(data_input), .config_0(configuration),
      .data_output_0(tanh_output));

  function automatic [32:0] ordered_bits(
      input [31:0] value, input integer width);
    reg [31:0] mask;
    reg [31:0] sign;
    begin
      mask = width == 32 ? 32'hffff_ffff : 32'h0000_ffff;
      sign = width == 32 ? 32'h8000_0000 : 32'h0000_8000;
      ordered_bits = (value & sign) != 0
                         ? {1'b0, (~value) & mask}
                         : {1'b0, (value | sign) & mask};
    end
  endfunction

  task automatic check_result(
      input [31:0] actual, input [31:0] expected,
      input [31:0] infinity, input integer width,
      input exact, input integer family);
    reg [31:0] mask;
    reg [32:0] actual_ordered;
    reg [32:0] expected_ordered;
    reg [32:0] distance;
    begin
      mask = width == 32 ? 32'hffff_ffff : 32'h0000_ffff;
      if ((actual & ~mask) != 0)
        $fatal(1, "hyperbolic padding mismatch family=%0d got=%h",
               family, actual);
      if (((actual & infinity) == infinity) !=
          ((expected & infinity) == infinity))
        $fatal(1, "hyperbolic exceptional-class mismatch family=%0d input=%h got=%h expected=%h",
               family, data_input, actual & mask, expected);
      if (exact != 0) begin
        if ((actual & mask) !== expected)
          $fatal(1, "hyperbolic oracle mismatch family=%0d input=%h got=%h expected=%h",
                 family, data_input, actual & mask, expected);
      end else begin
        actual_ordered = ordered_bits(actual & mask, width);
        expected_ordered = ordered_bits(expected, width);
        distance = actual_ordered >= expected_ordered
                       ? actual_ordered - expected_ordered
                       : expected_ordered - actual_ordered;
        if (distance > 4)
          $fatal(1, "hyperbolic oracle mismatch family=%0d input=%h got=%h expected=%h ulp=%0d",
                 family, data_input, actual & mask, expected, distance);
      end
    end
  endtask

  task automatic check_vector(
      input integer width, input [1:0] mode, input [31:0] infinity,
      input [31:0] value,
      input [31:0] sinh_expected, input [31:0] cosh_expected,
      input [31:0] tanh_expected, input [2:0] exact);
    begin
      data_input = width == 32 ? value : {16'ha5a5, value[15:0]};
      configuration = mode;
      #1;
      check_result(sinh_output, sinh_expected, infinity, width, exact[0], 0);
      check_result(cosh_output, cosh_expected, infinity, width, exact[1], 1);
      check_result(tanh_output, tanh_expected, infinity, width, exact[2], 2);
    end
  endtask

  task automatic check_parity(
      input integer width, input [1:0] mode, input [31:0] positive);
    reg [31:0] sign;
    reg [31:0] mask;
    reg [31:0] positive_sinh;
    reg [31:0] positive_cosh;
    reg [31:0] positive_tanh;
    begin
      sign = width == 32 ? 32'h8000_0000 : 32'h0000_8000;
      mask = width == 32 ? 32'hffff_ffff : 32'h0000_ffff;
      data_input = positive;
      configuration = mode;
      #1;
      positive_sinh = sinh_output & mask;
      positive_cosh = cosh_output & mask;
      positive_tanh = tanh_output & mask;
      data_input = (positive | sign) |
                   (width == 32 ? 32'd0 : 32'h5a5a_0000);
      #1;
      if ((sinh_output & mask) !== (positive_sinh | sign) ||
          (cosh_output & mask) !== positive_cosh ||
          (tanh_output & mask) !== (positive_tanh | sign))
        $fatal(1, "hyperbolic parity mismatch width=%0d mode=%0d",
               width, mode);
    end
  endtask

  initial begin
    data_input = '0;
    configuration = '0;
)sv";

  for (::fabric::FloatFormat format :
       {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
        ::fabric::FloatFormat::F32}) {
    const Mode &sinhMode = findMode("testbench", providers[0], format);
    const Mode &coshMode = findMode("testbench", providers[1], format);
    const Mode &tanhMode = findMode("testbench", providers[2], format);
    require("testbench",
            sinhMode.code == coshMode.code && coshMode.code == tanhMode.code,
            "families disagree on the explicit format codebook");
    const unsigned width = layout(format).width();
    std::vector<std::uint32_t> oracleInputs = inputs(format);
    appendOracleTransitions("testbench", oracleInputs, sinhMode, coshMode,
                            tanhMode);
    for (std::uint32_t input : oracleInputs) {
      const std::array expected = {evaluate("testbench", sinhMode, input),
                                   evaluate("testbench", coshMode, input),
                                   evaluate("testbench", tanhMode, input)};
      unsigned exact = 0;
      for (unsigned family = 0; family != expected.size(); ++family)
        if (requiresExactCheck(format, input, expected[family]))
          exact |= 1U << family;
      output << "    check_vector(" << width << ", 2'd"
             << static_cast<unsigned>(sinhMode.code) << ", 32'h";
      output.write_hex(layout(format).infinity());
      output << ", 32'h";
      output.write_hex(input);
      for (std::uint32_t value : expected) {
        output << ", 32'h";
        output.write_hex(value);
      }
      output << ", 3'd" << exact << ");\n";
    }
    for (double point : {0.125, 0.5, 1.0, 2.0}) {
      output << "    check_parity(" << width << ", 2'd"
             << static_cast<unsigned>(sinhMode.code) << ", 32'h";
      output.write_hex(bitsFromDouble(format, point));
      output << ");\n";
    }
  }
  output << R"sv(    $display("portable scalar hyperbolic PASS");
    $finish;
  end
endmodule
)sv";
  return output.str();
}

std::string synthesisTop() {
  return R"sv(module scalar_math_hyperbolic_synthesis_top(
    input [31:0] data_input,
    input [1:0] configuration,
    output [31:0] sinh_output,
    output [31:0] cosh_output,
    output [31:0] tanh_output);
  scalar_math_sinh sinh_dut(
      .data_input_0(data_input), .config_0(configuration),
      .data_output_0(sinh_output));
  scalar_math_cosh cosh_dut(
      .data_input_0(data_input), .config_0(configuration),
      .data_output_0(cosh_output));
  scalar_math_tanh tanh_dut(
      .data_input_0(data_input), .config_0(configuration),
      .data_output_0(tanh_output));
endmodule
)sv";
}

std::string yosysScript() {
  return R"ys(read_verilog -sv scalar_math_sinh.sv scalar_math_cosh.sv scalar_math_tanh.sv synthesis_top.sv
hierarchy -check -top scalar_math_hyperbolic_synthesis_top
proc
memory_map
opt_clean
check -assert
select -assert-none t:$dff t:$dlatch t:$memrd t:$memwr t:$meminit t:$mem_v2
synth -noabc -top scalar_math_hyperbolic_synthesis_top
check -assert
select -assert-none t:$_DFF_* t:$_SDFF_* t:$_DLATCH_*
stat
)ys";
}

void configuredSemanticsAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  std::array<EmittedProvider, 3> providers = {
      emit(test, store, HyperbolicFamily::Sinh, FixtureKind::Configured),
      emit(test, store, HyperbolicFamily::Cosh, FixtureKind::Configured),
      emit(test, store, HyperbolicFamily::Tanh, FixtureKind::Configured)};
  for (std::size_t index = 0; index != providers.size(); ++index) {
    HyperbolicFamily family = static_cast<HyperbolicFamily>(index);
    const EmittedProvider again =
        emit(test, store, family, FixtureKind::Configured);
    require(test, providers[index].rtl == again.rtl,
            "identical hyperbolic inputs produced different SystemVerilog");
    const llvm::StringRef rtl(providers[index].rtl);
    require(test,
            rtl.contains("function automatic") && rtl.contains("config_0") &&
                rtl.contains("loom_hyperbolic") && !rtl.contains("shortreal") &&
                !rtl.contains(" DPI") && !rtl.contains(" real"),
            "hyperbolic RTL is incomplete or not synthesizable bit logic");
  }
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{{"scalar_math_sinh.sv"}, providers[0].rtl},
           {{"scalar_math_cosh.sv"}, providers[1].rtl},
           {{"scalar_math_tanh.sv"}, providers[2].rtl},
           {{"testbench.sv"}, testbench(providers)},
           {{"synthesis_top.sv"}, synthesisTop()},
           {{"portable_scalar_math_hyperbolic.ys"}, yosysScript()}}))
    fail(test, llvm::toString(std::move(error)));
}

void physicalOccurrencesOwnConfiguration(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricFixture fixture = makeFabric(test, store, HyperbolicFamily::Sinh,
                                     FixtureKind::Configured, 2);
  require(test,
          fixture.physicalOccurrences.size() == 2 &&
              fixture.physicalOccurrences[0] != fixture.physicalOccurrences[1],
          "ABI2 fixture did not produce distinct physical occurrences");
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
  std::array<std::string, 2> rtl;
  for (std::size_t index = 0; index != rtl.size(); ++index) {
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeletonForOccurrence(
        test, *context, fixture, abi.abi(), fixture.physicalOccurrences[index],
        "scalar_math_sinh_occurrence_" + std::to_string(index));
    auto conformance =
        take(test, trySpecializeOccurrence(std::move(skeleton), abi,
                                           makeRegistry(test),
                                           fixture.physicalOccurrences[index]));
    rtl[index] = std::move(conformance.systemVerilog);
  }
  require(
      test,
      llvm::StringRef(rtl[0]).contains("2'd1: format_code = 2'd0;") &&
          llvm::StringRef(rtl[0]).contains("default: format_code = 2'd2;") &&
          llvm::StringRef(rtl[1]).contains("2'd3: format_code = 2'd0;") &&
          llvm::StringRef(rtl[1]).contains("default: format_code = 2'd0;"),
      "provider did not honor occurrence-qualified codebooks and inactive "
      "values");
}

void singletonRelationsAreFieldless(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  for (HyperbolicFamily family :
       {HyperbolicFamily::Sinh, HyperbolicFamily::Cosh,
        HyperbolicFamily::Tanh}) {
    const EmittedProvider provider =
        emit(test, store, family, FixtureKind::Singleton);
    require(test, !llvm::StringRef(provider.rtl).contains("config_0"),
            "singleton hyperbolic behavior retained a selector");
  }
}

void expectTypedUnsupported(llvm::StringRef test, const ArtifactStore &store,
                            HyperbolicFamily family, FixtureKind kind) {
  FabricFixture fixture = makeFabric(test, store, family, kind);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fixture, abi.abi());
  auto result =
      trySpecialize(std::move(skeleton), fixture, abi, makeRegistry(test));
  require(test, !result, "unsupported hyperbolic capability specialized");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() == familyId(family) &&
                     error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unexpected Unsupported error: " + error.message());
      });
  require(test, classified, "capability rejection was not typed Unsupported");
}

void unsupportedBoundaryIsTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  constexpr std::array unsupported = {
      std::pair{HyperbolicFamily::Sinh, FixtureKind::UnsupportedContract},
      std::pair{HyperbolicFamily::Cosh, FixtureKind::UnsupportedFormat},
      std::pair{HyperbolicFamily::Tanh, FixtureKind::UnsupportedAccuracy},
      std::pair{HyperbolicFamily::Cosh, FixtureKind::UnsupportedNaN},
      std::pair{HyperbolicFamily::Tanh, FixtureKind::UnsupportedSignedZero},
      std::pair{HyperbolicFamily::Sinh, FixtureKind::UnsupportedFastMath},
  };
  for (const auto &[family, kind] : unsupported)
    expectTypedUnsupported(test, store, family, kind);
}

llvm::Expected<FabricOperationProviderOutput>
placeholderProvider(FabricOperationProviderRequest) {
  return FabricOperationProviderOutput{};
}

bool hasCoverage(const FabricOperationProviderRegistry &registry,
                 ::fabric::ImplementationFamilyId family) {
  return llvm::any_of(registry.coverage(), [&](const auto &entry) {
    return entry.implementationFamily == family && entry.recipes.size() == 1 &&
           entry.recipes.front() == BackendRecipeKey::PortableSystemVerilog;
  });
}

void packageRegistrationRollsBack() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registry.add({::fabric::ImplementationFamilyId::ScalarMathTanh,
                        BackendRecipeKey::PortableSystemVerilog,
                        {},
                        placeholderProvider}))
    fail(test, llvm::toString(std::move(error)));
  llvm::Error error = registerPortableScalarMathHyperbolicProviders(registry);
  require(test, static_cast<bool>(error),
          "duplicate package registration unexpectedly succeeded");
  llvm::consumeError(std::move(error));
  require(test,
          !hasCoverage(registry,
                       ::fabric::ImplementationFamilyId::ScalarMathSinh) &&
              !hasCoverage(registry,
                           ::fabric::ImplementationFamilyId::ScalarMathCosh) &&
              hasCoverage(registry,
                          ::fabric::ImplementationFamilyId::ScalarMathTanh),
          "failed package registration did not roll back");
}

void malformedInputsFailClosed(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  const auto expectInvalidProfile = [&](FixtureKind kind,
                                        llvm::StringRef expected) {
    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        fabricSource(HyperbolicFamily::Sinh, kind), &fabricContext());
    require(test, static_cast<bool>(source),
            "could not parse invalid special-math profile");
    attachContract(test, *source, kind);
    ::fabric::ModuleOp invalidRoot;
    source->walk(
        [&](::fabric::ModuleOp candidate) { invalidRoot = candidate; });
    require(test, static_cast<bool>(invalidRoot),
            "invalid special-math profile has no root");
    expectError(test, loom::fabric::finalizeFabricRoot(invalidRoot, store),
                expected);
  };
  expectInvalidProfile(FixtureKind::UnsupportedRounding,
                       "requires exactly round-to-nearest-even");
  expectInvalidProfile(FixtureKind::InvalidSubnormal,
                       "subnormal behavior domain must contain only Preserve");

  FabricFixture fixture =
      makeFabric(test, store, HyperbolicFamily::Sinh, FixtureKind::Configured);
  expectError(test,
              finalizeConfigurationABI(makeConfigurationAbiDraft(
                                           test, fixture, AbiKind::MissingMode),
                                       store),
              "does not equal its Fabric relation");
  expectError(
      test,
      finalizeConfigurationABI(
          makeConfigurationAbiDraft(test, fixture, AbiKind::ExtraMode), store),
      "outside the finite behavior domain");

  expectError(
      test,
      finalizeConfigurationABI(
          makeConfigurationAbiDraft(test, fixture, AbiKind::DirectEncoding),
          store),
      "finite codebook");

  FinalizedConfigurationABI complete =
      makeConfigurationAbi(test, store, fixture);
  std::unique_ptr<mlir::MLIRContext> portContext = makeCirctContext();
  SkeletonFixture wrongPorts =
      makeSkeleton(test, *portContext, fixture, complete.abi(), true);
  expectError(test,
              trySpecialize(std::move(wrongPorts), fixture, complete,
                            makeRegistry(test)),
              "derived contract");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  configuredSemanticsAndArtifacts(root);
  physicalOccurrencesOwnConfiguration(root / "occurrences");
  singletonRelationsAreFieldless(root / "singleton");
  unsupportedBoundaryIsTransactional(root / "unsupported");
  packageRegistrationRollsBack();
  malformedInputsFailClosed(root / "malformed");
  return EXIT_SUCCESS;
}

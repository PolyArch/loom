#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FloatCompareMinMax.h"
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
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
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

enum class FamilyKind { Scalar, FixedVector };

enum class FixtureKind {
  Configured,
  ScalarSingleton,
  VectorSingleton,
  NnanSingleton,
  NnanOrderingSingleton,
  UnsupportedContract,
};

enum class AbiKind { Complete, MissingBehavior, ExtraBehavior, DirectBits };

struct FabricFixture final {
  FamilyKind family;
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

struct Mode final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  ::fabric::FloatFormat format;
  std::optional<mlir::arith::CmpFPredicate> predicate;
  unsigned laneCount;
  std::uint8_t physicalCode;

  bool isCompare() const {
    return actor.schema == ::dataflow::OperationSchemaId::ArithCmpF;
  }
};

struct EmittedProvider final {
  std::string systemVerilog;
  std::vector<Mode> modes;
  std::size_t inactiveMode = 0;
};

struct InputPair final {
  std::uint64_t lhs;
  std::uint64_t rhs;
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
    fail(test, "accepted malformed float compare/minmax input");
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

::fabric::ImplementationFamilyId familyId(FamilyKind family) {
  return family == FamilyKind::Scalar
             ? ::fabric::ImplementationFamilyId::ScalarFloatCompareMinMax
             : ::fabric::ImplementationFamilyId::FixedVectorFloatCompareMinMax;
}

llvm::StringRef familyKeyword(FamilyKind family) {
  return family == FamilyKind::Scalar ? "ScalarFloatCompareMinMax"
                                      : "FixedVectorFloatCompareMinMax";
}

llvm::StringRef configuredModuleName(FamilyKind family) {
  return family == FamilyKind::Scalar ? "scalar_float_compare_min_max"
                                      : "fixed_vector_float_compare_min_max";
}

std::string fabricSource(FamilyKind family, FixtureKind kind) {
  const bool scalarSingleton = kind == FixtureKind::ScalarSingleton;
  const bool vectorSingleton = kind == FixtureKind::VectorSingleton;
  const bool nnanSingleton = kind == FixtureKind::NnanSingleton;
  const bool nnanOrdering = kind == FixtureKind::NnanOrderingSingleton;
  const bool nnan = nnanSingleton || nnanOrdering;
  const bool singleton = scalarSingleton || vectorSingleton || nnan;
  const unsigned portWidth = family == FamilyKind::Scalar
                                 ? (scalarSingleton || nnanSingleton ? 32
                                    : nnanOrdering                   ? 16
                                                                     : 64)
                                 : 80;
  const unsigned resultWidth = portWidth;
  const llvm::StringRef schemas =
      scalarSingleton || nnan ? "@arith.cmpf"
      : vectorSingleton       ? "@arith.minimumf"
                        : "@arith.cmpf, @arith.minimumf, @arith.maximumf, "
                          "@arith.minnumf, @arith.maxnumf";
  const llvm::StringRef formats =
      nnanSingleton  ? R"mlir(["f16", "f32"])mlir"
      : nnanOrdering ? R"mlir(["f16", "bf16"])mlir"
      : singleton    ? R"mlir(["f32"])mlir"
                     : R"mlir(["f16", "bf16", "f32", "f64"])mlir";
  const llvm::StringRef nanBehaviors =
      singleton ? R"mlir(["ieee"])mlir"
                : R"mlir(["ieee", "number_preferred"])mlir";
  const llvm::StringRef predicates =
      scalarSingleton || nnanSingleton ? R"mlir(["uno"])mlir"
      : nnanOrdering                   ? R"mlir(["ugt"])mlir"
      : vectorSingleton
          ? R"mlir(["olt"])mlir"
          : R"mlir(["false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno", "true"])mlir";
  const llvm::StringRef formatField =
      family == FamilyKind::Scalar ? "float_formats" : "element_formats";

  std::string text;
  llvm::raw_string_ostream source(text);
  source << "module { fabric.module @" << configuredModuleName(family)
         << "(%a: !fabric.bits<" << portWidth << ">, %b: !fabric.bits<"
         << portWidth << ">) -> !fabric.bits<" << resultWidth
         << "> { %pe = fabric.pe [spatial]"
         << "(%pa = %a : !fabric.bits<" << portWidth
         << ">, %pb = %b : !fabric.bits<" << portWidth << ">) -> !fabric.bits<"
         << resultWidth << "> { %fu = fabric.fu"
         << "(%fa = %pa : !fabric.bits<" << portWidth
         << ">, %fb = %pb : !fabric.bits<" << portWidth << ">) -> !fabric.bits<"
         << resultWidth << "> { %value = fabric.op [" << schemas
         << "] (%fa, %fb) {implementation_family = "
         << "#fabric.implementation_family<" << familyKeyword(family)
         << ">, hw_params = {" << formatField << " = " << formats
         << ", behavior = {rounding_modes = [\"to_nearest_even\"], "
            "nan_behaviors = "
         << nanBehaviors
         << ", subnormal_behaviors = [\"preserve\"], "
            "signed_zero_behaviors = [\"preserve\"], fastmath = \""
         << (nnan ? "nnan" : "none") << "\"}, predicates = " << predicates;
  if (family == FamilyKind::FixedVector)
    source << ", max_payload_bits = 80 : i32";
  source << "}} : (!fabric.bits<" << portWidth << ">, !fabric.bits<"
         << portWidth << ">) -> !fabric.bits<" << resultWidth
         << "> fabric.yield %value : !fabric.bits<" << resultWidth
         << "> } } fabric.yield %pe : !fabric.bits<" << resultWidth << "> } }";
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
                         FamilyKind family,
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
              "System has no physical float compare/minmax occurrence");
      return FabricFixture{family, std::move(fabric), occurrence,
                           std::move(system), physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no float compare/minmax occurrence");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

::fabric::FloatFormat formatOf(llvm::StringRef test, mlir::Type type) {
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type))
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

unsigned bitWidth(::fabric::FloatFormat format) {
  switch (format) {
  case ::fabric::FloatFormat::F16:
  case ::fabric::FloatFormat::BF16:
    return 16;
  case ::fabric::FloatFormat::F32:
    return 32;
  case ::fabric::FloatFormat::F64:
    return 64;
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

unsigned laneCount(const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(actor.type.getInput(0)))
    return static_cast<unsigned>(vector.getNumElements());
  return 1;
}

std::optional<mlir::arith::CmpFPredicate>
predicateOf(llvm::StringRef test,
            const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (actor.schema != ::dataflow::OperationSchemaId::ArithCmpF)
    return std::nullopt;
  const auto *payload =
      std::get_if<::dataflow::FloatComparePayload>(&actor.payload);
  require(test, payload != nullptr,
          "Fabric projected comparison without a predicate");
  return payload->predicate;
}

std::uint8_t physicalCode(std::size_t ordinal) {
  return static_cast<std::uint8_t>(((ordinal * 73 + 19) % 251) + 1);
}

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "configured compare/minmax capability has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "configured compare/minmax relation is not finite");
  const auto &domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 74,
          "configured compare/minmax domain omitted a sealed behavior");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  entries.reserve(domain.size());
  for (auto [ordinal, point] : llvm::enumerate(domain)) {
    require(test, point.semanticConfiguration.has_value(),
            "configured compare/minmax behavior has no semantic value");
    if (kind == AbiKind::MissingBehavior && ordinal == 11)
      continue;
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (point.representativeActor.schema ==
            ::dataflow::OperationSchemaId::ArithMinimumF &&
        formatOf(test, point.representativeActor.type.getInput(0)) ==
            ::fabric::FloatFormat::F32)
      inactive = semantic;
    entries.push_back({std::move(semantic), {physicalCode(ordinal)}});
  }
  require(test, !inactive.empty(),
          "compare/minmax domain has no inactive minimum behavior");
  std::reverse(entries.begin(), entries.end());
  if (kind == AbiKind::ExtraBehavior)
    entries.push_back({{0xfe}, {0xff}});
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  SemanticFieldEncoding encoding =
      kind == AbiKind::DirectBits
          ? SemanticFieldEncoding{DirectBitsEncoding{8}}
          : SemanticFieldEncoding{
                FiniteCodebookEncoding{8, std::move(entries)}};
  if (kind == AbiKind::DirectBits)
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
                             const FabricFixture &fixture,
                             const ConfigurationABI &abi,
                             llvm::StringRef moduleName,
                             bool wrongConfigurationWidth = false) {
  const auto &resolved = capability(test, fixture);
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports =
      take(test, deriveFabricOperationLeafPorts(
                     builder, fixture.physicalOccurrence, resolved, abi));
  if (wrongConfigurationWidth) {
    auto field = llvm::find_if(
        ports, [](const auto &port) { return port.getName() == "config_0"; });
    require(test, field != ports.end(),
            "configured compare/minmax leaf has no selector");
    field->type = builder.getI1Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(moduleName), ports);
  return SkeletonFixture{std::move(module), leaf};
}

FabricOperationProviderRegistry makeProviderRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableFloatCompareMinMaxProviders(registry))
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
    SkeletonFixture &skeleton, const FabricFixture &fixture,
    const FinalizedConfigurationABI &abi,
    FabricOperationProviderRegistry &registry,
    BackendRecipeKey recipe = BackendRecipeKey::PortableSystemVerilog) {
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fixture.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fixture.physicalOccurrence, recipe, {}}};
  return specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                         recipes, registry, externalContracts);
}

std::string specialize(llvm::StringRef test, SkeletonFixture skeleton,
                       const FabricFixture &fixture,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, fixture.physicalOccurrence}}};
  auto conformance =
      take(test, loom::hardware::test::specializeAndExportPortableProvider(
                     std::move(module), abi, registry, externalContracts));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "portable compare/minmax emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

std::vector<Mode> projectedModes(llvm::StringRef test,
                                 const FabricFixture &fixture,
                                 const ConfigurationABI &abi) {
  const auto &resolved = capability(test, fixture);
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  const auto &domain = relation.finiteBehaviorDomain();
  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  if (!resolved.configurationFieldSchema.empty()) {
    field = abi.findOperationField(
        fixture.physicalOccurrence,
        resolved.configurationFieldSchema.front().ordinal);
    require(test, field != nullptr,
            "configured compare/minmax field is absent from ABI2");
    const ConfigurationEncodingRelation *encodingRelation =
        abi.findEncodingRelation(*field);
    require(test, encodingRelation != nullptr,
            "configured compare/minmax field has no encoding relation");
    codebook = std::get_if<FiniteCodebookEncoding>(
        &encodingRelation->semanticEncoding);
    require(test, codebook != nullptr,
            "configured compare/minmax field is not a finite codebook");
  }

  std::vector<Mode> modes;
  modes.reserve(domain.size());
  for (const auto &point : domain) {
    std::uint8_t code = 0;
    if (point.semanticConfiguration) {
      require(test, codebook != nullptr,
              "configured behavior has no physical codebook");
      const auto entry = llvm::find_if(
          codebook->entries, [&](const FiniteCodebookEntry &candidate) {
            return llvm::ArrayRef<std::uint8_t>(candidate.semanticValue)
                .equals(point.semanticConfiguration->bytes());
          });
      require(test,
              entry != codebook->entries.end() &&
                  entry->physicalCode.size() == 1,
              "sealed behavior is absent from the arbitrary codebook");
      code = entry->physicalCode.front();
    }
    modes.push_back({point.representativeActor,
                     formatOf(test, point.representativeActor.type.getInput(0)),
                     predicateOf(test, point.representativeActor),
                     laneCount(point.representativeActor), code});
  }
  return modes;
}

std::size_t inactiveMode(llvm::StringRef test, llvm::ArrayRef<Mode> modes,
                         const FabricFixture &fixture,
                         const ConfigurationABI &abi) {
  const auto &resolved = capability(test, fixture);
  require(test, resolved.configurationFieldSchema.size() == 1,
          "configured compare/minmax capability is fieldless");
  const ConfigurationFieldEncoding *field =
      abi.findOperationField(fixture.physicalOccurrence,
                             resolved.configurationFieldSchema.front().ordinal);
  require(test, field != nullptr,
          "configured compare/minmax field is absent from ABI2");
  const ConfigurationEncodingRelation *encodingRelation =
      abi.findEncodingRelation(*field);
  require(test, encodingRelation != nullptr,
          "configured compare/minmax field has no encoding relation");
  const auto *codebook =
      std::get_if<FiniteCodebookEncoding>(&encodingRelation->semanticEncoding);
  require(test, codebook != nullptr,
          "configured compare/minmax field is not a codebook");
  const auto entry = llvm::find_if(
      codebook->entries, [&](const FiniteCodebookEntry &candidate) {
        return llvm::ArrayRef<std::uint8_t>(candidate.semanticValue)
            .equals(encodingRelation->inactiveValue);
      });
  require(test, entry != codebook->entries.end(),
          "ABI2 inactive value is outside the behavior domain");
  const auto mode = llvm::find_if(modes, [&](const Mode &candidate) {
    return candidate.physicalCode == entry->physicalCode.front();
  });
  require(test, mode != modes.end(),
          "inactive physical code has no lowered behavior");
  return static_cast<std::size_t>(mode - modes.begin());
}

EmittedProvider emitProvider(llvm::StringRef test, const ArtifactStore &store,
                             FamilyKind family, FixtureKind kind,
                             llvm::StringRef moduleName) {
  FabricFixture fixture = makeFabric(test, store, family, kind);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
  std::vector<Mode> modes = projectedModes(test, fixture, abi.abi());
  std::size_t inactive = 0;
  if (!capability(test, fixture).configurationFieldSchema.empty())
    inactive = inactiveMode(test, modes, fixture, abi.abi());

  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first =
      makeSkeleton(test, *firstContext, fixture, abi.abi(), moduleName);
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  const bool configured = kind == FixtureKind::Configured;
  require(test,
          ports.size() == (configured ? 4 : 3) &&
              ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atOutput(0).getName() == "data_output_0",
          "compare/minmax leaf ports do not follow ABI2 geometry");
  if (configured)
    require(test,
            ports.atInput(2).getName() == "config_0" &&
                ports.atInput(2).type ==
                    mlir::IntegerType::get(firstContext.get(), 8),
            "configured compare/minmax selector is not the ABI2 field");
  std::string firstRtl = specialize(test, std::move(first), fixture, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fixture, abi.abi(), moduleName);
  const std::string secondRtl =
      specialize(test, std::move(second), fixture, abi);
  require(test, firstRtl == secondRtl,
          "identical compare/minmax inputs produced different SystemVerilog");
  const llvm::StringRef rtl(firstRtl);
  require(test,
          rtl.contains("function automatic") &&
              rtl.contains("loom_float_compare_min_max") &&
              rtl.contains("config_0") == configured &&
              !rtl.contains("shortreal") && !rtl.contains(" DPI") &&
              !rtl.contains(" real"),
          "compare/minmax RTL is incomplete or not synthesizable");
  return EmittedProvider{std::move(firstRtl), std::move(modes), inactive};
}

llvm::APInt evaluate(llvm::StringRef test, const Mode &mode,
                     std::uint64_t lhsBits, std::uint64_t rhsBits) {
  const unsigned width = bitWidth(mode.format);
  const llvm::APFloat lhs(semantics(mode.format), llvm::APInt(width, lhsBits));
  const llvm::APFloat rhs(semantics(mode.format), llvm::APInt(width, rhsBits));
  const loom::sim::PrimitiveOperationDescriptor descriptor{
      mode.actor, mode.isCompare() ? 1U : width, width};
  const std::array operands = {loom::sim::PrimitiveValue::floating(lhs),
                               loom::sim::PrimitiveValue::floating(rhs)};
  auto result =
      take(test, loom::sim::evaluatePrimitiveOperation(descriptor, operands));
  require(test, result.isDefined(),
          "strict APFloat/Simulator oracle produced a non-defined result");
  return *result.bits;
}

std::vector<InputPair> inputPairs(const Mode &mode) {
  const unsigned width = bitWidth(mode.format);
  const unsigned fractionBits = mode.format == ::fabric::FloatFormat::F16 ? 10
                                : mode.format == ::fabric::FloatFormat::BF16 ? 7
                                : mode.format == ::fabric::FloatFormat::F32
                                    ? 23
                                    : 52;
  const unsigned exponentBits = width - fractionBits - 1;
  const std::uint64_t sign = std::uint64_t{1} << (width - 1);
  const std::uint64_t exponentMask = (std::uint64_t{1} << exponentBits) - 1;
  const std::uint64_t infinity = exponentMask << fractionBits;
  const std::uint64_t one = ((std::uint64_t{1} << (exponentBits - 1)) - 1)
                            << fractionBits;
  const std::uint64_t two = one + (std::uint64_t{1} << fractionBits);
  const std::uint64_t quietBit = std::uint64_t{1} << (fractionBits - 1);
  const std::uint64_t quietNaN = infinity | quietBit | 5;
  const std::uint64_t otherQuietNaN = sign | infinity | quietBit | 3;
  const std::uint64_t signalingNaN = infinity | 3;
  const std::uint64_t otherSignalingNaN = sign | infinity | 1;
  const std::uint64_t negativeOne = sign | one;
  const std::uint64_t negativeTwo = sign | two;
  const std::uint64_t maximumFinite = infinity - 1;
  const std::uint64_t minimumNormal = std::uint64_t{1} << fractionBits;
  const std::uint64_t maximumSubnormal = minimumNormal - 1;

  if (mode.isCompare())
    return {{one, two},
            {two, one},
            {one, one},
            {0, sign},
            {negativeOne, one},
            {negativeOne, negativeTwo},
            {negativeTwo, negativeOne},
            {maximumSubnormal, minimumNormal},
            {maximumFinite, infinity},
            {quietNaN, one},
            {one, otherQuietNaN},
            {signalingNaN, one},
            {one, otherSignalingNaN}};
  return {{one, two},
          {two, one},
          {one, one},
          {0, sign},
          {sign, 0},
          {negativeOne, one},
          {negativeOne, negativeTwo},
          {negativeTwo, negativeOne},
          {maximumSubnormal, minimumNormal},
          {quietNaN, one},
          {one, otherQuietNaN},
          {signalingNaN, one},
          {one, otherSignalingNaN},
          {quietNaN, otherQuietNaN},
          {signalingNaN, otherSignalingNaN}};
}

std::string hexLiteral(unsigned width, const llvm::APInt &value) {
  llvm::SmallString<64> digits;
  value.toStringUnsigned(digits, 16);
  const unsigned digitCount = (width + 3) / 4;
  return std::to_string(width) + "'h" +
         std::string(digitCount - digits.size(), '0') + digits.str().str();
}

llvm::APInt paddedInput(unsigned physicalWidth, unsigned semanticWidth,
                        std::uint64_t value) {
  llvm::APInt result = llvm::APInt::getAllOnes(physicalWidth);
  result.insertBits(llvm::APInt(semanticWidth, value), 0);
  return result;
}

std::string
buildTestbench(llvm::ArrayRef<Mode> scalarModes, std::size_t scalarInactive,
               llvm::ArrayRef<Mode> vectorModes, std::size_t vectorInactive,
               const Mode &scalarSingleton, const Mode &vectorSingleton) {
  const llvm::StringRef test = "buildTestbench";
  std::string text;
  llvm::raw_string_ostream output(text);
  output << R"sv(
module testbench;
  logic [63:0] scalar_lhs;
  logic [63:0] scalar_rhs;
  logic [7:0] scalar_config;
  logic [63:0] scalar_result;
  logic [79:0] vector_lhs;
  logic [79:0] vector_rhs;
  logic [7:0] vector_config;
  logic [79:0] vector_result;
  logic [31:0] scalar_singleton_lhs;
  logic [31:0] scalar_singleton_rhs;
  logic [31:0] scalar_singleton_result;
  logic [79:0] vector_singleton_lhs;
  logic [79:0] vector_singleton_rhs;
  logic [79:0] vector_singleton_result;
  logic [31:0] scalar_nnan_lhs;
  logic [31:0] scalar_nnan_rhs;
  logic [31:0] scalar_nnan_result;
  logic [79:0] vector_nnan_lhs;
  logic [79:0] vector_nnan_rhs;
  logic [79:0] vector_nnan_result;
  logic [15:0] scalar_nnan_ordering_lhs;
  logic [15:0] scalar_nnan_ordering_rhs;
  logic [15:0] scalar_nnan_ordering_result;
  logic [79:0] vector_nnan_ordering_lhs;
  logic [79:0] vector_nnan_ordering_rhs;
  logic [79:0] vector_nnan_ordering_result;

  scalar_float_compare_min_max scalar_dut(
      .data_input_0(scalar_lhs),
      .data_input_1(scalar_rhs),
      .config_0(scalar_config),
      .data_output_0(scalar_result));
  fixed_vector_float_compare_min_max vector_dut(
      .data_input_0(vector_lhs),
      .data_input_1(vector_rhs),
      .config_0(vector_config),
      .data_output_0(vector_result));
  scalar_float_compare_singleton scalar_singleton_dut(
      .data_input_0(scalar_singleton_lhs),
      .data_input_1(scalar_singleton_rhs),
      .data_output_0(scalar_singleton_result));
  fixed_vector_float_minimum_singleton vector_singleton_dut(
      .data_input_0(vector_singleton_lhs),
      .data_input_1(vector_singleton_rhs),
      .data_output_0(vector_singleton_result));
  scalar_float_compare_nnan_singleton scalar_nnan_dut(
      .data_input_0(scalar_nnan_lhs),
      .data_input_1(scalar_nnan_rhs),
      .data_output_0(scalar_nnan_result));
  fixed_vector_float_compare_nnan_singleton vector_nnan_dut(
      .data_input_0(vector_nnan_lhs),
      .data_input_1(vector_nnan_rhs),
      .data_output_0(vector_nnan_result));
  scalar_float_compare_nnan_ordering_singleton scalar_nnan_ordering_dut(
      .data_input_0(scalar_nnan_ordering_lhs),
      .data_input_1(scalar_nnan_ordering_rhs),
      .data_output_0(scalar_nnan_ordering_result));
  fixed_vector_float_compare_nnan_ordering_singleton vector_nnan_ordering_dut(
      .data_input_0(vector_nnan_ordering_lhs),
      .data_input_1(vector_nnan_ordering_rhs),
      .data_output_0(vector_nnan_ordering_result));

  task automatic check_scalar(
      input logic [7:0] mode,
      input logic [63:0] lhs,
      input logic [63:0] rhs,
      input logic [63:0] expected);
    begin
      scalar_config = mode;
      scalar_lhs = lhs;
      scalar_rhs = rhs;
      #1;
      if (scalar_result !== expected)
        $fatal(1, "scalar mismatch mode=%0d lhs=%h rhs=%h got=%h expected=%h",
               mode, lhs, rhs, scalar_result, expected);
    end
  endtask

  task automatic check_vector(
      input logic [7:0] mode,
      input logic [79:0] lhs,
      input logic [79:0] rhs,
      input logic [79:0] expected);
    begin
      vector_config = mode;
      vector_lhs = lhs;
      vector_rhs = rhs;
      #1;
      if (vector_result !== expected)
        $fatal(1, "vector mismatch mode=%0d lhs=%h rhs=%h got=%h expected=%h",
               mode, lhs, rhs, vector_result, expected);
    end
  endtask

  initial begin
)sv";

  for (const Mode &mode : scalarModes) {
    for (const InputPair pair : inputPairs(mode)) {
      const unsigned width = bitWidth(mode.format);
      const llvm::APInt expected = evaluate(test, mode, pair.lhs, pair.rhs);
      output << "    check_scalar(8'd" << unsigned(mode.physicalCode) << ", "
             << hexLiteral(64, paddedInput(64, width, pair.lhs)) << ", "
             << hexLiteral(64, paddedInput(64, width, pair.rhs)) << ", "
             << hexLiteral(64, expected.zext(64)) << ");\n";
    }
  }

  for (auto [modeOrdinal, mode] : llvm::enumerate(vectorModes)) {
    const std::vector<InputPair> pairs = inputPairs(mode);
    const unsigned width = bitWidth(mode.format);
    const unsigned resultWidth = mode.isCompare() ? 1 : width;
    for (unsigned batch = 0; batch != 2; ++batch) {
      llvm::APInt lhs = llvm::APInt::getAllOnes(80);
      llvm::APInt rhs = llvm::APInt::getAllOnes(80);
      llvm::APInt expected(80, 0);
      for (unsigned lane = 0; lane != mode.laneCount; ++lane) {
        const InputPair pair =
            pairs[(modeOrdinal + batch * mode.laneCount + lane) % pairs.size()];
        lhs.insertBits(llvm::APInt(width, pair.lhs), lane * width);
        rhs.insertBits(llvm::APInt(width, pair.rhs), lane * width);
        expected.insertBits(evaluate(test, mode, pair.lhs, pair.rhs),
                            lane * resultWidth);
      }
      output << "    check_vector(8'd" << unsigned(mode.physicalCode) << ", "
             << hexLiteral(80, lhs) << ", " << hexLiteral(80, rhs) << ", "
             << hexLiteral(80, expected) << ");\n";
    }
  }

  const InputPair fallbackPair{0x3f800000U, 0xbf800000U};
  const Mode &scalarFallback = scalarModes[scalarInactive];
  const Mode &vectorFallback = vectorModes[vectorInactive];
  output << "    check_scalar(8'd0, "
         << hexLiteral(64, paddedInput(64, 32, fallbackPair.lhs)) << ", "
         << hexLiteral(64, paddedInput(64, 32, fallbackPair.rhs)) << ", "
         << hexLiteral(64, evaluate(test, scalarFallback, fallbackPair.lhs,
                                    fallbackPair.rhs)
                               .zext(64))
         << ");\n";
  llvm::APInt fallbackVectorLhs = llvm::APInt::getAllOnes(80);
  llvm::APInt fallbackVectorRhs = llvm::APInt::getAllOnes(80);
  llvm::APInt fallbackVectorExpected(80, 0);
  const unsigned fallbackWidth = bitWidth(vectorFallback.format);
  for (unsigned lane = 0; lane != vectorFallback.laneCount; ++lane) {
    fallbackVectorLhs.insertBits(llvm::APInt(fallbackWidth, fallbackPair.lhs),
                                 lane * fallbackWidth);
    fallbackVectorRhs.insertBits(llvm::APInt(fallbackWidth, fallbackPair.rhs),
                                 lane * fallbackWidth);
    fallbackVectorExpected.insertBits(
        evaluate(test, vectorFallback, fallbackPair.lhs, fallbackPair.rhs),
        lane * fallbackWidth);
  }
  output << "    check_vector(8'd0, " << hexLiteral(80, fallbackVectorLhs)
         << ", " << hexLiteral(80, fallbackVectorRhs) << ", "
         << hexLiteral(80, fallbackVectorExpected) << ");\n";

  const std::vector<InputPair> scalarSingletonPairs =
      inputPairs(scalarSingleton);
  const auto scalarQuietNaN =
      llvm::find_if(scalarSingletonPairs, [&](const InputPair &pair) {
        const llvm::APFloat lhs(
            semantics(scalarSingleton.format),
            llvm::APInt(bitWidth(scalarSingleton.format), pair.lhs));
        return lhs.isNaN() && !lhs.isSignaling();
      });
  require(test, scalarQuietNaN != scalarSingletonPairs.end(),
          "fieldless comparison vectors have no quiet NaN");
  const InputPair scalarSingletonPair = *scalarQuietNaN;
  const llvm::APInt scalarSingletonExpected = evaluate(
      test, scalarSingleton, scalarSingletonPair.lhs, scalarSingletonPair.rhs);
  output << "    scalar_singleton_lhs = "
         << hexLiteral(32, llvm::APInt(32, scalarSingletonPair.lhs)) << ";\n"
         << "    scalar_singleton_rhs = "
         << hexLiteral(32, llvm::APInt(32, scalarSingletonPair.rhs)) << ";\n"
         << "    #1;\n"
         << "    if (scalar_singleton_result !== 32'd"
         << (scalarSingletonExpected.isOne() ? '1' : '0')
         << ") $fatal(1, \"fieldless unordered compare failed\");\n";

  const std::vector<InputPair> singletonPairs = inputPairs(vectorSingleton);
  const std::uint64_t singletonSign = std::uint64_t{1}
                                      << (bitWidth(vectorSingleton.format) - 1);
  const auto signedZeros =
      llvm::find_if(singletonPairs, [&](const InputPair &pair) {
        return pair.lhs == 0 && pair.rhs == singletonSign;
      });
  const auto quietNaN =
      llvm::find_if(singletonPairs, [&](const InputPair &pair) {
        const llvm::APFloat lhs(
            semantics(vectorSingleton.format),
            llvm::APInt(bitWidth(vectorSingleton.format), pair.lhs));
        return lhs.isNaN() && !lhs.isSignaling();
      });
  require(test,
          vectorSingleton.laneCount == 2 &&
              signedZeros != singletonPairs.end() &&
              quietNaN != singletonPairs.end(),
          "fieldless vector minimum lacks its signed-zero or quiet-NaN case");
  llvm::APInt singletonLhs = llvm::APInt::getAllOnes(80);
  llvm::APInt singletonRhs = llvm::APInt::getAllOnes(80);
  llvm::APInt singletonExpected(80, 0);
  for (unsigned lane = 0; lane != vectorSingleton.laneCount; ++lane) {
    const InputPair pair = lane == 0 ? *signedZeros : *quietNaN;
    singletonLhs.insertBits(llvm::APInt(32, pair.lhs), lane * 32);
    singletonRhs.insertBits(llvm::APInt(32, pair.rhs), lane * 32);
    singletonExpected.insertBits(
        evaluate(test, vectorSingleton, pair.lhs, pair.rhs), lane * 32);
  }
  output << "    vector_singleton_lhs = " << hexLiteral(80, singletonLhs)
         << ";\n"
         << "    vector_singleton_rhs = " << hexLiteral(80, singletonRhs)
         << ";\n"
         << "    #1;\n"
         << "    if (vector_singleton_result !== "
         << hexLiteral(80, singletonExpected)
         << ") $fatal(1, \"fieldless vector minimum failed\");\n"
         << "    scalar_nnan_lhs = 32'h00007e01;\n"
         << "    scalar_nnan_rhs = 32'h3f800000;\n"
         << "    #1;\n"
         << "    if (scalar_nnan_result !== 32'd0) "
            "$fatal(1, \"fieldless scalar nnan normalization failed\");\n"
         << "    vector_nnan_lhs = 80'h00000000fe0100007e01;\n"
         << "    vector_nnan_rhs = 80'h0000bf8000003f800000;\n"
         << "    #1;\n"
         << "    if (vector_nnan_result !== 80'd0) "
            "$fatal(1, \"fieldless vector nnan normalization failed\");\n"
         << "    scalar_nnan_ordering_lhs = 16'hfc00;\n"
         << "    scalar_nnan_ordering_rhs = 16'h3f80;\n"
         << "    #1;\n"
         << "    if (scalar_nnan_ordering_result !== 16'd0) "
            "$fatal(1, \"fieldless scalar nnan ordering failed\");\n"
         << "    vector_nnan_ordering_lhs = "
            "80'hfc00fc00fc00fc00fc00;\n"
         << "    vector_nnan_ordering_rhs = "
            "80'h3f803f803f803f803f80;\n"
         << "    #1;\n"
         << "    if (vector_nnan_ordering_result !== 80'd0) "
            "$fatal(1, \"fieldless vector nnan ordering failed\");\n"
         << "    $finish;\n"
         << "  end\n"
         << "endmodule\n";
  return output.str();
}

std::string synthesisTop() {
  return R"sv(
module float_compare_min_max_synthesis_top(
    input logic [63:0] scalar_lhs,
    input logic [63:0] scalar_rhs,
    input logic [7:0] scalar_config,
    input logic [79:0] vector_lhs,
    input logic [79:0] vector_rhs,
    input logic [7:0] vector_config,
    output logic [63:0] scalar_result,
    output logic [79:0] vector_result,
    output logic [31:0] scalar_nnan_result,
    output logic [79:0] vector_nnan_result,
    output logic [15:0] scalar_nnan_ordering_result,
    output logic [79:0] vector_nnan_ordering_result);
  scalar_float_compare_min_max scalar_dut(
      .data_input_0(scalar_lhs), .data_input_1(scalar_rhs),
      .config_0(scalar_config), .data_output_0(scalar_result));
  fixed_vector_float_compare_min_max vector_dut(
      .data_input_0(vector_lhs), .data_input_1(vector_rhs),
      .config_0(vector_config), .data_output_0(vector_result));
  scalar_float_compare_nnan_singleton scalar_nnan_dut(
      .data_input_0(scalar_lhs[31:0]), .data_input_1(scalar_rhs[31:0]),
      .data_output_0(scalar_nnan_result));
  fixed_vector_float_compare_nnan_singleton vector_nnan_dut(
      .data_input_0(vector_lhs), .data_input_1(vector_rhs),
      .data_output_0(vector_nnan_result));
  scalar_float_compare_nnan_ordering_singleton scalar_nnan_ordering_dut(
      .data_input_0(scalar_lhs[15:0]), .data_input_1(scalar_rhs[15:0]),
      .data_output_0(scalar_nnan_ordering_result));
  fixed_vector_float_compare_nnan_ordering_singleton vector_nnan_ordering_dut(
      .data_input_0(vector_lhs), .data_input_1(vector_rhs),
      .data_output_0(vector_nnan_ordering_result));
endmodule
)sv";
}

std::string yosysScript() {
  return R"ys(
read_verilog -sv scalar_float_compare_min_max.sv
read_verilog -sv fixed_vector_float_compare_min_max.sv
read_verilog -sv scalar_float_compare_nnan_singleton.sv
read_verilog -sv fixed_vector_float_compare_nnan_singleton.sv
read_verilog -sv scalar_float_compare_nnan_ordering_singleton.sv
read_verilog -sv fixed_vector_float_compare_nnan_ordering_singleton.sv
read_verilog -sv synthesis_top.sv
hierarchy -check -top float_compare_min_max_synthesis_top
proc
opt
check -assert
select -assert-none t:$dff t:$dlatch t:$memrd t:$memwr t:$meminit t:$mem_v2
synth -noabc -top float_compare_min_max_synthesis_top
check -assert
select -assert-none t:$_DFF_* t:$_SDFF_* t:$_DLATCH_* t:$memrd t:$memwr t:$meminit t:$mem_v2
stat
)ys";
}

void providerRegistrationIsTransactional() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto before = registry.coverage();
  require(test,
          llvm::count_if(
              before,
              [](const auto &entry) { return !entry.recipes.empty(); }) == 2,
          "compare/minmax registration did not cover both families");
  for (FamilyKind family : {FamilyKind::Scalar, FamilyKind::FixedVector}) {
    const auto entry = llvm::find_if(before, [&](const auto &candidate) {
      return candidate.implementationFamily == familyId(family);
    });
    require(test,
            entry != before.end() &&
                entry->recipes ==
                    std::vector<BackendRecipeKey>{
                        BackendRecipeKey::PortableSystemVerilog},
            "compare/minmax registration has the wrong recipe coverage");
  }

  llvm::Error duplicate = registerPortableFloatCompareMinMaxProviders(registry);
  require(test, static_cast<bool>(duplicate),
          "duplicate compare/minmax registration succeeded");
  llvm::consumeError(std::move(duplicate));
  const auto after = registry.coverage();
  require(test, after.size() == before.size(),
          "failed registration changed provider coverage");
  for (auto [lhs, rhs] : llvm::zip(before, after))
    require(test,
            lhs.implementationFamily == rhs.implementationFamily &&
                lhs.recipes == rhs.recipes,
            "failed registration partially changed the provider registry");
}

void malformedAndUnsupportedInputsRollBack(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  for (FamilyKind family : {FamilyKind::Scalar, FamilyKind::FixedVector}) {
    FabricFixture valid = makeFabric(test, store, family);
    FinalizedConfigurationABI validAbi =
        makeConfigurationAbi(test, store, valid);
    std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
    SkeletonFixture malformed = makeSkeleton(
        test, *malformedContext, valid, validAbi.abi(),
        family == FamilyKind::Scalar ? "malformed_scalar" : "malformed_vector",
        true);
    const std::string before = moduleText(*malformed.module);
    expectError(test, trySpecialize(malformed, valid, validAbi, registry),
                "leaf port");
    require(test, moduleText(*malformed.module) == before,
            "malformed compare/minmax input mutated the caller skeleton");

    for (AbiKind kind : {AbiKind::MissingBehavior, AbiKind::ExtraBehavior,
                         AbiKind::DirectBits})
      expectError(test,
                  finalizeConfigurationABI(
                      makeConfigurationAbiDraft(test, valid, kind), store),
                  kind == AbiKind::ExtraBehavior
                      ? "outside the finite behavior domain"
                      : "finite codebook");

    FabricFixture unsupported =
        makeFabric(test, store, family, FixtureKind::UnsupportedContract);
    FinalizedConfigurationABI unsupportedAbi =
        makeConfigurationAbi(test, store, unsupported);
    std::unique_ptr<mlir::MLIRContext> unsupportedContext = makeCirctContext();
    SkeletonFixture unsupportedSkeleton = makeSkeleton(
        test, *unsupportedContext, unsupported, unsupportedAbi.abi(),
        family == FamilyKind::Scalar ? "unsupported_scalar"
                                     : "unsupported_vector");
    const std::string unsupportedBefore =
        moduleText(*unsupportedSkeleton.module);
    auto unsupportedResult = trySpecialize(unsupportedSkeleton, unsupported,
                                           unsupportedAbi, registry);
    require(test, !unsupportedResult,
            "unsupported compare/minmax resource contract specialized");
    bool classified = false;
    llvm::handleAllErrors(
        unsupportedResult.takeError(),
        [&](const FabricOperationProviderUnsupportedError &error) {
          classified =
              error.implementationFamily() == familyId(family) &&
              error.recipe() == BackendRecipeKey::PortableSystemVerilog;
        },
        [&](const llvm::ErrorInfoBase &error) {
          fail(test, "unsupported contract returned the wrong error class: " +
                         error.message());
        });
    require(test, classified,
            "unsupported contract lost its typed classification");
    require(test, moduleText(*unsupportedSkeleton.module) == unsupportedBefore,
            "unsupported contract mutated the caller skeleton");

    for (BackendRecipeKey recipe :
         {BackendRecipeKey::SynopsysDesignWare,
          BackendRecipeKey::CadenceChipWare, BackendRecipeKey::AmdXilinx,
          BackendRecipeKey::IntelAltera}) {
      std::unique_ptr<mlir::MLIRContext> nativeContext = makeCirctContext();
      SkeletonFixture native = makeSkeleton(
          test, *nativeContext, valid, validAbi.abi(),
          family == FamilyKind::Scalar ? "native_scalar" : "native_vector");
      const std::string nativeBefore = moduleText(*native.module);
      auto nativeResult =
          trySpecialize(native, valid, validAbi, registry, recipe);
      require(test, !nativeResult,
              "unregistered native compare/minmax recipe specialized");
      bool nativeClassified = false;
      llvm::handleAllErrors(
          nativeResult.takeError(),
          [&](const FabricOperationProviderUnsupportedError &error) {
            nativeClassified =
                error.implementationFamily() == familyId(family) &&
                error.recipe() == recipe;
          },
          [&](const llvm::ErrorInfoBase &error) {
            fail(test, "native recipe returned the wrong error class: " +
                           error.message());
          });
      require(test, nativeClassified,
              "native recipe lost its typed Unsupported classification");
      require(test, moduleText(*native.module) == nativeBefore,
              "unsupported native recipe mutated the caller skeleton");
    }
  }
}

void emitConformanceArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  EmittedProvider scalar =
      emitProvider(test, store, FamilyKind::Scalar, FixtureKind::Configured,
                   "scalar_float_compare_min_max");
  EmittedProvider vector = emitProvider(test, store, FamilyKind::FixedVector,
                                        FixtureKind::Configured,
                                        "fixed_vector_float_compare_min_max");
  EmittedProvider scalarSingleton = emitProvider(
      test, store, FamilyKind::Scalar, FixtureKind::ScalarSingleton,
      "scalar_float_compare_singleton");
  EmittedProvider vectorSingleton = emitProvider(
      test, store, FamilyKind::FixedVector, FixtureKind::VectorSingleton,
      "fixed_vector_float_minimum_singleton");
  EmittedProvider scalarNnan =
      emitProvider(test, store, FamilyKind::Scalar, FixtureKind::NnanSingleton,
                   "scalar_float_compare_nnan_singleton");
  EmittedProvider vectorNnan = emitProvider(
      test, store, FamilyKind::FixedVector, FixtureKind::NnanSingleton,
      "fixed_vector_float_compare_nnan_singleton");
  EmittedProvider scalarNnanOrdering = emitProvider(
      test, store, FamilyKind::Scalar, FixtureKind::NnanOrderingSingleton,
      "scalar_float_compare_nnan_ordering_singleton");
  EmittedProvider vectorNnanOrdering = emitProvider(
      test, store, FamilyKind::FixedVector, FixtureKind::NnanOrderingSingleton,
      "fixed_vector_float_compare_nnan_ordering_singleton");
  require(test,
          scalar.modes.size() == 74 && vector.modes.size() == 74 &&
              scalarSingleton.modes.size() == 1 &&
              vectorSingleton.modes.size() == 1 &&
              scalarNnan.modes.size() == 1 && vectorNnan.modes.size() == 1 &&
              scalarNnanOrdering.modes.size() == 1 &&
              vectorNnanOrdering.modes.size() == 1,
          "provider did not consume the complete sealed behavior domains");
  require(
      test,
      llvm::count_if(scalar.modes,
                     [](const Mode &mode) { return mode.isCompare(); }) == 58 &&
          llvm::count_if(vector.modes,
                         [](const Mode &mode) { return mode.isCompare(); }) ==
              58,
      "provider omitted a compare predicate or exact format");

  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"scalar_float_compare_min_max.sv", std::move(scalar.systemVerilog)},
           {"fixed_vector_float_compare_min_max.sv",
            std::move(vector.systemVerilog)},
           {"scalar_float_compare_singleton.sv",
            std::move(scalarSingleton.systemVerilog)},
           {"fixed_vector_float_minimum_singleton.sv",
            std::move(vectorSingleton.systemVerilog)},
           {"scalar_float_compare_nnan_singleton.sv",
            std::move(scalarNnan.systemVerilog)},
           {"fixed_vector_float_compare_nnan_singleton.sv",
            std::move(vectorNnan.systemVerilog)},
           {"scalar_float_compare_nnan_ordering_singleton.sv",
            std::move(scalarNnanOrdering.systemVerilog)},
           {"fixed_vector_float_compare_nnan_ordering_singleton.sv",
            std::move(vectorNnanOrdering.systemVerilog)},
           {"testbench.sv",
            buildTestbench(scalar.modes, scalar.inactiveMode, vector.modes,
                           vector.inactiveMode, scalarSingleton.modes.front(),
                           vectorSingleton.modes.front())},
           {"synthesis_top.sv", synthesisTop()},
           {"portable_float_compare_min_max.ys", yosysScript()}}))
    fail(test, llvm::toString(std::move(error)));
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  emitConformanceArtifacts(root);
  malformedAndUnsupportedInputsRollBack(root / "negative");
  providerRegistrationIsTransactional();
  return EXIT_SUCCESS;
}

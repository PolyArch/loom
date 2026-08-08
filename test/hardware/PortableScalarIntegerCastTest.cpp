#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/ScalarIntegerCast.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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

void expectInvalid(llvm::StringRef test,
                   llvm::Expected<FabricOperationProviderOutput> value,
                   llvm::StringRef expected) {
  require(test, !value,
          "provider accepted malformed scalar integer cast input");
  bool classified = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &) {
        fail(test, "malformed scalar integer cast input became Unsupported");
      },
      [&](const llvm::ErrorInfoBase &error) {
        require(test, llvm::StringRef(error.message()).contains(expected),
                error.message());
        classified = true;
      });
  require(test, classified, "malformed input lost its diagnostic");
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid scalar integer cast input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectTypedUnsupported(llvm::StringRef test,
                            llvm::Expected<FabricOperationProviderOutput> value,
                            BackendRecipeKey expectedRecipe,
                            llvm::StringRef description) {
  require(test, !value, description);
  bool classified = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() ==
                         ::fabric::ImplementationFamilyId::ScalarIntegerCast &&
                     error.recipe() == expectedRecipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, description.str() +
                       " returned the wrong error class: " + error.message());
      });
  require(test, classified,
          description.str() + " lost typed Unsupported classification");
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

std::string attributeText(mlir::Attribute attribute) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return text;
}

const ::fabric::ImplementationFamilyDescriptor &castDescriptor() {
  return ::fabric::implementationFamily(
      ::fabric::ImplementationFamilyId::ScalarIntegerCast);
}

enum class FixtureKind { Configured, IdentityOnly };
enum class ContractKind { OneCycleElastic, Wrong };

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
  FixtureKind kind;
  unsigned inputWidth = 0;
  unsigned outputWidth = 0;
};

::fabric::ScalarIntegerCastParams castParameters(FixtureKind kind) {
  if (kind == FixtureKind::IdentityOnly)
    return {::fabric::IntegerCastRelation{
        ::fabric::IntegerWidthRelation::get(
            {{::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I32}}),
        ::fabric::ResolvedIndexWidthSet::get(
            {::fabric::ResolvedIndexWidth::I32})}};
  return {::fabric::IntegerCastRelation{
      ::fabric::IntegerWidthRelation::get(
          {{::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I32},
           {::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I8},
           {::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I32}}),
      ::fabric::ResolvedIndexWidthSet::get(
          {::fabric::ResolvedIndexWidth::I32})}};
}

std::vector<::dataflow::OperationSchemaId> castSchemas(FixtureKind kind) {
  if (kind == FixtureKind::Configured)
    return std::vector<::dataflow::OperationSchemaId>(
        castDescriptor().admittedSchemas.begin(),
        castDescriptor().admittedSchemas.end());
  return {::dataflow::OperationSchemaId::ArithIndexCast,
          ::dataflow::OperationSchemaId::ArithIndexCastUI};
}

std::string
schemaListText(llvm::ArrayRef<::dataflow::OperationSchemaId> schemas) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  for (auto [index, schema] : llvm::enumerate(schemas)) {
    if (index != 0)
      stream << ", ";
    stream << '@' << ::dataflow::operationSchemaSpelling(schema);
  }
  return text;
}

void attachContract(llvm::StringRef test, mlir::ModuleOp module,
                    ContractKind kind) {
  const ::fabric::ResourceContract contract =
      kind == ContractKind::OneCycleElastic
          ? ::fabric::oneCycleElasticOperationResourceContract()
          : ::fabric::loopCarryOperationResourceContract();
  const std::vector<std::uint8_t> encoded =
      take(test, ::fabric::encodeResourceContractRecord(contract));
  const std::vector<std::int8_t> signedEncoding(encoded.begin(), encoded.end());
  module.walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedEncoding));
  });
}

FabricFixture
makeFabric(llvm::StringRef test, const ArtifactStore &store,
           FixtureKind kind = FixtureKind::Configured,
           ContractKind contract = ContractKind::OneCycleElastic) {
  const unsigned inputWidth = kind == FixtureKind::Configured ? 37 : 35;
  const unsigned outputWidth = kind == FixtureKind::Configured ? 40 : 36;
  const unsigned containerWidth = std::max(inputWidth, outputWidth);
  const ::fabric::FamilyCapabilityParams parameters = castParameters(kind);
  const std::string parametersText = attributeText(
      ::fabric::getFamilyCapabilityParamsAttr(&fabricContext(), parameters));
  const std::vector<::dataflow::OperationSchemaId> schemas = castSchemas(kind);

  std::string sourceText;
  llvm::raw_string_ostream source(sourceText);
  source << "module { fabric.module @scalar_integer_cast(%a: !fabric.bits<"
         << containerWidth << ">) -> !fabric.bits<" << containerWidth
         << "> { %pe = fabric.pe [spatial] (%pa = %a : !fabric.bits<"
         << containerWidth << ">) -> !fabric.bits<" << containerWidth
         << "> { %fu = fabric.fu (%fa = %pa : !fabric.bits<" << containerWidth
         << "> to !fabric.bits<" << inputWidth << ">) -> !fabric.bits<"
         << outputWidth << "> { %value = fabric.op [" << schemaListText(schemas)
         << "] (%fa) {implementation_family = "
            "#fabric.implementation_family<ScalarIntegerCast>, hw_params = "
         << parametersText << "} : (!fabric.bits<" << inputWidth
         << ">) -> !fabric.bits<" << outputWidth
         << "> fabric.yield %value : !fabric.bits<" << outputWidth
         << "> } } fabric.yield %pe : !fabric.bits<" << outputWidth << "> } }";

  auto parsed =
      mlir::parseSourceString<mlir::ModuleOp>(source.str(), &fabricContext());
  if (!parsed)
    fail(test,
         "could not parse scalar integer cast Fabric fixture:\n" + sourceText);
  attachContract(test, *parsed, contract);
  ::fabric::ModuleOp root;
  parsed->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
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
          ::fabric::ImplementationFamilyId::ScalarIntegerCast)
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
              "System has no physical scalar integer cast occurrence");
      return {std::move(fabric),
              occurrence,
              std::move(system),
              physical->physicalOccurrence,
              kind,
              inputWidth,
              outputWidth};
    }
  }
  fail(test, "Fabric fixture has no scalar integer cast occurrence");
}

::dataflow::CanonicalActorSchemaProjection
castActor(llvm::StringRef test, ::dataflow::OperationSchemaId schema,
          mlir::Type sourceType, mlir::Type destinationType) {
  mlir::Block block;
  const mlir::Location location = mlir::UnknownLoc::get(&fabricContext());
  mlir::Value input = block.addArgument(sourceType, location);
  mlir::OpBuilder builder(&fabricContext());
  builder.setInsertionPointToEnd(&block);
  mlir::Operation *operation = nullptr;
  using Schema = ::dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::ArithExtSI:
    operation =
        mlir::arith::ExtSIOp::create(builder, location, destinationType, input);
    break;
  case Schema::ArithExtUI:
    operation =
        mlir::arith::ExtUIOp::create(builder, location, destinationType, input);
    break;
  case Schema::ArithTruncI:
    operation = mlir::arith::TruncIOp::create(builder, location,
                                              destinationType, input);
    break;
  case Schema::ArithIndexCast:
    operation = mlir::arith::IndexCastOp::create(builder, location,
                                                 destinationType, input);
    break;
  case Schema::ArithIndexCastUI:
    operation = mlir::arith::IndexCastUIOp::create(builder, location,
                                                   destinationType, input);
    break;
  default:
    fail(test, "requested a non-cast actor fixture");
  }
  return take(test,
              ::dataflow::projectRegisteredActorSchemaProjection(operation));
}

std::vector<std::uint8_t>
semanticValue(llvm::StringRef test,
              const loom::fabric::ResolvedFabricOpCapabilityView &capability,
              const ::dataflow::CanonicalActorSchemaProjection &actor) {
  constexpr std::array<std::uint64_t, 1> operandPorts = {0};
  constexpr std::array<std::uint64_t, 1> resultPorts = {0};
  auto relation =
      take(test, capability.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "configured cast semantic field relation is not finite");
  const bool hasIndexEndpoint =
      llvm::isa<mlir::IndexType>(actor.type.getInput(0)) ||
      llvm::isa<mlir::IndexType>(actor.type.getResult(0));
  const loom::CanonicalSemanticBytes encoded = take(
      test, relation.projectSemanticValue(
                actor, operandPorts, resultPorts,
                hasIndexEndpoint ? std::optional<::fabric::ResolvedIndexWidth>(
                                       ::fabric::ResolvedIndexWidth::I32)
                                 : std::nullopt));
  return std::vector<std::uint8_t>(encoded.bytes().begin(),
                                   encoded.bytes().end());
}

struct CastSemanticValues final {
  std::vector<std::uint8_t> identity;
  std::vector<std::uint8_t> signExtend;
  std::vector<std::uint8_t> zeroExtend;
  std::vector<std::uint8_t> truncate;
};

CastSemanticValues configuredSemanticValues(
    llvm::StringRef test,
    const loom::fabric::ResolvedFabricOpCapabilityView &capability) {
  require(test, capability.configurationFieldSchema.size() == 1,
          "configured cast has an unexpected field count");
  mlir::Type i8 = mlir::IntegerType::get(&fabricContext(), 8);
  mlir::Type i32 = mlir::IntegerType::get(&fabricContext(), 32);
  mlir::Type index = mlir::IndexType::get(&fabricContext());

  CastSemanticValues values{
      semanticValue(test, capability,
                    castActor(test,
                              ::dataflow::OperationSchemaId::ArithIndexCast,
                              index, i32)),
      semanticValue(
          test, capability,
          castActor(test, ::dataflow::OperationSchemaId::ArithExtSI, i8, i32)),
      semanticValue(
          test, capability,
          castActor(test, ::dataflow::OperationSchemaId::ArithExtUI, i8, i32)),
      semanticValue(test, capability,
                    castActor(test, ::dataflow::OperationSchemaId::ArithTruncI,
                              i32, i8))};

  require(test,
          values.signExtend ==
              semanticValue(
                  test, capability,
                  castActor(test, ::dataflow::OperationSchemaId::ArithIndexCast,
                            i8, index)),
          "signed ordinary and index extension did not physically deduplicate");
  require(
      test,
      values.zeroExtend ==
          semanticValue(
              test, capability,
              castActor(test, ::dataflow::OperationSchemaId::ArithIndexCastUI,
                        i8, index)),
      "unsigned ordinary and index extension did not physically deduplicate");
  require(
      test,
      values.truncate ==
              semanticValue(
                  test, capability,
                  castActor(test, ::dataflow::OperationSchemaId::ArithIndexCast,
                            index, i8)) &&
          values.truncate ==
              semanticValue(
                  test, capability,
                  castActor(test,
                            ::dataflow::OperationSchemaId::ArithIndexCastUI,
                            index, i8)),
      "ordinary and index truncation did not physically deduplicate");
  require(
      test,
      values.identity ==
          semanticValue(
              test, capability,
              castActor(test, ::dataflow::OperationSchemaId::ArithIndexCastUI,
                        i32, index)),
      "signed and unsigned index identity did not physically deduplicate");
  return values;
}

enum class AbiKind { Complete, MissingTruncate };

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto *capability =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  if (capability->configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));

  const CastSemanticValues values = configuredSemanticValues(test, *capability);
  const auto fieldReference = capability->configurationFieldSchema.front();
  std::vector<FiniteCodebookEntry> entries{
      {values.identity, {0x05}},
      {values.signExtend, {0x01}},
      {values.zeroExtend, {0x06}},
      {kind == AbiKind::MissingTruncate ? std::vector<std::uint8_t>{0xff}
                                        : values.truncate,
       {0x02}}};
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence, fieldReference.ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, FiniteCodebookEncoding{3, std::move(entries)},
      values.identity};
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

enum class LeafMutation { None, WrongConfigurationWidth };

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fabric,
                             const ConfigurationABI &abi,
                             llvm::StringRef symbol,
                             LeafMutation mutation = LeafMutation::None) {
  const auto *capability =
      fabric.fabric.view().resolvedFabricOpCapability(fabric.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports =
      take(test, deriveFabricOperationLeafPorts(
                     builder, fabric.physicalOccurrence, *capability, abi));
  if (mutation == LeafMutation::WrongConfigurationWidth) {
    require(test, ports.size() == 3,
            "configured scalar integer cast has unexpected leaf ports");
    ports[1].type = builder.getIntegerType(2);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(symbol), ports);
  return {std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

llvm::Expected<FabricOperationProviderOutput>
trySpecialize(SkeletonFixture &skeleton, const FabricFixture &fabric,
              const FinalizedConfigurationABI &abi, BackendRecipeKey recipe,
              const FabricOperationProviderRegistry &registry) {
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, recipe, {}}};
  return specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                         recipes, registry, externalContracts);
}

std::string specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarIntegerCastProvider(registry))
    fail(test, llvm::toString(std::move(error)));
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
          "portable cast provider emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

void registrationIsPortableOnly() {
  const llvm::StringRef test = __func__;
  const auto &descriptor = castDescriptor();
  require(
      test,
      descriptor.familyId ==
              ::fabric::ImplementationFamilyId::ScalarIntegerCast &&
          descriptor.capabilityParamsSchema ==
              ::fabric::CapabilityParamsSchemaId::ScalarIntegerCastParams &&
          descriptor.typedAdmissionProvider ==
              ::fabric::TypedAdmissionProviderId::ScalarIntegerCastAdmission &&
          descriptor.admittedSchemas.size() == 5,
      "generated scalar integer cast descriptor changed");

  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarIntegerCastProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  const auto coverage = registry.coverage();
  require(test, coverage.size() == ::fabric::implementationFamilyCount(),
          "provider coverage lost the generated family closure");
  const auto found = llvm::find_if(coverage, [](const auto &entry) {
    return entry.implementationFamily ==
           ::fabric::ImplementationFamilyId::ScalarIntegerCast;
  });
  require(test,
          found != coverage.end() &&
              found->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "scalar integer cast provider registration is not portable-only");
  for (const auto &entry : coverage)
    if (entry.implementationFamily !=
        ::fabric::ImplementationFamilyId::ScalarIntegerCast)
      require(test, entry.recipes.empty(),
              "scalar integer cast registration covered another family");
}

void fabricOwnsBehaviorDomain(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture configured = makeFabric(test, store);
  const auto *capability = configured.fabric.view().resolvedFabricOpCapability(
      configured.occurrence);
  require(test, capability != nullptr, "configured capability did not resolve");
  require(test,
          std::holds_alternative<::fabric::ScalarIntegerCastParams>(
              capability->parameterizedCapability) &&
              capability->enabledOperationSchemas.size() == 5 &&
              capability->configurationFieldSchema.size() == 1,
          "configured capability lost its Fabric-owned cast domain");
  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const auto &port : capability->physicalPorts)
    (port.reference.direction == loom::fabric::FabricPortDirection::Input
         ? inputs
         : outputs)
        .push_back(&port);
  require(test,
          inputs.size() == 1 && outputs.size() == 1 &&
              inputs.front()->reference.ordinal == 0 &&
              outputs.front()->reference.ordinal == 0 &&
              inputs.front()->payloadWidthBits == configured.inputWidth &&
              outputs.front()->payloadWidthBits == configured.outputWidth,
          "configured capability changed its exact physical ports");
  const auto relation =
      take(test, capability->resolveSemanticFieldRelation(fabricContext()));
  const auto domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 4,
          "cross-schema cast behaviors did not collapse to four modes");
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured cast behavior has no semantic value");
    take(test, ::dataflow::encodeCanonicalActorSchemaProjection(
                   point.representativeActor));
  }
  const CastSemanticValues values = configuredSemanticValues(test, *capability);
  require(test,
          values.identity != values.signExtend &&
              values.identity != values.zeroExtend &&
              values.identity != values.truncate &&
              values.signExtend != values.zeroExtend &&
              values.signExtend != values.truncate &&
              values.zeroExtend != values.truncate,
          "distinct cast behaviors share configuration bytes");

  FabricFixture identity = makeFabric(test, store, FixtureKind::IdentityOnly);
  const auto *identityCapability =
      identity.fabric.view().resolvedFabricOpCapability(identity.occurrence);
  require(test, identityCapability != nullptr,
          "identity capability did not resolve");
  const auto identityRelation = take(
      test, identityCapability->resolveSemanticFieldRelation(fabricContext()));
  const auto identityDomain = identityRelation.finiteBehaviorDomain();
  require(test,
          identityCapability->configurationFieldSchema.empty() &&
              identityDomain.size() == 1 &&
              !identityDomain.front().semanticConfiguration &&
              identityDomain.front().resolvedIndexWidth ==
                  ::fabric::ResolvedIndexWidth::I32,
          "cross-schema identity did not become one configuration-free mode");
}

std::string emitDeterministically(llvm::StringRef test,
                                  const FabricFixture &fabric,
                                  const FinalizedConfigurationABI &abi,
                                  llvm::StringRef symbol) {
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first =
      makeSkeleton(test, *firstContext, fabric, abi.abi(), symbol);
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  const bool configured = fabric.kind == FixtureKind::Configured;
  require(
      test,
      ports.size() == (configured ? 3 : 2) &&
          ports.atInput(0).getName() == "data_input_0" &&
          mlir::cast<mlir::IntegerType>(ports.atInput(0).type).getWidth() ==
              fabric.inputWidth &&
          ports.atOutput(0).getName() == "data_output_0" &&
          mlir::cast<mlir::IntegerType>(ports.atOutput(0).type).getWidth() ==
              fabric.outputWidth,
      "derived cast leaf ports are not canonical");
  if (configured)
    require(
        test,
        ports.atInput(1).getName() == "config_0" &&
            mlir::cast<mlir::IntegerType>(ports.atInput(1).type).getWidth() ==
                3,
        "configured cast leaf has the wrong ABI port");
  const std::string firstRtl = specialize(test, first, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi(), symbol);
  const std::string secondRtl = specialize(test, second, fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical scalar integer cast inputs produced different RTL");
  return firstRtl;
}

void writeToolInputs(llvm::StringRef test, const std::filesystem::path &root,
                     const std::string &configured,
                     const std::string &identity) {
  const std::string testbench = R"sv(
module testbench;
  logic [36:0] cast_input;
  logic [2:0] cast_config;
  logic [39:0] cast_output;
  logic [34:0] identity_input;
  logic [35:0] identity_output;

  scalar_integer_cast cast_dut(
    .data_input_0(cast_input), .config_0(cast_config),
    .data_output_0(cast_output));
  scalar_integer_cast_identity identity_dut(
    .data_input_0(identity_input), .data_output_0(identity_output));

  initial begin
    cast_input = 37'h100000080;
    cast_config = 3'b001;
    #1;
    if (cast_output !== 40'h00ffffff80)
      $fatal(1, "sign extension or physical zero fill failed");

    cast_config = 3'b110;
    #1;
    if (cast_output !== 40'h0000000080)
      $fatal(1, "zero extension failed");

    cast_input = 37'h1deadbeef;
    cast_config = 3'b010;
    #1;
    if (cast_output !== 40'h00000000ef)
      $fatal(1, "truncation or low-bit input adaptation failed");

    cast_input = 37'h189abcdef;
    cast_config = 3'b101;
    #1;
    if (cast_output !== 40'h0089abcdef)
      $fatal(1, "configured identity failed");

    cast_config = 3'b000;
    #1;
    if (cast_output !== 40'h0089abcdef)
      $fatal(1, "unassigned code did not select the inactive identity");

    identity_input = 35'h589abcdef;
    #1;
    if (identity_output !== 36'h089abcdef)
      $fatal(1, "configuration-free index identity failed");
    $finish;
  end
endmodule
)sv";
  const std::string synthesisTop = R"sv(
module scalar_integer_cast_synthesis_top(
  input logic [36:0] cast_input,
  input logic [2:0] cast_config,
  input logic [34:0] identity_input,
  output logic [39:0] cast_output,
  output logic [35:0] identity_output);
  scalar_integer_cast cast_dut(
    .data_input_0(cast_input), .config_0(cast_config),
    .data_output_0(cast_output));
  scalar_integer_cast_identity identity_dut(
    .data_input_0(identity_input), .data_output_0(identity_output));
endmodule
)sv";
  const std::string yosysScript = R"ys(
read_verilog -sv scalar_integer_cast.sv scalar_integer_cast_identity.sv synthesis_top.sv
hierarchy -check -top scalar_integer_cast_synthesis_top
proc
opt
check -assert
synth -top scalar_integer_cast_synthesis_top
check -assert
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"scalar_integer_cast.sv", configured},
           {"scalar_integer_cast_identity.sv", identity},
           {"testbench.sv", testbench},
           {"synthesis_top.sv", synthesisTop},
           {"portable_scalar_integer_cast.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void configuredAndSingletonRtl(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture configured = makeFabric(test, store);
  FinalizedConfigurationABI configuredAbi =
      makeConfigurationAbi(test, store, configured);
  const std::string configuredRtl = emitDeterministically(
      test, configured, configuredAbi, "scalar_integer_cast");
  require(test,
          llvm::StringRef(configuredRtl).contains("config_0") &&
              llvm::StringRef(configuredRtl).contains("data_input_0[7:0]") &&
              llvm::StringRef(configuredRtl).contains("data_input_0[31:0]"),
          "configured RTL lost cast selection or low-bit semantics");

  FabricFixture identity = makeFabric(test, store, FixtureKind::IdentityOnly);
  FinalizedConfigurationABI identityAbi =
      makeConfigurationAbi(test, store, identity);
  const std::string identityRtl = emitDeterministically(
      test, identity, identityAbi, "scalar_integer_cast_identity");
  require(test,
          !llvm::StringRef(identityRtl).contains("config_") &&
              llvm::StringRef(identityRtl).contains("data_input_0[31:0]"),
          "configuration-free identity did not lower directly");
  writeToolInputs(test, root, configuredRtl, identityRtl);
}

void failuresAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableScalarIntegerCastProvider(registry))
    fail(test, llvm::toString(std::move(error)));

  FabricFixture valid = makeFabric(test, store);
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  std::unique_ptr<mlir::MLIRContext> leafContext = makeCirctContext();
  SkeletonFixture malformedLeaf = makeSkeleton(
      test, *leafContext, valid, validAbi.abi(), "malformed_cast_leaf",
      LeafMutation::WrongConfigurationWidth);
  const std::string leafBefore = moduleText(*malformedLeaf.module);
  expectInvalid(test,
                trySpecialize(malformedLeaf, valid, validAbi,
                              BackendRecipeKey::PortableSystemVerilog,
                              registry),
                "leaf port");
  require(test, moduleText(*malformedLeaf.module) == leafBefore,
          "malformed leaf mutated the caller module");

  expectError(
      test,
      finalizeConfigurationABI(
          makeConfigurationAbiDraft(test, valid, AbiKind::MissingTruncate),
          store),
      "semantic");

  FabricFixture wrongContract =
      makeFabric(test, store, FixtureKind::Configured, ContractKind::Wrong);
  FinalizedConfigurationABI wrongContractAbi =
      makeConfigurationAbi(test, store, wrongContract);
  std::unique_ptr<mlir::MLIRContext> contractContext = makeCirctContext();
  SkeletonFixture contractSkeleton =
      makeSkeleton(test, *contractContext, wrongContract,
                   wrongContractAbi.abi(), "unsupported_cast_contract");
  const std::string contractBefore = moduleText(*contractSkeleton.module);
  expectTypedUnsupported(
      test,
      trySpecialize(contractSkeleton, wrongContract, wrongContractAbi,
                    BackendRecipeKey::PortableSystemVerilog, registry),
      BackendRecipeKey::PortableSystemVerilog,
      "unsupported scalar integer cast resource contract");
  require(test, moduleText(*contractSkeleton.module) == contractBefore,
          "unsupported contract mutated the caller module");

  constexpr std::array nativeRecipes = {
      BackendRecipeKey::SynopsysDesignWare,
      BackendRecipeKey::CadenceChipWare,
      BackendRecipeKey::AmdXilinx,
      BackendRecipeKey::IntelAltera,
  };
  for (BackendRecipeKey recipe : nativeRecipes) {
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(
        test, *context, valid, validAbi.abi(), "unsupported_native_cast");
    const std::string before = moduleText(*skeleton.module);
    expectTypedUnsupported(
        test, trySpecialize(skeleton, valid, validAbi, recipe, registry),
        recipe, "backend-native scalar integer cast recipe");
    require(test, moduleText(*skeleton.module) == before,
            "unsupported native recipe mutated the caller module");
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  registrationIsPortableOnly();
  fabricOwnsBehaviorDomain(root / "domain");
  configuredAndSingletonRtl(root);
  failuresAreTransactional(root / "invalid");
  return 0;
}

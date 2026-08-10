#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/ScalarUnsignedIntegerDivRem.h"
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
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
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

using Family = ::fabric::ImplementationFamilyId;
using Schema = ::dataflow::OperationSchemaId;

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
    fail(test, "accepted malformed scalar unsigned div/rem input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

template <typename T>
void expectTypedUnsupported(llvm::StringRef test, llvm::Expected<T> value,
                            Family family, BackendRecipeKey recipe,
                            llvm::StringRef description) {
  require(test, !value, std::string("provider accepted ") + description.str());
  bool typedUnsupported = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        typedUnsupported =
            error.implementationFamily() == family && error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, description.str() +
                       " returned the wrong error class: " + error.message());
      });
  require(test, typedUnsupported,
          description.str() + " lost its typed Unsupported classification");
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

struct FabricSpec final {
  std::string name;
  std::string familyKeyword;
  Family family;
  std::vector<std::string> operations;
  std::string parameters;
  std::vector<unsigned> inputWidths;
  unsigned outputWidth = 0;
  unsigned semanticWidth = 0;
  bool unsupportedContract = false;
  bool inactiveQuotient = false;
};

FabricSpec div8PaddedSpec() {
  return {"scalar_unsigned_div_8_padded",
          "ScalarUnsignedIntegerDivRem",
          Family::ScalarUnsignedIntegerDivRem,
          {"arith.divui"},
          "integer_widths = [8 : i32]",
          {13, 13},
          13,
          8};
}

FabricSpec rem8PaddedSpec() {
  return {"scalar_unsigned_rem_8_padded",
          "ScalarUnsignedIntegerDivRem",
          Family::ScalarUnsignedIntegerDivRem,
          {"arith.remui"},
          "integer_widths = [8 : i32]",
          {13, 13},
          13,
          8};
}

FabricSpec configured8PaddedSpec() {
  return {"scalar_unsigned_div_rem_8_padded",
          "ScalarUnsignedIntegerDivRem",
          Family::ScalarUnsignedIntegerDivRem,
          {"arith.divui", "arith.remui"},
          "integer_widths = [8 : i32]",
          {13, 13},
          13,
          8};
}

FabricSpec configured8PaddedInactiveQuotientSpec() {
  FabricSpec spec = configured8PaddedSpec();
  spec.name = "scalar_unsigned_div_rem_8_padded_inactive_quotient";
  spec.inactiveQuotient = true;
  return spec;
}

FabricSpec configured64Spec() {
  return {"scalar_unsigned_div_rem_64",
          "ScalarUnsignedIntegerDivRem",
          Family::ScalarUnsignedIntegerDivRem,
          {"arith.divui", "arith.remui"},
          "integer_widths = [64 : i32]",
          {64, 64},
          64,
          64};
}

FabricSpec unsupportedContractSpec() {
  FabricSpec spec = configured8PaddedSpec();
  spec.name = "scalar_unsigned_div_rem_unsupported_contract";
  spec.unsupportedContract = true;
  return spec;
}

FabricSpec unsupportedShapeSpec() {
  FabricSpec spec = div8PaddedSpec();
  spec.name = "scalar_unsigned_div_unsupported_shape";
  spec.inputWidths.push_back(13);
  return spec;
}

FabricSpec multiWidthSpec() {
  return {"scalar_unsigned_div_multi_width",
          "ScalarUnsignedIntegerDivRem",
          Family::ScalarUnsignedIntegerDivRem,
          {"arith.divui"},
          "integer_widths = [8 : i32, 16 : i32]",
          {16, 16},
          16,
          16};
}

FabricSpec multiOperationWidthSpec() {
  return {"scalar_unsigned_div_rem_multi_width",
          "ScalarUnsignedIntegerDivRem",
          Family::ScalarUnsignedIntegerDivRem,
          {"arith.divui", "arith.remui"},
          "integer_widths = [8 : i32, 16 : i32]",
          {16, 16},
          16,
          16};
}

FabricSpec otherFamilySpec() {
  return {"scalar_integer_multiply_other_family",
          "ScalarIntegerMultiply",
          Family::ScalarIntegerMultiply,
          {"arith.muli"},
          "integer_widths = [8 : i32]",
          {8, 8},
          8,
          8};
}

std::string fabricSource(const FabricSpec &spec) {
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module { fabric.module @" << spec.name << '(';
  for (const auto [index, width] : llvm::enumerate(spec.inputWidths)) {
    if (index != 0)
      stream << ", ";
    stream << "%a" << index << ": !fabric.bits<" << width << '>';
  }
  stream << ") -> !fabric.bits<" << spec.outputWidth
         << "> { %pe = fabric.pe [spatial] (";
  for (const auto [index, width] : llvm::enumerate(spec.inputWidths)) {
    if (index != 0)
      stream << ", ";
    stream << "%pa" << index << " = %a" << index << " : !fabric.bits<" << width
           << '>';
  }
  stream << ") -> !fabric.bits<" << spec.outputWidth << "> { %fu = fabric.fu (";
  for (const auto [index, width] : llvm::enumerate(spec.inputWidths)) {
    if (index != 0)
      stream << ", ";
    stream << "%fa" << index << " = %pa" << index << " : !fabric.bits<" << width
           << '>';
  }
  stream << ") -> !fabric.bits<" << spec.outputWidth
         << "> { %value = fabric.op [";
  for (const auto [index, operation] : llvm::enumerate(spec.operations)) {
    if (index != 0)
      stream << ", ";
    stream << '@' << operation;
  }
  stream << "] (";
  for (std::size_t index = 0; index < spec.inputWidths.size(); ++index) {
    if (index != 0)
      stream << ", ";
    stream << "%fa" << index;
  }
  stream << ") {implementation_family = #fabric.implementation_family<"
         << spec.familyKeyword << ">, hw_params = {" << spec.parameters
         << "}} : (";
  for (const auto [index, width] : llvm::enumerate(spec.inputWidths)) {
    if (index != 0)
      stream << ", ";
    stream << "!fabric.bits<" << width << '>';
  }
  stream << ") -> !fabric.bits<" << spec.outputWidth
         << "> fabric.yield %value : !fabric.bits<" << spec.outputWidth
         << "> } } fabric.yield %pe : !fabric.bits<" << spec.outputWidth
         << "> } }";
  return source;
}

void attachResourceContract(llvm::StringRef test, mlir::ModuleOp source,
                            bool unsupported) {
  const ::fabric::ResourceContract &resourceContract =
      unsupported ? ::fabric::loopCarryOperationResourceContract()
                  : ::fabric::oneCycleElasticOperationResourceContract();
  const std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(resourceContract));
  const std::vector<std::int8_t> signedContract(contract.begin(),
                                                contract.end());
  source.walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedContract));
  });
}

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
  FabricSpec spec;
};

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         FabricSpec spec) {
  const std::string sourceText = fabricSource(spec);
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  attachResourceContract(test, *source, spec.unsupportedContract);

  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no root");
  FinalizedFabricRoot fabric =
      take(test, loom::fabric::finalizeFabricRoot(root, store));

  for (const auto fuOccurrence : fabric.view().fuOccurrences()) {
    const auto definition = fabric.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &candidate :
         fabric.view().resolvedFabricOpCapabilities(*definition)) {
      if (candidate.implementationFamily != spec.family)
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
              "System has no physical div/rem occurrence");
      return {std::move(fabric), occurrence, std::move(system),
              physical->physicalOccurrence, std::move(spec)};
    }
  }
  fail(test, "Fabric fixture has no expected operation occurrence");
}

void expectFabricRejected(llvm::StringRef test, const ArtifactStore &store,
                          const FabricSpec &spec, llvm::StringRef expected) {
  std::vector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      &fabricContext(), [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return mlir::success();
      });
  auto source = mlir::parseSourceString<mlir::ModuleOp>(fabricSource(spec),
                                                        &fabricContext());
  if (!source) {
    require(test,
            llvm::any_of(diagnostics,
                         [&](const std::string &message) {
                           return llvm::StringRef(message).contains(expected);
                         }),
            diagnostics.empty() ? "invalid Fabric produced no diagnostic"
                                : diagnostics.front());
    return;
  }
  attachResourceContract(test, *source, false);
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  auto finalized = loom::fabric::finalizeFabricRoot(root, store);
  require(test, !finalized, "malformed Fabric capability was finalized");
  const std::string message = llvm::toString(finalized.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

enum class ConfigurationAbiKind {
  Complete,
  MissingRemainder,
  ExtraSemanticValue,
  DirectBits,
  MissingField,
};

ConfigurationABIDraft makeConfigurationAbiDraft(
    llvm::StringRef test, const ArtifactStore &store,
    const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));

  require(test, resolved.configurationFieldSchema.size() == 1,
          "configured div/rem fixture has an unexpected field count");
  const auto &descriptor =
      ::fabric::implementationFamily(resolved.implementationFamily);
  require(test, descriptor.admittedSchemas.size() == 2,
          "generated div/rem family descriptor changed cardinality");
  const auto fieldReference = resolved.configurationFieldSchema.front();
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "configured div/rem relation is not finite");
  const auto domain = relation.finiteBehaviorDomain();
  require(test, !domain.empty(), "configured div/rem domain is empty");

  std::vector<FiniteCodebookEntry> entries;
  std::optional<std::size_t> quotientIndex;
  std::optional<std::size_t> remainderIndex;
  entries.reserve(domain.size() + 1);
  for (const auto [index, point] : llvm::enumerate(domain)) {
    require(test, point.semanticConfiguration.has_value(),
            "configured div/rem behavior has no semantic value");
    const Schema schema = point.representativeActor.schema;
    require(test,
            schema == descriptor.admittedSchemas[0] ||
                schema == descriptor.admittedSchemas[1],
            "configured div/rem domain contains a foreign schema");
    std::uint8_t code = static_cast<std::uint8_t>(index + 1);
    if (domain.size() == 2 && resolved.enabledOperationSchemas.size() == 2)
      code = schema == descriptor.admittedSchemas[0] ? 0x02 : 0x01;
    entries.push_back({{point.semanticConfiguration->bytes().begin(),
                        point.semanticConfiguration->bytes().end()},
                       {code}});
    if (schema == descriptor.admittedSchemas[0] && !quotientIndex)
      quotientIndex = index;
    if (schema == descriptor.admittedSchemas[1] && !remainderIndex)
      remainderIndex = index;
  }

  if (kind == ConfigurationAbiKind::MissingRemainder) {
    require(test, remainderIndex.has_value(),
            "configured div/rem domain has no remainder behavior");
    entries[*remainderIndex].semanticValue = {0xff};
  }
  if (kind == ConfigurationAbiKind::ExtraSemanticValue)
    entries.push_back({{0xfe}, {0x03}});

  SemanticFieldEncoding encoding;
  if (kind == ConfigurationAbiKind::DirectBits) {
    encoding = DirectBitsEncoding{2};
  } else {
    encoding = FiniteCodebookEncoding{2, std::move(entries)};
  }
  const std::size_t inactiveIndex = [&] {
    if ((kind == ConfigurationAbiKind::MissingRemainder ||
         fixture.spec.inactiveQuotient) &&
        quotientIndex)
      return *quotientIndex;
    if (remainderIndex)
      return *remainderIndex;
    return std::size_t{0};
  }();
  const std::vector<std::uint8_t> inactiveValue =
      kind == ConfigurationAbiKind::DirectBits
          ? std::vector<std::uint8_t>{0}
          : std::get<FiniteCodebookEncoding>(encoding)
                .entries[inactiveIndex]
                .semanticValue;
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence, fieldReference.ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, std::move(encoding), inactiveValue};
  ConfigurationABIDraft draft =
      take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                     fixture.system, {std::move(field)}));
  if (kind == ConfigurationAbiKind::MissingField) {
    bool removed = false;
    for (ProgrammingUnitDraft &unit : draft.programmingUnits) {
      const auto field = llvm::find_if(unit.fields, [&](const auto &candidate) {
        return loom::fabric::configurationField(candidate.slot) ==
               physicalField;
      });
      if (field == unit.fields.end())
        continue;
      unit.fields.erase(field);
      removed = true;
      break;
    }
    require(test, removed, "could not remove div/rem field from ABI draft");
  }
  return draft;
}

FinalizedConfigurationABI makeConfigurationAbi(
    llvm::StringRef test, const ArtifactStore &store,
    const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  return take(
      test, finalizeConfigurationABI(
                makeConfigurationAbiDraft(test, store, fixture, kind), store));
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

enum class LeafMutation { None, WrongFirstInputWidth };

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fixture,
                             const ConfigurationABI &abi,
                             LeafMutation mutation = LeafMutation::None) {
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
  if (mutation == LeafMutation::WrongFirstInputWidth) {
    require(test, !ports.empty(), "div/rem leaf has no data input");
    const unsigned width =
        mlir::cast<mlir::IntegerType>(ports.front().type).getWidth();
    ports.front().type = builder.getIntegerType(width + 1);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(fixture.spec.name), ports);
  return {std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

FabricOperationProviderRegistry makeProviderRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableScalarUnsignedIntegerDivRemProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

llvm::Expected<FabricOperationProviderOutput>
specializeNativeFor(SkeletonFixture &skeleton, const FabricFixture &fixture,
                    const FinalizedConfigurationABI &abi,
                    const FabricOperationProviderRegistry &registry,
                    BackendRecipeKey recipe) {
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fixture.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fixture.physicalOccurrence, recipe, {}}};
  return specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                         recipes, registry, externalContracts);
}

llvm::Expected<loom::hardware::test::PortableProviderConformance>
specializePortableFor(SkeletonFixture &skeleton, const FabricFixture &fixture,
                      const FinalizedConfigurationABI &abi,
                      const FabricOperationProviderRegistry &registry) {
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, fixture.physicalOccurrence}}};
  return loom::hardware::test::specializeAndExportPortableProvider(
      std::move(module), abi, registry, externalContracts);
}

std::string specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                       const FabricFixture &fixture,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  auto conformance =
      take(test, specializePortableFor(skeleton, fixture, abi, registry));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "portable div/rem emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

void generatedOwnerAndProviderCoverage() {
  const llvm::StringRef test = __func__;
  constexpr Family family = Family::ScalarUnsignedIntegerDivRem;
  const ::fabric::ImplementationFamilyDescriptor &descriptor =
      ::fabric::implementationFamily(family);
  require(test,
          descriptor.familyId == family &&
              descriptor.capabilityParamsSchema ==
                  ::fabric::CapabilityParamsSchemaId::ScalarIntegerParams &&
              descriptor.typedAdmissionProvider ==
                  ::fabric::TypedAdmissionProviderId::
                      ScalarOrdinaryIntegerAdmission &&
              descriptor.admittedSchemas.size() == 2 &&
              descriptor.admittedSchemas[0] == Schema::ArithDivUI &&
              descriptor.admittedSchemas[1] == Schema::ArithRemUI,
          "generated scalar unsigned div/rem descriptor changed");

  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  const auto found = llvm::find_if(coverage, [&](const auto &candidate) {
    return candidate.implementationFamily == family;
  });
  require(test,
          coverage.size() == ::fabric::implementationFamilyCount() &&
              found != coverage.end() &&
              found->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "scalar unsigned div/rem provider coverage is not portable-only");
}

void checkCapability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  require(test, resolved.implementationFamily == fixture.spec.family,
          "div/rem capability changed implementation family");
  const auto &descriptor =
      ::fabric::implementationFamily(resolved.implementationFamily);
  require(test,
          !resolved.enabledOperationSchemas.empty() &&
              llvm::all_of(resolved.enabledOperationSchemas,
                           [&](Schema schema) {
                             return llvm::is_contained(
                                 descriptor.admittedSchemas, schema);
                           }),
          "div/rem capability escaped its generated family descriptor");
  const auto *parameters = std::get_if<::fabric::ScalarIntegerParams>(
      &resolved.parameterizedCapability);
  require(test,
          parameters && parameters->integerWidths.valid() &&
              parameters->integerWidths.size() == 1 &&
              parameters->pointerFormats.empty(),
          "div/rem capability changed its scalar integer parameters");
  const auto width = llvm::find_if(
      ::fabric::integerWidthDomain, [&](::fabric::IntegerWidth candidate) {
        return ::fabric::getBitWidth(candidate) == fixture.spec.semanticWidth;
      });
  require(test,
          width != ::fabric::integerWidthDomain.end() &&
              parameters->integerWidths.contains(*width),
          "div/rem capability lost its semantic integer width");
  require(test,
          resolved.configurationFieldSchema.size() ==
              (resolved.enabledOperationSchemas.size() == 1 ? 0U : 1U),
          "div/rem capability changed selector cardinality");

  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const loom::fabric::ResolvedFabricOpPhysicalPortView *> outputs;
  for (const auto &port : resolved.physicalPorts)
    (port.reference.direction == loom::fabric::FabricPortDirection::Input
         ? inputs
         : outputs)
        .push_back(&port);
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  require(test,
          inputs.size() == fixture.spec.inputWidths.size() &&
              outputs.size() == 1 && outputs[0]->reference.ordinal == 0 &&
              outputs[0]->payloadWidthBits == fixture.spec.outputWidth,
          "div/rem capability changed its exact physical port inventory");
  for (const auto [index, input] : llvm::enumerate(inputs))
    require(test,
            input->reference.ordinal == index &&
                input->payloadWidthBits == fixture.spec.inputWidths[index],
            "div/rem capability changed an exact physical input");
}

void checkLeafPorts(llvm::StringRef test, SkeletonFixture &skeleton,
                    const FabricFixture &fixture, bool configured) {
  const circt::hw::ModulePortInfo ports(skeleton.leaf.getPortList());
  require(test, ports.size() == (configured ? 4U : 3U),
          "div/rem leaf has the wrong exact port count");
  require(
      test,
      ports.atInput(0).getName() == "data_input_0" &&
          ports.atInput(1).getName() == "data_input_1" &&
          ports.atOutput(0).getName() == "data_output_0" &&
          mlir::cast<mlir::IntegerType>(ports.atInput(0).type).getWidth() ==
              fixture.spec.inputWidths[0] &&
          mlir::cast<mlir::IntegerType>(ports.atInput(1).type).getWidth() ==
              fixture.spec.inputWidths[1] &&
          mlir::cast<mlir::IntegerType>(ports.atOutput(0).type).getWidth() ==
              fixture.spec.outputWidth,
      "div/rem leaf lost its exact physical data ports");
  if (configured)
    require(
        test,
        ports.atInput(2).getName() == "config_0" &&
            mlir::cast<mlir::IntegerType>(ports.atInput(2).type).getWidth() ==
                2,
        "configured div/rem leaf lost its ABI-owned selector port");
  else
    for (const auto &port : ports)
      require(test, !port.getName().starts_with("config_"),
              "singleton div/rem retained a selector port");
}

std::string emitDeterministically(llvm::StringRef test,
                                  const FabricFixture &fixture,
                                  const FinalizedConfigurationABI &abi) {
  const bool configured =
      !capability(test, fixture).configurationFieldSchema.empty();
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fixture, abi.abi());
  checkLeafPorts(test, first, fixture, configured);
  const std::string firstRtl = specialize(test, first, fixture, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fixture, abi.abi());
  const std::string secondRtl = specialize(test, second, fixture, abi);
  require(test, firstRtl == secondRtl,
          "identical div/rem inputs produced different SystemVerilog bytes");
  const llvm::StringRef rtl(firstRtl);
  require(test, rtl.count(" / ") == 1 && !rtl.contains(" % "),
          "div/rem provider did not emit exactly one unsigned divider");
  require(test, rtl.contains(" == ") && rtl.contains(" ? "),
          "div/rem provider lost deterministic divisor-zero refinement");
  if (configured)
    require(test, rtl.contains("config_0"),
            "configured div/rem emitted no ABI selector");
  else
    require(test, !rtl.contains("config_"),
            "singleton div/rem emitted a selector");
  return firstRtl;
}

std::string hexLiteral(unsigned width, std::uint64_t value) {
  std::ostringstream stream;
  stream << width << "'h" << std::hex << std::setfill('0')
         << std::setw(static_cast<int>((width + 3) / 4)) << value;
  return stream.str();
}

std::string testbenchSource() {
  std::ostringstream stream;
  stream << R"sv(module testbench;
  logic [12:0] div8_lhs;
  logic [12:0] div8_rhs;
  logic [12:0] div8_result;
  logic [12:0] rem8_lhs;
  logic [12:0] rem8_rhs;
  logic [12:0] rem8_result;
  logic [12:0] dual8_lhs;
  logic [12:0] dual8_rhs;
  logic [1:0] dual8_config;
  logic [12:0] dual8_result;
  logic [12:0] dual8q_lhs;
  logic [12:0] dual8q_rhs;
  logic [1:0] dual8q_config;
  logic [12:0] dual8q_result;
  logic [63:0] dual64_lhs;
  logic [63:0] dual64_rhs;
  logic [1:0] dual64_config;
  logic [63:0] dual64_result;

  scalar_unsigned_div_8_padded div8(
    .data_input_0(div8_lhs), .data_input_1(div8_rhs),
    .data_output_0(div8_result));
  scalar_unsigned_rem_8_padded rem8(
    .data_input_0(rem8_lhs), .data_input_1(rem8_rhs),
    .data_output_0(rem8_result));
  scalar_unsigned_div_rem_8_padded dual8(
    .data_input_0(dual8_lhs), .data_input_1(dual8_rhs),
    .config_0(dual8_config), .data_output_0(dual8_result));
  scalar_unsigned_div_rem_8_padded_inactive_quotient dual8q(
    .data_input_0(dual8q_lhs), .data_input_1(dual8q_rhs),
    .config_0(dual8q_config), .data_output_0(dual8q_result));
  scalar_unsigned_div_rem_64 dual64(
    .data_input_0(dual64_lhs), .data_input_1(dual64_rhs),
    .config_0(dual64_config), .data_output_0(dual64_result));

  task automatic check_div8(
      input logic [7:0] lhs, input logic [7:0] rhs,
      input logic [7:0] expected);
    begin
      div8_lhs = {5'h0, lhs};
      div8_rhs = {5'h0, rhs};
      #1;
      if (div8_result !== {5'h00, expected})
        $fatal(1, "8-bit unsigned division oracle mismatch");
    end
  endtask

  task automatic check_div8_padded(
      input logic [12:0] lhs, input logic [12:0] rhs,
      input logic [7:0] expected);
    begin
      div8_lhs = lhs;
      div8_rhs = rhs;
      #1;
      if (div8_result !== {5'h00, expected})
        $fatal(1, "8-bit padded unsigned division oracle mismatch");
    end
  endtask

  task automatic check_rem8(
      input logic [7:0] lhs, input logic [7:0] rhs,
      input logic [7:0] expected);
    begin
      rem8_lhs = {5'h0, lhs};
      rem8_rhs = {5'h0, rhs};
      #1;
      if (rem8_result !== {5'h00, expected})
        $fatal(1, "8-bit unsigned remainder oracle mismatch");
    end
  endtask

  task automatic check_rem8_padded(
      input logic [12:0] lhs, input logic [12:0] rhs,
      input logic [7:0] expected);
    begin
      rem8_lhs = lhs;
      rem8_rhs = rhs;
      #1;
      if (rem8_result !== {5'h00, expected})
        $fatal(1, "8-bit padded unsigned remainder oracle mismatch");
    end
  endtask

  task automatic check_dual8(
      input logic [7:0] lhs, input logic [7:0] rhs,
      input logic [1:0] configuration, input logic [7:0] expected);
    begin
      dual8_lhs = {5'h0, lhs};
      dual8_rhs = {5'h0, rhs};
      dual8_config = configuration;
      #1;
      if (dual8_result !== {5'h00, expected})
        $fatal(1, "8-bit configured div/rem oracle mismatch");
    end
  endtask

  task automatic check_dual8_padded(
      input logic [12:0] lhs, input logic [12:0] rhs,
      input logic [1:0] configuration, input logic [7:0] expected);
    begin
      dual8_lhs = lhs;
      dual8_rhs = rhs;
      dual8_config = configuration;
      #1;
      if (dual8_result !== {5'h00, expected})
        $fatal(1, "8-bit padded configured div/rem oracle mismatch");
    end
  endtask

  task automatic check_dual8q(
      input logic [12:0] lhs, input logic [12:0] rhs,
      input logic [1:0] configuration, input logic [7:0] expected);
    begin
      dual8q_lhs = lhs;
      dual8q_rhs = rhs;
      dual8q_config = configuration;
      #1;
      if (dual8q_result !== {5'h00, expected})
        $fatal(1, "8-bit quotient-inactive div/rem oracle mismatch");
    end
  endtask

  task automatic check_dual64(
      input logic [63:0] lhs, input logic [63:0] rhs,
      input logic [1:0] configuration, input logic [63:0] expected);
    begin
      dual64_lhs = lhs;
      dual64_rhs = rhs;
      dual64_config = configuration;
      #1;
      if (dual64_result !== expected)
        $fatal(1, "64-bit configured div/rem oracle mismatch");
    end
  endtask

  initial begin
    check_div8(8'h00, 8'h01, 8'h00);
    check_div8(8'hff, 8'h01, 8'hff);
    check_div8(8'hff, 8'hff, 8'h01);
    check_div8(8'hfe, 8'hff, 8'h00);
    check_div8(8'hff, 8'h00, 8'h00);
    check_div8_padded(13'h1fd3, 13'h1a11, 8'h0c);
    check_div8_padded(13'h1fd3, 13'h1a00, 8'h00);
    check_rem8(8'h00, 8'h01, 8'h00);
    check_rem8(8'hff, 8'h01, 8'h00);
    check_rem8(8'hff, 8'hfe, 8'h01);
    check_rem8(8'hff, 8'h00, 8'h00);
    check_rem8_padded(13'h1fd3, 13'h1a11, 8'h07);
    check_rem8_padded(13'h1fd3, 13'h1a00, 8'h00);
    check_dual8(8'hd3, 8'h11, 2'b10, 8'h0c);
    check_dual8(8'hd3, 8'h11, 2'b01, 8'h07);
    check_dual8(8'hd3, 8'h11, 2'b00, 8'h07);
    check_dual8(8'hd3, 8'h11, 2'b11, 8'h07);
    check_dual8(8'hd3, 8'h00, 2'b10, 8'h00);
    check_dual8(8'hd3, 8'h00, 2'b01, 8'h00);
    check_dual8_padded(13'h1fd3, 13'h1a11, 2'b10, 8'h0c);
    check_dual8_padded(13'h1fd3, 13'h1a11, 2'b01, 8'h07);
    check_dual8q(13'h1fd3, 13'h1a11, 2'b10, 8'h0c);
    check_dual8q(13'h1fd3, 13'h1a11, 2'b01, 8'h07);
    check_dual8q(13'h1fd3, 13'h1a11, 2'b00, 8'h0c);
    check_dual8q(13'h1fd3, 13'h1a11, 2'b11, 8'h0c);
    check_dual8q(13'h1fd3, 13'h1a00, 2'b00, 8'h00);
    check_dual64(64'hffffffffffffffff, 64'h0000000000000001,
                 2'b10, 64'hffffffffffffffff);
    check_dual64(64'hffffffffffffffff, 64'hffffffffffffffff,
                 2'b10, 64'h0000000000000001);
    check_dual64(64'hffffffffffffffff, 64'h8000000000000000,
                 2'b01, 64'h7fffffffffffffff);
    check_dual64(64'hffffffffffffffff, 64'h0000000000000000,
                 2'b10, 64'h0000000000000000);
)sv";

  std::mt19937_64 generator(0x6c6f6f6d64697672ULL);
  for (unsigned index = 0; index < 32; ++index) {
    const std::uint8_t lhs = static_cast<std::uint8_t>(generator());
    std::uint8_t rhs = static_cast<std::uint8_t>(generator());
    if (rhs == 0)
      rhs = 1;
    stream << "    check_div8(" << hexLiteral(8, lhs) << ", "
           << hexLiteral(8, rhs) << ", " << hexLiteral(8, lhs / rhs) << ");\n";
    stream << "    check_rem8(" << hexLiteral(8, lhs) << ", "
           << hexLiteral(8, rhs) << ", " << hexLiteral(8, lhs % rhs) << ");\n";
    stream << "    check_dual8(" << hexLiteral(8, lhs) << ", "
           << hexLiteral(8, rhs) << ", 2'b10, " << hexLiteral(8, lhs / rhs)
           << ");\n";
    stream << "    check_dual8(" << hexLiteral(8, lhs) << ", "
           << hexLiteral(8, rhs) << ", 2'b01, " << hexLiteral(8, lhs % rhs)
           << ");\n";
  }
  for (unsigned index = 0; index < 24; ++index) {
    const std::uint64_t lhs = generator();
    std::uint64_t rhs = generator();
    if (rhs == 0)
      rhs = 1;
    stream << "    check_dual64(" << hexLiteral(64, lhs) << ", "
           << hexLiteral(64, rhs) << ", 2'b10, " << hexLiteral(64, lhs / rhs)
           << ");\n";
    stream << "    check_dual64(" << hexLiteral(64, lhs) << ", "
           << hexLiteral(64, rhs) << ", 2'b01, " << hexLiteral(64, lhs % rhs)
           << ");\n";
  }
  stream << R"sv(    $finish;
  end
endmodule
)sv";
  return stream.str();
}

std::string synthesisTopSource() {
  return R"sv(module scalar_unsigned_div_rem_synthesis_top(
  input [12:0] div8_lhs,
  input [12:0] div8_rhs,
  input [12:0] rem8_lhs,
  input [12:0] rem8_rhs,
  input [12:0] dual8_lhs,
  input [12:0] dual8_rhs,
  input [1:0] dual8_config,
  input [12:0] dual8q_lhs,
  input [12:0] dual8q_rhs,
  input [1:0] dual8q_config,
  input [63:0] dual64_lhs,
  input [63:0] dual64_rhs,
  input [1:0] dual64_config,
  output [12:0] div8_result,
  output [12:0] rem8_result,
  output [12:0] dual8_result,
  output [12:0] dual8q_result,
  output [63:0] dual64_result
);
  scalar_unsigned_div_8_padded div8(
    .data_input_0(div8_lhs), .data_input_1(div8_rhs),
    .data_output_0(div8_result));
  scalar_unsigned_rem_8_padded rem8(
    .data_input_0(rem8_lhs), .data_input_1(rem8_rhs),
    .data_output_0(rem8_result));
  scalar_unsigned_div_rem_8_padded dual8(
    .data_input_0(dual8_lhs), .data_input_1(dual8_rhs),
    .config_0(dual8_config), .data_output_0(dual8_result));
  scalar_unsigned_div_rem_8_padded_inactive_quotient dual8q(
    .data_input_0(dual8q_lhs), .data_input_1(dual8q_rhs),
    .config_0(dual8q_config), .data_output_0(dual8q_result));
  scalar_unsigned_div_rem_64 dual64(
    .data_input_0(dual64_lhs), .data_input_1(dual64_rhs),
    .config_0(dual64_config), .data_output_0(dual64_result));
endmodule
)sv";
}

void validBehaviorAndToolInputs(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  const std::filesystem::path artifactRoot = root / "artifacts";
  std::filesystem::create_directories(artifactRoot);
  ArtifactStore store(artifactRoot.string());

  const std::array specs = {
      div8PaddedSpec(), rem8PaddedSpec(), configured8PaddedSpec(),
      configured8PaddedInactiveQuotientSpec(), configured64Spec()};
  std::vector<std::string> emitted;
  emitted.reserve(specs.size());
  for (const FabricSpec &spec : specs) {
    FabricFixture fixture = makeFabric(test, store, spec);
    checkCapability(test, fixture);
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
    emitted.push_back(emitDeterministically(test, fixture, abi));
  }

  const std::string yosysScript =
      R"ys(read_verilog -sv scalar_unsigned_div_8_padded.sv scalar_unsigned_rem_8_padded.sv scalar_unsigned_div_rem_8_padded.sv scalar_unsigned_div_rem_8_padded_inactive_quotient.sv scalar_unsigned_div_rem_64.sv synthesis_top.sv
hierarchy -check -top scalar_unsigned_div_rem_synthesis_top
proc
opt
check
select -assert-count 5 t:$div
select -assert-none t:$mod
synth -top scalar_unsigned_div_rem_synthesis_top -noabc
check -assert
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "provider_artifacts",
          {{"scalar_unsigned_div_8_padded.sv", emitted[0]},
           {"scalar_unsigned_rem_8_padded.sv", emitted[1]},
           {"scalar_unsigned_div_rem_8_padded.sv", emitted[2]},
           {"scalar_unsigned_div_rem_8_padded_inactive_quotient.sv",
            emitted[3]},
           {"scalar_unsigned_div_rem_64.sv", emitted[4]},
           {"testbench.sv", testbenchSource()},
           {"synthesis_top.sv", synthesisTopSource()},
           {"portable_scalar_unsigned_integer_div_rem.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void malformedInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  const std::filesystem::path artifactRoot = root / "artifacts";
  std::filesystem::create_directories(artifactRoot);
  ArtifactStore store(artifactRoot.string());

  FabricSpec wrongFamily = div8PaddedSpec();
  wrongFamily.name = "wrong_div_family";
  wrongFamily.familyKeyword = "ScalarIntegerMultiply";
  wrongFamily.family = Family::ScalarIntegerMultiply;
  expectFabricRejected(test, store, wrongFamily,
                       "not admitted by implementation family");
  FabricSpec wrongParameters = div8PaddedSpec();
  wrongParameters.name = "wrong_div_parameters";
  wrongParameters.parameters.clear();
  expectFabricRejected(test, store, wrongParameters, "integer_widths");

  FabricFixture valid = makeFabric(test, store, configured8PaddedSpec());
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  auto malformedParameters = capability(test, valid);
  malformedParameters.parameterizedCapability = ::fabric::ScalarFloatParams{
      ::fabric::FloatFormatSet::get({::fabric::FloatFormat::F32}),
      ::fabric::FloatBehaviorProfile::strictIEEE()};
  expectError(test,
              malformedParameters.resolveSemanticFieldRelation(fabricContext()),
              "parameter schema");

  std::unique_ptr<mlir::MLIRContext> portContext = makeCirctContext();
  SkeletonFixture wrongPorts =
      makeSkeleton(test, *portContext, valid, validAbi.abi(),
                   LeafMutation::WrongFirstInputWidth);
  expectError(test,
              specializePortableFor(wrongPorts, valid, validAbi, registry),
              "leaf port");

  for (ConfigurationAbiKind kind : {
           ConfigurationAbiKind::MissingRemainder,
           ConfigurationAbiKind::ExtraSemanticValue,
           ConfigurationAbiKind::DirectBits,
       }) {
    const llvm::StringRef expected = kind == ConfigurationAbiKind::DirectBits
                                         ? "finite codebook"
                                         : "semantic";
    expectError(test,
                finalizeConfigurationABI(
                    makeConfigurationAbiDraft(test, store, valid, kind), store),
                expected);
  }

  expectError(test,
              finalizeConfigurationABI(
                  makeConfigurationAbiDraft(test, store, valid,
                                            ConfigurationAbiKind::MissingField),
                  store),
              "cover");

  FabricFixture unsupportedContract =
      makeFabric(test, store, unsupportedContractSpec());
  FinalizedConfigurationABI contractAbi =
      makeConfigurationAbi(test, store, unsupportedContract);
  std::unique_ptr<mlir::MLIRContext> contractContext = makeCirctContext();
  SkeletonFixture contractSkeleton = makeSkeleton(
      test, *contractContext, unsupportedContract, contractAbi.abi());
  expectTypedUnsupported(test,
                         specializePortableFor(contractSkeleton,
                                               unsupportedContract, contractAbi,
                                               registry),
                         Family::ScalarUnsignedIntegerDivRem,
                         BackendRecipeKey::PortableSystemVerilog,
                         "unsupported div/rem resource contract");

  FabricFixture unsupportedShape =
      makeFabric(test, store, unsupportedShapeSpec());
  FinalizedConfigurationABI shapeAbi =
      makeConfigurationAbi(test, store, unsupportedShape);
  std::unique_ptr<mlir::MLIRContext> shapeContext = makeCirctContext();
  SkeletonFixture shapeSkeleton =
      makeSkeleton(test, *shapeContext, unsupportedShape, shapeAbi.abi());
  expectTypedUnsupported(test,
                         specializePortableFor(shapeSkeleton, unsupportedShape,
                                               shapeAbi, registry),
                         Family::ScalarUnsignedIntegerDivRem,
                         BackendRecipeKey::PortableSystemVerilog,
                         "unsupported div/rem physical shape");

  FabricFixture multiWidth = makeFabric(test, store, multiWidthSpec());
  auto multiWidthRelation =
      take(test, capability(test, multiWidth)
                     .resolveSemanticFieldRelation(fabricContext()));
  require(test,
          multiWidthRelation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              multiWidthRelation.finiteBehaviorDomain().size() == 2,
          "ABI 3.0 relation lost the div width dimension");

  FabricFixture multiOperationWidth =
      makeFabric(test, store, multiOperationWidthSpec());
  auto multiOperationWidthRelation =
      take(test, capability(test, multiOperationWidth)
                     .resolveSemanticFieldRelation(fabricContext()));
  require(test,
          multiOperationWidthRelation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              multiOperationWidthRelation.finiteBehaviorDomain().size() == 4,
          "ABI 3.0 relation lost the div/rem operation-width product");

  constexpr std::array nativeRecipes = {
      BackendRecipeKey::SynopsysDesignWare,
      BackendRecipeKey::CadenceChipWare,
      BackendRecipeKey::AmdXilinx,
      BackendRecipeKey::IntelAltera,
  };
  for (BackendRecipeKey recipe : nativeRecipes) {
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton =
        makeSkeleton(test, *context, valid, validAbi.abi());
    const std::string before = moduleText(*skeleton.module);
    expectTypedUnsupported(
        test, specializeNativeFor(skeleton, valid, validAbi, registry, recipe),
        Family::ScalarUnsignedIntegerDivRem, recipe,
        "backend-native div/rem recipe");
    require(test, moduleText(*skeleton.module) == before,
            "unsupported backend recipe partially mutated the caller module");
  }

  FabricFixture other = makeFabric(test, store, otherFamilySpec());
  FinalizedConfigurationABI otherAbi = makeConfigurationAbi(test, store, other);
  std::unique_ptr<mlir::MLIRContext> otherContext = makeCirctContext();
  SkeletonFixture otherSkeleton =
      makeSkeleton(test, *otherContext, other, otherAbi.abi());
  expectTypedUnsupported(
      test, specializePortableFor(otherSkeleton, other, otherAbi, registry),
      Family::ScalarIntegerMultiply, BackendRecipeKey::PortableSystemVerilog,
      "wrong-family capability");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  generatedOwnerAndProviderCoverage();
  validBehaviorAndToolInputs(root);
  malformedInputsAreTransactional(root / "malformed");
  return 0;
}

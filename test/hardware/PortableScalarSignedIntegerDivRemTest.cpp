#include "ConfigurationABI2TestSupport.h"
#include "PortableProviderTestSupport.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/ScalarSignedIntegerDivRem.h"

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
#include <memory>
#include <optional>
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
    fail(test, "accepted malformed signed integer div/rem input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectTypedUnsupported(llvm::StringRef test,
                            llvm::Expected<FabricOperationProviderOutput> value,
                            Family family, BackendRecipeKey recipe,
                            llvm::StringRef description) {
  require(test, !value, std::string("provider accepted ") + description.str());
  bool classified = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified =
            error.implementationFamily() == family && error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, description.str() +
                       " returned the wrong error class: " + error.message());
      });
  require(test, classified,
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
  bool unsupportedContract = false;
  bool inactiveQuotient = false;
};

FabricSpec div8PaddedSpec() {
  return {"scalar_signed_div_8_padded",
          "ScalarSignedIntegerDivRem",
          Family::ScalarSignedIntegerDivRem,
          {"arith.divsi"},
          "integer_widths = [8 : i32]",
          {13, 13},
          13};
}

FabricSpec rem8PaddedSpec() {
  return {"scalar_signed_rem_8_padded",
          "ScalarSignedIntegerDivRem",
          Family::ScalarSignedIntegerDivRem,
          {"arith.remsi"},
          "integer_widths = [8 : i32]",
          {13, 13},
          13};
}

FabricSpec configured8PaddedSpec() {
  return {"scalar_signed_div_rem_8_padded",
          "ScalarSignedIntegerDivRem",
          Family::ScalarSignedIntegerDivRem,
          {"arith.divsi", "arith.remsi"},
          "integer_widths = [8 : i32]",
          {13, 13},
          13};
}

FabricSpec configured8PaddedQuotientInactiveSpec() {
  FabricSpec spec = configured8PaddedSpec();
  spec.name = "scalar_signed_div_rem_8_padded_inactive_quotient";
  spec.inactiveQuotient = true;
  return spec;
}

FabricSpec activeWidthsSpec() {
  return {"scalar_signed_div_rem_active_widths",
          "ScalarSignedIntegerDivRem",
          Family::ScalarSignedIntegerDivRem,
          {"arith.divsi", "arith.remsi"},
          "integer_widths = [8 : i32, 16 : i32]",
          {21, 21},
          21};
}

FabricSpec configured64Spec() {
  return {"scalar_signed_div_rem_64",
          "ScalarSignedIntegerDivRem",
          Family::ScalarSignedIntegerDivRem,
          {"arith.divsi", "arith.remsi"},
          "integer_widths = [64 : i32]",
          {64, 64},
          64};
}

FabricSpec unsupportedContractSpec() {
  FabricSpec spec = configured8PaddedSpec();
  spec.name = "scalar_signed_div_rem_unsupported_contract";
  spec.unsupportedContract = true;
  return spec;
}

FabricSpec unsupportedShapeSpec() {
  FabricSpec spec = div8PaddedSpec();
  spec.name = "scalar_signed_div_unsupported_shape";
  spec.inputWidths.push_back(13);
  return spec;
}

FabricSpec otherFamilySpec() {
  return {"scalar_integer_multiply_other_family",
          "ScalarIntegerMultiply",
          Family::ScalarIntegerMultiply,
          {"arith.muli"},
          "integer_widths = [8 : i32]",
          {8, 8},
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
  auto source = mlir::parseSourceString<mlir::ModuleOp>(fabricSource(spec),
                                                        &fabricContext());
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
              "System has no physical signed div/rem occurrence");
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

unsigned
actorWidth(const ::dataflow::CanonicalActorSchemaProjection &representative) {
  return mlir::cast<mlir::IntegerType>(representative.type.getInput(0))
      .getWidth();
}

unsigned encodedBitCount(std::size_t behaviorCount) {
  unsigned width = 1;
  while ((std::size_t{1} << width) <= behaviorCount)
    ++width;
  return width;
}

std::uint8_t physicalModeCode(llvm::StringRef test, Schema schema,
                              unsigned width, std::size_t behaviorCount) {
  if (behaviorCount == 2)
    return schema == Schema::ArithDivSI ? 3 : 1;
  if (behaviorCount == 4 && width == 8)
    return schema == Schema::ArithDivSI ? 6 : 2;
  if (behaviorCount == 4 && width == 16)
    return schema == Schema::ArithDivSI ? 5 : 1;
  fail(test, "test ABI has no physical code for a relation mode");
}

enum class ConfigurationAbiKind {
  Complete,
  MissingSemanticValue,
  ExtraSemanticValue,
  DirectBits,
  MissingField,
};

ConfigurationABIDraft makeConfigurationAbiDraft(
    llvm::StringRef test, const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "configured signed div/rem fixture has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "configured signed div/rem relation is not finite");
  const auto domain = relation.finiteBehaviorDomain();
  require(test, !domain.empty(), "configured behavior domain is empty");

  std::vector<FiniteCodebookEntry> entries;
  entries.reserve(domain.size() + 1);
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured behavior has no semantic value");
    entries.push_back({{point.semanticConfiguration->bytes().begin(),
                        point.semanticConfiguration->bytes().end()},
                       {physicalModeCode(test, point.representativeActor.schema,
                                         actorWidth(point.representativeActor),
                                         domain.size())}});
  }
  if (kind == ConfigurationAbiKind::MissingSemanticValue)
    entries.back().semanticValue = {0xff};
  if (kind == ConfigurationAbiKind::ExtraSemanticValue)
    entries.push_back({{0xfe}, {0x02}});

  const unsigned bits = encodedBitCount(domain.size());
  SemanticFieldEncoding encoding =
      kind == ConfigurationAbiKind::DirectBits
          ? SemanticFieldEncoding{DirectBitsEncoding{bits}}
          : SemanticFieldEncoding{
                FiniteCodebookEncoding{bits, std::move(entries)}};
  const Schema inactiveSchema =
      fixture.spec.inactiveQuotient ? Schema::ArithDivSI : Schema::ArithRemSI;
  const auto inactive = llvm::find_if(domain, [&](const auto &point) {
    return point.representativeActor.schema == inactiveSchema;
  });
  const auto inactivePoint =
      inactive == domain.end() ? domain.begin() : inactive;
  const std::vector<std::uint8_t> inactiveValue =
      kind == ConfigurationAbiKind::DirectBits
          ? std::vector<std::uint8_t>{0}
          : std::vector<std::uint8_t>{
                inactivePoint->semanticConfiguration->bytes().begin(),
                inactivePoint->semanticConfiguration->bytes().end()};
  const auto fieldReference = resolved.configurationFieldSchema.front();
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
      const auto found = llvm::find_if(unit.fields, [&](const auto &candidate) {
        return candidate.field == physicalField;
      });
      if (found == unit.fields.end())
        continue;
      unit.fields.erase(found);
      removed = true;
      break;
    }
    require(test, removed, "could not remove signed div/rem field");
  }
  return draft;
}

FinalizedConfigurationABI makeConfigurationAbi(
    llvm::StringRef test, const ArtifactStore &store,
    const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  return take(test, finalizeConfigurationABI(
                        makeConfigurationAbiDraft(test, fixture, kind), store));
}

unsigned modeCode(llvm::StringRef test, const FabricFixture &fixture,
                  const FinalizedConfigurationABI &abi, Schema schema,
                  unsigned width) {
  auto relation = take(
      test,
      capability(test, fixture).resolveSemanticFieldRelation(fabricContext()));
  const auto point = llvm::find_if(
      relation.finiteBehaviorDomain(), [&](const auto &candidate) {
        return candidate.representativeActor.schema == schema &&
               actorWidth(candidate.representativeActor) == width;
      });
  require(test,
          point != relation.finiteBehaviorDomain().end() &&
              point->semanticConfiguration.has_value(),
          "configured relation has no requested signed div/rem mode");
  const auto &resolved = capability(test, fixture);
  require(test, resolved.configurationFieldSchema.size() == 1,
          "configured relation has no ABI field");
  const ConfigurationFieldEncoding *field = abi.abi().findOperationField(
      fixture.physicalOccurrence,
      resolved.configurationFieldSchema.front().ordinal);
  require(test, field != nullptr, "finalized ABI has no operation field");
  const auto *codebook =
      std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
  require(test, codebook != nullptr, "finalized ABI field has no codebook");
  const auto entry =
      llvm::find_if(codebook->entries, [&](const auto &candidate) {
        return llvm::ArrayRef<std::uint8_t>(candidate.semanticValue)
            .equals(point->semanticConfiguration->bytes());
      });
  require(test,
          entry != codebook->entries.end() && entry->physicalCode.size() == 1,
          "finalized ABI has no one-byte code for the requested mode");
  return entry->physicalCode.front();
}

unsigned spareCode(llvm::StringRef test, unsigned bitCount,
                   llvm::ArrayRef<unsigned> usedCodes) {
  for (unsigned code = (1U << bitCount) - 1; code != 0; --code)
    if (!llvm::is_contained(usedCodes, code))
      return code;
  fail(test, "test ABI has no spare physical code");
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
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports = take(
      test, deriveFabricOperationLeafPorts(builder, fixture.physicalOccurrence,
                                           capability(test, fixture), abi));
  if (mutation == LeafMutation::WrongFirstInputWidth) {
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
          registerPortableScalarSignedIntegerDivRemProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

llvm::Expected<FabricOperationProviderOutput> specializeFor(
    SkeletonFixture &skeleton, const FabricFixture &fixture,
    const FinalizedConfigurationABI &abi,
    const FabricOperationProviderRegistry &registry,
    BackendRecipeKey recipe = BackendRecipeKey::PortableSystemVerilog) {
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fixture.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fixture.physicalOccurrence, recipe, {}}};
  return specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                         recipes, registry, externalContracts);
}

std::string specializeWithSupport(llvm::StringRef test,
                                  SkeletonFixture skeleton,
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
          "portable signed div/rem emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

struct EmittedFixture final {
  FabricFixture fabric;
  FinalizedConfigurationABI abi;
  std::string rtl;
};

EmittedFixture emitDeterministically(llvm::StringRef test,
                                     const ArtifactStore &store,
                                     FabricSpec spec) {
  FabricFixture fixture = makeFabric(test, store, std::move(spec));
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fixture);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  std::string first = specializeWithSupport(
      test, makeSkeleton(test, *firstContext, fixture, abi.abi()), fixture,
      abi);
  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  std::string second = specializeWithSupport(
      test, makeSkeleton(test, *secondContext, fixture, abi.abi()), fixture,
      abi);
  require(test, first == second,
          "identical signed div/rem inputs produced different RTL bytes");

  auto relation = take(
      test,
      capability(test, fixture).resolveSemanticFieldRelation(fabricContext()));
  std::vector<unsigned> widths;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    const unsigned width = actorWidth(point.representativeActor);
    if (!llvm::is_contained(widths, width))
      widths.push_back(width);
  }
  require(test,
          llvm::StringRef(first).count(" / ") == widths.size() &&
              llvm::StringRef(first).count("$signed") >= widths.size() * 2 &&
              !llvm::StringRef(first).contains(" % "),
          "signed div/rem did not emit one shared signed divider per width");
  return {std::move(fixture), std::move(abi), std::move(first)};
}

void checkCapability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  require(test,
          resolved.implementationFamily == fixture.spec.family &&
              !resolved.enabledOperationSchemas.empty(),
          "signed div/rem capability changed its generated identity");
  const auto &descriptor =
      ::fabric::implementationFamily(resolved.implementationFamily);
  require(test,
          descriptor.familyId == resolved.implementationFamily &&
              descriptor.admittedSchemas.size() == 2 &&
              llvm::all_of(resolved.enabledOperationSchemas,
                           [&](Schema schema) {
                             return llvm::is_contained(
                                 descriptor.admittedSchemas, schema);
                           }),
          "signed div/rem capability escaped its family descriptor");
  const auto *parameters = std::get_if<::fabric::ScalarIntegerParams>(
      &resolved.parameterizedCapability);
  require(test,
          parameters && parameters->integerWidths.valid() &&
              !parameters->integerWidths.empty() &&
              parameters->pointerFormats.empty(),
          "signed div/rem capability changed its scalar integer parameters");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  const std::size_t expectedModes = resolved.enabledOperationSchemas.size() *
                                    parameters->integerWidths.size();
  require(test,
          relation.finiteBehaviorDomain().size() == expectedModes &&
              relation.kind() ==
                  (expectedModes == 1
                       ? ::fabric::FabricOpSemanticFieldRelationKind::None
                       : ::fabric::FabricOpSemanticFieldRelationKind::Finite),
          "sealed signed div/rem relation changed its operation-width product");
}

void generatedOwnerAndProviderCoverage() {
  const llvm::StringRef test = __func__;
  constexpr Family family = Family::ScalarSignedIntegerDivRem;
  const auto &descriptor = ::fabric::implementationFamily(family);
  require(
      test,
      descriptor.familyId == family &&
          descriptor.capabilityParamsSchema ==
              ::fabric::CapabilityParamsSchemaId::ScalarIntegerParams &&
          descriptor.typedAdmissionProvider ==
              ::fabric::TypedAdmissionProviderId::
                  ScalarOrdinaryIntegerAdmission &&
          llvm::is_contained(descriptor.admittedSchemas, Schema::ArithDivSI) &&
          llvm::is_contained(descriptor.admittedSchemas, Schema::ArithRemSI),
      "generated scalar signed div/rem descriptor changed");
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
          "signed div/rem provider coverage is not portable-only");
}

struct ModeCodes final {
  unsigned dualDiv = 0;
  unsigned dualRem = 0;
  unsigned dualSpare = 0;
  unsigned dualQuotientInactiveDiv = 0;
  unsigned dualQuotientInactiveRem = 0;
  unsigned dualQuotientInactiveSpare = 0;
  unsigned activeDiv8 = 0;
  unsigned activeRem8 = 0;
  unsigned activeDiv16 = 0;
  unsigned activeRem16 = 0;
  unsigned activeSpare = 0;
  unsigned dual64Div = 0;
  unsigned dual64Rem = 0;
};

std::string testbenchSource(const ModeCodes &codes) {
  std::ostringstream stream;
  stream << R"sv(module testbench;
  logic [12:0] div8_lhs, div8_rhs;
  logic [12:0] div8_result;
  logic [12:0] rem8_lhs, rem8_rhs;
  logic [12:0] rem8_result;
  logic [12:0] dual8_lhs, dual8_rhs;
  logic [1:0] dual8_config;
  logic [12:0] dual8_result;
  logic [12:0] dual8q_lhs, dual8q_rhs;
  logic [1:0] dual8q_config;
  logic [12:0] dual8q_result;
  logic [20:0] active_lhs, active_rhs;
  logic [2:0] active_config;
  logic [20:0] active_result;
  logic [63:0] dual64_lhs, dual64_rhs;
  logic [1:0] dual64_config;
  logic [63:0] dual64_result;

  scalar_signed_div_8_padded div8(
    .data_input_0(div8_lhs), .data_input_1(div8_rhs),
    .data_output_0(div8_result));
  scalar_signed_rem_8_padded rem8(
    .data_input_0(rem8_lhs), .data_input_1(rem8_rhs),
    .data_output_0(rem8_result));
  scalar_signed_div_rem_8_padded dual8(
    .data_input_0(dual8_lhs), .data_input_1(dual8_rhs),
    .config_0(dual8_config), .data_output_0(dual8_result));
  scalar_signed_div_rem_8_padded_inactive_quotient dual8q(
    .data_input_0(dual8q_lhs), .data_input_1(dual8q_rhs),
    .config_0(dual8q_config), .data_output_0(dual8q_result));
  scalar_signed_div_rem_active_widths active(
    .data_input_0(active_lhs), .data_input_1(active_rhs),
    .config_0(active_config), .data_output_0(active_result));
  scalar_signed_div_rem_64 dual64(
    .data_input_0(dual64_lhs), .data_input_1(dual64_rhs),
    .config_0(dual64_config), .data_output_0(dual64_result));

  task automatic check_div8(
      input logic [12:0] lhs, input logic [12:0] rhs,
      input logic [7:0] expected);
    begin
      div8_lhs = lhs;
      div8_rhs = rhs;
      #1;
      if (div8_result !== {5'h00, expected})
        $fatal(1, "signed quotient oracle mismatch");
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
        $fatal(1, "quotient-inactive signed div/rem oracle mismatch");
    end
  endtask

  task automatic check_rem8(
      input logic [12:0] lhs, input logic [12:0] rhs,
      input logic [7:0] expected);
    begin
      rem8_lhs = lhs;
      rem8_rhs = rhs;
      #1;
      if (rem8_result !== {5'h00, expected})
        $fatal(1, "signed remainder oracle mismatch");
    end
  endtask

  task automatic check_dual8(
      input logic [12:0] lhs, input logic [12:0] rhs,
      input logic [1:0] configuration, input logic [7:0] expected);
    begin
      dual8_lhs = lhs;
      dual8_rhs = rhs;
      dual8_config = configuration;
      #1;
      if (dual8_result !== {5'h00, expected})
        $fatal(1, "configured signed div/rem oracle mismatch");
    end
  endtask

  task automatic check_active(
      input logic [20:0] lhs, input logic [20:0] rhs,
      input logic [2:0] configuration, input logic [20:0] expected);
    begin
      active_lhs = lhs;
      active_rhs = rhs;
      active_config = configuration;
      #1;
      if (active_result !== expected)
        $fatal(1, "active-width signed div/rem oracle mismatch");
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
        $fatal(1, "64-bit signed div/rem oracle mismatch");
    end
  endtask

  task automatic check_div8_non_defined(
      input logic [12:0] lhs, input logic [12:0] rhs);
    begin
      div8_lhs = lhs;
      div8_rhs = rhs;
      #1;
      if ($isunknown(div8_result))
        $fatal(1, "non-defined signed quotient produced unknown bits");
    end
  endtask

  task automatic check_rem8_non_defined(
      input logic [12:0] lhs, input logic [12:0] rhs);
    begin
      rem8_lhs = lhs;
      rem8_rhs = rhs;
      #1;
      if ($isunknown(rem8_result))
        $fatal(1, "non-defined signed remainder produced unknown bits");
    end
  endtask

  initial begin
    check_div8(13'h0064, 13'h0007, 8'h0e);
    check_div8(13'h009c, 13'h0007, 8'hf2);
    check_div8(13'h0064, 13'h00f9, 8'hf2);
    check_div8(13'h009c, 13'h00f9, 8'h0e);
    check_div8(13'h0007, 13'h0003, 8'h02);
    check_div8(13'h00f9, 13'h0003, 8'hfe);
    check_div8(13'h1f9c, 13'h1a07, 8'hf2);

    check_rem8(13'h0064, 13'h0007, 8'h02);
    check_rem8(13'h009c, 13'h0007, 8'hfe);
    check_rem8(13'h0064, 13'h00f9, 8'h02);
    check_rem8(13'h009c, 13'h00f9, 8'hfe);
    check_rem8(13'h0007, 13'h0003, 8'h01);
    check_rem8(13'h00f9, 13'h0003, 8'hff);
    check_rem8(13'h1f9c, 13'h1a07, 8'hfe);
    check_rem8(13'h0080, 13'h00ff, 8'h00);
)sv";
  stream << "    check_dual8(13'h009c, 13'h0007, 2'd" << codes.dualDiv
         << ", 8'hf2);\n";
  stream << "    check_dual8(13'h009c, 13'h0007, 2'd" << codes.dualRem
         << ", 8'hfe);\n";
  stream << "    check_dual8(13'h009c, 13'h0007, 2'd0, 8'hfe);\n";
  stream << "    check_dual8(13'h009c, 13'h0007, 2'd" << codes.dualSpare
         << ", 8'hfe);\n";
  stream << "    check_dual8q(13'h009c, 13'h0007, 2'd"
         << codes.dualQuotientInactiveDiv << ", 8'hf2);\n";
  stream << "    check_dual8q(13'h009c, 13'h0007, 2'd"
         << codes.dualQuotientInactiveRem << ", 8'hfe);\n";
  stream << "    check_dual8q(13'h009c, 13'h0007, 2'd0, 8'hf2);\n";
  stream << "    check_dual8q(13'h009c, 13'h0007, 2'd"
         << codes.dualQuotientInactiveSpare << ", 8'hf2);\n";
  stream << "    check_active({13'h1fff, 8'h9c}, {13'h1555, 8'h07}, 3'd"
         << codes.activeDiv8 << ", {13'h0000, 8'hf2});\n";
  stream << "    check_active({13'h1fff, 8'h9c}, {13'h1555, 8'h07}, 3'd"
         << codes.activeRem8 << ", {13'h0000, 8'hfe});\n";
  stream << "    check_active({5'h1f, 16'h8ad0}, {5'h15, 16'h0007}, 3'd"
         << codes.activeDiv16 << ", {5'h00, 16'hef43});\n";
  stream << "    check_active({5'h1f, 16'h8ad0}, {5'h15, 16'h0007}, 3'd"
         << codes.activeRem16 << ", {5'h00, 16'hfffb});\n";
  stream << "    check_active({13'h1fff, 8'h9c}, {13'h1555, 8'h07}, 3'd0, "
            "{13'h0000, 8'hfe});\n";
  stream << "    check_active({13'h1fff, 8'h9c}, {13'h1555, 8'h07}, 3'd"
         << codes.activeSpare << ", {13'h0000, 8'hfe});\n";
  stream << "    check_dual64(64'hffffffffffffff9c, 64'h0000000000000007, 2'd"
         << codes.dual64Div << ", 64'hfffffffffffffff2);\n";
  stream << "    check_dual64(64'hffffffffffffff9c, 64'h0000000000000007, 2'd"
         << codes.dual64Rem << ", 64'hfffffffffffffffe);\n";
  stream << "    check_dual64(64'h8000000000000000, 64'hffffffffffffffff, 2'd"
         << codes.dual64Rem << ", 64'h0000000000000000);\n";
  stream << R"sv(
    check_div8_non_defined(13'h0064, 13'h0000);
    check_div8_non_defined(13'h0080, 13'h00ff);
    check_rem8_non_defined(13'h0064, 13'h0000);
    $finish;
  end
endmodule
)sv";
  return stream.str();
}

std::string synthesisTopSource() {
  return R"sv(module scalar_signed_div_rem_synthesis_top(
  input [12:0] div8_lhs, input [12:0] div8_rhs,
  input [12:0] rem8_lhs, input [12:0] rem8_rhs,
  input [12:0] dual8_lhs, input [12:0] dual8_rhs,
  input [1:0] dual8_config,
  input [12:0] dual8q_lhs, input [12:0] dual8q_rhs,
  input [1:0] dual8q_config,
  input [20:0] active_lhs, input [20:0] active_rhs,
  input [2:0] active_config,
  input [63:0] dual64_lhs, input [63:0] dual64_rhs,
  input [1:0] dual64_config,
  output [12:0] div8_result, output [12:0] rem8_result,
  output [12:0] dual8_result, output [12:0] dual8q_result,
  output [20:0] active_result,
  output [63:0] dual64_result
);
  scalar_signed_div_8_padded div8(
    .data_input_0(div8_lhs), .data_input_1(div8_rhs),
    .data_output_0(div8_result));
  scalar_signed_rem_8_padded rem8(
    .data_input_0(rem8_lhs), .data_input_1(rem8_rhs),
    .data_output_0(rem8_result));
  scalar_signed_div_rem_8_padded dual8(
    .data_input_0(dual8_lhs), .data_input_1(dual8_rhs),
    .config_0(dual8_config), .data_output_0(dual8_result));
  scalar_signed_div_rem_8_padded_inactive_quotient dual8q(
    .data_input_0(dual8q_lhs), .data_input_1(dual8q_rhs),
    .config_0(dual8q_config), .data_output_0(dual8q_result));
  scalar_signed_div_rem_active_widths active(
    .data_input_0(active_lhs), .data_input_1(active_rhs),
    .config_0(active_config), .data_output_0(active_result));
  scalar_signed_div_rem_64 dual64(
    .data_input_0(dual64_lhs), .data_input_1(dual64_rhs),
    .config_0(dual64_config), .data_output_0(dual64_result));
endmodule
)sv";
}

std::string yosysScriptSource() {
  return R"ys(read_verilog -sv scalar_signed_div_8_padded.sv scalar_signed_rem_8_padded.sv scalar_signed_div_rem_8_padded.sv scalar_signed_div_rem_8_padded_inactive_quotient.sv scalar_signed_div_rem_active_widths.sv scalar_signed_div_rem_64.sv synthesis_top.sv
hierarchy -check -top scalar_signed_div_rem_synthesis_top
proc
opt
check
select -assert-count 7 t:$div
select -assert-none t:$mod
synth -top scalar_signed_div_rem_synthesis_top -noabc
check -assert
stat
)ys";
}

void validBehaviorAndToolInputs(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());

  EmittedFixture div8 = emitDeterministically(test, store, div8PaddedSpec());
  EmittedFixture rem8 = emitDeterministically(test, store, rem8PaddedSpec());
  EmittedFixture dual8 =
      emitDeterministically(test, store, configured8PaddedSpec());
  EmittedFixture dual8q = emitDeterministically(
      test, store, configured8PaddedQuotientInactiveSpec());
  EmittedFixture active =
      emitDeterministically(test, store, activeWidthsSpec());
  EmittedFixture dual64 =
      emitDeterministically(test, store, configured64Spec());
  for (const FabricFixture *fixture :
       {&div8.fabric, &rem8.fabric, &dual8.fabric, &dual8q.fabric,
        &active.fabric, &dual64.fabric})
    checkCapability(test, *fixture);

  const unsigned dualDiv =
      modeCode(test, dual8.fabric, dual8.abi, Schema::ArithDivSI, 8);
  const unsigned dualRem =
      modeCode(test, dual8.fabric, dual8.abi, Schema::ArithRemSI, 8);
  const std::array dualUsed{dualDiv, dualRem};
  const unsigned dualQuotientInactiveDiv =
      modeCode(test, dual8q.fabric, dual8q.abi, Schema::ArithDivSI, 8);
  const unsigned dualQuotientInactiveRem =
      modeCode(test, dual8q.fabric, dual8q.abi, Schema::ArithRemSI, 8);
  const std::array dualQuotientInactiveUsed{dualQuotientInactiveDiv,
                                            dualQuotientInactiveRem};
  const unsigned activeDiv8 =
      modeCode(test, active.fabric, active.abi, Schema::ArithDivSI, 8);
  const unsigned activeRem8 =
      modeCode(test, active.fabric, active.abi, Schema::ArithRemSI, 8);
  const unsigned activeDiv16 =
      modeCode(test, active.fabric, active.abi, Schema::ArithDivSI, 16);
  const unsigned activeRem16 =
      modeCode(test, active.fabric, active.abi, Schema::ArithRemSI, 16);
  const std::array activeUsed{activeDiv8, activeRem8, activeDiv16, activeRem16};
  const ModeCodes codes{
      dualDiv,
      dualRem,
      spareCode(test, 2, dualUsed),
      dualQuotientInactiveDiv,
      dualQuotientInactiveRem,
      spareCode(test, 2, dualQuotientInactiveUsed),
      activeDiv8,
      activeRem8,
      activeDiv16,
      activeRem16,
      spareCode(test, 3, activeUsed),
      modeCode(test, dual64.fabric, dual64.abi, Schema::ArithDivSI, 64),
      modeCode(test, dual64.fabric, dual64.abi, Schema::ArithRemSI, 64)};

  const std::vector<loom::hardware::test::PortableProviderArtifact> artifacts =
      {{"scalar_signed_div_8_padded.sv", std::move(div8.rtl)},
       {"scalar_signed_rem_8_padded.sv", std::move(rem8.rtl)},
       {"scalar_signed_div_rem_8_padded.sv", std::move(dual8.rtl)},
       {"scalar_signed_div_rem_8_padded_inactive_quotient.sv",
        std::move(dual8q.rtl)},
       {"scalar_signed_div_rem_active_widths.sv", std::move(active.rtl)},
       {"scalar_signed_div_rem_64.sv", std::move(dual64.rtl)},
       {"testbench.sv", testbenchSource(codes)},
       {"synthesis_top.sv", synthesisTopSource()},
       {"portable_scalar_signed_integer_div_rem.ys", yosysScriptSource()}};
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts", artifacts))
    fail(test, llvm::toString(std::move(error)));
}

void malformedInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());

  FabricSpec wrongFamily = div8PaddedSpec();
  wrongFamily.name = "wrong_signed_div_family";
  wrongFamily.familyKeyword = "ScalarIntegerMultiply";
  wrongFamily.family = Family::ScalarIntegerMultiply;
  expectFabricRejected(test, store, wrongFamily,
                       "not admitted by implementation family");
  FabricSpec wrongParameters = div8PaddedSpec();
  wrongParameters.name = "wrong_signed_div_parameters";
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
  const std::string portBefore = moduleText(*wrongPorts.module);
  expectError(test, specializeFor(wrongPorts, valid, validAbi, registry),
              "leaf port");
  require(test, moduleText(*wrongPorts.module) == portBefore,
          "malformed signed div/rem leaf partially mutated the module");

  for (const auto [kind, expected] :
       std::array<std::pair<ConfigurationAbiKind, llvm::StringRef>, 4>{
           {{ConfigurationAbiKind::MissingSemanticValue, "finite codebook"},
            {ConfigurationAbiKind::ExtraSemanticValue, "semantic"},
            {ConfigurationAbiKind::DirectBits, "finite codebook"},
            {ConfigurationAbiKind::MissingField, "cover"}}})
    expectError(test,
                finalizeConfigurationABI(
                    makeConfigurationAbiDraft(test, valid, kind), store),
                expected);

  FabricFixture contract = makeFabric(test, store, unsupportedContractSpec());
  FinalizedConfigurationABI contractAbi =
      makeConfigurationAbi(test, store, contract);
  std::unique_ptr<mlir::MLIRContext> contractContext = makeCirctContext();
  SkeletonFixture contractSkeleton =
      makeSkeleton(test, *contractContext, contract, contractAbi.abi());
  const std::string contractBefore = moduleText(*contractSkeleton.module);
  expectTypedUnsupported(
      test, specializeFor(contractSkeleton, contract, contractAbi, registry),
      Family::ScalarSignedIntegerDivRem,
      BackendRecipeKey::PortableSystemVerilog, "unsupported resource contract");
  require(test, moduleText(*contractSkeleton.module) == contractBefore,
          "unsupported resource contract partially mutated the module");

  FabricFixture shape = makeFabric(test, store, unsupportedShapeSpec());
  FinalizedConfigurationABI shapeAbi = makeConfigurationAbi(test, store, shape);
  std::unique_ptr<mlir::MLIRContext> shapeContext = makeCirctContext();
  SkeletonFixture shapeSkeleton =
      makeSkeleton(test, *shapeContext, shape, shapeAbi.abi());
  const std::string shapeBefore = moduleText(*shapeSkeleton.module);
  expectTypedUnsupported(
      test, specializeFor(shapeSkeleton, shape, shapeAbi, registry),
      Family::ScalarSignedIntegerDivRem,
      BackendRecipeKey::PortableSystemVerilog, "unsupported physical shape");
  require(test, moduleText(*shapeSkeleton.module) == shapeBefore,
          "unsupported physical shape partially mutated the module");

  constexpr std::array nativeRecipes = {
      BackendRecipeKey::SynopsysDesignWare, BackendRecipeKey::CadenceChipWare,
      BackendRecipeKey::AmdXilinx, BackendRecipeKey::IntelAltera};
  for (BackendRecipeKey recipe : nativeRecipes) {
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton =
        makeSkeleton(test, *context, valid, validAbi.abi());
    const std::string before = moduleText(*skeleton.module);
    expectTypedUnsupported(
        test, specializeFor(skeleton, valid, validAbi, registry, recipe),
        Family::ScalarSignedIntegerDivRem, recipe, "backend-native recipe");
    require(test, moduleText(*skeleton.module) == before,
            "unsupported backend recipe partially mutated the module");
  }

  FabricFixture other = makeFabric(test, store, otherFamilySpec());
  FinalizedConfigurationABI otherAbi = makeConfigurationAbi(test, store, other);
  std::unique_ptr<mlir::MLIRContext> otherContext = makeCirctContext();
  SkeletonFixture otherSkeleton =
      makeSkeleton(test, *otherContext, other, otherAbi.abi());
  const std::string otherBefore = moduleText(*otherSkeleton.module);
  expectTypedUnsupported(
      test, specializeFor(otherSkeleton, other, otherAbi, registry),
      Family::ScalarIntegerMultiply, BackendRecipeKey::PortableSystemVerilog,
      "wrong-family capability");
  require(test, moduleText(*otherSkeleton.module) == otherBefore,
          "wrong-family capability partially mutated the module");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root = std::filesystem::absolute(argv[1]);
  generatedOwnerAndProviderCoverage();
  validBehaviorAndToolInputs(root);
  malformedInputsAreTransactional(root / "malformed");
  return 0;
}

#include "ADG/Builder.h"
#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FixedVectorValueSelect.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/OperationResourceContract.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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
using loom::adg::DesignBuilder;
using loom::adg::FinalizedFabricDesign;
using loom::adg::FuCapabilityTemplateSpec;
using loom::adg::FuSpec;
using loom::adg::OperationCapabilitySpec;
using loom::adg::PeSpec;
using loom::adg::PortType;
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

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted invalid portable fixed-vector select input");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid portable fixed-vector select input");
  expectError(test, value.takeError(), expected);
}

template <typename T>
void expectTypedUnsupported(llvm::StringRef test, llvm::Expected<T> value,
                            llvm::StringRef description) {
  require(test, !value, std::string("provider accepted ") + description.str());
  bool typedUnsupported = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        typedUnsupported =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::FixedVectorValueSelect &&
            error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, description.str() +
                       " returned the wrong error class: " + error.message());
      });
  require(test, typedUnsupported,
          description.str() + " lost its typed Unsupported classification");
}

::fabric::FixedVectorValueSelectParams configuredParameters() {
  return {::fabric::IntegerWidthSet::get(
              {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16}),
          ::fabric::FloatFormatSet::get(
              {::fabric::FloatFormat::F16, ::fabric::FloatFormat::F32}),
          130};
}

::fabric::FixedVectorValueSelectParams singletonParameters() {
  return {::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I16}),
          ::fabric::FloatFormatSet::get({::fabric::FloatFormat::F16}), 64};
}

struct FabricFixture final {
  FinalizedFabricDesign design;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;

  const loom::fabric::FinalizedFabricRoot &root() const {
    return design.roots().front();
  }
};

FabricFixture findFixture(llvm::StringRef test, FinalizedFabricDesign design,
                          const ArtifactStore &store) {
  require(test, design.roots().size() == 1,
          "Fabric fixture did not finalize one root");
  const auto &root = design.roots().front();
  for (const auto fuOccurrence : root.view().fuOccurrences()) {
    const auto definition = root.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &candidate :
         root.view().resolvedFabricOpCapabilities(*definition)) {
      if (candidate.implementationFamily !=
          ::fabric::ImplementationFamilyId::FixedVectorValueSelect)
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         root.view(), candidate.occurrence, fuOccurrence));
      FinalizedFabricRoot system = take(
          test, loom::hardware::test::makeSingleSpatialCoreSystem(root, store));
      auto systemView =
          take(test, loom::fabric::requireSystemRoot(system.view()));
      auto operations =
          take(test, enumerateFabricPhysicalOperations(systemView));
      const auto physical = llvm::find_if(operations, [&](const auto &entry) {
        return entry.localOccurrence == occurrence;
      });
      require(test, physical != operations.end(),
              "System has no physical vector select occurrence");
      return FabricFixture{std::move(design), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no fixed-vector value select occurrence");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.root().view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         llvm::StringRef label,
                         ::fabric::FixedVectorValueSelectParams parameters,
                         llvm::ArrayRef<unsigned> inputWidths,
                         unsigned outputWidth,
                         bool unsupportedContract = false) {
  DesignBuilder design(store);
  std::vector<PortType> operationInputTypes;
  operationInputTypes.reserve(inputWidths.size());
  for (unsigned width : inputWidths)
    operationInputTypes.push_back(take(test, PortType::bits(width)));
  const PortType operationOutputType = take(test, PortType::bits(outputWidth));
  const unsigned boundaryWidth = std::max(
      outputWidth, *std::max_element(inputWidths.begin(), inputWidths.end()));
  const PortType boundaryType = take(test, PortType::bits(boundaryWidth));
  const std::vector<PortType> boundaryInputTypes(inputWidths.size(),
                                                 boundaryType);

  auto spatial = take(test, design.createSpatialCore(label, boundaryInputTypes,
                                                     {boundaryType}));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != boundaryInputTypes.size(); ++ordinal)
    spatialInputs.push_back(take(test, spatial.input(ordinal)));
  auto pe = take(
      test, spatial.addPe(spatialInputs,
                          PeSpec::spatial(boundaryInputTypes, {boundaryType})));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != boundaryInputTypes.size(); ++ordinal)
    peInputs.push_back(take(test, pe.input(ordinal)));
  auto fu = take(
      test, pe.addFu(peInputs, FuSpec{operationInputTypes, {boundaryType}}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal != operationInputTypes.size();
       ++ordinal)
    fuInputs.push_back(take(test, fu.input(ordinal)));

  const auto family = ::fabric::ImplementationFamilyId::FixedVectorValueSelect;
  const auto &descriptor = ::fabric::implementationFamily(family);
  const ::fabric::ResourceContract contract =
      unsupportedContract
          ? ::fabric::loopCarryOperationResourceContract()
          : ::fabric::oneCycleElasticOperationResourceContract();
  auto operation =
      take(test, fu.addOperation(fuInputs,
                                 OperationCapabilitySpec{
                                     family,
                                     std::move(parameters),
                                     std::vector<::dataflow::OperationSchemaId>(
                                         descriptor.admittedSchemas.begin(),
                                         descriptor.admittedSchemas.end()),
                                     {operationOutputType},
                                     contract}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = fu.close({take(test, operation.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = spatial.close({take(test, pe.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  return findFixture(test, take(test, std::move(design).finalize()), store);
}

mlir::MLIRContext &fabricContext() {
  static mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  return context;
}

::dataflow::CanonicalActorSchemaProjection
selectActor(mlir::Type elementType, llvm::ArrayRef<std::int64_t> shape) {
  mlir::MLIRContext &context = fabricContext();
  mlir::VectorType values = mlir::VectorType::get(shape, elementType);
  mlir::VectorType condition =
      mlir::VectorType::get(shape, mlir::IntegerType::get(&context, 1));
  return {
      ::dataflow::OperationSchemaId::ArithSelect,
      mlir::FunctionType::get(&context, {condition, values, values}, {values}),
      ::dataflow::NoPayload{}};
}

std::vector<std::uint8_t>
configurationValue(llvm::StringRef test,
                   const loom::fabric::ResolvedFabricOpCapabilityView &resolved,
                   mlir::Type elementType, llvm::ArrayRef<std::int64_t> shape) {
  require(test, resolved.configurationFieldSchema.size() == 1,
          "configured select capability has an unexpected field count");
  constexpr std::array<std::uint64_t, 3> operandPorts = {0, 1, 2};
  constexpr std::array<std::uint64_t, 1> resultPorts = {0};
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  const loom::CanonicalSemanticBytes encoded =
      take(test, relation.projectSemanticValue(
                     selectActor(elementType, shape), operandPorts, resultPorts,
                     ::fabric::ResolvedIndexWidth::I64));
  return std::vector<std::uint8_t>(encoded.bytes().begin(),
                                   encoded.bytes().end());
}

unsigned behaviorElementWidth(
    const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  return mlir::cast<mlir::VectorType>(
             point.representativeActor.type.getInput(1))
      .getElementTypeBitWidth();
}

std::vector<FiniteCodebookEntry>
completeEntries(llvm::StringRef test,
                const loom::fabric::ResolvedFabricOpCapabilityView &resolved) {
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "vector select field relation is not finite");
  const auto domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 3,
          "Fabric did not collapse select behavior to three element widths");

  std::vector<FiniteCodebookEntry> entries;
  entries.reserve(domain.size());
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured Fabric behavior has no semantic value");
    require(test,
            point.representativeActor.schema ==
                ::dataflow::OperationSchemaId::ArithSelect,
            "Fabric projected a non-select behavior");
    const unsigned width = behaviorElementWidth(point);
    const std::uint8_t code = width == 8 ? 0x01 : width == 16 ? 0x02 : 0x04;
    require(test, width == 8 || width == 16 || width == 32,
            "Fabric projected an unexpected select element width");
    entries.push_back(
        {std::vector<std::uint8_t>(point.semanticConfiguration->bytes().begin(),
                                   point.semanticConfiguration->bytes().end()),
         {code}});
  }
  return entries;
}

enum class ConfigurationAbiKind {
  Complete,
  MissingWidth8,
  ExtraSemanticValue,
};

ConfigurationABIDraft makeConfigurationAbiDraft(
    llvm::StringRef test, const ArtifactStore &store,
    const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));

  std::vector<FiniteCodebookEntry> entries = completeEntries(test, resolved);
  if (kind == ConfigurationAbiKind::MissingWidth8) {
    const auto missing = llvm::find_if(entries, [](const auto &entry) {
      return entry.physicalCode == std::vector<std::uint8_t>{0x01};
    });
    require(test, missing != entries.end(),
            "eight-bit behavior is absent from the Fabric domain");
    missing->semanticValue = {0xfd};
  }
  if (kind == ConfigurationAbiKind::ExtraSemanticValue)
    entries.push_back({{0xfe}, {0x03}});
  const auto inactive = llvm::find_if(entries, [](const auto &entry) {
    return entry.physicalCode == std::vector<std::uint8_t>{0x02};
  });
  require(test, inactive != entries.end(),
          "sixteen-bit behavior is absent from the Fabric domain");
  const std::vector<std::uint8_t> inactiveValue = inactive->semanticValue;

  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, FiniteCodebookEncoding{3, std::move(entries)},
      inactiveValue};
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, {std::move(field)}));
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

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

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
    require(test, ports.size() == 5 && ports[3].getName() == "config_0",
            "configured select leaf did not expose its exact selector");
    ports[3].type = builder.getI2Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("fixed_vector_value_select"), ports);
  return SkeletonFixture{std::move(module), leaf};
}

std::string specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableFixedVectorValueSelectProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  auto conformance =
      take(test, loom::hardware::test::specializeAndExportPortableProvider(
                     {std::move(skeleton.module),
                      {{skeleton.leaf, fabric.physicalOccurrence}}},
                     abi, registry, externalContracts));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "portable vector select emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

void registrationIsPortableOnly() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableFixedVectorValueSelectProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  const auto coverage = registry.coverage();
  require(test, coverage.size() == ::fabric::implementationFamilyCount(),
          "provider coverage lost the generated family cardinality");
  const auto entry = llvm::find_if(coverage, [](const auto &candidate) {
    return candidate.implementationFamily ==
           ::fabric::ImplementationFamilyId::FixedVectorValueSelect;
  });
  require(test,
          entry != coverage.end() &&
              entry->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "fixed-vector select registered a non-portable recipe");
}

void writeToolInputs(const std::filesystem::path &root, llvm::StringRef rtl) {
  const llvm::StringRef test = __func__;
  const std::string testbench = R"sv(
module testbench;
  logic [129:0] data_input_0;
  logic [129:0] data_input_1;
  logic [129:0] data_input_2;
  logic [2:0] config_0;
  logic [129:0] data_output_0;

  fixed_vector_value_select dut(.*);

  initial begin
    data_input_0 = 130'h3_ffffffffffffffff_ffffffff00000005;
    data_input_1 = 130'h3_1111111111111111_1111111104030201;
    data_input_2 = 130'h2_eeeeeeeeeeeeeeee_eeeeeeeed4c3b2a1;

    config_0 = 3'b001;
    #1;
    if (data_output_0 !== 130'h0_eeeeeeeeeeeeeeee_eeeeeeeed403b201)
      $fatal(1, "eight-bit lane selection or low-bit transport failed");

    config_0 = 3'b010;
    #1;
    if (data_output_0 !== 130'h0_eeeeeeeeeeeeeeee_eeee1111d4c30201)
      $fatal(1, "sixteen-bit lane selection or low-bit transport failed");

    config_0 = 3'b100;
    #1;
    if (data_output_0 !== 130'h0_eeeeeeee11111111_eeeeeeee04030201)
      $fatal(1, "thirty-two-bit lane selection or low-bit transport failed");

    config_0 = 3'b000;
    #1;
    if (data_output_0 !== 130'h0_eeeeeeeeeeeeeeee_eeee1111d4c30201)
      $fatal(1, "unassigned code did not preserve inactive behavior");
    $finish;
  end
endmodule
)sv";
  const std::string yosysScript = R"ys(
read_verilog -sv fixed_vector_value_select.sv
hierarchy -check -top fixed_vector_value_select
proc
opt
check -assert
synth -top fixed_vector_value_select
check -assert
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "tool-artifacts",
          {{"fixed_vector_value_select.sv", rtl.str()},
           {"testbench.sv", testbench},
           {"portable_fixed_vector_value_select.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void configuredBehaviorAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric =
      makeFabric(test, store, "fixed-vector-value-select",
                 configuredParameters(), {130, 130, 130}, 130);
  const auto &resolved = capability(test, fabric);
  require(test,
          resolved.implementationFamily ==
                  ::fabric::ImplementationFamilyId::FixedVectorValueSelect &&
              resolved.enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{
                      ::dataflow::OperationSchemaId::ArithSelect} &&
              std::holds_alternative<::fabric::FixedVectorValueSelectParams>(
                  resolved.parameterizedCapability) &&
              resolved.configurationFieldSchema.size() == 1,
          "resolved select capability escaped its generated Fabric contract");
  require(
      test,
      configurationValue(test, resolved,
                         mlir::IntegerType::get(&fabricContext(), 16), {8}) ==
              configurationValue(test, resolved,
                                 mlir::Float16Type::get(&fabricContext()),
                                 {2, 2}) &&
          configurationValue(test, resolved,
                             mlir::IntegerType::get(&fabricContext(), 16),
                             {8}) !=
              configurationValue(test, resolved,
                                 mlir::IntegerType::get(&fabricContext(), 8),
                                 {16}),
      "configuration did not distinguish only element bit width");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 5 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atInput(2).getName() == "data_input_2" &&
              ports.atInput(3).getName() == "config_0" &&
              mlir::cast<mlir::IntegerType>(ports.atInput(3).type).getWidth() ==
                  3 &&
              ports.atOutput(0).getName() == "data_output_0",
          "derived select leaf ports are not canonical");
  const std::string firstRtl = specialize(test, first, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string secondRtl = specialize(test, second, fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical vector select inputs produced different RTL");
  require(test,
          llvm::StringRef(firstRtl).contains("config_0") &&
              llvm::StringRef(firstRtl).contains("data_input_0[0]") &&
              llvm::StringRef(firstRtl).contains("data_input_0[15]") &&
              !llvm::StringRef(firstRtl).contains("data_input_0[16]"),
          "portable select did not use the low condition lane bits");
  writeToolInputs(root, firstRtl);
}

void sameWidthSingletonNeedsNoConfiguration(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric =
      makeFabric(test, store, "fixed-vector-value-select-singleton",
                 singletonParameters(), {64, 64, 64}, 64);
  const auto &resolved = capability(test, fabric);
  require(test, resolved.configurationFieldSchema.empty(),
          "same-width integer/float select gained a configuration field");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() == ::fabric::FabricOpSemanticFieldRelationKind::None,
          "singleton vector select unexpectedly has a field relation");
  const auto domain = relation.finiteBehaviorDomain();
  require(test,
          domain.size() == 1 && !domain.front().semanticConfiguration &&
              behaviorElementWidth(domain.front()) == 16,
          "same-width integer/float select did not collapse to one behavior");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  require(test, skeleton.leaf.getPortList().size() == 4,
          "singleton vector select retained a selector port");
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test, !llvm::StringRef(rtl).contains("config_"),
          "singleton vector select emitted configuration logic");
}

llvm::Expected<loom::hardware::test::PortableProviderConformance>
specializeForFailure(SkeletonFixture &skeleton, const FabricFixture &fabric,
                     const FinalizedConfigurationABI &abi,
                     const FabricOperationProviderRegistry &registry) {
  ExternalImplementationContractCatalog externalContracts;
  return loom::hardware::test::specializeAndExportPortableProvider(
      {std::move(skeleton.module),
       {{skeleton.leaf, fabric.physicalOccurrence}}},
      abi, registry, externalContracts);
}

void malformedInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric =
      makeFabric(test, store, "malformed-vector-select", configuredParameters(),
                 {128, 128, 128}, 128);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableFixedVectorValueSelectProvider(registry))
    fail(test, llvm::toString(std::move(error)));

  std::unique_ptr<mlir::MLIRContext> portContext = makeCirctContext();
  SkeletonFixture wrongPorts =
      makeSkeleton(test, *portContext, fabric, abi.abi(), true);
  expectError(test, specializeForFailure(wrongPorts, fabric, abi, registry),
              "leaf port");

  expectError(test,
              finalizeConfigurationABI(
                  makeConfigurationAbiDraft(
                      test, store, fabric, ConfigurationAbiKind::MissingWidth8),
                  store),
              "semantic");

  expectError(
      test,
      finalizeConfigurationABI(
          makeConfigurationAbiDraft(test, store, fabric,
                                    ConfigurationAbiKind::ExtraSemanticValue),
          store),
      "semantic");
}

void malformedAndUnsupportedCapabilitiesFailClosed(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture configured =
      makeFabric(test, store, "capability-vector-select",
                 configuredParameters(), {128, 128, 128}, 128);
  auto malformed = capability(test, configured);
  malformed.parameterizedCapability = ::fabric::ScalarIntegerParams{
      ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I8})};
  expectError(test, malformed.resolveSemanticFieldRelation(fabricContext()),
              "parameter schema");

  FabricFixture undersized = makeFabric(
      test, store, "undersized-vector-select",
      ::fabric::FixedVectorValueSelectParams{
          ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I8}),
          ::fabric::FloatFormatSet{}, 128},
      {8, 64, 64}, 64);
  const auto &narrowedCapability = capability(test, undersized);
  auto narrowedRelation = take(
      test, narrowedCapability.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          narrowedRelation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::None,
          "narrowed vector select unexpectedly has a field relation");
  const auto narrowedDomain = narrowedRelation.finiteBehaviorDomain();
  require(test,
          narrowedCapability.configurationFieldSchema.empty() &&
              narrowedDomain.size() == 1 &&
              !narrowedDomain.front().semanticConfiguration &&
              behaviorElementWidth(narrowedDomain.front()) == 8,
          "physical ports did not narrow select to vector<8xi8>");
  FinalizedConfigurationABI undersizedAbi =
      makeConfigurationAbi(test, store, undersized);
  std::unique_ptr<mlir::MLIRContext> undersizedContext = makeCirctContext();
  SkeletonFixture undersizedSkeleton =
      makeSkeleton(test, *undersizedContext, undersized, undersizedAbi.abi());
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableFixedVectorValueSelectProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  const std::string narrowedRtl =
      specialize(test, undersizedSkeleton, undersized, undersizedAbi);
  require(test,
          !llvm::StringRef(narrowedRtl).contains("config_0") &&
              llvm::StringRef(narrowedRtl).contains("data_input_0[7]") &&
              !llvm::StringRef(narrowedRtl).contains("data_input_0[8]"),
          "narrow select did not use exactly eight condition lanes");

  FabricFixture unsupportedContract =
      makeFabric(test, store, "unsupported-contract-vector-select",
                 singletonParameters(), {64, 64, 64}, 64, true);
  FinalizedConfigurationABI contractAbi =
      makeConfigurationAbi(test, store, unsupportedContract);
  std::unique_ptr<mlir::MLIRContext> contractContext = makeCirctContext();
  SkeletonFixture contractSkeleton = makeSkeleton(
      test, *contractContext, unsupportedContract, contractAbi.abi());
  expectTypedUnsupported(test,
                         specializeForFailure(contractSkeleton,
                                              unsupportedContract, contractAbi,
                                              registry),
                         "unsupported select resource contract");

  FabricFixture unsupportedShape =
      makeFabric(test, store, "unsupported-shape-vector-select",
                 singletonParameters(), {64, 64, 64, 64}, 64);
  FinalizedConfigurationABI shapeAbi =
      makeConfigurationAbi(test, store, unsupportedShape);
  std::unique_ptr<mlir::MLIRContext> shapeContext = makeCirctContext();
  SkeletonFixture shapeSkeleton =
      makeSkeleton(test, *shapeContext, unsupportedShape, shapeAbi.abi());
  expectTypedUnsupported(
      test,
      specializeForFailure(shapeSkeleton, unsupportedShape, shapeAbi, registry),
      "unsupported select physical shape");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  registrationIsPortableOnly();
  configuredBehaviorAndDeterminism(root);
  sameWidthSingletonNeedsNoConfiguration(root / "singleton");
  malformedInputsAreTransactional(root / "malformed");
  malformedAndUnsupportedCapabilitiesFailClosed(root / "capabilities");
  return 0;
}

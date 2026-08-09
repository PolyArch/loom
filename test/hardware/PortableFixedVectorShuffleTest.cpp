#include "ADG/Builder.h"
#include "ConfigurationABI3TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FixedVectorShuffle.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/ImplementationFamily.h"
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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <sstream>
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
using namespace loom::hardware;
using namespace loom::hardware::rtl;

constexpr unsigned kPhysicalWidth = 130;

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
    fail(test, "accepted invalid portable fixed-vector shuffle input");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid portable fixed-vector shuffle input");
  expectError(test, value.takeError(), expected);
}

template <typename T>
void expectTypedUnsupported(llvm::StringRef test, llvm::Expected<T> value,
                            BackendRecipeKey recipe,
                            llvm::StringRef description) {
  require(test, !value, std::string("provider accepted ") + description.str());
  bool typedUnsupported = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        typedUnsupported =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::FixedVectorShuffle &&
            error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, description.str() +
                       " returned the wrong error class: " + error.message());
      });
  require(test, typedUnsupported,
          description.str() + " lost its typed Unsupported classification");
}

::fabric::FixedVectorShuffleParams shuffleParameters() {
  return {::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I8}),
          ::fabric::FloatFormatSet{},
          kPhysicalWidth,
          kPhysicalWidth,
          24,
          7,
          5};
}

struct FabricFixture final {
  FinalizedFabricDesign design;
  FabricFuOccurrenceNodeRef occurrence;
  loom::fabric::FinalizedFabricRoot system;
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
          ::fabric::ImplementationFamilyId::FixedVectorShuffle)
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         root.view(), candidate.occurrence, fuOccurrence));
      auto system = take(
          test, loom::hardware::test::makeSingleSpatialCoreSystem(root, store));
      auto systemView =
          take(test, loom::fabric::requireSystemRoot(system.view()));
      auto operations =
          take(test, enumerateFabricPhysicalOperations(systemView));
      const auto physical = llvm::find_if(operations, [&](const auto &entry) {
        return entry.localOccurrence == occurrence;
      });
      require(test, physical != operations.end(),
              "System has no physical fixed-vector shuffle occurrence");
      return FabricFixture{std::move(design), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no fixed-vector shuffle occurrence");
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
                         ::fabric::FixedVectorShuffleParams parameters,
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

  const auto family = ::fabric::ImplementationFamilyId::FixedVectorShuffle;
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
shuffleActor(unsigned leftBlocks, unsigned rightBlocks, unsigned resultBlocks,
             unsigned blockElements, llvm::ArrayRef<std::int64_t> mask) {
  mlir::MLIRContext &context = fabricContext();
  mlir::Type element = mlir::IntegerType::get(&context, 8);
  mlir::VectorType left =
      mlir::VectorType::get({static_cast<std::int64_t>(leftBlocks),
                             static_cast<std::int64_t>(blockElements)},
                            element);
  mlir::VectorType right =
      mlir::VectorType::get({static_cast<std::int64_t>(rightBlocks),
                             static_cast<std::int64_t>(blockElements)},
                            element);
  mlir::VectorType result =
      mlir::VectorType::get({static_cast<std::int64_t>(resultBlocks),
                             static_cast<std::int64_t>(blockElements)},
                            element);
  return {::dataflow::OperationSchemaId::VectorShuffle,
          mlir::FunctionType::get(&context, {left, right}, {result}),
          ::dataflow::VectorShuffleMaskPayload{
              std::vector<std::int64_t>(mask.begin(), mask.end())}};
}

::dataflow::CanonicalActorSchemaProjection
singleBitShuffleActor(llvm::ArrayRef<std::int64_t> mask) {
  mlir::MLIRContext &context = fabricContext();
  mlir::Type element = mlir::IntegerType::get(&context, 1);
  mlir::VectorType left = mlir::VectorType::get({2}, element);
  mlir::VectorType right = mlir::VectorType::get({3}, element);
  mlir::VectorType result = mlir::VectorType::get({3}, element);
  return {::dataflow::OperationSchemaId::VectorShuffle,
          mlir::FunctionType::get(&context, {left, right}, {result}),
          ::dataflow::VectorShuffleMaskPayload{
              std::vector<std::int64_t>(mask.begin(), mask.end())}};
}

std::vector<std::uint8_t> semanticConfiguration(
    llvm::StringRef test,
    const loom::fabric::ResolvedFabricOpCapabilityView &resolved,
    const ::dataflow::CanonicalActorSchemaProjection &actor) {
  require(test, resolved.configurationFieldSchema.size() == 1,
          "shuffle capability does not own exactly one configuration field");
  constexpr std::array<std::uint64_t, 2> operandPorts = {0, 1};
  constexpr std::array<std::uint64_t, 1> resultPorts = {0};
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Direct,
          "shuffle semantic field relation is not direct");
  const loom::CanonicalSemanticBytes encoded = take(
      test, relation.projectSemanticValue(actor, operandPorts, resultPorts,
                                          ::fabric::ResolvedIndexWidth::I64));
  if (llvm::Error error = relation.validateSemanticValue(encoded.bytes()))
    fail(test, llvm::toString(std::move(error)));
  return std::vector<std::uint8_t>(encoded.bytes().begin(),
                                   encoded.bytes().end());
}

enum class ConfigurationAbiKind {
  Complete,
  WrongBitCount,
  FiniteCodebook,
};

ConfigurationABIDraft makeConfigurationAbiDraft(
    llvm::StringRef test, const FabricFixture &fixture,
    const ::dataflow::CanonicalActorSchemaProjection &inactiveActor,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Direct &&
              relation.directEncodedBitCount().has_value(),
          "shuffle capability did not resolve an exact direct relation");
  const std::vector<std::uint8_t> semantic =
      semanticConfiguration(test, resolved, inactiveActor);
  const std::uint64_t encodedBits = kind == ConfigurationAbiKind::WrongBitCount
                                        ? *relation.directEncodedBitCount() - 1
                                        : *relation.directEncodedBitCount();
  std::vector<std::uint8_t> inactive = semantic;
  if (kind == ConfigurationAbiKind::WrongBitCount)
    inactive.assign(static_cast<std::size_t>((encodedBits + 7) / 8), 0);

  SemanticFieldEncoding encoding = DirectBitsEncoding{encodedBits};
  if (kind == ConfigurationAbiKind::FiniteCodebook)
    encoding = FiniteCodebookEncoding{
        encodedBits, {FiniteCodebookEntry{semantic, semantic}}};
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, std::move(encoding), std::move(inactive)};
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, {std::move(field)}));
}

FinalizedConfigurationABI makeConfigurationAbi(
    llvm::StringRef test, const ArtifactStore &store,
    const FabricFixture &fixture,
    const ::dataflow::CanonicalActorSchemaProjection &inactiveActor,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  return take(test,
              finalizeConfigurationABI(
                  makeConfigurationAbiDraft(test, fixture, inactiveActor, kind),
                  store));
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
    require(test, ports.size() == 4 && ports[2].getName() == "config_0",
            "shuffle leaf did not expose its exact configuration field");
    const unsigned width =
        mlir::cast<mlir::IntegerType>(ports[2].type).getWidth();
    ports[2].type = builder.getIntegerType(width - 1);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("fixed_vector_shuffle"), ports);
  return SkeletonFixture{std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

FabricOperationProviderRegistry makeProviderRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableFixedVectorShuffleProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

llvm::Expected<FabricOperationProviderOutput>
specializeWithRecipe(SkeletonFixture &skeleton, const FabricFixture &fabric,
                     const FinalizedConfigurationABI &abi,
                     const FabricOperationProviderRegistry &registry,
                     BackendRecipeKey recipe) {
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, recipe, {}}};
  return specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                         recipes, registry, externalContracts);
}

llvm::Expected<loom::hardware::test::PortableProviderConformance>
specializePortableForFailure(SkeletonFixture &skeleton,
                             const FabricFixture &fabric,
                             const FinalizedConfigurationABI &abi,
                             const FabricOperationProviderRegistry &registry) {
  ExternalImplementationContractCatalog externalContracts;
  return loom::hardware::test::specializeAndExportPortableProvider(
      {std::move(skeleton.module),
       {{skeleton.leaf, fabric.physicalOccurrence}}},
      abi, registry, externalContracts);
}

std::string specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
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
          "portable shuffle emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

std::uint64_t readPackedBits(llvm::ArrayRef<std::uint8_t> bytes,
                             std::uint32_t offset, std::uint32_t count) {
  std::uint64_t result = 0;
  for (std::uint32_t bit = 0; bit != count; ++bit)
    if (((bytes[(offset + bit) / 8] >> ((offset + bit) % 8)) & 1U) != 0)
      result |= std::uint64_t{1} << bit;
  return result;
}

llvm::APInt decodeBits(llvm::ArrayRef<std::uint8_t> bytes, unsigned bitCount) {
  llvm::APInt result(bitCount, 0);
  for (unsigned bit = 0; bit != bitCount; ++bit)
    if (((bytes[bit / 8] >> (bit % 8)) & 1U) != 0)
      result.setBit(bit);
  return result;
}

std::string svLiteral(const llvm::APInt &value) {
  llvm::SmallString<64> digits;
  value.toString(digits, 16, false, false);
  return std::to_string(value.getBitWidth()) + "'h" + digits.str().str();
}

llvm::APInt operandBits(unsigned blockWidth,
                        llvm::ArrayRef<std::uint64_t> blocks) {
  llvm::APInt result(kPhysicalWidth, 0);
  for (auto [ordinal, value] : llvm::enumerate(blocks))
    result.insertBits(llvm::APInt(blockWidth, value),
                      static_cast<unsigned>(ordinal) * blockWidth);
  const unsigned payloadBits = blocks.size() * blockWidth;
  if (payloadBits < kPhysicalWidth)
    result.setBits(payloadBits, kPhysicalWidth);
  return result;
}

struct ExpectedBits final {
  llvm::APInt value;
  llvm::APInt definedMask;
};

ExpectedBits expectedShuffle(const llvm::APInt &left, const llvm::APInt &right,
                             unsigned blockWidth, unsigned leftBlocks,
                             llvm::ArrayRef<std::int64_t> selectors) {
  llvm::APInt value(kPhysicalWidth, 0);
  llvm::APInt definedMask(kPhysicalWidth, 0);
  for (auto [ordinal, selector] : llvm::enumerate(selectors)) {
    if (selector < 0)
      continue;
    const unsigned selected = static_cast<unsigned>(selector);
    const bool fromLeft = selected < leftBlocks;
    const unsigned sourceOrdinal = fromLeft ? selected : selected - leftBlocks;
    const llvm::APInt block =
        (fromLeft ? left : right)
            .extractBits(blockWidth, sourceOrdinal * blockWidth);
    const unsigned destination = static_cast<unsigned>(ordinal) * blockWidth;
    value.insertBits(block, destination);
    definedMask.setBits(destination, destination + blockWidth);
  }
  return {std::move(value), std::move(definedMask)};
}

llvm::APInt highPaddingMask(unsigned resultBits) {
  llvm::APInt result(kPhysicalWidth, 0);
  result.setBits(resultBits, kPhysicalWidth);
  return result;
}

std::vector<std::uint8_t> encodedConfiguration(
    llvm::StringRef test,
    const loom::fabric::ResolvedFabricOpCapabilityView &resolved,
    const loom::fabric::FabricPhysicalOccurrenceOwnerRef &physicalOccurrence,
    const ConfigurationABI &abi,
    const ::dataflow::CanonicalActorSchemaProjection &actor) {
  const std::vector<std::uint8_t> semantic =
      semanticConfiguration(test, resolved, actor);
  const ConfigurationFieldEncoding *field = abi.findOperationField(
      physicalOccurrence, resolved.configurationFieldSchema.front().ordinal);
  require(test, field != nullptr,
          "shuffle operation field is absent from the ABI");
  const std::vector<SemanticConfigurationValue> values = {
      {field->slot, semantic}};
  const auto unit = llvm::find_if(
      abi.programmingUnits(), [&](const ProgrammingUnit &candidate) {
        return llvm::any_of(candidate.fields, [&](const auto &candidateField) {
          return candidateField.slot == field->slot;
        });
      });
  require(test, unit != abi.programmingUnits().end(),
          "shuffle operation field has no programming unit");
  const std::vector<std::uint8_t> payload =
      take(test, abi.encode(unit->id, values));
  std::vector<std::uint8_t> physical(
      static_cast<std::size_t>((field->encodedBitCount() + 7) / 8), 0);
  for (const DestinationSlice &slice : field->destinationSlices)
    for (std::uint64_t bit = 0; bit != slice.bitCount; ++bit)
      if (((payload[(slice.destinationBitOffset + bit) / 8] >>
            ((slice.destinationBitOffset + bit) % 8)) &
           1U) != 0)
        physical[(slice.sourceBitOffset + bit) / 8] |=
            std::uint8_t{1} << ((slice.sourceBitOffset + bit) % 8);
  require(test, physical == semantic,
          "identity DirectBits destination changed the semantic field");
  return physical;
}

struct ToolCase final {
  std::string description;
  llvm::APInt left;
  llvm::APInt right;
  llvm::APInt configuration;
  ExpectedBits expected;
  llvm::APInt paddingMask;
};

void writeToolInputs(const std::filesystem::path &root, llvm::StringRef rtl,
                     llvm::ArrayRef<ToolCase> cases) {
  const llvm::StringRef test = __func__;
  std::ostringstream testbench;
  testbench << R"sv(module testbench;
  logic [129:0] data_input_0;
  logic [129:0] data_input_1;
  logic [25:0] config_0;
  logic [129:0] data_output_0;

  fixed_vector_shuffle dut(.*);

  initial begin
)sv";
  for (const ToolCase &testCase : cases) {
    testbench << "    data_input_0 = " << svLiteral(testCase.left) << ";\n"
              << "    data_input_1 = " << svLiteral(testCase.right) << ";\n"
              << "    config_0 = " << svLiteral(testCase.configuration) << ";\n"
              << "    #1;\n"
              << "    if ((data_output_0 & "
              << svLiteral(testCase.expected.definedMask)
              << ") !== " << svLiteral(testCase.expected.value) << ")\n"
              << "      $fatal(1, \"" << testCase.description
              << " defined blocks failed\");\n"
              << "    if ((data_output_0 & " << svLiteral(testCase.paddingMask)
              << ") !== 130'h0)\n"
              << "      $fatal(1, \"" << testCase.description
              << " high padding failed\");\n\n";
  }
  testbench << R"sv(    $finish;
  end
endmodule
)sv";
  const std::string yosysScript = R"ys(
read_verilog -sv fixed_vector_shuffle.sv
hierarchy -check -top fixed_vector_shuffle
proc
opt
check -assert
synth -top fixed_vector_shuffle
check -assert
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "tool-artifacts",
          {{"fixed_vector_shuffle.sv", rtl.str()},
           {"testbench.sv", testbench.str()},
           {"portable_fixed_vector_shuffle.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void registrationIsPortableOnly() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  require(test, coverage.size() == ::fabric::implementationFamilyCount(),
          "provider coverage lost the generated family cardinality");
  const auto entry = llvm::find_if(coverage, [](const auto &candidate) {
    return candidate.implementationFamily ==
           ::fabric::ImplementationFamilyId::FixedVectorShuffle;
  });
  require(test,
          entry != coverage.end() &&
              entry->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "fixed-vector shuffle registered a non-portable recipe");
}

void configuredBehaviorAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store, "fixed-vector-shuffle",
                                    shuffleParameters(), {130, 130}, 130);
  const auto &resolved = capability(test, fabric);
  require(test,
          resolved.implementationFamily ==
                  ::fabric::ImplementationFamilyId::FixedVectorShuffle &&
              resolved.enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{
                      ::dataflow::OperationSchemaId::VectorShuffle} &&
              std::holds_alternative<::fabric::FixedVectorShuffleParams>(
                  resolved.parameterizedCapability) &&
              resolved.configurationFieldSchema.size() == 1,
          "resolved shuffle capability escaped its generated Fabric contract");
  const auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  const auto *sealedLayout = relation.fixedVectorShuffleLayout();
  require(
      test,
      relation.kind() == ::fabric::FabricOpSemanticFieldRelationKind::Direct &&
          sealedLayout != nullptr && relation.directEncodedBitCount() &&
          *relation.directEncodedBitCount() == sealedLayout->encodedBitCount,
      "shuffle capability did not seal an exact Direct layout");
  const auto &layout = *sealedLayout;
  require(test, layout.encodedBitCount == 26 && layout.selectorCount == 5,
          "shuffle configuration layout did not match the shared owner");

  const std::vector<std::int64_t> firstMask = {4, 2, 4, -1};
  const auto firstActor = shuffleActor(3, 2, 4, 3, firstMask);
  FinalizedConfigurationABI abi =
      makeConfigurationAbi(test, store, fabric, firstActor);
  const ConfigurationFieldEncoding *field = abi.abi().findOperationField(
      fabric.physicalOccurrence,
      resolved.configurationFieldSchema.front().ordinal);
  const auto *direct =
      field ? std::get_if<DirectBitsEncoding>(&field->semanticEncoding)
            : nullptr;
  require(test, direct && direct->encodedBitCount == layout.encodedBitCount,
          "ConfigurationABI did not preserve the exact DirectBits field");

  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 4 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atInput(2).getName() == "config_0" &&
              mlir::cast<mlir::IntegerType>(ports.atInput(2).type).getWidth() ==
                  layout.encodedBitCount &&
              ports.atOutput(0).getName() == "data_output_0",
          "derived shuffle leaf ports are not canonical vector-token ports");
  const std::string firstRtl = specialize(test, first, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string secondRtl = specialize(test, second, fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical shuffle inputs produced different RTL");
  require(test,
          llvm::StringRef(firstRtl).contains("data_input_0") &&
              llvm::StringRef(firstRtl).contains("data_input_1") &&
              llvm::StringRef(firstRtl).contains("config_0") &&
              llvm::StringRef(firstRtl).contains(">>") &&
              llvm::StringRef(firstRtl).contains("<<"),
          "portable shuffle did not emit the generic selection network");

  const std::vector<std::uint8_t> firstConfiguration = encodedConfiguration(
      test, resolved, fabric.physicalOccurrence, abi.abi(), firstActor);
  require(
      test,
      readPackedBits(firstConfiguration, layout.blockWidthBitOffset,
                     layout.blockWidthBitCount) == 23 &&
          readPackedBits(firstConfiguration, layout.leftBlockCountBitOffset,
                         layout.blockCountBitCount) == 2 &&
          readPackedBits(firstConfiguration, layout.resultBlockCountBitOffset,
                         layout.resultBlockCountBitCount) == 3 &&
          readPackedBits(firstConfiguration, layout.selectorBitOffset,
                         layout.selectorBitCount) == 4 &&
          readPackedBits(firstConfiguration,
                         layout.selectorBitOffset + layout.selectorBitCount,
                         layout.selectorBitCount) == 2 &&
          readPackedBits(firstConfiguration,
                         layout.selectorBitOffset + 3 * layout.selectorBitCount,
                         layout.selectorBitCount) == 0,
      "Fabric did not encode the expected shuffle geometry and selectors");

  const llvm::APInt firstLeft = operandBits(24, {0x010203, 0x111213, 0x212223});
  const llvm::APInt firstRight = operandBits(24, {0xa0b0c0, 0xd0e0f0});
  ToolCase poisonAndDuplicate{
      "unequal duplicate poison",
      firstLeft,
      firstRight,
      decodeBits(firstConfiguration, layout.encodedBitCount),
      expectedShuffle(firstLeft, firstRight, 24, 3, firstMask),
      highPaddingMask(96)};

  const std::vector<std::int64_t> secondMask = {3, 1};
  const auto secondActor = shuffleActor(1, 3, 2, 2, secondMask);
  const std::vector<std::uint8_t> secondConfiguration = encodedConfiguration(
      test, resolved, fabric.physicalOccurrence, abi.abi(), secondActor);
  const llvm::APInt secondLeft = operandBits(16, {0x1234});
  const llvm::APInt secondRight = operandBits(16, {0x5678, 0x9abc, 0xdef0});
  ToolCase smallerResult{
      "smaller result capacity",
      secondLeft,
      secondRight,
      decodeBits(secondConfiguration, layout.encodedBitCount),
      expectedShuffle(secondLeft, secondRight, 16, 1, secondMask),
      highPaddingMask(32)};
  const std::vector<ToolCase> cases = {poisonAndDuplicate, smallerResult};
  writeToolInputs(root, firstRtl, cases);
}

void singleBitBlockWidth(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  const ::fabric::FixedVectorShuffleParams parameters{
      ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I1}),
      ::fabric::FloatFormatSet{},
      7,
      7,
      1,
      5,
      3};
  FabricFixture fabric =
      makeFabric(test, store, "single-bit-shuffle", parameters, {7, 7}, 7);
  const auto actor = singleBitShuffleActor({4, 0, -1});
  FinalizedConfigurationABI abi =
      makeConfigurationAbi(test, store, fabric, actor);
  const auto &resolved = capability(test, fabric);
  const auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  const auto *sealedLayout = relation.fixedVectorShuffleLayout();
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Direct &&
              sealedLayout != nullptr,
          "single-bit shuffle did not seal its Direct layout");
  const auto &layout = *sealedLayout;
  require(test, layout.blockWidthBitCount == 0,
          "single-bit shuffle unexpectedly encoded its fixed block width");
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "tool-artifacts", {{"fixed_vector_shuffle.sv", rtl}}))
    fail(test, llvm::toString(std::move(error)));
  const llvm::StringRef emitted(rtl);
  require(test,
          emitted.contains("module fixed_vector_shuffle") &&
              emitted.contains("input  [6:0]") &&
              emitted.contains("output [6:0]") && !emitted.contains("[-1:0]") &&
              !emitted.contains("'x"),
          "single-bit shuffle emitted malformed RTL");
}

void malformedInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  const std::vector<std::int64_t> mask = {0, 2, -1};
  const auto actor = shuffleActor(2, 2, 3, 2, mask);
  FabricFixture fabric = makeFabric(test, store, "malformed-shuffle",
                                    shuffleParameters(), {130, 130}, 130);
  FinalizedConfigurationABI abi =
      makeConfigurationAbi(test, store, fabric, actor);
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  std::unique_ptr<mlir::MLIRContext> portContext = makeCirctContext();
  SkeletonFixture wrongPorts =
      makeSkeleton(test, *portContext, fabric, abi.abi(), true);
  expectError(test,
              specializePortableForFailure(wrongPorts, fabric, abi, registry),
              "port");

  expectError(test,
              finalizeConfigurationABI(
                  makeConfigurationAbiDraft(
                      test, fabric, actor, ConfigurationAbiKind::WrongBitCount),
                  store),
              "DirectBits width");

  expectError(
      test,
      finalizeConfigurationABI(
          makeConfigurationAbiDraft(test, fabric, actor,
                                    ConfigurationAbiKind::FiniteCodebook),
          store),
      "DirectBits");
}

void unsupportedCapabilitiesAreTransactional(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  const std::vector<std::int64_t> mask = {0, 2};
  const auto actor = shuffleActor(1, 3, 2, 2, mask);
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  FabricFixture unsupportedContract =
      makeFabric(test, store, "unsupported-contract-shuffle",
                 shuffleParameters(), {130, 130}, 130, true);
  FinalizedConfigurationABI contractAbi =
      makeConfigurationAbi(test, store, unsupportedContract, actor);
  std::unique_ptr<mlir::MLIRContext> contractContext = makeCirctContext();
  SkeletonFixture contractSkeleton = makeSkeleton(
      test, *contractContext, unsupportedContract, contractAbi.abi());
  expectTypedUnsupported(test,
                         specializePortableForFailure(contractSkeleton,
                                                      unsupportedContract,
                                                      contractAbi, registry),
                         BackendRecipeKey::PortableSystemVerilog,
                         "unsupported shuffle resource contract");

  FabricFixture unsupportedShape =
      makeFabric(test, store, "unsupported-shape-shuffle", shuffleParameters(),
                 {130, 130, 130}, 130);
  FinalizedConfigurationABI shapeAbi =
      makeConfigurationAbi(test, store, unsupportedShape, actor);
  std::unique_ptr<mlir::MLIRContext> shapeContext = makeCirctContext();
  SkeletonFixture shapeSkeleton =
      makeSkeleton(test, *shapeContext, unsupportedShape, shapeAbi.abi());
  expectTypedUnsupported(test,
                         specializePortableForFailure(shapeSkeleton,
                                                      unsupportedShape,
                                                      shapeAbi, registry),
                         BackendRecipeKey::PortableSystemVerilog,
                         "unsupported shuffle physical shape");

  FabricFixture undersized = makeFabric(test, store, "undersized-shuffle",
                                        shuffleParameters(), {64, 64}, 64);
  FinalizedConfigurationABI undersizedAbi =
      makeConfigurationAbi(test, store, undersized, actor);
  std::unique_ptr<mlir::MLIRContext> undersizedContext = makeCirctContext();
  SkeletonFixture undersizedSkeleton =
      makeSkeleton(test, *undersizedContext, undersized, undersizedAbi.abi());
  expectTypedUnsupported(test,
                         specializePortableForFailure(undersizedSkeleton,
                                                      undersized, undersizedAbi,
                                                      registry),
                         BackendRecipeKey::PortableSystemVerilog,
                         "undersized shuffle physical datapath");

  FabricFixture native = makeFabric(test, store, "native-recipe-shuffle",
                                    shuffleParameters(), {130, 130}, 130);
  FinalizedConfigurationABI nativeAbi =
      makeConfigurationAbi(test, store, native, actor);
  std::unique_ptr<mlir::MLIRContext> nativeContext = makeCirctContext();
  SkeletonFixture nativeSkeleton =
      makeSkeleton(test, *nativeContext, native, nativeAbi.abi());
  const std::string nativeBefore = moduleText(*nativeSkeleton.module);
  expectTypedUnsupported(
      test,
      specializeWithRecipe(nativeSkeleton, native, nativeAbi, registry,
                           BackendRecipeKey::SynopsysDesignWare),
      BackendRecipeKey::SynopsysDesignWare, "native shuffle recipe");
  require(test, moduleText(*nativeSkeleton.module) == nativeBefore,
          "unsupported native shuffle partially mutated the caller module");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  registrationIsPortableOnly();
  configuredBehaviorAndDeterminism(root);
  singleBitBlockWidth(root / "single-bit");
  malformedInputsAreTransactional(root / "malformed");
  unsupportedCapabilitiesAreTransactional(root / "unsupported");
  return 0;
}

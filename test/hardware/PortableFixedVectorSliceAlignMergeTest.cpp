#include "ADG/Builder.h"
#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FixedVectorSliceAlignMerge.h"

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
#include <fstream>
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
using namespace loom::hardware;
using namespace loom::hardware::rtl;

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

void expectError(llvm::StringRef test, llvm::Error error,
                 llvm::StringRef expected) {
  if (!error)
    fail(test, "accepted invalid fixed-vector slice provider input");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid fixed-vector slice provider input");
  expectError(test, value.takeError(), expected);
}

void expectTypedUnsupported(llvm::StringRef test,
                            llvm::Expected<FabricOperationProviderOutput> value,
                            BackendRecipeKey recipe,
                            llvm::StringRef description) {
  require(test, !value, std::string("provider accepted ") + description.str());
  bool typedUnsupported = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        typedUnsupported =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::FixedVectorSliceAlignMerge &&
            error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, description.str() +
                       " returned the wrong error class: " + error.message());
      });
  require(test, typedUnsupported,
          description.str() + " lost its typed Unsupported classification");
}

::fabric::FixedVectorSliceAlignMergeParams parameters() {
  return {
      ::fabric::IntegerWidthSet::get(
          {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16}),
      ::fabric::FloatFormatSet::get({::fabric::FloatFormat::F32}),
      130,
      64,
      2,
      ::fabric::ResolvedIndexWidthSet::get({::fabric::ResolvedIndexWidth::I64}),
  };
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
          ::fabric::ImplementationFamilyId::FixedVectorSliceAlignMerge)
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
              "System has no physical fixed-vector slice occurrence");
      return FabricFixture{std::move(design), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no fixed-vector slice occurrence");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.root().view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

FabricFixture makeFabric(
    llvm::StringRef test, const ArtifactStore &store, llvm::StringRef label,
    llvm::ArrayRef<unsigned> inputWidths = std::array<unsigned, 4>{130, 130, 64,
                                                                   64},
    unsigned outputWidth = 130, bool unsupportedContract = false,
    llvm::ArrayRef<Schema> schemas =
        std::array<Schema, 2>{Schema::VectorExtract, Schema::VectorInsert},
    const ::fabric::FixedVectorSliceAlignMergeParams &capabilityParameters =
        parameters()) {
  DesignBuilder design(store);
  std::vector<PortType> operationInputs;
  operationInputs.reserve(inputWidths.size());
  for (unsigned width : inputWidths)
    operationInputs.push_back(take(test, PortType::bits(width)));
  const PortType operationOutput = take(test, PortType::bits(outputWidth));
  const unsigned boundaryWidth = std::max(
      outputWidth, *std::max_element(inputWidths.begin(), inputWidths.end()));
  const PortType boundary = take(test, PortType::bits(boundaryWidth));
  const std::vector<PortType> boundaryInputs(inputWidths.size(), boundary);

  auto spatial =
      take(test, design.createSpatialCore(label, boundaryInputs, {boundary}));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != boundaryInputs.size(); ++ordinal)
    spatialInputs.push_back(take(test, spatial.input(ordinal)));
  auto pe =
      take(test, spatial.addPe(spatialInputs,
                               PeSpec::spatial(boundaryInputs, {boundary})));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != boundaryInputs.size(); ++ordinal)
    peInputs.push_back(take(test, pe.input(ordinal)));
  auto fu = take(test, pe.addFu(peInputs, FuSpec{operationInputs, {boundary}}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal != operationInputs.size(); ++ordinal)
    fuInputs.push_back(take(test, fu.input(ordinal)));

  const auto family =
      ::fabric::ImplementationFamilyId::FixedVectorSliceAlignMerge;
  const ::fabric::ResourceContract contract =
      unsupportedContract
          ? ::fabric::loopCarryOperationResourceContract()
          : ::fabric::oneCycleElasticOperationResourceContract();
  auto operation = take(
      test, fu.addOperation(fuInputs, OperationCapabilitySpec{
                                          family,
                                          capabilityParameters,
                                          std::vector<Schema>(schemas.begin(),
                                                              schemas.end()),
                                          {operationOutput},
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
makeActor(Schema schema, unsigned elementWidth,
          llvm::ArrayRef<std::int64_t> containerShape,
          llvm::ArrayRef<std::int64_t> sliceShape,
          llvm::ArrayRef<std::int64_t> position) {
  mlir::MLIRContext &context = fabricContext();
  mlir::Type element = mlir::IntegerType::get(&context, elementWidth);
  mlir::Type container = mlir::VectorType::get(containerShape, element);
  mlir::Type slice =
      sliceShape.empty() ? element : mlir::VectorType::get(sliceShape, element);
  std::vector<mlir::Type> inputs;
  if (schema == Schema::VectorExtract)
    inputs.push_back(container);
  else {
    inputs.push_back(slice);
    inputs.push_back(container);
  }
  for (std::int64_t component : position)
    if (component == mlir::ShapedType::kDynamic)
      inputs.push_back(mlir::IndexType::get(&context));
  return {
      schema,
      mlir::FunctionType::get(
          &context, inputs,
          {schema == Schema::VectorExtract ? slice : container}),
      ::dataflow::VectorStaticPositionPayload{
          std::vector<std::int64_t>(position.begin(), position.end())},
  };
}

std::vector<std::uint8_t>
configurationValue(llvm::StringRef test,
                   const loom::fabric::ResolvedFabricOpCapabilityView &resolved,
                   const ::dataflow::CanonicalActorSchemaProjection &actor) {
  require(test, resolved.configurationFieldSchema.size() == 1,
          "slice capability does not have one direct field");
  std::vector<std::uint64_t> operandPorts;
  if (actor.schema == Schema::VectorExtract) {
    operandPorts.push_back(0);
    for (std::uint64_t ordinal = 1; ordinal != actor.type.getNumInputs();
         ++ordinal)
      operandPorts.push_back(ordinal + 1);
  } else {
    for (std::uint64_t ordinal = 0; ordinal != actor.type.getNumInputs();
         ++ordinal)
      operandPorts.push_back(ordinal);
  }
  constexpr std::array<std::uint64_t, 1> resultPorts = {0};
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Direct,
          "slice semantic field relation is not direct");
  const loom::CanonicalSemanticBytes encoded = take(
      test, relation.projectSemanticValue(actor, operandPorts, resultPorts,
                                          ::fabric::ResolvedIndexWidth::I64));
  if (llvm::Error error = relation.validateSemanticValue(encoded.bytes()))
    fail(test, llvm::toString(std::move(error)));
  return std::vector<std::uint8_t>(encoded.bytes().begin(),
                                   encoded.bytes().end());
}

std::vector<std::uint8_t> zeroBits(std::uint64_t bitCount) {
  return std::vector<std::uint8_t>((bitCount + 7) / 8, 0);
}

enum class ConfigurationAbiKind { Valid, WrongWidth, FiniteCodebook, Missing };

ConfigurationABIDraft makeConfigurationAbiDraft(
    llvm::StringRef test, const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Valid) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty()) {
    require(test, kind == ConfigurationAbiKind::Valid,
            "fieldless slice requested a malformed ABI variant");
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  }
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Direct &&
              relation.directEncodedBitCount().has_value() &&
              relation.fixedVectorSliceAlignMergeLayout() != nullptr,
          "slice capability did not resolve an exact direct relation");
  const std::uint64_t bitCount = kind == ConfigurationAbiKind::WrongWidth
                                     ? *relation.directEncodedBitCount() - 1
                                     : *relation.directEncodedBitCount();
  const auto inactiveActor =
      makeActor(resolved.enabledOperationSchemas.front(), 8, {1}, {}, {0});
  const std::vector<std::uint8_t> validInactive =
      configurationValue(test, resolved, inactiveActor);
  const std::vector<std::uint8_t> inactive =
      kind == ConfigurationAbiKind::WrongWidth ? zeroBits(bitCount)
                                               : validInactive;
  SemanticFieldEncoding encoding = DirectBitsEncoding{bitCount};
  if (kind == ConfigurationAbiKind::FiniteCodebook)
    encoding = FiniteCodebookEncoding{
        bitCount, {FiniteCodebookEntry{validInactive, validInactive}}};
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, std::move(encoding), inactive};
  ConfigurationABIDraft draft =
      take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                     fixture.system, {std::move(field)}));
  if (kind == ConfigurationAbiKind::Missing) {
    bool removed = false;
    for (ProgrammingUnitDraft &unit : draft.programmingUnits)
      for (auto field = unit.fields.begin(); field != unit.fields.end();
           ++field)
        if (field->field == physicalField) {
          unit.fields.erase(field);
          removed = true;
          break;
        }
    require(test, removed,
            "could not remove the slice field from the ABI draft");
  }
  return draft;
}

FinalizedConfigurationABI
makeConfigurationAbi(llvm::StringRef test, const ArtifactStore &store,
                     const FabricFixture &fixture,
                     ConfigurationAbiKind kind = ConfigurationAbiKind::Valid) {
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

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fabric,
                             const ConfigurationABI &abi,
                             bool malformedInput = false) {
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
  if (malformedInput) {
    const unsigned width =
        mlir::cast<mlir::IntegerType>(ports.front().type).getWidth();
    ports.front().type = builder.getIntegerType(width - 1);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("fixed_vector_slice_align_merge"), ports);
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
  if (llvm::Error error =
          registerPortableFixedVectorSliceAlignMergeProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

llvm::Expected<FabricOperationProviderOutput>
specializeFor(SkeletonFixture &skeleton, const FabricFixture &fabric,
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

std::string specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  FabricOperationProviderOutput output =
      take(test, specializeFor(skeleton, fabric, abi, registry,
                               BackendRecipeKey::PortableSystemVerilog));
  require(test,
          output.payloads.empty() && output.activityPoints.empty() &&
              output.externalImplementationBindings.empty(),
          "portable slice provider emitted external implementation state");
  return take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
}

std::uint64_t readPackedBits(llvm::ArrayRef<std::uint8_t> bytes,
                             std::uint32_t offset, std::uint32_t count) {
  std::uint64_t value = 0;
  for (std::uint32_t bit = 0; bit != count; ++bit)
    value |=
        std::uint64_t((bytes[(offset + bit) / 8] >> ((offset + bit) % 8)) & 1U)
        << bit;
  return value;
}

llvm::APInt decodeBits(llvm::ArrayRef<std::uint8_t> bytes, unsigned bitCount) {
  llvm::APInt value(bitCount, 0);
  for (unsigned bit = 0; bit != bitCount; ++bit)
    if (((bytes[bit / 8] >> (bit % 8)) & 1U) != 0)
      value.setBit(bit);
  return value;
}

std::string hex(const llvm::APInt &value) {
  llvm::SmallString<64> digits;
  value.toString(digits, 16, false, false);
  return digits.str().str();
}

llvm::APInt sourceBits() {
  llvm::APInt value(130, "fedcba98765432100123456789abcdef", 16);
  value.setBit(128);
  value.setBit(129);
  return value;
}

llvm::APInt insertedBits() {
  llvm::APInt value(130, "0123456789abcdeffedcba9876543210", 16);
  value.setBit(128);
  value.setBit(129);
  return value;
}

llvm::APInt destinationBits() {
  llvm::APInt value(130, "55aa55aa55aa55aaaa55aa55aa55aa55", 16);
  value.setBit(128);
  return value;
}

llvm::APInt oracleExtract(const llvm::APInt &source, std::uint64_t offset,
                          unsigned sliceWidth) {
  return source.lshr(static_cast<unsigned>(offset)) &
         llvm::APInt::getLowBitsSet(source.getBitWidth(), sliceWidth);
}

llvm::APInt oracleInsert(const llvm::APInt &inserted,
                         const llvm::APInt &destination, std::uint64_t offset,
                         unsigned sliceWidth) {
  llvm::APInt lowMask =
      llvm::APInt::getLowBitsSet(destination.getBitWidth(), sliceWidth);
  llvm::APInt mask = lowMask.shl(static_cast<unsigned>(offset));
  return (destination & ~mask) |
         ((inserted & lowMask).shl(static_cast<unsigned>(offset)) & mask);
}

struct BehaviorCase final {
  std::string name;
  ::dataflow::CanonicalActorSchemaProjection actor;
  std::uint64_t staticOffset = 0;
  unsigned sliceWidth = 0;
  std::vector<std::uint64_t> strides;
  std::vector<std::uint64_t> indices;
  bool undefinedResult = false;
};

std::vector<BehaviorCase> behaviorCases() {
  constexpr std::int64_t dyn = mlir::ShapedType::kDynamic;
  return {
      {"static scalar extract",
       makeActor(Schema::VectorExtract, 16, {2, 4}, {}, {1, 2}),
       96,
       16,
       {},
       {}},
      {"static trailing-subvector extract",
       makeActor(Schema::VectorExtract, 16, {2, 4}, {4}, {1}),
       64,
       64,
       {},
       {}},
      {"two-position dynamic scalar extract",
       makeActor(Schema::VectorExtract, 16, {2, 4}, {}, {dyn, dyn}),
       0,
       16,
       {64, 16},
       {1, 2}},
      {"mixed-position trailing-subvector extract",
       makeActor(Schema::VectorExtract, 8, {2, 2, 4}, {4}, {dyn, 1}),
       32,
       32,
       {64},
       {1}},
      {"static scalar insert",
       makeActor(Schema::VectorInsert, 16, {2, 4}, {}, {0, 1}),
       16,
       16,
       {},
       {}},
      {"static trailing-subvector insert",
       makeActor(Schema::VectorInsert, 16, {2, 4}, {4}, {1}),
       64,
       64,
       {},
       {}},
      {"mixed-position dynamic scalar insert",
       makeActor(Schema::VectorInsert, 16, {2, 4}, {}, {dyn, 3}),
       48,
       16,
       {64},
       {1}},
      {"dynamic trailing-subvector insert",
       makeActor(Schema::VectorInsert, 16, {2, 4}, {4}, {dyn}),
       0,
       64,
       {64},
       {1}},
      {"out-of-range dynamic extract",
       makeActor(Schema::VectorExtract, 16, {2, 4}, {4}, {dyn}),
       0,
       64,
       {64},
       {7},
       true},
  };
}

std::uint64_t effectiveOffset(const BehaviorCase &behavior) {
  std::uint64_t offset = behavior.staticOffset;
  for (auto [index, stride] : llvm::zip(behavior.indices, behavior.strides))
    offset += index * stride;
  return offset;
}

void writeToolInputs(
    const std::filesystem::path &root, llvm::StringRef rtl,
    const loom::fabric::ResolvedFabricOpCapabilityView &resolved,
    const ::fabric::FixedVectorSliceAlignMergeConfigurationLayout &layout) {
  std::ofstream(root / "fixed_vector_slice_align_merge.sv") << rtl.str();
  std::ofstream testbench(root / "testbench.sv");
  testbench << "module testbench;\n"
               "  logic [129:0] data_input_0;\n"
               "  logic [129:0] data_input_1;\n"
               "  logic [63:0] data_input_2;\n"
               "  logic [63:0] data_input_3;\n"
            << "  logic [" << layout.encodedBitCount - 1
            << ":0] config_0;\n"
               "  logic [129:0] data_output_0;\n\n"
               "  fixed_vector_slice_align_merge dut(.*);\n\n"
               "  initial begin\n";
  const llvm::APInt source = sourceBits();
  const llvm::APInt inserted = insertedBits();
  const llvm::APInt destination = destinationBits();
  for (const BehaviorCase &behavior : behaviorCases()) {
    const std::vector<std::uint8_t> configuration =
        configurationValue("writeToolInputs", resolved, behavior.actor);
    const bool insert = behavior.actor.schema == Schema::VectorInsert;
    testbench << "    data_input_0 = 130'h" << hex(insert ? inserted : source)
              << ";\n"
              << "    data_input_1 = 130'h" << hex(destination) << ";\n"
              << "    data_input_2 = 64'd"
              << (behavior.indices.empty() ? 0 : behavior.indices[0]) << ";\n"
              << "    data_input_3 = 64'd"
              << (behavior.indices.size() < 2 ? 0 : behavior.indices[1])
              << ";\n"
              << "    config_0 = " << layout.encodedBitCount << "'h"
              << hex(decodeBits(configuration, layout.encodedBitCount)) << ";\n"
              << "    #1;\n";
    if (behavior.undefinedResult) {
      testbench << "    if (^data_output_0 === 1'bx)\n"
                   "      $fatal(1, \"out-of-range position introduced "
                   "X-filled success\");\n";
    } else {
      const std::uint64_t offset = effectiveOffset(behavior);
      const llvm::APInt expected =
          insert
              ? oracleInsert(inserted, destination, offset, behavior.sliceWidth)
              : oracleExtract(source, offset, behavior.sliceWidth);
      testbench << "    if (data_output_0 !== 130'h" << hex(expected) << ")\n"
                << "      $fatal(1, \"" << behavior.name << " failed\");\n";
    }
  }
  testbench << "    $finish;\n"
               "  end\n"
               "endmodule\n";

  std::ofstream(root / "portable_fixed_vector_slice_align_merge.ys")
      << R"ys(read_verilog -sv fixed_vector_slice_align_merge.sv
hierarchy -check -top fixed_vector_slice_align_merge
proc
opt
check -assert
select -assert-none t:$dlatch t:$memrd t:$memwr t:$meminit t:$mem_v2
stat
synth -top fixed_vector_slice_align_merge
check -assert
select -assert-none t:$dlatch t:$memrd t:$memwr t:$meminit t:$mem_v2
stat
)ys";
}

void writeZeroBitToolInputs(const std::filesystem::path &root,
                            llvm::StringRef rtl, Schema schema) {
  std::ofstream(root / "fixed_vector_slice_align_merge.sv") << rtl.str();
  std::ofstream testbench(root / "testbench.sv");
  testbench
      << "module testbench;\n"
         "  logic data_input_0;\n"
         "  logic data_input_1;\n"
         "  logic data_output_0;\n\n"
         "  fixed_vector_slice_align_merge dut(.*);\n\n"
         "  initial begin\n"
         "    data_input_0 = 1'b0; data_input_1 = 1'b0; #1;\n"
         "    if (data_output_0 !== 1'b0) $fatal(1, \"zero failed\");\n"
         "    data_input_0 = 1'b0; data_input_1 = 1'b1; #1;\n"
         "    if (data_output_0 !== 1'b0) $fatal(1, \"replace zero failed\");\n"
         "    data_input_0 = 1'b1; data_input_1 = 1'b0; #1;\n"
         "    if (data_output_0 !== 1'b1) $fatal(1, \"one failed\");\n"
         "    data_input_0 = 1'b1; data_input_1 = 1'b1; #1;\n"
         "    if (data_output_0 !== 1'b1) $fatal(1, \"replace one failed\");\n"
         "    $finish;\n"
         "  end\n"
         "endmodule\n";
  std::ofstream(root / "portable_fixed_vector_slice_align_merge.ys")
      << R"ys(read_verilog -sv fixed_vector_slice_align_merge.sv
hierarchy -check -top fixed_vector_slice_align_merge
proc
opt
check -assert
select -assert-none t:$dlatch t:$memrd t:$memwr t:$meminit t:$mem_v2
synth -top fixed_vector_slice_align_merge
check -assert
)ys";
  require("writeZeroBitToolInputs",
          schema == Schema::VectorExtract || schema == Schema::VectorInsert,
          "zero-bit tool fixture has an unknown schema");
}

void registrationIsPortableOnly() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);
  const auto coverage = registry.coverage();
  require(test, coverage.size() == ::fabric::implementationFamilyCount(),
          "provider coverage lost the generated family cardinality");
  const auto entry = llvm::find_if(coverage, [](const auto &candidate) {
    return candidate.implementationFamily ==
           ::fabric::ImplementationFamilyId::FixedVectorSliceAlignMerge;
  });
  require(test,
          entry != coverage.end() &&
              entry->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "fixed-vector slice provider registered a native recipe");
}

void configuredBehaviorAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  const std::filesystem::path artifactRoot = root / "artifacts";
  std::filesystem::create_directories(artifactRoot);
  ArtifactStore store(artifactRoot.string());
  FabricFixture fabric = makeFabric(test, store, "fixed-vector-slice");
  const auto &resolved = capability(test, fabric);
  require(
      test,
      resolved.implementationFamily ==
              ::fabric::ImplementationFamilyId::FixedVectorSliceAlignMerge &&
          resolved.enabledOperationSchemas ==
              std::vector<Schema>{Schema::VectorExtract,
                                  Schema::VectorInsert} &&
          std::holds_alternative<::fabric::FixedVectorSliceAlignMergeParams>(
              resolved.parameterizedCapability) &&
          resolved.configurationFieldSchema.size() == 1 &&
          resolved.physicalPorts.size() == 5,
      "resolved slice capability escaped its generated Fabric contract");

  const auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  const auto *sealedLayout = relation.fixedVectorSliceAlignMergeLayout();
  require(
      test,
      relation.kind() == ::fabric::FabricOpSemanticFieldRelationKind::Direct &&
          sealedLayout != nullptr && relation.directEncodedBitCount() &&
          *relation.directEncodedBitCount() == sealedLayout->encodedBitCount,
      "slice capability did not seal an exact Direct layout");
  const auto &layout = *sealedLayout;
  require(test,
          layout.encodesMode && layout.offsetBitCount == 8 &&
              layout.sliceWidthBitCount == 6 &&
              layout.dynamicStrideBitCount == 8 &&
              layout.dynamicStrideCount == 2 && layout.encodedBitCount == 31,
          "slice configuration layout changed unexpectedly");
  for (const BehaviorCase &behavior : behaviorCases()) {
    const std::vector<std::uint8_t> encoded =
        configurationValue(test, resolved, behavior.actor);
    require(test,
            readPackedBits(encoded, layout.modeBitOffset, 1) ==
                    (behavior.actor.schema == Schema::VectorInsert ? 1U : 0U) &&
                readPackedBits(encoded, layout.staticOffsetBitOffset,
                               layout.offsetBitCount) ==
                    behavior.staticOffset &&
                readPackedBits(encoded, layout.sliceWidthBitOffset,
                               layout.sliceWidthBitCount) ==
                    behavior.sliceWidth - 1,
            "actor-owned slice configuration projection changed");
    for (std::uint32_t ordinal = 0; ordinal != layout.dynamicStrideCount;
         ++ordinal) {
      const std::uint64_t expected =
          ordinal < behavior.strides.size() ? behavior.strides[ordinal] : 0;
      require(test,
              readPackedBits(encoded,
                             layout.dynamicStrideBitOffset +
                                 ordinal * layout.dynamicStrideBitCount,
                             layout.dynamicStrideBitCount) == expected,
              "actor-owned dynamic stride projection changed");
    }
  }
  require(test, relation.finiteBehaviorDomain().empty(),
          "Direct slice relation exposed a finite behavior domain");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 6 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atInput(2).getName() == "data_input_2" &&
              ports.atInput(3).getName() == "data_input_3" &&
              ports.atInput(4).getName() == "config_0" &&
              mlir::cast<mlir::IntegerType>(ports.atInput(4).type).getWidth() ==
                  layout.encodedBitCount &&
              ports.atOutput(0).getName() == "data_output_0",
          "derived slice leaf ports are not canonical");
  const std::string firstRtl = specialize(test, first, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string secondRtl = specialize(test, second, fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical slice inputs produced different RTL bytes");
  require(test,
          llvm::StringRef(firstRtl).contains("data_input_2") &&
              llvm::StringRef(firstRtl).contains("data_input_3") &&
              llvm::StringRef(firstRtl).contains("config_0") &&
              !llvm::StringRef(firstRtl).contains("poison"),
          "portable slice RTL lost runtime positions or gained poison state");
  writeToolInputs(root, firstRtl, resolved, layout);
}

void generatedOwnersRejectMalformedCapabilities() {
  const llvm::StringRef test = __func__;
  constexpr std::array<std::uint32_t, 4> inputWidths = {130, 130, 64, 64};
  constexpr std::array<std::uint32_t, 1> resultWidths = {130};
  auto wrongParameters = parameters();
  wrongParameters.maxSlicePayloadBits = 131;
  expectError(test,
              ::fabric::resolveFixedVectorSliceAlignMergeConfigurationLayout(
                  wrongParameters, {Schema::VectorExtract}),
              "payload capacities");
  expectError(test,
              ::fabric::resolveFixedVectorSliceAlignMergeConfigurationLayout(
                  parameters(), {Schema::VectorShuffle}),
              "foreign schema");
  expectError(
      test,
      ::fabric::resolveFabricOpSemanticFieldRelation(
          ::fabric::ImplementationFamilyId::FixedVectorSliceAlignMerge,
          ::fabric::ScalarIntegerParams{
              ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I16})},
          {Schema::VectorExtract}, inputWidths, resultWidths, fabricContext()),
      "parameter schema");
  const auto insertRelation = take(
      test, ::fabric::resolveFabricOpSemanticFieldRelation(
                ::fabric::ImplementationFamilyId::FixedVectorSliceAlignMerge,
                parameters(), {Schema::VectorInsert}, inputWidths, resultWidths,
                fabricContext()));
  expectError(test,
              insertRelation.projectSemanticValue(
                  makeActor(Schema::VectorExtract, 16, {2, 4}, {4}, {1}),
                  std::array<std::uint64_t, 1>{0},
                  std::array<std::uint64_t, 1>{0},
                  ::fabric::ResolvedIndexWidth::I64),
              "not enabled");
}

void singleSchemaResourcesDoNotManufacturePaths(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  const std::filesystem::path artifactRoot = root / "artifacts";
  std::filesystem::create_directories(artifactRoot);
  ArtifactStore store(artifactRoot.string());

  const std::array extractSchema = {Schema::VectorExtract};
  FabricFixture extract =
      makeFabric(test, store, "extract-only", {130, 130, 64, 64}, 130, false,
                 extractSchema);
  FinalizedConfigurationABI extractAbi =
      makeConfigurationAbi(test, store, extract);
  std::unique_ptr<mlir::MLIRContext> extractContext = makeCirctContext();
  SkeletonFixture extractSkeleton =
      makeSkeleton(test, *extractContext, extract, extractAbi.abi());
  const std::string extractRtl =
      specialize(test, extractSkeleton, extract, extractAbi);
  require(test,
          llvm::StringRef(extractRtl).contains("data_input_0 >>") &&
              !llvm::StringRef(extractRtl).contains("data_input_1 &") &&
              !llvm::StringRef(extractRtl).contains("config_0[0] ?"),
          "extract-only capability manufactured an insert path or mode");

  const std::array insertSchema = {Schema::VectorInsert};
  FabricFixture insert = makeFabric(
      test, store, "insert-only", {130, 130, 64, 64}, 130, false, insertSchema);
  FinalizedConfigurationABI insertAbi =
      makeConfigurationAbi(test, store, insert);
  std::unique_ptr<mlir::MLIRContext> insertContext = makeCirctContext();
  SkeletonFixture insertSkeleton =
      makeSkeleton(test, *insertContext, insert, insertAbi.abi());
  const std::string insertRtl =
      specialize(test, insertSkeleton, insert, insertAbi);
  require(test,
          llvm::StringRef(insertRtl).contains("data_input_1 &") &&
              !llvm::StringRef(insertRtl).contains("data_input_0 >>") &&
              !llvm::StringRef(insertRtl).contains("config_0[0] ?"),
          "insert-only capability manufactured an extract path or mode");
}

void zeroBitSliceLowersWithoutConfiguration(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  const ::fabric::FixedVectorSliceAlignMergeParams zeroBitParameters{
      ::fabric::IntegerWidthSet::get({::fabric::IntegerWidth::I1}),
      ::fabric::FloatFormatSet{},
      1,
      1,
      0,
      ::fabric::ResolvedIndexWidthSet{}};
  for (auto [schema, label] :
       std::array{std::pair{Schema::VectorExtract, "extract"},
                  std::pair{Schema::VectorInsert, "insert"}}) {
    const std::filesystem::path caseRoot = root / label;
    std::filesystem::create_directories(caseRoot);
    const std::filesystem::path artifactRoot = caseRoot / "artifacts";
    std::filesystem::create_directories(artifactRoot);
    ArtifactStore store(artifactRoot.string());
    const std::array schemas = {schema};
    FabricFixture fabric = makeFabric(test, store, label, {1, 1}, 1, false,
                                      schemas, zeroBitParameters);
    const auto &resolved = capability(test, fabric);
    require(test, resolved.configurationFieldSchema.empty(),
            "zero-bit slice manufactured a configuration field");
    const auto relation =
        take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
    const auto &finite = relation.finiteBehaviorDomain();
    require(
        test,
        relation.kind() == ::fabric::FabricOpSemanticFieldRelationKind::None &&
            relation.fixedVectorSliceAlignMergeLayout() != nullptr &&
            relation.fixedVectorSliceAlignMergeLayout()->encodedBitCount == 0 &&
            finite.size() == 1 && !finite.front().semanticConfiguration &&
            finite.front().representativeActor.schema == schema,
        "zero-bit slice lost its unique fieldless behavior");

    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
    const circt::hw::ModulePortInfo ports(skeleton.leaf.getPortList());
    require(test,
            ports.size() == 3 && ports.atInput(0).getName() == "data_input_0" &&
                ports.atInput(1).getName() == "data_input_1" &&
                ports.atOutput(0).getName() == "data_output_0",
            "zero-bit slice derived a noncanonical fieldless leaf");
    const std::string rtl = specialize(test, skeleton, fabric, abi);
    require(test, !llvm::StringRef(rtl).contains("config_"),
            "zero-bit slice RTL depends on absent configuration");
    writeZeroBitToolInputs(caseRoot, rtl, schema);
  }
}

void malformedAndUnsupportedInputsAreTransactional(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  const std::filesystem::path artifactRoot = root / "artifacts";
  std::filesystem::create_directories(artifactRoot);
  ArtifactStore store(artifactRoot.string());
  FabricFixture valid = makeFabric(test, store, "valid-slice");
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  FabricOperationProviderRegistry registry = makeProviderRegistry(test);

  std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
  SkeletonFixture malformed =
      makeSkeleton(test, *malformedContext, valid, validAbi.abi(), true);
  const std::string malformedBefore = moduleText(*malformed.module);
  expectError(test,
              specializeFor(malformed, valid, validAbi, registry,
                            BackendRecipeKey::PortableSystemVerilog),
              "leaf port");
  require(test, moduleText(*malformed.module) == malformedBefore,
          "malformed slice leaf partially mutated the caller module");

  for (ConfigurationAbiKind kind : {ConfigurationAbiKind::WrongWidth,
                                    ConfigurationAbiKind::FiniteCodebook}) {
    expectError(test,
                finalizeConfigurationABI(
                    makeConfigurationAbiDraft(test, valid, kind), store),
                kind == ConfigurationAbiKind::WrongWidth ? "DirectBits width"
                                                         : "DirectBits");
  }

  expectError(
      test,
      finalizeConfigurationABI(
          makeConfigurationAbiDraft(test, valid, ConfigurationAbiKind::Missing),
          store),
      "cover every Fabric configuration field");

  FabricFixture unsupportedContract = makeFabric(
      test, store, "unsupported-contract", {130, 130, 64, 64}, 130, true);
  FinalizedConfigurationABI contractAbi =
      makeConfigurationAbi(test, store, unsupportedContract);
  std::unique_ptr<mlir::MLIRContext> contractContext = makeCirctContext();
  SkeletonFixture contractSkeleton = makeSkeleton(
      test, *contractContext, unsupportedContract, contractAbi.abi());
  const std::string contractBefore = moduleText(*contractSkeleton.module);
  expectTypedUnsupported(test,
                         specializeFor(contractSkeleton, unsupportedContract,
                                       contractAbi, registry,
                                       BackendRecipeKey::PortableSystemVerilog),
                         BackendRecipeKey::PortableSystemVerilog,
                         "unsupported slice resource contract");
  require(test, moduleText(*contractSkeleton.module) == contractBefore,
          "unsupported slice contract partially mutated the caller module");

  FabricFixture unsupportedShape =
      makeFabric(test, store, "unsupported-shape", {130, 130, 64, 64, 64});
  FinalizedConfigurationABI shapeAbi =
      makeConfigurationAbi(test, store, unsupportedShape);
  std::unique_ptr<mlir::MLIRContext> shapeContext = makeCirctContext();
  SkeletonFixture shapeSkeleton =
      makeSkeleton(test, *shapeContext, unsupportedShape, shapeAbi.abi());
  const std::string shapeBefore = moduleText(*shapeSkeleton.module);
  expectTypedUnsupported(test,
                         specializeFor(shapeSkeleton, unsupportedShape,
                                       shapeAbi, registry,
                                       BackendRecipeKey::PortableSystemVerilog),
                         BackendRecipeKey::PortableSystemVerilog,
                         "unsupported slice physical shape");
  require(test, moduleText(*shapeSkeleton.module) == shapeBefore,
          "unsupported slice shape partially mutated the caller module");

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
        test, specializeFor(skeleton, valid, validAbi, registry, recipe),
        recipe, "native slice recipe");
    require(test, moduleText(*skeleton.module) == before,
            "unsupported native recipe partially mutated the caller module");
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  registrationIsPortableOnly();
  configuredBehaviorAndDeterminism(root);
  generatedOwnersRejectMalformedCapabilities();
  singleSchemaResourcesDoNotManufacturePaths(root / "single-schema");
  zeroBitSliceLowersWithoutConfiguration(root / "zero-bit");
  malformedAndUnsupportedInputsAreTransactional(root / "failures");
  return 0;
}

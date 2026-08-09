#include "ADG/Builder.h"
#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/TokenMuxDemux.h"
#include "Hardware/RTL/Specialization.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricPeConfiguration.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FinalizedFabricRoot;
using loom::hardware::ConfigurationABI;
using loom::hardware::ConfigurationFieldEncoding;
using loom::hardware::DestinationSlice;
using loom::hardware::ExternalImplementationContractCatalog;
using loom::hardware::FinalizedConfigurationABI;
using loom::hardware::FiniteCodebookEncoding;
using loom::hardware::FiniteCodebookEntry;
using loom::hardware::ProgrammingUnit;
using loom::hardware::SemanticConfigurationValue;
using loom::hardware::rtl::BackendRecipeKey;
using loom::hardware::rtl::FabricOperationLeafAssociation;
using loom::hardware::rtl::FabricOperationProviderOutput;
using loom::hardware::rtl::FabricOperationProviderRegistry;
using loom::hardware::rtl::FabricOperationProviderUnsupportedError;
using loom::hardware::rtl::FabricOperationRecipeBinding;
using loom::hardware::rtl::ModuleRootCirctSkeleton;
using loom::hardware::rtl::ResolvedFabricPhysicalOperation;

enum class FamilyKind { Mux, Demux };

::fabric::ImplementationFamilyId familyId(FamilyKind family) {
  return family == FamilyKind::Mux
             ? ::fabric::ImplementationFamilyId::TokenMux
             : ::fabric::ImplementationFamilyId::TokenDemux;
}

llvm::StringRef familyName(FamilyKind family) {
  return family == FamilyKind::Mux ? "token_mux" : "token_demux";
}

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
    fail(test, "accepted invalid token routing input");
  const std::string message = llvm::toString(std::move(error));
  require(test, llvm::StringRef(message).contains(expected), message);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid token routing input");
  expectError(test, value.takeError(), expected);
}

void expectTypedUnsupported(llvm::StringRef test,
                            llvm::Expected<FabricOperationProviderOutput> value,
                            FamilyKind family, BackendRecipeKey recipe) {
  require(test, !value, "unsupported token routing recipe was accepted");
  bool matched = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        matched = error.implementationFamily() == familyId(family) &&
                  error.recipe() == recipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unsupported recipe returned the wrong error class: " +
                       error.message());
      });
  require(test, matched, "typed Unsupported lost its family or recipe");
}

mlir::MLIRContext &relationContext() {
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

std::unique_ptr<mlir::MLIRContext> makeCirctContext() {
  mlir::DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect,
                  circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
  context->loadAllAvailableDialects();
  return context;
}

FinalizedFabricRoot makeModule(llvm::StringRef test, ArtifactStore &store,
                               FamilyKind family) {
  using loom::adg::DesignBuilder;
  using loom::adg::FuCapabilityTemplateSpec;
  using loom::adg::FuSpec;
  using loom::adg::OperationCapabilitySpec;
  using loom::adg::PeSpec;
  using loom::adg::PortType;

  const PortType outer = take(test, PortType::bits(32));
  const std::vector<PortType> inputTypes =
      family == FamilyKind::Mux ? std::vector<PortType>(5, outer)
                                : std::vector<PortType>(2, outer);
  const std::vector<PortType> outputTypes =
      family == FamilyKind::Mux ? std::vector<PortType>(1, outer)
                                : std::vector<PortType>(4, outer);
  const std::vector<PortType> outerInputTypes(inputTypes.size(), outer);
  const std::vector<PortType> outerOutputTypes(outputTypes.size(), outer);

  DesignBuilder builder(store);
  auto spatial = take(test, builder.createSpatialCore(
                                (familyName(family) + "_portable_test").str(),
                                outerInputTypes, outerOutputTypes));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != outerInputTypes.size(); ++ordinal)
    spatialInputs.push_back(take(test, spatial.input(ordinal)));
  auto pe = take(
      test, spatial.addPe(spatialInputs,
                          PeSpec::spatial(outerInputTypes, outerOutputTypes)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != outerInputTypes.size(); ++ordinal)
    peInputs.push_back(take(test, pe.input(ordinal)));
  auto fu = take(test, pe.addFu(peInputs, FuSpec{inputTypes, outputTypes}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal != inputTypes.size(); ++ordinal)
    fuInputs.push_back(take(test, fu.input(ordinal)));
  auto operation = take(
      test,
      fu.addOperation(
          fuInputs, OperationCapabilitySpec{
                        familyId(family),
                        ::fabric::RoutedTokenParams{8, 4},
                        {family == FamilyKind::Mux
                             ? ::dataflow::OperationSchemaId::DataflowMux
                             : ::dataflow::OperationSchemaId::DataflowDemux},
                        outputTypes,
                        ::fabric::oneCycleElasticOperationResourceContract()}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  std::vector<loom::adg::FuValue> operationOutputs;
  for (std::size_t ordinal = 0; ordinal != outerOutputTypes.size(); ++ordinal)
    operationOutputs.push_back(take(test, operation.output(ordinal)));
  if (llvm::Error error = fu.close(operationOutputs))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  std::vector<loom::adg::SpatialValue> spatialOutputs;
  for (std::size_t ordinal = 0; ordinal != outputTypes.size(); ++ordinal)
    spatialOutputs.push_back(take(test, pe.output(ordinal)));
  if (llvm::Error error = spatial.close(spatialOutputs))
    fail(test, llvm::toString(std::move(error)));
  auto design = take(test, std::move(builder).finalize());
  require(test, design.roots().size() == 1,
          "token routing fixture did not publish one Module root");
  return design.roots().front();
}

struct Codebook final {
  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> targetSemantic;
  std::vector<std::uint8_t> inactiveSemantic;
};

bool hasPorts(const ::fabric::FiniteImplementationFamilyBehaviorPoint &point,
              FamilyKind family, llvm::ArrayRef<std::uint64_t> ports) {
  return family == FamilyKind::Mux
             ? llvm::ArrayRef<std::uint64_t>(point.operandPorts).equals(ports)
             : llvm::ArrayRef<std::uint64_t>(point.resultPorts).equals(ports);
}

Codebook
makeCodebook(llvm::StringRef test, FamilyKind family,
             const loom::fabric::ResolvedFabricOpCapabilityView &capability) {
  auto relation =
      take(test, capability.resolveSemanticFieldRelation(relationContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 11,
          "routed-token relation is not the sealed eleven-mode domain");

  const std::vector<std::uint64_t> target =
      family == FamilyKind::Mux ? std::vector<std::uint64_t>{0, 1, 3, 4}
                                : std::vector<std::uint64_t>{0, 2, 3};
  const std::vector<std::uint64_t> inactive =
      family == FamilyKind::Mux ? std::vector<std::uint64_t>{0, 1, 2}
                                : std::vector<std::uint64_t>{0, 1};
  std::set<std::vector<std::uint8_t>> semanticValues;
  std::set<std::uint8_t> physicalCodes;
  std::uint8_t nextCode = 0;
  Codebook result;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "configured routed-token mode has no semantic value");
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    std::uint8_t code = 0;
    if (hasPorts(point, family, target)) {
      code = 10;
      result.targetSemantic = semantic;
    } else if (hasPorts(point, family, inactive)) {
      code = 1;
      result.inactiveSemantic = semantic;
    } else {
      while (nextCode == 1 || nextCode == 10 || nextCode == 15)
        ++nextCode;
      code = nextCode++;
    }
    require(test, semanticValues.insert(semantic).second,
            "routed-token relation duplicated a semantic value");
    require(test, physicalCodes.insert(code).second,
            "test ABI duplicated a physical code");
    result.entries.push_back({std::move(semantic), {code}});
  }
  require(test,
          !result.targetSemantic.empty() && !result.inactiveSemantic.empty(),
          "sealed relation omitted a required lane image");
  return result;
}

struct Fixture final {
  FamilyKind family;
  FinalizedFabricRoot module;
  FinalizedFabricRoot system;
  FinalizedConfigurationABI abi;
  loom::fabric::SpatialCoreOccurrenceRef spatialCore;
  std::vector<ResolvedFabricPhysicalOperation> operations;
  std::vector<std::uint8_t> targetSemantic;

  const ResolvedFabricPhysicalOperation &operation() const {
    return operations.front();
  }
};

Fixture makeFixture(llvm::StringRef test, ArtifactStore &store,
                    FamilyKind family) {
  FinalizedFabricRoot module = makeModule(test, store, family);
  FinalizedFabricRoot system = take(
      test, loom::hardware::test::makeSingleSpatialCoreSystem(module, store));
  auto systemView = take(test, loom::fabric::requireSystemRoot(system.view()));
  require(test, systemView.artifact().accCoreOccurrences().size() == 1,
          "token routing System did not publish one SpatialCore");
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore{
      systemView.artifact().accCoreOccurrences().front()};
  std::vector<ResolvedFabricPhysicalOperation> operations = take(
      test, loom::hardware::rtl::enumerateFabricPhysicalOperations(systemView));
  llvm::erase_if(operations, [&](const auto &operation) {
    return operation.capability->implementationFamily != familyId(family);
  });
  require(test, operations.size() == 1,
          "token routing System did not resolve one physical operation");
  const auto &operation = operations.front();
  require(test, operation.capability->configurationFieldSchema.size() == 1,
          "token routing capability does not own one semantic field");
  Codebook codebook = makeCodebook(test, family, *operation.capability);
  auto physicalField =
      take(test,
           loom::hardware::test::qualifyPhysicalConfigurationField(
               operation.physicalOccurrence,
               operation.capability->configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride override{
      physicalField, FiniteCodebookEncoding{4, std::move(codebook.entries)},
      codebook.inactiveSemantic};
  FinalizedConfigurationABI abi = take(
      test,
      loom::hardware::finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         system, {std::move(override)})),
          store));
  return Fixture{family,
                 std::move(module),
                 std::move(system),
                 std::move(abi),
                 spatialCore,
                 std::move(operations),
                 std::move(codebook.targetSemantic)};
}

struct LeafSkeleton final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

LeafSkeleton makeLeafSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                              const Fixture &fixture, bool malformed = false) {
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location,
      loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol,
      loom::hardware::rtl::fabricOperationGeneratorDescriptor,
      builder.getArrayAttr({}));
  std::vector<circt::hw::PortInfo> ports =
      take(test, loom::hardware::rtl::deriveFabricOperationLeafPorts(
                     builder, fixture.operation().physicalOccurrence,
                     *fixture.operation().capability, fixture.abi.abi()));
  if (malformed) {
    auto config = llvm::find_if(
        ports, [](const auto &port) { return port.getName() == "config_0"; });
    require(test, config != ports.end(),
            "token routing leaf omitted its configuration field");
    config->type = builder.getI8Type();
  }
  auto leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(
          &context, loom::hardware::rtl::fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(familyName(fixture.family)), ports);
  return {std::move(module), leaf};
}

FabricOperationProviderRegistry makeRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          loom::hardware::rtl::registerPortableTokenMuxDemuxProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

std::string specialize(llvm::StringRef test, ModuleRootCirctSkeleton skeleton,
                       const Fixture &fixture) {
  FabricOperationProviderRegistry registry = makeRegistry(test);
  ExternalImplementationContractCatalog externalContracts;
  auto conformance = take(
      test, loom::hardware::test::specializeAndExportPortableProvider(
                std::move(skeleton), fixture.abi, registry, externalContracts));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "portable token routing provider emitted implementation metadata");
  return std::move(conformance.systemVerilog);
}

std::string specializeLeaf(llvm::StringRef test, LeafSkeleton skeleton,
                           const Fixture &fixture) {
  return specialize(
      test,
      ModuleRootCirctSkeleton{
          std::move(skeleton.module),
          {{skeleton.leaf, fixture.operation().physicalOccurrence}}},
      fixture);
}

std::string specializeSystem(llvm::StringRef test, mlir::MLIRContext &context,
                             const Fixture &fixture) {
  auto skeleton =
      take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                     context, fixture.spatialCore, fixture.abi.abi()));
  require(test, skeleton.operationLeaves.size() == 1,
          "CommonSkeleton did not expose one token routing leaf");
  return specialize(test, std::move(skeleton), fixture);
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream(result) << module;
  return result;
}

const loom::fabric::FabricTransportEndpointRef &
boundaryEndpoint(llvm::StringRef test,
                 const loom::fabric::FabricArtifactView &module,
                 loom::fabric::FabricPortDirection direction,
                 loom::fabric::FabricOrdinal ordinal) {
  const loom::fabric::FabricTransportEndpointRef *result = nullptr;
  for (const auto &attachment : module.moduleBoundaryTransportAttachments()) {
    if (attachment.boundary.direction != direction ||
        attachment.boundary.ordinal != ordinal)
      continue;
    require(test, result == nullptr,
            "Module boundary endpoint has duplicate attachments");
    result = &attachment.endpoint;
  }
  require(test, result != nullptr, "Module boundary endpoint is unattached");
  return *result;
}

loom::fabric::FabricPhysicalConfigurationFieldRef qualifyConfigurationField(
    llvm::StringRef test, loom::fabric::SpatialCoreOccurrenceRef spatialCore,
    const loom::fabric::FabricSemanticConfigFieldRef &field) {
  auto target =
      take(test, loom::fabric::FabricModulePhysicalTargetRef::create(field));
  return take(test, loom::fabric::FabricPhysicalConfigurationFieldRef::create(
                        loom::fabric::SpatialCoreInternalOccurrenceRef{
                            spatialCore, std::move(target)}));
}

const ProgrammingUnit *
fieldOwner(const ConfigurationABI &abi,
           const loom::fabric::FabricPhysicalConfigurationFieldRef &field) {
  for (const ProgrammingUnit &unit : abi.programmingUnits())
    if (llvm::any_of(unit.fields, [&](const auto &candidate) {
          return candidate.field == field;
        }))
      return &unit;
  return nullptr;
}

void writeBit(std::vector<std::uint8_t> &bytes, std::uint64_t bit, bool value) {
  const std::uint8_t mask = std::uint8_t{1} << (bit % 8);
  if (value)
    bytes[bit / 8] |= mask;
  else
    bytes[bit / 8] &= static_cast<std::uint8_t>(~mask);
}

struct ConfigurationImages final {
  std::string portName;
  std::uint64_t bitCount = 0;
  std::vector<std::uint8_t> target;
  std::vector<std::uint8_t> invalid;
};

ConfigurationImages makeConfigurationImages(llvm::StringRef test,
                                            const Fixture &fixture) {
  require(test,
          fixture.module.view().peOccurrences().size() == 1 &&
              fixture.module.view().fuOccurrences().size() == 1,
          "token routing fixture changed its PE/FU shape");
  const auto pe = fixture.module.view().peOccurrences().front();
  const auto fu = fixture.module.view().fuOccurrences().front();
  auto schema =
      take(test, fixture.module.view().spatialPeConfigurationSchema(pe));
  const ProgrammingUnit *owner = nullptr;
  std::vector<SemanticConfigurationValue> values;
  for (const auto &descriptor : schema.fields()) {
    loom::fabric::FabricPeConfigurationValue value;
    if (descriptor.kind ==
        loom::fabric::FabricPeConfigurationFieldKind::Activation) {
      value = loom::fabric::FabricPeActive{fu};
    } else {
      require(test, descriptor.port.has_value(),
              "PE selector field has no FU port");
      value = loom::fabric::FabricPeRoute{boundaryEndpoint(
          test, fixture.module.view(), descriptor.port->direction,
          descriptor.port->ordinal)};
    }
    const auto physical = qualifyConfigurationField(test, fixture.spatialCore,
                                                    descriptor.reference);
    const ProgrammingUnit *candidate = fieldOwner(fixture.abi.abi(), physical);
    require(test, candidate != nullptr,
            "PE configuration field has no programming owner");
    if (owner)
      require(test, owner->id == candidate->id,
              "token routing configuration spans programming units");
    else
      owner = candidate;
    auto encoded = take(test, schema.encode(descriptor.reference, value));
    values.push_back(
        {physical, std::vector<std::uint8_t>(encoded.bytes().begin(),
                                             encoded.bytes().end())});
  }

  const auto &operation = fixture.operation();
  const auto ordinal =
      operation.capability->configurationFieldSchema.front().ordinal;
  const ConfigurationFieldEncoding *operationField =
      fixture.abi.abi().findOperationField(operation.physicalOccurrence,
                                           ordinal);
  require(test, operationField != nullptr,
          "operation configuration field is absent from ABI 2.0");
  const ProgrammingUnit *operationOwner =
      fieldOwner(fixture.abi.abi(), operationField->field);
  require(test, operationOwner != nullptr,
          "operation field has no programming owner");
  if (owner)
    require(test, owner->id == operationOwner->id,
            "operation and PE configuration have different owners");
  else
    owner = operationOwner;
  values.push_back({operationField->field, fixture.targetSemantic});
  std::vector<std::uint8_t> target =
      take(test, fixture.abi.abi().encode(owner->id, values));
  std::vector<std::uint8_t> invalid = target;
  for (const DestinationSlice &slice : operationField->destinationSlices)
    for (std::uint64_t bit = 0; bit != slice.bitCount; ++bit) {
      const std::uint64_t source = slice.sourceBitOffset + bit;
      writeBit(invalid, slice.destinationBitOffset + bit,
               ((std::uint64_t{15} >> source) & 1U) != 0);
    }
  return {"configuration_" + std::to_string(owner->id), owner->payloadBitCount,
          std::move(target), std::move(invalid)};
}

std::string bitLiteral(llvm::ArrayRef<std::uint8_t> bytes,
                       std::uint64_t bitCount) {
  std::string result;
  result.reserve(static_cast<std::size_t>(bitCount));
  for (std::uint64_t bit = bitCount; bit > 0; --bit) {
    const std::uint64_t index = bit - 1;
    result.push_back(((bytes[index / 8] >> (index % 8)) & 1U) ? '1' : '0');
  }
  return result;
}

std::string leafTestbench() {
  return R"sv(module testbench;
  logic [31:0] mux_data_input_0;
  logic [31:0] mux_data_input_1, mux_data_input_2, mux_data_input_3, mux_data_input_4;
  logic mux_valid_input_0, mux_valid_input_1, mux_valid_input_2;
  logic mux_valid_input_3, mux_valid_input_4, mux_ready_output_0;
  logic [3:0] mux_config_0;
  logic mux_ready_input_0, mux_ready_input_1, mux_ready_input_2;
  logic mux_ready_input_3, mux_ready_input_4;
  logic [31:0] mux_data_output_0;
  logic mux_valid_output_0;

  logic [31:0] demux_data_input_0;
  logic [31:0] demux_data_input_1;
  logic demux_valid_input_0, demux_valid_input_1;
  logic demux_ready_output_0, demux_ready_output_1;
  logic demux_ready_output_2, demux_ready_output_3;
  logic [3:0] demux_config_0;
  logic demux_ready_input_0, demux_ready_input_1;
  logic [31:0] demux_data_output_0, demux_data_output_1;
  logic [31:0] demux_data_output_2, demux_data_output_3;
  logic demux_valid_output_0, demux_valid_output_1;
  logic demux_valid_output_2, demux_valid_output_3;

  token_mux mux(
    .data_input_0(mux_data_input_0), .data_input_1(mux_data_input_1),
    .data_input_2(mux_data_input_2), .data_input_3(mux_data_input_3),
    .data_input_4(mux_data_input_4), .valid_input_0(mux_valid_input_0),
    .valid_input_1(mux_valid_input_1), .valid_input_2(mux_valid_input_2),
    .valid_input_3(mux_valid_input_3), .valid_input_4(mux_valid_input_4),
    .ready_output_0(mux_ready_output_0), .config_0(mux_config_0),
    .ready_input_0(mux_ready_input_0), .ready_input_1(mux_ready_input_1),
    .ready_input_2(mux_ready_input_2), .ready_input_3(mux_ready_input_3),
    .ready_input_4(mux_ready_input_4), .data_output_0(mux_data_output_0),
    .valid_output_0(mux_valid_output_0));

  token_demux demux(
    .data_input_0(demux_data_input_0), .data_input_1(demux_data_input_1),
    .valid_input_0(demux_valid_input_0), .valid_input_1(demux_valid_input_1),
    .ready_output_0(demux_ready_output_0), .ready_output_1(demux_ready_output_1),
    .ready_output_2(demux_ready_output_2), .ready_output_3(demux_ready_output_3),
    .config_0(demux_config_0), .ready_input_0(demux_ready_input_0),
    .ready_input_1(demux_ready_input_1), .data_output_0(demux_data_output_0),
    .data_output_1(demux_data_output_1), .data_output_2(demux_data_output_2),
    .data_output_3(demux_data_output_3), .valid_output_0(demux_valid_output_0),
    .valid_output_1(demux_valid_output_1), .valid_output_2(demux_valid_output_2),
    .valid_output_3(demux_valid_output_3));

  task automatic check(input bit condition, input string message);
    if (!condition) $fatal(1, "%s", message);
  endtask

  initial begin
    mux_data_input_0 = 0;
    mux_data_input_1 = 8'h11;
    mux_data_input_2 = 8'h22;
    mux_data_input_3 = 32'habcd0033;
    mux_data_input_4 = 8'h44;
    mux_valid_input_0 = 1;
    mux_valid_input_1 = 1;
    mux_valid_input_2 = 1;
    mux_valid_input_3 = 1;
    mux_valid_input_4 = 1;
    mux_ready_output_0 = 1;
    mux_config_0 = 4'ha;
    #1;
    check(mux_valid_output_0 && mux_data_output_0 == 8'h11 &&
              mux_ready_input_0 && mux_ready_input_1 &&
              !mux_ready_input_2 && !mux_ready_input_3 && !mux_ready_input_4,
          "mux local lane zero did not use physical input one");
    mux_data_input_0 = 1;
    #1;
    check(mux_data_output_0 == 8'h33 && mux_ready_input_0 &&
              !mux_ready_input_1 && !mux_ready_input_2 &&
              mux_ready_input_3 && !mux_ready_input_4,
          "mux noncontiguous local lane one changed physical identity");
    mux_data_input_0 = 2;
    #1;
    check(mux_data_output_0 == 8'h44 && mux_ready_input_4 &&
              !mux_ready_input_1 && !mux_ready_input_2 && !mux_ready_input_3,
          "mux local lane two did not use physical input four");
    mux_data_input_0 = 7;
    #1;
    check(mux_data_output_0 == 8'h11 && mux_ready_input_1 &&
              !mux_ready_input_2 && !mux_ready_input_3 && !mux_ready_input_4,
          "mux poison selector did not take its deterministic refinement");
    mux_data_input_0 = 1;
    mux_ready_output_0 = 0;
    #1;
    check(mux_valid_output_0 && mux_data_output_0 == 8'h33 &&
              !mux_ready_input_0 && !mux_ready_input_1 &&
              !mux_ready_input_2 && !mux_ready_input_3 && !mux_ready_input_4,
          "stalled mux consumed an input or changed its payload");
    mux_ready_output_0 = 1;
    mux_valid_input_3 = 0;
    #1;
    check(!mux_valid_output_0 && !mux_ready_input_0 && mux_ready_input_3 &&
              !mux_ready_input_1 && !mux_ready_input_2 && !mux_ready_input_4,
          "mux consumed a selector without its selected input");
    mux_valid_input_3 = 1;
    mux_valid_input_0 = 0;
    #1;
    check(!mux_valid_output_0 && mux_ready_input_0 && !mux_ready_input_3,
          "mux consumed selected data without its selector");
    mux_valid_input_0 = 1;
    mux_config_0 = 4'hf;
    #1;
    check(mux_valid_output_0 && mux_data_output_0 == 8'h22 &&
              mux_ready_input_2 && !mux_ready_input_1 &&
              !mux_ready_input_3 && !mux_ready_input_4,
          "mux unmatched code did not use the ABI inactive mode");

    demux_data_input_0 = 1;
    demux_data_input_1 = 32'hffff005a;
    demux_valid_input_0 = 1;
    demux_valid_input_1 = 1;
    demux_ready_output_0 = 1;
    demux_ready_output_1 = 1;
    demux_ready_output_2 = 1;
    demux_ready_output_3 = 1;
    demux_config_0 = 4'ha;
    #1;
    check(demux_ready_input_0 && demux_ready_input_1 &&
              !demux_valid_output_0 && !demux_valid_output_1 &&
              demux_valid_output_2 && !demux_valid_output_3 &&
              demux_data_output_2 == 8'h5a,
          "demux noncontiguous local lane one changed physical identity");
    demux_ready_output_2 = 0;
    #1;
    check(!demux_ready_input_0 && !demux_ready_input_1 &&
              demux_valid_output_2 && demux_data_output_2 == 8'h5a &&
              !demux_valid_output_0 && !demux_valid_output_1 &&
              !demux_valid_output_3,
          "stalled demux consumed or partially published a token");
    demux_ready_output_2 = 1;
    demux_data_input_0 = 2;
    #1;
    check(demux_valid_output_3 && !demux_valid_output_0 &&
              !demux_valid_output_1 && !demux_valid_output_2,
          "demux local lane two did not use physical output three");
    demux_data_input_0 = 7;
    #1;
    check(demux_valid_output_0 && !demux_valid_output_1 &&
              !demux_valid_output_2 && !demux_valid_output_3,
          "demux poison selector did not take its deterministic refinement");
    demux_data_input_0 = 1;
    demux_valid_input_0 = 0;
    #1;
    check(!demux_valid_output_2 && demux_ready_input_0 &&
              !demux_ready_input_1,
          "demux consumed data without its selector");
    demux_valid_input_0 = 1;
    demux_valid_input_1 = 0;
    #1;
    check(!demux_valid_output_2 && !demux_ready_input_0 &&
              demux_ready_input_1,
          "demux consumed a selector without its data");
    demux_valid_input_1 = 1;
    demux_config_0 = 4'hf;
    #1;
    check(demux_valid_output_1 && !demux_valid_output_0 &&
              !demux_valid_output_2 && !demux_valid_output_3,
          "demux unmatched code did not use the ABI inactive mode");
    $finish;
  end
endmodule
)sv";
}

std::string directYosysScript(llvm::StringRef top, llvm::StringRef source) {
  std::string script;
  llvm::raw_string_ostream output(script);
  output << "read_verilog -sv " << source << '\n'
         << "hierarchy -check -top " << top << '\n'
         << "proc\nopt\ncheck -assert\n"
         << "select -assert-none t:$*ff* t:$*latch* t:$_*FF* "
            "t:$_*LATCH* t:$mem*\n"
         << "synth -top " << top << '\n'
         << "check -assert\n";
  return output.str();
}

std::string muxSystemTestbench(const ConfigurationImages &configuration) {
  std::string result = R"sv(module testbench;
  logic clock, reset;
  logic [31:0] input_0_data;
  logic input_0_valid, input_0_ready;
  logic [31:0] input_1_data, input_2_data, input_3_data, input_4_data;
  logic input_1_valid, input_2_valid, input_3_valid, input_4_valid;
  logic input_1_ready, input_2_ready, input_3_ready, input_4_ready;
  logic [31:0] output_0_data;
  logic output_0_valid, output_0_ready;
)sv";
  result += "  logic [" + std::to_string(configuration.bitCount - 1) + ":0] " +
            configuration.portName + ";\n";
  result += R"sv(  loom_module dut(.*);
  always #5 clock = ~clock;
  task automatic check(input bit condition, input string message);
    if (!condition) $fatal(1, "%s", message);
  endtask
  initial begin
    clock = 0; reset = 1;
    input_0_data = 0; input_0_valid = 0;
    input_1_data = 8'h11; input_1_valid = 0;
    input_2_data = 8'h22; input_2_valid = 0;
    input_3_data = 8'h33; input_3_valid = 0;
    input_4_data = 8'h44; input_4_valid = 0;
    output_0_ready = 1;
)sv";
  result += "    " + configuration.portName + " = " +
            std::to_string(configuration.bitCount) + "'b" +
            bitLiteral(configuration.target, configuration.bitCount) + ";\n";
  result += R"sv(    repeat (2) @(posedge clock);
    @(negedge clock); reset = 0;
    input_0_data = 1; input_0_valid = 1;
    input_3_data = 32'habcd003c; input_3_valid = 1;
    #1;
    check(input_0_ready && input_3_ready && !input_1_ready &&
              !input_2_ready && !input_4_ready && !output_0_valid,
          "mux shell did not admit only the complete selected tuple");
    @(posedge clock); #1;
    check(output_0_valid && output_0_data == 8'h3c,
          "mux shell did not publish the accepted tuple");
    output_0_ready = 0;
    input_0_data = 2; input_3_valid = 0;
    input_4_data = 32'hffff00a5; input_4_valid = 1;
    repeat (3) begin
      @(negedge clock); #1;
      check(output_0_valid && output_0_data == 8'h3c &&
                !input_0_ready && !input_4_ready,
            "mux shell did not hold a stalled token and backpressure");
    end
    output_0_ready = 1; #1;
    check(input_0_ready && input_4_ready,
          "mux shell did not admit a same-cycle replacement");
    @(posedge clock); #1;
    check(output_0_valid && output_0_data == 8'ha5,
          "mux shell replacement changed the selected token");
    input_0_valid = 0; input_4_valid = 0;
    @(posedge clock); #1;
    check(!output_0_valid, "mux shell retained a released token");
    @(negedge clock); reset = 1; #1;
    check(!output_0_valid && !input_0_ready && !input_1_ready &&
              !input_2_ready && !input_3_ready && !input_4_ready,
          "mux shell reset did not quiesce the boundary");
    reset = 0;
)sv";
  result += "    " + configuration.portName + " = " +
            std::to_string(configuration.bitCount) + "'b" +
            bitLiteral(configuration.invalid, configuration.bitCount) + ";\n";
  result += R"sv(    input_0_data = 1; input_0_valid = 1;
    input_2_data = 32'h12340066; input_2_valid = 1; #1;
    check(input_0_ready && input_2_ready && !input_1_ready &&
              !input_3_ready && !input_4_ready,
          "mux shell unmatched code did not select the ABI inactive lane");
    @(posedge clock); #1;
    check(output_0_valid && output_0_data == 8'h66,
          "mux shell inactive fallback changed the token");
    $finish;
  end
endmodule
)sv";
  return result;
}

std::string demuxSystemTestbench(const ConfigurationImages &configuration) {
  std::string result = R"sv(module testbench;
  logic clock, reset;
  logic [31:0] input_0_data;
  logic input_0_valid, input_0_ready;
  logic [31:0] input_1_data;
  logic input_1_valid, input_1_ready;
  logic [31:0] output_0_data, output_1_data, output_2_data, output_3_data;
  logic output_0_valid, output_1_valid, output_2_valid, output_3_valid;
  logic output_0_ready, output_1_ready, output_2_ready, output_3_ready;
)sv";
  result += "  logic [" + std::to_string(configuration.bitCount - 1) + ":0] " +
            configuration.portName + ";\n";
  result += R"sv(  loom_module dut(.*);
  always #5 clock = ~clock;
  task automatic check(input bit condition, input string message);
    if (!condition) $fatal(1, "%s", message);
  endtask
  initial begin
    clock = 0; reset = 1;
    input_0_data = 1; input_0_valid = 0;
    input_1_data = 32'habcd003c; input_1_valid = 0;
    output_0_ready = 1; output_1_ready = 1;
    output_2_ready = 1; output_3_ready = 1;
)sv";
  result += "    " + configuration.portName + " = " +
            std::to_string(configuration.bitCount) + "'b" +
            bitLiteral(configuration.target, configuration.bitCount) + ";\n";
  result += R"sv(    repeat (2) @(posedge clock);
    @(negedge clock); reset = 0;
    input_0_valid = 1; input_1_valid = 1; #1;
    check(input_0_ready && input_1_ready && !output_0_valid &&
              !output_1_valid && !output_2_valid && !output_3_valid,
          "demux shell did not admit one complete tuple");
    @(posedge clock); #1;
    check(!output_0_valid && !output_1_valid && output_2_valid &&
              !output_3_valid && output_2_data == 8'h3c,
          "demux shell partially published or misrouted a tuple");
    output_2_ready = 0; input_1_data = 32'hffff00a5;
    repeat (3) begin
      @(negedge clock); #1;
      check(output_2_valid && output_2_data == 8'h3c &&
                !input_0_ready && !input_1_ready &&
                !output_0_valid && !output_1_valid && !output_3_valid,
            "demux shell did not hold selected-output backpressure");
    end
    output_2_ready = 1; #1;
    check(input_0_ready && input_1_ready,
          "demux shell did not admit a same-cycle replacement");
    @(posedge clock); #1;
    check(output_2_valid && output_2_data == 8'ha5,
          "demux shell replacement changed the token");
    input_0_valid = 0; input_1_valid = 0;
    @(posedge clock); #1;
    check(!output_0_valid && !output_1_valid && !output_2_valid &&
              !output_3_valid,
          "demux shell retained a released token");
    @(negedge clock); reset = 1; #1;
    check(!input_0_ready && !input_1_ready && !output_0_valid &&
              !output_1_valid && !output_2_valid && !output_3_valid,
          "demux shell reset did not quiesce the boundary");
    reset = 0;
)sv";
  result += "    " + configuration.portName + " = " +
            std::to_string(configuration.bitCount) + "'b" +
            bitLiteral(configuration.invalid, configuration.bitCount) + ";\n";
  result += R"sv(    input_0_data = 1; input_0_valid = 1;
    input_1_data = 32'h12340077; input_1_valid = 1; #1;
    check(input_0_ready && input_1_ready,
          "demux shell unmatched code did not admit the inactive lane");
    @(posedge clock); #1;
    check(!output_0_valid && output_1_valid && !output_2_valid &&
              !output_3_valid && output_1_data == 8'h77,
          "demux shell inactive fallback changed the token");
    $finish;
  end
endmodule
)sv";
  return result;
}

std::string systemYosysScript(llvm::StringRef rtlName) {
  return (llvm::Twine("read_verilog -sv ") + rtlName + R"ys(
hierarchy -check -top loom_module
check -assert
proc
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys")
      .str();
}

void validateAndWrite(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  Fixture mux = makeFixture(test, store, FamilyKind::Mux);
  Fixture demux = makeFixture(test, store, FamilyKind::Demux);

  FabricOperationProviderRegistry registry = makeRegistry(test);
  const auto coverage = registry.coverage();
  const auto registered = llvm::count_if(
      coverage, [](const auto &entry) { return !entry.recipes.empty(); });
  require(test,
          registered == 2 &&
              coverage[static_cast<std::size_t>(familyId(FamilyKind::Mux))]
                      .recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog} &&
              coverage[static_cast<std::size_t>(familyId(FamilyKind::Demux))]
                      .recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "token routing registration does not expose exactly two portable "
          "families");
  const auto beforeDuplicate = registry.coverage();
  expectError(
      test,
      loom::hardware::rtl::registerPortableTokenMuxDemuxProviders(registry),
      "duplicate");
  const auto afterDuplicate = registry.coverage();
  require(test, afterDuplicate.size() == beforeDuplicate.size(),
          "failed registration changed provider coverage size");
  for (std::size_t index = 0; index != beforeDuplicate.size(); ++index)
    require(test,
            afterDuplicate[index].implementationFamily ==
                    beforeDuplicate[index].implementationFamily &&
                afterDuplicate[index].recipes == beforeDuplicate[index].recipes,
            "failed registration partially mutated the provider registry");

  std::unique_ptr<mlir::MLIRContext> muxLeafContext = makeCirctContext();
  const std::string muxRtl =
      specializeLeaf(test, makeLeafSkeleton(test, *muxLeafContext, mux), mux);
  std::unique_ptr<mlir::MLIRContext> muxLeafRepeatContext = makeCirctContext();
  const std::string muxRepeat = specializeLeaf(
      test, makeLeafSkeleton(test, *muxLeafRepeatContext, mux), mux);
  require(test, muxRtl == muxRepeat,
          "mux RTL changed across identical specializations");

  std::unique_ptr<mlir::MLIRContext> demuxLeafContext = makeCirctContext();
  const std::string demuxRtl = specializeLeaf(
      test, makeLeafSkeleton(test, *demuxLeafContext, demux), demux);
  std::unique_ptr<mlir::MLIRContext> demuxLeafRepeatContext =
      makeCirctContext();
  const std::string demuxRepeat = specializeLeaf(
      test, makeLeafSkeleton(test, *demuxLeafRepeatContext, demux), demux);
  require(test, demuxRtl == demuxRepeat,
          "demux RTL changed across identical specializations");

  std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
  LeafSkeleton malformed = makeLeafSkeleton(test, *malformedContext, mux, true);
  const std::string beforeMalformed = moduleText(*malformed.module);
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {malformed.leaf, mux.operation().physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {mux.operation().physicalOccurrence,
       BackendRecipeKey::PortableSystemVerilog,
       {}}};
  expectError(test,
              loom::hardware::rtl::specializeFabricOperationLeaves(
                  *malformed.module, mux.abi, associations, recipes, registry,
                  externalContracts),
              "leaf port");
  require(test, moduleText(*malformed.module) == beforeMalformed,
          "malformed leaf rejection partially mutated the caller module");

  std::unique_ptr<mlir::MLIRContext> nativeContext = makeCirctContext();
  LeafSkeleton native = makeLeafSkeleton(test, *nativeContext, demux);
  const std::string beforeNative = moduleText(*native.module);
  const std::vector<FabricOperationLeafAssociation> nativeAssociations = {
      {native.leaf, demux.operation().physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> nativeRecipes = {
      {demux.operation().physicalOccurrence,
       BackendRecipeKey::SynopsysDesignWare,
       {}}};
  expectTypedUnsupported(test,
                         loom::hardware::rtl::specializeFabricOperationLeaves(
                             *native.module, demux.abi, nativeAssociations,
                             nativeRecipes, registry, externalContracts),
                         FamilyKind::Demux,
                         BackendRecipeKey::SynopsysDesignWare);
  require(test, moduleText(*native.module) == beforeNative,
          "Unsupported recipe partially mutated the caller module");

  std::unique_ptr<mlir::MLIRContext> muxSystemContext = makeCirctContext();
  const std::string muxSystem = specializeSystem(test, *muxSystemContext, mux);
  std::unique_ptr<mlir::MLIRContext> muxSystemRepeatContext =
      makeCirctContext();
  require(test,
          muxSystem == specializeSystem(test, *muxSystemRepeatContext, mux),
          "mux CommonSkeleton RTL changed across identical specializations");
  std::unique_ptr<mlir::MLIRContext> demuxSystemContext = makeCirctContext();
  const std::string demuxSystem =
      specializeSystem(test, *demuxSystemContext, demux);
  std::unique_ptr<mlir::MLIRContext> demuxSystemRepeatContext =
      makeCirctContext();
  require(test,
          demuxSystem ==
              specializeSystem(test, *demuxSystemRepeatContext, demux),
          "demux CommonSkeleton RTL changed across identical specializations");

  const ConfigurationImages muxConfiguration =
      makeConfigurationImages(test, mux);
  const ConfigurationImages demuxConfiguration =
      makeConfigurationImages(test, demux);
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "leaf",
          {{"token_mux.sv", muxRtl},
           {"token_demux.sv", demuxRtl},
           {"testbench.sv", leafTestbench()},
           {"token_mux.ys", directYosysScript("token_mux", "token_mux.sv")},
           {"token_demux.ys",
            directYosysScript("token_demux", "token_demux.sv")}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "mux_system",
          {{"token_mux_system.sv", muxSystem},
           {"testbench.sv", muxSystemTestbench(muxConfiguration)},
           {"token_mux_system.ys", systemYosysScript("token_mux_system.sv")}}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "demux_system",
          {{"token_demux_system.sv", demuxSystem},
           {"testbench.sv", demuxSystemTestbench(demuxConfiguration)},
           {"token_demux_system.ys",
            systemYosysScript("token_demux_system.sv")}}))
    fail(test, llvm::toString(std::move(error)));
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  validateAndWrite(argv[1]);
  return 0;
}

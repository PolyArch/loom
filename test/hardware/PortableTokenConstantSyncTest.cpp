#include "ADG/Builder.h"
#include "ConfigurationABI2TestSupport.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/TokenConstantSync.h"
#include "PortableProviderTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/OperationResourceContract.h"

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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
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

struct FabricFixture final {
  FinalizedFabricRoot module;
  FabricFuOccurrenceNodeRef localOccurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

std::vector<loom::adg::PortType>
portTypes(llvm::StringRef test, llvm::ArrayRef<std::uint32_t> widths) {
  std::vector<loom::adg::PortType> result;
  result.reserve(widths.size());
  for (std::uint32_t width : widths)
    result.push_back(take(test, loom::adg::PortType::bits(width)));
  return result;
}

FabricFixture
makeFabric(llvm::StringRef test, ArtifactStore &store, llvm::StringRef name,
           ::fabric::ImplementationFamilyId family,
           ::fabric::FamilyCapabilityParams parameters,
           ::dataflow::OperationSchemaId schema,
           llvm::ArrayRef<std::uint32_t> inputWidths,
           llvm::ArrayRef<std::uint32_t> outputWidths,
           const ::fabric::ResourceContract &resourceContract =
               ::fabric::oneCycleElasticOperationResourceContract()) {
  using loom::adg::DesignBuilder;
  using loom::adg::FuCapabilityTemplateSpec;
  using loom::adg::FuSpec;
  using loom::adg::OperationCapabilitySpec;
  using loom::adg::PeSpec;

  std::vector<loom::adg::PortType> inputs = portTypes(test, inputWidths);
  std::vector<loom::adg::PortType> outputs = portTypes(test, outputWidths);
  const std::uint32_t boundaryWidth =
      std::max(*std::max_element(inputWidths.begin(), inputWidths.end()),
               *std::max_element(outputWidths.begin(), outputWidths.end()));
  const loom::adg::PortType boundary =
      take(test, loom::adg::PortType::bits(boundaryWidth));
  const std::vector<loom::adg::PortType> boundaryInputs(inputs.size(),
                                                        boundary);
  const std::vector<loom::adg::PortType> boundaryOutputs(outputs.size(),
                                                         boundary);
  DesignBuilder builder(store);
  auto spatial = take(
      test, builder.createSpatialCore(name, boundaryInputs, boundaryOutputs));
  std::vector<loom::adg::SpatialValue> spatialInputs;
  for (std::size_t ordinal = 0; ordinal != boundaryInputs.size(); ++ordinal)
    spatialInputs.push_back(take(test, spatial.input(ordinal)));
  auto pe = take(
      test, spatial.addPe(spatialInputs,
                          PeSpec::spatial(boundaryInputs, boundaryOutputs)));
  std::vector<loom::adg::PeValue> peInputs;
  for (std::size_t ordinal = 0; ordinal != boundaryInputs.size(); ++ordinal)
    peInputs.push_back(take(test, pe.input(ordinal)));
  auto fu = take(test, pe.addFu(peInputs, FuSpec{inputs, boundaryOutputs}));
  std::vector<loom::adg::FuValue> fuInputs;
  for (std::size_t ordinal = 0; ordinal != inputs.size(); ++ordinal)
    fuInputs.push_back(take(test, fu.input(ordinal)));
  auto operation =
      take(test, fu.addOperation(fuInputs,
                                 OperationCapabilitySpec{family,
                                                         std::move(parameters),
                                                         {schema},
                                                         outputs,
                                                         resourceContract}));
  if (llvm::Error error =
          fu.addCapabilityTemplate(FuCapabilityTemplateSpec{{operation}, {}}))
    fail(test, llvm::toString(std::move(error)));
  std::vector<loom::adg::FuValue> operationOutputs;
  for (std::size_t ordinal = 0; ordinal != outputs.size(); ++ordinal)
    operationOutputs.push_back(take(test, operation.output(ordinal)));
  if (llvm::Error error = fu.close(operationOutputs))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  std::vector<loom::adg::SpatialValue> spatialOutputs;
  for (std::size_t ordinal = 0; ordinal != boundaryOutputs.size(); ++ordinal)
    spatialOutputs.push_back(take(test, pe.output(ordinal)));
  if (llvm::Error error = spatial.close(spatialOutputs))
    fail(test, llvm::toString(std::move(error)));
  auto design = take(test, std::move(builder).finalize());
  require(test, design.roots().size() == 1,
          "provider fixture did not publish one Module root");
  FinalizedFabricRoot module = design.roots().front();

  for (const auto fuOccurrence : module.view().fuOccurrences()) {
    const auto definition = module.view().fuTemplateOf(fuOccurrence);
    if (!definition)
      continue;
    for (const auto &capability :
         module.view().resolvedFabricOpCapabilities(*definition)) {
      if (capability.implementationFamily != family)
        continue;
      FabricFuOccurrenceNodeRef local =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         module.view(), capability.occurrence, fuOccurrence));
      FinalizedFabricRoot system =
          take(test, loom::hardware::test::makeSingleSpatialCoreSystem(module,
                                                                       store));
      auto systemView =
          take(test, loom::fabric::requireSystemRoot(system.view()));
      auto operations =
          take(test, enumerateFabricPhysicalOperations(systemView));
      const auto physical = llvm::find_if(operations, [&](const auto &entry) {
        return entry.localOccurrence == local;
      });
      require(test, physical != operations.end(),
              "System has no physical provider occurrence");
      return FabricFixture{std::move(module), local, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "fixture has no requested operation family");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.module.view().resolvedFabricOpCapability(fixture.localOccurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

std::vector<std::uint8_t> packedValue(std::uint64_t value,
                                      std::uint64_t bitCount) {
  std::vector<std::uint8_t> result((bitCount + 7) / 8, 0);
  for (std::uint64_t bit = 0; bit != bitCount; ++bit)
    if ((value & (std::uint64_t{1} << bit)) != 0)
      result[static_cast<std::size_t>(bit / 8)] |=
          static_cast<std::uint8_t>(1U << (bit % 8));
  return result;
}

FinalizedConfigurationABI makeConstantAbi(llvm::StringRef test,
                                          const ArtifactStore &store,
                                          const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(relationContext()));
  require(test,
          relation.kind() ==
                  fabric::FabricOpSemanticFieldRelationKind::Direct &&
              relation.directEncodedBitCount().has_value() &&
              resolved.configurationFieldSchema.size() == 1,
          "constant fixture does not expose one direct carrier");
  const std::uint64_t width = *relation.directEncodedBitCount();
  auto physical =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physical, DirectBitsEncoding{width}, packedValue(0, width)};
  return take(
      test,
      finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         fixture.system, {std::move(field)})),
          store));
}

std::uint64_t
laneMask(llvm::StringRef test,
         const fabric::FiniteImplementationFamilyBehaviorPoint &point,
         std::size_t physicalLaneCount) {
  require(test,
          !point.operandPorts.empty() &&
              point.operandPorts == point.resultPorts,
          "sync relation does not preserve its positional lane image");
  require(test, llvm::is_sorted(point.operandPorts),
          "sync relation lane image is not ordered");
  std::uint64_t mask = 0;
  for (std::uint64_t ordinal : point.operandPorts) {
    require(test, ordinal < physicalLaneCount && ordinal < 64,
            "sync relation lane image is out of range");
    mask |= std::uint64_t{1} << ordinal;
  }
  return mask;
}

FinalizedConfigurationABI makeSyncAbi(llvm::StringRef test,
                                      const ArtifactStore &store,
                                      const FabricFixture &fixture,
                                      std::uint64_t inactiveMask) {
  const auto &resolved = capability(test, fixture);
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(relationContext()));
  if (relation.kind() == fabric::FabricOpSemanticFieldRelationKind::None) {
    require(test, resolved.configurationFieldSchema.empty(),
            "singleton sync relation retained a configuration field");
    return take(
        test,
        finalizeConfigurationABI(
            take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                           fixture.system)),
            store));
  }
  require(test,
          relation.kind() ==
                  fabric::FabricOpSemanticFieldRelationKind::Finite &&
              resolved.configurationFieldSchema.size() == 1,
          "configured sync fixture does not expose one finite field");
  const std::size_t laneCount = resolved.physicalPorts.size() / 2;
  require(test, laneCount > 1 && laneCount < 64,
          "configured sync fixture has an unsupported lane count");
  const std::uint64_t encodedBitCount = laneCount + 1;
  const std::uint64_t laneBits = (std::uint64_t{1} << laneCount) - 1;
  const std::uint64_t physicalTag = std::uint64_t{1} << laneCount;
  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactive;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "configured sync mode has no semantic key");
    const std::uint64_t mask = laneMask(test, point, laneCount);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (mask == inactiveMask)
      inactive = semantic;
    const std::uint64_t physicalCode = physicalTag | (mask ^ laneBits);
    require(test, physicalCode != mask,
            "sync test ABI reused a semantic lane mask as a physical code");
    entries.push_back(
        {std::move(semantic), packedValue(physicalCode, encodedBitCount)});
  }
  require(test, !inactive.empty(),
          "requested inactive sync lane image is not reachable");
  auto physical =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physical, FiniteCodebookEncoding{encodedBitCount, std::move(entries)},
      std::move(inactive)};
  return take(
      test,
      finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         fixture.system, {std::move(field)})),
          store));
}

std::uint64_t unpackedValue(llvm::StringRef test,
                            llvm::ArrayRef<std::uint8_t> bytes,
                            std::uint64_t bitCount) {
  require(test, bitCount <= 64 && bytes.size() == (bitCount + 7) / 8,
          "test ABI code has an unsupported packed width");
  std::uint64_t result = 0;
  for (std::uint64_t bit = 0; bit != bitCount; ++bit)
    if ((bytes[static_cast<std::size_t>(bit / 8)] &
         static_cast<std::uint8_t>(1U << (bit % 8))) != 0)
      result |= std::uint64_t{1} << bit;
  return result;
}

struct LeafSyncCodes final {
  std::uint64_t bitCount = 0;
  std::uint64_t spare = 0;
  std::uint64_t inactive = 0;
  std::uint64_t singleLane = 0;
  std::uint64_t allLanes = 0;
};

LeafSyncCodes leafSyncCodes(llvm::StringRef test, const FabricFixture &fixture,
                            const FinalizedConfigurationABI &abi,
                            std::uint64_t inactiveMask,
                            std::uint64_t singleLaneMask,
                            std::uint64_t allLanesMask) {
  const auto &resolved = capability(test, fixture);
  require(test, resolved.configurationFieldSchema.size() == 1,
          "configured sync fixture does not have one field");
  const ConfigurationFieldEncoding *field = abi.abi().findOperationField(
      fixture.physicalOccurrence,
      resolved.configurationFieldSchema.front().ordinal);
  require(test, field != nullptr,
          "finalized ABI does not own the sync operation field");
  const auto *codebook =
      std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
  require(test, codebook != nullptr && codebook->encodedBitCount < 16,
          "finalized sync field does not have a small finite codebook");

  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(relationContext()));
  require(test,
          relation.kind() == fabric::FabricOpSemanticFieldRelationKind::Finite,
          "configured sync relation is not finite");
  const std::size_t laneCount = resolved.physicalPorts.size() / 2;
  auto codeForMask = [&](std::uint64_t requestedMask) {
    const auto point = llvm::find_if(
        relation.finiteBehaviorDomain(), [&](const auto &candidate) {
          return laneMask(test, candidate, laneCount) == requestedMask;
        });
    require(test, point != relation.finiteBehaviorDomain().end(),
            "requested sync lane image is absent from the relation");
    const auto entry = llvm::find_if(
        codebook->entries, [&](const FiniteCodebookEntry &candidate) {
          return llvm::ArrayRef<std::uint8_t>(candidate.semanticValue)
              .equals(point->semanticConfiguration->bytes());
        });
    require(test, entry != codebook->entries.end(),
            "finalized ABI omitted a sync relation point");
    return unpackedValue(test, entry->physicalCode, codebook->encodedBitCount);
  };

  const std::uint64_t codeCount = std::uint64_t{1} << codebook->encodedBitCount;
  std::vector<bool> used(static_cast<std::size_t>(codeCount), false);
  for (const auto &entry : codebook->entries)
    used[static_cast<std::size_t>(unpackedValue(
        test, entry.physicalCode, codebook->encodedBitCount))] = true;
  const auto spare = llvm::find(used, false);
  require(test, spare != used.end(),
          "sync test ABI unexpectedly exhausted its physical carrier");
  return LeafSyncCodes{
      codebook->encodedBitCount,
      static_cast<std::uint64_t>(std::distance(used.begin(), spare)),
      codeForMask(inactiveMask), codeForMask(singleLaneMask),
      codeForMask(allLanesMask)};
}

struct LeafFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  circt::hw::HWModuleGeneratedOp leaf;
};

LeafFixture makeLeaf(llvm::StringRef test, mlir::MLIRContext &context,
                     const FabricFixture &fixture, const ConfigurationABI &abi,
                     llvm::StringRef name) {
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
  auto leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(name), ports);
  return LeafFixture{std::move(module), leaf};
}

std::string specializeLeaf(llvm::StringRef test, const FabricFixture &fixture,
                           const FinalizedConfigurationABI &abi,
                           llvm::StringRef name) {
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  LeafFixture leaf = makeLeaf(test, *context, fixture, abi.abi(), name);
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableTokenConstantSyncProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton skeleton{std::move(leaf.module),
                                   {{leaf.leaf, fixture.physicalOccurrence}}};
  auto conformance =
      take(test, loom::hardware::test::specializeAndExportPortableProvider(
                     std::move(skeleton), abi, registry, externalContracts));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "portable token provider emitted implementation metadata");
  return std::move(conformance.systemVerilog);
}

std::string leafTestbench(const LeafSyncCodes &codes,
                          const LeafSyncCodes &zeroPayloadCodes) {
  std::ostringstream output;
  output << R"sv(
module provider_testbench;
  logic [7:0]  constant_data_input_0;
  logic        constant_valid_input_0;
  logic        constant_ready_output_0;
  logic [7:0]  constant_config_0;
  logic        constant_ready_input_0;
  logic [11:0] constant_data_output_0;
  logic        constant_valid_output_0;

  token_constant constant_dut(
    .data_input_0(constant_data_input_0),
    .valid_input_0(constant_valid_input_0),
    .ready_output_0(constant_ready_output_0),
    .config_0(constant_config_0),
    .ready_input_0(constant_ready_input_0),
    .data_output_0(constant_data_output_0),
    .valid_output_0(constant_valid_output_0));

  logic [7:0]  sync_data_input_0;
  logic [15:0] sync_data_input_1;
  logic [23:0] sync_data_input_2;
  logic        sync_valid_input_0;
  logic        sync_valid_input_1;
  logic        sync_valid_input_2;
  logic        sync_ready_output_0;
  logic        sync_ready_output_1;
  logic        sync_ready_output_2;
)sv";
  output << "  logic [" << codes.bitCount - 1 << ":0] sync_config_0;\n"
         << "  localparam logic [" << codes.bitCount - 1
         << ":0] SYNC_SPARE_CODE = " << codes.bitCount << "'d" << codes.spare
         << ";\n"
         << "  localparam logic [" << codes.bitCount - 1
         << ":0] SYNC_INACTIVE_CODE = " << codes.bitCount << "'d"
         << codes.inactive << ";\n"
         << "  localparam logic [" << codes.bitCount - 1
         << ":0] SYNC_SINGLE_LANE_CODE = " << codes.bitCount << "'d"
         << codes.singleLane << ";\n"
         << "  localparam logic [" << codes.bitCount - 1
         << ":0] SYNC_ALL_LANES_CODE = " << codes.bitCount << "'d"
         << codes.allLanes << ";\n"
         << "  logic [" << zeroPayloadCodes.bitCount - 1
         << ":0] zero_config_0;\n"
         << "  localparam logic [" << zeroPayloadCodes.bitCount - 1
         << ":0] ZERO_ALL_LANES_CODE = " << zeroPayloadCodes.bitCount << "'d"
         << zeroPayloadCodes.allLanes << ";\n";
  output << R"sv(
  logic        sync_ready_input_0;
  logic        sync_ready_input_1;
  logic        sync_ready_input_2;
  logic [11:0] sync_data_output_0;
  logic [7:0]  sync_data_output_1;
  logic [23:0] sync_data_output_2;
  logic        sync_valid_output_0;
  logic        sync_valid_output_1;
  logic        sync_valid_output_2;

  token_sync sync_dut(
    .data_input_0(sync_data_input_0),
    .data_input_1(sync_data_input_1),
    .data_input_2(sync_data_input_2),
    .valid_input_0(sync_valid_input_0),
    .valid_input_1(sync_valid_input_1),
    .valid_input_2(sync_valid_input_2),
    .ready_output_0(sync_ready_output_0),
    .ready_output_1(sync_ready_output_1),
    .ready_output_2(sync_ready_output_2),
    .config_0(sync_config_0),
    .ready_input_0(sync_ready_input_0),
    .ready_input_1(sync_ready_input_1),
    .ready_input_2(sync_ready_input_2),
    .data_output_0(sync_data_output_0),
    .data_output_1(sync_data_output_1),
    .data_output_2(sync_data_output_2),
    .valid_output_0(sync_valid_output_0),
    .valid_output_1(sync_valid_output_1),
    .valid_output_2(sync_valid_output_2));

  logic [7:0]  zero_data_input_1;
  logic        zero_valid_input_0;
  logic        zero_valid_input_1;
  logic        zero_ready_output_0;
  logic        zero_ready_output_1;
  logic        zero_ready_input_0;
  logic        zero_ready_input_1;
  logic        zero_valid_output_0;
  logic [11:0] zero_data_output_1;
  logic        zero_valid_output_1;

  token_sync_zero_payload zero_dut(
    .valid_input_0(zero_valid_input_0),
    .data_input_1(zero_data_input_1),
    .valid_input_1(zero_valid_input_1),
    .ready_output_0(zero_ready_output_0),
    .ready_output_1(zero_ready_output_1),
    .config_0(zero_config_0),
    .ready_input_0(zero_ready_input_0),
    .ready_input_1(zero_ready_input_1),
    .valid_output_0(zero_valid_output_0),
    .data_output_1(zero_data_output_1),
    .valid_output_1(zero_valid_output_1));

  logic [7:0]  singleton_data_input_0;
  logic        singleton_valid_input_0;
  logic        singleton_ready_output_0;
  logic        singleton_ready_input_0;
  logic [11:0] singleton_data_output_0;
  logic        singleton_valid_output_0;

  token_sync_singleton singleton_dut(
    .data_input_0(singleton_data_input_0),
    .valid_input_0(singleton_valid_input_0),
    .ready_output_0(singleton_ready_output_0),
    .ready_input_0(singleton_ready_input_0),
    .data_output_0(singleton_data_output_0),
    .valid_output_0(singleton_valid_output_0));

  task automatic check(input bit condition, input string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

  initial begin
    constant_data_input_0 = 8'hff;
    constant_config_0 = 8'ha5;
    constant_valid_input_0 = 0;
    constant_ready_output_0 = 1;
    #1;
    check(constant_ready_input_0 && !constant_valid_output_0 &&
              constant_data_output_0 == 12'h0a5,
          "constant did not preserve payload and control activity");
    constant_data_input_0 = 8'h00;
    constant_valid_input_0 = 1;
    constant_ready_output_0 = 0;
    #1;
    check(!constant_ready_input_0 && constant_valid_output_0 &&
              constant_data_output_0 == 12'h0a5,
          "constant payload depended on control data or readiness");
    constant_ready_output_0 = 1;
    #1;
    check(constant_ready_input_0 && constant_valid_output_0,
          "constant did not complete its one-token transfer");

    sync_data_input_0 = 8'h5a;
    sync_data_input_1 = 16'habcd;
    sync_data_input_2 = 24'habcdef;
    sync_valid_input_0 = 1;
    sync_valid_input_1 = 1;
    sync_valid_input_2 = 0;
    sync_ready_output_0 = 1;
    sync_ready_output_1 = 1;
    sync_ready_output_2 = 1;
    sync_config_0 = SYNC_SPARE_CODE;
    #1;
    check(!sync_ready_input_0 && !sync_ready_input_1 &&
              sync_ready_input_2 && !sync_valid_output_0 &&
              !sync_valid_output_1 && !sync_valid_output_2,
          "spare sync code did not use the ABI inactive lane image");

    sync_valid_input_2 = 1;
    sync_ready_output_2 = 0;
    sync_config_0 = SYNC_INACTIVE_CODE;
    #1;
    check(!sync_ready_input_0 && !sync_ready_input_1 &&
              !sync_ready_input_2 && !sync_valid_output_0 &&
              !sync_valid_output_1 && sync_valid_output_2,
          "sync atomic fork exposed a partial transfer");
    sync_ready_output_2 = 1;
    #1;
    check(sync_ready_input_0 && !sync_ready_input_1 &&
              sync_ready_input_2 && sync_valid_output_0 &&
              !sync_valid_output_1 && sync_valid_output_2 &&
              sync_data_output_0 == 12'h05a &&
              sync_data_output_2 == 24'h00cdef,
          "noncontiguous sync lane image changed order or payload width");

    sync_config_0 = SYNC_SINGLE_LANE_CODE;
    #1;
    check(!sync_ready_input_0 && sync_ready_input_1 &&
              !sync_ready_input_2 && !sync_valid_output_0 &&
              sync_valid_output_1 && !sync_valid_output_2 &&
              sync_data_output_1 == 8'hcd,
          "single selected sync lane did not preserve its positional payload");

    sync_config_0 = SYNC_ALL_LANES_CODE;
    sync_valid_input_1 = 0;
    #1;
    check(!sync_ready_input_0 && sync_ready_input_1 &&
              !sync_ready_input_2 && !sync_valid_output_0 &&
              !sync_valid_output_1 && !sync_valid_output_2,
          "three-lane sync consumed an incomplete input tuple");

    zero_data_input_1 = 8'hc7;
    zero_valid_input_0 = 1;
    zero_valid_input_1 = 1;
    zero_ready_output_0 = 1;
    zero_ready_output_1 = 0;
    zero_config_0 = ZERO_ALL_LANES_CODE;
    #1;
    check(!zero_ready_input_0 && !zero_ready_input_1 &&
              !zero_valid_output_0 && zero_valid_output_1 &&
              zero_data_output_1 == 12'h0c7,
          "zero-payload sync lane escaped atomic tuple backpressure");
    zero_ready_output_1 = 1;
    #1;
    check(zero_ready_input_0 && zero_ready_input_1 &&
              zero_valid_output_0 && zero_valid_output_1,
          "zero-payload sync lane did not complete its tuple transfer");

    singleton_data_input_0 = 8'hd3;
    singleton_valid_input_0 = 1;
    singleton_ready_output_0 = 1;
    #1;
    check(singleton_ready_input_0 && singleton_valid_output_0 &&
              singleton_data_output_0 == 12'h0d3,
          "fieldless singleton sync did not forward its token");
    $finish;
  end
endmodule
)sv";
  return output.str();
}

struct SystemConfiguration final {
  std::string portName;
  std::uint64_t bitCount = 0;
  std::vector<std::uint8_t> payload;
};

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
  if (!result)
    fail(test, "Module boundary endpoint is unattached");
  return *result;
}

loom::fabric::FabricPhysicalConfigurationFieldRef
qualifyField(llvm::StringRef test,
             loom::fabric::SpatialCoreOccurrenceRef spatialCore,
             const loom::fabric::FabricSemanticConfigFieldRef &field) {
  auto target =
      take(test, loom::fabric::FabricModulePhysicalTargetRef::create(field));
  return take(test, loom::fabric::FabricPhysicalConfigurationFieldRef::create(
                        loom::fabric::SpatialCoreInternalOccurrenceRef{
                            spatialCore, std::move(target)}));
}

SystemConfiguration
makeSystemConfiguration(llvm::StringRef test, const FabricFixture &fixture,
                        loom::fabric::SpatialCoreOccurrenceRef spatialCore,
                        const ConfigurationABI &abi) {
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
              "PE selector field has no physical port");
      value = loom::fabric::FabricPeRoute{boundaryEndpoint(
          test, fixture.module.view(), descriptor.port->direction,
          descriptor.port->ordinal)};
    }
    const auto physical = qualifyField(test, spatialCore, descriptor.reference);
    const ProgrammingUnit *fieldOwner = nullptr;
    for (const ProgrammingUnit &unit : abi.programmingUnits())
      for (const ConfigurationFieldEncoding &field : unit.fields)
        if (field.field == physical)
          fieldOwner = &unit;
    require(test, fieldOwner != nullptr, "PE field has no programming unit");
    if (owner)
      require(test, owner->id == fieldOwner->id,
              "PE route configuration spans programming units");
    else
      owner = fieldOwner;
    auto semantic = take(test, schema.encode(descriptor.reference, value));
    values.push_back(
        {physical, std::vector<std::uint8_t>(semantic.bytes().begin(),
                                             semantic.bytes().end())});
  }
  require(test, owner != nullptr, "route configuration has no owner");
  return SystemConfiguration{"configuration_" + std::to_string(owner->id),
                             owner->payloadBitCount,
                             take(test, abi.encode(owner->id, values))};
}

std::string bitLiteral(llvm::ArrayRef<std::uint8_t> bytes,
                       std::uint64_t bitCount) {
  std::string result;
  for (std::uint64_t bit = bitCount; bit > 0; --bit) {
    const std::uint64_t index = bit - 1;
    result.push_back(
        ((bytes[static_cast<std::size_t>(index / 8)] >> (index % 8)) & 1U) != 0
            ? '1'
            : '0');
  }
  return result;
}

std::pair<std::string, SystemConfiguration>
specializeSystem(llvm::StringRef test, const FabricFixture &fixture,
                 const FinalizedConfigurationABI &abi) {
  auto systemView =
      take(test, loom::fabric::requireSystemRoot(fixture.system.view()));
  require(test, systemView.artifact().accCoreOccurrences().size() == 1,
          "sync System does not contain one SpatialCore");
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore{
      systemView.artifact().accCoreOccurrences().front()};
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  auto skeleton = take(
      test, buildModuleRootCirctSkeleton(*context, spatialCore, abi.abi()));
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableTokenConstantSyncProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  auto conformance =
      take(test, loom::hardware::test::specializeAndExportPortableProvider(
                     std::move(skeleton), abi, registry, externalContracts));
  return {std::move(conformance.systemVerilog),
          makeSystemConfiguration(test, fixture, spatialCore, abi.abi())};
}

std::string systemTestbench(const SystemConfiguration &configuration) {
  std::ostringstream output;
  output << R"sv(
module system_testbench;
  logic       clock;
  logic       reset;
  logic [7:0] input_0_data;
  logic       input_0_valid;
  logic       input_0_ready;
  logic [7:0] input_1_data;
  logic       input_1_valid;
  logic       input_1_ready;
  logic [7:0] output_0_data;
  logic       output_0_valid;
  logic       output_0_ready;
  logic [7:0] output_1_data;
  logic       output_1_valid;
  logic       output_1_ready;
)sv";
  output << "  logic [" << configuration.bitCount - 1 << ":0] "
         << configuration.portName << ";\n\n";
  output << R"sv(  loom_module dut(.*);

  always #5 clock = ~clock;

  task automatic check(input bit condition, input string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

  initial begin
    clock = 0;
    reset = 1;
    input_0_data = 8'h12;
    input_1_data = 8'h34;
    input_0_valid = 1;
    input_1_valid = 0;
    output_0_ready = 1;
    output_1_ready = 1;
)sv";
  output << "    " << configuration.portName << " = " << configuration.bitCount
         << "'b" << bitLiteral(configuration.payload, configuration.bitCount)
         << ";\n";
  output << R"sv(    #1;
    check(!input_0_ready && !input_1_ready &&
              !output_0_valid && !output_1_valid,
          "reset did not quiesce the elastic token boundary");
    repeat (2) @(posedge clock);
    @(negedge clock);
    reset = 0;
    #1;
    check(!input_0_ready && input_1_ready &&
              !output_0_valid && !output_1_valid,
          "sync consumed an incomplete tuple after reset");

    input_1_valid = 1;
    #1;
    check(input_0_ready && input_1_ready &&
              !output_0_valid && !output_1_valid,
          "sync did not atomically admit a complete tuple");
    @(posedge clock);
    #1;
    check(output_0_valid && output_1_valid &&
              output_0_data == 8'h12 && output_1_data == 8'h34,
          "common result slot did not publish the captured tuple");

    @(negedge clock);
    input_0_data = 8'h56;
    input_1_data = 8'h78;
    output_1_ready = 0;
    #1;
    check(!input_0_ready && !input_1_ready &&
              !output_0_valid && output_1_valid &&
              output_0_data == 8'h12 && output_1_data == 8'h34,
          "backpressure exposed a partial tuple or changed held payload");
    @(posedge clock);
    #1;
    check(!input_0_ready && !input_1_ready &&
              !output_0_valid && output_1_valid &&
              output_0_data == 8'h12 && output_1_data == 8'h34,
          "stalled tuple did not remain stable");

    @(negedge clock);
    output_1_ready = 1;
    #1;
    check(input_0_ready && input_1_ready &&
              output_0_valid && output_1_valid,
          "released tuple did not admit a bubble-free replacement");
    @(posedge clock);
    #1;
    check(output_0_valid && output_1_valid &&
              output_0_data == 8'h56 && output_1_data == 8'h78,
          "bubble-free replacement did not commit atomically");

    @(negedge clock);
    output_0_ready = 0;
    output_1_ready = 0;
    reset = 1;
    #1;
    check(!input_0_ready && !input_1_ready &&
              !output_0_valid && !output_1_valid,
          "asynchronous reset did not clear the held tuple");
    reset = 0;
    input_1_valid = 0;
    output_0_ready = 1;
    output_1_ready = 1;
    #1;
    check(!input_0_ready && input_1_ready &&
              !output_0_valid && !output_1_valid,
          "reset did not restore empty atomic-join state");
    $finish;
  end
endmodule
)sv";
  return output.str();
}

std::string yosysScript(llvm::StringRef top, llvm::StringRef source,
                        bool allowFlipFlops = false) {
  std::string script;
  llvm::raw_string_ostream output(script);
  const std::string forbidden =
      allowFlipFlops
          ? (top + "/t:$*latch* " + top + "/t:$mem*").str()
          : (top + "/t:$*ff* " + top + "/t:$*latch* " + top + "/t:$mem*").str();
  output << "read_verilog -sv " << source << '\n'
         << "hierarchy -check -top " << top << '\n'
         << "proc\ncheck -assert\n"
         << "select -assert-none " << forbidden << '\n'
         << "synth -top " << top << '\n'
         << "check -assert\n"
         << "select -assert-none " << forbidden << '\n';
  return output.str();
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

void expectTypedUnsupported(
    llvm::StringRef test, ::fabric::ImplementationFamilyId expectedFamily,
    llvm::Expected<FabricOperationProviderOutput> value) {
  require(test, !value, "unsupported resource contract specialized");
  bool classified = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() == expectedFamily &&
                     error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unsupported contract returned the wrong error class: " +
                       error.message());
      });
  require(test, classified,
          "resource contract lost its typed Unsupported classification");
}

bool sameCoverage(llvm::ArrayRef<FabricOperationProviderCoverage> lhs,
                  llvm::ArrayRef<FabricOperationProviderCoverage> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left.implementationFamily != right.implementationFamily ||
        left.recipes != right.recipes)
      return false;
  return true;
}

llvm::Expected<FabricOperationProviderOutput>
standInProvider(FabricOperationProviderRequest) {
  return FabricOperationProviderOutput{};
}

void registrationRollsBack() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registry.add({::fabric::ImplementationFamilyId::TokenSync,
                        BackendRecipeKey::PortableSystemVerilog,
                        {},
                        standInProvider}))
    fail(test, llvm::toString(std::move(error)));
  const auto before = registry.coverage();
  llvm::Error error = registerPortableTokenConstantSyncProviders(registry);
  require(test, static_cast<bool>(error),
          "duplicate package registration succeeded");
  llvm::consumeError(std::move(error));
  require(test, sameCoverage(before, registry.coverage()),
          "failed package registration changed provider coverage");
  const auto constant = llvm::find_if(before, [](const auto &entry) {
    return entry.implementationFamily ==
           ::fabric::ImplementationFamilyId::TokenConstant;
  });
  require(test, constant != before.end() && constant->recipes.empty(),
          "failed registration retained a partial constant provider");
}

void unsupportedContractRollsBack(const std::filesystem::path &root,
                                  ::fabric::ImplementationFamilyId family) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  const bool isConstant =
      family == ::fabric::ImplementationFamilyId::TokenConstant;
  FabricFixture fixture = [&] {
    if (isConstant)
      return makeFabric(test, store, "unsupported-token-constant", family,
                        ::fabric::PayloadCapacityParams{8},
                        ::dataflow::OperationSchemaId::DataflowConstant, {8},
                        {12}, ::fabric::loopCarryOperationResourceContract());
    return makeFabric(test, store, "unsupported-token-sync", family,
                      ::fabric::RoutedTokenParams{8, 2},
                      ::dataflow::OperationSchemaId::DataflowSync, {8, 8},
                      {8, 8}, ::fabric::loopCarryOperationResourceContract());
  }();
  FinalizedConfigurationABI abi = [&] {
    if (isConstant)
      return makeConstantAbi(test, store, fixture);
    return makeSyncAbi(test, store, fixture, 3);
  }();
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  LeafFixture leaf =
      makeLeaf(test, *context, fixture, abi.abi(),
               isConstant ? "unsupported_constant" : "unsupported_sync");
  const std::string before = moduleText(*leaf.module);
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableTokenConstantSyncProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {leaf.leaf, fixture.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fixture.physicalOccurrence,
       BackendRecipeKey::PortableSystemVerilog,
       {}}};
  expectTypedUnsupported(
      test, family,
      specializeFabricOperationLeaves(*leaf.module, abi, associations, recipes,
                                      registry, externalContracts));
  require(test, moduleText(*leaf.module) == before,
          "unsupported specialization partially mutated the caller module");
}

void emitArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture constant =
      makeFabric(test, store, "portable-token-constant",
                 fabric::ImplementationFamilyId::TokenConstant,
                 fabric::PayloadCapacityParams{8},
                 dataflow::OperationSchemaId::DataflowConstant, {8}, {12});
  FinalizedConfigurationABI constantAbi =
      makeConstantAbi(test, store, constant);
  FabricFixture sync = makeFabric(test, store, "portable-token-sync",
                                  fabric::ImplementationFamilyId::TokenSync,
                                  fabric::RoutedTokenParams{16, 3},
                                  dataflow::OperationSchemaId::DataflowSync,
                                  {8, 16, 24}, {12, 8, 24});
  FinalizedConfigurationABI syncAbi = makeSyncAbi(test, store, sync, 5);
  const LeafSyncCodes syncCodes = leafSyncCodes(test, sync, syncAbi, 5, 2, 7);
  FabricFixture singleton =
      makeFabric(test, store, "portable-token-sync-singleton",
                 fabric::ImplementationFamilyId::TokenSync,
                 fabric::RoutedTokenParams{16, 2},
                 dataflow::OperationSchemaId::DataflowSync, {8}, {12});
  FinalizedConfigurationABI singletonAbi =
      makeSyncAbi(test, store, singleton, 1);
  FabricFixture zeroPayload =
      makeFabric(test, store, "portable-token-sync-zero-payload",
                 fabric::ImplementationFamilyId::TokenSync,
                 fabric::RoutedTokenParams{8, 2},
                 dataflow::OperationSchemaId::DataflowSync, {0, 8}, {0, 12});
  FinalizedConfigurationABI zeroPayloadAbi =
      makeSyncAbi(test, store, zeroPayload, 1);
  const LeafSyncCodes zeroPayloadCodes =
      leafSyncCodes(test, zeroPayload, zeroPayloadAbi, 1, 2, 3);

  const std::string constantRtl =
      specializeLeaf(test, constant, constantAbi, "token_constant");
  const std::string syncRtl = specializeLeaf(test, sync, syncAbi, "token_sync");
  const std::string singletonRtl =
      specializeLeaf(test, singleton, singletonAbi, "token_sync_singleton");
  const std::string zeroPayloadRtl = specializeLeaf(
      test, zeroPayload, zeroPayloadAbi, "token_sync_zero_payload");
  require(test,
          constantRtl == specializeLeaf(test, constant, constantAbi,
                                        "token_constant") &&
              syncRtl == specializeLeaf(test, sync, syncAbi, "token_sync") &&
              singletonRtl == specializeLeaf(test, singleton, singletonAbi,
                                             "token_sync_singleton") &&
              zeroPayloadRtl == specializeLeaf(test, zeroPayload,
                                               zeroPayloadAbi,
                                               "token_sync_zero_payload"),
          "identical provider inputs produced different SystemVerilog");

  FabricFixture systemFixture =
      makeFabric(test, store, "portable-token-sync-system",
                 fabric::ImplementationFamilyId::TokenSync,
                 fabric::RoutedTokenParams{8, 2},
                 dataflow::OperationSchemaId::DataflowSync, {8, 8}, {8, 8});
  FinalizedConfigurationABI systemAbi =
      makeSyncAbi(test, store, systemFixture, 3);
  auto system = specializeSystem(test, systemFixture, systemAbi);
  auto systemAgain = specializeSystem(test, systemFixture, systemAbi);
  require(test,
          system.first == systemAgain.first &&
              system.second.portName == systemAgain.second.portName &&
              system.second.payload == systemAgain.second.payload,
          "identical common skeleton inputs were not deterministic");

  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"token_constant.sv", constantRtl},
           {"token_sync.sv", syncRtl},
           {"token_sync_singleton.sv", singletonRtl},
           {"token_sync_zero_payload.sv", zeroPayloadRtl},
           {"provider_testbench.sv",
            leafTestbench(syncCodes, zeroPayloadCodes)},
           {"token_sync_system.sv", system.first},
           {"system_testbench.sv", systemTestbench(system.second)},
           {"portable_token_constant.ys",
            yosysScript("token_constant", "token_constant.sv")},
           {"portable_token_sync.ys",
            yosysScript("token_sync", "token_sync.sv")},
           {"portable_token_sync_singleton.ys",
            yosysScript("token_sync_singleton", "token_sync_singleton.sv")},
           {"portable_token_sync_zero_payload.ys",
            yosysScript("token_sync_zero_payload",
                        "token_sync_zero_payload.sv")},
           {"portable_token_sync_system.ys",
            yosysScript("loom_module", "token_sync_system.sv", true)}}))
    fail(test, llvm::toString(std::move(error)));
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  emitArtifacts(root);
  registrationRollsBack();
  unsupportedContractRollsBack(root / "rollback_constant",
                               ::fabric::ImplementationFamilyId::TokenConstant);
  unsupportedContractRollsBack(root / "rollback_sync",
                               ::fabric::ImplementationFamilyId::TokenSync);
  return 0;
}

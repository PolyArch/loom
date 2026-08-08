#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/Providers/FixedVectorIntegerAddSub.h"

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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
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

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid portable fixed-vector add/sub input");
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

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
};

enum class FabricFixtureKind {
  Configured,
  Singleton,
  Undersized,
  UnsupportedContract,
};

enum class ConfigurationAbiKind {
  Complete,
  MissingSubtractI8,
  ExtraSemanticValue,
};

FabricFixture
makeFabric(llvm::StringRef test, const ArtifactStore &store,
           FabricFixtureKind kind = FabricFixtureKind::Configured) {
  llvm::StringRef sourceText;
  switch (kind) {
  case FabricFixtureKind::Configured:
  case FabricFixtureKind::UnsupportedContract:
    sourceText = R"mlir(
    module {
      fabric.module @fixed_vector_integer_add_sub(
          %a: !fabric.bits<128>, %b: !fabric.bits<128>)
          -> !fabric.bits<128> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<128>, %pb = %b : !fabric.bits<128>)
            -> !fabric.bits<128> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<128>,
               %fb = %pb : !fabric.bits<128>) -> !fabric.bits<128> {
            %value = fabric.op [@arith.addi, @arith.subi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorIntegerAddSub>,
               hw_params = {
                 element_widths = [8 : i32, 16 : i32],
                 max_payload_bits = 128 : i32}}
              : (!fabric.bits<128>, !fabric.bits<128>) -> !fabric.bits<128>
            fabric.yield %value : !fabric.bits<128>
          }
        }
        fabric.yield %pe : !fabric.bits<128>
      }
    }
  )mlir";
    break;
  case FabricFixtureKind::Singleton:
    sourceText = R"mlir(
    module {
      fabric.module @fixed_vector_integer_add(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>,
               %fb = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
            %value = fabric.op [@arith.addi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorIntegerAddSub>,
               hw_params = {
                 element_widths = [8 : i32],
                 max_payload_bits = 32 : i32}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %value : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir";
    break;
  case FabricFixtureKind::Undersized:
    sourceText = R"mlir(
    module {
      fabric.module @undersized_fixed_vector_integer_add(
          %a: !fabric.bits<64>, %b: !fabric.bits<64>)
          -> !fabric.bits<64> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<64>, %pb = %b : !fabric.bits<64>)
            -> !fabric.bits<64> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<64>,
               %fb = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
            %value = fabric.op [@arith.addi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorIntegerAddSub>,
               hw_params = {
                 element_widths = [8 : i32],
                 max_payload_bits = 128 : i32}}
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
            fabric.yield %value : !fabric.bits<64>
          }
        }
        fabric.yield %pe : !fabric.bits<64>
      }
    }
  )mlir";
    break;
  }

  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  const ::fabric::ResourceContract &resourceContract =
      kind == FabricFixtureKind::UnsupportedContract
          ? ::fabric::loopCarryOperationResourceContract()
          : ::fabric::oneCycleElasticOperationResourceContract();
  const std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(resourceContract));
  const std::vector<std::int8_t> signedContract(contract.begin(),
                                                contract.end());
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(
        ::fabric::kResourceContractRecordAttrName,
        mlir::DenseI8ArrayAttr::get(&fabricContext(), signedContract));
  });

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
      if (capability.implementationFamily !=
          ::fabric::ImplementationFamilyId::FixedVectorIntegerAddSub)
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), capability.occurrence, fuOccurrence));
      return FabricFixture{std::move(fabric), occurrence};
    }
  }
  fail(test, "Fabric fixture has no fixed-vector integer add/sub occurrence");
}

dataflow::CanonicalActorSchemaProjection
actor(std::initializer_list<std::int64_t> shape, unsigned elementWidth,
      dataflow::OperationSchemaId schema,
      mlir::arith::IntegerOverflowFlags flags =
          mlir::arith::IntegerOverflowFlags::none) {
  mlir::MLIRContext &context = fabricContext();
  mlir::Type element = mlir::IntegerType::get(&context, elementWidth);
  mlir::VectorType vector = mlir::VectorType::get(shape, element);
  mlir::FunctionType type =
      mlir::FunctionType::get(&context, {vector, vector}, {vector});
  return {schema, type, dataflow::IntegerOverflowPayload{flags}};
}

std::vector<std::uint8_t> configurationValue(
    llvm::StringRef test,
    const loom::fabric::ResolvedFabricOpCapabilityView &capability,
    const loom::fabric::FabricSemanticConfigFieldRef &field,
    const dataflow::CanonicalActorSchemaProjection &projection) {
  constexpr std::array<std::uint64_t, 2> operandPorts = {0, 1};
  constexpr std::array<std::uint64_t, 1> resultPorts = {0};
  const loom::CanonicalSemanticBytes encoded =
      take(test, capability.encodeSemanticConfiguration(
                     field, projection, 64, operandPorts, resultPorts));
  return std::vector<std::uint8_t>(encoded.bytes().begin(),
                                   encoded.bytes().end());
}

unsigned behaviorElementWidth(
    const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  return mlir::cast<mlir::IntegerType>(
             mlir::cast<mlir::VectorType>(
                 point.representativeActor.type.getInput(0))
                 .getElementType())
      .getWidth();
}

std::vector<FiniteCodebookEntry> completeEntries(
    llvm::StringRef test,
    const loom::fabric::ResolvedFabricOpCapabilityView &capability) {
  const auto domain =
      take(test, capability.resolveFiniteBehaviorDomain(fabricContext()));
  require(test, domain.size() == 4,
          "Fabric did not project the exact vector add/sub behavior domain");

  std::vector<FiniteCodebookEntry> entries;
  entries.reserve(domain.size());
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured Fabric behavior has no semantic value");
    const unsigned width = behaviorElementWidth(point);
    const bool add = point.representativeActor.schema ==
                     dataflow::OperationSchemaId::ArithAddI;
    const bool subtract = point.representativeActor.schema ==
                          dataflow::OperationSchemaId::ArithSubI;
    require(test, add || subtract,
            "Fabric projected an unexpected fixed-vector behavior");
    const std::uint8_t code =
        add ? (width == 8 ? 0x01 : 0x03) : (width == 8 ? 0x02 : 0x04);
    require(test, width == 8 || width == 16,
            "Fabric projected an unexpected element width");
    entries.push_back(
        {std::vector<std::uint8_t>(point.semanticConfiguration->bytes().begin(),
                                   point.semanticConfiguration->bytes().end()),
         {code}});
  }
  return entries;
}

FinalizedConfigurationABI makeConfigurationAbi(
    llvm::StringRef test, const ArtifactStore &store,
    const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  const auto *capability =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  if (capability->configurationFieldSchema.empty())
    return take(test, finalizeConfigurationABI(
                          ConfigurationABIDraft{fixture.fabric.reference(), {}},
                          store));
  require(test, capability->configurationFieldSchema.size() == 1,
          "vector add/sub fixture has an unexpected field count");
  std::vector<FiniteCodebookEntry> entries = completeEntries(test, *capability);
  if (kind == ConfigurationAbiKind::MissingSubtractI8) {
    const auto missing = llvm::find_if(entries, [](const auto &entry) {
      return entry.physicalCode == std::vector<std::uint8_t>{0x02};
    });
    require(test, missing != entries.end(),
            "i8 subtract behavior is absent from the Fabric domain");
    missing->semanticValue = {0xfd};
  }
  if (kind == ConfigurationAbiKind::ExtraSemanticValue)
    entries.push_back({{0xfe}, {0x05}});
  const auto inactive = llvm::find_if(entries, [](const auto &entry) {
    return entry.physicalCode == std::vector<std::uint8_t>{0x04};
  });
  require(test, inactive != entries.end(),
          "i16 subtract behavior is absent from the Fabric domain");
  const std::vector<std::uint8_t> inactiveValue = inactive->semanticValue;
  ConfigurationFieldEncoding field{
      capability->configurationFieldSchema.front(),
      FiniteCodebookEncoding{3, std::move(entries)},
      {{0, 0, 3}},
      inactiveValue};
  ProgrammingUnitDraft unit{{field.field.owner.catalog()}, 3, {field}};
  return take(test, finalizeConfigurationABI(
                        ConfigurationABIDraft{fixture.fabric.reference(),
                                              {std::move(unit)}},
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
      take(test, deriveFabricOperationLeafPorts(builder, *capability, abi));
  if (wrongConfigurationWidth) {
    require(test, ports.size() == 4,
            "configured vector add/sub leaf did not have four ports");
    ports[2].type = builder.getI2Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("fixed_vector_integer_add_sub"), ports);
  return SkeletonFixture{std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

std::string specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableFixedVectorIntegerAddSubProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.occurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  FabricOperationProviderOutput output =
      take(test, specializeFabricOperationLeaves(
                     *skeleton.module, fabric.fabric, abi, associations,
                     recipes, registry, externalContracts));
  require(test,
          output.payloads.empty() && output.activityPoints.empty() &&
              output.externalImplementationBindings.empty(),
          "portable vector add/sub emitted external implementation state");
  return take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
}

void configuredLaneBehaviorAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  const auto *capability =
      fabric.fabric.view().resolvedFabricOpCapability(fabric.occurrence);
  require(test, capability && capability->configurationFieldSchema.size() == 1,
          "configured vector capability has no exact field");
  const auto field = capability->configurationFieldSchema.front();
  require(test,
          configurationValue(
              test, *capability, field,
              actor({4}, 8, dataflow::OperationSchemaId::ArithAddI)) ==
              configurationValue(test, *capability, field,
                                 actor({2, 2}, 8,
                                       dataflow::OperationSchemaId::ArithAddI,
                                       mlir::arith::IntegerOverflowFlags::nsw)),
          "vector shape or overflow promise entered configuration bytes");
  require(test,
          configurationValue(
              test, *capability, field,
              actor({4}, 8, dataflow::OperationSchemaId::ArithAddI)) !=
              configurationValue(
                  test, *capability, field,
                  actor({2}, 16, dataflow::OperationSchemaId::ArithAddI)),
          "distinct lane boundaries collapsed to one configuration value");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 4 && ports.atInput(2).getName() == "config_0" &&
              ports.atInput(2).type ==
                  mlir::IntegerType::get(firstContext.get(), 3),
          "derived vector add/sub leaf ports are not canonical");
  const std::string firstRtl = specialize(test, first, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string secondRtl = specialize(test, second, fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical vector add/sub inputs produced different RTL");
  require(test,
          llvm::StringRef(firstRtl).contains("config_0") &&
              llvm::StringRef(firstRtl).contains(" + ") &&
              !llvm::StringRef(firstRtl).contains(" - "),
          "portable provider did not share configured lane datapaths");

  std::ofstream(root / "fixed_vector_integer_add_sub.sv") << firstRtl;
  std::ofstream(root / "testbench.sv") << R"sv(
module testbench;
  logic [127:0] data_input_0;
  logic [127:0] data_input_1;
  logic [2:0] config_0;
  logic [127:0] data_output_0;

  fixed_vector_integer_add_sub dut(.*);

  initial begin
    data_input_0 = 128'h00000000000000000000000000ff00ff;
    data_input_1 = 128'h00000000000000000000000000010001;
    config_0 = 3'b001;
    #1;
    if (data_output_0 !== 128'h0)
      $fatal(1, "i8 add leaked carry across lanes");

    config_0 = 3'b011;
    #1;
    if (data_output_0 !== 128'h00000000000000000000000001000100)
      $fatal(1, "i16 add did not retain an intra-lane carry");

    data_input_0 = 128'h0;
    data_input_1 = 128'h00000000000000000000000000010001;
    config_0 = 3'b010;
    #1;
    if (data_output_0 !== 128'h00000000000000000000000000ff00ff)
      $fatal(1, "i8 subtract leaked borrow across lanes");

    config_0 = 3'b100;
    #1;
    if (data_output_0 !== 128'h000000000000000000000000ffffffff)
      $fatal(1, "i16 subtract did not retain intra-lane borrow");

    config_0 = 3'b000;
    #1;
    if (data_output_0 !== 128'h000000000000000000000000ffffffff)
      $fatal(1, "unassigned code did not preserve inactive behavior");
    $finish;
  end
endmodule
)sv";
  std::ofstream(root / "portable_fixed_vector_integer_add_sub.ys") << R"ys(
read_verilog fixed_vector_integer_add_sub.sv
hierarchy -check -top fixed_vector_integer_add_sub
proc
opt
check
select -assert-none t:$sub
select -assert-count 48 t:$add
synth -top fixed_vector_integer_add_sub
check
stat
)ys";
}

void singletonNeedsNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store, FabricFixtureKind::Singleton);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  require(test, skeleton.leaf.getPortList().size() == 3,
          "singleton vector add retained a redundant selector");
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test,
          !llvm::StringRef(rtl).contains("config_0") &&
              llvm::StringRef(rtl).contains(" + ") &&
              !llvm::StringRef(rtl).contains(" - "),
          "singleton vector add did not lower directly");
}

void fixedSchemaStaysOutsideWidthValues() {
  const llvm::StringRef test = __func__;
  const ::fabric::FamilyCapabilityParams parameters =
      ::fabric::FixedVectorIntegerParams{
          ::fabric::IntegerWidthSet::get(
              {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16}),
          128};
  const dataflow::OperationSchemaId addSchema =
      dataflow::OperationSchemaId::ArithAddI;
  const dataflow::OperationSchemaId subtractSchema =
      dataflow::OperationSchemaId::ArithSubI;
  constexpr std::array<std::uint32_t, 2> inputWidths = {128, 128};
  constexpr std::array<std::uint32_t, 1> resultWidths = {128};
  const auto addRelation =
      take(test, ::fabric::resolveFabricOpSemanticFieldRelation(
                     ::fabric::ImplementationFamilyId::FixedVectorIntegerAddSub,
                     parameters, llvm::ArrayRef(addSchema), inputWidths,
                     resultWidths, fabricContext()));
  const auto subtractRelation =
      take(test, ::fabric::resolveFabricOpSemanticFieldRelation(
                     ::fabric::ImplementationFamilyId::FixedVectorIntegerAddSub,
                     parameters, llvm::ArrayRef(subtractSchema), inputWidths,
                     resultWidths, fabricContext()));
  require(test,
          addRelation.finiteBehaviorDomain().size() == 2 &&
              subtractRelation.finiteBehaviorDomain().size() == 2,
          "single-schema vector relations lost their width domain");
  constexpr std::array<std::uint64_t, 2> operandPorts = {0, 1};
  constexpr std::array<std::uint64_t, 1> resultPorts = {0};
  for (unsigned width : {8U, 16U}) {
    const loom::CanonicalSemanticBytes add = take(
        test,
        addRelation.projectSemanticValue(
            actor({static_cast<std::int64_t>(128 / width)}, width, addSchema),
            operandPorts, resultPorts));
    const loom::CanonicalSemanticBytes subtract =
        take(test, subtractRelation.projectSemanticValue(
                       actor({static_cast<std::int64_t>(128 / width)}, width,
                             subtractSchema),
                       operandPorts, resultPorts));
    require(test, add.bytes().equals(subtract.bytes()),
            "fixed operation schema entered width-selector bytes");
  }
}

void physicalCapacityNarrowsTheBehaviorDomain(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture configured = makeFabric(test, store);
  const auto *capability = configured.fabric.view().resolvedFabricOpCapability(
      configured.occurrence);
  require(test, capability != nullptr,
          "configured vector capability did not resolve");
  auto narrow = *capability;
  for (auto &port : narrow.physicalPorts)
    port.payloadWidthBits = 64;
  const auto domain =
      take(test, narrow.resolveFiniteBehaviorDomain(fabricContext()));
  require(test, domain.size() == 4,
          "64-bit physical ports lost a reachable add/sub behavior");
  for (const auto &point : domain) {
    const auto vector = mlir::cast<mlir::VectorType>(
        point.representativeActor.type.getInput(0));
    require(test,
            vector.getNumElements() * vector.getElementTypeBitWidth() == 64,
            "behavior witness exceeded the narrowed physical datapath");
  }

  FabricFixture fabric = makeFabric(test, store, FabricFixtureKind::Undersized);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  require(test, skeleton.leaf.getPortList().size() == 3,
          "narrow singleton add retained a redundant selector");
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test,
          !llvm::StringRef(rtl).contains("config_0") &&
              llvm::StringRef(rtl).contains(" + "),
          "narrow singleton add did not lower its reachable datapath");
}

void unsupportedResourceContractIsTransactional(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric =
      makeFabric(test, store, FabricFixtureKind::UnsupportedContract);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const std::string before = moduleText(*skeleton.module);

  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableFixedVectorIntegerAddSubProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.occurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  auto result = specializeFabricOperationLeaves(*skeleton.module, fabric.fabric,
                                                abi, associations, recipes,
                                                registry, externalContracts);
  require(test, !result, "unsupported resource contract specialized");
  bool classifiedUnsupported = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classifiedUnsupported =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::FixedVectorIntegerAddSub &&
            error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "resource contract returned the wrong error class: " +
                       error.message());
      });
  require(test, classifiedUnsupported,
          "resource contract lost its typed Unsupported classification");
  require(test, moduleText(*skeleton.module) == before,
          "unsupported resource contract mutated the common skeleton");
}

void malformedInputsFailClosed(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableFixedVectorIntegerAddSubProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};

  std::unique_ptr<mlir::MLIRContext> portContext = makeCirctContext();
  SkeletonFixture wrongPorts =
      makeSkeleton(test, *portContext, fabric, abi.abi(), true);
  const std::string portBefore = moduleText(*wrongPorts.module);
  const std::vector<FabricOperationLeafAssociation> portAssociations = {
      {wrongPorts.leaf, fabric.occurrence}};
  expectError(test,
              specializeFabricOperationLeaves(*wrongPorts.module, fabric.fabric,
                                              abi, portAssociations, recipes,
                                              registry, externalContracts),
              "leaf port");
  require(test, moduleText(*wrongPorts.module) == portBefore,
          "invalid vector leaf ports partially mutated the common skeleton");

  FinalizedConfigurationABI missing = makeConfigurationAbi(
      test, store, fabric, ConfigurationAbiKind::MissingSubtractI8);
  std::unique_ptr<mlir::MLIRContext> missingContext = makeCirctContext();
  SkeletonFixture missingCodebook =
      makeSkeleton(test, *missingContext, fabric, missing.abi());
  const std::string missingBefore = moduleText(*missingCodebook.module);
  const std::vector<FabricOperationLeafAssociation> missingAssociations = {
      {missingCodebook.leaf, fabric.occurrence}};
  expectError(test,
              specializeFabricOperationLeaves(
                  *missingCodebook.module, fabric.fabric, missing,
                  missingAssociations, recipes, registry, externalContracts),
              "semantic value");
  require(test, moduleText(*missingCodebook.module) == missingBefore,
          "incomplete vector codebook partially mutated the common skeleton");

  FinalizedConfigurationABI extra = makeConfigurationAbi(
      test, store, fabric, ConfigurationAbiKind::ExtraSemanticValue);
  std::unique_ptr<mlir::MLIRContext> extraContext = makeCirctContext();
  SkeletonFixture extraCodebook =
      makeSkeleton(test, *extraContext, fabric, extra.abi());
  const std::string extraBefore = moduleText(*extraCodebook.module);
  const std::vector<FabricOperationLeafAssociation> extraAssociations = {
      {extraCodebook.leaf, fabric.occurrence}};
  expectError(test,
              specializeFabricOperationLeaves(
                  *extraCodebook.module, fabric.fabric, extra,
                  extraAssociations, recipes, registry, externalContracts),
              "configuration domain");
  require(test, moduleText(*extraCodebook.module) == extraBefore,
          "overcomplete vector codebook partially mutated the common skeleton");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  configuredLaneBehaviorAndDeterminism(root);
  singletonNeedsNoSelector(root / "singleton");
  fixedSchemaStaysOutsideWidthValues();
  physicalCapacityNarrowsTheBehaviorDomain(root / "physical_capacity");
  unsupportedResourceContractIsTransactional(root / "resource_contract");
  malformedInputsFailClosed(root / "malformed");
  return 0;
}

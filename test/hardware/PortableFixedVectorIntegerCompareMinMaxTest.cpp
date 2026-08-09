#include "ConfigurationABI3TestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FixedVectorIntegerCompareMinMax.h"
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
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
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

using Schema = dataflow::OperationSchemaId;
using Predicate = mlir::arith::CmpIPredicate;

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
    fail(test, "accepted malformed fixed-vector compare/min-max input");
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
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

enum class FabricFixtureKind {
  Configured,
  SingletonUnsignedMinimum,
  UnsupportedContract,
};

enum class ConfigurationAbiKind {
  Complete,
  MissingSignedMinimumI8,
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
      fabric.module @fixed_vector_integer_compare_min_max(
          %a: !fabric.bits<48>, %b: !fabric.bits<48>)
          -> !fabric.bits<48> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<48>, %pb = %b : !fabric.bits<48>)
            -> !fabric.bits<48> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<48>,
               %fb = %pb : !fabric.bits<48>) -> !fabric.bits<48> {
            %value = fabric.op
              [@arith.cmpi, @arith.minsi, @arith.maxsi,
               @arith.minui, @arith.maxui] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorIntegerCompareMinMax>,
               hw_params = {
                 element_widths = [8 : i32, 16 : i32],
                 predicates = ["eq", "ne", "slt", "sle", "sgt", "sge",
                               "ult", "ule", "ugt", "uge"],
                 max_payload_bits = 48 : i32}}
              : (!fabric.bits<48>, !fabric.bits<48>) -> !fabric.bits<48>
            fabric.yield %value : !fabric.bits<48>
          }
        }
        fabric.yield %pe : !fabric.bits<48>
      }
    }
  )mlir";
    break;
  case FabricFixtureKind::SingletonUnsignedMinimum:
    sourceText = R"mlir(
    module {
      fabric.module @fixed_vector_integer_unsigned_minimum(
          %a: !fabric.bits<24>, %b: !fabric.bits<24>)
          -> !fabric.bits<24> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<24>, %pb = %b : !fabric.bits<24>)
            -> !fabric.bits<24> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<24>,
               %fb = %pb : !fabric.bits<24>) -> !fabric.bits<24> {
            %value = fabric.op [@arith.minui] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorIntegerCompareMinMax>,
               hw_params = {
                 element_widths = [8 : i32],
                 predicates = ["ult"],
                 max_payload_bits = 24 : i32}}
              : (!fabric.bits<24>, !fabric.bits<24>) -> !fabric.bits<24>
            fabric.yield %value : !fabric.bits<24>
          }
        }
        fabric.yield %pe : !fabric.bits<24>
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
          ::fabric::ImplementationFamilyId::FixedVectorIntegerCompareMinMax)
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
              "System has no physical fixed-vector compare/min-max occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no fixed-vector compare/min-max occurrence");
}

unsigned
elementWidth(const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  auto vector =
      mlir::cast<mlir::VectorType>(point.representativeActor.type.getInput(0));
  return mlir::cast<mlir::IntegerType>(vector.getElementType()).getWidth();
}

std::uint8_t
behaviorCode(llvm::StringRef test,
             const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  const unsigned width = elementWidth(point);
  require(test, width == 8 || width == 16,
          "Fabric relation exposed an unexpected element width");
  const std::uint8_t widthOffset = width == 8 ? 0 : 1;
  switch (point.representativeActor.schema) {
  case Schema::ArithCmpI: {
    const auto *payload = std::get_if<dataflow::IntegerComparePayload>(
        &point.representativeActor.payload);
    require(test, payload != nullptr,
            "Fabric comparison behavior has no typed predicate");
    return static_cast<std::uint8_t>(
        (width == 8 ? 1 : 11) + static_cast<std::uint8_t>(payload->predicate));
  }
  case Schema::ArithMinSI:
    return static_cast<std::uint8_t>(21 + widthOffset);
  case Schema::ArithMaxSI:
    return static_cast<std::uint8_t>(23 + widthOffset);
  case Schema::ArithMinUI:
    return static_cast<std::uint8_t>(25 + widthOffset);
  case Schema::ArithMaxUI:
    return static_cast<std::uint8_t>(27 + widthOffset);
  default:
    fail(test, "Fabric relation exposed a foreign behavior schema");
  }
}

std::vector<FiniteCodebookEntry> completeEntries(
    llvm::StringRef test,
    const loom::fabric::ResolvedFabricOpCapabilityView &capability) {
  auto relation =
      take(test, capability.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "fixed-vector compare/min-max relation is not finite");
  const auto domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 28,
          "Fabric did not project the exact compare/min-max behavior domain");

  std::array<std::array<bool, 10>, 2> comparisons{};
  std::array<std::array<bool, 4>, 2> minMax{};
  std::set<std::vector<std::uint8_t>> semanticValues;
  std::set<std::uint8_t> physicalCodes;
  std::vector<FiniteCodebookEntry> entries;
  entries.reserve(domain.size());
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured Fabric behavior has no semantic value");
    require(test,
            point.operandPorts == std::vector<std::uint64_t>({0, 1}) &&
                point.resultPorts == std::vector<std::uint64_t>({0}),
            "Fabric behavior witness changed physical port correspondence");
    auto input = mlir::cast<mlir::VectorType>(
        point.representativeActor.type.getInput(0));
    auto result = mlir::cast<mlir::VectorType>(
        point.representativeActor.type.getResult(0));
    const unsigned width = elementWidth(point);
    const std::size_t widthIndex = width == 8 ? 0 : 1;
    require(test,
            input.getRank() == 1 && !input.isScalable() &&
                input.getNumElements() == 48 / width &&
                point.representativeActor.type.getInput(1) == input &&
                result.getShape() == input.getShape(),
            "Fabric behavior witness has the wrong vector geometry");
    if (point.representativeActor.schema == Schema::ArithCmpI) {
      const auto *payload = std::get_if<dataflow::IntegerComparePayload>(
          &point.representativeActor.payload);
      require(test, payload != nullptr,
              "Fabric comparison behavior has no predicate");
      const std::size_t predicate =
          static_cast<std::size_t>(payload->predicate);
      require(test, predicate < comparisons[widthIndex].size(),
              "Fabric comparison predicate is outside the closed domain");
      comparisons[widthIndex][predicate] = true;
      require(test, result.getElementType().isInteger(1),
              "Fabric comparison result is not one bit per lane");
    } else {
      const std::size_t role =
          point.representativeActor.schema == Schema::ArithMinSI   ? 0
          : point.representativeActor.schema == Schema::ArithMaxSI ? 1
          : point.representativeActor.schema == Schema::ArithMinUI ? 2
                                                                   : 3;
      minMax[widthIndex][role] = true;
      require(test, result == input,
              "Fabric min/max result changed active element geometry");
    }

    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    const std::uint8_t code = behaviorCode(test, point);
    require(test, semanticValues.insert(semantic).second,
            "Fabric relation contains duplicate semantic values");
    require(test, physicalCodes.insert(code).second,
            "test ABI assigned a physical code twice");
    entries.push_back({std::move(semantic), {code}});
  }
  for (const auto &width : comparisons)
    for (bool present : width)
      require(test, present,
              "Fabric relation omitted an integer comparison predicate");
  for (const auto &width : minMax)
    for (bool present : width)
      require(test, present, "Fabric relation omitted a min/max behavior");
  return entries;
}

dataflow::CanonicalActorSchemaProjection
actor(unsigned width, Schema schema,
      std::optional<Predicate> predicate = std::nullopt) {
  mlir::MLIRContext &context = fabricContext();
  mlir::Type element = mlir::IntegerType::get(&context, width);
  mlir::Type values = mlir::VectorType::get({48 / width}, element);
  mlir::Type result =
      schema == Schema::ArithCmpI
          ? mlir::VectorType::get({48 / width},
                                  mlir::IntegerType::get(&context, 1))
          : values;
  if (schema == Schema::ArithCmpI)
    return {schema,
            mlir::FunctionType::get(&context, {values, values}, {result}),
            dataflow::IntegerComparePayload{predicate.value_or(Predicate::eq)}};
  return {schema, mlir::FunctionType::get(&context, {values, values}, {result}),
          dataflow::NoPayload{}};
}

ConfigurationABIDraft makeConfigurationAbiDraft(
    llvm::StringRef test, const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
  const auto *capability =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  if (capability->configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, capability->configurationFieldSchema.size() == 1,
          "compare/min-max fixture has an unexpected field count");
  const auto fieldReference = capability->configurationFieldSchema.front();
  std::vector<FiniteCodebookEntry> entries = completeEntries(test, *capability);
  if (kind == ConfigurationAbiKind::MissingSignedMinimumI8) {
    const auto missing = llvm::find_if(entries, [](const auto &entry) {
      return entry.physicalCode == std::vector<std::uint8_t>{21};
    });
    require(test, missing != entries.end(),
            "signed i8 minimum is absent from the Fabric domain");
    missing->semanticValue = {0xfd};
  }
  if (kind == ConfigurationAbiKind::ExtraSemanticValue)
    entries.push_back({{0xfe}, {0x1f}});
  const auto inactive = llvm::find_if(entries, [](const auto &entry) {
    return entry.physicalCode == std::vector<std::uint8_t>{28};
  });
  require(test, inactive != entries.end(),
          "unsigned i16 maximum is absent from the Fabric domain");
  const std::vector<std::uint8_t> inactiveValue = inactive->semanticValue;
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence, fieldReference.ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, FiniteCodebookEncoding{5, std::move(entries)},
      inactiveValue};
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, {std::move(field)}));
}

FinalizedConfigurationABI makeConfigurationAbi(
    llvm::StringRef test, const ArtifactStore &store,
    const FabricFixture &fixture,
    ConfigurationAbiKind kind = ConfigurationAbiKind::Complete) {
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
      take(test, deriveFabricOperationLeafPorts(
                     builder, fabric.physicalOccurrence, *capability, abi));
  if (wrongConfigurationWidth) {
    require(test, ports.size() == 4,
            "configured compare/min-max leaf did not have four ports");
    ports[2].type = builder.getI4Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("fixed_vector_integer_compare_min_max"), ports);
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
          registerPortableFixedVectorIntegerCompareMinMaxProvider(registry))
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
          "portable compare/min-max provider emitted implementation metadata");
  return std::move(conformance.systemVerilog);
}

void configuredSemanticsAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  const auto *capability =
      fabric.fabric.view().resolvedFabricOpCapability(fabric.occurrence);
  require(test, capability && capability->configurationFieldSchema.size() == 1,
          "configured capability has no exact semantic field");
  completeEntries(test, *capability);
  const auto field = capability->configurationFieldSchema.front();
  expectError(test,
              capability->encodeSemanticConfiguration(
                  field, actor(32, Schema::ArithCmpI, Predicate::slt), 64,
                  std::array<std::uint64_t, 2>{0, 1},
                  std::array<std::uint64_t, 1>{0}),
              "not admitted");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 4 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atInput(2).getName() == "config_0" &&
              ports.atInput(2).type ==
                  mlir::IntegerType::get(firstContext.get(), 5) &&
              ports.atOutput(0).getName() == "data_output_0",
          "derived compare/min-max leaf ports are not canonical");
  const std::string firstRtl = specialize(test, first, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string secondRtl = specialize(test, second, fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical compare/min-max inputs produced different SystemVerilog");
  require(test,
          llvm::StringRef(firstRtl).contains("config_0") &&
              llvm::StringRef(firstRtl).contains("$signed"),
          "portable provider omitted configured signed comparison logic");

  const std::string testbench = R"sv(
module testbench;
  logic [47:0] data_input_0;
  logic [47:0] data_input_1;
  logic [4:0] config_0;
  logic [47:0] data_output_0;

  fixed_vector_integer_compare_min_max dut(.*);

  task automatic expect_value(
      input logic [4:0] code,
      input logic [47:0] expected,
      input string message);
    config_0 = code;
    #1;
    if (data_output_0 !== expected) $fatal(1, "%s", message);
  endtask

  initial begin
    data_input_0 = 48'hfe0100ff7f80;
    data_input_1 = 48'hfd01ff00807f;
    expect_value(5'd1,  48'h000000000010, "i8 equality lanes failed");
    expect_value(5'd2,  48'h00000000002f, "i8 inequality lanes failed");
    expect_value(5'd3,  48'h000000000005, "i8 signed less-than lanes failed");
    expect_value(5'd4,  48'h000000000015, "i8 signed less-equal lanes failed");
    expect_value(5'd5,  48'h00000000002a, "i8 signed greater-than lanes failed");
    expect_value(5'd6,  48'h00000000003a, "i8 signed greater-equal lanes failed");
    expect_value(5'd7,  48'h00000000000a, "i8 unsigned less-than lanes failed");
    expect_value(5'd8,  48'h00000000001a, "i8 unsigned less-equal lanes failed");
    expect_value(5'd9,  48'h000000000025, "i8 unsigned greater-than lanes failed");
    expect_value(5'd10, 48'h000000000035, "i8 unsigned greater-equal lanes failed");
    expect_value(5'd21, 48'hfd01ffff8080, "i8 signed minimum lanes failed");
    expect_value(5'd23, 48'hfe0100007f7f, "i8 signed maximum lanes failed");
    expect_value(5'd25, 48'hfd0100007f7f, "i8 unsigned minimum lanes failed");
    expect_value(5'd27, 48'hfe01ffff8080, "i8 unsigned maximum lanes failed");

    data_input_0 = 48'hfffe7fff8000;
    data_input_1 = 48'hfffdffff0001;
    expect_value(5'd13, 48'h000000000001, "i16 signed comparison lanes failed");
    expect_value(5'd19, 48'h000000000005, "i16 unsigned comparison lanes failed");
    expect_value(5'd22, 48'hfffdffff8000, "i16 signed minimum lanes failed");
    expect_value(5'd24, 48'hfffe7fff0001, "i16 signed maximum lanes failed");
    expect_value(5'd26, 48'hfffd7fff0001, "i16 unsigned minimum lanes failed");
    expect_value(5'd28, 48'hfffeffff8000, "i16 unsigned maximum lanes failed");
    expect_value(5'd0,  48'hfffeffff8000,
                 "unassigned code did not preserve inactive behavior");
    $finish;
  end
endmodule
)sv";
  const std::string yosysScript = R"ys(
read_verilog -sv fixed_vector_integer_compare_min_max.sv
hierarchy -check -top fixed_vector_integer_compare_min_max
proc
opt
check
synth -top fixed_vector_integer_compare_min_max
check
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"fixed_vector_integer_compare_min_max.sv", firstRtl},
           {"testbench.sv", testbench},
           {"portable_fixed_vector_integer_compare_min_max.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void singletonUnsignedMinimumNeedsNoSelector(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric =
      makeFabric(test, store, FabricFixtureKind::SingletonUnsignedMinimum);
  const auto *capability =
      fabric.fabric.view().resolvedFabricOpCapability(fabric.occurrence);
  require(test, capability && capability->configurationFieldSchema.empty(),
          "singleton unsigned minimum retained a semantic field");
  const auto relation =
      take(test, capability->resolveSemanticFieldRelation(fabricContext()));
  require(
      test,
      relation.kind() == ::fabric::FabricOpSemanticFieldRelationKind::None &&
          relation.finiteBehaviorDomain().size() == 1 &&
          !relation.finiteBehaviorDomain().front().semanticConfiguration,
      "singleton unsigned minimum did not collapse to a fieldless relation");
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  require(test, skeleton.leaf.getPortList().size() == 3,
          "singleton unsigned minimum retained a selector port");
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test, !llvm::StringRef(rtl).contains("config_0"),
          "singleton unsigned minimum emitted selector logic");
}

void malformedInputsFailClosed(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);

  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture wrongPorts =
      makeSkeleton(test, *context, fabric, abi.abi(), true);
  const std::string before = moduleText(*wrongPorts.module);
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableFixedVectorIntegerCompareMinMaxProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {wrongPorts.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectError(test,
              specializeFabricOperationLeaves(*wrongPorts.module, abi,
                                              associations, recipes, registry,
                                              externalContracts),
              "leaf port");
  require(test, moduleText(*wrongPorts.module) == before,
          "malformed leaf ports partially mutated the common skeleton");

  expectError(
      test,
      finalizeConfigurationABI(
          makeConfigurationAbiDraft(
              test, fabric, ConfigurationAbiKind::MissingSignedMinimumI8),
          store),
      "semantic");
  expectError(test,
              finalizeConfigurationABI(
                  makeConfigurationAbiDraft(
                      test, fabric, ConfigurationAbiKind::ExtraSemanticValue),
                  store),
              "semantic");
}

void unsupportedResourceContractIsTypedAndTransactional(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric =
      makeFabric(test, store, FabricFixtureKind::UnsupportedContract);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());

  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableFixedVectorIntegerCompareMinMaxProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, fabric.physicalOccurrence}}};
  auto result = loom::hardware::test::specializeAndExportPortableProvider(
      std::move(module), abi, registry, externalContracts);
  require(test, !result, "unsupported resource contract specialized");
  bool classifiedUnsupported = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classifiedUnsupported =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::
                    FixedVectorIntegerCompareMinMax &&
            error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "resource contract returned the wrong error class: " +
                       error.message());
      });
  require(test, classifiedUnsupported,
          "resource contract lost its typed Unsupported classification");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  configuredSemanticsAndDeterminism(root);
  singletonUnsignedMinimumNeedsNoSelector(root / "singleton");
  malformedInputsFailClosed(root / "malformed");
  unsupportedResourceContractIsTypedAndTransactional(root /
                                                     "unsupported_contract");
  return 0;
}

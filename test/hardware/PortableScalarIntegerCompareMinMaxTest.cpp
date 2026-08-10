#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/ScalarIntegerCompareMinMax.h"
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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

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

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test,
         "accepted invalid portable compare/min/max input expected to report " +
             expected.str());
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
  ConfiguredCompareMinMax,
  SingletonEquality,
  SingletonSignedLessThanWidths,
  SingletonSignedGreaterThanWidths,
  SingletonUnsignedMinimum,
  NarrowUnsignedMinimum,
  UnsupportedContract,
};

enum class ConfigurationAbiKind {
  Complete,
  MissingSignedMinimum,
  ExtraSemanticValue,
};

FabricFixture makeFabric(
    llvm::StringRef test, const ArtifactStore &store,
    FabricFixtureKind kind = FabricFixtureKind::ConfiguredCompareMinMax) {
  const llvm::StringRef sourceText =
      (kind == FabricFixtureKind::ConfiguredCompareMinMax ||
       kind == FabricFixtureKind::UnsupportedContract)
          ? R"mlir(
    module {
      fabric.module @integer_compare_min_max(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>, %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %value = fabric.op
              [@arith.cmpi, @arith.minsi, @arith.maxui] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerCompareMinMax>,
               hw_params = {
                 integer_widths = [8 : i32, 32 : i32],
                 predicates = ["eq", "slt", "ugt"]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %value : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir"
      : kind == FabricFixtureKind::SingletonEquality                ? R"mlir(
    module {
      fabric.module @integer_equality(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>, %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %value = fabric.op [@arith.cmpi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerCompareMinMax>,
               hw_params = {
                 integer_widths = [8 : i32, 32 : i32],
                 predicates = ["eq"]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %value : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir"
      : kind == FabricFixtureKind::SingletonSignedLessThanWidths    ? R"mlir(
    module {
      fabric.module @signed_less_than_widths(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>, %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %value = fabric.op [@arith.cmpi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerCompareMinMax>,
               hw_params = {
                 integer_widths = [8 : i32, 32 : i32],
                 predicates = ["slt"]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %value : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir"
      : kind == FabricFixtureKind::SingletonSignedGreaterThanWidths ? R"mlir(
    module {
      fabric.module @signed_greater_than_widths(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>, %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %value = fabric.op [@arith.cmpi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerCompareMinMax>,
               hw_params = {
                 integer_widths = [8 : i32, 32 : i32],
                 predicates = ["sgt"]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %value : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir"
      : kind == FabricFixtureKind::SingletonUnsignedMinimum         ? R"mlir(
    module {
      fabric.module @unsigned_minimum(
          %a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>, %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %value = fabric.op [@arith.minui] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerCompareMinMax>,
               hw_params = {
                 integer_widths = [8 : i32, 32 : i32],
                 predicates = ["eq", "slt", "ult"]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %value : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir"
                                                                    : R"mlir(
    module {
      fabric.module @narrow_unsigned_minimum(
          %a: !fabric.bits<8>, %b: !fabric.bits<8>)
          -> !fabric.bits<8> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<8>, %pb = %b : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<8>, %fb = %pb : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value = fabric.op [@arith.minui] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerCompareMinMax>,
               hw_params = {
                 integer_widths = [8 : i32, 32 : i32],
                 predicates = ["ult"]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value : !fabric.bits<8>
          }
        }
        fabric.yield %pe : !fabric.bits<8>
      }
    }
  )mlir";

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
          ::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax)
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
              "System has no physical scalar integer compare/min/max "
              "occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no scalar integer compare/min/max occurrence");
}

dataflow::CanonicalActorSchemaProjection
actor(unsigned width, dataflow::OperationSchemaId schema,
      std::optional<mlir::arith::CmpIPredicate> predicate = std::nullopt) {
  mlir::MLIRContext &context = fabricContext();
  mlir::Type operand = mlir::IntegerType::get(&context, width);
  const bool comparison = schema == dataflow::OperationSchemaId::ArithCmpI;
  mlir::FunctionType type = mlir::FunctionType::get(
      &context, {operand, operand},
      {comparison ? mlir::IntegerType::get(&context, 1) : operand});
  if (comparison)
    return {schema, type,
            dataflow::IntegerComparePayload{
                predicate.value_or(mlir::arith::CmpIPredicate::eq)}};
  return {schema, type, dataflow::NoPayload{}};
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

std::vector<FiniteCodebookEntry> completeEntries(
    llvm::StringRef test,
    const loom::fabric::ResolvedFabricOpCapabilityView &capability) {
  auto relation =
      take(test, capability.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "compare/min/max semantic field relation is not finite");
  const auto &domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 7,
          "Fabric did not project the exact compare/min/max behavior domain");

  std::vector<FiniteCodebookEntry> entries;
  entries.reserve(domain.size());
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured Fabric behavior has no semantic value");
    const auto width = mlir::cast<mlir::IntegerType>(
                           point.representativeActor.type.getInput(0))
                           .getWidth();
    std::uint8_t code = 0;
    switch (point.representativeActor.schema) {
    case dataflow::OperationSchemaId::ArithCmpI: {
      const auto *payload = std::get_if<dataflow::IntegerComparePayload>(
          &point.representativeActor.payload);
      require(test, payload != nullptr,
              "Fabric comparison behavior has no predicate");
      if (payload->predicate == mlir::arith::CmpIPredicate::eq)
        code = 0x01;
      else if (payload->predicate == mlir::arith::CmpIPredicate::slt &&
               width == 8)
        code = 0x02;
      else if (payload->predicate == mlir::arith::CmpIPredicate::slt &&
               width == 32)
        code = 0x03;
      else if (payload->predicate == mlir::arith::CmpIPredicate::ugt)
        code = 0x04;
      break;
    }
    case dataflow::OperationSchemaId::ArithMinSI:
      code = width == 8 ? 0x05 : width == 32 ? 0x06 : 0;
      break;
    case dataflow::OperationSchemaId::ArithMaxUI:
      code = 0x07;
      break;
    default:
      break;
    }
    require(test, code != 0,
            "Fabric projected an unexpected compare/min/max behavior");
    entries.push_back(
        {std::vector<std::uint8_t>(point.semanticConfiguration->bytes().begin(),
                                   point.semanticConfiguration->bytes().end()),
         {code}});
  }
  return entries;
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
          "compare/min/max fixture has an unexpected field count");
  const auto fieldReference = capability->configurationFieldSchema.front();
  std::vector<FiniteCodebookEntry> entries = completeEntries(test, *capability);
  if (kind == ConfigurationAbiKind::MissingSignedMinimum) {
    const auto missing = llvm::find_if(entries, [](const auto &entry) {
      return entry.physicalCode == std::vector<std::uint8_t>{0x05};
    });
    require(test, missing != entries.end(),
            "signed minimum behavior is absent from the Fabric domain");
    missing->semanticValue = {0xfd};
  }
  if (kind == ConfigurationAbiKind::ExtraSemanticValue)
    entries.push_back({{0xfe}, {0x00}});
  const auto inactive = llvm::find_if(entries, [](const auto &entry) {
    return entry.physicalCode == std::vector<std::uint8_t>{0x07};
  });
  require(test, inactive != entries.end(),
          "unsigned maximum behavior is absent from the Fabric domain");
  const std::vector<std::uint8_t> inactiveValue = inactive->semanticValue;
  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence, fieldReference.ordinal));
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
            "configured compare/min/max leaf did not have four ports");
    ports[2].type = builder.getI2Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("scalar_integer_compare_min_max"), ports);
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
          registerPortableScalarIntegerCompareMinMaxProvider(registry))
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
          "portable compare/min/max emitted external implementation state");
  return std::move(conformance.systemVerilog);
}

void compactSemanticFieldAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  const auto *capability =
      fabric.fabric.view().resolvedFabricOpCapability(fabric.occurrence);
  require(test, capability && capability->configurationFieldSchema.size() == 1,
          "configured capability has no exact field");
  const auto field = capability->configurationFieldSchema.front();
  require(
      test,
      configurationValue(test, *capability, field,
                         actor(8, dataflow::OperationSchemaId::ArithCmpI,
                               mlir::arith::CmpIPredicate::eq)) ==
          configurationValue(test, *capability, field,
                             actor(32, dataflow::OperationSchemaId::ArithCmpI,
                                   mlir::arith::CmpIPredicate::eq)),
      "width-independent equality did not collapse to one field value");
  require(
      test,
      configurationValue(test, *capability, field,
                         actor(8, dataflow::OperationSchemaId::ArithCmpI,
                               mlir::arith::CmpIPredicate::slt)) !=
          configurationValue(test, *capability, field,
                             actor(32, dataflow::OperationSchemaId::ArithCmpI,
                                   mlir::arith::CmpIPredicate::slt)),
      "signed comparison widths collapsed despite distinct sign bits");
  expectError(test,
              capability->encodeSemanticConfiguration(
                  field,
                  actor(8, dataflow::OperationSchemaId::ArithCmpI,
                        mlir::arith::CmpIPredicate::ne),
                  64, std::array<std::uint64_t, 2>{0, 1},
                  std::array<std::uint64_t, 1>{0}),
              "predicate");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fabric, abi.abi());
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 4 && ports.atInput(2).getName() == "config_0" &&
              ports.atInput(2).type ==
                  mlir::IntegerType::get(firstContext.get(), 3),
          "derived compare/min/max leaf ports are not canonical");
  const std::string firstRtl = specialize(test, first, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi());
  const std::string secondRtl = specialize(test, second, fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical compare/min/max inputs produced different RTL");
  require(test,
          llvm::StringRef(firstRtl).contains("config_0") &&
              llvm::StringRef(firstRtl).contains("$signed"),
          "portable provider omitted configured signed comparison logic");

  const std::string testbench = R"sv(
module testbench;
  logic [31:0] data_input_0;
  logic [31:0] data_input_1;
  logic [2:0] config_0;
  logic [31:0] data_output_0;

  scalar_integer_compare_min_max dut(.*);

  initial begin
    data_input_0 = 32'h00000080;
    data_input_1 = 32'h00000080;
    config_0 = 3'b001;
    #1;
    if (data_output_0 !== 32'h1) $fatal(1, "equality failed");

    data_input_1 = 32'h0000007f;
    config_0 = 3'b010;
    #1;
    if (data_output_0 !== 32'h1) $fatal(1, "signed i8 compare failed");

    config_0 = 3'b011;
    #1;
    if (data_output_0 !== 32'h0) $fatal(1, "signed i32 compare failed");

    config_0 = 3'b100;
    #1;
    if (data_output_0 !== 32'h1) $fatal(1, "unsigned compare failed");

    config_0 = 3'b101;
    #1;
    if (data_output_0 !== 32'h00000080) $fatal(1, "signed i8 min failed");

    config_0 = 3'b110;
    #1;
    if (data_output_0 !== 32'h0000007f) $fatal(1, "signed i32 min failed");

    config_0 = 3'b111;
    #1;
    if (data_output_0 !== 32'h00000080) $fatal(1, "unsigned max failed");

    config_0 = 3'b000;
    #1;
    if (data_output_0 !== 32'h00000080)
      $fatal(1, "unassigned code did not preserve inactive behavior");
    $finish;
  end
endmodule
)sv";
  const std::string yosysScript = R"ys(
read_verilog scalar_integer_compare_min_max.sv
hierarchy -check -top scalar_integer_compare_min_max
proc
opt
check
synth -top scalar_integer_compare_min_max
check
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"scalar_integer_compare_min_max.sv", firstRtl},
           {"testbench.sv", testbench},
           {"portable_scalar_integer_compare_min_max.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void singletonEqualityNeedsNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric =
      makeFabric(test, store, FabricFixtureKind::SingletonEquality);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  require(test, skeleton.leaf.getPortList().size() == 3,
          "singleton equality retained a redundant selector");
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test,
          !llvm::StringRef(rtl).contains("config_0") &&
              llvm::StringRef(rtl).contains("=="),
          "singleton equality did not lower directly");
}

void fixedFactsStayOutsideSemanticValues(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());

  FabricFixture less =
      makeFabric(test, store, FabricFixtureKind::SingletonSignedLessThanWidths);
  FabricFixture greater = makeFabric(
      test, store, FabricFixtureKind::SingletonSignedGreaterThanWidths);
  const auto *lessCapability =
      less.fabric.view().resolvedFabricOpCapability(less.occurrence);
  const auto *greaterCapability =
      greater.fabric.view().resolvedFabricOpCapability(greater.occurrence);
  require(test,
          lessCapability && greaterCapability &&
              lessCapability->configurationFieldSchema.size() == 1 &&
              greaterCapability->configurationFieldSchema.size() == 1,
          "signed width domains did not expose one semantic field");
  const auto lessRelation =
      take(test, lessCapability->resolveSemanticFieldRelation(fabricContext()));
  const auto greaterRelation = take(
      test, greaterCapability->resolveSemanticFieldRelation(fabricContext()));
  const auto lessDomain = lessRelation.finiteBehaviorDomain();
  const auto greaterDomain = greaterRelation.finiteBehaviorDomain();
  require(test, lessDomain.size() == 2 && greaterDomain.size() == 2,
          "signed width domains did not retain both configured widths");
  for (std::size_t index = 0; index < lessDomain.size(); ++index) {
    require(test,
            lessDomain[index].semanticConfiguration &&
                greaterDomain[index].semanticConfiguration &&
                lessDomain[index].semanticConfiguration->bytes().equals(
                    greaterDomain[index].semanticConfiguration->bytes()),
            "fixed comparison predicate entered width-selector bytes");
  }

  FabricFixture unsignedMinimum =
      makeFabric(test, store, FabricFixtureKind::SingletonUnsignedMinimum);
  const auto *unsignedCapability =
      unsignedMinimum.fabric.view().resolvedFabricOpCapability(
          unsignedMinimum.occurrence);
  require(test,
          unsignedCapability &&
              unsignedCapability->configurationFieldSchema.empty(),
          "irrelevant compare predicates created an unsigned minimum field");
  const auto unsignedRelation = take(
      test, unsignedCapability->resolveSemanticFieldRelation(fabricContext()));
  const auto unsignedDomain = unsignedRelation.finiteBehaviorDomain();
  require(test,
          unsignedDomain.size() == 1 &&
              !unsignedDomain.front().semanticConfiguration,
          "configuration-free unsigned minimum retained a semantic value");
}

void physicalPortsNarrowTheBehaviorDomain(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture narrow =
      makeFabric(test, store, FabricFixtureKind::NarrowUnsignedMinimum);
  const auto *capability =
      narrow.fabric.view().resolvedFabricOpCapability(narrow.occurrence);
  require(test, capability != nullptr,
          "narrow unsigned-min capability did not resolve");
  const auto relation =
      take(test, capability->resolveSemanticFieldRelation(fabricContext()));
  const auto domain = relation.finiteBehaviorDomain();
  require(test,
          domain.size() == 1 && !domain.front().semanticConfiguration &&
              mlir::cast<mlir::IntegerType>(
                  domain.front().representativeActor.type.getResult(0))
                      .getWidth() == 8,
          "8-bit physical ports did not retain only the reachable behavior");

  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, narrow);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, narrow, abi.abi());
  const circt::hw::ModulePortInfo ports(skeleton.leaf.getPortList());
  require(
      test,
      ports.size() == 3 &&
          ports.atInput(0).type == mlir::IntegerType::get(context.get(), 8) &&
          ports.atInput(1).type == mlir::IntegerType::get(context.get(), 8) &&
          ports.atOutput(0).type == mlir::IntegerType::get(context.get(), 8),
      "narrow unsigned-min leaf retained a selector or wide port");
  const std::string rtl = specialize(test, skeleton, narrow, abi);
  require(test, !llvm::StringRef(rtl).contains("config_0"),
          "narrow unsigned-min RTL retained a redundant selector");
}

void malformedInputsFailClosed(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableScalarIntegerCompareMinMaxProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, BackendRecipeKey::PortableSystemVerilog, {}}};

  std::unique_ptr<mlir::MLIRContext> portContext = makeCirctContext();
  SkeletonFixture wrongPorts =
      makeSkeleton(test, *portContext, fabric, abi.abi(), true);
  const std::string portBefore = moduleText(*wrongPorts.module);
  const std::vector<FabricOperationLeafAssociation> portAssociations = {
      {wrongPorts.leaf, fabric.physicalOccurrence}};
  expectError(test,
              specializeFabricOperationLeaves(*wrongPorts.module, abi,
                                              portAssociations, recipes,
                                              registry, externalContracts),
              "leaf port");
  require(test, moduleText(*wrongPorts.module) == portBefore,
          "invalid leaf ports partially mutated the common skeleton");

  expectError(test,
              finalizeConfigurationABI(
                  makeConfigurationAbiDraft(
                      test, fabric, ConfigurationAbiKind::MissingSignedMinimum),
                  store),
              "semantic");

  expectError(test,
              finalizeConfigurationABI(
                  makeConfigurationAbiDraft(
                      test, fabric, ConfigurationAbiKind::ExtraSemanticValue),
                  store),
              "semantic");
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
          registerPortableScalarIntegerCompareMinMaxProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  auto result =
      specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                      recipes, registry, externalContracts);
  require(test, !result, "unsupported resource contract specialized");
  bool classifiedUnsupported = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classifiedUnsupported =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax &&
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

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  compactSemanticFieldAndDeterminism(root);
  singletonEqualityNeedsNoSelector(root / "singleton");
  fixedFactsStayOutsideSemanticValues(root / "fixed_facts");
  physicalPortsNarrowTheBehaviorDomain(root / "narrowed_ports");
  malformedInputsFailClosed(root / "malformed");
  unsupportedResourceContractIsTransactional(root / "resource_contract");
  return 0;
}

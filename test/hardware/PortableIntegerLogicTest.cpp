#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/IntegerLogic.h"
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
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

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value,
                    llvm::StringRef expected) {
  require(test, !value, "accepted malformed integer logic input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

enum class FamilyKind { Scalar, FixedVector };
enum class BehaviorSet { AllLogic, EquivalentOr };
enum class AbiKind { Complete, MissingXor, ExtraSemanticValue };
enum class LogicBehavior { And, Or, Xor };

::fabric::ImplementationFamilyId familyId(FamilyKind family) {
  return family == FamilyKind::Scalar
             ? ::fabric::ImplementationFamilyId::ScalarIntegerLogic
             : ::fabric::ImplementationFamilyId::FixedVectorIntegerLogic;
}

LogicBehavior behaviorOf(llvm::StringRef test,
                         ::dataflow::OperationSchemaId schema) {
  using Schema = ::dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::ArithAndI:
    return LogicBehavior::And;
  case Schema::ArithOrI:
  case Schema::LLVMOrDisjoint:
    return LogicBehavior::Or;
  case Schema::ArithXOrI:
    return LogicBehavior::Xor;
  default:
    fail(test, "Fabric returned a non-logic behavior witness");
  }
}

std::uint8_t physicalCode(LogicBehavior behavior) {
  switch (behavior) {
  case LogicBehavior::And:
    return 0x05;
  case LogicBehavior::Or:
    return 0x03;
  case LogicBehavior::Xor:
    return 0x06;
  }
  llvm_unreachable("unknown logic behavior");
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
  FamilyKind family;
};

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         FamilyKind family,
                         BehaviorSet behaviors = BehaviorSet::AllLogic,
                         bool unsupportedContract = false) {
  llvm::StringRef sourceText;
  if (family == FamilyKind::Scalar && behaviors == BehaviorSet::AllLogic) {
    sourceText = R"mlir(
    module {
      fabric.module @scalar_integer_logic(
          %a: !fabric.bits<64>, %b: !fabric.bits<64>)
          -> !fabric.bits<64> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<64>, %pb = %b : !fabric.bits<64>)
            -> !fabric.bits<64> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<64>,
               %fb = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
            %value = fabric.op
              [@arith.andi, @arith.ori, @arith.xori, @llvm.or] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerLogic>,
               hw_params = {integer_widths = [1 : i32, 8 : i32, 16 : i32,
                                              32 : i32, 64 : i32]}}
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
            fabric.yield %value : !fabric.bits<64>
          }
        }
        fabric.yield %pe : !fabric.bits<64>
      }
    }
  )mlir";
  } else if (family == FamilyKind::Scalar) {
    sourceText = R"mlir(
    module {
      fabric.module @scalar_integer_or(
          %a: !fabric.bits<64>, %b: !fabric.bits<64>)
          -> !fabric.bits<64> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<64>, %pb = %b : !fabric.bits<64>)
            -> !fabric.bits<64> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<64>,
               %fb = %pb : !fabric.bits<64>) -> !fabric.bits<64> {
            %value = fabric.op [@arith.ori, @llvm.or] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerLogic>,
               hw_params = {integer_widths = [1 : i32, 8 : i32, 16 : i32,
                                              32 : i32, 64 : i32]}}
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
            fabric.yield %value : !fabric.bits<64>
          }
        }
        fabric.yield %pe : !fabric.bits<64>
      }
    }
  )mlir";
  } else if (behaviors == BehaviorSet::AllLogic) {
    sourceText = R"mlir(
    module {
      fabric.module @fixed_vector_integer_logic(
          %a: !fabric.bits<128>, %b: !fabric.bits<128>)
          -> !fabric.bits<128> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<128>, %pb = %b : !fabric.bits<128>)
            -> !fabric.bits<128> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<128>,
               %fb = %pb : !fabric.bits<128>) -> !fabric.bits<128> {
            %value = fabric.op
              [@arith.andi, @arith.ori, @arith.xori, @llvm.or] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorIntegerLogic>,
               hw_params = {
                 element_widths = [1 : i32, 8 : i32, 16 : i32, 32 : i32,
                                   64 : i32],
                 max_payload_bits = 128 : i32}}
              : (!fabric.bits<128>, !fabric.bits<128>) -> !fabric.bits<128>
            fabric.yield %value : !fabric.bits<128>
          }
        }
        fabric.yield %pe : !fabric.bits<128>
      }
    }
  )mlir";
  } else {
    sourceText = R"mlir(
    module {
      fabric.module @fixed_vector_integer_or(
          %a: !fabric.bits<128>, %b: !fabric.bits<128>)
          -> !fabric.bits<128> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<128>, %pb = %b : !fabric.bits<128>)
            -> !fabric.bits<128> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<128>,
               %fb = %pb : !fabric.bits<128>) -> !fabric.bits<128> {
            %value = fabric.op [@arith.ori, @llvm.or] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorIntegerLogic>,
               hw_params = {
                 element_widths = [1 : i32, 8 : i32, 16 : i32, 32 : i32,
                                   64 : i32],
                 max_payload_bits = 128 : i32}}
              : (!fabric.bits<128>, !fabric.bits<128>) -> !fabric.bits<128>
            fabric.yield %value : !fabric.bits<128>
          }
        }
        fabric.yield %pe : !fabric.bits<128>
      }
    }
  )mlir";
  }

  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  const ::fabric::ResourceContract &contract =
      unsupportedContract
          ? ::fabric::loopGateOperationResourceContract()
          : ::fabric::oneCycleElasticOperationResourceContract();
  const std::vector<std::uint8_t> encoded =
      take(test, ::fabric::encodeResourceContractRecord(contract));
  const std::vector<std::int8_t> signedContract(encoded.begin(), encoded.end());
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
      if (capability.implementationFamily != familyId(family))
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
              "System has no physical integer logic occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence, family};
    }
  }
  fail(test, "Fabric fixture has no integer logic occurrence");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const ArtifactStore &store,
                          const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "logic capability has more than one configuration field");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "configured logic relation is not finite");
  const auto domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 3,
          "full logic capability did not resolve three physical behaviors");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactiveValue;
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured logic behavior has no semantic value");
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    const LogicBehavior behavior =
        behaviorOf(test, point.representativeActor.schema);
    if (behavior == LogicBehavior::Or)
      inactiveValue = semantic;
    if (kind == AbiKind::MissingXor && behavior == LogicBehavior::Xor)
      semantic = {0xfd};
    entries.push_back({std::move(semantic), {physicalCode(behavior)}});
  }
  require(test, !inactiveValue.empty(), "logic domain has no OR behavior");
  if (kind == AbiKind::ExtraSemanticValue)
    entries.push_back({{0xfe}, {0x07}});

  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     resolved.configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      std::move(physicalField), FiniteCodebookEncoding{3, std::move(entries)},
      std::move(inactiveValue)};
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, {std::move(field)}));
}

FinalizedConfigurationABI
makeConfigurationAbi(llvm::StringRef test, const ArtifactStore &store,
                     const FabricFixture &fixture,
                     AbiKind kind = AbiKind::Complete) {
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
                             const ConfigurationABI &abi, llvm::StringRef name,
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
    const auto config = llvm::find_if(ports, [](const auto &port) {
      return port.getName().starts_with("config_");
    });
    require(test, config != ports.end(), "logic leaf has no selector port");
    config->type = builder.getIntegerType(2);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(name), ports);
  return SkeletonFixture{std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

struct LogicCounts final {
  unsigned andCount = 0;
  unsigned orCount = 0;
  unsigned xorCount = 0;
  unsigned muxCount = 0;
};

struct SpecializedRtl final {
  std::string text;
  LogicCounts counts;
};

SpecializedRtl specialize(llvm::StringRef test, SkeletonFixture &skeleton,
                          const FabricFixture &fabric,
                          const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableIntegerLogicProviders(registry))
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
          "portable logic provider emitted external implementation state");

  const llvm::StringRef text(conformance.systemVerilog);
  LogicCounts counts;
  counts.andCount = text.count(" & ");
  counts.orCount = text.count(" | ");
  counts.xorCount = text.count(" ^ ");
  counts.muxCount = text.count(" ? ");
  return {std::move(conformance.systemVerilog), counts};
}

void checkFullDomain(llvm::StringRef test, const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  const auto &descriptor =
      ::fabric::implementationFamily(resolved.implementationFamily);
  require(test,
          resolved.implementationFamily == familyId(fixture.family) &&
              resolved.enabledOperationSchemas.size() ==
                  descriptor.admittedSchemas.size() &&
              resolved.configurationFieldSchema.size() == 1,
          "sealed logic capability does not match its generated family");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "configured logic relation is not finite");
  const auto domain = relation.finiteBehaviorDomain();
  require(test, domain.size() == 3,
          "Fabric did not collapse logic witnesses to three behaviors");
  const auto orBehavior = llvm::find_if(domain, [](const auto &point) {
    return point.representativeActor.schema ==
           ::dataflow::OperationSchemaId::ArithOrI;
  });
  require(test, orBehavior != domain.end() && orBehavior->semanticConfiguration,
          "aliased OR schemas did not collapse to one semantic behavior");

  auto distinctSelection = resolved;
  distinctSelection.enabledOperationSchemas = {
      ::dataflow::OperationSchemaId::ArithAndI,
      ::dataflow::OperationSchemaId::ArithOrI};
  auto distinctRelation = take(
      test, distinctSelection.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          distinctRelation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "distinct logic relation is not finite");
  const auto distinctDomain = distinctRelation.finiteBehaviorDomain();
  require(test, distinctDomain.size() == 2,
          "distinct logic selection did not expose two behaviors");
}

SpecializedRtl emitDeterministically(llvm::StringRef test,
                                     const FabricFixture &fabric,
                                     const FinalizedConfigurationABI &abi,
                                     llvm::StringRef moduleName) {
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first =
      makeSkeleton(test, *firstContext, fabric, abi.abi(), moduleName);
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(test,
          ports.size() == 4 && ports.atInput(0).getName() == "data_input_0" &&
              ports.atInput(1).getName() == "data_input_1" &&
              ports.atInput(2).getName() == "config_0" &&
              mlir::cast<mlir::IntegerType>(ports.atInput(2).type).getWidth() ==
                  3 &&
              ports.atOutput(0).getName() == "data_output_0",
          "derived configured logic ports are not canonical");
  SpecializedRtl firstRtl = specialize(test, first, fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi(), moduleName);
  SpecializedRtl secondRtl = specialize(test, second, fabric, abi);
  require(test, firstRtl.text == secondRtl.text,
          "identical logic inputs produced different SystemVerilog");
  require(
      test,
      firstRtl.counts.andCount == 1 && firstRtl.counts.orCount == 1 &&
          firstRtl.counts.xorCount == 1 && firstRtl.counts.muxCount == 2,
      "configured logic did not materialize one shared datapath per behavior");
  const llvm::StringRef rtl(firstRtl.text);
  require(test,
          rtl.contains("config_0") && !rtl.contains("config_1") &&
              !rtl.contains_insensitive("poison") &&
              !rtl.contains_insensitive("trap") &&
              !rtl.contains_insensitive("stall") && !rtl.contains("valid_") &&
              !rtl.contains("ready_"),
          "logic RTL added semantic checking or protocol sidebands");
  return firstRtl;
}

void equivalentOrUsesOneUnconfiguredDatapath(llvm::StringRef test,
                                             const ArtifactStore &store,
                                             FamilyKind family) {
  FabricFixture fabric =
      makeFabric(test, store, family, BehaviorSet::EquivalentOr);
  const auto &resolved = capability(test, fabric);
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  const auto domain = relation.finiteBehaviorDomain();
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::None &&
              resolved.configurationFieldSchema.empty() && domain.size() == 1 &&
              !domain.front().semanticConfiguration,
          "equivalent OR schemas did not collapse before RTL lowering");
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  const llvm::StringRef name = family == FamilyKind::Scalar
                                   ? "scalar_integer_or_shared"
                                   : "fixed_vector_integer_or_shared";
  SkeletonFixture skeleton =
      makeSkeleton(test, *context, fabric, abi.abi(), name);
  require(test, skeleton.leaf.getPortList().size() == 3,
          "equivalent OR capability retained a selector");
  SpecializedRtl rtl = specialize(test, skeleton, fabric, abi);
  require(test,
          rtl.counts.andCount == 0 && rtl.counts.orCount == 1 &&
              rtl.counts.xorCount == 0 && rtl.counts.muxCount == 0 &&
              !llvm::StringRef(rtl.text).contains("config_") &&
              !llvm::StringRef(rtl.text).contains_insensitive("poison"),
          "ordinary and disjoint OR did not share one direct physical OR");
}

void writeToolInputs(const std::filesystem::path &root,
                     const SpecializedRtl &scalar,
                     const SpecializedRtl &vector) {
  const llvm::StringRef test = __func__;
  const std::string testbench = R"sv(
module testbench;
  logic [63:0] scalar_input_0;
  logic [63:0] scalar_input_1;
  logic [2:0] scalar_config;
  logic [63:0] scalar_output;
  logic [127:0] vector_input_0;
  logic [127:0] vector_input_1;
  logic [2:0] vector_config;
  logic [127:0] vector_output;
  logic [63:0] expected_scalar;
  logic [127:0] expected_vector;
  integer sample;

  scalar_integer_logic scalar_dut(
    .data_input_0(scalar_input_0), .data_input_1(scalar_input_1),
    .config_0(scalar_config), .data_output_0(scalar_output));
  fixed_vector_integer_logic vector_dut(
    .data_input_0(vector_input_0), .data_input_1(vector_input_1),
    .config_0(vector_config), .data_output_0(vector_output));

  initial begin
    for (sample = 0; sample < 64; sample = sample + 1) begin
      scalar_input_0 = 64'hf0c35aa596693cc3 ^ sample;
      scalar_input_1 = 64'h3cf00ff0c33ca55a ^ (sample << 7);
      vector_input_0 =
          128'hff00f00faaaa55550123456789abcdef ^ sample;
      vector_input_1 =
          128'h0ff00ff03333ccccfedcba9876543210 ^ (sample << 11);

      scalar_config = 3'b101;
      vector_config = 3'b101;
      expected_scalar = scalar_input_0 & scalar_input_1;
      expected_vector = vector_input_0 & vector_input_1;
      #1;
      if (scalar_output !== expected_scalar || vector_output !== expected_vector)
        $fatal(1, "AND oracle mismatch");

      scalar_config = 3'b011;
      vector_config = 3'b011;
      expected_scalar = scalar_input_0 | scalar_input_1;
      expected_vector = vector_input_0 | vector_input_1;
      #1;
      if (scalar_output !== expected_scalar || vector_output !== expected_vector)
        $fatal(1, "shared OR oracle mismatch");

      scalar_config = 3'b110;
      vector_config = 3'b110;
      expected_scalar = scalar_input_0 ^ scalar_input_1;
      expected_vector = vector_input_0 ^ vector_input_1;
      #1;
      if (scalar_output !== expected_scalar || vector_output !== expected_vector)
        $fatal(1, "XOR oracle mismatch");

      scalar_config = 3'b000;
      vector_config = 3'b000;
      expected_scalar = scalar_input_0 | scalar_input_1;
      expected_vector = vector_input_0 | vector_input_1;
      #1;
      if (scalar_output !== expected_scalar || vector_output !== expected_vector)
        $fatal(1, "inactive OR fallback mismatch");
    end
    $finish;
  end
endmodule
)sv";
  const std::string synthesisTop = R"sv(
module integer_logic_synthesis_top(
  input logic [63:0] scalar_input_0,
  input logic [63:0] scalar_input_1,
  input logic [2:0] scalar_config,
  input logic [127:0] vector_input_0,
  input logic [127:0] vector_input_1,
  input logic [2:0] vector_config,
  output logic [63:0] scalar_output,
  output logic [127:0] vector_output);
  scalar_integer_logic scalar_logic(
    .data_input_0(scalar_input_0), .data_input_1(scalar_input_1),
    .config_0(scalar_config), .data_output_0(scalar_output));
  fixed_vector_integer_logic vector_logic(
    .data_input_0(vector_input_0), .data_input_1(vector_input_1),
    .config_0(vector_config), .data_output_0(vector_output));
endmodule
)sv";
  const std::string yosysScript = R"ys(
read_verilog -sv scalar_integer_logic.sv fixed_vector_integer_logic.sv synthesis_top.sv
hierarchy -check -top integer_logic_synthesis_top
proc
opt
check -assert
select -assert-count 2 t:$and
select -assert-count 2 t:$or
select -assert-count 2 t:$xor
synth -top integer_logic_synthesis_top
check -assert
stat
)ys";
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "tool-artifacts",
          {{"scalar_integer_logic.sv", scalar.text},
           {"fixed_vector_integer_logic.sv", vector.text},
           {"testbench.sv", testbench},
           {"synthesis_top.sv", synthesisTop},
           {"portable_integer_logic.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void registrationIsPortableOnly() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableIntegerLogicProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  const auto coverage = registry.coverage();
  require(test, coverage.size() == ::fabric::implementationFamilyCount(),
          "provider coverage lost the generated family closure");
  for (FamilyKind family : {FamilyKind::Scalar, FamilyKind::FixedVector}) {
    const auto found = llvm::find_if(coverage, [&](const auto &entry) {
      return entry.implementationFamily == familyId(family);
    });
    require(test,
            found != coverage.end() &&
                found->recipes ==
                    std::vector<BackendRecipeKey>{
                        BackendRecipeKey::PortableSystemVerilog},
            "integer logic provider registered a native recipe alias");
  }
  for (const auto &entry : coverage)
    if (entry.implementationFamily != familyId(FamilyKind::Scalar) &&
        entry.implementationFamily != familyId(FamilyKind::FixedVector))
      require(test, entry.recipes.empty(),
              "integer logic registration covered an unrelated family");
}

template <typename T>
void expectTypedUnsupported(llvm::StringRef test, llvm::Expected<T> result,
                            ::fabric::ImplementationFamilyId expectedFamily,
                            BackendRecipeKey expectedRecipe,
                            llvm::StringRef description) {
  require(test, !result, description);
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() == expectedFamily &&
                     error.recipe() == expectedRecipe;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, description.str() +
                       " returned the wrong error class: " + error.message());
      });
  require(test, classified,
          description.str() + " lost typed Unsupported classification");
}

template <typename T>
void expectInvalid(llvm::StringRef test, llvm::Expected<T> result,
                   llvm::StringRef expected) {
  require(test, !result, "provider accepted malformed integer logic input");
  bool invalid = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &) {
        fail(test, "malformed integer logic input became typed Unsupported");
      },
      [&](const llvm::ErrorInfoBase &error) {
        require(test, llvm::StringRef(error.message()).contains(expected),
                error.message());
        invalid = true;
      });
  require(test, invalid, "malformed integer logic input lost its error");
}

llvm::Expected<FabricOperationProviderOutput>
trySpecializeRecipe(SkeletonFixture &skeleton, const FabricFixture &fabric,
                    const FinalizedConfigurationABI &abi,
                    BackendRecipeKey recipe,
                    const FabricOperationProviderRegistry &registry) {
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, recipe, {}}};
  return specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                         recipes, registry, externalContracts);
}

llvm::Expected<loom::hardware::test::PortableProviderConformance>
tryPortableSpecialize(SkeletonFixture &skeleton, const FabricFixture &fabric,
                      const FinalizedConfigurationABI &abi,
                      const FabricOperationProviderRegistry &registry) {
  ExternalImplementationContractCatalog externalContracts;
  return loom::hardware::test::specializeAndExportPortableProvider(
      {std::move(skeleton.module),
       {{skeleton.leaf, fabric.physicalOccurrence}}},
      abi, registry, externalContracts);
}

void failuresAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableIntegerLogicProviders(registry))
    fail(test, llvm::toString(std::move(error)));

  for (FamilyKind family : {FamilyKind::Scalar, FamilyKind::FixedVector}) {
    FabricFixture unsupported =
        makeFabric(test, store, family, BehaviorSet::AllLogic, true);
    FinalizedConfigurationABI unsupportedAbi =
        makeConfigurationAbi(test, store, unsupported);
    std::unique_ptr<mlir::MLIRContext> unsupportedContext = makeCirctContext();
    SkeletonFixture unsupportedSkeleton =
        makeSkeleton(test, *unsupportedContext, unsupported,
                     unsupportedAbi.abi(), "unsupported_integer_logic");
    expectTypedUnsupported(
        test,
        tryPortableSpecialize(unsupportedSkeleton, unsupported, unsupportedAbi,
                              registry),
        familyId(family), BackendRecipeKey::PortableSystemVerilog,
        "unsupported resource contract");

    FabricFixture valid = makeFabric(test, store, family);
    FinalizedConfigurationABI validAbi =
        makeConfigurationAbi(test, store, valid);
    std::unique_ptr<mlir::MLIRContext> leafContext = makeCirctContext();
    SkeletonFixture malformedLeaf =
        makeSkeleton(test, *leafContext, valid, validAbi.abi(),
                     "malformed_integer_logic", true);
    expectInvalid(
        test, tryPortableSpecialize(malformedLeaf, valid, validAbi, registry),
        "leaf port");

    for (AbiKind kind : {AbiKind::MissingXor, AbiKind::ExtraSemanticValue}) {
      expectRejected(
          test,
          finalizeConfigurationABI(
              makeConfigurationAbiDraft(test, store, valid, kind), store),
          "semantic");
    }

    constexpr std::array nativeRecipes = {
        BackendRecipeKey::SynopsysDesignWare, BackendRecipeKey::CadenceChipWare,
        BackendRecipeKey::AmdXilinx, BackendRecipeKey::IntelAltera};
    for (BackendRecipeKey recipe : nativeRecipes) {
      std::unique_ptr<mlir::MLIRContext> nativeContext = makeCirctContext();
      SkeletonFixture native = makeSkeleton(test, *nativeContext, valid,
                                            validAbi.abi(), "native_logic");
      const std::string before = moduleText(*native.module);
      expectTypedUnsupported(
          test, trySpecializeRecipe(native, valid, validAbi, recipe, registry),
          familyId(family), recipe, "native recipe");
      require(test, moduleText(*native.module) == before,
              "unsupported native recipe mutated the caller module");
    }
  }
}

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture scalar = makeFabric(test, store, FamilyKind::Scalar);
  FabricFixture vector = makeFabric(test, store, FamilyKind::FixedVector);
  checkFullDomain(test, scalar);
  checkFullDomain(test, vector);
  FinalizedConfigurationABI scalarAbi =
      makeConfigurationAbi(test, store, scalar);
  FinalizedConfigurationABI vectorAbi =
      makeConfigurationAbi(test, store, vector);
  SpecializedRtl scalarRtl =
      emitDeterministically(test, scalar, scalarAbi, "scalar_integer_logic");
  SpecializedRtl vectorRtl = emitDeterministically(
      test, vector, vectorAbi, "fixed_vector_integer_logic");
  equivalentOrUsesOneUnconfiguredDatapath(test, store, FamilyKind::Scalar);
  equivalentOrUsesOneUnconfiguredDatapath(test, store, FamilyKind::FixedVector);
  writeToolInputs(root, scalarRtl, vectorRtl);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  registrationIsPortableOnly();
  configuredBehaviorAndArtifacts(root);
  failuresAreTransactional(root / "failures");
  return 0;
}

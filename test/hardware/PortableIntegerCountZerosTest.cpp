#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/IntegerCountZeros.h"
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
  require(test, !value, "accepted malformed count-zero input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

enum class FamilyKind { Scalar, FixedVector };
enum class FixtureKind { Configured, Singleton, UnsupportedContract };
enum class AbiKind { Complete, MissingBehavior, ExtraBehavior };
enum class Direction { Leading, Trailing };

struct Behavior final {
  Direction direction;
  unsigned width;
};

::fabric::ImplementationFamilyId familyId(FamilyKind family) {
  return family == FamilyKind::Scalar
             ? ::fabric::ImplementationFamilyId::ScalarIntegerCountZeros
             : ::fabric::ImplementationFamilyId::FixedVectorIntegerCountZeros;
}

unsigned wideWidth(FamilyKind family) {
  return family == FamilyKind::Scalar ? 32 : 16;
}

Direction directionOf(llvm::StringRef test,
                      ::dataflow::OperationSchemaId schema) {
  using Schema = ::dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::MathCountLeadingZeros:
  case Schema::LLVMCountLeadingZeros:
    return Direction::Leading;
  case Schema::MathCountTrailingZeros:
  case Schema::LLVMCountTrailingZeros:
    return Direction::Trailing;
  default:
    fail(test, "Fabric returned a non-count-zero behavior witness");
  }
}

Behavior
behaviorOf(llvm::StringRef test, FamilyKind family,
           const ::dataflow::CanonicalActorSchemaProjection &representative) {
  require(test,
          representative.type.getNumInputs() == 1 &&
              representative.type.getNumResults() == 1,
          "count-zero behavior has the wrong arity");
  unsigned width = 0;
  if (family == FamilyKind::Scalar) {
    auto input =
        llvm::dyn_cast<mlir::IntegerType>(representative.type.getInput(0));
    require(test,
            input && input.isSignless() &&
                representative.type.getResult(0) == input,
            "scalar count-zero behavior is not uniform signless integer");
    width = input.getWidth();
  } else {
    auto input =
        llvm::dyn_cast<mlir::VectorType>(representative.type.getInput(0));
    require(test,
            input && !input.isScalable() && input.getNumElements() != 0 &&
                representative.type.getResult(0) == input,
            "vector count-zero behavior is not a uniform fixed vector");
    auto element = llvm::dyn_cast<mlir::IntegerType>(input.getElementType());
    require(test, element && element.isSignless(),
            "vector count-zero element is not signless integer");
    width = element.getWidth();
  }
  return Behavior{directionOf(test, representative.schema), width};
}

std::uint8_t physicalCode(const Behavior &behavior) {
  const bool narrow = behavior.width == 8;
  if (behavior.direction == Direction::Leading)
    return narrow ? 0x01 : 0x03;
  return narrow ? 0x06 : 0x04;
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
                         FixtureKind kind = FixtureKind::Configured) {
  llvm::StringRef sourceText;
  if (family == FamilyKind::Scalar && kind == FixtureKind::Singleton) {
    sourceText = R"mlir(
    module {
      fabric.module @scalar_count_trailing_zeros(
          %value: !fabric.bits<67>) -> !fabric.bits<67> {
        %pe = fabric.pe [spatial]
            (%input = %value : !fabric.bits<67>) -> !fabric.bits<67> {
          %fu = fabric.fu
              (%operand = %input : !fabric.bits<67>) -> !fabric.bits<67> {
            %count = fabric.op [@math.cttz] (%operand)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerCountZeros>,
               hw_params = {integer_widths = [8 : i32]}}
              : (!fabric.bits<67>) -> !fabric.bits<67>
            fabric.yield %count : !fabric.bits<67>
          }
        }
        fabric.yield %pe : !fabric.bits<67>
      }
    }
  )mlir";
  } else if (family == FamilyKind::Scalar) {
    sourceText = R"mlir(
    module {
      fabric.module @scalar_integer_count_zeros(
          %value: !fabric.bits<67>) -> !fabric.bits<67> {
        %pe = fabric.pe [spatial]
            (%input = %value : !fabric.bits<67>) -> !fabric.bits<67> {
          %fu = fabric.fu
              (%operand = %input : !fabric.bits<67>) -> !fabric.bits<67> {
            %count = fabric.op
              [@math.ctlz, @math.cttz, @llvm.intr.ctlz, @llvm.intr.cttz]
              (%operand)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerCountZeros>,
               hw_params = {integer_widths = [8 : i32, 32 : i32]}}
              : (!fabric.bits<67>) -> !fabric.bits<67>
            fabric.yield %count : !fabric.bits<67>
          }
        }
        fabric.yield %pe : !fabric.bits<67>
      }
    }
  )mlir";
  } else if (kind == FixtureKind::Singleton) {
    sourceText = R"mlir(
    module {
      fabric.module @vector_count_leading_zeros(
          %value: !fabric.bits<120>) -> !fabric.bits<120> {
        %pe = fabric.pe [spatial]
            (%input = %value : !fabric.bits<120>) -> !fabric.bits<120> {
          %fu = fabric.fu
              (%operand = %input : !fabric.bits<120>) -> !fabric.bits<120> {
            %count = fabric.op [@math.ctlz] (%operand)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorIntegerCountZeros>,
               hw_params = {
                 element_widths = [16 : i32],
                 max_payload_bits = 120 : i32}}
              : (!fabric.bits<120>) -> !fabric.bits<120>
            fabric.yield %count : !fabric.bits<120>
          }
        }
        fabric.yield %pe : !fabric.bits<120>
      }
    }
  )mlir";
  } else {
    sourceText = R"mlir(
    module {
      fabric.module @fixed_vector_integer_count_zeros(
          %value: !fabric.bits<120>) -> !fabric.bits<120> {
        %pe = fabric.pe [spatial]
            (%input = %value : !fabric.bits<120>) -> !fabric.bits<120> {
          %fu = fabric.fu
              (%operand = %input : !fabric.bits<120>) -> !fabric.bits<120> {
            %count = fabric.op
              [@math.ctlz, @math.cttz, @llvm.intr.ctlz, @llvm.intr.cttz]
              (%operand)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorIntegerCountZeros>,
               hw_params = {
                 element_widths = [8 : i32, 16 : i32],
                 max_payload_bits = 120 : i32}}
              : (!fabric.bits<120>) -> !fabric.bits<120>
            fabric.yield %count : !fabric.bits<120>
          }
        }
        fabric.yield %pe : !fabric.bits<120>
      }
    }
  )mlir";
  }

  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, static_cast<bool>(source),
          "could not parse count-zero Fabric fixture");
  const ::fabric::ResourceContract &contract =
      kind == FixtureKind::UnsupportedContract
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
  require(test, static_cast<bool>(root), "Fabric fixture has no module root");
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
              "System has no physical count-zero occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence, family};
    }
  }
  fail(test, "Fabric fixture has no count-zero capability");
}

const loom::fabric::ResolvedFabricOpCapabilityView &
capability(llvm::StringRef test, const FabricFixture &fixture) {
  const auto *result =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, result != nullptr, "Fabric capability did not resolve");
  return *result;
}

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto &resolved = capability(test, fixture);
  if (resolved.configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, resolved.configurationFieldSchema.size() == 1,
          "count-zero capability has an unexpected field count");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 4,
          "configured count-zero relation is not the expected finite domain");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactiveValue;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "configured count-zero behavior has no semantic value");
    const Behavior behavior =
        behaviorOf(test, fixture.family, point.representativeActor);
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    if (behavior.direction == Direction::Leading && behavior.width == 8)
      inactiveValue = semantic;
    if (kind == AbiKind::MissingBehavior &&
        behavior.direction == Direction::Trailing &&
        behavior.width == wideWidth(fixture.family))
      continue;
    entries.push_back({std::move(semantic), {physicalCode(behavior)}});
  }
  require(test, !inactiveValue.empty(),
          "count-zero domain has no inactive leading-i8 behavior");
  if (kind == AbiKind::ExtraBehavior)
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
                             llvm::StringRef moduleName,
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
    const auto field = llvm::find_if(ports, [](const auto &port) {
      return port.getName().starts_with("config_");
    });
    require(test, field != ports.end(),
            "configured count-zero leaf has no selector port");
    field->type = builder.getI2Type();
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(moduleName), ports);
  return SkeletonFixture{std::move(module), leaf};
}

std::string moduleText(mlir::ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

std::string specialize(llvm::StringRef test, SkeletonFixture skeleton,
                       const FabricFixture &fabric,
                       const FinalizedConfigurationABI &abi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableIntegerCountZerosProviders(registry))
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
          "portable count-zero provider emitted implementation metadata");
  return std::move(conformance.systemVerilog);
}

void checkSealedRelation(llvm::StringRef test, const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  const auto &descriptor =
      ::fabric::implementationFamily(resolved.implementationFamily);
  require(test,
          resolved.implementationFamily == familyId(fixture.family) &&
              resolved.enabledOperationSchemas.size() ==
                  descriptor.admittedSchemas.size() &&
              resolved.configurationFieldSchema.size() == 1,
          "count-zero capability does not match its generated family");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 4,
          "count-zero aliases did not collapse by direction and active width");

  bool leadingNarrow = false;
  bool trailingNarrow = false;
  bool leadingWide = false;
  bool trailingWide = false;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "configured count-zero behavior has no semantic key");
    const Behavior behavior =
        behaviorOf(test, fixture.family, point.representativeActor);
    const bool narrow = behavior.width == 8;
    const bool wide = behavior.width == wideWidth(fixture.family);
    require(test, narrow || wide,
            "count-zero relation exposed an unexpected active width");
    if (behavior.direction == Direction::Leading && narrow)
      leadingNarrow = true;
    else if (behavior.direction == Direction::Trailing && narrow)
      trailingNarrow = true;
    else if (behavior.direction == Direction::Leading && wide)
      leadingWide = true;
    else if (behavior.direction == Direction::Trailing && wide)
      trailingWide = true;
  }
  require(test, leadingNarrow && trailingNarrow && leadingWide && trailingWide,
          "sealed count-zero domain lost a direction or active width");
}

std::string emitDeterministically(llvm::StringRef test,
                                  const FabricFixture &fabric,
                                  const FinalizedConfigurationABI &abi,
                                  llvm::StringRef moduleName,
                                  unsigned physicalWidth) {
  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first =
      makeSkeleton(test, *firstContext, fabric, abi.abi(), moduleName);
  const circt::hw::ModulePortInfo ports(first.leaf.getPortList());
  require(
      test,
      ports.size() == 3 && ports.atInput(0).getName() == "data_input_0" &&
          mlir::cast<mlir::IntegerType>(ports.atInput(0).type).getWidth() ==
              physicalWidth &&
          ports.atInput(1).getName() == "config_0" &&
          mlir::cast<mlir::IntegerType>(ports.atInput(1).type).getWidth() ==
              3 &&
          ports.atOutput(0).getName() == "data_output_0" &&
          mlir::cast<mlir::IntegerType>(ports.atOutput(0).type).getWidth() ==
              physicalWidth,
      "derived count-zero leaf ports are not canonical");
  const std::string firstRtl = specialize(test, std::move(first), fabric, abi);

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fabric, abi.abi(), moduleName);
  const std::string secondRtl =
      specialize(test, std::move(second), fabric, abi);
  require(test, firstRtl == secondRtl,
          "identical count-zero inputs produced different SystemVerilog");
  const llvm::StringRef rtl(firstRtl);
  require(test,
          rtl.contains("config_0") && !rtl.contains_insensitive("poison") &&
              !rtl.contains_insensitive("trap") &&
              !rtl.contains_insensitive("stall") && !rtl.contains("valid_") &&
              !rtl.contains("ready_") && !rtl.contains("reset") &&
              !rtl.contains("clock"),
          "count-zero RTL added semantic checks or protocol state");
  return firstRtl;
}

const std::string testbench = R"sv(
module testbench;
  logic [66:0] scalar_input;
  logic [2:0] scalar_config;
  logic [66:0] scalar_output;
  logic [66:0] expected_scalar;
  logic [119:0] vector_input;
  logic [2:0] vector_config;
  logic [119:0] vector_output;
  logic [119:0] expected_vector;
  logic [119:0] sibling_output;
  integer lane;
  integer sample;

  scalar_integer_count_zeros scalar_dut(
    .data_input_0(scalar_input), .config_0(scalar_config),
    .data_output_0(scalar_output));
  fixed_vector_integer_count_zeros vector_dut(
    .data_input_0(vector_input), .config_0(vector_config),
    .data_output_0(vector_output));

  function automatic [7:0] clz8(input logic [7:0] value);
    integer bit_index;
    begin
      clz8 = 8;
      for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1)
        if (value[bit_index]) clz8 = 7 - bit_index;
    end
  endfunction

  function automatic [7:0] ctz8(input logic [7:0] value);
    integer bit_index;
    begin
      ctz8 = 8;
      for (bit_index = 7; bit_index >= 0; bit_index = bit_index - 1)
        if (value[bit_index]) ctz8 = bit_index;
    end
  endfunction

  function automatic [15:0] clz16(input logic [15:0] value);
    integer bit_index;
    begin
      clz16 = 16;
      for (bit_index = 0; bit_index < 16; bit_index = bit_index + 1)
        if (value[bit_index]) clz16 = 15 - bit_index;
    end
  endfunction

  function automatic [15:0] ctz16(input logic [15:0] value);
    integer bit_index;
    begin
      ctz16 = 16;
      for (bit_index = 15; bit_index >= 0; bit_index = bit_index - 1)
        if (value[bit_index]) ctz16 = bit_index;
    end
  endfunction

  function automatic [31:0] clz32(input logic [31:0] value);
    integer bit_index;
    begin
      clz32 = 32;
      for (bit_index = 0; bit_index < 32; bit_index = bit_index + 1)
        if (value[bit_index]) clz32 = 31 - bit_index;
    end
  endfunction

  function automatic [31:0] ctz32(input logic [31:0] value);
    integer bit_index;
    begin
      ctz32 = 32;
      for (bit_index = 31; bit_index >= 0; bit_index = bit_index - 1)
        if (value[bit_index]) ctz32 = bit_index;
    end
  endfunction

  initial begin
    scalar_input = '0;
    scalar_config = 3'b001;
    #1;
    if (scalar_output !== 67'd8) $fatal(1, "scalar clz zero mismatch");
    scalar_config = 3'b110;
    #1;
    if (scalar_output !== 67'd8) $fatal(1, "scalar ctz zero mismatch");
    scalar_config = 3'b011;
    #1;
    if (scalar_output !== 67'd32) $fatal(1, "scalar wide clz zero mismatch");
    scalar_config = 3'b100;
    #1;
    if (scalar_output !== 67'd32) $fatal(1, "scalar wide ctz zero mismatch");

    scalar_input = '1;
    scalar_input[7:0] = 8'h01;
    scalar_config = 3'b001;
    #1;
    if (scalar_output !== 67'd7)
      $fatal(1, "scalar clz active-width mismatch");
    scalar_input[7:0] = 8'h80;
    scalar_config = 3'b110;
    #1;
    if (scalar_output !== 67'd7)
      $fatal(1, "scalar ctz active-width mismatch");

    scalar_input = '0;
    scalar_input[7:0] = 8'h84;
    scalar_config = 3'b001;
    #1;
    if (scalar_output !== 67'd0)
      $fatal(1, "scalar clz multi-hot mismatch");
    scalar_config = 3'b110;
    #1;
    if (scalar_output !== 67'd2)
      $fatal(1, "scalar ctz multi-hot mismatch");

    scalar_input = '0;
    scalar_input[5] = 1'b1;
    scalar_input[66:32] = '1;
    scalar_config = 3'b011;
    #1;
    if (scalar_output !== 67'd26)
      $fatal(1, "scalar clz output encoding mismatch");
    scalar_input = '0;
    scalar_input[29] = 1'b1;
    scalar_input[66:32] = '1;
    scalar_config = 3'b100;
    #1;
    if (scalar_output !== 67'd29)
      $fatal(1, "scalar ctz output encoding mismatch");

    for (sample = 0; sample < 32; sample = sample + 1) begin
      scalar_input = '0;
      scalar_input[31:0] = 32'h80000001 ^ (sample << 3);
      scalar_config = 3'b011;
      expected_scalar = '0;
      expected_scalar[31:0] = clz32(scalar_input[31:0]);
      #1;
      if (scalar_output !== expected_scalar)
        $fatal(1, "scalar clz oracle mismatch");
      scalar_config = 3'b100;
      expected_scalar[31:0] = ctz32(scalar_input[31:0]);
      #1;
      if (scalar_output !== expected_scalar)
        $fatal(1, "scalar ctz oracle mismatch");
    end

    vector_input = '0;
    vector_config = 3'b001;
    expected_vector = '0;
    for (lane = 0; lane < 15; lane = lane + 1)
      expected_vector[lane * 8 +: 8] = 8'd8;
    #1;
    if (vector_output !== expected_vector)
      $fatal(1, "vector clz zero mismatch");
    vector_config = 3'b110;
    #1;
    if (vector_output !== expected_vector)
      $fatal(1, "vector ctz zero mismatch");

    for (lane = 0; lane < 15; lane = lane + 1)
      vector_input[lane * 8 +: 8] = 8'h01 << (lane % 8);
    vector_input[4 * 8 +: 8] = 8'h84;
    vector_config = 3'b001;
    expected_vector = '0;
    for (lane = 0; lane < 15; lane = lane + 1)
      expected_vector[lane * 8 +: 8] =
          clz8(vector_input[lane * 8 +: 8]);
    #1;
    if (vector_output !== expected_vector)
      $fatal(1, "vector clz lane oracle mismatch");
    if (vector_output[4 * 8 +: 8] !== 8'd0)
      $fatal(1, "vector clz multi-hot mismatch");
    vector_config = 3'b110;
    for (lane = 0; lane < 15; lane = lane + 1)
      expected_vector[lane * 8 +: 8] =
          ctz8(vector_input[lane * 8 +: 8]);
    #1;
    if (vector_output !== expected_vector)
      $fatal(1, "vector ctz lane oracle mismatch");
    if (vector_output[4 * 8 +: 8] !== 8'd2)
      $fatal(1, "vector ctz multi-hot mismatch");

    sibling_output = vector_output;
    vector_input[4 * 8 +: 8] = 8'h80;
    #1;
    if (vector_output[4 * 8 +: 8] !== 8'd7)
      $fatal(1, "vector changed lane mismatch");
    for (lane = 0; lane < 15; lane = lane + 1)
      if (lane != 4 && vector_output[lane * 8 +: 8] !==
                           sibling_output[lane * 8 +: 8])
        $fatal(1, "vector sibling lane changed");

    vector_input = '0;
    for (lane = 0; lane < 7; lane = lane + 1)
      vector_input[lane * 16 +: 16] = 16'h0001 << ((lane * 3) % 16);
    vector_input[119:112] = 8'hff;
    vector_config = 3'b011;
    expected_vector = '0;
    for (lane = 0; lane < 7; lane = lane + 1)
      expected_vector[lane * 16 +: 16] =
          clz16(vector_input[lane * 16 +: 16]);
    #1;
    if (vector_output !== expected_vector)
      $fatal(1, "vector clz element-width mismatch");
    vector_config = 3'b100;
    for (lane = 0; lane < 7; lane = lane + 1)
      expected_vector[lane * 16 +: 16] =
          ctz16(vector_input[lane * 16 +: 16]);
    #1;
    if (vector_output !== expected_vector)
      $fatal(1, "vector ctz element-width mismatch");

    vector_input = '0;
    vector_input[119:112] = 8'hff;
    vector_config = 3'b011;
    expected_vector = '0;
    for (lane = 0; lane < 7; lane = lane + 1)
      expected_vector[lane * 16 +: 16] = 16'd16;
    #1;
    if (vector_output !== expected_vector)
      $fatal(1, "vector wide clz zero mismatch");
    vector_config = 3'b100;
    #1;
    if (vector_output !== expected_vector)
      $fatal(1, "vector wide ctz zero mismatch");

    scalar_input = '0;
    scalar_input[7:0] = 8'h04;
    scalar_config = 3'b000;
    #1;
    if (scalar_output !== 67'd5)
      $fatal(1, "inactive scalar code mismatch");
    vector_input = '0;
    for (lane = 0; lane < 15; lane = lane + 1)
      vector_input[lane * 8 +: 8] = 8'h04;
    vector_config = 3'b000;
    expected_vector = '0;
    for (lane = 0; lane < 15; lane = lane + 1)
      expected_vector[lane * 8 +: 8] = 8'd5;
    #1;
    if (vector_output !== expected_vector)
      $fatal(1, "inactive vector code mismatch");
    $finish;
  end
endmodule
)sv";

const std::string synthesisTop = R"sv(
module integer_count_zeros_synthesis_top(
  input logic [66:0] scalar_input,
  input logic [2:0] scalar_config,
  input logic [119:0] vector_input,
  input logic [2:0] vector_config,
  output logic [66:0] scalar_output,
  output logic [119:0] vector_output);
  scalar_integer_count_zeros scalar_count(
    .data_input_0(scalar_input), .config_0(scalar_config),
    .data_output_0(scalar_output));
  fixed_vector_integer_count_zeros vector_count(
    .data_input_0(vector_input), .config_0(vector_config),
    .data_output_0(vector_output));
endmodule
)sv";

const std::string yosysScript = R"ys(
read_verilog -sv scalar_integer_count_zeros.sv fixed_vector_integer_count_zeros.sv synthesis_top.sv
hierarchy -check -top integer_count_zeros_synthesis_top
proc
opt
check -assert
select -assert-none t:$dff t:$adff t:$sdff t:$dlatch
synth -top integer_count_zeros_synthesis_top
check -assert
stat
)ys";

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricFixture scalar =
      makeFabric(test, store, FamilyKind::Scalar, FixtureKind::Configured);
  FabricFixture vector =
      makeFabric(test, store, FamilyKind::FixedVector, FixtureKind::Configured);
  checkSealedRelation(test, scalar);
  checkSealedRelation(test, vector);
  FinalizedConfigurationABI scalarAbi =
      makeConfigurationAbi(test, store, scalar);
  FinalizedConfigurationABI vectorAbi =
      makeConfigurationAbi(test, store, vector);
  const std::string scalarRtl = emitDeterministically(
      test, scalar, scalarAbi, "scalar_integer_count_zeros", 67);
  const std::string vectorRtl = emitDeterministically(
      test, vector, vectorAbi, "fixed_vector_integer_count_zeros", 120);
  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"scalar_integer_count_zeros.sv", scalarRtl},
           {"fixed_vector_integer_count_zeros.sv", vectorRtl},
           {"testbench.sv", testbench},
           {"synthesis_top.sv", synthesisTop},
           {"portable_integer_count_zeros.ys", yosysScript}}))
    fail(test, llvm::toString(std::move(error)));
}

void registrationIsPortableOnly() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableIntegerCountZerosProviders(registry))
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
            "count-zero provider registered a native recipe alias");
  }
  for (const auto &entry : coverage)
    if (entry.implementationFamily != familyId(FamilyKind::Scalar) &&
        entry.implementationFamily != familyId(FamilyKind::FixedVector))
      require(test, entry.recipes.empty(),
              "count-zero registration covered an unrelated family");
}

void singletonNeedsNoSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  for (FamilyKind family : {FamilyKind::Scalar, FamilyKind::FixedVector}) {
    FabricFixture fabric =
        makeFabric(test, store, family, FixtureKind::Singleton);
    const auto &resolved = capability(test, fabric);
    auto relation =
        take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
    require(test,
            resolved.configurationFieldSchema.empty() &&
                relation.kind() ==
                    ::fabric::FabricOpSemanticFieldRelationKind::None &&
                relation.finiteBehaviorDomain().size() == 1 &&
                !relation.finiteBehaviorDomain().front().semanticConfiguration,
            "singleton count-zero capability retained a semantic selector");
    FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi(),
                                            family == FamilyKind::Scalar
                                                ? "scalar_count_trailing_zeros"
                                                : "vector_count_leading_zeros");
    require(test, skeleton.leaf.getPortList().size() == 2,
            "singleton count-zero leaf retained a selector port");
    const std::string rtl = specialize(test, std::move(skeleton), fabric, abi);
    require(test, !llvm::StringRef(rtl).contains("config_"),
            "singleton count-zero RTL retained a selector");
  }
}

llvm::Expected<FabricOperationProviderOutput>
trySpecialize(SkeletonFixture &skeleton, const FabricFixture &fabric,
              const FinalizedConfigurationABI &abi, BackendRecipeKey recipe,
              const FabricOperationProviderRegistry &registry) {
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> associations = {
      {skeleton.leaf, fabric.physicalOccurrence}};
  const std::vector<FabricOperationRecipeBinding> recipes = {
      {fabric.physicalOccurrence, recipe, {}}};
  return specializeFabricOperationLeaves(*skeleton.module, abi, associations,
                                         recipes, registry, externalContracts);
}

void expectTypedUnsupported(
    llvm::StringRef test, llvm::Expected<FabricOperationProviderOutput> result,
    ::fabric::ImplementationFamilyId expectedFamily,
    BackendRecipeKey expectedRecipe, llvm::StringRef description) {
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

void expectInvalid(llvm::StringRef test,
                   llvm::Expected<FabricOperationProviderOutput> result,
                   llvm::StringRef expected) {
  require(test, !result, "provider accepted malformed count-zero input");
  bool invalid = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &) {
        fail(test, "malformed count-zero input became typed Unsupported");
      },
      [&](const llvm::ErrorInfoBase &error) {
        require(test, llvm::StringRef(error.message()).contains(expected),
                error.message());
        invalid = true;
      });
  require(test, invalid, "malformed count-zero input lost its error");
}

void failuresAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableIntegerCountZerosProviders(registry))
    fail(test, llvm::toString(std::move(error)));

  constexpr std::array nativeRecipes = {
      BackendRecipeKey::SynopsysDesignWare, BackendRecipeKey::CadenceChipWare,
      BackendRecipeKey::AmdXilinx, BackendRecipeKey::IntelAltera};
  for (FamilyKind family : {FamilyKind::Scalar, FamilyKind::FixedVector}) {
    FabricFixture unsupported =
        makeFabric(test, store, family, FixtureKind::UnsupportedContract);
    FinalizedConfigurationABI unsupportedAbi =
        makeConfigurationAbi(test, store, unsupported);
    std::unique_ptr<mlir::MLIRContext> unsupportedContext = makeCirctContext();
    SkeletonFixture unsupportedSkeleton =
        makeSkeleton(test, *unsupportedContext, unsupported,
                     unsupportedAbi.abi(), "unsupported_integer_count_zeros");
    const std::string unsupportedBefore =
        moduleText(*unsupportedSkeleton.module);
    expectTypedUnsupported(
        test,
        trySpecialize(unsupportedSkeleton, unsupported, unsupportedAbi,
                      BackendRecipeKey::PortableSystemVerilog, registry),
        familyId(family), BackendRecipeKey::PortableSystemVerilog,
        "unsupported resource contract");
    require(test, moduleText(*unsupportedSkeleton.module) == unsupportedBefore,
            "unsupported capability mutated the caller module");

    FabricFixture valid =
        makeFabric(test, store, family, FixtureKind::Configured);
    FinalizedConfigurationABI validAbi =
        makeConfigurationAbi(test, store, valid);
    std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
    SkeletonFixture malformed =
        makeSkeleton(test, *malformedContext, valid, validAbi.abi(),
                     "malformed_integer_count_zeros", true);
    const std::string malformedBefore = moduleText(*malformed.module);
    expectInvalid(test,
                  trySpecialize(malformed, valid, validAbi,
                                BackendRecipeKey::PortableSystemVerilog,
                                registry),
                  "leaf port");
    require(test, moduleText(*malformed.module) == malformedBefore,
            "malformed leaf partially mutated the caller module");

    expectRejected(
        test,
        finalizeConfigurationABI(
            makeConfigurationAbiDraft(test, valid, AbiKind::MissingBehavior),
            store),
        "finite codebook");
    expectRejected(
        test,
        finalizeConfigurationABI(
            makeConfigurationAbiDraft(test, valid, AbiKind::ExtraBehavior),
            store),
        "semantic value");

    for (BackendRecipeKey recipe : nativeRecipes) {
      std::unique_ptr<mlir::MLIRContext> nativeContext = makeCirctContext();
      SkeletonFixture native =
          makeSkeleton(test, *nativeContext, valid, validAbi.abi(),
                       "native_integer_count_zeros");
      const std::string before = moduleText(*native.module);
      expectTypedUnsupported(
          test, trySpecialize(native, valid, validAbi, recipe, registry),
          familyId(family), recipe, "native recipe");
      require(test, moduleText(*native.module) == before,
              "unsupported native recipe mutated the caller module");
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  registrationIsPortableOnly();
  configuredBehaviorAndArtifacts(root);
  singletonNeedsNoSelector(root / "singletons");
  failuresAreTransactional(root / "failures");
  return 0;
}

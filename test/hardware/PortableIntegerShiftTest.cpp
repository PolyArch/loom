#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/IntegerShift.h"
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
#include <optional>
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
  require(test, !value, "accepted malformed integer shift input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

enum class FamilyKind { Scalar, FixedVector };
enum class AbiKind { Complete, MissingMode, ExtraSemanticValue };
enum class ShiftOperation { Left, LogicalRight, ArithmeticRight };

struct ShiftMode final {
  ShiftOperation operation;
  unsigned activeWidth = 0;
};

::fabric::ImplementationFamilyId familyId(FamilyKind family) {
  return family == FamilyKind::Scalar
             ? ::fabric::ImplementationFamilyId::ScalarIntegerShift
             : ::fabric::ImplementationFamilyId::FixedVectorIntegerShift;
}

bool sameMode(const ShiftMode &lhs, const ShiftMode &rhs) {
  return lhs.operation == rhs.operation && lhs.activeWidth == rhs.activeWidth;
}

ShiftMode
modeOf(llvm::StringRef test, FamilyKind family,
       const ::dataflow::CanonicalActorSchemaProjection &representative) {
  ShiftOperation operation;
  using Schema = ::dataflow::OperationSchemaId;
  switch (representative.schema) {
  case Schema::ArithShLI:
    operation = ShiftOperation::Left;
    break;
  case Schema::ArithShRUI:
    operation = ShiftOperation::LogicalRight;
    break;
  case Schema::ArithShRSI:
    operation = ShiftOperation::ArithmeticRight;
    break;
  default:
    fail(test, "Fabric returned a non-shift behavior witness");
  }

  if (family == FamilyKind::Scalar &&
      operation != ShiftOperation::ArithmeticRight)
    return {operation, 0};
  mlir::Type input = representative.type.getInput(0);
  if (family == FamilyKind::Scalar) {
    auto integer = llvm::dyn_cast<mlir::IntegerType>(input);
    require(test, static_cast<bool>(integer),
            "scalar shift witness has a non-integer input");
    return {operation, integer.getWidth()};
  }
  auto vector = llvm::dyn_cast<mlir::VectorType>(input);
  require(test, static_cast<bool>(vector),
          "vector shift witness has a non-vector input");
  auto element = llvm::dyn_cast<mlir::IntegerType>(vector.getElementType());
  require(test, static_cast<bool>(element),
          "vector shift witness has a non-integer element");
  return {operation, element.getWidth()};
}

std::uint8_t physicalCode(llvm::StringRef test, FamilyKind family,
                          const ShiftMode &mode) {
  if (family == FamilyKind::Scalar) {
    if (mode.operation == ShiftOperation::Left && mode.activeWidth == 0)
      return 1;
    if (mode.operation == ShiftOperation::LogicalRight && mode.activeWidth == 0)
      return 2;
    if (mode.operation == ShiftOperation::ArithmeticRight) {
      if (mode.activeWidth == 8)
        return 4;
      if (mode.activeWidth == 16)
        return 6;
      if (mode.activeWidth == 32)
        return 7;
    }
  } else {
    if (mode.operation == ShiftOperation::Left && mode.activeWidth == 8)
      return 1;
    if (mode.operation == ShiftOperation::Left && mode.activeWidth == 16)
      return 2;
    if (mode.operation == ShiftOperation::LogicalRight && mode.activeWidth == 8)
      return 3;
    if (mode.operation == ShiftOperation::LogicalRight &&
        mode.activeWidth == 16)
      return 4;
    if (mode.operation == ShiftOperation::ArithmeticRight &&
        mode.activeWidth == 8)
      return 5;
    if (mode.operation == ShiftOperation::ArithmeticRight &&
        mode.activeWidth == 16)
      return 6;
  }
  fail(test, "sealed shift relation produced an unexpected mode");
}

bool isInactiveMode(FamilyKind family, const ShiftMode &mode) {
  if (mode.operation != ShiftOperation::LogicalRight)
    return false;
  return family == FamilyKind::Scalar ? mode.activeWidth == 0
                                      : mode.activeWidth == 16;
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
                         FamilyKind family, bool unsupportedContract = false,
                         bool singleton = false) {
  require(test, !singleton || family == FamilyKind::FixedVector,
          "scalar singleton fixture is not defined");
  const llvm::StringRef sourceText = family == FamilyKind::Scalar ? R"mlir(
    module {
      fabric.module @scalar_integer_shift(
          %value: !fabric.bits<32>, %amount: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pv = %value : !fabric.bits<32>,
             %pa = %amount : !fabric.bits<32>) -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fv = %pv : !fabric.bits<32>,
               %fa = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
            %shifted = fabric.op
              [@arith.shli, @arith.shrsi, @arith.shrui] (%fv, %fa)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerShift>,
               hw_params = {
                 integer_widths = [8 : i32, 16 : i32, 32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %shifted : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir"
                                     : singleton                  ? R"mlir(
    module {
      fabric.module @fixed_vector_integer_shift_singleton(
          %value: !fabric.bits<24>, %amount: !fabric.bits<24>)
          -> !fabric.bits<24> {
        %pe = fabric.pe [spatial]
            (%pv = %value : !fabric.bits<24>,
             %pa = %amount : !fabric.bits<24>) -> !fabric.bits<24> {
          %fu = fabric.fu
              (%fv = %pv : !fabric.bits<24>,
               %fa = %pa : !fabric.bits<24>) -> !fabric.bits<24> {
            %shifted = fabric.op
              [@arith.shrsi] (%fv, %fa)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorIntegerShift>,
               hw_params = {
                 element_widths = [8 : i32],
                 max_payload_bits = 24 : i32}}
              : (!fabric.bits<24>, !fabric.bits<24>) -> !fabric.bits<24>
            fabric.yield %shifted : !fabric.bits<24>
          }
        }
        fabric.yield %pe : !fabric.bits<24>
      }
    }
  )mlir"
                                                                  : R"mlir(
    module {
      fabric.module @fixed_vector_integer_shift(
          %value: !fabric.bits<32>, %amount: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pv = %value : !fabric.bits<32>,
             %pa = %amount : !fabric.bits<32>) -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fv = %pv : !fabric.bits<32>,
               %fa = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
            %shifted = fabric.op
              [@arith.shli, @arith.shrsi, @arith.shrui] (%fv, %fa)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorIntegerShift>,
               hw_params = {
                 element_widths = [8 : i32, 16 : i32],
                 max_payload_bits = 32 : i32}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %shifted : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir";

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
              "System has no physical integer shift occurrence");
      return FabricFixture{std::move(fabric), occurrence, std::move(system),
                           physical->physicalOccurrence, family};
    }
  }
  fail(test, "Fabric fixture has no integer shift occurrence");
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
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  if (resolved.configurationFieldSchema.empty()) {
    const auto domain = relation.finiteBehaviorDomain();
    require(test,
            kind == AbiKind::Complete &&
                relation.kind() ==
                    ::fabric::FabricOpSemanticFieldRelationKind::None &&
                domain.size() == 1 && !domain.front().semanticConfiguration,
            "configuration-free shift relation is not a singleton");
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system, {}));
  }
  require(test, resolved.configurationFieldSchema.size() == 1,
          "configured shift capability does not have one field");
  require(test,
          relation.kind() ==
              ::fabric::FabricOpSemanticFieldRelationKind::Finite,
          "configured shift relation is not finite");
  const auto domain = relation.finiteBehaviorDomain();
  const std::size_t expectedSize = fixture.family == FamilyKind::Scalar ? 5 : 6;
  require(test, domain.size() == expectedSize,
          "shift capability has the wrong behavior domain size");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactiveValue;
  for (const auto &point : domain) {
    require(test, point.semanticConfiguration.has_value(),
            "configured shift behavior has no semantic value");
    std::vector<std::uint8_t> semantic(
        point.semanticConfiguration->bytes().begin(),
        point.semanticConfiguration->bytes().end());
    const ShiftMode mode =
        modeOf(test, fixture.family, point.representativeActor);
    if (isInactiveMode(fixture.family, mode))
      inactiveValue = semantic;
    if (kind == AbiKind::MissingMode &&
        sameMode(mode, {ShiftOperation::ArithmeticRight, 8}))
      semantic = {0xfd};
    entries.push_back(
        {std::move(semantic), {physicalCode(test, fixture.family, mode)}});
  }
  require(test, !inactiveValue.empty(),
          "shift domain has no inactive behavior");
  if (kind == AbiKind::ExtraSemanticValue)
    entries.push_back({{0xfe}, {0}});

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
    require(test, config != ports.end(), "shift leaf has no selector port");
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

std::string emit(llvm::StringRef test, const FabricFixture &fabric,
                 const FinalizedConfigurationABI &abi,
                 llvm::StringRef moduleName) {
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton =
      makeSkeleton(test, *context, fabric, abi.abi(), moduleName);
  const circt::hw::ModulePortInfo ports(skeleton.leaf.getPortList());
  const bool configured =
      !capability(test, fabric).configurationFieldSchema.empty();
  bool canonical = ports.size() == (configured ? 4U : 3U) &&
                   ports.atInput(0).getName() == "data_input_0" &&
                   ports.atInput(1).getName() == "data_input_1" &&
                   ports.atOutput(0).getName() == "data_output_0";
  if (configured)
    canonical =
        canonical && ports.atInput(2).getName() == "config_0" &&
        mlir::cast<mlir::IntegerType>(ports.atInput(2).type).getWidth() == 3;
  require(test, canonical, "derived shift ports are not canonical");

  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableIntegerShiftProviders(registry))
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
          "portable shift provider emitted implementation metadata");
  return std::move(conformance.systemVerilog);
}

void checkSealedDomain(llvm::StringRef test, const FabricFixture &fixture) {
  const auto &resolved = capability(test, fixture);
  const auto &descriptor =
      ::fabric::implementationFamily(resolved.implementationFamily);
  require(test,
          resolved.implementationFamily == familyId(fixture.family) &&
              resolved.enabledOperationSchemas.size() ==
                  descriptor.admittedSchemas.size() &&
              resolved.configurationFieldSchema.size() == 1,
          "sealed shift capability does not match its generated family");
  auto relation =
      take(test, resolved.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() ==
                  (fixture.family == FamilyKind::Scalar ? 5U : 6U),
          "sealed shift relation has the wrong finite quotient");
}

std::string testbenchText() {
  return R"sv(
module testbench;
  logic [31:0] scalar_input_0;
  logic [31:0] scalar_input_1;
  logic [2:0] scalar_config;
  logic [31:0] scalar_output;
  logic [31:0] vector_input_0;
  logic [31:0] vector_input_1;
  logic [2:0] vector_config;
  logic [31:0] vector_output;
  logic [23:0] singleton_input_0;
  logic [23:0] singleton_input_1;
  logic [23:0] singleton_output;

  scalar_integer_shift scalar_dut(
    .data_input_0(scalar_input_0), .data_input_1(scalar_input_1),
    .config_0(scalar_config), .data_output_0(scalar_output));
  fixed_vector_integer_shift vector_dut(
    .data_input_0(vector_input_0), .data_input_1(vector_input_1),
    .config_0(vector_config), .data_output_0(vector_output));
  fixed_vector_integer_shift_singleton singleton_dut(
    .data_input_0(singleton_input_0), .data_input_1(singleton_input_1),
    .data_output_0(singleton_output));

  task automatic check_scalar(
      input logic [31:0] value, input logic [31:0] amount,
      input logic [2:0] mode, input logic [31:0] expected,
      input string description);
    begin
      scalar_input_0 = value;
      scalar_input_1 = amount;
      scalar_config = mode;
      #1;
      if (scalar_output !== expected) $fatal(1, "%s", description);
    end
  endtask

  task automatic check_vector(
      input logic [31:0] value, input logic [31:0] amount,
      input logic [2:0] mode, input logic [31:0] expected,
      input string description);
    begin
      vector_input_0 = value;
      vector_input_1 = amount;
      vector_config = mode;
      #1;
      if (vector_output !== expected) $fatal(1, "%s", description);
    end
  endtask

  initial begin
    check_scalar(32'h12345678, 0, 3'b001, 32'h12345678,
                 "scalar left shift by zero failed");
    check_scalar(1, 31, 3'b001, 32'h80000000,
                 "scalar left shift at width minus one failed");
    check_scalar(1, 32, 3'b001, 0,
                 "scalar left shift at width failed");
    check_scalar(32'h80000000, 31, 3'b010, 1,
                 "scalar logical right at width minus one failed");
    check_scalar(32'h80000000, 32, 3'b010, 0,
                 "scalar logical right at width failed");
    check_scalar(32'hf0000000, 4, 3'b000, 32'h0f000000,
                 "scalar inactive logical right behavior failed");

    check_scalar(32'h00000080, 0, 3'b100, 32'h00000080,
                 "scalar i8 arithmetic right zero-amount padding failed");
    check_scalar(32'h00000080, 1, 3'b100, 32'h000000c0,
                 "scalar i8 arithmetic right sign propagation failed");
    check_scalar(32'h00000080, 7, 3'b100, 32'h000000ff,
                 "scalar i8 arithmetic right boundary failed");
    check_scalar(32'h00000080, 8, 3'b100, 32'h000000ff,
                 "scalar i8 arithmetic right overshift failed");
    check_scalar(32'h00008000, 15, 3'b110, 32'h0000ffff,
                 "scalar i16 arithmetic right boundary failed");
    check_scalar(32'h00007fff, 16, 3'b110, 0,
                 "scalar i16 arithmetic right overshift failed");
    check_scalar(32'h80000000, 31, 3'b111, 32'hffffffff,
                 "scalar i32 arithmetic right boundary failed");
    check_scalar(32'h80000000, 32, 3'b111, 32'hffffffff,
                 "scalar i32 arithmetic right overshift failed");

    check_vector(32'h01804081, 32'h07010800, 3'b001, 32'h80000081,
                 "vector i8 left shifts crossed lanes");
    check_vector(32'h80010001, 32'h0010000f, 3'b010, 32'h00008000,
                 "vector i16 left shift boundaries failed");
    check_vector(32'h80ff0180, 32'h01000807, 3'b011, 32'h40ff0001,
                 "vector i8 logical shifts crossed lanes");
    check_vector(32'h8000ffff, 32'h0010000f, 3'b100, 32'h00000001,
                 "vector i16 logical shift boundaries failed");
    check_vector(32'h807f0180, 32'h01000807, 3'b101, 32'hc07f00ff,
                 "vector i8 arithmetic shifts crossed lanes");
    check_vector(32'h80007fff, 32'h0010000f, 3'b110, 32'hffff0000,
                 "vector i16 arithmetic shift boundaries failed");
    check_vector(32'h8000ffff, 32'h0010000f, 3'b000, 32'h00000001,
                 "vector inactive logical right behavior failed");
    singleton_input_0 = 24'h807f80;
    singleton_input_1 = 24'h010701;
    #1;
    if (singleton_output !== 24'hc000c0)
      $fatal(1, "three-lane singleton arithmetic shift failed");
    $finish;
  end
endmodule
)sv";
}

std::string synthesisTopText() {
  return R"sv(
module integer_shift_synthesis_top(
  input logic [31:0] scalar_input_0,
  input logic [31:0] scalar_input_1,
  input logic [2:0] scalar_config,
  input logic [31:0] vector_input_0,
  input logic [31:0] vector_input_1,
  input logic [2:0] vector_config,
  input logic [23:0] singleton_input_0,
  input logic [23:0] singleton_input_1,
  output logic [31:0] scalar_output,
  output logic [31:0] vector_output,
  output logic [23:0] singleton_output);
  scalar_integer_shift scalar_shift(
    .data_input_0(scalar_input_0), .data_input_1(scalar_input_1),
    .config_0(scalar_config), .data_output_0(scalar_output));
  fixed_vector_integer_shift vector_shift(
    .data_input_0(vector_input_0), .data_input_1(vector_input_1),
    .config_0(vector_config), .data_output_0(vector_output));
  fixed_vector_integer_shift_singleton singleton_shift(
    .data_input_0(singleton_input_0), .data_input_1(singleton_input_1),
    .data_output_0(singleton_output));
endmodule
)sv";
}

std::string yosysScriptText() {
  return R"ys(
read_verilog -sv scalar_integer_shift.sv
read_verilog -sv fixed_vector_integer_shift.sv
read_verilog -sv fixed_vector_integer_shift_singleton.sv
read_verilog -sv synthesis_top.sv
hierarchy -check -top integer_shift_synthesis_top
proc
opt
check -assert
select -assert-count 7 t:$shl
select -assert-count 7 t:$shr
select -assert-count 12 t:$sshr
synth -top integer_shift_synthesis_top
check -assert
stat
)ys";
}

void configuredBehaviorAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture scalar = makeFabric(test, store, FamilyKind::Scalar);
  FabricFixture vector = makeFabric(test, store, FamilyKind::FixedVector);
  FabricFixture singleton =
      makeFabric(test, store, FamilyKind::FixedVector,
                 /*unsupportedContract=*/false, /*singleton=*/true);
  checkSealedDomain(test, scalar);
  checkSealedDomain(test, vector);
  const auto &singletonCapability = capability(test, singleton);
  auto singletonRelation = take(
      test, singletonCapability.resolveSemanticFieldRelation(fabricContext()));
  require(test,
          singletonCapability.configurationFieldSchema.empty() &&
              singletonRelation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::None &&
              singletonRelation.finiteBehaviorDomain().size() == 1,
          "singleton vector shift retained a configuration field");
  FinalizedConfigurationABI scalarAbi =
      makeConfigurationAbi(test, store, scalar);
  FinalizedConfigurationABI vectorAbi =
      makeConfigurationAbi(test, store, vector);
  FinalizedConfigurationABI singletonAbi =
      makeConfigurationAbi(test, store, singleton);

  const std::string scalarFirst =
      emit(test, scalar, scalarAbi, "scalar_integer_shift");
  const std::string scalarSecond =
      emit(test, scalar, scalarAbi, "scalar_integer_shift");
  const std::string vectorFirst =
      emit(test, vector, vectorAbi, "fixed_vector_integer_shift");
  const std::string vectorSecond =
      emit(test, vector, vectorAbi, "fixed_vector_integer_shift");
  const std::string singletonFirst = emit(
      test, singleton, singletonAbi, "fixed_vector_integer_shift_singleton");
  const std::string singletonSecond = emit(
      test, singleton, singletonAbi, "fixed_vector_integer_shift_singleton");
  require(test,
          scalarFirst == scalarSecond && vectorFirst == vectorSecond &&
              singletonFirst == singletonSecond,
          "identical shift inputs produced different SystemVerilog");
  require(test,
          llvm::StringRef(scalarFirst).contains("config_0") &&
              llvm::StringRef(vectorFirst).contains("config_0") &&
              !llvm::StringRef(singletonFirst).contains("config_") &&
              !llvm::StringRef(scalarFirst).contains("valid_") &&
              !llvm::StringRef(vectorFirst).contains("ready_"),
          "shift RTL changed the operation-leaf protocol");

  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts",
          {{"scalar_integer_shift.sv", scalarFirst},
           {"fixed_vector_integer_shift.sv", vectorFirst},
           {"fixed_vector_integer_shift_singleton.sv", singletonFirst},
           {"testbench.sv", testbenchText()},
           {"synthesis_top.sv", synthesisTopText()},
           {"portable_integer_shift.ys", yosysScriptText()}}))
    fail(test, llvm::toString(std::move(error)));
}

void registrationIsPortableOnly() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableIntegerShiftProviders(registry))
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
            "integer shift provider registered a native recipe alias");
  }
  for (const auto &entry : coverage)
    if (entry.implementationFamily != familyId(FamilyKind::Scalar) &&
        entry.implementationFamily != familyId(FamilyKind::FixedVector))
      require(test, entry.recipes.empty(),
              "integer shift registration covered an unrelated family");
}

llvm::Expected<FabricOperationProviderOutput>
placeholderProvider(FabricOperationProviderRequest) {
  return FabricOperationProviderOutput{};
}

void failedRegistrationRollsBack() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registry.add(
          {::fabric::ImplementationFamilyId::FixedVectorIntegerShift,
           BackendRecipeKey::PortableSystemVerilog,
           {},
           placeholderProvider}))
    fail(test, llvm::toString(std::move(error)));
  llvm::Error error = registerPortableIntegerShiftProviders(registry);
  require(test, static_cast<bool>(error),
          "duplicate vector shift registration succeeded");
  llvm::consumeError(std::move(error));
  const auto coverage = registry.coverage();
  const auto scalar = llvm::find_if(coverage, [](const auto &entry) {
    return entry.implementationFamily ==
           ::fabric::ImplementationFamilyId::ScalarIntegerShift;
  });
  const auto vector = llvm::find_if(coverage, [](const auto &entry) {
    return entry.implementationFamily ==
           ::fabric::ImplementationFamilyId::FixedVectorIntegerShift;
  });
  require(test,
          scalar != coverage.end() && scalar->recipes.empty() &&
              vector != coverage.end() &&
              vector->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog},
          "failed package registration changed the registry");
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
  require(test, !result, "provider accepted malformed integer shift input");
  bool invalid = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &) {
        fail(test, "malformed integer shift input became typed Unsupported");
      },
      [&](const llvm::ErrorInfoBase &error) {
        require(test, llvm::StringRef(error.message()).contains(expected),
                error.message());
        invalid = true;
      });
  require(test, invalid, "malformed integer shift input lost its error");
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

void failuresAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableIntegerShiftProviders(registry))
    fail(test, llvm::toString(std::move(error)));

  for (FamilyKind family : {FamilyKind::Scalar, FamilyKind::FixedVector}) {
    FabricFixture unsupported = makeFabric(test, store, family, true);
    FinalizedConfigurationABI unsupportedAbi =
        makeConfigurationAbi(test, store, unsupported);
    std::unique_ptr<mlir::MLIRContext> unsupportedContext = makeCirctContext();
    SkeletonFixture unsupportedSkeleton =
        makeSkeleton(test, *unsupportedContext, unsupported,
                     unsupportedAbi.abi(), "unsupported_integer_shift");
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

    FabricFixture valid = makeFabric(test, store, family);
    FinalizedConfigurationABI validAbi =
        makeConfigurationAbi(test, store, valid);
    std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
    SkeletonFixture malformed =
        makeSkeleton(test, *malformedContext, valid, validAbi.abi(),
                     "malformed_integer_shift", true);
    const std::string malformedBefore = moduleText(*malformed.module);
    expectInvalid(test,
                  trySpecialize(malformed, valid, validAbi,
                                BackendRecipeKey::PortableSystemVerilog,
                                registry),
                  "leaf port");
    require(test, moduleText(*malformed.module) == malformedBefore,
            "malformed leaf mutated the caller module");

    for (AbiKind kind : {AbiKind::MissingMode, AbiKind::ExtraSemanticValue})
      expectRejected(
          test,
          finalizeConfigurationABI(
              makeConfigurationAbiDraft(test, store, valid, kind), store),
          "semantic");

    constexpr std::array nativeRecipes = {
        BackendRecipeKey::SynopsysDesignWare, BackendRecipeKey::CadenceChipWare,
        BackendRecipeKey::AmdXilinx, BackendRecipeKey::IntelAltera};
    for (BackendRecipeKey recipe : nativeRecipes) {
      std::unique_ptr<mlir::MLIRContext> nativeContext = makeCirctContext();
      SkeletonFixture native = makeSkeleton(test, *nativeContext, valid,
                                            validAbi.abi(), "native_shift");
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
  registrationIsPortableOnly();
  failedRegistrationRollsBack();
  configuredBehaviorAndArtifacts(root);
  failuresAreTransactional(root / "failures");
  return 0;
}

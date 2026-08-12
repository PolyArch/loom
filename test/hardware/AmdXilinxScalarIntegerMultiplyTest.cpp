#include "Hardware/RTL/Providers/AmdXilinxScalarIntegerMultiply.h"
#include "ConfigurationABITestSupport.h"
#include "Hardware/Implementation/FpgaNativeExternalContracts.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/ScalarIntegerMultiply.h"
#include "ImplementationPlatform/ImplementationPlatform.h"
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
#include <variant>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FinalizedFabricRoot;
using namespace loom::hardware;
using namespace loom::hardware::rtl;
namespace platform = loom::platform;

const FpgaNativeExternalModuleContract &nativeContract() {
  return amdXilinxDsp58ExternalModuleContract();
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

std::string moduleText(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

void expectUnsupportedError(llvm::StringRef test, llvm::Error error,
                            ::fabric::ImplementationFamilyId family) {
  require(test, static_cast<bool>(error),
          "accepted an unverified AMD/Xilinx recipe");
  bool classified = false;
  llvm::handleAllErrors(
      std::move(error),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() == family &&
                     error.recipe() == BackendRecipeKey::AmdXilinx;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unverified recipe returned the wrong error class: " +
                       error.message());
      });
  require(test, classified, "unverified recipe lost typed Unsupported");
}

void expectUnsupported(llvm::StringRef test,
                       llvm::Expected<FabricOperationProviderOutput> value,
                       ::fabric::ImplementationFamilyId family) {
  require(test, !value, "accepted an unverified AMD/Xilinx recipe");
  expectUnsupportedError(test, value.takeError(), family);
}

void expectExternalContractError(llvm::StringRef test, llvm::Error error) {
  require(test, static_cast<bool>(error),
          "accepted an invalid native FPGA external binding");
  const std::string message = llvm::toString(std::move(error));
  require(test,
          llvm::StringRef(message).starts_with(
              "fpga_native_external_contract_invalid:"),
          "invalid external binding lost its owner error classification");
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

enum class FixtureKind {
  Multiply,
  MultiplyAllWidths,
  TwoMultiplies,
  Add,
  UnsupportedContract,
};

struct FabricFixture final {
  FinalizedFabricRoot system;
  std::vector<ResolvedFabricPhysicalOperation> operations;
};

std::string fabricSource(unsigned width, FixtureKind kind) {
  if (kind == FixtureKind::TwoMultiplies)
    return R"mlir(module {
      fabric.module @two_amd_xilinx_tests(
          %a0: !fabric.bits<16>, %b0: !fabric.bits<16>,
          %a1: !fabric.bits<16>, %b1: !fabric.bits<16>)
          -> (!fabric.bits<16>, !fabric.bits<16>) {
        %pe0 = fabric.pe [spatial]
            (%pa0 = %a0 : !fabric.bits<16>, %pb0 = %b0 : !fabric.bits<16>)
            -> !fabric.bits<16> {
          %fu0 = fabric.fu
              (%fa0 = %pa0 : !fabric.bits<16>,
               %fb0 = %pb0 : !fabric.bits<16>) -> !fabric.bits<16> {
            %value0 = fabric.op [@arith.muli] (%fa0, %fb0)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerMultiply>,
               hw_params = {integer_widths = [16 : i32]}}
              : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
            fabric.yield %value0 : !fabric.bits<16>
          }
        }
        %pe1 = fabric.pe [spatial]
            (%pa1 = %a1 : !fabric.bits<16>, %pb1 = %b1 : !fabric.bits<16>)
            -> !fabric.bits<16> {
          %fu1 = fabric.fu
              (%fa1 = %pa1 : !fabric.bits<16>,
               %fb1 = %pb1 : !fabric.bits<16>) -> !fabric.bits<16> {
            %value1 = fabric.op [@arith.muli] (%fa1, %fb1)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerMultiply>,
               hw_params = {integer_widths = [16 : i32]}}
              : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
            fabric.yield %value1 : !fabric.bits<16>
          }
        }
        fabric.yield %pe0, %pe1 : !fabric.bits<16>, !fabric.bits<16>
      }
    })mlir";
  const bool add = kind == FixtureKind::Add;
  const llvm::StringRef operation = add ? "arith.addi" : "arith.muli";
  const llvm::StringRef family =
      add ? "ScalarIntegerAddSub" : "ScalarIntegerMultiply";
  const std::string widths = kind == FixtureKind::MultiplyAllWidths
                                 ? "8 : i32, 16 : i32"
                                 : std::to_string(width) + " : i32";
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module { fabric.module @amd_xilinx_test(%a: !fabric.bits<" << width
         << ">, %b: !fabric.bits<" << width << ">) -> !fabric.bits<" << width
         << "> { %pe = fabric.pe [spatial] (%pa = %a : !fabric.bits<" << width
         << ">, %pb = %b : !fabric.bits<" << width << ">) -> !fabric.bits<"
         << width << "> { %fu = fabric.fu (%fa = %pa : !fabric.bits<" << width
         << ">, %fb = %pb : !fabric.bits<" << width << ">) -> !fabric.bits<"
         << width << "> { %value = fabric.op [@" << operation
         << "] (%fa, %fb) {implementation_family = "
         << "#fabric.implementation_family<" << family
         << ">, hw_params = {integer_widths = [" << widths
         << "]}} : (!fabric.bits<" << width << ">, !fabric.bits<" << width
         << ">) -> !fabric.bits<" << width
         << "> fabric.yield %value : !fabric.bits<" << width
         << "> } } fabric.yield %pe : !fabric.bits<" << width << "> } }";
  return source;
}

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         unsigned width,
                         FixtureKind kind = FixtureKind::Multiply) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      fabricSource(width, kind), &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  const ::fabric::ResourceContract &resourceContract =
      kind == FixtureKind::UnsupportedContract
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
  FinalizedFabricRoot module =
      take(test, loom::fabric::finalizeFabricRoot(root, store));
  FinalizedFabricRoot system =
      take(test, loom::hardware::test::makeSpatialCoreSystem(module, store, 1));
  auto systemView = take(test, loom::fabric::requireSystemRoot(system.view()));
  auto operations = take(test, enumerateFabricPhysicalOperations(systemView));
  const std::uint64_t expectedOperations =
      kind == FixtureKind::TwoMultiplies ? 2 : 1;
  require(test, operations.size() == expectedOperations,
          "Fabric fixture has the wrong physical operation count");
  return {std::move(system), std::move(operations)};
}

FinalizedConfigurationABI makeAbi(llvm::StringRef test,
                                  const ArtifactStore &store,
                                  const FabricFixture &fixture) {
  return take(
      test,
      finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         fixture.system)),
          store));
}

platform::FinalizedImplementationPlatform
makePlatform(llvm::StringRef test, const ArtifactStore &store,
             platform::ImplementationTarget target = platform::FpgaTarget{
                 platform::FpgaVendor::AmdXilinx,
                 nativeContract().deviceOrderingCode.str()}) {
  return take(test, platform::finalizeImplementationPlatform(
                        {std::move(target), {"default"}}, store));
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

enum class LeafMutation { None, WrongInputWidth };

struct SkeletonFixture final {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::vector<circt::hw::HWModuleGeneratedOp> leaves;
};

SkeletonFixture makeSkeleton(llvm::StringRef test, mlir::MLIRContext &context,
                             const FabricFixture &fixture,
                             const ConfigurationABI &abi,
                             LeafMutation mutation = LeafMutation::None) {
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  std::vector<circt::hw::HWModuleGeneratedOp> leaves;
  for (const ResolvedFabricPhysicalOperation &operation : fixture.operations) {
    std::vector<circt::hw::PortInfo> ports =
        take(test, deriveFabricOperationLeafPorts(builder,
                                                  operation.physicalOccurrence,
                                                  *operation.capability, abi));
    if (mutation == LeafMutation::WrongInputWidth)
      ports.front().type = builder.getIntegerType(8);
    leaves.push_back(circt::hw::HWModuleGeneratedOp::create(
        builder, location,
        mlir::FlatSymbolRefAttr::get(&context,
                                     fabricOperationGeneratorSchemaSymbol),
        builder.getStringAttr("amd_xilinx_multiply_" +
                              std::to_string(leaves.size())),
        ports));
  }
  return {std::move(module), std::move(leaves)};
}

ExternalInputBinding exactInput() {
  const FpgaNativeExternalModuleContract &contract = nativeContract();
  return {
      contract.providerInputSlotRef.str(),
      ToolBundledResourceDependency{contract.stableProviderBuildIdentity.str(),
                                    contract.resourceKey.str()}};
}

ExternalImplementationContractCatalog makeContracts(llvm::StringRef test) {
  ExternalImplementationContractCatalog catalog;
  if (llvm::Error error =
          registerAmdXilinxDsp58ExternalImplementationContract(catalog))
    fail(test, llvm::toString(std::move(error)));
  return catalog;
}

FabricOperationProviderRegistry makeRegistry(llvm::StringRef test) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registerPortableScalarIntegerMultiplyProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerAmdXilinxScalarIntegerMultiplyProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  return registry;
}

std::vector<FabricOperationLeafAssociation>
associations(const SkeletonFixture &skeleton, const FabricFixture &fixture) {
  std::vector<FabricOperationLeafAssociation> result;
  for (const auto [leaf, operation] :
       llvm::zip_equal(skeleton.leaves, fixture.operations))
    result.push_back({leaf, operation.physicalOccurrence});
  return result;
}

std::vector<FabricOperationRecipeBinding>
recipes(const FabricFixture &fixture,
        llvm::ArrayRef<ExternalInputBinding> inputs = {}) {
  std::vector<FabricOperationRecipeBinding> result;
  for (const ResolvedFabricPhysicalOperation &operation : fixture.operations)
    result.push_back(
        {operation.physicalOccurrence, BackendRecipeKey::AmdXilinx,
         std::vector<ExternalInputBinding>(inputs.begin(), inputs.end())});
  return result;
}

std::string syntheticDsp58() {
  return R"sv(module DSP58 #(
  parameter integer ACASCREG = 1,
  parameter integer ADREG = 1,
  parameter integer ALUMODEREG = 1,
  parameter AMULTSEL = "A",
  parameter integer AREG = 1,
  parameter AUTORESET_PATDET = "NO_RESET",
  parameter AUTORESET_PRIORITY = "RESET",
  parameter A_INPUT = "DIRECT",
  parameter integer BCASCREG = 1,
  parameter BMULTSEL = "B",
  parameter integer BREG = 1,
  parameter B_INPUT = "DIRECT",
  parameter integer CARRYINREG = 1,
  parameter integer CARRYINSELREG = 1,
  parameter integer CREG = 1,
  parameter integer DREG = 1,
  parameter DSP_MODE = "INT24",
  parameter integer INMODEREG = 1,
  parameter [3:0] IS_ALUMODE_INVERTED = 4'b0000,
  parameter [0:0] IS_ASYNC_RST_INVERTED = 1'b0,
  parameter [0:0] IS_CARRYIN_INVERTED = 1'b0,
  parameter [0:0] IS_CLK_INVERTED = 1'b0,
  parameter [4:0] IS_INMODE_INVERTED = 5'b00000,
  parameter [2:0] IS_NEGATE_INVERTED = 3'b000,
  parameter [8:0] IS_OPMODE_INVERTED = 9'b000000000,
  parameter [0:0] IS_RSTALLCARRYIN_INVERTED = 1'b0,
  parameter [0:0] IS_RSTALUMODE_INVERTED = 1'b0,
  parameter [0:0] IS_RSTA_INVERTED = 1'b0,
  parameter [0:0] IS_RSTB_INVERTED = 1'b0,
  parameter [0:0] IS_RSTCTRL_INVERTED = 1'b0,
  parameter [0:0] IS_RSTC_INVERTED = 1'b0,
  parameter [0:0] IS_RSTD_INVERTED = 1'b0,
  parameter [0:0] IS_RSTINMODE_INVERTED = 1'b0,
  parameter [0:0] IS_RSTM_INVERTED = 1'b0,
  parameter [0:0] IS_RSTP_INVERTED = 1'b0,
  parameter [57:0] MASK = 58'h0FFFFFFFFFFFFFF,
  parameter integer MREG = 1,
  parameter integer OPMODEREG = 1,
  parameter [57:0] PATTERN = 58'h000000000000000,
  parameter PREADDINSEL = "A",
  parameter integer PREG = 1,
  parameter RESET_MODE = "SYNC",
  parameter [57:0] RND = 58'h000000000000000,
  parameter SEL_MASK = "MASK",
  parameter SEL_PATTERN = "PATTERN",
  parameter USE_MULT = "MULTIPLY",
  parameter USE_PATTERN_DETECT = "NO_PATDET",
  parameter USE_SIMD = "ONE58",
  parameter USE_WIDEXOR = "FALSE",
  parameter XORSIMD = "XOR24_34_58_116"
)(
  output [33:0] ACOUT,
  output [23:0] BCOUT,
  output CARRYCASCOUT,
  output [3:0] CARRYOUT,
  output MULTSIGNOUT,
  output OVERFLOW,
  output [57:0] P,
  output PATTERNBDETECT,
  output PATTERNDETECT,
  output [57:0] PCOUT,
  output UNDERFLOW,
  output [7:0] XOROUT,
  input [33:0] A,
  input [33:0] ACIN,
  input [3:0] ALUMODE,
  input ASYNC_RST,
  input [23:0] B,
  input [23:0] BCIN,
  input [57:0] C,
  input CARRYCASCIN,
  input CARRYIN,
  input [2:0] CARRYINSEL,
  input CEA1,
  input CEA2,
  input CEAD,
  input CEALUMODE,
  input CEB1,
  input CEB2,
  input CEC,
  input CECARRYIN,
  input CECTRL,
  input CED,
  input CEINMODE,
  input CEM,
  input CEP,
  input CLK,
  input [26:0] D,
  input [4:0] INMODE,
  input MULTSIGNIN,
  input [2:0] NEGATE,
  input [8:0] OPMODE,
  input [57:0] PCIN,
  input RSTA,
  input RSTALLCARRYIN,
  input RSTALUMODE,
  input RSTB,
  input RSTC,
  input RSTCTRL,
  input RSTD,
  input RSTINMODE,
  input RSTM,
  input RSTP
);
  logic signed [26:0] multiplier_a;
  logic signed [23:0] multiplier_b;
  logic signed [50:0] product;

  assign multiplier_a = A[26:0];
  assign multiplier_b = B;
  assign product = multiplier_a * multiplier_b;
  assign P = {{7{product[50]}}, product};
  assign ACOUT = '0;
  assign BCOUT = '0;
  assign CARRYCASCOUT = '0;
  assign CARRYOUT = '0;
  assign MULTSIGNOUT = '0;
  assign OVERFLOW = '0;
  assign PATTERNBDETECT = '0;
  assign PATTERNDETECT = '0;
  assign PCOUT = '0;
  assign UNDERFLOW = '0;
  assign XOROUT = '0;
endmodule
)sv";
}

std::string syntheticTestbench() {
  return R"sv(module elastic_i16_multiply(
  input logic clock,
  input logic reset,
  input logic input_valid,
  output logic input_ready,
  input logic output_ready,
  output logic output_valid,
  input logic [15:0] lhs,
  input logic [15:0] rhs,
  output logic [15:0] result
);
  logic [15:0] next_result;

  amd_xilinx_multiply_0 operation(
    .data_input_0(lhs),
    .data_input_1(rhs),
    .data_output_0(next_result)
  );

  assign input_ready = !output_valid || output_ready;
  always_ff @(posedge clock) begin
    if (reset) begin
      output_valid <= 1'b0;
      result <= '0;
    end else if (input_ready) begin
      output_valid <= input_valid;
      if (input_valid)
        result <= next_result;
    end
  end
endmodule

module testbench;
  logic clock;
  logic reset;
  logic input_valid;
  logic input_ready;
  logic output_ready;
  logic output_valid;
  logic [15:0] lhs;
  logic [15:0] rhs;
  logic [15:0] result;
  logic [15:0] direct_lhs;
  logic [15:0] direct_rhs;
  logic [15:0] direct_result;

  elastic_i16_multiply elastic_dut(.*);
  amd_xilinx_multiply_0 direct_dut(
    .data_input_0(direct_lhs),
    .data_input_1(direct_rhs),
    .data_output_0(direct_result)
  );

  initial clock = 0;
  always #5 clock = !clock;

  task automatic check_direct(
      input logic [15:0] left,
      input logic [15:0] right,
      input logic [15:0] expected);
    begin
      direct_lhs = left;
      direct_rhs = right;
      #1;
      if (direct_result !== expected)
        $fatal(1, "direct modular product mismatch");
    end
  endtask

  initial begin
    check_direct(16'h0000, 16'h5a5a, 16'h0000);
    check_direct(16'h0013, 16'h0007, 16'h0085);
    check_direct(16'hffff, 16'h0002, 16'hfffe);
    check_direct(16'h8001, 16'h8001, 16'h0001);

    reset = 1;
    input_valid = 0;
    output_ready = 0;
    lhs = 16'h0003;
    rhs = 16'h0007;
    repeat (2) @(posedge clock);
    #1;
    if (output_valid)
      $fatal(1, "reset did not clear the elastic result slot");

    reset = 0;
    input_valid = 1;
    output_ready = 1;
    @(posedge clock);
    #1;
    if (!output_valid || result !== 16'h0015)
      $fatal(1, "product was not published after one cycle");

    output_ready = 0;
    lhs = 16'hffff;
    rhs = 16'h0002;
    repeat (3) begin
      @(posedge clock);
      #1;
      if (!output_valid || input_ready || result !== 16'h0015)
        $fatal(1, "backpressure changed a retained product");
    end

    output_ready = 1;
    #1;
    if (!input_ready)
      $fatal(1, "release did not admit a same-cycle replacement");
    @(posedge clock);
    #1;
    if (!output_valid || result !== 16'hfffe)
      $fatal(1, "same-cycle replacement lost the modular product");

    input_valid = 0;
    @(posedge clock);
    #1;
    if (output_valid)
      $fatal(1, "released elastic result slot remained occupied");
    $finish;
  end
endmodule
)sv";
}

llvm::Expected<FabricOperationProviderOutput>
specialize(SkeletonFixture &skeleton, const FabricFixture &fixture,
           const FinalizedConfigurationABI &abi,
           const FabricOperationProviderRegistry &registry,
           const ExternalImplementationContractCatalog &contracts,
           const platform::ImplementationPlatform *platform,
           llvm::ArrayRef<ExternalInputBinding> inputs) {
  return specializeFabricOperationLeaves(
      *skeleton.module, abi, associations(skeleton, fixture),
      recipes(fixture, inputs), registry, contracts, platform);
}

void sealedCapabilityAndRegistration(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  const FpgaNativeExternalModuleContract &definition = nativeContract();
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fixture = makeFabric(test, store, 16);
  const auto *capability = fixture.operations.front().capability;
  require(test,
          capability->implementationFamily ==
                  ::fabric::ImplementationFamilyId::ScalarIntegerMultiply &&
              capability->enabledOperationSchemas ==
                  std::vector<::dataflow::OperationSchemaId>{
                      ::dataflow::OperationSchemaId::ArithMulI} &&
              capability->configurationFieldSchema.empty(),
          "sealed Fabric capability changed semantic identity");
  const auto *parameters = std::get_if<::fabric::ScalarIntegerParams>(
      &capability->parameterizedCapability);
  require(test,
          parameters && parameters->integerWidths.valid() &&
              parameters->integerWidths.size() == 1 &&
              parameters->integerWidths.contains(::fabric::IntegerWidth::I16) &&
              parameters->pointerFormats.empty(),
          "sealed Fabric capability changed its exact I16 parameters");
  const std::vector<std::uint8_t> actual =
      take(test, ::fabric::encodeResourceContractRecord(
                     capability->resourceStateAndTimingContract));
  const std::vector<std::uint8_t> expected =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  require(test, actual == expected,
          "sealed Fabric capability changed latency or progress semantics");

  FabricOperationProviderRegistry registry = makeRegistry(test);
  const auto coverage = registry.coverage();
  const auto entry = llvm::find_if(coverage, [](const auto &candidate) {
    return candidate.implementationFamily ==
           ::fabric::ImplementationFamilyId::ScalarIntegerMultiply;
  });
  require(test,
          entry != coverage.end() &&
              entry->recipes ==
                  std::vector<BackendRecipeKey>{
                      BackendRecipeKey::PortableSystemVerilog,
                      BackendRecipeKey::AmdXilinx},
          "portable and AMD/Xilinx recipes do not coexist explicitly");

  ExternalImplementationContractCatalog contracts = makeContracts(test);
  const auto contract = contracts.find(definition.contractRef);
  require(test,
          contract && contract->inputSlots.size() == 1 &&
              contract->inputSlots.front().providerInputSlotRef ==
                  definition.providerInputSlotRef &&
              contract->inputSlots.front().acceptedDependencyKinds ==
                  std::vector<ExternalDependencyKind>{
                      ExternalDependencyKind::ToolBundledResource} &&
              contract->supportedRepresentations ==
                  std::vector<RepresentationRootVariant>{
                      RepresentationRootVariant::Rtl,
                      RepresentationRootVariant::FpgaPhysical} &&
              contract->blackBoxContractRequired &&
              !contract->memoryMacroCapable && contract->validator,
          "DSP58 external contract changed its exact closure");
  for (const std::vector<ExternalInputBinding> &invalidInputs :
       std::array<std::vector<ExternalInputBinding>, 3>{
           std::vector<ExternalInputBinding>{},
           std::vector<ExternalInputBinding>{
               {"wrong_slot",
                ToolBundledResourceDependency{
                    definition.stableProviderBuildIdentity.str(),
                    definition.resourceKey.str()}}},
           std::vector<ExternalInputBinding>{exactInput(), exactInput()}}) {
    auto canonical = contracts.canonicalizeAndValidateInputs(
        definition.contractRef, invalidInputs, RepresentationRootVariant::Rtl);
    require(test, !canonical,
            "DSP58 external contract accepted an invalid input closure");
    llvm::consumeError(canonical.takeError());
  }
}

void exactOccurrenceMaterializesDeterministically(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  const FpgaNativeExternalModuleContract &definition = nativeContract();
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fixture = makeFabric(test, store, 16);
  FinalizedConfigurationABI abi = makeAbi(test, store, fixture);
  auto platform = makePlatform(test, store);
  FabricOperationProviderRegistry registry = makeRegistry(test);
  ExternalImplementationContractCatalog contracts = makeContracts(test);
  const std::array<ExternalInputBinding, 1> inputs{exactInput()};

  std::unique_ptr<mlir::MLIRContext> firstContext = makeCirctContext();
  SkeletonFixture first = makeSkeleton(test, *firstContext, fixture, abi.abi());
  FabricOperationProviderOutput firstOutput =
      take(test, specialize(first, fixture, abi, registry, contracts,
                            &platform.platform(), inputs));
  const std::vector<std::uint8_t> expectedBlackBox(
      definition.blackBoxContractBytes.bytes_begin(),
      definition.blackBoxContractBytes.bytes_end());
  require(test,
          firstOutput.payloads.size() == 1 &&
              firstOutput.payloads.front().role ==
                  PayloadRole::BlackBoxContract &&
              firstOutput.payloads.front().canonicalLogicalName ==
                  definition.blackBoxPayloadLogicalName &&
              firstOutput.payloads.front().bytes == expectedBlackBox &&
              firstOutput.externalImplementationBindings.size() == 1,
          "DSP58 provider omitted its BlackBoxContract or external binding");
  const ExternalImplementationBindingDraft &binding =
      firstOutput.externalImplementationBindings.front();
  require(test,
          binding.providerContractRef == definition.contractRef &&
              binding.externalInputs ==
                  std::vector<ExternalInputBinding>{exactInput()} &&
              binding.fabricResourceRefs ==
                  std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef>{
                      fixture.operations.front().physicalOccurrence} &&
              binding.representationLocators ==
                  std::vector<RepresentationLocator>{
                      {RepresentationObjectKind::Module,
                       definition.moduleName.str()}} &&
              binding.blackBoxContractPayload ==
                  ImplementationPayloadKey{
                      PayloadRole::BlackBoxContract,
                      definition.blackBoxPayloadLogicalName.str()},
          "DSP58 provider changed its exact external binding");
  const auto format =
      take(test, RepresentationFormatDescriptorRef::get(
                     RepresentationFormatKind::SystemVerilogRtl));
  const ImplementationPayload contractPayload{
      PayloadRole::BlackBoxContract,
      definition.blackBoxPayloadLogicalName.str(),
      loom::computeBlobDigest(expectedBlackBox)};
  const ImplementationRepresentationRoot representation{
      RepresentationRootVariant::Rtl,
      std::nullopt,
      format,
      {RepresentationObjectKind::Module, "amd_xilinx_multiply_0"},
      {contractPayload}};
  auto system =
      take(test, loom::fabric::requireSystemRoot(fixture.system.view()));
  std::vector<ExternalImplementationBindingDraft> validatedBindings =
      firstOutput.externalImplementationBindings;
  if (llvm::Error error = contracts.canonicalizeAndValidateBindings(
          validatedBindings, representation, &platform.platform(), system))
    fail(test, llvm::toString(std::move(error)));
  const ImplementationRepresentationRoot physicalRepresentation{
      RepresentationRootVariant::FpgaPhysical,
      RepresentationPhysicalStage::Routed,
      format,
      {RepresentationObjectKind::DeviceResource,
       "device_78637670313830322d76737661353630312d3348502d652d53"},
      {contractPayload}};
  validatedBindings = firstOutput.externalImplementationBindings;
  if (llvm::Error error = contracts.canonicalizeAndValidateBindings(
          validatedBindings, physicalRepresentation, &platform.platform(),
          system))
    fail(test, llvm::toString(std::move(error)));
  std::vector<ExternalImplementationBindingDraft> wrongBindings =
      firstOutput.externalImplementationBindings;
  wrongBindings.front().representationLocators.front().canonicalName =
      "DSP48E2";
  expectExternalContractError(
      test, contracts.canonicalizeAndValidateBindings(
                wrongBindings, representation, &platform.platform(), system));

  auto primitive =
      first.module->lookupSymbol<circt::hw::HWModuleExternOp>("DSP58");
  auto wrapper = first.module->lookupSymbol<circt::hw::HWModuleOp>(
      "amd_xilinx_multiply_0");
  require(test, primitive && wrapper,
          "DSP58 provider did not materialize primitive and wrapper modules");
  require(test,
          primitive.getParametersAttr().size() == 50 &&
              primitive.getPortList().size() == 52,
          "DSP58 declaration changed its exact parameters or ports");
  const auto wrapperPortStorage = wrapper.getPortList();
  const circt::hw::ModulePortInfo wrapperPorts(wrapperPortStorage);
  require(test,
          wrapperPorts.size() == 3 &&
              std::distance(wrapperPorts.getInputs().begin(),
                            wrapperPorts.getInputs().end()) == 2 &&
              std::distance(wrapperPorts.getOutputs().begin(),
                            wrapperPorts.getOutputs().end()) == 1 &&
              mlir::cast<mlir::IntegerType>(wrapperPorts.atInput(0).type)
                      .getWidth() == 16 &&
              mlir::cast<mlir::IntegerType>(wrapperPorts.atInput(1).type)
                      .getWidth() == 16 &&
              mlir::cast<mlir::IntegerType>(wrapperPorts.atOutput(0).type)
                      .getWidth() == 16,
          "DSP58 wrapper changed its exact ABI2 leaf ports");
  std::string firstRtl =
      take(test, lowerAndExportSpecializedSystemVerilog(*first.module));

  std::unique_ptr<mlir::MLIRContext> secondContext = makeCirctContext();
  SkeletonFixture second =
      makeSkeleton(test, *secondContext, fixture, abi.abi());
  FabricOperationProviderOutput secondOutput =
      take(test, specialize(second, fixture, abi, registry, contracts,
                            &platform.platform(), inputs));
  std::string secondRtl =
      take(test, lowerAndExportSpecializedSystemVerilog(*second.module));
  require(test,
          firstRtl == secondRtl && firstOutput.payloads.front().bytes ==
                                       secondOutput.payloads.front().bytes,
          "identical occurrence inputs produced nondeterministic output");
  const llvm::StringRef rtl(firstRtl);
  require(test,
          rtl.contains("DSP58 #(") && rtl.contains(".DSP_MODE(\"INT24\")") &&
              rtl.contains(".USE_MULT(\"MULTIPLY\")") &&
              rtl.contains(".USE_SIMD(\"ONE58\")") &&
              rtl.contains(".AREG(0)") && rtl.contains(".BREG(0)") &&
              rtl.contains(".MREG(0)") && rtl.contains(".PREG(0)") &&
              rtl.contains(".OPMODE") && rtl.contains("9'h5") &&
              !rtl.contains("always"),
          "emitted wrapper changed the verified combinational DSP58 mode");

  auto systemView =
      take(test, loom::fabric::requireSystemRoot(fixture.system.view()));
  require(test, systemView.artifact().accCoreOccurrences().size() == 1,
          "exact fixture changed its SpatialCore occurrence count");
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore{
      systemView.artifact().accCoreOccurrences().front()};
  std::unique_ptr<mlir::MLIRContext> systemContext = makeCirctContext();
  ModuleRootCirctSkeleton systemSkeleton = take(
      test, buildModuleRootCirctSkeleton(*systemContext, spatialCore, abi));
  require(test, systemSkeleton.operationLeaves.size() == 1,
          "common skeleton changed its exact operation leaf count");
  const std::vector<FabricOperationRecipeBinding> systemRecipes = {
      {systemSkeleton.operationLeaves.front().occurrence,
       BackendRecipeKey::AmdXilinx,
       {exactInput()}},
  };
  take(test, specializeFabricOperationLeaves(
                 *systemSkeleton.module, abi, systemSkeleton.operationLeaves,
                 systemRecipes, registry, contracts, &platform.platform()));
  const std::string systemRtl = take(
      test, lowerAndExportSpecializedSystemVerilog(*systemSkeleton.module));

  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "provider_artifacts",
          {{"amd_xilinx_scalar_integer_multiply.sv", firstRtl},
           {"amd_xilinx_scalar_integer_multiply_system.sv", systemRtl},
           {"synthetic_dsp58.sv", syntheticDsp58()},
           {"testbench.sv", syntheticTestbench()}}))
    fail(test, llvm::toString(std::move(error)));
}

void unsupportedBoundaryIsTyped(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  const FpgaNativeExternalModuleContract &definition = nativeContract();
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeRegistry(test);
  ExternalImplementationContractCatalog contracts = makeContracts(test);
  const std::array<ExternalInputBinding, 1> inputs{exactInput()};

  const auto runUnsupported = [&](FabricFixture fixture,
                                  platform::ImplementationTarget target,
                                  llvm::ArrayRef<ExternalInputBinding> bound) {
    FinalizedConfigurationABI abi = makeAbi(test, store, fixture);
    auto platform = makePlatform(test, store, std::move(target));
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton = makeSkeleton(test, *context, fixture, abi.abi());
    expectUnsupported(
        test,
        specialize(skeleton, fixture, abi, registry, contracts,
                   &platform.platform(), bound),
        fixture.operations.front().capability->implementationFamily);
  };

  runUnsupported(makeFabric(test, store, 8),
                 platform::FpgaTarget{platform::FpgaVendor::AmdXilinx,
                                      definition.deviceOrderingCode.str()},
                 inputs);
  runUnsupported(makeFabric(test, store, 16, FixtureKind::MultiplyAllWidths),
                 platform::FpgaTarget{platform::FpgaVendor::AmdXilinx,
                                      definition.deviceOrderingCode.str()},
                 inputs);
  runUnsupported(makeFabric(test, store, 16, FixtureKind::UnsupportedContract),
                 platform::FpgaTarget{platform::FpgaVendor::AmdXilinx,
                                      definition.deviceOrderingCode.str()},
                 inputs);
  runUnsupported(makeFabric(test, store, 16),
                 platform::FpgaTarget{platform::FpgaVendor::AmdXilinx,
                                      "xcvh1782-vsva3340-2MP-e-S"},
                 inputs);
  runUnsupported(makeFabric(test, store, 16),
                 platform::FpgaTarget{platform::FpgaVendor::IntelAltera,
                                      "AGFB027R24C2E2V"},
                 inputs);
  runUnsupported(makeFabric(test, store, 16),
                 platform::AsicTarget{"saed32", "EDK_08_2025"}, inputs);

  FabricFixture valid = makeFabric(test, store, 16);
  FinalizedConfigurationABI validAbi = makeAbi(test, store, valid);
  std::unique_ptr<mlir::MLIRContext> noPlatformContext = makeCirctContext();
  SkeletonFixture noPlatform =
      makeSkeleton(test, *noPlatformContext, valid, validAbi.abi());
  expectUnsupported(test,
                    specialize(noPlatform, valid, validAbi, registry, contracts,
                               nullptr, inputs),
                    ::fabric::ImplementationFamilyId::ScalarIntegerMultiply);

  auto exactPlatform = makePlatform(test, store);
  for (const ExternalInputBinding &wrong : std::array<ExternalInputBinding, 2>{
           ExternalInputBinding{
               definition.providerInputSlotRef.str(),
               ToolBundledResourceDependency{"wrong_provider_build",
                                             definition.resourceKey.str()}},
           ExternalInputBinding{
               definition.providerInputSlotRef.str(),
               ToolBundledResourceDependency{
                   definition.stableProviderBuildIdentity.str(),
                   "unisim:versal:DSP48E2"}}}) {
    std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
    SkeletonFixture skeleton =
        makeSkeleton(test, *context, valid, validAbi.abi());
    const std::array<ExternalInputBinding, 1> wrongInputs{wrong};
    expectUnsupported(test,
                      specialize(skeleton, valid, validAbi, registry, contracts,
                                 &exactPlatform.platform(), wrongInputs),
                      ::fabric::ImplementationFamilyId::ScalarIntegerMultiply);
  }

  FabricFixture add = makeFabric(test, store, 16, FixtureKind::Add);
  FinalizedConfigurationABI addAbi = makeAbi(test, store, add);
  std::unique_ptr<mlir::MLIRContext> addContext = makeCirctContext();
  SkeletonFixture addSkeleton =
      makeSkeleton(test, *addContext, add, addAbi.abi());
  expectUnsupported(test,
                    specialize(addSkeleton, add, addAbi, registry, contracts,
                               &exactPlatform.platform(), inputs),
                    ::fabric::ImplementationFamilyId::ScalarIntegerAddSub);
}

void malformedAbiAndTransactionRollBack(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  const FpgaNativeExternalModuleContract &definition = nativeContract();
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricOperationProviderRegistry registry = makeRegistry(test);
  ExternalImplementationContractCatalog contracts = makeContracts(test);
  auto platform = makePlatform(test, store);
  const std::array<ExternalInputBinding, 1> inputs{exactInput()};

  FabricFixture single = makeFabric(test, store, 16);
  FinalizedConfigurationABI singleAbi = makeAbi(test, store, single);
  std::unique_ptr<mlir::MLIRContext> malformedContext = makeCirctContext();
  SkeletonFixture malformed =
      makeSkeleton(test, *malformedContext, single, singleAbi.abi(),
                   LeafMutation::WrongInputWidth);
  const std::string malformedBefore = moduleText(*malformed.module);
  auto malformedResult = specialize(malformed, single, singleAbi, registry,
                                    contracts, &platform.platform(), inputs);
  require(test, !malformedResult,
          "AMD/Xilinx provider accepted a malformed ABI2 leaf");
  llvm::consumeError(malformedResult.takeError());
  require(test, moduleText(*malformed.module) == malformedBefore,
          "malformed ABI2 leaf mutated the common skeleton");

  FabricFixture repeated =
      makeFabric(test, store, 16, FixtureKind::TwoMultiplies);
  FinalizedConfigurationABI repeatedAbi = makeAbi(test, store, repeated);
  std::unique_ptr<mlir::MLIRContext> repeatedContext = makeCirctContext();
  SkeletonFixture repeatedSkeleton =
      makeSkeleton(test, *repeatedContext, repeated, repeatedAbi.abi());
  std::vector<FabricOperationRecipeBinding> repeatedRecipes =
      recipes(repeated, inputs);
  repeatedRecipes.back().externalInputs.front().dependencyIdentity =
      ToolBundledResourceDependency{
          definition.stableProviderBuildIdentity.str(),
          "unisim:versal:DSP48E2"};
  const std::string repeatedBefore = moduleText(*repeatedSkeleton.module);
  auto result = specializeFabricOperationLeaves(
      *repeatedSkeleton.module, repeatedAbi,
      associations(repeatedSkeleton, repeated), repeatedRecipes, registry,
      contracts, &platform.platform());
  expectUnsupported(test, std::move(result),
                    ::fabric::ImplementationFamilyId::ScalarIntegerMultiply);
  require(test, moduleText(*repeatedSkeleton.module) == repeatedBefore,
          "multi-occurrence failure partially committed specialization");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  sealedCapabilityAndRegistration(root / "sealed");
  exactOccurrenceMaterializesDeterministically(root / "exact");
  unsupportedBoundaryIsTyped(root / "unsupported");
  malformedAbiAndTransactionRollBack(root / "transaction");
  return 0;
}

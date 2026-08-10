#include "ConfigurationABITestSupport.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Providers/FloatSign.h"
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
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted malformed floating sign input");
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

enum class FixtureKind {
  ScalarConfigured,
  ScalarSingleton,
  ScalarUnsupportedContract,
  VectorConfigured,
};

struct FabricFixture final {
  ::fabric::ImplementationFamilyId family;
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
  FinalizedFabricRoot system;
  loom::fabric::FabricPhysicalOccurrenceOwnerRef physicalOccurrence;
};

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store,
                         FixtureKind kind) {
  llvm::StringRef sourceText;
  ::fabric::ImplementationFamilyId family;
  switch (kind) {
  case FixtureKind::ScalarConfigured:
  case FixtureKind::ScalarUnsupportedContract:
    family = ::fabric::ImplementationFamilyId::ScalarFloatSign;
    sourceText = R"mlir(
    module {
      fabric.module @scalar_float_sign(%input: !fabric.bits<64>)
          -> !fabric.bits<64> {
        %pe = fabric.pe [spatial]
            (%pe_input = %input : !fabric.bits<64>) -> !fabric.bits<64> {
          %fu = fabric.fu
              (%fu_input = %pe_input : !fabric.bits<64>)
              -> !fabric.bits<64> {
            %value = fabric.op [@arith.negf, @math.absf] (%fu_input)
              {implementation_family =
                 #fabric.implementation_family<ScalarFloatSign>,
               hw_params = {
                 float_formats = ["f16", "bf16", "f32", "f64"],
                 behavior = {
                   rounding_modes = ["to_nearest_even"],
                   nan_behaviors = ["ieee"],
                   subnormal_behaviors = ["preserve"],
                   signed_zero_behaviors = ["preserve"],
                   fastmath = "none"}}}
              : (!fabric.bits<64>) -> !fabric.bits<64>
            fabric.yield %value : !fabric.bits<64>
          }
        }
        fabric.yield %pe : !fabric.bits<64>
      }
    }
  )mlir";
    break;
  case FixtureKind::ScalarSingleton:
    family = ::fabric::ImplementationFamilyId::ScalarFloatSign;
    sourceText = R"mlir(
    module {
      fabric.module @scalar_float_absolute(%input: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pe_input = %input : !fabric.bits<32>) -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fu_input = %pe_input : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %value = fabric.op [@math.absf] (%fu_input)
              {implementation_family =
                 #fabric.implementation_family<ScalarFloatSign>,
               hw_params = {
                 float_formats = ["f32"],
                 behavior = {
                   rounding_modes = ["to_nearest_even"],
                   nan_behaviors = ["ieee"],
                   subnormal_behaviors = ["preserve"],
                   signed_zero_behaviors = ["preserve"],
                   fastmath = "none"}}}
              : (!fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %value : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir";
    break;
  case FixtureKind::VectorConfigured:
    family = ::fabric::ImplementationFamilyId::FixedVectorFloatSign;
    sourceText = R"mlir(
    module {
      fabric.module @fixed_vector_float_sign(%input: !fabric.bits<128>)
          -> !fabric.bits<128> {
        %pe = fabric.pe [spatial]
            (%pe_input = %input : !fabric.bits<128>) -> !fabric.bits<128> {
          %fu = fabric.fu
              (%fu_input = %pe_input : !fabric.bits<128>)
              -> !fabric.bits<128> {
            %value = fabric.op [@arith.negf, @math.absf] (%fu_input)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorFloatSign>,
               hw_params = {
                 element_formats = ["f16", "bf16", "f32", "f64"],
                 behavior = {
                   rounding_modes = ["to_nearest_even"],
                   nan_behaviors = ["ieee"],
                   subnormal_behaviors = ["preserve"],
                   signed_zero_behaviors = ["preserve"],
                   fastmath = "none"},
                 max_payload_bits = 128 : i32}}
              : (!fabric.bits<128>) -> !fabric.bits<128>
            fabric.yield %value : !fabric.bits<128>
          }
        }
        fabric.yield %pe : !fabric.bits<128>
      }
    }
  )mlir";
    break;
  }

  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  const ::fabric::ResourceContract &resourceContract =
      kind == FixtureKind::ScalarUnsupportedContract
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
      if (capability.implementationFamily != family)
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
              "System has no physical floating sign occurrence");
      return FabricFixture{family, std::move(fabric), occurrence,
                           std::move(system), physical->physicalOccurrence};
    }
  }
  fail(test, "Fabric fixture has no floating sign occurrence");
}

unsigned activeRepresentationWidth(
    const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  mlir::Type type = point.representativeActor.type.getInput(0);
  if (auto vector = llvm::dyn_cast<mlir::VectorType>(type))
    type = vector.getElementType();
  return mlir::cast<mlir::FloatType>(type).getWidth();
}

std::uint8_t
physicalCode(const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  const unsigned width = activeRepresentationWidth(point);
  const bool absolute =
      point.representativeActor.schema == dataflow::OperationSchemaId::MathAbsF;
  const unsigned widthOrdinal = width == 16 ? 0 : width == 32 ? 1 : 2;
  return static_cast<std::uint8_t>(1 + 2 * widthOrdinal + absolute);
}

enum class AbiKind { Complete, MissingMode, ExtraMode };

ConfigurationABIDraft
makeConfigurationAbiDraft(llvm::StringRef test, const FabricFixture &fixture,
                          AbiKind kind = AbiKind::Complete) {
  const auto *capability =
      fixture.fabric.view().resolvedFabricOpCapability(fixture.occurrence);
  require(test, capability != nullptr, "Fabric capability did not resolve");
  if (capability->configurationFieldSchema.empty())
    return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                          fixture.system));
  require(test, capability->configurationFieldSchema.size() == 1,
          "floating sign fixture has an unexpected field count");
  auto relation =
      take(test, capability->resolveSemanticFieldRelation(fabricContext()));
  require(test,
          relation.kind() ==
                  ::fabric::FabricOpSemanticFieldRelationKind::Finite &&
              relation.finiteBehaviorDomain().size() == 6,
          "floating sign relation did not expose six width/role modes");

  std::vector<FiniteCodebookEntry> entries;
  std::vector<std::uint8_t> inactiveValue;
  for (const auto &point : relation.finiteBehaviorDomain()) {
    require(test, point.semanticConfiguration.has_value(),
            "configured floating sign mode has no semantic value");
    const std::uint8_t code = physicalCode(point);
    if (code == 6)
      inactiveValue.assign(point.semanticConfiguration->bytes().begin(),
                           point.semanticConfiguration->bytes().end());
    if (kind == AbiKind::MissingMode && code == 5)
      continue;
    entries.push_back(
        {std::vector<std::uint8_t>(point.semanticConfiguration->bytes().begin(),
                                   point.semanticConfiguration->bytes().end()),
         {code}});
  }
  require(test, !inactiveValue.empty(),
          "floating sign relation has no absolute f64 mode");
  if (kind == AbiKind::ExtraMode)
    entries.push_back({{0xff}, {0x00}});

  auto physicalField =
      take(test, loom::hardware::test::qualifyPhysicalConfigurationField(
                     fixture.physicalOccurrence,
                     capability->configurationFieldSchema.front().ordinal));
  loom::hardware::test::ConfigurationFieldEncodingOverride field{
      physicalField, FiniteCodebookEncoding{3, std::move(entries)},
      std::move(inactiveValue)};
  return take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                        fixture.system, {std::move(field)}));
}

FinalizedConfigurationABI makeConfigurationAbi(llvm::StringRef test,
                                               const ArtifactStore &store,
                                               const FabricFixture &fixture) {
  return take(test, finalizeConfigurationABI(
                        makeConfigurationAbiDraft(test, fixture), store));
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
    require(test, ports.size() == 3,
            "configured floating sign leaf did not have three ports");
    ports[1].type = builder.getI2Type();
  }
  const llvm::StringRef name =
      fabric.family == ::fabric::ImplementationFamilyId::ScalarFloatSign
          ? "scalar_float_sign"
          : "fixed_vector_float_sign";
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr(name), ports);
  return SkeletonFixture{std::move(module), leaf};
}

loom::hardware::test::PortableProviderConformance
specialize(llvm::StringRef test, SkeletonFixture skeleton,
           const FabricFixture &fabric,
           const FinalizedConfigurationABI &configurationAbi) {
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableFloatSignProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, fabric.physicalOccurrence}}};
  auto conformance =
      take(test, loom::hardware::test::specializeAndExportPortableProvider(
                     std::move(module), configurationAbi, registry,
                     externalContracts));
  require(test,
          conformance.providerOutput.payloads.empty() &&
              conformance.providerOutput.activityPoints.empty() &&
              conformance.providerOutput.externalImplementationBindings.empty(),
          "portable floating sign provider emitted implementation metadata");
  return conformance;
}

std::string emit(llvm::StringRef test, const ArtifactStore &store,
                 FixtureKind kind) {
  FabricFixture fabric = makeFabric(test, store, kind);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  return specialize(test, std::move(skeleton), fabric, abi).systemVerilog;
}

std::string testbench() {
  return R"sv(module testbench;
  logic [63:0] scalar_input;
  logic [2:0] scalar_config;
  wire [63:0] scalar_output;
  logic [127:0] vector_input;
  logic [2:0] vector_config;
  wire [127:0] vector_output;

  scalar_float_sign scalar_dut(
      .data_input_0(scalar_input), .config_0(scalar_config),
      .data_output_0(scalar_output));
  fixed_vector_float_sign vector_dut(
      .data_input_0(vector_input), .config_0(vector_config),
      .data_output_0(vector_output));

  task automatic check_scalar(
      input logic [63:0] value, input logic [2:0] mode,
      input logic [63:0] expected);
    begin
      scalar_input = value;
      scalar_config = mode;
      #1;
      if (scalar_output !== expected)
        $fatal(1, "scalar sign oracle mismatch mode=%0d input=%h got=%h expected=%h",
               mode, value, scalar_output, expected);
    end
  endtask

  task automatic check_vector16(
      input logic [127:0] value, input logic [2:0] mode,
      input logic absolute);
    logic [127:0] expected;
    integer lane;
    begin
      expected = value;
      for (lane = 0; lane < 8; lane = lane + 1)
        expected[lane * 16 + 15] =
            absolute ? 1'b0 : ~expected[lane * 16 + 15];
      vector_input = value;
      vector_config = mode;
      #1;
      if (vector_output !== expected)
        $fatal(1, "vector f16 sign oracle mismatch mode=%0d", mode);
    end
  endtask

  task automatic check_vector32(
      input logic [127:0] value, input logic [2:0] mode,
      input logic absolute);
    logic [127:0] expected;
    integer lane;
    begin
      expected = value;
      for (lane = 0; lane < 4; lane = lane + 1)
        expected[lane * 32 + 31] =
            absolute ? 1'b0 : ~expected[lane * 32 + 31];
      vector_input = value;
      vector_config = mode;
      #1;
      if (vector_output !== expected)
        $fatal(1, "vector f32 sign oracle mismatch mode=%0d", mode);
    end
  endtask

  task automatic check_vector64(
      input logic [127:0] value, input logic [2:0] mode,
      input logic absolute);
    logic [127:0] expected;
    integer lane;
    begin
      expected = value;
      for (lane = 0; lane < 2; lane = lane + 1)
        expected[lane * 64 + 63] =
            absolute ? 1'b0 : ~expected[lane * 64 + 63];
      vector_input = value;
      vector_config = mode;
      #1;
      if (vector_output !== expected)
        $fatal(1, "vector f64 sign oracle mismatch mode=%0d", mode);
    end
  endtask

  initial begin
    scalar_input = '0;
    scalar_config = '0;
    vector_input = '0;
    vector_config = '0;

    check_scalar(64'hffff_ffff_ffff_0000, 3'd1,
                 64'h0000_0000_0000_8000);
    check_scalar(64'hffff_ffff_ffff_8000, 3'd1,
                 64'h0000_0000_0000_0000);
    check_scalar(64'hffff_ffff_ffff_7e35, 3'd1,
                 64'h0000_0000_0000_fe35);
    check_scalar(64'hffff_ffff_ffff_fe35, 3'd2,
                 64'h0000_0000_0000_7e35);
    check_scalar(64'hffff_ffff_8000_0000, 3'd3,
                 64'h0000_0000_0000_0000);
    check_scalar(64'hffff_ffff_7fc1_2345, 3'd3,
                 64'h0000_0000_ffc1_2345);
    check_scalar(64'hffff_ffff_ffc1_2345, 3'd4,
                 64'h0000_0000_7fc1_2345);
    check_scalar(64'h0000_0000_0000_0000, 3'd5,
                 64'h8000_0000_0000_0000);
    check_scalar(64'h7ff8_1234_5678_9abc, 3'd5,
                 64'hfff8_1234_5678_9abc);
    check_scalar(64'hfff8_1234_5678_9abc, 3'd6,
                 64'h7ff8_1234_5678_9abc);
    check_scalar(64'hbff0_0000_0000_0000, 3'd0,
                 64'h3ff0_0000_0000_0000);

    check_vector16(
        {16'hfe35, 16'h7e35, 16'h8000, 16'h0000,
         16'hbc00, 16'h3c00, 16'hffff, 16'h0001}, 3'd1, 1'b0);
    check_vector16(
        {16'hfe35, 16'h7e35, 16'h8000, 16'h0000,
         16'hbc00, 16'h3c00, 16'hffff, 16'h0001}, 3'd2, 1'b1);
    check_vector32(
        {32'hffc12345, 32'h7fc12345, 32'h80000000, 32'h3f800000},
        3'd3, 1'b0);
    check_vector32(
        {32'hffc12345, 32'h7fc12345, 32'h80000000, 32'h3f800000},
        3'd4, 1'b1);
    check_vector64(
        {64'hfff8123456789abc, 64'h0000000000000000}, 3'd5, 1'b0);
    check_vector64(
        {64'hfff8123456789abc, 64'h8000000000000000}, 3'd6, 1'b1);
    $display("portable float sign PASS");
    $finish;
  end
endmodule
)sv";
}

std::string synthesisTop() {
  return R"sv(module float_sign_synthesis_top(
    input [63:0] scalar_input,
    input [2:0] scalar_config,
    input [127:0] vector_input,
    input [2:0] vector_config,
    output [63:0] scalar_output,
    output [127:0] vector_output);
  scalar_float_sign scalar_dut(
      .data_input_0(scalar_input), .config_0(scalar_config),
      .data_output_0(scalar_output));
  fixed_vector_float_sign vector_dut(
      .data_input_0(vector_input), .config_0(vector_config),
      .data_output_0(vector_output));
endmodule
)sv";
}

std::string yosysScript() {
  return R"ys(read_verilog -sv scalar_float_sign.sv fixed_vector_float_sign.sv synthesis_top.sv
hierarchy -check -top float_sign_synthesis_top
proc
opt_clean
check -assert
select -assert-count 6 t:$not
select -assert-count 10 t:$eq
select -assert-count 10 t:$mux
select -assert-none t:$dff t:$dlatch t:$memrd t:$memwr t:$meminit t:$mem_v2
synth -top float_sign_synthesis_top
check -assert
select -assert-none t:$_DFF_* t:$_SDFF_* t:$_DLATCH_*
stat
)ys";
}

void configuredSemanticsAndDeterminism(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  const std::string scalarFirst =
      emit(test, store, FixtureKind::ScalarConfigured);
  const std::string scalarSecond =
      emit(test, store, FixtureKind::ScalarConfigured);
  const std::string scalarThird =
      emit(test, store, FixtureKind::ScalarConfigured);
  require(test, scalarFirst == scalarSecond && scalarSecond == scalarThird,
          "scalar floating sign RTL is not deterministic");
  const std::string vectorFirst =
      emit(test, store, FixtureKind::VectorConfigured);
  const std::string vectorSecond =
      emit(test, store, FixtureKind::VectorConfigured);
  const std::string vectorThird =
      emit(test, store, FixtureKind::VectorConfigured);
  require(test, vectorFirst == vectorSecond && vectorSecond == vectorThird,
          "vector floating sign RTL is not deterministic");

  if (llvm::Error error = loom::hardware::test::writePortableProviderArtifacts(
          root / "artifacts", {{{"scalar_float_sign.sv"}, scalarFirst},
                               {{"fixed_vector_float_sign.sv"}, vectorFirst},
                               {{"testbench.sv"}, testbench()},
                               {{"synthesis_top.sv"}, synthesisTop()},
                               {{"portable_float_sign.ys"}, yosysScript()}}))
    fail(test, llvm::toString(std::move(error)));
}

void singletonHasNoConfigurationSelector(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  const std::string rtl = emit(test, store, FixtureKind::ScalarSingleton);
  require(test, !llvm::StringRef(rtl).contains("config_0"),
          "singleton floating sign behavior retained a selector");
}

void typedUnsupportedIsTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricFixture fabric =
      makeFabric(test, store, FixtureKind::ScalarUnsupportedContract);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());

  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableFloatSignProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  ModuleRootCirctSkeleton module{std::move(skeleton.module),
                                 {{skeleton.leaf, fabric.physicalOccurrence}}};
  auto result = loom::hardware::test::specializeAndExportPortableProvider(
      std::move(module), abi, registry, externalContracts);
  require(test, !result, "unsupported resource contract specialized");
  bool classified = false;
  llvm::handleAllErrors(
      result.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        classified = error.implementationFamily() == fabric.family &&
                     error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, "unexpected unsupported error: " + error.message());
      });
  require(test, classified, "resource rejection was not typed Unsupported");
}

llvm::Expected<FabricOperationProviderOutput>
placeholderProvider(FabricOperationProviderRequest) {
  return FabricOperationProviderOutput{};
}

bool hasCoverage(const FabricOperationProviderRegistry &registry,
                 ::fabric::ImplementationFamilyId family) {
  return llvm::any_of(registry.coverage(), [&](const auto &entry) {
    return entry.implementationFamily == family && entry.recipes.size() == 1 &&
           entry.recipes.front() == BackendRecipeKey::PortableSystemVerilog;
  });
}

void packageRegistrationRollsBack() {
  const llvm::StringRef test = __func__;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error =
          registry.add({::fabric::ImplementationFamilyId::FixedVectorFloatSign,
                        BackendRecipeKey::PortableSystemVerilog,
                        {},
                        placeholderProvider}))
    fail(test, llvm::toString(std::move(error)));
  llvm::Error error = registerPortableFloatSignProviders(registry);
  require(test, static_cast<bool>(error),
          "duplicate package registration unexpectedly succeeded");
  llvm::consumeError(std::move(error));
  require(
      test,
      !hasCoverage(registry,
                   ::fabric::ImplementationFamilyId::ScalarFloatSign) &&
          hasCoverage(registry,
                      ::fabric::ImplementationFamilyId::FixedVectorFloatSign),
      "failed package registration did not roll back");
}

void malformedInputsFailClosed(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root / "store");
  ArtifactStore store((root / "store").string());
  FabricFixture fabric = makeFabric(test, store, FixtureKind::ScalarConfigured);

  expectError(
      test,
      finalizeConfigurationABI(
          makeConfigurationAbiDraft(test, fabric, AbiKind::MissingMode), store),
      "does not equal its Fabric relation");
  expectError(
      test,
      finalizeConfigurationABI(
          makeConfigurationAbiDraft(test, fabric, AbiKind::ExtraMode), store),
      "outside the finite behavior domain");

  FinalizedConfigurationABI complete =
      makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> portContext = makeCirctContext();
  SkeletonFixture wrongPorts =
      makeSkeleton(test, *portContext, fabric, complete.abi(), true);
  ModuleRootCirctSkeleton wrongPortModule{
      std::move(wrongPorts.module),
      {{wrongPorts.leaf, fabric.physicalOccurrence}}};
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableFloatSignProviders(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  expectError(
      test,
      loom::hardware::test::specializeAndExportPortableProvider(
          std::move(wrongPortModule), complete, registry, externalContracts),
      "derived contract");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  configuredSemanticsAndDeterminism(root / "configured");
  singletonHasNoConfigurationSelector(root / "singleton");
  typedUnsupportedIsTransactional(root / "unsupported");
  packageRegistrationRollsBack();
  malformedInputsFailClosed(root / "malformed");
  return 0;
}

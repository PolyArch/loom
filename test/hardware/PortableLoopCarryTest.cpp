#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/Providers/LoopCarry.h"

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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    fail(test, "accepted invalid portable loop carry input");
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

enum class ResourceContractKind { LoopCarry, OneCycleElastic };

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
};

FabricFixture makeFabricWithWidths(
    llvm::StringRef test, const ArtifactStore &store, unsigned phaseWidth,
    unsigned initWidth, unsigned nextWidth, unsigned outputWidth,
    ResourceContractKind contractKind = ResourceContractKind::LoopCarry) {
  const std::string phase = std::to_string(phaseWidth);
  const std::string init = std::to_string(initWidth);
  const std::string next = std::to_string(nextWidth);
  const std::string output = std::to_string(outputWidth);
  const std::string outer = std::to_string(
      std::max({1U, phaseWidth, initWidth, nextWidth, outputWidth}));
  const auto inputBinding = [&](const std::string &inner) {
    const std::string boundary = ": !fabric.bits<" + outer + ">";
    return inner == outer ? boundary
                          : boundary + " to !fabric.bits<" + inner + ">";
  };
  const std::string phaseBinding = inputBinding(phase);
  const std::string initBinding = inputBinding(init);
  const std::string nextBinding = inputBinding(next);
  const std::string yieldBinding =
      output == outer
          ? ": !fabric.bits<" + output + ">"
          : ": !fabric.bits<" + output + "> to !fabric.bits<" + outer + ">";
  const std::string sourceText =
      "module { fabric.module @loop_carry(%phase: !fabric.bits<" + outer +
      ">, %init: !fabric.bits<" + outer + ">, %next: !fabric.bits<" + outer +
      ">) -> !fabric.bits<" + outer +
      "> { %pe = fabric.pe [spatial] (%p = %phase : !fabric.bits<" + outer +
      ">, %i = %init : !fabric.bits<" + outer +
      ">, %n = %next : !fabric.bits<" + outer + ">) -> !fabric.bits<" + outer +
      "> { %fu = fabric.fu (%fp = %p " + phaseBinding + ", %fi = %i " +
      initBinding + ", %fn = %n " + nextBinding + ") -> !fabric.bits<" + outer +
      "> { %value = fabric.op [@dataflow.carry] (%fp, %fi, %fn) "
      "{implementation_family = #fabric.implementation_family<LoopCarry>, "
      "hw_params = {}} : (!fabric.bits<" +
      phase + ">, !fabric.bits<" + init + ">, !fabric.bits<" + next +
      ">) -> !fabric.bits<" + output + "> fabric.yield %value " + yieldBinding +
      " } } fabric.yield %pe : !fabric.bits<" + outer + "> } }";
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  if (!source)
    fail(test, "could not parse Fabric fixture: " + sourceText);

  const ::fabric::ResourceContract &contract =
      contractKind == ResourceContractKind::LoopCarry
          ? ::fabric::loopCarryOperationResourceContract()
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
      if (capability.implementationFamily !=
          ::fabric::ImplementationFamilyId::LoopCarry)
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), capability.occurrence, fuOccurrence));
      return FabricFixture{std::move(fabric), occurrence};
    }
  }
  fail(test, "Fabric fixture has no loop carry occurrence");
}

FabricFixture makeFabric(
    llvm::StringRef test, const ArtifactStore &store, unsigned payloadWidth = 8,
    unsigned phaseWidth = 8,
    ResourceContractKind contractKind = ResourceContractKind::LoopCarry) {
  return makeFabricWithWidths(test, store, phaseWidth, payloadWidth,
                              payloadWidth, payloadWidth, contractKind);
}

FinalizedConfigurationABI makeConfigurationAbi(llvm::StringRef test,
                                               const ArtifactStore &store,
                                               const FabricFixture &fixture) {
  return take(
      test, finalizeConfigurationABI(
                ConfigurationABIDraft{fixture.fabric.reference(), {}}, store));
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
                             bool wrongStateWidth = false) {
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
  if (wrongStateWidth) {
    const auto state = llvm::find_if(ports, [](const auto &port) {
      return port.getName() == "state_current";
    });
    require(test, state != ports.end(), "loop carry leaf has no current state");
    state->type = builder.getIntegerType(2);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("loop_carry"), ports);
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
  if (llvm::Error error = registerPortableLoopCarryProvider(registry))
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
          "portable loop carry emitted external implementation state");
  return take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
}

void writeYosysScript(const std::filesystem::path &root) {
  std::ofstream(root / "portable_loop_carry.ys") << R"ys(
read_verilog loop_carry.sv
hierarchy -check -top loop_carry
proc
opt
check -assert
select -assert-none loop_carry/t:$*ff* loop_carry/t:$*latch* loop_carry/t:$_*FF* loop_carry/t:$_*LATCH* loop_carry/t:$mem* loop_carry/m:*
synth -top loop_carry
check -assert
select -assert-none loop_carry/t:$*ff* loop_carry/t:$*latch* loop_carry/t:$_*FF* loop_carry/t:$_*LATCH* loop_carry/t:$mem* loop_carry/m:*
stat
)ys";
}

void canonicalBoundaryAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const auto ports = skeleton.leaf.getPortList();
  const std::vector<llvm::StringRef> expected{
      "data_input_0",   "data_input_1",  "data_input_2",   "valid_input_0",
      "valid_input_1",  "valid_input_2", "ready_output_0", "state_current",
      "ready_input_0",  "ready_input_1", "ready_input_2",  "data_output_0",
      "valid_output_0", "state_next",    "state_write"};
  require(test, ports.size() == expected.size(),
          "loop carry leaf has the wrong port count");
  for (auto [index, name] : llvm::enumerate(expected))
    require(test, ports[index].getName() == name,
            "loop carry leaf ports are not canonical");
  require(test,
          ports[7].type == mlir::IntegerType::get(context.get(), 1) &&
              ports[13].type == mlir::IntegerType::get(context.get(), 1) &&
              ports[14].type == mlir::IntegerType::get(context.get(), 1),
          "loop carry state boundary is not one bit");
  require(test,
          encodeLoopCarryOperationLeafState(
              ::dataflow::semantics::CarrySemanticState::Initial)
                  .isZero() &&
              encodeLoopCarryOperationLeafState(
                  ::dataflow::semantics::CarrySemanticState::Running)
                  .isOne(),
          "loop carry state encoding does not preserve canonical reset");

  const std::string rtl = specialize(test, skeleton, fabric, abi);
  const llvm::StringRef text(rtl);
  require(test,
          text.contains("state_current") && text.contains("state_next") &&
              text.contains("state_write") && !text.contains("always_ff") &&
              !text.contains("posedge"),
          "loop carry provider did not remain a combinational state transform");
  std::ofstream(root / "loop_carry.sv") << rtl;
  std::ofstream(root / "testbench.sv") << R"sv(
module testbench;
  logic [7:0] data_input_0;
  logic [7:0] data_input_1;
  logic [7:0] data_input_2;
  logic       valid_input_0;
  logic       valid_input_1;
  logic       valid_input_2;
  logic       ready_output_0;
  logic       state_current;
  logic       ready_input_0;
  logic       ready_input_1;
  logic       ready_input_2;
  logic [7:0] data_output_0;
  logic       valid_output_0;
  logic       state_next;
  logic       state_write;
  integer     control;
  logic       expected_phase_ready;
  logic       expected_init_ready;
  logic       expected_next_ready;
  logic       expected_output_valid;
  logic       expected_state_write;
  logic       expected_state_next;

  loop_carry dut(.*);

  task automatic expect_signals(
      input logic phase_ready,
      input logic init_ready,
      input logic next_ready,
      input logic output_valid,
      input logic write_state,
      input logic next_state);
    #1;
    if (ready_input_0 !== phase_ready || ready_input_1 !== init_ready ||
        ready_input_2 !== next_ready || valid_output_0 !== output_valid ||
        state_write !== write_state || state_next !== next_state)
      $fatal(1, "unexpected carry control signals");
  endtask

  initial begin
    for (control = 0; control < 64; control = control + 1) begin
      state_current = control[5];
      data_input_0 = {control[3:0], 3'b101, control[4]};
      valid_input_0 = control[3];
      valid_input_1 = control[2];
      valid_input_2 = control[1];
      ready_output_0 = control[0];
      data_input_1 = 8'h20 ^ control[7:0];
      data_input_2 = 8'h80 ^ control[7:0];
      expected_phase_ready = state_current &&
          (!data_input_0[0] || (valid_input_2 && ready_output_0));
      expected_init_ready = !state_current && ready_output_0;
      expected_next_ready = state_current && data_input_0[0] &&
          valid_input_0 && ready_output_0;
      expected_output_valid = !state_current ? valid_input_1 :
          (data_input_0[0] && valid_input_0 && valid_input_2);
      expected_state_write =
          (!state_current && valid_input_1 && ready_output_0) ||
          (state_current && valid_input_0 &&
           (!data_input_0[0] || (valid_input_2 && ready_output_0)));
      expected_state_next = expected_state_write
          ? (state_current ? data_input_0[0] : 1'b1)
          : state_current;
      expect_signals(expected_phase_ready, expected_init_ready,
                     expected_next_ready, expected_output_valid,
                     expected_state_write, expected_state_next);
      if (valid_output_0 &&
          data_output_0 !== (state_current ? data_input_2 : data_input_1))
        $fatal(1, "carry forwarded the wrong payload source");
    end

    data_input_0 = 8'h80;
    data_input_1 = 8'h12;
    data_input_2 = 8'h34;
    valid_input_0 = 0;
    valid_input_1 = 0;
    valid_input_2 = 0;
    ready_output_0 = 1;
    state_current = 0;
    expect_signals(0, 1, 0, 0, 0, 0);

    valid_input_1 = 1;
    ready_output_0 = 0;
    expect_signals(0, 0, 0, 1, 0, 0);
    if (data_output_0 !== 8'h12) $fatal(1, "stalled init payload changed");

    ready_output_0 = 1;
    expect_signals(0, 1, 0, 1, 1, 1);
    if (data_output_0 !== 8'h12) $fatal(1, "init payload was not forwarded");

    state_current = state_next;
    valid_input_1 = 1;
    valid_input_0 = 0;
    expect_signals(1, 0, 0, 0, 0, 1);

    data_input_0 = 8'h81;
    valid_input_0 = 1;
    valid_input_2 = 0;
    expect_signals(0, 0, 1, 0, 0, 1);

    valid_input_2 = 1;
    ready_output_0 = 0;
    expect_signals(0, 0, 0, 1, 0, 1);
    if (data_output_0 !== 8'h34) $fatal(1, "stalled next payload changed");

    ready_output_0 = 1;
    expect_signals(1, 0, 1, 1, 1, 1);
    if (data_output_0 !== 8'h34) $fatal(1, "next payload was not forwarded");

    data_input_0 = 8'h80;
    ready_output_0 = 0;
    expect_signals(1, 0, 0, 0, 1, 0);

    state_current = state_next;
    valid_input_0 = 1;
    valid_input_1 = 0;
    valid_input_2 = 1;
    ready_output_0 = 1;
    expect_signals(0, 1, 0, 0, 0, 0);
    $finish;
  end
endmodule
)sv";
  writeYosysScript(root);
}

void mixedPhysicalWidthsPreserveLowBits(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabricWithWidths(test, store, 8, 16, 4, 8);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  std::ofstream(root / "loop_carry.sv") << rtl;
  std::ofstream(root / "testbench.sv") << R"sv(
module testbench;
  logic [7:0]  data_input_0;
  logic [15:0] data_input_1;
  logic [3:0]  data_input_2;
  logic        valid_input_0;
  logic        valid_input_1;
  logic        valid_input_2;
  logic        ready_output_0;
  logic        state_current;
  logic        ready_input_0;
  logic        ready_input_1;
  logic        ready_input_2;
  logic [7:0]  data_output_0;
  logic        valid_output_0;
  logic        state_next;
  logic        state_write;

  loop_carry dut(.*);

  initial begin
    data_input_0 = 8'h80;
    data_input_1 = 16'habcd;
    data_input_2 = 4'ha;
    valid_input_0 = 1;
    valid_input_1 = 1;
    valid_input_2 = 1;
    ready_output_0 = 1;
    state_current = 0;
    #1;
    if (!ready_input_1 || ready_input_0 || ready_input_2 ||
        !valid_output_0 || data_output_0 !== 8'hcd ||
        !state_write || !state_next)
      $fatal(1, "init low-bit truncation failed");

    state_current = 1;
    data_input_0 = 8'h81;
    #1;
    if (!ready_input_0 || ready_input_1 || !ready_input_2 ||
        !valid_output_0 || data_output_0 !== 8'h0a ||
        !state_write || !state_next)
      $fatal(1, "next zero extension failed");

    data_input_0 = 8'h80;
    ready_output_0 = 0;
    #1;
    if (!ready_input_0 || ready_input_1 || ready_input_2 ||
        valid_output_0 || !state_write || state_next)
      $fatal(1, "close depended on inactive output capacity");
    $finish;
  end
endmodule
)sv";
  writeYosysScript(root);
}

void zeroWidthPayloadNeedsOnlyHandshake(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store, 0);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const auto ports = skeleton.leaf.getPortList();
  require(test,
          llvm::none_of(ports,
                        [](const auto &port) {
                          return port.getName() == "data_input_1" ||
                                 port.getName() == "data_input_2" ||
                                 port.getName() == "data_output_0";
                        }),
          "zero-width carry retained payload signals");
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test,
          llvm::StringRef(rtl).contains("valid_output_0") &&
              !llvm::StringRef(rtl).contains("data_output_0"),
          "zero-width carry did not preserve token-only handshaking");
}

void invalidInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());

  FabricFixture malformedContract =
      makeFabric(test, store, 8, 1, ResourceContractKind::OneCycleElastic);
  FinalizedConfigurationABI malformedAbi =
      makeConfigurationAbi(test, store, malformedContract);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture contractSkeleton =
      makeSkeleton(test, *context, malformedContract, malformedAbi.abi());
  const std::string beforeContract = moduleText(*contractSkeleton.module);
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableLoopCarryProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;
  const std::vector<FabricOperationLeafAssociation> contractAssociations = {
      {contractSkeleton.leaf, malformedContract.occurrence}};
  const std::vector<FabricOperationRecipeBinding> contractRecipes = {
      {malformedContract.occurrence,
       BackendRecipeKey::PortableSystemVerilog,
       {}}};
  expectError(test,
              specializeFabricOperationLeaves(
                  *contractSkeleton.module, malformedContract.fabric,
                  malformedAbi, contractAssociations, contractRecipes, registry,
                  externalContracts),
              "resource contract");
  require(test, moduleText(*contractSkeleton.module) == beforeContract,
          "resource-contract failure modified the common skeleton");

  FabricFixture valid = makeFabric(test, store);
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  SkeletonFixture malformedLeaf =
      makeSkeleton(test, *context, valid, validAbi.abi(), true);
  const std::string beforeLeaf = moduleText(*malformedLeaf.module);
  const std::vector<FabricOperationLeafAssociation> leafAssociations = {
      {malformedLeaf.leaf, valid.occurrence}};
  const std::vector<FabricOperationRecipeBinding> leafRecipes = {
      {valid.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectError(test,
              specializeFabricOperationLeaves(
                  *malformedLeaf.module, valid.fabric, validAbi,
                  leafAssociations, leafRecipes, registry, externalContracts),
              "leaf port");
  require(test, moduleText(*malformedLeaf.module) == beforeLeaf,
          "leaf-contract failure modified the common skeleton");

  FabricFixture zeroPhase = makeFabric(test, store, 8, 0);
  FinalizedConfigurationABI zeroPhaseAbi =
      makeConfigurationAbi(test, store, zeroPhase);
  SkeletonFixture zeroPhaseSkeleton =
      makeSkeleton(test, *context, zeroPhase, zeroPhaseAbi.abi());
  const std::vector<FabricOperationLeafAssociation> phaseAssociations = {
      {zeroPhaseSkeleton.leaf, zeroPhase.occurrence}};
  const std::vector<FabricOperationRecipeBinding> phaseRecipes = {
      {zeroPhase.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectError(test,
              specializeFabricOperationLeaves(
                  *zeroPhaseSkeleton.module, zeroPhase.fabric, zeroPhaseAbi,
                  phaseAssociations, phaseRecipes, registry, externalContracts),
              "phase input");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  canonicalBoundaryAndArtifacts(root);
  mixedPhysicalWidthsPreserveLowBits(root / "mixed_width");
  zeroWidthPayloadNeedsOnlyHandshake(root / "zero_width");
  invalidInputsAreTransactional(root / "invalid");
  return 0;
}

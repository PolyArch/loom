#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/Providers/LoopInvariant.h"

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
    fail(test, "accepted invalid portable loop invariant input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

void expectTypedUnsupported(llvm::StringRef test,
                            llvm::Expected<FabricOperationProviderOutput> value,
                            llvm::StringRef description) {
  require(test, !value, std::string("provider accepted ") + description.str());
  bool typedUnsupported = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const FabricOperationProviderUnsupportedError &error) {
        typedUnsupported =
            error.implementationFamily() ==
                ::fabric::ImplementationFamilyId::LoopInvariant &&
            error.recipe() == BackendRecipeKey::PortableSystemVerilog;
      },
      [&](const llvm::ErrorInfoBase &error) {
        fail(test, description.str() +
                       " returned the wrong error class: " + error.message());
      });
  require(test, typedUnsupported,
          description.str() + " lost its typed Unsupported classification");
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

enum class ResourceContractKind { LoopInvariant, OneCycleElastic };

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
};

FabricFixture makeFabric(
    llvm::StringRef test, const ArtifactStore &store, unsigned phaseWidth,
    unsigned initWidth, unsigned outputWidth,
    ResourceContractKind contractKind = ResourceContractKind::LoopInvariant) {
  const std::string phase = std::to_string(phaseWidth);
  const std::string init = std::to_string(initWidth);
  const std::string output = std::to_string(outputWidth);
  const std::string outer =
      std::to_string(std::max({1U, phaseWidth, initWidth, outputWidth}));
  const auto inputBinding = [&](const std::string &inner) {
    const std::string boundary = ": !fabric.bits<" + outer + ">";
    return inner == outer ? boundary
                          : boundary + " to !fabric.bits<" + inner + ">";
  };
  const std::string yieldBinding =
      output == outer
          ? ": !fabric.bits<" + output + ">"
          : ": !fabric.bits<" + output + "> to !fabric.bits<" + outer + ">";
  const std::string sourceText =
      "module { fabric.module @loop_invariant(%phase: !fabric.bits<" + outer +
      ">, %init: !fabric.bits<" + outer + ">) -> !fabric.bits<" + outer +
      "> { %pe = fabric.pe [spatial] (%p = %phase : !fabric.bits<" + outer +
      ">, %i = %init : !fabric.bits<" + outer + ">) -> !fabric.bits<" + outer +
      "> { %fu = fabric.fu (%fp = %p " + inputBinding(phase) + ", %fi = %i " +
      inputBinding(init) + ") -> !fabric.bits<" + outer +
      "> { %value = fabric.op [@dataflow.invariant] (%fp, %fi) "
      "{implementation_family = #fabric.implementation_family<LoopInvariant>, "
      "hw_params = {}} : (!fabric.bits<" +
      phase + ">, !fabric.bits<" + init + ">) -> !fabric.bits<" + output +
      "> fabric.yield %value " + yieldBinding +
      " } } fabric.yield %pe : !fabric.bits<" + outer + "> } }";
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  if (!source)
    fail(test, "could not parse Fabric fixture: " + sourceText);

  const ::fabric::ResourceContract &contract =
      contractKind == ResourceContractKind::LoopInvariant
          ? ::fabric::loopInvariantOperationResourceContract()
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
          ::fabric::ImplementationFamilyId::LoopInvariant)
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), capability.occurrence, fuOccurrence));
      return FabricFixture{std::move(fabric), occurrence};
    }
  }
  fail(test, "Fabric fixture has no loop invariant occurrence");
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
    require(test, state != ports.end(), "loop invariant leaf has no state");
    state->type = builder.getIntegerType(
        mlir::cast<mlir::IntegerType>(state->type).getWidth() + 1);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("loop_invariant"), ports);
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
  if (llvm::Error error = registerPortableLoopInvariantProvider(registry))
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
          "portable loop invariant emitted external implementation state");
  return take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
}

void writeYosysScript(const std::filesystem::path &root) {
  std::ofstream(root / "portable_loop_invariant.ys") << R"ys(
read_verilog loop_invariant.sv
hierarchy -check -top loop_invariant
proc
opt
check -assert
select -assert-none loop_invariant/t:$*ff* loop_invariant/t:$*latch* loop_invariant/t:$_*FF* loop_invariant/t:$_*LATCH* loop_invariant/t:$mem* loop_invariant/m:*
synth -top loop_invariant
check -assert
select -assert-none loop_invariant/t:$*ff* loop_invariant/t:$*latch* loop_invariant/t:$_*FF* loop_invariant/t:$_*LATCH* loop_invariant/t:$mem* loop_invariant/m:*
stat
)ys";
}

void schemaCasesRemainAuthoritative(llvm::StringRef test) {
  using namespace ::dataflow::semantics;
  const InvariantCaseDescriptor init = invariantCaseDescriptor(
      selectInvariantCase(InvariantSemanticState::Initial, false));
  const InvariantCaseDescriptor replay = invariantCaseDescriptor(
      selectInvariantCase(InvariantSemanticState::Running, true));
  const InvariantCaseDescriptor close = invariantCaseDescriptor(
      selectInvariantCase(InvariantSemanticState::Running, false));
  require(test,
          init.output == InvariantOutputSource::InitInput && init.latchInput &&
              *init.latchInput == InvariantInput::Init && !init.clearLatch &&
              replay.output == InvariantOutputSource::Latched &&
              !replay.latchInput && !replay.clearLatch &&
              close.output == InvariantOutputSource::None &&
              !close.latchInput && close.clearLatch,
          "schema-owned invariant transition descriptors changed");
}

void canonicalBoundaryAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  std::filesystem::create_directories(root / "artifacts");
  ArtifactStore store((root / "artifacts").string());
  FabricFixture fabric = makeFabric(test, store, 8, 8, 8);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const auto ports = skeleton.leaf.getPortList();
  const std::vector<llvm::StringRef> expected{
      "data_input_0",   "data_input_1",   "valid_input_0", "valid_input_1",
      "ready_output_0", "state_current",  "ready_input_0", "ready_input_1",
      "data_output_0",  "valid_output_0", "state_next",    "state_write"};
  require(test, ports.size() == expected.size(),
          "loop invariant leaf has the wrong port count");
  for (auto [index, name] : llvm::enumerate(expected))
    require(test, ports[index].getName() == name,
            "loop invariant leaf ports are not canonical");
  require(test,
          ports[5].type == mlir::IntegerType::get(context.get(), 9) &&
              ports[10].type == mlir::IntegerType::get(context.get(), 9),
          "loop invariant state boundary does not contain mode and payload");

  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test,
          llvm::StringRef(rtl).contains("state_current") &&
              llvm::StringRef(rtl).contains("state_next") &&
              llvm::StringRef(rtl).contains("state_write") &&
              !llvm::StringRef(rtl).contains("always_ff") &&
              !llvm::StringRef(rtl).contains("posedge"),
          "loop invariant provider did not remain a combinational transform");
  std::ofstream(root / "loop_invariant.sv") << rtl;
  std::ofstream(root / "testbench.sv") << R"sv(
module testbench;
  logic [7:0] data_input_0;
  logic [7:0] data_input_1;
  logic       valid_input_0;
  logic       valid_input_1;
  logic       ready_output_0;
  logic [8:0] state_current;
  logic       ready_input_0;
  logic       ready_input_1;
  logic [7:0] data_output_0;
  logic       valid_output_0;
  logic [8:0] state_next;
  logic       state_write;

  loop_invariant dut(.*);

  task automatic expect_control(
      input logic phase_ready,
      input logic init_ready,
      input logic output_valid,
      input logic write_state,
      input logic [8:0] next_state);
    #1;
    if (ready_input_0 !== phase_ready || ready_input_1 !== init_ready ||
        valid_output_0 !== output_valid || state_write !== write_state ||
        state_next !== next_state)
      $fatal(1, "unexpected invariant control signals");
  endtask

  initial begin
    data_input_0 = 8'hff;
    data_input_1 = 8'h3c;
    valid_input_0 = 0;
    valid_input_1 = 1;
    ready_output_0 = 0;
    state_current = 9'h000;
    expect_control(0, 0, 1, 0, 9'h000);
    if (data_output_0 !== 8'h3c)
      $fatal(1, "stalled Init payload changed");

    ready_output_0 = 1;
    expect_control(0, 1, 1, 1, {8'h3c, 1'b1});
    if (data_output_0 !== 8'h3c)
      $fatal(1, "Init did not forward its input");

    state_current = state_next;
    data_input_1 = 8'ha5;
    data_input_0 = 8'h01;
    valid_input_0 = 1;
    ready_output_0 = 0;
    expect_control(0, 0, 1, 0, {8'h3c, 1'b1});
    if (data_output_0 !== 8'h3c)
      $fatal(1, "stalled Replay did not use the latch");

    ready_output_0 = 1;
    expect_control(1, 0, 1, 1, {8'h3c, 1'b1});
    if (data_output_0 !== 8'h3c)
      $fatal(1, "Replay did not use the latch");

    data_input_0 = 8'hfe;
    ready_output_0 = 0;
    expect_control(1, 0, 0, 1, 9'h000);

    state_current = state_next;
    valid_input_0 = 0;
    valid_input_1 = 0;
    ready_output_0 = 1;
    expect_control(0, 1, 0, 0, 9'h000);
    $finish;
  end
endmodule
)sv";
  writeYosysScript(root);
}

void mixedPhysicalWidthsPreserveLowBits(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  std::filesystem::create_directories(root / "artifacts");
  ArtifactStore store((root / "artifacts").string());
  FabricFixture fabric = makeFabric(test, store, 8, 16, 8);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  std::ofstream(root / "loop_invariant.sv") << rtl;
  std::ofstream(root / "testbench.sv") << R"sv(
module testbench;
  logic [7:0]  data_input_0;
  logic [15:0] data_input_1;
  logic        valid_input_0;
  logic        valid_input_1;
  logic        ready_output_0;
  logic [8:0]  state_current;
  logic        ready_input_0;
  logic        ready_input_1;
  logic [7:0]  data_output_0;
  logic        valid_output_0;
  logic [8:0]  state_next;
  logic        state_write;

  loop_invariant dut(.*);

  initial begin
    data_input_0 = 8'h80;
    data_input_1 = 16'habcd;
    valid_input_0 = 1;
    valid_input_1 = 1;
    ready_output_0 = 1;
    state_current = 9'h000;
    #1;
    if (!ready_input_1 || ready_input_0 || !valid_output_0 ||
        data_output_0 !== 8'hcd || !state_write ||
        state_next !== {8'hcd, 1'b1})
      $fatal(1, "Init low-bit truncation failed");

    state_current = {8'h5a, 1'b1};
    data_input_0 = 8'h81;
    #1;
    if (!ready_input_0 || ready_input_1 || !valid_output_0 ||
        data_output_0 !== 8'h5a || !state_write || state_next !== state_current)
      $fatal(1, "Replay did not preserve the retained low bits");
    $finish;
  end
endmodule
)sv";
  writeYosysScript(root);
}

void zeroWidthPayloadNeedsOnlyHandshake(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  std::filesystem::create_directories(root / "artifacts");
  ArtifactStore store((root / "artifacts").string());
  FabricFixture fabric = makeFabric(test, store, 8, 0, 0);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const auto ports = skeleton.leaf.getPortList();
  require(test,
          llvm::none_of(ports,
                        [](const auto &port) {
                          return port.getName() == "data_input_1" ||
                                 port.getName() == "data_output_0";
                        }),
          "zero-width invariant retained payload signals");
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test,
          llvm::StringRef(rtl).contains("valid_output_0") &&
              !llvm::StringRef(rtl).contains("data_output_0"),
          "zero-width invariant did not preserve token-only handshaking");
  std::ofstream(root / "loop_invariant.sv") << rtl;
  std::ofstream(root / "testbench.sv") << R"sv(
module testbench;
  logic [7:0] data_input_0;
  logic       valid_input_0;
  logic       valid_input_1;
  logic       ready_output_0;
  logic       state_current;
  logic       ready_input_0;
  logic       ready_input_1;
  logic       valid_output_0;
  logic       state_next;
  logic       state_write;

  loop_invariant dut(.*);

  initial begin
    data_input_0 = 8'h00;
    valid_input_0 = 0;
    valid_input_1 = 1;
    ready_output_0 = 1;
    state_current = 0;
    #1;
    if (ready_input_0 || !ready_input_1 || !valid_output_0 ||
        !state_write || !state_next)
      $fatal(1, "zero-payload Init failed");

    state_current = 1;
    data_input_0 = 8'h01;
    valid_input_0 = 1;
    #1;
    if (!ready_input_0 || ready_input_1 || !valid_output_0 ||
        !state_write || !state_next)
      $fatal(1, "zero-payload Replay failed");

    data_input_0 = 8'h00;
    ready_output_0 = 0;
    #1;
    if (!ready_input_0 || ready_input_1 || valid_output_0 ||
        !state_write || state_next)
      $fatal(1, "zero-payload Close failed");
    $finish;
  end
endmodule
)sv";
  writeYosysScript(root);
}

void invalidInputsAreTransactional(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableLoopInvariantProvider(registry))
    fail(test, llvm::toString(std::move(error)));
  ExternalImplementationContractCatalog externalContracts;

  FabricFixture unsupported =
      makeFabric(test, store, 8, 8, 8, ResourceContractKind::OneCycleElastic);
  FinalizedConfigurationABI unsupportedAbi =
      makeConfigurationAbi(test, store, unsupported);
  SkeletonFixture unsupportedSkeleton =
      makeSkeleton(test, *context, unsupported, unsupportedAbi.abi());
  const std::string beforeContract = moduleText(*unsupportedSkeleton.module);
  const std::vector<FabricOperationLeafAssociation> unsupportedAssociations = {
      {unsupportedSkeleton.leaf, unsupported.occurrence}};
  const std::vector<FabricOperationRecipeBinding> unsupportedRecipes = {
      {unsupported.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectTypedUnsupported(test,
                         specializeFabricOperationLeaves(
                             *unsupportedSkeleton.module, unsupported.fabric,
                             unsupportedAbi, unsupportedAssociations,
                             unsupportedRecipes, registry, externalContracts),
                         "unsupported invariant resource contract");
  require(test, moduleText(*unsupportedSkeleton.module) == beforeContract,
          "unsupported contract partially mutated the caller module");

  FabricFixture valid = makeFabric(test, store, 8, 8, 8);
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  SkeletonFixture malformed =
      makeSkeleton(test, *context, valid, validAbi.abi(), true);
  const std::string beforeLeaf = moduleText(*malformed.module);
  const std::vector<FabricOperationLeafAssociation> malformedAssociations = {
      {malformed.leaf, valid.occurrence}};
  const std::vector<FabricOperationRecipeBinding> validRecipes = {
      {valid.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectError(test,
              specializeFabricOperationLeaves(*malformed.module, valid.fabric,
                                              validAbi, malformedAssociations,
                                              validRecipes, registry,
                                              externalContracts),
              "leaf port");
  require(test, moduleText(*malformed.module) == beforeLeaf,
          "malformed leaf partially mutated the caller module");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  schemaCasesRemainAuthoritative("schemaCasesRemainAuthoritative");
  canonicalBoundaryAndArtifacts(root);
  mixedPhysicalWidthsPreserveLowBits(root / "mixed_width");
  zeroWidthPayloadNeedsOnlyHandshake(root / "zero_width");
  invalidInputsAreTransactional(root / "invalid");
  return 0;
}

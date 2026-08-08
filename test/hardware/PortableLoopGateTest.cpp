#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/Providers/LoopGate.h"

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
    fail(test, "accepted invalid portable loop gate input");
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
                ::fabric::ImplementationFamilyId::LoopGate &&
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

enum class ResourceContractKind { LoopGate, OneCycleElastic };

struct FabricFixture final {
  FinalizedFabricRoot fabric;
  FabricFuOccurrenceNodeRef occurrence;
};

FabricFixture
makeFabric(llvm::StringRef test, const ArtifactStore &store,
           unsigned phaseInputWidth, unsigned valueInputWidth,
           unsigned phaseOutputWidth, unsigned valueOutputWidth,
           ResourceContractKind contractKind = ResourceContractKind::LoopGate) {
  const unsigned outerWidth = std::max({1U, phaseInputWidth, valueInputWidth,
                                        phaseOutputWidth, valueOutputWidth});
  const auto inputBinding = [&](unsigned width) {
    const std::string outer =
        ": !fabric.bits<" + std::to_string(outerWidth) + ">";
    return width == outerWidth
               ? outer
               : outer + " to !fabric.bits<" + std::to_string(width) + ">";
  };
  const auto outputBinding = [&](unsigned width) {
    const std::string inner = ": !fabric.bits<" + std::to_string(width) + ">";
    return width == outerWidth
               ? inner
               : inner + " to !fabric.bits<" + std::to_string(outerWidth) + ">";
  };
  const std::string sourceText =
      "module { fabric.module @loop_gate(%phase : !fabric.bits<" +
      std::to_string(outerWidth) + ">, %value : !fabric.bits<" +
      std::to_string(outerWidth) +
      ">) { fabric.pe [spatial] (%p = %phase : !fabric.bits<" +
      std::to_string(outerWidth) + ">, %v = %value : !fabric.bits<" +
      std::to_string(outerWidth) + ">) -> (!fabric.bits<" +
      std::to_string(outerWidth) + ">, !fabric.bits<" +
      std::to_string(outerWidth) + ">) { fabric.fu (%fp = %p " +
      inputBinding(phaseInputWidth) + ", %fv = %v " +
      inputBinding(valueInputWidth) + ") -> (!fabric.bits<" +
      std::to_string(outerWidth) + ">, !fabric.bits<" +
      std::to_string(outerWidth) +
      ">) { %after_phase, %after_value = fabric.op [@dataflow.gate] "
      "(%fp, %fv) {implementation_family = "
      "#fabric.implementation_family<LoopGate>, hw_params = {}} : "
      "(!fabric.bits<" +
      std::to_string(phaseInputWidth) + ">, !fabric.bits<" +
      std::to_string(valueInputWidth) + ">) -> (!fabric.bits<" +
      std::to_string(phaseOutputWidth) + ">, !fabric.bits<" +
      std::to_string(valueOutputWidth) + ">) fabric.yield %after_phase " +
      outputBinding(phaseOutputWidth) + ", %after_value " +
      outputBinding(valueOutputWidth) + " } } fabric.yield } }";
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &fabricContext());
  if (!source)
    fail(test, "could not parse Fabric fixture: " + sourceText);

  const ::fabric::ResourceContract &contract =
      contractKind == ResourceContractKind::LoopGate
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
      if (capability.implementationFamily !=
          ::fabric::ImplementationFamilyId::LoopGate)
        continue;
      FabricFuOccurrenceNodeRef occurrence =
          take(test, loom::fabric::deriveFabricFuOccurrenceNode(
                         fabric.view(), capability.occurrence, fuOccurrence));
      return FabricFixture{std::move(fabric), occurrence};
    }
  }
  fail(test, "Fabric fixture has no loop gate occurrence");
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
    require(test, state != ports.end(), "loop gate leaf has no current state");
    state->type = builder.getIntegerType(2);
  }
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("loop_gate"), ports);
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
  if (llvm::Error error = registerPortableLoopGateProvider(registry))
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
          "portable loop gate emitted external implementation state");
  return take(test, lowerAndExportSpecializedSystemVerilog(*skeleton.module));
}

void writeYosysScript(const std::filesystem::path &root) {
  std::ofstream(root / "portable_loop_gate.ys") << R"ys(
read_verilog loop_gate.sv
hierarchy -check -top loop_gate
proc
opt
check -assert
select -assert-none loop_gate/t:$*ff* loop_gate/t:$*latch* loop_gate/t:$_*FF* loop_gate/t:$_*LATCH* loop_gate/t:$mem* loop_gate/m:*
synth -top loop_gate
check -assert
select -assert-none loop_gate/t:$*ff* loop_gate/t:$*latch* loop_gate/t:$_*FF* loop_gate/t:$_*LATCH* loop_gate/t:$mem* loop_gate/m:*
stat
)ys";
}

void writeDenseTestbench(const std::filesystem::path &root) {
  std::ofstream(root / "testbench.sv") << R"sv(
module testbench;
  logic [7:0]  data_input_0;
  logic [15:0] data_input_1;
  logic        valid_input_0;
  logic        valid_input_1;
  logic        ready_output_0;
  logic        ready_output_1;
  logic        state_current;
  logic        ready_input_0;
  logic        ready_input_1;
  logic [7:0]  data_output_0;
  logic [7:0]  data_output_1;
  logic        valid_output_0;
  logic        valid_output_1;
  logic        state_next;
  logic        state_write;
  integer      control;
  logic        phase;
  logic        both_valid;
  logic        output_capacity;
  logic        expected_phase_valid;
  logic        expected_value_valid;
  logic        expected_fire;
  logic        expected_next;

  loop_gate dut(.*);

  initial begin
    for (control = 0; control < 64; control = control + 1) begin
      state_current = control[5];
      data_input_0 = {7'h55, control[4]};
      data_input_1 = 16'ha500 ^ control[15:0];
      valid_input_0 = control[3];
      valid_input_1 = control[2];
      ready_output_0 = control[1];
      ready_output_1 = control[0];
      phase = data_input_0[0];
      both_valid = valid_input_0 && valid_input_1;
      expected_phase_valid =
          both_valid && state_current && (!phase || ready_output_1);
      expected_value_valid =
          both_valid && phase && (!state_current || ready_output_0);
      if (!state_current && !phase)
        output_capacity = 1'b1;
      else if (!state_current && phase)
        output_capacity = ready_output_1;
      else if (state_current && phase)
        output_capacity = ready_output_0 && ready_output_1;
      else
        output_capacity = ready_output_0;
      expected_fire = both_valid && output_capacity;
      expected_next = expected_fire ? phase : state_current;
      #1;
      if (ready_input_0 !== (valid_input_1 && output_capacity) ||
          ready_input_1 !== (valid_input_0 && output_capacity) ||
          valid_output_0 !== expected_phase_valid ||
          valid_output_1 !== expected_value_valid ||
          state_write !== expected_fire || state_next !== expected_next)
        $fatal(1, "unexpected gate handshake or state transition");
      if (valid_output_0 &&
          (data_output_0[0] !== phase || data_output_0[7:1] !== 7'b0))
        $fatal(1, "gate emitted the wrong phase payload");
      if (valid_output_1 && data_output_1 !== data_input_1[7:0])
        $fatal(1, "gate did not preserve the low value bits");
      if (state_current && phase && both_valid &&
          ((valid_output_0 && ready_output_0) !==
           (valid_output_1 && ready_output_1)))
        $fatal(1, "gate partially published an atomic result pair");
    end
    $finish;
  end
endmodule
)sv";
}

void writeZeroPayloadTestbench(const std::filesystem::path &root) {
  std::ofstream(root / "testbench.sv") << R"sv(
module testbench;
  logic data_input_0;
  logic valid_input_0;
  logic valid_input_1;
  logic ready_output_0;
  logic ready_output_1;
  logic state_current;
  logic ready_input_0;
  logic ready_input_1;
  logic data_output_0;
  logic valid_output_0;
  logic valid_output_1;
  logic state_next;
  logic state_write;
  integer control;
  logic both_valid;
  logic output_capacity;
  logic expected_fire;

  loop_gate dut(.*);

  initial begin
    for (control = 0; control < 64; control = control + 1) begin
      state_current = control[5];
      data_input_0 = control[4];
      valid_input_0 = control[3];
      valid_input_1 = control[2];
      ready_output_0 = control[1];
      ready_output_1 = control[0];
      both_valid = valid_input_0 && valid_input_1;
      if (!state_current && !data_input_0)
        output_capacity = 1'b1;
      else if (!state_current && data_input_0)
        output_capacity = ready_output_1;
      else if (state_current && data_input_0)
        output_capacity = ready_output_0 && ready_output_1;
      else
        output_capacity = ready_output_0;
      expected_fire = both_valid && output_capacity;
      #1;
      if (ready_input_0 !== (valid_input_1 && output_capacity) ||
          ready_input_1 !== (valid_input_0 && output_capacity) ||
          valid_output_0 !==
              (both_valid && state_current &&
               (!data_input_0 || ready_output_1)) ||
          valid_output_1 !==
              (both_valid && data_input_0 &&
               (!state_current || ready_output_0)) ||
          state_write !== expected_fire ||
          state_next !== (expected_fire ? data_input_0 : state_current))
        $fatal(1, "zero-payload gate changed handshake semantics");
      if (valid_output_0 && data_output_0 !== data_input_0)
        $fatal(1, "zero-payload gate emitted the wrong phase");
      if (state_current && data_input_0 && both_valid &&
          ((valid_output_0 && ready_output_0) !==
           (valid_output_1 && ready_output_1)))
        $fatal(1, "zero-payload gate partially published a result pair");
    end
    $finish;
  end
endmodule
)sv";
}

void canonicalTransitionsAndArtifacts(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store, 8, 16, 8, 8);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const auto ports = skeleton.leaf.getPortList();
  const std::vector<llvm::StringRef> expected{
      "data_input_0",   "data_input_1",   "valid_input_0", "valid_input_1",
      "ready_output_0", "ready_output_1", "state_current", "ready_input_0",
      "ready_input_1",  "data_output_0",  "data_output_1", "valid_output_0",
      "valid_output_1", "state_next",     "state_write"};
  require(test, ports.size() == expected.size(),
          "loop gate leaf has the wrong port count");
  for (auto [index, name] : llvm::enumerate(expected))
    require(test, ports[index].getName() == name,
            "loop gate leaf ports are not canonical");
  require(test,
          ports[6].type == mlir::IntegerType::get(context.get(), 1) &&
              ports[13].type == mlir::IntegerType::get(context.get(), 1) &&
              ports[14].type == mlir::IntegerType::get(context.get(), 1),
          "loop gate state boundary is not one bit");

  const std::string rtl = specialize(test, skeleton, fabric, abi);
  const llvm::StringRef text(rtl);
  require(test,
          text.contains("state_current") && text.contains("state_next") &&
              text.contains("state_write") && !text.contains("always_ff") &&
              !text.contains("posedge"),
          "loop gate provider did not remain a combinational state transform");
  std::ofstream(root / "loop_gate.sv") << rtl;
  writeDenseTestbench(root);
  writeYosysScript(root);
}

void zeroWidthValueKeepsTokenHandshake(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store, 1, 0, 1, 0);
  FinalizedConfigurationABI abi = makeConfigurationAbi(test, store, fabric);
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  SkeletonFixture skeleton = makeSkeleton(test, *context, fabric, abi.abi());
  const auto ports = skeleton.leaf.getPortList();
  require(test,
          llvm::none_of(ports,
                        [](const auto &port) {
                          return port.getName() == "data_input_1" ||
                                 port.getName() == "data_output_1";
                        }),
          "zero-width gate retained value payload signals");
  const std::string rtl = specialize(test, skeleton, fabric, abi);
  require(test,
          llvm::StringRef(rtl).contains("valid_output_1") &&
              !llvm::StringRef(rtl).contains("data_output_1"),
          "zero-width gate did not preserve token-only handshaking");
  std::ofstream(root / "loop_gate.sv") << rtl;
  writeZeroPayloadTestbench(root);
  writeYosysScript(root);
}

void invalidAndUnsupportedInputsAreTransactional(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  std::unique_ptr<mlir::MLIRContext> context = makeCirctContext();
  ExternalImplementationContractCatalog externalContracts;
  FabricOperationProviderRegistry registry;
  if (llvm::Error error = registerPortableLoopGateProvider(registry))
    fail(test, llvm::toString(std::move(error)));

  FabricFixture unsupported = makeFabric(test, store, 1, 8, 1, 8,
                                         ResourceContractKind::OneCycleElastic);
  FinalizedConfigurationABI unsupportedAbi =
      makeConfigurationAbi(test, store, unsupported);
  SkeletonFixture unsupportedSkeleton =
      makeSkeleton(test, *context, unsupported, unsupportedAbi.abi());
  const std::string unsupportedBefore = moduleText(*unsupportedSkeleton.module);
  const std::vector<FabricOperationLeafAssociation> unsupportedAssociations = {
      {unsupportedSkeleton.leaf, unsupported.occurrence}};
  const std::vector<FabricOperationRecipeBinding> unsupportedRecipes = {
      {unsupported.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectTypedUnsupported(test,
                         specializeFabricOperationLeaves(
                             *unsupportedSkeleton.module, unsupported.fabric,
                             unsupportedAbi, unsupportedAssociations,
                             unsupportedRecipes, registry, externalContracts),
                         "unsupported loop gate resource contract");
  require(test, moduleText(*unsupportedSkeleton.module) == unsupportedBefore,
          "unsupported contract partially mutated the caller module");

  FabricFixture valid = makeFabric(test, store, 1, 8, 1, 8);
  FinalizedConfigurationABI validAbi = makeConfigurationAbi(test, store, valid);
  SkeletonFixture malformed =
      makeSkeleton(test, *context, valid, validAbi.abi(), true);
  const std::string malformedBefore = moduleText(*malformed.module);
  const std::vector<FabricOperationLeafAssociation> malformedAssociations = {
      {malformed.leaf, valid.occurrence}};
  const std::vector<FabricOperationRecipeBinding> malformedRecipes = {
      {valid.occurrence, BackendRecipeKey::PortableSystemVerilog, {}}};
  expectError(test,
              specializeFabricOperationLeaves(*malformed.module, valid.fabric,
                                              validAbi, malformedAssociations,
                                              malformedRecipes, registry,
                                              externalContracts),
              "leaf port");
  require(test, moduleText(*malformed.module) == malformedBefore,
          "invalid leaf partially mutated the caller module");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one output directory");
  const std::filesystem::path root(argv[1]);
  canonicalTransitionsAndArtifacts(root);
  zeroWidthValueKeepsTokenHandshake(root / "zero_payload");
  invalidAndUnsupportedInputsAreTransactional(root / "invalid");
  return 0;
}

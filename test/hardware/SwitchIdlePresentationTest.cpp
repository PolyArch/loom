// Anchors the Temporal switch idle-presentation contract of
// docs/spec-rtl-lowering.md with the exact generated RTL of one temporal
// switch whose two resident rows either select disjoint outputs or contend
// for one output:
//  - with disjoint outputs both rows are presented together, so each input
//    is ready exactly while its output is ready and never observes its own
//    valid, and a token on either input retires without waiting for the
//    idle rotation;
//  - with one shared output the idle rotation presents one candidate at a
//    time, a valid requester is presented by the grant policy even while the
//    rotation shows the other input, and simultaneous requesters retire one
//    per handoff.
// The first fixture is the joint whose absence closed a combinational cycle
// through the reusable switch fabric of a mapped Matmul under an event-driven
// simulator; the second keeps the contention semantics unchanged.
#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "ConfigurationABITestSupport.h"
#include "ConfigurationTransportTestSupport.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FinalizedFabricRoot;
using loom::hardware::FinalizedConfigurationABI;

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

enum class Fixture { Disjoint, Contending };

llvm::StringRef fixtureName(Fixture fixture) {
  return fixture == Fixture::Disjoint ? "disjoint" : "contending";
}

/// One resident row: the input, the output it selects, and its tag.
struct ResidentRow final {
  std::uint32_t input = 0;
  std::uint32_t output = 0;
  std::uint32_t tag = 0;
};

std::vector<ResidentRow> residentRows(Fixture fixture) {
  if (fixture == Fixture::Disjoint)
    return {{0, 0, 2}, {1, 1, 1}};
  return {{0, 0, 2}, {1, 0, 1}};
}

/// A 2 x 2 fully connected temporal switch with a two-entry route table and
/// a round-robin grant, the smallest fabric in which two resident rows share
/// one arbitration component.
FinalizedFabricRoot makeFixtureModule(llvm::StringRef test,
                                      const ArtifactStore &store,
                                      Fixture fixture) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType tagged8x2 = take(test, PortType::taggedBits(8, 2));
  auto spatial =
      take(test, design.createSpatialCore(
                     ("switch-idle-presentation-" + fixtureName(fixture)).str(),
                     {tagged8x2, tagged8x2}, {tagged8x2, tagged8x2}));
  auto routed = take(
      test,
      spatial.addSwitch(
          {take(test, spatial.input(0)), take(test, spatial.input(1))},
          SwitchSpec::temporal({tagged8x2, tagged8x2}, {tagged8x2, tagged8x2},
                               {{0, 1}, {0, 1}}, 2,
                               ::fabric::TemporalSwitchRoundRobin{{0, 1}, 0})));
  if (llvm::Error error = spatial.close(routed.values()))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "switch fixture did not finalize one Module");
  return std::move(finalized.roots().front());
}

loom::fabric::FabricPhysicalConfigurationFieldRef qualifyConfigurationField(
    llvm::StringRef test, loom::fabric::SpatialCoreOccurrenceRef spatialCore,
    const loom::fabric::FabricSemanticConfigFieldRef &field) {
  auto target =
      take(test, loom::fabric::FabricModulePhysicalTargetRef::create(field));
  return take(test, loom::fabric::FabricPhysicalConfigurationFieldRef::create(
                        loom::fabric::SpatialCoreInternalOccurrenceRef{
                            spatialCore, std::move(target)}));
}

const loom::hardware::ProgrammingUnit *findProgrammingOwner(
    llvm::StringRef test, const loom::hardware::ConfigurationABI &abi,
    const loom::fabric::FabricPhysicalConfigurationSlotRef &slot) {
  const loom::hardware::ProgrammingUnit *result = nullptr;
  for (const auto &unit : abi.programmingUnits())
    for (const auto &field : unit.fields)
      if (field.slot == slot) {
        require(test, result == nullptr,
                "configuration field has duplicate programming owners");
        result = &unit;
      }
  require(test, result != nullptr,
          "configuration field has no programming owner");
  return result;
}

struct SwitchArtifact final {
  Fixture fixture = Fixture::Disjoint;
  std::string systemVerilog;
  loom::hardware::test::PortableConfigurationTarget target;
  std::vector<std::uint8_t> image;
};

SwitchArtifact buildArtifact(const std::filesystem::path &root,
                             Fixture fixture) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FinalizedFabricRoot module = makeFixtureModule(test, store, fixture);
  FinalizedFabricRoot system = take(
      test, loom::hardware::test::makeSingleSpatialCoreSystem(module, store));
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore = take(
      test, loom::hardware::test::requireSingleSpatialCoreOccurrence(system));
  const auto &view = module.view();
  require(test, view.switchOccurrences().size() == 1,
          "switch fixture changed its switch count");
  const auto sw = view.switchOccurrences().front();

  // The switch carrier is a Direct field; the ABI draft needs its exact bit
  // width like every other direct carrier.
  std::vector<loom::hardware::test::ConfigurationFieldEncodingOverride>
      overrides;
  const loom::fabric::FabricInventoryOwnerRef owner =
      loom::fabric::FabricInventoryOwnerRef::of(sw);
  const std::uint64_t fieldCount = view.inventorySize(
      owner, loom::fabric::FabricInventoryKind::SemanticConfigField);
  for (std::uint64_t ordinal = 0; ordinal < fieldCount; ++ordinal) {
    const loom::fabric::FabricSemanticConfigFieldRef field{
        loom::fabric::FabricConfigurationOwnerRef(owner), ordinal};
    auto relation = take(
        test, view.semanticFieldRelation(field, *const_cast<mlir::Operation *>(
                                                     view.canonicalOperation())
                                                     ->getContext()));
    if (relation.kind() !=
        loom::fabric::FabricSemanticFieldRelationKind::Direct)
      continue;
    const std::uint64_t bitCount = *relation.directEncodedBitCount();
    overrides.push_back({qualifyConfigurationField(test, spatialCore, field),
                         loom::hardware::DirectBitsEncoding{bitCount},
                         std::vector<std::uint8_t>((bitCount + 7) / 8, 0)});
  }
  FinalizedConfigurationABI abi = take(
      test,
      loom::hardware::finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         system, overrides)),
          store));

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto skeleton = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                                 context, spatialCore, abi));
  require(test, skeleton.operationLeaves.empty(),
          "switch fixture unexpectedly owns operation leaves");
  // The fixture has no operation leaf to specialize, so the common skeleton
  // lowers and exports directly.
  auto systemVerilog =
      take(test, loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(
                     *skeleton.module));

  // Every resident row selects exactly one admitted traversal of the switch.
  const auto traversalOf = [&](std::uint32_t input, std::uint32_t output) {
    std::optional<loom::fabric::FabricPhysicalTraversalRef> found;
    for (const auto &traversal : view.physicalTraversals()) {
      const auto *payload =
          std::get_if<loom::fabric::FabricSwitchTraversalPayload>(
              &traversal.reference.payload);
      if (payload && payload->owner == sw && payload->input == input &&
          payload->output == output) {
        require(test, !found.has_value(),
                "switch fixture admits one traversal twice");
        found = traversal.reference;
      }
    }
    require(test, found.has_value(), "switch fixture lacks a traversal");
    return *found;
  };
  std::vector<loom::fabric::FabricTemporalSwitchRouteEntry> entries;
  for (const ResidentRow &row : residentRows(fixture))
    entries.push_back(
        {llvm::APInt(2, row.tag), {traversalOf(row.input, row.output)}});
  const loom::fabric::FabricSemanticConfigFieldRef field{
      loom::fabric::FabricConfigurationOwnerRef(owner), 0};
  auto semantic = take(test, loom::fabric::encodeTemporalSwitchConfiguration(
                                 view, field, entries));
  const auto slot =
      take(test, loom::fabric::qualifyFabricConfigurationSlot(
                     qualifyConfigurationField(test, spatialCore, field),
                     loom::fabric::FabricStaticConfigurationResidency{}));
  const loom::hardware::ProgrammingUnit *programming =
      findProgrammingOwner(test, abi.abi(), slot);
  std::vector<loom::hardware::SemanticConfigurationValue> values;
  values.push_back({slot, std::vector<std::uint8_t>(semantic.bytes().begin(),
                                                    semantic.bytes().end())});
  return SwitchArtifact{
      fixture, std::move(systemVerilog),
      take(test, loom::hardware::test::derivePortableConfigurationTarget(
                     abi, spatialCore, programming->id)),
      take(test, abi.abi().encode(programming->id, values))};
}

void writeArtifacts(const std::filesystem::path &root,
                    const SwitchArtifact &artifact) {
  const llvm::StringRef test = __func__;
  const std::string prefix =
      ("switch_idle_presentation_" + fixtureName(artifact.fixture)).str();
  const bool disjoint = artifact.fixture == Fixture::Disjoint;
  std::ofstream(root / (prefix + "_module.sv")) << artifact.systemVerilog;
  std::ofstream testbench(root / (prefix + "_testbench.sv"));
  testbench << "\nmodule " << prefix << R"sv(_testbench;
  logic       clock;
  logic       reset;
  logic [7:0] input_0_data;
  logic [1:0] input_0_tag;
  logic       input_0_valid;
  logic       input_0_ready;
  logic [7:0] input_1_data;
  logic [1:0] input_1_tag;
  logic       input_1_valid;
  logic       input_1_ready;
  logic [7:0] output_0_data;
  logic [1:0] output_0_tag;
  logic       output_0_valid;
  logic       output_0_ready;
  logic [7:0] output_1_data;
  logic [1:0] output_1_tag;
  logic       output_1_valid;
  logic       output_1_ready;
)sv";
  testbench << loom::hardware::test::portableAxiLiteSignalDeclarations()
            << "\n";
  testbench << R"sv(
  loom_module dut(.*);
  always #5 clock = ~clock;

  task automatic check(bit condition, string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

  // Offers one token on an input port, returns the number of clock edges
  // it waited before the switch accepted it, and checks that the token
  // appears on its output in the accepting cycle: a temporal switch forwards
  // combinationally, so the output handshake coincides with the input's.
  task automatic send_expect(input int in_port, input int out_port,
                             input logic [1:0] tag, input logic [7:0] data,
                             output int waited);
    begin
      @(negedge clock);
      if (in_port == 0) begin
        input_0_data = data;
        input_0_tag = tag;
        input_0_valid = 1;
      end else begin
        input_1_data = data;
        input_1_tag = tag;
        input_1_valid = 1;
      end
      waited = 0;
      do begin
        @(posedge clock);
        waited = waited + 1;
        if (waited == 16)
          $fatal(1, "Input port handshake timed out");
      end while (!(in_port == 0 ? input_0_ready : input_1_ready));
      check(out_port == 0
                ? (output_0_valid && output_0_ready && output_0_tag == tag &&
                   output_0_data == data)
                : (output_1_valid && output_1_ready && output_1_tag == tag &&
                   output_1_data == data),
            "Accepted token did not appear on its output in the same cycle");
      @(negedge clock);
      if (in_port == 0)
        input_0_valid = 0;
      else
        input_1_valid = 0;
    end
  endtask

  task automatic expect_silence(string message);
    begin
      repeat (6) begin
        @(posedge clock);
        #1;
        check(!output_0_valid && !output_1_valid, message);
      end
    end
  endtask

)sv";
  testbench << loom::hardware::test::portableAxiLiteDriverTasks();
  testbench << loom::hardware::test::portableCycleWatchdog();
  testbench << R"sv(

  int waited;
  int ready_0_cycles;
  int ready_1_cycles;
  int both_ready_cycles;
  int neither_ready_cycles;

  initial begin
    clock = 0;
    reset = 1;
    input_0_data = 0;
    input_0_tag = 0;
    input_0_valid = 0;
    input_1_data = 0;
    input_1_tag = 0;
    input_1_valid = 0;
    output_0_ready = 1;
    output_1_ready = 1;
)sv";
  testbench << loom::hardware::test::portableAxiLiteInitialization();
  testbench << R"sv(    repeat (2) @(posedge clock);
    #1 reset = 0;
    #1;
    check(!input_0_ready && !input_1_ready && !output_0_valid &&
              !output_1_valid,
          "Disabled Temporal switch presented readiness or a token");

)sv";
  testbench << take(test, loom::hardware::test::portableAxiLiteProgramAndVerify(
                              artifact.target, artifact.image));
  // Both inputs present their resident tags without a token so the idle
  // presentation of the two rows is observable.
  testbench << R"sv(    @(negedge clock);
    input_0_tag = 2'd2;
    input_1_tag = 2'd1;
    ready_0_cycles = 0;
    ready_1_cycles = 0;
    both_ready_cycles = 0;
    neither_ready_cycles = 0;
    repeat (6) begin
      @(posedge clock);
      #1;
      if (input_0_ready) ready_0_cycles = ready_0_cycles + 1;
      if (input_1_ready) ready_1_cycles = ready_1_cycles + 1;
      if (input_0_ready && input_1_ready)
        both_ready_cycles = both_ready_cycles + 1;
      if (!input_0_ready && !input_1_ready)
        neither_ready_cycles = neither_ready_cycles + 1;
    end
)sv";
  if (disjoint)
    testbench << R"sv(    check(both_ready_cycles == 6,
          "Rows with disjoint outputs were not presented together while idle");
    @(negedge clock);
    output_0_ready = 0;
    repeat (3) begin
      @(posedge clock);
      #1;
      check(!input_0_ready && input_1_ready,
            "Readiness did not follow the row's own output alone");
    end
    @(negedge clock);
    output_0_ready = 1;
    send_expect(0, 0, 2'd2, 8'd165, waited);
    check(waited == 1, "Idle row 0 was not ready for its first token");
    expect_silence("Row 0 token leaked to another output or repeated");
    send_expect(1, 1, 2'd1, 8'd90, waited);
    check(waited == 1, "Idle row 1 was not ready for its first token");
    expect_silence("Row 1 token leaked to another output or repeated");
    $finish;
  end
endmodule
)sv";
  else
    testbench
        << R"sv(    check(both_ready_cycles == 0 && neither_ready_cycles == 0,
          "Rows contending for one output were not presented one at a time");
    check(ready_0_cycles >= 2 && ready_1_cycles >= 2,
          "The idle rotation did not present each contending row");
    send_expect(1, 0, 2'd1, 8'd90, waited);
    check(waited <= 2, "A valid requester was not presented by the grant");
    send_expect(0, 0, 2'd2, 8'd165, waited);
    check(waited <= 2, "A valid requester was not presented by the grant");
    expect_silence("Contending tokens repeated");
    // Simultaneous requesters retire one per handoff, in two cycles.
    @(negedge clock);
    input_0_data = 8'd7;
    input_0_valid = 1;
    input_1_data = 8'd9;
    input_1_valid = 1;
    ready_0_cycles = 0;
    ready_1_cycles = 0;
    repeat (3) begin
      @(posedge clock);
      #1;
      check(!(input_0_ready && input_1_ready),
            "Two contending requesters were granted in one cycle");
      if (input_0_valid && input_0_ready) begin
        ready_0_cycles = ready_0_cycles + 1;
        @(negedge clock);
        input_0_valid = 0;
      end else if (input_1_valid && input_1_ready) begin
        ready_1_cycles = ready_1_cycles + 1;
        @(negedge clock);
        input_1_valid = 0;
      end
    end
    check(ready_0_cycles == 1 && ready_1_cycles == 1,
          "Simultaneous contending requesters did not both retire");
    $finish;
  end
endmodule
)sv";
  std::ofstream(root / (prefix + ".ys"))
      << "\nread_verilog -sv " << prefix << R"ys(_module.sv
hierarchy -check -top loom_module
check -assert
proc
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys";
}

} // namespace

int main(int argc, char **argv) {
  require("main", argc == 2, "expected exactly one output directory");
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  for (const Fixture fixture : {Fixture::Disjoint, Fixture::Contending})
    writeArtifacts(
        root,
        buildArtifact(root / "store" / fixtureName(fixture).str(), fixture));
  return 0;
}

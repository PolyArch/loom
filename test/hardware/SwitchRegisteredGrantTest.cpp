// Anchors the registered grant of `docs/spec-fabric-switch.md` with exact
// generated RTL for Temporal switches, over three fixtures that share one
// generated testbench shape:
//  - `disjoint` gives every input its own physical arbitration component, so
//    no component arbitrates: each input is ready exactly while the outputs
//    its resident row selects are ready, an atomic multicast still withholds
//    its source's readiness and a selected output's validity until every
//    sibling output is ready, and both components transfer in the cycle their
//    valid arrives;
//  - `round_robin` and `fixed_priority` put three inputs in one physical
//    component, so that component's grant pointer names the one input that
//    may transfer: an idle component holds its pointer at its reset
//    requester, a fresh request is granted in the cycle after its valid rises,
//    a lone continuous stream keeps its turn every cycle, two continuous
//    contenders alternate one transfer per cycle under either policy, and a
//    pointed input whose downstream refuses holds its turn.
// Every fixture sweeps the complete input valid vector inside one cycle and
// requires every input ready to be unchanged, which is the invariant that no
// ready of a Temporal switch has a combinational path from any valid: the
// pointer's next value observes that vector, but only through the register.
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
#include <optional>
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

enum class Fixture { Disjoint, RoundRobin, FixedPriority };

llvm::StringRef fixtureName(Fixture fixture) {
  switch (fixture) {
  case Fixture::Disjoint:
    return "disjoint";
  case Fixture::RoundRobin:
    return "round_robin";
  case Fixture::FixedPriority:
    return "fixed_priority";
  }
  llvm_unreachable("closed switch fixture domain");
}

/// One resident row: its input, the outputs it selects, and its tag.
struct ResidentRow final {
  std::uint32_t input = 0;
  std::vector<std::uint32_t> outputs;
  std::uint32_t tag = 0;
};

/// The `disjoint` fixture owns one multicast row and one unicast row on
/// separate physical components; the contended fixtures put every input on one
/// component and let two of the three rows select one output.
std::vector<ResidentRow> residentRows(Fixture fixture) {
  if (fixture == Fixture::Disjoint)
    return {{0, {0, 1}, 2}, {1, {2}, 1}};
  return {{0, {0}, 0}, {1, {1}, 1}, {2, {0}, 2}};
}

std::vector<std::vector<std::uint32_t>> connectivity(Fixture fixture) {
  if (fixture == Fixture::Disjoint)
    return {{0}, {0}, {1}};
  return {{0, 1, 2}, {0, 1, 2}};
}

std::uint32_t inputCount(Fixture fixture) {
  return fixture == Fixture::Disjoint ? 2 : 3;
}

std::uint32_t outputCount(Fixture fixture) {
  return fixture == Fixture::Disjoint ? 3 : 2;
}

FinalizedFabricRoot makeFixtureModule(llvm::StringRef test,
                                      const ArtifactStore &store,
                                      Fixture fixture) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType tagged8x2 = take(test, PortType::taggedBits(8, 2));
  std::vector<PortType> inputTypes(inputCount(fixture), tagged8x2);
  std::vector<PortType> outputTypes(outputCount(fixture), tagged8x2);
  auto spatial =
      take(test, design.createSpatialCore(
                     ("switch-registered-grant-" + fixtureName(fixture)).str(),
                     inputTypes, outputTypes));
  // A switch whose physical connectivity admits no fan-in must omit the
  // policy; a contended one carries exactly one.
  std::optional<::fabric::TemporalSwitchGrantPolicy> grantPolicy;
  if (fixture == Fixture::RoundRobin)
    grantPolicy = ::fabric::TemporalSwitchGrantPolicy(
        ::fabric::TemporalSwitchRoundRobin{{0, 1, 2}, 0});
  else if (fixture == Fixture::FixedPriority)
    grantPolicy = ::fabric::TemporalSwitchGrantPolicy(
        ::fabric::TemporalSwitchFixedPriority{{1, 0, 2}});
  std::vector<SpatialValue> switchInputs;
  switchInputs.reserve(inputTypes.size());
  for (std::uint32_t input = 0; input != inputCount(fixture); ++input)
    switchInputs.push_back(take(test, spatial.input(input)));
  auto routed = take(
      test,
      spatial.addSwitch(
          switchInputs,
          SwitchSpec::temporal(
              inputTypes, outputTypes, connectivity(fixture),
              static_cast<std::uint32_t>(residentRows(fixture).size()),
              std::move(grantPolicy))));
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

  // Every selection of a resident row is one admitted traversal.
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
  for (const ResidentRow &row : residentRows(fixture)) {
    std::vector<loom::fabric::FabricPhysicalTraversalRef> selected;
    for (std::uint32_t output : row.outputs)
      selected.push_back(traversalOf(row.input, output));
    entries.push_back({llvm::APInt(2, row.tag), std::move(selected)});
  }
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

/// The generated port declarations a fixture's testbench binds by name.
std::string portDeclarations(Fixture fixture) {
  std::string result;
  llvm::raw_string_ostream out(result);
  for (std::uint32_t input = 0; input != inputCount(fixture); ++input)
    out << "  logic [7:0] input_" << input << "_data;\n"
        << "  logic [1:0] input_" << input << "_tag;\n"
        << "  logic       input_" << input << "_valid;\n"
        << "  logic       input_" << input << "_ready;\n";
  for (std::uint32_t output = 0; output != outputCount(fixture); ++output)
    out << "  logic [7:0] output_" << output << "_data;\n"
        << "  logic [1:0] output_" << output << "_tag;\n"
        << "  logic       output_" << output << "_valid;\n"
        << "  logic       output_" << output << "_ready;\n";
  return result;
}

/// The generated accessors and drivers that keep every scenario free of port
/// arity.
std::string portAccessors(Fixture fixture) {
  std::string result;
  llvm::raw_string_ostream out(result);
  out << "  function automatic int unsigned ready_vector();\n"
      << "    ready_vector = 0;\n";
  for (std::uint32_t input = 0; input != inputCount(fixture); ++input)
    out << "    if (input_" << input << "_ready) ready_vector |= (1 << "
        << input << ");\n";
  out << "  endfunction\n\n"
      << "  // A grant is a readiness; a transfer is that grant taken by a\n"
      << "  // valid token, and is what retires the token at the next edge.\n"
      << "  function automatic int unsigned transfer_vector();\n"
      << "    transfer_vector = 0;\n";
  for (std::uint32_t input = 0; input != inputCount(fixture); ++input)
    out << "    if (input_" << input << "_ready && input_" << input
        << "_valid) transfer_vector |= (1 << " << input << ");\n";
  out << "  endfunction\n\n"
      << "  function automatic int unsigned output_valid_vector();\n"
      << "    output_valid_vector = 0;\n";
  for (std::uint32_t output = 0; output != outputCount(fixture); ++output)
    out << "    if (output_" << output
        << "_valid) output_valid_vector |= (1 << " << output << ");\n";
  out << "  endfunction\n\n"
      << "  function automatic int unsigned popcount(input int unsigned "
         "word);\n"
      << "    popcount = 0;\n"
      << "    for (int index = 0; index < 32; index++)\n"
      << "      if (word[index]) popcount = popcount + 1;\n"
      << "  endfunction\n\n"
      << "  task automatic set_valid(input int unsigned word);\n"
      << "    begin\n";
  for (std::uint32_t input = 0; input != inputCount(fixture); ++input)
    out << "      input_" << input << "_valid = word[" << input << "];\n";
  out << "    end\n"
      << "  endtask\n\n"
      << "  task automatic offer(input int unsigned port,\n"
      << "                       input logic [7:0] payload);\n"
      << "    begin\n"
      << "      case (port)\n";
  for (std::uint32_t input = 0; input != inputCount(fixture); ++input)
    out << "        " << input << ": begin input_" << input
        << "_data = payload; input_" << input << "_valid = 1; end\n";
  out << "        default: $fatal(1, \"offer on an absent input port\");\n"
      << "      endcase\n"
      << "    end\n"
      << "  endtask\n\n"
      << "  // Every input presents its resident row's tag at all times: the\n"
      << "  // tag is the switch's capacity term, never a request.\n"
      << "  task automatic present_tags();\n"
      << "    begin\n";
  for (const ResidentRow &row : residentRows(fixture))
    out << "      input_" << row.input << "_tag = 2'd" << row.tag << ";\n";
  out << "    end\n"
      << "  endtask\n\n"
      << "  task automatic release_outputs(input bit released);\n"
      << "    begin\n";
  for (std::uint32_t output = 0; output != outputCount(fixture); ++output)
    out << "      output_" << output << "_ready = released;\n";
  out << "    end\n"
      << "  endtask\n";
  return result;
}

/// The invariant every fixture proves: with the switch's state held, sweeping
/// the complete input valid vector inside one cycle changes no input ready, so
/// no ready of a Temporal switch has a combinational path from any valid. The
/// sweep restores the idle vector before every clock edge, so the grant
/// pointer, whose next value does observe that vector, never moves. Its final
/// reading leaves `idle_ready` naming the idle component's pointer.
std::string validSweep(Fixture fixture) {
  std::string result;
  llvm::raw_string_ostream out(result);
  out << "    for (int combination = 1; combination < "
      << (1u << inputCount(fixture)) << "; combination++) begin\n"
      << "      @(negedge clock);\n"
      << "      set_valid(0);\n"
      << "      #1;\n"
      << "      idle_ready = ready_vector();\n"
      << "      set_valid(combination);\n"
      << "      #1;\n"
      << "      check(ready_vector() == idle_ready,\n"
      << "            $sformatf(\"Input readiness observed valid vector "
         "%0d\",\n"
      << "                      combination));\n"
      << "      set_valid(0);\n"
      << "    end\n";
  return result;
}

void writeArtifacts(const std::filesystem::path &root,
                    const SwitchArtifact &artifact) {
  const llvm::StringRef test = __func__;
  const Fixture fixture = artifact.fixture;
  const std::string prefix =
      ("switch_registered_grant_" + fixtureName(fixture)).str();
  std::ofstream(root / (prefix + ".ys"))
      << "\nread_verilog -sv " << prefix << R"ys(_module.sv
hierarchy -check -top loom_module
check -assert
proc
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys";
  std::ofstream(root / (prefix + "_module.sv")) << artifact.systemVerilog;
  std::ofstream testbench(root / (prefix + "_testbench.sv"));
  testbench << "\nmodule " << prefix << "_testbench;\n"
            << "  logic       clock;\n"
            << "  logic       reset;\n"
            << portDeclarations(fixture)
            << loom::hardware::test::portableAxiLiteSignalDeclarations()
            << R"sv(
  loom_module dut(.*);
  always #5 clock = ~clock;

  task automatic check(bit condition, string message);
    if (!condition)
      $fatal(1, "%s", message);
  endtask

)sv" << portAccessors(fixture)
            << loom::hardware::test::portableAxiLiteDriverTasks()
            << loom::hardware::test::portableCycleWatchdog()
            << R"sv(
  int unsigned idle_ready;
  int unsigned observed;
  int unsigned waited;
  int unsigned granted;

  initial begin
    clock = 0;
    reset = 1;
)sv";
  for (std::uint32_t input = 0; input != inputCount(fixture); ++input)
    testbench << "    input_" << input << "_data = 0;\n"
              << "    input_" << input << "_tag = 0;\n"
              << "    input_" << input << "_valid = 0;\n";
  testbench << "    release_outputs(1);\n"
            << loom::hardware::test::portableAxiLiteInitialization()
            << R"sv(    repeat (2) @(posedge clock);
    #1 reset = 0;
    #1;
    check(ready_vector() == 0 && output_valid_vector() == 0,
          "Disabled Temporal switch presented readiness or a token");

)sv";
  testbench << take(test, loom::hardware::test::portableAxiLiteProgramAndVerify(
                              artifact.target, artifact.image));
  testbench << "    @(negedge clock);\n"
            << "    present_tags();\n\n"
            << validSweep(fixture);

  if (fixture == Fixture::Disjoint) {
    testbench << R"sv(
    // Each input owns its own physical component, so no pointer rotates and
    // both inputs stay ready while their selected outputs are.
    repeat (4) begin
      @(posedge clock);
      #1;
      check(ready_vector() == 'b11,
            "A one-input component withheld readiness");
      check(output_valid_vector() == 0, "An idle component asserted valid");
    end

    // The multicast is atomic: its source is not ready, and the selected
    // output whose sibling refuses is not valid, until every selected output
    // is ready. The independent component is untouched.
    @(negedge clock);
    output_1_ready = 0;
    offer(0, 8'd165);
    repeat (3) begin
      @(posedge clock);
      #1;
      check(ready_vector() == 'b10,
            "The multicast source retired before a selected output was ready");
      check(output_valid_vector() == 'b010,
            "A multicast output fired without its sibling's readiness");
      check(output_0_data == 8'd165 && output_0_tag == 2'd2 &&
                output_1_data == 8'd165 && output_1_tag == 2'd2,
            "The multicast did not present its source on every selection");
    end
    @(negedge clock);
    set_valid(0);
    output_1_ready = 1;

    // With every selected output ready both components transfer in the cycle
    // their valid arrives, together and without waiting for a grant.
    @(negedge clock);
    offer(0, 8'd165);
    offer(1, 8'd90);
    #1;
    check(ready_vector() == 'b11 && output_valid_vector() == 'b111,
          "Output-disjoint components did not transfer together");
    check(output_0_data == 8'd165 && output_0_tag == 2'd2 &&
              output_1_data == 8'd165 && output_1_tag == 2'd2 &&
              output_2_data == 8'd90 && output_2_tag == 2'd1,
          "A transfer did not carry its exact payload to every selection");
    @(posedge clock);
    @(negedge clock);
    set_valid(0);
    repeat (4) begin
      @(posedge clock);
      #1;
      check(output_valid_vector() == 0, "A retired token repeated");
    end
    $finish;
  end
endmodule
)sv";
    return;
  }

  // Both contended fixtures put the three inputs in one physical component.
  testbench << R"sv(
    // An idle component holds its pointer at its reset requester, and that
    // pointer names exactly one input whatever the inputs are doing.
    repeat (4) begin
      @(posedge clock);
      #1;
      check(popcount(ready_vector()) == 1,
            "A contention component did not name exactly one requester");
      check(ready_vector() == idle_ready,
            "An idle component moved its grant pointer");
      check(output_valid_vector() == 0, "An idle component asserted valid");
    end

)sv";

  if (fixture == Fixture::RoundRobin) {
    testbench << R"sv(
    check(idle_ready == 'b001,
          "RoundRobin did not reset its pointer at its reset requester");

    // A fresh request on an idle component waits one cycle: the pointer
    // cannot name it in the cycle its valid rises, and takes it in the next.
    @(negedge clock);
    offer(1, 8'd11);
    #1;
    check(ready_vector() == 'b001 && transfer_vector() == 0,
          "A grant observed a valid of the same cycle");
    @(posedge clock);
    #1;
    check(ready_vector() == 'b010 && output_1_valid &&
              output_1_data == 8'd11 && output_1_tag == 2'd1,
          "A fresh request was not granted in the cycle after its valid");

    // A lone continuous stream keeps its own turn and transfers every cycle.
    repeat (4) begin
      @(posedge clock);
      #1;
      check(transfer_vector() == 'b010 && output_1_valid,
            "A lone continuous requester lost a cycle to the rotation");
    end
    @(negedge clock);
    set_valid(0);

    // Two continuous contenders alternate, one transfer per cycle.
    @(negedge clock);
    offer(0, 8'd7);
    offer(2, 8'd9);
    granted = 0;
    for (waited = 0; waited < 4 && granted == 0; waited = waited + 1) begin
      @(posedge clock);
      #1;
      granted = transfer_vector();
    end
    check(granted == 'b001 || granted == 'b100,
          "Continuous contenders never reached a transfer");
    repeat (4) begin
      @(posedge clock);
      #1;
      observed = transfer_vector();
      check(popcount(ready_vector()) == 1,
            "A contention component granted more than one input");
      check(observed == (granted == 'b001 ? 'b100 : 'b001),
            "Two continuous contenders did not alternate");
      check(output_0_valid && popcount(output_valid_vector()) == 1 &&
                output_0_data == (observed[0] ? 8'd7 : 8'd9),
            "An alternating transfer did not carry its exact payload");
      granted = observed;
    end
    @(negedge clock);
    input_2_valid = 0;
)sv";
  } else {
    testbench << R"sv(
    check(idle_ready == 'b010,
          "FixedPriority did not reset its pointer at its first requester");

    // A fresh request waits one cycle: the pointer cannot name it in the
    // cycle its valid rises, and takes it in the next.
    @(negedge clock);
    offer(0, 8'd7);
    #1;
    check(ready_vector() == 'b010 && transfer_vector() == 0,
          "A grant observed a valid of the same cycle");
    @(posedge clock);
    #1;
    check(ready_vector() == 'b001 && output_0_valid && output_0_data == 8'd7,
          "A fresh request was not granted in the cycle after its valid");

    // A lone continuous stream keeps its turn and transfers every cycle.
    repeat (3) begin
      @(posedge clock);
      #1;
      check(transfer_vector() == 'b001 && output_0_valid,
            "A lone continuous requester lost a cycle");
    end

    // The order is 1, 0, 2: a higher-priority requester takes the pointer in
    // the cycle after its valid, and the lower one resumes when it retires.
    @(negedge clock);
    offer(1, 8'd11);
    #1;
    check(transfer_vector() == 'b001,
          "FixedPriority reordered inside the request cycle");
    @(posedge clock);
    #1;
    check(transfer_vector() == 'b010 && output_1_valid &&
              output_1_data == 8'd11,
          "FixedPriority did not take its highest-priority requester");
    check(popcount(ready_vector()) == 1,
          "FixedPriority named more than one requester");
    @(posedge clock);
    @(negedge clock);
    input_1_valid = 0;
    @(posedge clock);
    @(posedge clock);
    #1;
    check(transfer_vector() == 'b001 && output_0_valid,
          "FixedPriority did not resume its remaining requester");
)sv";
  }

  testbench << R"sv(
    // A pointed requester whose downstream refuses holds its turn: no other
    // input of the component is granted while it waits, and its output stays
    // valid waiting for that readiness.
    @(negedge clock);
    input_1_valid = 0;
    input_2_valid = 0;
    offer(0, 8'd21);
    output_0_ready = 0;
    repeat (3) @(posedge clock);
    repeat (6) begin
      @(posedge clock);
      #1;
      check(ready_vector() == 0,
            "A blocked requester released its contention component");
      check(output_valid_vector() == 'b01 && output_0_data == 8'd21,
            "A blocked requester did not hold its exact output");
    end
    @(negedge clock);
    output_0_ready = 1;
    #1;
    check(input_0_ready && output_0_valid && output_0_data == 8'd21 &&
              output_0_tag == 2'd0,
          "The held grant did not transfer when its output accepted");
    @(posedge clock);
    @(negedge clock);
    set_valid(0);

)sv";

  testbench << R"sv(
    repeat (4) begin
      @(posedge clock);
      #1;
      check(output_valid_vector() == 0, "A retired token repeated");
    end
    $finish;
  end
endmodule
)sv";
}

} // namespace

int main(int argc, char **argv) {
  require("main", argc == 3,
          "expected an output directory and one fixture name");
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  std::optional<Fixture> selected;
  for (const Fixture fixture :
       {Fixture::Disjoint, Fixture::RoundRobin, Fixture::FixedPriority})
    if (fixtureName(fixture) == argv[2]) {
      selected = fixture;
      break;
    }
  require("main", selected.has_value(), "unknown switch fixture name");
  writeArtifacts(root, buildArtifact(root / "store", *selected));
  return 0;
}

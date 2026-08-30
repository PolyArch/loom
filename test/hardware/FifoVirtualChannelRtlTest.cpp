// Builds one tagged per-tag-virtual-channel FIFO Fabric fixture, exports its
// portable SystemVerilog, and emits a cycle-exact conformance testbench whose
// expected trace is computed by the simulator's own transport storage queue
// (loom::sim::detail::CgraTransportStorageRuntime). The testbench drives the
// exported RTL cycle by cycle and compares the presented tag value, valid,
// input ready, dequeue grants, occupancy, and the offer cursor register
// against the oracle trace.

#include "ConfigurationABITestSupport.h"
#include "ConfigurationTransportTestSupport.h"

#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/RTL/CommonSkeleton.h"

#include "CgraTransportStorageRuntime.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kFifoDepth = 4;
constexpr std::uint32_t kTagWidthBits = 4;
constexpr std::uint32_t kPayloadWidthBits = 8;
constexpr std::uint32_t kTagMask = (1U << kTagWidthBits) - 1;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test.str() << ": " << message << '\n';
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

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

template <typename T>
void takeCommit(llvm::StringRef test, llvm::Expected<T> result) {
  if (!result)
    fail(test, llvm::toString(result.takeError()));
}

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fifo-virtual-channel-rtl-test", path))
      fail(test, error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << test_ << ": unable to remove temporary directory: "
                   << error.message() << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string test_;
  std::string path_;
};

/// One scripted stimulus cycle: an optional enqueue offer (tag and payload)
/// and the downstream ready level.
struct StimulusCycle final {
  std::optional<std::uint32_t> enqueueTag;
  std::uint32_t enqueueData = 0;
  bool outputReady = false;
};

/// The expected cycle-start observation of the RTL under one stimulus cycle.
struct ExpectedCycle final {
  bool inputReady = false;
  bool outputValid = false;
  std::uint32_t outputTag = 0;
  std::uint32_t outputData = 0;
  std::uint32_t occupancy = 0;
  std::uint32_t cursor = 0;
  bool grant = false;
};

struct Fixture final {
  loom::fabric::FinalizedFabricRoot module;
  loom::fabric::FinalizedFabricRoot system;
  loom::hardware::FinalizedConfigurationABI abi;
  loom::fabric::SpatialCoreOccurrenceRef spatialCore;
  loom::fabric::FabricFifoOccurrenceRef fifo;
};

/// The programming-unit target and image bytes that activate the fixture.
using ConfigurationImage =
    std::pair<loom::hardware::test::PortableConfigurationTarget,
              std::vector<std::uint8_t>>;

loom::fabric::FinalizedFabricRoot makeVirtualChannelFifoFabric(
    llvm::StringRef test, const loom::ArtifactStore &store) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType tagged = take(
      test, PortType::taggedBits(kPayloadWidthBits, kTagWidthBits));
  auto spatial = take(test, design.createSpatialCore("fifo-virtual-channel",
                                                     {tagged}, {tagged}));
  auto fifo = take(test,
                   spatial.addFifo(take(test, spatial.input(0)),
                                   FifoSpec{tagged, kFifoDepth, false,
                                            ::fabric::FifoQueueDiscipline::
                                                PerTagVirtualChannel}));
  requireSuccess(test, spatial.close({fifo.value()}));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "virtual-channel fixture did not finalize one Module");
  return std::move(finalized.roots().front());
}

Fixture makeFixture(llvm::StringRef test, const loom::ArtifactStore &store) {
  loom::fabric::FinalizedFabricRoot module =
      makeVirtualChannelFifoFabric(test, store);
  require(test, module.view().fifoOccurrences().size() == 1,
          "virtual-channel fixture changed its FIFO inventory");
  const loom::fabric::FabricFifoOccurrenceRef fifo =
      module.view().fifoOccurrences().front();
  require(test,
          module.view().fifoQueueDiscipline(fifo) ==
              ::fabric::FifoQueueDiscipline::PerTagVirtualChannel,
          "virtual-channel fixture lost its queue discipline");
  loom::fabric::FinalizedFabricRoot system =
      take(test, loom::hardware::test::makeSpatialCoreSystem(module, store, 1));
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore =
      take(test, loom::hardware::test::requireSingleSpatialCoreOccurrence(
                     system));
  auto abiDraft =
      take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(system));
  loom::hardware::FinalizedConfigurationABI abi =
      take(test, loom::hardware::finalizeConfigurationABI(std::move(abiDraft),
                                                          store));
  return Fixture{std::move(module), std::move(system), std::move(abi),
                 spatialCore, fifo};
}

loom::fabric::FabricPhysicalConfigurationFieldRef
qualifyConfigurationField(llvm::StringRef test,
                          loom::fabric::SpatialCoreOccurrenceRef spatialCore,
                          const loom::fabric::FabricSemanticConfigFieldRef &field) {
  auto target =
      take(test, loom::fabric::FabricModulePhysicalTargetRef::create(field));
  return take(test, loom::fabric::FabricPhysicalConfigurationFieldRef::create(
                        loom::fabric::SpatialCoreInternalOccurrenceRef{
                            spatialCore, std::move(target)}));
}

/// Computes the programming image that sets the fixture FIFO's configuration
/// field to Buffered, per programming unit.
std::vector<ConfigurationImage>
bufferedConfigurationImages(llvm::StringRef test, const Fixture &fixture) {
  const loom::fabric::FabricSemanticConfigFieldRef field{
      loom::fabric::FabricConfigurationOwnerRef(
          loom::fabric::FabricInventoryOwnerRef::of(fixture.fifo)),
      0};
  const auto bufferedBytes = take(
      test, loom::fabric::encodeFabricFifoConfiguration(
                fixture.module.view(), field,
                loom::fabric::FabricFifoTraversalMode::Buffered));
  const auto physical =
      qualifyConfigurationField(test, fixture.spatialCore, field);
  const auto slot = take(test, loom::fabric::qualifyFabricConfigurationSlot(
                                   physical, loom::fabric::
                                                 FabricStaticConfigurationResidency{}));
  std::vector<ConfigurationImage> images;
  for (const auto &unit : fixture.abi.abi().programmingUnits()) {
    std::vector<loom::hardware::SemanticConfigurationValue> values;
    for (const auto &unitField : unit.fields)
      if (unitField.slot == slot)
        values.push_back({slot,
                          std::vector<std::uint8_t>(bufferedBytes.bytes().begin(),
                                                    bufferedBytes.bytes().end())});
    images.emplace_back(
        take(test, loom::hardware::test::derivePortableConfigurationTarget(
                       fixture.abi, fixture.spatialCore, unit.id)),
        take(test, fixture.abi.abi().encode(unit.id, values)));
  }
  require(test, !images.empty(), "fixture has no programming unit");
  return images;
}

/// Drives the simulator storage queue with the scripted stimulus and records
/// the cycle-start expectation of every cycle. Coverage counters prove the
/// scenario reaches the discipline's interesting states.
struct OracleTrace final {
  std::vector<ExpectedCycle> cycles;
  std::uint64_t refusedOffers = 0;
  std::uint64_t nonHeadGrants = 0;
  std::uint64_t fullCyclesWithRejectedEnqueue = 0;
  std::uint64_t simultaneousGrantEnqueue = 0;
  std::uint64_t drainToEmpty = 0;
  std::uint64_t cursorWrapAtMaximumTag = 0;
  std::uint32_t maxResidentChannels = 0;
};

OracleTrace computeOracleTrace(llvm::StringRef test,
                               llvm::ArrayRef<StimulusCycle> stimulus) {
  using loom::sim::detail::CgraTransportStorageEntry;
  using loom::sim::detail::CgraTransportStorageRuntime;
  auto oracle = take(test, CgraTransportStorageRuntime::create(
                               kFifoDepth, false,
                               ::fabric::FifoQueueDiscipline::
                                   PerTagVirtualChannel));
  OracleTrace trace;
  std::vector<std::uint32_t> tokenData;
  std::uint64_t nextToken = 0;
  std::uint64_t previousOccupancy = 0;
  for (const StimulusCycle &cycle : stimulus) {
    const std::optional<CgraTransportStorageEntry> offered =
        oracle.offeredEntry();
    ExpectedCycle expected;
    expected.inputReady = !oracle.full();
    expected.outputValid = offered.has_value();
    if (offered) {
      expected.outputTag = offered->virtualChannelKey;
      expected.outputData = tokenData[offered->transferSlot];
    }
    expected.occupancy = oracle.occupancy();
    expected.cursor = oracle.offerCursor() & kTagMask;
    expected.grant = expected.outputValid && cycle.outputReady;
    trace.cycles.push_back(expected);

    trace.maxResidentChannels =
        std::max(trace.maxResidentChannels, oracle.distinctResidentChannels());
    if (expected.outputValid && !cycle.outputReady)
      ++trace.refusedOffers;
    if (oracle.full() && cycle.enqueueTag)
      ++trace.fullCyclesWithRejectedEnqueue;
    if (expected.grant) {
      std::vector<CgraTransportStorageEntry> order;
      oracle.appendQueueOrder(order);
      if (order.size() > 1 && order.front().transferSlot != offered->transferSlot)
        ++trace.nonHeadGrants;
      if (offered->virtualChannelKey == kTagMask)
        ++trace.cursorWrapAtMaximumTag;
    }
    if (expected.grant && cycle.enqueueTag && expected.inputReady)
      ++trace.simultaneousGrantEnqueue;

    std::optional<CgraTransportStorageEntry> enqueue;
    if (cycle.enqueueTag && expected.inputReady) {
      CgraTransportStorageEntry entry;
      entry.transferSlot = nextToken++;
      entry.virtualChannelKey = *cycle.enqueueTag;
      enqueue = entry;
      tokenData.push_back(cycle.enqueueData);
    }
    if (expected.grant) {
      // commit dequeues (advancing the cursor past the granted channel)
      // before it appends the enqueue.
      takeCommit(test, oracle.commit(enqueue, offered));
    } else {
      // A refused offer names the channel presented from the cycle-start
      // state, so the cursor advance must not observe this cycle's enqueue.
      if (expected.outputValid && !cycle.outputReady)
        oracle.advanceOffer();
      if (enqueue)
        takeCommit(test, oracle.commit(enqueue, std::nullopt));
    }
    if (oracle.occupancy() == 0 && previousOccupancy != 0)
      ++trace.drainToEmpty;
    previousOccupancy = oracle.occupancy();
  }
  require(test, oracle.occupancy() == 0, "scenario does not drain the queue");
  require(test, trace.refusedOffers >= 4,
          "scenario does not exercise refused-offer rotation");
  require(test, trace.nonHeadGrants >= 2,
          "scenario does not exercise non-head compaction");
  require(test, trace.fullCyclesWithRejectedEnqueue >= 1,
          "scenario does not exercise a full queue rejecting an enqueue");
  require(test, trace.simultaneousGrantEnqueue >= 1,
          "scenario does not exercise simultaneous grant and enqueue");
  require(test, trace.drainToEmpty >= 1,
          "scenario does not exercise drain to the empty state");
  require(test, trace.cursorWrapAtMaximumTag >= 1,
          "scenario does not exercise cursor wrap at the maximum tag value");
  require(test, trace.maxResidentChannels >= 3,
          "scenario does not exercise three resident channels");
  return trace;
}

std::vector<StimulusCycle> makeStimulus() {
  const auto enqueue = [](std::uint32_t tag, std::uint32_t data,
                          bool ready) {
    return StimulusCycle{tag, data, ready};
  };
  const auto idle = [](bool ready) { return StimulusCycle{std::nullopt, 0, ready}; };
  return {
      enqueue(5, 0, false),  // first token enters the empty queue
      enqueue(3, 1, false),  // tag 5 presented and refused; cursor moves past 5
      enqueue(5, 2, false),  // wrap presents tag 3; refused
      enqueue(9, 3, false),  // tag 5 presented; refused; queue becomes full
      enqueue(1, 4, false),  // full queue rejects the enqueue; tag 9 refused
      idle(false),           // tag 3 refused; cursor wraps to the lowest value
      idle(true),            // tag 5 granted (head slot)
      enqueue(1, 4, true),   // rejected token reoffered; tag 9 refused
      idle(true),            // tag 3 granted
      enqueue(15, 5, true),  // tag 5 granted with a simultaneous enqueue
      idle(true),            // tag 9 granted
      idle(true),            // tag 15 granted from a non-head slot; cursor wraps
      idle(true),            // tag 1 granted; queue drains empty
      idle(false),           // empty queue presents nothing
      enqueue(2, 6, false),  // refill after drain; tag 2 presented and refused
      enqueue(1, 7, false),  // tag 1 re-enters below the cursor; presented
      idle(true),            // tag 1 granted from a non-head slot
      idle(true),            // tag 2 granted; queue drains empty again
      idle(false),
  };
}

std::string decimalConstant(std::uint32_t value, unsigned width) {
  return std::to_string(width) + "'d" + std::to_string(value);
}

/// Renders the conformance testbench. The DUT register probes use the exact
/// instance and register names the RTL emitter assigns.
std::string renderTestbench(
    const Fixture &fixture,
    const std::vector<ConfigurationImage> &images,
    llvm::ArrayRef<StimulusCycle> stimulus, const OracleTrace &trace) {
  const std::string fifoInstance = "fifo_" + std::to_string(fixture.fifo.id());
  const std::string occupancyProbe = "dut." + fifoInstance + ".occupancy_reg";
  const std::string cursorProbe = "dut." + fifoInstance + ".offer_cursor_reg";
  std::ostringstream out;
  out << "module vc_fifo_testbench;\n"
         "  logic clock;\n"
         "  logic reset;\n"
      << loom::hardware::test::portableAxiLiteSignalDeclarations() << '\n'
      << "  logic [" << (kPayloadWidthBits - 1) << ":0] input_0_data;\n"
         "  logic ["
      << (kTagWidthBits - 1)
      << ":0] input_0_tag;\n"
         "  logic input_0_valid;\n"
         "  logic input_0_ready;\n"
         "  logic ["
      << (kPayloadWidthBits - 1) << ":0] output_0_data;\n"
      << "  logic [" << (kTagWidthBits - 1)
      << ":0] output_0_tag;\n"
         "  logic output_0_valid;\n"
         "  logic output_0_ready;\n"
         "\n"
         "  loom_module dut(.*);\n"
         "\n"
         "  always #5 clock = ~clock;\n"
      << loom::hardware::test::portableAxiLiteDriverTasks()
      << loom::hardware::test::portableCycleWatchdog() << '\n'
      << R"sv(
  task automatic check_cycle(
    input integer cycle,
    input logic expected_input_ready,
    input logic expected_output_valid,
    input logic [)sv"
      << (kTagWidthBits - 1) << R"sv(:0] expected_output_tag,
    input logic [)sv"
      << (kPayloadWidthBits - 1) << R"sv(:0] expected_output_data,
    input logic [)sv"
      << 2 << R"sv(:0] expected_occupancy,
    input logic [)sv"
      << (kTagWidthBits - 1) << R"sv(:0] expected_cursor,
    input logic expected_grant
  );
    begin
      if (input_0_ready !== expected_input_ready)
        $fatal(1, "cycle %0d: input_ready %b, expected %b", cycle,
               input_0_ready, expected_input_ready);
      if (output_0_valid !== expected_output_valid)
        $fatal(1, "cycle %0d: output_valid %b, expected %b", cycle,
               output_0_valid, expected_output_valid);
      if (expected_output_valid) begin
        if (output_0_tag !== expected_output_tag)
          $fatal(1, "cycle %0d: output tag %0d, expected %0d", cycle,
                 output_0_tag, expected_output_tag);
        if (output_0_data !== expected_output_data)
          $fatal(1, "cycle %0d: output data %0d, expected %0d", cycle,
                 output_0_data, expected_output_data);
      end
      if ((output_0_valid & output_0_ready) !== expected_grant)
        $fatal(1, "cycle %0d: dequeue grant changed", cycle);
      if ()sv"
      << occupancyProbe << R"sv( !== expected_occupancy)
        $fatal(1, "cycle %0d: occupancy %0d, expected %0d", cycle,
               )sv"
      << occupancyProbe << ", expected_occupancy);\n"
      << "      if (" << cursorProbe << R"sv( !== expected_cursor)
        $fatal(1, "cycle %0d: offer cursor %0d, expected %0d", cycle,
               )sv"
      << cursorProbe << ", expected_cursor);\n"
      << R"sv(    end
  endtask

  integer cycle;
  initial begin
    clock = 0;
    reset = 1;
    input_0_data = 0;
    input_0_tag = 0;
    input_0_valid = 0;
    output_0_ready = 0;
)sv"
      << loom::hardware::test::portableAxiLiteInitialization() << R"sv(
    repeat (2) @(posedge clock);
    @(negedge clock);
    #1;
    if (output_0_valid !== 1'b0 || input_0_ready !== 1'b0)
      $fatal(1, "reset did not clear the queue offers");
    if ()sv"
      << occupancyProbe << " !== 0 || " << cursorProbe << R"sv( !== 0)
      $fatal(1, "reset did not return the queue to the canonical empty state");
    reset = 0;
)sv";
  for (const auto &[target, image] : images) {
    auto program =
        loom::hardware::test::portableAxiLiteProgramAndVerify(target, image);
    if (!program)
      fail("renderTestbench", llvm::toString(program.takeError()));
    out << *program;
  }
  out << "    repeat (2) @(posedge clock);\n";
  for (std::size_t ordinal = 0; ordinal != trace.cycles.size(); ++ordinal) {
    const ExpectedCycle &expected = trace.cycles[ordinal];
    const StimulusCycle &stimulusCycle = stimulus[ordinal];
    out << "    @(negedge clock);\n"
        << "    input_0_valid = "
        << (stimulusCycle.enqueueTag ? "1'b1" : "1'b0") << ";\n"
        << "    input_0_tag = " << kTagWidthBits << "'d"
        << stimulusCycle.enqueueTag.value_or(0) << ";\n"
        << "    input_0_data = " << kPayloadWidthBits << "'d"
        << stimulusCycle.enqueueData << ";\n"
        << "    output_0_ready = " << (stimulusCycle.outputReady ? "1'b1" : "1'b0")
        << ";\n"
        << "    #1;\n"
        << "    check_cycle(" << ordinal << ", "
        << (expected.inputReady ? "1'b1" : "1'b0") << ", "
        << (expected.outputValid ? "1'b1" : "1'b0") << ", "
        << decimalConstant(expected.outputTag, kTagWidthBits) << ", "
        << decimalConstant(expected.outputData, kPayloadWidthBits) << ", "
        << decimalConstant(expected.occupancy, 3) << ", "
        << decimalConstant(expected.cursor, kTagWidthBits) << ", "
        << (expected.grant ? "1'b1" : "1'b0") << ");\n"
        << "    @(posedge clock);\n";
  }
  out << R"sv(    @(negedge clock);
    #1;
    if (output_0_valid !== 1'b0)
      $fatal(1, "queue did not stay drained after the scenario");
    $write("vc_fifo_conformance_passed\n");
    $finish;
  end
endmodule
)sv";
  return out.str();
}

llvm::Error writeConformanceArtifacts(const std::filesystem::path &root,
                                      llvm::StringRef systemVerilog,
                                      llvm::StringRef testbench) {
  std::filesystem::create_directories(root);
  std::ofstream(root / "vc_fifo_module.sv") << systemVerilog.str();
  std::ofstream(root / "vc_fifo_testbench.sv") << testbench.str();
  std::ofstream(root / "vc_fifo.ys") << R"ys(
read_verilog -sv vc_fifo_module.sv
hierarchy -check -top loom_module
check -assert
proc
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys";
  return llvm::Error::success();
}

} // namespace

int main(int argc, char **argv) {
  const llvm::StringRef test = "fifo_virtual_channel_rtl";
  if (argc != 1 && argc != 2)
    fail(test, "expected at most one output directory");
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  const Fixture fixture = makeFixture(test, store);

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto skeleton = take(test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                                 context, fixture.spatialCore, fixture.abi));
  require(test, skeleton.operationLeaves.empty(),
          "virtual-channel fixture unexpectedly owns a physical operation");
  const std::string systemVerilog = take(
      test, loom::hardware::rtl::lowerAndExportSpecializedSystemVerilog(
                *skeleton.module));
  require(test,
          llvm::StringRef(systemVerilog).contains("offer_cursor_reg") &&
              llvm::StringRef(systemVerilog).contains("occupancy_reg") &&
              !llvm::StringRef(systemVerilog).contains("head_reg") &&
              !llvm::StringRef(systemVerilog).contains("tail_reg"),
          "virtual-channel FIFO RTL kept the strict pointer registers");
  require(test,
          !llvm::StringRef(systemVerilog).contains("bypass"),
          "virtual-channel FIFO RTL kept a bypass path");

  const std::vector<StimulusCycle> stimulus = makeStimulus();
  const OracleTrace trace = computeOracleTrace(test, stimulus);
  const std::vector<ConfigurationImage> images =
      bufferedConfigurationImages(test, fixture);
  const std::string testbench = renderTestbench(fixture, images, stimulus, trace);

  if (argc == 2)
    requireSuccess(test, writeConformanceArtifacts(argv[1], systemVerilog,
                                                   testbench));
  return EXIT_SUCCESS;
}

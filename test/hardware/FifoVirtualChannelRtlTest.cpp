// Builds one tagged per-tag-virtual-channel FIFO Fabric fixture, exports its
// portable SystemVerilog, and emits a cycle-exact conformance testbench whose
// expected trace is computed by the simulator's own transport storage queue
// (loom::sim::detail::CgraTransportStorageRuntime). The stimulus is not a
// hand-written scenario: the test enumerates every queue state reachable
// from reset under every single-cycle stimulus (one optional enqueue tag and
// the downstream ready level) and walks a stimulus sequence that exercises
// every reachable state/stimulus pair. The testbench drives the exported RTL
// cycle by cycle and compares the presented tag value, valid, input ready,
// dequeue grants, occupancy, and the offer cursor register against the
// oracle trace, so the RTL offer tournament and its hole-closing dequeue are
// checked against the simulator on every reachable arbitration situation.

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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

/// An odd slot count exercises the carry-over leaf of the balanced offer
/// tournament; a two-bit tag keeps the reachable state space small enough
/// to enumerate completely while still wrapping the cursor at the top value.
constexpr std::uint32_t kFifoDepth = 5;
constexpr std::uint32_t kTagWidthBits = 2;
constexpr std::uint32_t kPayloadWidthBits = 8;
constexpr std::uint32_t kTagMask = (1U << kTagWidthBits) - 1;
constexpr std::uint32_t kTagValueCount = 1U << kTagWidthBits;
/// The single-cycle stimulus alphabet: no enqueue or one of the tag values,
/// each with the downstream ready level low or high.
constexpr std::uint32_t kStimulusCount = 2 * (1 + kTagValueCount);
constexpr std::uint32_t kOccupancyBits = 3;
static_assert((1U << kOccupancyBits) > kFifoDepth,
              "occupancy field must hold the full queue");

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

/// One single-cycle stimulus: an optional enqueue offer (tag and payload)
/// and the downstream ready level.
struct StimulusCycle final {
  std::optional<std::uint32_t> enqueueTag;
  std::uint32_t enqueueData = 0;
  bool outputReady = false;
};

/// The stimulus alphabet index: the low bit is the ready level; the rest
/// selects no enqueue (0) or the enqueue tag value plus one.
StimulusCycle stimulusOf(std::uint32_t index) {
  StimulusCycle cycle;
  cycle.outputReady = (index & 1U) != 0;
  const std::uint32_t enqueue = index >> 1;
  if (enqueue != 0)
    cycle.enqueueTag = enqueue - 1;
  return cycle;
}

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
    llvm::StringRef test, const loom::ArtifactStore &store,
    std::uint32_t depth = kFifoDepth) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType tagged = take(
      test, PortType::taggedBits(kPayloadWidthBits, kTagWidthBits));
  auto spatial = take(test, design.createSpatialCore("fifo-virtual-channel",
                                                     {tagged}, {tagged}));
  auto fifo = take(test,
                   spatial.addFifo(take(test, spatial.input(0)),
                                   FifoSpec{tagged, depth, false,
                                            ::fabric::FifoQueueDiscipline::
                                                PerTagVirtualChannel}));
  requireSuccess(test, spatial.close({fifo.value()}));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "virtual-channel fixture did not finalize one Module");
  return std::move(finalized.roots().front());
}

Fixture makeFixture(llvm::StringRef test, const loom::ArtifactStore &store,
                    std::uint32_t depth = kFifoDepth) {
  loom::fabric::FinalizedFabricRoot module =
      makeVirtualChannelFifoFabric(test, store, depth);
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

using OracleQueue = loom::sim::detail::CgraTransportStorageRuntime;
using OracleEntry = loom::sim::detail::CgraTransportStorageEntry;

OracleQueue makeOracle(llvm::StringRef test) {
  return take(test, OracleQueue::create(
                        kFifoDepth, false,
                        ::fabric::FifoQueueDiscipline::PerTagVirtualChannel));
}

/// The arbitration state of the oracle: the resident tag values in arrival
/// order and the cursor reduced to the tag width. A cursor past the highest
/// tag value and the zero cursor present the same channel, exactly as the
/// RTL cursor register wraps.
std::string queueStateKey(const OracleQueue &queue) {
  std::vector<OracleEntry> entries;
  queue.appendQueueOrder(entries);
  std::string key;
  for (const OracleEntry &entry : entries)
    key += static_cast<char>('0' + entry.virtualChannelKey);
  key += '/';
  key += static_cast<char>('0' + (queue.offerCursor() & kTagMask));
  return key;
}

/// Applies one stimulus cycle to the oracle queue: the cycle-start
/// observation the RTL must show, then the queue transition the cycle
/// commits. Payloads are tracked per token so the presented data is checked
/// as well as the presented tag.
struct OracleStep final {
  ExpectedCycle expected;
  std::optional<OracleEntry> offered;
  bool enqueued = false;
  bool drainedToEmpty = false;
  bool nonHeadGrant = false;
};

OracleStep stepOracle(llvm::StringRef test, OracleQueue &queue,
                      const StimulusCycle &cycle, std::uint64_t &nextToken,
                      std::vector<std::uint32_t> &tokenData) {
  OracleStep step;
  step.offered = queue.offeredEntry();
  ExpectedCycle &expected = step.expected;
  expected.inputReady = !queue.full();
  expected.outputValid = step.offered.has_value();
  if (step.offered) {
    expected.outputTag = step.offered->virtualChannelKey;
    expected.outputData = tokenData[step.offered->transferSlot];
  }
  expected.occupancy = queue.occupancy();
  expected.cursor = queue.offerCursor() & kTagMask;
  expected.grant = expected.outputValid && cycle.outputReady;
  const std::uint32_t occupancyBefore = queue.occupancy();
  if (expected.grant) {
    std::vector<OracleEntry> order;
    queue.appendQueueOrder(order);
    step.nonHeadGrant =
        order.size() > 1 && order.front().transferSlot != step.offered->transferSlot;
  }
  std::optional<OracleEntry> enqueue;
  if (cycle.enqueueTag && expected.inputReady) {
    OracleEntry entry;
    entry.transferSlot = nextToken++;
    entry.virtualChannelKey = *cycle.enqueueTag;
    enqueue = entry;
    tokenData.push_back(cycle.enqueueData);
    step.enqueued = true;
  }
  if (expected.grant) {
    // commit dequeues (advancing the cursor past the granted channel) before
    // it appends the enqueue.
    takeCommit(test, queue.commit(enqueue, step.offered));
  } else {
    // A refused offer names the channel presented from the cycle-start
    // state, so the cursor advance must not observe this cycle's enqueue.
    if (expected.outputValid && !cycle.outputReady)
      queue.advanceOffer();
    if (enqueue)
      takeCommit(test, queue.commit(enqueue, std::nullopt));
  }
  step.drainedToEmpty = occupancyBefore != 0 && queue.occupancy() == 0;
  return step;
}

/// One reachable oracle state with its successor under every stimulus.
struct ReachableState final {
  OracleQueue queue;
  std::array<std::size_t, kStimulusCount> successors{};
  std::bitset<kStimulusCount> covered;
};

/// Enumerates every queue state reachable from reset under the stimulus
/// alphabet. Payloads do not affect arbitration, so the enumeration tracks
/// tags and the cursor only.
std::vector<ReachableState> enumerateReachableStates(llvm::StringRef test) {
  std::vector<ReachableState> states;
  std::map<std::string, std::size_t> index;
  const OracleQueue reset = makeOracle(test);
  index.emplace(queueStateKey(reset), 0);
  states.push_back(ReachableState{reset, {}, {}});
  for (std::size_t ordinal = 0; ordinal != states.size(); ++ordinal) {
    for (std::uint32_t input = 0; input != kStimulusCount; ++input) {
      OracleQueue successor = states[ordinal].queue;
      std::vector<OracleEntry> resident;
      successor.appendQueueOrder(resident);
      std::uint64_t nextToken = 0;
      for (const OracleEntry &entry : resident)
        nextToken = std::max<std::uint64_t>(nextToken, entry.transferSlot + 1);
      std::vector<std::uint32_t> tokenData(nextToken, 0);
      (void)stepOracle(test, successor, stimulusOf(input), nextToken,
                       tokenData);
      const std::string successorKey = queueStateKey(successor);
      auto [position, inserted] = index.emplace(successorKey, states.size());
      if (inserted)
        states.push_back(ReachableState{std::move(successor), {}, {}});
      states[ordinal].successors[input] = position->second;
    }
  }
  return states;
}

/// The complete conformance stimulus: a walk from reset that applies every
/// stimulus in every reachable state at least once and ends with the queue
/// drained. Between coverage steps the walk follows a shortest known path to
/// the nearest state with an uncovered stimulus.
struct ExhaustiveWalk final {
  std::vector<StimulusCycle> stimulus;
  std::vector<ExpectedCycle> expected;
  std::uint64_t reachableStates = 0;
  std::uint64_t coveredTransitions = 0;
  std::uint64_t refusedOffers = 0;
  std::uint64_t nonHeadGrants = 0;
  std::uint64_t fullCyclesWithRejectedEnqueue = 0;
  std::uint64_t simultaneousGrantEnqueue = 0;
  std::uint64_t drainToEmpty = 0;
  std::uint64_t cursorWrapAtMaximumTag = 0;
  std::uint32_t maxResidentChannels = 0;
};

ExhaustiveWalk computeExhaustiveWalk(llvm::StringRef test) {
  std::vector<ReachableState> states = enumerateReachableStates(test);
  const auto hasUncovered = [](const ReachableState &state) {
    return !state.covered.all();
  };
  std::vector<std::uint32_t> inputs;
  std::size_t current = 0;
  std::vector<std::size_t> parentState(states.size());
  std::vector<std::uint32_t> parentInput(states.size());
  std::vector<bool> visited(states.size());
  while (true) {
    ReachableState &state = states[current];
    if (hasUncovered(state)) {
      std::uint32_t input = 0;
      while (state.covered.test(input))
        ++input;
      state.covered.set(input);
      inputs.push_back(input);
      current = state.successors[input];
      continue;
    }
    // Breadth-first search for the nearest state with an uncovered stimulus.
    std::fill(visited.begin(), visited.end(), false);
    std::deque<std::size_t> frontier{current};
    visited[current] = true;
    std::optional<std::size_t> target;
    while (!frontier.empty() && !target) {
      const std::size_t ordinal = frontier.front();
      frontier.pop_front();
      for (std::uint32_t input = 0; input != kStimulusCount; ++input) {
        const std::size_t next = states[ordinal].successors[input];
        if (visited[next])
          continue;
        visited[next] = true;
        parentState[next] = ordinal;
        parentInput[next] = input;
        if (hasUncovered(states[next])) {
          target = next;
          break;
        }
        frontier.push_back(next);
      }
    }
    if (!target)
      break;
    std::vector<std::uint32_t> path;
    for (std::size_t ordinal = *target; ordinal != current;
         ordinal = parentState[ordinal])
      path.push_back(parentInput[ordinal]);
    std::reverse(path.begin(), path.end());
    inputs.insert(inputs.end(), path.begin(), path.end());
    current = *target;
  }
  // Drain: grant every remaining channel head.
  for (std::uint32_t cycle = 0; cycle != kFifoDepth; ++cycle)
    inputs.push_back(1);

  ExhaustiveWalk walk;
  walk.reachableStates = states.size();
  for (const ReachableState &state : states)
    walk.coveredTransitions += state.covered.count();

  // Replay the walk on one continuous oracle run with payload tracking; this
  // replay, not the enumeration, is the expected trace.
  OracleQueue oracle = makeOracle(test);
  std::vector<std::uint32_t> tokenData;
  std::uint64_t nextToken = 0;
  std::uint32_t nextPayload = 1;
  for (std::uint32_t input : inputs) {
    StimulusCycle cycle = stimulusOf(input);
    if (cycle.enqueueTag) {
      cycle.enqueueData = nextPayload;
      nextPayload = (nextPayload + 1) & ((1U << kPayloadWidthBits) - 1);
    }
    const bool fullBefore = oracle.full();
    const OracleStep step =
        stepOracle(test, oracle, cycle, nextToken, tokenData);
    walk.maxResidentChannels =
        std::max(walk.maxResidentChannels, oracle.distinctResidentChannels());
    walk.refusedOffers += step.expected.outputValid && !cycle.outputReady;
    walk.fullCyclesWithRejectedEnqueue += fullBefore && cycle.enqueueTag;
    walk.nonHeadGrants += step.nonHeadGrant;
    walk.cursorWrapAtMaximumTag +=
        step.expected.grant && step.offered->virtualChannelKey == kTagMask;
    walk.simultaneousGrantEnqueue += step.expected.grant && step.enqueued;
    walk.drainToEmpty += step.drainedToEmpty;
    walk.stimulus.push_back(cycle);
    walk.expected.push_back(step.expected);
  }
  require(test, oracle.occupancy() == 0, "walk does not drain the queue");
  require(test,
          walk.coveredTransitions == walk.reachableStates * kStimulusCount,
          "walk does not cover every reachable state and stimulus");
  require(test, walk.refusedOffers != 0 && walk.nonHeadGrants != 0 &&
                    walk.fullCyclesWithRejectedEnqueue != 0 &&
                    walk.simultaneousGrantEnqueue != 0 &&
                    walk.drainToEmpty != 0 && walk.cursorWrapAtMaximumTag != 0 &&
                    walk.maxResidentChannels == kTagValueCount,
          "walk does not reach every discipline state");
  return walk;
}

/// Packed trace words consumed by the testbench through `$readmemh`.
constexpr unsigned kStimulusWordBits = 2 + kTagWidthBits + kPayloadWidthBits;
constexpr unsigned kExpectationWordBits =
    3 + kTagWidthBits + kPayloadWidthBits + kOccupancyBits + kTagWidthBits;

std::uint32_t packStimulus(const StimulusCycle &cycle) {
  std::uint32_t word = cycle.outputReady ? 1U : 0U;
  word |= (cycle.enqueueTag ? 1U : 0U) << 1;
  word |= cycle.enqueueTag.value_or(0) << 2;
  word |= cycle.enqueueData << (2 + kTagWidthBits);
  return word;
}

std::uint32_t packExpectation(const ExpectedCycle &expected) {
  std::uint32_t word = expected.inputReady ? 1U : 0U;
  word |= (expected.outputValid ? 1U : 0U) << 1;
  word |= (expected.grant ? 1U : 0U) << 2;
  unsigned offset = 3;
  word |= expected.outputTag << offset;
  offset += kTagWidthBits;
  word |= expected.outputData << offset;
  offset += kPayloadWidthBits;
  word |= expected.occupancy << offset;
  offset += kOccupancyBits;
  word |= expected.cursor << offset;
  return word;
}

std::string renderTraceFile(llvm::ArrayRef<std::uint32_t> words,
                            unsigned wordBits) {
  std::string text;
  llvm::raw_string_ostream out(text);
  for (std::uint32_t word : words)
    out << llvm::format_hex_no_prefix(word, (wordBits + 3) / 4, true) << '\n';
  return text;
}

/// Renders the conformance testbench. The DUT register probes use the exact
/// instance and register names the RTL emitter assigns; the trace words are
/// loaded from the two trace files beside the testbench.
std::string renderTestbench(const Fixture &fixture,
                            const std::vector<ConfigurationImage> &images,
                            const ExhaustiveWalk &walk,
                            const std::filesystem::path &stimulusPath,
                            const std::filesystem::path &expectationPath) {
  const std::string fifoInstance = "fifo_" + std::to_string(fixture.fifo.id());
  const std::string occupancyProbe = "dut." + fifoInstance + ".occupancy_reg";
  const std::string cursorProbe = "dut." + fifoInstance + ".offer_cursor_reg";
  const std::size_t cycleCount = walk.stimulus.size();
  const unsigned tagOffset = 3;
  const unsigned dataOffset = tagOffset + kTagWidthBits;
  const unsigned occupancyOffset = dataOffset + kPayloadWidthBits;
  const unsigned cursorOffset = occupancyOffset + kOccupancyBits;
  const unsigned stimulusDataOffset = 2 + kTagWidthBits;
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
         "  logic ["
      << (kStimulusWordBits - 1) << ":0] stimulus [0:" << (cycleCount - 1)
      << "];\n"
      << "  logic [" << (kExpectationWordBits - 1) << ":0] expectation [0:"
      << (cycleCount - 1) << "];\n"
      << "  logic [" << (kStimulusWordBits - 1) << ":0] stimulus_word;\n"
      << "  logic [" << (kExpectationWordBits - 1) << ":0] expected_word;\n"
         "\n"
         "  loom_module dut(.*);\n"
         "\n"
         "  always #5 clock = ~clock;\n"
      << loom::hardware::test::portableAxiLiteDriverTasks()
      << loom::hardware::test::portableCycleWatchdog(cycleCount + 4096) << '\n'
      << R"sv(
  task automatic check_cycle(
    input integer cycle,
    input logic [)sv"
      << (kExpectationWordBits - 1) << R"sv(:0] expected
  );
    begin
      if (input_0_ready !== expected[0])
        $fatal(1, "cycle %0d: input_ready %b, expected %b", cycle,
               input_0_ready, expected[0]);
      if (output_0_valid !== expected[1])
        $fatal(1, "cycle %0d: output_valid %b, expected %b", cycle,
               output_0_valid, expected[1]);
      if (expected[1]) begin
        if (output_0_tag !== expected[)sv"
      << (tagOffset + kTagWidthBits - 1) << ':' << tagOffset << R"sv(])
          $fatal(1, "cycle %0d: output tag %0d, expected %0d", cycle,
                 output_0_tag, expected[)sv"
      << (tagOffset + kTagWidthBits - 1) << ':' << tagOffset << R"sv(]);
        if (output_0_data !== expected[)sv"
      << (dataOffset + kPayloadWidthBits - 1) << ':' << dataOffset << R"sv(])
          $fatal(1, "cycle %0d: output data %0d, expected %0d", cycle,
                 output_0_data, expected[)sv"
      << (dataOffset + kPayloadWidthBits - 1) << ':' << dataOffset << R"sv(]);
      end
      if ((output_0_valid & output_0_ready) !== expected[2])
        $fatal(1, "cycle %0d: dequeue grant changed", cycle);
      if ()sv"
      << occupancyProbe << " !== expected[" << (occupancyOffset + kOccupancyBits - 1)
      << ':' << occupancyOffset << R"sv(])
        $fatal(1, "cycle %0d: occupancy %0d, expected %0d", cycle,
               )sv"
      << occupancyProbe << ", expected[" << (occupancyOffset + kOccupancyBits - 1)
      << ':' << occupancyOffset << "]);\n"
      << "      if (" << cursorProbe << " !== expected["
      << (cursorOffset + kTagWidthBits - 1) << ':' << cursorOffset << R"sv(])
        $fatal(1, "cycle %0d: offer cursor %0d, expected %0d", cycle,
               )sv"
      << cursorProbe << ", expected[" << (cursorOffset + kTagWidthBits - 1)
      << ':' << cursorOffset << "]);\n"
      << R"sv(    end
  endtask

  integer cycle;
  initial begin
    $readmemh(")sv"
      << stimulusPath.generic_string() << "\", stimulus);\n"
      << "    $readmemh(\"" << expectationPath.generic_string()
      << R"sv(", expectation);
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
  out << "    repeat (2) @(posedge clock);\n"
      << "    for (cycle = 0; cycle < " << cycleCount
      << "; cycle = cycle + 1) begin\n"
      << "      @(negedge clock);\n"
      << "      stimulus_word = stimulus[cycle];\n"
      << "      expected_word = expectation[cycle];\n"
      << "      input_0_valid = stimulus_word[1];\n"
      << "      input_0_tag = stimulus_word[" << (2 + kTagWidthBits - 1)
      << ":2];\n"
      << "      input_0_data = stimulus_word["
      << (stimulusDataOffset + kPayloadWidthBits - 1) << ':'
      << stimulusDataOffset << "];\n"
      << "      output_0_ready = stimulus_word[0];\n"
      << "      #1;\n"
      << "      check_cycle(cycle, expected_word);\n"
      << "      @(posedge clock);\n"
      << "    end\n"
      << R"sv(    @(negedge clock);
    input_0_valid = 1'b0;
    output_0_ready = 1'b0;
    #1;
    if (output_0_valid !== 1'b0)
      $fatal(1, "queue did not stay drained after the walk");
    $write("vc_fifo_conformance_passed cycles=%0d\n", )sv"
      << cycleCount << R"sv();
    $finish;
  end
endmodule
)sv";
  return out.str();
}

llvm::Error writeConformanceArtifacts(const std::filesystem::path &root,
                                      llvm::StringRef systemVerilog,
                                      llvm::StringRef testbench,
                                      llvm::StringRef stimulusTrace,
                                      llvm::StringRef expectationTrace) {
  std::filesystem::create_directories(root);
  std::ofstream(root / "vc_fifo_module.sv") << systemVerilog.str();
  std::ofstream(root / "vc_fifo_testbench.sv") << testbench.str();
  std::ofstream(root / "vc_fifo_stimulus.hex") << stimulusTrace.str();
  std::ofstream(root / "vc_fifo_expectation.hex") << expectationTrace.str();
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

unsigned combinationalDepth(mlir::Value value,
                            llvm::DenseMap<mlir::Value, unsigned> &memo) {
  if (!value || mlir::isa<mlir::BlockArgument>(value))
    return 0;
  const auto cached = memo.find(value);
  if (cached != memo.end())
    return cached->second;
  mlir::Operation *operation = value.getDefiningOp();
  if (!operation)
    return 0;
  const bool combinational =
      operation->getName().getDialectNamespace() == "comb" ||
      mlir::isa<circt::hw::ArrayCreateOp, circt::hw::ArrayGetOp>(operation);
  if (!combinational)
    return 0;
  unsigned depth = 0;
  for (mlir::Value operand : operation->getOperands())
    depth = std::max(depth, combinationalDepth(operand, memo));
  return memo.try_emplace(value, depth + 1).first->second;
}

unsigned fifoCombinationalDepth(llvm::StringRef test, mlir::ModuleOp module) {
  circt::hw::HWModuleOp fifo;
  module.walk([&](circt::hw::HWModuleOp candidate) {
    if (!candidate.getSymName().starts_with("loom_fabric_fifo_"))
      return;
    require(test, !fifo, "fixture contains multiple FIFO definitions");
    fifo = candidate;
  });
  require(test, static_cast<bool>(fifo), "fixture omitted its FIFO definition");
  llvm::DenseMap<mlir::Value, unsigned> memo;
  unsigned depth = 0;
  fifo.walk([&](mlir::Operation *operation) {
    for (mlir::Value result : operation->getResults())
      depth = std::max(depth, combinationalDepth(result, memo));
  });
  return depth;
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
  const unsigned fixtureDepth = fifoCombinationalDepth(test, *skeleton.module);

  constexpr std::uint32_t structuralFifoDepth = 32;
  const Fixture structuralFixture =
      makeFixture(test, store, structuralFifoDepth);
  mlir::MLIRContext structuralContext;
  structuralContext.loadDialect<
      circt::comb::CombDialect, circt::hw::HWDialect,
      circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto structuralSkeleton = take(
      test, loom::hardware::rtl::buildModuleRootCirctSkeleton(
                structuralContext, structuralFixture.spatialCore,
                structuralFixture.abi));
  const unsigned structuralDepth =
      fifoCombinationalDepth(test, *structuralSkeleton.module);
  constexpr unsigned reductionDepthPerLevel = 6;
  const unsigned additionalLevels =
      llvm::Log2_64_Ceil(structuralFifoDepth) -
      llvm::Log2_64_Ceil(kFifoDepth);
  require(test,
          structuralDepth <=
              fixtureDepth + reductionDepthPerLevel * additionalLevels,
          "virtual-channel FIFO selector depth did not scale logarithmically");
  llvm::outs() << "fifo_selector_comb_depth depth" << kFifoDepth << '='
               << fixtureDepth << " depth" << structuralFifoDepth << '='
               << structuralDepth << '\n';

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

  const ExhaustiveWalk walk = computeExhaustiveWalk(test);
  llvm::outs() << "vc_fifo_exhaustive_walk states=" << walk.reachableStates
               << " transitions=" << walk.coveredTransitions
               << " cycles=" << walk.stimulus.size()
               << " refused_offers=" << walk.refusedOffers
               << " non_head_grants=" << walk.nonHeadGrants
               << " simultaneous_grant_enqueue="
               << walk.simultaneousGrantEnqueue << '\n';

  if (argc == 2) {
    const std::filesystem::path root =
        std::filesystem::absolute(std::filesystem::path(argv[1]));
    const std::vector<ConfigurationImage> images =
        bufferedConfigurationImages(test, fixture);
    std::vector<std::uint32_t> stimulusWords;
    std::vector<std::uint32_t> expectationWords;
    stimulusWords.reserve(walk.stimulus.size());
    expectationWords.reserve(walk.expected.size());
    for (const StimulusCycle &cycle : walk.stimulus)
      stimulusWords.push_back(packStimulus(cycle));
    for (const ExpectedCycle &expected : walk.expected)
      expectationWords.push_back(packExpectation(expected));
    const std::string testbench =
        renderTestbench(fixture, images, walk, root / "vc_fifo_stimulus.hex",
                        root / "vc_fifo_expectation.hex");
    requireSuccess(
        test, writeConformanceArtifacts(
                  root, systemVerilog, testbench,
                  renderTraceFile(stimulusWords, kStimulusWordBits),
                  renderTraceFile(expectationWords, kExpectationWordBits)));
  }
  return EXIT_SUCCESS;
}

#include "DFGSimulatorInternal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace loom::sim::detail;

namespace {

namespace sem = dataflow::semantics;

constexpr llvm::StringLiteral fixture = R"mlir(
module {
  func.func @parallelize(%data: i8, %phase: i1) {
    %vector, %mask, %group_phase =
      dataflow.parallelize %data, %phase
        : (i8, i1) -> (vector<2xi8>, vector<2xi1>, i1)
    %packed = dataflow.pack %vector : vector<2xi8> -> i16
    return
  }

  func.func @wide_parallelize(%data: i8, %phase: i1) {
    %vector, %mask, %group_phase =
      dataflow.parallelize %data, %phase
        : (i8, i1) -> (vector<8xi8>, vector<8xi1>, i1)
    return
  }

  func.func @serialize(%packed: i32, %packed_mask: i4, %group_phase: i1) {
    %vector = dataflow.unpack %packed : i32 -> vector<4xi8>
    %mask = dataflow.unpack %packed_mask : i4 -> vector<4xi1>
    %data, %scalar_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<4xi8>, vector<4xi1>, i1) -> (i8, i1)
    return
  }

  func.func @rank_two(%packed: i48, %vector: vector<2x3xi8>) {
    %lanes = dataflow.unpack %packed : i48 -> vector<2x3xi8>
    %bits = dataflow.pack %vector : vector<2x3xi8> -> i48
    return
  }

  func.func @gate(%cond: i1, %value: i8) {
    %after_cond, %after_value = dataflow.gate %cond, %value : i8
    return
  }

  func.func @stream(%init: i16, %limit: i16, %step: i16) {
    %iv, %phase = dataflow.stream %init, %limit, %step step add while slt : i16
    return
  }

  func.func @memory(%mem: memref<?xi8>, %addr: index,
                    %addresses: vector<2xindex>, %data: vector<2xi8>,
                    %ctrl: none) {
    %loaded, %read = dataflow.load %mem[%addr] %ctrl : memref<?xi8>
    %written = dataflow.store %mem[%addresses] %data %ctrl
        : memref<?xi8>, vector<2xindex>, vector<2xi8>
    return
  }

  func.func @llvm_load(%ptr: !llvm.ptr) {
    %data = llvm.load %ptr {alignment = 4 : i64} : !llvm.ptr -> i32
    return
  }
}
)mlir";

// Row-major flattened lane order for the fixture's vector<2x3xi8>: element
// [row][column] owns flattened lane row * 3 + column, lane zero owns the least
// significant byte, and the whole value packs to 0x060504030201.
constexpr unsigned kRows = 2;
constexpr unsigned kColumns = 3;
constexpr unsigned kElementWidth = 8;
constexpr unsigned kRankTwoWidth = kRows * kColumns * kElementWidth;
constexpr uint64_t kRankTwoPacked = 0x060504030201ULL;
constexpr uint8_t kLanes[kRows][kColumns] = {{0x01, 0x02, 0x03},
                                             {0x04, 0x05, 0x06}};

unsigned rowMajorLane(unsigned row, unsigned column) {
  return row * kColumns + column;
}

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "DFGVectorBoundaryTest: " << message << "\n";
  std::exit(1);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename Input>
sem::SemanticInputMask consumedMask(std::initializer_list<Input> inputs) {
  sem::SemanticInputMask mask = 0;
  for (Input input : inputs)
    mask |= sem::semanticInput(input);
  return mask;
}

// The registered operation schema for each stateful actor owns a closed set of
// named transition-case descriptors. Each descriptor is the sole statement of
// its case's required logical state, consumed operand heads, active results and
// their value sources, and next logical state. These assertions pin all
// fourteen descriptors in the closed contract; one representative blocked
// readiness per actor confirms that an unsatisfied case consumes nothing and
// holds its state, which is not itself a transition case.
void actorTransitionDescriptorContract() {
  using sem::StreamCase;
  using sem::StreamInput;
  using sem::StreamMode;
  using sem::StreamOutputSource;
  const sem::SemanticInputMask streamActivation = consumedMask<StreamInput>(
      {StreamInput::Init, StreamInput::Limit, StreamInput::Step});
  {
    auto d = sem::streamCaseDescriptor(StreamCase::StartTrue);
    require(d.requiredMode == StreamMode::Idle &&
                d.consumedInputs == streamActivation &&
                d.ivSource == StreamOutputSource::Current && d.emitPhase &&
                d.phase && d.nextMode == StreamMode::Running,
            "stream StartTrue descriptor");
  }
  {
    auto d = sem::streamCaseDescriptor(StreamCase::StartClose);
    require(d.requiredMode == StreamMode::Idle &&
                d.consumedInputs == streamActivation &&
                d.ivSource == StreamOutputSource::None && d.emitPhase &&
                !d.phase && d.nextMode == StreamMode::Idle,
            "stream StartClose descriptor");
  }
  {
    auto d = sem::streamCaseDescriptor(StreamCase::ContinueTrue);
    require(d.requiredMode == StreamMode::Running && d.consumedInputs == 0 &&
                d.ivSource == StreamOutputSource::Current && d.emitPhase &&
                d.phase && d.nextMode == StreamMode::Running,
            "stream ContinueTrue descriptor");
  }
  {
    auto d = sem::streamCaseDescriptor(StreamCase::ContinueClose);
    require(d.requiredMode == StreamMode::Running && d.consumedInputs == 0 &&
                d.ivSource == StreamOutputSource::None && d.emitPhase &&
                !d.phase && d.nextMode == StreamMode::Idle,
            "stream ContinueClose descriptor");
  }

  using sem::CarryCase;
  using sem::CarryInput;
  using sem::CarrySemanticState;
  {
    auto d = sem::carryCaseDescriptor(CarryCase::Init);
    require(d.requiredState == CarrySemanticState::Initial &&
                d.consumedInputs ==
                    consumedMask<CarryInput>({CarryInput::Init}) &&
                d.forwardedInput == CarryInput::Init &&
                d.nextState == CarrySemanticState::Running,
            "carry Init descriptor");
  }
  {
    auto d = sem::carryCaseDescriptor(CarryCase::Next);
    require(d.requiredState == CarrySemanticState::Running &&
                d.consumedInputs ==
                    consumedMask<CarryInput>(
                        {CarryInput::Phase, CarryInput::Next}) &&
                d.forwardedInput == CarryInput::Next &&
                d.nextState == CarrySemanticState::Running,
            "carry Next descriptor");
  }
  {
    auto d = sem::carryCaseDescriptor(CarryCase::Close);
    require(d.requiredState == CarrySemanticState::Running &&
                d.consumedInputs ==
                    consumedMask<CarryInput>({CarryInput::Phase}) &&
                !d.forwardedInput && d.nextState == CarrySemanticState::Initial,
            "carry Close descriptor");
  }

  using sem::InvariantCase;
  using sem::InvariantInput;
  using sem::InvariantOutputSource;
  using sem::InvariantSemanticState;
  {
    auto d = sem::invariantCaseDescriptor(InvariantCase::Init);
    require(d.requiredState == InvariantSemanticState::Initial &&
                d.consumedInputs ==
                    consumedMask<InvariantInput>({InvariantInput::Init}) &&
                d.output == InvariantOutputSource::InitInput &&
                d.latchInput == InvariantInput::Init && !d.clearLatch &&
                d.nextState == InvariantSemanticState::Running,
            "invariant Init descriptor");
  }
  {
    auto d = sem::invariantCaseDescriptor(InvariantCase::Replay);
    require(d.requiredState == InvariantSemanticState::Running &&
                d.consumedInputs ==
                    consumedMask<InvariantInput>({InvariantInput::Phase}) &&
                d.output == InvariantOutputSource::Latched && !d.latchInput &&
                !d.clearLatch && d.nextState == InvariantSemanticState::Running,
            "invariant Replay descriptor");
  }
  {
    auto d = sem::invariantCaseDescriptor(InvariantCase::Close);
    require(d.requiredState == InvariantSemanticState::Running &&
                d.consumedInputs ==
                    consumedMask<InvariantInput>({InvariantInput::Phase}) &&
                d.output == InvariantOutputSource::None && !d.latchInput &&
                d.clearLatch && d.nextState == InvariantSemanticState::Initial,
            "invariant Close descriptor");
  }

  using sem::GateCase;
  using sem::GateInput;
  using sem::GateSemanticState;
  const sem::SemanticInputMask gateHeads =
      consumedMask<GateInput>({GateInput::Phase, GateInput::Value});
  {
    auto d = sem::gateCaseDescriptor(GateCase::ClosedDrop);
    require(d.requiredState == GateSemanticState::Closed &&
                d.consumedInputs == gateHeads && !d.emitPhase &&
                !d.forwardedInput && d.nextState == GateSemanticState::Closed,
            "gate ClosedDrop descriptor");
  }
  {
    auto d = sem::gateCaseDescriptor(GateCase::FirstTrue);
    require(d.requiredState == GateSemanticState::Closed &&
                d.consumedInputs == gateHeads && !d.emitPhase &&
                d.forwardedInput == GateInput::Value &&
                d.nextState == GateSemanticState::Open,
            "gate FirstTrue descriptor");
  }
  {
    auto d = sem::gateCaseDescriptor(GateCase::ContinueTrue);
    require(d.requiredState == GateSemanticState::Open &&
                d.consumedInputs == gateHeads && d.emitPhase && d.phase &&
                d.forwardedInput == GateInput::Value &&
                d.nextState == GateSemanticState::Open,
            "gate ContinueTrue descriptor");
  }
  {
    auto d = sem::gateCaseDescriptor(GateCase::Close);
    require(d.requiredState == GateSemanticState::Open &&
                d.consumedInputs == gateHeads && d.emitPhase && !d.phase &&
                !d.forwardedInput && d.nextState == GateSemanticState::Closed,
            "gate Close descriptor");
  }

  // Blocked readiness: an unsatisfied case is not another transition case. Each
  // evaluator consumes nothing and holds its logical state.
  {
    sem::StreamSemanticState idle;
    auto blocked = sem::evaluateStreamTransition(
        idle, sem::StreamSemanticConfig{}, std::nullopt);
    require(static_cast<bool>(blocked), "stream blocked readiness evaluated");
    require(!blocked->firing.ready &&
                blocked->firing.consumedInputCount() == 0 && !blocked->emitIv &&
                !blocked->emitPhase &&
                blocked->nextState.mode == StreamMode::Idle,
            "stream blocked readiness consumed nothing and held state");
  }
  {
    auto blocked = sem::evaluateCarryTransition(CarrySemanticState::Initial,
                                                std::nullopt, false, false);
    require(!blocked.firing.ready && blocked.firing.consumedInputCount() == 0 &&
                !blocked.forwardedInput &&
                blocked.nextState == CarrySemanticState::Initial,
            "carry blocked readiness consumed nothing and held state");
  }
  {
    auto blocked = sem::evaluateInvariantTransition(
        InvariantSemanticState::Initial, std::nullopt, false);
    require(!blocked.firing.ready && blocked.firing.consumedInputCount() == 0 &&
                blocked.output == InvariantOutputSource::None &&
                !blocked.latchInput && !blocked.clearLatch &&
                blocked.nextState == InvariantSemanticState::Initial,
            "invariant blocked readiness consumed nothing and held state");
  }
  {
    auto blocked = sem::evaluateGateTransition(GateSemanticState::Closed,
                                               std::nullopt, false);
    require(!blocked.firing.ready && blocked.firing.consumedInputCount() == 0 &&
                !blocked.emitPhase && !blocked.forwardedInput &&
                blocked.nextState == GateSemanticState::Closed,
            "gate blocked readiness consumed nothing and held state");
  }
}

// A token keeps its bit pattern in one APInt, whose width is an unsigned. An
// exact vector width past that is reported at this boundary instead of being
// narrowed into a legal one. The exact width is named in the diagnostic and no
// value of that width is ever built, so the check stays arithmetic.
void tokenWidthNarrowsAtTokenBoundary() {
  mlir::MLIRContext context;
  mlir::Type wide =
      mlir::VectorType::get({257}, mlir::IntegerType::get(&context, 16777215));
  llvm::Expected<unsigned> width = tokenTypeBitWidth(wide);
  require(!width, "a width past the token representation was accepted");
  const std::string message = llvm::toString(width.takeError());
  require(message == "bit width 4311744255 exceeds the token representation",
          "the token boundary did not report its own narrowing: " + message);
}

template <typename T> T takeExpected(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

Token tokenWithBits(mlir::Type type, uint64_t value) {
  unsigned width = takeExpected(tokenTypeBitWidth(type));
  return takeExpected(tokenFromBitPattern(llvm::APInt(width, value), type));
}

Token malformedToken(TokenKind kind, unsigned width) {
  Token token;
  token.kind = kind;
  token.bitPattern = llvm::APInt(width, 0);
  return token;
}

uint64_t bitsOf(const Token &token, mlir::Type type) {
  return takeExpected(tokenBitPattern(token, type)).getZExtValue();
}

void expectBits(llvm::ArrayRef<Token> tokens, mlir::Type type,
                std::initializer_list<uint64_t> expected,
                llvm::StringRef message) {
  require(tokens.size() == expected.size(), message);
  for (auto [token, value] : llvm::zip_equal(tokens, expected))
    require(bitsOf(token, type) == value, message);
}

void expectPhases(llvm::ArrayRef<Token> tokens,
                  std::initializer_list<bool> expected,
                  llvm::StringRef message) {
  require(tokens.size() == expected.size(), message);
  for (auto [token, value] : llvm::zip_equal(tokens, expected))
    require(boolToken(token) == value, message);
}

void flushPending(SimulatorState &state) {
  for (auto &entry : state.pendingChannels) {
    auto &target = state.channels[entry.first];
    while (!entry.second.empty()) {
      target.push_back(entry.second.front());
      entry.second.pop_front();
    }
  }
  state.pendingChannels.clear();
  for (auto &entry : state.pendingObservedOutputs) {
    auto &target = state.observedOutputs[entry.first];
    target.append(entry.second.begin(), entry.second.end());
  }
  state.pendingObservedOutputs.clear();
}

void parallelizePreservesQueuedActivation(dataflow::ParallelizeOp op) {
  SimulatorState state;
  auto &data = state.channels[&op.getDataMutable()];
  data.push_back(tokenWithBits(op.getData().getType(), 17));
  data.push_back(tokenWithBits(op.getData().getType(), 18));
  auto &phase = state.channels[&op.getScalarPhaseMutable()];
  phase.push_back(boolValueToken(true));
  phase.push_back(boolValueToken(false));
  phase.push_back(boolValueToken(true));
  phase.push_back(boolValueToken(false));

  require(fireActorOperation(op, state), "first scalar true did not fire");
  require(data.size() == 1, "first scalar true consumed the wrong payload");
  require(fireActorOperation(op, state), "first scalar false did not fire");
  require(data.size() == 1,
          "scalar false consumed the next activation payload");
  require(fireActorOperation(op, state), "second scalar true did not fire");
  require(fireActorOperation(op, state), "second scalar false did not fire");

  expectBits(state.pendingObservedOutputs[op.getVector()],
             op.getVector().getType(), {17, 18},
             "parallelize did not reset zero-filled lanes");
  expectBits(state.pendingObservedOutputs[op.getMask()], op.getMask().getType(),
             {1, 1}, "parallelize did not reset active masks");
  expectPhases(state.pendingObservedOutputs[op.getGroupPhase()],
               {true, false, true, false},
               "parallelize emitted the wrong activation phases");
}

// One firing attempt owns one frontier: the scheduler clears the accumulator
// before every attempt, so an actor never inherits the order of the previous
// one. Tests that drive an actor directly must reproduce that reset or the
// accumulator silently carries stale effects between firings.
bool fireOnce(mlir::Operation *op, SimulatorState &state) {
  state.firingMemoryOrderFrontier.clear();
  return fireActorOperation(op, state);
}

// A group assembles its memory order across several firings and publishes it
// once, on the firing that emits. The order it is still assembling belongs to
// the retained actor state, not to the firing, so it must survive the
// per-attempt reset and must not be interned before a token carries it.
void parallelizeGroupPublishesItsAssembledOrderOnce(
    dataflow::ParallelizeOp op) {
  SimulatorState state;
  state.memoryOrder = std::make_unique<loom::sim::MemoryAtomicOrder>();
  state.memorySync =
      std::make_unique<loom::sim::MemorySynchronization>(*state.memoryOrder);

  const unsigned lanes = static_cast<unsigned>(
      mlir::cast<mlir::VectorType>(op.getVector().getType())
          .getShape()
          .front());
  auto &data = state.channels[&op.getDataMutable()];
  auto &phase = state.channels[&op.getScalarPhaseMutable()];
  // One true scalar phase per lane fills the group and emits it on the last
  // firing. Each phase carries its own unrelated effect, so the assembled
  // union grows by one every firing and no reduction can shrink it.
  llvm::SmallVector<loom::sim::SyncEffectId> seeded;
  for (unsigned lane = 0; lane < lanes; ++lane) {
    data.push_back(tokenWithBits(op.getData().getType(), 20 + lane));
    Token carrier = boolValueToken(true);
    const loom::sim::SyncEffectId effect = state.memorySync->declareEffect();
    seeded.push_back(effect);
    carrier.memoryOrder = state.memoryOrderFrontiers.internCanonical(effect);
    phase.push_back(std::move(carrier));
  }

  const std::size_t seededFrontiers =
      state.memoryOrderFrontiers.frontierCount();
  for (unsigned firing = 0; firing < lanes; ++firing)
    require(fireOnce(op, state), "a group firing did not fire");

  const auto vector = state.pendingObservedOutputs.find(op.getVector());
  require(vector != state.pendingObservedOutputs.end() &&
              vector->second.size() == 1,
          "the group did not publish exactly one vector token");
  // Every scalar phase consumed while assembling the group is present in the
  // one frontier the group publishes. Without retained phase order the earlier
  // firings' effects would be lost to the per-attempt reset.
  llvm::ArrayRef<loom::sim::SyncEffectId> published =
      state.memoryOrderFrontiers.elements(vector->second.front().memoryOrder);
  require(published.size() == lanes,
          "the published group frontier lost a scalar phase effect");
  for (loom::sim::SyncEffectId effect : seeded)
    require(llvm::is_contained(published, effect),
            "the published group frontier dropped a consumed phase effect");
  // Only the firing that emits may intern, and only the assembled union: a
  // partial cumulative prefix in the arena would raise the retained count
  // above the one seeded singleton per phase plus this one publication.
  require(state.memoryOrderFrontiers.frontierCount() == seededFrontiers + 1,
          "the group interned a partial prefix or more than one publication");
}

// A transition that consumes order but emits nothing must retain no frontier:
// a closed gate drops its inputs, so the order it absorbed is never carried by
// any token and must not reach the arena. Repeating the drop with a fresh
// effect each time would otherwise retain one composite frontier per drop.
void droppedOrderIsNeverRetained(dataflow::GateOp op) {
  SimulatorState state;
  state.memoryOrder = std::make_unique<loom::sim::MemoryAtomicOrder>();
  state.memorySync =
      std::make_unique<loom::sim::MemorySynchronization>(*state.memoryOrder);

  // Each drop consumes two inputs carrying distinct unrelated effects, so the
  // order it absorbs is a composite that no seeded frontier already names.
  constexpr unsigned kDrops = 8;
  auto &cond = state.channels[&op.getBeforeCondMutable()];
  auto &value = state.channels[&op.getBeforeValueMutable()];
  for (unsigned drop = 0; drop < kDrops; ++drop) {
    Token phase = boolValueToken(false);
    phase.memoryOrder = state.memoryOrderFrontiers.internCanonical(
        state.memorySync->declareEffect());
    cond.push_back(std::move(phase));
    Token payload = tokenWithBits(op.getBeforeValue().getType(), drop);
    payload.memoryOrder = state.memoryOrderFrontiers.internCanonical(
        state.memorySync->declareEffect());
    value.push_back(std::move(payload));
  }
  const std::size_t seededFrontiers =
      state.memoryOrderFrontiers.frontierCount();

  for (unsigned drop = 0; drop < kDrops; ++drop)
    require(fireOnce(op, state), "a closed gate drop did not fire");
  require(state.pendingObservedOutputs.empty(),
          "a closed gate drop published a token");
  require(state.memoryOrderFrontiers.frontierCount() == seededFrontiers,
          "a transition that emitted nothing retained a frontier");
}

// Scale gate for the retained activation frontier.
//
// A stateful actor retains its activation's memory-order union and publishes
// it on every firing the activation makes. A stream consumes nothing after
// its start firing, so the retained union does not change across iterations
// and each publication must be a memo lookup: a handoff that drops the
// union's memos reduces and rehashes the whole width on every iteration.
//
// The gate is the elapsed time, not the arena shape: interning deduplicates
// to the same handle either way, so the content assertions below cannot
// reject repeated work. At this width and length the memo-dropping handoff
// measures well beyond the budget, while a memoized handoff costs
// milliseconds. The lit runner additionally bounds the whole process.
void streamRepublishesRetainedOrderWithoutRework(dataflow::StreamOp op) {
  constexpr unsigned kWidth = 32768;
  constexpr unsigned kIterations = 32767;
  constexpr double kBudgetSeconds = 15.0;
  const auto start = std::chrono::steady_clock::now();

  SimulatorState state;
  state.memoryOrder = std::make_unique<loom::sim::MemoryAtomicOrder>();
  state.memorySync =
      std::make_unique<loom::sim::MemorySynchronization>(*state.memoryOrder);

  // The activation's init carries a wide frontier of mutually unrelated
  // effects and its step carries one more, so the retained union is the whole
  // width plus one, no reduction can shrink it, and it names a frontier no
  // seeded token already carries. Declared effect identities ascend, so the
  // seeded frontier is already canonical.
  llvm::SmallVector<loom::sim::SyncEffectId> wide;
  wide.reserve(kWidth);
  for (unsigned index = 0; index < kWidth; ++index)
    wide.push_back(state.memorySync->declareEffect());
  Token init = tokenWithBits(op.getInit().getType(), 0);
  init.memoryOrder = state.memoryOrderFrontiers.internCanonical(wide);
  state.channels[&op.getInitMutable()].push_back(std::move(init));
  state.channels[&op.getLimitMutable()].push_back(
      tokenWithBits(op.getLimit().getType(), kIterations));
  Token step = tokenWithBits(op.getStep().getType(), 1);
  step.memoryOrder = state.memoryOrderFrontiers.internCanonical(
      state.memorySync->declareEffect());
  state.channels[&op.getStepMutable()].push_back(std::move(step));
  const std::size_t seededFrontiers =
      state.memoryOrderFrontiers.frontierCount();

  // One firing starts the activation, one fires per iteration, and one closes
  // it; the close emits only a phase token and retires the frontier.
  for (unsigned firing = 0; firing <= kIterations; ++firing)
    require(fireOnce(op, state), "a stream firing did not fire");

  const auto ivs = state.pendingObservedOutputs.find(op.getIv());
  require(ivs != state.pendingObservedOutputs.end() &&
              ivs->second.size() == kIterations,
          "the stream did not publish one iv per iteration");
  const MemoryOrderFrontierId published = ivs->second.front().memoryOrder;
  require(!published.empty(), "the stream published no memory order");
  for (const Token &iv : ivs->second)
    require(iv.memoryOrder == published,
            "stream iterations published separately resolved frontiers");
  require(state.memoryOrderFrontiers.elements(published).size() == kWidth + 1,
          "the published stream frontier lost members of its activation");
  // The activation interns its union exactly once, on the firing that first
  // publishes it; the retiring close may add nothing to the arena.
  require(state.memoryOrderFrontiers.frontierCount() == seededFrontiers + 1,
          "the stream retained more than its one published frontier");

  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  require(elapsed < kBudgetSeconds, "the stream exceeded its scale budget");
}

void serializePreservesQueuedActivation(dataflow::SerializeOp op,
                                        dataflow::UnpackOp vectorUnpack,
                                        dataflow::UnpackOp maskUnpack) {
  SimulatorState state;
  auto &packed = state.channels[&vectorUnpack.getPackedMutable()];
  packed.push_back(
      tokenWithBits(vectorUnpack.getPacked().getType(), 0x44332211U));
  packed.push_back(
      tokenWithBits(vectorUnpack.getPacked().getType(), 0x44332211U));
  auto &packedMask = state.channels[&maskUnpack.getPackedMutable()];
  packedMask.push_back(tokenWithBits(maskUnpack.getPacked().getType(), 0));
  packedMask.push_back(tokenWithBits(maskUnpack.getPacked().getType(), 5));

  require(fireActorOperation(vectorUnpack, state),
          "first vector unpack did not fire");
  require(fireActorOperation(vectorUnpack, state),
          "second vector unpack did not fire");
  require(fireActorOperation(maskUnpack, state),
          "first mask unpack did not fire");
  require(fireActorOperation(maskUnpack, state),
          "second mask unpack did not fire");
  flushPending(state);

  auto &vectors = state.channels[&op.getVectorMutable()];
  auto &masks = state.channels[&op.getMaskMutable()];
  require(vectors.size() == 2 && masks.size() == 2,
          "unpack did not queue both activation payloads");
  auto &phase = state.channels[&op.getGroupPhaseMutable()];
  phase.push_back(boolValueToken(true));
  phase.push_back(boolValueToken(false));
  phase.push_back(boolValueToken(true));
  phase.push_back(boolValueToken(false));

  require(fireActorOperation(op, state), "all-zero true group did not fire");
  require(vectors.size() == 1 && masks.size() == 1,
          "all-zero true group did not consume its payload");
  require(state.pendingObservedOutputs[op.getData()].empty(),
          "all-zero group emitted scalar data");
  require(fireActorOperation(op, state), "first group false did not fire");
  require(vectors.size() == 1 && masks.size() == 1,
          "group false consumed the next activation payload");
  require(fireActorOperation(op, state), "sparse true group did not fire");
  require(fireActorOperation(op, state), "second group false did not fire");

  expectBits(state.pendingObservedOutputs[op.getData()], op.getData().getType(),
             {0x11, 0x33}, "serialize did not preserve low-slice lane order");
  expectPhases(state.pendingObservedOutputs[op.getScalarPhase()],
               {false, true, true, false},
               "serialize emitted the wrong activation phases");
}

void parallelizeFailureIsAtomic(dataflow::ParallelizeOp op) {
  {
    SimulatorState state;
    state.channels[&op.getDataMutable()].push_back(
        malformedToken(TokenKind::Integer, 16));
    state.channels[&op.getScalarPhaseMutable()].push_back(boolValueToken(true));

    require(!fireActorOperation(op, state),
            "parallelize accepted a malformed scalar token");
    require(state.channels[&op.getDataMutable()].size() == 1 &&
                state.channels[&op.getScalarPhaseMutable()].size() == 1,
            "parallelize consumed input on conversion failure");
    require(!state.parallelizeStates.contains(op.getOperation()),
            "parallelize changed actor state on conversion failure");
    require(state.pendingChannels.empty() &&
                state.pendingObservedOutputs.empty() &&
                state.actorMutationEpoch == 0,
            "parallelize published output on conversion failure");
  }

  {
    SimulatorState state;
    ParallelizeState pending;
    pending.semanticState.pendingItems = 1;
    pending.slots.resize(2);
    pending.slots[0] = malformedToken(TokenKind::Integer, 16);
    state.parallelizeStates[op.getOperation()] = pending;
    state.channels[&op.getScalarPhaseMutable()].push_back(
        boolValueToken(false));

    require(!fireActorOperation(op, state),
            "parallelize assembled a malformed pending group");
    const ParallelizeState &preserved =
        state.parallelizeStates.find(op.getOperation())->second;
    require(preserved.semanticState.pendingItems == 1 && preserved.slots[0] &&
                preserved.slots[0]->bitPattern->getBitWidth() == 16,
            "parallelize changed pending state on group construction failure");
    require(state.channels[&op.getScalarPhaseMutable()].size() == 1,
            "parallelize consumed phase on group construction failure");
    require(state.pendingChannels.empty() &&
                state.pendingObservedOutputs.empty() &&
                state.actorMutationEpoch == 0,
            "parallelize published a malformed pending group");
  }
}

// Independent oracle for the rank-two bit representation: the expected packed
// value is assembled here from the two-dimensional lane table, so a permuted or
// padded lane placement cannot cancel out against its own inverse.
llvm::APInt rowMajorLaneBits() {
  llvm::APInt bits(kRankTwoWidth, 0);
  for (unsigned row = 0; row < kRows; ++row)
    for (unsigned column = 0; column < kColumns; ++column)
      bits.insertBits(llvm::APInt(kElementWidth, kLanes[row][column]),
                      rowMajorLane(row, column) * kElementWidth);
  return bits;
}

void unpackPlacesRowMajorLanes(dataflow::UnpackOp op) {
  SimulatorState state;
  state.channels[&op.getPackedMutable()].push_back(
      tokenWithBits(op.getPacked().getType(), kRankTwoPacked));

  require(fireActorOperation(op, state), "rank-two unpack did not fire");
  auto &published = state.pendingObservedOutputs[op.getVector()];
  require(published.size() == 1,
          "rank-two unpack did not publish exactly one vector token");
  llvm::APInt bits = takeExpected(
      tokenBitPattern(published.front(), op.getVector().getType()));
  require(bits.getBitWidth() == kRankTwoWidth,
          "rank-two unpack published the wrong token bit width");
  for (unsigned row = 0; row < kRows; ++row)
    for (unsigned column = 0; column < kColumns; ++column) {
      llvm::APInt lane = bits.extractBits(
          kElementWidth, rowMajorLane(row, column) * kElementWidth);
      require(lane.getZExtValue() == kLanes[row][column],
              "rank-two unpack placed a lane outside its row-major bit slice");
    }
}

void packFlattensRowMajorLanes(dataflow::PackOp op) {
  SimulatorState state;
  state.channels[&op.getVectorMutable()].push_back(takeExpected(
      tokenFromBitPattern(rowMajorLaneBits(), op.getVector().getType())));

  require(fireActorOperation(op, state), "rank-two pack did not fire");
  auto &published = state.pendingObservedOutputs[op.getPacked()];
  require(published.size() == 1,
          "rank-two pack did not publish exactly one packed token");
  llvm::APInt bits = takeExpected(
      tokenBitPattern(published.front(), op.getPacked().getType()));
  require(bits.getBitWidth() == kRankTwoWidth,
          "rank-two pack published the wrong packed bit width");
  require(bits.getZExtValue() == kRankTwoPacked,
          "rank-two pack did not flatten lanes row-major with lane zero low");
}

void packFailureIsAtomic(dataflow::PackOp op) {
  SimulatorState state;
  state.channels[&op.getVectorMutable()].push_back(
      malformedToken(TokenKind::Vector, 8));

  require(!fireActorOperation(op, state), "pack accepted a malformed vector");
  require(state.channels[&op.getVectorMutable()].size() == 1,
          "pack consumed input on conversion failure");
  require(state.pendingChannels.empty() &&
              state.pendingObservedOutputs.empty() &&
              state.actorMutationEpoch == 0,
          "pack published output on conversion failure");
}

void unpackFailureIsAtomic(dataflow::UnpackOp op) {
  SimulatorState state;
  state.channels[&op.getPackedMutable()].push_back(
      malformedToken(TokenKind::Integer, 8));

  require(!fireActorOperation(op, state),
          "unpack accepted a malformed packed token");
  require(state.channels[&op.getPackedMutable()].size() == 1,
          "unpack consumed input on conversion failure");
  require(state.pendingChannels.empty() &&
              state.pendingObservedOutputs.empty() &&
              state.actorMutationEpoch == 0,
          "unpack published output on conversion failure");
}

void serializeFailureIsAtomic(dataflow::SerializeOp op) {
  SimulatorState state;
  state.channels[&op.getVectorMutable()].push_back(
      malformedToken(TokenKind::Vector, 8));
  state.channels[&op.getMaskMutable()].push_back(
      tokenWithBits(op.getMask().getType(), 1));
  state.channels[&op.getGroupPhaseMutable()].push_back(boolValueToken(true));

  require(!fireActorOperation(op, state),
          "serialize accepted a malformed vector");
  require(state.channels[&op.getVectorMutable()].size() == 1 &&
              state.channels[&op.getMaskMutable()].size() == 1 &&
              state.channels[&op.getGroupPhaseMutable()].size() == 1,
          "serialize consumed input on conversion failure");
  require(state.pendingChannels.empty() &&
              state.pendingObservedOutputs.empty() &&
              state.actorMutationEpoch == 0,
          "serialize published output on conversion failure");
}

// A memory actor rejects an access entirely on peeked inputs. Only its reason
// and the unsupported-capability outcome may change; inputs, outputs, actor
// mutation state, events, fire counts, and memory may not.
std::shared_ptr<MemoryValue>
makeMemory(mlir::Type elementType, std::initializer_list<uint64_t> values) {
  auto memory = std::make_shared<MemoryValue>();
  for (uint64_t value : values)
    memory->elements.push_back(tokenWithBits(elementType, value));
  memory->elementType = elementType;
  memory->initialized = llvm::SmallBitVector(memory->elements.size(), true);
  return memory;
}

unsigned resolvedIndexBits(mlir::Operation *scope) {
  return takeExpected(loom::getIndexBitWidth(scope));
}

Token indexVectorToken(unsigned indexBits,
                       std::initializer_list<uint64_t> lanes) {
  llvm::APInt bits(indexBits * lanes.size(), 0);
  unsigned lane = 0;
  for (uint64_t value : lanes)
    bits.insertBits(llvm::APInt(indexBits, value), indexBits * lane++);
  Token token;
  token.kind = TokenKind::Vector;
  token.bitPattern = bits;
  return token;
}

void expectUntouchedRun(SimulatorState &state, const MemoryValue &memory,
                        llvm::ArrayRef<uint64_t> elements,
                        llvm::StringRef message) {
  require(state.pendingChannels.empty() && state.pendingObservedOutputs.empty(),
          message);
  require(state.actorMutationEpoch == 0 && state.eventCount == 0 &&
              state.operationFireCounts.empty(),
          message);
  require(state.terminalPrimitiveOps.empty(), message);
  require(memory.elements.size() == elements.size(), message);
  for (auto [token, value] : llvm::zip_equal(memory.elements, elements))
    require(bitsOf(token, memory.elementType) == value, message);
}

// A value read out of memory carries a witness the loading firing never
// consumed: the firing consumed only its address token, yet the loaded token
// must keep its resident order. The firing's own order still applies, so the
// loaded token publishes their union, resolved once as one arena entry.
void llvmLoadKeepsItsMemoryResidentWitness(mlir::LLVM::LoadOp op) {
  SimulatorState state;
  state.memoryOrder = std::make_unique<loom::sim::MemoryAtomicOrder>();
  state.memorySync =
      std::make_unique<loom::sim::MemorySynchronization>(*state.memoryOrder);

  auto memory = makeMemory(op.getType(), {0x2A});
  const loom::sim::SyncEffectId witness = state.memorySync->declareEffect();
  memory->elements[0].memoryOrder =
      state.memoryOrderFrontiers.internCanonical(witness);
  Token address = pointerToken(op.getAddr(), memory, 0);
  const loom::sim::SyncEffectId consumed = state.memorySync->declareEffect();
  address.memoryOrder = state.memoryOrderFrontiers.internCanonical(consumed);
  state.channels[&op->getOpOperand(0)].push_back(std::move(address));
  const std::size_t seededFrontiers =
      state.memoryOrderFrontiers.frontierCount();

  require(fireOnce(op, state), "llvm.load of a resident witness did not fire");
  const auto loaded = state.pendingObservedOutputs.find(op.getResult());
  require(loaded != state.pendingObservedOutputs.end() &&
              loaded->second.size() == 1,
          "llvm.load did not publish exactly one data token");
  llvm::ArrayRef<loom::sim::SyncEffectId> published =
      state.memoryOrderFrontiers.elements(loaded->second.front().memoryOrder);
  require(published.size() == 2 && llvm::is_contained(published, witness) &&
              llvm::is_contained(published, consumed),
          "llvm.load dropped its resident witness or its firing's order");
  // The firing's own frontier deduplicates against the seeded singleton, so
  // the merged union is the only new entry.
  require(state.memoryOrderFrontiers.frontierCount() == seededFrontiers + 1,
          "llvm.load interned more than its one merged publication");
}

void loadRejectionIsAtomic(dataflow::LoadOp op) {
  SimulatorState state;
  auto memoryType = mlir::cast<mlir::MemRefType>(op.getMem().getType());
  auto memory = makeMemory(memoryType.getElementType(), {0x11, 0x22});
  state.channels[&op.getMemMutable()].push_back(
      pointerToken(op.getMem(), memory, 0));
  state.channels[&op.getAddrMutable()].push_back(
      indexToken(llvm::APInt(resolvedIndexBits(op.getOperation()), 99)));
  state.channels[&op.getCtrlMutable()].push_back(noneToken());
  state.admittedPlainMemoryActions.try_emplace(op.getOperation(),
                                               ReadyPlainMemoryAction{});

  require(!fireActorOperation(op, state),
          "load accepted an out-of-range address");
  require(state.diagnostics.size() == 1, "load recorded no rejection reason");
  require(!fireActorOperation(op, state),
          "load accepted an out-of-range address when re-polled");
  require(state.diagnostics.size() == 2,
          "a re-polled load rejection is not detectable as a failed attempt");
  require(state.channels[&op.getMemMutable()].size() == 1 &&
              state.channels[&op.getAddrMutable()].size() == 1 &&
              state.channels[&op.getCtrlMutable()].size() == 1,
          "load consumed input on a rejected access");
  expectUntouchedRun(state, *memory, {0x11, 0x22},
                     "load changed run or memory state on a rejected access");
}

void storeDuplicateScatterIsAtomic(dataflow::StoreOp op) {
  SimulatorState state;
  auto memoryType = mlir::cast<mlir::MemRefType>(op.getMem().getType());
  auto memory = makeMemory(memoryType.getElementType(), {0x11, 0x22});
  state.channels[&op.getMemMutable()].push_back(
      pointerToken(op.getMem(), memory, 0));
  state.channels[&op.getAddrMutable()].push_back(
      indexVectorToken(resolvedIndexBits(op.getOperation()), {1, 1}));
  state.channels[&op.getDataMutable()].push_back(
      tokenWithBits(op.getData().getType(), 0xAB43));
  state.channels[&op.getCtrlMutable()].push_back(noneToken());
  state.admittedPlainMemoryActions.try_emplace(op.getOperation(),
                                               ReadyPlainMemoryAction{});

  require(!fireActorOperation(op, state),
          "store accepted duplicate active destinations");
  require(state.diagnostics.size() == 1, "store recorded no rejection reason");
  require(!fireActorOperation(op, state),
          "store accepted duplicate active destinations when re-polled");
  require(state.diagnostics.size() == 2,
          "a re-polled store rejection is not detectable as a failed attempt");
  require(state.channels[&op.getMemMutable()].size() == 1 &&
              state.channels[&op.getAddrMutable()].size() == 1 &&
              state.channels[&op.getDataMutable()].size() == 1 &&
              state.channels[&op.getCtrlMutable()].size() == 1,
          "store consumed input on a rejected access");
  require(state.runtimeUnsupportedCapability,
          "duplicate active scatter did not report an unsupported capability");
  expectUntouchedRun(state, *memory, {0x11, 0x22},
                     "store changed run or memory state on a rejected access");
}

void storeSynchronizationFailureIsAtomic(dataflow::StoreOp op) {
  SimulatorState state;
  auto memoryType = mlir::cast<mlir::MemRefType>(op.getMem().getType());
  auto memory = makeMemory(memoryType.getElementType(), {0x11, 0x22});
  state.channels[&op.getMemMutable()].push_back(
      pointerToken(op.getMem(), memory, 0));
  state.channels[&op.getAddrMutable()].push_back(
      indexVectorToken(resolvedIndexBits(op.getOperation()), {0, 1}));
  state.channels[&op.getDataMutable()].push_back(
      tokenWithBits(op.getData().getType(), 0xAB43));
  Token ctrl = noneToken();
  ctrl.memoryOrder =
      state.memoryOrderFrontiers.internCanonical(loom::sim::SyncEffectId(99));
  state.channels[&op.getCtrlMutable()].push_back(std::move(ctrl));
  PlainMemoryActionProjection projected =
      projectReadyPlainMemoryAction(op.getOperation(), state);
  require(projected.ready && projected.diagnostics.empty(),
          "unable to project synchronization-failure store");
  state.admittedPlainMemoryActions.try_emplace(op.getOperation(),
                                               std::move(*projected.ready));

  require(!fireActorOperation(op, state),
          "store fired after synchronization insertion failed");
  require(state.runtimeUnsupportedCapability && state.diagnostics.size() == 1,
          "synchronization insertion failure did not report unsupported");
  require(state.channels[&op.getMemMutable()].size() == 1 &&
              state.channels[&op.getAddrMutable()].size() == 1 &&
              state.channels[&op.getDataMutable()].size() == 1 &&
              state.channels[&op.getCtrlMutable()].size() == 1,
          "store consumed input on synchronization insertion failure");
  require(state.memoryActions.empty() &&
              state.firingMemoryOrderFrontier.empty() &&
              state.admittedPlainMemoryActions.contains(op.getOperation()),
          "store partially issued on synchronization insertion failure");
  expectUntouchedRun(
      state, *memory, {0x11, 0x22},
      "store changed run or memory state on synchronization insertion failure");
}

void disjointPlainMemoryHistoryHasBoundedQueryWork() {
  loom::sim::MemoryAtomicOrder order;
  loom::sim::MemorySynchronization sync(order);
  PlainMemoryConflictIndex history;
  constexpr std::uint64_t kRoot = 7;
  constexpr unsigned kIntervals = 16384;
  llvm::SmallVector<ReadyPlainMemoryAction> ready;
  ready.reserve(kIntervals + 1);

  for (unsigned index = 0; index < kIntervals; ++index) {
    const std::int64_t begin = static_cast<std::int64_t>(index) * 2;
    MemoryActionRecord read{kRoot, {{begin, begin + 1}}, false};
    history.retain(read, sync.declareEffect(), sync);
    ready.push_back(ReadyPlainMemoryAction{
        MemoryActionRecord{kRoot, {{begin, begin + 1}}, true}, {}});
  }
  require(history.intervalCount(kRoot) == kIntervals,
          "disjoint history did not retain one interval per access");

  MemoryActionRecord gap{kRoot, {{1, 2}}, true};
  PlainMemoryConflictQuery gapQuery = history.query(gap);
  require(gapQuery.effects.empty() && gapQuery.inspectedIntervals == 0,
          "a disjoint history query inspected unrelated intervals");

  const std::int64_t lastBegin = static_cast<std::int64_t>(kIntervals - 1) * 2;
  MemoryActionRecord overlap{kRoot, {{lastBegin, lastBegin + 1}}, true};
  PlainMemoryConflictQuery overlapQuery = history.query(overlap);
  require(overlapQuery.effects.size() == 1 &&
              overlapQuery.effects.front() ==
                  loom::sim::SyncEffectId(kIntervals - 1) &&
              overlapQuery.inspectedIntervals == 1,
          "a point conflict query inspected unrelated history");

  ReadyPlainMemoryConflictScan disjoint = scanReadyPlainMemoryConflicts(ready);
  require(!disjoint.hasConflict && disjoint.inspectedRanges == kIntervals,
          "ready disjoint actions were not scanned once per range");
  ready.push_back(ReadyPlainMemoryAction{overlap, {}});
  require(scanReadyPlainMemoryConflicts(ready).hasConflict,
          "ready overlapping writes were not rejected");
}

// Scale gate for the memory-order frontier representation.
//
// A wide dataflow.sync publishes one memory-order frontier to every result, so
// the run must compute that frontier once and hand each result a handle to it.
//
// The gate is the elapsed time, not the arena shape: an implementation that
// recomputes and reinterns the same frontier per output still deduplicates to
// one arena entry, so counting entries would not reject it. Merging a
// width-k frontier into each of k results is cubic, and at this width the old
// per-output merge measured 37.3 seconds against the budget below, while the
// shared-handle path costs milliseconds. The frontier assertions that follow
// only pin the published content; the budget is what rejects the old
// behaviour.
//
// The measured region covers fixture construction, seeding and firing, so the
// gate is externally observable rather than a post-hoc timer around one call.
// The lit runner additionally bounds the whole process.
void wideSyncSharesOnePublishedFrontier(mlir::MLIRContext &context) {
  constexpr unsigned kWidth = 8192;
  constexpr double kBudgetSeconds = 15.0;
  const auto start = std::chrono::steady_clock::now();

  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "func.func @wide_sync(";
  for (unsigned index = 0; index < kWidth; ++index)
    stream << (index ? ", " : "") << "%in" << index << ": none";
  stream << ") {\n  %joined:" << kWidth << " = dataflow.sync ";
  for (unsigned index = 0; index < kWidth; ++index)
    stream << (index ? ", " : "") << "%in" << index;
  stream << " : (";
  for (unsigned index = 0; index < kWidth; ++index)
    stream << (index ? ", " : "") << "none";
  stream << ") -> (";
  for (unsigned index = 0; index < kWidth; ++index)
    stream << (index ? ", " : "") << "none";
  stream << ")\n  return\n}\n";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(stream.str(), &context);
  require(static_cast<bool>(module), "unable to parse the wide sync fixture");
  dataflow::SyncOp sync;
  module->walk([&](dataflow::SyncOp op) { sync = op; });
  require(sync && sync->getNumOperands() == kWidth,
          "the wide sync fixture is missing its actor");

  // Each input carries a distinct effect, so the published frontier is the
  // whole width and no reduction can shrink it. The state owns both causality
  // engines, as a run does, so the bound reference outlives every use.
  SimulatorState state;
  state.memoryOrder = std::make_unique<loom::sim::MemoryAtomicOrder>();
  state.memorySync =
      std::make_unique<loom::sim::MemorySynchronization>(*state.memoryOrder);
  for (mlir::OpOperand &operand : sync->getOpOperands()) {
    Token token = noneToken();
    token.memoryOrder = state.memoryOrderFrontiers.internCanonical(
        state.memorySync->declareEffect());
    state.channels[&operand].push_back(std::move(token));
  }
  const std::size_t seededFrontiers =
      state.memoryOrderFrontiers.frontierCount();

  require(fireOnce(sync, state), "the wide sync did not fire");

  require(state.pendingObservedOutputs.size() == kWidth,
          "the wide sync did not publish one token per result");
  MemoryOrderFrontierId published;
  for (mlir::Value result : sync->getResults()) {
    const auto tokens = state.pendingObservedOutputs.find(result);
    require(tokens != state.pendingObservedOutputs.end() &&
                tokens->second.size() == 1,
            "a wide sync result published the wrong token count");
    const MemoryOrderFrontierId frontier = tokens->second.front().memoryOrder;
    require(!frontier.empty(), "a wide sync result published no memory order");
    if (published.empty())
      published = frontier;
    require(frontier == published,
            "wide sync results published separately owned frontiers");
  }
  require(state.memoryOrderFrontiers.elements(published).size() == kWidth,
          "the published frontier lost members of its firing");
  // One frontier per seeded input plus exactly one for the whole publication.
  require(state.memoryOrderFrontiers.frontierCount() == seededFrontiers + 1,
          "the wide sync retained more than one published frontier");

  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  require(elapsed < kBudgetSeconds, "the wide sync exceeded its scale budget");
}

} // namespace

int main() {
  tokenWidthNarrowsAtTokenBoundary();
  actorTransitionDescriptorContract();
  disjointPlainMemoryHistoryHasBoundedQueryWork();

  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(fixture, &context);
  require(static_cast<bool>(module), "unable to parse fixture");

  auto parallelizeFunc =
      module->lookupSymbol<mlir::func::FuncOp>("parallelize");
  auto wideParallelizeFunc =
      module->lookupSymbol<mlir::func::FuncOp>("wide_parallelize");
  auto serializeFunc = module->lookupSymbol<mlir::func::FuncOp>("serialize");
  auto rankTwoFunc = module->lookupSymbol<mlir::func::FuncOp>("rank_two");
  auto gateFunc = module->lookupSymbol<mlir::func::FuncOp>("gate");
  auto streamFunc = module->lookupSymbol<mlir::func::FuncOp>("stream");
  auto memoryFunc = module->lookupSymbol<mlir::func::FuncOp>("memory");
  auto llvmLoadFunc = module->lookupSymbol<mlir::func::FuncOp>("llvm_load");
  require(parallelizeFunc && wideParallelizeFunc && serializeFunc &&
              rankTwoFunc && gateFunc && streamFunc && memoryFunc &&
              llvmLoadFunc,
          "fixture functions are missing");

  dataflow::GateOp gate;
  gateFunc.walk([&](dataflow::GateOp op) { gate = op; });
  require(gate, "the gate fixture actor is missing");

  dataflow::StreamOp stream;
  streamFunc.walk([&](dataflow::StreamOp op) { stream = op; });
  require(stream, "the stream fixture actor is missing");

  mlir::LLVM::LoadOp llvmLoad;
  llvmLoadFunc.walk([&](mlir::LLVM::LoadOp op) { llvmLoad = op; });
  require(llvmLoad, "the llvm.load fixture actor is missing");

  dataflow::ParallelizeOp wideParallelize;
  wideParallelizeFunc.walk(
      [&](dataflow::ParallelizeOp op) { wideParallelize = op; });
  require(wideParallelize, "the wide parallelize fixture actor is missing");

  dataflow::ParallelizeOp parallelize;
  dataflow::PackOp pack;
  dataflow::SerializeOp serialize;
  dataflow::PackOp rankTwoPack;
  dataflow::UnpackOp rankTwoUnpack;
  llvm::SmallVector<dataflow::UnpackOp, 2> unpacks;
  parallelizeFunc.walk([&](dataflow::ParallelizeOp op) { parallelize = op; });
  parallelizeFunc.walk([&](dataflow::PackOp op) { pack = op; });
  serializeFunc.walk([&](dataflow::SerializeOp op) { serialize = op; });
  serializeFunc.walk([&](dataflow::UnpackOp op) { unpacks.push_back(op); });
  rankTwoFunc.walk([&](dataflow::PackOp op) { rankTwoPack = op; });
  rankTwoFunc.walk([&](dataflow::UnpackOp op) { rankTwoUnpack = op; });
  dataflow::LoadOp load;
  dataflow::StoreOp store;
  memoryFunc.walk([&](dataflow::LoadOp op) { load = op; });
  memoryFunc.walk([&](dataflow::StoreOp op) { store = op; });
  require(parallelize && pack && serialize && unpacks.size() == 2 &&
              rankTwoPack && rankTwoUnpack && load && store,
          "fixture actors are missing");

  dataflow::UnpackOp vectorUnpack = unpacks[0];
  dataflow::UnpackOp maskUnpack = unpacks[1];
  if (mlir::cast<mlir::VectorType>(vectorUnpack.getVector().getType())
          .getElementType()
          .isInteger(1))
    std::swap(vectorUnpack, maskUnpack);

  parallelizePreservesQueuedActivation(parallelize);
  parallelizeGroupPublishesItsAssembledOrderOnce(wideParallelize);
  droppedOrderIsNeverRetained(gate);
  llvmLoadKeepsItsMemoryResidentWitness(llvmLoad);
  streamRepublishesRetainedOrderWithoutRework(stream);
  serializePreservesQueuedActivation(serialize, vectorUnpack, maskUnpack);
  unpackPlacesRowMajorLanes(rankTwoUnpack);
  packFlattensRowMajorLanes(rankTwoPack);
  parallelizeFailureIsAtomic(parallelize);
  packFailureIsAtomic(pack);
  unpackFailureIsAtomic(vectorUnpack);
  serializeFailureIsAtomic(serialize);
  loadRejectionIsAtomic(load);
  storeDuplicateScatterIsAtomic(store);
  storeSynchronizationFailureIsAtomic(store);
  wideSyncSharesOnePublishedFrontier(context);
  return 0;
}

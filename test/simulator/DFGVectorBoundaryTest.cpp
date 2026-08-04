#include "DFGSimulatorInternal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

  func.func @memory(%mem: memref<?xi8>, %addr: index,
                    %addresses: vector<2xindex>, %data: vector<2xi8>,
                    %ctrl: none) {
    %loaded, %read = dataflow.load %mem[%addr] %ctrl : memref<?xi8>
    %written = dataflow.store %mem[%addresses] %data %ctrl
        : memref<?xi8>, vector<2xindex>, vector<2xi8>
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

void publishedWideFrontierForwardsByHandle() {
  MemoryOrderFrontierArena arena;
  const llvm::SmallVector<loom::sim::SyncEffectId, 2> effects = {
      loom::sim::SyncEffectId(1), loom::sim::SyncEffectId(2)};
  const MemoryOrderFrontierId frontier = arena.internCanonical(effects);

  MemoryOrderAccumulator accumulator;
  accumulator.absorb(frontier);

  require(accumulator.published() == frontier,
          "a stored wide frontier was expanded instead of forwarded");
}

void growingFrontierSharesPublishedPrefixes() {
  constexpr std::uint64_t kEffects = 2048;
  MemoryOrderFrontierArena arena;
  MemoryOrderFrontierId frontier;
  for (std::uint64_t ordinal = 0; ordinal < kEffects; ++ordinal) {
    const MemoryOrderFrontierId effect =
        arena.internCanonical(loom::sim::SyncEffectId(ordinal));
    frontier = arena.internUnion({frontier, effect});
  }

  llvm::SmallVector<loom::sim::SyncEffectId> effects;
  arena.appendCanonicalEffects(frontier, effects);
  require(effects.size() == kEffects && effects.front().value() == 0 &&
              effects.back().value() == kEffects - 1,
          "a persistent frontier union changed its effect set");
  require(arena.retainedEffectReferences() == kEffects,
          "a growing frontier copied previously published effects");
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
    require(
        d.requiredState == CarrySemanticState::Initial && !d.requiredPhase &&
            d.consumedInputs == consumedMask<CarryInput>({CarryInput::Init}) &&
            d.forwardedInput == CarryInput::Init &&
            d.nextState == CarrySemanticState::Running,
        "carry Init descriptor");
  }
  {
    auto d = sem::carryCaseDescriptor(CarryCase::Next);
    require(d.requiredState == CarrySemanticState::Running &&
                d.requiredPhase == true &&
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
                d.requiredPhase == false &&
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
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
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

struct TestSimulatorState : SimulatorState {
  TestSimulatorState() { execution = &prepared; }

  PreparedGraphExecution prepared;
};

TokenQueue &testChannelQueue(TestSimulatorState &state,
                             mlir::OpOperand &operand) {
  auto found = state.prepared.channelOrdinals.find(&operand);
  if (found != state.prepared.channelOrdinals.end())
    return state.channelSlots[found->second].ready;
  const ChannelOrdinal ordinal = state.channelSlots.size();
  state.prepared.channelOrdinals.try_emplace(&operand, ordinal);
  state.prepared.channels.push_back({&operand, InvalidActorOrdinal});
  state.channelSlots.push_back(
      ChannelSlot{&operand, InvalidActorOrdinal, {}, {}});
  return state.channelSlots.back().ready;
}

ActorExecutionPlan &installActorPlan(TestSimulatorState &state,
                                     mlir::Operation *operation) {
  for (ActorExecutionPlan &plan : state.prepared.actorPlans)
    if (plan.operation == operation)
      return plan;

  auto projection = dataflow::projectRegisteredActorSchemaProjection(operation);
  if (!projection)
    fail("actor admission failed: " + llvm::toString(projection.takeError()));
  if (auto unsupported = unsupportedActorProvider(operation, *projection))
    fail("actor provider admission failed: " + unsupported->label +
         (unsupported->reason.empty() ? "" : ": " + unsupported->reason));

  ChannelOrdinal firstInput = 0;
  if (operation->getNumOperands() != 0) {
    firstInput = state.channelSlots.size();
    for (mlir::OpOperand &operand : operation->getOpOperands()) {
      TokenQueue ready;
      TokenQueue pending;
      auto prior = state.prepared.channelOrdinals.find(&operand);
      if (prior != state.prepared.channelOrdinals.end()) {
        const ChannelOrdinal oldOrdinal = prior->second;
        ready = std::move(state.channelSlots[oldOrdinal].ready);
        pending = std::move(state.channelSlots[oldOrdinal].pending);
        for (ActorExecutionPlan &plan : state.prepared.actorPlans)
          for (ActorExecutionPlan::Output &output : plan.outputs)
            for (ChannelOrdinal &channel : output.channels)
              if (channel == oldOrdinal)
                channel = state.channelSlots.size();
        for (ChannelOrdinal &channel : state.pendingChannelOrdinals)
          if (channel == oldOrdinal)
            channel = state.channelSlots.size();
      }
      const ChannelOrdinal ordinal = state.channelSlots.size();
      state.prepared.channelOrdinals[&operand] = ordinal;
      state.channelSlots.push_back(ChannelSlot{
          &operand, InvalidActorOrdinal, std::move(ready), std::move(pending)});
    }
  }

  llvm::SmallVector<ActorExecutionPlan::Output, 2> outputs;
  for (mlir::Value result : operation->getResults()) {
    ActorExecutionPlan::Output output;
    output.value = result;
    output.observed = true;
    for (mlir::OpOperand &use : result.getUses()) {
      (void)testChannelQueue(state, use);
      output.channels.push_back(
          state.prepared.channelOrdinals.find(&use)->second);
    }
    outputs.push_back(std::move(output));
  }

  std::optional<loom::sim::PrimitiveOperationDescriptor> primitive;
  if (loom::sim::isSupportedPrimitiveOperation(projection->schema))
    primitive =
        takeExpected(primitiveDescriptorForActor(*projection, operation));
  std::optional<MemoryActorExecutionPlan> memory;
  if (mlir::isa<dataflow::LoadOp, dataflow::StoreOp>(operation))
    memory =
        takeExpected(memoryActorExecutionPlan(operation, state.graphScope));
  auto runtimeProvider = actorRuntimeProvider(projection->schema);
  if (!runtimeProvider)
    fail("actor runtime provider is unavailable");
  auto handshakeCases =
      takeExpected(dataflow::semantics::projectActorHandshakeCases(
          projection->schema, operation->getNumOperands(),
          operation->getNumResults()));
  state.prepared.actorPlans.push_back(ActorExecutionPlan{
      operation, std::move(*projection), runtimeProvider->commit, firstInput,
      static_cast<std::uint32_t>(operation->getNumOperands()),
      std::move(outputs), std::move(primitive), std::move(memory), std::nullopt,
      std::move(handshakeCases), runtimeProvider->probe});
  return state.prepared.actorPlans.back();
}

template <typename Op>
bool fireAdmittedActorOperation(Op op, TestSimulatorState &state) {
  mlir::Operation *operation = op.getOperation();
  for (mlir::Value result : operation->getResults())
    state.prepared.observedValues.insert(result);
  ActorExecutionPlan &plan = installActorPlan(state, operation);
  const ActorExecutionPlan *prior = state.currentActorPlan;
  state.currentActorPlan = &plan;
  bool fired = loom::sim::detail::fireActorOperation(plan, state);
  state.currentActorPlan = prior;
  return fired;
}

Token tokenWithBits(mlir::Type type, uint64_t value) {
  unsigned width = takeExpected(tokenTypeBitWidth(type));
  return takeExpected(tokenFromBitPattern(llvm::APInt(width, value), type));
}

Token malformedToken(TokenKind kind, unsigned width) {
  Token token;
  token.kind = kind;
  token.setExactBitPattern(llvm::APInt(width, 0));
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

void flushPending(SimulatorState &state) { flushPendingTokens(state); }

void parallelizePreservesQueuedActivation(dataflow::ParallelizeOp op) {
  TestSimulatorState state;
  testChannelQueue(state, op.getDataMutable())
      .push_back(tokenWithBits(op.getData().getType(), 17));
  testChannelQueue(state, op.getDataMutable())
      .push_back(tokenWithBits(op.getData().getType(), 18));
  testChannelQueue(state, op.getScalarPhaseMutable())
      .push_back(boolValueToken(true));
  testChannelQueue(state, op.getScalarPhaseMutable())
      .push_back(boolValueToken(false));
  testChannelQueue(state, op.getScalarPhaseMutable())
      .push_back(boolValueToken(true));
  testChannelQueue(state, op.getScalarPhaseMutable())
      .push_back(boolValueToken(false));

  require(fireAdmittedActorOperation(op, state),
          "first scalar true did not fire");
  require(testChannelQueue(state, op.getDataMutable()).size() == 1,
          "first scalar true consumed the wrong payload");
  require(fireAdmittedActorOperation(op, state),
          "first scalar false did not fire");
  require(testChannelQueue(state, op.getDataMutable()).size() == 1,
          "scalar false consumed the next activation payload");
  require(fireAdmittedActorOperation(op, state),
          "second scalar true did not fire");
  require(fireAdmittedActorOperation(op, state),
          "second scalar false did not fire");

  expectBits(state.pendingObservedOutputs[op.getVector()],
             op.getVector().getType(), {17, 18},
             "parallelize did not reset zero-filled lanes");
  expectBits(state.pendingObservedOutputs[op.getMask()], op.getMask().getType(),
             {1, 1}, "parallelize did not reset active masks");
  expectPhases(state.pendingObservedOutputs[op.getGroupPhase()],
               {true, false, true, false},
               "parallelize emitted the wrong activation phases");
}

void serializePreservesQueuedActivation(dataflow::SerializeOp op,
                                        dataflow::UnpackOp vectorUnpack,
                                        dataflow::UnpackOp maskUnpack) {
  TestSimulatorState state;
  testChannelQueue(state, vectorUnpack.getPackedMutable())
      .push_back(
          tokenWithBits(vectorUnpack.getPacked().getType(), 0x44332211U));
  testChannelQueue(state, vectorUnpack.getPackedMutable())
      .push_back(
          tokenWithBits(vectorUnpack.getPacked().getType(), 0x44332211U));
  testChannelQueue(state, maskUnpack.getPackedMutable())
      .push_back(tokenWithBits(maskUnpack.getPacked().getType(), 0));
  testChannelQueue(state, maskUnpack.getPackedMutable())
      .push_back(tokenWithBits(maskUnpack.getPacked().getType(), 5));

  require(fireAdmittedActorOperation(vectorUnpack, state),
          "first vector unpack did not fire");
  require(fireAdmittedActorOperation(vectorUnpack, state),
          "second vector unpack did not fire");
  require(fireAdmittedActorOperation(maskUnpack, state),
          "first mask unpack did not fire");
  require(fireAdmittedActorOperation(maskUnpack, state),
          "second mask unpack did not fire");
  flushPending(state);

  require(testChannelQueue(state, op.getVectorMutable()).size() == 2 &&
              testChannelQueue(state, op.getMaskMutable()).size() == 2,
          "unpack did not queue both activation payloads");
  testChannelQueue(state, op.getGroupPhaseMutable())
      .push_back(boolValueToken(true));
  testChannelQueue(state, op.getGroupPhaseMutable())
      .push_back(boolValueToken(false));
  testChannelQueue(state, op.getGroupPhaseMutable())
      .push_back(boolValueToken(true));
  testChannelQueue(state, op.getGroupPhaseMutable())
      .push_back(boolValueToken(false));

  require(fireAdmittedActorOperation(op, state),
          "all-zero true group did not fire");
  require(testChannelQueue(state, op.getVectorMutable()).size() == 1 &&
              testChannelQueue(state, op.getMaskMutable()).size() == 1,
          "all-zero true group did not consume its payload");
  require(state.pendingObservedOutputs[op.getData()].empty(),
          "all-zero group emitted scalar data");
  require(fireAdmittedActorOperation(op, state),
          "first group false did not fire");
  require(testChannelQueue(state, op.getVectorMutable()).size() == 1 &&
              testChannelQueue(state, op.getMaskMutable()).size() == 1,
          "group false consumed the next activation payload");
  require(fireAdmittedActorOperation(op, state),
          "sparse true group did not fire");
  require(fireAdmittedActorOperation(op, state),
          "second group false did not fire");

  expectBits(state.pendingObservedOutputs[op.getData()], op.getData().getType(),
             {0x11, 0x33}, "serialize did not preserve low-slice lane order");
  expectPhases(state.pendingObservedOutputs[op.getScalarPhase()],
               {false, true, true, false},
               "serialize emitted the wrong activation phases");
}

void parallelizeFailureIsAtomic(dataflow::ParallelizeOp op) {
  {
    TestSimulatorState state;
    testChannelQueue(state, op.getDataMutable())
        .push_back(malformedToken(TokenKind::Integer, 16));
    testChannelQueue(state, op.getScalarPhaseMutable())
        .push_back(boolValueToken(true));

    require(!fireAdmittedActorOperation(op, state),
            "parallelize accepted a malformed scalar token");
    require(testChannelQueue(state, op.getDataMutable()).size() == 1 &&
                testChannelQueue(state, op.getScalarPhaseMutable()).size() == 1,
            "parallelize consumed input on conversion failure");
    require(!state.parallelizeStates.contains(op.getOperation()),
            "parallelize changed actor state on conversion failure");
    require(state.pendingChannelOrdinals.empty() &&
                state.pendingObservedOutputs.empty() &&
                state.actorMutationEpoch == 0,
            "parallelize published output on conversion failure");
  }

  {
    TestSimulatorState state;
    ParallelizeState pending;
    pending.semanticState.pendingItems = 1;
    pending.slots.resize(2);
    pending.slots[0] = malformedToken(TokenKind::Integer, 16);
    state.parallelizeStates[op.getOperation()] = pending;
    testChannelQueue(state, op.getScalarPhaseMutable())
        .push_back(boolValueToken(false));

    require(!fireAdmittedActorOperation(op, state),
            "parallelize assembled a malformed pending group");
    const ParallelizeState &preserved =
        state.parallelizeStates.find(op.getOperation())->second;
    require(preserved.semanticState.pendingItems == 1 && preserved.slots[0] &&
                preserved.slots[0]->exactBitWidth() == 16,
            "parallelize changed pending state on group construction failure");
    require(testChannelQueue(state, op.getScalarPhaseMutable()).size() == 1,
            "parallelize consumed phase on group construction failure");
    require(state.pendingChannelOrdinals.empty() &&
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
  TestSimulatorState state;
  testChannelQueue(state, op.getPackedMutable())
      .push_back(tokenWithBits(op.getPacked().getType(), kRankTwoPacked));

  require(fireAdmittedActorOperation(op, state),
          "rank-two unpack did not fire");
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
  TestSimulatorState state;
  testChannelQueue(state, op.getVectorMutable())
      .push_back(takeExpected(
          tokenFromBitPattern(rowMajorLaneBits(), op.getVector().getType())));

  require(fireAdmittedActorOperation(op, state), "rank-two pack did not fire");
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
  TestSimulatorState state;
  testChannelQueue(state, op.getVectorMutable())
      .push_back(malformedToken(TokenKind::Vector, 8));

  require(!fireAdmittedActorOperation(op, state),
          "pack accepted a malformed vector");
  require(testChannelQueue(state, op.getVectorMutable()).size() == 1,
          "pack consumed input on conversion failure");
  require(state.pendingChannelOrdinals.empty() &&
              state.pendingObservedOutputs.empty() &&
              state.actorMutationEpoch == 0,
          "pack published output on conversion failure");
}

void unpackFailureIsAtomic(dataflow::UnpackOp op) {
  TestSimulatorState state;
  testChannelQueue(state, op.getPackedMutable())
      .push_back(malformedToken(TokenKind::Integer, 8));

  require(!fireAdmittedActorOperation(op, state),
          "unpack accepted a malformed packed token");
  require(testChannelQueue(state, op.getPackedMutable()).size() == 1,
          "unpack consumed input on conversion failure");
  require(state.pendingChannelOrdinals.empty() &&
              state.pendingObservedOutputs.empty() &&
              state.actorMutationEpoch == 0,
          "unpack published output on conversion failure");
}

void serializeFailureIsAtomic(dataflow::SerializeOp op) {
  TestSimulatorState state;
  testChannelQueue(state, op.getVectorMutable())
      .push_back(malformedToken(TokenKind::Vector, 8));
  testChannelQueue(state, op.getMaskMutable())
      .push_back(tokenWithBits(op.getMask().getType(), 1));
  testChannelQueue(state, op.getGroupPhaseMutable())
      .push_back(boolValueToken(true));

  require(!fireAdmittedActorOperation(op, state),
          "serialize accepted a malformed vector");
  require(testChannelQueue(state, op.getVectorMutable()).size() == 1 &&
              testChannelQueue(state, op.getMaskMutable()).size() == 1 &&
              testChannelQueue(state, op.getGroupPhaseMutable()).size() == 1,
          "serialize consumed input on conversion failure");
  require(state.pendingChannelOrdinals.empty() &&
              state.pendingObservedOutputs.empty() &&
              state.actorMutationEpoch == 0,
          "serialize published output on conversion failure");
}

// A memory actor rejects an access entirely on peeked inputs. Only its reason
// and the run's retained failure may change; inputs, outputs, actor mutation
// state, events, fire counts, and memory may not.
std::shared_ptr<MemoryValue> makeMemory(mlir::Type elementType,
                                        std::initializer_list<uint64_t> values,
                                        mlir::Operation *scope) {
  auto memory = std::make_shared<MemoryValue>();
  for (uint64_t value : values) {
    auto bytes = takeExpected(encodeMemoryElement(
        tokenWithBits(elementType, value), elementType, scope));
    memory->bytes.append(bytes.begin(), bytes.end());
  }
  memory->initialized = llvm::SmallBitVector(memory->bytes.size(), true);
  return memory;
}

unsigned resolvedIndexBits(mlir::Operation *scope) {
  return takeExpected(loom::getIndexBitWidth(scope));
}

void installMemoryActorPlan(TestSimulatorState &state, mlir::Operation *op) {
  state.graphScope = op;
  state.currentActorPlan = &installActorPlan(state, op);
}

Token indexVectorToken(unsigned indexBits,
                       std::initializer_list<uint64_t> lanes) {
  llvm::APInt bits(indexBits * lanes.size(), 0);
  unsigned lane = 0;
  for (uint64_t value : lanes)
    bits.insertBits(llvm::APInt(indexBits, value), indexBits * lane++);
  Token token;
  token.kind = TokenKind::Vector;
  token.setExactBitPattern(bits);
  return token;
}

void expectUntouchedRun(SimulatorState &state, const MemoryValue &memory,
                        mlir::Type elementType, mlir::Operation *scope,
                        llvm::ArrayRef<uint64_t> elements,
                        llvm::StringRef message) {
  require(state.pendingChannelOrdinals.empty() &&
              state.pendingObservedOutputs.empty(),
          message);
  require(state.actorMutationEpoch == 0 && state.eventCount == 0 &&
              llvm::all_of(state.operationFireCounts,
                           [](std::uint64_t count) { return count == 0; }),
          message);
  require(state.terminalPrimitiveOps.empty(), message);
  llvm::SmallVector<loom::sim::SemanticMemoryByte> expected;
  for (uint64_t value : elements) {
    auto bytes = takeExpected(encodeMemoryElement(
        tokenWithBits(elementType, value), elementType, scope));
    expected.append(bytes.begin(), bytes.end());
  }
  require(memory.bytes.size() == expected.size(), message);
  for (auto [actual, wanted] : llvm::zip_equal(memory.bytes, expected))
    require(actual.state == wanted.state && actual.value == wanted.value,
            message);
}

void loadRejectionIsAtomic(dataflow::LoadOp op) {
  TestSimulatorState state;
  installMemoryActorPlan(state, op.getOperation());
  auto memoryType = mlir::cast<mlir::MemRefType>(op.getMem().getType());
  auto memory =
      makeMemory(memoryType.getElementType(), {0x11, 0x22}, op.getOperation());
  testChannelQueue(state, op.getMemMutable())
      .push_back(memoryCapabilityToken(op.getMem(), memory, 0));
  testChannelQueue(state, op.getAddrMutable())
      .push_back(
          indexToken(llvm::APInt(resolvedIndexBits(op.getOperation()), 99)));
  testChannelQueue(state, op.getCtrlMutable()).push_back(noneToken());

  PlainMemoryActionProjection first =
      projectReadyPlainMemoryAction(op.getOperation(), state);
  require(!first.ready && first.diagnostics.size() == 1,
          "load admission accepted an out-of-range address");
  PlainMemoryActionProjection second =
      projectReadyPlainMemoryAction(op.getOperation(), state);
  require(!second.ready && second.diagnostics.size() == 1,
          "a re-polled load rejection is not detectable as a failed attempt");
  require(testChannelQueue(state, op.getMemMutable()).size() == 1 &&
              testChannelQueue(state, op.getAddrMutable()).size() == 1 &&
              testChannelQueue(state, op.getCtrlMutable()).size() == 1,
          "load consumed input on a rejected access");
  expectUntouchedRun(state, *memory, memoryType.getElementType(),
                     op.getOperation(), {0x11, 0x22},
                     "load changed run or memory state on a rejected access");
}

void storeSynchronizationFailureIsAtomic(dataflow::StoreOp op) {
  TestSimulatorState state;
  installMemoryActorPlan(state, op.getOperation());
  auto memoryType = mlir::cast<mlir::MemRefType>(op.getMem().getType());
  auto memory =
      makeMemory(memoryType.getElementType(), {0x11, 0x22}, op.getOperation());
  testChannelQueue(state, op.getMemMutable())
      .push_back(memoryCapabilityToken(op.getMem(), memory, 0));
  testChannelQueue(state, op.getAddrMutable())
      .push_back(
          indexVectorToken(resolvedIndexBits(op.getOperation()), {0, 1}));
  testChannelQueue(state, op.getDataMutable())
      .push_back(tokenWithBits(op.getData().getType(), 0xAB43));
  Token ctrl = noneToken();
  ctrl.memoryOrder =
      state.memoryOrderFrontiers.internCanonical(loom::sim::SyncEffectId(99));
  testChannelQueue(state, op.getCtrlMutable()).push_back(std::move(ctrl));
  PlainMemoryActionProjection projected =
      projectReadyPlainMemoryAction(op.getOperation(), state);
  require(projected.ready && projected.diagnostics.empty(),
          "unable to project synchronization-failure store");
  state.admittedPlainMemoryActions.try_emplace(op.getOperation(),
                                               std::move(*projected.ready));

  require(!fireAdmittedActorOperation(op, state),
          "store fired after synchronization insertion failed");
  require(state.diagnostics.size() == 1,
          "store recorded no synchronization failure reason");
  require(state.failure == RunFailure::ProviderInvariant,
          "a synchronization provider failure was not classified as an "
          "execution failure");
  require(testChannelQueue(state, op.getMemMutable()).size() == 1 &&
              testChannelQueue(state, op.getAddrMutable()).size() == 1 &&
              testChannelQueue(state, op.getDataMutable()).size() == 1 &&
              testChannelQueue(state, op.getCtrlMutable()).size() == 1,
          "store consumed input on synchronization insertion failure");
  require(state.memoryActions.empty() &&
              state.firingMemoryOrderFrontier.empty() &&
              state.admittedPlainMemoryActions.contains(op.getOperation()),
          "store partially issued on synchronization insertion failure");
  expectUntouchedRun(
      state, *memory, memoryType.getElementType(), op.getOperation(),
      {0x11, 0x22},
      "store changed run or memory state on synchronization insertion failure");
}

// A finalized plain scatter carries the distinctness its program already
// proved (docs/spec-dataflow-vectorization.md), so admission neither inspects
// nor guesses that legality. Resolved duplicates therefore break an invariant
// the provider guarantees rather than exposing a capability the model lacks,
// and the refused firing consumes, retains, and publishes nothing.
void storeDuplicateScatterIsProviderFailure(dataflow::StoreOp op) {
  TestSimulatorState state;
  installMemoryActorPlan(state, op.getOperation());
  auto memoryType = mlir::cast<mlir::MemRefType>(op.getMem().getType());
  auto memory =
      makeMemory(memoryType.getElementType(), {0x11, 0x22}, op.getOperation());
  testChannelQueue(state, op.getMemMutable())
      .push_back(memoryCapabilityToken(op.getMem(), memory, 0));
  testChannelQueue(state, op.getAddrMutable())
      .push_back(
          indexVectorToken(resolvedIndexBits(op.getOperation()), {1, 1}));
  testChannelQueue(state, op.getDataMutable())
      .push_back(tokenWithBits(op.getData().getType(), 0xAB43));
  testChannelQueue(state, op.getCtrlMutable()).push_back(noneToken());
  PlainMemoryActionProjection projected =
      projectReadyPlainMemoryAction(op.getOperation(), state);
  require(projected.ready && projected.diagnostics.empty(),
          "admission rejected a duplicate plain scatter it does not own");

  state.admittedPlainMemoryActions.try_emplace(op.getOperation(),
                                               std::move(*projected.ready));
  require(!fireAdmittedActorOperation(op, state),
          "store resolved a duplicate active plain scatter destination");
  require(state.diagnostics.size() == 1,
          "store recorded no duplicate-destination reason");
  require(state.failure == RunFailure::ProviderInvariant,
          "a duplicate plain scatter was not classified as an execution "
          "failure");
  require(testChannelQueue(state, op.getMemMutable()).size() == 1 &&
              testChannelQueue(state, op.getAddrMutable()).size() == 1 &&
              testChannelQueue(state, op.getDataMutable()).size() == 1 &&
              testChannelQueue(state, op.getCtrlMutable()).size() == 1,
          "store consumed input on a duplicate active destination");
  require(state.memoryActions.empty() &&
              state.firingMemoryOrderFrontier.empty() &&
              state.admittedPlainMemoryActions.contains(op.getOperation()),
          "store partially issued a duplicate active destination");
  expectUntouchedRun(
      state, *memory, memoryType.getElementType(), op.getOperation(),
      {0x11, 0x22},
      "store changed run or memory state on a duplicate active destination");
}

// The duplicate-scatter and synchronization-failure stores above set the
// ProviderInvariant terminal through the firing path; this pins the single
// projection from a retained failure to the run report. A run that retained no
// failure keeps the driver's own status.
void runFailureProjectsOnceToExecutionFailed() {
  {
    TestSimulatorState state;
    state.failure = RunFailure::ProviderInvariant;
    loom::sim::DFGSimulationReport report;
    report.status = "pass";
    require(applyRunFailureTerminal(state, report),
            "a retained provider invariant did not take the run terminal");
    require(report.status == "execution_failed",
            "a provider invariant did not project to execution_failed");
  }
  {
    TestSimulatorState state;
    loom::sim::DFGSimulationReport report;
    report.status = "pass";
    require(!applyRunFailureTerminal(state, report),
            "a run with no retained failure took a terminal");
    require(report.status == "pass",
            "a run with no retained failure had its status rewritten");
  }
}

} // namespace

int main() {
  tokenWidthNarrowsAtTokenBoundary();
  actorTransitionDescriptorContract();
  publishedWideFrontierForwardsByHandle();
  growingFrontierSharesPublishedPrefixes();

  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(fixture, &context);
  require(static_cast<bool>(module), "unable to parse fixture");

  auto parallelizeFunc =
      module->lookupSymbol<mlir::func::FuncOp>("parallelize");
  auto serializeFunc = module->lookupSymbol<mlir::func::FuncOp>("serialize");
  auto rankTwoFunc = module->lookupSymbol<mlir::func::FuncOp>("rank_two");
  auto memoryFunc = module->lookupSymbol<mlir::func::FuncOp>("memory");
  require(parallelizeFunc && serializeFunc && rankTwoFunc && memoryFunc,
          "fixture functions are missing");

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
  serializePreservesQueuedActivation(serialize, vectorUnpack, maskUnpack);
  unpackPlacesRowMajorLanes(rankTwoUnpack);
  packFlattensRowMajorLanes(rankTwoPack);
  parallelizeFailureIsAtomic(parallelize);
  packFailureIsAtomic(pack);
  unpackFailureIsAtomic(vectorUnpack);
  serializeFailureIsAtomic(serialize);
  loadRejectionIsAtomic(load);
  storeSynchronizationFailureIsAtomic(store);
  storeDuplicateScatterIsProviderFailure(store);
  runFailureProjectsOnceToExecutionFailed();
  return 0;
}

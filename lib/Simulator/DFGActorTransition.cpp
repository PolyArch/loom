#include "DFGSimulatorInternal.h"

#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <system_error>
#include <utility>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {
namespace {

using namespace dataflow::semantics;

const ChannelSlot &inputSlot(const ActorExecutionPlan &plan,
                             const SimulatorState &state,
                             unsigned operandOrdinal) {
  assert(operandOrdinal < plan.inputChannelCount &&
         "probe input is outside the actor plan");
  return state.channelSlots[plan.firstInputChannel + operandOrdinal];
}

bool hasInput(const ActorExecutionPlan &plan, const SimulatorState &state,
              unsigned operandOrdinal) {
  return !inputSlot(plan, state, operandOrdinal).ready.empty();
}

const Token &peekInput(const ActorExecutionPlan &plan,
                       const SimulatorState &state, unsigned operandOrdinal) {
  const TokenQueue &queue = inputSlot(plan, state, operandOrdinal).ready;
  assert(!queue.empty() && "probe peek requires a ready input");
  return queue.front();
}

std::optional<bool> peekBool(const ActorExecutionPlan &plan,
                             const SimulatorState &state,
                             unsigned operandOrdinal) {
  if (!hasInput(plan, state, operandOrdinal))
    return std::nullopt;
  return boolToken(peekInput(plan, state, operandOrdinal));
}

ActorTransitionShape allPorts(const ActorExecutionPlan &plan) {
  ActorTransitionShape shape;
  shape.consumedInputs.reserve(plan.inputChannelCount);
  for (std::uint32_t ordinal = 0; ordinal < plan.inputChannelCount; ++ordinal)
    shape.consumedInputs.push_back(ordinal);
  shape.activeResults.reserve(plan.outputs.size());
  for (std::uint32_t ordinal = 0; ordinal < plan.outputs.size(); ++ordinal)
    shape.activeResults.push_back(ordinal);
  return shape;
}

llvm::Expected<std::optional<ActorTransitionShape>>
ready(ActorTransitionShape shape) {
  return std::optional<ActorTransitionShape>(std::move(shape));
}

llvm::Expected<std::optional<ActorTransitionShape>> blocked() {
  return std::optional<ActorTransitionShape>();
}

llvm::Expected<std::optional<ActorTransitionShape>>
probeAllInputs(const ActorExecutionPlan &plan, const SimulatorState &state) {
  for (unsigned ordinal = 0; ordinal < plan.inputChannelCount; ++ordinal)
    if (!hasInput(plan, state, ordinal))
      return blocked();
  return ready(allPorts(plan));
}

ActorTransitionShape
shapeFromDecision(const SemanticFiringDecision &decision,
                  llvm::ArrayRef<std::uint32_t> activeResults) {
  ActorTransitionShape shape;
  for (std::uint32_t ordinal = 0; ordinal < 8; ++ordinal)
    if ((decision.consumedInputs & (SemanticInputMask{1} << ordinal)) != 0)
      shape.consumedInputs.push_back(ordinal);
  shape.activeResults.assign(activeResults.begin(), activeResults.end());
  return shape;
}

unsigned streamIntegerBitWidth(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  return 0;
}

llvm::Expected<std::optional<ActorTransitionShape>>
probeStream(const ActorExecutionPlan &plan, const SimulatorState &state) {
  const auto *payload =
      std::get_if<dataflow::StreamRecurrencePayload>(&plan.projection.payload);
  if (!payload)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "stream probe received an incompatible semantic payload");
  if (state.failedStreamOps.contains(plan.operation))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "stream actor is in a failed state");

  StreamSemanticState stream;
  auto current = state.streamStates.find(plan.operation);
  if (current != state.streamStates.end())
    stream = current->second;
  std::optional<StreamActivation> activation;
  if (stream.mode == StreamMode::Idle && hasInput(plan, state, 0) &&
      hasInput(plan, state, 1) && hasInput(plan, state, 2)) {
    auto init = tokenBitPattern(peekInput(plan, state, 0),
                                plan.operation->getOperand(0).getType());
    if (!init)
      return init.takeError();
    auto limit = tokenBitPattern(peekInput(plan, state, 1),
                                 plan.operation->getOperand(1).getType());
    if (!limit)
      return limit.takeError();
    auto step = tokenBitPattern(peekInput(plan, state, 2),
                                plan.operation->getOperand(2).getType());
    if (!step)
      return step.takeError();
    activation =
        StreamActivation{std::move(*init), std::move(*limit), std::move(*step)};
  }
  auto transition = evaluateStreamTransition(
      stream,
      StreamSemanticConfig{
          payload->stepKind, payload->predicate,
          streamIntegerBitWidth(plan.operation->getOperand(0).getType())},
      activation);
  if (!transition)
    return transition.takeError();
  if (!transition->firing.ready)
    return blocked();
  llvm::SmallVector<std::uint32_t, 2> active;
  if (transition->emitIv)
    active.push_back(0);
  if (transition->emitPhase)
    active.push_back(1);
  return ready(shapeFromDecision(transition->firing, active));
}

llvm::Expected<std::optional<ActorTransitionShape>>
probeCarry(const ActorExecutionPlan &plan, const SimulatorState &state) {
  CarrySemanticState semanticState = CarrySemanticState::Initial;
  auto current = state.carryStates.find(plan.operation);
  if (current != state.carryStates.end())
    semanticState = current->second.semanticState;
  const CarryTransition transition = evaluateCarryTransition(
      semanticState, peekBool(plan, state, 0), hasInput(plan, state, 1),
      hasInput(plan, state, 2));
  if (!transition.firing.ready)
    return blocked();
  llvm::SmallVector<std::uint32_t, 1> active;
  if (transition.forwardedInput)
    active.push_back(0);
  return ready(shapeFromDecision(transition.firing, active));
}

llvm::Expected<std::optional<ActorTransitionShape>>
probeInvariant(const ActorExecutionPlan &plan, const SimulatorState &state) {
  InvariantSemanticState semanticState = InvariantSemanticState::Initial;
  auto current = state.invariantStates.find(plan.operation);
  if (current != state.invariantStates.end())
    semanticState = current->second.semanticState;
  const InvariantTransition transition = evaluateInvariantTransition(
      semanticState, peekBool(plan, state, 0), hasInput(plan, state, 1));
  if (!transition.firing.ready)
    return blocked();
  llvm::SmallVector<std::uint32_t, 1> active;
  if (transition.output != InvariantOutputSource::None)
    active.push_back(0);
  return ready(shapeFromDecision(transition.firing, active));
}

llvm::Expected<std::optional<ActorTransitionShape>>
probeGate(const ActorExecutionPlan &plan, const SimulatorState &state) {
  const GateSemanticState gate =
      state.gateContinueStates.contains(plan.operation)
          ? GateSemanticState::Open
          : GateSemanticState::Closed;
  const GateTransition transition = evaluateGateTransition(
      gate, peekBool(plan, state, 0), hasInput(plan, state, 1));
  if (!transition.firing.ready)
    return blocked();
  llvm::SmallVector<std::uint32_t, 2> active;
  if (transition.emitPhase)
    active.push_back(0);
  if (transition.forwardedInput)
    active.push_back(1);
  return ready(shapeFromDecision(transition.firing, active));
}

std::int64_t selectorLane(mlir::Type selectorType, const Token &selector) {
  return mlir::isa<mlir::IntegerType>(selectorType) ? boolToken(selector)
                                                    : integerToken(selector);
}

llvm::Expected<std::optional<ActorTransitionShape>>
probeMux(const ActorExecutionPlan &plan, const SimulatorState &state) {
  if (!hasInput(plan, state, 0))
    return blocked();
  auto op = mlir::cast<dataflow::MuxOp>(plan.operation);
  const std::int64_t lane =
      selectorLane(op.getSel().getType(), peekInput(plan, state, 0));
  if (lane < 0 || static_cast<std::size_t>(lane) >= op.getInputs().size())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "dataflow.mux selector is out of range");
  const std::uint32_t selected = static_cast<std::uint32_t>(lane) + 1;
  if (!hasInput(plan, state, selected))
    return blocked();
  return ready(ActorTransitionShape{{0, selected}, {0}});
}

llvm::Expected<std::optional<ActorTransitionShape>>
probeDemux(const ActorExecutionPlan &plan, const SimulatorState &state) {
  if (!hasInput(plan, state, 0) || !hasInput(plan, state, 1))
    return blocked();
  auto op = mlir::cast<dataflow::DemuxOp>(plan.operation);
  const std::int64_t lane =
      selectorLane(op.getSel().getType(), peekInput(plan, state, 0));
  if (lane < 0 || static_cast<std::size_t>(lane) >= op.getOutputs().size())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "dataflow.demux selector is out of range");
  return ready(
      ActorTransitionShape{{0, 1}, {static_cast<std::uint32_t>(lane)}});
}

llvm::Expected<std::optional<ActorTransitionShape>>
probeParallelize(const ActorExecutionPlan &plan, const SimulatorState &state) {
  auto op = mlir::cast<dataflow::ParallelizeOp>(plan.operation);
  const std::uint64_t vectorLength =
      op.getVector().getType().getShape().front();
  ParallelizeSemanticState semanticState;
  auto current = state.parallelizeStates.find(plan.operation);
  if (current != state.parallelizeStates.end()) {
    semanticState = current->second.semanticState;
    if (current->second.slots.size() != vectorLength &&
        semanticState.pendingItems != 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "dataflow.parallelize state does not match its vector length");
  }
  const ParallelizeTransition transition = evaluateParallelizeTransition(
      semanticState, vectorLength, peekBool(plan, state, 1),
      hasInput(plan, state, 0));
  if (!transition.firing.ready)
    return blocked();
  llvm::SmallVector<std::uint32_t, 3> active;
  if (transition.emitGroup) {
    active.push_back(0);
    active.push_back(1);
  }
  if (transition.emitTruePhase || transition.emitFalsePhase)
    active.push_back(2);
  return ready(shapeFromDecision(transition.firing, active));
}

llvm::Expected<std::optional<ActorTransitionShape>>
probeSerialize(const ActorExecutionPlan &plan, const SimulatorState &state) {
  const SerializeTransition transition = evaluateSerializeTransition(
      peekBool(plan, state, 2), hasInput(plan, state, 0),
      hasInput(plan, state, 1));
  if (!transition.firing.ready)
    return blocked();
  llvm::SmallVector<std::uint32_t, 2> active;
  if (transition.emitActiveItems) {
    active.push_back(0);
    active.push_back(1);
  } else if (transition.emitFalsePhase) {
    active.push_back(1);
  }
  return ready(shapeFromDecision(transition.firing, active));
}

llvm::Expected<std::optional<ActorTransitionShape>>
selectDynamicShape(const ActorExecutionPlan &plan,
                   const SimulatorState &state) {
  using Kind = ActorTransitionProbeKind;
  switch (plan.transitionProbe) {
  case Kind::Unavailable:
    return llvm::createStringError(
        std::errc::not_supported,
        "actor transition probe is unavailable for this provider");
  case Kind::AllInputs:
    return probeAllInputs(plan, state);
  case Kind::OneShot:
    if (state.oneShotOps.contains(plan.operation))
      return blocked();
    return probeAllInputs(plan, state);
  case Kind::Primitive:
    if (state.terminalPrimitiveOps.contains(plan.operation) ||
        (plan.inputChannelCount == 0 &&
         state.oneShotOps.contains(plan.operation)))
      return blocked();
    return probeAllInputs(plan, state);
  case Kind::GetElementPtr:
    if (state.terminalPrimitiveOps.contains(plan.operation))
      return blocked();
    return probeAllInputs(plan, state);
  case Kind::Stream:
    return probeStream(plan, state);
  case Kind::Carry:
    return probeCarry(plan, state);
  case Kind::Invariant:
    return probeInvariant(plan, state);
  case Kind::Gate:
    return probeGate(plan, state);
  case Kind::Mux:
    return probeMux(plan, state);
  case Kind::Demux:
    return probeDemux(plan, state);
  case Kind::Parallelize:
    return probeParallelize(plan, state);
  case Kind::Serialize:
    return probeSerialize(plan, state);
  }
  llvm_unreachable("closed actor transition probe kind");
}

} // namespace

llvm::Expected<std::optional<std::uint32_t>>
probeActorTransition(const ActorExecutionPlan &plan,
                     const SimulatorState &state) {
  auto shape = selectDynamicShape(plan, state);
  if (!shape)
    return shape.takeError();
  if (!*shape)
    return std::nullopt;

  std::optional<std::uint32_t> selected;
  for (const dataflow::semantics::ActorHandshakeCase &candidate :
       plan.handshakeCases) {
    if (candidate.consumedInputs != (*shape)->consumedInputs ||
        candidate.activeResults != (*shape)->activeResults)
      continue;
    if (selected)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "dynamic actor transition matches multiple schema cases");
    selected = candidate.ordinal;
  }
  if (!selected)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dynamic actor transition is absent from its schema cases");
  return selected;
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

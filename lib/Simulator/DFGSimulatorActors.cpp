#include "DFGSimulatorInternal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <limits>
#include <system_error>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {
static unsigned streamIntegerBitWidth(mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return intType.getWidth();
  return 0;
}

static std::optional<bool> peekBoolToken(const SimulatorState &state,
                                         unsigned operandOrdinal) {
  if (!hasInputToken(state, operandOrdinal))
    return std::nullopt;
  return boolToken(peekInputToken(state, operandOrdinal));
}

static std::optional<llvm::APInt> streamOperandBits(SimulatorState &state,
                                                    unsigned operandOrdinal,
                                                    mlir::Type type) {
  auto bits = tokenBitPattern(peekInputToken(state, operandOrdinal), type);
  if (!bits) {
    state.diagnostics.push_back(llvm::toString(bits.takeError()));
    return std::nullopt;
  }
  return *bits;
}

static llvm::Expected<Token> streamIvToken(const llvm::APInt &bits,
                                           mlir::Type type) {
  auto token = tokenFromBitPattern(bits, type);
  if (!token)
    return token;
  // Widths 2 through 64 still project their text through the narrow signed
  // value. An i1 keeps the canonical boolean text and wider widths print
  // from the exact bit pattern.
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  if (integer && integer.getWidth() >= 2 && integer.getWidth() <= 64)
    token->scalarValue = static_cast<std::uint64_t>(bits.getSExtValue());
  return token;
}

static bool
fireStream(dataflow::StreamOp op,
           const dataflow::CanonicalActorSchemaProjection &projection,
           SimulatorState &state) {
  const auto *payload =
      std::get_if<dataflow::StreamRecurrencePayload>(&projection.payload);
  assert(payload && "stream provider received the wrong semantic payload");
  if (state.failedStreamOps.contains(op.getOperation()))
    return false;

  StreamSemanticState &stream = state.streamStates[op.getOperation()];
  std::optional<StreamActivation> activation;
  if (stream.mode == StreamMode::Idle && hasInputToken(state, 0) &&
      hasInputToken(state, 1) && hasInputToken(state, 2)) {
    auto init = streamOperandBits(state, 0, op.getInit().getType());
    auto limit = streamOperandBits(state, 1, op.getLimit().getType());
    auto step = streamOperandBits(state, 2, op.getStep().getType());
    if (!init || !limit || !step) {
      state.failedStreamOps.insert(op.getOperation());
      return false;
    }
    activation =
        StreamActivation{std::move(*init), std::move(*limit), std::move(*step)};
  }

  auto transition = evaluateStreamTransition(
      stream,
      StreamSemanticConfig{payload->stepKind, payload->predicate,
                           streamIntegerBitWidth(op.getInit().getType())},
      activation);
  if (!transition) {
    state.diagnostics.push_back(llvm::toString(transition.takeError()));
    state.failedStreamOps.insert(op.getOperation());
    return false;
  }
  if (!transition->firing.ready)
    return false;

  std::optional<Token> iv;
  if (transition->emitIv) {
    auto token = streamIvToken(transition->iv, op.getIv().getType());
    if (!token) {
      state.diagnostics.push_back(llvm::toString(token.takeError()));
      state.failedStreamOps.insert(op.getOperation());
      return false;
    }
    iv = std::move(*token);
  }

  if (selectsSemanticInput(transition->firing.consumedInputs,
                           StreamInput::Init))
    (void)popInputToken(state, 0);
  if (selectsSemanticInput(transition->firing.consumedInputs,
                           StreamInput::Limit))
    (void)popInputToken(state, 1);
  if (selectsSemanticInput(transition->firing.consumedInputs,
                           StreamInput::Step))
    (void)popInputToken(state, 2);

  retainAndPublishActivationMemoryOrder(state, op.getOperation());
  if (iv) {
    emitResultToken(state, 0, std::move(*iv));
    ++state.streamTrueEmissionCounts[op.getOperation()];
  }
  if (transition->emitPhase)
    emitResultToken(state, 1, boolValueToken(transition->phase));
  stream = transition->nextState;
  releaseActivationMemoryOrder(state, op.getOperation(),
                               stream.mode == StreamMode::Idle);
  return true;
}

static bool
fireConstant(dataflow::ConstantOp op,
             const dataflow::CanonicalActorSchemaProjection &projection,
             SimulatorState &state) {
  if (!hasInputToken(state, 0))
    return false;
  const auto *payload =
      std::get_if<dataflow::ConstantValuePayload>(&projection.payload);
  assert(payload && "constant provider received the wrong semantic payload");
  auto tokenOrErr = tokenFromTypedAttr(payload->value);
  if (!tokenOrErr) {
    state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
    return false;
  }
  popInputToken(state, 0);
  emitResultToken(state, 0, *tokenOrErr);
  return true;
}

static bool fireCarry(dataflow::CarryOp op, SimulatorState &state) {
  LoopState &carry = state.carryStates[op.getOperation()];
  auto transition =
      evaluateCarryTransition(carry.semanticState, peekBoolToken(state, 0),
                              hasInputToken(state, 1), hasInputToken(state, 2));
  if (!transition.firing.ready)
    return false;

  std::optional<Token> forwarded;
  if (selectsSemanticInput(transition.firing.consumedInputs, CarryInput::Phase))
    (void)popInputToken(state, 0);
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           CarryInput::Init)) {
    Token value = popInputToken(state, 1);
    if (transition.forwardedInput == CarryInput::Init)
      forwarded = value;
  }
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           CarryInput::Next)) {
    Token value = popInputToken(state, 2);
    if (transition.forwardedInput == CarryInput::Next)
      forwarded = value;
  }
  retainAndPublishActivationMemoryOrder(state, op.getOperation());
  if (forwarded)
    emitResultToken(state, 0, *forwarded);
  carry.semanticState = transition.nextState;
  releaseActivationMemoryOrder(state, op.getOperation(),
                               carry.semanticState ==
                                   PhaseSemanticState::Initial);
  return true;
}

static bool fireInvariant(dataflow::InvariantOp op, SimulatorState &state) {
  LoopState &invariant = state.invariantStates[op.getOperation()];
  auto transition = evaluateInvariantTransition(invariant.semanticState,
                                                peekBoolToken(state, 0),
                                                hasInputToken(state, 1));
  if (!transition.firing.ready)
    return false;

  if (selectsSemanticInput(transition.firing.consumedInputs,
                           InvariantInput::Phase))
    (void)popInputToken(state, 0);
  std::optional<Token> init;
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           InvariantInput::Init))
    init = popInputToken(state, 1);
  if (transition.latchInput == InvariantInput::Init)
    invariant.latched = *init;

  retainAndPublishActivationMemoryOrder(state, op.getOperation());
  if (transition.output == InvariantOutputSource::InitInput)
    emitResultToken(state, 0, *init);
  else if (transition.output == InvariantOutputSource::Latched)
    emitResultToken(state, 0, *invariant.latched);
  if (transition.clearLatch)
    invariant.latched.reset();
  invariant.semanticState = transition.nextState;
  releaseActivationMemoryOrder(state, op.getOperation(),
                               invariant.semanticState ==
                                   PhaseSemanticState::Initial);
  return true;
}

static bool fireGate(dataflow::GateOp op, SimulatorState &state) {
  const GateSemanticState gate =
      state.gateContinueStates.contains(op.getOperation())
          ? GateSemanticState::Open
          : GateSemanticState::Closed;
  auto transition = evaluateGateTransition(gate, peekBoolToken(state, 0),
                                           hasInputToken(state, 1));
  if (!transition.firing.ready)
    return false;

  if (selectsSemanticInput(transition.firing.consumedInputs, GateInput::Phase))
    (void)popInputToken(state, 0);
  std::optional<Token> value;
  if (selectsSemanticInput(transition.firing.consumedInputs, GateInput::Value))
    value = popInputToken(state, 1);
  retainAndPublishActivationMemoryOrder(state, op.getOperation());
  if (transition.emitPhase)
    emitResultToken(state, 0, boolValueToken(transition.phase));
  if (transition.forwardedInput == GateInput::Value)
    emitResultToken(state, 1, *value);
  if (transition.nextState == GateSemanticState::Open)
    state.gateContinueStates.insert(op.getOperation());
  else
    state.gateContinueStates.erase(op.getOperation());
  releaseActivationMemoryOrder(state, op.getOperation(),
                               transition.nextState ==
                                   GateSemanticState::Closed);
  return true;
}

static bool fireSync(dataflow::SyncOp op, SimulatorState &state) {
  for (unsigned operand = 0; operand < op->getNumOperands(); ++operand) {
    if (!hasInputToken(state, operand))
      return false;
  }

  llvm::SmallVector<Token, 4> consumed;
  consumed.reserve(op->getNumOperands());
  for (unsigned operand = 0; operand < op->getNumOperands(); ++operand)
    consumed.push_back(popInputToken(state, operand));

  for (auto [resultOrdinal, token] : llvm::enumerate(consumed))
    emitResultToken(state, resultOrdinal, token);
  return true;
}

static bool fireMux(dataflow::MuxOp op, SimulatorState &state) {
  if (!hasInputToken(state, 0))
    return false;

  const Token &sel = peekInputToken(state, 0);
  const std::int64_t lane = mlir::isa<mlir::IntegerType>(op.getSel().getType())
                                ? boolToken(sel)
                                : integerToken(sel);
  if (lane < 0 || static_cast<std::size_t>(lane) >= op.getInputs().size()) {
    (void)popInputToken(state, 0);
    state.diagnostics.push_back("dataflow.mux selector is out of range");
    return false;
  }

  const unsigned selectedOperand = static_cast<unsigned>(lane) + 1;
  if (!hasInputToken(state, selectedOperand))
    return false;

  (void)popInputToken(state, 0);
  Token value = popInputToken(state, selectedOperand);
  emitResultToken(state, 0, value);
  return true;
}

static bool fireDemux(dataflow::DemuxOp op, SimulatorState &state) {
  if (!hasInputToken(state, 0) || !hasInputToken(state, 1))
    return false;

  const Token &sel = peekInputToken(state, 0);
  const std::int64_t lane = mlir::isa<mlir::IntegerType>(op.getSel().getType())
                                ? boolToken(sel)
                                : integerToken(sel);
  if (lane < 0 || static_cast<std::size_t>(lane) >= op.getOutputs().size()) {
    (void)popInputToken(state, 0);
    (void)popInputToken(state, 1);
    state.diagnostics.push_back("dataflow.demux selector is out of range");
    return false;
  }

  (void)popInputToken(state, 0);
  Token value = popInputToken(state, 1);
  emitResultToken(state, static_cast<unsigned>(lane), value);
  return true;
}

struct ParallelizeGroup {
  Token vector;
  Token mask;
  // Memory order from every active lane assembled across prior firings.
  MemoryOrderAccumulator frontier;
};

static llvm::Expected<ParallelizeGroup>
buildParallelizeGroup(dataflow::ParallelizeOp op, SimulatorState &state,
                      const ParallelizeState &parallel,
                      std::uint64_t activeItems) {
  mlir::VectorType vectorType = op.getVector().getType();
  auto maskWidth = tokenTypeBitWidth(op.getMask().getType());
  if (!maskWidth)
    return maskWidth.takeError();
  if (activeItems > parallel.slots.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dataflow.parallelize active lane count exceeds actor state");

  auto laneWidth = resolvedTokenTypeBitWidth(vectorType.getElementType(), op);
  if (!laneWidth)
    return laneWidth.takeError();
  llvm::SmallVector<PrimitiveValue, 8> vectorLanes(
      static_cast<std::size_t>(vectorType.getNumElements()),
      PrimitiveValue::integer(llvm::APInt(*laneWidth, 0)));
  llvm::APInt maskBits(*maskWidth, 0);
  MemoryOrderAccumulator frontier;
  for (std::uint64_t lane = 0; lane < activeItems; ++lane) {
    if (!parallel.slots[lane]) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "dataflow.parallelize active lane has no scalar token");
    }
    auto laneValue = primitiveValueFromToken(
        *parallel.slots[lane], vectorType.getElementType(), *laneWidth);
    if (!laneValue)
      return laneValue.takeError();
    vectorLanes[lane] = std::move(*laneValue);
    maskBits.setBit(static_cast<unsigned>(lane));
    frontier.absorb(parallel.slots[lane]->memoryOrder);
  }

  auto vectorToken =
      tokenFromVectorPrimitiveValues(vectorLanes, vectorType, op);
  if (!vectorToken)
    return vectorToken.takeError();
  auto maskToken = tokenFromBitPattern(maskBits, op.getMask().getType());
  if (!maskToken)
    return maskToken.takeError();
  return ParallelizeGroup{*vectorToken, *maskToken, std::move(frontier)};
}

static bool fireParallelize(dataflow::ParallelizeOp op, SimulatorState &state) {
  mlir::VectorType vectorType = op.getVector().getType();
  const std::uint64_t vectorLength = vectorType.getShape().front();
  ParallelizeState next;
  auto current = state.parallelizeStates.find(op.getOperation());
  if (current != state.parallelizeStates.end())
    next = current->second;
  if (next.slots.size() != vectorLength) {
    if (next.semanticState.pendingItems != 0) {
      state.diagnostics.push_back(
          "dataflow.parallelize state does not match its vector length");
      return false;
    }
    next.slots.assign(vectorLength, std::nullopt);
  }

  auto transition = evaluateParallelizeTransition(
      next.semanticState, vectorLength, peekBoolToken(state, 1),
      hasInputToken(state, 0));
  if (!transition.firing.ready)
    return false;

  // Retain each scalar phase's memory order across the multi-firing group. The
  // group publishes nothing until it emits, so this only accumulates.
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           ParallelizeInput::Phase))
    next.phaseFrontier.absorb(peekInputToken(state, 1).memoryOrder);

  std::optional<ParallelizeGroup> group;
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           ParallelizeInput::Data)) {
    const Token data = peekInputToken(state, 0);
    auto laneWidth = resolvedTokenTypeBitWidth(vectorType.getElementType(), op);
    if (!laneWidth) {
      state.diagnostics.push_back(llvm::toString(laneWidth.takeError()));
      return false;
    }
    auto laneValue =
        primitiveValueFromToken(data, vectorType.getElementType(), *laneWidth);
    if (!laneValue) {
      state.diagnostics.push_back(llvm::toString(laneValue.takeError()));
      return false;
    }
    const std::uint64_t lane = next.semanticState.pendingItems;
    if (lane >= next.slots.size()) {
      state.diagnostics.push_back(
          "dataflow.parallelize pending lane is out of range");
      return false;
    }
    next.slots[lane] = data;
  }
  if (transition.emitGroup) {
    auto groupOrErr =
        buildParallelizeGroup(op, state, next, transition.activeItems);
    if (!groupOrErr) {
      state.diagnostics.push_back(llvm::toString(groupOrErr.takeError()));
      return false;
    }
    group = *groupOrErr;
    group->frontier.absorbAll(next.phaseFrontier);
    next.slots.assign(vectorLength, std::nullopt);
    next.phaseFrontier.clear();
  }
  if (transition.emitFalsePhase)
    next.phaseFrontier.clear();
  next.semanticState = transition.nextState;

  if (selectsSemanticInput(transition.firing.consumedInputs,
                           ParallelizeInput::Phase))
    (void)popInputToken(state, 1);
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           ParallelizeInput::Data))
    (void)popInputToken(state, 0);
  state.parallelizeStates[op.getOperation()] = std::move(next);
  if (group) {
    state.firingMemoryOrderFrontier.absorbAll(group->frontier);
    emitResultToken(state, 0, group->vector);
    emitResultToken(state, 1, group->mask);
  }
  if (transition.emitTruePhase)
    emitResultToken(state, 2, boolValueToken(true));
  if (transition.emitFalsePhase)
    emitResultToken(state, 2, boolValueToken(false));
  return true;
}

static bool firePack(dataflow::PackOp op, SimulatorState &state) {
  if (!hasInputToken(state, 0))
    return false;
  Token vector = peekInputToken(state, 0);
  auto lanes = vectorPrimitiveValues(vector, op.getVector().getType(), op);
  if (!lanes) {
    state.diagnostics.push_back(llvm::toString(lanes.takeError()));
    return false;
  }
  const bool hasPoison = llvm::any_of(*lanes, [](const PrimitiveValue &lane) {
    return lane.state == PrimitiveValueState::Poison;
  });
  const bool hasUndef = llvm::any_of(*lanes, [](const PrimitiveValue &lane) {
    return lane.state == PrimitiveValueState::Undef;
  });
  llvm::Expected<Token> packed =
      hasPoison  ? exceptionalValueToken(PrimitiveValueState::Poison,
                                         op.getPacked().getType())
      : hasUndef ? exceptionalValueToken(PrimitiveValueState::Undef,
                                         op.getPacked().getType())
                 : [&]() -> llvm::Expected<Token> {
    auto bits = resolvedTokenBitPattern(vector, op.getVector().getType(), op);
    if (!bits)
      return bits.takeError();
    return tokenFromBitPattern(*bits, op.getPacked().getType());
  }();
  if (!packed) {
    state.diagnostics.push_back(llvm::toString(packed.takeError()));
    return false;
  }
  (void)popInputToken(state, 0);
  emitResultToken(state, 0, *packed);
  return true;
}

static bool fireUnpack(dataflow::UnpackOp op, SimulatorState &state) {
  if (!hasInputToken(state, 0))
    return false;
  Token packedToken = peekInputToken(state, 0);
  llvm::Expected<Token> vector = [&]() -> llvm::Expected<Token> {
    if (packedToken.valueState != PrimitiveValueState::Defined)
      return exceptionalValueToken(packedToken.valueState,
                                   op.getVector().getType());
    auto bits = tokenBitPattern(packedToken, op.getPacked().getType());
    if (!bits)
      return bits.takeError();
    return tokenFromResolvedBitPattern(*bits, op.getVector().getType(), op);
  }();
  if (!vector) {
    state.diagnostics.push_back(llvm::toString(vector.takeError()));
    return false;
  }
  (void)popInputToken(state, 0);
  emitResultToken(state, 0, *vector);
  return true;
}

static bool fireSerialize(dataflow::SerializeOp op, SimulatorState &state) {
  auto transition = evaluateSerializeTransition(peekBoolToken(state, 2),
                                                hasInputToken(state, 0),
                                                hasInputToken(state, 1));
  if (!transition.firing.ready)
    return false;

  llvm::SmallVector<Token, 8> activeLanes;
  if (transition.emitActiveItems) {
    Token vectorToken = peekInputToken(state, 0);
    Token maskToken = peekInputToken(state, 1);
    mlir::VectorType vectorType = op.getVector().getType();
    auto vectorLanes = vectorPrimitiveValues(vectorToken, vectorType, op);
    auto maskLanes =
        vectorPrimitiveValues(maskToken, op.getMask().getType(), op);
    if (!vectorLanes || !maskLanes) {
      if (!vectorLanes)
        state.diagnostics.push_back(llvm::toString(vectorLanes.takeError()));
      if (!maskLanes)
        state.diagnostics.push_back(llvm::toString(maskLanes.takeError()));
      return false;
    }
    for (unsigned lane = 0; lane < vectorType.getShape().front(); ++lane) {
      const PrimitiveValue &mask = (*maskLanes)[lane];
      if (!mask.isDefined()) {
        state.diagnostics.push_back(
            "dataflow.serialize exceptional mask cardinality has no exact "
            "single-path provider");
        state.failure = RunFailure::UnsupportedCapability;
        return false;
      }
      if (mask.bits->isZero())
        continue;
      auto laneToken = tokenFromPrimitiveValue((*vectorLanes)[lane],
                                               vectorType.getElementType());
      if (!laneToken) {
        state.diagnostics.push_back(llvm::toString(laneToken.takeError()));
        return false;
      }
      activeLanes.push_back(*laneToken);
    }
  }

  if (selectsSemanticInput(transition.firing.consumedInputs,
                           SerializeInput::Phase))
    (void)popInputToken(state, 2);
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           SerializeInput::Vector))
    (void)popInputToken(state, 0);
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           SerializeInput::Mask))
    (void)popInputToken(state, 1);
  for (const Token &lane : activeLanes) {
    emitResultToken(state, 0, lane);
    emitResultToken(state, 1, boolValueToken(true));
  }
  if (transition.emitFalsePhase)
    emitResultToken(state, 1, boolValueToken(false));
  return true;
}

static bool hasVectorPrimitiveType(mlir::Operation *op) {
  return llvm::any_of(op->getOperandTypes(),
                      [](mlir::Type type) {
                        return mlir::isa<mlir::VectorType>(type);
                      }) ||
         llvm::any_of(op->getResultTypes(), [](mlir::Type type) {
           return mlir::isa<mlir::VectorType>(type);
         });
}

static llvm::Error validatePrimitiveElementType(mlir::Type type,
                                                llvm::StringRef role) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (integer.getWidth() == 0 || integer.getWidth() > 64)
      return llvm::createStringError(
          std::errc::not_supported,
          "%s element type i%u has width %u; scalar primitive evaluator "
          "supports integer lane widths from 1 to 64",
          role.str().c_str(), integer.getWidth(), integer.getWidth());
    return llvm::Error::success();
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (!llvm::APFloat::isRepresentableBy(floating.getFloatSemantics(),
                                          llvm::APFloat::IEEEdouble())) {
      std::string typeName = llvm::formatv("{0}", floating).str();
      return llvm::createStringError(
          std::errc::not_supported,
          "%s element type %s has %u-bit floating-point semantics not exactly "
          "representable by the scalar evaluator's f64 lane model",
          role.str().c_str(), typeName.c_str(), floating.getWidth());
    }
    return llvm::Error::success();
  }
  std::string typeName = llvm::formatv("{0}", type).str();
  return llvm::createStringError(
      std::errc::not_supported,
      "%s element type %s has no scalar primitive representation",
      role.str().c_str(), typeName.c_str());
}

static llvm::Expected<mlir::VectorType>
validateElementwiseVectorPrimitive(mlir::Operation *op, mlir::Value result) {
  auto resultType = mlir::dyn_cast<mlir::VectorType>(result.getType());
  if (!resultType)
    return llvm::createStringError(
        std::errc::not_supported,
        "vector primitive must produce a vector result");
  if (resultType.getRank() < 1 || resultType.isScalable())
    return llvm::createStringError(
        std::errc::not_supported,
        "vector primitive result must be fixed-size and positive-rank");
  if (llvm::Error error =
          validatePrimitiveElementType(resultType.getElementType(), "result"))
    return std::move(error);
  if (op->getNumOperands() == 0)
    return llvm::createStringError(std::errc::not_supported,
                                   "vector primitive has no operands");

  for (mlir::Type type : op->getOperandTypes()) {
    auto vectorType = mlir::dyn_cast<mlir::VectorType>(type);
    if (!vectorType || vectorType.getRank() < 1 || vectorType.isScalable())
      return llvm::createStringError(
          std::errc::not_supported,
          "vector primitive operands must be fixed-size and positive-rank");
    if (vectorType.getShape() != resultType.getShape())
      return llvm::createStringError(
          std::errc::not_supported,
          "vector primitive operand and result shapes must match");
    if (llvm::Error error = validatePrimitiveElementType(
            vectorType.getElementType(), "operand"))
      return std::move(error);
  }
  return resultType;
}

llvm::Expected<PrimitiveOperationDescriptor> primitiveDescriptorForActor(
    const dataflow::CanonicalActorSchemaProjection &projection,
    mlir::Operation *op) {
  if (!hasVectorPrimitiveType(op))
    return primitiveDescriptor(projection, op, op->getResult(0));

  auto resultType = mlir::cast<mlir::VectorType>(op->getResult(0).getType());
  auto operandType = mlir::cast<mlir::VectorType>(op->getOperand(0).getType());
  return primitiveDescriptor(projection, op, resultType.getElementType(),
                             operandType.getElementType());
}

llvm::Error validatePrimitiveTokenTypes(mlir::Operation *op,
                                        mlir::Value result) {
  if (!hasVectorPrimitiveType(op))
    return llvm::Error::success();
  auto vectorType = validateElementwiseVectorPrimitive(op, result);
  if (!vectorType)
    return vectorType.takeError();
  return llvm::Error::success();
}

static llvm::Expected<Token> evaluateElementwiseVectorPrimitive(
    mlir::Operation *op, const PrimitiveOperationDescriptor &descriptor,
    mlir::Value result, llvm::ArrayRef<Token> inputTokens) {
  mlir::VectorType resultType = mlir::cast<mlir::VectorType>(result.getType());

  llvm::SmallVector<llvm::SmallVector<PrimitiveValue, 8>, 4> operandLanes;
  operandLanes.reserve(inputTokens.size());
  for (auto [operand, token] :
       llvm::zip_equal(op->getOpOperands(), inputTokens)) {
    auto vectorType = mlir::cast<mlir::VectorType>(operand.get().getType());
    auto lanes = vectorPrimitiveValues(token, vectorType, op);
    if (!lanes)
      return lanes.takeError();
    operandLanes.push_back(std::move(*lanes));
  }

  llvm::SmallVector<PrimitiveValue, 8> resultLanes;
  resultLanes.reserve(static_cast<std::size_t>(resultType.getNumElements()));
  // Operands and result share one shape, so the flattened lane ordinal names
  // the same logical element in each of them.
  for (std::size_t lane = 0;
       lane < static_cast<std::size_t>(resultType.getNumElements()); ++lane) {
    llvm::SmallVector<PrimitiveValue, 4> laneOperands;
    laneOperands.reserve(inputTokens.size());
    for (const auto &lanes : operandLanes)
      laneOperands.push_back(lanes[lane]);

    auto laneResult = evaluatePrimitiveOperation(descriptor, laneOperands);
    if (!laneResult)
      return llvm::joinErrors(
          llvm::createStringError(
              std::errc::invalid_argument, "%s failed for vector lane %zu",
              dataflow::operationSchemaSpelling(descriptor.actor.schema)
                  .str()
                  .c_str(),
              lane),
          laneResult.takeError());
    resultLanes.push_back(std::move(*laneResult));
  }
  return tokenFromVectorPrimitiveValues(resultLanes, resultType, op);
}

llvm::Expected<Token>
evaluatePrimitiveToken(mlir::Operation *op,
                       const PrimitiveOperationDescriptor &descriptor,
                       mlir::Value result, llvm::ArrayRef<Token> inputTokens) {
  if (inputTokens.size() != op->getNumOperands())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "primitive token count does not match operation operands");
  if (hasVectorPrimitiveType(op))
    return evaluateElementwiseVectorPrimitive(op, descriptor, result,
                                              inputTokens);
  llvm::SmallVector<PrimitiveValue, 4> operands;
  operands.reserve(inputTokens.size());
  for (auto [operand, token] :
       llvm::zip_equal(op->getOpOperands(), inputTokens)) {
    auto value = primitiveValueFromToken(token, operand.get().getType(),
                                         descriptor.operandBitWidth);
    if (!value)
      return value.takeError();
    operands.push_back(*value);
  }
  auto value = evaluatePrimitiveOperation(descriptor, operands);
  if (!value)
    return value.takeError();
  return tokenFromPrimitiveValue(*value, result.getType());
}

static bool firePrimitiveOperation(mlir::Operation *op, mlir::Value result,
                                   SimulatorState &state) {
  if (state.terminalComputeOps.contains(op))
    return false;
  if (op->getNumOperands() == 0 && state.oneShotOps.contains(op))
    return false;
  for (unsigned operand = 0; operand < op->getNumOperands(); ++operand) {
    if (!hasInputToken(state, operand))
      return false;
  }

  llvm::SmallVector<Token, 4> operands;
  operands.reserve(op->getNumOperands());
  for (unsigned operand = 0; operand < op->getNumOperands(); ++operand)
    operands.push_back(peekInputToken(state, operand));
  assert(state.currentActorPlan && state.currentActorPlan->operation == op &&
         state.currentActorPlan->primitive &&
         "admitted primitive actor has no execution descriptor");
  auto resultToken = evaluatePrimitiveToken(
      op, *state.currentActorPlan->primitive, result, operands);
  if (!resultToken) {
    state.diagnostics.push_back(llvm::toString(resultToken.takeError()));
    state.terminalComputeOps.insert(op);
    return false;
  }
  for (unsigned operand = 0; operand < op->getNumOperands(); ++operand)
    (void)popInputToken(state, operand);
  emitResultToken(state, 0, *resultToken);
  if (op->getNumOperands() == 0)
    state.oneShotOps.insert(op);
  return true;
}

static bool
fireArithConstant(mlir::arith::ConstantOp op,
                  const dataflow::CanonicalActorSchemaProjection &projection,
                  SimulatorState &state) {
  if (state.oneShotOps.contains(op.getOperation()))
    return false;
  const auto *payload =
      std::get_if<dataflow::ConstantValuePayload>(&projection.payload);
  assert(payload && "constant provider received the wrong semantic payload");
  auto tokenOrErr = tokenFromTypedAttr(payload->value);
  if (!tokenOrErr) {
    state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
    return false;
  }
  emitResultToken(state, 0, *tokenOrErr);
  state.oneShotOps.insert(op.getOperation());
  return true;
}

template <typename OpT, bool (*Fire)(OpT, SimulatorState &)>
static bool
fireTypedActor(mlir::Operation *op,
               const dataflow::CanonicalActorSchemaProjection &projection,
               SimulatorState &state) {
  (void)projection;
  return Fire(mlir::cast<OpT>(op), state);
}

static bool
fireStreamActor(mlir::Operation *op,
                const dataflow::CanonicalActorSchemaProjection &projection,
                SimulatorState &state) {
  return fireStream(mlir::cast<dataflow::StreamOp>(op), projection, state);
}

static bool fireDataflowConstantActor(
    mlir::Operation *op,
    const dataflow::CanonicalActorSchemaProjection &projection,
    SimulatorState &state) {
  return fireConstant(mlir::cast<dataflow::ConstantOp>(op), projection, state);
}

static bool fireArithConstantActor(
    mlir::Operation *op,
    const dataflow::CanonicalActorSchemaProjection &projection,
    SimulatorState &state) {
  return fireArithConstant(mlir::cast<mlir::arith::ConstantOp>(op), projection,
                           state);
}

static bool
firePrimitiveActor(mlir::Operation *op,
                   const dataflow::CanonicalActorSchemaProjection &projection,
                   SimulatorState &state) {
  (void)projection;
  return firePrimitiveOperation(op, op->getResult(0), state);
}

std::optional<ActorRuntimeProvider>
actorRuntimeProvider(dataflow::OperationSchemaId schema) {
  using Probe = ActorTransitionProbeKind;
  if (isSupportedPrimitiveOperation(schema))
    return ActorRuntimeProvider{firePrimitiveActor, Probe::StatelessCompute};

  using Schema = dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::VectorExtract:
  case Schema::VectorInsert:
  case Schema::VectorShuffle:
    return ActorRuntimeProvider{fireVectorStructuralActor,
                                Probe::StatelessCompute};
  case Schema::LLVMGetElementPtr:
    return ActorRuntimeProvider{fireGetElementPtr, Probe::GetElementPtr};
  case Schema::ArithConstant:
    return ActorRuntimeProvider{fireArithConstantActor, Probe::OneShot};
  case Schema::DataflowStream:
    return ActorRuntimeProvider{fireStreamActor, Probe::Stream};
  case Schema::DataflowConstant:
    return ActorRuntimeProvider{fireDataflowConstantActor, Probe::AllInputs};
  case Schema::DataflowCarry:
    return ActorRuntimeProvider{fireTypedActor<dataflow::CarryOp, fireCarry>,
                                Probe::Carry};
  case Schema::DataflowInvariant:
    return ActorRuntimeProvider{
        fireTypedActor<dataflow::InvariantOp, fireInvariant>, Probe::Invariant};
  case Schema::DataflowGate:
    return ActorRuntimeProvider{fireTypedActor<dataflow::GateOp, fireGate>,
                                Probe::Gate};
  case Schema::DataflowSync:
    return ActorRuntimeProvider{fireTypedActor<dataflow::SyncOp, fireSync>,
                                Probe::AllInputs};
  case Schema::DataflowMux:
    return ActorRuntimeProvider{fireTypedActor<dataflow::MuxOp, fireMux>,
                                Probe::Mux};
  case Schema::DataflowDemux:
    return ActorRuntimeProvider{fireTypedActor<dataflow::DemuxOp, fireDemux>,
                                Probe::Demux};
  case Schema::DataflowParallelize:
    return ActorRuntimeProvider{
        fireTypedActor<dataflow::ParallelizeOp, fireParallelize>,
        Probe::Parallelize};
  case Schema::DataflowPack:
    return ActorRuntimeProvider{fireTypedActor<dataflow::PackOp, firePack>,
                                Probe::AllInputs};
  case Schema::DataflowUnpack:
    return ActorRuntimeProvider{fireTypedActor<dataflow::UnpackOp, fireUnpack>,
                                Probe::AllInputs};
  case Schema::DataflowSerialize:
    return ActorRuntimeProvider{
        fireTypedActor<dataflow::SerializeOp, fireSerialize>, Probe::Serialize};
  case Schema::DataflowLoad:
    return ActorRuntimeProvider{fireTypedActor<dataflow::LoadOp, fireLoad>,
                                Probe::MemoryInputs};
  case Schema::DataflowStore:
    return ActorRuntimeProvider{fireTypedActor<dataflow::StoreOp, fireStore>,
                                Probe::MemoryInputs};
  default:
    return std::nullopt;
  }
}

ActorProvider actorProvider(dataflow::OperationSchemaId schema) {
  auto runtime = actorRuntimeProvider(schema);
  return runtime ? runtime->commit : nullptr;
}

static bool hasUnsupportedMemoryContract(
    const dataflow::CanonicalActorSchemaProjection &projection) {
  const auto *memory =
      std::get_if<dataflow::MemoryContractPayload>(&projection.payload);
  if (!memory)
    return false;
  const auto *plain = std::get_if<dataflow::PlainAccessProjection>(memory);
  return !plain || plain->isVolatile;
}

bool fireActorOperation(const ActorExecutionPlan &plan, SimulatorState &state) {
  assert(plan.provider && "admitted actor has no simulator provider");
  if (!plan.provider(plan.operation, plan.projection, state))
    return false;
  return recordEvent(state, plan.projection.schema);
}

std::optional<UnsupportedOperation> unsupportedActorProvider(
    mlir::Operation *op,
    const dataflow::CanonicalActorSchemaProjection &projection) {
  if (hasUnsupportedMemoryContract(projection))
    return UnsupportedOperation{
        unsupportedOperationLabel(op),
        "atomic, volatile, and fence memory contracts have no dynamic "
        "consistency-domain semantics"};

  if (auto reason = unsupportedMemoryActorRepresentation(op))
    return UnsupportedOperation{unsupportedOperationLabel(op),
                                std::move(*reason)};

  if (!actorProvider(projection.schema))
    return UnsupportedOperation{unsupportedOperationLabel(op), ""};

  if (isSupportedPrimitiveOperation(projection.schema)) {
    if (op->getNumResults() != 1)
      return UnsupportedOperation{unsupportedOperationLabel(op),
                                  "primitive provider requires one result"};
    if (llvm::Error error = validatePrimitiveTokenTypes(op, op->getResult(0)))
      return UnsupportedOperation{unsupportedOperationLabel(op),
                                  llvm::toString(std::move(error))};
  }
  return std::nullopt;
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

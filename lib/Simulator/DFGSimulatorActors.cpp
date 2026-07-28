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
                                         mlir::OpOperand &operand) {
  if (!hasToken(state, operand))
    return std::nullopt;
  return boolToken(peekToken(state, operand));
}

static std::optional<llvm::APInt> streamOperandBits(SimulatorState &state,
                                                    mlir::OpOperand &operand) {
  auto bits =
      tokenBitPattern(peekToken(state, operand), operand.get().getType());
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
    token->intValue = bits.getSExtValue();
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
  if (stream.mode == StreamMode::Idle && hasToken(state, op->getOpOperand(0)) &&
      hasToken(state, op->getOpOperand(1)) &&
      hasToken(state, op->getOpOperand(2))) {
    auto init = streamOperandBits(state, op->getOpOperand(0));
    auto limit = streamOperandBits(state, op->getOpOperand(1));
    auto step = streamOperandBits(state, op->getOpOperand(2));
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
    (void)popToken(state, op->getOpOperand(0));
  if (selectsSemanticInput(transition->firing.consumedInputs,
                           StreamInput::Limit))
    (void)popToken(state, op->getOpOperand(1));
  if (selectsSemanticInput(transition->firing.consumedInputs,
                           StreamInput::Step))
    (void)popToken(state, op->getOpOperand(2));

  retainAndPublishActivationMemoryOrder(state, op.getOperation());
  if (iv) {
    emitToken(state, op.getIv(), std::move(*iv));
    ++state.streamTrueEmissionCounts[op.getOperation()];
  }
  if (transition->emitPhase)
    emitToken(state, op.getPhase(), boolValueToken(transition->phase));
  stream = transition->nextState;
  releaseActivationMemoryOrder(state, op.getOperation(),
                               stream.mode == StreamMode::Idle);
  return true;
}

static bool
fireConstant(dataflow::ConstantOp op,
             const dataflow::CanonicalActorSchemaProjection &projection,
             SimulatorState &state) {
  if (!hasToken(state, op->getOpOperand(0)))
    return false;
  const auto *payload =
      std::get_if<dataflow::ConstantValuePayload>(&projection.payload);
  assert(payload && "constant provider received the wrong semantic payload");
  auto tokenOrErr = tokenFromTypedAttr(payload->value);
  if (!tokenOrErr) {
    state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
    return false;
  }
  popToken(state, op->getOpOperand(0));
  emitToken(state, op.getValue(), *tokenOrErr);
  return true;
}

static bool fireCarry(dataflow::CarryOp op, SimulatorState &state) {
  LoopState &carry = state.carryStates[op.getOperation()];
  auto transition = evaluateCarryTransition(
      carry.semanticState, peekBoolToken(state, op->getOpOperand(0)),
      hasToken(state, op->getOpOperand(1)),
      hasToken(state, op->getOpOperand(2)));
  if (!transition.firing.ready)
    return false;

  std::optional<Token> forwarded;
  if (selectsSemanticInput(transition.firing.consumedInputs, CarryInput::Phase))
    (void)popToken(state, op->getOpOperand(0));
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           CarryInput::Init)) {
    Token value = popToken(state, op->getOpOperand(1));
    if (transition.forwardedInput == CarryInput::Init)
      forwarded = value;
  }
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           CarryInput::Next)) {
    Token value = popToken(state, op->getOpOperand(2));
    if (transition.forwardedInput == CarryInput::Next)
      forwarded = value;
  }
  retainAndPublishActivationMemoryOrder(state, op.getOperation());
  if (forwarded)
    emitToken(state, op.getOutput(), *forwarded);
  carry.semanticState = transition.nextState;
  releaseActivationMemoryOrder(state, op.getOperation(),
                               carry.semanticState ==
                                   PhaseSemanticState::Initial);
  return true;
}

static bool fireInvariant(dataflow::InvariantOp op, SimulatorState &state) {
  LoopState &invariant = state.invariantStates[op.getOperation()];
  auto transition = evaluateInvariantTransition(
      invariant.semanticState, peekBoolToken(state, op->getOpOperand(0)),
      hasToken(state, op->getOpOperand(1)));
  if (!transition.firing.ready)
    return false;

  if (selectsSemanticInput(transition.firing.consumedInputs,
                           InvariantInput::Phase))
    (void)popToken(state, op->getOpOperand(0));
  std::optional<Token> init;
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           InvariantInput::Init))
    init = popToken(state, op->getOpOperand(1));
  if (transition.latchInput == InvariantInput::Init)
    invariant.latched = *init;

  retainAndPublishActivationMemoryOrder(state, op.getOperation());
  if (transition.output == InvariantOutputSource::InitInput)
    emitToken(state, op.getOutput(), *init);
  else if (transition.output == InvariantOutputSource::Latched)
    emitToken(state, op.getOutput(), *invariant.latched);
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
  auto transition = evaluateGateTransition(
      gate, peekBoolToken(state, op.getBeforeCondMutable()),
      hasToken(state, op.getBeforeValueMutable()));
  if (!transition.firing.ready)
    return false;

  if (selectsSemanticInput(transition.firing.consumedInputs, GateInput::Phase))
    (void)popToken(state, op.getBeforeCondMutable());
  std::optional<Token> value;
  if (selectsSemanticInput(transition.firing.consumedInputs, GateInput::Value))
    value = popToken(state, op.getBeforeValueMutable());
  retainAndPublishActivationMemoryOrder(state, op.getOperation());
  if (transition.emitPhase)
    emitToken(state, op.getAfterCond(), boolValueToken(transition.phase));
  if (transition.forwardedInput == GateInput::Value)
    emitToken(state, op.getAfterValue(), *value);
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
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    if (!hasToken(state, operand))
      return false;
  }

  llvm::SmallVector<Token> consumed;
  consumed.reserve(op->getNumOperands());
  for (mlir::OpOperand &operand : op->getOpOperands())
    consumed.push_back(popToken(state, operand));

  for (auto [result, token] : llvm::zip_equal(op->getResults(), consumed))
    emitToken(state, result, token);
  return true;
}

static bool fireMux(dataflow::MuxOp op, SimulatorState &state) {
  mlir::OpOperand &selOperand = op->getOpOperand(0);
  if (!hasToken(state, selOperand))
    return false;

  const Token &sel = peekToken(state, selOperand);
  const std::int64_t lane = mlir::isa<mlir::IntegerType>(op.getSel().getType())
                                ? boolToken(sel)
                                : integerToken(sel);
  if (lane < 0 || static_cast<std::size_t>(lane) >= op.getInputs().size()) {
    (void)popToken(state, selOperand);
    state.diagnostics.push_back("dataflow.mux selector is out of range");
    return false;
  }

  mlir::OpOperand &selectedOperand =
      op->getOpOperand(static_cast<unsigned>(lane) + 1);
  if (!hasToken(state, selectedOperand))
    return false;

  (void)popToken(state, selOperand);
  Token value = popToken(state, selectedOperand);
  emitToken(state, op.getOutput(), value);
  return true;
}

static bool fireDemux(dataflow::DemuxOp op, SimulatorState &state) {
  mlir::OpOperand &selOperand = op->getOpOperand(0);
  mlir::OpOperand &inputOperand = op->getOpOperand(1);
  if (!hasToken(state, selOperand) || !hasToken(state, inputOperand))
    return false;

  const Token &sel = peekToken(state, selOperand);
  const std::int64_t lane = mlir::isa<mlir::IntegerType>(op.getSel().getType())
                                ? boolToken(sel)
                                : integerToken(sel);
  if (lane < 0 || static_cast<std::size_t>(lane) >= op.getOutputs().size()) {
    (void)popToken(state, selOperand);
    (void)popToken(state, inputOperand);
    state.diagnostics.push_back("dataflow.demux selector is out of range");
    return false;
  }

  (void)popToken(state, selOperand);
  Token value = popToken(state, inputOperand);
  emitToken(state, op.getOutputs()[static_cast<unsigned>(lane)], value);
  return true;
}

struct ParallelizeGroup {
  Token vector;
  Token mask;
  // Memory order from every active lane assembled across prior firings.
  llvm::SmallVector<SyncEffectId, 2> frontier;
};

static llvm::Expected<ParallelizeGroup>
buildParallelizeGroup(dataflow::ParallelizeOp op, SimulatorState &state,
                      const ParallelizeState &parallel,
                      std::uint64_t activeItems) {
  mlir::VectorType vectorType = op.getVector().getType();
  auto laneWidth = tokenTypeBitWidth(vectorType.getElementType());
  if (!laneWidth)
    return laneWidth.takeError();
  auto totalWidth = tokenTypeBitWidth(vectorType);
  if (!totalWidth)
    return totalWidth.takeError();
  auto maskWidth = tokenTypeBitWidth(op.getMask().getType());
  if (!maskWidth)
    return maskWidth.takeError();
  if (activeItems > parallel.slots.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "dataflow.parallelize active lane count exceeds actor state");

  llvm::APInt vectorBits(*totalWidth, 0);
  llvm::APInt maskBits(*maskWidth, 0);
  llvm::SmallVector<SyncEffectId, 2> frontier;
  for (std::uint64_t lane = 0; lane < activeItems; ++lane) {
    if (!parallel.slots[lane]) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "dataflow.parallelize active lane has no scalar token");
    }
    auto laneBits =
        tokenBitPattern(*parallel.slots[lane], vectorType.getElementType());
    if (!laneBits)
      return laneBits.takeError();
    vectorBits.insertBits(*laneBits, *laneWidth * static_cast<unsigned>(lane));
    maskBits.setBit(static_cast<unsigned>(lane));
    llvm::ArrayRef<SyncEffectId> laneOrder =
        state.memoryOrderFrontiers.elements(parallel.slots[lane]->memoryOrder);
    frontier.append(laneOrder.begin(), laneOrder.end());
  }

  auto vectorToken = tokenFromBitPattern(vectorBits, vectorType);
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
      next.semanticState, vectorLength,
      peekBoolToken(state, op.getScalarPhaseMutable()),
      hasToken(state, op.getDataMutable()));
  if (!transition.firing.ready)
    return false;

  // Retain each scalar phase's memory order across the multi-firing group. The
  // group publishes nothing until it emits, so this only accumulates.
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           ParallelizeInput::Phase))
    next.phaseFrontier.append(state.memoryOrderFrontiers.elements(
        peekToken(state, op.getScalarPhaseMutable()).memoryOrder));

  std::optional<ParallelizeGroup> group;
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           ParallelizeInput::Data)) {
    const Token data = peekToken(state, op.getDataMutable());
    auto laneBits = tokenBitPattern(data, vectorType.getElementType());
    if (!laneBits) {
      state.diagnostics.push_back(llvm::toString(laneBits.takeError()));
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
    llvm::ArrayRef<SyncEffectId> phaseOrder = next.phaseFrontier.elements();
    group->frontier.append(phaseOrder.begin(), phaseOrder.end());
    next.slots.assign(vectorLength, std::nullopt);
    next.phaseFrontier.clear();
  }
  if (transition.emitFalsePhase)
    next.phaseFrontier.clear();
  next.semanticState = transition.nextState;

  if (selectsSemanticInput(transition.firing.consumedInputs,
                           ParallelizeInput::Phase))
    (void)popToken(state, op.getScalarPhaseMutable());
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           ParallelizeInput::Data))
    (void)popToken(state, op.getDataMutable());
  state.parallelizeStates[op.getOperation()] = std::move(next);
  if (group) {
    state.firingMemoryOrderFrontier.append(group->frontier);
    emitToken(state, op.getVector(), group->vector);
    emitToken(state, op.getMask(), group->mask);
  }
  if (transition.emitTruePhase)
    emitToken(state, op.getGroupPhase(), boolValueToken(true));
  if (transition.emitFalsePhase)
    emitToken(state, op.getGroupPhase(), boolValueToken(false));
  return true;
}

static bool firePack(dataflow::PackOp op, SimulatorState &state) {
  if (!hasToken(state, op.getVectorMutable()))
    return false;
  Token vector = peekToken(state, op.getVectorMutable());
  auto bits = tokenBitPattern(vector, op.getVector().getType());
  if (!bits) {
    state.diagnostics.push_back(llvm::toString(bits.takeError()));
    return false;
  }
  auto packed = tokenFromBitPattern(*bits, op.getPacked().getType());
  if (!packed) {
    state.diagnostics.push_back(llvm::toString(packed.takeError()));
    return false;
  }
  (void)popToken(state, op.getVectorMutable());
  emitToken(state, op.getPacked(), *packed);
  return true;
}

static bool fireUnpack(dataflow::UnpackOp op, SimulatorState &state) {
  if (!hasToken(state, op.getPackedMutable()))
    return false;
  Token packedToken = peekToken(state, op.getPackedMutable());
  auto bits = tokenBitPattern(packedToken, op.getPacked().getType());
  if (!bits) {
    state.diagnostics.push_back(llvm::toString(bits.takeError()));
    return false;
  }
  auto vector = tokenFromBitPattern(*bits, op.getVector().getType());
  if (!vector) {
    state.diagnostics.push_back(llvm::toString(vector.takeError()));
    return false;
  }
  (void)popToken(state, op.getPackedMutable());
  emitToken(state, op.getVector(), *vector);
  return true;
}

static bool fireSerialize(dataflow::SerializeOp op, SimulatorState &state) {
  auto transition = evaluateSerializeTransition(
      peekBoolToken(state, op.getGroupPhaseMutable()),
      hasToken(state, op.getVectorMutable()),
      hasToken(state, op.getMaskMutable()));
  if (!transition.firing.ready)
    return false;

  llvm::SmallVector<Token> activeLanes;
  if (transition.emitActiveItems) {
    Token vectorToken = peekToken(state, op.getVectorMutable());
    Token maskToken = peekToken(state, op.getMaskMutable());
    mlir::VectorType vectorType = op.getVector().getType();
    auto vectorBits = tokenBitPattern(vectorToken, vectorType);
    auto maskBits = tokenBitPattern(maskToken, op.getMask().getType());
    if (!vectorBits || !maskBits) {
      if (!vectorBits)
        state.diagnostics.push_back(llvm::toString(vectorBits.takeError()));
      if (!maskBits)
        state.diagnostics.push_back(llvm::toString(maskBits.takeError()));
      return false;
    }

    auto laneWidth = tokenTypeBitWidth(vectorType.getElementType());
    if (!laneWidth) {
      state.diagnostics.push_back(llvm::toString(laneWidth.takeError()));
      return false;
    }
    for (unsigned lane = 0; lane < vectorType.getShape().front(); ++lane) {
      if (!(*maskBits)[lane])
        continue;
      llvm::APInt laneBits =
          vectorBits->extractBits(*laneWidth, *laneWidth * lane);
      auto laneToken =
          tokenFromBitPattern(laneBits, vectorType.getElementType());
      if (!laneToken) {
        state.diagnostics.push_back(llvm::toString(laneToken.takeError()));
        return false;
      }
      activeLanes.push_back(*laneToken);
    }
  }

  if (selectsSemanticInput(transition.firing.consumedInputs,
                           SerializeInput::Phase))
    (void)popToken(state, op.getGroupPhaseMutable());
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           SerializeInput::Vector))
    (void)popToken(state, op.getVectorMutable());
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           SerializeInput::Mask))
    (void)popToken(state, op.getMaskMutable());
  for (const Token &lane : activeLanes) {
    emitToken(state, op.getData(), lane);
    emitToken(state, op.getScalarPhase(), boolValueToken(true));
  }
  if (transition.emitFalsePhase)
    emitToken(state, op.getScalarPhase(), boolValueToken(false));
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

  llvm::SmallVector<llvm::APInt> operandBits;
  llvm::SmallVector<unsigned> operandWidths;
  operandBits.reserve(inputTokens.size());
  operandWidths.reserve(inputTokens.size());
  for (auto [operand, token] :
       llvm::zip_equal(op->getOpOperands(), inputTokens)) {
    auto vectorType = mlir::cast<mlir::VectorType>(operand.get().getType());
    auto bits = tokenBitPattern(token, vectorType);
    if (!bits)
      return bits.takeError();
    auto width = tokenTypeBitWidth(vectorType.getElementType());
    if (!width)
      return width.takeError();
    operandBits.push_back(*bits);
    operandWidths.push_back(*width);
  }

  auto resultWidth = tokenTypeBitWidth(resultType);
  if (!resultWidth)
    return resultWidth.takeError();
  auto resultElementWidth = tokenTypeBitWidth(resultType.getElementType());
  if (!resultElementWidth)
    return resultElementWidth.takeError();
  llvm::APInt resultBits(*resultWidth, 0);
  // Operands and result share one shape, so the flattened lane ordinal names
  // the same logical element in each of them. The canonical row-major order
  // comes from that shared bit layout rather than from per-axis strides.
  for (unsigned lane = 0; lane < resultType.getNumElements(); ++lane) {
    llvm::SmallVector<PrimitiveValue> laneOperands;
    laneOperands.reserve(inputTokens.size());
    for (auto [operand, bits, width] :
         llvm::zip_equal(op->getOpOperands(), operandBits, operandWidths)) {
      auto vectorType = mlir::cast<mlir::VectorType>(operand.get().getType());
      llvm::APInt laneBits = bits.extractBits(width, width * lane);
      auto laneToken =
          tokenFromBitPattern(laneBits, vectorType.getElementType());
      if (!laneToken)
        return laneToken.takeError();
      auto laneValue = primitiveValueFromToken(
          *laneToken, vectorType.getElementType(), descriptor.operandBitWidth);
      if (!laneValue)
        return laneValue.takeError();
      laneOperands.push_back(*laneValue);
    }

    auto laneResult = evaluatePrimitiveOperation(descriptor, laneOperands);
    if (!laneResult)
      return llvm::joinErrors(
          llvm::createStringError(
              std::errc::invalid_argument, "%s failed for vector lane %u",
              dataflow::operationSchemaSpelling(descriptor.actor.schema)
                  .str()
                  .c_str(),
              lane),
          laneResult.takeError());
    auto laneToken =
        tokenFromPrimitiveValue(*laneResult, resultType.getElementType());
    if (!laneToken)
      return laneToken.takeError();
    auto laneBits = tokenBitPattern(*laneToken, resultType.getElementType());
    if (!laneBits)
      return laneBits.takeError();
    resultBits.insertBits(*laneBits, *resultElementWidth * lane);
  }
  return tokenFromBitPattern(resultBits, resultType);
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
  llvm::SmallVector<PrimitiveValue> operands;
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
  if (state.terminalPrimitiveOps.contains(op))
    return false;
  if (op->getNumOperands() == 0 && state.oneShotOps.contains(op))
    return false;
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    if (!hasToken(state, operand))
      return false;
  }

  llvm::SmallVector<Token> operands;
  operands.reserve(op->getNumOperands());
  for (mlir::OpOperand &operand : op->getOpOperands())
    operands.push_back(peekToken(state, operand));
  auto descriptor = state.primitiveDescriptors.find(op);
  assert(descriptor != state.primitiveDescriptors.end() &&
         "admitted primitive actor has no execution descriptor");
  auto resultToken =
      evaluatePrimitiveToken(op, descriptor->second, result, operands);
  if (!resultToken) {
    state.diagnostics.push_back(llvm::toString(resultToken.takeError()));
    state.terminalPrimitiveOps.insert(op);
    return false;
  }
  for (mlir::OpOperand &operand : op->getOpOperands())
    (void)popToken(state, operand);
  emitToken(state, result, *resultToken);
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
  emitToken(state, op.getResult(), *tokenOrErr);
  state.oneShotOps.insert(op.getOperation());
  return true;
}

using ActorProvider = bool (*)(mlir::Operation *,
                               const dataflow::CanonicalActorSchemaProjection &,
                               SimulatorState &);

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

static ActorProvider actorProvider(dataflow::OperationSchemaId schema) {
  if (isSupportedPrimitiveOperation(schema))
    return firePrimitiveActor;

  using Schema = dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::ArithConstant:
    return fireArithConstantActor;
  case Schema::DataflowStream:
    return fireStreamActor;
  case Schema::DataflowConstant:
    return fireDataflowConstantActor;
  case Schema::DataflowCarry:
    return fireTypedActor<dataflow::CarryOp, fireCarry>;
  case Schema::DataflowInvariant:
    return fireTypedActor<dataflow::InvariantOp, fireInvariant>;
  case Schema::DataflowGate:
    return fireTypedActor<dataflow::GateOp, fireGate>;
  case Schema::DataflowSync:
    return fireTypedActor<dataflow::SyncOp, fireSync>;
  case Schema::DataflowMux:
    return fireTypedActor<dataflow::MuxOp, fireMux>;
  case Schema::DataflowDemux:
    return fireTypedActor<dataflow::DemuxOp, fireDemux>;
  case Schema::DataflowParallelize:
    return fireTypedActor<dataflow::ParallelizeOp, fireParallelize>;
  case Schema::DataflowPack:
    return fireTypedActor<dataflow::PackOp, firePack>;
  case Schema::DataflowUnpack:
    return fireTypedActor<dataflow::UnpackOp, fireUnpack>;
  case Schema::DataflowSerialize:
    return fireTypedActor<dataflow::SerializeOp, fireSerialize>;
  case Schema::DataflowLoad:
    return fireTypedActor<dataflow::LoadOp, fireLoad>;
  case Schema::DataflowStore:
    return fireTypedActor<dataflow::StoreOp, fireStore>;
  default:
    return nullptr;
  }
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

bool fireActorOperation(mlir::Operation *op, SimulatorState &state) {
  const auto &projection = actorProjection(state, op);
  ActorProvider provider = actorProvider(projection.schema);
  assert(provider && "admitted actor has no simulator provider");
  if (!provider(op, projection, state))
    return false;
  return recordEvent(state, projection.schema);
}

std::optional<UnsupportedOperation> unsupportedActorProvider(
    mlir::Operation *op,
    const dataflow::CanonicalActorSchemaProjection &projection) {
  if (hasUnsupportedMemoryContract(projection))
    return UnsupportedOperation{
        unsupportedOperationLabel(op),
        "atomic, volatile, and fence memory contracts have no dynamic "
        "consistency-domain semantics"};

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

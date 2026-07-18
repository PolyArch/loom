#include "DFGSimulatorInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include <system_error>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

static unsigned streamIntegerBitWidth(mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return intType.getWidth();
  return 0;
}

static std::optional<bool> peekBoolToken(ChannelMap &channels,
                                         mlir::OpOperand &operand) {
  if (!hasToken(channels, operand))
    return std::nullopt;
  return boolToken(peekToken(channels, operand));
}

static bool fireStream(dataflow::StreamOp op, SimulatorState &state) {
  if (state.failedStreamOps.contains(op.getOperation()))
    return false;

  StreamSemanticState &stream = state.streamStates[op.getOperation()];
  std::optional<StreamActivation> activation;
  if (stream.mode == StreamMode::Idle &&
      hasToken(state.channels, op->getOpOperand(0)) &&
      hasToken(state.channels, op->getOpOperand(1)) &&
      hasToken(state.channels, op->getOpOperand(2))) {
    activation = StreamActivation{
        integerToken(peekToken(state.channels, op->getOpOperand(0))),
        integerToken(peekToken(state.channels, op->getOpOperand(1))),
        integerToken(peekToken(state.channels, op->getOpOperand(2)))};
  }

  auto transition = evaluateStreamTransition(
      stream,
      StreamSemanticConfig{op.getStepKind(), op.getPredicate(),
                           streamIntegerBitWidth(op.getInit().getType())},
      activation);
  if (!transition) {
    state.diagnostics.push_back(llvm::toString(transition.takeError()));
    state.failedStreamOps.insert(op.getOperation());
    return false;
  }
  if (!transition->firing.ready)
    return false;

  if (selectsSemanticInput(transition->firing.consumedInputs,
                           StreamInput::Init))
    (void)popToken(state, op->getOpOperand(0));
  if (selectsSemanticInput(transition->firing.consumedInputs,
                           StreamInput::Limit))
    (void)popToken(state, op->getOpOperand(1));
  if (selectsSemanticInput(transition->firing.consumedInputs,
                           StreamInput::Step))
    (void)popToken(state, op->getOpOperand(2));

  if (transition->emitIv) {
    emitToken(state, op.getIv(), integerValueToken(transition->iv));
    ++state.streamTrueEmissionCounts[op.getOperation()];
  }
  if (transition->emitPhase)
    emitToken(state, op.getPhase(), boolValueToken(transition->phase));
  stream = transition->nextState;
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireConstant(dataflow::ConstantOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op->getOpOperand(0)))
    return false;
  auto attr = mlir::dyn_cast<mlir::TypedAttr>(op.getConstValue());
  if (!attr) {
    state.diagnostics.push_back("dataflow.constant has untyped const_value");
    return false;
  }
  auto tokenOrErr = tokenFromTypedAttr(attr);
  if (!tokenOrErr) {
    state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
    return false;
  }
  popToken(state, op->getOpOperand(0));
  emitToken(state, op.getValue(), *tokenOrErr);
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireCarry(dataflow::CarryOp op, SimulatorState &state) {
  LoopState &carry = state.carryStates[op.getOperation()];
  auto transition = evaluateCarryTransition(
      carry.semanticState, peekBoolToken(state.channels, op->getOpOperand(0)),
      hasToken(state.channels, op->getOpOperand(1)),
      hasToken(state.channels, op->getOpOperand(2)));
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
  if (forwarded)
    emitToken(state, op.getOutput(), *forwarded);
  carry.semanticState = transition.nextState;
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireInvariant(dataflow::InvariantOp op, SimulatorState &state) {
  LoopState &invariant = state.invariantStates[op.getOperation()];
  auto transition = evaluateInvariantTransition(
      invariant.semanticState,
      peekBoolToken(state.channels, op->getOpOperand(0)),
      hasToken(state.channels, op->getOpOperand(1)));
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

  if (transition.output == InvariantOutputSource::InitInput)
    emitToken(state, op.getOutput(), *init);
  else if (transition.output == InvariantOutputSource::Latched)
    emitToken(state, op.getOutput(), *invariant.latched);
  if (transition.clearLatch)
    invariant.latched.reset();
  invariant.semanticState = transition.nextState;
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireGate(dataflow::GateOp op, SimulatorState &state) {
  const GateSemanticState gate =
      state.gateContinueStates.contains(op.getOperation())
          ? GateSemanticState::Open
          : GateSemanticState::Closed;
  auto transition = evaluateGateTransition(
      gate, peekBoolToken(state.channels, op.getBeforeCondMutable()),
      hasToken(state.channels, op.getBeforeValueMutable()));
  if (!transition.firing.ready)
    return false;

  if (selectsSemanticInput(transition.firing.consumedInputs, GateInput::Phase))
    (void)popToken(state, op.getBeforeCondMutable());
  std::optional<Token> value;
  if (selectsSemanticInput(transition.firing.consumedInputs, GateInput::Value))
    value = popToken(state, op.getBeforeValueMutable());
  if (transition.emitPhase)
    emitToken(state, op.getAfterCond(), boolValueToken(transition.phase));
  if (transition.forwardedInput == GateInput::Value)
    emitToken(state, op.getAfterValue(), *value);
  if (transition.nextState == GateSemanticState::Open)
    state.gateContinueStates.insert(op.getOperation());
  else
    state.gateContinueStates.erase(op.getOperation());
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireSync(dataflow::SyncOp op, SimulatorState &state) {
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    if (!hasToken(state.channels, operand))
      return false;
  }

  llvm::SmallVector<Token> consumed;
  consumed.reserve(op->getNumOperands());
  for (mlir::OpOperand &operand : op->getOpOperands())
    consumed.push_back(popToken(state, operand));

  for (auto [result, token] : llvm::zip_equal(op->getResults(), consumed))
    emitToken(state, result, token);
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireMux(dataflow::MuxOp op, SimulatorState &state) {
  mlir::OpOperand &selOperand = op->getOpOperand(0);
  if (!hasToken(state.channels, selOperand))
    return false;

  const Token &sel = state.channels[&selOperand].front();
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
  if (!hasToken(state.channels, selectedOperand))
    return false;

  (void)popToken(state, selOperand);
  Token value = popToken(state, selectedOperand);
  emitToken(state, op.getOutput(), value);
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireDemux(dataflow::DemuxOp op, SimulatorState &state) {
  mlir::OpOperand &selOperand = op->getOpOperand(0);
  mlir::OpOperand &inputOperand = op->getOpOperand(1);
  if (!hasToken(state.channels, selOperand) ||
      !hasToken(state.channels, inputOperand))
    return false;

  const Token &sel = state.channels[&selOperand].front();
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
  return recordEvent(state, op->getName().getStringRef());
}

struct ParallelizeGroup {
  Token vector;
  Token mask;
};

static llvm::Expected<ParallelizeGroup>
buildParallelizeGroup(dataflow::ParallelizeOp op,
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
  }

  auto vectorToken = tokenFromBitPattern(vectorBits, vectorType);
  if (!vectorToken)
    return vectorToken.takeError();
  auto maskToken = tokenFromBitPattern(maskBits, op.getMask().getType());
  if (!maskToken)
    return maskToken.takeError();
  return ParallelizeGroup{*vectorToken, *maskToken};
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
      peekBoolToken(state.channels, op.getScalarPhaseMutable()),
      hasToken(state.channels, op.getDataMutable()));
  if (!transition.firing.ready)
    return false;

  std::optional<ParallelizeGroup> group;
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           ParallelizeInput::Data)) {
    const Token data = peekToken(state.channels, op.getDataMutable());
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
    auto groupOrErr = buildParallelizeGroup(op, next, transition.activeItems);
    if (!groupOrErr) {
      state.diagnostics.push_back(llvm::toString(groupOrErr.takeError()));
      return false;
    }
    group = *groupOrErr;
    next.slots.assign(vectorLength, std::nullopt);
  }
  next.semanticState = transition.nextState;

  if (selectsSemanticInput(transition.firing.consumedInputs,
                           ParallelizeInput::Phase))
    (void)popToken(state, op.getScalarPhaseMutable());
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           ParallelizeInput::Data))
    (void)popToken(state, op.getDataMutable());
  state.parallelizeStates[op.getOperation()] = std::move(next);
  if (group) {
    emitToken(state, op.getVector(), group->vector);
    emitToken(state, op.getMask(), group->mask);
  }
  if (transition.emitTruePhase)
    emitToken(state, op.getGroupPhase(), boolValueToken(true));
  if (transition.emitFalsePhase)
    emitToken(state, op.getGroupPhase(), boolValueToken(false));
  return recordEvent(state, op->getName().getStringRef());
}

static bool firePack(dataflow::PackOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getVectorMutable()))
    return false;
  Token vector = peekToken(state.channels, op.getVectorMutable());
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
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireUnpack(dataflow::UnpackOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getPackedMutable()))
    return false;
  Token packedToken = peekToken(state.channels, op.getPackedMutable());
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
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireSerialize(dataflow::SerializeOp op, SimulatorState &state) {
  auto transition = evaluateSerializeTransition(
      peekBoolToken(state.channels, op.getGroupPhaseMutable()),
      hasToken(state.channels, op.getVectorMutable()),
      hasToken(state.channels, op.getMaskMutable()));
  if (!transition.firing.ready)
    return false;

  llvm::SmallVector<Token> activeLanes;
  if (transition.emitActiveItems) {
    Token vectorToken = peekToken(state.channels, op.getVectorMutable());
    Token maskToken = peekToken(state.channels, op.getMaskMutable());
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
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireCast(mlir::UnrealizedConversionCastOp op,
                     SimulatorState &state) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  mlir::OpOperand &operand = op->getOpOperand(0);
  if (!hasToken(state.channels, operand))
    return false;
  Token token = popToken(state, operand);
  if (auto memrefType =
          mlir::dyn_cast<mlir::MemRefType>(op.getResult(0).getType())) {
    auto tokenOrErr =
        ensurePointerMemory(state, token, memrefType.getElementType());
    if (!tokenOrErr) {
      state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
      return false;
    }
    emitToken(state, op.getResult(0), *tokenOrErr);
    return true;
  }
  emitToken(state, op.getResult(0), token);
  return true;
}

static bool fireGEP(mlir::LLVM::GEPOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getBaseMutable()))
    return false;
  for (unsigned i = 1, e = op->getNumOperands(); i < e; ++i) {
    if (!hasToken(state.channels, op->getOpOperand(i)))
      return false;
  }
  Token base = popToken(state, op.getBaseMutable());
  if (base.kind != TokenKind::Pointer) {
    state.diagnostics.push_back("llvm.getelementptr base is not a pointer");
    return false;
  }
  llvm::SmallVector<Token> dynamicTokens;
  for (unsigned i = 1, e = op->getNumOperands(); i < e; ++i)
    dynamicTokens.push_back(popToken(state, op->getOpOperand(i)));
  auto offsetOrErr = gepByteOffset(op, dynamicTokens);
  if (!offsetOrErr) {
    state.diagnostics.push_back(llvm::toString(offsetOrErr.takeError()));
    return false;
  }
  base.pointer.byteOffset += *offsetOrErr;
  emitToken(state, op.getResult(), base);
  return recordEvent(state, op->getName().getStringRef());
}

static std::optional<MemoryView>
resolveMemoryView(SimulatorState &state, mlir::Value mem,
                  mlir::OpOperand &memOperand) {
  if (hasToken(state.channels, memOperand)) {
    Token token = popToken(state, memOperand);
    if (token.kind != TokenKind::Pointer || !token.pointer.memory) {
      state.diagnostics.push_back(
          "dataflow memory operand is not a memory view");
      return std::nullopt;
    }
    return token.pointer;
  }
  auto memIt = state.memories.find(mem);
  if (memIt != state.memories.end())
    return MemoryView{memIt->second, mem, 0};
  return std::nullopt;
}

std::optional<std::size_t> resolveElementIndex(const MemoryView &view,
                                               const Token &addr,
                                               SimulatorState &state,
                                               mlir::Operation *scope,
                                               llvm::StringRef opName) {
  auto elementSizeOrErr = byteSizeOfType(view.memory->elementType, scope);
  if (!elementSizeOrErr) {
    state.diagnostics.push_back(llvm::toString(elementSizeOrErr.takeError()));
    return std::nullopt;
  }
  if (*elementSizeOrErr == 0 || view.byteOffset % *elementSizeOrErr != 0) {
    state.diagnostics.push_back(
        "memory view byte offset is not element-aligned");
    return std::nullopt;
  }
  const std::int64_t baseIndex = view.byteOffset / *elementSizeOrErr;
  const std::int64_t index = baseIndex + integerToken(addr);
  if (index < 0 ||
      static_cast<std::size_t>(index) >= view.memory->elements.size()) {
    state.diagnostics.push_back((opName + " address is out of range").str());
    return std::nullopt;
  }
  return static_cast<std::size_t>(index);
}

std::optional<Token> readMemoryElement(const MemoryView &view,
                                       std::size_t index, SimulatorState &state,
                                       llvm::StringRef opName) {
  if (!view.memory->initialized[index]) {
    state.diagnostics.push_back((opName + " reads uninitialized memory").str());
    return std::nullopt;
  }
  return view.memory->elements[index];
}

void writeMemoryElement(const MemoryView &view, std::size_t index,
                        Token value) {
  view.memory->elements[index] = value;
  view.memory->initialized.set(index);
}

static bool fireLoad(dataflow::LoadOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getAddrMutable()) ||
      !hasToken(state.channels, op.getCtrlMutable()))
    return false;
  std::optional<MemoryView> view =
      resolveMemoryView(state, op.getMem(), op.getMemMutable());
  if (!view)
    return false;
  Token addr = popToken(state, op.getAddrMutable());
  popToken(state, op.getCtrlMutable());
  std::optional<std::size_t> index = resolveElementIndex(
      *view, addr, state, op.getOperation(), "dataflow.load");
  if (!index)
    return false;
  std::optional<Token> value =
      readMemoryElement(*view, *index, state, "dataflow.load");
  if (!value)
    return false;
  emitToken(state, op.getData(), *value);
  emitToken(state, op.getDone(), noneToken());
  if (hasComputedAddress(op.getAddr()))
    state.memoryAddressScore += kLoadAddressScore;
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireLLVMLoad(mlir::LLVM::LoadOp op, SimulatorState &state) {
  mlir::OpOperand &addrOperand = op->getOpOperand(0);
  if (!hasToken(state.channels, addrOperand))
    return false;
  Token ptr = popToken(state, addrOperand);
  auto viewOrErr = ensurePointerMemory(state, ptr, op->getResult(0).getType());
  if (!viewOrErr) {
    state.diagnostics.push_back(llvm::toString(viewOrErr.takeError()));
    return false;
  }
  std::optional<std::size_t> index =
      resolveElementIndex(viewOrErr->pointer, integerValueToken(0), state,
                          op.getOperation(), "llvm.load");
  if (!index)
    return false;
  std::optional<Token> value =
      readMemoryElement(viewOrErr->pointer, *index, state, "llvm.load");
  if (!value)
    return false;
  emitToken(state, op->getResult(0), *value);
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireLLVMStore(mlir::LLVM::StoreOp op, SimulatorState &state) {
  mlir::OpOperand &valueOperand = op->getOpOperand(0);
  mlir::OpOperand &addrOperand = op->getOpOperand(1);
  if (!hasToken(state.channels, valueOperand) ||
      !hasToken(state.channels, addrOperand))
    return false;
  Token value = popToken(state, valueOperand);
  Token ptr = popToken(state, addrOperand);
  auto viewOrErr = ensurePointerMemory(state, ptr, op->getOperand(0).getType());
  if (!viewOrErr) {
    state.diagnostics.push_back(llvm::toString(viewOrErr.takeError()));
    return false;
  }
  std::optional<std::size_t> index =
      resolveElementIndex(viewOrErr->pointer, integerValueToken(0), state,
                          op.getOperation(), "llvm.store");
  if (!index)
    return false;
  writeMemoryElement(viewOrErr->pointer, *index, value);
  return recordEvent(state, op->getName().getStringRef());
}

static std::optional<std::size_t>
resolveByteRangeStart(const MemoryView &view, std::int64_t byteLength,
                      SimulatorState &state, mlir::Operation *scope,
                      llvm::StringRef opName, llvm::StringRef role) {
  if (byteLength < 0) {
    state.diagnostics.push_back((opName + " length is negative").str());
    return std::nullopt;
  }
  if (view.byteOffset < 0) {
    state.diagnostics.push_back(
        (opName + " " + role + " byte offset is negative").str());
    return std::nullopt;
  }
  auto elementSizeOrErr = byteSizeOfType(view.memory->elementType, scope);
  if (!elementSizeOrErr) {
    state.diagnostics.push_back(llvm::toString(elementSizeOrErr.takeError()));
    return std::nullopt;
  }
  if (*elementSizeOrErr != 1) {
    state.diagnostics.push_back(
        (opName + " requires byte-addressable i8 memory fixtures").str());
    return std::nullopt;
  }
  const std::uint64_t start = static_cast<std::uint64_t>(view.byteOffset);
  const std::uint64_t length = static_cast<std::uint64_t>(byteLength);
  const std::uint64_t size = view.memory->elements.size();
  if (start > size || length > size - start) {
    state.diagnostics.push_back(
        (opName + " " + role + " range is out of range").str());
    return std::nullopt;
  }
  return static_cast<std::size_t>(start);
}

bool executeLLVMMemcpy(mlir::LLVM::MemcpyOp op, SimulatorState &state,
                       const Token &dst, const Token &src, const Token &len) {
  if (op.getIsVolatile()) {
    state.diagnostics.push_back("volatile llvm.intr.memcpy is unsupported");
    return false;
  }

  if (len.kind != TokenKind::Integer && len.kind != TokenKind::Bool) {
    state.diagnostics.push_back("llvm.intr.memcpy length is not integer-like");
    return false;
  }

  mlir::Type byteType = mlir::IntegerType::get(op.getContext(), 8);
  auto dstOrErr = ensurePointerMemory(state, dst, byteType);
  if (!dstOrErr) {
    state.diagnostics.push_back(llvm::toString(dstOrErr.takeError()));
    return false;
  }
  auto srcOrErr = ensurePointerMemory(state, src, byteType);
  if (!srcOrErr) {
    state.diagnostics.push_back(llvm::toString(srcOrErr.takeError()));
    return false;
  }

  const std::int64_t byteLength = integerToken(len);
  std::optional<std::size_t> dstStart = resolveByteRangeStart(
      dstOrErr->pointer, byteLength, state, op.getOperation(),
      "llvm.intr.memcpy", "destination");
  std::optional<std::size_t> srcStart =
      resolveByteRangeStart(srcOrErr->pointer, byteLength, state,
                            op.getOperation(), "llvm.intr.memcpy", "source");
  if (!dstStart || !srcStart)
    return false;

  if (dstOrErr->pointer.memory == srcOrErr->pointer.memory) {
    std::size_t length = static_cast<std::size_t>(byteLength);
    bool overlaps =
        *dstStart < *srcStart + length && *srcStart < *dstStart + length;
    if (overlaps && dstStart != srcStart) {
      state.diagnostics.push_back(
          "llvm.intr.memcpy overlapping ranges are unsupported");
      return false;
    }
  }

  llvm::SmallVector<Token> copied;
  copied.reserve(static_cast<std::size_t>(byteLength));
  for (std::int64_t i = 0; i < byteLength; ++i) {
    std::optional<Token> value = readMemoryElement(
        srcOrErr->pointer, *srcStart + i, state, "llvm.intr.memcpy");
    if (!value)
      return false;
    copied.push_back(*value);
  }
  for (auto [offset, token] : llvm::enumerate(copied))
    writeMemoryElement(dstOrErr->pointer, *dstStart + offset, token);
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireLLVMMemcpy(mlir::LLVM::MemcpyOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getDstMutable()) ||
      !hasToken(state.channels, op.getSrcMutable()) ||
      !hasToken(state.channels, op.getLenMutable()))
    return false;

  Token dst = popToken(state, op.getDstMutable());
  Token src = popToken(state, op.getSrcMutable());
  Token len = popToken(state, op.getLenMutable());
  return executeLLVMMemcpy(op, state, dst, src, len);
}

static bool fireLLVMCall(mlir::LLVM::CallOp op, SimulatorState &state) {
  if (!isSupportedLLVMCall(op))
    return false;
  llvm::SmallVector<Token> operands;
  operands.reserve(op->getNumOperands());
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    if (!hasToken(state.channels, operand))
      return false;
  }
  for (mlir::OpOperand &operand : op->getOpOperands())
    operands.push_back(popToken(state, operand));

  Token result;
  if (!executeCmsisNNVecMatMultTS8(op, state, operands, result))
    return false;
  emitToken(state, op->getResult(0), result);
  return true;
}

static bool fireStore(dataflow::StoreOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getAddrMutable()) ||
      !hasToken(state.channels, op.getDataMutable()) ||
      !hasToken(state.channels, op.getCtrlMutable()))
    return false;
  std::optional<MemoryView> view =
      resolveMemoryView(state, op.getMem(), op.getMemMutable());
  if (!view)
    return false;
  Token addr = popToken(state, op.getAddrMutable());
  Token data = popToken(state, op.getDataMutable());
  popToken(state, op.getCtrlMutable());
  std::optional<std::size_t> index = resolveElementIndex(
      *view, addr, state, op.getOperation(), "dataflow.store");
  if (!index)
    return false;
  writeMemoryElement(*view, *index, data);
  emitToken(state, op.getDone(), noneToken());
  if (hasComputedAddress(op.getAddr()))
    state.memoryAddressScore += kStoreAddressScore;
  return recordEvent(state, op->getName().getStringRef());
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
  if (resultType.getRank() != 1 || resultType.isScalable())
    return llvm::createStringError(
        std::errc::not_supported,
        "vector primitive result must be fixed-size and rank-1");
  if (llvm::Error error =
          validatePrimitiveElementType(resultType.getElementType(), "result"))
    return std::move(error);
  if (op->getNumOperands() == 0)
    return llvm::createStringError(std::errc::not_supported,
                                   "vector primitive has no operands");

  for (mlir::Type type : op->getOperandTypes()) {
    auto vectorType = mlir::dyn_cast<mlir::VectorType>(type);
    if (!vectorType || vectorType.getRank() != 1 || vectorType.isScalable())
      return llvm::createStringError(
          std::errc::not_supported,
          "vector primitive operands must be fixed-size and rank-1");
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

llvm::Error validatePrimitiveTokenTypes(mlir::Operation *op,
                                        mlir::Value result) {
  if (!hasVectorPrimitiveType(op))
    return llvm::Error::success();
  auto vectorType = validateElementwiseVectorPrimitive(op, result);
  if (!vectorType)
    return vectorType.takeError();
  return llvm::Error::success();
}

static llvm::Expected<Token>
evaluateElementwiseVectorPrimitive(mlir::Operation *op, mlir::Value result,
                                   llvm::ArrayRef<Token> inputTokens) {
  auto resultTypeOrErr = validateElementwiseVectorPrimitive(op, result);
  if (!resultTypeOrErr)
    return resultTypeOrErr.takeError();
  mlir::VectorType resultType = *resultTypeOrErr;
  std::string predicate = primitivePredicate(op);
  auto firstOperandType =
      mlir::cast<mlir::VectorType>(op->getOperand(0).getType());
  auto descriptor =
      primitiveDescriptor(op, predicate, resultType.getElementType(),
                          firstOperandType.getElementType());
  if (!descriptor)
    return descriptor.takeError();

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
  for (unsigned lane = 0; lane < resultType.getShape().front(); ++lane) {
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
      auto laneValue =
          primitiveValueFromToken(*laneToken, vectorType.getElementType());
      if (!laneValue)
        return laneValue.takeError();
      laneOperands.push_back(*laneValue);
    }

    auto laneResult = evaluatePrimitiveOperation(*descriptor, laneOperands);
    if (!laneResult)
      return llvm::joinErrors(
          llvm::createStringError(std::errc::invalid_argument,
                                  "%s failed for vector lane %u",
                                  descriptor->name.c_str(), lane),
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
evaluatePrimitiveToken(mlir::Operation *op, mlir::Value result,
                       llvm::ArrayRef<Token> inputTokens) {
  if (inputTokens.size() != op->getNumOperands())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "primitive token count does not match operation operands");
  if (hasVectorPrimitiveType(op))
    return evaluateElementwiseVectorPrimitive(op, result, inputTokens);

  std::string predicate = primitivePredicate(op);
  auto descriptor = primitiveDescriptor(op, predicate, result);
  if (!descriptor)
    return descriptor.takeError();
  llvm::SmallVector<PrimitiveValue> operands;
  operands.reserve(inputTokens.size());
  for (auto [operand, token] :
       llvm::zip_equal(op->getOpOperands(), inputTokens)) {
    auto value = primitiveValueFromToken(token, operand.get().getType());
    if (!value)
      return value.takeError();
    operands.push_back(*value);
  }
  auto value = evaluatePrimitiveOperation(*descriptor, operands);
  if (!value)
    return value.takeError();
  return tokenFromPrimitiveValue(*value, result.getType());
}

static bool firePrimitiveOperation(mlir::Operation *op, mlir::Value result,
                                   SimulatorState &state) {
  if (state.terminalPrimitiveOps.contains(op))
    return false;
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    if (!hasToken(state.channels, operand))
      return false;
  }

  llvm::SmallVector<Token> operands;
  operands.reserve(op->getNumOperands());
  for (mlir::OpOperand &operand : op->getOpOperands())
    operands.push_back(peekToken(state.channels, operand));
  auto resultToken = evaluatePrimitiveToken(op, result, operands);
  if (!resultToken) {
    state.diagnostics.push_back(llvm::toString(resultToken.takeError()));
    state.terminalPrimitiveOps.insert(op);
    return false;
  }
  for (mlir::OpOperand &operand : op->getOpOperands())
    (void)popToken(state, operand);
  emitToken(state, result, *resultToken);
  return recordEvent(state, primitiveOperationName(op));
}

static bool fireArithConstant(mlir::arith::ConstantOp op,
                              SimulatorState &state) {
  if (state.oneShotOps.contains(op.getOperation()))
    return false;
  auto attr = mlir::dyn_cast<mlir::TypedAttr>(op.getValue());
  if (!attr) {
    state.diagnostics.push_back("arith.constant has untyped value");
    return false;
  }
  auto tokenOrErr = tokenFromTypedAttr(attr);
  if (!tokenOrErr) {
    state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
    return false;
  }
  emitToken(state, op.getResult(), *tokenOrErr);
  state.oneShotOps.insert(op.getOperation());
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireLLVMZero(mlir::LLVM::ZeroOp op, SimulatorState &state) {
  if (state.oneShotOps.contains(op.getOperation()))
    return false;
  auto tokenOrErr = zeroToken(op->getResult(0).getType());
  if (!tokenOrErr) {
    state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
    return false;
  }
  emitToken(state, op->getResult(0), *tokenOrErr);
  state.oneShotOps.insert(op.getOperation());
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireLLVMAddressOf(mlir::LLVM::AddressOfOp op,
                              SimulatorState &state) {
  if (state.oneShotOps.contains(op.getOperation()))
    return false;
  mlir::Value result = op->getResult(0);
  emitToken(state, result, pointerToken(result));
  state.oneShotOps.insert(op.getOperation());
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireUBPoison(mlir::ub::PoisonOp op, SimulatorState &state) {
  if (state.oneShotOps.contains(op.getOperation()))
    return false;
  auto tokenOrErr = zeroToken(op->getResult(0).getType());
  if (!tokenOrErr) {
    state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
    return false;
  }
  emitToken(state, op->getResult(0), *tokenOrErr);
  state.oneShotOps.insert(op.getOperation());
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireLLVMICmp(mlir::LLVM::ICmpOp op, SimulatorState &state) {
  mlir::OpOperand &lhsOperand = op->getOpOperand(0);
  mlir::OpOperand &rhsOperand = op->getOpOperand(1);
  if (!hasToken(state.channels, lhsOperand) ||
      !hasToken(state.channels, rhsOperand))
    return false;
  Token lhs = popToken(state, lhsOperand);
  Token rhs = popToken(state, rhsOperand);
  auto resultOrErr = evaluatePointerICmp(op, lhs, rhs);
  if (!resultOrErr) {
    state.diagnostics.push_back(llvm::toString(resultOrErr.takeError()));
    return false;
  }
  emitToken(state, op->getResult(0), *resultOrErr);
  return recordEvent(state, op->getName().getStringRef());
}

bool isPointerSelect(mlir::LLVM::SelectOp op) {
  return op->getNumOperands() == 3 && op->getNumResults() == 1 &&
         mlir::isa<mlir::LLVM::LLVMPointerType>(op->getResult(0).getType());
}

std::optional<Token> evaluatePointerSelect(mlir::LLVM::SelectOp op,
                                           const Token &condition,
                                           const Token &trueValue,
                                           const Token &falseValue,
                                           SimulatorState &state) {
  if (!isPointerSelect(op))
    return std::nullopt;
  if (trueValue.kind != TokenKind::Pointer ||
      falseValue.kind != TokenKind::Pointer) {
    state.diagnostics.push_back(
        "llvm.select pointer operands are not pointers");
    return std::nullopt;
  }
  return boolToken(condition) ? trueValue : falseValue;
}

static bool fireLLVMSelect(mlir::LLVM::SelectOp op, SimulatorState &state) {
  if (!isPointerSelect(op))
    return firePrimitiveOperation(op.getOperation(), op->getResult(0), state);
  mlir::OpOperand &conditionOperand = op->getOpOperand(0);
  mlir::OpOperand &trueOperand = op->getOpOperand(1);
  mlir::OpOperand &falseOperand = op->getOpOperand(2);
  if (!hasToken(state.channels, conditionOperand) ||
      !hasToken(state.channels, trueOperand) ||
      !hasToken(state.channels, falseOperand))
    return false;
  Token condition = popToken(state, conditionOperand);
  Token trueValue = popToken(state, trueOperand);
  Token falseValue = popToken(state, falseOperand);
  std::optional<Token> selected =
      evaluatePointerSelect(op, condition, trueValue, falseValue, state);
  if (!selected)
    return false;
  emitToken(state, op->getResult(0), *selected);
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireGenericPrimitive(mlir::Operation *op, SimulatorState &state) {
  if (op->getNumResults() != 1)
    return false;
  if (!isSupportedPrimitiveOperation(primitiveOperationName(op)))
    return false;
  return firePrimitiveOperation(op, op->getResult(0), state);
}

bool fireActorOperation(mlir::Operation *op, SimulatorState &state) {
  return llvm::TypeSwitch<mlir::Operation *, bool>(op)
      .Case<dataflow::StreamOp>(
          [&](auto typedOp) { return fireStream(typedOp, state); })
      .Case<dataflow::ConstantOp>(
          [&](auto typedOp) { return fireConstant(typedOp, state); })
      .Case<dataflow::CarryOp>(
          [&](auto typedOp) { return fireCarry(typedOp, state); })
      .Case<dataflow::InvariantOp>(
          [&](auto typedOp) { return fireInvariant(typedOp, state); })
      .Case<dataflow::GateOp>(
          [&](auto typedOp) { return fireGate(typedOp, state); })
      .Case<dataflow::SyncOp>(
          [&](auto typedOp) { return fireSync(typedOp, state); })
      .Case<dataflow::MuxOp>(
          [&](auto typedOp) { return fireMux(typedOp, state); })
      .Case<dataflow::DemuxOp>(
          [&](auto typedOp) { return fireDemux(typedOp, state); })
      .Case<dataflow::ParallelizeOp>(
          [&](auto typedOp) { return fireParallelize(typedOp, state); })
      .Case<dataflow::PackOp>(
          [&](auto typedOp) { return firePack(typedOp, state); })
      .Case<dataflow::UnpackOp>(
          [&](auto typedOp) { return fireUnpack(typedOp, state); })
      .Case<dataflow::SerializeOp>(
          [&](auto typedOp) { return fireSerialize(typedOp, state); })
      .Case<dataflow::LoadOp>(
          [&](auto typedOp) { return fireLoad(typedOp, state); })
      .Case<dataflow::StoreOp>(
          [&](auto typedOp) { return fireStore(typedOp, state); })
      .Case<mlir::UnrealizedConversionCastOp>(
          [&](auto typedOp) { return fireCast(typedOp, state); })
      .Case<mlir::LLVM::GEPOp>(
          [&](auto typedOp) { return fireGEP(typedOp, state); })
      .Case<mlir::LLVM::AddressOfOp>(
          [&](auto typedOp) { return fireLLVMAddressOf(typedOp, state); })
      .Case<mlir::LLVM::ZeroOp>(
          [&](auto typedOp) { return fireLLVMZero(typedOp, state); })
      .Case<mlir::ub::PoisonOp>(
          [&](auto typedOp) { return fireUBPoison(typedOp, state); })
      .Case<mlir::LLVM::ICmpOp>(
          [&](auto typedOp) { return fireLLVMICmp(typedOp, state); })
      .Case<mlir::LLVM::SelectOp>(
          [&](auto typedOp) { return fireLLVMSelect(typedOp, state); })
      .Case<mlir::LLVM::LoadOp>(
          [&](auto typedOp) { return fireLLVMLoad(typedOp, state); })
      .Case<mlir::LLVM::StoreOp>(
          [&](auto typedOp) { return fireLLVMStore(typedOp, state); })
      .Case<mlir::LLVM::MemcpyOp>(
          [&](auto typedOp) { return fireLLVMMemcpy(typedOp, state); })
      .Case<mlir::LLVM::CallOp>(
          [&](auto typedOp) { return fireLLVMCall(typedOp, state); })
      .Case<mlir::arith::ConstantOp>(
          [&](auto typedOp) { return fireArithConstant(typedOp, state); })
      .Default([&](mlir::Operation *genericOp) {
        return fireGenericPrimitive(genericOp, state);
      });
}

std::optional<UnsupportedOperation>
unsupportedActorOperation(mlir::Operation *op) {
  if (op->getNumResults() == 1 &&
      isSupportedPrimitiveOperation(primitiveOperationName(op))) {
    if (llvm::Error error = validatePrimitiveTokenTypes(op, op->getResult(0))) {
      return UnsupportedOperation{unsupportedOperationLabel(op),
                                  llvm::toString(std::move(error))};
    }
    return std::nullopt;
  }
  if (auto icmp = mlir::dyn_cast<mlir::LLVM::ICmpOp>(op)) {
    if (isSupportedPointerICmp(icmp))
      return std::nullopt;
    return UnsupportedOperation{unsupportedOperationLabel(op), ""};
  }
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    if (isSupportedLLVMCall(call))
      return std::nullopt;
    return UnsupportedOperation{unsupportedOperationLabel(op), ""};
  }
  if (mlir::isa<dataflow::StreamOp, dataflow::ConstantOp, dataflow::CarryOp,
                dataflow::InvariantOp, dataflow::GateOp, dataflow::SyncOp,
                dataflow::MuxOp, dataflow::DemuxOp, dataflow::ParallelizeOp,
                dataflow::PackOp, dataflow::UnpackOp, dataflow::SerializeOp,
                dataflow::LoadOp, dataflow::StoreOp,
                mlir::UnrealizedConversionCastOp, mlir::LLVM::AddressOfOp,
                mlir::LLVM::GEPOp, mlir::LLVM::ZeroOp, mlir::LLVM::LoadOp,
                mlir::LLVM::StoreOp, mlir::LLVM::MemcpyOp,
                mlir::arith::ConstantOp, mlir::ub::PoisonOp>(op))
    return std::nullopt;
  return UnsupportedOperation{unsupportedOperationLabel(op), ""};
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

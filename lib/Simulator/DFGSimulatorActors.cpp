#include "DFGSimulatorInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "llvm/ADT/TypeSwitch.h"

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

static std::optional<unsigned> vectorSizeAttr(mlir::Operation *op,
                                              SimulatorState &state) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>("vec_size");
  if (!attr) {
    state.diagnostics.push_back(
        (op->getName().getStringRef() + " missing vec_size attribute").str());
    return std::nullopt;
  }
  int64_t value = attr.getInt();
  if (value < 1 || value > 64 || (value & (value - 1)) != 0) {
    state.diagnostics.push_back(
        (op->getName().getStringRef() + " has invalid vec_size").str());
    return std::nullopt;
  }
  return static_cast<unsigned>(value);
}

static std::optional<unsigned>
signlessIntegerBitWidthForVector(mlir::Type type, SimulatorState &state,
                                 llvm::StringRef op) {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType || !intType.isSignless()) {
    state.diagnostics.push_back(
        (op + " requires signless integer lanes").str());
    return std::nullopt;
  }
  return intType.getWidth();
}

static std::uint64_t lowBitsMask(unsigned width) {
  if (width >= 64)
    return ~std::uint64_t{0};
  return (std::uint64_t{1} << width) - 1;
}

static std::uint64_t tokenBits(const Token &token, unsigned width) {
  return static_cast<std::uint64_t>(integerToken(token)) & lowBitsMask(width);
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
    (void)popToken(state.channels, op->getOpOperand(0));
  if (selectsSemanticInput(transition->firing.consumedInputs,
                           StreamInput::Limit))
    (void)popToken(state.channels, op->getOpOperand(1));
  if (selectsSemanticInput(transition->firing.consumedInputs,
                           StreamInput::Step))
    (void)popToken(state.channels, op->getOpOperand(2));

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
  popToken(state.channels, op->getOpOperand(0));
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
    (void)popToken(state.channels, op->getOpOperand(0));
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           CarryInput::Init)) {
    Token value = popToken(state.channels, op->getOpOperand(1));
    if (transition.forwardedInput == CarryInput::Init)
      forwarded = value;
  }
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           CarryInput::Next)) {
    Token value = popToken(state.channels, op->getOpOperand(2));
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
    (void)popToken(state.channels, op->getOpOperand(0));
  std::optional<Token> init;
  if (selectsSemanticInput(transition.firing.consumedInputs,
                           InvariantInput::Init))
    init = popToken(state.channels, op->getOpOperand(1));
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
    (void)popToken(state.channels, op.getBeforeCondMutable());
  std::optional<Token> value;
  if (selectsSemanticInput(transition.firing.consumedInputs, GateInput::Value))
    value = popToken(state.channels, op.getBeforeValueMutable());
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
  llvm::SmallVector<Token> tokens;
  for (mlir::OpOperand &operand : op->getOpOperands())
    tokens.push_back(popToken(state.channels, operand));
  for (auto [result, token] : llvm::zip(op->getResults(), tokens))
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
    (void)popToken(state.channels, selOperand);
    state.diagnostics.push_back("dataflow.mux selector is out of range");
    return false;
  }

  mlir::OpOperand &selectedOperand =
      op->getOpOperand(static_cast<unsigned>(lane) + 1);
  if (!hasToken(state.channels, selectedOperand))
    return false;

  (void)popToken(state.channels, selOperand);
  Token value = popToken(state.channels, selectedOperand);
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
    (void)popToken(state.channels, selOperand);
    (void)popToken(state.channels, inputOperand);
    state.diagnostics.push_back("dataflow.demux selector is out of range");
    return false;
  }

  (void)popToken(state.channels, selOperand);
  Token value = popToken(state.channels, inputOperand);
  emitToken(state, op.getOutputs()[static_cast<unsigned>(lane)], value);
  return recordEvent(state, op->getName().getStringRef());
}

static void emitParallelizeGroup(dataflow::ParallelizeOp op,
                                 SimulatorState &state,
                                 ParallelizeState &parallel) {
  for (auto [i, output] : llvm::enumerate(op.getOutputs())) {
    if (i < parallel.slots.size() && parallel.slots[i])
      emitToken(state, output, *parallel.slots[i]);
  }
  emitToken(state, op.getMask(), integerValueToken(parallel.mask));
  parallel.slots.assign(op.getOutputs().size(), std::nullopt);
  parallel.mask = 0;
}

static bool fireParallelize(dataflow::ParallelizeOp op, SimulatorState &state) {
  std::optional<unsigned> vecSize = vectorSizeAttr(op.getOperation(), state);
  if (!vecSize)
    return false;
  ParallelizeState &parallel = state.parallelizeStates[op.getOperation()];
  if (parallel.slots.size() != *vecSize)
    parallel.slots.assign(*vecSize, std::nullopt);

  if (!hasToken(state.channels, op.getContMutable()))
    return false;
  const Token &contToken = peekToken(state.channels, op.getContMutable());
  const bool cont = boolToken(contToken);
  if (!hasToken(state.channels, op.getDataMutable()))
    return false;
  if (cont && op.getStride() && !hasToken(state.channels, op->getOpOperand(2)))
    return false;

  (void)popToken(state.channels, op.getContMutable());
  Token data = popToken(state.channels, op.getDataMutable());
  if (!cont) {
    if (parallel.mask != 0)
      emitParallelizeGroup(op, state, parallel);
    parallel.pointer = 0;
    return recordEvent(state, op->getName().getStringRef());
  }

  std::uint64_t stride = 1;
  if (op.getStride()) {
    Token strideToken = popToken(state.channels, op->getOpOperand(2));
    std::int64_t strideValue = integerToken(strideToken);
    if (strideValue <= 0) {
      state.diagnostics.push_back(
          "dataflow.parallelize stride must be positive");
      return false;
    }
    stride = static_cast<std::uint64_t>(strideValue);
  }

  const std::uint64_t slot = parallel.pointer % *vecSize;
  parallel.slots[slot] = data;
  parallel.mask |= std::uint64_t{1} << slot;
  parallel.pointer += stride;
  if (parallel.pointer >= *vecSize) {
    emitParallelizeGroup(op, state, parallel);
    parallel.pointer %= *vecSize;
  }
  return recordEvent(state, op->getName().getStringRef());
}

static bool firePack(dataflow::PackOp op, SimulatorState &state) {
  std::optional<unsigned> vecSize = vectorSizeAttr(op.getOperation(), state);
  if (!vecSize)
    return false;
  if (!hasToken(state.channels, op.getMaskMutable()))
    return false;
  const Token &maskToken = peekToken(state.channels, op.getMaskMutable());
  const std::uint64_t mask = tokenBits(maskToken, *vecSize);
  for (unsigned i = 0; i < *vecSize; ++i) {
    if ((mask & (std::uint64_t{1} << i)) == 0)
      continue;
    if (!hasToken(state.channels, op->getOpOperand(i)))
      return false;
  }
  std::optional<unsigned> laneWidth = signlessIntegerBitWidthForVector(
      op.getInputs().front().getType(), state, "dataflow.pack");
  if (!laneWidth)
    return false;
  if ((*laneWidth) * (*vecSize) > 64) {
    state.diagnostics.push_back(
        "dataflow.pack DFG-sim supports packed widths up to 64 bits");
    return false;
  }

  (void)popToken(state.channels, op.getMaskMutable());
  std::uint64_t packed = 0;
  for (unsigned i = 0; i < *vecSize; ++i) {
    if ((mask & (std::uint64_t{1} << i)) == 0)
      continue;
    Token lane = popToken(state.channels, op->getOpOperand(i));
    packed |= tokenBits(lane, *laneWidth) << ((*laneWidth) * i);
  }
  emitToken(state, op.getPacked(),
            integerValueToken(static_cast<std::int64_t>(packed)));
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireUnpack(dataflow::UnpackOp op, SimulatorState &state) {
  std::optional<unsigned> vecSize = vectorSizeAttr(op.getOperation(), state);
  if (!vecSize)
    return false;
  if (!hasToken(state.channels, op.getPackedMutable()) ||
      !hasToken(state.channels, op.getMaskMutable()))
    return false;
  std::optional<unsigned> laneWidth = signlessIntegerBitWidthForVector(
      op.getOutputs().front().getType(), state, "dataflow.unpack");
  if (!laneWidth)
    return false;
  if ((*laneWidth) * (*vecSize) > 64) {
    state.diagnostics.push_back(
        "dataflow.unpack DFG-sim supports packed widths up to 64 bits");
    return false;
  }

  Token packedToken = popToken(state.channels, op.getPackedMutable());
  Token maskToken = popToken(state.channels, op.getMaskMutable());
  const std::uint64_t packed =
      tokenBits(packedToken, (*laneWidth) * (*vecSize));
  const std::uint64_t mask = tokenBits(maskToken, *vecSize);
  const std::uint64_t laneMask = lowBitsMask(*laneWidth);
  for (unsigned i = 0; i < *vecSize; ++i) {
    if ((mask & (std::uint64_t{1} << i)) == 0)
      continue;
    std::uint64_t laneBits = (packed >> ((*laneWidth) * i)) & laneMask;
    emitToken(state, op.getOutputs()[i],
              integerValueToken(static_cast<std::int64_t>(laneBits)));
  }
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireSerialize(dataflow::SerializeOp op, SimulatorState &state) {
  std::optional<unsigned> vecSize = vectorSizeAttr(op.getOperation(), state);
  if (!vecSize)
    return false;
  if (!hasToken(state.channels, op.getMaskMutable()))
    return false;
  const Token &maskToken = peekToken(state.channels, op.getMaskMutable());
  const std::uint64_t mask = tokenBits(maskToken, *vecSize);
  for (unsigned i = 0; i < *vecSize; ++i) {
    if ((mask & (std::uint64_t{1} << i)) == 0)
      continue;
    if (!hasToken(state.channels, op->getOpOperand(i)))
      return false;
  }

  (void)popToken(state.channels, op.getMaskMutable());
  for (unsigned i = 0; i < *vecSize; ++i) {
    if ((mask & (std::uint64_t{1} << i)) == 0)
      continue;
    Token lane = popToken(state.channels, op->getOpOperand(i));
    emitToken(state, op.getData(), lane);
    emitToken(state, op.getCont(), boolValueToken(true));
  }
  emitToken(state, op.getCont(), boolValueToken(false));
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireCast(mlir::UnrealizedConversionCastOp op,
                     SimulatorState &state) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  mlir::OpOperand &operand = op->getOpOperand(0);
  if (!hasToken(state.channels, operand))
    return false;
  Token token = popToken(state.channels, operand);
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
  Token base = popToken(state.channels, op.getBaseMutable());
  if (base.kind != TokenKind::Pointer) {
    state.diagnostics.push_back("llvm.getelementptr base is not a pointer");
    return false;
  }
  llvm::SmallVector<Token> dynamicTokens;
  for (unsigned i = 1, e = op->getNumOperands(); i < e; ++i)
    dynamicTokens.push_back(popToken(state.channels, op->getOpOperand(i)));
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
    Token token = popToken(state.channels, memOperand);
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

static bool fireLoad(dataflow::LoadOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getAddrMutable()) ||
      !hasToken(state.channels, op.getCtrlMutable()))
    return false;
  std::optional<MemoryView> view =
      resolveMemoryView(state, op.getMem(), op.getMemMutable());
  if (!view)
    return false;
  Token addr = popToken(state.channels, op.getAddrMutable());
  popToken(state.channels, op.getCtrlMutable());
  std::optional<std::size_t> index = resolveElementIndex(
      *view, addr, state, op.getOperation(), "dataflow.load");
  if (!index)
    return false;
  emitToken(state, op.getData(), view->memory->elements[*index]);
  emitToken(state, op.getDone(), noneToken());
  ++state.loadFireCounts[op.getOperation()];
  if (hasComputedAddress(op.getAddr()))
    state.memoryAddressScore += kLoadAddressScore;
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireLLVMLoad(mlir::LLVM::LoadOp op, SimulatorState &state) {
  mlir::OpOperand &addrOperand = op->getOpOperand(0);
  if (!hasToken(state.channels, addrOperand))
    return false;
  Token ptr = popToken(state.channels, addrOperand);
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
  emitToken(state, op->getResult(0),
            viewOrErr->pointer.memory->elements[*index]);
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireLLVMStore(mlir::LLVM::StoreOp op, SimulatorState &state) {
  mlir::OpOperand &valueOperand = op->getOpOperand(0);
  mlir::OpOperand &addrOperand = op->getOpOperand(1);
  if (!hasToken(state.channels, valueOperand) ||
      !hasToken(state.channels, addrOperand))
    return false;
  Token value = popToken(state.channels, valueOperand);
  Token ptr = popToken(state.channels, addrOperand);
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
  viewOrErr->pointer.memory->elements[*index] = value;
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
  for (std::int64_t i = 0; i < byteLength; ++i)
    copied.push_back(srcOrErr->pointer.memory->elements[*srcStart + i]);
  for (auto [offset, token] : llvm::enumerate(copied))
    dstOrErr->pointer.memory->elements[*dstStart + offset] = token;
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireLLVMMemcpy(mlir::LLVM::MemcpyOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getDstMutable()) ||
      !hasToken(state.channels, op.getSrcMutable()) ||
      !hasToken(state.channels, op.getLenMutable()))
    return false;

  Token dst = popToken(state.channels, op.getDstMutable());
  Token src = popToken(state.channels, op.getSrcMutable());
  Token len = popToken(state.channels, op.getLenMutable());
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
    operands.push_back(popToken(state.channels, operand));

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
  Token addr = popToken(state.channels, op.getAddrMutable());
  Token data = popToken(state.channels, op.getDataMutable());
  popToken(state.channels, op.getCtrlMutable());
  std::optional<std::size_t> index = resolveElementIndex(
      *view, addr, state, op.getOperation(), "dataflow.store");
  if (!index)
    return false;
  view->memory->elements[*index] = data;
  emitToken(state, op.getDone(), noneToken());
  if (hasComputedAddress(op.getAddr()))
    state.memoryAddressScore += kStoreAddressScore;
  return recordEvent(state, op->getName().getStringRef());
}

static bool firePrimitiveOperation(mlir::Operation *op, mlir::Value result,
                                   SimulatorState &state) {
  if (state.terminalPrimitiveOps.contains(op))
    return false;
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    if (!hasToken(state.channels, operand))
      return false;
  }
  std::string predicate = primitivePredicate(op);
  auto descriptor = primitiveDescriptor(op, predicate, result);
  if (!descriptor) {
    state.diagnostics.push_back(llvm::toString(descriptor.takeError()));
    state.terminalPrimitiveOps.insert(op);
    return false;
  }
  llvm::SmallVector<PrimitiveValue> operands;
  for (mlir::OpOperand &operand : op->getOpOperands())
    operands.push_back(
        primitiveValueFromToken(popToken(state.channels, operand)));
  auto valueOrErr = evaluatePrimitiveOperation(*descriptor, operands);
  if (!valueOrErr) {
    state.diagnostics.push_back(llvm::toString(valueOrErr.takeError()));
    return false;
  }
  emitToken(state, result, tokenFromPrimitiveValue(*valueOrErr));
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
  std::int64_t byteOffset = 0;
  auto fixtureIt = state.globalMemoryFixtures.find(op.getGlobalName());
  if (fixtureIt != state.globalMemoryFixtures.end()) {
    state.rawMemoryFixtures[result] = fixtureIt->second;
    byteOffset = fixtureIt->second.byteOffset;
  }
  emitToken(state, result, pointerToken(result, {}, byteOffset));
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
  Token lhs = popToken(state.channels, lhsOperand);
  Token rhs = popToken(state.channels, rhsOperand);
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
  Token condition = popToken(state.channels, conditionOperand);
  Token trueValue = popToken(state.channels, trueOperand);
  Token falseValue = popToken(state.channels, falseOperand);
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

std::optional<std::string> unsupportedActorOperation(mlir::Operation *op) {
  if (op->getNumResults() == 1 &&
      isSupportedPrimitiveOperation(primitiveOperationName(op)))
    return std::nullopt;
  if (auto icmp = mlir::dyn_cast<mlir::LLVM::ICmpOp>(op)) {
    if (isSupportedPointerICmp(icmp))
      return std::nullopt;
    return unsupportedOperationLabel(op);
  }
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    if (isSupportedLLVMCall(call))
      return std::nullopt;
    return unsupportedOperationLabel(op);
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
  return unsupportedOperationLabel(op);
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

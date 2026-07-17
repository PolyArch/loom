#include "DFGSimulatorInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <functional>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

using MemoryCloneMap =
    llvm::DenseMap<const MemoryValue *, std::shared_ptr<MemoryValue>>;

static std::shared_ptr<MemoryValue>
cloneMemoryHandle(const std::shared_ptr<MemoryValue> &memory,
                  MemoryCloneMap &clones) {
  if (!memory)
    return {};
  auto [it, inserted] =
      clones.try_emplace(memory.get(), std::shared_ptr<MemoryValue>());
  if (inserted)
    it->second = std::make_shared<MemoryValue>(*memory);
  return it->second;
}

static void retargetTokenMemory(Token &token, MemoryCloneMap &clones) {
  if (token.kind != TokenKind::Pointer || !token.pointer.memory)
    return;
  token.pointer.memory = cloneMemoryHandle(token.pointer.memory, clones);
}

static void retargetChannelMap(ChannelMap &channels, MemoryCloneMap &clones) {
  for (auto &entry : channels)
    for (Token &token : entry.second)
      retargetTokenMemory(token, clones);
}

static void retargetOutputMap(OutputMap &outputs, MemoryCloneMap &clones) {
  for (auto &entry : outputs)
    for (Token &token : entry.second)
      retargetTokenMemory(token, clones);
}

static void retargetTokenVector(llvm::SmallVectorImpl<Token> &tokens,
                                MemoryCloneMap &clones) {
  for (Token &token : tokens)
    retargetTokenMemory(token, clones);
}

static void
retargetLoopStates(llvm::DenseMap<mlir::Operation *, LoopState> &states,
                   MemoryCloneMap &clones) {
  for (auto &entry : states)
    if (entry.second.latched)
      retargetTokenMemory(*entry.second.latched, clones);
}

static MemoryCloneMap isolateProbeStateMemory(SimulatorState &state) {
  MemoryCloneMap clones;
  for (auto &entry : state.memories)
    entry.second = cloneMemoryHandle(entry.second, clones);
  retargetChannelMap(state.channels, clones);
  retargetChannelMap(state.pendingChannels, clones);
  retargetOutputMap(state.observedOutputs, clones);
  retargetOutputMap(state.pendingObservedOutputs, clones);
  retargetLoopStates(state.carryStates, clones);
  retargetLoopStates(state.invariantStates, clones);
  return clones;
}

static void appendProbeDiagnostics(SimulatorState &state,
                                   const SimulatorState &probeState) {
  for (const std::string &diagnostic : probeState.diagnostics) {
    if (std::find(state.diagnostics.begin(), state.diagnostics.end(),
                  diagnostic) != state.diagnostics.end())
      continue;
    state.diagnostics.push_back(diagnostic);
  }
}

using LocalValueMap = llvm::DenseMap<mlir::Value, Token>;

static void retargetLocalValueMap(LocalValueMap &locals,
                                  MemoryCloneMap &clones) {
  for (auto &entry : locals)
    retargetTokenMemory(entry.second, clones);
}

static unsigned observedTokenCount(mlir::Value value,
                                   const SimulatorState &state) {
  unsigned count = 0;
  auto observedIt = state.observedOutputs.find(value);
  if (observedIt != state.observedOutputs.end())
    count += observedIt->second.size();
  auto pendingIt = state.pendingObservedOutputs.find(value);
  if (pendingIt != state.pendingObservedOutputs.end())
    count += pendingIt->second.size();
  return count;
}

static unsigned structuredOpFireIndex(mlir::Operation *op,
                                      const SimulatorState &state) {
  if (op->getNumResults() != 0)
    return observedTokenCount(op->getResult(0), state);
  auto effectIt = state.structuredEffectFireCounts.find(op);
  return effectIt == state.structuredEffectFireCounts.end()
             ? 0
             : static_cast<unsigned>(effectIt->second);
}

static unsigned structuredInputTokenCount(mlir::Operation *op,
                                          const SimulatorState &state) {
  if (op->getNumOperands() == 0)
    return 0;
  return observedTokenCount(op->getOperand(0), state);
}

bool hasPendingOrderedStructuredFire(mlir::Operation *op,
                                     const SimulatorState &state) {
  if (!isStructuredOperation(op))
    return false;
  return structuredInputTokenCount(op, state) >
         structuredOpFireIndex(op, state);
}

static void recordStructuredEffectFire(SimulatorState &state,
                                       mlir::Operation *op) {
  if (op->getNumResults() == 0)
    ++state.structuredEffectFireCounts[op];
}

static bool hasMemRefValue(mlir::ValueRange values) {
  return llvm::any_of(values, [](mlir::Value value) {
    return mlir::isa<mlir::MemRefType>(value.getType());
  });
}

static bool isSupportedStructuredCast(mlir::UnrealizedConversionCastOp cast) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return false;
  if (mlir::isa<mlir::MemRefType>(cast.getResult(0).getType()))
    return mlir::isa<mlir::LLVM::LLVMPointerType>(cast.getOperand(0).getType());
  return !hasMemRefValue(cast->getOperands()) &&
         !hasMemRefValue(cast->getResults());
}

static bool canBroadcastStructuredForCapture(mlir::Value value) {
  if (mlir::isa<mlir::BlockArgument>(value))
    return true;
  return mlir::isa_and_nonnull<mlir::arith::ConstantOp, dataflow::ConstantOp,
                               mlir::ub::PoisonOp>(value.getDefiningOp());
}
static std::optional<Token> lookupToken(mlir::Value value,
                                        SimulatorState &state,
                                        const LocalValueMap &locals,
                                        unsigned captureIndex = 0) {
  auto localIt = locals.find(value);
  if (localIt != locals.end())
    return localIt->second;
  const unsigned count = observedTokenCount(value, state);
  if (count == 0)
    return std::nullopt;
  if (count == 1 &&
      (captureIndex == 0 || canBroadcastStructuredForCapture(value)))
    captureIndex = 0;
  auto pendingIt = state.pendingObservedOutputs.find(value);
  auto observedIt = state.observedOutputs.find(value);
  const unsigned observedCount =
      observedIt == state.observedOutputs.end() ? 0 : observedIt->second.size();
  if (captureIndex >= count)
    return std::nullopt;
  if (captureIndex >= observedCount)
    return pendingIt->second[captureIndex - observedCount];
  return observedIt->second[captureIndex];
}

static bool valueAvailableForStructuredRegion(mlir::Value value,
                                              SimulatorState &state,
                                              const LocalValueMap &locals,
                                              unsigned captureIndex) {
  if (locals.contains(value))
    return true;
  if (state.memories.contains(value))
    return true;
  return lookupToken(value, state, locals, captureIndex).has_value();
}

static bool structuredRegionCapturesAvailable(mlir::Region &region,
                                              SimulatorState &state,
                                              const LocalValueMap &locals,
                                              unsigned captureIndex) {
  llvm::SetVector<mlir::Value> captures;
  mlir::getUsedValuesDefinedAbove(region, captures);
  for (mlir::Value value : captures) {
    if (!valueAvailableForStructuredRegion(value, state, locals, captureIndex))
      return false;
  }
  return true;
}

static mlir::Region *selectedIfRegion(mlir::scf::IfOp op, const Token &cond) {
  if (boolToken(cond))
    return &op.getThenRegion();
  if (op.getElseRegion().empty())
    return nullptr;
  return &op.getElseRegion();
}

static bool selectedIfCapturesAvailable(mlir::scf::IfOp op,
                                        SimulatorState &state,
                                        const LocalValueMap &locals,
                                        unsigned captureIndex) {
  std::optional<Token> cond =
      lookupToken(op.getCondition(), state, locals, captureIndex);
  if (!cond)
    return false;
  mlir::Region *selected = selectedIfRegion(op, *cond);
  return !selected || structuredRegionCapturesAvailable(*selected, state,
                                                        locals, captureIndex);
}

static unsigned structuredForFireIndex(mlir::scf::ForOp op,
                                       const SimulatorState &state) {
  return structuredOpFireIndex(op.getOperation(), state);
}
static bool assignLocalPrimitiveResult(mlir::Operation *op, mlir::Value result,
                                       SimulatorState &state,
                                       LocalValueMap &locals,
                                       unsigned captureIndex) {
  llvm::SmallVector<PrimitiveValue> operands;
  for (mlir::Value operand : op->getOperands()) {
    std::optional<Token> token =
        lookupToken(operand, state, locals, captureIndex);
    if (!token)
      return false;
    operands.push_back(primitiveValueFromToken(*token));
  }
  std::string predicate = primitivePredicate(op);
  auto descriptor = primitiveDescriptor(op, predicate, result);
  if (!descriptor) {
    state.diagnostics.push_back(llvm::toString(descriptor.takeError()));
    return false;
  }
  auto valueOrErr = evaluatePrimitiveOperation(*descriptor, operands);
  if (!valueOrErr) {
    state.diagnostics.push_back(llvm::toString(valueOrErr.takeError()));
    return false;
  }
  locals[result] = tokenFromPrimitiveValue(*valueOrErr);
  return recordEvent(state, primitiveOperationName(op));
}

static bool assignLocalLLVMZero(mlir::LLVM::ZeroOp op, SimulatorState &state,
                                LocalValueMap &locals) {
  auto tokenOrErr = zeroToken(op->getResult(0).getType());
  if (!tokenOrErr) {
    state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
    return false;
  }
  locals[op->getResult(0)] = *tokenOrErr;
  return recordEvent(state, op->getName().getStringRef());
}

static bool assignLocalLLVMAddressOf(mlir::LLVM::AddressOfOp op,
                                     SimulatorState &state,
                                     LocalValueMap &locals) {
  mlir::Value result = op->getResult(0);
  locals[result] = pointerToken(result);
  return recordEvent(state, op->getName().getStringRef());
}

static bool assignLocalUBPoison(mlir::ub::PoisonOp op, SimulatorState &state,
                                LocalValueMap &locals) {
  auto tokenOrErr = zeroToken(op->getResult(0).getType());
  if (!tokenOrErr) {
    state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
    return false;
  }
  locals[op->getResult(0)] = *tokenOrErr;
  return recordEvent(state, op->getName().getStringRef());
}

static bool assignLocalLLVMICmp(mlir::LLVM::ICmpOp op, SimulatorState &state,
                                LocalValueMap &locals, unsigned captureIndex) {
  std::optional<Token> lhs =
      lookupToken(op.getLhs(), state, locals, captureIndex);
  std::optional<Token> rhs =
      lookupToken(op.getRhs(), state, locals, captureIndex);
  if (!lhs || !rhs)
    return false;
  auto resultOrErr = evaluatePointerICmp(op, *lhs, *rhs);
  if (!resultOrErr) {
    state.diagnostics.push_back(llvm::toString(resultOrErr.takeError()));
    return false;
  }
  locals[op->getResult(0)] = *resultOrErr;
  return recordEvent(state, op->getName().getStringRef());
}

static bool assignLocalLLVMSelect(mlir::LLVM::SelectOp op,
                                  SimulatorState &state, LocalValueMap &locals,
                                  unsigned captureIndex) {
  if (!isPointerSelect(op))
    return assignLocalPrimitiveResult(op.getOperation(), op->getResult(0),
                                      state, locals, captureIndex);
  std::optional<Token> condition =
      lookupToken(op->getOperand(0), state, locals, captureIndex);
  std::optional<Token> trueValue =
      lookupToken(op->getOperand(1), state, locals, captureIndex);
  std::optional<Token> falseValue =
      lookupToken(op->getOperand(2), state, locals, captureIndex);
  if (!condition || !trueValue || !falseValue)
    return false;
  std::optional<Token> selected =
      evaluatePointerSelect(op, *condition, *trueValue, *falseValue, state);
  if (!selected)
    return false;
  locals[op->getResult(0)] = *selected;
  return recordEvent(state, op->getName().getStringRef());
}

static bool assignLocalDataflowConstant(dataflow::ConstantOp op,
                                        SimulatorState &state,
                                        LocalValueMap &locals,
                                        unsigned captureIndex) {
  if (!lookupToken(op.getCtrl(), state, locals, captureIndex))
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
  locals[op.getValue()] = *tokenOrErr;
  return recordEvent(state, op->getName().getStringRef());
}

static bool assignLocalCast(mlir::UnrealizedConversionCastOp cast,
                            SimulatorState &state, LocalValueMap &locals,
                            unsigned captureIndex) {
  if (!isSupportedStructuredCast(cast))
    return false;
  std::optional<Token> token =
      lookupToken(cast.getOperand(0), state, locals, captureIndex);
  if (!token)
    return false;
  if (auto memrefType =
          mlir::dyn_cast<mlir::MemRefType>(cast.getResult(0).getType())) {
    auto tokenOrErr =
        ensurePointerMemory(state, *token, memrefType.getElementType());
    if (!tokenOrErr) {
      state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
      return false;
    }
    locals[cast.getResult(0)] = *tokenOrErr;
    return true;
  }
  locals[cast.getResult(0)] = *token;
  return true;
}

static bool assignLocalGEP(mlir::LLVM::GEPOp op, SimulatorState &state,
                           LocalValueMap &locals, unsigned captureIndex) {
  std::optional<Token> base =
      lookupToken(op.getBase(), state, locals, captureIndex);
  if (!base)
    return false;
  if (base->kind != TokenKind::Pointer) {
    state.diagnostics.push_back("llvm.getelementptr base is not a pointer");
    return false;
  }
  llvm::SmallVector<Token> dynamicTokens;
  for (unsigned i = 1, e = op->getNumOperands(); i < e; ++i) {
    std::optional<Token> token =
        lookupToken(op->getOperand(i), state, locals, captureIndex);
    if (!token)
      return false;
    dynamicTokens.push_back(*token);
  }
  auto offsetOrErr = gepByteOffset(op, dynamicTokens);
  if (!offsetOrErr) {
    state.diagnostics.push_back(llvm::toString(offsetOrErr.takeError()));
    return false;
  }
  Token result = *base;
  result.pointer.byteOffset += *offsetOrErr;
  locals[op.getResult()] = result;
  return recordEvent(state, op->getName().getStringRef());
}

static std::optional<MemoryView> lookupLocalMemoryView(mlir::Value mem,
                                                       SimulatorState &state,
                                                       LocalValueMap &locals,
                                                       unsigned captureIndex) {
  if (std::optional<Token> token =
          lookupToken(mem, state, locals, captureIndex)) {
    if (token->kind != TokenKind::Pointer || !token->pointer.memory) {
      state.diagnostics.push_back(
          "dataflow memory operand is not a memory view");
      return std::nullopt;
    }
    return token->pointer;
  }
  auto memIt = state.memories.find(mem);
  if (memIt != state.memories.end())
    return MemoryView{memIt->second, mem, 0};
  return std::nullopt;
}

static bool assignLocalDataflowLoad(dataflow::LoadOp op, SimulatorState &state,
                                    LocalValueMap &locals,
                                    unsigned captureIndex) {
  std::optional<MemoryView> view =
      lookupLocalMemoryView(op.getMem(), state, locals, captureIndex);
  std::optional<Token> addr =
      lookupToken(op.getAddr(), state, locals, captureIndex);
  std::optional<Token> ctrl =
      lookupToken(op.getCtrl(), state, locals, captureIndex);
  if (!view || !addr || !ctrl)
    return false;
  std::optional<std::size_t> index = resolveElementIndex(
      *view, *addr, state, op.getOperation(), "dataflow.load");
  if (!index)
    return false;
  std::optional<Token> value =
      readMemoryElement(*view, *index, state, "dataflow.load");
  if (!value)
    return false;
  locals[op.getData()] = *value;
  locals[op.getDone()] = noneToken();
  ++state.loadFireCounts[op.getOperation()];
  if (hasComputedAddress(op.getAddr()))
    state.memoryAddressScore += kLoadAddressScore;
  return recordEvent(state, op->getName().getStringRef());
}

static std::optional<unsigned> demuxResultIndex(dataflow::DemuxOp op,
                                                mlir::Value value) {
  for (auto [index, result] : llvm::enumerate(op.getOutputs()))
    if (result == value)
      return static_cast<unsigned>(index);
  return std::nullopt;
}

static std::optional<bool> isInactiveDemuxResult(mlir::Value value,
                                                 SimulatorState &state,
                                                 const LocalValueMap &locals,
                                                 unsigned captureIndex) {
  auto demux = value.getDefiningOp<dataflow::DemuxOp>();
  if (!demux)
    return std::nullopt;
  std::optional<unsigned> outputIndex = demuxResultIndex(demux, value);
  if (!outputIndex)
    return std::nullopt;
  std::optional<Token> sel =
      lookupToken(demux.getSel(), state, locals, captureIndex);
  if (!sel)
    return std::nullopt;
  const std::int64_t lane =
      mlir::isa<mlir::IntegerType>(demux.getSel().getType())
          ? boolToken(*sel)
          : integerToken(*sel);
  if (lane < 0 || static_cast<std::size_t>(lane) >= demux.getOutputs().size()) {
    state.diagnostics.push_back("dataflow.demux selector is out of range");
    return std::nullopt;
  }
  return static_cast<unsigned>(lane) != *outputIndex;
}

static bool assignLocalDataflowStore(dataflow::StoreOp op,
                                     SimulatorState &state,
                                     LocalValueMap &locals,
                                     unsigned captureIndex) {
  std::optional<MemoryView> view =
      lookupLocalMemoryView(op.getMem(), state, locals, captureIndex);
  std::optional<Token> addr =
      lookupToken(op.getAddr(), state, locals, captureIndex);
  std::optional<Token> data =
      lookupToken(op.getData(), state, locals, captureIndex);
  std::optional<Token> ctrl =
      lookupToken(op.getCtrl(), state, locals, captureIndex);
  if (!ctrl) {
    std::optional<bool> inactive =
        isInactiveDemuxResult(op.getCtrl(), state, locals, captureIndex);
    if (inactive && *inactive) {
      locals[op.getDone()] = noneToken();
      return true;
    }
  }
  if (!view || !addr || !data || !ctrl)
    return false;
  std::optional<std::size_t> index = resolveElementIndex(
      *view, *addr, state, op.getOperation(), "dataflow.store");
  if (!index)
    return false;
  writeMemoryElement(*view, *index, *data);
  locals[op.getDone()] = noneToken();
  if (hasComputedAddress(op.getAddr()))
    state.memoryAddressScore += kStoreAddressScore;
  return recordEvent(state, op->getName().getStringRef());
}

static bool assignLocalLLVMMemcpy(mlir::LLVM::MemcpyOp op,
                                  SimulatorState &state, LocalValueMap &locals,
                                  unsigned captureIndex) {
  std::optional<Token> dst =
      lookupToken(op.getDst(), state, locals, captureIndex);
  std::optional<Token> src =
      lookupToken(op.getSrc(), state, locals, captureIndex);
  std::optional<Token> len =
      lookupToken(op.getLen(), state, locals, captureIndex);
  if (!dst || !src || !len)
    return false;
  return executeLLVMMemcpy(op, state, *dst, *src, *len);
}

static bool assignLocalLLVMCall(mlir::LLVM::CallOp op, SimulatorState &state,
                                LocalValueMap &locals, unsigned captureIndex) {
  if (!isSupportedLLVMCall(op))
    return false;
  llvm::SmallVector<Token> operands;
  operands.reserve(op->getNumOperands());
  for (mlir::Value operand : op->getOperands()) {
    std::optional<Token> token =
        lookupToken(operand, state, locals, captureIndex);
    if (!token)
      return false;
    operands.push_back(*token);
  }
  Token result;
  if (!executeCmsisNNVecMatMultTS8(op, state, operands, result))
    return false;
  locals[op->getResult(0)] = result;
  return true;
}

static bool assignLocalGate(dataflow::GateOp op, SimulatorState &state,
                            LocalValueMap &locals, unsigned captureIndex) {
  std::optional<Token> cond =
      lookupToken(op.getBeforeCond(), state, locals, captureIndex);
  std::optional<Token> value =
      lookupToken(op.getBeforeValue(), state, locals, captureIndex);
  if (!cond || !value)
    return false;
  const GateSemanticState gate =
      state.gateContinueStates.contains(op.getOperation())
          ? GateSemanticState::Open
          : GateSemanticState::Closed;
  GateTransition transition = evaluateGateTransition(
      gate, std::optional<bool>{boolToken(*cond)}, /*valueAvailable=*/true);
  if (!transition.firing.ready)
    return false;

  if (transition.emitPhase)
    locals[op.getAfterCond()] = boolValueToken(transition.phase);
  if (transition.forwardedInput == GateInput::Value)
    locals[op.getAfterValue()] = *value;
  if (transition.nextState == GateSemanticState::Open)
    state.gateContinueStates.insert(op.getOperation());
  else
    state.gateContinueStates.erase(op.getOperation());
  return recordEvent(state, op->getName().getStringRef());
}

static bool assignLocalMux(dataflow::MuxOp op, SimulatorState &state,
                           LocalValueMap &locals, unsigned captureIndex) {
  std::optional<Token> sel =
      lookupToken(op.getSel(), state, locals, captureIndex);
  if (!sel)
    return false;
  const std::int64_t lane = mlir::isa<mlir::IntegerType>(op.getSel().getType())
                                ? boolToken(*sel)
                                : integerToken(*sel);
  if (lane < 0 || static_cast<std::size_t>(lane) >= op.getInputs().size()) {
    state.diagnostics.push_back("dataflow.mux selector is out of range");
    return false;
  }
  std::optional<Token> selected = lookupToken(
      op.getInputs()[static_cast<unsigned>(lane)], state, locals, captureIndex);
  if (!selected)
    return false;
  locals[op.getOutput()] = *selected;
  return recordEvent(state, op->getName().getStringRef());
}

static bool assignLocalDemux(dataflow::DemuxOp op, SimulatorState &state,
                             LocalValueMap &locals, unsigned captureIndex) {
  std::optional<Token> sel =
      lookupToken(op.getSel(), state, locals, captureIndex);
  std::optional<Token> input =
      lookupToken(op.getInput(), state, locals, captureIndex);
  if (!sel || !input)
    return false;
  const std::int64_t lane = mlir::isa<mlir::IntegerType>(op.getSel().getType())
                                ? boolToken(*sel)
                                : integerToken(*sel);
  if (lane < 0 || static_cast<std::size_t>(lane) >= op.getOutputs().size()) {
    state.diagnostics.push_back("dataflow.demux selector is out of range");
    return false;
  }
  locals[op.getOutputs()[static_cast<unsigned>(lane)]] = *input;
  return recordEvent(state, op->getName().getStringRef());
}

static bool executeStructuredForBodyOp(mlir::Operation *op,
                                       SimulatorState &state,
                                       LocalValueMap &locals,
                                       unsigned captureIndex);
static bool executeStructuredFor(mlir::scf::ForOp op, SimulatorState &state,
                                 llvm::ArrayRef<Token> operands,
                                 unsigned captureIndex,
                                 llvm::SmallVectorImpl<Token> &results,
                                 const LocalValueMap *captures = nullptr);
static bool executeStructuredWhile(mlir::scf::WhileOp op, SimulatorState &state,
                                   llvm::ArrayRef<Token> operands,
                                   unsigned captureIndex,
                                   llvm::SmallVectorImpl<Token> &results,
                                   const LocalValueMap *captures = nullptr);
static bool executeStructuredForall(mlir::scf::ForallOp op,
                                    SimulatorState &state,
                                    LocalValueMap &captures,
                                    unsigned captureIndex = 0);

static bool evaluateStructuredYieldRegion(
    mlir::Operation *parent, mlir::Block *block, llvm::StringRef opName,
    SimulatorState &state, LocalValueMap &parentLocals, unsigned captureIndex,
    llvm::SmallVectorImpl<Token> &yielded) {
  if (!block)
    return parent->getNumResults() == 0;
  LocalValueMap locals = parentLocals;
  for (mlir::Operation &bodyOp : block->getOperations()) {
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(bodyOp)) {
      if (yield.getNumOperands() != parent->getNumResults())
        return false;
      for (mlir::Value value : yield.getResults()) {
        std::optional<Token> token =
            lookupToken(value, state, locals, captureIndex);
        if (!token)
          return false;
        yielded.push_back(*token);
      }
      return true;
    }
    if (!executeStructuredForBodyOp(&bodyOp, state, locals, captureIndex)) {
      state.diagnostics.push_back(("structured " + opName +
                                   " failed to execute " +
                                   bodyOp.getName().getStringRef())
                                      .str());
      return false;
    }
  }
  return parent->getNumResults() == 0;
}

static bool evaluateStructuredIf(mlir::scf::IfOp op, SimulatorState &state,
                                 LocalValueMap &locals, unsigned captureIndex,
                                 llvm::SmallVectorImpl<Token> &yielded) {
  std::optional<Token> cond =
      lookupToken(op.getCondition(), state, locals, captureIndex);
  if (!cond)
    return false;
  mlir::Block *selected =
      boolToken(*cond)
          ? op.thenBlock()
          : (op.getElseRegion().empty() ? nullptr : op.elseBlock());
  return evaluateStructuredYieldRegion(op.getOperation(), selected, "scf.if",
                                       state, locals, captureIndex, yielded);
}

static bool executeStructuredIfLocally(mlir::scf::IfOp op,
                                       SimulatorState &state,
                                       LocalValueMap &locals,
                                       unsigned captureIndex) {
  if (!selectedIfCapturesAvailable(op, state, locals, captureIndex))
    return false;
  llvm::SmallVector<Token> yielded;
  if (!evaluateStructuredIf(op, state, locals, captureIndex, yielded))
    return false;
  if (yielded.size() != op->getNumResults())
    return false;
  for (auto [result, token] : llvm::zip(op->getResults(), yielded))
    locals[result] = token;
  return recordEvent(state, op->getName().getStringRef());
}

static mlir::Block *
selectStructuredIndexSwitchBlock(mlir::scf::IndexSwitchOp op,
                                 SimulatorState &state, LocalValueMap &locals,
                                 unsigned captureIndex) {
  std::optional<Token> selector =
      lookupToken(op.getArg(), state, locals, captureIndex);
  if (!selector)
    return nullptr;
  const std::int64_t selected = integerToken(*selector);
  for (auto [index, value] : llvm::enumerate(op.getCases()))
    if (selected == value)
      return &op.getCaseBlock(index);
  return &op.getDefaultBlock();
}

static bool evaluateStructuredIndexSwitch(
    mlir::scf::IndexSwitchOp op, SimulatorState &state, LocalValueMap &locals,
    unsigned captureIndex, llvm::SmallVectorImpl<Token> &yielded) {
  mlir::Block *selected =
      selectStructuredIndexSwitchBlock(op, state, locals, captureIndex);
  if (!selected)
    return false;
  return evaluateStructuredYieldRegion(op.getOperation(), selected,
                                       "scf.index_switch", state, locals,
                                       captureIndex, yielded);
}

static bool executeStructuredIndexSwitchLocally(mlir::scf::IndexSwitchOp op,
                                                SimulatorState &state,
                                                LocalValueMap &locals,
                                                unsigned captureIndex) {
  llvm::SmallVector<Token> yielded;
  if (!evaluateStructuredIndexSwitch(op, state, locals, captureIndex, yielded))
    return false;
  if (yielded.size() != op->getNumResults())
    return false;
  for (auto [result, token] : llvm::zip(op->getResults(), yielded))
    locals[result] = token;
  return recordEvent(state, op->getName().getStringRef());
}

static bool executeStructuredForBodyOp(mlir::Operation *op,
                                       SimulatorState &state,
                                       LocalValueMap &locals,
                                       unsigned captureIndex) {
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op))
    return executeStructuredIfLocally(ifOp, state, locals, captureIndex);
  if (auto switchOp = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(op))
    return executeStructuredIndexSwitchLocally(switchOp, state, locals,
                                               captureIndex);
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
    llvm::SmallVector<Token> operands;
    operands.reserve(forOp->getNumOperands());
    for (mlir::Value operand : forOp->getOperands()) {
      std::optional<Token> token =
          lookupToken(operand, state, locals, captureIndex);
      if (!token)
        return false;
      operands.push_back(*token);
    }
    llvm::SmallVector<Token> results;
    if (!executeStructuredFor(forOp, state, operands, captureIndex, results,
                              &locals))
      return false;
    if (results.size() != forOp->getNumResults())
      return false;
    for (auto [result, token] : llvm::zip(forOp->getResults(), results))
      locals[result] = token;
    return true;
  }
  if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
    llvm::SmallVector<Token> operands;
    operands.reserve(whileOp->getNumOperands());
    for (mlir::Value operand : whileOp->getOperands()) {
      std::optional<Token> token =
          lookupToken(operand, state, locals, captureIndex);
      if (!token)
        return false;
      operands.push_back(*token);
    }
    llvm::SmallVector<Token> results;
    if (!executeStructuredWhile(whileOp, state, operands, captureIndex, results,
                                &locals))
      return false;
    if (results.size() != whileOp->getNumResults())
      return false;
    for (auto [result, token] : llvm::zip(whileOp->getResults(), results))
      locals[result] = token;
    return true;
  }
  if (auto forallOp = mlir::dyn_cast<mlir::scf::ForallOp>(op))
    return executeStructuredForall(forallOp, state, locals, captureIndex);
  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
    auto attr = mlir::dyn_cast<mlir::TypedAttr>(constant.getValue());
    if (!attr) {
      state.diagnostics.push_back("arith.constant has untyped value");
      return false;
    }
    auto tokenOrErr = tokenFromTypedAttr(attr);
    if (!tokenOrErr) {
      state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
      return false;
    }
    locals[constant.getResult()] = *tokenOrErr;
    return recordEvent(state, constant->getName().getStringRef());
  }
  if (auto constant = mlir::dyn_cast<dataflow::ConstantOp>(op))
    return assignLocalDataflowConstant(constant, state, locals, captureIndex);
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op))
    return assignLocalCast(cast, state, locals, captureIndex);
  if (auto gep = mlir::dyn_cast<mlir::LLVM::GEPOp>(op))
    return assignLocalGEP(gep, state, locals, captureIndex);
  if (auto zero = mlir::dyn_cast<mlir::LLVM::ZeroOp>(op))
    return assignLocalLLVMZero(zero, state, locals);
  if (auto addressOf = mlir::dyn_cast<mlir::LLVM::AddressOfOp>(op))
    return assignLocalLLVMAddressOf(addressOf, state, locals);
  if (auto poison = mlir::dyn_cast<mlir::ub::PoisonOp>(op))
    return assignLocalUBPoison(poison, state, locals);
  if (auto icmp = mlir::dyn_cast<mlir::LLVM::ICmpOp>(op))
    return assignLocalLLVMICmp(icmp, state, locals, captureIndex);
  if (auto select = mlir::dyn_cast<mlir::LLVM::SelectOp>(op))
    return assignLocalLLVMSelect(select, state, locals, captureIndex);
  if (auto load = mlir::dyn_cast<dataflow::LoadOp>(op))
    return assignLocalDataflowLoad(load, state, locals, captureIndex);
  if (auto store = mlir::dyn_cast<dataflow::StoreOp>(op))
    return assignLocalDataflowStore(store, state, locals, captureIndex);
  if (auto memcpy = mlir::dyn_cast<mlir::LLVM::MemcpyOp>(op))
    return assignLocalLLVMMemcpy(memcpy, state, locals, captureIndex);
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op))
    return assignLocalLLVMCall(call, state, locals, captureIndex);
  if (auto gate = mlir::dyn_cast<dataflow::GateOp>(op))
    return assignLocalGate(gate, state, locals, captureIndex);
  if (auto mux = mlir::dyn_cast<dataflow::MuxOp>(op))
    return assignLocalMux(mux, state, locals, captureIndex);
  if (auto demux = mlir::dyn_cast<dataflow::DemuxOp>(op))
    return assignLocalDemux(demux, state, locals, captureIndex);
  if (op->getNumResults() == 1 &&
      isSupportedPrimitiveOperation(primitiveOperationName(op)))
    return assignLocalPrimitiveResult(op, op->getResult(0), state, locals,
                                      captureIndex);
  return false;
}

static std::optional<std::string>
unsupportedStructuredIfOperation(mlir::scf::IfOp op);

static std::optional<std::string>
unsupportedStructuredIndexSwitchOperation(mlir::scf::IndexSwitchOp op);

static std::optional<std::string>
unsupportedStructuredForOperation(mlir::scf::ForOp op);

static std::optional<std::string>
unsupportedStructuredWhileOperation(mlir::scf::WhileOp op);

static std::optional<std::string>
unsupportedStructuredForallOperation(mlir::scf::ForallOp op);

static std::optional<std::string>
unsupportedStructuredBodyOperation(mlir::Operation *op) {
  if (mlir::isa<mlir::arith::ConstantOp>(op))
    return std::nullopt;
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op))
    return unsupportedStructuredIfOperation(ifOp);
  if (auto switchOp = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(op))
    return unsupportedStructuredIndexSwitchOperation(switchOp);
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op))
    return unsupportedStructuredForOperation(forOp);
  if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op))
    return unsupportedStructuredWhileOperation(whileOp);
  if (auto forallOp = mlir::dyn_cast<mlir::scf::ForallOp>(op))
    return unsupportedStructuredForallOperation(forallOp);
  if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
    if (isSupportedStructuredCast(cast))
      return std::nullopt;
    return unsupportedOperationLabel(op);
  }
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
  if (mlir::isa<dataflow::ConstantOp, dataflow::LoadOp, dataflow::StoreOp,
                dataflow::GateOp, dataflow::MuxOp, dataflow::DemuxOp,
                mlir::LLVM::AddressOfOp, mlir::LLVM::GEPOp, mlir::LLVM::ZeroOp,
                mlir::LLVM::MemcpyOp, mlir::ub::PoisonOp>(op))
    return std::nullopt;
  if (op->getNumResults() == 1 &&
      isSupportedPrimitiveOperation(primitiveOperationName(op)))
    return std::nullopt;
  return unsupportedOperationLabel(op);
}

static std::optional<std::string>
unsupportedStructuredBodyOperations(mlir::Block *block) {
  for (mlir::Operation &bodyOp : block->without_terminator())
    if (auto name = unsupportedStructuredBodyOperation(&bodyOp))
      return name;
  return std::nullopt;
}

static std::optional<std::string>
unsupportedStructuredYieldRegion(mlir::Operation *parent, mlir::Block *block,
                                 llvm::StringRef opName) {
  if (!block)
    return parent->getNumResults() == 0
               ? std::nullopt
               : std::optional<std::string>(opName.str());
  auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(block->getTerminator());
  if (!yield || yield.getNumOperands() != parent->getNumResults())
    return "scf.yield";
  return unsupportedStructuredBodyOperations(block);
}

static std::optional<std::string>
unsupportedStructuredIfOperation(mlir::scf::IfOp op) {
  if (op.getThenRegion().empty())
    return "scf.if";
  if (auto name = unsupportedStructuredYieldRegion(op.getOperation(),
                                                   op.thenBlock(), "scf.if"))
    return name;
  if (op.getElseRegion().empty())
    return op->getNumResults() == 0 ? std::nullopt
                                    : std::optional<std::string>("scf.if");
  return unsupportedStructuredYieldRegion(op.getOperation(), op.elseBlock(),
                                          "scf.if");
}

static std::optional<std::string>
unsupportedStructuredIndexSwitchOperation(mlir::scf::IndexSwitchOp op) {
  if (op.getDefaultRegion().empty())
    return "scf.index_switch";
  if (auto name = unsupportedStructuredYieldRegion(
          op.getOperation(), &op.getDefaultBlock(), "scf.index_switch"))
    return name;
  for (unsigned index = 0, end = op.getNumCases(); index < end; ++index)
    if (auto name = unsupportedStructuredYieldRegion(
            op.getOperation(), &op.getCaseBlock(index), "scf.index_switch"))
      return name;
  return std::nullopt;
}

static std::optional<std::string>
unsupportedStructuredForOperation(mlir::scf::ForOp op) {
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(op.getBody()->getTerminator());
  if (!yield)
    return "scf.for";
  if (yield.getNumOperands() != op->getNumResults())
    return "scf.yield";
  return unsupportedStructuredBodyOperations(op.getBody());
}

static std::optional<std::string>
unsupportedStructuredWhileBody(mlir::Block *block,
                               llvm::StringRef terminatorName) {
  if (!block)
    return "scf.while";
  mlir::Operation *terminator = block->getTerminator();
  if (!terminator || terminator->getName().getStringRef() != terminatorName)
    return terminatorName.str();
  return unsupportedStructuredBodyOperations(block);
}

static std::optional<std::string>
unsupportedStructuredWhileOperation(mlir::scf::WhileOp op) {
  if (!op.getBeforeBody() || !op.getAfterBody())
    return "scf.while";
  auto condition = op.getConditionOp();
  if (!condition || condition.getArgs().size() != op->getNumResults())
    return "scf.condition";
  auto yield = op.getYieldOp();
  if (!yield || yield.getResults().size() != op->getNumOperands())
    return "scf.yield";
  if (auto name =
          unsupportedStructuredWhileBody(op.getBeforeBody(), "scf.condition"))
    return name;
  return unsupportedStructuredWhileBody(op.getAfterBody(), "scf.yield");
}

static std::optional<std::string>
unsupportedStructuredForallOperation(mlir::scf::ForallOp op) {
  if (!op.getOutputs().empty() || op->getNumResults() != 0)
    return "scf.forall";
  auto inParallel = op.getTerminator();
  if (inParallel.getRegion().empty() || !inParallel.getRegion().front().empty())
    return "scf.forall.in_parallel";
  return unsupportedStructuredBodyOperations(op.getBody());
}

static unsigned structuredIfFireIndex(mlir::scf::IfOp op,
                                      const SimulatorState &state) {
  return structuredOpFireIndex(op.getOperation(), state);
}

static bool fireStructuredIf(mlir::scf::IfOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op->getOpOperand(0)))
    return false;
  const unsigned captureIndex = structuredIfFireIndex(op, state);
  Token cond = peekToken(state.channels, op->getOpOperand(0));
  LocalValueMap probeLocals;
  probeLocals[op.getCondition()] = cond;
  if (!selectedIfCapturesAvailable(op, state, probeLocals, captureIndex))
    return false;
  llvm::SmallVector<Token> probeYielded;
  SimulatorState probeState = state;
  (void)isolateProbeStateMemory(probeState);
  if (!evaluateStructuredIf(op, probeState, probeLocals, captureIndex,
                            probeYielded)) {
    appendProbeDiagnostics(state, probeState);
    return false;
  }

  LocalValueMap locals;
  locals[op.getCondition()] = popToken(state.channels, op->getOpOperand(0));
  llvm::SmallVector<Token> yielded;
  if (!evaluateStructuredIf(op, state, locals, captureIndex, yielded))
    return false;
  if (yielded.size() != op->getNumResults())
    return false;
  for (auto [result, token] : llvm::zip(op->getResults(), yielded))
    emitToken(state, result, token);
  recordStructuredEffectFire(state, op.getOperation());
  return recordEvent(state, op->getName().getStringRef());
}

static unsigned structuredIndexSwitchFireIndex(mlir::scf::IndexSwitchOp op,
                                               const SimulatorState &state) {
  return structuredOpFireIndex(op.getOperation(), state);
}

static bool fireStructuredIndexSwitch(mlir::scf::IndexSwitchOp op,
                                      SimulatorState &state) {
  if (!hasToken(state.channels, op->getOpOperand(0)))
    return false;
  const unsigned captureIndex = structuredIndexSwitchFireIndex(op, state);
  Token selector = peekToken(state.channels, op->getOpOperand(0));
  LocalValueMap probeLocals;
  probeLocals[op.getArg()] = selector;
  llvm::SmallVector<Token> probeYielded;
  SimulatorState probeState = state;
  (void)isolateProbeStateMemory(probeState);
  if (!evaluateStructuredIndexSwitch(op, probeState, probeLocals, captureIndex,
                                     probeYielded)) {
    appendProbeDiagnostics(state, probeState);
    return false;
  }

  LocalValueMap locals;
  locals[op.getArg()] = popToken(state.channels, op->getOpOperand(0));
  llvm::SmallVector<Token> yielded;
  if (!evaluateStructuredIndexSwitch(op, state, locals, captureIndex, yielded))
    return false;
  if (yielded.size() != op->getNumResults())
    return false;
  for (auto [result, token] : llvm::zip(op->getResults(), yielded))
    emitToken(state, result, token);
  recordStructuredEffectFire(state, op.getOperation());
  return recordEvent(state, op->getName().getStringRef());
}

static bool executeStructuredFor(mlir::scf::ForOp op, SimulatorState &state,
                                 llvm::ArrayRef<Token> operands,
                                 unsigned captureIndex,
                                 llvm::SmallVectorImpl<Token> &results,
                                 const LocalValueMap *captures) {
  std::int64_t iv = integerToken(operands[0]);
  const std::int64_t ub = integerToken(operands[1]);
  const std::int64_t step = integerToken(operands[2]);
  if (step == 0) {
    state.diagnostics.push_back("scf.for step is zero");
    return false;
  }
  llvm::SmallVector<Token> carried;
  carried.append(operands.begin() + 3, operands.end());
  if (carried.size() != op->getNumResults()) {
    state.diagnostics.push_back("scf.for iter_args/result count mismatch");
    return false;
  }
  auto keepRunning = [&]() { return step > 0 ? iv < ub : iv > ub; };
  std::uint64_t iterations = 0;
  while (keepRunning()) {
    if (state.maxStructuredLoopIterations != 0 &&
        iterations >= state.maxStructuredLoopIterations) {
      state.diagnostics.push_back(
          "maximum structured scf.for iterations reached");
      return false;
    }
    LocalValueMap locals;
    if (captures)
      locals = *captures;
    locals[op.getInductionVar()] = integerValueToken(iv);
    for (auto [arg, token] : llvm::zip(op.getRegionIterArgs(), carried))
      locals[arg] = token;
    llvm::SmallVector<Token> yielded;
    bool sawYield = false;
    for (mlir::Operation &bodyOp : op.getBody()->getOperations()) {
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(bodyOp)) {
        for (mlir::Value value : yield.getResults()) {
          std::optional<Token> token =
              lookupToken(value, state, locals, captureIndex);
          if (!token)
            return false;
          yielded.push_back(*token);
        }
        sawYield = true;
        break;
      }
      if (!executeStructuredForBodyOp(&bodyOp, state, locals, captureIndex)) {
        state.diagnostics.push_back(("structured scf.for failed to execute " +
                                     bodyOp.getName().getStringRef())
                                        .str());
        return false;
      }
    }
    if (!sawYield)
      return false;
    if (yielded.size() != carried.size())
      return false;
    carried = std::move(yielded);
    iv += step;
    ++iterations;
  }
  state.structuredLoopIterations =
      std::max(state.structuredLoopIterations, iterations);
  results.append(carried.begin(), carried.end());
  return true;
}

static bool fireStructuredFor(mlir::scf::ForOp op, SimulatorState &state) {
  const unsigned operandCount = op->getNumOperands();
  for (unsigned operandIndex = 0; operandIndex < operandCount; ++operandIndex) {
    if (!hasToken(state.channels, op->getOpOperand(operandIndex)))
      return false;
  }
  const unsigned captureIndex = structuredForFireIndex(op, state);
  llvm::SmallVector<Token> operands;
  operands.reserve(operandCount);
  for (unsigned operandIndex = 0; operandIndex < operandCount; ++operandIndex)
    operands.push_back(
        peekToken(state.channels, op->getOpOperand(operandIndex)));

  llvm::SmallVector<Token> probeResults;
  SimulatorState probeState = state;
  MemoryCloneMap probeClones = isolateProbeStateMemory(probeState);
  llvm::SmallVector<Token> probeOperands(operands.begin(), operands.end());
  retargetTokenVector(probeOperands, probeClones);
  if (!executeStructuredFor(op, probeState, probeOperands, captureIndex,
                            probeResults)) {
    appendProbeDiagnostics(state, probeState);
    return false;
  }

  for (unsigned operandIndex = 0; operandIndex < operandCount; ++operandIndex)
    (void)popToken(state.channels, op->getOpOperand(operandIndex));
  llvm::SmallVector<Token> results;
  if (!executeStructuredFor(op, state, operands, captureIndex, results))
    return false;
  for (auto [result, token] : llvm::zip(op->getResults(), results))
    emitToken(state, result, token);
  recordStructuredEffectFire(state, op.getOperation());
  return true;
}

static unsigned structuredWhileFireIndex(mlir::scf::WhileOp op,
                                         const SimulatorState &state) {
  return structuredOpFireIndex(op.getOperation(), state);
}

static bool executeStructuredWhile(mlir::scf::WhileOp op, SimulatorState &state,
                                   llvm::ArrayRef<Token> operands,
                                   unsigned captureIndex,
                                   llvm::SmallVectorImpl<Token> &results,
                                   const LocalValueMap *captures) {
  llvm::SmallVector<Token> carried(operands.begin(), operands.end());
  if (carried.size() != op.getBeforeBody()->getNumArguments()) {
    state.diagnostics.push_back("scf.while init/before-arg count mismatch");
    return false;
  }

  std::uint64_t beforeExecutions = 0;
  while (true) {
    if (state.maxStructuredLoopIterations != 0 &&
        beforeExecutions >= state.maxStructuredLoopIterations) {
      state.diagnostics.push_back(
          "maximum structured scf.while iterations reached");
      return false;
    }

    LocalValueMap beforeLocals;
    if (captures)
      beforeLocals = *captures;
    for (auto [arg, token] :
         llvm::zip(op.getBeforeBody()->getArguments(), carried))
      beforeLocals[arg] = token;
    ++beforeExecutions;

    llvm::SmallVector<Token> conditionArgs;
    bool sawCondition = false;
    bool keepRunning = false;
    for (mlir::Operation &bodyOp : op.getBeforeBody()->getOperations()) {
      if (auto condition = mlir::dyn_cast<mlir::scf::ConditionOp>(bodyOp)) {
        std::optional<Token> conditionToken = lookupToken(
            condition.getCondition(), state, beforeLocals, captureIndex);
        if (!conditionToken)
          return false;
        keepRunning = boolToken(*conditionToken);
        for (mlir::Value value : condition.getArgs()) {
          std::optional<Token> token =
              lookupToken(value, state, beforeLocals, captureIndex);
          if (!token)
            return false;
          conditionArgs.push_back(*token);
        }
        sawCondition = true;
        break;
      }
      if (!executeStructuredForBodyOp(&bodyOp, state, beforeLocals,
                                      captureIndex)) {
        state.diagnostics.push_back(("structured scf.while failed to execute " +
                                     bodyOp.getName().getStringRef())
                                        .str());
        return false;
      }
    }
    if (!sawCondition)
      return false;
    if (!keepRunning) {
      if (conditionArgs.size() != op->getNumResults()) {
        state.diagnostics.push_back("scf.condition/result count mismatch");
        return false;
      }
      state.structuredLoopIterations =
          std::max(state.structuredLoopIterations, beforeExecutions);
      results.append(conditionArgs.begin(), conditionArgs.end());
      return true;
    }
    if (conditionArgs.size() != op.getAfterBody()->getNumArguments()) {
      state.diagnostics.push_back("scf.condition/after-arg count mismatch");
      return false;
    }

    LocalValueMap afterLocals;
    if (captures)
      afterLocals = *captures;
    for (auto [arg, token] :
         llvm::zip(op.getAfterBody()->getArguments(), conditionArgs))
      afterLocals[arg] = token;

    llvm::SmallVector<Token> yielded;
    bool sawYield = false;
    for (mlir::Operation &bodyOp : op.getAfterBody()->getOperations()) {
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(bodyOp)) {
        for (mlir::Value value : yield.getResults()) {
          std::optional<Token> token =
              lookupToken(value, state, afterLocals, captureIndex);
          if (!token)
            return false;
          yielded.push_back(*token);
        }
        sawYield = true;
        break;
      }
      if (!executeStructuredForBodyOp(&bodyOp, state, afterLocals,
                                      captureIndex)) {
        state.diagnostics.push_back(("structured scf.while failed to execute " +
                                     bodyOp.getName().getStringRef())
                                        .str());
        return false;
      }
    }
    if (!sawYield)
      return false;
    if (yielded.size() != op->getNumOperands()) {
      state.diagnostics.push_back("scf.yield/init count mismatch");
      return false;
    }
    carried = std::move(yielded);
  }
}

static bool fireStructuredWhile(mlir::scf::WhileOp op, SimulatorState &state) {
  const unsigned operandCount = op->getNumOperands();
  for (unsigned operandIndex = 0; operandIndex < operandCount; ++operandIndex) {
    if (!hasToken(state.channels, op->getOpOperand(operandIndex)))
      return false;
  }
  const unsigned captureIndex = structuredWhileFireIndex(op, state);
  llvm::SmallVector<Token> operands;
  operands.reserve(operandCount);
  for (unsigned operandIndex = 0; operandIndex < operandCount; ++operandIndex)
    operands.push_back(
        peekToken(state.channels, op->getOpOperand(operandIndex)));

  llvm::SmallVector<Token> probeResults;
  SimulatorState probeState = state;
  MemoryCloneMap probeClones = isolateProbeStateMemory(probeState);
  llvm::SmallVector<Token> probeOperands(operands.begin(), operands.end());
  retargetTokenVector(probeOperands, probeClones);
  if (!executeStructuredWhile(op, probeState, probeOperands, captureIndex,
                              probeResults)) {
    appendProbeDiagnostics(state, probeState);
    return false;
  }

  for (unsigned operandIndex = 0; operandIndex < operandCount; ++operandIndex)
    (void)popToken(state.channels, op->getOpOperand(operandIndex));
  llvm::SmallVector<Token> results;
  if (!executeStructuredWhile(op, state, operands, captureIndex, results))
    return false;
  for (auto [result, token] : llvm::zip(op->getResults(), results))
    emitToken(state, result, token);
  recordStructuredEffectFire(state, op.getOperation());
  return recordEvent(state, op->getName().getStringRef());
}

static std::optional<std::int64_t>
resolveStructuredBound(mlir::OpFoldResult bound, SimulatorState &state,
                       LocalValueMap &locals, unsigned captureIndex,
                       llvm::StringRef opName) {
  if (auto attr = llvm::dyn_cast<mlir::Attribute>(bound)) {
    auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr);
    if (!intAttr) {
      state.diagnostics.push_back((opName + " bound is not an integer").str());
      return std::nullopt;
    }
    return intAttr.getInt();
  }
  auto value = llvm::dyn_cast<mlir::Value>(bound);
  if (!value) {
    state.diagnostics.push_back((opName + " bound is not a value").str());
    return std::nullopt;
  }
  std::optional<Token> token = lookupToken(value, state, locals, captureIndex);
  if (!token)
    return std::nullopt;
  return integerToken(*token);
}

static bool executeStructuredForall(mlir::scf::ForallOp op,
                                    SimulatorState &state,
                                    LocalValueMap &captures,
                                    unsigned captureIndex) {
  if (!op.getOutputs().empty() || op->getNumResults() != 0) {
    state.diagnostics.push_back(
        "scf.forall shared_out/result form is unsupported");
    return false;
  }
  auto inParallel = op.getTerminator();
  if (inParallel.getRegion().empty() ||
      !inParallel.getRegion().front().empty()) {
    state.diagnostics.push_back(
        "scf.forall.in_parallel aggregation is unsupported");
    return false;
  }

  llvm::SmallVector<mlir::OpFoldResult> mixedLbs = op.getMixedLowerBound();
  llvm::SmallVector<mlir::OpFoldResult> mixedUbs = op.getMixedUpperBound();
  llvm::SmallVector<mlir::OpFoldResult> mixedSteps = op.getMixedStep();
  llvm::SmallVector<mlir::Value> ivs = op.getInductionVars();
  const unsigned rank = ivs.size();
  if (rank == 0 || mixedLbs.size() != rank || mixedUbs.size() != rank ||
      mixedSteps.size() != rank) {
    state.diagnostics.push_back("scf.forall rank/bounds mismatch");
    return false;
  }

  llvm::SmallVector<std::int64_t> lbs;
  llvm::SmallVector<std::int64_t> ubs;
  llvm::SmallVector<std::int64_t> steps;
  lbs.reserve(rank);
  ubs.reserve(rank);
  steps.reserve(rank);
  for (unsigned dim = 0; dim < rank; ++dim) {
    std::optional<std::int64_t> lb = resolveStructuredBound(
        mixedLbs[dim], state, captures, captureIndex, "scf.forall");
    std::optional<std::int64_t> ub = resolveStructuredBound(
        mixedUbs[dim], state, captures, captureIndex, "scf.forall");
    std::optional<std::int64_t> step = resolveStructuredBound(
        mixedSteps[dim], state, captures, captureIndex, "scf.forall");
    if (!lb || !ub || !step)
      return false;
    if (*step == 0) {
      state.diagnostics.push_back("scf.forall step is zero");
      return false;
    }
    lbs.push_back(*lb);
    ubs.push_back(*ub);
    steps.push_back(*step);
  }

  std::uint64_t iterations = 0;
  auto executeAtRank = [&](LocalValueMap &locals) -> bool {
    for (mlir::Operation &bodyOp : op.getBody()->without_terminator()) {
      if (!executeStructuredForBodyOp(&bodyOp, state, locals, captureIndex)) {
        state.diagnostics.push_back(
            ("structured scf.forall failed to execute " +
             bodyOp.getName().getStringRef())
                .str());
        return false;
      }
    }
    ++iterations;
    return true;
  };

  std::function<bool(unsigned, LocalValueMap &)> visitDim =
      [&](unsigned dim, LocalValueMap &locals) -> bool {
    if (dim == rank) {
      if (state.maxStructuredLoopIterations != 0 &&
          iterations >= state.maxStructuredLoopIterations) {
        state.diagnostics.push_back(
            "maximum structured scf.forall iterations reached");
        return false;
      }
      return executeAtRank(locals);
    }
    auto keepRunning = [&](std::int64_t iv) {
      return steps[dim] > 0 ? iv < ubs[dim] : iv > ubs[dim];
    };
    for (std::int64_t iv = lbs[dim]; keepRunning(iv); iv += steps[dim]) {
      LocalValueMap nested = locals;
      nested[ivs[dim]] = integerValueToken(iv);
      if (!visitDim(dim + 1, nested))
        return false;
    }
    return true;
  };

  LocalValueMap locals = captures;
  if (!visitDim(0, locals))
    return false;
  state.structuredLoopIterations =
      std::max(state.structuredLoopIterations, iterations);
  return recordEvent(state, op->getName().getStringRef());
}

static bool fireStructuredForall(mlir::scf::ForallOp op,
                                 SimulatorState &state) {
  if (state.oneShotOps.contains(op.getOperation()))
    return false;
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    if (!hasToken(state.channels, operand))
      return false;
  }

  LocalValueMap operands;
  for (mlir::OpOperand &operand : op->getOpOperands())
    operands[operand.get()] = peekToken(state.channels, operand);
  if (!structuredRegionCapturesAvailable(op.getRegion(), state, operands, 0))
    return false;

  SimulatorState probeState = state;
  MemoryCloneMap probeClones = isolateProbeStateMemory(probeState);
  LocalValueMap probeOperands = operands;
  retargetLocalValueMap(probeOperands, probeClones);
  if (!executeStructuredForall(op, probeState, probeOperands)) {
    appendProbeDiagnostics(state, probeState);
    return false;
  }

  for (mlir::OpOperand &operand : op->getOpOperands())
    operands[operand.get()] = popToken(state.channels, operand);
  if (!executeStructuredForall(op, state, operands))
    return false;
  recordStructuredEffectFire(state, op.getOperation());
  state.oneShotOps.insert(op.getOperation());
  return true;
}

bool isStructuredOperation(mlir::Operation *op) {
  return mlir::isa<mlir::scf::IfOp, mlir::scf::IndexSwitchOp, mlir::scf::ForOp,
                   mlir::scf::WhileOp, mlir::scf::ForallOp>(op);
}

bool fireStructuredOperation(mlir::Operation *op, SimulatorState &state) {
  return llvm::TypeSwitch<mlir::Operation *, bool>(op)
      .Case<mlir::scf::IfOp>(
          [&](auto typedOp) { return fireStructuredIf(typedOp, state); })
      .Case<mlir::scf::IndexSwitchOp>([&](auto typedOp) {
        return fireStructuredIndexSwitch(typedOp, state);
      })
      .Case<mlir::scf::ForOp>(
          [&](auto typedOp) { return fireStructuredFor(typedOp, state); })
      .Case<mlir::scf::WhileOp>(
          [&](auto typedOp) { return fireStructuredWhile(typedOp, state); })
      .Case<mlir::scf::ForallOp>(
          [&](auto typedOp) { return fireStructuredForall(typedOp, state); })
      .Default([](mlir::Operation *) { return false; });
}

std::optional<std::string> unsupportedStructuredOperation(mlir::Operation *op) {
  return llvm::TypeSwitch<mlir::Operation *, std::optional<std::string>>(op)
      .Case<mlir::scf::IfOp>(unsupportedStructuredIfOperation)
      .Case<mlir::scf::IndexSwitchOp>(unsupportedStructuredIndexSwitchOperation)
      .Case<mlir::scf::ForOp>(unsupportedStructuredForOperation)
      .Case<mlir::scf::WhileOp>(unsupportedStructuredWhileOperation)
      .Case<mlir::scf::ForallOp>(unsupportedStructuredForallOperation)
      .Default([](mlir::Operation *unknown) {
        return std::optional<std::string>(unsupportedOperationLabel(unknown));
      });
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

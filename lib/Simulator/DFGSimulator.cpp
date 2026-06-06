#include "Simulator/DFGSimulator.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>

using namespace loom::sim;

namespace {

enum class TokenKind { None, Integer, Float, Bool };

struct Token {
  TokenKind kind = TokenKind::None;
  std::int64_t intValue = 0;
  double floatValue = 0.0;
  bool boolValue = false;
};

using ChannelMap = llvm::DenseMap<const mlir::OpOperand *, std::deque<Token>>;
using OutputMap = llvm::DenseMap<mlir::Value, llvm::SmallVector<Token>>;

struct StreamState {
  bool initialized = false;
  bool done = false;
  std::uint64_t trueEmissions = 0;
  std::int64_t current = 0;
  std::int64_t ub = 0;
  std::int64_t step = 0;
};

struct LoopState {
  bool initialized = false;
  std::optional<Token> latched;
};

struct MemoryValue {
  mlir::Type elementType;
  llvm::SmallVector<Token> elements;
};

struct SimulatorState {
  ChannelMap channels;
  ChannelMap pendingChannels;
  OutputMap observedOutputs;
  OutputMap pendingObservedOutputs;
  llvm::DenseMap<mlir::Value, std::shared_ptr<MemoryValue>> memories;
  llvm::DenseMap<mlir::Value, std::string> rawMemoryFixtures;
  llvm::DenseMap<mlir::Operation *, StreamState> streamStates;
  llvm::DenseMap<mlir::Operation *, LoopState> carryStates;
  llvm::DenseMap<mlir::Operation *, LoopState> invariantStates;
  llvm::DenseSet<mlir::Operation *> gateContinueStates;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> loadFireCounts;
  llvm::DenseSet<mlir::Operation *> oneShotOps;
  llvm::DenseMap<mlir::Value, std::uint64_t> seededTokenCounts;
  llvm::SmallVector<std::string> diagnostics;
  std::map<std::string, std::uint64_t> operationFireCounts;
  std::uint64_t eventCount = 0;
};

std::string typePrefix(mlir::Type type) {
  if (mlir::isa<mlir::NoneType>(type))
    return "none";
  if (mlir::isa<mlir::IndexType>(type))
    return "index";
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return llvm::formatv("i{0}", intType.getWidth()).str();
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floatType.isF16())
      return "f16";
    if (floatType.isF32())
      return "f32";
    if (floatType.isF64())
      return "f64";
  }
  std::string storage;
  llvm::raw_string_ostream os(storage);
  type.print(os);
  return storage;
}

std::string tokenToString(const Token &token, mlir::Type type) {
  if (token.kind == TokenKind::None)
    return "none";
  if (token.kind == TokenKind::Bool)
    return typePrefix(type) + ":" + (token.boolValue ? "true" : "false");
  if (token.kind == TokenKind::Integer)
    return typePrefix(type) + ":" + std::to_string(token.intValue);
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << typePrefix(type) << ':';
  if (std::floor(token.floatValue) == token.floatValue)
    os << static_cast<std::int64_t>(token.floatValue);
  else
    os << llvm::formatv("{0:f6}", token.floatValue);
  return os.str();
}

llvm::Expected<Token> tokenFromTypedAttr(mlir::TypedAttr attr) {
  if (mlir::isa<mlir::NoneType>(attr.getType()))
    return Token{TokenKind::None};
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    if (intAttr.getType().isInteger(1))
      return Token{TokenKind::Bool, 0, 0.0, intAttr.getValue().isOne()};
    return Token{TokenKind::Integer, intAttr.getValue().getSExtValue()};
  }
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr))
    return Token{TokenKind::Float, 0, floatAttr.getValueAsDouble()};
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported dataflow.constant attribute");
}

llvm::Expected<Token> parseRuntimeToken(llvm::StringRef raw, mlir::Type type) {
  raw = raw.trim();
  if (mlir::isa<mlir::NoneType>(type)) {
    if (raw == "none")
      return Token{TokenKind::None};
    return llvm::createStringError(std::errc::invalid_argument,
                                   "none argument expects value 'none'");
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (intType.getWidth() == 1) {
      if (raw == "true" || raw == "1")
        return Token{TokenKind::Bool, 0, 0.0, true};
      if (raw == "false" || raw == "0")
        return Token{TokenKind::Bool, 0, 0.0, false};
      return llvm::createStringError(std::errc::invalid_argument,
                                     "i1 argument expects true/false/0/1");
    }
    std::int64_t value = 0;
    if (raw.getAsInteger(10, value))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "integer argument is not base-10");
    return Token{TokenKind::Integer, value};
  }
  if (mlir::isa<mlir::IndexType>(type)) {
    std::int64_t value = 0;
    if (raw.getAsInteger(10, value))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "index argument is not base-10");
    return Token{TokenKind::Integer, value};
  }
  if (mlir::isa<mlir::FloatType>(type)) {
    double value = 0.0;
    if (raw.getAsDouble(value))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "float argument is not parseable");
    return Token{TokenKind::Float, 0, value};
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported runtime argument type");
}

llvm::Expected<llvm::SmallVector<Token>> parseMemoryTokens(llvm::StringRef raw,
                                                           mlir::Type type) {
  llvm::SmallVector<Token> tokens;
  llvm::SmallVector<llvm::StringRef> parts;
  raw.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (parts.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "memref fixture must contain values");
  for (llvm::StringRef part : parts) {
    auto tokenOrErr = parseRuntimeToken(part, type);
    if (!tokenOrErr)
      return tokenOrErr.takeError();
    tokens.push_back(*tokenOrErr);
  }
  return tokens;
}

bool hasToken(ChannelMap &channels, mlir::OpOperand &operand) {
  auto it = channels.find(&operand);
  return it != channels.end() && !it->second.empty();
}

Token popToken(ChannelMap &channels, mlir::OpOperand &operand) {
  auto &queue = channels[&operand];
  Token token = queue.front();
  queue.pop_front();
  return token;
}

void emitToken(SimulatorState &state, mlir::Value value, Token token) {
  for (mlir::OpOperand &use : value.getUses())
    state.pendingChannels[&use].push_back(token);
  state.pendingObservedOutputs[value].push_back(token);
}

bool recordEvent(SimulatorState &state, llvm::StringRef opName) {
  auto costOrErr = estimateOperationCost(opName);
  if (!costOrErr) {
    state.diagnostics.push_back(llvm::toString(costOrErr.takeError()));
    return false;
  }
  ++state.eventCount;
  ++state.operationFireCounts[opName.str()];
  return true;
}

std::uint64_t estimateDynamicPipelineCycles(
    const std::map<std::string, std::uint64_t> &operationFireCounts,
    llvm::SmallVectorImpl<std::string> &diagnostics) {
  std::uint64_t cycles = 0;
  for (const auto &[opName, fireCount] : operationFireCounts) {
    if (fireCount == 0)
      continue;
    auto costOrErr = estimateOperationCost(opName);
    if (!costOrErr) {
      diagnostics.push_back(llvm::toString(costOrErr.takeError()));
      continue;
    }
    cycles += costOrErr->latencyCycles;
    if (fireCount > 1)
      cycles += (fireCount - 1) * costOrErr->reciprocalThroughput;
  }
  return cycles;
}

std::uint64_t dynamicWorkItems(const SimulatorState &state) {
  std::uint64_t maxStreamItems = 0;
  for (const auto &entry : state.streamStates)
    maxStreamItems = std::max(maxStreamItems, entry.second.trueEmissions);
  std::uint64_t maxSeededItems = 0;
  for (const auto &entry : state.seededTokenCounts)
    maxSeededItems = std::max(maxSeededItems, entry.second);
  const std::uint64_t workItems = std::max(maxStreamItems, maxSeededItems);
  if (workItems == 0 && state.eventCount > 0)
    return 1;
  return workItems;
}

bool requiresCompleteDynamicReturn(mlir::Value value) {
  if (!mlir::isa<mlir::NoneType>(value.getType()))
    return true;
  return value.getDefiningOp<dataflow::StoreOp>() != nullptr;
}

void flushPendingTokens(SimulatorState &state) {
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

std::int64_t integerToken(const Token &token) { return token.intValue; }

bool boolToken(const Token &token) {
  if (token.kind == TokenKind::Bool)
    return token.boolValue;
  return token.intValue != 0;
}

PrimitiveValue primitiveValueFromToken(const Token &token) {
  switch (token.kind) {
  case TokenKind::None:
    return PrimitiveValue::none();
  case TokenKind::Integer:
    return PrimitiveValue::integer(token.intValue);
  case TokenKind::Float:
    return PrimitiveValue::floating(token.floatValue);
  case TokenKind::Bool:
    return PrimitiveValue::boolean(token.boolValue);
  }
  return PrimitiveValue::none();
}

Token tokenFromPrimitiveValue(const PrimitiveValue &value) {
  switch (value.kind) {
  case PrimitiveValueKind::None:
    return Token{TokenKind::None};
  case PrimitiveValueKind::Integer:
    return Token{TokenKind::Integer, value.intValue};
  case PrimitiveValueKind::Float:
    return Token{TokenKind::Float, 0, value.floatValue};
  case PrimitiveValueKind::Bool:
    return Token{TokenKind::Bool, 0, 0.0, value.boolValue};
  }
  return Token{TokenKind::None};
}

unsigned integerBitWidth(mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return intType.getWidth();
  if (mlir::isa<mlir::IndexType>(type))
    return 64;
  return 0;
}

std::string primitivePredicate(mlir::Operation *op) {
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(op))
    return mlir::arith::stringifyCmpIPredicate(cmp.getPredicate()).str();
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(op))
    return mlir::arith::stringifyCmpFPredicate(cmp.getPredicate()).str();
  return "";
}

bool evaluateCont(std::int64_t current, std::int64_t ub, llvm::StringRef pred) {
  if (pred == "<")
    return current < ub;
  if (pred == "<=")
    return current <= ub;
  if (pred == ">")
    return current > ub;
  if (pred == ">=")
    return current >= ub;
  if (pred == "!=")
    return current != ub;
  return false;
}

std::int64_t stepIndex(std::int64_t current, std::int64_t step,
                       llvm::StringRef stepOp) {
  if (stepOp == "+=")
    return current + step;
  if (stepOp == "-=")
    return current - step;
  if (stepOp == "*=")
    return current * step;
  if (stepOp == "/=")
    return step == 0 ? current : current / step;
  if (stepOp == "<<=")
    return current << step;
  if (stepOp == ">>=")
    return current >> step;
  return current;
}

bool fireStream(dataflow::StreamOp op, SimulatorState &state) {
  StreamState &stream = state.streamStates[op.getOperation()];
  if (stream.done)
    return false;

  if (!stream.initialized) {
    if (!hasToken(state.channels, op->getOpOperand(0)) ||
        !hasToken(state.channels, op->getOpOperand(1)) ||
        !hasToken(state.channels, op->getOpOperand(2)))
      return false;
    stream.current =
        integerToken(popToken(state.channels, op->getOpOperand(0)));
    stream.ub = integerToken(popToken(state.channels, op->getOpOperand(1)));
    stream.step = integerToken(popToken(state.channels, op->getOpOperand(2)));
    stream.initialized = true;
  }

  const bool cont = evaluateCont(stream.current, stream.ub, op.getContCond());
  emitToken(state, op.getIndex(), Token{TokenKind::Integer, stream.current});
  emitToken(state, op.getRwc(), Token{TokenKind::Bool, 0, 0.0, cont});
  if (cont) {
    ++stream.trueEmissions;
    stream.current = stepIndex(stream.current, stream.step, op.getStepOp());
  } else {
    stream.done = true;
  }
  return recordEvent(state, op->getName().getStringRef());
}

bool fireConstant(dataflow::ConstantOp op, SimulatorState &state) {
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

bool fireCarry(dataflow::CarryOp op, SimulatorState &state) {
  LoopState &carry = state.carryStates[op.getOperation()];
  if (!carry.initialized) {
    if (!hasToken(state.channels, op->getOpOperand(1)))
      return false;
    Token init = popToken(state.channels, op->getOpOperand(1));
    emitToken(state, op.getOutput(), init);
    carry.initialized = true;
    return recordEvent(state, op->getName().getStringRef());
  }

  if (!hasToken(state.channels, op->getOpOperand(0)) ||
      !hasToken(state.channels, op->getOpOperand(2)))
    return false;
  Token cond = popToken(state.channels, op->getOpOperand(0));
  Token value = popToken(state.channels, op->getOpOperand(2));
  if (boolToken(cond)) {
    emitToken(state, op.getOutput(), value);
  } else {
    carry.initialized = false;
  }
  return recordEvent(state, op->getName().getStringRef());
}

bool fireInvariant(dataflow::InvariantOp op, SimulatorState &state) {
  LoopState &invariant = state.invariantStates[op.getOperation()];
  if (!invariant.initialized) {
    if (!hasToken(state.channels, op->getOpOperand(1)))
      return false;
    Token init = popToken(state.channels, op->getOpOperand(1));
    invariant.latched = init;
    invariant.initialized = true;
    emitToken(state, op.getOutput(), init);
    return recordEvent(state, op->getName().getStringRef());
  }

  if (!hasToken(state.channels, op->getOpOperand(0)))
    return false;
  Token cond = popToken(state.channels, op->getOpOperand(0));
  if (boolToken(cond)) {
    emitToken(state, op.getOutput(), *invariant.latched);
  } else {
    invariant.initialized = false;
    invariant.latched.reset();
  }
  return recordEvent(state, op->getName().getStringRef());
}

bool fireGate(dataflow::GateOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getBeforeCondMutable()) ||
      !hasToken(state.channels, op.getBeforeValueMutable()))
    return false;
  Token cond = popToken(state.channels, op.getBeforeCondMutable());
  Token value = popToken(state.channels, op.getBeforeValueMutable());
  const bool isContinue = state.gateContinueStates.contains(op.getOperation());
  const bool open = boolToken(cond);

  if (!isContinue) {
    if (open) {
      emitToken(state, op.getAfterValue(), value);
      state.gateContinueStates.insert(op.getOperation());
    }
    return recordEvent(state, op->getName().getStringRef());
  }

  if (open) {
    emitToken(state, op.getAfterCond(), Token{TokenKind::Bool, 0, 0.0, true});
    emitToken(state, op.getAfterValue(), value);
  } else {
    emitToken(state, op.getAfterCond(), Token{TokenKind::Bool, 0, 0.0, false});
    state.gateContinueStates.erase(op.getOperation());
  }
  return recordEvent(state, op->getName().getStringRef());
}

bool fireSync(dataflow::SyncOp op, SimulatorState &state) {
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

bool fireLoad(dataflow::LoadOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getAddrMutable()) ||
      !hasToken(state.channels, op.getCtrlMutable()))
    return false;
  auto memIt = state.memories.find(op.getMem());
  if (memIt == state.memories.end()) {
    state.diagnostics.push_back("dataflow.load has no memref fixture");
    return false;
  }
  Token addr = popToken(state.channels, op.getAddrMutable());
  popToken(state.channels, op.getCtrlMutable());
  const std::int64_t index = integerToken(addr);
  if (index < 0 ||
      static_cast<std::size_t>(index) >= memIt->second->elements.size()) {
    state.diagnostics.push_back("dataflow.load address is out of range");
    return false;
  }
  emitToken(state, op.getData(),
            memIt->second->elements[static_cast<std::size_t>(index)]);
  emitToken(state, op.getDone(), Token{TokenKind::None});
  ++state.loadFireCounts[op.getOperation()];
  return recordEvent(state, op->getName().getStringRef());
}

bool fireStore(dataflow::StoreOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getAddrMutable()) ||
      !hasToken(state.channels, op.getDataMutable()) ||
      !hasToken(state.channels, op.getCtrlMutable()))
    return false;
  auto memIt = state.memories.find(op.getMem());
  if (memIt == state.memories.end()) {
    state.diagnostics.push_back("dataflow.store has no memref fixture");
    return false;
  }
  Token addr = popToken(state.channels, op.getAddrMutable());
  Token data = popToken(state.channels, op.getDataMutable());
  popToken(state.channels, op.getCtrlMutable());
  const std::int64_t index = integerToken(addr);
  if (index < 0 ||
      static_cast<std::size_t>(index) >= memIt->second->elements.size()) {
    state.diagnostics.push_back("dataflow.store address is out of range");
    return false;
  }
  memIt->second->elements[static_cast<std::size_t>(index)] = data;
  emitToken(state, op.getDone(), Token{TokenKind::None});
  return recordEvent(state, op->getName().getStringRef());
}

bool firePrimitiveOperation(mlir::Operation *op, mlir::Value result,
                            SimulatorState &state) {
  for (mlir::OpOperand &operand : op->getOpOperands()) {
    if (!hasToken(state.channels, operand))
      return false;
  }
  llvm::SmallVector<PrimitiveValue> operands;
  for (mlir::OpOperand &operand : op->getOpOperands())
    operands.push_back(
        primitiveValueFromToken(popToken(state.channels, operand)));
  auto valueOrErr =
      evaluatePrimitiveOperation(
          PrimitiveOperationDescriptor{op->getName().getStringRef(),
                                       primitivePredicate(op),
                                       integerBitWidth(result.getType()),
                                       integerBitWidth(
                                           op->getOperand(0).getType())},
          operands);
  if (!valueOrErr) {
    state.diagnostics.push_back(llvm::toString(valueOrErr.takeError()));
    return false;
  }
  emitToken(state, result, tokenFromPrimitiveValue(*valueOrErr));
  return recordEvent(state, op->getName().getStringRef());
}

bool fireArithConstant(mlir::arith::ConstantOp op, SimulatorState &state) {
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

bool fireGenericPrimitive(mlir::Operation *op, SimulatorState &state) {
  if (!isSupportedPrimitiveOperation(op->getName().getStringRef()) ||
      op->getNumResults() != 1)
    return false;
  return firePrimitiveOperation(op, op->getResult(0), state);
}

bool isSupportedNonEvent(mlir::Operation *op) {
  return mlir::isa<dataflow::GraphReturnOp, mlir::UnrealizedConversionCastOp>(
      op);
}

dataflow::StreamOp findStreamIndexSource(mlir::Value value) {
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    value = cast.getIn();
  auto stream = value.getDefiningOp<dataflow::StreamOp>();
  if (!stream || stream.getIndex() != value)
    return {};
  return stream;
}

bool fireOperation(mlir::Operation *op, SimulatorState &state) {
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
      .Case<dataflow::LoadOp>(
          [&](auto typedOp) { return fireLoad(typedOp, state); })
      .Case<dataflow::StoreOp>(
          [&](auto typedOp) { return fireStore(typedOp, state); })
      .Case<mlir::arith::ConstantOp>(
          [&](auto typedOp) { return fireArithConstant(typedOp, state); })
      .Default([&](mlir::Operation *genericOp) {
        return fireGenericPrimitive(genericOp, state);
      });
}

std::optional<std::string> unsupportedOperation(mlir::Operation *op) {
  if (isSupportedNonEvent(op))
    return std::nullopt;
  if (isSupportedPrimitiveOperation(op->getName().getStringRef()) &&
      op->getNumResults() == 1)
    return std::nullopt;
  if (mlir::isa<dataflow::StreamOp, dataflow::ConstantOp, dataflow::CarryOp,
                dataflow::InvariantOp, dataflow::GateOp, dataflow::SyncOp,
                dataflow::LoadOp, dataflow::StoreOp,
                mlir::arith::ConstantOp>(op))
    return std::nullopt;
  return op->getName().getStringRef().str();
}

dataflow::GraphFuncOp findGraph(mlir::ModuleOp module, llvm::StringRef name) {
  if (name.starts_with("@"))
    name = name.drop_front();
  dataflow::GraphFuncOp match;
  module.walk([&](dataflow::GraphFuncOp graph) {
    if (!match && graph.getSymName() == name)
      match = graph;
  });
  return match;
}

llvm::Expected<llvm::StringMap<llvm::SmallVector<std::string>>>
indexRuntimeArgs(llvm::ArrayRef<DFGRuntimeArg> args, unsigned argCount) {
  llvm::StringMap<llvm::SmallVector<std::string>> byIndex;
  for (const DFGRuntimeArg &arg : args) {
    if (arg.index >= argCount)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "argument index %u is out of range",
                                     arg.index);
    std::string key = std::to_string(arg.index);
    byIndex[key].push_back(arg.value);
  }
  return byIndex;
}

llvm::Expected<llvm::StringMap<std::string>>
indexMemoryArgs(llvm::ArrayRef<DFGMemoryArg> args, unsigned argCount) {
  llvm::StringMap<std::string> byIndex;
  for (const DFGMemoryArg &arg : args) {
    if (arg.index >= argCount)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "memref index %u is out of range",
                                     arg.index);
    std::string key = std::to_string(arg.index);
    if (byIndex.contains(key))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "memref index %u is repeated", arg.index);
    byIndex.try_emplace(key, arg.values);
  }
  return byIndex;
}

void observeReturnOperands(dataflow::GraphFuncOp graph,
                           llvm::SmallVectorImpl<mlir::Value> &returns) {
  auto ret = mlir::dyn_cast_or_null<dataflow::GraphReturnOp>(
      graph.getBody().front().getTerminator());
  if (!ret)
    return;
  returns.append(ret.getValues().begin(), ret.getValues().end());
}

void seedBlockArgument(SimulatorState &state, mlir::BlockArgument arg,
                       const Token &token) {
  for (mlir::OpOperand &use : arg.getUses())
    state.channels[&use].push_back(token);
  state.observedOutputs[arg].push_back(token);
  ++state.seededTokenCounts[arg];
}

llvm::Error propagateMemoryAliases(mlir::Block &entry, SimulatorState &state) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Operation &op : entry.getOperations()) {
      if (!mlir::isa<mlir::UnrealizedConversionCastOp>(op) ||
          op.getNumOperands() != 1 || op.getNumResults() != 1)
        continue;
      mlir::Value source = op.getOperand(0);
      mlir::Value target = op.getResult(0);
      if (state.memories.contains(target))
        continue;
      auto memoryIt = state.memories.find(source);
      if (memoryIt != state.memories.end()) {
        state.memories[target] = memoryIt->second;
        changed = true;
        continue;
      }
      auto rawIt = state.rawMemoryFixtures.find(source);
      auto targetMemref = mlir::dyn_cast<mlir::MemRefType>(target.getType());
      if (rawIt == state.rawMemoryFixtures.end() || !targetMemref)
        continue;
      auto tokensOrErr =
          parseMemoryTokens(rawIt->second, targetMemref.getElementType());
      if (!tokensOrErr)
        return tokensOrErr.takeError();
      auto memory = std::make_shared<MemoryValue>(
          MemoryValue{targetMemref.getElementType(), std::move(*tokensOrErr)});
      state.memories[source] = memory;
      state.memories[target] = memory;
      state.rawMemoryFixtures[target] = rawIt->second;
      changed = true;
    }
  }
  return llvm::Error::success();
}

bool hasIncompleteStreamLoads(mlir::Block &entry, SimulatorState &state) {
  bool incomplete = false;
  for (mlir::Operation &op : entry.getOperations()) {
    auto load = mlir::dyn_cast<dataflow::LoadOp>(op);
    if (!load)
      continue;
    dataflow::StreamOp stream = findStreamIndexSource(load.getAddr());
    if (!stream)
      continue;
    const std::uint64_t required =
        state.streamStates[stream.getOperation()].trueEmissions;
    const std::uint64_t actual = state.loadFireCounts[load.getOperation()];
    if (actual >= required)
      continue;
    incomplete = true;
    state.diagnostics.push_back(
        llvm::formatv("dataflow.load consumed {0} of {1} true stream indices",
                      actual, required)
            .str());
  }
  return incomplete;
}

} // namespace

llvm::Expected<DFGSimulationReport>
loom::sim::simulateDataflowGraph(mlir::ModuleOp module,
                                 const DFGSimulationOptions &options) {
  DFGSimulationReport report;
  report.graph = options.graphName;
  report.workload =
      options.workloadName.empty() ? options.graphName : options.workloadName;
  report.status = "pass";

  dataflow::GraphFuncOp graph = findGraph(module, options.graphName);
  if (!graph)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "dataflow.graph.func '%s' was not found",
                                   options.graphName.c_str());
  if (graph.isExternal())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "dataflow.graph.func '%s' is external",
                                   options.graphName.c_str());

  mlir::Block &entry = graph.getBody().front();
  auto argsOrErr = indexRuntimeArgs(options.args, entry.getNumArguments());
  if (!argsOrErr)
    return argsOrErr.takeError();
  llvm::StringMap<llvm::SmallVector<std::string>> args = std::move(*argsOrErr);
  auto memoriesOrErr =
      indexMemoryArgs(options.memories, entry.getNumArguments());
  if (!memoriesOrErr)
    return memoriesOrErr.takeError();
  llvm::StringMap<std::string> memories = std::move(*memoriesOrErr);

  SimulatorState state;
  llvm::SmallVector<mlir::Value> returnValues;
  observeReturnOperands(graph, returnValues);

  for (auto [index, arg] : llvm::enumerate(entry.getArguments())) {
    std::string key = std::to_string(index);
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(arg.getType())) {
      if (!memories.contains(key))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "missing memref fixture for argument %u",
                                       unsigned(index));
      if (args.contains(key))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "memref argument %u must use --memref",
                                       unsigned(index));
      auto tokensOrErr =
          parseMemoryTokens(memories.lookup(key), memrefType.getElementType());
      if (!tokensOrErr)
        return llvm::joinErrors(
            llvm::createStringError(std::errc::invalid_argument,
                                    "invalid memref argument %u",
                                    unsigned(index)),
            tokensOrErr.takeError());
      state.memories[arg] = std::make_shared<MemoryValue>(
          MemoryValue{memrefType.getElementType(), std::move(*tokensOrErr)});
      continue;
    }

    if (memories.contains(key)) {
      if (args.contains(key))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "memory fixture argument %u must not also use --arg",
            unsigned(index));
      state.rawMemoryFixtures[arg] = memories.lookup(key);
      continue;
    }
    auto argIt = args.find(key);
    if (argIt == args.end())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "missing runtime argument %u",
                                     unsigned(index));
    for (llvm::StringRef rawToken : argIt->second) {
      auto tokenOrErr = parseRuntimeToken(rawToken, arg.getType());
      if (!tokenOrErr)
        return llvm::joinErrors(
            llvm::createStringError(std::errc::invalid_argument,
                                    "invalid argument %u", unsigned(index)),
            tokenOrErr.takeError());
      seedBlockArgument(state, arg, *tokenOrErr);
    }
  }

  if (llvm::Error err = propagateMemoryAliases(entry, state))
    return std::move(err);

  llvm::StringSet<> unsupported;
  for (mlir::Operation &op : entry.getOperations()) {
    if (auto name = unsupportedOperation(&op))
      unsupported.insert(*name);
  }
  if (!unsupported.empty()) {
    report.status = "unsupported";
    for (const auto &entry : unsupported)
      report.diagnostics.push_back("unsupported op: " + entry.getKey().str());
    return report;
  }

  for (std::uint64_t step = 0; step < options.maxEventSteps; ++step) {
    bool fired = false;
    for (mlir::Operation &op : entry.getOperations()) {
      if (isSupportedNonEvent(&op))
        continue;
      fired |= fireOperation(&op, state);
    }
    if (!fired)
      break;
    flushPendingTokens(state);
    ++report.wavefrontSteps;
  }
  if (report.wavefrontSteps == options.maxEventSteps) {
    report.status = "blocked";
    report.diagnostics.push_back("maximum optimistic event steps reached");
  }

  bool missingReturn = false;
  report.dynamicWorkItems = dynamicWorkItems(state);
  for (mlir::Value value : returnValues) {
    auto it = state.observedOutputs.find(value);
    if (it == state.observedOutputs.end() || it->second.empty()) {
      report.finalOutputs.push_back("missing");
      missingReturn = true;
      continue;
    }
    if (report.dynamicWorkItems > 1 &&
        requiresCompleteDynamicReturn(value) &&
        it->second.size() < report.dynamicWorkItems) {
      missingReturn = true;
      state.diagnostics.push_back(llvm::formatv(
                                      "dataflow.graph.return value produced "
                                      "{0} of {1} dynamic work items",
                                      it->second.size(), report.dynamicWorkItems)
                                      .str());
    }
    report.finalOutputs.push_back(
        tokenToString(it->second.back(), value.getType()));
  }
  const bool incompleteLoads = hasIncompleteStreamLoads(entry, state);
  if (report.status == "pass" && !state.diagnostics.empty()) {
    report.status = "blocked";
    report.diagnostics.push_back("DFG-sim stopped with runtime diagnostics");
  }
  if (report.status == "pass" && (missingReturn || incompleteLoads)) {
    report.status = "blocked";
    report.diagnostics.push_back(
        "DFG-sim stopped before all returned values produced complete outputs");
  } else if (report.status == "blocked" && (missingReturn || incompleteLoads)) {
    report.diagnostics.push_back(
        "DFG-sim stopped before all returned values produced complete outputs");
  }
  report.eventCount = state.eventCount;
  report.operationFireCounts = state.operationFireCounts;
  report.optimisticCycles =
      estimateDynamicPipelineCycles(state.operationFireCounts, state.diagnostics);
  report.diagnostics.append(state.diagnostics.begin(), state.diagnostics.end());
  return report;
}

llvm::Error
loom::sim::writeDFGSimulationReportJson(llvm::StringRef outputPath,
                                        const DFGSimulationReport &report) {
  llvm::SmallString<256> parent(outputPath);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty()) {
    if (std::error_code ec = llvm::sys::fs::create_directories(parent))
      return llvm::createStringError(ec, "could not create %s", parent.c_str());
  }

  llvm::json::Object root;
  root["schema_version"] = report.schemaVersion;
  root["kind"] = report.kind;
  root["workload"] = report.workload;
  root["graph"] = report.graph;
  root["status"] = report.status;
  root["metric_definition"] = report.metricDefinition;
  root["operation_semantics_source"] = report.operationSemanticsSource;
  root["operation_cost_model_source"] = report.operationCostModelSource;
  root["optimistic_cycles"] = report.optimisticCycles;
  root["wavefront_steps"] = report.wavefrontSteps;
  root["event_count"] = report.eventCount;
  root["dynamic_work_items"] = report.dynamicWorkItems;

  llvm::json::Object fireCounts;
  for (const auto &[opName, count] : report.operationFireCounts)
    fireCounts[opName] = count;
  root["operation_fire_counts"] = std::move(fireCounts);

  llvm::json::Array outputs;
  for (const std::string &value : report.finalOutputs)
    outputs.push_back(value);
  root["final_outputs"] = std::move(outputs);

  llvm::json::Array diagnostics;
  for (const std::string &diagnostic : report.diagnostics)
    diagnostics.push_back(diagnostic);
  root["diagnostics"] = std::move(diagnostics);

  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return llvm::createStringError(ec, "could not open %s",
                                   outputPath.str().c_str());
  out << llvm::formatv("{0:2}", llvm::json::Value(std::move(root))) << "\n";
  return llvm::Error::success();
}

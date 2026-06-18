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

struct MemoryValue;

struct MemoryView {
  std::shared_ptr<MemoryValue> memory;
  mlir::Value root;
  std::int64_t byteOffset = 0;
};

enum class TokenKind { None, Integer, Float, Bool, Pointer };

struct Token {
  TokenKind kind = TokenKind::None;
  std::int64_t intValue = 0;
  double floatValue = 0.0;
  bool boolValue = false;
  MemoryView pointer;
};

Token noneToken() { return Token{}; }

Token integerValueToken(std::int64_t value) {
  Token token;
  token.kind = TokenKind::Integer;
  token.intValue = value;
  return token;
}

Token floatValueToken(double value) {
  Token token;
  token.kind = TokenKind::Float;
  token.floatValue = value;
  return token;
}

Token boolValueToken(bool value) {
  Token token;
  token.kind = TokenKind::Bool;
  token.boolValue = value;
  return token;
}

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
  if (token.kind == TokenKind::Pointer)
    return typePrefix(type) + ":ptr+" + std::to_string(token.pointer.byteOffset);
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << typePrefix(type) << ':';
  if (token.floatValue == 0.0 && std::signbit(token.floatValue))
    os << "-0";
  else if (std::floor(token.floatValue) == token.floatValue)
    os << static_cast<std::int64_t>(token.floatValue);
  else
    os << llvm::formatv("{0:f6}", token.floatValue);
  return os.str();
}

llvm::Expected<Token> tokenFromTypedAttr(mlir::TypedAttr attr) {
  if (mlir::isa<mlir::NoneType>(attr.getType()))
    return noneToken();
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    if (intAttr.getType().isInteger(1))
      return boolValueToken(intAttr.getValue().isOne());
    return integerValueToken(intAttr.getValue().getSExtValue());
  }
  if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr))
    return floatValueToken(floatAttr.getValueAsDouble());
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported dataflow.constant attribute");
}

llvm::Expected<Token> parseRuntimeToken(llvm::StringRef raw, mlir::Type type) {
  raw = raw.trim();
  if (mlir::isa<mlir::NoneType>(type)) {
    if (raw == "none")
      return noneToken();
    return llvm::createStringError(std::errc::invalid_argument,
                                   "none argument expects value 'none'");
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (intType.getWidth() == 1) {
      if (raw == "true" || raw == "1")
        return boolValueToken(true);
      if (raw == "false" || raw == "0")
        return boolValueToken(false);
      return llvm::createStringError(std::errc::invalid_argument,
                                     "i1 argument expects true/false/0/1");
    }
    std::int64_t value = 0;
    if (raw.getAsInteger(10, value))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "integer argument is not base-10");
    return integerValueToken(value);
  }
  if (mlir::isa<mlir::IndexType>(type)) {
    std::int64_t value = 0;
    if (raw.getAsInteger(10, value))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "index argument is not base-10");
    return integerValueToken(value);
  }
  if (mlir::isa<mlir::FloatType>(type)) {
    double value = 0.0;
    if (raw.getAsDouble(value))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "float argument is not parseable");
    return floatValueToken(value);
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

std::int64_t integerToken(const Token &token);

std::string typeToString(mlir::Type type) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  type.print(os);
  return os.str();
}

llvm::Expected<std::int64_t> byteSizeOfType(mlir::Type type) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return std::max<std::int64_t>(1, (intType.getWidth() + 7) / 8);
  if (mlir::isa<mlir::IndexType>(type))
    return 8;
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floatType.isF16())
      return 2;
    if (floatType.isF32())
      return 4;
    if (floatType.isF64())
      return 8;
  }
  if (auto arrayType = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type)) {
    auto elementSizeOrErr = byteSizeOfType(arrayType.getElementType());
    if (!elementSizeOrErr)
      return elementSizeOrErr.takeError();
    return static_cast<std::int64_t>(arrayType.getNumElements()) *
           *elementSizeOrErr;
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported llvm.getelementptr element type: %s",
                                 typeToString(type).c_str());
}

llvm::Expected<std::shared_ptr<MemoryValue>>
materializeMemory(SimulatorState &state, mlir::Value root, llvm::StringRef raw,
                  mlir::Type elementType) {
  auto existing = state.memories.find(root);
  if (existing != state.memories.end()) {
    if (existing->second->elementType != elementType)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "memory fixture type mismatch: existing %s, requested %s",
          typeToString(existing->second->elementType).c_str(),
          typeToString(elementType).c_str());
    return existing->second;
  }
  auto tokensOrErr = parseMemoryTokens(raw, elementType);
  if (!tokensOrErr)
    return tokensOrErr.takeError();
  auto memory = std::make_shared<MemoryValue>(
      MemoryValue{elementType, std::move(*tokensOrErr)});
  state.memories[root] = memory;
  return memory;
}

Token pointerToken(mlir::Value root, std::shared_ptr<MemoryValue> memory = {},
                   std::int64_t byteOffset = 0) {
  Token token;
  token.kind = TokenKind::Pointer;
  token.pointer = MemoryView{std::move(memory), root, byteOffset};
  return token;
}

llvm::Expected<Token> ensurePointerMemory(SimulatorState &state, Token token,
                                          mlir::Type elementType) {
  if (token.kind != TokenKind::Pointer)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "memory view operand is not a pointer");
  if (token.pointer.memory) {
    if (token.pointer.memory->elementType != elementType)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "memory view type mismatch: existing %s, requested %s",
          typeToString(token.pointer.memory->elementType).c_str(),
          typeToString(elementType).c_str());
    return token;
  }
  auto rawIt = state.rawMemoryFixtures.find(token.pointer.root);
  if (rawIt == state.rawMemoryFixtures.end())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "pointer memory fixture is missing");
  auto memoryOrErr =
      materializeMemory(state, token.pointer.root, rawIt->second, elementType);
  if (!memoryOrErr)
    return memoryOrErr.takeError();
  token.pointer.memory = *memoryOrErr;
  return token;
}

llvm::Expected<std::int64_t>
gepByteOffset(mlir::LLVM::GEPOp op, llvm::ArrayRef<Token> dynamicTokens) {
  mlir::Type currentType = op.getElemType();
  std::int64_t offset = 0;
  unsigned dynamicIndex = 0;
  bool firstIndex = true;
  for (std::int32_t rawIndex : op.getRawConstantIndices()) {
    mlir::Type strideType = currentType;
    if (!firstIndex) {
      if (auto arrayType =
              mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(currentType)) {
        strideType = arrayType.getElementType();
      } else {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "unsupported llvm.getelementptr aggregate index over type: %s",
            typeToString(currentType).c_str());
      }
    }
    auto strideOrErr = byteSizeOfType(strideType);
    if (!strideOrErr)
      return strideOrErr.takeError();
    std::int64_t index = rawIndex;
    if (rawIndex == mlir::LLVM::GEPOp::kDynamicIndex) {
      if (dynamicIndex >= dynamicTokens.size())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "llvm.getelementptr dynamic index is missing");
      const Token &token = dynamicTokens[dynamicIndex++];
      if (token.kind != TokenKind::Integer && token.kind != TokenKind::Bool)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "llvm.getelementptr dynamic index must be integer-like");
      index = integerToken(token);
    }
    offset += index * *strideOrErr;
    currentType = strideType;
    firstIndex = false;
  }
  return offset;
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

std::int64_t integerToken(const Token &token) {
  if (token.kind == TokenKind::Bool)
    return token.boolValue ? 1 : 0;
  return token.intValue;
}

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
  case TokenKind::Pointer:
    return PrimitiveValue::none();
  }
  return PrimitiveValue::none();
}

Token tokenFromPrimitiveValue(const PrimitiveValue &value) {
  switch (value.kind) {
  case PrimitiveValueKind::None:
    return noneToken();
  case PrimitiveValueKind::Integer:
    return integerValueToken(value.intValue);
  case PrimitiveValueKind::Float:
    return floatValueToken(value.floatValue);
  case PrimitiveValueKind::Bool:
    return boolValueToken(value.boolValue);
  }
  return noneToken();
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

std::string primitiveOperationName(mlir::Operation *op) {
  if (auto intrinsic = mlir::dyn_cast<mlir::LLVM::CallIntrinsicOp>(op))
    return intrinsic.getIntrin().str();
  return op->getName().getStringRef().str();
}

PrimitiveOperationDescriptor primitiveDescriptor(mlir::Operation *op,
                                                 llvm::StringRef predicate,
                                                 mlir::Value result) {
  std::string opName = primitiveOperationName(op);
  PrimitiveOperationDescriptor descriptor{
      opName,
      predicate,
      integerBitWidth(result.getType()),
      integerBitWidth(op->getOperand(0).getType())};
  if (auto div = mlir::dyn_cast<mlir::arith::DivSIOp>(op))
    descriptor.isExact = div.getIsExact();
  if (auto shift = mlir::dyn_cast<mlir::arith::ShRSIOp>(op))
    descriptor.isExact = shift.getIsExact();
  if (auto shift = mlir::dyn_cast<mlir::arith::ShRUIOp>(op))
    descriptor.isExact = shift.getIsExact();
  if (auto trunc = mlir::dyn_cast<mlir::arith::TruncIOp>(op)) {
    mlir::arith::IntegerOverflowFlags flags = trunc.getOverflowFlags();
    descriptor.noSignedWrap = mlir::arith::bitEnumContainsAll(
        flags, mlir::arith::IntegerOverflowFlags::nsw);
    descriptor.noUnsignedWrap = mlir::arith::bitEnumContainsAll(
        flags, mlir::arith::IntegerOverflowFlags::nuw);
  }
  return descriptor;
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
  emitToken(state, op.getIndex(), integerValueToken(stream.current));
  emitToken(state, op.getRwc(), boolValueToken(cont));
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
    emitToken(state, op.getAfterCond(), boolValueToken(true));
    emitToken(state, op.getAfterValue(), value);
  } else {
    emitToken(state, op.getAfterCond(), boolValueToken(false));
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

bool fireCast(mlir::UnrealizedConversionCastOp op, SimulatorState &state) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  mlir::OpOperand &operand = op->getOpOperand(0);
  if (!hasToken(state.channels, operand))
    return false;
  Token token = popToken(state.channels, operand);
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(op.getResult(0).getType())) {
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

bool fireGEP(mlir::LLVM::GEPOp op, SimulatorState &state) {
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

std::optional<MemoryView> resolveMemoryView(SimulatorState &state,
                                            mlir::Value mem,
                                            mlir::OpOperand &memOperand) {
  if (hasToken(state.channels, memOperand)) {
    Token token = popToken(state.channels, memOperand);
    if (token.kind != TokenKind::Pointer || !token.pointer.memory) {
      state.diagnostics.push_back("dataflow memory operand is not a memory view");
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
                                               llvm::StringRef opName) {
  auto elementSizeOrErr = byteSizeOfType(view.memory->elementType);
  if (!elementSizeOrErr) {
    state.diagnostics.push_back(llvm::toString(elementSizeOrErr.takeError()));
    return std::nullopt;
  }
  if (*elementSizeOrErr == 0 || view.byteOffset % *elementSizeOrErr != 0) {
    state.diagnostics.push_back("memory view byte offset is not element-aligned");
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

bool fireLoad(dataflow::LoadOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getAddrMutable()) ||
      !hasToken(state.channels, op.getCtrlMutable()))
    return false;
  std::optional<MemoryView> view =
      resolveMemoryView(state, op.getMem(), op.getMemMutable());
  if (!view)
    return false;
  Token addr = popToken(state.channels, op.getAddrMutable());
  popToken(state.channels, op.getCtrlMutable());
  std::optional<std::size_t> index =
      resolveElementIndex(*view, addr, state, "dataflow.load");
  if (!index)
    return false;
  emitToken(state, op.getData(), view->memory->elements[*index]);
  emitToken(state, op.getDone(), noneToken());
  ++state.loadFireCounts[op.getOperation()];
  return recordEvent(state, op->getName().getStringRef());
}

bool fireLLVMLoad(mlir::LLVM::LoadOp op, SimulatorState &state) {
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
                          "llvm.load");
  if (!index)
    return false;
  emitToken(state, op->getResult(0),
            viewOrErr->pointer.memory->elements[*index]);
  return recordEvent(state, op->getName().getStringRef());
}

std::optional<std::size_t> resolveByteRangeStart(const MemoryView &view,
                                                 std::int64_t byteLength,
                                                 SimulatorState &state,
                                                 llvm::StringRef opName,
                                                 llvm::StringRef role) {
  if (byteLength < 0) {
    state.diagnostics.push_back((opName + " length is negative").str());
    return std::nullopt;
  }
  if (view.byteOffset < 0) {
    state.diagnostics.push_back(
        (opName + " " + role + " byte offset is negative").str());
    return std::nullopt;
  }
  auto elementSizeOrErr = byteSizeOfType(view.memory->elementType);
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

bool fireLLVMMemcpy(mlir::LLVM::MemcpyOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getDstMutable()) ||
      !hasToken(state.channels, op.getSrcMutable()) ||
      !hasToken(state.channels, op.getLenMutable()))
    return false;
  if (op.getIsVolatile()) {
    state.diagnostics.push_back("volatile llvm.intr.memcpy is unsupported");
    return false;
  }

  Token dst = popToken(state.channels, op.getDstMutable());
  Token src = popToken(state.channels, op.getSrcMutable());
  Token len = popToken(state.channels, op.getLenMutable());
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
      dstOrErr->pointer, byteLength, state, "llvm.intr.memcpy", "destination");
  std::optional<std::size_t> srcStart = resolveByteRangeStart(
      srcOrErr->pointer, byteLength, state, "llvm.intr.memcpy", "source");
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

bool fireStore(dataflow::StoreOp op, SimulatorState &state) {
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
  std::optional<std::size_t> index =
      resolveElementIndex(*view, addr, state, "dataflow.store");
  if (!index)
    return false;
  view->memory->elements[*index] = data;
  emitToken(state, op.getDone(), noneToken());
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
  std::string predicate = primitivePredicate(op);
  auto valueOrErr =
      evaluatePrimitiveOperation(primitiveDescriptor(op, predicate, result),
                                 operands);
  if (!valueOrErr) {
    state.diagnostics.push_back(llvm::toString(valueOrErr.takeError()));
    return false;
  }
  emitToken(state, result, tokenFromPrimitiveValue(*valueOrErr));
  return recordEvent(state, primitiveOperationName(op));
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
  if (op->getNumResults() != 1)
    return false;
  if (!isSupportedPrimitiveOperation(primitiveOperationName(op)))
    return false;
  return firePrimitiveOperation(op, op->getResult(0), state);
}

bool isSupportedNonEvent(mlir::Operation *op) {
  return mlir::isa<dataflow::GraphReturnOp>(op);
}

void collectStreamIndexSources(mlir::Value value,
                               llvm::DenseSet<mlir::Operation *> &sources,
                               llvm::DenseSet<mlir::Value> &seen,
                               unsigned depth = 0) {
  if (!value || depth > 8 || !seen.insert(value).second)
    return;
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    return collectStreamIndexSources(cast.getIn(), sources, seen, depth + 1);
  if (auto stream = value.getDefiningOp<dataflow::StreamOp>()) {
    if (stream.getIndex() == value || stream.getRwc() == value)
      sources.insert(stream.getOperation());
    return;
  }
  auto carry = value.getDefiningOp<dataflow::CarryOp>();
  if (carry && carry.getOutput() == value) {
    collectStreamIndexSources(carry->getOperand(0), sources, seen, depth + 1);
    return;
  }
  auto invariant = value.getDefiningOp<dataflow::InvariantOp>();
  if (invariant && invariant.getOutput() == value) {
    collectStreamIndexSources(invariant.getCond(), sources, seen, depth + 1);
    return;
  }
  auto gate = value.getDefiningOp<dataflow::GateOp>();
  if (gate &&
      (gate.getAfterValue() == value || gate.getAfterCond() == value)) {
    collectStreamIndexSources(gate.getBeforeCond(), sources, seen, depth + 1);
    return;
  }
  mlir::Operation *owner = value.getDefiningOp();
  if (!owner)
    return;
  if (!mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
                 mlir::arith::MulIOp, mlir::arith::DivSIOp,
                 mlir::arith::RemSIOp>(owner))
    return;
  for (mlir::Value operand : owner->getOperands())
    collectStreamIndexSources(operand, sources, seen, depth + 1);
}

dataflow::StreamOp findStreamIndexSource(mlir::Value value) {
  llvm::DenseSet<mlir::Operation *> sources;
  llvm::DenseSet<mlir::Value> seen;
  collectStreamIndexSources(value, sources, seen);
  if (sources.size() != 1)
    return {};
  return mlir::cast<dataflow::StreamOp>(*sources.begin());
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
      .Case<mlir::UnrealizedConversionCastOp>(
          [&](auto typedOp) { return fireCast(typedOp, state); })
      .Case<mlir::LLVM::GEPOp>(
          [&](auto typedOp) { return fireGEP(typedOp, state); })
      .Case<mlir::LLVM::LoadOp>(
          [&](auto typedOp) { return fireLLVMLoad(typedOp, state); })
      .Case<mlir::LLVM::MemcpyOp>(
          [&](auto typedOp) { return fireLLVMMemcpy(typedOp, state); })
      .Case<mlir::arith::ConstantOp>(
          [&](auto typedOp) { return fireArithConstant(typedOp, state); })
      .Default([&](mlir::Operation *genericOp) {
        return fireGenericPrimitive(genericOp, state);
      });
}

std::optional<std::string> unsupportedOperation(mlir::Operation *op) {
  if (isSupportedNonEvent(op))
    return std::nullopt;
  if (op->getNumResults() == 1 &&
      isSupportedPrimitiveOperation(primitiveOperationName(op)))
    return std::nullopt;
  if (mlir::isa<dataflow::StreamOp, dataflow::ConstantOp, dataflow::CarryOp,
                dataflow::InvariantOp, dataflow::GateOp, dataflow::SyncOp,
                dataflow::LoadOp, dataflow::StoreOp,
                mlir::UnrealizedConversionCastOp, mlir::LLVM::GEPOp,
                mlir::LLVM::LoadOp, mlir::LLVM::MemcpyOp,
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

bool hasDirectLLVMAddressUse(mlir::BlockArgument arg) {
  for (mlir::OpOperand &use : arg.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (mlir::isa<mlir::LLVM::LoadOp>(owner) ||
        mlir::isa<mlir::LLVM::GEPOp>(owner) ||
        mlir::isa<mlir::LLVM::MemcpyOp>(owner))
      return true;
  }
  return false;
}

void broadcastRawPointerArguments(mlir::Block &entry, SimulatorState &state) {
  std::uint64_t targetCount = 0;
  for (const auto &seeded : state.seededTokenCounts)
    targetCount = std::max(targetCount, seeded.second);
  if (targetCount <= 1)
    return;

  for (mlir::BlockArgument arg : entry.getArguments()) {
    if (!mlir::isa<mlir::LLVM::LLVMPointerType>(arg.getType()))
      continue;
    if (!state.rawMemoryFixtures.contains(arg) || !hasDirectLLVMAddressUse(arg))
      continue;
    std::uint64_t current = state.seededTokenCounts[arg];
    while (current < targetCount) {
      seedBlockArgument(state, arg, pointerToken(arg));
      ++current;
    }
  }
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
      auto memoryOrErr = materializeMemory(state, source, rawIt->second,
                                           targetMemref.getElementType());
      if (!memoryOrErr)
        return memoryOrErr.takeError();
      auto memory = *memoryOrErr;
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

llvm::SmallVector<std::string> serializeMemoryValue(const MemoryValue &memory) {
  llvm::SmallVector<std::string> values;
  for (const Token &token : memory.elements)
    values.push_back(tokenToString(token, memory.elementType));
  return values;
}

void captureFinalMemoryState(mlir::Block &entry, SimulatorState &state,
                             DFGSimulationReport &report) {
  for (auto [index, arg] : llvm::enumerate(entry.getArguments())) {
    auto memory = state.memories.find(arg);
    if (memory == state.memories.end())
      continue;
    report.finalMemoryState[llvm::formatv("arg{0}", index).str()] =
        serializeMemoryValue(*memory->second);
  }
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
      if (!mlir::isa<mlir::LLVM::LLVMPointerType>(arg.getType()))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "memory fixture argument %u must be memref or !llvm.ptr",
            unsigned(index));
      state.rawMemoryFixtures[arg] = memories.lookup(key);
      seedBlockArgument(state, arg, pointerToken(arg));
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

  broadcastRawPointerArguments(entry, state);

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
  captureFinalMemoryState(entry, state, report);
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

  llvm::json::Object finalMemoryState;
  for (const auto &[argument, values] : report.finalMemoryState) {
    llvm::json::Array memoryValues;
    for (const std::string &value : values)
      memoryValues.push_back(value);
    finalMemoryState[argument] = std::move(memoryValues);
  }
  root["final_memory_state"] = std::move(finalMemoryState);

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

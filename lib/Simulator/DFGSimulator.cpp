#include "Simulator/DFGSimulator.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
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
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>

using namespace loom::sim;

namespace {

constexpr std::uint64_t kLoadAddressSetupCycles = 1;
constexpr std::uint64_t kStoreAddressSetupCycles = 2;

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

struct ParallelizeState {
  std::uint64_t pointer = 0;
  llvm::SmallVector<std::optional<Token>, 8> slots;
  std::uint64_t mask = 0;
};

struct MemoryValue {
  mlir::Type elementType;
  llvm::SmallVector<Token> elements;
};

struct MemoryFixture {
  std::string values;
  std::int64_t byteOffset = 0;
};

struct SimulatorState {
  ChannelMap channels;
  ChannelMap pendingChannels;
  OutputMap observedOutputs;
  OutputMap pendingObservedOutputs;
  llvm::DenseMap<mlir::Value, std::shared_ptr<MemoryValue>> memories;
  llvm::DenseMap<mlir::Value, MemoryFixture> rawMemoryFixtures;
  llvm::DenseMap<mlir::Operation *, StreamState> streamStates;
  llvm::DenseMap<mlir::Operation *, LoopState> carryStates;
  llvm::DenseMap<mlir::Operation *, LoopState> invariantStates;
  llvm::DenseMap<mlir::Operation *, ParallelizeState> parallelizeStates;
  llvm::DenseSet<mlir::Operation *> gateContinueStates;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> loadFireCounts;
  llvm::DenseSet<mlir::Operation *> oneShotOps;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> structuredEffectFireCounts;
  llvm::DenseMap<mlir::Value, std::uint64_t> seededTokenCounts;
  llvm::SmallVector<std::string> diagnostics;
  std::map<std::string, std::uint64_t> operationFireCounts;
  std::uint64_t eventCount = 0;
  std::uint64_t memoryAddressSetupCycles = 0;
  std::uint64_t structuredLoopIterations = 0;
  std::uint64_t maxStructuredLoopIterations = 0;
};

using MemoryCloneMap =
    llvm::DenseMap<const MemoryValue *, std::shared_ptr<MemoryValue>>;

std::shared_ptr<MemoryValue>
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

void retargetTokenMemory(Token &token, MemoryCloneMap &clones) {
  if (token.kind != TokenKind::Pointer || !token.pointer.memory)
    return;
  token.pointer.memory = cloneMemoryHandle(token.pointer.memory, clones);
}

void retargetChannelMap(ChannelMap &channels, MemoryCloneMap &clones) {
  for (auto &entry : channels)
    for (Token &token : entry.second)
      retargetTokenMemory(token, clones);
}

void retargetOutputMap(OutputMap &outputs, MemoryCloneMap &clones) {
  for (auto &entry : outputs)
    for (Token &token : entry.second)
      retargetTokenMemory(token, clones);
}

void retargetTokenVector(llvm::SmallVectorImpl<Token> &tokens,
                         MemoryCloneMap &clones) {
  for (Token &token : tokens)
    retargetTokenMemory(token, clones);
}

void retargetLoopStates(llvm::DenseMap<mlir::Operation *, LoopState> &states,
                        MemoryCloneMap &clones) {
  for (auto &entry : states)
    if (entry.second.latched)
      retargetTokenMemory(*entry.second.latched, clones);
}

MemoryCloneMap isolateProbeStateMemory(SimulatorState &state) {
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

void appendProbeDiagnostics(SimulatorState &state,
                            const SimulatorState &probeState) {
  for (const std::string &diagnostic : probeState.diagnostics) {
    if (std::find(state.diagnostics.begin(), state.diagnostics.end(),
                  diagnostic) != state.diagnostics.end())
      continue;
    state.diagnostics.push_back(diagnostic);
  }
}

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
    return typePrefix(type) + ":ptr+" +
           std::to_string(token.pointer.byteOffset);
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
  return llvm::createStringError(
      std::errc::invalid_argument,
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

llvm::Expected<Token> zeroToken(mlir::Type type) {
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(type))
    return pointerToken(mlir::Value{});
  if (mlir::isa<mlir::IndexType>(type))
    return integerValueToken(0);
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (intType.getWidth() == 1)
      return boolValueToken(false);
    return integerValueToken(0);
  }
  if (mlir::isa<mlir::FloatType>(type))
    return floatValueToken(0.0);
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported llvm.mlir.zero type: %s",
                                 typeToString(type).c_str());
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
  auto memoryOrErr = materializeMemory(state, token.pointer.root,
                                       rawIt->second.values, elementType);
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
        return llvm::createStringError(
            std::errc::invalid_argument,
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

Token peekToken(ChannelMap &channels, mlir::OpOperand &operand) {
  return channels[&operand].front();
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

bool hasComputedAddress(mlir::Value value) {
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  return def->getName().getStringRef() != "dataflow.stream";
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

std::uint64_t nonStructuredDynamicWorkItems(const SimulatorState &state) {
  std::uint64_t maxStreamItems = 0;
  for (const auto &entry : state.streamStates)
    maxStreamItems = std::max(maxStreamItems, entry.second.trueEmissions);
  std::uint64_t maxSeededItems = 0;
  for (const auto &entry : state.seededTokenCounts)
    maxSeededItems = std::max(maxSeededItems, entry.second);
  return std::max(maxStreamItems, maxSeededItems);
}

std::uint64_t dynamicWorkItems(const SimulatorState &state) {
  const std::uint64_t workItems = std::max(nonStructuredDynamicWorkItems(state),
                                           state.structuredLoopIterations);
  if (workItems == 0 && state.eventCount > 0)
    return 1;
  return workItems;
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

bool samePointer(const MemoryView &lhs, const MemoryView &rhs) {
  return lhs.root == rhs.root && lhs.byteOffset == rhs.byteOffset;
}

std::optional<std::size_t> resolveElementIndex(const MemoryView &view,
                                               const Token &addr,
                                               SimulatorState &state,
                                               llvm::StringRef opName);

constexpr llvm::StringLiteral kCmsisNNVecMatMultTS8 =
    "arm_nn_vec_mat_mult_t_s8";

bool isSupportedLLVMCall(mlir::LLVM::CallOp op) {
  auto callee = op.getCallee();
  if (!callee || *callee != kCmsisNNVecMatMultTS8)
    return false;
  if (op->getNumOperands() != 15 || op->getNumResults() != 1)
    return false;
  for (unsigned i = 0; i < 5; ++i)
    if (!mlir::isa<mlir::LLVM::LLVMPointerType>(op->getOperand(i).getType()))
      return false;
  return mlir::isa<mlir::IntegerType>(op->getResult(0).getType());
}

bool isNullPointerToken(const Token &token) {
  return token.kind == TokenKind::Pointer && !token.pointer.root &&
         !token.pointer.memory;
}

std::int64_t signExtend(std::int64_t value, unsigned width) {
  if (width == 0 || width >= 64)
    return value;
  const std::uint64_t mask = (std::uint64_t{1} << width) - 1;
  std::uint64_t bits = static_cast<std::uint64_t>(value) & mask;
  const std::uint64_t signBit = std::uint64_t{1} << (width - 1);
  if ((bits & signBit) == 0)
    return static_cast<std::int64_t>(bits);
  return static_cast<std::int64_t>(bits | ~mask);
}

std::int32_t wrapI32(std::int64_t value) {
  return static_cast<std::int32_t>(static_cast<std::uint32_t>(value));
}

std::int32_t doublingHighMultNoSat(std::int32_t lhs, std::int32_t rhs) {
  std::int64_t product =
      (std::int64_t{1} << 30) + static_cast<std::int64_t>(lhs) * rhs;
  return wrapI32(product >> 31);
}

std::int32_t divideByPowerOfTwo(std::int32_t dividend, std::int32_t exponent) {
  if (exponent <= 0)
    return dividend;
  if (exponent >= 31)
    exponent = 30;
  const std::int32_t remainderMask =
      static_cast<std::int32_t>((std::int64_t{1} << exponent) - 1);
  const std::int32_t remainder = remainderMask & dividend;
  std::int32_t result = dividend >> exponent;
  std::int32_t threshold = remainderMask >> 1;
  if (result < 0)
    ++threshold;
  if (remainder > threshold)
    ++result;
  return result;
}

std::int32_t cmsisRequantize(std::int32_t value, std::int32_t multiplier,
                             std::int32_t shift) {
  const std::int32_t leftShift = shift > 0 ? shift : 0;
  const std::int32_t rightShift = shift > 0 ? 0 : -shift;
  std::int64_t shifted = static_cast<std::int64_t>(value)
                         << static_cast<unsigned>(leftShift);
  std::int32_t multiplied =
      doublingHighMultNoSat(wrapI32(shifted), multiplier);
  return divideByPowerOfTwo(multiplied, rightShift);
}

std::optional<std::int64_t>
integerOperand(llvm::ArrayRef<Token> operands, unsigned index,
               llvm::StringRef name, SimulatorState &state) {
  if (index >= operands.size()) {
    state.diagnostics.push_back((name + " operand is missing").str());
    return std::nullopt;
  }
  const Token &token = operands[index];
  if (token.kind != TokenKind::Integer && token.kind != TokenKind::Bool) {
    state.diagnostics.push_back((name + " operand is not integer-like").str());
    return std::nullopt;
  }
  return integerToken(token);
}

std::optional<std::int64_t>
loadIntegerPointerElement(SimulatorState &state, const Token &ptr,
                          mlir::Type elementType, std::int64_t elementOffset,
                          unsigned signedWidth, llvm::StringRef opName) {
  auto viewOrErr = ensurePointerMemory(state, ptr, elementType);
  if (!viewOrErr) {
    state.diagnostics.push_back(llvm::toString(viewOrErr.takeError()));
    return std::nullopt;
  }
  std::optional<std::size_t> index = resolveElementIndex(
      viewOrErr->pointer, integerValueToken(elementOffset), state, opName);
  if (!index)
    return std::nullopt;
  if (!recordEvent(state, "llvm.load"))
    return std::nullopt;
  return signExtend(integerToken(viewOrErr->pointer.memory->elements[*index]),
                    signedWidth);
}

bool storeIntegerPointerElement(SimulatorState &state, const Token &ptr,
                                mlir::Type elementType,
                                std::int64_t elementOffset,
                                std::int64_t value, llvm::StringRef opName) {
  auto viewOrErr = ensurePointerMemory(state, ptr, elementType);
  if (!viewOrErr) {
    state.diagnostics.push_back(llvm::toString(viewOrErr.takeError()));
    return false;
  }
  std::optional<std::size_t> index = resolveElementIndex(
      viewOrErr->pointer, integerValueToken(elementOffset), state, opName);
  if (!index)
    return false;
  viewOrErr->pointer.memory->elements[*index] =
      integerValueToken(signExtend(value, 8));
  return recordEvent(state, "llvm.store");
}

bool recordCmsisArithmetic(SimulatorState &state, llvm::StringRef opName) {
  return recordEvent(state, opName);
}

bool executeCmsisNNVecMatMultTS8(mlir::LLVM::CallOp op, SimulatorState &state,
                                 llvm::ArrayRef<Token> operands,
                                 Token &result) {
  if (operands.size() != 15) {
    state.diagnostics.push_back(
        "arm_nn_vec_mat_mult_t_s8 expects 15 operands");
    return false;
  }

  std::optional<std::int64_t> lhsOffset =
      integerOperand(operands, 5, "lhs_offset", state);
  std::optional<std::int64_t> dstOffset =
      integerOperand(operands, 6, "dst_offset", state);
  std::optional<std::int64_t> dstMultiplier =
      integerOperand(operands, 7, "dst_multiplier", state);
  std::optional<std::int64_t> dstShift =
      integerOperand(operands, 8, "dst_shift", state);
  std::optional<std::int64_t> rhsCols =
      integerOperand(operands, 9, "rhs_cols", state);
  std::optional<std::int64_t> rhsRows =
      integerOperand(operands, 10, "rhs_rows", state);
  std::optional<std::int64_t> activationMin =
      integerOperand(operands, 11, "activation_min", state);
  std::optional<std::int64_t> activationMax =
      integerOperand(operands, 12, "activation_max", state);
  std::optional<std::int64_t> addressOffset =
      integerOperand(operands, 13, "address_offset", state);
  std::optional<std::int64_t> rhsOffset =
      integerOperand(operands, 14, "rhs_offset", state);
  if (!lhsOffset || !dstOffset || !dstMultiplier || !dstShift || !rhsCols ||
      !rhsRows || !activationMin || !activationMax || !addressOffset ||
      !rhsOffset)
    return false;
  if (*rhsCols < 0 || *rhsRows < 0 || *addressOffset <= 0) {
    state.diagnostics.push_back(
        "arm_nn_vec_mat_mult_t_s8 has invalid dimensions");
    return false;
  }

  mlir::Type i8Type = mlir::IntegerType::get(op.getContext(), 8);
  mlir::Type i32Type = mlir::IntegerType::get(op.getContext(), 32);
  const bool hasBias = !isNullPointerToken(operands[3]);

  for (std::int64_t row = 0; row < *rhsRows; ++row) {
    std::int64_t acc = 0;
    if (hasBias) {
      std::optional<std::int64_t> bias = loadIntegerPointerElement(
          state, operands[3], i32Type, row, 32, "arm_nn_vec_mat_mult_t_s8");
      if (!bias)
        return false;
      acc = *bias;
    }
    for (std::int64_t col = 0; col < *rhsCols; ++col) {
      std::optional<std::int64_t> lhsValue = loadIntegerPointerElement(
          state, operands[0], i8Type, col, 8, "arm_nn_vec_mat_mult_t_s8");
      std::optional<std::int64_t> rhsValue = loadIntegerPointerElement(
          state, operands[1], i8Type, row * *rhsCols + col, 8,
          "arm_nn_vec_mat_mult_t_s8");
      if (!lhsValue || !rhsValue)
        return false;
      if (!recordCmsisArithmetic(state, "arith.addi"))
        return false;
      *lhsValue += *lhsOffset;
      if (!recordCmsisArithmetic(state, "arith.addi"))
        return false;
      *rhsValue += *rhsOffset;
      if (!recordCmsisArithmetic(state, "arith.muli"))
        return false;
      const std::int64_t product = *lhsValue * *rhsValue;
      if (!recordCmsisArithmetic(state, "arith.addi"))
        return false;
      acc += product;
    }

    if (!recordCmsisArithmetic(state, "arith.muli") ||
        !recordCmsisArithmetic(state, "arith.shrsi"))
      return false;
    acc = cmsisRequantize(wrapI32(acc), wrapI32(*dstMultiplier),
                          wrapI32(*dstShift));
    if (!recordCmsisArithmetic(state, "arith.addi"))
      return false;
    acc += *dstOffset;
    if (!recordCmsisArithmetic(state, "arith.select"))
      return false;
    acc = std::max(acc, *activationMin);
    if (!recordCmsisArithmetic(state, "arith.select"))
      return false;
    acc = std::min(acc, *activationMax);
    if (!storeIntegerPointerElement(state, operands[4], i8Type,
                                    row * *addressOffset, acc,
                                    "arm_nn_vec_mat_mult_t_s8"))
      return false;
  }

  result = integerValueToken(0);
  return true;
}

bool isSupportedPointerICmp(mlir::LLVM::ICmpOp op) {
  if (!mlir::isa<mlir::LLVM::LLVMPointerType>(op.getLhs().getType()) ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(op.getRhs().getType()))
    return false;
  return op.getPredicate() == mlir::LLVM::ICmpPredicate::eq ||
         op.getPredicate() == mlir::LLVM::ICmpPredicate::ne;
}

llvm::Expected<Token> evaluatePointerICmp(mlir::LLVM::ICmpOp op,
                                          const Token &lhs, const Token &rhs) {
  if (!isSupportedPointerICmp(op))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "llvm.icmp supports only pointer eq/ne in DFG-sim");
  if (lhs.kind != TokenKind::Pointer || rhs.kind != TokenKind::Pointer)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "llvm.icmp operands are not pointers");
  bool equal = samePointer(lhs.pointer, rhs.pointer);
  if (op.getPredicate() == mlir::LLVM::ICmpPredicate::ne)
    equal = !equal;
  return boolValueToken(equal);
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

std::optional<unsigned> vectorSizeAttr(mlir::Operation *op,
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

std::optional<unsigned> signlessIntegerBitWidthForVector(mlir::Type type,
                                                         SimulatorState &state,
                                                         llvm::StringRef op) {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType || !intType.isSignless()) {
    state.diagnostics.push_back(
        (op + " requires signless integer lanes").str());
    return std::nullopt;
  }
  return intType.getWidth();
}

std::uint64_t lowBitsMask(unsigned width) {
  if (width >= 64)
    return ~std::uint64_t{0};
  return (std::uint64_t{1} << width) - 1;
}

std::uint64_t tokenBits(const Token &token, unsigned width) {
  return static_cast<std::uint64_t>(integerToken(token)) & lowBitsMask(width);
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
      opName, predicate, integerBitWidth(result.getType()),
      integerBitWidth(op->getOperand(0).getType())};
  if (auto div = mlir::dyn_cast<mlir::arith::DivSIOp>(op))
    descriptor.isExact = div.getIsExact();
  if (auto div = mlir::dyn_cast<mlir::arith::DivUIOp>(op))
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

bool fireMux(dataflow::MuxOp op, SimulatorState &state) {
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

bool fireDemux(dataflow::DemuxOp op, SimulatorState &state) {
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

void emitParallelizeGroup(dataflow::ParallelizeOp op, SimulatorState &state,
                          ParallelizeState &parallel) {
  for (auto [i, output] : llvm::enumerate(op.getOutputs())) {
    if (i < parallel.slots.size() && parallel.slots[i])
      emitToken(state, output, *parallel.slots[i]);
  }
  emitToken(state, op.getMask(), integerValueToken(parallel.mask));
  parallel.slots.assign(op.getOutputs().size(), std::nullopt);
  parallel.mask = 0;
}

bool fireParallelize(dataflow::ParallelizeOp op, SimulatorState &state) {
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

bool firePack(dataflow::PackOp op, SimulatorState &state) {
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

bool fireUnpack(dataflow::UnpackOp op, SimulatorState &state) {
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

bool fireSerialize(dataflow::SerializeOp op, SimulatorState &state) {
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

bool fireCast(mlir::UnrealizedConversionCastOp op, SimulatorState &state) {
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
                                               llvm::StringRef opName) {
  auto elementSizeOrErr = byteSizeOfType(view.memory->elementType);
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
  if (hasComputedAddress(op.getAddr()))
    state.memoryAddressSetupCycles += kLoadAddressSetupCycles;
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
  std::optional<std::size_t> index = resolveElementIndex(
      viewOrErr->pointer, integerValueToken(0), state, "llvm.load");
  if (!index)
    return false;
  emitToken(state, op->getResult(0),
            viewOrErr->pointer.memory->elements[*index]);
  return recordEvent(state, op->getName().getStringRef());
}

bool fireLLVMStore(mlir::LLVM::StoreOp op, SimulatorState &state) {
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
  std::optional<std::size_t> index = resolveElementIndex(
      viewOrErr->pointer, integerValueToken(0), state, "llvm.store");
  if (!index)
    return false;
  viewOrErr->pointer.memory->elements[*index] = value;
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

bool fireLLVMMemcpy(mlir::LLVM::MemcpyOp op, SimulatorState &state) {
  if (!hasToken(state.channels, op.getDstMutable()) ||
      !hasToken(state.channels, op.getSrcMutable()) ||
      !hasToken(state.channels, op.getLenMutable()))
    return false;

  Token dst = popToken(state.channels, op.getDstMutable());
  Token src = popToken(state.channels, op.getSrcMutable());
  Token len = popToken(state.channels, op.getLenMutable());
  return executeLLVMMemcpy(op, state, dst, src, len);
}

bool fireLLVMCall(mlir::LLVM::CallOp op, SimulatorState &state) {
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
  if (hasComputedAddress(op.getAddr()))
    state.memoryAddressSetupCycles += kStoreAddressSetupCycles;
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
  auto valueOrErr = evaluatePrimitiveOperation(
      primitiveDescriptor(op, predicate, result), operands);
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

bool fireLLVMZero(mlir::LLVM::ZeroOp op, SimulatorState &state) {
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

bool fireUBPoison(mlir::ub::PoisonOp op, SimulatorState &state) {
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

bool fireLLVMICmp(mlir::LLVM::ICmpOp op, SimulatorState &state) {
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

bool fireLLVMSelect(mlir::LLVM::SelectOp op, SimulatorState &state) {
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

bool fireGenericPrimitive(mlir::Operation *op, SimulatorState &state) {
  if (op->getNumResults() != 1)
    return false;
  if (!isSupportedPrimitiveOperation(primitiveOperationName(op)))
    return false;
  return firePrimitiveOperation(op, op->getResult(0), state);
}
using LocalValueMap = llvm::DenseMap<mlir::Value, Token>;

void retargetLocalValueMap(LocalValueMap &locals, MemoryCloneMap &clones) {
  for (auto &entry : locals)
    retargetTokenMemory(entry.second, clones);
}

unsigned observedTokenCount(mlir::Value value, const SimulatorState &state) {
  unsigned count = 0;
  auto observedIt = state.observedOutputs.find(value);
  if (observedIt != state.observedOutputs.end())
    count += observedIt->second.size();
  auto pendingIt = state.pendingObservedOutputs.find(value);
  if (pendingIt != state.pendingObservedOutputs.end())
    count += pendingIt->second.size();
  return count;
}

unsigned structuredOpFireIndex(mlir::Operation *op,
                               const SimulatorState &state) {
  if (op->getNumResults() != 0)
    return observedTokenCount(op->getResult(0), state);
  auto effectIt = state.structuredEffectFireCounts.find(op);
  return effectIt == state.structuredEffectFireCounts.end()
             ? 0
             : static_cast<unsigned>(effectIt->second);
}

void recordStructuredEffectFire(SimulatorState &state, mlir::Operation *op) {
  if (op->getNumResults() == 0)
    ++state.structuredEffectFireCounts[op];
}

bool hasMemRefValue(mlir::ValueRange values) {
  return llvm::any_of(values, [](mlir::Value value) {
    return mlir::isa<mlir::MemRefType>(value.getType());
  });
}

bool isSupportedStructuredCast(mlir::UnrealizedConversionCastOp cast) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return false;
  if (mlir::isa<mlir::MemRefType>(cast.getResult(0).getType()))
    return mlir::isa<mlir::LLVM::LLVMPointerType>(cast.getOperand(0).getType());
  return !hasMemRefValue(cast->getOperands()) &&
         !hasMemRefValue(cast->getResults());
}

bool canBroadcastStructuredForCapture(mlir::Value value) {
  if (mlir::isa<mlir::BlockArgument>(value))
    return true;
  return mlir::isa_and_nonnull<mlir::arith::ConstantOp, dataflow::ConstantOp,
                               mlir::ub::PoisonOp>(value.getDefiningOp());
}
std::optional<Token> lookupToken(mlir::Value value, SimulatorState &state,
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

bool valueAvailableForStructuredRegion(mlir::Value value, SimulatorState &state,
                                       const LocalValueMap &locals,
                                       unsigned captureIndex) {
  if (locals.contains(value))
    return true;
  if (state.memories.contains(value))
    return true;
  return lookupToken(value, state, locals, captureIndex).has_value();
}

bool structuredRegionCapturesAvailable(mlir::Region &region,
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

mlir::Region *selectedIfRegion(mlir::scf::IfOp op, const Token &cond) {
  if (boolToken(cond))
    return &op.getThenRegion();
  if (op.getElseRegion().empty())
    return nullptr;
  return &op.getElseRegion();
}

bool selectedIfCapturesAvailable(mlir::scf::IfOp op, SimulatorState &state,
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

unsigned structuredForFireIndex(mlir::scf::ForOp op,
                                const SimulatorState &state) {
  return structuredOpFireIndex(op.getOperation(), state);
}
bool assignLocalPrimitiveResult(mlir::Operation *op, mlir::Value result,
                                SimulatorState &state, LocalValueMap &locals,
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
  auto valueOrErr = evaluatePrimitiveOperation(
      primitiveDescriptor(op, predicate, result), operands);
  if (!valueOrErr) {
    state.diagnostics.push_back(llvm::toString(valueOrErr.takeError()));
    return false;
  }
  locals[result] = tokenFromPrimitiveValue(*valueOrErr);
  return recordEvent(state, primitiveOperationName(op));
}

bool assignLocalLLVMZero(mlir::LLVM::ZeroOp op, SimulatorState &state,
                         LocalValueMap &locals) {
  auto tokenOrErr = zeroToken(op->getResult(0).getType());
  if (!tokenOrErr) {
    state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
    return false;
  }
  locals[op->getResult(0)] = *tokenOrErr;
  return recordEvent(state, op->getName().getStringRef());
}

bool assignLocalUBPoison(mlir::ub::PoisonOp op, SimulatorState &state,
                         LocalValueMap &locals) {
  auto tokenOrErr = zeroToken(op->getResult(0).getType());
  if (!tokenOrErr) {
    state.diagnostics.push_back(llvm::toString(tokenOrErr.takeError()));
    return false;
  }
  locals[op->getResult(0)] = *tokenOrErr;
  return recordEvent(state, op->getName().getStringRef());
}

bool assignLocalLLVMICmp(mlir::LLVM::ICmpOp op, SimulatorState &state,
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

bool assignLocalLLVMSelect(mlir::LLVM::SelectOp op, SimulatorState &state,
                           LocalValueMap &locals, unsigned captureIndex) {
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

bool assignLocalDataflowConstant(dataflow::ConstantOp op, SimulatorState &state,
                                 LocalValueMap &locals, unsigned captureIndex) {
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

bool assignLocalCast(mlir::UnrealizedConversionCastOp cast,
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

bool assignLocalGEP(mlir::LLVM::GEPOp op, SimulatorState &state,
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

std::optional<MemoryView> lookupLocalMemoryView(mlir::Value mem,
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

bool assignLocalDataflowLoad(dataflow::LoadOp op, SimulatorState &state,
                             LocalValueMap &locals, unsigned captureIndex) {
  std::optional<MemoryView> view =
      lookupLocalMemoryView(op.getMem(), state, locals, captureIndex);
  std::optional<Token> addr =
      lookupToken(op.getAddr(), state, locals, captureIndex);
  std::optional<Token> ctrl =
      lookupToken(op.getCtrl(), state, locals, captureIndex);
  if (!view || !addr || !ctrl)
    return false;
  std::optional<std::size_t> index =
      resolveElementIndex(*view, *addr, state, "dataflow.load");
  if (!index)
    return false;
  locals[op.getData()] = view->memory->elements[*index];
  locals[op.getDone()] = noneToken();
  ++state.loadFireCounts[op.getOperation()];
  if (hasComputedAddress(op.getAddr()))
    state.memoryAddressSetupCycles += kLoadAddressSetupCycles;
  return recordEvent(state, op->getName().getStringRef());
}

bool assignLocalDataflowStore(dataflow::StoreOp op, SimulatorState &state,
                              LocalValueMap &locals, unsigned captureIndex) {
  std::optional<MemoryView> view =
      lookupLocalMemoryView(op.getMem(), state, locals, captureIndex);
  std::optional<Token> addr =
      lookupToken(op.getAddr(), state, locals, captureIndex);
  std::optional<Token> data =
      lookupToken(op.getData(), state, locals, captureIndex);
  std::optional<Token> ctrl =
      lookupToken(op.getCtrl(), state, locals, captureIndex);
  if (!view || !addr || !data || !ctrl)
    return false;
  std::optional<std::size_t> index =
      resolveElementIndex(*view, *addr, state, "dataflow.store");
  if (!index)
    return false;
  view->memory->elements[*index] = *data;
  locals[op.getDone()] = noneToken();
  if (hasComputedAddress(op.getAddr()))
    state.memoryAddressSetupCycles += kStoreAddressSetupCycles;
  return recordEvent(state, op->getName().getStringRef());
}

bool assignLocalLLVMMemcpy(mlir::LLVM::MemcpyOp op, SimulatorState &state,
                           LocalValueMap &locals, unsigned captureIndex) {
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

bool assignLocalLLVMCall(mlir::LLVM::CallOp op, SimulatorState &state,
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

bool assignLocalGate(dataflow::GateOp op, SimulatorState &state,
                     LocalValueMap &locals, unsigned captureIndex) {
  std::optional<Token> cond =
      lookupToken(op.getBeforeCond(), state, locals, captureIndex);
  std::optional<Token> value =
      lookupToken(op.getBeforeValue(), state, locals, captureIndex);
  if (!cond || !value)
    return false;
  const bool isContinue = state.gateContinueStates.contains(op.getOperation());
  const bool open = boolToken(*cond);
  if (!isContinue) {
    if (open) {
      locals[op.getAfterValue()] = *value;
      state.gateContinueStates.insert(op.getOperation());
    }
    return recordEvent(state, op->getName().getStringRef());
  }
  if (open) {
    locals[op.getAfterCond()] = boolValueToken(true);
    locals[op.getAfterValue()] = *value;
  } else {
    locals[op.getAfterCond()] = boolValueToken(false);
    state.gateContinueStates.erase(op.getOperation());
  }
  return recordEvent(state, op->getName().getStringRef());
}

bool assignLocalMux(dataflow::MuxOp op, SimulatorState &state,
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

bool assignLocalDemux(dataflow::DemuxOp op, SimulatorState &state,
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

bool executeStructuredForBodyOp(mlir::Operation *op, SimulatorState &state,
                                LocalValueMap &locals, unsigned captureIndex);
bool executeStructuredFor(mlir::scf::ForOp op, SimulatorState &state,
                          llvm::ArrayRef<Token> operands, unsigned captureIndex,
                          llvm::SmallVectorImpl<Token> &results,
                          const LocalValueMap *captures = nullptr);
bool executeStructuredWhile(mlir::scf::WhileOp op, SimulatorState &state,
                            llvm::ArrayRef<Token> operands,
                            unsigned captureIndex,
                            llvm::SmallVectorImpl<Token> &results,
                            const LocalValueMap *captures = nullptr);
bool executeStructuredForall(mlir::scf::ForallOp op, SimulatorState &state,
                             LocalValueMap &captures,
                             unsigned captureIndex = 0);

bool evaluateStructuredYieldRegion(mlir::Operation *parent, mlir::Block *block,
                                   llvm::StringRef opName,
                                   SimulatorState &state,
                                   LocalValueMap &parentLocals,
                                   unsigned captureIndex,
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

bool evaluateStructuredIf(mlir::scf::IfOp op, SimulatorState &state,
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

bool executeStructuredIfLocally(mlir::scf::IfOp op, SimulatorState &state,
                                LocalValueMap &locals, unsigned captureIndex) {
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

mlir::Block *selectStructuredIndexSwitchBlock(mlir::scf::IndexSwitchOp op,
                                              SimulatorState &state,
                                              LocalValueMap &locals,
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

bool evaluateStructuredIndexSwitch(mlir::scf::IndexSwitchOp op,
                                   SimulatorState &state, LocalValueMap &locals,
                                   unsigned captureIndex,
                                   llvm::SmallVectorImpl<Token> &yielded) {
  mlir::Block *selected =
      selectStructuredIndexSwitchBlock(op, state, locals, captureIndex);
  if (!selected)
    return false;
  return evaluateStructuredYieldRegion(op.getOperation(), selected,
                                       "scf.index_switch", state, locals,
                                       captureIndex, yielded);
}

bool executeStructuredIndexSwitchLocally(mlir::scf::IndexSwitchOp op,
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

bool executeStructuredForBodyOp(mlir::Operation *op, SimulatorState &state,
                                LocalValueMap &locals, unsigned captureIndex) {
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

std::string unsupportedOperationLabel(mlir::Operation *op);

std::optional<std::string> unsupportedStructuredIfOperation(mlir::scf::IfOp op);

std::optional<std::string>
unsupportedStructuredIndexSwitchOperation(mlir::scf::IndexSwitchOp op);

std::optional<std::string>
unsupportedStructuredForOperation(mlir::scf::ForOp op);

std::optional<std::string>
unsupportedStructuredWhileOperation(mlir::scf::WhileOp op);

std::optional<std::string>
unsupportedStructuredForallOperation(mlir::scf::ForallOp op);

std::optional<std::string>
unsupportedStructuredYieldRegion(mlir::Operation *parent, mlir::Block *block,
                                 llvm::StringRef opName) {
  if (!block)
    return parent->getNumResults() == 0
               ? std::nullopt
               : std::optional<std::string>(opName.str());
  auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(block->getTerminator());
  if (!yield || yield.getNumOperands() != parent->getNumResults())
    return "scf.yield";
  for (mlir::Operation &bodyOp : block->without_terminator()) {
    if (mlir::isa<mlir::arith::ConstantOp>(bodyOp))
      continue;
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(bodyOp)) {
      if (auto name = unsupportedStructuredIfOperation(ifOp))
        return name;
      continue;
    }
    if (auto switchOp = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(bodyOp)) {
      if (auto name = unsupportedStructuredIndexSwitchOperation(switchOp))
        return name;
      continue;
    }
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(bodyOp)) {
      if (auto name = unsupportedStructuredForOperation(forOp))
        return name;
      continue;
    }
    if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(bodyOp)) {
      if (auto name = unsupportedStructuredWhileOperation(whileOp))
        return name;
      continue;
    }
    if (auto forallOp = mlir::dyn_cast<mlir::scf::ForallOp>(bodyOp)) {
      if (auto name = unsupportedStructuredForallOperation(forallOp))
        return name;
      continue;
    }
    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(bodyOp)) {
      if (isSupportedStructuredCast(cast))
        continue;
      return unsupportedOperationLabel(cast.getOperation());
    }
    if (auto icmp = mlir::dyn_cast<mlir::LLVM::ICmpOp>(bodyOp)) {
      if (isSupportedPointerICmp(icmp))
        continue;
      return unsupportedOperationLabel(&bodyOp);
    }
    if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(bodyOp)) {
      if (isSupportedLLVMCall(call))
        continue;
      return unsupportedOperationLabel(&bodyOp);
    }
    if (mlir::isa<dataflow::ConstantOp, dataflow::LoadOp, dataflow::StoreOp,
                  dataflow::GateOp, dataflow::MuxOp, dataflow::DemuxOp,
                  mlir::LLVM::GEPOp, mlir::LLVM::ZeroOp, mlir::LLVM::MemcpyOp,
                  mlir::ub::PoisonOp>(bodyOp))
      continue;
    if (bodyOp.getNumResults() == 1 &&
        isSupportedPrimitiveOperation(primitiveOperationName(&bodyOp)))
      continue;
    return unsupportedOperationLabel(&bodyOp);
  }
  return std::nullopt;
}

std::optional<std::string>
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

std::optional<std::string>
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

std::optional<std::string>
unsupportedStructuredForOperation(mlir::scf::ForOp op) {
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(op.getBody()->getTerminator());
  if (!yield)
    return "scf.for";
  if (yield.getNumOperands() != op->getNumResults())
    return "scf.yield";
  for (mlir::Operation &bodyOp : op.getBody()->without_terminator()) {
    if (mlir::isa<mlir::arith::ConstantOp>(bodyOp))
      continue;
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(bodyOp)) {
      if (auto name = unsupportedStructuredIfOperation(ifOp))
        return name;
      continue;
    }
    if (auto switchOp = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(bodyOp)) {
      if (auto name = unsupportedStructuredIndexSwitchOperation(switchOp))
        return name;
      continue;
    }
    if (auto nestedFor = mlir::dyn_cast<mlir::scf::ForOp>(bodyOp)) {
      if (auto name = unsupportedStructuredForOperation(nestedFor))
        return name;
      continue;
    }
    if (auto nestedForall = mlir::dyn_cast<mlir::scf::ForallOp>(bodyOp)) {
      if (auto name = unsupportedStructuredForallOperation(nestedForall))
        return name;
      continue;
    }
    if (auto nestedWhile = mlir::dyn_cast<mlir::scf::WhileOp>(bodyOp)) {
      if (auto name = unsupportedStructuredWhileOperation(nestedWhile))
        return name;
      continue;
    }
    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(bodyOp)) {
      if (isSupportedStructuredCast(cast))
        continue;
      return unsupportedOperationLabel(cast.getOperation());
    }
    if (auto icmp = mlir::dyn_cast<mlir::LLVM::ICmpOp>(bodyOp)) {
      if (isSupportedPointerICmp(icmp))
        continue;
      return unsupportedOperationLabel(&bodyOp);
    }
    if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(bodyOp)) {
      if (isSupportedLLVMCall(call))
        continue;
      return unsupportedOperationLabel(&bodyOp);
    }
    if (mlir::isa<dataflow::ConstantOp, dataflow::LoadOp, dataflow::StoreOp,
                  dataflow::GateOp, dataflow::MuxOp, dataflow::DemuxOp,
                  mlir::LLVM::GEPOp, mlir::LLVM::ZeroOp, mlir::LLVM::MemcpyOp,
                  mlir::ub::PoisonOp>(bodyOp))
      continue;
    if (bodyOp.getNumResults() == 1 &&
        isSupportedPrimitiveOperation(primitiveOperationName(&bodyOp)))
      continue;
    return unsupportedOperationLabel(&bodyOp);
  }
  return std::nullopt;
}

std::optional<std::string>
unsupportedStructuredWhileBody(mlir::Block *block,
                               llvm::StringRef terminatorName) {
  if (!block)
    return "scf.while";
  mlir::Operation *terminator = block->getTerminator();
  if (!terminator || terminator->getName().getStringRef() != terminatorName)
    return terminatorName.str();
  for (mlir::Operation &bodyOp : block->without_terminator()) {
    if (mlir::isa<mlir::arith::ConstantOp>(bodyOp))
      continue;
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(bodyOp)) {
      if (auto name = unsupportedStructuredIfOperation(ifOp))
        return name;
      continue;
    }
    if (auto switchOp = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(bodyOp)) {
      if (auto name = unsupportedStructuredIndexSwitchOperation(switchOp))
        return name;
      continue;
    }
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(bodyOp)) {
      if (auto name = unsupportedStructuredForOperation(forOp))
        return name;
      continue;
    }
    if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(bodyOp)) {
      if (auto name = unsupportedStructuredWhileOperation(whileOp))
        return name;
      continue;
    }
    if (auto forallOp = mlir::dyn_cast<mlir::scf::ForallOp>(bodyOp)) {
      if (auto name = unsupportedStructuredForallOperation(forallOp))
        return name;
      continue;
    }
    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(bodyOp)) {
      if (isSupportedStructuredCast(cast))
        continue;
      return unsupportedOperationLabel(cast.getOperation());
    }
    if (auto icmp = mlir::dyn_cast<mlir::LLVM::ICmpOp>(bodyOp)) {
      if (isSupportedPointerICmp(icmp))
        continue;
      return unsupportedOperationLabel(&bodyOp);
    }
    if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(bodyOp)) {
      if (isSupportedLLVMCall(call))
        continue;
      return unsupportedOperationLabel(&bodyOp);
    }
    if (mlir::isa<dataflow::ConstantOp, dataflow::LoadOp, dataflow::StoreOp,
                  dataflow::GateOp, dataflow::MuxOp, dataflow::DemuxOp,
                  mlir::LLVM::GEPOp, mlir::LLVM::ZeroOp, mlir::LLVM::MemcpyOp,
                  mlir::ub::PoisonOp>(bodyOp))
      continue;
    if (bodyOp.getNumResults() == 1 &&
        isSupportedPrimitiveOperation(primitiveOperationName(&bodyOp)))
      continue;
    return unsupportedOperationLabel(&bodyOp);
  }
  return std::nullopt;
}

std::optional<std::string>
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

std::optional<std::string>
unsupportedStructuredForallOperation(mlir::scf::ForallOp op) {
  if (!op.getOutputs().empty() || op->getNumResults() != 0)
    return "scf.forall";
  auto inParallel = op.getTerminator();
  if (inParallel.getRegion().empty() || !inParallel.getRegion().front().empty())
    return "scf.forall.in_parallel";
  for (mlir::Operation &bodyOp : op.getBody()->without_terminator()) {
    if (mlir::isa<mlir::arith::ConstantOp>(bodyOp))
      continue;
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(bodyOp)) {
      if (auto name = unsupportedStructuredIfOperation(ifOp))
        return name;
      continue;
    }
    if (auto switchOp = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(bodyOp)) {
      if (auto name = unsupportedStructuredIndexSwitchOperation(switchOp))
        return name;
      continue;
    }
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(bodyOp)) {
      if (auto name = unsupportedStructuredForOperation(forOp))
        return name;
      continue;
    }
    if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(bodyOp)) {
      if (auto name = unsupportedStructuredWhileOperation(whileOp))
        return name;
      continue;
    }
    if (auto nestedForall = mlir::dyn_cast<mlir::scf::ForallOp>(bodyOp)) {
      if (auto name = unsupportedStructuredForallOperation(nestedForall))
        return name;
      continue;
    }
    if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(bodyOp)) {
      if (isSupportedStructuredCast(cast))
        continue;
      return unsupportedOperationLabel(cast.getOperation());
    }
    if (auto icmp = mlir::dyn_cast<mlir::LLVM::ICmpOp>(bodyOp)) {
      if (isSupportedPointerICmp(icmp))
        continue;
      return unsupportedOperationLabel(&bodyOp);
    }
    if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(bodyOp)) {
      if (isSupportedLLVMCall(call))
        continue;
      return unsupportedOperationLabel(&bodyOp);
    }
    if (mlir::isa<dataflow::ConstantOp, dataflow::LoadOp, dataflow::StoreOp,
                  dataflow::GateOp, dataflow::MuxOp, dataflow::DemuxOp,
                  mlir::LLVM::GEPOp, mlir::LLVM::ZeroOp, mlir::LLVM::MemcpyOp,
                  mlir::ub::PoisonOp>(bodyOp))
      continue;
    if (bodyOp.getNumResults() == 1 &&
        isSupportedPrimitiveOperation(primitiveOperationName(&bodyOp)))
      continue;
    return unsupportedOperationLabel(&bodyOp);
  }
  return std::nullopt;
}

unsigned structuredIfFireIndex(mlir::scf::IfOp op,
                               const SimulatorState &state) {
  return structuredOpFireIndex(op.getOperation(), state);
}

bool fireStructuredIf(mlir::scf::IfOp op, SimulatorState &state) {
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

unsigned structuredIndexSwitchFireIndex(mlir::scf::IndexSwitchOp op,
                                        const SimulatorState &state) {
  return structuredOpFireIndex(op.getOperation(), state);
}

bool fireStructuredIndexSwitch(mlir::scf::IndexSwitchOp op,
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

bool executeStructuredFor(mlir::scf::ForOp op, SimulatorState &state,
                          llvm::ArrayRef<Token> operands, unsigned captureIndex,
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

bool fireStructuredFor(mlir::scf::ForOp op, SimulatorState &state) {
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

unsigned structuredWhileFireIndex(mlir::scf::WhileOp op,
                                  const SimulatorState &state) {
  return structuredOpFireIndex(op.getOperation(), state);
}

bool executeStructuredWhile(mlir::scf::WhileOp op, SimulatorState &state,
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

bool fireStructuredWhile(mlir::scf::WhileOp op, SimulatorState &state) {
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

std::optional<std::int64_t> resolveStructuredBound(mlir::OpFoldResult bound,
                                                   SimulatorState &state,
                                                   LocalValueMap &locals,
                                                   unsigned captureIndex,
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

bool executeStructuredForall(mlir::scf::ForallOp op, SimulatorState &state,
                             LocalValueMap &captures, unsigned captureIndex) {
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

bool fireStructuredForall(mlir::scf::ForallOp op, SimulatorState &state) {
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

bool isSupportedNonEvent(mlir::Operation *op) {
  return mlir::isa<dataflow::GraphReturnOp>(op);
}

bool isOrderedStructuredOperation(mlir::Operation *op) {
  return mlir::isa<mlir::scf::IfOp, mlir::scf::IndexSwitchOp, mlir::scf::ForOp,
                   mlir::scf::WhileOp, mlir::scf::ForallOp>(op);
}

unsigned structuredInputTokenCount(mlir::Operation *op,
                                   const SimulatorState &state) {
  if (op->getNumOperands() == 0)
    return 0;
  return observedTokenCount(op->getOperand(0), state);
}

bool hasPendingOrderedStructuredFire(mlir::Operation *op,
                                     const SimulatorState &state) {
  if (!isOrderedStructuredOperation(op))
    return false;
  return structuredInputTokenCount(op, state) >
         structuredOpFireIndex(op, state);
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
  if (gate && (gate.getAfterValue() == value || gate.getAfterCond() == value)) {
    collectStreamIndexSources(gate.getBeforeCond(), sources, seen, depth + 1);
    return;
  }
  mlir::Operation *owner = value.getDefiningOp();
  if (!owner)
    return;
  if (!mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp,
                 mlir::arith::DivSIOp, mlir::arith::DivUIOp,
                 mlir::arith::RemSIOp, mlir::arith::RemUIOp>(owner))
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
      .Default([&](mlir::Operation *genericOp) {
        return fireGenericPrimitive(genericOp, state);
      });
}

std::string unsupportedOperationLabel(mlir::Operation *op) {
  if (auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(op)) {
    auto callee = call.getCallee();
    if (callee.has_value() && !callee->empty())
      return llvm::formatv("{0} @{1}", op->getName().getStringRef(), *callee)
          .str();
  }
  return op->getName().getStringRef().str();
}

std::optional<std::string> unsupportedOperation(mlir::Operation *op) {
  if (isSupportedNonEvent(op))
    return std::nullopt;
  if (op->getNumResults() == 1 &&
      isSupportedPrimitiveOperation(primitiveOperationName(op)))
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
                mlir::UnrealizedConversionCastOp, mlir::LLVM::GEPOp,
                mlir::LLVM::ZeroOp, mlir::LLVM::LoadOp, mlir::LLVM::StoreOp,
                mlir::LLVM::MemcpyOp, mlir::arith::ConstantOp,
                mlir::ub::PoisonOp>(op))
    return std::nullopt;
  return unsupportedOperationLabel(op);
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

llvm::Expected<llvm::StringMap<MemoryFixture>>
indexMemoryArgs(llvm::ArrayRef<DFGMemoryArg> args, unsigned argCount) {
  llvm::StringMap<MemoryFixture> byIndex;
  for (const DFGMemoryArg &arg : args) {
    if (arg.index >= argCount)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "memref index %u is out of range",
                                     arg.index);
    std::string key = std::to_string(arg.index);
    if (byIndex.contains(key))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "memref index %u is repeated", arg.index);
    byIndex.try_emplace(key, MemoryFixture{arg.values, arg.byteOffset});
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
        mlir::isa<mlir::LLVM::StoreOp>(owner) ||
        mlir::isa<mlir::LLVM::GEPOp>(owner) ||
        mlir::isa<mlir::LLVM::MemcpyOp>(owner))
      return true;
  }
  return false;
}

bool hasDataflowStreamUse(mlir::BlockArgument arg) {
  for (mlir::OpOperand &use : arg.getUses()) {
    if (mlir::isa<dataflow::StreamOp>(use.getOwner()))
      return true;
  }
  return false;
}

bool hasDataflowCarryUse(mlir::BlockArgument arg) {
  for (mlir::OpOperand &use : arg.getUses()) {
    if (mlir::isa<dataflow::CarryOp>(use.getOwner()))
      return true;
  }
  return false;
}

bool isScalarBroadcastArgument(mlir::BlockArgument arg) {
  mlir::Type type = arg.getType();
  if (mlir::isa<mlir::NoneType, mlir::MemRefType, mlir::LLVM::LLVMPointerType>(
          type))
    return false;
  if (hasDataflowStreamUse(arg) || hasDataflowCarryUse(arg))
    return false;
  return true;
}

std::uint64_t maxSeededArgumentCardinality(const SimulatorState &state) {
  std::uint64_t targetCount = 0;
  for (const auto &seeded : state.seededTokenCounts)
    targetCount = std::max(targetCount, seeded.second);
  return targetCount;
}

void broadcastScalarArguments(mlir::Block &entry, SimulatorState &state) {
  const std::uint64_t targetCount = maxSeededArgumentCardinality(state);
  if (targetCount <= 1)
    return;

  for (mlir::BlockArgument arg : entry.getArguments()) {
    if (!isScalarBroadcastArgument(arg))
      continue;
    auto countIt = state.seededTokenCounts.find(arg);
    if (countIt == state.seededTokenCounts.end() || countIt->second != 1)
      continue;
    auto observedIt = state.observedOutputs.find(arg);
    if (observedIt == state.observedOutputs.end() || observedIt->second.empty())
      continue;
    const Token token = observedIt->second.front();
    while (state.seededTokenCounts[arg] < targetCount)
      seedBlockArgument(state, arg, token);
  }
}

void broadcastRawPointerArguments(mlir::Block &entry, SimulatorState &state) {
  const std::uint64_t targetCount = maxSeededArgumentCardinality(state);
  if (targetCount <= 1)
    return;

  for (mlir::BlockArgument arg : entry.getArguments()) {
    if (!mlir::isa<mlir::LLVM::LLVMPointerType>(arg.getType()))
      continue;
    if (!state.rawMemoryFixtures.contains(arg) || !hasDirectLLVMAddressUse(arg))
      continue;
    std::uint64_t current = state.seededTokenCounts[arg];
    while (current < targetCount) {
      seedBlockArgument(
          state, arg,
          pointerToken(arg, {}, state.rawMemoryFixtures[arg].byteOffset));
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
      auto memoryOrErr = materializeMemory(state, source, rawIt->second.values,
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

bool isVectorCardinalityBoundaryValue(mlir::Value value) {
  mlir::Operation *def = value.getDefiningOp();
  return mlir::isa_and_nonnull<dataflow::ParallelizeOp, dataflow::PackOp,
                               dataflow::UnpackOp, dataflow::SerializeOp>(def);
}

bool hasPendingVectorGroups(SimulatorState &state) {
  bool pending = false;
  for (auto &entry : state.parallelizeStates) {
    if (entry.second.mask == 0)
      continue;
    pending = true;
    state.diagnostics.push_back(
        "dataflow.parallelize ended with pending lanes; emit a false "
        "continuation token to flush the partial vector group");
  }
  return pending;
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
  llvm::StringMap<MemoryFixture> memories = std::move(*memoriesOrErr);

  SimulatorState state;
  state.maxStructuredLoopIterations = options.maxEventSteps;
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
      if (memories.lookup(key).byteOffset != 0)
        return llvm::createStringError(std::errc::invalid_argument,
                                       "memref argument %u cannot use a "
                                       "nonzero memory fixture byte offset",
                                       unsigned(index));
      auto tokensOrErr = parseMemoryTokens(memories.lookup(key).values,
                                           memrefType.getElementType());
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
      seedBlockArgument(state, arg,
                        pointerToken(arg, {}, memories.lookup(key).byteOffset));
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
  broadcastScalarArguments(entry, state);
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
    bool orderedStructuredBarrier = false;
    for (mlir::Operation &op : entry.getOperations()) {
      if (isSupportedNonEvent(&op))
        continue;
      if (orderedStructuredBarrier && isOrderedStructuredOperation(&op))
        continue;
      bool opFired = fireOperation(&op, state);
      fired |= opFired;
      if (hasPendingOrderedStructuredFire(&op, state))
        orderedStructuredBarrier = true;
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
  std::uint64_t requiredReturnItems = nonStructuredDynamicWorkItems(state);
  if (requiredReturnItems == 0 && state.eventCount > 0)
    requiredReturnItems = 1;
  for (mlir::Value value : returnValues) {
    auto it = state.observedOutputs.find(value);
    if (it == state.observedOutputs.end() || it->second.empty()) {
      report.finalOutputs.push_back("missing");
      missingReturn = true;
      continue;
    }
    const bool requiresComplete =
        (!mlir::isa<mlir::NoneType>(value.getType()) &&
         !isVectorCardinalityBoundaryValue(value)) ||
        value.getDefiningOp<dataflow::StoreOp>() != nullptr;
    if (requiredReturnItems > 1 && requiresComplete &&
        it->second.size() < requiredReturnItems) {
      missingReturn = true;
      state.diagnostics.push_back(
          llvm::formatv("dataflow.graph.return value produced "
                        "{0} of {1} dynamic work items",
                        it->second.size(), requiredReturnItems)
              .str());
    }
    report.finalOutputs.push_back(
        tokenToString(it->second.back(), value.getType()));
  }
  captureFinalMemoryState(entry, state, report);
  const bool pendingVectorGroups = hasPendingVectorGroups(state);
  const bool incompleteLoads = hasIncompleteStreamLoads(entry, state);
  if (report.status == "pass" && !state.diagnostics.empty()) {
    report.status = "blocked";
    report.diagnostics.push_back("DFG-sim stopped with runtime diagnostics");
  }
  if (report.status == "pass" &&
      (missingReturn || incompleteLoads || pendingVectorGroups)) {
    report.status = "blocked";
    report.diagnostics.push_back(
        "DFG-sim stopped before all returned values produced complete outputs");
  } else if (report.status == "blocked" &&
             (missingReturn || incompleteLoads || pendingVectorGroups)) {
    report.diagnostics.push_back(
        "DFG-sim stopped before all returned values produced complete outputs");
  }
  report.eventCount = state.eventCount;
  report.operationFireCounts = state.operationFireCounts;
  report.pipelineLatencyThroughputCycles = estimateDynamicPipelineCycles(
      state.operationFireCounts, state.diagnostics);
  report.optimisticCycles = report.pipelineLatencyThroughputCycles;
  report.operationMixCycles = report.operationFireCounts.size();
  report.optimisticCycles += report.operationMixCycles;
  report.memoryAddressSetupCycles = state.memoryAddressSetupCycles;
  report.optimisticCycles += report.memoryAddressSetupCycles;
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
  root["pipeline_latency_throughput_cycles"] =
      static_cast<int64_t>(report.pipelineLatencyThroughputCycles);
  root["operation_mix_cycles"] = report.operationMixCycles;
  root["memory_address_setup_cycles"] = report.memoryAddressSetupCycles;
  llvm::json::Array cycleBreakdown;
  cycleBreakdown.push_back(llvm::json::Object{
      {"category", "pipeline_latency_throughput"},
      {"cycles", static_cast<int64_t>(report.pipelineLatencyThroughputCycles)},
      {"evidence", "operation_fire_counts"},
      {"modeled", true},
  });
  cycleBreakdown.push_back(llvm::json::Object{
      {"category", "operation_mix"},
      {"cycles", static_cast<int64_t>(report.operationMixCycles)},
      {"evidence", "distinct operation_fire_counts keys"},
      {"modeled", true},
  });
  cycleBreakdown.push_back(llvm::json::Object{
      {"category", "memory_address_setup"},
      {"cycles", static_cast<int64_t>(report.memoryAddressSetupCycles)},
      {"evidence", "computed dataflow.load/store address operands"},
      {"modeled", true},
  });
  root["cycle_breakdown"] = std::move(cycleBreakdown);
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

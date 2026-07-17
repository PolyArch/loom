#include "Simulator/DFGSimulator.h"
#include "DFGSimulatorInternal.h"

#include "Dataflow/IR/DataflowGraphValidation.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <system_error>

using namespace loom::sim;
using namespace loom::sim::detail;

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

Token noneToken() { return Token{}; }

Token integerValueToken(std::int64_t value) {
  Token token;
  token.kind = TokenKind::Integer;
  token.intValue = value;
  return token;
}

static Token floatValueToken(double value) {
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

static std::string typePrefix(mlir::Type type) {
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

static std::string tokenToString(const Token &token, mlir::Type type) {
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

static llvm::Expected<Token> parseRuntimeToken(llvm::StringRef raw,
                                               mlir::Type type) {
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

static llvm::Expected<llvm::SmallVector<Token>>
parseMemoryTokens(llvm::StringRef raw, mlir::Type type) {
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

static std::string typeToString(mlir::Type type) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  type.print(os);
  return os.str();
}

static llvm::Expected<unsigned> supportedBitWidth(std::uint64_t width,
                                                  llvm::StringRef label) {
  if (width == 0 || width > 64)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "%s bit width must be in [1, 64], got %llu",
                                   label.str().c_str(),
                                   static_cast<unsigned long long>(width));
  return static_cast<unsigned>(width);
}

static bool hasExplicitIndexLayout(mlir::Operation *scope) {
  for (mlir::Operation *op = scope; op; op = op->getParentOp()) {
    mlir::DataLayoutSpecInterface spec;
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(op))
      spec = module.getDataLayoutSpec();
    else if (auto layoutOp = mlir::dyn_cast<mlir::DataLayoutOpInterface>(op))
      spec = layoutOp.getDataLayoutSpec();
    if (spec && !spec.getSpecForType<mlir::IndexType>().empty())
      return true;
  }
  return false;
}

static llvm::Expected<unsigned> indexBitWidth(mlir::Operation *scope) {
  if (!hasExplicitIndexLayout(scope))
    return supportedBitWidth(loom::getIndexWidth(), "configured index");

  llvm::TypeSize width = mlir::DataLayout::closest(scope).getTypeSizeInBits(
      mlir::IndexType::get(scope->getContext()));
  if (width.isScalable())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "scalable index widths are unsupported");
  return supportedBitWidth(width.getFixedValue(), "index");
}

llvm::Expected<std::int64_t> byteSizeOfType(mlir::Type type,
                                            mlir::Operation *scope) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return std::max<std::int64_t>(1, (intType.getWidth() + 7) / 8);
  if (mlir::isa<mlir::IndexType>(type)) {
    auto width = indexBitWidth(scope);
    if (!width)
      return width.takeError();
    return std::max<std::int64_t>(1, (*width + 7) / 8);
  }
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floatType.isF16())
      return 2;
    if (floatType.isF32())
      return 4;
    if (floatType.isF64())
      return 8;
  }
  if (auto arrayType = mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(type)) {
    auto elementSizeOrErr = byteSizeOfType(arrayType.getElementType(), scope);
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

static llvm::Expected<std::shared_ptr<MemoryValue>>
materializeMemory(SimulatorState &state, mlir::Value root, llvm::StringRef raw,
                  mlir::Type elementType) {
  auto [rootIt, inserted] =
      state.memoryRootIds.try_emplace(root, state.nextMemoryRootId);
  if (inserted)
    ++state.nextMemoryRootId;
  auto existing = state.memories.find(root);
  if (existing != state.memories.end()) {
    if (existing->second->elementType != elementType)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "memory fixture type mismatch: existing %s, requested %s",
          typeToString(existing->second->elementType).c_str(),
          typeToString(elementType).c_str());
    if (existing->second->logicalRootId != rootIt->second)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "memory root identity mismatch");
    return existing->second;
  }
  auto tokensOrErr = parseMemoryTokens(raw, elementType);
  if (!tokensOrErr)
    return tokensOrErr.takeError();
  llvm::SmallVector<Token> tokens = std::move(*tokensOrErr);
  llvm::SmallBitVector initialized(tokens.size(), /*t=*/true);
  auto memory = std::make_shared<MemoryValue>(MemoryValue{
      rootIt->second, elementType, std::move(tokens), std::move(initialized)});
  state.memories[root] = memory;
  return memory;
}

Token pointerToken(mlir::Value root, std::shared_ptr<MemoryValue> memory,
                   std::int64_t byteOffset) {
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
    auto strideOrErr = byteSizeOfType(strideType, op.getOperation());
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

static std::uint64_t estimateWeightedOperationScore(
    const std::map<std::string, std::uint64_t> &operationFireCounts,
    llvm::SmallVectorImpl<std::string> &diagnostics) {
  std::uint64_t score = 0;
  for (const auto &[opName, fireCount] : operationFireCounts) {
    if (fireCount == 0)
      continue;
    auto costOrErr = estimateOperationCost(opName);
    if (!costOrErr) {
      diagnostics.push_back(llvm::toString(costOrErr.takeError()));
      continue;
    }
    score += costOrErr->baseScore;
    if (fireCount > 1)
      score += (fireCount - 1) * costOrErr->repeatScore;
  }
  return score;
}

static std::uint64_t
nonStructuredDynamicWorkItems(const SimulatorState &state) {
  std::uint64_t maxStreamItems = 0;
  for (const auto &entry : state.streamTrueEmissionCounts)
    maxStreamItems = std::max(maxStreamItems, entry.second);
  std::uint64_t maxSeededItems = 0;
  for (const auto &entry : state.seededTokenCounts) {
    if (mlir::isa<mlir::NoneType>(entry.first.getType()))
      continue;
    maxSeededItems = std::max(maxSeededItems, entry.second);
  }
  return std::max(maxStreamItems, maxSeededItems);
}

static std::uint64_t dynamicWorkItems(const SimulatorState &state) {
  const std::uint64_t workItems = std::max(nonStructuredDynamicWorkItems(state),
                                           state.structuredLoopIterations);
  if (workItems == 0 && state.eventCount > 0)
    return 1;
  return workItems;
}
static void flushPendingTokens(SimulatorState &state) {
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

static bool samePointer(const MemoryView &lhs, const MemoryView &rhs) {
  return lhs.root == rhs.root && lhs.byteOffset == rhs.byteOffset;
}

std::optional<std::size_t> resolveElementIndex(const MemoryView &view,
                                               const Token &addr,
                                               SimulatorState &state,
                                               mlir::Operation *scope,
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

static bool isNullPointerToken(const Token &token) {
  return token.kind == TokenKind::Pointer && !token.pointer.root &&
         !token.pointer.memory;
}

static std::int64_t signExtend(std::int64_t value, unsigned width) {
  if (width == 0 || width >= 64)
    return value;
  const std::uint64_t mask = (std::uint64_t{1} << width) - 1;
  std::uint64_t bits = static_cast<std::uint64_t>(value) & mask;
  const std::uint64_t signBit = std::uint64_t{1} << (width - 1);
  if ((bits & signBit) == 0)
    return static_cast<std::int64_t>(bits);
  return static_cast<std::int64_t>(bits | ~mask);
}

static std::int32_t wrapI32(std::int64_t value) {
  return static_cast<std::int32_t>(static_cast<std::uint32_t>(value));
}

static std::int32_t doublingHighMultNoSat(std::int32_t lhs, std::int32_t rhs) {
  std::int64_t product =
      (std::int64_t{1} << 30) + static_cast<std::int64_t>(lhs) * rhs;
  return wrapI32(product >> 31);
}

static std::int32_t divideByPowerOfTwo(std::int32_t dividend,
                                       std::int32_t exponent) {
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

static std::int32_t cmsisRequantize(std::int32_t value, std::int32_t multiplier,
                                    std::int32_t shift) {
  const std::int32_t leftShift = shift > 0 ? shift : 0;
  const std::int32_t rightShift = shift > 0 ? 0 : -shift;
  std::int64_t shifted = static_cast<std::int64_t>(value)
                         << static_cast<unsigned>(leftShift);
  std::int32_t multiplied = doublingHighMultNoSat(wrapI32(shifted), multiplier);
  return divideByPowerOfTwo(multiplied, rightShift);
}

static std::optional<std::int64_t>
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

static std::optional<std::int64_t>
loadIntegerPointerElement(SimulatorState &state, const Token &ptr,
                          mlir::Type elementType, std::int64_t elementOffset,
                          unsigned signedWidth, mlir::Operation *scope,
                          llvm::StringRef opName) {
  auto viewOrErr = ensurePointerMemory(state, ptr, elementType);
  if (!viewOrErr) {
    state.diagnostics.push_back(llvm::toString(viewOrErr.takeError()));
    return std::nullopt;
  }
  std::optional<std::size_t> index =
      resolveElementIndex(viewOrErr->pointer, integerValueToken(elementOffset),
                          state, scope, opName);
  if (!index)
    return std::nullopt;
  std::optional<Token> value =
      readMemoryElement(viewOrErr->pointer, *index, state, opName);
  if (!value)
    return std::nullopt;
  return signExtend(integerToken(*value), signedWidth);
}

static bool storeIntegerPointerElement(SimulatorState &state, const Token &ptr,
                                       mlir::Type elementType,
                                       std::int64_t elementOffset,
                                       std::int64_t value,
                                       mlir::Operation *scope,
                                       llvm::StringRef opName) {
  auto viewOrErr = ensurePointerMemory(state, ptr, elementType);
  if (!viewOrErr) {
    state.diagnostics.push_back(llvm::toString(viewOrErr.takeError()));
    return false;
  }
  std::optional<std::size_t> index =
      resolveElementIndex(viewOrErr->pointer, integerValueToken(elementOffset),
                          state, scope, opName);
  if (!index)
    return false;
  writeMemoryElement(viewOrErr->pointer, *index,
                     integerValueToken(signExtend(value, 8)));
  return true;
}

bool executeCmsisNNVecMatMultTS8(mlir::LLVM::CallOp op, SimulatorState &state,
                                 llvm::ArrayRef<Token> operands,
                                 Token &result) {
  if (operands.size() != 15) {
    state.diagnostics.push_back("arm_nn_vec_mat_mult_t_s8 expects 15 operands");
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
          state, operands[3], i32Type, row, 32, op.getOperation(),
          "arm_nn_vec_mat_mult_t_s8");
      if (!bias)
        return false;
      acc = *bias;
    }
    for (std::int64_t col = 0; col < *rhsCols; ++col) {
      std::optional<std::int64_t> lhsValue = loadIntegerPointerElement(
          state, operands[0], i8Type, col, 8, op.getOperation(),
          "arm_nn_vec_mat_mult_t_s8");
      std::optional<std::int64_t> rhsValue = loadIntegerPointerElement(
          state, operands[1], i8Type, row * *rhsCols + col, 8,
          op.getOperation(), "arm_nn_vec_mat_mult_t_s8");
      if (!lhsValue || !rhsValue)
        return false;
      *lhsValue += *lhsOffset;
      *rhsValue += *rhsOffset;
      const std::int64_t product = *lhsValue * *rhsValue;
      acc += product;
    }

    acc = cmsisRequantize(wrapI32(acc), wrapI32(*dstMultiplier),
                          wrapI32(*dstShift));
    acc += *dstOffset;
    acc = std::max(acc, *activationMin);
    acc = std::min(acc, *activationMax);
    if (!storeIntegerPointerElement(
            state, operands[4], i8Type, row * *addressOffset, acc,
            op.getOperation(), "arm_nn_vec_mat_mult_t_s8"))
      return false;
  }

  ++state.eventCount;
  ++state.modeledLibraryCalls[kCmsisNNVecMatMultTS8.str()];
  state.modeledLibraryScore += static_cast<std::uint64_t>(*rhsRows) *
                               static_cast<std::uint64_t>(*rhsCols);
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

static llvm::Expected<unsigned> integerBitWidth(mlir::Type type,
                                                mlir::Operation *scope) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return supportedBitWidth(intType.getWidth(), "integer");
  if (mlir::isa<mlir::IndexType>(type))
    return indexBitWidth(scope);
  return 0u;
}

std::string primitivePredicate(mlir::Operation *op) {
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(op))
    return mlir::arith::stringifyCmpIPredicate(cmp.getPredicate()).str();
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(op))
    return mlir::arith::stringifyCmpFPredicate(cmp.getPredicate()).str();
  return "";
}

std::string primitiveOperationName(mlir::Operation *op) {
  if (op->getName().getStringRef() == "llvm.inline_asm") {
    auto asmString = op->getAttrOfType<mlir::StringAttr>("asm_string");
    if (asmString) {
      llvm::StringRef text = asmString.getValue();
      if (text == "pkhbt $0, $1, $2, lsl $3")
        return "llvm.arm.pkhbt";
      if (text == "pkhtb $0, $1, $2, asr $3")
        return "llvm.arm.pkhtb";
      if (text == "sxtab16 $0, $1, $2")
        return "llvm.arm.sxtab16";
      if (text == "sxtb16 $0, $1")
        return "llvm.arm.sxtb16";
    }
  }
  if (auto intrinsic = mlir::dyn_cast<mlir::LLVM::CallIntrinsicOp>(op))
    return intrinsic.getIntrin().str();
  return op->getName().getStringRef().str();
}

llvm::Expected<PrimitiveOperationDescriptor>
primitiveDescriptor(mlir::Operation *op, llvm::StringRef predicate,
                    mlir::Value result) {
  std::string opName = primitiveOperationName(op);
  auto resultBitWidth = integerBitWidth(result.getType(), op);
  if (!resultBitWidth)
    return resultBitWidth.takeError();
  auto operandBitWidth = integerBitWidth(op->getOperand(0).getType(), op);
  if (!operandBitWidth)
    return operandBitWidth.takeError();
  PrimitiveOperationDescriptor descriptor{opName, predicate, *resultBitWidth,
                                          *operandBitWidth};
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

static bool isSupportedNonEvent(mlir::Operation *op) {
  return mlir::isa<dataflow::GraphReturnOp, mlir::memref::AllocOp,
                   mlir::memref::CastOp>(op);
}

static void collectStreamIndexSources(
    mlir::Value value, llvm::DenseSet<mlir::Operation *> &sources,
    llvm::DenseSet<mlir::Value> &seen, unsigned depth = 0) {
  if (!value || depth > 8 || !seen.insert(value).second)
    return;
  if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    return collectStreamIndexSources(cast.getIn(), sources, seen, depth + 1);
  if (auto stream = value.getDefiningOp<dataflow::StreamOp>()) {
    if (stream.getIv() == value || stream.getPhase() == value)
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

static dataflow::StreamOp findStreamIndexSource(mlir::Value value) {
  llvm::DenseSet<mlir::Operation *> sources;
  llvm::DenseSet<mlir::Value> seen;
  collectStreamIndexSources(value, sources, seen);
  if (sources.size() != 1)
    return {};
  return mlir::cast<dataflow::StreamOp>(*sources.begin());
}

static bool fireOperation(mlir::Operation *op, SimulatorState &state) {
  if (isStructuredOperation(op))
    return fireStructuredOperation(op, state);
  return fireActorOperation(op, state);
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

static std::optional<std::string> unsupportedOperation(mlir::Operation *op) {
  if (isSupportedNonEvent(op))
    return std::nullopt;
  if (isStructuredOperation(op))
    return unsupportedStructuredOperation(op);
  return unsupportedActorOperation(op);
}

static dataflow::GraphOp findGraph(mlir::ModuleOp module,
                                   llvm::StringRef name) {
  if (name.starts_with("@"))
    name = name.drop_front();
  dataflow::GraphOp match;
  module.walk([&](dataflow::GraphOp graph) {
    if (!match && graph.getSymName() == name)
      match = graph;
  });
  return match;
}

static llvm::Expected<llvm::StringMap<llvm::SmallVector<std::string>>>
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

static llvm::Expected<llvm::StringMap<MemoryFixture>>
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

struct GraphReturnObservation {
  llvm::SmallVector<mlir::Value> complete;
  llvm::SmallVector<mlir::Value> values;
  llvm::SmallVector<mlir::Value> streams;
  llvm::SmallVector<mlir::Value> memories;
};

static GraphReturnObservation observeReturnOperands(dataflow::GraphOp graph) {
  GraphReturnObservation observation;
  auto ret = mlir::dyn_cast_or_null<dataflow::GraphReturnOp>(
      graph.getBody().front().getTerminator());
  if (!ret)
    return observation;
  observation.complete.append(ret.getComplete().begin(),
                              ret.getComplete().end());
  observation.values.append(ret.getValues().begin(), ret.getValues().end());
  observation.streams.append(ret.getStreams().begin(), ret.getStreams().end());
  observation.memories.append(ret.getMemories().begin(),
                              ret.getMemories().end());
  return observation;
}

static void seedBlockArgument(SimulatorState &state, mlir::BlockArgument arg,
                              const Token &token) {
  for (mlir::OpOperand &use : arg.getUses())
    state.channels[&use].push_back(token);
  state.observedOutputs[arg].push_back(token);
  ++state.seededTokenCounts[arg];
}

static llvm::Expected<std::size_t>
allocationElementCount(mlir::memref::AllocOp alloc, SimulatorState &state) {
  mlir::MemRefType type = alloc.getType();
  mlir::ValueRange dynamicSizes = alloc.getDynamicSizes();
  std::uint64_t count = 1;
  unsigned dynamicIndex = 0;
  for (unsigned dimension = 0; dimension < type.getRank(); ++dimension) {
    std::uint64_t extent = 0;
    if (type.isDynamicDim(dimension)) {
      if (dynamicIndex == dynamicSizes.size())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "memref.alloc has fewer dynamic extents than its result type");
      auto observed = state.observedOutputs.find(dynamicSizes[dynamicIndex++]);
      if (observed == state.observedOutputs.end() ||
          observed->second.size() != 1 ||
          observed->second.front().kind != TokenKind::Integer)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "memref.alloc dynamic extent is not an exact-one launch value");
      if (observed->second.front().intValue < 0)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "memref.alloc dynamic extent is negative");
      extent = static_cast<std::uint64_t>(observed->second.front().intValue);
    } else {
      extent = static_cast<std::uint64_t>(type.getDimSize(dimension));
    }
    if (extent != 0 && count > std::numeric_limits<std::size_t>::max() / extent)
      return llvm::createStringError(std::errc::value_too_large,
                                     "memref.alloc is too large for DFG-sim");
    count *= extent;
  }
  if (dynamicIndex != dynamicSizes.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "memref.alloc has more dynamic extents than its result type");
  if (count > std::numeric_limits<unsigned>::max())
    return llvm::createStringError(std::errc::value_too_large,
                                   "memref.alloc is too large for DFG-sim");
  return static_cast<std::size_t>(count);
}

static llvm::Error initializeFreshMemoryRoots(mlir::Block &entry,
                                              SimulatorState &state) {
  for (mlir::Operation &op : entry.getOperations()) {
    auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(op);
    if (!alloc)
      continue;
    if (state.memories.contains(alloc.getResult()))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "memref.alloc root is repeated");
    auto countOrErr = allocationElementCount(alloc, state);
    if (!countOrErr)
      return countOrErr.takeError();
    auto zeroOrErr = zeroToken(alloc.getType().getElementType());
    if (!zeroOrErr)
      return zeroOrErr.takeError();
    llvm::SmallVector<Token> elements(*countOrErr, *zeroOrErr);
    auto [rootIt, inserted] = state.memoryRootIds.try_emplace(
        alloc.getResult(), state.nextMemoryRootId);
    if (inserted)
      ++state.nextMemoryRootId;
    state.memories[alloc.getResult()] = std::make_shared<MemoryValue>(
        MemoryValue{rootIt->second, alloc.getType().getElementType(),
                    std::move(elements),
                    llvm::SmallBitVector(static_cast<unsigned>(*countOrErr),
                                         /*t=*/false)});
  }
  return llvm::Error::success();
}

static llvm::Error propagateMemoryAliases(mlir::Block &entry,
                                          SimulatorState &state) {
  llvm::DenseMap<mlir::Value, mlir::Type> fixtureElementTypes;
  for (mlir::Operation &op : entry.getOperations()) {
    if (!mlir::isa<mlir::UnrealizedConversionCastOp>(op) ||
        op.getNumOperands() != 1 || op.getNumResults() != 1)
      continue;
    mlir::Value source = op.getOperand(0);
    auto targetMemref =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult(0).getType());
    if (!targetMemref || !state.rawMemoryFixtures.contains(source))
      continue;
    auto [it, inserted] =
        fixtureElementTypes.try_emplace(source, targetMemref.getElementType());
    if (!inserted && it->second != targetMemref.getElementType())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "memory fixture type mismatch: existing %s, requested %s",
          typeToString(it->second).c_str(),
          typeToString(targetMemref.getElementType()).c_str());
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Operation &op : entry.getOperations()) {
      mlir::Value source;
      mlir::Value target;
      if (auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
        if (cast.getInputs().size() != 1 || cast.getResults().size() != 1)
          continue;
        source = cast.getInputs().front();
        target = cast.getResults().front();
      } else if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(op)) {
        source = cast.getSource();
        target = cast.getDest();
      } else {
        continue;
      }
      if (auto rootIt = state.memoryRootIds.find(source);
          rootIt != state.memoryRootIds.end())
        state.memoryRootIds.try_emplace(target, rootIt->second);
      if (state.memories.contains(target))
        continue;
      auto memoryIt = state.memories.find(source);
      if (memoryIt != state.memories.end()) {
        auto targetMemref = mlir::dyn_cast<mlir::MemRefType>(target.getType());
        if (targetMemref &&
            memoryIt->second->elementType != targetMemref.getElementType())
          return llvm::createStringError(
              std::errc::invalid_argument,
              "memory fixture type mismatch: existing %s, requested %s",
              typeToString(memoryIt->second->elementType).c_str(),
              typeToString(targetMemref.getElementType()).c_str());
        state.memoryRootIds.try_emplace(target,
                                        memoryIt->second->logicalRootId);
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
      state.memoryRootIds.try_emplace(target, memory->logicalRootId);
      state.rawMemoryFixtures[target] = rawIt->second;
      changed = true;
    }
  }
  return llvm::Error::success();
}

static std::shared_ptr<MemoryValue> memoryForValue(SimulatorState &state,
                                                   mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    auto memory = state.memories.find(value);
    if (memory != state.memories.end())
      return memory->second;
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1)
        return {};
      value = cast.getInputs().front();
      continue;
    }
    return {};
  }
  return {};
}

static std::optional<std::uint64_t> memoryRootIdForValue(SimulatorState &state,
                                                         mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  while (value && visited.insert(value).second) {
    auto root = state.memoryRootIds.find(value);
    if (root != state.memoryRootIds.end())
      return root->second;
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() != 1)
        return std::nullopt;
      value = cast.getInputs().front();
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

static bool hasIncompleteStreamLoads(mlir::Block &entry,
                                     SimulatorState &state) {
  bool incomplete = false;
  for (mlir::Operation &op : entry.getOperations()) {
    auto load = mlir::dyn_cast<dataflow::LoadOp>(op);
    if (!load)
      continue;
    dataflow::StreamOp stream = findStreamIndexSource(load.getAddr());
    if (!stream)
      continue;
    const std::uint64_t required =
        state.streamTrueEmissionCounts[stream.getOperation()];
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

static bool hasPendingVectorGroups(SimulatorState &state) {
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

static llvm::SmallVector<std::string>
serializeMemoryValue(const MemoryValue &memory) {
  llvm::SmallVector<std::string> values;
  for (auto [index, token] : llvm::enumerate(memory.elements)) {
    if (!memory.initialized[index]) {
      values.push_back("uninitialized");
      continue;
    }
    values.push_back(tokenToString(token, memory.elementType));
  }
  return values;
}

static llvm::Expected<std::string>
memoryFixtureFromSerializedValues(llvm::ArrayRef<std::string> values) {
  std::string fixture;
  llvm::raw_string_ostream os(fixture);
  for (auto [index, value] : llvm::enumerate(values)) {
    llvm::StringRef serialized(value);
    size_t separator = serialized.find(':');
    if (separator == llvm::StringRef::npos)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "cannot reuse serialized memory value '%s' for another invocation",
          value.c_str());
    if (index != 0)
      os << ',';
    os << serialized.drop_front(separator + 1);
  }
  return os.str();
}

static void captureFinalMemoryState(dataflow::GraphOp graph,
                                    SimulatorState &state,
                                    DFGSimulationReport &report) {
  mlir::Block &entry = graph.getBody().front();
  for (unsigned index = 0, end = graph.getFunctionType().getNumInputs();
       index < end; ++index) {
    mlir::BlockArgument arg = entry.getArgument(index + 1);
    std::shared_ptr<MemoryValue> memory = memoryForValue(state, arg);
    std::string port = llvm::formatv("arg{0}", index).str();
    if (memory)
      report.finalMemoryState[port] = serializeMemoryValue(*memory);
    if (auto rootId = memoryRootIdForValue(state, arg))
      report.finalMemoryRoots[port] =
          llvm::formatv("memory_root{0}", *rootId).str();
  }
  auto ret = mlir::cast<dataflow::GraphReturnOp>(entry.getTerminator());
  for (auto [index, memoryResult] : llvm::enumerate(ret.getMemories())) {
    std::shared_ptr<MemoryValue> memory = memoryForValue(state, memoryResult);
    std::string port = llvm::formatv("memory_result{0}", index).str();
    if (memory)
      report.finalMemoryState[port] = serializeMemoryValue(*memory);
    if (auto rootId = memoryRootIdForValue(state, memoryResult))
      report.finalMemoryRoots[port] =
          llvm::formatv("memory_root{0}", *rootId).str();
  }
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim
llvm::Expected<DFGSimulationReport>
loom::sim::simulateDataflowGraph(mlir::ModuleOp module,
                                 const DFGSimulationOptions &options) {
  DFGSimulationReport report;
  report.graph = options.graphName;
  report.workload =
      options.workloadName.empty() ? options.graphName : options.workloadName;
  report.status = "pass";

  dataflow::GraphOp graph = findGraph(module, options.graphName);
  if (!graph) {
    report.status = "unsupported";
    report.diagnostics.push_back(
        llvm::formatv("dataflow.graph '{0}' was not found", options.graphName)
            .str());
    return report;
  }
  if (graph.isExternal()) {
    report.status = "unsupported";
    report.diagnostics.push_back(
        llvm::formatv("dataflow.graph '{0}' is external", options.graphName)
            .str());
    return report;
  }

  if (llvm::Error error = dataflow::validateFinalizedProgram(module))
    return std::move(error);

  llvm::ArrayRef<int32_t> resultSegments = graph.getResultSegmentSizes();
  if (options.invocations == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "invocations must be nonzero");
  if (options.invocations > 1) {
    if (resultSegments[0] != 0 || resultSegments[1] != 0)
      return llvm::createStringError(
          std::errc::not_supported,
          "multiple invocations with value or stream results are unsupported");

    unsigned applicationInputCount = graph.getFunctionType().getNumInputs();
    auto groupedArgsOrErr =
        indexRuntimeArgs(options.args, applicationInputCount);
    if (!groupedArgsOrErr)
      return groupedArgsOrErr.takeError();
    llvm::StringMap<llvm::SmallVector<std::string>> groupedArgs =
        std::move(*groupedArgsOrErr);
    for (unsigned index = 0; index < applicationInputCount; ++index) {
      dataflow::GraphPortKind kind = graph.getInputPortKind(index);
      if (kind == dataflow::GraphPortKind::Memory)
        continue;
      if (kind == dataflow::GraphPortKind::Stream)
        return llvm::createStringError(
            std::errc::not_supported,
            "multiple invocations with stream inputs are unsupported");
      std::string key = std::to_string(index);
      auto it = groupedArgs.find(key);
      if (it == groupedArgs.end() || it->second.size() != options.invocations)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "value argument %u requires exactly one token per invocation",
            index);
    }

    DFGSimulationReport aggregate = report;
    llvm::SmallVector<DFGMemoryArg> currentMemories = options.memories;
    for (const DFGMemoryArg &memory : currentMemories)
      if (memory.byteOffset != 0)
        return llvm::createStringError(
            std::errc::not_supported,
            "multiple invocations with nonzero memory fixture offsets are "
            "unsupported");

    for (std::uint64_t invocation = 0; invocation < options.invocations;
         ++invocation) {
      DFGSimulationOptions single = options;
      single.invocations = 1;
      single.args.clear();
      single.memories = currentMemories;
      for (unsigned index = 0; index < applicationInputCount; ++index) {
        if (graph.getInputPortKind(index) != dataflow::GraphPortKind::Value)
          continue;
        std::string key = std::to_string(index);
        single.args.push_back({index, groupedArgs.lookup(key)[invocation]});
      }

      auto singleReportOrErr = simulateDataflowGraph(module, single);
      if (!singleReportOrErr)
        return singleReportOrErr.takeError();
      DFGSimulationReport singleReport = std::move(*singleReportOrErr);
      aggregate.wavefrontSteps += singleReport.wavefrontSteps;
      aggregate.eventCount += singleReport.eventCount;
      aggregate.dynamicWorkItems += singleReport.dynamicWorkItems;
      aggregate.modeledLibraryScore += singleReport.modeledLibraryScore;
      aggregate.memoryAddressScore += singleReport.memoryAddressScore;
      for (const auto &[name, count] : singleReport.operationFireCounts)
        aggregate.operationFireCounts[name] += count;
      for (const auto &[name, count] : singleReport.modeledLibraryCalls)
        aggregate.modeledLibraryCalls[name] += count;
      aggregate.finalOutputs = std::move(singleReport.finalOutputs);
      aggregate.finalMemoryState = std::move(singleReport.finalMemoryState);
      aggregate.finalMemoryRoots = std::move(singleReport.finalMemoryRoots);

      for (const std::string &diagnostic : singleReport.diagnostics)
        aggregate.diagnostics.push_back(
            llvm::formatv("invocation {0}: {1}", invocation, diagnostic).str());
      if (singleReport.status != "pass") {
        aggregate.status = singleReport.status;
        break;
      }

      for (DFGMemoryArg &memory : currentMemories) {
        std::string key = llvm::formatv("arg{0}", memory.index).str();
        auto stateIt = aggregate.finalMemoryState.find(key);
        if (stateIt == aggregate.finalMemoryState.end())
          return llvm::createStringError(
              std::errc::invalid_argument,
              "memory argument %u was not materialized by invocation %llu",
              memory.index, static_cast<unsigned long long>(invocation));
        auto fixtureOrErr = memoryFixtureFromSerializedValues(stateIt->second);
        if (!fixtureOrErr)
          return fixtureOrErr.takeError();
        memory.values = std::move(*fixtureOrErr);
      }
    }

    aggregate.weightedOperationScore = estimateWeightedOperationScore(
        aggregate.operationFireCounts, aggregate.diagnostics);
    aggregate.operationDiversityScore = aggregate.operationFireCounts.size();
    aggregate.operationCostScore =
        aggregate.weightedOperationScore + aggregate.modeledLibraryScore +
        aggregate.operationDiversityScore + aggregate.memoryAddressScore;
    return aggregate;
  }

  mlir::Block &entry = graph.getBody().front();
  unsigned applicationInputCount = graph.getFunctionType().getNumInputs();
  auto argsOrErr = indexRuntimeArgs(options.args, applicationInputCount);
  if (!argsOrErr)
    return argsOrErr.takeError();
  llvm::StringMap<llvm::SmallVector<std::string>> args = std::move(*argsOrErr);
  auto memoriesOrErr = indexMemoryArgs(options.memories, applicationInputCount);
  if (!memoriesOrErr)
    return memoriesOrErr.takeError();
  llvm::StringMap<MemoryFixture> memories = std::move(*memoriesOrErr);

  SimulatorState state;
  state.maxStructuredLoopIterations = options.maxEventSteps;
  GraphReturnObservation returnObservation = observeReturnOperands(graph);
  seedBlockArgument(state, graph.getStart(), noneToken());

  for (unsigned index = 0; index < applicationInputCount; ++index) {
    mlir::BlockArgument arg = entry.getArgument(index + 1);
    std::string key = std::to_string(index);
    dataflow::GraphPortKind kind = graph.getInputPortKind(index);
    if (kind == dataflow::GraphPortKind::Memory) {
      if (!memories.contains(key))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "missing memory fixture for argument %u",
                                       unsigned(index));
      if (args.contains(key))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "memory argument %u must use --memref",
                                       unsigned(index));
      if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(arg.getType())) {
        if (memories.lookup(key).byteOffset != 0)
          return llvm::createStringError(
              std::errc::invalid_argument,
              "memref argument %u cannot use a nonzero memory fixture byte "
              "offset",
              unsigned(index));
        auto tokensOrErr = parseMemoryTokens(memories.lookup(key).values,
                                             memrefType.getElementType());
        if (!tokensOrErr)
          return llvm::joinErrors(
              llvm::createStringError(std::errc::invalid_argument,
                                      "invalid memref argument %u",
                                      unsigned(index)),
              tokensOrErr.takeError());
        llvm::SmallVector<Token> tokens = std::move(*tokensOrErr);
        llvm::SmallBitVector initialized(tokens.size(), /*t=*/true);
        auto [rootIt, inserted] =
            state.memoryRootIds.try_emplace(arg, state.nextMemoryRootId);
        if (inserted)
          ++state.nextMemoryRootId;
        state.memories[arg] = std::make_shared<MemoryValue>(
            MemoryValue{rootIt->second, memrefType.getElementType(),
                        std::move(tokens), std::move(initialized)});
      } else {
        if (!state.memoryRootIds.contains(arg))
          state.memoryRootIds[arg] = state.nextMemoryRootId++;
        state.rawMemoryFixtures[arg] = memories.lookup(key);
        seedBlockArgument(
            state, arg, pointerToken(arg, {}, memories.lookup(key).byteOffset));
      }
      continue;
    }

    if (memories.contains(key))
      return llvm::createStringError(std::errc::invalid_argument,
                                     "value argument %u must not use --memref",
                                     unsigned(index));
    auto argIt = args.find(key);
    if (argIt == args.end())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "missing runtime argument %u",
                                     unsigned(index));
    if (kind == dataflow::GraphPortKind::Value && argIt->second.size() != 1)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "value argument %u requires exactly one token", unsigned(index));
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

  if (llvm::Error err = initializeFreshMemoryRoots(entry, state))
    return std::move(err);
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

  auto outputCount = [&](mlir::Value value) -> size_t {
    auto it = state.observedOutputs.find(value);
    return it == state.observedOutputs.end() ? 0 : it->second.size();
  };
  auto completionReady = [&]() {
    return !returnObservation.complete.empty() &&
           llvm::all_of(returnObservation.complete, [&](mlir::Value witness) {
             return outputCount(witness) != 0;
           });
  };

  auto unclosedStatefulActor = [&]() -> mlir::Operation * {
    for (mlir::Operation &op : entry.without_terminator()) {
      if (auto stream = mlir::dyn_cast<dataflow::StreamOp>(op)) {
        auto it = state.streamStates.find(stream.getOperation());
        if (it != state.streamStates.end() &&
            it->second.mode != StreamMode::Idle)
          return &op;
        continue;
      }
      if (auto carry = mlir::dyn_cast<dataflow::CarryOp>(op)) {
        auto it = state.carryStates.find(carry.getOperation());
        if (it != state.carryStates.end() &&
            it->second.semanticState != PhaseSemanticState::Initial)
          return &op;
        continue;
      }
      if (auto invariant = mlir::dyn_cast<dataflow::InvariantOp>(op)) {
        auto it = state.invariantStates.find(invariant.getOperation());
        if (it != state.invariantStates.end() &&
            (it->second.semanticState != PhaseSemanticState::Initial ||
             it->second.latched.has_value()))
          return &op;
        continue;
      }
      if (auto gate = mlir::dyn_cast<dataflow::GateOp>(op))
        if (state.gateContinueStates.contains(gate.getOperation()))
          return &op;
    }
    return nullptr;
  };

  bool retired = false;
  auto streamInputsCommitted = [&]() {
    for (unsigned index = 0; index < applicationInputCount; ++index) {
      if (graph.getInputPortKind(index) != dataflow::GraphPortKind::Stream)
        continue;
      mlir::BlockArgument arg = entry.getArgument(index + 1);
      for (mlir::OpOperand &use : arg.getUses()) {
        auto channel = state.channels.find(&use);
        if (channel != state.channels.end() && !channel->second.empty())
          return false;
      }
    }
    return true;
  };
  auto observeRetirement = [&]() {
    if (retired || !completionReady())
      return;
    retired = true;
    for (mlir::Value witness : returnObservation.complete) {
      if (outputCount(witness) != 1) {
        report.status = "invalid";
        report.diagnostics.push_back(
            "completion witness produced multiple tokens before retirement");
        return;
      }
    }
    if (!streamInputsCommitted()) {
      report.status = "invalid";
      report.diagnostics.push_back(
          "graph retired before all stream input tokens were committed");
      return;
    }
    if (mlir::Operation *actor = unclosedStatefulActor()) {
      report.status = "invalid";
      report.diagnostics.push_back(
          ("graph retired before stateful actor close/reset: " +
           actor->getName().getStringRef())
              .str());
      return;
    }
    for (auto [index, value] : llvm::enumerate(returnObservation.values)) {
      size_t count = outputCount(value);
      if (count == 1)
        continue;
      report.status = "invalid";
      report.diagnostics.push_back(
          llvm::formatv("value output #{0} produced {1} tokens at retirement",
                        index, count)
              .str());
      return;
    }
  };
  observeRetirement();

  for (std::uint64_t step = 0; step < options.maxEventSteps; ++step) {
    bool fired = false;
    bool orderedStructuredBarrier = false;
    for (mlir::Operation &op : entry.getOperations()) {
      if (isSupportedNonEvent(&op))
        continue;
      if (orderedStructuredBarrier && isStructuredOperation(&op))
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
    observeRetirement();
    if (retired || report.status == "invalid")
      break;
  }
  if (!retired && report.wavefrontSteps == options.maxEventSteps) {
    report.status = "blocked";
    report.diagnostics.push_back("maximum event steps reached");
  }

  bool missingReturn = false;
  report.dynamicWorkItems = dynamicWorkItems(state);
  if (!retired) {
    report.finalOutputs.push_back("missing");
    missingReturn = true;
  } else {
    mlir::Value witness = returnObservation.complete.front();
    report.finalOutputs.push_back(
        tokenToString(state.observedOutputs.find(witness)->second.front(),
                      witness.getType()));
  }
  for (mlir::Value value : returnObservation.values) {
    auto it = state.observedOutputs.find(value);
    if (it == state.observedOutputs.end() || it->second.empty()) {
      report.finalOutputs.push_back("missing");
      missingReturn = true;
      continue;
    }
    report.finalOutputs.push_back(
        tokenToString(it->second.front(), value.getType()));
  }
  for (mlir::Value stream : returnObservation.streams) {
    llvm::SmallVector<std::string> tokens;
    auto it = state.observedOutputs.find(stream);
    if (it != state.observedOutputs.end())
      for (const Token &token : it->second)
        tokens.push_back(tokenToString(token, stream.getType()));
    report.finalStreamOutputs.push_back(std::move(tokens));
  }
  captureFinalMemoryState(graph, state, report);
  const bool pendingVectorGroups = hasPendingVectorGroups(state);
  const bool incompleteLoads = hasIncompleteStreamLoads(entry, state);
  if (report.status == "pass" && !retired) {
    report.status = "blocked";
    report.diagnostics.push_back("graph did not fire its retirement frontier");
  }
  if (report.status == "pass" && !state.diagnostics.empty()) {
    report.status = "blocked";
    report.diagnostics.push_back("DFG-sim stopped with runtime diagnostics");
  }
  if (report.status == "pass" &&
      (missingReturn || incompleteLoads || pendingVectorGroups)) {
    report.status = retired ? "invalid" : "blocked";
    report.diagnostics.push_back(
        retired ? "graph retired with incomplete internal state"
                : "graph stopped before retirement outputs were complete");
  }
  report.eventCount = state.eventCount;
  report.operationFireCounts = state.operationFireCounts;
  report.modeledLibraryCalls = state.modeledLibraryCalls;
  report.weightedOperationScore = estimateWeightedOperationScore(
      state.operationFireCounts, state.diagnostics);
  report.operationCostScore = report.weightedOperationScore;
  report.modeledLibraryScore = state.modeledLibraryScore;
  report.operationCostScore += report.modeledLibraryScore;
  report.operationDiversityScore = report.operationFireCounts.size();
  report.operationCostScore += report.operationDiversityScore;
  report.memoryAddressScore = state.memoryAddressScore;
  report.operationCostScore += report.memoryAddressScore;
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
  if (report.status == "pass") {
    root["operation_cost_score"] = report.operationCostScore;
    root["weighted_operation_score"] =
        static_cast<int64_t>(report.weightedOperationScore);
    root["modeled_library_score"] = report.modeledLibraryScore;
    root["operation_diversity_score"] = report.operationDiversityScore;
    root["memory_address_score"] = report.memoryAddressScore;
    llvm::json::Array scoreBreakdown;
    scoreBreakdown.push_back(llvm::json::Object{
        {"category", "weighted_operations"},
        {"score", static_cast<int64_t>(report.weightedOperationScore)},
        {"evidence", "operation_fire_counts"},
        {"heuristic", true},
    });
    scoreBreakdown.push_back(llvm::json::Object{
        {"category", "modeled_library_work"},
        {"score", static_cast<int64_t>(report.modeledLibraryScore)},
        {"evidence", "modeled_library_calls and modeled workload dimensions"},
        {"heuristic", true},
    });
    scoreBreakdown.push_back(llvm::json::Object{
        {"category", "operation_diversity"},
        {"score", static_cast<int64_t>(report.operationDiversityScore)},
        {"evidence", "distinct operation_fire_counts keys"},
        {"heuristic", true},
    });
    scoreBreakdown.push_back(llvm::json::Object{
        {"category", "computed_memory_address"},
        {"score", static_cast<int64_t>(report.memoryAddressScore)},
        {"evidence", "computed dataflow.load/store address operands"},
        {"heuristic", true},
    });
    root["score_breakdown"] = std::move(scoreBreakdown);
  }
  root["wavefront_steps"] = report.wavefrontSteps;
  root["event_count"] = report.eventCount;
  root["dynamic_work_items"] = report.dynamicWorkItems;

  llvm::json::Object fireCounts;
  for (const auto &[opName, count] : report.operationFireCounts)
    fireCounts[opName] = count;
  root["operation_fire_counts"] = std::move(fireCounts);

  llvm::json::Object libraryCalls;
  for (const auto &[callee, count] : report.modeledLibraryCalls)
    libraryCalls[callee] = count;
  root["modeled_library_calls"] = std::move(libraryCalls);

  llvm::json::Array outputs;
  for (const std::string &value : report.finalOutputs)
    outputs.push_back(value);
  root["final_outputs"] = std::move(outputs);

  llvm::json::Array streamOutputs;
  for (const auto &stream : report.finalStreamOutputs) {
    llvm::json::Array streamValues;
    for (const std::string &value : stream)
      streamValues.push_back(value);
    streamOutputs.push_back(std::move(streamValues));
  }
  root["final_stream_outputs"] = std::move(streamOutputs);

  llvm::json::Object finalMemoryState;
  for (const auto &[argument, values] : report.finalMemoryState) {
    llvm::json::Array memoryValues;
    for (const std::string &value : values)
      memoryValues.push_back(value);
    finalMemoryState[argument] = std::move(memoryValues);
  }
  root["final_memory_state"] = std::move(finalMemoryState);

  llvm::json::Object finalMemoryRoots;
  for (const auto &[port, rootId] : report.finalMemoryRoots)
    finalMemoryRoots[port] = rootId;
  root["final_memory_roots"] = std::move(finalMemoryRoots);

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

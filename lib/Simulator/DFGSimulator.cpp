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
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>

using namespace loom::sim;
using namespace loom::sim::detail;

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

// A memory fixture is an operand of the graph that owns it, so its element
// tokens are encoded against that same scope as every other runtime token.
static llvm::Expected<llvm::SmallVector<Token>>
parseMemoryTokens(llvm::StringRef raw, mlir::Type type,
                  mlir::Operation *scope) {
  llvm::SmallVector<Token> tokens;
  llvm::SmallVector<llvm::StringRef> parts;
  raw.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (parts.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "memref fixture must contain values");
  for (llvm::StringRef part : parts) {
    auto tokenOrErr = parseRuntimeToken(part, type, scope);
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

static llvm::Expected<std::int64_t> byteSizeForBitWidth(std::uint64_t width) {
  if (width == 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "token bit width must be nonzero");
  const std::uint64_t bytes = llvm::divideCeil(width, std::uint64_t{8});
  if (bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return llvm::createStringError(std::errc::value_too_large,
                                   "token byte size is unsupported");
  return static_cast<std::int64_t>(bytes);
}

llvm::Expected<std::int64_t> byteSizeOfType(mlir::Type type,
                                            mlir::Operation *scope) {
  if (mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::VectorType>(type)) {
    auto width = tokenTypeBitWidth(type);
    if (!width)
      return width.takeError();
    return byteSizeForBitWidth(*width);
  }
  if (mlir::isa<mlir::IndexType>(type)) {
    auto width = loom::getIndexBitWidth(scope);
    if (!width)
      return width.takeError();
    return byteSizeForBitWidth(*width);
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported memory element type: %s",
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
  auto tokensOrErr = parseMemoryTokens(raw, elementType, state.graphScope);
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
  if (mlir::isa<mlir::IndexType>(type))
    return integerValueToken(0);
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return tokenFromBitPattern(llvm::APInt(intType.getWidth(), 0), intType);
  if (mlir::isa<mlir::FloatType>(type))
    return floatValueToken(0.0);
  if (mlir::isa<mlir::VectorType>(type)) {
    auto width = tokenTypeBitWidth(type);
    if (!width)
      return width.takeError();
    return tokenFromBitPattern(llvm::APInt(*width, 0), type);
  }
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported zero-initialized memory type: %s",
                                 typeToString(type).c_str());
}

bool hasToken(ChannelMap &channels, mlir::OpOperand &operand) {
  auto it = channels.find(&operand);
  return it != channels.end() && !it->second.empty();
}

Token popToken(SimulatorState &state, mlir::OpOperand &operand) {
  auto &queue = state.channels[&operand];
  Token token = queue.front();
  queue.pop_front();
  state.firingMemoryOrderFrontier.absorb(
      state.memoryOrderFrontiers.elements(token.memoryOrder),
      token.memoryOrder);
  ++state.actorMutationEpoch;
  return token;
}

Token peekToken(ChannelMap &channels, mlir::OpOperand &operand) {
  return channels[&operand].front();
}

static void publishToken(SimulatorState &state, mlir::Value value,
                         const Token &token) {
  for (mlir::OpOperand &use : value.getUses())
    state.pendingChannels[&use].push_back(token);
  state.pendingObservedOutputs[value].push_back(token);
  ++state.actorMutationEpoch;
}

void emitToken(SimulatorState &state, mlir::Value value, Token token) {
  token.memoryOrder = publishFiredMemoryOrder(state, token.memoryOrder);
  publishToken(state, value, token);
}

void emitTokenWithMemoryOrder(SimulatorState &state, mlir::Value value,
                              Token token, MemoryOrderFrontierId memoryOrder) {
  token.memoryOrder = memoryOrder;
  publishToken(state, value, token);
}

bool recordEvent(SimulatorState &state, dataflow::OperationSchemaId schema) {
  ++state.eventCount;
  ++state.operationFireCounts[schema];
  return true;
}

const dataflow::CanonicalActorSchemaProjection &
actorProjection(const SimulatorState &state, mlir::Operation *op) {
  auto found = state.actorProjections.find(op);
  assert(found != state.actorProjections.end() &&
         "admitted actor has no cached schema projection");
  return found->second;
}

bool recordActorEvent(SimulatorState &state, mlir::Operation *op) {
  return recordEvent(state, actorProjection(state, op).schema);
}

static void flushPendingTokens(SimulatorState &state) {
  for (auto &entry : state.pendingChannels) {
    if (!entry.second.empty()) {
      mlir::Operation *owner = entry.first->getOwner();
      auto ordinal = state.plainMemoryOperationOrder.find(owner);
      if (ordinal != state.plainMemoryOperationOrder.end())
        state.plainMemoryCandidates.try_emplace(ordinal->second, owner);
    }
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
  if (token.bitPattern)
    return token.bitPattern->sextOrTrunc(64).getSExtValue();
  return token.intValue;
}

bool boolToken(const Token &token) {
  if (token.kind == TokenKind::Bool)
    return token.boolValue;
  if (token.bitPattern)
    return !token.bitPattern->isZero();
  return token.intValue != 0;
}

llvm::Expected<PrimitiveValue> primitiveValueFromToken(const Token &token,
                                                       mlir::Type type,
                                                       unsigned indexBitWidth) {
  if (token.valueState == PrimitiveValueState::Poison)
    return PrimitiveValue::poison();
  if (token.valueState == PrimitiveValueState::Undef)
    return PrimitiveValue::undef();
  if (mlir::isa<mlir::IndexType>(type)) {
    if (indexBitWidth == 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "index primitive operand has no resolved bit width");
    auto bits = indexTokenBitPattern(token, indexBitWidth);
    if (!bits)
      return bits.takeError();
    return PrimitiveValue::integer(*bits);
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    auto bits = tokenBitPattern(token, intType);
    if (!bits)
      return bits.takeError();
    return PrimitiveValue::integer(*bits);
  }
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    auto bits = tokenBitPattern(token, floatType);
    if (!bits)
      return bits.takeError();
    return PrimitiveValue::floating(
        llvm::APFloat(floatType.getFloatSemantics(), *bits));
  }
  return llvm::createStringError(
      std::errc::invalid_argument,
      "primitive operand type has no scalar simulator representation");
}

llvm::Expected<Token> tokenFromPrimitiveValue(const PrimitiveValue &value,
                                              mlir::Type type) {
  if (value.state != PrimitiveValueState::Defined)
    return exceptionalValueToken(value.state, type);
  if (!value.bits)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "defined primitive result has no bits");
  if (mlir::isa<mlir::IndexType>(type)) {
    return indexToken(*value.bits);
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    auto token = tokenFromBitPattern(*value.bits, intType);
    if (!token)
      return token.takeError();
    if (intType.getWidth() >= 2 && intType.getWidth() <= 64)
      token->intValue = value.bits->getSExtValue();
    return *token;
  }
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    return tokenFromBitPattern(*value.bits, floatType);
  }
  return llvm::createStringError(
      std::errc::invalid_argument,
      "primitive result type has no scalar simulator representation");
}

static llvm::Expected<unsigned> integerBitWidth(mlir::Type type,
                                                mlir::Operation *scope) {
  if (!type)
    return 0u;
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return intType.getWidth();
  if (mlir::isa<mlir::IndexType>(type)) {
    auto width = loom::getIndexBitWidth(scope);
    if (!width)
      return width.takeError();
    return *width;
  }
  return 0u;
}

llvm::Expected<PrimitiveOperationDescriptor>
primitiveDescriptor(const dataflow::CanonicalActorSchemaProjection &projection,
                    mlir::Operation *op, mlir::Value result) {
  mlir::Type operandType =
      op->getNumOperands() == 0 ? mlir::Type{} : op->getOperand(0).getType();
  return primitiveDescriptor(projection, op, result.getType(), operandType);
}

llvm::Expected<PrimitiveOperationDescriptor>
primitiveDescriptor(const dataflow::CanonicalActorSchemaProjection &projection,
                    mlir::Operation *op, mlir::Type resultType,
                    mlir::Type operandType) {
  auto resultBitWidth = integerBitWidth(resultType, op);
  if (!resultBitWidth)
    return resultBitWidth.takeError();
  auto operandBitWidth = integerBitWidth(operandType, op);
  if (!operandBitWidth)
    return operandBitWidth.takeError();
  return PrimitiveOperationDescriptor{projection, *resultBitWidth,
                                      *operandBitWidth};
}

static bool isSupportedNonEvent(mlir::Operation *op) {
  return mlir::isa<dataflow::GraphReturnOp, mlir::memref::AllocOp,
                   mlir::memref::CastOp, mlir::UnrealizedConversionCastOp>(op);
}

enum class FireOutcome {
  NotReady,
  Fired,
  Failed,
};

static FireOutcome fireOperation(mlir::Operation *op, SimulatorState &state) {
  const std::uint64_t mutationEpoch = state.actorMutationEpoch;
  const std::size_t diagnosticCount = state.diagnostics.size();
  // One attempt owns one frontier. Clearing it here keeps a NotReady or Failed
  // attempt from lending its consumed order to the next actor.
  state.firingMemoryOrderFrontier.clear();
  bool fired = fireActorOperation(op, state);
  if (fired)
    return FireOutcome::Fired;
  if (state.actorMutationEpoch != mutationEpoch ||
      state.diagnostics.size() != diagnosticCount)
    return FireOutcome::Failed;
  return FireOutcome::NotReady;
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
      const std::int64_t dynamicExtent = integerToken(observed->second.front());
      if (dynamicExtent < 0)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "memref.alloc dynamic extent is negative");
      extent = static_cast<std::uint64_t>(dynamicExtent);
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
    auto memory = std::make_shared<MemoryValue>(MemoryValue{
        rootIt->second, alloc.getType().getElementType(), std::move(elements),
        llvm::SmallBitVector(static_cast<unsigned>(*countOrErr),
                             /*t=*/false)});
    state.memories[alloc.getResult()] = memory;
    state.memoryViews[alloc.getResult()] =
        MemoryView{memory, alloc.getResult(), 0};
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
      if (state.memoryViews.contains(target))
        continue;
      auto viewIt = state.memoryViews.find(source);
      if (viewIt != state.memoryViews.end()) {
        MemoryView view = viewIt->second;
        auto targetMemref = mlir::dyn_cast<mlir::MemRefType>(target.getType());
        if (targetMemref &&
            view.memory->elementType != targetMemref.getElementType())
          return llvm::createStringError(
              std::errc::invalid_argument,
              "memory fixture type mismatch: existing %s, requested %s",
              typeToString(view.memory->elementType).c_str(),
              typeToString(targetMemref.getElementType()).c_str());
        state.memories[target] = view.memory;
        state.memoryViews[target] = view;
        state.memoryRootIds.try_emplace(target, view.memory->logicalRootId);
        changed = true;
        continue;
      }
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
        state.memoryViews[target] = MemoryView{memoryIt->second, source, 0};
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
      MemoryView view{memory, source, rawIt->second.byteOffset};
      state.memories[source] = memory;
      state.memories[target] = memory;
      state.memoryViews[source] = view;
      state.memoryViews[target] = view;
      state.memoryRootIds.try_emplace(target, memory->logicalRootId);
      state.rawMemoryFixtures[target] = rawIt->second;
      changed = true;
    }
  }
  return llvm::Error::success();
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
  state.graphScope = graph.getOperation();
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
        auto tokensOrErr = parseMemoryTokens(
            memories.lookup(key).values, memrefType.getElementType(), graph);
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
        auto memory = std::make_shared<MemoryValue>(
            MemoryValue{rootIt->second, memrefType.getElementType(),
                        std::move(tokens), std::move(initialized)});
        state.memories[arg] = memory;
        state.memoryViews[arg] = MemoryView{memory, arg, 0};
      } else {
        if (!state.memoryRootIds.contains(arg))
          state.memoryRootIds[arg] = state.nextMemoryRootId++;
        state.rawMemoryFixtures[arg] = memories.lookup(key);
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
      auto tokenOrErr = parseRuntimeToken(rawToken, arg.getType(), graph);
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

  std::set<std::pair<std::string, std::string>> unsupported;
  std::uint64_t operationOrdinal = 0;
  for (mlir::Operation &op : entry.getOperations()) {
    if (mlir::isa<dataflow::LoadOp, dataflow::StoreOp>(op)) {
      state.plainMemoryOperationOrder.try_emplace(&op, operationOrdinal);
      state.plainMemoryCandidates.try_emplace(operationOrdinal, &op);
    }
    if (isSupportedNonEvent(&op)) {
      ++operationOrdinal;
      continue;
    }
    if (!dataflow::operationSchemaOf(&op)) {
      unsupported.emplace(unsupportedOperationLabel(&op), "");
      ++operationOrdinal;
      continue;
    }
    auto projection = dataflow::projectRegisteredActorSchemaProjection(&op);
    if (!projection) {
      unsupported.emplace(unsupportedOperationLabel(&op),
                          llvm::toString(projection.takeError()));
      ++operationOrdinal;
      continue;
    }
    if (auto diagnostic = unsupportedActorProvider(&op, *projection)) {
      unsupported.emplace(diagnostic->label, diagnostic->reason);
      ++operationOrdinal;
      continue;
    }
    state.actorProjections.try_emplace(&op, std::move(*projection));
    ++operationOrdinal;
  }
  if (!unsupported.empty()) {
    report.status = "unsupported";
    for (const auto &[label, reason] : unsupported) {
      std::string diagnostic = "unsupported op: " + label;
      if (!reason.empty())
        diagnostic += ": " + reason;
      report.diagnostics.push_back(std::move(diagnostic));
    }
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

  while ((report.wavefrontSteps < options.maxEventSteps || retired) &&
         report.status != "invalid") {
    if (!admitReadyPlainMemoryActions(state))
      break;
    bool fired = false;
    for (mlir::Operation &op : entry.getOperations()) {
      if (isSupportedNonEvent(&op))
        continue;
      FireOutcome outcome = fireOperation(&op, state);
      // The run has already failed at runtime, so it leaves the wave here,
      // before any later actor observes or mutates state and before this wave
      // publishes anything. The retained failure overrides the lifecycle
      // classification below, so the run never continues into a deadlock
      // witness, an exhausted event budget, or a static-invalid diagnosis
      // that would relabel it.
      if (state.failure != RunFailure::None)
        break;
      if (retired && outcome != FireOutcome::NotReady) {
        report.status = "invalid";
        report.diagnostics.push_back(("actor '" + op.getName().getStringRef() +
                                      (outcome == FireOutcome::Fired
                                           ? "' fired after graph retirement"
                                           : "' failed after graph retirement"))
                                         .str());
        break;
      }
      fired |= outcome == FireOutcome::Fired;
    }
    if (report.status == "invalid" || state.failure != RunFailure::None ||
        !fired)
      break;
    flushPendingTokens(state);
    ++report.wavefrontSteps;
    observeRetirement();
  }
  // A runtime failure is definitive: once a plain conflicting access is
  // rejected or a provider invariant breaks, the run does not become a
  // deadlock, so an exhausted event budget must not mask it as blocked.
  if (!retired && report.wavefrontSteps == options.maxEventSteps &&
      state.failure == RunFailure::None) {
    report.status = "blocked";
    report.diagnostics.push_back("maximum event steps reached");
  }

  bool missingReturn = false;
  bool pendingVectorGroups = false;
  // A failed run has no result, whichever way it failed: diagnostics and
  // execution evidence remain reportable, but outputs and terminal memory are
  // not fabricated from a prefix the failed decision never committed.
  if (!applyRunFailureTerminal(state, report)) {
    if (!retired) {
      report.finalOutputs.push_back("missing");
      missingReturn = true;
    } else {
      mlir::Value witness = returnObservation.complete.front();
      auto serialized =
          tokenToString(state.observedOutputs.find(witness)->second.front(),
                        witness.getType(), graph);
      if (!serialized)
        return serialized.takeError();
      report.finalOutputs.push_back(std::move(*serialized));
    }
    for (mlir::Value value : returnObservation.values) {
      auto it = state.observedOutputs.find(value);
      if (it == state.observedOutputs.end() || it->second.empty()) {
        report.finalOutputs.push_back("missing");
        missingReturn = true;
        continue;
      }
      auto serialized =
          tokenToString(it->second.front(), value.getType(), graph);
      if (!serialized)
        return serialized.takeError();
      report.finalOutputs.push_back(std::move(*serialized));
    }
    for (mlir::Value stream : returnObservation.streams) {
      llvm::SmallVector<std::string> tokens;
      auto it = state.observedOutputs.find(stream);
      if (it != state.observedOutputs.end())
        for (const Token &token : it->second) {
          auto serialized = tokenToString(token, stream.getType(), graph);
          if (!serialized)
            return serialized.takeError();
          tokens.push_back(std::move(*serialized));
        }
      report.finalStreamOutputs.push_back(std::move(tokens));
    }
    if (llvm::Error error = captureFinalMemoryState(graph, state, report))
      return std::move(error);
    pendingVectorGroups = hasPendingVectorGroups(state);
  }
  if (report.status == "pass" && !retired) {
    report.status = "blocked";
    report.diagnostics.push_back("graph did not fire its retirement frontier");
  }
  if (report.status == "pass" && !state.diagnostics.empty()) {
    report.status = "blocked";
    report.diagnostics.push_back("DFG-sim stopped with runtime diagnostics");
  }
  if (report.status == "pass" && (missingReturn || pendingVectorGroups)) {
    report.status = retired ? "invalid" : "blocked";
    report.diagnostics.push_back(
        retired ? "graph retired with incomplete internal state"
                : "graph stopped before retirement outputs were complete");
  }
  projectRunObservations(state, report);
  return report;
}

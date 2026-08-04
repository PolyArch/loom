#include "DFGSimulatorInternal.h"
#include "SimulationWireInternal.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <limits>
#include <system_error>
#include <utility>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {
namespace {

llvm::Expected<std::size_t> allocationElementCount(mlir::memref::AllocOp alloc,
                                                   SimulatorState &state) {
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

} // namespace

void initializeRunState(SimulatorState &state,
                        const PreparedGraphExecution &execution) {
  state.execution = &execution;
  state.channelSlots.reserve(execution.channels.size());
  for (const PreparedGraphExecution::Channel &channel : execution.channels)
    state.channelSlots.push_back(
        ChannelSlot{channel.operand, channel.ownerActorOrdinal, {}, {}});
  state.nextActorCandidates.resize(execution.actorPlans.size(), true);
  state.plainMemoryCandidates = execution.initialPlainMemoryCandidates;
}

llvm::Error initializeFreshMemoryRoots(mlir::Block &entry,
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
    auto elementBytes = encodeMemoryElement(
        *zeroOrErr, alloc.getType().getElementType(), state.graphScope);
    if (!elementBytes)
      return elementBytes.takeError();
    if (*countOrErr >
        std::numeric_limits<std::size_t>::max() / elementBytes->size())
      return llvm::createStringError(std::errc::value_too_large,
                                     "memref.alloc is too large for DFG-sim");
    llvm::SmallVector<SemanticMemoryByte> bytes;
    bytes.reserve(*countOrErr * elementBytes->size());
    for (std::size_t index = 0; index < *countOrErr; ++index)
      bytes.append(elementBytes->begin(), elementBytes->end());
    auto [rootIt, inserted] = state.memoryRootIds.try_emplace(
        alloc.getResult(), state.nextMemoryRootId);
    if (inserted)
      ++state.nextMemoryRootId;
    const std::size_t totalBytes = *countOrErr * elementBytes->size();
    if (totalBytes > std::numeric_limits<unsigned>::max())
      return llvm::createStringError(std::errc::value_too_large,
                                     "memref.alloc is too large for DFG-sim");
    auto memory = std::make_shared<MemoryValue>(
        MemoryValue{rootIt->second,
                    std::move(bytes),
                    llvm::SmallBitVector(static_cast<unsigned>(totalBytes),
                                         /*t=*/false),
                    {}});
    state.memories[alloc.getResult()] = memory;
    state.memoryViews[alloc.getResult()] = MemoryView{
        memory, alloc.getResult(), 0, alloc.getType().getElementType()};
  }
  return llvm::Error::success();
}

llvm::Error propagateMemoryAliases(mlir::Block &entry, SimulatorState &state) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Operation &op : entry.getOperations()) {
      mlir::Value source;
      mlir::Value target;
      if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(op)) {
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
        if (targetMemref) {
          view.elementType = targetMemref.getElementType();
          MemoryView &sourceView = viewIt->second;
          if (!sourceView.elementType)
            sourceView.elementType = view.elementType;
          else if (sourceView.elementType != view.elementType)
            sourceView.elementType = {};
        }
        state.memories[target] = view.memory;
        state.memoryViews[target] = view;
        state.memoryRootIds.try_emplace(target, view.memory->logicalRootId);
        changed = true;
        continue;
      }
      auto memoryIt = state.memories.find(source);
      if (memoryIt != state.memories.end()) {
        auto targetMemref = mlir::dyn_cast<mlir::MemRefType>(target.getType());
        state.memoryRootIds.try_emplace(target,
                                        memoryIt->second->logicalRootId);
        state.memories[target] = memoryIt->second;
        state.memoryViews[target] = MemoryView{
            memoryIt->second, source, 0,
            targetMemref ? targetMemref.getElementType() : mlir::Type{}};
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
      MemoryView view{memory, source, rawIt->second.byteOffset,
                      targetMemref.getElementType()};
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

llvm::Error initializeTypedGraphExecutionState(
    SimulatorState &state, const PreparedGraphExecution &execution,
    dataflow::GraphOp graph, const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    const ResolvedLaunchContext &context) {
  state.graphScope = graph.getOperation();
  initializeRunState(state, execution);
  seedBlockArgument(state, graph.getStart(), noneToken());
  if (llvm::Error error =
          seedTypedDfgInputs(state, graph, workload, runtimeInput, context))
    return error;
  mlir::Block &entry = graph.getBody().front();
  if (llvm::Error error = initializeFreshMemoryRoots(entry, state))
    return error;
  return propagateMemoryAliases(entry, state);
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

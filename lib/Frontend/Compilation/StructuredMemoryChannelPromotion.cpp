#include "StructuredMemoryCommunicationDetail.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

namespace loom::frontend::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "structured_memory_channel_promotion_invalid: " + message);
}

mlir::Value exactMemoryRoot(mlir::Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<mlir::memref::CastOp>()) {
      value = cast.getSource();
      continue;
    }
    if (auto view = value.getDefiningOp<mlir::memref::SubViewOp>()) {
      value = view.getSource();
      continue;
    }
    if (auto reinterpret =
            value.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
      value = reinterpret.getSource();
      continue;
    }
    return value;
  }
}

enum class EndpointKind { Producer, Consumer };

struct EndpointPlan final {
  EndpointKind kind = EndpointKind::Producer;
  dataflow::ThreadLaunchOp launch;
  dataflow::ThreadOp thread;
  unsigned formalOrdinal = 0;
  loom::SpatialRegionOp spatial;
  unsigned memoryOrdinal = 0;
  mlir::BlockArgument memoryArgument;
  mlir::Operation *event = nullptr;
  llvm::SmallVector<unsigned, 4> readOrdinals;
  llvm::SmallVector<unsigned, 4> writeOrdinals;
};

struct ChannelPlan final {
  mlir::memref::AllocOp allocation;
  mlir::memref::DeallocOp deallocation;
  EndpointPlan producer;
  EndpointPlan consumer;
  dataflow::ThreadWaitOp producerWait;
  dataflow::ThreadWaitOp consumerWait;
};

std::optional<unsigned> spatialMemoryOrdinal(loom::SpatialRegionOp spatial,
                                             mlir::Value value) {
  for (auto [ordinal, input] : llvm::enumerate(spatial.getMemoryInputs()))
    if (input == value)
      return ordinal;
  return std::nullopt;
}

std::optional<unsigned>
threadFormalForSpatialMemory(loom::SpatialRegionOp spatial,
                             mlir::Value memory) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(memory);
  if (!argument || argument.getOwner() != &spatial.getBody().front())
    return std::nullopt;
  const std::uint64_t memoryBase =
      spatial.getValueInputs().size() + spatial.getStreamInputs().size();
  if (argument.getArgNumber() < memoryBase)
    return std::nullopt;
  const std::uint64_t memoryOrdinal = argument.getArgNumber() - memoryBase;
  if (memoryOrdinal >= spatial.getMemoryInputs().size())
    return std::nullopt;
  auto thread = spatial->getParentOfType<dataflow::ThreadOp>();
  auto formal = llvm::dyn_cast<mlir::BlockArgument>(
      spatial.getMemoryInputs()[memoryOrdinal]);
  if (!thread || !formal || formal.getOwner() != &thread.getBody().front() ||
      formal.getArgNumber() >= thread.getFunctionType().getNumInputs())
    return std::nullopt;
  return formal.getArgNumber();
}

bool isNormalizedDimension(mlir::scf::ForOp loop, std::int64_t extent) {
  if (!loop.getInitArgs().empty() || loop->getNumResults() != 0 || extent <= 0)
    return false;
  llvm::APInt lower;
  llvm::APInt upper;
  llvm::APInt step;
  return mlir::matchPattern(loop.getLowerBound(),
                            mlir::m_ConstantInt(&lower)) &&
         mlir::matchPattern(loop.getUpperBound(),
                            mlir::m_ConstantInt(&upper)) &&
         mlir::matchPattern(loop.getStep(), mlir::m_ConstantInt(&step)) &&
         lower.isZero() && upper.isSignedIntN(64) &&
         upper.getSExtValue() == extent && step.getLimitedValue(2) == 1;
}

bool coversExactLogicalDomain(mlir::Operation *event, mlir::ValueRange indices,
                              mlir::MemRefType type,
                              loom::SpatialRegionOp spatial) {
  if (indices.size() != static_cast<std::size_t>(type.getRank()))
    return false;
  llvm::SmallVector<mlir::scf::ForOp, 4> loops;
  for (mlir::Operation *parent = event->getParentOp(); parent != spatial;
       parent = parent ? parent->getParentOp() : nullptr) {
    auto loop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(parent);
    if (!loop)
      return false;
    loops.push_back(loop);
  }
  std::reverse(loops.begin(), loops.end());
  if (loops.size() != static_cast<std::size_t>(type.getRank()))
    return false;
  for (auto [dimension, loop, index] :
       llvm::zip(type.getShape(), loops, indices))
    if (!isNormalizedDimension(loop, dimension) ||
        index != loop.getInductionVar())
      return false;
  return true;
}

bool isAllowedStructure(mlir::Operation *operation) {
  return llvm::isa<mlir::scf::ForOp, mlir::scf::YieldOp, loom::SpatialYieldOp>(
             operation) ||
         mlir::isMemoryEffectFree(operation);
}

std::optional<EndpointPlan> analyzeEndpoint(dataflow::ThreadLaunchOp launch,
                                            unsigned formalOrdinal,
                                            mlir::MemRefType allocationType) {
  auto thread = mlir::SymbolTable::lookupNearestSymbolFrom<dataflow::ThreadOp>(
      launch, launch.getCalleeAttr());
  if (!thread || thread.isExternal() ||
      thread.getDomain().getKind() !=
          dataflow::ThreadDomainKind::DenseRectangular ||
      formalOrdinal >= thread.getFunctionType().getNumInputs() ||
      thread.getFunctionType().getInput(formalOrdinal) != allocationType ||
      thread.getBody().front().getNumArguments() !=
          thread.getFunctionType().getNumInputs() + 1)
    return std::nullopt;

  mlir::BlockArgument formal =
      thread.getBody().front().getArgument(formalOrdinal);
  if (!llvm::hasSingleElement(formal.getUses()))
    return std::nullopt;
  auto spatial =
      llvm::dyn_cast<loom::SpatialRegionOp>(formal.use_begin()->getOwner());
  if (!spatial || spatial->getBlock() != &thread.getBody().front())
    return std::nullopt;
  std::optional<unsigned> memoryOrdinal = spatialMemoryOrdinal(spatial, formal);
  if (!memoryOrdinal)
    return std::nullopt;
  const unsigned argumentOrdinal = spatial.getValueInputs().size() +
                                   spatial.getStreamInputs().size() +
                                   *memoryOrdinal;
  mlir::BlockArgument memoryArgument =
      spatial.getBody().front().getArgument(argumentOrdinal);
  if (memoryArgument.getType() != allocationType ||
      !llvm::hasSingleElement(memoryArgument.getUses()))
    return std::nullopt;

  mlir::Operation *event = memoryArgument.use_begin()->getOwner();
  EndpointKind kind;
  mlir::ValueRange eventIndices;
  if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(event)) {
    if (store.getMemref() != memoryArgument ||
        store.getValue().getType() != allocationType.getElementType())
      return std::nullopt;
    kind = EndpointKind::Producer;
    eventIndices = store.getIndices();
  } else if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(event)) {
    if (load.getMemref() != memoryArgument ||
        load.getResult().getType() != allocationType.getElementType())
      return std::nullopt;
    kind = EndpointKind::Consumer;
    eventIndices = load.getIndices();
  } else {
    return std::nullopt;
  }
  if (!coversExactLogicalDomain(event, eventIndices, allocationType, spatial))
    return std::nullopt;

  for (mlir::Operation &operation :
       thread.getBody().front().without_terminator())
    if (&operation != spatial.getOperation() &&
        !mlir::isMemoryEffectFree(&operation))
      return std::nullopt;

  EndpointPlan plan{kind,
                    launch,
                    thread,
                    formalOrdinal,
                    spatial,
                    *memoryOrdinal,
                    memoryArgument,
                    event,
                    {},
                    {}};
  bool legal = true;
  bool beforeReceive = kind == EndpointKind::Consumer;
  spatial.getBody().walk([&](mlir::Operation *operation) {
    if (!legal)
      return mlir::WalkResult::interrupt();
    if (operation == event) {
      beforeReceive = false;
      return mlir::WalkResult::advance();
    }
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation)) {
      std::optional<unsigned> ordinal =
          threadFormalForSpatialMemory(spatial, load.getMemref());
      if (!ordinal || beforeReceive) {
        legal = false;
        return mlir::WalkResult::interrupt();
      }
      plan.readOrdinals.push_back(*ordinal);
      return mlir::WalkResult::advance();
    }
    if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(operation)) {
      std::optional<unsigned> ordinal =
          threadFormalForSpatialMemory(spatial, store.getMemref());
      if (!ordinal || beforeReceive || kind == EndpointKind::Producer) {
        legal = false;
        return mlir::WalkResult::interrupt();
      }
      plan.writeOrdinals.push_back(*ordinal);
      return mlir::WalkResult::advance();
    }
    if (!isAllowedStructure(operation)) {
      legal = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!legal)
    return std::nullopt;

  llvm::sort(plan.readOrdinals);
  plan.readOrdinals.erase(
      std::unique(plan.readOrdinals.begin(), plan.readOrdinals.end()),
      plan.readOrdinals.end());
  llvm::sort(plan.writeOrdinals);
  plan.writeOrdinals.erase(
      std::unique(plan.writeOrdinals.begin(), plan.writeOrdinals.end()),
      plan.writeOrdinals.end());
  return plan;
}

std::optional<unsigned> launchBodyOrdinal(mlir::OpOperand &use,
                                          dataflow::ThreadLaunchOp launch) {
  if (use.getOperandNumber() >= launch.getBodyOperands().size())
    return std::nullopt;
  return use.getOperandNumber();
}

std::optional<dataflow::ThreadWaitOp>
uniqueWait(dataflow::ThreadLaunchOp launch) {
  if (!llvm::hasSingleElement(launch.getAsyncToken().getUses()))
    return std::nullopt;
  auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(
      launch.getAsyncToken().use_begin()->getOwner());
  if (!wait || wait.getAsyncDependencies().size() != 1 ||
      wait.getAsyncDependencies().front() != launch.getAsyncToken())
    return std::nullopt;
  return wait;
}

bool hasOnlyPureOperationsBetween(mlir::Operation *first,
                                  mlir::Operation *last) {
  for (mlir::Operation *operation = first->getNextNode(); operation != last;
       operation = operation ? operation->getNextNode() : nullptr)
    if (!operation || !mlir::isMemoryEffectFree(operation))
      return false;
  return true;
}

std::optional<ChannelPlan> analyzeChannel(mlir::memref::AllocOp allocation) {
  mlir::MemRefType type = allocation.getType();
  if (!type.hasStaticShape() || !type.getLayout().isIdentity() ||
      type.getRank() == 0 ||
      llvm::any_of(type.getShape(),
                   [](std::int64_t extent) { return extent <= 0; }) ||
      !allocation.getDynamicSizes().empty() ||
      !allocation.getSymbolOperands().empty() ||
      allocation->getParentOfType<dataflow::ThreadOp>() ||
      allocation->getParentOfType<dataflow::GraphOp>() ||
      allocation->getParentOfType<loom::SpatialRegionOp>())
    return std::nullopt;

  mlir::memref::DeallocOp deallocation;
  llvm::SmallVector<std::pair<dataflow::ThreadLaunchOp, unsigned>, 2> launches;
  for (mlir::OpOperand &use : allocation.getResult().getUses()) {
    if (auto dealloc =
            llvm::dyn_cast<mlir::memref::DeallocOp>(use.getOwner())) {
      if (deallocation || dealloc.getMemref() != allocation.getResult())
        return std::nullopt;
      deallocation = dealloc;
      continue;
    }
    auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(use.getOwner());
    std::optional<unsigned> ordinal =
        launch ? launchBodyOrdinal(use, launch) : std::nullopt;
    if (!ordinal)
      return std::nullopt;
    launches.emplace_back(launch, *ordinal);
  }
  if (!deallocation || launches.size() != 2 ||
      launches.front().first == launches.back().first)
    return std::nullopt;

  auto first = analyzeEndpoint(launches[0].first, launches[0].second, type);
  auto second = analyzeEndpoint(launches[1].first, launches[1].second, type);
  if (!first || !second || first->kind == second->kind)
    return std::nullopt;
  EndpointPlan producer =
      first->kind == EndpointKind::Producer ? *first : *second;
  EndpointPlan consumer =
      first->kind == EndpointKind::Consumer ? *first : *second;
  if (!producer.launch.getAsyncDependencies().empty() ||
      !consumer.launch.getAsyncDependencies().empty())
    return std::nullopt;
  std::optional<dataflow::ThreadWaitOp> producerWait =
      uniqueWait(producer.launch);
  std::optional<dataflow::ThreadWaitOp> consumerWait =
      uniqueWait(consumer.launch);
  mlir::Block *block = allocation->getBlock();
  if (!producerWait || !consumerWait || producer.launch->getBlock() != block ||
      consumer.launch->getBlock() != block ||
      (*producerWait)->getBlock() != block ||
      (*consumerWait)->getBlock() != block ||
      deallocation->getBlock() != block ||
      !allocation->isBeforeInBlock(producer.launch) ||
      !producer.launch->isBeforeInBlock(*producerWait) ||
      !(*producerWait)->isBeforeInBlock(consumer.launch) ||
      !consumer.launch->isBeforeInBlock(*consumerWait) ||
      !(*consumerWait)->isBeforeInBlock(deallocation) ||
      !hasOnlyPureOperationsBetween(*producerWait, consumer.launch))
    return std::nullopt;

  mlir::DominanceInfo dominance(block->getParentOp());
  if (llvm::any_of(consumer.launch->getOperands(), [&](mlir::Value operand) {
        return !dominance.dominates(operand, producerWait->getOperation());
      }))
    return std::nullopt;

  for (unsigned producerRead : producer.readOrdinals) {
    if (producerRead >= producer.launch.getBodyOperands().size())
      return std::nullopt;
    for (unsigned consumerWrite : consumer.writeOrdinals) {
      if (consumerWrite >= consumer.launch.getBodyOperands().size() ||
          !areKnownDistinctMemoryRoots(
              producer.launch.getBodyOperands()[producerRead],
              consumer.launch.getBodyOperands()[consumerWrite]))
        return std::nullopt;
    }
  }
  return ChannelPlan{allocation, deallocation,  producer,
                     consumer,   *producerWait, *consumerWait};
}

std::string freshThreadName(mlir::ModuleOp module, llvm::StringRef base,
                            EndpointKind kind) {
  std::string name = (llvm::Twine(base) + (kind == EndpointKind::Producer
                                               ? "_channel_producer"
                                               : "_channel_consumer"))
                         .str();
  while (mlir::SymbolTable::lookupSymbolIn(module, name))
    name.push_back('_');
  return name;
}

llvm::Expected<EndpointPlan>
specializeIfShared(EndpointPlan plan, mlir::MemRefType allocationType) {
  mlir::ModuleOp module = plan.thread->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return invalid("selected thread has no module owner");
  std::optional<mlir::SymbolTable::UseRange> uses =
      mlir::SymbolTable::getSymbolUses(plan.thread, module);
  if (uses && llvm::hasSingleElement(*uses) &&
      uses->begin()->getUser() == plan.launch.getOperation())
    return plan;

  auto clone = llvm::cast<dataflow::ThreadOp>(plan.thread->clone());
  clone.setSymName(
      freshThreadName(module, plan.thread.getSymName(), plan.kind));
  mlir::OpBuilder builder(plan.thread);
  builder.setInsertionPointAfter(plan.thread);
  builder.insert(clone.getOperation());
  plan.launch.setCallee(clone.getSymName());
  auto specialized =
      analyzeEndpoint(plan.launch, plan.formalOrdinal, allocationType);
  if (!specialized || specialized->kind != plan.kind)
    return invalid("specialized thread no longer has the selected endpoint");
  return *specialized;
}

void setThreadFormalType(EndpointPlan &plan, mlir::Type type) {
  llvm::SmallVector<mlir::Type, 4> inputs(
      plan.thread.getFunctionType().getInputs());
  inputs[plan.formalOrdinal] = type;
  plan.thread.setFunctionType(
      mlir::FunctionType::get(plan.thread.getContext(), inputs, {}));
  plan.thread.getBody().front().getArgument(plan.formalOrdinal).setType(type);
}

llvm::Error rewriteEndpoint(EndpointPlan &plan, mlir::Type channelType) {
  setThreadFormalType(plan, channelType);
  mlir::BlockArgument formal =
      plan.thread.getBody().front().getArgument(plan.formalOrdinal);
  mlir::Block &entry = plan.spatial.getBody().front();
  mlir::BlockArgument oldMemoryArgument = plan.memoryArgument;
  mlir::BlockArgument channelArgument;

  if (plan.kind == EndpointKind::Producer) {
    plan.spatial.getStreamOutputsMutable().append(formal);
    plan.spatial.getMemoryInputsMutable().erase(plan.memoryOrdinal);
    channelArgument =
        entry.addArgument(channelType, oldMemoryArgument.getLoc());
    auto store = llvm::dyn_cast<mlir::memref::StoreOp>(plan.event);
    if (!store)
      return invalid("producer endpoint is no longer a store");
    mlir::OpBuilder builder(store);
    dataflow::ChannelSendOp::create(builder, store.getLoc(), channelArgument,
                                    store.getValue());
    store.erase();
  } else {
    const unsigned streamArgumentOrdinal =
        plan.spatial.getValueInputs().size() +
        plan.spatial.getStreamInputs().size();
    plan.spatial.getStreamInputsMutable().append(formal);
    plan.spatial.getMemoryInputsMutable().erase(plan.memoryOrdinal);
    channelArgument = entry.insertArgument(streamArgumentOrdinal, channelType,
                                           oldMemoryArgument.getLoc());
    llvm::SmallVector<mlir::Attribute, 4> sourceMaps(
        plan.spatial.getSourceMaps().begin(),
        plan.spatial.getSourceMaps().end());
    sourceMaps.push_back(mlir::AffineMapAttr::get(
        mlir::AffineMap::get(0, 0, {}, plan.spatial.getContext())));
    plan.spatial.setSourceMapsAttr(
        mlir::ArrayAttr::get(plan.spatial.getContext(), sourceMaps));
    auto load = llvm::dyn_cast<mlir::memref::LoadOp>(plan.event);
    if (!load)
      return invalid("consumer endpoint is no longer a load");
    mlir::OpBuilder builder(load);
    auto receive = dataflow::ChannelReceiveOp::create(builder, load.getLoc(),
                                                      channelArgument);
    load.getResult().replaceAllUsesWith(receive.getMessage());
    load.erase();
  }
  if (!oldMemoryArgument.use_empty())
    return invalid("selected memory argument retained an unproved use");
  entry.eraseArgument(oldMemoryArgument.getArgNumber());
  return llvm::Error::success();
}

} // namespace

bool areKnownDistinctMemoryRoots(mlir::Value lhs, mlir::Value rhs) {
  lhs = exactMemoryRoot(lhs);
  rhs = exactMemoryRoot(rhs);
  if (lhs == rhs)
    return false;
  auto lhsGlobal = lhs.getDefiningOp<mlir::memref::GetGlobalOp>();
  auto rhsGlobal = rhs.getDefiningOp<mlir::memref::GetGlobalOp>();
  auto lhsAlloc = lhs.getDefiningOp<mlir::memref::AllocOp>();
  auto rhsAlloc = rhs.getDefiningOp<mlir::memref::AllocOp>();
  if (lhsGlobal && rhsGlobal)
    return lhsGlobal.getName() != rhsGlobal.getName();
  if (lhsAlloc && rhsAlloc)
    return lhsAlloc != rhsAlloc;
  return (lhsGlobal && rhsAlloc) || (lhsAlloc && rhsGlobal);
}

bool canPromoteSpscBufferToChannel(mlir::memref::AllocOp allocation) {
  return analyzeChannel(allocation).has_value();
}

llvm::Error promoteSpscBufferToChannel(mlir::memref::AllocOp allocation) {
  std::optional<ChannelPlan> analyzed = analyzeChannel(allocation);
  if (!analyzed)
    return invalid(
        "selected allocation is not an exact promotable SPSC buffer");
  ChannelPlan plan = *analyzed;
  mlir::MemRefType allocationType = allocation.getType();
  auto producer = specializeIfShared(plan.producer, allocationType);
  if (!producer)
    return producer.takeError();
  plan.producer = *producer;
  auto consumer = specializeIfShared(plan.consumer, allocationType);
  if (!consumer)
    return consumer.takeError();
  plan.consumer = *consumer;

  mlir::Type channelType = dataflow::ChannelType::get(
      allocation.getContext(), allocationType.getElementType());
  mlir::OpBuilder builder(allocation);
  auto channel = dataflow::ChannelCreateOp::create(builder, allocation.getLoc(),
                                                   channelType);
  if (llvm::Error error = rewriteEndpoint(plan.producer, channelType))
    return error;
  if (llvm::Error error = rewriteEndpoint(plan.consumer, channelType))
    return error;
  plan.producer.launch->setOperand(plan.producer.formalOrdinal,
                                   channel.getChannel());
  plan.consumer.launch->setOperand(plan.consumer.formalOrdinal,
                                   channel.getChannel());
  plan.consumer.launch->moveBefore(plan.producerWait);
  plan.deallocation.erase();
  if (!allocation.getResult().use_empty())
    return invalid("promoted allocation retained an unproved use");
  allocation.erase();
  return llvm::Error::success();
}

} // namespace loom::frontend::detail

#include "StructuredMemoryCommunicationDetail.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
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
    if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>()) {
      value = gep.getBase();
      continue;
    }
    return value;
  }
}

bool isEnclosingBlockArgument(mlir::Value value, mlir::Operation *operation) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument)
    return false;
  for (mlir::Operation *current = operation; current;
       current = current->getParentOp())
    if (current->getBlock() == argument.getOwner())
      return true;
  return false;
}

bool areDistinctNoAliasFunctionArguments(mlir::Value lhs, mlir::Value rhs) {
  auto lhsArgument = llvm::dyn_cast<mlir::BlockArgument>(lhs);
  auto rhsArgument = llvm::dyn_cast<mlir::BlockArgument>(rhs);
  if (!lhsArgument || !rhsArgument ||
      lhsArgument.getOwner() != rhsArgument.getOwner() ||
      lhsArgument.getArgNumber() == rhsArgument.getArgNumber())
    return false;
  auto function = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
      lhsArgument.getOwner()->getParentOp());
  if (!function)
    return false;
  mlir::DictionaryAttr lhsAttrs = mlir::function_interface_impl::getArgAttrDict(
      function, lhsArgument.getArgNumber());
  mlir::DictionaryAttr rhsAttrs = mlir::function_interface_impl::getArgAttrDict(
      function, rhsArgument.getArgNumber());
  llvm::StringRef noAlias = mlir::LLVM::LLVMDialect::getNoAliasAttrName();
  return lhsAttrs && rhsAttrs && lhsAttrs.contains(noAlias) &&
         rhsAttrs.contains(noAlias);
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
  llvm::SmallVector<EndpointPlan, 4> consumers;
  dataflow::ThreadWaitOp producerWait;
  llvm::SmallVector<dataflow::ThreadWaitOp, 4> consumerWaits;
};

struct SourceAllocationShape final {
  mlir::Type elementType;
  std::uint64_t elementCount = 0;
  std::uint64_t elementBytes = 0;
};

struct SourcePointerAccess final {
  mlir::Operation *event = nullptr;
  mlir::LLVM::GEPOp address;
  std::uint64_t byteOffset = 0;
};

struct SourceEndpointPlan final {
  EndpointKind kind = EndpointKind::Producer;
  dataflow::ThreadLaunchOp launch;
  dataflow::ThreadOp thread;
  unsigned formalOrdinal = 0;
  loom::SpatialRegionOp spatial;
  unsigned valueOrdinal = 0;
  mlir::BlockArgument pointerArgument;
  llvm::SmallVector<SourcePointerAccess, 8> accesses;
  llvm::SmallVector<unsigned, 4> readOrdinals;
  llvm::SmallVector<unsigned, 4> writeOrdinals;
};

struct SourceChannelPlan final {
  mlir::LLVM::AllocaOp allocation;
  llvm::SmallVector<mlir::Operation *, 2> lifetimeMarkers;
  SourceAllocationShape shape;
  SourceEndpointPlan producer;
  llvm::SmallVector<SourceEndpointPlan, 4> consumers;
  dataflow::ThreadWaitOp producerWait;
  llvm::SmallVector<dataflow::ThreadLaunchOp, 4> launchesToMove;
};

struct ThreadEffectPlan final {
  dataflow::ThreadLaunchOp launch;
  llvm::SmallVector<unsigned, 4> readOrdinals;
  llvm::SmallVector<unsigned, 4> writeOrdinals;
};

struct TransparentOwnedThreadWrapper final {
  mlir::LLVM::LLVMFuncOp function;
  dataflow::ThreadLaunchOp launch;
  dataflow::ThreadWaitOp wait;
};

bool isAllowedStructure(mlir::Operation *operation);
std::optional<unsigned> launchBodyOrdinal(mlir::OpOperand &use,
                                          dataflow::ThreadLaunchOp launch);

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

std::optional<unsigned>
threadFormalForSpatialValue(loom::SpatialRegionOp spatial, mlir::Value value) {
  value = exactMemoryRoot(value);
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner() != &spatial.getBody().front() ||
      argument.getArgNumber() >= spatial.getValueInputs().size())
    return std::nullopt;
  auto thread = spatial->getParentOfType<dataflow::ThreadOp>();
  auto formal = llvm::dyn_cast<mlir::BlockArgument>(
      spatial.getValueInputs()[argument.getArgNumber()]);
  if (!thread || !formal || formal.getOwner() != &thread.getBody().front() ||
      formal.getArgNumber() >= thread.getFunctionType().getNumInputs())
    return std::nullopt;
  return formal.getArgNumber();
}

std::optional<SourceAllocationShape>
analyzeSourceAllocationShape(mlir::LLVM::AllocaOp allocation) {
  llvm::APInt arraySize;
  if (allocation.getInalloca() ||
      !mlir::matchPattern(allocation.getArraySize(),
                          mlir::m_ConstantInt(&arraySize)) ||
      !arraySize.isOne())
    return std::nullopt;

  mlir::Type element = allocation.getElemType();
  std::uint64_t elementCount = 1;
  bool aggregate = false;
  while (auto array = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(element)) {
    aggregate = true;
    if (array.getNumElements() == 0 ||
        elementCount >
            std::numeric_limits<std::uint64_t>::max() / array.getNumElements())
      return std::nullopt;
    elementCount *= array.getNumElements();
    element = array.getElementType();
  }
  if (!aggregate || !llvm::isa<mlir::IntegerType, mlir::FloatType>(element))
    return std::nullopt;
  mlir::DataLayout layout = mlir::DataLayout::closest(allocation);
  llvm::TypeSize bytes = layout.getTypeSize(element);
  if (bytes.isScalable() || bytes.getFixedValue() == 0 ||
      elementCount >
          std::numeric_limits<std::uint64_t>::max() / bytes.getFixedValue())
    return std::nullopt;
  llvm::TypeSize aggregateBytes = layout.getTypeSize(allocation.getElemType());
  if (aggregateBytes.isScalable() ||
      aggregateBytes.getFixedValue() != elementCount * bytes.getFixedValue())
    return std::nullopt;
  return SourceAllocationShape{element, elementCount, bytes.getFixedValue()};
}

std::optional<std::uint64_t>
sourcePointerByteOffset(mlir::Value address, mlir::Value root,
                        mlir::LLVM::GEPOp &addressOp) {
  if (address == root) {
    addressOp = {};
    return 0;
  }
  auto gep = address.getDefiningOp<mlir::LLVM::GEPOp>();
  if (!gep || gep.getBase() != root ||
      !mlir::LLVM::bitEnumContainsAny(
          gep.getNoWrapFlags(), mlir::LLVM::GEPNoWrapFlags::inboundsFlag) ||
      gep.getRawConstantIndices().size() != 1)
    return std::nullopt;
  auto integerElement = llvm::dyn_cast<mlir::IntegerType>(gep.getElemType());
  if ((!integerElement || integerElement.getWidth() != 8) &&
      !llvm::isa<mlir::LLVM::LLVMByteType>(gep.getElemType()))
    return std::nullopt;

  const std::int32_t raw = gep.getRawConstantIndices().front();
  std::int64_t offset = raw;
  if (raw == mlir::LLVM::GEPOp::kDynamicIndex) {
    if (!llvm::hasSingleElement(gep.getDynamicIndices()))
      return std::nullopt;
    llvm::APInt constant;
    if (!mlir::matchPattern(gep.getDynamicIndices().front(),
                            mlir::m_ConstantInt(&constant)) ||
        !constant.isSignedIntN(64))
      return std::nullopt;
    offset = constant.getSExtValue();
  } else if (!gep.getDynamicIndices().empty()) {
    return std::nullopt;
  }
  if (offset <= 0 || !llvm::hasSingleElement(gep.getRes().getUses()))
    return std::nullopt;
  addressOp = gep;
  return static_cast<std::uint64_t>(offset);
}

mlir::Value llvmMemoryAddress(mlir::Operation *operation) {
  if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(operation))
    return load.getAddr();
  if (auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(operation))
    return store.getAddr();
  return {};
}

std::optional<TransparentOwnedThreadWrapper>
transparentOwnedThreadWrapper(mlir::LLVM::CallOp call) {
  if (!call || !call.getCalleeAttr() || call.getNumResults() != 0)
    return std::nullopt;
  auto function =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
          call, call.getCalleeAttr());
  if (!function || function.isExternal() || function.isVarArg() ||
      !function.getBody().hasOneBlock() ||
      !llvm::isa<mlir::LLVM::LLVMVoidType>(
          function.getFunctionType().getReturnType()) ||
      call.getArgOperands().size() !=
          function.getFunctionType().getParams().size())
    return std::nullopt;

  mlir::Block &block = function.getBody().front();
  if (block.getNumArguments() != call.getArgOperands().size())
    return std::nullopt;
  auto operation = block.begin();
  if (operation == block.end())
    return std::nullopt;
  auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(&*operation++);
  if (!launch || !launch.getGridUpperBounds().empty() ||
      !launch.getAsyncDependencies().empty() ||
      launch.getBodyOperands().size() != block.getNumArguments())
    return std::nullopt;
  for (auto [operand, argument] :
       llvm::zip_equal(launch.getBodyOperands(), block.getArguments()))
    if (operand != argument)
      return std::nullopt;
  if (operation == block.end())
    return std::nullopt;
  auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(&*operation++);
  if (!wait || wait.getAsyncDependencies().size() != 1 ||
      wait.getAsyncDependencies().front() != launch.getAsyncToken() ||
      !llvm::hasSingleElement(launch.getAsyncToken().getUses()))
    return std::nullopt;
  if (operation == block.end())
    return std::nullopt;
  auto returned = llvm::dyn_cast<mlir::LLVM::ReturnOp>(&*operation++);
  if (!returned || returned.getNumOperands() != 0 || operation != block.end())
    return std::nullopt;
  return TransparentOwnedThreadWrapper{function, launch, wait};
}

llvm::Error
liftTransparentOwnedThreadWrapperCalls(mlir::LLVM::AllocaOp allocation) {
  llvm::SmallVector<mlir::LLVM::CallOp, 4> calls;
  llvm::SmallPtrSet<mlir::Operation *, 4> seenCalls;
  for (mlir::OpOperand &use : allocation.getRes().getUses()) {
    if (llvm::isa<mlir::LLVM::LifetimeStartOp, mlir::LLVM::LifetimeEndOp,
                  dataflow::ThreadLaunchOp>(use.getOwner()))
      continue;
    auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(use.getOwner());
    if (!call || !transparentOwnedThreadWrapper(call))
      return invalid("source allocation has a non-transparent callable use");
    if (seenCalls.insert(call.getOperation()).second)
      calls.push_back(call);
  }

  for (mlir::LLVM::CallOp call : calls) {
    auto wrapper = transparentOwnedThreadWrapper(call);
    if (!wrapper)
      return invalid("source allocation wrapper changed during lifting");
    mlir::OpBuilder builder(call);
    auto launch = dataflow::ThreadLaunchOp::create(
        builder, call.getLoc(), wrapper->launch.getCalleeAttr(),
        call.getArgOperands(), mlir::ValueRange{}, mlir::ValueRange{});
    dataflow::ThreadWaitOp::create(builder, call.getLoc(),
                                   mlir::ValueRange{launch.getAsyncToken()});
    call.erase();
  }
  return llvm::Error::success();
}

std::optional<SourceEndpointPlan>
analyzeSourceEndpoint(dataflow::ThreadLaunchOp launch, unsigned formalOrdinal,
                      const SourceAllocationShape &shape,
                      mlir::Type pointerType) {
  auto thread = mlir::SymbolTable::lookupNearestSymbolFrom<dataflow::ThreadOp>(
      launch, launch.getCalleeAttr());
  if (!thread || thread.isExternal() ||
      thread.getDomain().getKind() !=
          dataflow::ThreadDomainKind::DenseRectangular ||
      formalOrdinal >= thread.getFunctionType().getNumInputs() ||
      thread.getFunctionType().getInput(formalOrdinal) != pointerType ||
      thread.getBody().front().getNumArguments() !=
          thread.getFunctionType().getNumInputs() + 1) {
    return std::nullopt;
  }

  mlir::BlockArgument formal =
      thread.getBody().front().getArgument(formalOrdinal);
  if (!llvm::hasSingleElement(formal.getUses())) {
    return std::nullopt;
  }
  auto spatial =
      llvm::dyn_cast<loom::SpatialRegionOp>(formal.use_begin()->getOwner());
  if (!spatial || spatial->getBlock() != &thread.getBody().front()) {
    return std::nullopt;
  }
  auto value = llvm::find(spatial.getValueInputs(), formal);
  if (value == spatial.getValueInputs().end()) {
    return std::nullopt;
  }
  const unsigned valueOrdinal = value - spatial.getValueInputs().begin();
  mlir::BlockArgument pointerArgument =
      spatial.getBody().front().getArgument(valueOrdinal);
  if (pointerArgument.getType() != pointerType) {
    return std::nullopt;
  }

  std::optional<EndpointKind> endpointKind;
  for (mlir::OpOperand &use : pointerArgument.getUses()) {
    mlir::Operation *owner = use.getOwner();
    mlir::Operation *event = owner;
    if (!(llvm::isa<mlir::LLVM::LoadOp, mlir::LLVM::StoreOp>(owner) &&
          llvmMemoryAddress(owner) == pointerArgument)) {
      auto gep = llvm::dyn_cast<mlir::LLVM::GEPOp>(owner);
      mlir::LLVM::GEPOp ignored;
      if (!gep || gep.getBase() != pointerArgument ||
          !sourcePointerByteOffset(gep.getRes(), pointerArgument, ignored)) {
        return std::nullopt;
      }
      event = gep.getRes().use_begin()->getOwner();
    }
    if (!llvm::isa<mlir::LLVM::LoadOp, mlir::LLVM::StoreOp>(event) ||
        exactMemoryRoot(llvmMemoryAddress(event)) != pointerArgument) {
      return std::nullopt;
    }
    const EndpointKind kind = llvm::isa<mlir::LLVM::StoreOp>(event)
                                  ? EndpointKind::Producer
                                  : EndpointKind::Consumer;
    if (endpointKind && *endpointKind != kind) {
      return std::nullopt;
    }
    endpointKind = kind;
  }
  if (!endpointKind) {
    return std::nullopt;
  }

  SourceEndpointPlan plan;
  plan.launch = launch;
  plan.thread = thread;
  plan.formalOrdinal = formalOrdinal;
  plan.spatial = spatial;
  plan.valueOrdinal = valueOrdinal;
  plan.pointerArgument = pointerArgument;
  plan.kind = *endpointKind;
  bool legal = true;
  bool sawSelectedEvent = false;
  bool beforeReceive = plan.kind == EndpointKind::Consumer;
  spatial.getBody().walk([&](mlir::Operation *operation) {
    if (!legal)
      return mlir::WalkResult::interrupt();
    if (mlir::Value address = llvmMemoryAddress(operation)) {
      mlir::LLVM::GEPOp addressOp;
      std::optional<std::uint64_t> offset =
          sourcePointerByteOffset(address, pointerArgument, addressOp);
      if (offset) {
        if (operation->getBlock() != &spatial.getBody().front()) {
          legal = false;
          return mlir::WalkResult::interrupt();
        }
        EndpointKind kind;
        mlir::Type elementType;
        if (auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(operation)) {
          kind = EndpointKind::Producer;
          elementType = store.getValue().getType();
        } else {
          kind = EndpointKind::Consumer;
          elementType = llvm::cast<mlir::LLVM::LoadOp>(operation).getType();
        }
        if (elementType != shape.elementType || plan.kind != kind) {
          legal = false;
          return mlir::WalkResult::interrupt();
        }
        sawSelectedEvent = true;
        beforeReceive = false;
        plan.accesses.push_back({operation, addressOp, *offset});
        return mlir::WalkResult::advance();
      }

      std::optional<unsigned> ordinal =
          threadFormalForSpatialValue(spatial, address);
      if (!ordinal) {
        legal = false;
        return mlir::WalkResult::interrupt();
      }
      if (llvm::isa<mlir::LLVM::LoadOp>(operation)) {
        plan.readOrdinals.push_back(*ordinal);
      } else {
        if (beforeReceive || plan.kind == EndpointKind::Producer) {
          legal = false;
          return mlir::WalkResult::interrupt();
        }
        plan.writeOrdinals.push_back(*ordinal);
      }
      return mlir::WalkResult::advance();
    }
    if (!isAllowedStructure(operation)) {
      legal = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!legal || !sawSelectedEvent ||
      plan.accesses.size() != shape.elementCount) {
    return std::nullopt;
  }
  for (auto [ordinal, access] : llvm::enumerate(plan.accesses)) {
    if (access.byteOffset != ordinal * shape.elementBytes ||
        (ordinal == 0) != !access.address) {
      return std::nullopt;
    }
  }
  for (mlir::Operation &operation :
       thread.getBody().front().without_terminator())
    if (&operation != spatial.getOperation() &&
        !mlir::isMemoryEffectFree(&operation)) {
      return std::nullopt;
    }
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
         llvm::isa<dataflow::ChannelSendOp, dataflow::ChannelReceiveOp>(
             operation) ||
         mlir::isMemoryEffectFree(operation);
}

std::optional<ThreadEffectPlan>
analyzeThreadEffects(dataflow::ThreadLaunchOp launch) {
  auto thread = mlir::SymbolTable::lookupNearestSymbolFrom<dataflow::ThreadOp>(
      launch, launch.getCalleeAttr());
  if (!thread || thread.isExternal() ||
      launch.getBodyOperands().size() !=
          thread.getFunctionType().getNumInputs() ||
      thread.getBody().front().getNumArguments() !=
          thread.getFunctionType().getNumInputs() + 1)
    return std::nullopt;

  loom::SpatialRegionOp spatial;
  for (mlir::Operation &operation :
       thread.getBody().front().without_terminator()) {
    if (auto candidate = llvm::dyn_cast<loom::SpatialRegionOp>(operation)) {
      if (spatial)
        return std::nullopt;
      spatial = candidate;
      continue;
    }
    if (!mlir::isMemoryEffectFree(&operation))
      return std::nullopt;
  }
  if (!spatial)
    return std::nullopt;

  ThreadEffectPlan plan{launch, {}, {}};
  bool legal = true;
  spatial.getBody().walk([&](mlir::Operation *operation) {
    if (!legal)
      return mlir::WalkResult::interrupt();
    if (mlir::Value address = llvmMemoryAddress(operation)) {
      std::optional<unsigned> ordinal =
          threadFormalForSpatialValue(spatial, address);
      if (!ordinal) {
        legal = false;
        return mlir::WalkResult::interrupt();
      }
      if (llvm::isa<mlir::LLVM::LoadOp>(operation))
        plan.readOrdinals.push_back(*ordinal);
      else
        plan.writeOrdinals.push_back(*ordinal);
      return mlir::WalkResult::advance();
    }
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation)) {
      std::optional<unsigned> ordinal =
          threadFormalForSpatialMemory(spatial, load.getMemref());
      if (!ordinal) {
        legal = false;
        return mlir::WalkResult::interrupt();
      }
      plan.readOrdinals.push_back(*ordinal);
      return mlir::WalkResult::advance();
    }
    if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(operation)) {
      std::optional<unsigned> ordinal =
          threadFormalForSpatialMemory(spatial, store.getMemref());
      if (!ordinal) {
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

std::optional<EndpointKind> channelBindingKind(dataflow::ThreadLaunchOp launch,
                                               unsigned formalOrdinal) {
  auto thread = mlir::SymbolTable::lookupNearestSymbolFrom<dataflow::ThreadOp>(
      launch, launch.getCalleeAttr());
  if (!thread || thread.isExternal() ||
      formalOrdinal >= thread.getFunctionType().getNumInputs())
    return std::nullopt;
  mlir::BlockArgument formal =
      thread.getBody().front().getArgument(formalOrdinal);
  if (!llvm::hasSingleElement(formal.getUses()))
    return std::nullopt;
  auto spatial =
      llvm::dyn_cast<loom::SpatialRegionOp>(formal.use_begin()->getOwner());
  if (!spatial)
    return std::nullopt;
  if (llvm::is_contained(spatial.getStreamOutputs(), formal))
    return EndpointKind::Producer;
  if (llvm::is_contained(spatial.getStreamInputs(), formal))
    return EndpointKind::Consumer;
  return std::nullopt;
}

std::optional<dataflow::ThreadLaunchOp>
channelProducerLaunch(dataflow::ThreadLaunchOp consumer,
                      unsigned consumerOrdinal) {
  if (consumerOrdinal >= consumer.getBodyOperands().size() ||
      channelBindingKind(consumer, consumerOrdinal) != EndpointKind::Consumer)
    return std::nullopt;
  mlir::Value channel = consumer.getBodyOperands()[consumerOrdinal];
  if (!llvm::isa<dataflow::ChannelType>(channel.getType()))
    return std::nullopt;

  dataflow::ThreadLaunchOp producer;
  for (mlir::OpOperand &use : channel.getUses()) {
    auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(use.getOwner());
    std::optional<unsigned> ordinal =
        launch ? launchBodyOrdinal(use, launch) : std::nullopt;
    if (!ordinal ||
        channelBindingKind(launch, *ordinal) != EndpointKind::Producer)
      continue;
    if (producer && producer != launch)
      return std::nullopt;
    producer = launch;
  }
  return producer ? std::optional(producer) : std::nullopt;
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

template <typename Plan>
std::optional<mlir::Value> launchOperand(const Plan &plan, unsigned ordinal) {
  dataflow::ThreadLaunchOp launch = plan.launch;
  if (ordinal >= launch.getBodyOperands().size())
    return std::nullopt;
  return launch.getBodyOperands()[ordinal];
}

template <typename LhsPlan, typename RhsPlan>
bool haveIndependentRemainingEffects(const LhsPlan &lhs, const RhsPlan &rhs) {
  auto areDistinct = [&](unsigned lhsOrdinal, unsigned rhsOrdinal) {
    std::optional<mlir::Value> lhsRoot = launchOperand(lhs, lhsOrdinal);
    std::optional<mlir::Value> rhsRoot = launchOperand(rhs, rhsOrdinal);
    return lhsRoot && rhsRoot &&
           areKnownDistinctMemoryRoots(*lhsRoot, *rhsRoot);
  };
  for (unsigned write : lhs.writeOrdinals) {
    for (unsigned read : rhs.readOrdinals)
      if (!areDistinct(write, read))
        return false;
    for (unsigned otherWrite : rhs.writeOrdinals)
      if (!areDistinct(write, otherWrite))
        return false;
  }
  for (unsigned read : lhs.readOrdinals)
    for (unsigned write : rhs.writeOrdinals)
      if (!areDistinct(read, write))
        return false;
  return true;
}

template <typename Plan>
bool haveIndependentFanoutEffects(const Plan &producer,
                                  llvm::ArrayRef<Plan> consumers) {
  for (const Plan &consumer : consumers)
    if (!haveIndependentRemainingEffects(producer, consumer))
      return false;
  for (std::size_t left = 0; left < consumers.size(); ++left)
    for (std::size_t right = left + 1; right < consumers.size(); ++right)
      if (!haveIndependentRemainingEffects(consumers[left], consumers[right]))
        return false;
  return true;
}

std::optional<llvm::SmallVector<dataflow::ThreadLaunchOp, 4>>
sourceLaunchesToMove(const SourceEndpointPlan &producer,
                     llvm::ArrayRef<SourceEndpointPlan> consumers,
                     dataflow::ThreadWaitOp producerWait) {
  mlir::Block *block = producer.launch->getBlock();
  if (!block || producerWait->getBlock() != block)
    return std::nullopt;

  llvm::SmallVector<dataflow::ThreadLaunchOp, 8> pending;
  llvm::SmallPtrSet<mlir::Operation *, 8> closure;
  for (const SourceEndpointPlan &consumer : consumers) {
    dataflow::ThreadLaunchOp consumerLaunch = consumer.launch;
    if (consumerLaunch->getBlock() != block ||
        !producer.launch->isBeforeInBlock(consumerLaunch))
      return std::nullopt;
    if (closure.insert(consumerLaunch.getOperation()).second)
      pending.push_back(consumerLaunch);
  }

  auto appendChannelDependencies = [&](std::size_t begin) {
    for (std::size_t index = begin; index < pending.size(); ++index) {
      dataflow::ThreadLaunchOp launch = pending[index];
      for (auto [ordinal, operand] :
           llvm::enumerate(launch.getBodyOperands())) {
        if (!llvm::isa<dataflow::ChannelType>(operand.getType()) ||
            channelBindingKind(launch, ordinal) != EndpointKind::Consumer)
          continue;
        auto dependency = channelProducerLaunch(launch, ordinal);
        if (!dependency || (*dependency)->getBlock() != block ||
            *dependency == producer.launch ||
            !(*dependency)->isBeforeInBlock(launch))
          return false;
        if (closure.insert(dependency->getOperation()).second)
          pending.push_back(*dependency);
      }
    }
    return true;
  };
  if (!appendChannelDependencies(0))
    return std::nullopt;

  llvm::sort(pending,
             [](dataflow::ThreadLaunchOp lhs, dataflow::ThreadLaunchOp rhs) {
               return lhs->isBeforeInBlock(rhs);
             });
  auto reversedPending = llvm::reverse(pending);
  auto lastAfterWait =
      llvm::find_if(reversedPending, [&](dataflow::ThreadLaunchOp launch) {
        return producerWait->isBeforeInBlock(launch);
      });
  if (lastAfterWait != reversedPending.end()) {
    const std::size_t dependencyCount = pending.size();
    mlir::Operation *last = lastAfterWait->getOperation();
    for (mlir::Operation *operation = producerWait->getNextNode(); operation;
         operation = operation->getNextNode()) {
      if (auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(operation);
          launch && closure.insert(operation).second)
        pending.push_back(launch);
      if (operation == last)
        break;
    }
    if (!appendChannelDependencies(dependencyCount))
      return std::nullopt;
    llvm::sort(pending,
               [](dataflow::ThreadLaunchOp lhs, dataflow::ThreadLaunchOp rhs) {
                 return lhs->isBeforeInBlock(rhs);
               });
  }

  llvm::SmallVector<dataflow::ThreadLaunchOp, 4> moved;
  for (dataflow::ThreadLaunchOp launch : pending) {
    std::optional<dataflow::ThreadWaitOp> wait = uniqueWait(launch);
    if (!wait || !launch->isBeforeInBlock(*wait))
      return std::nullopt;
    if (producerWait->isBeforeInBlock(launch))
      moved.push_back(launch);
  }

  mlir::DominanceInfo dominance(block->getParentOp());
  for (dataflow::ThreadLaunchOp launch : moved)
    if (llvm::any_of(launch->getOperands(), [&](mlir::Value operand) {
          return !dominance.dominates(operand, producerWait.getOperation());
        }))
      return std::nullopt;

  for (dataflow::ThreadLaunchOp launch : pending) {
    auto selected =
        llvm::find_if(consumers, [&](const SourceEndpointPlan &plan) {
          return plan.launch == launch;
        });
    if (selected != consumers.end()) {
      if (!haveIndependentRemainingEffects(producer, *selected))
        return std::nullopt;
      continue;
    }
    auto effects = analyzeThreadEffects(launch);
    if (!effects || !haveIndependentRemainingEffects(producer, *effects))
      return std::nullopt;
  }

  if (moved.empty())
    return moved;

  llvm::SmallPtrSet<mlir::Operation *, 8> movedOperations;
  for (dataflow::ThreadLaunchOp launch : moved)
    movedOperations.insert(launch.getOperation());
  mlir::Operation *last = moved.back().getOperation();
  for (mlir::Operation *operation = producerWait->getNextNode(); operation;
       operation = operation->getNextNode()) {
    if (movedOperations.contains(operation)) {
      if (operation == last)
        break;
      continue;
    }
    if (mlir::isMemoryEffectFree(operation)) {
      if (operation == last)
        break;
      continue;
    }
    auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(operation);
    if (!wait ||
        llvm::any_of(wait.getAsyncDependencies(), [&](mlir::Value token) {
          auto launch = token.getDefiningOp<dataflow::ThreadLaunchOp>();
          return !launch || !closure.contains(launch.getOperation());
        }))
      return std::nullopt;
    if (operation == last)
      break;
  }
  return moved;
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
  llvm::SmallVector<std::pair<dataflow::ThreadLaunchOp, unsigned>, 4> launches;
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
  if (!deallocation || launches.size() < 2)
    return std::nullopt;
  mlir::Block *block = allocation->getBlock();
  llvm::SmallPtrSet<mlir::Operation *, 8> uniqueLaunches;
  for (auto [launch, ordinal] : launches) {
    (void)ordinal;
    if (launch->getBlock() != block ||
        !uniqueLaunches.insert(launch.getOperation()).second)
      return std::nullopt;
  }
  llvm::sort(launches, [](auto lhs, auto rhs) {
    return lhs.first->isBeforeInBlock(rhs.first);
  });

  std::optional<EndpointPlan> producer;
  llvm::SmallVector<EndpointPlan, 4> consumers;
  for (auto [launch, ordinal] : launches) {
    auto endpoint = analyzeEndpoint(launch, ordinal, type);
    if (!endpoint || !launch.getAsyncDependencies().empty())
      return std::nullopt;
    if (endpoint->kind == EndpointKind::Producer) {
      if (producer)
        return std::nullopt;
      producer = std::move(*endpoint);
    } else {
      consumers.push_back(std::move(*endpoint));
    }
  }
  if (!producer || consumers.empty() ||
      launches.front().first != producer->launch)
    return std::nullopt;

  std::optional<dataflow::ThreadWaitOp> producerWait =
      uniqueWait(producer->launch);
  if (!producerWait || (*producerWait)->getBlock() != block ||
      deallocation->getBlock() != block ||
      !allocation->isBeforeInBlock(producer->launch) ||
      !producer->launch->isBeforeInBlock(*producerWait))
    return std::nullopt;

  llvm::SmallVector<dataflow::ThreadWaitOp, 4> consumerWaits;
  mlir::Operation *previousWait = producerWait->getOperation();
  for (const EndpointPlan &consumer : consumers) {
    std::optional<dataflow::ThreadWaitOp> wait = uniqueWait(consumer.launch);
    if (!wait || (*wait)->getBlock() != block ||
        !previousWait->isBeforeInBlock(consumer.launch) ||
        !consumer.launch->isBeforeInBlock(*wait) ||
        !hasOnlyPureOperationsBetween(previousWait, consumer.launch))
      return std::nullopt;
    consumerWaits.push_back(*wait);
    previousWait = wait->getOperation();
  }
  if (!previousWait->isBeforeInBlock(deallocation))
    return std::nullopt;

  mlir::DominanceInfo dominance(block->getParentOp());
  for (const EndpointPlan &consumer : consumers)
    if (llvm::any_of(consumer.launch->getOperands(), [&](mlir::Value operand) {
          return !dominance.dominates(operand, producerWait->getOperation());
        }))
      return std::nullopt;
  if (!haveIndependentFanoutEffects(*producer,
                                    llvm::ArrayRef<EndpointPlan>(consumers)))
    return std::nullopt;
  return ChannelPlan{allocation,           deallocation,
                     std::move(*producer), std::move(consumers),
                     *producerWait,        std::move(consumerWaits)};
}

std::optional<SourceChannelPlan>
analyzeSourceChannel(mlir::LLVM::AllocaOp allocation) {
  std::optional<SourceAllocationShape> shape =
      analyzeSourceAllocationShape(allocation);
  if (!shape || allocation->getParentOfType<dataflow::ThreadOp>() ||
      allocation->getParentOfType<dataflow::GraphOp>() ||
      allocation->getParentOfType<loom::SpatialRegionOp>())
    return std::nullopt;

  mlir::LLVM::LifetimeStartOp lifetimeStart;
  mlir::LLVM::LifetimeEndOp lifetimeEnd;
  llvm::SmallVector<std::pair<dataflow::ThreadLaunchOp, unsigned>, 4> launches;
  for (mlir::OpOperand &use : allocation.getRes().getUses()) {
    if (auto start =
            llvm::dyn_cast<mlir::LLVM::LifetimeStartOp>(use.getOwner())) {
      if (lifetimeStart || start.getPtr() != allocation.getRes())
        return std::nullopt;
      lifetimeStart = start;
      continue;
    }
    if (auto end = llvm::dyn_cast<mlir::LLVM::LifetimeEndOp>(use.getOwner())) {
      if (lifetimeEnd || end.getPtr() != allocation.getRes())
        return std::nullopt;
      lifetimeEnd = end;
      continue;
    }
    auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(use.getOwner());
    std::optional<unsigned> ordinal =
        launch ? launchBodyOrdinal(use, launch) : std::nullopt;
    if (!ordinal)
      return std::nullopt;
    launches.emplace_back(launch, *ordinal);
  }
  if (static_cast<bool>(lifetimeStart) != static_cast<bool>(lifetimeEnd) ||
      launches.size() < 2)
    return std::nullopt;
  mlir::Block *block = allocation->getBlock();
  llvm::SmallPtrSet<mlir::Operation *, 8> uniqueLaunches;
  for (auto [launch, ordinal] : launches) {
    (void)ordinal;
    if (launch->getBlock() != block ||
        !uniqueLaunches.insert(launch.getOperation()).second)
      return std::nullopt;
  }
  llvm::sort(launches, [](auto lhs, auto rhs) {
    return lhs.first->isBeforeInBlock(rhs.first);
  });

  std::optional<SourceEndpointPlan> producer;
  llvm::SmallVector<SourceEndpointPlan, 4> consumers;
  for (auto [launch, ordinal] : launches) {
    auto endpoint = analyzeSourceEndpoint(launch, ordinal, *shape,
                                          allocation.getRes().getType());
    if (!endpoint || !launch.getAsyncDependencies().empty())
      return std::nullopt;
    if (endpoint->kind == EndpointKind::Producer) {
      if (producer)
        return std::nullopt;
      producer = std::move(*endpoint);
    } else {
      consumers.push_back(std::move(*endpoint));
    }
  }
  if (!producer || consumers.empty() ||
      launches.front().first != producer->launch)
    return std::nullopt;

  std::optional<dataflow::ThreadWaitOp> producerWait =
      uniqueWait(producer->launch);
  if (!producerWait || (*producerWait)->getBlock() != block ||
      (lifetimeStart && lifetimeStart->getBlock() != block) ||
      (lifetimeEnd && lifetimeEnd->getBlock() != block) ||
      !allocation->isBeforeInBlock(producer->launch) ||
      (lifetimeStart && (!allocation->isBeforeInBlock(lifetimeStart) ||
                         !lifetimeStart->isBeforeInBlock(producer->launch))) ||
      !producer->launch->isBeforeInBlock(*producerWait))
    return std::nullopt;

  for (const SourceEndpointPlan &consumer : consumers) {
    std::optional<dataflow::ThreadWaitOp> wait = uniqueWait(consumer.launch);
    if (!wait || (*wait)->getBlock() != block ||
        !producer->launch->isBeforeInBlock(consumer.launch) ||
        !consumer.launch->isBeforeInBlock(*wait) ||
        (lifetimeEnd && !(*wait)->isBeforeInBlock(lifetimeEnd)))
      return std::nullopt;
  }
  auto launchesToMove = sourceLaunchesToMove(
      *producer, llvm::ArrayRef<SourceEndpointPlan>(consumers), *producerWait);
  if (!launchesToMove)
    return std::nullopt;
  llvm::SmallVector<mlir::Operation *, 2> lifetimeMarkers;
  if (lifetimeStart)
    lifetimeMarkers.push_back(lifetimeStart);
  if (lifetimeEnd)
    lifetimeMarkers.push_back(lifetimeEnd);
  return SourceChannelPlan{allocation,
                           std::move(lifetimeMarkers),
                           *shape,
                           std::move(*producer),
                           std::move(consumers),
                           *producerWait,
                           std::move(*launchesToMove)};
}

bool canPromoteSourceOrderedBuffer(mlir::LLVM::AllocaOp allocation) {
  mlir::ModuleOp module = allocation->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return false;
  mlir::IRMapping mapping;
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      llvm::cast<mlir::ModuleOp>(module->clone(mapping)));
  auto clonedAllocation = llvm::dyn_cast_or_null<mlir::LLVM::AllocaOp>(
      mapping.lookupOrNull(allocation.getOperation()));
  if (!clonedAllocation)
    return false;
  if (llvm::Error error =
          liftTransparentOwnedThreadWrapperCalls(clonedAllocation)) {
    llvm::consumeError(std::move(error));
    return false;
  }
  return analyzeSourceChannel(clonedAllocation).has_value();
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
specializeIfShared(EndpointPlan plan, mlir::MemRefType allocationType,
                   mlir::Operation *&trackedSpatialRegion) {
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
  if (trackedSpatialRegion == plan.spatial.getOperation())
    trackedSpatialRegion = specialized->spatial.getOperation();
  return *specialized;
}

llvm::Expected<SourceEndpointPlan> specializeSourceIfShared(
    SourceEndpointPlan plan, const SourceAllocationShape &shape,
    mlir::Type pointerType, mlir::Operation *&trackedSpatialRegion) {
  mlir::ModuleOp module = plan.thread->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return invalid("selected source thread has no module owner");
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
  auto specialized = analyzeSourceEndpoint(plan.launch, plan.formalOrdinal,
                                           shape, pointerType);
  if (!specialized || specialized->kind != plan.kind)
    return invalid(
        "specialized source thread no longer has the selected endpoint");
  if (trackedSpatialRegion == plan.spatial.getOperation())
    trackedSpatialRegion = specialized->spatial.getOperation();
  return *specialized;
}

template <typename Plan> void setThreadFormalType(Plan &plan, mlir::Type type) {
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

llvm::Error rewriteSourceEndpoint(SourceEndpointPlan &plan,
                                  mlir::Type channelType) {
  setThreadFormalType(plan, channelType);
  mlir::BlockArgument formal =
      plan.thread.getBody().front().getArgument(plan.formalOrdinal);
  mlir::Block &entry = plan.spatial.getBody().front();
  mlir::BlockArgument oldPointerArgument = plan.pointerArgument;
  mlir::BlockArgument channelArgument;

  if (plan.kind == EndpointKind::Producer) {
    plan.spatial.getValueInputsMutable().erase(plan.valueOrdinal);
    plan.spatial.getStreamOutputsMutable().append(formal);
    channelArgument =
        entry.addArgument(channelType, oldPointerArgument.getLoc());
    for (const SourcePointerAccess &access : plan.accesses) {
      auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(access.event);
      if (!store)
        return invalid("source producer endpoint is no longer a store");
      mlir::OpBuilder builder(store);
      dataflow::ChannelSendOp::create(builder, store.getLoc(), channelArgument,
                                      store.getValue());
      store.erase();
    }
  } else {
    plan.spatial.getValueInputsMutable().erase(plan.valueOrdinal);
    const unsigned streamArgumentOrdinal =
        plan.spatial.getValueInputs().size() +
        plan.spatial.getStreamInputs().size() + 1;
    plan.spatial.getStreamInputsMutable().append(formal);
    channelArgument = entry.insertArgument(streamArgumentOrdinal, channelType,
                                           oldPointerArgument.getLoc());
    llvm::SmallVector<mlir::Attribute, 4> sourceMaps(
        plan.spatial.getSourceMaps().begin(),
        plan.spatial.getSourceMaps().end());
    sourceMaps.push_back(mlir::AffineMapAttr::get(
        mlir::AffineMap::get(0, 0, {}, plan.spatial.getContext())));
    plan.spatial.setSourceMapsAttr(
        mlir::ArrayAttr::get(plan.spatial.getContext(), sourceMaps));
    for (const SourcePointerAccess &access : plan.accesses) {
      auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(access.event);
      if (!load)
        return invalid("source consumer endpoint is no longer a load");
      mlir::OpBuilder builder(load);
      auto receive = dataflow::ChannelReceiveOp::create(builder, load.getLoc(),
                                                        channelArgument);
      load.getResult().replaceAllUsesWith(receive.getMessage());
      load.erase();
    }
  }
  for (SourcePointerAccess &access : llvm::reverse(plan.accesses))
    if (access.address) {
      if (!access.address.getRes().use_empty())
        return invalid("source endpoint address retained an unproved use");
      access.address.erase();
    }
  if (!oldPointerArgument.use_empty())
    return invalid("selected source pointer retained an unproved use");
  entry.eraseArgument(oldPointerArgument.getArgNumber());
  return llvm::Error::success();
}

} // namespace

bool areKnownDistinctMemoryRoots(mlir::Value lhs, mlir::Value rhs) {
  lhs = exactMemoryRoot(lhs);
  rhs = exactMemoryRoot(rhs);
  if (lhs == rhs)
    return false;
  if (areDistinctNoAliasFunctionArguments(lhs, rhs))
    return true;
  auto lhsGlobal = lhs.getDefiningOp<mlir::memref::GetGlobalOp>();
  auto rhsGlobal = rhs.getDefiningOp<mlir::memref::GetGlobalOp>();
  auto lhsAlloc = lhs.getDefiningOp<mlir::memref::AllocOp>();
  auto rhsAlloc = rhs.getDefiningOp<mlir::memref::AllocOp>();
  auto lhsAlloca = lhs.getDefiningOp<mlir::LLVM::AllocaOp>();
  auto rhsAlloca = rhs.getDefiningOp<mlir::LLVM::AllocaOp>();
  auto lhsAddress = lhs.getDefiningOp<mlir::LLVM::AddressOfOp>();
  auto rhsAddress = rhs.getDefiningOp<mlir::LLVM::AddressOfOp>();
  if (lhsAlloca && rhsAlloca)
    return true;
  if (lhsAlloca)
    return rhsAlloc || rhsGlobal || rhsAddress ||
           isEnclosingBlockArgument(rhs, lhsAlloca);
  if (rhsAlloca)
    return lhsAlloc || lhsGlobal || lhsAddress ||
           isEnclosingBlockArgument(lhs, rhsAlloca);
  if (lhsAddress && rhsAddress)
    return lhsAddress.getGlobalName() != rhsAddress.getGlobalName();
  if (lhsGlobal && rhsGlobal)
    return lhsGlobal.getName() != rhsGlobal.getName();
  if (lhsAlloc && rhsAlloc)
    return lhsAlloc != rhsAlloc;
  return (lhsGlobal && rhsAlloc) || (lhsAlloc && rhsGlobal);
}

bool canPromoteOrderedBufferToChannel(mlir::memref::AllocOp allocation) {
  return analyzeChannel(allocation).has_value();
}

bool canPromoteOrderedBufferToChannel(mlir::LLVM::AllocaOp allocation) {
  return canPromoteSourceOrderedBuffer(allocation);
}

llvm::Error
promoteOrderedBufferToChannel(mlir::memref::AllocOp allocation,
                              mlir::Operation *&trackedSpatialRegion) {
  std::optional<ChannelPlan> analyzed = analyzeChannel(allocation);
  if (!analyzed)
    return invalid("selected allocation is not an exact promotable ordered "
                   "producer-consumer buffer");
  ChannelPlan plan = *analyzed;
  mlir::MemRefType allocationType = allocation.getType();
  auto producer =
      specializeIfShared(plan.producer, allocationType, trackedSpatialRegion);
  if (!producer)
    return producer.takeError();
  plan.producer = *producer;
  for (EndpointPlan &consumer : plan.consumers) {
    auto specialized =
        specializeIfShared(consumer, allocationType, trackedSpatialRegion);
    if (!specialized)
      return specialized.takeError();
    consumer = std::move(*specialized);
  }

  mlir::Type channelType = dataflow::ChannelType::get(
      allocation.getContext(), allocationType.getElementType());
  mlir::OpBuilder builder(allocation);
  auto channel = dataflow::ChannelCreateOp::create(builder, allocation.getLoc(),
                                                   channelType);
  if (llvm::Error error = rewriteEndpoint(plan.producer, channelType))
    return error;
  for (EndpointPlan &consumer : plan.consumers)
    if (llvm::Error error = rewriteEndpoint(consumer, channelType))
      return error;
  plan.producer.launch->setOperand(plan.producer.formalOrdinal,
                                   channel.getChannel());
  for (EndpointPlan &consumer : plan.consumers) {
    consumer.launch->setOperand(consumer.formalOrdinal, channel.getChannel());
    consumer.launch->moveBefore(plan.producerWait);
  }
  plan.deallocation.erase();
  if (!allocation.getResult().use_empty())
    return invalid("promoted allocation retained an unproved use");
  allocation.erase();
  return llvm::Error::success();
}

llvm::Error
promoteOrderedBufferToChannel(mlir::LLVM::AllocaOp allocation,
                              mlir::Operation *&trackedSpatialRegion) {
  if (llvm::Error error = liftTransparentOwnedThreadWrapperCalls(allocation))
    return error;
  std::optional<SourceChannelPlan> analyzed = analyzeSourceChannel(allocation);
  if (!analyzed)
    return invalid("selected source allocation is not an exact promotable "
                   "ordered producer-consumer buffer");
  SourceChannelPlan plan = *analyzed;
  mlir::Type pointerType = allocation.getRes().getType();
  auto producer = specializeSourceIfShared(plan.producer, plan.shape,
                                           pointerType, trackedSpatialRegion);
  if (!producer)
    return producer.takeError();
  plan.producer = *producer;
  for (SourceEndpointPlan &consumer : plan.consumers) {
    auto specialized = specializeSourceIfShared(
        consumer, plan.shape, pointerType, trackedSpatialRegion);
    if (!specialized)
      return specialized.takeError();
    consumer = std::move(*specialized);
  }

  mlir::Type channelType = dataflow::ChannelType::get(allocation.getContext(),
                                                      plan.shape.elementType);
  mlir::OpBuilder builder(allocation);
  auto channel = dataflow::ChannelCreateOp::create(builder, allocation.getLoc(),
                                                   channelType);
  if (llvm::Error error = rewriteSourceEndpoint(plan.producer, channelType))
    return error;
  for (SourceEndpointPlan &consumer : plan.consumers)
    if (llvm::Error error = rewriteSourceEndpoint(consumer, channelType))
      return error;
  plan.producer.launch->setOperand(plan.producer.formalOrdinal,
                                   channel.getChannel());
  for (SourceEndpointPlan &consumer : plan.consumers)
    consumer.launch->setOperand(consumer.formalOrdinal, channel.getChannel());
  for (dataflow::ThreadLaunchOp launch : plan.launchesToMove)
    launch->moveBefore(plan.producerWait);
  for (mlir::Operation *marker : plan.lifetimeMarkers)
    marker->erase();
  if (!allocation.getRes().use_empty())
    return invalid("promoted source allocation retained an unproved use");
  allocation.erase();
  return llvm::Error::success();
}

} // namespace loom::frontend::detail

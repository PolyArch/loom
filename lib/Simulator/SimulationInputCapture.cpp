#include "Simulator/SimulationInputCapture.h"

#include "SimulationPointerCapture.h"
#include "SimulationWireInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <limits>
#include <optional>
#include <system_error>
#include <tuple>

namespace loom::sim {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("simulation_input_capture_invalid: ") + message);
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      llvm::Twine("simulation_input_capture_unsupported: ") + message);
}

mlir::FloatType floatingLaneType(mlir::Type elementType) {
  if (auto floating = llvm::dyn_cast<mlir::FloatType>(elementType))
    return floating;
  if (auto vector = llvm::dyn_cast<mlir::VectorType>(elementType))
    return llvm::dyn_cast<mlir::FloatType>(vector.getElementType());
  return {};
}

struct FloatingWriteProjection {
  mlir::FloatType laneType;
  bool sawWrite = false;
  bool conflict = false;
};

void recordFloatingWrite(mlir::Value memory,
                         FloatingWriteProjection &projection) {
  projection.sawWrite = true;
  auto memoryType = llvm::dyn_cast<mlir::MemRefType>(memory.getType());
  mlir::FloatType laneType = memoryType
                                 ? floatingLaneType(memoryType.getElementType())
                                 : mlir::FloatType{};
  if (!laneType || (projection.laneType && projection.laneType != laneType)) {
    projection.conflict = true;
    return;
  }
  projection.laneType = laneType;
}

void collectFloatingWrites(mlir::Value memory,
                           llvm::DenseSet<mlir::Value> &visited,
                           FloatingWriteProjection &projection) {
  if (!visited.insert(memory).second)
    return;
  for (mlir::OpOperand &use : memory.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (auto store = llvm::dyn_cast<dataflow::StoreOp>(owner)) {
      if (store.getMem() == memory)
        recordFloatingWrite(memory, projection);
      continue;
    }
    if (auto atomic = llvm::dyn_cast<dataflow::AtomicRmwOp>(owner)) {
      if (atomic.getMem() == memory)
        recordFloatingWrite(memory, projection);
      continue;
    }
    if (auto cmpxchg = llvm::dyn_cast<dataflow::CmpXchgOp>(owner)) {
      if (cmpxchg.getMem() == memory)
        recordFloatingWrite(memory, projection);
      continue;
    }
    if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(owner)) {
      if (cast.getSource() == memory)
        collectFloatingWrites(cast.getResult(), visited, projection);
      continue;
    }
  }
}

mlir::FloatType
uniformFloatingWriteLaneType(detail::ResolvedLaunchContext &context,
                             dataflow::LogicalMemoryRootRef root) {
  FloatingWriteProjection projection;
  llvm::DenseSet<mlir::Value> visited;
  const unsigned firstMemory = 1 +
                               static_cast<unsigned>(context.numValueInputs) +
                               static_cast<unsigned>(context.numStreamInputs);
  mlir::Block &entry = context.graphOp.getBody().front();
  for (auto [ordinal, candidate] : llvm::enumerate(context.memoryInputRoots)) {
    if (!candidate || *candidate != root)
      continue;
    collectFloatingWrites(entry.getArgument(firstMemory + ordinal), visited,
                          projection);
  }
  if (!projection.sawWrite || projection.conflict)
    return {};
  return projection.laneType;
}

void collectInitialStateReads(mlir::Value memory,
                              llvm::DenseSet<mlir::Value> &visited,
                              bool &requiresInitialState) {
  if (requiresInitialState || !visited.insert(memory).second)
    return;
  for (mlir::OpOperand &use : memory.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (auto load = llvm::dyn_cast<dataflow::LoadOp>(owner)) {
      if (load.getMem() == memory)
        requiresInitialState = true;
      continue;
    }
    if (auto atomic = llvm::dyn_cast<dataflow::AtomicRmwOp>(owner)) {
      if (atomic.getMem() == memory)
        requiresInitialState = true;
      continue;
    }
    if (auto cmpxchg = llvm::dyn_cast<dataflow::CmpXchgOp>(owner)) {
      if (cmpxchg.getMem() == memory)
        requiresInitialState = true;
      continue;
    }
    if (auto store = llvm::dyn_cast<dataflow::StoreOp>(owner)) {
      if (store.getMem() == memory)
        continue;
    }
    if (auto cast = llvm::dyn_cast<mlir::memref::CastOp>(owner)) {
      if (cast.getSource() == memory) {
        collectInitialStateReads(cast.getResult(), visited,
                                 requiresInitialState);
        continue;
      }
    }
    // Unknown capability consumers may observe pre-activation state. The
    // source-backed projection remains conservative rather than inventing a
    // write-only proof.
    requiresInitialState = true;
  }
}

bool memoryRequiresInitialState(detail::ResolvedLaunchContext &context,
                                dataflow::LogicalMemoryRootRef root) {
  bool requiresInitialState = false;
  llvm::DenseSet<mlir::Value> visited;
  const unsigned firstMemory = 1 +
                               static_cast<unsigned>(context.numValueInputs) +
                               static_cast<unsigned>(context.numStreamInputs);
  mlir::Block &entry = context.graphOp.getBody().front();
  for (auto [ordinal, candidate] : llvm::enumerate(context.memoryInputRoots)) {
    if (!candidate || *candidate != root)
      continue;
    collectInitialStateReads(entry.getArgument(firstMemory + ordinal), visited,
                             requiresInitialState);
  }
  return requiresInitialState;
}

std::optional<std::uint64_t> constantUnsigned(mlir::Value value) {
  mlir::Attribute attribute;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    attribute = constant.getValue();
  else if (auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>())
    attribute = constant.getValue();
  auto integer = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attribute);
  if (!integer || integer.getValue().isNegative() ||
      integer.getValue().getActiveBits() > 64)
    return std::nullopt;
  return integer.getValue().getZExtValue();
}

std::optional<std::int64_t> constantSigned(mlir::Value value) {
  mlir::Attribute attribute;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    attribute = constant.getValue();
  else if (auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>())
    attribute = constant.getValue();
  auto integer = llvm::dyn_cast_if_present<mlir::IntegerAttr>(attribute);
  if (!integer || !integer.getValue().isSignedIntN(64))
    return std::nullopt;
  return integer.getValue().getSExtValue();
}

llvm::Expected<std::uint64_t> fixedTypeByteCount(mlir::Operation *scope,
                                                 mlir::Type type) {
  llvm::TypeSize bytes = mlir::DataLayout::closest(scope).getTypeSize(type);
  if (bytes.isScalable() || bytes.getFixedValue() == 0)
    return unsupported("LLVM type has no fixed nonzero byte size");
  return bytes.getFixedValue();
}

bool fitsStorageExtent(const detail::LaneShape &shape,
                       std::uint64_t storageBytes) {
  if (shape.lanesPerToken == 0 || shape.laneBitWidth == 0 ||
      shape.lanesPerToken >
          std::numeric_limits<std::uint64_t>::max() / shape.laneBitWidth)
    return false;
  const std::uint64_t semanticBits = shape.lanesPerToken * shape.laneBitWidth;
  const std::uint64_t requiredBytes =
      semanticBits / 8 + static_cast<std::uint64_t>(semanticBits % 8 != 0);
  return requiredBytes <= storageBytes;
}

llvm::Expected<std::uint64_t>
fixedAllocationByteCount(mlir::LLVM::AllocaOp allocation) {
  std::optional<std::uint64_t> count =
      constantUnsigned(allocation.getArraySize());
  if (!count || *count == 0)
    return unsupported("LLVM alloca has no positive constant element count");

  llvm::Expected<std::uint64_t> elementBytes =
      fixedTypeByteCount(allocation.getOperation(), allocation.getElemType());
  if (!elementBytes)
    return elementBytes.takeError();
  if (*count > std::numeric_limits<std::uint64_t>::max() / *elementBytes)
    return unsupported("LLVM alloca byte count overflows uint64");
  return *elementBytes * *count;
}

struct ResolvedObject {
  mlir::Value base;
  mlir::Operation *owner = nullptr;
  std::uint64_t byteCount = 0;
  std::uint64_t byteOffset = 0;
};

struct ResolvedDirectCallObject {
  ResolvedObject object;
  DirectCallMemorySource source;
  mlir::Value pointer;
};

llvm::Expected<std::uint64_t> constantGepByteOffset(mlir::LLVM::GEPOp gep) {
  mlir::Type indexedType = gep.getElemType();
  std::uint64_t offset = 0;
  unsigned dynamicIndex = 0;
  for (auto [position, raw] : llvm::enumerate(gep.getRawConstantIndices())) {
    std::optional<std::int64_t> index;
    if (raw == mlir::LLVM::GEPOp::kDynamicIndex) {
      if (dynamicIndex >= gep.getDynamicIndices().size())
        return invalid("LLVM GEP dynamic-index table is malformed");
      index = constantSigned(gep.getDynamicIndices()[dynamicIndex++]);
    } else {
      index = raw;
    }
    if (!index)
      return unsupported("LLVM GEP index is not constant");
    if (*index < 0)
      return unsupported(llvm::Twine("LLVM GEP has negative constant index ") +
                         llvm::Twine(*index));

    mlir::Type strideType = indexedType;
    if (position != 0) {
      auto array = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(indexedType);
      if (!array)
        return unsupported("LLVM GEP aggregate indexing is not an array");
      strideType = array.getElementType();
      indexedType = strideType;
    }
    llvm::Expected<std::uint64_t> stride =
        fixedTypeByteCount(gep.getOperation(), strideType);
    if (!stride)
      return stride.takeError();
    const std::uint64_t unsignedIndex = static_cast<std::uint64_t>(*index);
    if (unsignedIndex > std::numeric_limits<std::uint64_t>::max() / *stride)
      return unsupported("LLVM GEP byte offset overflows uint64");
    const std::uint64_t increment = unsignedIndex * *stride;
    if (offset > std::numeric_limits<std::uint64_t>::max() - increment)
      return unsupported("LLVM GEP byte offset overflows uint64");
    offset += increment;
  }
  if (dynamicIndex != gep.getDynamicIndices().size())
    return invalid("LLVM GEP has unused dynamic indices");
  return offset;
}

llvm::Expected<ResolvedObject> resolveObject(mlir::Value pointer) {
  if (auto gep = pointer.getDefiningOp<mlir::LLVM::GEPOp>()) {
    llvm::Expected<ResolvedObject> base = resolveObject(gep.getBase());
    if (!base)
      return base.takeError();
    llvm::Expected<std::uint64_t> offset = constantGepByteOffset(gep);
    if (!offset)
      return offset.takeError();
    if (base->byteOffset > std::numeric_limits<std::uint64_t>::max() - *offset)
      return unsupported("LLVM GEP chain byte offset overflows uint64");
    base->byteOffset += *offset;
    if (base->byteOffset >= base->byteCount)
      return invalid("LLVM GEP points outside its finite allocation");
    return *base;
  }
  if (auto cast = pointer.getDefiningOp<mlir::LLVM::BitcastOp>())
    return resolveObject(cast.getArg());
  if (auto cast = pointer.getDefiningOp<mlir::LLVM::AddrSpaceCastOp>())
    return resolveObject(cast.getArg());
  if (auto cast = pointer.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1)
      return unsupported("finite object has a non-unary pointer cast");
    return resolveObject(cast.getInputs().front());
  }
  if (auto allocation = pointer.getDefiningOp<mlir::LLVM::AllocaOp>()) {
    llvm::Expected<std::uint64_t> byteCount =
        fixedAllocationByteCount(allocation);
    if (!byteCount)
      return byteCount.takeError();
    return ResolvedObject{allocation.getResult(), allocation.getOperation(),
                          *byteCount, 0};
  }
  if (auto address = pointer.getDefiningOp<mlir::LLVM::AddressOfOp>()) {
    auto global =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::GlobalOp>(
            address, address.getGlobalNameAttr());
    if (!global)
      return unsupported(
          "LLVM address does not resolve to a finite global object");
    llvm::Expected<std::uint64_t> byteCount =
        fixedTypeByteCount(global.getOperation(), global.getGlobalType());
    if (!byteCount)
      return byteCount.takeError();
    return ResolvedObject{address.getResult(), global.getOperation(),
                          *byteCount, 0};
  }
  return unsupported("call operand does not resolve to a finite LLVM object");
}

llvm::Expected<ResolvedObject> resolveObjectThroughCallPath(
    mlir::Value pointer, mlir::LLVM::LLVMFuncOp enclosingCallable,
    llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath, std::size_t pathIndex) {
  if (auto gep = pointer.getDefiningOp<mlir::LLVM::GEPOp>()) {
    llvm::Expected<ResolvedObject> base = resolveObjectThroughCallPath(
        gep.getBase(), enclosingCallable, invocationPath, pathIndex);
    if (!base)
      return base.takeError();
    llvm::Expected<std::uint64_t> offset = constantGepByteOffset(gep);
    if (!offset)
      return offset.takeError();
    if (base->byteOffset > std::numeric_limits<std::uint64_t>::max() - *offset)
      return unsupported("LLVM GEP chain byte offset overflows uint64");
    base->byteOffset += *offset;
    if (base->byteOffset >= base->byteCount)
      return invalid("LLVM GEP points outside its finite allocation");
    return *base;
  }
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(pointer)) {
    if (argument.getOwner() != &enclosingCallable.getBody().front()) {
      mlir::Operation *owner = argument.getOwner()->getParentOp();
      auto branch =
          llvm::dyn_cast_or_null<mlir::RegionBranchOpInterface>(owner);
      return unsupported(
          llvm::Twine("memory boundary block argument ") +
          llvm::Twine(argument.getArgNumber()) + " is owned by '" +
          (owner ? owner->getName().getStringRef()
                 : llvm::StringRef("<unknown>")) +
          "', not callable '" + enclosingCallable.getSymName() +
          "'; region-branch interface=" + (branch ? "present" : "absent"));
    }
    if (pathIndex >= invocationPath.size())
      return invalid("operation invocation path is incomplete");
    mlir::LLVM::CallOp hostCall = invocationPath[pathIndex];
    if (!hostCall.getCalleeAttr() ||
        hostCall.getCalleeAttr().getValue() != enclosingCallable.getSymName())
      return invalid("operation invocation path has a broken callee edge");
    const unsigned ordinal = argument.getArgNumber();
    if (ordinal >= hostCall.getCalleeOperands().size())
      return invalid("callable argument exceeds host call operands");
    mlir::Value callerValue = hostCall.getCalleeOperands()[ordinal];
    if (pathIndex == 0)
      return resolveObject(callerValue);
    auto caller = hostCall->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (!caller)
      return invalid("operation invocation path call has no caller");
    return resolveObjectThroughCallPath(callerValue, caller, invocationPath,
                                        pathIndex - 1);
  }
  return resolveObject(pointer);
}

struct RegionValueSource {
  mlir::RegionBranchOpInterface branch;
  mlir::RegionSuccessor successor;
  unsigned inputIndex = 0;
};

std::optional<RegionValueSource> getRegionValueSource(mlir::Value value) {
  if (auto result = llvm::dyn_cast<mlir::OpResult>(value)) {
    auto branch =
        llvm::dyn_cast<mlir::RegionBranchOpInterface>(result.getOwner());
    if (!branch)
      return std::nullopt;
    mlir::RegionSuccessor successor(result.getOwner());
    mlir::ValueRange inputs = branch.getSuccessorInputs(successor);
    auto position = llvm::find(inputs, value);
    if (position == inputs.end())
      return std::nullopt;
    return RegionValueSource{
        branch, successor,
        static_cast<unsigned>(std::distance(inputs.begin(), position))};
  }

  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument)
    return std::nullopt;
  auto branch = llvm::dyn_cast_or_null<mlir::RegionBranchOpInterface>(
      argument.getOwner()->getParentOp());
  if (!branch)
    return std::nullopt;
  mlir::RegionSuccessor successor(argument.getOwner()->getParent());
  mlir::ValueRange inputs = branch.getSuccessorInputs(successor);
  auto position = llvm::find(inputs, value);
  if (position == inputs.end())
    return std::nullopt;
  return RegionValueSource{
      branch, successor,
      static_cast<unsigned>(std::distance(inputs.begin(), position))};
}

llvm::Expected<ResolvedObject>
resolveObjectAtInvocation(mlir::Value pointer, mlir::LLVM::LLVMFuncOp callable,
                          llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  if (!callable || invocationPath.empty())
    return resolveObject(pointer);

  for (std::size_t pathIndex = 0; pathIndex < invocationPath.size();
       ++pathIndex) {
    mlir::LLVM::CallOp call = invocationPath[pathIndex];
    if (!call.getCalleeAttr())
      return invalid("operation invocation path contains an indirect call");
    auto callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
            call, call.getCalleeAttr());
    if (callee == callable)
      return resolveObjectThroughCallPath(pointer, callable, invocationPath,
                                          pathIndex);
  }

  auto rootCaller =
      invocationPath.front()->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (rootCaller == callable)
    return resolveObject(pointer);
  return invalid("pointer value is outside the exact invocation path");
}

struct InvocationSegment {
  mlir::LLVM::LLVMFuncOp callable;
  mlir::Operation *anchor = nullptr;
};

llvm::Expected<llvm::SmallVector<InvocationSegment, 4>>
invocationSegmentsTo(mlir::LLVM::LoadOp load,
                     llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  auto loadCallable = load->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!loadCallable)
    return invalid("pointer load is not enclosed by an LLVM callable");
  if (invocationPath.empty())
    return llvm::SmallVector<InvocationSegment, 4>{
        InvocationSegment{loadCallable, load.getOperation()}};

  llvm::SmallVector<InvocationSegment, 4> segments;
  auto rootCaller =
      invocationPath.front()->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!rootCaller)
    return invalid("operation invocation path has no root caller");
  if (rootCaller == loadCallable)
    return llvm::SmallVector<InvocationSegment, 4>{
        InvocationSegment{rootCaller, load.getOperation()}};

  segments.push_back(InvocationSegment{
      rootCaller, mlir::LLVM::CallOp(invocationPath.front()).getOperation()});
  for (std::size_t index = 0; index < invocationPath.size(); ++index) {
    mlir::LLVM::CallOp call = invocationPath[index];
    if (!call.getCalleeAttr())
      return invalid("operation invocation path contains an indirect call");
    auto callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
            call, call.getCalleeAttr());
    if (!callee)
      return invalid("operation invocation path callee does not resolve");
    if (callee == loadCallable) {
      segments.push_back(InvocationSegment{callee, load.getOperation()});
      return segments;
    }
    if (index + 1 == invocationPath.size())
      break;
    mlir::Operation *anchor =
        mlir::LLVM::CallOp(invocationPath[index + 1]).getOperation();
    segments.push_back(InvocationSegment{callee, anchor});
  }
  return invalid("operation invocation path does not reach pointer load");
}

llvm::Expected<std::optional<mlir::Value>> findSegmentReachingPointerStore(
    const ResolvedObject &slot, InvocationSegment segment,
    llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  mlir::DominanceInfo dominance(segment.callable);
  mlir::LLVM::StoreOp winner;
  bool ambiguous = false;
  llvm::Error resolutionError = llvm::Error::success();
  segment.callable.walk([&](mlir::LLVM::StoreOp store) {
    if (ambiguous || resolutionError)
      return;
    llvm::Expected<ResolvedObject> target = resolveObjectAtInvocation(
        store.getAddr(), segment.callable, invocationPath);
    if (!target) {
      llvm::Error error = target.takeError();
      std::error_code code = llvm::errorToErrorCode(std::move(error));
      if (code == std::make_error_code(std::errc::not_supported))
        return;
      resolutionError = invalid(
          "malformed store address reached pointer descriptor analysis");
      return;
    }
    if (target->owner != slot.owner || target->byteOffset != slot.byteOffset)
      return;
    if (dominance.properlyDominates(segment.anchor, store.getOperation()))
      return;
    if (!dominance.properlyDominates(store.getOperation(), segment.anchor)) {
      ambiguous = true;
      return;
    }
    if (!winner) {
      winner = store;
      return;
    }
    if (dominance.properlyDominates(winner.getOperation(),
                                    store.getOperation())) {
      winner = store;
      return;
    }
    if (!dominance.properlyDominates(store.getOperation(),
                                     winner.getOperation()))
      ambiguous = true;
  });
  if (resolutionError)
    return std::move(resolutionError);
  if (ambiguous)
    return unsupported(
        "pointer descriptor slot has an ambiguous reaching store");
  if (!winner)
    return std::optional<mlir::Value>{};
  if (!llvm::isa<mlir::LLVM::LLVMPointerType>(winner.getValue().getType()))
    return unsupported("pointer descriptor slot is written by a non-pointer");
  return std::optional<mlir::Value>(winner.getValue());
}

llvm::Expected<InvocationSegment>
segmentFor(mlir::Operation *operation,
           llvm::ArrayRef<InvocationSegment> segments) {
  auto callable = operation->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  for (InvocationSegment segment : segments)
    if (segment.callable == callable)
      return segment;
  return invalid("descriptor use is outside the exact invocation path");
}

llvm::Expected<std::uint64_t> directCallOrdinal(mlir::LLVM::LLVMFuncOp caller,
                                                mlir::LLVM::CallOp target,
                                                llvm::StringRef callee);

llvm::Expected<bool>
isExactInvocationCall(mlir::LLVM::CallOp call,
                      llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  if (!call.getCalleeAttr())
    return false;
  auto caller = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!caller)
    return invalid("invocation call has no enclosing callable");
  llvm::Expected<std::uint64_t> ordinal =
      directCallOrdinal(caller, call, call.getCalleeAttr().getValue());
  if (!ordinal)
    return ordinal.takeError();
  for (mlir::LLVM::CallOp candidate : invocationPath) {
    if (!candidate.getCalleeAttr() ||
        candidate.getCalleeAttr().getValue() != call.getCalleeAttr().getValue())
      continue;
    auto candidateCaller = candidate->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (!candidateCaller || candidateCaller.getSymName() != caller.getSymName())
      continue;
    llvm::Expected<std::uint64_t> candidateOrdinal = directCallOrdinal(
        candidateCaller, candidate, candidate.getCalleeAttr().getValue());
    if (!candidateOrdinal)
      return candidateOrdinal.takeError();
    if (*candidateOrdinal == *ordinal)
      return true;
  }
  return false;
}

bool appendRegionBranchSuccessorInputs(
    mlir::OpOperand &use, llvm::SmallVectorImpl<mlir::Value> &worklist) {
  mlir::Operation *owner = use.getOwner();
  auto branch = llvm::dyn_cast<mlir::RegionBranchOpInterface>(owner);
  if (!branch && llvm::isa<mlir::RegionBranchTerminatorOpInterface>(owner))
    branch = llvm::dyn_cast_or_null<mlir::RegionBranchOpInterface>(
        owner->getParentOp());
  if (!branch)
    return false;

  mlir::RegionBranchSuccessorMapping mapping;
  branch.getSuccessorOperandInputMapping(mapping);
  auto forwarded = mapping.find(&use);
  if (forwarded == mapping.end())
    return false;
  worklist.append(forwarded->second.begin(), forwarded->second.end());
  return true;
}

llvm::Error validateDescriptorUseClosure(
    const ResolvedObject &slot, llvm::ArrayRef<InvocationSegment> segments,
    llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  auto allocation = llvm::dyn_cast<mlir::LLVM::AllocaOp>(slot.owner);
  if (!allocation)
    return unsupported(
        "pointer descriptor slot is not owned by a finite stack allocation");

  llvm::SmallVector<mlir::Value, 8> worklist{allocation.getResult()};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *owner = use.getOwner();
      llvm::Expected<InvocationSegment> segment = segmentFor(owner, segments);
      if (!segment)
        return segment.takeError();
      if (appendRegionBranchSuccessorInputs(use, worklist))
        continue;
      mlir::DominanceInfo dominance(segment->callable);
      if (dominance.properlyDominates(segment->anchor, owner))
        continue;
      if (owner != segment->anchor &&
          !dominance.properlyDominates(owner, segment->anchor))
        return unsupported(
            "pointer descriptor has a control-dependent use before capture");

      if (auto gep = llvm::dyn_cast<mlir::LLVM::GEPOp>(owner)) {
        if (gep.getBase() != value)
          return invalid("descriptor pointer is not the LLVM GEP base");
        llvm::Expected<std::uint64_t> offset = constantGepByteOffset(gep);
        if (!offset)
          return offset.takeError();
        worklist.push_back(gep.getResult());
        continue;
      }
      if (auto cast = llvm::dyn_cast<mlir::LLVM::BitcastOp>(owner)) {
        if (cast.getArg() != value)
          return invalid("descriptor pointer is not the LLVM bitcast input");
        worklist.push_back(cast.getResult());
        continue;
      }
      if (auto cast = llvm::dyn_cast<mlir::LLVM::AddrSpaceCastOp>(owner)) {
        if (cast.getArg() != value)
          return invalid(
              "descriptor pointer is not the address-space cast input");
        worklist.push_back(cast.getResult());
        continue;
      }
      if (auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(owner)) {
        if (cast.getInputs().size() != 1 || cast.getResults().size() != 1 ||
            cast.getInputs().front() != value)
          return unsupported(
              "descriptor pointer has a non-unary conversion cast");
        worklist.push_back(cast.getResults().front());
        continue;
      }
      if (auto store = llvm::dyn_cast<mlir::LLVM::StoreOp>(owner)) {
        if (store.getValue() == value)
          return unsupported("pointer descriptor allocation escapes by store");
        if (store.getAddr() != value)
          return invalid("descriptor pointer has a malformed store use");
        continue;
      }
      if (auto load = llvm::dyn_cast<mlir::LLVM::LoadOp>(owner)) {
        if (load.getAddr() != value)
          return invalid("descriptor pointer has a malformed load use");
        continue;
      }
      if (llvm::isa<mlir::LLVM::LifetimeStartOp, mlir::LLVM::LifetimeEndOp>(
              owner))
        continue;
      if (auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(owner)) {
        llvm::Expected<bool> exact =
            isExactInvocationCall(call, invocationPath);
        if (!exact)
          return exact.takeError();
        if (!*exact) {
          llvm::StringRef callee = call.getCalleeAttr()
                                       ? call.getCalleeAttr().getValue()
                                       : llvm::StringRef("<indirect>");
          auto caller = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
          return unsupported(
              llvm::Twine("pointer descriptor escapes through non-selected '") +
              (caller ? caller.getSymName() : llvm::StringRef("<unknown>")) +
              " -> " + callee + "'");
        }
        if (!call.getCalleeAttr())
          return invalid("exact invocation call has no direct callee");
        auto callee =
            mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
                call, call.getCalleeAttr());
        if (!callee || callee.getBody().empty())
          return invalid("exact invocation callee has no body");
        bool forwarded = false;
        for (auto [ordinal, operand] :
             llvm::enumerate(call.getCalleeOperands())) {
          if (operand != value)
            continue;
          if (ordinal >= callee.getBody().front().getNumArguments())
            return invalid("call operand exceeds callee arguments");
          worklist.push_back(callee.getBody().front().getArgument(ordinal));
          forwarded = true;
        }
        if (!forwarded)
          return invalid("descriptor pointer is not a callee operand");
        continue;
      }
      return unsupported(
          llvm::Twine("pointer descriptor has an unsupported use by '") +
          owner->getName().getStringRef() + "'");
    }
  }
  return llvm::Error::success();
}

struct ResolvedOperationObject {
  ResolvedObject object;
  mlir::Value captureBase;
};

llvm::Expected<ResolvedOperationObject>
resolveOperationObject(mlir::Value pointer,
                       llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath);

llvm::Expected<mlir::Value>
reachingStoredPointer(mlir::LLVM::LoadOp load,
                      llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  auto loadCallable = load->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!loadCallable)
    return invalid("pointer load is not enclosed by an LLVM callable");
  llvm::Expected<ResolvedOperationObject> resolvedSlot =
      resolveOperationObject(load.getAddr(), invocationPath);
  if (!resolvedSlot)
    return resolvedSlot.takeError();
  ResolvedObject &slot = resolvedSlot->object;
  llvm::Expected<llvm::SmallVector<InvocationSegment, 4>> segments =
      invocationSegmentsTo(load, invocationPath);
  if (!segments)
    return segments.takeError();
  if (llvm::Error error =
          validateDescriptorUseClosure(slot, *segments, invocationPath))
    return std::move(error);

  mlir::Value reachingValue;
  for (const InvocationSegment &segment : *segments) {
    llvm::Expected<std::optional<mlir::Value>> local =
        findSegmentReachingPointerStore(slot, segment, invocationPath);
    if (!local)
      return local.takeError();
    if (*local)
      reachingValue = **local;
  }
  if (!reachingValue)
    return unsupported("pointer descriptor load has no unique reaching store");
  return reachingValue;
}

struct OperationPointerOrigin {
  mlir::Value staticRoot;
  mlir::Value captureBase;
  std::uint64_t staticByteOffset = 0;
};

using OperationPointerVisit =
    std::tuple<mlir::Value, mlir::Value, mlir::Operation *>;

mlir::LLVM::LLVMFuncOp enclosingLlvmCallable(mlir::Value value) {
  mlir::Operation *scope = value.getDefiningOp();
  if (!scope) {
    auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
    if (!argument)
      return {};
    scope = argument.getOwner()->getParentOp();
  }
  if (auto callable = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(scope))
    return callable;
  return scope->getParentOfType<mlir::LLVM::LLVMFuncOp>();
}

mlir::LLVM::CallOp returnProjectionBinding(
    mlir::LLVM::LLVMFuncOp callable,
    llvm::ArrayRef<mlir::LLVM::CallOp> returnProjectionCalls) {
  for (mlir::LLVM::CallOp call : llvm::reverse(returnProjectionCalls)) {
    if (!call.getCalleeAttr())
      continue;
    auto callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
            call, call.getCalleeAttr());
    if (callee == callable)
      return call;
  }
  return {};
}

llvm::Expected<std::optional<std::uint64_t>>
baseBindingCallOrdinal(mlir::Value base, mlir::Value boundaryPointer,
                       llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  mlir::LLVM::LLVMFuncOp baseCallable = enclosingLlvmCallable(base);
  mlir::LLVM::LLVMFuncOp selectedCallable =
      enclosingLlvmCallable(boundaryPointer);
  if (!baseCallable || !selectedCallable)
    return invalid("operation memory base is outside an LLVM callable");
  if (baseCallable == selectedCallable)
    return std::optional<std::uint64_t>{};

  std::optional<std::uint64_t> result;
  for (std::uint64_t ordinal = 0; ordinal < invocationPath.size(); ++ordinal) {
    mlir::LLVM::CallOp call = invocationPath[ordinal];
    auto caller = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
    if (caller != baseCallable)
      continue;
    if (result)
      return invalid("operation memory base has multiple invocation bindings");
    mlir::DominanceInfo dominance(baseCallable);
    if (!dominance.dominates(base, call.getOperation()))
      return unsupported(
          "operation memory base does not dominate its invocation edge");
    result = ordinal;
  }
  if (!result)
    return invalid(
        "operation memory base is outside the exact invocation path");
  return result;
}

llvm::Error collectOperationPointerRoots(
    mlir::Value pointer,
    llvm::DenseMap<OperationPointerVisit, std::uint64_t> &visited,
    llvm::SmallVectorImpl<OperationPointerOrigin> &roots,
    llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath,
    llvm::ArrayRef<mlir::LLVM::CallOp> returnProjectionCalls = {},
    mlir::Value captureBase = {}, std::uint64_t staticByteOffset = 0) {
  if (!pointer)
    return invalid("operation memory boundary contains an absent pointer");
  mlir::LLVM::CallOp returnBinding = returnProjectionBinding(
      enclosingLlvmCallable(pointer), returnProjectionCalls);
  OperationPointerVisit visit{pointer, captureBase,
                              returnBinding ? returnBinding.getOperation()
                                            : nullptr};
  auto [position, inserted] = visited.try_emplace(visit, staticByteOffset);
  if (!inserted && position->second != staticByteOffset)
    return unsupported(
        "operation memory boundary has inconsistent descriptor offsets");
  if (!inserted)
    return llvm::Error::success();

  auto collect = [&](mlir::Value source) {
    return collectOperationPointerRoots(source, visited, roots, invocationPath,
                                        returnProjectionCalls, captureBase,
                                        staticByteOffset);
  };
  if (auto gep = pointer.getDefiningOp<mlir::LLVM::GEPOp>()) {
    if (!captureBase)
      return collect(gep.getBase());
    llvm::Expected<std::uint64_t> offset = constantGepByteOffset(gep);
    if (!offset)
      return offset.takeError();
    if (staticByteOffset > std::numeric_limits<std::uint64_t>::max() - *offset)
      return invalid("descriptor pointer byte offset overflows uint64");
    return collectOperationPointerRoots(
        gep.getBase(), visited, roots, invocationPath, returnProjectionCalls,
        captureBase, staticByteOffset + *offset);
  }
  if (auto cast = pointer.getDefiningOp<mlir::LLVM::BitcastOp>())
    return collect(cast.getArg());
  if (auto cast = pointer.getDefiningOp<mlir::LLVM::AddrSpaceCastOp>())
    return collect(cast.getArg());
  if (auto cast = pointer.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1)
      return unsupported(
          "operation memory boundary has a non-unary pointer cast");
    return collect(cast.getInputs().front());
  }
  if (auto select = pointer.getDefiningOp<mlir::arith::SelectOp>()) {
    if (llvm::Error error = collectOperationPointerRoots(
            select.getTrueValue(), visited, roots, invocationPath,
            returnProjectionCalls, captureBase, staticByteOffset))
      return error;
    return collectOperationPointerRoots(select.getFalseValue(), visited, roots,
                                        invocationPath, returnProjectionCalls,
                                        captureBase, staticByteOffset);
  }
  if (auto select = pointer.getDefiningOp<mlir::LLVM::SelectOp>()) {
    if (llvm::Error error = collectOperationPointerRoots(
            select.getTrueValue(), visited, roots, invocationPath,
            returnProjectionCalls, captureBase, staticByteOffset))
      return error;
    return collectOperationPointerRoots(select.getFalseValue(), visited, roots,
                                        invocationPath, returnProjectionCalls,
                                        captureBase, staticByteOffset);
  }
  if (auto load = pointer.getDefiningOp<mlir::LLVM::LoadOp>()) {
    if (!llvm::isa<mlir::LLVM::LLVMPointerType>(load.getResult().getType()))
      return invalid("non-pointer LLVM load reached pointer-origin analysis");
    llvm::Expected<mlir::Value> stored =
        reachingStoredPointer(load, invocationPath);
    if (!stored)
      return stored.takeError();
    return collectOperationPointerRoots(
        *stored, visited, roots, invocationPath, returnProjectionCalls,
        captureBase ? captureBase : pointer, staticByteOffset);
  }

  if (auto call = pointer.getDefiningOp<mlir::LLVM::CallOp>()) {
    if (!call.getCalleeAttr())
      return unsupported(
          "operation memory boundary depends on an indirect pointer return");
    auto callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
            call, call.getCalleeAttr());
    if (!callee || callee.getBody().empty())
      return unsupported(
          llvm::Twine("operation memory boundary depends on external pointer "
                      "return from '") +
          call.getCalleeAttr().getValue() + "'");
    auto result = llvm::dyn_cast<mlir::OpResult>(pointer);
    if (!result)
      return invalid("LLVM call pointer is not an operation result");

    llvm::SmallVector<mlir::LLVM::CallOp, 4> projectedCalls(
        returnProjectionCalls);
    projectedCalls.push_back(call);
    bool sawReturn = false;
    for (mlir::Block &block : callee.getBody()) {
      auto returned =
          llvm::dyn_cast<mlir::LLVM::ReturnOp>(block.getTerminator());
      if (!returned)
        continue;
      sawReturn = true;
      if (result.getResultNumber() >= returned.getNumOperands())
        return invalid("LLVM callee return arity differs from its call result");
      if (llvm::Error error = collectOperationPointerRoots(
              returned.getOperand(result.getResultNumber()), visited, roots,
              invocationPath, projectedCalls, captureBase, staticByteOffset))
        return error;
    }
    if (!sawReturn)
      return invalid("pointer-returning LLVM callee has no return path");
    return llvm::Error::success();
  }

  if (std::optional<RegionValueSource> source = getRegionValueSource(pointer)) {
    llvm::SmallVector<mlir::RegionBranchPoint, 4> predecessors;
    llvm::SmallVector<mlir::Value, 4> values;
    source->branch.getPredecessors(source->successor, predecessors);
    source->branch.getPredecessorValues(source->successor, source->inputIndex,
                                        values);
    if (predecessors.size() != values.size())
      return invalid("region pointer predecessor table is malformed");
    for (mlir::Value predecessor : values)
      if (llvm::Error error = collectOperationPointerRoots(
              predecessor, visited, roots, invocationPath,
              returnProjectionCalls, captureBase, staticByteOffset))
        return error;
    return llvm::Error::success();
  }

  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(pointer)) {
    auto callable = llvm::dyn_cast_or_null<mlir::LLVM::LLVMFuncOp>(
        argument.getOwner()->getParentOp());
    if (callable && argument.getOwner() == &callable.getBody().front()) {
      mlir::LLVM::CallOp incomingCall =
          returnProjectionBinding(callable, returnProjectionCalls);
      if (!incomingCall) {
        for (mlir::LLVM::CallOp call : invocationPath) {
          if (!call.getCalleeAttr())
            return invalid(
                "operation invocation path contains an indirect call");
          auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<
              mlir::LLVM::LLVMFuncOp>(call, call.getCalleeAttr());
          if (callee != callable)
            continue;
          if (incomingCall)
            return invalid(
                "recursive operation invocation path is unsupported");
          incomingCall = call;
        }
      }
      if (incomingCall) {
        const unsigned ordinal = argument.getArgNumber();
        if (ordinal >= incomingCall.getCalleeOperands().size())
          return invalid("callable argument exceeds exact call operands");
        return collectOperationPointerRoots(
            incomingCall.getCalleeOperands()[ordinal], visited, roots,
            invocationPath, returnProjectionCalls, captureBase,
            staticByteOffset);
      }
    }
  }

  if (pointer.getDefiningOp<mlir::LLVM::AllocaOp>() ||
      pointer.getDefiningOp<mlir::LLVM::AddressOfOp>() ||
      llvm::isa<mlir::BlockArgument>(pointer)) {
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(pointer)) {
      if (auto branch = llvm::dyn_cast_or_null<mlir::RegionBranchOpInterface>(
              argument.getOwner()->getParentOp())) {
        mlir::RegionSuccessor successor(argument.getOwner()->getParent());
        return invalid(
            llvm::Twine("region branch block argument ") +
            llvm::Twine(argument.getArgNumber()) + " of '" +
            branch->getName().getStringRef() +
            "' is absent from its canonical successor-input range of " +
            llvm::Twine(branch.getSuccessorInputs(successor).size()));
      }
    }
    OperationPointerOrigin origin{pointer, captureBase ? captureBase : pointer,
                                  staticByteOffset};
    if (!llvm::any_of(roots, [&](const OperationPointerOrigin &candidate) {
          return candidate.staticRoot == origin.staticRoot &&
                 candidate.captureBase == origin.captureBase &&
                 candidate.staticByteOffset == origin.staticByteOffset;
        }))
      roots.push_back(origin);
    return llvm::Error::success();
  }
  if (pointer.getDefiningOp<mlir::LLVM::ZeroOp>() ||
      pointer.getDefiningOp<mlir::LLVM::UndefOp>() ||
      pointer.getDefiningOp<mlir::LLVM::PoisonOp>())
    return llvm::Error::success();
  mlir::Operation *definition = pointer.getDefiningOp();
  return unsupported(
      llvm::Twine("operation memory boundary has no closed pointer-origin "
                  "projection through '") +
      definition->getName().getStringRef() + "'");
}

llvm::Expected<ResolvedOperationObject>
resolveOperationObject(mlir::Value pointer,
                       llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  llvm::DenseMap<OperationPointerVisit, std::uint64_t> visited;
  llvm::SmallVector<OperationPointerOrigin, 4> roots;
  if (llvm::Error error =
          collectOperationPointerRoots(pointer, visited, roots, invocationPath))
    return std::move(error);
  if (roots.empty())
    return unsupported(
        "operation memory boundary has no concrete finite pointer origin");

  std::optional<ResolvedOperationObject> result;
  for (const OperationPointerOrigin &origin : roots) {
    llvm::Expected<ResolvedObject> object = resolveObject(origin.staticRoot);
    if (!object)
      return object.takeError();
    if (object->byteOffset >
        std::numeric_limits<std::uint64_t>::max() - origin.staticByteOffset)
      return invalid("operation memory byte offset overflows uint64");
    object->byteOffset += origin.staticByteOffset;
    if (object->byteOffset >= object->byteCount)
      return invalid("operation memory view begins outside its finite object");
    if (!result) {
      result = ResolvedOperationObject{*object, origin.captureBase};
      continue;
    }
    if (result->object.owner != object->owner ||
        result->object.byteCount != object->byteCount ||
        result->object.byteOffset != object->byteOffset ||
        result->captureBase != origin.captureBase)
      return unsupported(
          "operation memory boundary may select distinct backing views");
  }
  return *result;
}

CanonicalValueSequence exceptionalValue(const detail::LaneShape &shape,
                                        SemanticState state) {
  CanonicalValueSequence value;
  value.tokenCount = 1;
  value.lanes.reserve(shape.lanesPerToken);
  for (std::uint64_t lane = 0; lane < shape.lanesPerToken; ++lane)
    value.lanes.push_back(state == SemanticState::Poison
                              ? SemanticLane::poison()
                              : SemanticLane::undef());
  return value;
}

CanonicalValueSequence definedZeroValue(const detail::LaneShape &shape) {
  CanonicalValueSequence value;
  value.tokenCount = 1;
  value.lanes.reserve(shape.lanesPerToken);
  for (std::uint64_t lane = 0; lane < shape.lanesPerToken; ++lane)
    value.lanes.push_back(
        SemanticLane::defined(llvm::APInt(shape.laneBitWidth, 0)));
  return value;
}

bool graphScalarInputIsUnused(detail::ResolvedLaunchContext &context,
                              std::uint64_t ordinal) {
  if (ordinal >= context.numValueInputs ||
      ordinal >= context.valueInputShapes.size() ||
      context.valueInputShapes[ordinal].pointerLayout)
    return false;
  mlir::Block &entry = context.graphOp.getBody().front();
  return ordinal + 1 < entry.getNumArguments() &&
         entry.getArgument(static_cast<unsigned>(ordinal + 1)).use_empty();
}

bool exceptionalSequence(const CanonicalValueSequence &value) {
  return !value.lanes.empty() &&
         llvm::all_of(value.lanes, [](const SemanticLane &lane) {
           return lane.state == SemanticState::Undef ||
                  lane.state == SemanticState::Poison;
         });
}

llvm::Expected<llvm::APInt> attributeBits(mlir::Attribute attribute,
                                          std::uint32_t width) {
  if (auto integer = llvm::dyn_cast<mlir::IntegerAttr>(attribute)) {
    if (integer.getValue().getBitWidth() != width)
      return invalid("constant integer width differs from its graph input");
    return integer.getValue();
  }
  if (auto floating = llvm::dyn_cast<mlir::FloatAttr>(attribute)) {
    llvm::APInt bits = floating.getValue().bitcastToAPInt();
    if (bits.getBitWidth() != width)
      return invalid("constant float width differs from its graph input");
    return bits;
  }
  return unsupported("constant graph value has no integer or float bits");
}

llvm::Expected<std::optional<CanonicalValueSequence>>
fixedValueOf(mlir::Value source, const detail::LaneShape &shape) {
  if (source.getDefiningOp<mlir::LLVM::UndefOp>())
    return std::optional<CanonicalValueSequence>(
        exceptionalValue(shape, SemanticState::Undef));
  if (source.getDefiningOp<mlir::LLVM::PoisonOp>())
    return std::optional<CanonicalValueSequence>(
        exceptionalValue(shape, SemanticState::Poison));

  mlir::Attribute attribute;
  if (auto constant = source.getDefiningOp<mlir::arith::ConstantOp>())
    attribute = constant.getValue();
  else if (auto constant = source.getDefiningOp<mlir::LLVM::ConstantOp>())
    attribute = constant.getValue();
  if (!attribute)
    return std::optional<CanonicalValueSequence>{};

  CanonicalValueSequence value;
  value.tokenCount = 1;
  value.lanes.reserve(shape.lanesPerToken);
  if (shape.lanesPerToken == 1 && !llvm::isa<mlir::ElementsAttr>(attribute)) {
    llvm::Expected<llvm::APInt> bits =
        attributeBits(attribute, shape.laneBitWidth);
    if (!bits)
      return bits.takeError();
    value.lanes.push_back(SemanticLane::defined(std::move(*bits)));
    return std::optional<CanonicalValueSequence>(std::move(value));
  }

  auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(attribute);
  if (!dense || dense.getNumElements() < 0 ||
      static_cast<std::uint64_t>(dense.getNumElements()) != shape.lanesPerToken)
    return unsupported("vector constant does not match its graph input shape");
  if (llvm::isa<mlir::IntegerType>(dense.getElementType())) {
    for (const llvm::APInt &raw : dense.getValues<llvm::APInt>()) {
      if (raw.getBitWidth() != shape.laneBitWidth)
        return invalid("vector integer lane width differs from graph input");
      value.lanes.push_back(SemanticLane::defined(raw));
    }
  } else if (llvm::isa<mlir::FloatType>(dense.getElementType())) {
    for (const llvm::APFloat &raw : dense.getValues<llvm::APFloat>()) {
      llvm::APInt bits = raw.bitcastToAPInt();
      if (bits.getBitWidth() != shape.laneBitWidth)
        return invalid("vector float lane width differs from graph input");
      value.lanes.push_back(SemanticLane::defined(std::move(bits)));
    }
  } else {
    return unsupported("vector constant element type is not bit-valued");
  }
  return std::optional<CanonicalValueSequence>(std::move(value));
}

llvm::Expected<SimulationValueInputCapture>
operationValueInputCapture(detail::ResolvedLaunchContext &context,
                           std::uint64_t valueInputOrdinal,
                           mlir::ValueRange boundaryInputs) {
  mlir::Value graphSource =
      context.graphLaunchOp.getValueInputs()[valueInputOrdinal];
  mlir::Type graphType =
      context.graphOp.getFunctionType().getInput(valueInputOrdinal);
  llvm::Expected<detail::LaneShape> shape =
      detail::laneShapeOf(graphType, context.graphOp.getOperation());
  if (!shape)
    return shape.takeError();

  mlir::Value boundaryValue;
  std::optional<std::uint64_t> boundaryOrdinal;
  std::optional<std::uint64_t> coordinateDimension;
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(graphSource)) {
    const std::uint64_t inputCount =
        context.thread.getFunctionType().getNumInputs();
    if (argument.getOwner()->getParentOp() != context.thread.getOperation())
      return invalid("graph value input is not a thread body formal");
    if (argument.getArgNumber() < inputCount) {
      boundaryOrdinal = argument.getArgNumber();
      if (*boundaryOrdinal >= boundaryInputs.size())
        return invalid("thread value formal exceeds source boundary inputs");
      boundaryValue = boundaryInputs[*boundaryOrdinal];
    } else {
      if (argument.getArgNumber() <= inputCount ||
          argument.getArgNumber() - inputCount - 1 >= context.threadRank)
        return invalid("graph value input is not a dense coordinate formal");
      coordinateDimension = argument.getArgNumber() - inputCount - 1;
      boundaryValue = graphSource;
    }
  } else {
    boundaryValue = graphSource;
  }

  llvm::Expected<std::optional<CanonicalValueSequence>> fixed =
      fixedValueOf(boundaryValue, *shape);
  if (!fixed)
    return fixed.takeError();
  bool unusedByGraph = false;
  // CFG-to-SCF may leave a dead carry placeholder as a graph input. Preserve
  // the ABI, but project only a proven-unobserved scalar to a defined wire
  // value; any semantic use keeps the exceptional state intact.
  if (*fixed && graphScalarInputIsUnused(context, valueInputOrdinal) &&
      exceptionalSequence(**fixed)) {
    *fixed = definedZeroValue(*shape);
    unusedByGraph = true;
  }
  if (*fixed)
    return SimulationValueInputCapture{
        valueInputOrdinal, std::nullopt, boundaryValue, shape->lanesPerToken,
        shape->laneBitWidth, 0, std::move(*fixed), std::nullopt, unusedByGraph,
        std::nullopt};
  if (!*fixed && !boundaryOrdinal && !coordinateDimension)
    return unsupported(
        "graph value input is neither fixed nor a boundary or coordinate "
        "formal");

  std::uint64_t byteCount = 0;
  if (!*fixed) {
    llvm::Expected<std::uint64_t> bytes = fixedTypeByteCount(
        boundaryValue.getDefiningOp() ? boundaryValue.getDefiningOp()
                                      : context.graphOp.getOperation(),
        boundaryValue.getType());
    if (!bytes)
      return bytes.takeError();
    if (!fitsStorageExtent(*shape, *bytes))
      return invalid("runtime graph value does not fit its storage extent");
    byteCount = *bytes;
  }
  return SimulationValueInputCapture{valueInputOrdinal,   boundaryOrdinal,
                                     boundaryValue,       shape->lanesPerToken,
                                     shape->laneBitWidth, byteCount,
                                     std::move(*fixed),   std::nullopt, false,
                                     coordinateDimension};
}

llvm::Expected<SimulationValueResultCapture>
operationValueResultCapture(detail::ResolvedLaunchContext &context,
                            std::uint64_t valueResultOrdinal,
                            mlir::ValueRange boundaryResults) {
  if (valueResultOrdinal >= boundaryResults.size() ||
      valueResultOrdinal >= context.numValueResults)
    return invalid("graph value result exceeds its Structured boundary");
  mlir::Value boundaryValue = boundaryResults[valueResultOrdinal];
  mlir::Type graphType =
      context.graphOp.getFunctionType().getResult(valueResultOrdinal);
  llvm::Expected<loom::CanonicalSemanticBytes> graphTypeBytes =
      dataflow::encodeCanonicalType(graphType);
  if (!graphTypeBytes)
    return graphTypeBytes.takeError();
  llvm::Expected<loom::CanonicalSemanticBytes> boundaryTypeBytes =
      dataflow::encodeCanonicalType(boundaryValue.getType());
  if (!boundaryTypeBytes)
    return boundaryTypeBytes.takeError();
  if (boundaryTypeBytes->bytes() != graphTypeBytes->bytes()) {
    std::string graphTypeText;
    llvm::raw_string_ostream graphTypeStream(graphTypeText);
    graphType.print(graphTypeStream);
    std::string boundaryTypeText;
    llvm::raw_string_ostream boundaryTypeStream(boundaryTypeText);
    boundaryValue.getType().print(boundaryTypeStream);
    return invalid("graph value result type " + graphTypeText +
                   " differs from its Structured source " + boundaryTypeText);
  }
  llvm::Expected<detail::LaneShape> shape =
      detail::laneShapeOf(graphType, context.graphOp.getOperation());
  if (!shape)
    return shape.takeError();
  llvm::Expected<std::uint64_t> bytes = fixedTypeByteCount(
      boundaryValue.getDefiningOp() ? boundaryValue.getDefiningOp()
                                    : context.graphOp.getOperation(),
      boundaryValue.getType());
  if (!bytes)
    return bytes.takeError();
  if (!fitsStorageExtent(*shape, *bytes))
    return invalid("graph value result does not fit its storage extent");
  return SimulationValueResultCapture{valueResultOrdinal, boundaryValue,
                                      shape->lanesPerToken, shape->laneBitWidth,
                                      *bytes};
}

llvm::Expected<unsigned>
enclosingCallableArgument(mlir::Value value, mlir::LLVM::LLVMFuncOp function) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner() != &function.getBody().front())
    return unsupported("root launch memory is not an enclosing callable "
                       "argument");
  return argument.getArgNumber();
}

llvm::Expected<SimulationValueInputCapture> directCallValueInputCapture(
    detail::ResolvedLaunchContext &context, std::uint64_t valueInputOrdinal,
    mlir::LLVM::LLVMFuncOp enclosingCallable, mlir::LLVM::CallOp hostCall) {
  mlir::Value graphSource =
      context.graphLaunchOp.getValueInputs()[valueInputOrdinal];
  mlir::Type graphType =
      context.graphOp.getFunctionType().getInput(valueInputOrdinal);
  llvm::Expected<detail::LaneShape> shape =
      detail::laneShapeOf(graphType, context.graphOp.getOperation());
  if (!shape)
    return shape.takeError();

  mlir::Value callableSource = graphSource;
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(graphSource)) {
    if (argument.getOwner()->getParentOp() != context.thread.getOperation() ||
        argument.getArgNumber() >=
            context.thread.getFunctionType().getNumInputs() ||
        argument.getArgNumber() >=
            context.rootLaunchOp.getBodyOperands().size())
      return invalid("graph value input is not a rooted thread formal");
    callableSource =
        context.rootLaunchOp.getBodyOperands()[argument.getArgNumber()];
  }

  llvm::Expected<std::optional<CanonicalValueSequence>> fixed =
      fixedValueOf(callableSource, *shape);
  if (!fixed)
    return fixed.takeError();
  bool unusedByGraph = false;
  if (*fixed && graphScalarInputIsUnused(context, valueInputOrdinal) &&
      exceptionalSequence(**fixed)) {
    *fixed = definedZeroValue(*shape);
    unusedByGraph = true;
  }
  if (*fixed)
    return SimulationValueInputCapture{
        valueInputOrdinal,    std::nullopt,        callableSource,
        shape->lanesPerToken, shape->laneBitWidth, 0,
        std::move(*fixed),    std::nullopt,        unusedByGraph, std::nullopt};

  if (llvm::isa<mlir::LLVM::LLVMPointerType>(callableSource.getType()) &&
      callableSource.getDefiningOp<mlir::LLVM::AddressOfOp>()) {
    llvm::Expected<std::uint64_t> bytes = fixedTypeByteCount(
        callableSource.getDefiningOp(), callableSource.getType());
    if (!bytes)
      return bytes.takeError();
    if (!fitsStorageExtent(*shape, *bytes))
      return invalid("runtime global pointer does not fit its storage extent");
    return SimulationValueInputCapture{
        valueInputOrdinal,   std::nullopt, callableSource, shape->lanesPerToken,
        shape->laneBitWidth, *bytes,       std::nullopt,   std::nullopt,
        false,               std::nullopt};
  }

  llvm::Expected<unsigned> callableArgument =
      enclosingCallableArgument(callableSource, enclosingCallable);
  if (!callableArgument)
    return callableArgument.takeError();
  if (*callableArgument >= hostCall.getCalleeOperands().size())
    return invalid("callable value argument exceeds host call operands");
  mlir::Value hostOperand = hostCall.getCalleeOperands()[*callableArgument];

  fixed = fixedValueOf(hostOperand, *shape);
  if (!fixed)
    return fixed.takeError();
  if (*fixed && graphScalarInputIsUnused(context, valueInputOrdinal) &&
      exceptionalSequence(**fixed)) {
    *fixed = definedZeroValue(*shape);
    unusedByGraph = true;
  }
  if (*fixed)
    return SimulationValueInputCapture{
        valueInputOrdinal,    std::nullopt,        hostOperand,
        shape->lanesPerToken, shape->laneBitWidth, 0,
        std::move(*fixed),    std::nullopt,        unusedByGraph, std::nullopt};

  llvm::Expected<std::uint64_t> bytes = fixedTypeByteCount(
      hostOperand.getDefiningOp() ? hostOperand.getDefiningOp()
                                  : hostCall.getOperation(),
      hostOperand.getType());
  if (!bytes)
    return bytes.takeError();
  if (shape->lanesPerToken >
          std::numeric_limits<std::uint64_t>::max() / shape->laneBitWidth ||
      shape->lanesPerToken * shape->laneBitWidth > *bytes * 8)
    return invalid("runtime graph value does not fit its storage extent");
  return SimulationValueInputCapture{valueInputOrdinal,   *callableArgument,
                                     hostOperand,         shape->lanesPerToken,
                                     shape->laneBitWidth, *bytes,
                                     std::nullopt,        std::nullopt,
                                     false,               std::nullopt};
}

llvm::Expected<SimulationValueResultCapture> directCallValueResultCapture(
    detail::ResolvedLaunchContext &context, std::uint64_t valueResultOrdinal,
    mlir::LLVM::LLVMFuncOp enclosingCallable, mlir::LLVM::CallOp hostCall) {
  if (context.numValueResults != 1 || valueResultOrdinal != 0 ||
      context.graphLaunchOp.getValueResults().size() != 1 ||
      hostCall.getNumResults() != 1)
    return invalid(
        "whole-callable capture requires one exact LLVM value result");

  mlir::Value graphResult = context.graphLaunchOp.getValueResults().front();
  if (!graphResult.hasOneUse())
    return invalid("graph value result does not have one thread publication");
  auto store =
      llvm::dyn_cast<mlir::LLVM::StoreOp>(graphResult.use_begin()->getOwner());
  if (!store || store.getValue() != graphResult)
    return invalid("graph value result is not stored into caller-owned state");

  auto threadSlot = llvm::dyn_cast<mlir::BlockArgument>(store.getAddr());
  if (!threadSlot ||
      threadSlot.getOwner() != &context.thread.getBody().front() ||
      threadSlot.getArgNumber() >=
          context.thread.getFunctionType().getNumInputs() ||
      threadSlot.getArgNumber() >=
          context.rootLaunchOp.getBodyOperands().size())
    return invalid("graph value result is not stored through a thread formal");
  mlir::Value callerSlot =
      context.rootLaunchOp.getBodyOperands()[threadSlot.getArgNumber()];
  if (!callerSlot.getDefiningOp<mlir::LLVM::AllocaOp>())
    return invalid("graph value result has no caller-owned result slot");

  auto returnOp = llvm::dyn_cast<mlir::LLVM::ReturnOp>(
      enclosingCallable.getBody().front().getTerminator());
  if (!returnOp || returnOp.getNumOperands() != 1)
    return invalid("whole-callable result has no direct LLVM return");
  auto load = returnOp.getOperand(0).getDefiningOp<mlir::LLVM::LoadOp>();
  if (!load || load.getAddr() != callerSlot)
    return invalid("whole-callable result is not loaded from its result slot");

  if (!context.rootLaunchOp.getAsyncToken().hasOneUse())
    return invalid("whole-callable result has no unique thread wait");
  auto wait = llvm::dyn_cast<dataflow::ThreadWaitOp>(
      context.rootLaunchOp.getAsyncToken().use_begin()->getOwner());
  if (!wait || wait->getBlock() != load->getBlock() ||
      !wait->isBeforeInBlock(load))
    return invalid("whole-callable result is loaded before thread retirement");

  mlir::Type graphType =
      context.graphOp.getFunctionType().getResult(valueResultOrdinal);
  mlir::Type hostType = hostCall.getResult().getType();
  llvm::Expected<loom::CanonicalSemanticBytes> graphTypeBytes =
      dataflow::encodeCanonicalType(graphType);
  if (!graphTypeBytes)
    return graphTypeBytes.takeError();
  llvm::Expected<loom::CanonicalSemanticBytes> hostTypeBytes =
      dataflow::encodeCanonicalType(hostType);
  if (!hostTypeBytes)
    return hostTypeBytes.takeError();
  if (graphTypeBytes->bytes() != hostTypeBytes->bytes() ||
      load.getResult().getType() != hostType)
    return invalid("whole-callable graph and host result types differ");

  llvm::Expected<detail::LaneShape> shape =
      detail::laneShapeOf(graphType, context.graphOp.getOperation());
  if (!shape)
    return shape.takeError();
  llvm::Expected<std::uint64_t> bytes =
      fixedTypeByteCount(hostCall.getOperation(), hostType);
  if (!bytes)
    return bytes.takeError();
  if (!fitsStorageExtent(*shape, *bytes))
    return invalid("whole-callable result does not fit its storage extent");
  return SimulationValueResultCapture{valueResultOrdinal, hostCall.getResult(),
                                      shape->lanesPerToken, shape->laneBitWidth,
                                      *bytes};
}

llvm::Expected<ResolvedDirectCallObject>
directCallMemoryObject(mlir::Value callableSource,
                       mlir::LLVM::LLVMFuncOp enclosingCallable,
                       llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  if (invocationPath.empty())
    return invalid("direct-call memory capture has no invocation path");
  mlir::LLVM::CallOp hostCall = invocationPath.back();
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(callableSource)) {
    if (argument.getOwner() != &enclosingCallable.getBody().front()) {
      mlir::Operation *owner = argument.getOwner()->getParentOp();
      return unsupported(
          llvm::Twine("root launch memory block argument belongs to '") +
          (owner ? owner->getName().getStringRef()
                 : llvm::StringRef("<unknown>")) +
          "', not enclosing callable '" + enclosingCallable.getSymName() + "'");
    }
    const unsigned ordinal = argument.getArgNumber();
    if (ordinal >= hostCall.getCalleeOperands().size())
      return invalid("callable argument exceeds host call operands");
    llvm::Expected<ResolvedObject> object =
        resolveObjectThroughCallPath(callableSource, enclosingCallable,
                                     invocationPath, invocationPath.size() - 1);
    if (!object)
      return object.takeError();
    return ResolvedDirectCallObject{std::move(*object),
                                    DirectCallOperandMemorySource{ordinal},
                                    hostCall.getCalleeOperands()[ordinal]};
  }

  llvm::Expected<ResolvedObject> object = resolveObject(callableSource);
  if (!object)
    return object.takeError();
  auto address = object->base.getDefiningOp<mlir::LLVM::AddressOfOp>();
  if (!address)
    return unsupported(
        "root launch memory is neither a callable argument nor a global");
  return ResolvedDirectCallObject{
      std::move(*object),
      DirectCallGlobalMemorySource{address.getGlobalName().str()},
      callableSource};
}

llvm::Expected<std::uint64_t> directCallOrdinal(mlir::LLVM::LLVMFuncOp caller,
                                                mlir::LLVM::CallOp target,
                                                llvm::StringRef callee) {
  std::uint64_t ordinal = 0;
  bool found = false;
  caller.walk([&](mlir::LLVM::CallOp call) {
    if (found || !call.getCalleeAttr() ||
        call.getCalleeAttr().getValue() != callee)
      return;
    if (call.getOperation() == target.getOperation()) {
      found = true;
      return;
    }
    ++ordinal;
  });
  if (!found)
    return invalid("host call is not owned by its enclosing callable");
  return ordinal;
}

llvm::Expected<DirectCallCaptureSite>
directCallCaptureSite(mlir::LLVM::CallOp hostCall,
                      mlir::LLVM::LLVMFuncOp expectedCallee) {
  if (!hostCall.getCalleeAttr())
    return unsupported("indirect host call has no exact callable relation");
  auto called =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
          hostCall, hostCall.getCalleeAttr());
  if (!called || called != expectedCallee)
    return invalid("host call does not target the selected callable");
  auto caller = hostCall->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!caller)
    return invalid("host call is not enclosed by an LLVM callable");
  llvm::Expected<std::uint64_t> ordinal =
      directCallOrdinal(caller, hostCall, hostCall.getCalleeAttr().getValue());
  if (!ordinal)
    return ordinal.takeError();
  return DirectCallCaptureSite{hostCall, caller.getSymName().str(),
                               expectedCallee.getSymName().str(), *ordinal};
}

llvm::Expected<std::vector<DirectCallCaptureSite>>
directCallCapturePath(llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath,
                      mlir::LLVM::LLVMFuncOp expectedLeaf = {}) {
  if (invocationPath.empty())
    return unsupported("source-backed capture has no direct invocation path");

  std::vector<DirectCallCaptureSite> sites;
  sites.reserve(invocationPath.size());
  for (mlir::LLVM::CallOp hostCall : invocationPath) {
    if (!hostCall.getCalleeAttr())
      return unsupported("indirect host call has no exact callable relation");
    auto callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
            hostCall, hostCall.getCalleeAttr());
    if (!callee)
      return invalid("invocation path callee does not resolve");
    llvm::Expected<DirectCallCaptureSite> site =
        directCallCaptureSite(hostCall, callee);
    if (!site)
      return site.takeError();
    if (!sites.empty() &&
        sites.back().hostCalleeSymbol != site->hostCallerSymbol)
      return invalid("invocation path is not caller-callee contiguous");
    if (site->hostCallerSymbol == site->hostCalleeSymbol ||
        (!sites.empty() &&
         sites.front().hostCallerSymbol == site->hostCalleeSymbol) ||
        llvm::any_of(sites, [&](const DirectCallCaptureSite &prior) {
          return prior.hostCalleeSymbol == site->hostCalleeSymbol;
        }))
      return unsupported("recursive source-backed invocation is unsupported");
    sites.push_back(std::move(*site));
  }
  if (expectedLeaf &&
      sites.back().hostCalleeSymbol != expectedLeaf.getSymName())
    return invalid("invocation path does not reach the selected callable");
  return sites;
}

} // namespace

llvm::Expected<DirectCallSimulationInputCapturePlan>
deriveSimulationInputCapturePlan(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch, mlir::LLVM::CallOp hostCall) {
  return deriveSimulationInputCapturePlan(
      program, launch, llvm::ArrayRef<mlir::LLVM::CallOp>(hostCall));
}

llvm::Expected<DirectCallSimulationInputCapturePlan>
deriveSimulationInputCapturePlan(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch,
    llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  llvm::Expected<detail::ResolvedLaunchContext> context =
      detail::resolveLaunchContext(program, launch);
  if (!context)
    return context.takeError();

  auto enclosingCallable =
      context->rootLaunchOp->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!enclosingCallable)
    return unsupported("root thread launch is not enclosed by an LLVM "
                       "callable");
  llvm::Expected<std::vector<DirectCallCaptureSite>> invocationSites =
      directCallCapturePath(invocationPath, enclosingCallable);
  if (!invocationSites)
    return invocationSites.takeError();
  mlir::LLVM::CallOp hostCall = invocationPath.back();

  DirectCallSimulationInputCapturePlan plan{
      SimulationInputCapturePlan{launch, {}, {}, {}, {}},
      {},
      std::move(*invocationSites)};
  plan.input.valueInputs.reserve(context->numValueInputs);
  for (std::uint64_t ordinal = 0; ordinal < context->numValueInputs;
       ++ordinal) {
    llvm::Expected<SimulationValueInputCapture> value =
        directCallValueInputCapture(*context, ordinal, enclosingCallable,
                                    hostCall);
    if (!value)
      return value.takeError();
    plan.input.valueInputs.push_back(std::move(*value));
  }
  plan.input.valueResults.reserve(context->numValueResults);
  for (std::uint64_t ordinal = 0; ordinal < context->numValueResults;
       ++ordinal) {
    llvm::Expected<SimulationValueResultCapture> value =
        directCallValueResultCapture(*context, ordinal, enclosingCallable,
                                     hostCall);
    if (!value)
      return value.takeError();
    plan.input.valueResults.push_back(std::move(*value));
  }
  std::vector<mlir::Operation *> objectOwners;
  for (const dataflow::LogicalMemoryRootRef &root : context->importedRoots) {
    llvm::Expected<dataflow::CanonicalLogicalMemoryRootView> resolvedRoot =
        program.resolve(root);
    if (!resolvedRoot)
      return resolvedRoot.takeError();
    auto threadSource =
        capture_detail::threadMemorySourceForRoot(*resolvedRoot, *context);
    if (!threadSource)
      return threadSource.takeError();
    mlir::Value callableSource = *threadSource;
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(callableSource)) {
      if (argument.getArgNumber() >=
          context->rootLaunchOp.getBodyOperands().size())
        return invalid("thread memory formal exceeds root launch operands");
      callableSource =
          context->rootLaunchOp.getBodyOperands()[argument.getArgNumber()];
    }
    llvm::Expected<ResolvedDirectCallObject> resolved = directCallMemoryObject(
        callableSource, enclosingCallable, invocationPath);
    if (!resolved)
      return resolved.takeError();
    ResolvedObject &object = resolved->object;
    std::uint64_t objectIndex = 0;
    while (objectIndex < objectOwners.size()) {
      if (objectOwners[objectIndex] == object.owner)
        break;
      ++objectIndex;
    }
    if (objectIndex == plan.input.objects.size()) {
      plan.input.objects.push_back(
          SimulationMemoryCaptureObject{resolved->pointer, object.byteCount,
                                        object.byteOffset, std::nullopt});
      plan.memoryObjectSources.push_back(std::move(resolved->source));
      objectOwners.push_back(object.owner);
    } else if (plan.input.objects[objectIndex].byteCount != object.byteCount)
      return invalid("one host allocation resolved to inconsistent extents");
    if (object.byteOffset >= object.byteCount)
      return invalid("logical root offset is outside its host allocation");
    plan.input.memoryRootBindings.push_back(SimulationMemoryRootCapture{
        root, objectIndex, object.byteOffset,
        memoryRequiresInitialState(*context, root),
        uniformFloatingWriteLaneType(*context, root), resolved->pointer});
  }
  if (llvm::Error error = capture_detail::attachPointerValueTargets(
          program, *context, plan.input))
    return std::move(error);
  return plan;
}

llvm::Expected<OperationSimulationInputCapturePlan>
deriveOperationSimulationInputCapturePlanImpl(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch, mlir::ValueRange boundaryInputs,
    mlir::ValueRange boundaryResults,
    llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  llvm::Expected<detail::ResolvedLaunchContext> context =
      detail::resolveLaunchContext(program, launch);
  if (!context)
    return context.takeError();

  std::vector<DirectCallCaptureSite> invocationSites;
  if (!invocationPath.empty()) {
    llvm::Expected<std::vector<DirectCallCaptureSite>> resolvedPath =
        directCallCapturePath(invocationPath);
    if (!resolvedPath)
      return resolvedPath.takeError();
    invocationSites = std::move(*resolvedPath);
  }

  if (boundaryResults.size() != context->numValueResults)
    return invalid(
        "Structured live-out count differs from graph value results");
  OperationSimulationInputCapturePlan plan{
      SimulationInputCapturePlan{launch, {}, {}, {}, {}},
      std::move(invocationSites)};
  plan.input.valueInputs.reserve(context->numValueInputs);
  for (std::uint64_t ordinal = 0; ordinal < context->numValueInputs;
       ++ordinal) {
    llvm::Expected<SimulationValueInputCapture> value =
        operationValueInputCapture(*context, ordinal, boundaryInputs);
    if (!value)
      return value.takeError();
    plan.input.valueInputs.push_back(std::move(*value));
  }
  plan.input.valueResults.reserve(context->numValueResults);
  for (std::uint64_t ordinal = 0; ordinal < context->numValueResults;
       ++ordinal) {
    llvm::Expected<SimulationValueResultCapture> value =
        operationValueResultCapture(*context, ordinal, boundaryResults);
    if (!value)
      return value.takeError();
    plan.input.valueResults.push_back(std::move(*value));
  }
  std::vector<mlir::Operation *> objectOwners;
  for (const dataflow::LogicalMemoryRootRef &root : context->importedRoots) {
    llvm::Expected<dataflow::CanonicalLogicalMemoryRootView> resolvedRoot =
        program.resolve(root);
    if (!resolvedRoot)
      return resolvedRoot.takeError();
    auto threadSource =
        capture_detail::threadMemorySourceForRoot(*resolvedRoot, *context);
    if (!threadSource)
      return threadSource.takeError();
    mlir::Value pointer = *threadSource;
    if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(pointer)) {
      if (argument.getArgNumber() >= boundaryInputs.size())
        return invalid("thread memory formal exceeds source boundary inputs");
      pointer = boundaryInputs[argument.getArgNumber()];
    }
    llvm::Expected<ResolvedOperationObject> resolved =
        resolveOperationObject(pointer, invocationPath);
    if (!resolved)
      return resolved.takeError();
    llvm::Expected<std::optional<std::uint64_t>> bindingCallOrdinal =
        baseBindingCallOrdinal(resolved->captureBase, pointer, invocationPath);
    if (!bindingCallOrdinal)
      return bindingCallOrdinal.takeError();
    ResolvedObject &object = resolved->object;
    std::uint64_t objectIndex = 0;
    while (objectIndex < objectOwners.size()) {
      if (objectOwners[objectIndex] == object.owner)
        break;
      ++objectIndex;
    }
    if (objectIndex == plan.input.objects.size()) {
      plan.input.objects.push_back(SimulationMemoryCaptureObject{
          resolved->captureBase, object.byteCount, object.byteOffset,
          *bindingCallOrdinal});
      objectOwners.push_back(object.owner);
    } else {
      const SimulationMemoryCaptureObject &existing =
          plan.input.objects[objectIndex];
      if (existing.byteCount != object.byteCount)
        return invalid(
            "one source allocation resolved to inconsistent extents");
      if (existing.base != resolved->captureBase ||
          existing.operandByteOffset != object.byteOffset ||
          existing.baseBindingCallOrdinal != *bindingCallOrdinal)
        return unsupported(
            "one source allocation has inconsistent backing-base projections");
    }
    plan.input.memoryRootBindings.push_back(SimulationMemoryRootCapture{
        root, objectIndex, object.byteOffset,
        memoryRequiresInitialState(*context, root),
        uniformFloatingWriteLaneType(*context, root), pointer});
  }
  if (llvm::Error error = capture_detail::attachPointerValueTargets(
          program, *context, plan.input))
    return std::move(error);
  return plan;
}

llvm::Expected<OperationSimulationInputCapturePlan>
deriveOperationSimulationInputCapturePlan(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch, mlir::ValueRange boundaryInputs,
    mlir::ValueRange boundaryResults) {
  return deriveOperationSimulationInputCapturePlanImpl(
      program, launch, boundaryInputs, boundaryResults, {});
}

llvm::Expected<OperationSimulationInputCapturePlan>
deriveOperationSimulationInputCapturePlan(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch, mlir::ValueRange boundaryInputs,
    mlir::ValueRange boundaryResults, mlir::LLVM::CallOp invocation) {
  return deriveOperationSimulationInputCapturePlanImpl(
      program, launch, boundaryInputs, boundaryResults,
      llvm::ArrayRef<mlir::LLVM::CallOp>(invocation));
}

llvm::Expected<OperationSimulationInputCapturePlan>
deriveOperationSimulationInputCapturePlan(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch, mlir::ValueRange boundaryInputs,
    mlir::ValueRange boundaryResults,
    llvm::ArrayRef<mlir::LLVM::CallOp> invocationPath) {
  return deriveOperationSimulationInputCapturePlanImpl(
      program, launch, boundaryInputs, boundaryResults, invocationPath);
}

} // namespace loom::sim

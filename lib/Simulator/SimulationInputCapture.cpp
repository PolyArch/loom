#include "Simulator/SimulationInputCapture.h"

#include "SimulationWireInternal.h"

#include "Dataflow/IR/DataflowOps.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/APFloat.h"
#include <limits>
#include <optional>
#include <system_error>

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
    if (auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(owner)) {
      if (cast.getInputs().size() == 1 && cast.getResults().size() == 1 &&
          cast.getInputs().front() == memory)
        collectFloatingWrites(cast.getResults().front(), visited, projection);
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
    if (auto cast = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(owner)) {
      if (cast.getInputs().size() == 1 && cast.getResults().size() == 1 &&
          cast.getInputs().front() == memory) {
        collectInitialStateReads(cast.getResults().front(), visited,
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
    if (!index || *index < 0)
      return unsupported("LLVM GEP has no nonnegative constant index");

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
    if (argument.getOwner() != &enclosingCallable.getBody().front())
      return unsupported(
          "memory boundary is not owned by its enclosing callable");
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
  if (auto argument = llvm::dyn_cast<mlir::BlockArgument>(graphSource)) {
    if (argument.getOwner()->getParentOp() != context.thread.getOperation() ||
        argument.getArgNumber() >=
            context.thread.getFunctionType().getNumInputs())
      return invalid("graph value input is not a thread value formal");
    boundaryOrdinal = argument.getArgNumber();
    if (*boundaryOrdinal >= boundaryInputs.size())
      return invalid("thread value formal exceeds source boundary inputs");
    boundaryValue = boundaryInputs[*boundaryOrdinal];
  } else {
    boundaryValue = graphSource;
  }

  llvm::Expected<std::optional<CanonicalValueSequence>> fixed =
      fixedValueOf(boundaryValue, *shape);
  if (!fixed)
    return fixed.takeError();
  if (!*fixed && !boundaryOrdinal)
    return unsupported(
        "graph value input is neither fixed nor a boundary formal");

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
                                     std::move(*fixed)};
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
  if (*fixed)
    return SimulationValueInputCapture{
        valueInputOrdinal,    std::nullopt,        callableSource,
        shape->lanesPerToken, shape->laneBitWidth, 0,
        std::move(*fixed)};

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
  if (*fixed)
    return SimulationValueInputCapture{
        valueInputOrdinal,    std::nullopt,        hostOperand,
        shape->lanesPerToken, shape->laneBitWidth, 0,
        std::move(*fixed)};

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
                                     std::nullopt};
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
    if (argument.getOwner() != &enclosingCallable.getBody().front())
      return unsupported("root launch memory is not owned by its callable");
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
    if (resolvedRoot->op != context->thread.getOperation() ||
        !resolvedRoot->formalArgIndex)
      return invalid("imported root is not owned by the rooted thread");
    const unsigned threadFormal = *resolvedRoot->formalArgIndex;
    if (threadFormal >= context->rootLaunchOp.getBodyOperands().size())
      return invalid("thread memory formal exceeds root launch operands");
    llvm::Expected<ResolvedDirectCallObject> resolved = directCallMemoryObject(
        context->rootLaunchOp.getBodyOperands()[threadFormal],
        enclosingCallable, invocationPath);
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
      plan.input.objects.push_back(SimulationMemoryCaptureObject{
          resolved->pointer, object.byteCount, object.byteOffset});
      plan.memoryObjectSources.push_back(std::move(resolved->source));
      objectOwners.push_back(object.owner);
    } else if (plan.input.objects[objectIndex].byteCount != object.byteCount)
      return invalid("one host allocation resolved to inconsistent extents");
    if (object.byteOffset >= object.byteCount)
      return invalid("logical root offset is outside its host allocation");
    plan.input.memoryRootBindings.push_back(SimulationMemoryRootCapture{
        root, objectIndex, object.byteOffset,
        memoryRequiresInitialState(*context, root),
        uniformFloatingWriteLaneType(*context, root)});
  }
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

  mlir::LLVM::LLVMFuncOp enclosingCallable;
  std::vector<DirectCallCaptureSite> invocationSites;
  if (!invocationPath.empty()) {
    llvm::Expected<std::vector<DirectCallCaptureSite>> resolvedPath =
        directCallCapturePath(invocationPath);
    if (!resolvedPath)
      return resolvedPath.takeError();
    invocationSites = std::move(*resolvedPath);
    mlir::LLVM::CallOp leafCall = invocationPath.back();
    enclosingCallable =
        mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
            leafCall, leafCall.getCalleeAttr());
    if (!enclosingCallable)
      return invalid("operation invocation leaf does not resolve");
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
    if (resolvedRoot->op != context->thread.getOperation() ||
        !resolvedRoot->formalArgIndex)
      return invalid("imported root is not owned by the rooted thread");
    const std::uint64_t boundaryOrdinal = *resolvedRoot->formalArgIndex;
    if (boundaryOrdinal >= boundaryInputs.size())
      return invalid("thread memory formal exceeds source boundary inputs");

    mlir::Value pointer = boundaryInputs[boundaryOrdinal];
    llvm::Expected<ResolvedObject> object =
        invocationPath.empty()
            ? resolveObject(pointer)
            : resolveObjectThroughCallPath(pointer, enclosingCallable,
                                           invocationPath,
                                           invocationPath.size() - 1);
    if (!object)
      return object.takeError();
    std::uint64_t objectIndex = 0;
    while (objectIndex < objectOwners.size()) {
      if (objectOwners[objectIndex] == object->owner)
        break;
      ++objectIndex;
    }
    if (objectIndex == plan.input.objects.size()) {
      plan.input.objects.push_back(SimulationMemoryCaptureObject{
          pointer, object->byteCount, object->byteOffset});
      objectOwners.push_back(object->owner);
    } else if (plan.input.objects[objectIndex].byteCount != object->byteCount) {
      return invalid("one source allocation resolved to inconsistent extents");
    }
    plan.input.memoryRootBindings.push_back(SimulationMemoryRootCapture{
        root, objectIndex, object->byteOffset,
        memoryRequiresInitialState(*context, root),
        uniformFloatingWriteLaneType(*context, root)});
  }
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

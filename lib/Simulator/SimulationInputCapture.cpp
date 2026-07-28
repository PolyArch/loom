#include "Simulator/SimulationInputCapture.h"

#include "SimulationWireInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
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
    if (shape->lanesPerToken >
            std::numeric_limits<std::uint64_t>::max() / shape->laneBitWidth ||
        shape->lanesPerToken * shape->laneBitWidth > *bytes * 8)
      return invalid("runtime graph value does not fit its storage extent");
    byteCount = *bytes;
  }
  return SimulationValueInputCapture{valueInputOrdinal,   boundaryOrdinal,
                                     boundaryValue,       shape->lanesPerToken,
                                     shape->laneBitWidth, byteCount,
                                     std::move(*fixed)};
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

} // namespace

llvm::Expected<DirectCallSimulationInputCapturePlan>
deriveSimulationInputCapturePlan(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch, mlir::LLVM::CallOp hostCall) {
  llvm::Expected<detail::ResolvedLaunchContext> context =
      detail::resolveLaunchContext(program, launch);
  if (!context)
    return context.takeError();

  auto enclosingCallable =
      context->rootLaunchOp->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!enclosingCallable)
    return unsupported("root thread launch is not enclosed by an LLVM "
                       "callable");
  if (!hostCall.getCalleeAttr())
    return unsupported("indirect host call has no exact callable relation");
  auto called =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::LLVM::LLVMFuncOp>(
          hostCall, hostCall.getCalleeAttr());
  if (!called || called != enclosingCallable)
    return invalid("host call does not target the rooted launch callable");

  auto hostCaller = hostCall->getParentOfType<mlir::LLVM::LLVMFuncOp>();
  if (!hostCaller)
    return invalid("host call is not enclosed by an LLVM callable");
  llvm::Expected<std::uint64_t> callOrdinal = directCallOrdinal(
      hostCaller, hostCall, hostCall.getCalleeAttr().getValue());
  if (!callOrdinal)
    return callOrdinal.takeError();

  DirectCallSimulationInputCapturePlan plan{
      SimulationInputCapturePlan{launch, {}, {}, {}}, hostCall,
      hostCaller.getSymName().str(), hostCall.getCalleeAttr().getValue().str(),
      *callOrdinal};
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
    llvm::Expected<unsigned> callableArgument = enclosingCallableArgument(
        context->rootLaunchOp.getBodyOperands()[threadFormal],
        enclosingCallable);
    if (!callableArgument)
      return callableArgument.takeError();
    if (*callableArgument >= hostCall.getCalleeOperands().size())
      return invalid("callable argument exceeds host call operands");

    llvm::Expected<ResolvedObject> object =
        resolveObject(hostCall.getCalleeOperands()[*callableArgument]);
    if (!object)
      return object.takeError();
    std::uint64_t objectIndex = 0;
    while (objectIndex < plan.input.objects.size()) {
      llvm::Expected<ResolvedObject> existing =
          resolveObject(plan.input.objects[objectIndex].base);
      if (!existing)
        return existing.takeError();
      if (existing->owner == object->owner)
        break;
      ++objectIndex;
    }
    if (objectIndex == plan.input.objects.size()) {
      SimulationMemoryCaptureObject capture;
      capture.base = object->base;
      capture.byteCount = object->byteCount;
      capture.boundaryOperandOrdinal = *callableArgument;
      capture.operandByteOffset = object->byteOffset;
      plan.input.objects.push_back(capture);
    } else if (plan.input.objects[objectIndex].byteCount != object->byteCount)
      return invalid("one host allocation resolved to inconsistent extents");
    if (object->byteOffset >= object->byteCount)
      return invalid("logical root offset is outside its host allocation");
    plan.input.memoryRootBindings.push_back(
        SimulationMemoryRootCapture{root, objectIndex, object->byteOffset});
  }
  return plan;
}

llvm::Expected<SimulationInputCapturePlan>
deriveOperationSimulationInputCapturePlan(
    const dataflow::CanonicalDataflowProgramView &program,
    dataflow::RootedGraphLaunchRef launch, mlir::ValueRange boundaryInputs) {
  llvm::Expected<detail::ResolvedLaunchContext> context =
      detail::resolveLaunchContext(program, launch);
  if (!context)
    return context.takeError();

  SimulationInputCapturePlan plan{launch, {}, {}, {}};
  plan.valueInputs.reserve(context->numValueInputs);
  for (std::uint64_t ordinal = 0; ordinal < context->numValueInputs;
       ++ordinal) {
    llvm::Expected<SimulationValueInputCapture> value =
        operationValueInputCapture(*context, ordinal, boundaryInputs);
    if (!value)
      return value.takeError();
    plan.valueInputs.push_back(std::move(*value));
  }
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

    llvm::Expected<ResolvedObject> object =
        resolveObject(boundaryInputs[boundaryOrdinal]);
    if (!object)
      return object.takeError();
    std::uint64_t objectIndex = 0;
    while (objectIndex < plan.objects.size()) {
      llvm::Expected<ResolvedObject> existing =
          resolveObject(plan.objects[objectIndex].base);
      if (!existing)
        return existing.takeError();
      if (existing->owner == object->owner)
        break;
      ++objectIndex;
    }
    if (objectIndex == plan.objects.size()) {
      plan.objects.push_back(
          SimulationMemoryCaptureObject{object->base, object->byteCount,
                                        boundaryOrdinal, object->byteOffset});
    } else if (plan.objects[objectIndex].byteCount != object->byteCount) {
      return invalid("one source allocation resolved to inconsistent extents");
    }
    plan.memoryRootBindings.push_back(
        SimulationMemoryRootCapture{root, objectIndex, object->byteOffset});
  }
  return plan;
}

} // namespace loom::sim

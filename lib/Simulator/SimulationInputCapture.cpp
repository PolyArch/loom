#include "Simulator/SimulationInputCapture.h"

#include "SimulationWireInternal.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include <limits>
#include <optional>
#include <system_error>

namespace loom::sim {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("simulation_memory_capture_invalid: ") + message);
}

llvm::Error unsupported(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::not_supported),
      llvm::Twine("simulation_memory_capture_unsupported: ") + message);
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
    return ResolvedObject{allocation.getResult(), *byteCount, 0};
  }
  return unsupported("call operand does not resolve to a finite LLVM alloca");
}

llvm::Expected<unsigned>
enclosingCallableArgument(mlir::Value value, mlir::LLVM::LLVMFuncOp function) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner() != &function.getBody().front())
    return unsupported("root launch memory is not an enclosing callable "
                       "argument");
  return argument.getArgNumber();
}

} // namespace

llvm::Expected<SimulationMemoryCapturePlan> deriveSimulationMemoryCapturePlan(
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

  SimulationMemoryCapturePlan plan{launch, hostCall, {}, {}};
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
    while (objectIndex < plan.objects.size() &&
           plan.objects[objectIndex].base != object->base)
      ++objectIndex;
    if (objectIndex == plan.objects.size())
      plan.objects.push_back(
          SimulationMemoryCaptureObject{object->base, object->byteCount});
    else if (plan.objects[objectIndex].byteCount != object->byteCount)
      return invalid("one host allocation resolved to inconsistent extents");
    if (object->byteOffset >= object->byteCount)
      return invalid("logical root offset is outside its host allocation");
    plan.memoryRootBindings.push_back(
        SimulationMemoryRootCapture{root, objectIndex, object->byteOffset});
  }
  return plan;
}

} // namespace loom::sim

#include "Frontend/Compilation/StructuredMemoryCommunication.h"

#include "StructuredMemoryCommunicationDetail.h"

#include "Common/IndexWidth.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/Lowering/ExactMemRefLayout.h"
#include "Frontend/Raising/MemoryProvenance.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::frontend {
namespace {

constexpr llvm::StringLiteral decisionSchema =
    "loom.structured_memory_communication.decision.3.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_memory_communication_invalid: " +
                                     message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Expected<std::uint32_t> takeU32(llvm::ArrayRef<std::uint8_t> &bytes,
                                      llvm::StringRef name) {
  if (bytes.size() < 4)
    return invalid("truncated " + name);
  std::uint32_t value = 0;
  for (std::uint8_t byte : bytes.take_front(4))
    value = (value << 8) | byte;
  bytes = bytes.drop_front(4);
  return value;
}

llvm::Expected<std::uint64_t> takeU64(llvm::ArrayRef<std::uint8_t> &bytes,
                                      llvm::StringRef name) {
  if (bytes.size() < 8)
    return invalid("truncated " + name);
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes.take_front(8))
    value = (value << 8) | byte;
  bytes = bytes.drop_front(8);
  return value;
}

llvm::Expected<StructuredEntityRef>
takeEntityRef(llvm::ArrayRef<std::uint8_t> &bytes) {
  if (bytes.size() < structuredEntityRefWireSize)
    return invalid("truncated decision anchor");
  auto reference =
      decodeStructuredEntityRef(bytes.take_front(structuredEntityRefWireSize));
  if (!reference)
    return reference.takeError();
  bytes = bytes.drop_front(structuredEntityRefWireSize);
  return reference;
}

llvm::Expected<mlir::Operation *>
mapTrackedSpatialRegion(const StructuredProgramCandidate &parent,
                        std::optional<StructuredEntityRef> trackedSpatialRegion,
                        mlir::IRMapping &mapping) {
  if (!trackedSpatialRegion)
    return nullptr;
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto entity = view->resolve(*trackedSpatialRegion);
  if (!entity)
    return entity.takeError();
  if (!llvm::isa_and_nonnull<loom::SpatialRegionOp>(entity->operation))
    return invalid("tracked operation is not a Spatial region");
  mlir::Operation *mapped = mapping.lookupOrNull(entity->operation);
  if (!mapped)
    return invalid("tracked Spatial region was not mapped into the clone");
  return mapped;
}

bool isLoadAlignmentEstablishedByBase(mlir::memref::LoadOp load,
                                      mlir::MemRefType type) {
  const std::optional<std::uint64_t> alignment = load.getAlignment();
  if (!alignment || *alignment <= 1)
    return true;
  if (*alignment > static_cast<std::uint64_t>(
                       std::numeric_limits<std::int64_t>::max()) ||
      load.getIndices().size() != static_cast<std::size_t>(type.getRank()))
    return false;

  const llvm::TypeSize elementBytes =
      mlir::DataLayout::closest(load).getTypeSize(type.getElementType());
  if (elementBytes.isScalable() || elementBytes.getFixedValue() == 0)
    return false;

  std::uint64_t linearOffsetModulo = 0;
  for (auto [index, extent] : llvm::zip(load.getIndices(), type.getShape())) {
    llvm::APInt constant;
    if (!mlir::matchPattern(index, mlir::m_ConstantInt(&constant)) ||
        !constant.isSignedIntN(64))
      return false;
    const std::int64_t signedIndex = constant.getSExtValue();
    const std::int64_t signedAlignment = static_cast<std::int64_t>(*alignment);
    const std::int64_t signedRemainder = signedIndex % signedAlignment;
    const std::uint64_t indexModulo = static_cast<std::uint64_t>(
        signedRemainder < 0 ? signedRemainder + signedAlignment
                            : signedRemainder);
    linearOffsetModulo = static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(linearOffsetModulo) *
             static_cast<std::uint64_t>(extent) +
         indexModulo) %
        *alignment);
  }
  return static_cast<std::uint64_t>(
             static_cast<unsigned __int128>(linearOffsetModulo) *
             elementBytes.getFixedValue() % *alignment) == 0;
}

bool hasOnlyDirectLoads(mlir::Value memory, loom::SpatialRegionOp spatial) {
  auto type = llvm::dyn_cast<mlir::MemRefType>(memory.getType());
  if (!type)
    return false;
  bool hasLoad = false;
  for (mlir::OpOperand &use : memory.getUses()) {
    auto load = llvm::dyn_cast<mlir::memref::LoadOp>(use.getOwner());
    if (!load || load.getMemref() != memory ||
        !spatial.getBody().isAncestor(load->getParentRegion()) ||
        !isLoadAlignmentEstablishedByBase(load, type))
      return false;
    hasLoad = true;
  }
  return hasLoad;
}

bool isInitializedConstantGlobalValue(mlir::Value value) {
  auto access = value.getDefiningOp<mlir::memref::GetGlobalOp>();
  if (!access)
    return false;
  auto global =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::memref::GlobalOp>(
          access, access.getNameAttr());
  if (!global || !global.getConstant())
    return false;
  std::optional<mlir::Attribute> initializer = global.getInitialValue();
  return initializer && llvm::isa<mlir::ElementsAttr>(*initializer);
}

bool isConstantAtEveryRootLaunch(dataflow::ThreadOp thread,
                                 std::uint64_t inputOrdinal) {
  auto module = thread->getParentOfType<mlir::ModuleOp>();
  if (!module || inputOrdinal >= thread.getFunctionType().getNumInputs())
    return false;
  std::uint64_t launchCount = 0;
  bool allConstant = true;
  module.walk([&](dataflow::ThreadLaunchOp launch) {
    auto callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<dataflow::ThreadOp>(
            launch, launch.getCalleeAttr());
    if (callee != thread)
      return;
    ++launchCount;
    if (inputOrdinal >= launch.getBodyOperands().size() ||
        !isInitializedConstantGlobalValue(
            launch.getBodyOperands()[inputOrdinal]))
      allConstant = false;
  });
  return launchCount != 0 && allConstant;
}

mlir::BlockArgument memoryArgument(loom::SpatialRegionOp spatial,
                                   std::uint64_t memoryInputOrdinal) {
  if (memoryInputOrdinal >= spatial.getMemoryInputs().size())
    return {};
  const std::uint64_t argumentOrdinal = spatial.getValueInputs().size() +
                                        spatial.getStreamInputs().size() +
                                        memoryInputOrdinal;
  return spatial.getBody().front().getArgument(argumentOrdinal);
}

bool isStageableConstantInput(loom::SpatialRegionOp spatial,
                              std::uint64_t memoryInputOrdinal) {
  mlir::BlockArgument argument = memoryArgument(spatial, memoryInputOrdinal);
  if (!argument)
    return false;
  auto type = llvm::dyn_cast<mlir::MemRefType>(argument.getType());
  if (!type || !type.hasStaticShape() || !type.getLayout().isIdentity() ||
      !hasOnlyDirectLoads(argument, spatial))
    return false;
  auto thread = spatial->getParentOfType<dataflow::ThreadOp>();
  auto threadInput = llvm::dyn_cast<mlir::BlockArgument>(
      spatial.getMemoryInputs()[memoryInputOrdinal]);
  if (!thread || !threadInput ||
      threadInput.getOwner() != &thread.getBody().front() ||
      threadInput.getType() != argument.getType())
    return false;
  return isConstantAtEveryRootLaunch(thread, threadInput.getArgNumber());
}

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> cloneAndResolveMemoryInput(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &reference,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance,
    mlir::BlockArgument &clonedInput, mlir::Operation *&clonedSpatialRegion) {
  if (reference.kind != StructuredEntityKind::Value)
    return invalid("memory decision does not reference a value");
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto entity = view->resolve(reference);
  if (!entity)
    return entity.takeError();
  auto sourceInput = llvm::dyn_cast<mlir::BlockArgument>(entity->value);
  if (!sourceInput)
    return invalid("memory decision does not reference a block argument");

  mlir::IRMapping mapping;
  auto privateClone = cloneStructuredProgramWithSourceLocations(
      parent, sourceProvenance, mapping);
  if (!privateClone)
    return privateClone.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> clone = std::move(*privateClone);
  clonedInput = llvm::dyn_cast_or_null<mlir::BlockArgument>(
      mapping.lookupOrNull(sourceInput));
  if (!clonedInput)
    return invalid("selected memory input was not mapped into the clone");
  auto spatial = mapTrackedSpatialRegion(parent, trackedSpatialRegion, mapping);
  if (!spatial)
    return spatial.takeError();
  clonedSpatialRegion = *spatial;
  return clone;
}

llvm::Error stageConstantGlobal(mlir::BlockArgument input) {
  auto spatial = llvm::dyn_cast_or_null<loom::SpatialRegionOp>(
      input.getOwner()->getParentOp());
  if (!spatial || input.getOwner() != &spatial.getBody().front())
    return invalid("memory input is outside loom.spatial_region");
  const std::uint64_t memoryArgumentBase =
      spatial.getValueInputs().size() + spatial.getStreamInputs().size();
  if (input.getArgNumber() < memoryArgumentBase)
    return invalid("selected value is not a Spatial memory input");
  const std::uint64_t memoryInputOrdinal =
      input.getArgNumber() - memoryArgumentBase;
  if (!isStageableConstantInput(spatial, memoryInputOrdinal))
    return invalid("constant-global staging preconditions are not satisfied");

  llvm::SmallVector<mlir::memref::LoadOp, 8> loads;
  spatial.getBody().walk([&](mlir::memref::LoadOp load) {
    if (load.getMemref() == input)
      loads.push_back(load);
  });
  if (loads.empty())
    return invalid("constant-global staging has no direct load");

  std::uint64_t requiredAlignment = 0;
  for (mlir::memref::LoadOp load : loads)
    requiredAlignment =
        std::max(requiredAlignment, load.getAlignment().value_or(0));

  mlir::Block &entry = spatial.getBody().front();
  mlir::OpBuilder builder(&entry, entry.begin());
  mlir::Location location = spatial.getLoc();
  mlir::IntegerAttr alignment;
  if (requiredAlignment)
    alignment = builder.getI64IntegerAttr(requiredAlignment);
  auto buffer = mlir::memref::AllocOp::create(
      builder, location, llvm::cast<mlir::MemRefType>(input.getType()),
      alignment);
  mlir::memref::CopyOp::create(builder, location, input, buffer);
  for (mlir::memref::LoadOp load : loads)
    load->setOperand(0, buffer);
  return llvm::Error::success();
}

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> cloneAndResolveAllocation(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &reference,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance,
    mlir::memref::AllocOp &clonedAlloc, mlir::Operation *&clonedSpatialRegion) {
  if (reference.kind != StructuredEntityKind::Value)
    return invalid("layout decision does not reference a value");
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto entity = view->resolve(reference);
  if (!entity)
    return entity.takeError();
  auto sourceAlloc = entity->value.getDefiningOp<mlir::memref::AllocOp>();
  if (!sourceAlloc || sourceAlloc.getResult() != entity->value)
    return invalid("layout decision does not reference an allocation result");

  mlir::IRMapping mapping;
  auto privateClone = cloneStructuredProgramWithSourceLocations(
      parent, sourceProvenance, mapping);
  if (!privateClone)
    return privateClone.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> clone = std::move(*privateClone);
  mlir::Value clonedValue = mapping.lookupOrNull(entity->value);
  clonedAlloc = clonedValue.getDefiningOp<mlir::memref::AllocOp>();
  if (!clonedAlloc)
    return invalid("selected allocation was not mapped into the clone");
  auto spatial = mapTrackedSpatialRegion(parent, trackedSpatialRegion, mapping);
  if (!spatial)
    return spatial.takeError();
  clonedSpatialRegion = *spatial;
  return clone;
}

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
cloneAndResolveChannelAllocation(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &reference,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance,
    mlir::Value &clonedValue, mlir::Operation *&clonedSpatialRegion) {
  if (reference.kind != StructuredEntityKind::Value)
    return invalid("channel decision does not reference a value");
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto entity = view->resolve(reference);
  if (!entity)
    return entity.takeError();
  if (!entity->value ||
      (!entity->value.getDefiningOp<mlir::memref::AllocOp>() &&
       !entity->value.getDefiningOp<mlir::LLVM::AllocaOp>()))
    return invalid(
        "channel decision does not reference a supported allocation result");

  mlir::IRMapping mapping;
  auto privateClone = cloneStructuredProgramWithSourceLocations(
      parent, sourceProvenance, mapping);
  if (!privateClone)
    return privateClone.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> clone = std::move(*privateClone);
  clonedValue = mapping.lookupOrNull(entity->value);
  if (!clonedValue)
    return invalid("selected channel allocation was not mapped into the clone");
  auto spatial = mapTrackedSpatialRegion(parent, trackedSpatialRegion, mapping);
  if (!spatial)
    return spatial.takeError();
  clonedSpatialRegion = *spatial;
  return clone;
}

bool hasExactCompatibleCopyEndpoint(mlir::Value other,
                                    mlir::MemRefType selectedType,
                                    unsigned indexBits) {
  auto otherType = llvm::dyn_cast<mlir::MemRefType>(other.getType());
  if (!otherType || otherType.getShape() != selectedType.getShape() ||
      otherType.getElementType() != selectedType.getElementType())
    return false;
  auto layout = lowering::resolveExactMemRefLayout(otherType, indexBits);
  if (!layout) {
    llvm::consumeError(layout.takeError());
    return false;
  }
  return true;
}

llvm::Expected<std::optional<llvm::SmallVector<unsigned, 4>>>
legalLocalLayoutOrder(mlir::memref::AllocOp alloc) {
  auto spatial = alloc->getParentOfType<loom::SpatialRegionOp>();
  if (!spatial || alloc->getParentRegion() != &spatial.getBody())
    return std::nullopt;
  auto indexBits = loom::getIndexBitWidth(alloc);
  if (!indexBits)
    return indexBits.takeError();
  auto order =
      lowering::resolveDenseMemRefStorageOrder(alloc.getType(), *indexBits);
  if (!order) {
    llvm::consumeError(order.takeError());
    return std::nullopt;
  }

  bool hasAddressedUse = false;
  mlir::Value memory = alloc.getResult();
  for (mlir::OpOperand &use : memory.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (owner->getParentRegion() != alloc->getParentRegion())
      return std::nullopt;
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(owner)) {
      if (load.getMemref() != memory)
        return std::nullopt;
      hasAddressedUse = true;
      continue;
    }
    if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(owner)) {
      if (store.getMemref() != memory)
        return std::nullopt;
      hasAddressedUse = true;
      continue;
    }
    if (auto copy = llvm::dyn_cast<mlir::memref::CopyOp>(owner)) {
      mlir::Value other;
      if (copy.getSource() == memory)
        other = copy.getTarget();
      else if (copy.getTarget() == memory)
        other = copy.getSource();
      else
        return std::nullopt;
      if (!hasExactCompatibleCopyEndpoint(other, alloc.getType(), *indexBits))
        return std::nullopt;
      hasAddressedUse = true;
      continue;
    }
    if (auto dealloc = llvm::dyn_cast<mlir::memref::DeallocOp>(owner)) {
      if (dealloc.getMemref() != memory)
        return std::nullopt;
      continue;
    }
    return std::nullopt;
  }
  if (!hasAddressedUse)
    return std::nullopt;
  return std::optional<llvm::SmallVector<unsigned, 4>>(std::move(*order));
}

llvm::Expected<mlir::MemRefType>
exchangeAdjacentStoragePositions(mlir::memref::AllocOp alloc,
                                 std::uint64_t position) {
  auto order = legalLocalLayoutOrder(alloc);
  if (!order)
    return order.takeError();
  if (!*order)
    return invalid("local allocation layout preconditions are not satisfied");
  if (position >= (*order)->size() - 1)
    return invalid("adjacent storage position is out of range");
  std::swap((**order)[position], (**order)[position + 1]);

  llvm::SmallVector<std::int64_t, 4> strides(alloc.getType().getRank());
  std::uint64_t running = 1;
  for (unsigned dimension : llvm::reverse(**order)) {
    if (running >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      return invalid("dense layout stride exceeds int64");
    strides[dimension] = static_cast<std::int64_t>(running);
    const std::uint64_t extent =
        static_cast<std::uint64_t>(alloc.getType().getDimSize(dimension));
    if (extent != 0 &&
        running > std::numeric_limits<std::uint64_t>::max() / extent)
      return invalid("dense layout extent product overflows u64");
    running *= extent;
  }
  auto layout = mlir::StridedLayoutAttr::get(alloc.getContext(), 0, strides);
  auto type = mlir::MemRefType::get(alloc.getType().getShape(),
                                    alloc.getType().getElementType(), layout,
                                    alloc.getType().getMemorySpace());
  if (type == alloc.getType())
    return invalid("adjacent storage exchange does not change the layout");
  return type;
}

llvm::Error
permuteLocalBufferLayout(mlir::memref::AllocOp alloc,
                         const PermuteLocalBufferLayoutDecision &decision) {
  auto type =
      exchangeAdjacentStoragePositions(alloc, decision.adjacentStoragePosition);
  if (!type)
    return type.takeError();
  alloc.getResult().setType(*type);
  return llvm::Error::success();
}

struct PipelineLoopPlan final {
  mlir::scf::ForOp loop;
  mlir::memref::AllocOp buffer;
  mlir::memref::CopyOp copy;
  mlir::memref::DeallocOp dealloc;
  mlir::Value source;
  mlir::memref::SubViewOp sourceView;
  std::uint64_t tripCount = 0;
};

std::optional<std::uint64_t> normalizedStaticTripCount(mlir::scf::ForOp loop) {
  if (!loop.getInitArgs().empty() || loop->getNumResults() != 0)
    return std::nullopt;
  llvm::APInt lower;
  llvm::APInt step;
  if (!mlir::matchPattern(loop.getLowerBound(), mlir::m_ConstantInt(&lower)) ||
      !mlir::matchPattern(loop.getStep(), mlir::m_ConstantInt(&step)) ||
      !lower.isZero() || step.getLimitedValue(2) != 1)
    return std::nullopt;
  std::optional<llvm::APInt> count = loop.getStaticTripCount();
  if (!count || count->getActiveBits() > 64)
    return std::nullopt;
  return count->getZExtValue();
}

bool isDefinedOutside(mlir::Value value, mlir::scf::ForOp loop) {
  mlir::Region *region = value.getParentRegion();
  return !region || !loop.getRegion().isAncestor(region);
}

bool isExactStaticMemRef(mlir::Value value, unsigned indexBits) {
  auto type = llvm::dyn_cast<mlir::MemRefType>(value.getType());
  if (!type || !type.hasStaticShape())
    return false;
  auto layout = lowering::resolveExactMemRefLayout(type, indexBits);
  if (!layout) {
    llvm::consumeError(layout.takeError());
    return false;
  }
  return true;
}

bool hasOneExactIterationDimension(mlir::ValueRange indices,
                                   mlir::MemRefType type, mlir::Value iv,
                                   std::uint64_t tripCount) {
  if (indices.size() != static_cast<std::size_t>(type.getRank()))
    return false;
  bool foundIteration = false;
  for (auto [index, extent] : llvm::zip(indices, type.getShape())) {
    if (index == iv) {
      if (foundIteration || extent < 0 ||
          tripCount > static_cast<std::uint64_t>(extent))
        return false;
      foundIteration = true;
      continue;
    }
    llvm::APInt constant;
    if (!mlir::matchPattern(index, mlir::m_ConstantInt(&constant)) ||
        !constant.isSignedIntN(64))
      return false;
    const std::int64_t coordinate = constant.getSExtValue();
    if (coordinate < 0 || extent < 0 || coordinate >= extent)
      return false;
  }
  return foundIteration;
}

bool isClosedOutput(mlir::Value output, mlir::scf::ForOp loop,
                    std::uint64_t tripCount) {
  auto type = llvm::dyn_cast<mlir::MemRefType>(output.getType());
  if (!type || !type.hasStaticShape() || !isDefinedOutside(output, loop))
    return false;
  bool hasStore = false;
  for (mlir::OpOperand &use : output.getUses()) {
    auto store = llvm::dyn_cast<mlir::memref::StoreOp>(use.getOwner());
    if (!store || store.getMemref() != output ||
        !loop.getRegion().isAncestor(store->getParentRegion()) ||
        !hasOneExactIterationDimension(store.getIndices(), type,
                                       loop.getInductionVar(), tripCount))
      return false;
    hasStore = true;
  }
  return hasStore;
}

std::optional<unsigned>
spatialMemoryThreadInputOrdinal(mlir::Value value,
                                loom::SpatialRegionOp spatial,
                                dataflow::ThreadOp thread) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner() != &spatial.getBody().front())
    return std::nullopt;
  const std::uint64_t memoryBase =
      spatial.getValueInputs().size() + spatial.getStreamInputs().size();
  if (argument.getArgNumber() < memoryBase)
    return std::nullopt;
  const std::uint64_t memoryOrdinal = argument.getArgNumber() - memoryBase;
  if (memoryOrdinal >= spatial.getMemoryInputs().size())
    return std::nullopt;
  auto threadArgument = llvm::dyn_cast<mlir::BlockArgument>(
      spatial.getMemoryInputs()[memoryOrdinal]);
  if (!threadArgument || threadArgument.getOwner() != &thread.getBody().front())
    return std::nullopt;
  return threadArgument.getArgNumber();
}

bool areDistinctAtEveryRootLaunch(mlir::Value source, mlir::Value output,
                                  loom::SpatialRegionOp spatial) {
  auto thread = spatial->getParentOfType<dataflow::ThreadOp>();
  auto module = spatial->getParentOfType<mlir::ModuleOp>();
  if (!thread || !module)
    return false;
  std::optional<unsigned> sourceOrdinal =
      spatialMemoryThreadInputOrdinal(source, spatial, thread);
  std::optional<unsigned> outputOrdinal =
      spatialMemoryThreadInputOrdinal(output, spatial, thread);
  if (!sourceOrdinal || !outputOrdinal || *sourceOrdinal == *outputOrdinal)
    return false;

  std::uint64_t launchCount = 0;
  bool distinct = true;
  module.walk([&](dataflow::ThreadLaunchOp launch) {
    auto callee =
        mlir::SymbolTable::lookupNearestSymbolFrom<dataflow::ThreadOp>(
            launch, launch.getCalleeAttr());
    if (callee != thread)
      return;
    ++launchCount;
    if (*sourceOrdinal >= launch.getBodyOperands().size() ||
        *outputOrdinal >= launch.getBodyOperands().size() ||
        !raising::haveProvenDistinctMemoryRoots(
            launch.getBodyOperands()[*sourceOrdinal],
            launch.getBodyOperands()[*outputOrdinal]))
      distinct = false;
  });
  return launchCount != 0 && distinct;
}

bool isConstantAtEveryRootLaunch(mlir::Value source,
                                 loom::SpatialRegionOp spatial) {
  auto thread = spatial->getParentOfType<dataflow::ThreadOp>();
  if (!thread)
    return false;
  std::optional<unsigned> sourceOrdinal =
      spatialMemoryThreadInputOrdinal(source, spatial, thread);
  return sourceOrdinal && isConstantAtEveryRootLaunch(thread, *sourceOrdinal);
}

bool isClosedSubviewSource(mlir::memref::SubViewOp view,
                           mlir::MemRefType bufferType, mlir::scf::ForOp loop,
                           std::uint64_t tripCount, unsigned indexBits) {
  auto sourceType =
      llvm::dyn_cast<mlir::MemRefType>(view.getSource().getType());
  auto viewType = llvm::dyn_cast<mlir::MemRefType>(view.getType());
  if (!sourceType || !viewType || !sourceType.hasStaticShape() ||
      viewType.getRank() != sourceType.getRank() ||
      viewType.getShape() != bufferType.getShape() ||
      viewType.getElementType() != bufferType.getElementType() ||
      !isDefinedOutside(view.getSource(), loop) ||
      !isExactStaticMemRef(view.getSource(), indexBits))
    return false;

  bool foundIteration = false;
  for (auto [offset, size, stride, sourceExtent, bufferExtent] : llvm::zip(
           view.getMixedOffsets(), view.getMixedSizes(), view.getMixedStrides(),
           sourceType.getShape(), bufferType.getShape())) {
    const std::optional<std::int64_t> staticSize =
        mlir::getConstantIntValue(size);
    const std::optional<std::int64_t> staticStride =
        mlir::getConstantIntValue(stride);
    if (!staticSize || !staticStride || *staticSize != bufferExtent ||
        *staticSize <= 0 || *staticStride != 1)
      return false;
    if (mlir::Value dynamic = offset.dyn_cast<mlir::Value>()) {
      if (foundIteration || dynamic != loop.getInductionVar() ||
          *staticSize != 1 || sourceExtent < 0 ||
          tripCount > static_cast<std::uint64_t>(sourceExtent))
        return false;
      foundIteration = true;
      continue;
    }
    const std::optional<std::int64_t> staticOffset =
        mlir::getConstantIntValue(offset);
    if (!staticOffset || *staticOffset < 0 || sourceExtent < 0 ||
        *staticOffset > sourceExtent - *staticSize)
      return false;
  }
  if (!foundIteration)
    return false;
  if (!llvm::hasSingleElement(view.getResult().getUses()))
    return false;
  return llvm::hasSingleElement(view.getSource().getUses()) &&
         view.getSource().use_begin()->getOwner() == view.getOperation();
}

std::optional<PipelineLoopPlan> analyzePipelineLoop(mlir::scf::ForOp loop) {
  std::optional<std::uint64_t> tripCount = normalizedStaticTripCount(loop);
  auto spatial = loop->getParentOfType<loom::SpatialRegionOp>();
  if (!tripCount || *tripCount < 2 || !spatial ||
      loop->getParentRegion() != &spatial.getBody())
    return std::nullopt;
  auto indexBits = loom::getIndexBitWidth(loop);
  if (!indexBits) {
    llvm::consumeError(indexBits.takeError());
    return std::nullopt;
  }

  PipelineLoopPlan plan;
  plan.loop = loop;
  plan.tripCount = *tripCount;
  llvm::SmallVector<mlir::Operation *, 16> operations;
  for (mlir::Operation &operation : loop.getBody()->without_terminator()) {
    if (operation.getNumRegions() != 0)
      return std::nullopt;
    operations.push_back(&operation);
    if (auto alloc = llvm::dyn_cast<mlir::memref::AllocOp>(operation)) {
      if (plan.buffer)
        return std::nullopt;
      plan.buffer = alloc;
    } else if (auto copy = llvm::dyn_cast<mlir::memref::CopyOp>(operation)) {
      if (plan.copy)
        return std::nullopt;
      plan.copy = copy;
    } else if (auto dealloc =
                   llvm::dyn_cast<mlir::memref::DeallocOp>(operation)) {
      if (plan.dealloc)
        return std::nullopt;
      plan.dealloc = dealloc;
    }
  }
  if (!plan.buffer || !plan.copy || !plan.dealloc ||
      plan.copy.getTarget() != plan.buffer.getResult() ||
      plan.dealloc.getMemref() != plan.buffer.getResult())
    return std::nullopt;
  auto bufferType = plan.buffer.getType();
  if (!bufferType.hasStaticShape() || !bufferType.getLayout().isIdentity() ||
      llvm::any_of(bufferType.getShape(),
                   [](std::int64_t extent) { return extent <= 0; }) ||
      !plan.buffer.getDynamicSizes().empty() ||
      !plan.buffer.getSymbolOperands().empty())
    return std::nullopt;

  plan.sourceView =
      plan.copy.getSource().getDefiningOp<mlir::memref::SubViewOp>();
  if (plan.sourceView) {
    plan.source = plan.sourceView.getSource();
    if (!isClosedSubviewSource(plan.sourceView, bufferType, loop, *tripCount,
                               *indexBits))
      return std::nullopt;
  } else {
    plan.source = plan.copy.getSource();
    auto sourceType = llvm::dyn_cast<mlir::MemRefType>(plan.source.getType());
    if (!sourceType || sourceType.getShape() != bufferType.getShape() ||
        sourceType.getElementType() != bufferType.getElementType() ||
        !isDefinedOutside(plan.source, loop) ||
        !isExactStaticMemRef(plan.source, *indexBits) ||
        !isConstantAtEveryRootLaunch(plan.source, spatial) ||
        !llvm::hasSingleElement(plan.source.getUses()) ||
        plan.source.use_begin()->getOwner() != plan.copy.getOperation())
      return std::nullopt;
  }

  auto allocPosition = llvm::find(operations, plan.buffer.getOperation());
  auto viewPosition =
      plan.sourceView ? llvm::find(operations, plan.sourceView.getOperation())
                      : operations.end();
  auto copyPosition = llvm::find(operations, plan.copy.getOperation());
  auto deallocPosition = llvm::find(operations, plan.dealloc.getOperation());
  if (allocPosition == operations.end() || copyPosition == operations.end() ||
      deallocPosition == operations.end() || allocPosition >= copyPosition ||
      (plan.sourceView && viewPosition >= copyPosition) ||
      copyPosition >= deallocPosition ||
      std::next(deallocPosition) != operations.end())
    return std::nullopt;
  for (auto position = operations.begin(); position != copyPosition; ++position)
    if (*position != plan.buffer.getOperation() &&
        (!plan.sourceView || *position != plan.sourceView.getOperation()))
      return std::nullopt;

  bool hasBufferLoad = false;
  llvm::SmallVector<mlir::Value, 4> outputs;
  for (auto position = std::next(copyPosition); position != operations.end();
       ++position) {
    mlir::Operation *operation = *position;
    if (operation == plan.dealloc.getOperation())
      continue;
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation)) {
      if (load.getMemref() != plan.buffer.getResult())
        return std::nullopt;
      hasBufferLoad = true;
      continue;
    }
    if (auto store = llvm::dyn_cast<mlir::memref::StoreOp>(operation)) {
      if (store.getMemref() == plan.buffer.getResult() ||
          !hasOneExactIterationDimension(store.getIndices(),
                                         store.getMemRefType(),
                                         loop.getInductionVar(), *tripCount))
        return std::nullopt;
      outputs.push_back(store.getMemref());
      continue;
    }
    if (!mlir::isMemoryEffectFree(operation))
      return std::nullopt;
    for (mlir::Value operand : operation->getOperands()) {
      if (operand == loop.getInductionVar() || isDefinedOutside(operand, loop))
        continue;
      mlir::Operation *definition = operand.getDefiningOp();
      if (!definition || llvm::find(operations, definition) >= position ||
          llvm::find(operations, definition) <= copyPosition)
        return std::nullopt;
    }
  }
  if (!hasBufferLoad || outputs.empty())
    return std::nullopt;
  llvm::SmallVector<mlir::Value, 4> uniqueOutputs;
  for (mlir::Value output : outputs)
    if (!llvm::is_contained(uniqueOutputs, output))
      uniqueOutputs.push_back(output);
  for (mlir::Value output : uniqueOutputs)
    if (output == plan.source || !isExactStaticMemRef(output, *indexBits) ||
        !isClosedOutput(output, loop, *tripCount) ||
        !areDistinctAtEveryRootLaunch(plan.source, output, spatial))
      return std::nullopt;

  unsigned copyUses = 0;
  unsigned loadUses = 0;
  unsigned deallocUses = 0;
  for (mlir::OpOperand &use : plan.buffer.getResult().getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (owner == plan.copy.getOperation() &&
        plan.copy.getTarget() == plan.buffer.getResult())
      ++copyUses;
    else if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(owner);
             load && load.getMemref() == plan.buffer.getResult())
      ++loadUses;
    else if (owner == plan.dealloc.getOperation())
      ++deallocUses;
    else
      return std::nullopt;
  }
  if (copyUses != 1 || loadUses == 0 || deallocUses != 1)
    return std::nullopt;
  return plan;
}

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> cloneAndResolvePipelineLoop(
    const StructuredProgramCandidate &parent,
    const StructuredEntityRef &reference,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance,
    mlir::scf::ForOp &clonedLoop, mlir::Operation *&clonedSpatialRegion) {
  if (reference.kind != StructuredEntityKind::Operation)
    return invalid("pipeline decision does not reference an operation");
  auto view = parent.view();
  if (!view)
    return view.takeError();
  auto entity = view->resolve(reference);
  if (!entity)
    return entity.takeError();
  auto sourceLoop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(entity->operation);
  if (!sourceLoop)
    return invalid("pipeline decision does not reference scf.for");

  mlir::IRMapping mapping;
  auto privateClone = cloneStructuredProgramWithSourceLocations(
      parent, sourceProvenance, mapping);
  if (!privateClone)
    return privateClone.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> clone = std::move(*privateClone);
  clonedLoop = llvm::dyn_cast_or_null<mlir::scf::ForOp>(
      mapping.lookupOrNull(sourceLoop.getOperation()));
  if (!clonedLoop)
    return invalid("selected pipeline loop was not mapped into the clone");
  auto spatial = mapTrackedSpatialRegion(parent, trackedSpatialRegion, mapping);
  if (!spatial)
    return spatial.takeError();
  clonedSpatialRegion = *spatial;
  return clone;
}

mlir::Value constantIndex(mlir::OpBuilder &builder, mlir::Location location,
                          std::uint64_t value) {
  return mlir::arith::ConstantOp::create(
             builder, location,
             builder.getIntegerAttr(builder.getIndexType(), value))
      .getResult();
}

void emitStagingCopy(mlir::OpBuilder &builder, PipelineLoopPlan &plan,
                     mlir::Value ring, mlir::Value logicalIteration,
                     mlir::Value slot) {
  const auto bufferType = plan.buffer.getType();
  llvm::SmallVector<mlir::Value, 4> logicalIndices;
  std::function<void(unsigned)> emitDimension = [&](unsigned dimension) {
    if (dimension == static_cast<unsigned>(bufferType.getRank())) {
      llvm::SmallVector<mlir::Value, 4> sourceIndices;
      if (plan.sourceView) {
        for (auto [offset, logicalIndex] :
             llvm::zip(plan.sourceView.getMixedOffsets(), logicalIndices)) {
          mlir::Value base;
          if (mlir::Value dynamic = offset.dyn_cast<mlir::Value>()) {
            base = logicalIteration;
          } else {
            base = constantIndex(
                builder, plan.loop.getLoc(),
                static_cast<std::uint64_t>(*mlir::getConstantIntValue(offset)));
          }
          sourceIndices.push_back(
              mlir::arith::AddIOp::create(builder, plan.loop.getLoc(), base,
                                          logicalIndex)
                  .getResult());
        }
      } else {
        sourceIndices.assign(logicalIndices.begin(), logicalIndices.end());
      }
      llvm::SmallVector<mlir::Value, 5> ringIndices{slot};
      ringIndices.append(logicalIndices.begin(), logicalIndices.end());
      mlir::Value element =
          mlir::memref::LoadOp::create(builder, plan.loop.getLoc(), plan.source,
                                       sourceIndices)
              .getResult();
      mlir::memref::StoreOp::create(builder, plan.loop.getLoc(), element, ring,
                                    ringIndices);
      return;
    }
    mlir::Value lower = constantIndex(builder, plan.loop.getLoc(), 0);
    mlir::Value upper = constantIndex(
        builder, plan.loop.getLoc(),
        static_cast<std::uint64_t>(bufferType.getDimSize(dimension)));
    mlir::Value step = constantIndex(builder, plan.loop.getLoc(), 1);
    auto loop = mlir::scf::ForOp::create(builder, plan.loop.getLoc(), lower,
                                         upper, step);
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(loop.getBody()->getTerminator());
    logicalIndices.push_back(loop.getInductionVar());
    emitDimension(dimension + 1);
    logicalIndices.pop_back();
  };
  emitDimension(0);
}

void emitComputeSuffix(mlir::OpBuilder &builder, PipelineLoopPlan &plan,
                       mlir::Value ring, mlir::Value logicalIteration,
                       mlir::Value slot) {
  mlir::IRMapping mapping;
  mapping.map(plan.loop.getInductionVar(), logicalIteration);
  bool afterCopy = false;
  for (mlir::Operation &operation : plan.loop.getBody()->without_terminator()) {
    if (&operation == plan.copy.getOperation()) {
      afterCopy = true;
      continue;
    }
    if (!afterCopy || &operation == plan.dealloc.getOperation())
      continue;
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(operation)) {
      llvm::SmallVector<mlir::Value, 5> indices{slot};
      for (mlir::Value index : load.getIndices())
        indices.push_back(mapping.lookupOrDefault(index));
      auto replacement =
          mlir::memref::LoadOp::create(builder, load.getLoc(), ring, indices);
      replacement->setAttrs(load->getAttrs());
      mapping.map(load.getResult(), replacement.getResult());
      continue;
    }
    builder.clone(operation, mapping);
  }
}

llvm::Error materializePipelineLoop(mlir::scf::ForOp loop) {
  std::optional<PipelineLoopPlan> plan = analyzePipelineLoop(loop);
  if (!plan)
    return invalid("pipeline preconditions are not satisfied");
  mlir::OpBuilder builder(loop);
  mlir::Location location = loop.getLoc();
  llvm::SmallVector<std::int64_t, 5> ringShape{2};
  ringShape.append(plan->buffer.getType().getShape().begin(),
                   plan->buffer.getType().getShape().end());
  auto ringType = mlir::MemRefType::get(
      ringShape, plan->buffer.getType().getElementType(), mlir::AffineMap(),
      plan->buffer.getType().getMemorySpace());
  auto ring = mlir::memref::AllocOp::create(builder, location, ringType,
                                            plan->buffer.getAlignmentAttr());

  mlir::Value zero = constantIndex(builder, location, 0);
  mlir::Value one = constantIndex(builder, location, 1);
  mlir::Value two = constantIndex(builder, location, 2);
  mlir::Value last = constantIndex(builder, location, plan->tripCount - 1);
  emitStagingCopy(builder, *plan, ring, zero, zero);

  auto kernel = mlir::scf::ForOp::create(builder, location, zero, last, one);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(kernel.getBody()->getTerminator());
    mlir::Value iteration = kernel.getInductionVar();
    mlir::Value next =
        mlir::arith::AddIOp::create(builder, location, iteration, one)
            .getResult();
    mlir::Value nextSlot =
        mlir::arith::RemUIOp::create(builder, location, next, two).getResult();
    emitStagingCopy(builder, *plan, ring, next, nextSlot);
    mlir::Value slot =
        mlir::arith::RemUIOp::create(builder, location, iteration, two)
            .getResult();
    emitComputeSuffix(builder, *plan, ring, iteration, slot);
  }
  mlir::Value lastSlot =
      constantIndex(builder, location, (plan->tripCount - 1) % 2);
  emitComputeSuffix(builder, *plan, ring, last, lastSlot);
  mlir::memref::DeallocOp::create(builder, location, ring);
  loop.erase();
  return llvm::Error::success();
}

} // namespace

StructuredMemoryCommunicationDecisionKind
structuredMemoryCommunicationDecisionKind(
    const StructuredMemoryCommunicationDecision &decision) {
  return std::visit(
      [](const auto &typed) -> StructuredMemoryCommunicationDecisionKind {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, StageConstantGlobalDecision>)
          return StructuredMemoryCommunicationDecisionKind::StageConstantGlobal;
        if constexpr (std::is_same_v<T, PermuteLocalBufferLayoutDecision>)
          return StructuredMemoryCommunicationDecisionKind::
              PermuteLocalBufferLayout;
        if constexpr (std::is_same_v<T, PipelineStagedLoopDecision>)
          return StructuredMemoryCommunicationDecisionKind::PipelineStagedLoop;
        return StructuredMemoryCommunicationDecisionKind::
            PromoteOrderedBufferToChannel;
      },
      decision);
}

const StructuredEntityRef &structuredMemoryCommunicationDecisionAnchor(
    const StructuredMemoryCommunicationDecision &decision) {
  return std::visit(
      [](const auto &typed) -> const StructuredEntityRef & {
        return typed.anchor;
      },
      decision);
}

llvm::ArrayRef<std::uint8_t>
structuredMemoryCommunicationDecisionSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(decisionSchema.data()),
          decisionSchema.size()};
}

llvm::Expected<std::vector<std::uint8_t>>
encodeStructuredMemoryCommunicationDecision(
    const StructuredMemoryCommunicationDecision &decision) {
  const StructuredMemoryCommunicationDecisionKind kind =
      structuredMemoryCommunicationDecisionKind(decision);
  const StructuredEntityRef &anchor =
      structuredMemoryCommunicationDecisionAnchor(decision);
  const StructuredEntityKind expectedKind =
      kind == StructuredMemoryCommunicationDecisionKind::PipelineStagedLoop
          ? StructuredEntityKind::Operation
          : StructuredEntityKind::Value;
  if (anchor.kind != expectedKind)
    return invalid("decision anchor has the wrong entity kind");

  std::vector<std::uint8_t> bytes;
  appendU32(bytes, static_cast<std::uint32_t>(kind));
  std::vector<std::uint8_t> encodedAnchor = encodeStructuredEntityRef(anchor);
  bytes.insert(bytes.end(), encodedAnchor.begin(), encodedAnchor.end());
  if (const auto *layout =
          std::get_if<PermuteLocalBufferLayoutDecision>(&decision))
    appendU64(bytes, layout->adjacentStoragePosition);
  return bytes;
}

llvm::Expected<StructuredMemoryCommunicationDecision>
adoptStructuredMemoryCommunicationDecision(
    llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  llvm::ArrayRef<std::uint8_t> remaining = canonicalBytes;
  auto rawKind = takeU32(remaining, "decision kind");
  if (!rawKind)
    return rawKind.takeError();
  if (*rawKind >
      static_cast<std::uint32_t>(StructuredMemoryCommunicationDecisionKind::
                                     PromoteOrderedBufferToChannel))
    return invalid("decision payload has an unknown kind");
  auto anchor = takeEntityRef(remaining);
  if (!anchor)
    return anchor.takeError();

  const auto kind =
      static_cast<StructuredMemoryCommunicationDecisionKind>(*rawKind);
  std::optional<StructuredMemoryCommunicationDecision> decision;
  switch (kind) {
  case StructuredMemoryCommunicationDecisionKind::StageConstantGlobal:
    if (anchor->kind != StructuredEntityKind::Value)
      return invalid("constant-staging anchor is not a value");
    decision = StageConstantGlobalDecision{*anchor};
    break;
  case StructuredMemoryCommunicationDecisionKind::PermuteLocalBufferLayout: {
    if (anchor->kind != StructuredEntityKind::Value)
      return invalid("layout anchor is not a value");
    auto position = takeU64(remaining, "adjacent storage position");
    if (!position)
      return position.takeError();
    decision = PermuteLocalBufferLayoutDecision{*anchor, *position};
    break;
  }
  case StructuredMemoryCommunicationDecisionKind::PipelineStagedLoop:
    if (anchor->kind != StructuredEntityKind::Operation)
      return invalid("pipeline anchor is not an operation");
    decision = PipelineStagedLoopDecision{*anchor};
    break;
  case StructuredMemoryCommunicationDecisionKind::PromoteOrderedBufferToChannel:
    if (anchor->kind != StructuredEntityKind::Value)
      return invalid("channel-promotion anchor is not a value");
    decision = PromoteOrderedBufferToChannelDecision{*anchor};
    break;
  }
  if (!remaining.empty())
    return invalid("decision payload has trailing bytes");
  auto reencoded = encodeStructuredMemoryCommunicationDecision(*decision);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*reencoded) != canonicalBytes)
    return invalid("decision payload does not re-encode exactly");
  return std::move(*decision);
}

llvm::Expected<StructuredMemoryCommunicationDecisionDomain>
enumerateStructuredMemoryCommunicationDecisions(
    const StructuredProgramCandidate &parent,
    std::uint64_t scopeExpansionLimit) {
  if (scopeExpansionLimit == 0)
    return invalid("scope expansion limit must be positive");
  auto view = parent.view();
  if (!view)
    return view.takeError();

  llvm::DenseMap<mlir::Value, StructuredEntityRef> valueReferences;
  for (const StructuredEntity &entity :
       view->entities(StructuredEntityKind::Value))
    valueReferences.try_emplace(entity.value, entity.reference);
  std::vector<StructuredMemoryCommunicationDecision> decisions;
  std::uint64_t inspectedMemoryScopes = 0;
  for (const StructuredEntity &entity :
       view->entities(StructuredEntityKind::Operation)) {
    if (auto spatial =
            llvm::dyn_cast_or_null<loom::SpatialRegionOp>(entity.operation)) {
      for (std::uint64_t ordinal = 0;
           ordinal != spatial.getMemoryInputs().size(); ++ordinal) {
        if (inspectedMemoryScopes == scopeExpansionLimit)
          return StructuredMemoryCommunicationDecisionDomain{
              std::move(decisions), inspectedMemoryScopes};
        ++inspectedMemoryScopes;
        if (!isStageableConstantInput(spatial, ordinal))
          continue;
        mlir::BlockArgument argument = memoryArgument(spatial, ordinal);
        auto found = valueReferences.find(argument);
        if (found == valueReferences.end())
          return invalid(
              "Spatial memory input has no canonical value reference");
        decisions.emplace_back(StageConstantGlobalDecision{found->second});
      }
      continue;
    }

    if (auto loop =
            llvm::dyn_cast_or_null<mlir::scf::ForOp>(entity.operation)) {
      if (inspectedMemoryScopes == scopeExpansionLimit)
        return StructuredMemoryCommunicationDecisionDomain{
            std::move(decisions), inspectedMemoryScopes};
      ++inspectedMemoryScopes;
      if (analyzePipelineLoop(loop))
        decisions.emplace_back(PipelineStagedLoopDecision{entity.reference});
      continue;
    }

    if (auto alloca =
            llvm::dyn_cast_or_null<mlir::LLVM::AllocaOp>(entity.operation)) {
      if (inspectedMemoryScopes == scopeExpansionLimit)
        return StructuredMemoryCommunicationDecisionDomain{
            std::move(decisions), inspectedMemoryScopes};
      ++inspectedMemoryScopes;
      auto found = valueReferences.find(alloca.getRes());
      if (detail::canPromoteOrderedBufferToChannel(alloca)) {
        if (found == valueReferences.end())
          return invalid("source allocation has no canonical value reference");
        decisions.emplace_back(
            PromoteOrderedBufferToChannelDecision{found->second});
      }
      continue;
    }

    auto alloc =
        llvm::dyn_cast_or_null<mlir::memref::AllocOp>(entity.operation);
    if (!alloc)
      continue;
    if (inspectedMemoryScopes == scopeExpansionLimit)
      return StructuredMemoryCommunicationDecisionDomain{std::move(decisions),
                                                         inspectedMemoryScopes};
    ++inspectedMemoryScopes;
    auto order = legalLocalLayoutOrder(alloc);
    if (!order)
      return order.takeError();
    auto found = valueReferences.find(alloc.getResult());
    if ((*order || detail::canPromoteOrderedBufferToChannel(alloc)) &&
        found == valueReferences.end())
      return invalid("local allocation has no canonical value reference");
    if (*order) {
      for (std::uint64_t position = 0; position + 1 < (*order)->size();
           ++position) {
        auto changed = exchangeAdjacentStoragePositions(alloc, position);
        if (!changed) {
          llvm::consumeError(changed.takeError());
          continue;
        }
        decisions.emplace_back(
            PermuteLocalBufferLayoutDecision{found->second, position});
      }
    }
    if (detail::canPromoteOrderedBufferToChannel(alloc))
      decisions.emplace_back(
          PromoteOrderedBufferToChannelDecision{found->second});
  }
  return StructuredMemoryCommunicationDecisionDomain{std::move(decisions),
                                                     inspectedMemoryScopes};
}

llvm::Expected<MaterializedStructuredMemoryCommunicationCandidate>
materializeStructuredMemoryCommunicationDecision(
    const StructuredProgramCandidate &parent,
    const StructuredMemoryCommunicationDecision &decision,
    std::optional<StructuredEntityRef> trackedSpatialRegion,
    llvm::ArrayRef<StructuredOperationSourceProvenance> sourceProvenance) {
  auto encoded = encodeStructuredMemoryCommunicationDecision(decision);
  if (!encoded)
    return encoded.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> clone;
  mlir::Operation *clonedSpatialRegion = nullptr;
  if (const auto *stage = std::get_if<StageConstantGlobalDecision>(&decision)) {
    mlir::BlockArgument input;
    auto resolved = cloneAndResolveMemoryInput(
        parent, stage->anchor, trackedSpatialRegion, sourceProvenance, input,
        clonedSpatialRegion);
    if (!resolved)
      return resolved.takeError();
    clone = std::move(*resolved);
    if (llvm::Error error = stageConstantGlobal(input))
      return std::move(error);
  } else if (const auto *layout =
                 std::get_if<PermuteLocalBufferLayoutDecision>(&decision)) {
    mlir::memref::AllocOp alloc;
    auto resolved =
        cloneAndResolveAllocation(parent, layout->anchor, trackedSpatialRegion,
                                  sourceProvenance, alloc, clonedSpatialRegion);
    if (!resolved)
      return resolved.takeError();
    clone = std::move(*resolved);
    if (llvm::Error error = permuteLocalBufferLayout(alloc, *layout))
      return std::move(error);
  } else if (const auto *pipeline =
                 std::get_if<PipelineStagedLoopDecision>(&decision)) {
    mlir::scf::ForOp loop;
    auto resolved = cloneAndResolvePipelineLoop(
        parent, pipeline->anchor, trackedSpatialRegion, sourceProvenance, loop,
        clonedSpatialRegion);
    if (!resolved)
      return resolved.takeError();
    clone = std::move(*resolved);
    if (llvm::Error error = materializePipelineLoop(loop))
      return std::move(error);
  } else if (const auto *channel =
                 std::get_if<PromoteOrderedBufferToChannelDecision>(
                     &decision)) {
    mlir::Value allocation;
    auto resolved = cloneAndResolveChannelAllocation(
        parent, channel->anchor, trackedSpatialRegion, sourceProvenance,
        allocation, clonedSpatialRegion);
    if (!resolved)
      return resolved.takeError();
    clone = std::move(*resolved);
    if (auto alloc = allocation.getDefiningOp<mlir::memref::AllocOp>()) {
      if (llvm::Error error =
              detail::promoteOrderedBufferToChannel(alloc, clonedSpatialRegion))
        return std::move(error);
    } else if (auto alloca = allocation.getDefiningOp<mlir::LLVM::AllocaOp>()) {
      if (llvm::Error error = detail::promoteOrderedBufferToChannel(
              alloca, clonedSpatialRegion))
        return std::move(error);
    } else {
      return invalid("channel allocation changed representation in the clone");
    }
  } else {
    return invalid("unknown memory communication decision");
  }
  if (mlir::failed(mlir::verify(*clone)))
    return invalid("materialized memory candidate does not verify");
  auto finalized = finalizeStructuredProgramWithTrackedEntities(
      clone.get(), {},
      clonedSpatialRegion ? llvm::ArrayRef(&clonedSpatialRegion, 1)
                          : llvm::ArrayRef<mlir::Operation *>{});
  if (!finalized)
    return finalized.takeError();
  if (finalized->trackedOperations.size() !=
      static_cast<std::size_t>(clonedSpatialRegion != nullptr))
    return invalid("tracked Spatial region projection changed cardinality");
  return MaterializedStructuredMemoryCommunicationCandidate{
      std::move(finalized->artifact),
      finalized->trackedOperations.empty()
          ? std::nullopt
          : std::optional(finalized->trackedOperations.front()),
      std::move(finalized->sourceProvenance)};
}

} // namespace loom::frontend

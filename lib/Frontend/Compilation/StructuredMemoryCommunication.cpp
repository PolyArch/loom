#include "Frontend/Compilation/StructuredMemoryCommunication.h"

#include "Common/IndexWidth.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/Lowering/ExactMemRefLayout.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
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
    "loom.structured_memory_communication.decision.2.0";

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

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
cloneAndResolveMemoryInput(const StructuredProgramCandidate &parent,
                           const StructuredEntityRef &reference,
                           mlir::BlockArgument &clonedInput) {
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
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      llvm::cast<mlir::ModuleOp>(parent.module()->clone(mapping)));
  clonedInput = llvm::dyn_cast_or_null<mlir::BlockArgument>(
      mapping.lookupOrNull(sourceInput));
  if (!clonedInput)
    return invalid("selected memory input was not mapped into the clone");
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

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
cloneAndResolveAllocation(const StructuredProgramCandidate &parent,
                          const StructuredEntityRef &reference,
                          mlir::memref::AllocOp &clonedAlloc) {
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
  mlir::OwningOpRef<mlir::ModuleOp> clone(
      llvm::cast<mlir::ModuleOp>(parent.module()->clone(mapping)));
  mlir::Value clonedValue = mapping.lookupOrNull(entity->value);
  clonedAlloc = clonedValue.getDefiningOp<mlir::memref::AllocOp>();
  if (!clonedAlloc)
    return invalid("selected allocation was not mapped into the clone");
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
            PromoteSpscBufferToChannel;
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
                                     PromoteSpscBufferToChannel))
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
  case StructuredMemoryCommunicationDecisionKind::PromoteSpscBufferToChannel:
    if (anchor->kind != StructuredEntityKind::Value)
      return invalid("channel-promotion anchor is not a value");
    decision = PromoteSpscBufferToChannelDecision{*anchor};
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
    if (!*order)
      continue;
    auto found = valueReferences.find(alloc.getResult());
    if (found == valueReferences.end())
      return invalid("local allocation has no canonical value reference");
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
  return StructuredMemoryCommunicationDecisionDomain{std::move(decisions),
                                                     inspectedMemoryScopes};
}

llvm::Expected<MaterializedStructuredMemoryCommunicationCandidate>
materializeStructuredMemoryCommunicationDecision(
    const StructuredProgramCandidate &parent,
    const StructuredMemoryCommunicationDecision &decision) {
  auto encoded = encodeStructuredMemoryCommunicationDecision(decision);
  if (!encoded)
    return encoded.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> clone;
  if (const auto *stage = std::get_if<StageConstantGlobalDecision>(&decision)) {
    mlir::BlockArgument input;
    auto resolved = cloneAndResolveMemoryInput(parent, stage->anchor, input);
    if (!resolved)
      return resolved.takeError();
    clone = std::move(*resolved);
    if (llvm::Error error = stageConstantGlobal(input))
      return std::move(error);
  } else if (const auto *layout =
                 std::get_if<PermuteLocalBufferLayoutDecision>(&decision)) {
    mlir::memref::AllocOp alloc;
    auto resolved = cloneAndResolveAllocation(parent, layout->anchor, alloc);
    if (!resolved)
      return resolved.takeError();
    clone = std::move(*resolved);
    if (llvm::Error error = permuteLocalBufferLayout(alloc, *layout))
      return std::move(error);
  } else if (std::holds_alternative<PipelineStagedLoopDecision>(decision)) {
    return invalid("pipeline materialization is not implemented");
  } else {
    return invalid("channel-promotion materialization is not implemented");
  }
  if (mlir::failed(mlir::verify(*clone)))
    return invalid("materialized memory candidate does not verify");
  auto finalized = finalizeStructuredProgramWithTrackedBlocks(clone.get(), {});
  if (!finalized)
    return finalized.takeError();
  return MaterializedStructuredMemoryCommunicationCandidate{
      std::move(finalized->artifact), std::move(finalized->sourceProvenance)};
}

} // namespace loom::frontend

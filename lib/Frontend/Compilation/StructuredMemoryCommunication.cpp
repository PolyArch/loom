#include "Frontend/Compilation/StructuredMemoryCommunication.h"

#include "Frontend/IR/LoomOps.h"

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
#include <optional>
#include <utility>
#include <vector>

namespace loom::frontend {
namespace {

constexpr llvm::StringLiteral decisionSchema =
    "loom.structured_memory_communication.decision.1.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_memory_communication_invalid: " +
                                     message);
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

} // namespace

llvm::ArrayRef<std::uint8_t>
structuredMemoryCommunicationDecisionSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(decisionSchema.data()),
          decisionSchema.size()};
}

llvm::Expected<std::vector<std::uint8_t>>
encodeStructuredMemoryCommunicationDecision(
    const StructuredMemoryCommunicationDecision &decision) {
  if (decision.memoryInput.kind != StructuredEntityKind::Value)
    return invalid("decision does not reference a value");
  if (decision.kind !=
      StructuredMemoryCommunicationDecisionKind::StageConstantGlobal)
    return invalid("decision has an unknown kind");
  std::vector<std::uint8_t> bytes =
      encodeStructuredEntityRef(decision.memoryInput);
  const std::uint32_t kind = static_cast<std::uint32_t>(decision.kind);
  bytes.push_back(static_cast<std::uint8_t>(kind >> 24));
  bytes.push_back(static_cast<std::uint8_t>(kind >> 16));
  bytes.push_back(static_cast<std::uint8_t>(kind >> 8));
  bytes.push_back(static_cast<std::uint8_t>(kind));
  return bytes;
}

llvm::Expected<StructuredMemoryCommunicationDecision>
adoptStructuredMemoryCommunicationDecision(
    llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  constexpr std::size_t wireSize = structuredEntityRefWireSize + 4;
  if (canonicalBytes.size() != wireSize)
    return invalid("decision payload has the wrong size");
  auto memoryInput = decodeStructuredEntityRef(
      canonicalBytes.take_front(structuredEntityRefWireSize));
  if (!memoryInput)
    return memoryInput.takeError();
  if (memoryInput->kind != StructuredEntityKind::Value)
    return invalid("decision does not reference a value");
  llvm::ArrayRef<std::uint8_t> encodedKind =
      canonicalBytes.drop_front(structuredEntityRefWireSize);
  const std::uint32_t kind =
      (static_cast<std::uint32_t>(encodedKind[0]) << 24) |
      (static_cast<std::uint32_t>(encodedKind[1]) << 16) |
      (static_cast<std::uint32_t>(encodedKind[2]) << 8) |
      static_cast<std::uint32_t>(encodedKind[3]);
  if (kind !=
      static_cast<std::uint32_t>(
          StructuredMemoryCommunicationDecisionKind::StageConstantGlobal))
    return invalid("decision payload has an unknown kind");
  StructuredMemoryCommunicationDecision decision{
      *memoryInput,
      static_cast<StructuredMemoryCommunicationDecisionKind>(kind)};
  auto reencoded = encodeStructuredMemoryCommunicationDecision(decision);
  if (!reencoded)
    return reencoded.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*reencoded) != canonicalBytes)
    return invalid("decision payload does not re-encode exactly");
  return decision;
}

llvm::Expected<std::vector<StructuredMemoryCommunicationDecision>>
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
  for (const StructuredEntity &entity :
       view->entities(StructuredEntityKind::Operation)) {
    auto spatial =
        llvm::dyn_cast_or_null<loom::SpatialRegionOp>(entity.operation);
    if (!spatial)
      continue;
    for (std::uint64_t ordinal = 0; ordinal != spatial.getMemoryInputs().size();
         ++ordinal) {
      if (!isStageableConstantInput(spatial, ordinal))
        continue;
      if (decisions.size() == scopeExpansionLimit)
        return decisions;
      mlir::BlockArgument argument = memoryArgument(spatial, ordinal);
      auto found = valueReferences.find(argument);
      if (found == valueReferences.end())
        return invalid("Spatial memory input has no canonical value reference");
      decisions.push_back(
          {found->second,
           StructuredMemoryCommunicationDecisionKind::StageConstantGlobal});
    }
  }
  return decisions;
}

llvm::Expected<MaterializedStructuredMemoryCommunicationCandidate>
materializeStructuredMemoryCommunicationDecision(
    const StructuredProgramCandidate &parent,
    const StructuredMemoryCommunicationDecision &decision) {
  auto encoded = encodeStructuredMemoryCommunicationDecision(decision);
  if (!encoded)
    return encoded.takeError();
  mlir::BlockArgument input;
  auto clone = cloneAndResolveMemoryInput(parent, decision.memoryInput, input);
  if (!clone)
    return clone.takeError();

  switch (decision.kind) {
  case StructuredMemoryCommunicationDecisionKind::StageConstantGlobal:
    if (llvm::Error error = stageConstantGlobal(input))
      return std::move(error);
    break;
  }
  if (mlir::failed(mlir::verify(**clone)))
    return invalid("materialized memory candidate does not verify");
  auto finalized = finalizeStructuredProgramWithTrackedBlocks(clone->get(), {});
  if (!finalized)
    return finalized.takeError();
  return MaterializedStructuredMemoryCommunicationCandidate{
      std::move(finalized->artifact), std::move(finalized->sourceProvenance)};
}

} // namespace loom::frontend

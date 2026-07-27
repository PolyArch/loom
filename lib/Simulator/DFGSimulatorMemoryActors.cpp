//===- DFGSimulatorMemoryActors.cpp - Memory actor semantics --------------===//
//
// One owner for how a memory actor observes and changes logical memory: the
// view it addresses, the active lanes and element slots it resolves, the read
// or write it prepares before anything commits, the action it projects for
// admission, and the effect it issues on commit. The load and store path
// projects, prepares, and issues from peeked inputs alone, so a rejection
// anywhere along it is atomic: no input is consumed, nothing is published,
// and memory is unchanged.
//
// MemorySynchronization remains the sole authority over the order between two
// effects; this module only declares them.
//
//===----------------------------------------------------------------------===//

#include "DFGSimulatorInternal.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
#include "Dataflow/IR/DataflowOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

// The memory an access addresses, read without consuming a channel-backed
// view token, so a rejected access leaves that operand queued like every
// other input.
static std::optional<MemoryView>
peekMemoryView(SimulatorState &state, mlir::Value mem,
               mlir::OpOperand &memOperand,
               llvm::SmallVectorImpl<std::string> &diagnostics) {
  if (hasToken(state.channels, memOperand)) {
    Token token = peekToken(state.channels, memOperand);
    if (token.kind != TokenKind::Pointer || !token.pointer.memory) {
      diagnostics.push_back("dataflow memory operand is not a memory view");
      return std::nullopt;
    }
    return token.pointer;
  }
  auto viewIt = state.memoryViews.find(mem);
  if (viewIt != state.memoryViews.end())
    return viewIt->second;
  auto memIt = state.memories.find(mem);
  if (memIt != state.memories.end())
    return MemoryView{memIt->second, mem, 0};
  return std::nullopt;
}

static void consumeMemoryView(SimulatorState &state,
                              mlir::OpOperand &memOperand) {
  if (hasToken(state.channels, memOperand))
    (void)popToken(state, memOperand);
}

static std::optional<std::size_t>
resolveElementIndex(const MemoryView &view, const llvm::APInt &address,
                    llvm::SmallVectorImpl<std::string> &diagnostics,
                    mlir::Operation *scope, llvm::StringRef diagnosticLabel) {
  auto elementSizeOrErr = byteSizeOfType(view.memory->elementType, scope);
  if (!elementSizeOrErr) {
    diagnostics.push_back(llvm::toString(elementSizeOrErr.takeError()));
    return std::nullopt;
  }
  if (*elementSizeOrErr == 0 || view.byteOffset % *elementSizeOrErr != 0) {
    diagnostics.push_back("memory view byte offset is not element-aligned");
    return std::nullopt;
  }
  const std::int64_t baseIndex = view.byteOffset / *elementSizeOrErr;
  // The semantic address is signed at its own width. Widening both terms past
  // that width keeps the sum exact, so a host slot is named only after the
  // sign and bound checks below prove it exists.
  const unsigned width = std::max(address.getBitWidth(), 64u) + 1;
  const llvm::APInt slot =
      address.sext(width) + llvm::APInt(width, baseIndex, /*isSigned=*/true);
  const llvm::APInt limit(width, view.memory->elements.size());
  if (slot.isNegative() || slot.uge(limit)) {
    diagnostics.push_back((diagnosticLabel + " address is out of range").str());
    return std::nullopt;
  }
  return static_cast<std::size_t>(slot.getZExtValue());
}

std::optional<std::size_t>
resolveElementIndex(const MemoryView &view, const llvm::APInt &address,
                    SimulatorState &state, mlir::Operation *scope,
                    llvm::StringRef diagnosticLabel) {
  return resolveElementIndex(view, address, state.diagnostics, scope,
                             diagnosticLabel);
}

std::optional<std::size_t>
resolveElementIndex(const MemoryView &view, const Token &addr,
                    SimulatorState &state, mlir::Operation *scope,
                    llvm::StringRef diagnosticLabel) {
  return resolveElementIndex(
      view,
      llvm::APInt(64, static_cast<std::uint64_t>(integerToken(addr)),
                  /*isSigned=*/true),
      state, scope, diagnosticLabel);
}

std::optional<Token> readMemoryElement(const MemoryView &view,
                                       std::size_t index, SimulatorState &state,
                                       llvm::StringRef diagnosticLabel) {
  if (!view.memory->initialized[index]) {
    state.diagnostics.push_back(
        (diagnosticLabel + " reads uninitialized memory").str());
    return std::nullopt;
  }
  return view.memory->elements[index];
}

void writeMemoryElement(const MemoryView &view, std::size_t index,
                        Token value) {
  view.memory->elements[index] = value;
  view.memory->initialized.set(index);
}

static std::optional<llvm::APInt>
getActiveMemoryLanes(const dataflow::semantics::MemoryAccessType &access,
                     const Token *mask, mlir::Type maskType,
                     llvm::SmallVectorImpl<std::string> &diagnostics) {
  if (access.laneCount() > std::numeric_limits<unsigned>::max()) {
    diagnostics.push_back(
        "vector lane count exceeds the simulator bit-vector limit");
    return std::nullopt;
  }
  const unsigned lanes = static_cast<unsigned>(access.laneCount());
  if (!mask)
    return llvm::APInt::getAllOnes(lanes);
  auto maskBits = tokenBitPattern(*mask, maskType);
  if (!maskBits) {
    diagnostics.push_back(llvm::toString(maskBits.takeError()));
    return std::nullopt;
  }
  return *maskBits;
}

// Resolves the address of every active lane before any element is read or
// written, so a firing that cannot complete leaves memory untouched. A
// contiguous access adds the lane ordinal at the declared index width, exactly
// as an explicit index-typed `arith.addi` would; an indexed access takes each
// lane address from its own address-vector slice. Inactive lanes are skipped
// entirely, so their addresses are never evaluated.
static std::optional<llvm::SmallVector<std::size_t>>
resolveActiveLaneSlots(const MemoryView &view, const Token &addr,
                       const dataflow::semantics::MemoryAccessType &access,
                       const llvm::APInt &activeLanes,
                       llvm::SmallVectorImpl<std::string> &diagnostics,
                       mlir::Operation *scope,
                       llvm::StringRef diagnosticLabel) {
  // The index width is structural: every access has one whether or not a lane
  // is active, so it is resolved before the inactive shortcut. No address is
  // parsed or evaluated for an inactive lane.
  auto width = loom::getIndexBitWidth(scope);
  if (!width) {
    diagnostics.push_back(llvm::toString(width.takeError()));
    return std::nullopt;
  }
  llvm::SmallVector<std::size_t> slots;
  if (activeLanes.isZero())
    return slots;

  llvm::APInt base;
  std::optional<llvm::APInt> addressBits;
  if (access.isGather()) {
    auto bits =
        vectorIndexTokenBitPattern(addr, access.addressVectorType, scope);
    if (!bits) {
      diagnostics.push_back(llvm::toString(bits.takeError()));
      return std::nullopt;
    }
    addressBits = *bits;
  } else {
    auto bits = indexTokenBitPattern(addr, *width);
    if (!bits) {
      diagnostics.push_back(llvm::toString(bits.takeError()));
      return std::nullopt;
    }
    base = *bits;
  }

  slots.reserve(activeLanes.popcount());
  for (unsigned lane = 0; lane < access.laneCount(); ++lane) {
    if (!activeLanes[lane])
      continue;
    llvm::APInt laneAddress =
        access.isGather() ? addressBits->extractBits(*width, *width * lane)
                          : base + llvm::APInt(*width, lane, /*isSigned=*/false,
                                               /*implicitTrunc=*/true);
    auto slot = resolveElementIndex(view, laneAddress, diagnostics, scope,
                                    diagnosticLabel);
    if (!slot)
      return std::nullopt;
    slots.push_back(*slot);
  }
  return slots;
}

struct ProjectedDataflowMemoryAccess {
  dataflow::semantics::MemoryAccessType type;
  llvm::APInt activeLanes;
  llvm::SmallVector<std::size_t> slots;
};

static std::optional<ProjectedDataflowMemoryAccess> projectDataflowMemoryAccess(
    const MemoryView &view, const Token &addr, mlir::MemRefType memoryType,
    mlir::Type addressType, mlir::Type dataType, const Token *mask,
    mlir::Type maskType, llvm::SmallVectorImpl<std::string> &diagnostics,
    mlir::Operation *scope, llvm::StringRef diagnosticLabel) {
  auto access = dataflow::semantics::analyzeMemoryAccessType(
      memoryType, dataType, addressType, maskType);
  if (!access) {
    diagnostics.push_back(llvm::toString(access.takeError()));
    return std::nullopt;
  }
  const bool vectorAccess = access->isVector();
  auto activeLanes =
      getActiveMemoryLanes(*access, vectorAccess ? mask : nullptr,
                           vectorAccess ? maskType : mlir::Type{}, diagnostics);
  auto slots = activeLanes
                   ? resolveActiveLaneSlots(view, addr, *access, *activeLanes,
                                            diagnostics, scope, diagnosticLabel)
                   : std::nullopt;
  if (!activeLanes || !slots)
    return std::nullopt;
  return ProjectedDataflowMemoryAccess{
      std::move(*access), std::move(*activeLanes), std::move(*slots)};
}

static std::optional<DataflowMemoryRead>
prepareDataflowMemoryRead(const MemoryView &view,
                          const ProjectedDataflowMemoryAccess &projection,
                          SimulatorState &state) {
  const auto &access = projection.type;
  // An element access is one complete memref element, even when that element
  // is itself a vector: one lane, one address, and no mask.
  if (!access.isVector()) {
    auto value = readMemoryElement(view, projection.slots.front(), state,
                                   "dataflow.load");
    if (!value)
      return std::nullopt;
    return DataflowMemoryRead{*value, true};
  }

  const llvm::APInt &activeLanes = projection.activeLanes;
  auto elementWidth = tokenTypeBitWidth(access.elementType);
  auto vectorWidth = tokenTypeBitWidth(access.vectorType);
  if (!elementWidth || !vectorWidth) {
    if (!elementWidth)
      state.diagnostics.push_back(llvm::toString(elementWidth.takeError()));
    if (!vectorWidth)
      state.diagnostics.push_back(llvm::toString(vectorWidth.takeError()));
    return std::nullopt;
  }

  // Inactive lanes keep the element type's all-zero bit representation.
  llvm::APInt resultBits(*vectorWidth, 0);
  unsigned active = 0;
  for (unsigned lane = 0; lane < access.laneCount(); ++lane) {
    if (!activeLanes[lane])
      continue;
    auto element = readMemoryElement(view, projection.slots[active++], state,
                                     "dataflow.load");
    if (!element)
      return std::nullopt;
    auto elementBits = tokenBitPattern(*element, access.elementType);
    if (!elementBits) {
      state.diagnostics.push_back(llvm::toString(elementBits.takeError()));
      return std::nullopt;
    }
    resultBits.insertBits(*elementBits, *elementWidth * lane);
  }

  auto result = tokenFromBitPattern(resultBits, access.vectorType);
  if (!result) {
    state.diagnostics.push_back(llvm::toString(result.takeError()));
    return std::nullopt;
  }
  return DataflowMemoryRead{*result, !activeLanes.isZero()};
}

static std::optional<DataflowMemoryWrite>
prepareDataflowMemoryWrite(const Token &data,
                           const ProjectedDataflowMemoryAccess &projection,
                           SimulatorState &state) {
  const auto &access = projection.type;
  if (!access.isVector()) {
    return DataflowMemoryWrite{{{projection.slots.front(), data}}, true};
  }

  const llvm::APInt &activeLanes = projection.activeLanes;
  if (activeLanes.isZero())
    return DataflowMemoryWrite{};

  auto elementWidth = tokenTypeBitWidth(access.elementType);
  auto dataBits = tokenBitPattern(data, access.vectorType);
  if (!elementWidth || !dataBits) {
    if (!elementWidth)
      state.diagnostics.push_back(llvm::toString(elementWidth.takeError()));
    if (!dataBits)
      state.diagnostics.push_back(llvm::toString(dataBits.takeError()));
    return std::nullopt;
  }

  // A plain scatter has no lane order for duplicate active destinations, so a
  // finalized actor is only legal because its program already proved them
  // distinct or lowered the access to an explicit program order. Distinctness
  // is that program's invariant, not a scheduler decision: admission neither
  // inspects nor guesses it. A finalized firing that still resolves duplicates
  // has therefore broken the invariant its provider guarantees, which fails
  // the run rather than reporting a capability the model lacks. The refusal is
  // atomic: no input is consumed, no memory changes, and nothing is published.
  if (access.isGather()) {
    llvm::SmallDenseSet<std::size_t, 8> destinations;
    for (std::size_t slot : projection.slots) {
      if (destinations.insert(slot).second)
        continue;
      state.diagnostics.push_back(
          "finalized plain dataflow.store resolved duplicate active addresses");
      state.failure = RunFailure::ProviderInvariant;
      return std::nullopt;
    }
  }

  DataflowMemoryWrite write;
  write.accessedMemory = true;
  write.elements.reserve(projection.slots.size());
  unsigned active = 0;
  for (unsigned lane = 0; lane < access.laneCount(); ++lane) {
    if (!activeLanes[lane])
      continue;
    llvm::APInt elementBits =
        dataBits->extractBits(*elementWidth, *elementWidth * lane);
    auto element = tokenFromBitPattern(elementBits, access.elementType);
    if (!element) {
      state.diagnostics.push_back(llvm::toString(element.takeError()));
      return std::nullopt;
    }
    write.elements.emplace_back(projection.slots[active++], *element);
  }
  return write;
}

void commitDataflowMemoryWrite(const MemoryView &view,
                               const DataflowMemoryWrite &write) {
  for (const auto &[index, value] : write.elements)
    writeMemoryElement(view, index, value);
}

// The byte ranges one issued access covers, derived from the active element
// slots it already resolved. An inactive lane resolves no slot, so it
// contributes no range and derives no access.
static std::optional<MemoryActionRecord> projectMemoryAction(
    const MemoryView &view, llvm::ArrayRef<std::size_t> slots, bool isWrite,
    llvm::SmallVectorImpl<std::string> &diagnostics, mlir::Operation *scope) {
  MemoryActionRecord action;
  action.rootId = view.memory->logicalRootId;
  action.isWrite = isWrite;
  if (slots.empty())
    return action;
  auto elementSize = byteSizeOfType(view.memory->elementType, scope);
  if (!elementSize) {
    diagnostics.push_back(llvm::toString(elementSize.takeError()));
    return std::nullopt;
  }
  action.byteRanges.reserve(slots.size());
  for (std::size_t slot : slots) {
    const std::int64_t begin = static_cast<std::int64_t>(slot) * *elementSize;
    action.byteRanges.emplace_back(begin, begin + *elementSize);
  }
  canonicalizeMemoryActionRanges(action.byteRanges);
  return action;
}

static MemorySynchronization &memorySynchronization(SimulatorState &state) {
  if (!state.memorySync) {
    state.memoryOrder = std::make_unique<MemoryAtomicOrder>();
    state.memorySync =
        std::make_unique<MemorySynchronization>(*state.memoryOrder);
  }
  return *state.memorySync;
}

// Memory order enters an access only through its explicit ctrl token. Ordinary
// memory, address, data, and mask dependencies remain actor operands but do not
// create sequenced-before facts.
static llvm::SmallVector<SyncEffectId, 2>
peekMemoryOrderFrontier(SimulatorState &state, mlir::OpOperand &ctrl) {
  llvm::SmallVector<SyncEffectId, 2> frontier;
  if (hasToken(state.channels, ctrl))
    frontier.assign(state.memoryOrderFrontiers.elements(
        peekToken(state.channels, ctrl).memoryOrder));
  return frontier;
}

// Empty accesses pass through only ctrl order; others publish a new effect.
static std::optional<MemoryOrderFrontierId>
issueMemoryAction(const MemoryActionRecord &action,
                  llvm::ArrayRef<SyncEffectId> orderFrontier,
                  SimulatorState &state) {
  if (action.byteRanges.empty()) {
    // The admitted ctrl frontier is already canonical and reduced.
    return state.memoryOrderFrontiers.internCanonical(orderFrontier);
  }
  MemorySynchronization &sync = memorySynchronization(state);
  auto effect = sync.declareEffectSequencedAfter(orderFrontier);
  if (!effect) {
    // The admitted frontier is canonical output of this run's own
    // bookkeeping, so the authority rejecting it is a provider invariant
    // violation, not a missing capability. The rejection is atomic: no input
    // is consumed, no action is retained, and no frontier is published.
    state.diagnostics.push_back(llvm::toString(effect.takeError()));
    state.failure = RunFailure::ProviderInvariant;
    return std::nullopt;
  }
  state.memoryActions.retain(action, *effect, sync);
  return state.memoryOrderFrontiers.internCanonical(*effect);
}

static mlir::OpOperand *getOptionalMaskOperand(mlir::Operation *op,
                                               mlir::Value mask) {
  if (!mask)
    return nullptr;
  return &op->getOpOperand(op->getNumOperands() - 1);
}

struct PreparedLoadFiring {
  MemoryView view;
  DataflowMemoryRead read;
  mlir::OpOperand *maskOperand = nullptr;
};

struct ProjectedMemoryFiring {
  MemoryView view;
  ProjectedDataflowMemoryAccess access;
  mlir::OpOperand *maskOperand = nullptr;
};

static std::optional<ProjectedMemoryFiring>
projectLoadFiring(dataflow::LoadOp op, SimulatorState &state,
                  llvm::SmallVectorImpl<std::string> &diagnostics) {
  mlir::OpOperand *maskOperand =
      getOptionalMaskOperand(op.getOperation(), op.getMask());
  if (!hasToken(state.channels, op.getAddrMutable()) ||
      !hasToken(state.channels, op.getCtrlMutable()) ||
      (maskOperand && !hasToken(state.channels, *maskOperand)))
    return std::nullopt;
  std::optional<MemoryView> view =
      peekMemoryView(state, op.getMem(), op.getMemMutable(), diagnostics);
  if (!view)
    return std::nullopt;
  Token addr = peekToken(state.channels, op.getAddrMutable());
  std::optional<Token> mask;
  if (maskOperand)
    mask = peekToken(state.channels, *maskOperand);
  auto access = projectDataflowMemoryAccess(
      *view, addr, mlir::cast<mlir::MemRefType>(op.getMem().getType()),
      op.getAddr().getType(), op.getData().getType(), mask ? &*mask : nullptr,
      op.getMask() ? op.getMask().getType() : mlir::Type{}, diagnostics,
      op.getOperation(), "dataflow.load");
  if (!access)
    return std::nullopt;
  return ProjectedMemoryFiring{std::move(*view), std::move(*access),
                               maskOperand};
}

static std::optional<PreparedLoadFiring>
prepareLoadFiring(dataflow::LoadOp op, SimulatorState &state) {
  auto projected = projectLoadFiring(op, state, state.diagnostics);
  if (!projected)
    return std::nullopt;
  auto read =
      prepareDataflowMemoryRead(projected->view, projected->access, state);
  if (!read)
    return std::nullopt;
  return PreparedLoadFiring{std::move(projected->view), std::move(*read),
                            projected->maskOperand};
}

struct PreparedStoreFiring {
  MemoryView view;
  DataflowMemoryWrite write;
  mlir::OpOperand *maskOperand = nullptr;
};

static std::optional<ProjectedMemoryFiring>
projectStoreFiring(dataflow::StoreOp op, SimulatorState &state,
                   llvm::SmallVectorImpl<std::string> &diagnostics) {
  mlir::OpOperand *maskOperand =
      getOptionalMaskOperand(op.getOperation(), op.getMask());
  if (!hasToken(state.channels, op.getAddrMutable()) ||
      !hasToken(state.channels, op.getDataMutable()) ||
      !hasToken(state.channels, op.getCtrlMutable()) ||
      (maskOperand && !hasToken(state.channels, *maskOperand)))
    return std::nullopt;
  std::optional<MemoryView> view =
      peekMemoryView(state, op.getMem(), op.getMemMutable(), diagnostics);
  if (!view)
    return std::nullopt;
  Token addr = peekToken(state.channels, op.getAddrMutable());
  std::optional<Token> mask;
  if (maskOperand)
    mask = peekToken(state.channels, *maskOperand);
  auto access = projectDataflowMemoryAccess(
      *view, addr, mlir::cast<mlir::MemRefType>(op.getMem().getType()),
      op.getAddr().getType(), op.getData().getType(), mask ? &*mask : nullptr,
      op.getMask() ? op.getMask().getType() : mlir::Type{}, diagnostics,
      op.getOperation(), "dataflow.store");
  if (!access)
    return std::nullopt;
  return ProjectedMemoryFiring{std::move(*view), std::move(*access),
                               maskOperand};
}

static std::optional<PreparedStoreFiring>
prepareStoreFiring(dataflow::StoreOp op, SimulatorState &state) {
  auto projected = projectStoreFiring(op, state, state.diagnostics);
  if (!projected)
    return std::nullopt;
  Token data = peekToken(state.channels, op.getDataMutable());
  auto write = prepareDataflowMemoryWrite(data, projected->access, state);
  if (!write)
    return std::nullopt;
  return PreparedStoreFiring{std::move(projected->view), std::move(*write),
                             projected->maskOperand};
}

PlainMemoryActionProjection
projectReadyPlainMemoryAction(mlir::Operation *operation,
                              SimulatorState &state) {
  PlainMemoryActionProjection result;
  if (auto op = mlir::dyn_cast<dataflow::LoadOp>(operation)) {
    auto projected = projectLoadFiring(op, state, result.diagnostics);
    if (!projected)
      return result;
    auto action =
        projectMemoryAction(projected->view, projected->access.slots,
                            /*isWrite=*/false, result.diagnostics, operation);
    if (!action)
      return result;
    result.ready = ReadyPlainMemoryAction{
        std::move(*action),
        peekMemoryOrderFrontier(state, op.getCtrlMutable())};
    return result;
  }

  auto op = mlir::dyn_cast<dataflow::StoreOp>(operation);
  if (!op)
    return result;
  auto projected = projectStoreFiring(op, state, result.diagnostics);
  if (!projected)
    return result;
  auto action =
      projectMemoryAction(projected->view, projected->access.slots,
                          /*isWrite=*/true, result.diagnostics, operation);
  if (!action)
    return result;
  result.ready = ReadyPlainMemoryAction{
      std::move(*action), peekMemoryOrderFrontier(state, op.getCtrlMutable())};
  return result;
}

bool fireLoad(dataflow::LoadOp op, SimulatorState &state) {
  auto admitted = state.admittedPlainMemoryActions.find(op.getOperation());
  if (admitted == state.admittedPlainMemoryActions.end())
    return false;
  auto prepared = prepareLoadFiring(op, state);
  if (!prepared)
    return false;
  auto publication = issueMemoryAction(admitted->second.action,
                                       admitted->second.ctrlFrontier, state);
  if (!publication)
    return false;
  state.admittedPlainMemoryActions.erase(op.getOperation());

  consumeMemoryView(state, op.getMemMutable());
  popToken(state, op.getAddrMutable());
  popToken(state, op.getCtrlMutable());
  if (prepared->maskOperand)
    popToken(state, *prepared->maskOperand);
  emitTokenWithMemoryOrder(state, op.getData(), prepared->read.data,
                           MemoryOrderFrontierId());
  emitTokenWithMemoryOrder(state, op.getDone(), noneToken(), *publication);
  return recordActorEvent(state, op.getOperation());
}

bool fireStore(dataflow::StoreOp op, SimulatorState &state) {
  auto admitted = state.admittedPlainMemoryActions.find(op.getOperation());
  if (admitted == state.admittedPlainMemoryActions.end())
    return false;
  auto prepared = prepareStoreFiring(op, state);
  if (!prepared)
    return false;
  auto publication = issueMemoryAction(admitted->second.action,
                                       admitted->second.ctrlFrontier, state);
  if (!publication)
    return false;
  state.admittedPlainMemoryActions.erase(op.getOperation());

  consumeMemoryView(state, op.getMemMutable());
  popToken(state, op.getAddrMutable());
  popToken(state, op.getDataMutable());
  popToken(state, op.getCtrlMutable());
  if (prepared->maskOperand)
    popToken(state, *prepared->maskOperand);
  commitDataflowMemoryWrite(prepared->view, prepared->write);
  emitTokenWithMemoryOrder(state, op.getDone(), noneToken(), *publication);
  return recordActorEvent(state, op.getOperation());
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

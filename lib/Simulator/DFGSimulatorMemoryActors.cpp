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
#include "mlir/Interfaces/DataLayoutInterfaces.h"
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
               unsigned memoryOperandOrdinal,
               llvm::SmallVectorImpl<std::string> &diagnostics) {
  if (hasInputToken(state, memoryOperandOrdinal)) {
    Token token = peekInputToken(state, memoryOperandOrdinal);
    const MemoryView *view = token.memoryView();
    if (token.kind != TokenKind::Pointer || !view || !view->memory) {
      diagnostics.push_back("dataflow memory operand is not a memory view");
      return std::nullopt;
    }
    return *view;
  }
  auto viewIt = state.memoryViews.find(mem);
  if (viewIt != state.memoryViews.end())
    return viewIt->second;
  auto memIt = state.memories.find(mem);
  if (memIt != state.memories.end()) {
    mlir::Type elementType;
    if (auto type = mlir::dyn_cast<mlir::MemRefType>(mem.getType()))
      elementType = type.getElementType();
    return MemoryView{memIt->second, mem, 0, elementType};
  }
  return std::nullopt;
}

static void consumeMemoryView(SimulatorState &state,
                              unsigned memoryOperandOrdinal) {
  if (hasInputToken(state, memoryOperandOrdinal))
    (void)popInputToken(state, memoryOperandOrdinal);
}

static llvm::Expected<MemoryByteOrder> memoryByteOrder(mlir::Operation *scope) {
  mlir::Attribute endianness = mlir::DataLayout::closest(scope).getEndianness();
  if (!endianness)
    return MemoryByteOrder::Little;
  auto spelling = mlir::dyn_cast<mlir::StringAttr>(endianness);
  if (!spelling)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "memory endianness is not a string");
  if (spelling.getValue() == "little")
    return MemoryByteOrder::Little;
  if (spelling.getValue() == "big")
    return MemoryByteOrder::Big;
  return llvm::createStringError(std::errc::invalid_argument,
                                 "unsupported memory endianness '%s'",
                                 spelling.getValue().str().c_str());
}

llvm::Expected<ResolvedMemoryElementLayout>
resolveMemoryElementLayout(mlir::Type type, mlir::Operation *scope) {
  auto byteCount = byteSizeOfType(type, scope);
  if (!byteCount)
    return byteCount.takeError();
  if (*byteCount <= 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "memory element has zero storage size");
  auto bitWidth = resolvedTokenTypeBitWidth(type, scope);
  if (!bitWidth)
    return bitWidth.takeError();
  auto order = memoryByteOrder(scope);
  if (!order)
    return order.takeError();
  return ResolvedMemoryElementLayout{static_cast<std::size_t>(*byteCount),
                                     *bitWidth, *order};
}

llvm::Expected<MemoryActorExecutionPlan>
memoryActorExecutionPlan(mlir::Operation *operation,
                         mlir::Operation *graphScope) {
  mlir::Value memory;
  mlir::Value address;
  mlir::Type dataType;
  mlir::Value mask;
  unsigned memoryOperandOrdinal = 0;
  unsigned addressOperandOrdinal = 0;
  std::optional<unsigned> dataOperandOrdinal;
  unsigned controlOperandOrdinal = 0;
  std::optional<unsigned> maskOperandOrdinal;
  if (auto load = mlir::dyn_cast<dataflow::LoadOp>(operation)) {
    memory = load.getMem();
    address = load.getAddr();
    dataType = load.getData().getType();
    mask = load.getMask();
    memoryOperandOrdinal = load.getMemMutable().getOperandNumber();
    addressOperandOrdinal = load.getAddrMutable().getOperandNumber();
    controlOperandOrdinal = load.getCtrlMutable().getOperandNumber();
  } else if (auto store = mlir::dyn_cast<dataflow::StoreOp>(operation)) {
    memory = store.getMem();
    address = store.getAddr();
    dataType = store.getData().getType();
    mask = store.getMask();
    memoryOperandOrdinal = store.getMemMutable().getOperandNumber();
    addressOperandOrdinal = store.getAddrMutable().getOperandNumber();
    dataOperandOrdinal = store.getDataMutable().getOperandNumber();
    controlOperandOrdinal = store.getCtrlMutable().getOperandNumber();
  } else {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "memory execution plan requires dataflow.load or dataflow.store");
  }
  if (mask)
    maskOperandOrdinal = operation->getNumOperands() - 1;

  auto access = dataflow::semantics::analyzeMemoryAccessType(
      mlir::cast<mlir::MemRefType>(memory.getType()), dataType,
      address.getType(), mask ? mask.getType() : mlir::Type{});
  if (!access)
    return access.takeError();
  auto indexWidth = loom::getIndexBitWidth(graphScope);
  if (!indexWidth)
    return indexWidth.takeError();
  auto elementLayout =
      resolveMemoryElementLayout(access->elementType, graphScope);
  if (!elementLayout)
    return elementLayout.takeError();
  auto addressWidth = resolvedTokenTypeBitWidth(address.getType(), graphScope);
  if (!addressWidth)
    return addressWidth.takeError();
  auto dataWidth = resolvedTokenTypeBitWidth(dataType, graphScope);
  if (!dataWidth)
    return dataWidth.takeError();
  return MemoryActorExecutionPlan{
      std::move(*access), memoryOperandOrdinal,  addressOperandOrdinal,
      dataOperandOrdinal, controlOperandOrdinal, maskOperandOrdinal,
      *indexWidth,        *addressWidth,         *dataWidth,
      *elementLayout};
}

static std::optional<std::size_t>
resolveElementByteOffset(const MemoryView &view, const llvm::APInt &address,
                         std::size_t elementByteCount,
                         llvm::SmallVectorImpl<std::string> &diagnostics,
                         llvm::StringRef diagnosticLabel) {
  if (elementByteCount == 0) {
    diagnostics.push_back("memory element has zero storage size");
    return std::nullopt;
  }
  if (view.byteOffset < 0 ||
      view.byteOffset % static_cast<std::int64_t>(elementByteCount) != 0) {
    diagnostics.push_back("memory view byte offset is not element-aligned");
    return std::nullopt;
  }
  // The semantic address is signed at its own width. Convert it to an exact
  // byte offset before any host-size projection.
  const unsigned width = std::max(address.getBitWidth(), 64u) + 1;
  const llvm::APInt byteOffset =
      address.sext(width) *
          llvm::APInt(width, static_cast<std::uint64_t>(elementByteCount)) +
      llvm::APInt(width, static_cast<std::uint64_t>(view.byteOffset));
  const llvm::APInt end =
      byteOffset +
      llvm::APInt(width, static_cast<std::uint64_t>(elementByteCount));
  const llvm::APInt limit(width, view.memory->bytes.size());
  if (byteOffset.isNegative() || end.ugt(limit)) {
    diagnostics.push_back((diagnosticLabel + " address is out of range").str());
    return std::nullopt;
  }
  return static_cast<std::size_t>(byteOffset.getZExtValue());
}

std::optional<std::size_t>
resolveElementByteOffset(const MemoryView &view, const llvm::APInt &address,
                         mlir::Type elementType, SimulatorState &state,
                         mlir::Operation *scope,
                         llvm::StringRef diagnosticLabel) {
  auto elementByteCount = byteSizeOfType(elementType, scope);
  if (!elementByteCount) {
    state.diagnostics.push_back(llvm::toString(elementByteCount.takeError()));
    return std::nullopt;
  }
  return resolveElementByteOffset(view, address,
                                  static_cast<std::size_t>(*elementByteCount),
                                  state.diagnostics, diagnosticLabel);
}

std::optional<std::size_t>
resolveElementByteOffset(const MemoryView &view, const Token &addr,
                         mlir::Type elementType, SimulatorState &state,
                         mlir::Operation *scope,
                         llvm::StringRef diagnosticLabel) {
  return resolveElementByteOffset(
      view,
      llvm::APInt(64, static_cast<std::uint64_t>(integerToken(addr)),
                  /*isSigned=*/true),
      elementType, state, scope, diagnosticLabel);
}

static llvm::Expected<Token> tokenFromMemoryBits(const llvm::APInt &bits,
                                                 mlir::Type type) {
  if (mlir::isa<mlir::IndexType>(type))
    return indexToken(bits);
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
    if (mlir::isa<mlir::IndexType>(vector.getElementType())) {
      Token token;
      token.kind = TokenKind::Vector;
      token.setExactBitPattern(bits);
      return token;
    }
  }
  return tokenFromBitPattern(bits, type);
}

static llvm::Expected<llvm::APInt>
memoryTokenBits(const Token &token, mlir::Type type, unsigned bitWidth) {
  if (mlir::isa<mlir::IndexType>(type))
    return indexTokenBitPattern(token, bitWidth);
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(type)) {
    if (mlir::isa<mlir::IndexType>(vector.getElementType())) {
      if (token.valueState != PrimitiveValueState::Defined ||
          token.kind != TokenKind::Vector || !token.hasExactBitPattern() ||
          token.exactBitWidth() != bitWidth)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "index-vector memory token does not match its resolved width");
      return token.exactBitPattern();
    }
  }
  return tokenBitPattern(token, type);
}

std::optional<Token> readMemoryElementResolved(
    const MemoryView &view, std::size_t byteOffset, mlir::Type elementType,
    const ResolvedMemoryElementLayout &layout, SimulatorState &state,
    llvm::StringRef diagnosticLabel) {
  const std::size_t byteCount = layout.byteCount;
  const unsigned bitWidth = layout.bitWidth;
  const MemoryByteOrder order = layout.byteOrder;
  bool poison = false;
  bool undef = false;
  llvm::APInt bits(bitWidth, 0);
  for (std::size_t index = 0; index < byteCount; ++index) {
    const std::size_t position = byteOffset + index;
    if (!view.memory->initialized[position]) {
      state.diagnostics.push_back(
          (diagnosticLabel + " reads uninitialized memory").str());
      return std::nullopt;
    }
    const SemanticMemoryByte &byte = view.memory->bytes[position];
    poison |= byte.state == SemanticState::Poison;
    undef |= byte.state == SemanticState::Undef;
    if (byte.state != SemanticState::Defined)
      continue;
    const std::size_t semanticByte =
        order == MemoryByteOrder::Little ? index : byteCount - 1 - index;
    const unsigned low = static_cast<unsigned>(semanticByte * 8);
    if (low >= bitWidth)
      continue;
    const unsigned width = std::min(8u, bitWidth - low);
    bits.insertBits(llvm::APInt(width, byte.value), low);
  }
  if (poison || undef) {
    auto token = exceptionalValueToken(poison ? PrimitiveValueState::Poison
                                              : PrimitiveValueState::Undef,
                                       elementType);
    if (!token) {
      state.diagnostics.push_back(llvm::toString(token.takeError()));
      return std::nullopt;
    }
    return *token;
  }
  auto token = tokenFromMemoryBits(bits, elementType);
  if (!token) {
    state.diagnostics.push_back(llvm::toString(token.takeError()));
    return std::nullopt;
  }
  return *token;
}

std::optional<Token>
readMemoryElement(const MemoryView &view, std::size_t byteOffset,
                  mlir::Type elementType, SimulatorState &state,
                  mlir::Operation *scope, llvm::StringRef diagnosticLabel) {
  auto layout = resolveMemoryElementLayout(elementType, scope);
  if (!layout) {
    state.diagnostics.push_back(llvm::toString(layout.takeError()));
    return std::nullopt;
  }
  return readMemoryElementResolved(view, byteOffset, elementType, *layout,
                                   state, diagnosticLabel);
}

static llvm::Expected<llvm::SmallVector<SemanticMemoryByte, 8>>
encodeMemoryElementResolved(const Token &value, mlir::Type elementType,
                            std::size_t byteCount, unsigned bitWidth,
                            MemoryByteOrder order) {
  llvm::SmallVector<SemanticMemoryByte, 8> bytes(byteCount);
  if (value.valueState != PrimitiveValueState::Defined) {
    const SemanticState state = value.valueState == PrimitiveValueState::Poison
                                    ? SemanticState::Poison
                                    : SemanticState::Undef;
    std::fill(bytes.begin(), bytes.end(), SemanticMemoryByte{state, 0});
    return bytes;
  }
  auto bits = memoryTokenBits(value, elementType, bitWidth);
  if (!bits)
    return bits.takeError();
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    const std::size_t semanticByte =
        order == MemoryByteOrder::Little ? index : bytes.size() - 1 - index;
    const unsigned low = static_cast<unsigned>(semanticByte * 8);
    const unsigned width = low >= bitWidth ? 0 : std::min(8u, bitWidth - low);
    bytes[index] = SemanticMemoryByte{
        SemanticState::Defined,
        width == 0 ? std::uint8_t{0}
                   : static_cast<std::uint8_t>(
                         bits->extractBitsAsZExtValue(width, low))};
  }
  return bytes;
}

llvm::Expected<llvm::SmallVector<SemanticMemoryByte, 8>>
encodeMemoryElement(const Token &value, mlir::Type elementType,
                    mlir::Operation *scope) {
  auto byteCount = byteSizeOfType(elementType, scope);
  if (!byteCount)
    return byteCount.takeError();
  auto bitWidth = resolvedTokenTypeBitWidth(elementType, scope);
  if (!bitWidth)
    return bitWidth.takeError();
  auto order = memoryByteOrder(scope);
  if (!order)
    return order.takeError();
  return encodeMemoryElementResolved(value, elementType,
                                     static_cast<std::size_t>(*byteCount),
                                     *bitWidth, *order);
}

void writeMemoryElement(const MemoryView &view, std::size_t byteOffset,
                        llvm::ArrayRef<SemanticMemoryByte> bytes) {
  std::copy(bytes.begin(), bytes.end(),
            view.memory->bytes.begin() + byteOffset);
  view.memory->initialized.set(byteOffset, byteOffset + bytes.size());
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
                       const MemoryActorExecutionPlan &plan,
                       const llvm::APInt &activeLanes,
                       llvm::SmallVectorImpl<std::string> &diagnostics,
                       llvm::StringRef diagnosticLabel) {
  // The index width is structural: every access has one whether or not a lane
  // is active, so it is resolved before the inactive shortcut. No address is
  // parsed or evaluated for an inactive lane.
  const auto &access = plan.access;
  const unsigned width = plan.indexBitWidth;
  llvm::SmallVector<std::size_t> slots;
  if (activeLanes.isZero())
    return slots;

  llvm::APInt base;
  std::optional<llvm::APInt> addressBits;
  if (access.isGather()) {
    auto bits =
        memoryTokenBits(addr, access.addressVectorType, plan.addressBitWidth);
    if (!bits) {
      diagnostics.push_back(llvm::toString(bits.takeError()));
      return std::nullopt;
    }
    addressBits = *bits;
  } else {
    auto bits = indexTokenBitPattern(addr, width);
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
        access.isGather() ? addressBits->extractBits(width, width * lane)
                          : base + llvm::APInt(width, lane, /*isSigned=*/false,
                                               /*implicitTrunc=*/true);
    auto slot = resolveElementByteOffset(view, laneAddress,
                                         plan.elementLayout.byteCount,
                                         diagnostics, diagnosticLabel);
    if (!slot)
      return std::nullopt;
    slots.push_back(*slot);
  }
  return slots;
}

struct ProjectedDataflowMemoryAccess {
  llvm::APInt activeLanes;
  llvm::SmallVector<std::size_t> slots;
};

static std::optional<ProjectedDataflowMemoryAccess>
projectDataflowMemoryAccess(const MemoryView &view, const Token &addr,
                            const MemoryActorExecutionPlan &plan,
                            const Token *mask, mlir::Type maskType,
                            llvm::SmallVectorImpl<std::string> &diagnostics,
                            llvm::StringRef diagnosticLabel) {
  const bool vectorAccess = plan.access.isVector();
  auto activeLanes =
      getActiveMemoryLanes(plan.access, vectorAccess ? mask : nullptr,
                           vectorAccess ? maskType : mlir::Type{}, diagnostics);
  auto slots = activeLanes
                   ? resolveActiveLaneSlots(view, addr, plan, *activeLanes,
                                            diagnostics, diagnosticLabel)
                   : std::nullopt;
  if (!activeLanes || !slots)
    return std::nullopt;
  return ProjectedDataflowMemoryAccess{std::move(*activeLanes),
                                       std::move(*slots)};
}

static std::optional<DataflowMemoryRead> prepareDataflowMemoryRead(
    const MemoryView &view, const llvm::APInt &activeLanes,
    llvm::ArrayRef<std::size_t> slots, const MemoryActorExecutionPlan &plan,
    SimulatorState &state) {
  const auto &access = plan.access;
  // An element access is one complete memref element, even when that element
  // is itself a vector: one lane, one address, and no mask.
  if (!access.isVector()) {
    auto value =
        readMemoryElementResolved(view, slots.front(), access.elementType,
                                  plan.elementLayout, state, "dataflow.load");
    if (!value)
      return std::nullopt;
    return DataflowMemoryRead{*value, true};
  }

  // Inactive lanes keep the element type's all-zero bit representation.
  llvm::APInt resultBits(plan.dataBitWidth, 0);
  unsigned active = 0;
  for (unsigned lane = 0; lane < access.laneCount(); ++lane) {
    if (!activeLanes[lane])
      continue;
    auto element =
        readMemoryElementResolved(view, slots[active++], access.elementType,
                                  plan.elementLayout, state, "dataflow.load");
    if (!element)
      return std::nullopt;
    auto elementBits = memoryTokenBits(*element, access.elementType,
                                       plan.elementLayout.bitWidth);
    if (!elementBits) {
      state.diagnostics.push_back(llvm::toString(elementBits.takeError()));
      return std::nullopt;
    }
    resultBits.insertBits(*elementBits, plan.elementLayout.bitWidth * lane);
  }

  auto result = tokenFromMemoryBits(resultBits, access.vectorType);
  if (!result) {
    state.diagnostics.push_back(llvm::toString(result.takeError()));
    return std::nullopt;
  }
  return DataflowMemoryRead{*result, !activeLanes.isZero()};
}

static std::optional<DataflowMemoryWrite>
prepareDataflowMemoryWrite(const Token &data, const llvm::APInt &activeLanes,
                           llvm::ArrayRef<std::size_t> slots,
                           const MemoryActorExecutionPlan &plan,
                           SimulatorState &state) {
  const auto &access = plan.access;
  if (!access.isVector()) {
    auto bytes = encodeMemoryElementResolved(
        data, access.elementType, plan.elementLayout.byteCount,
        plan.elementLayout.bitWidth, plan.elementLayout.byteOrder);
    if (!bytes) {
      state.diagnostics.push_back(llvm::toString(bytes.takeError()));
      return std::nullopt;
    }
    DataflowMemoryWrite write;
    write.accessedMemory = true;
    write.elements.push_back(
        DataflowMemoryWrite::Element{slots.front(), std::move(*bytes)});
    return write;
  }

  if (activeLanes.isZero())
    return DataflowMemoryWrite{};

  auto dataBits = memoryTokenBits(data, access.vectorType, plan.dataBitWidth);
  if (!dataBits) {
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
    for (std::size_t slot : slots) {
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
  write.elements.reserve(slots.size());
  unsigned active = 0;
  for (unsigned lane = 0; lane < access.laneCount(); ++lane) {
    if (!activeLanes[lane])
      continue;
    llvm::APInt elementBits = dataBits->extractBits(
        plan.elementLayout.bitWidth, plan.elementLayout.bitWidth * lane);
    auto element = tokenFromMemoryBits(elementBits, access.elementType);
    if (!element) {
      state.diagnostics.push_back(llvm::toString(element.takeError()));
      return std::nullopt;
    }
    auto bytes = encodeMemoryElementResolved(
        *element, access.elementType, plan.elementLayout.byteCount,
        plan.elementLayout.bitWidth, plan.elementLayout.byteOrder);
    if (!bytes) {
      state.diagnostics.push_back(llvm::toString(bytes.takeError()));
      return std::nullopt;
    }
    write.elements.push_back(
        DataflowMemoryWrite::Element{slots[active++], std::move(*bytes)});
  }
  return write;
}

void commitDataflowMemoryWrite(const MemoryView &view,
                               const DataflowMemoryWrite &write) {
  for (const DataflowMemoryWrite::Element &element : write.elements)
    writeMemoryElement(view, element.byteOffset, element.bytes);
}

// The byte ranges one issued access covers, derived from the active element
// slots it already resolved. An inactive lane resolves no slot, so it
// contributes no range and derives no access.
static MemoryActionRecord projectMemoryAction(const MemoryView &view,
                                              llvm::ArrayRef<std::size_t> slots,
                                              std::size_t elementByteCount,
                                              bool isWrite) {
  MemoryActionRecord action;
  action.rootId = view.memory->logicalRootId;
  action.isWrite = isWrite;
  if (slots.empty())
    return action;
  action.byteRanges.reserve(slots.size());
  for (std::size_t slot : slots) {
    const std::int64_t begin = static_cast<std::int64_t>(slot);
    action.byteRanges.emplace_back(
        begin, begin + static_cast<std::int64_t>(elementByteCount));
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
peekMemoryOrderFrontier(SimulatorState &state, unsigned controlOperandOrdinal) {
  llvm::SmallVector<SyncEffectId, 2> frontier;
  if (hasInputToken(state, controlOperandOrdinal))
    state.memoryOrderFrontiers.appendCanonicalEffects(
        peekInputToken(state, controlOperandOrdinal).memoryOrder, frontier);
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

struct ProjectedMemoryFiring {
  MemoryView view;
  ProjectedDataflowMemoryAccess access;
  std::optional<unsigned> maskOperandOrdinal;
};

static std::optional<ProjectedMemoryFiring>
projectLoadFiring(dataflow::LoadOp op, SimulatorState &state,
                  llvm::SmallVectorImpl<std::string> &diagnostics) {
  assert(state.currentActorPlan && state.currentActorPlan->memory &&
         "admitted load has no execution plan");
  const MemoryActorExecutionPlan &plan = *state.currentActorPlan->memory;
  if (!hasInputToken(state, plan.addressOperandOrdinal) ||
      !hasInputToken(state, plan.controlOperandOrdinal) ||
      (plan.maskOperandOrdinal &&
       !hasInputToken(state, *plan.maskOperandOrdinal)))
    return std::nullopt;
  std::optional<MemoryView> view = peekMemoryView(
      state, op.getMem(), plan.memoryOperandOrdinal, diagnostics);
  if (!view)
    return std::nullopt;
  Token addr = peekInputToken(state, plan.addressOperandOrdinal);
  std::optional<Token> mask;
  if (plan.maskOperandOrdinal)
    mask = peekInputToken(state, *plan.maskOperandOrdinal);
  assert(state.currentActorPlan->operation == op.getOperation() &&
         "active load plan does not match the operation");
  auto access = projectDataflowMemoryAccess(
      *view, addr, plan, mask ? &*mask : nullptr,
      op.getMask() ? op.getMask().getType() : mlir::Type{}, diagnostics,
      "dataflow.load");
  if (!access)
    return std::nullopt;
  return ProjectedMemoryFiring{std::move(*view), std::move(*access),
                               plan.maskOperandOrdinal};
}

static std::optional<ProjectedMemoryFiring>
projectStoreFiring(dataflow::StoreOp op, SimulatorState &state,
                   llvm::SmallVectorImpl<std::string> &diagnostics) {
  assert(state.currentActorPlan && state.currentActorPlan->memory &&
         state.currentActorPlan->memory->dataOperandOrdinal &&
         "admitted store has no execution plan");
  const MemoryActorExecutionPlan &plan = *state.currentActorPlan->memory;
  if (!hasInputToken(state, plan.addressOperandOrdinal) ||
      !hasInputToken(state, *plan.dataOperandOrdinal) ||
      !hasInputToken(state, plan.controlOperandOrdinal) ||
      (plan.maskOperandOrdinal &&
       !hasInputToken(state, *plan.maskOperandOrdinal)))
    return std::nullopt;
  std::optional<MemoryView> view = peekMemoryView(
      state, op.getMem(), plan.memoryOperandOrdinal, diagnostics);
  if (!view)
    return std::nullopt;
  Token addr = peekInputToken(state, plan.addressOperandOrdinal);
  std::optional<Token> mask;
  if (plan.maskOperandOrdinal)
    mask = peekInputToken(state, *plan.maskOperandOrdinal);
  assert(state.currentActorPlan->operation == op.getOperation() &&
         "active store plan does not match the operation");
  auto access = projectDataflowMemoryAccess(
      *view, addr, plan, mask ? &*mask : nullptr,
      op.getMask() ? op.getMask().getType() : mlir::Type{}, diagnostics,
      "dataflow.store");
  if (!access)
    return std::nullopt;
  return ProjectedMemoryFiring{std::move(*view), std::move(*access),
                               plan.maskOperandOrdinal};
}

PlainMemoryActionProjection
projectReadyPlainMemoryAction(mlir::Operation *operation,
                              SimulatorState &state) {
  PlainMemoryActionProjection result;
  if (auto op = mlir::dyn_cast<dataflow::LoadOp>(operation)) {
    auto projected = projectLoadFiring(op, state, result.diagnostics);
    if (!projected)
      return result;
    const auto &plan = *state.currentActorPlan->memory;
    MemoryActionRecord action = projectMemoryAction(
        projected->view, projected->access.slots, plan.elementLayout.byteCount,
        /*isWrite=*/false);
    result.ready = ReadyPlainMemoryAction{
        std::move(action),
        peekMemoryOrderFrontier(state, plan.controlOperandOrdinal),
        std::move(projected->view),
        std::move(projected->access.activeLanes),
        std::move(projected->access.slots),
        projected->maskOperandOrdinal};
    return result;
  }

  auto op = mlir::dyn_cast<dataflow::StoreOp>(operation);
  if (!op)
    return result;
  auto projected = projectStoreFiring(op, state, result.diagnostics);
  if (!projected)
    return result;
  const auto &plan = *state.currentActorPlan->memory;
  MemoryActionRecord action = projectMemoryAction(
      projected->view, projected->access.slots, plan.elementLayout.byteCount,
      /*isWrite=*/true);
  result.ready = ReadyPlainMemoryAction{
      std::move(action),
      peekMemoryOrderFrontier(state, plan.controlOperandOrdinal),
      std::move(projected->view),
      std::move(projected->access.activeLanes),
      std::move(projected->access.slots),
      projected->maskOperandOrdinal};
  return result;
}

bool fireLoad(dataflow::LoadOp op, SimulatorState &state) {
  auto admitted = state.admittedPlainMemoryActions.find(op.getOperation());
  if (admitted == state.admittedPlainMemoryActions.end())
    return false;
  ReadyPlainMemoryAction &ready = admitted->second;
  const auto &plan = *state.currentActorPlan->memory;
  auto read = prepareDataflowMemoryRead(ready.view, ready.activeLanes,
                                        ready.slots, plan, state);
  if (!read)
    return false;
  auto publication = issueMemoryAction(ready.action, ready.ctrlFrontier, state);
  if (!publication)
    return false;
  std::optional<unsigned> maskOperandOrdinal = ready.maskOperandOrdinal;
  state.admittedPlainMemoryActions.erase(op.getOperation());

  consumeMemoryView(state, plan.memoryOperandOrdinal);
  (void)popInputToken(state, plan.addressOperandOrdinal);
  (void)popInputToken(state, plan.controlOperandOrdinal);
  if (maskOperandOrdinal)
    (void)popInputToken(state, *maskOperandOrdinal);
  emitResultTokenWithMemoryOrder(state, 0, read->data, MemoryOrderFrontierId());
  emitResultTokenWithMemoryOrder(state, 1, noneToken(), *publication);
  return true;
}

bool fireStore(dataflow::StoreOp op, SimulatorState &state) {
  auto admitted = state.admittedPlainMemoryActions.find(op.getOperation());
  if (admitted == state.admittedPlainMemoryActions.end())
    return false;
  ReadyPlainMemoryAction &ready = admitted->second;
  const auto &plan = *state.currentActorPlan->memory;
  assert(plan.dataOperandOrdinal && "store plan has no data operand");
  Token data = peekInputToken(state, *plan.dataOperandOrdinal);
  auto write = prepareDataflowMemoryWrite(data, ready.activeLanes, ready.slots,
                                          plan, state);
  if (!write)
    return false;
  auto publication = issueMemoryAction(ready.action, ready.ctrlFrontier, state);
  if (!publication)
    return false;
  MemoryView view = ready.view;
  std::optional<unsigned> maskOperandOrdinal = ready.maskOperandOrdinal;
  state.admittedPlainMemoryActions.erase(op.getOperation());

  consumeMemoryView(state, plan.memoryOperandOrdinal);
  (void)popInputToken(state, plan.addressOperandOrdinal);
  (void)popInputToken(state, *plan.dataOperandOrdinal);
  (void)popInputToken(state, plan.controlOperandOrdinal);
  if (maskOperandOrdinal)
    (void)popInputToken(state, *maskOperandOrdinal);
  commitDataflowMemoryWrite(view, *write);
  emitResultTokenWithMemoryOrder(state, 0, noneToken(), *publication);
  return true;
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

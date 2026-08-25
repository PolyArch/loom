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
#include <limits>
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
    if (token.kind != TokenKind::MemoryCapability || !view || !view->memory) {
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
  std::optional<unsigned> desiredOperandOrdinal;
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
  } else if (auto rmw = mlir::dyn_cast<dataflow::AtomicRmwOp>(operation)) {
    memory = rmw.getMem();
    address = rmw.getAddr();
    dataType = rmw.getValue().getType();
    mask = rmw.getMask();
    memoryOperandOrdinal = rmw.getMemMutable().getOperandNumber();
    addressOperandOrdinal = rmw.getAddrMutable().getOperandNumber();
    dataOperandOrdinal = rmw.getValueMutable().getOperandNumber();
    controlOperandOrdinal = rmw.getCtrlMutable().getOperandNumber();
  } else if (auto cmpxchg = mlir::dyn_cast<dataflow::CmpXchgOp>(operation)) {
    memory = cmpxchg.getMem();
    address = cmpxchg.getAddr();
    dataType = cmpxchg.getExpected().getType();
    mask = cmpxchg.getMask();
    memoryOperandOrdinal = cmpxchg.getMemMutable().getOperandNumber();
    addressOperandOrdinal = cmpxchg.getAddrMutable().getOperandNumber();
    dataOperandOrdinal = cmpxchg.getExpectedMutable().getOperandNumber();
    desiredOperandOrdinal = cmpxchg.getDesiredMutable().getOperandNumber();
    controlOperandOrdinal = cmpxchg.getCtrlMutable().getOperandNumber();
  } else {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "memory execution plan requires an addressed memory actor");
  }
  if (mask)
    maskOperandOrdinal = operation->getNumOperands() - 1;

  auto access = dataflow::semantics::analyzeMemoryAccessType(
      mlir::cast<mlir::MemRefType>(memory.getType()), dataType,
      address.getType(), operation, mask ? mask.getType() : mlir::Type{});
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
      dataOperandOrdinal, desiredOperandOrdinal, controlOperandOrdinal,
      maskOperandOrdinal, *indexWidth,           *addressWidth,
      *dataWidth,         *elementLayout};
}

std::optional<std::string>
unsupportedMemoryActorRepresentation(mlir::Operation *operation) {
  std::optional<dataflow::semantics::MemoryActorContract> contract =
      dataflow::semantics::getMemoryActorContract(operation);
  if (contract && contract->syncScope &&
      contract->syncScope.getKind() == dataflow::SyncScopeKind::Target)
    return "target-specific synchronization scope has no DFG-sim domain "
           "resolver";
  if (mlir::isa<dataflow::FenceOp>(operation))
    return std::nullopt;

  mlir::Value memory;
  mlir::Value data;
  mlir::Value mask;
  if (auto load = mlir::dyn_cast<dataflow::LoadOp>(operation)) {
    memory = load.getMem();
    data = load.getData();
    mask = load.getMask();
  } else if (auto store = mlir::dyn_cast<dataflow::StoreOp>(operation)) {
    memory = store.getMem();
    data = store.getData();
    mask = store.getMask();
  } else if (auto rmw = mlir::dyn_cast<dataflow::AtomicRmwOp>(operation)) {
    memory = rmw.getMem();
    data = rmw.getValue();
    mask = rmw.getMask();
  } else if (auto cmpxchg = mlir::dyn_cast<dataflow::CmpXchgOp>(operation)) {
    memory = cmpxchg.getMem();
    data = cmpxchg.getExpected();
    mask = cmpxchg.getMask();
  } else {
    return std::nullopt;
  }

  mlir::Value address = operation->getOperand(1);
  if (contract && (contract->atomic || contract->isVolatile) &&
      (mlir::isa<mlir::VectorType>(data.getType()) ||
       mlir::isa<mlir::VectorType>(address.getType()) || mask))
    return "DFG-sim atomic and volatile memory provider supports scalar "
           "addressed actions only";
  if (auto addressVector = mlir::dyn_cast<mlir::VectorType>(address.getType()))
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(addressVector.getElementType()))
      return "vector pointer memory addresses have no lane-local pointer "
             "provenance provider";

  auto memref = mlir::dyn_cast<mlir::MemRefType>(memory.getType());
  auto vector = memref
                    ? mlir::dyn_cast<mlir::VectorType>(memref.getElementType())
                    : mlir::VectorType{};
  if (!vector)
    return std::nullopt;
  auto laneWidth =
      resolvedTokenTypeBitWidth(vector.getElementType(), operation);
  if (!laneWidth)
    return llvm::toString(laneWidth.takeError());
  if (*laneWidth % 8 != 0)
    return "vector-valued memory elements with sub-byte-aligned lanes cannot "
           "preserve lane-local exceptional state in byte-addressed memory";
  return std::nullopt;
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
  if (auto pointer = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(type)) {
    const PointerValue *value = token.pointerValue();
    if (token.valueState != PrimitiveValueState::Defined ||
        token.kind != TokenKind::Pointer || !value ||
        value->addressSpace != pointer.getAddressSpace() ||
        value->representation.getBitWidth() != bitWidth)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "pointer memory token does not match its exact representation");
    return value->representation;
  }
  return tokenBitPattern(token, type);
}

static PrimitiveValueState mergeMemoryLaneState(PrimitiveValueState current,
                                                SemanticState observed) {
  if (current == PrimitiveValueState::Poison ||
      observed == SemanticState::Poison)
    return PrimitiveValueState::Poison;
  if (current == PrimitiveValueState::Undef || observed == SemanticState::Undef)
    return PrimitiveValueState::Undef;
  return PrimitiveValueState::Defined;
}

static llvm::Expected<Token>
tokenFromMemoryVector(mlir::VectorType type,
                      const ResolvedMemoryElementLayout &layout,
                      llvm::ArrayRef<SemanticMemoryByte> bytes) {
  const std::size_t laneCount = static_cast<std::size_t>(type.getNumElements());
  if (laneCount == 0 || layout.bitWidth % laneCount != 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector memory layout has no uniform semantic lane width");
  const unsigned laneBitWidth =
      layout.bitWidth / static_cast<unsigned>(laneCount);
  if (laneBitWidth == 0)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "vector memory layout has zero-width semantic lanes");
  llvm::SmallVector<PrimitiveValueState, 4> states(
      laneCount, PrimitiveValueState::Defined);
  llvm::APInt packedBits(layout.bitWidth, 0);
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    const std::size_t semanticByte = layout.byteOrder == MemoryByteOrder::Little
                                         ? index
                                         : bytes.size() - 1 - index;
    const unsigned low = static_cast<unsigned>(semanticByte * 8);
    if (low >= layout.bitWidth)
      continue;
    const unsigned width = std::min(8u, layout.bitWidth - low);
    const unsigned firstLane = low / laneBitWidth;
    const unsigned lastLane = (low + width - 1) / laneBitWidth;
    for (unsigned lane = firstLane; lane <= lastLane; ++lane)
      states[lane] = mergeMemoryLaneState(states[lane], bytes[index].state);
    if (bytes[index].state == SemanticState::Defined)
      packedBits.insertBits(llvm::APInt(width, bytes[index].value), low);
  }

  bool allDefined = true;
  bool allPoison = true;
  bool allUndef = true;
  for (PrimitiveValueState state : states) {
    allDefined &= state == PrimitiveValueState::Defined;
    allPoison &= state == PrimitiveValueState::Poison;
    allUndef &= state == PrimitiveValueState::Undef;
  }
  if (allDefined)
    return tokenFromMemoryBits(packedBits, type);
  if (allPoison)
    return exceptionalValueToken(PrimitiveValueState::Poison, type);
  if (allUndef)
    return exceptionalValueToken(PrimitiveValueState::Undef, type);
  Token token;
  token.kind = TokenKind::Vector;
  token.setVectorLanePayload(VectorLanePayload{laneBitWidth, std::move(states),
                                               std::move(packedBits)});
  return token;
}

std::optional<Token> readMemoryElementResolved(
    const MemoryView &view, std::size_t byteOffset, mlir::Type dataType,
    const ResolvedMemoryElementLayout &layout,
    const std::optional<::loom::PointerLayout> &pointerLayout,
    SimulatorState &state, llvm::StringRef diagnosticLabel) {
  const std::size_t byteCount = layout.byteCount;
  const unsigned bitWidth = layout.bitWidth;
  const MemoryByteOrder order = layout.byteOrder;
  for (std::size_t index = 0; index < byteCount; ++index) {
    if (!view.memory->initialized[byteOffset + index]) {
      state.diagnostics.push_back(
          (diagnosticLabel + " reads uninitialized memory").str());
      return std::nullopt;
    }
  }
  llvm::ArrayRef<SemanticMemoryByte> storedBytes(
      view.memory->bytes.data() + byteOffset, byteCount);
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(dataType)) {
    auto token = tokenFromMemoryVector(vector, layout, storedBytes);
    if (!token) {
      state.diagnostics.push_back(llvm::toString(token.takeError()));
      return std::nullopt;
    }
    return *token;
  }
  bool poison = false;
  bool undef = false;
  llvm::APInt bits(bitWidth, 0);
  for (std::size_t index = 0; index < byteCount; ++index) {
    const std::size_t position = byteOffset + index;
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
                                       dataType);
    if (!token) {
      state.diagnostics.push_back(llvm::toString(token.takeError()));
      return std::nullopt;
    }
    return *token;
  }
  if (pointerLayout) {
    auto stored = view.memory->pointerValues.find(byteOffset);
    if (stored == view.memory->pointerValues.end()) {
      state.diagnostics.push_back(
          (diagnosticLabel +
           " reads defined pointer bits without pointer provenance")
              .str());
      return std::nullopt;
    }
    const PointerValue &pointer = stored->second;
    if (pointer.addressSpace != pointerLayout->addressSpace ||
        pointer.representation.getBitWidth() !=
            pointerLayout->representationBits ||
        pointer.byteOffset.getBitWidth() != pointerLayout->addressBits ||
        pointer.representation != bits || !pointer.memory) {
      state.diagnostics.push_back(
          (diagnosticLabel + " reads inconsistent pointer provenance").str());
      return std::nullopt;
    }
    Token token;
    token.kind = TokenKind::Pointer;
    token.setPointerValue(pointer);
    return token;
  }
  auto token = tokenFromMemoryBits(bits, dataType);
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
                                   std::nullopt, state, diagnosticLabel);
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
  if (auto vector = mlir::dyn_cast<mlir::VectorType>(elementType)) {
    if (const VectorLanePayload *payload = value.vectorLanePayload()) {
      const std::size_t laneCount =
          static_cast<std::size_t>(vector.getNumElements());
      if (laneCount == 0 || payload->laneStates.size() != laneCount ||
          payload->laneBitWidth == 0 ||
          laneCount >
              std::numeric_limits<unsigned>::max() / payload->laneBitWidth ||
          payload->laneBitWidth * laneCount != bitWidth ||
          payload->packedBits.getBitWidth() != bitWidth)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "mixed-lane memory token does not match its vector layout");
      if (payload->laneBitWidth % 8 != 0)
        return llvm::createStringError(
            std::errc::not_supported,
            "mixed vector memory token has sub-byte-aligned lanes that "
            "cannot preserve lane-local exceptional state");
      for (std::size_t index = 0; index < bytes.size(); ++index) {
        const std::size_t semanticByte =
            order == MemoryByteOrder::Little ? index : bytes.size() - 1 - index;
        const unsigned low = static_cast<unsigned>(semanticByte * 8);
        const unsigned width =
            low >= bitWidth ? 0 : std::min(8u, bitWidth - low);
        if (width == 0) {
          bytes[index] = SemanticMemoryByte{SemanticState::Defined, 0};
          continue;
        }
        const unsigned firstLane = low / payload->laneBitWidth;
        const unsigned lastLane = (low + width - 1) / payload->laneBitWidth;
        PrimitiveValueState state = PrimitiveValueState::Defined;
        for (unsigned lane = firstLane; lane <= lastLane; ++lane) {
          if (payload->laneStates[lane] == PrimitiveValueState::Poison) {
            state = PrimitiveValueState::Poison;
            break;
          }
          if (payload->laneStates[lane] == PrimitiveValueState::Undef)
            state = PrimitiveValueState::Undef;
        }
        const SemanticState semanticState =
            state == PrimitiveValueState::Poison  ? SemanticState::Poison
            : state == PrimitiveValueState::Undef ? SemanticState::Undef
                                                  : SemanticState::Defined;
        bytes[index] = SemanticMemoryByte{
            semanticState,
            semanticState == SemanticState::Defined
                ? static_cast<std::uint8_t>(
                      payload->packedBits.extractBitsAsZExtValue(width, low))
                : std::uint8_t{0}};
      }
      return bytes;
    }
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
  const std::size_t writeEnd = byteOffset + bytes.size();
  for (auto stored = view.memory->pointerValues.begin();
       stored != view.memory->pointerValues.end();) {
    const std::size_t pointerBytes =
        (stored->second.representation.getBitWidth() + 7) / 8;
    const std::size_t pointerEnd = stored->first + pointerBytes;
    if (stored->first < writeEnd && byteOffset < pointerEnd)
      stored = view.memory->pointerValues.erase(stored);
    else
      ++stored;
  }
  std::copy(bytes.begin(), bytes.end(),
            view.memory->bytes.begin() + byteOffset);
  view.memory->initialized.set(byteOffset, byteOffset + bytes.size());
}

static std::optional<llvm::APInt>
getActiveMemoryLanes(const dataflow::semantics::MemoryAccessType &access,
                     const Token *mask, mlir::Type maskType,
                     llvm::SmallVectorImpl<std::string> &diagnostics,
                     SimulatorState &state, mlir::Operation *scope) {
  if (access.laneCount() > std::numeric_limits<unsigned>::max()) {
    diagnostics.push_back(
        "vector lane count exceeds the simulator bit-vector limit");
    state.failure = RunFailure::ProviderInvariant;
    return std::nullopt;
  }
  const unsigned lanes = static_cast<unsigned>(access.laneCount());
  if (!mask)
    return llvm::APInt::getAllOnes(lanes);

  auto maskValues = vectorPrimitiveValues(
      *mask, mlir::cast<mlir::VectorType>(maskType), scope);
  if (!maskValues) {
    diagnostics.push_back(llvm::toString(maskValues.takeError()));
    state.failure = RunFailure::ProviderInvariant;
    return std::nullopt;
  }
  if (maskValues->size() != lanes) {
    diagnostics.push_back(
        "memory mask lane count does not match its access shape");
    state.failure = RunFailure::ProviderInvariant;
    return std::nullopt;
  }
  llvm::APInt active(lanes, 0);
  for (auto [lane, value] : llvm::enumerate(*maskValues)) {
    if (!value.isDefined()) {
      diagnostics.push_back(
          "memory exceptional mask activity has no exact single-path "
          "provider");
      state.failure = RunFailure::UnsupportedCapability;
      return std::nullopt;
    }
    if (!value.bits->isZero())
      active.setBit(static_cast<unsigned>(lane));
  }
  return active;
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
                       const llvm::APInt &activeLanes, mlir::Operation *scope,
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

  if (access.addressForm ==
      dataflow::semantics::MemoryAddressForm::PointerAddressed) {
    if (access.isGather()) {
      diagnostics.push_back(
          "vector pointer memory addresses have no DFG-sim token provider");
      return std::nullopt;
    }
    const PointerValue *pointer = addr.pointerValue();
    if (addr.valueState != PrimitiveValueState::Defined ||
        addr.kind != TokenKind::Pointer || !pointer || !pointer->memory) {
      diagnostics.push_back(
          (diagnosticLabel + " pointer address has no object provenance")
              .str());
      return std::nullopt;
    }
    if (!access.pointerLayout ||
        pointer->addressSpace != access.pointerLayout->addressSpace ||
        pointer->representation.getBitWidth() !=
            access.pointerLayout->representationBits ||
        pointer->byteOffset.getBitWidth() !=
            access.pointerLayout->addressBits) {
      diagnostics.push_back(
          (diagnosticLabel + " pointer address has the wrong layout").str());
      return std::nullopt;
    }
    if (pointer->memory != view.memory) {
      diagnostics.push_back(
          (diagnosticLabel +
           " pointer does not resolve through the selected memory service")
              .str());
      return std::nullopt;
    }
    if (view.byteOffset < 0 || pointer->byteOffset.isNegative()) {
      diagnostics.push_back(
          (diagnosticLabel + " pointer address is out of range").str());
      return std::nullopt;
    }

    const std::uint64_t base = pointer->byteOffset.getLimitedValue();
    slots.reserve(activeLanes.popcount());
    for (unsigned lane = 0; lane < access.laneCount(); ++lane) {
      if (!activeLanes[lane])
        continue;
      if (lane > std::numeric_limits<std::uint64_t>::max() /
                     plan.elementLayout.byteCount) {
        diagnostics.push_back(
            (diagnosticLabel + " pointer lane offset overflows").str());
        return std::nullopt;
      }
      const std::uint64_t laneOffset =
          static_cast<std::uint64_t>(lane) * plan.elementLayout.byteCount;
      if (base > std::numeric_limits<std::uint64_t>::max() - laneOffset) {
        diagnostics.push_back(
            (diagnosticLabel + " pointer lane address overflows").str());
        return std::nullopt;
      }
      const std::uint64_t slot = base + laneOffset;
      if (slot > view.memory->bytes.size() ||
          plan.elementLayout.byteCount > view.memory->bytes.size() - slot) {
        diagnostics.push_back(
            (diagnosticLabel + " pointer address is out of range").str());
        return std::nullopt;
      }
      slots.push_back(static_cast<std::size_t>(slot));
    }
    return slots;
  }

  llvm::APInt base;
  std::optional<llvm::SmallVector<PrimitiveValue, 8>> addressLanes;
  if (access.isGather()) {
    auto lanes = vectorPrimitiveValues(addr, access.addressVectorType, scope);
    if (!lanes) {
      diagnostics.push_back(llvm::toString(lanes.takeError()));
      return std::nullopt;
    }
    addressLanes = std::move(*lanes);
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
    llvm::APInt laneAddress = base;
    if (access.isGather()) {
      const PrimitiveValue &address = (*addressLanes)[lane];
      if (!address.isDefined() || address.bits->getBitWidth() != width) {
        diagnostics.push_back(
            (diagnosticLabel + " active lane address is not defined").str());
        return std::nullopt;
      }
      laneAddress = *address.bits;
    } else {
      laneAddress += llvm::APInt(width, lane, /*isSigned=*/false,
                                 /*implicitTrunc=*/true);
    }
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

static std::optional<ProjectedDataflowMemoryAccess> projectDataflowMemoryAccess(
    const MemoryView &view, const Token &addr,
    const MemoryActorExecutionPlan &plan, const Token *mask,
    mlir::Type maskType, mlir::Operation *scope,
    llvm::SmallVectorImpl<std::string> &diagnostics, SimulatorState &state,
    llvm::StringRef diagnosticLabel) {
  const bool vectorAccess = plan.access.isVector();
  auto activeLanes = getActiveMemoryLanes(
      plan.access, vectorAccess ? mask : nullptr,
      vectorAccess ? maskType : mlir::Type{}, diagnostics, state, scope);
  auto slots = activeLanes
                   ? resolveActiveLaneSlots(view, addr, plan, *activeLanes,
                                            scope, diagnostics, diagnosticLabel)
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
    auto value = readMemoryElementResolved(
        view, slots.front(), access.dataType, plan.elementLayout,
        access.dataPointerLayout, state, "dataflow.load");
    if (!value)
      return std::nullopt;
    return DataflowMemoryRead{*value, true};
  }

  // Inactive lanes keep the element type's defined all-zero value.
  llvm::SmallVector<PrimitiveValue, 8> resultLanes(
      access.laneCount(),
      PrimitiveValue::integer(llvm::APInt(plan.elementLayout.bitWidth, 0)));
  unsigned active = 0;
  for (unsigned lane = 0; lane < access.laneCount(); ++lane) {
    if (!activeLanes[lane])
      continue;
    auto element = readMemoryElementResolved(
        view, slots[active++], access.elementType, plan.elementLayout,
        std::nullopt, state, "dataflow.load");
    if (!element)
      return std::nullopt;
    auto elementValue = primitiveValueFromToken(*element, access.elementType,
                                                plan.indexBitWidth);
    if (!elementValue) {
      state.diagnostics.push_back(llvm::toString(elementValue.takeError()));
      return std::nullopt;
    }
    resultLanes[lane] = std::move(*elementValue);
  }

  auto result = tokenFromVectorPrimitiveValues(
      resultLanes, access.vectorType, state.currentActorPlan->operation);
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
        data, access.dataType, plan.elementLayout.byteCount,
        plan.elementLayout.bitWidth, plan.elementLayout.byteOrder);
    if (!bytes) {
      state.diagnostics.push_back(llvm::toString(bytes.takeError()));
      return std::nullopt;
    }
    DataflowMemoryWrite write;
    write.accessedMemory = true;
    std::optional<PointerValue> pointer;
    if (access.dataPointerLayout &&
        data.valueState == PrimitiveValueState::Defined) {
      const PointerValue *value = data.pointerValue();
      if (!value ||
          value->addressSpace != access.dataPointerLayout->addressSpace ||
          value->representation.getBitWidth() !=
              access.dataPointerLayout->representationBits ||
          value->byteOffset.getBitWidth() !=
              access.dataPointerLayout->addressBits ||
          !value->memory) {
        state.diagnostics.push_back(
            "dataflow.store pointer data has the wrong provenance layout");
        return std::nullopt;
      }
      pointer = *value;
    }
    write.elements.push_back(DataflowMemoryWrite::Element{
        slots.front(), std::move(*bytes), std::move(pointer)});
    return write;
  }

  if (activeLanes.isZero())
    return DataflowMemoryWrite{};

  auto dataLanes = vectorPrimitiveValues(data, access.vectorType,
                                         state.currentActorPlan->operation);
  if (!dataLanes) {
    state.diagnostics.push_back(llvm::toString(dataLanes.takeError()));
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
    auto element =
        tokenFromPrimitiveValue((*dataLanes)[lane], access.elementType);
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
    write.elements.push_back(DataflowMemoryWrite::Element{
        slots[active++], std::move(*bytes), std::nullopt});
  }
  return write;
}

void commitDataflowMemoryWrite(const MemoryView &view,
                               const DataflowMemoryWrite &write) {
  for (const DataflowMemoryWrite::Element &element : write.elements) {
    writeMemoryElement(view, element.byteOffset, element.bytes);
    if (element.pointer)
      view.memory->pointerValues.emplace(element.byteOffset, *element.pointer);
  }
}

// The byte ranges one issued access covers, derived from the active element
// slots it already resolved. An inactive lane resolves no slot, so it
// contributes no range and derives no access.
static MemoryActionRecord projectMemoryAction(const MemoryView &view,
                                              llvm::ArrayRef<std::size_t> slots,
                                              std::size_t elementByteCount,
                                              bool isWrite, bool isAtomic) {
  MemoryActionRecord action;
  action.rootId = view.memory->logicalRootId;
  action.isWrite = isWrite;
  action.isAtomic = isAtomic;
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

MemorySynchronization &memorySynchronization(SimulatorState &state) {
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

std::optional<DataflowMemoryRead>
prepareMemoryRead(const ReadyMemoryAction &ready,
                  const MemoryActorExecutionPlan &plan, SimulatorState &state) {
  return prepareDataflowMemoryRead(ready.view, ready.activeLanes, ready.slots,
                                   plan, state);
}

std::optional<DataflowMemoryWrite>
prepareMemoryWrite(const Token &data, const ReadyMemoryAction &ready,
                   const MemoryActorExecutionPlan &plan,
                   SimulatorState &state) {
  return prepareDataflowMemoryWrite(data, ready.activeLanes, ready.slots, plan,
                                    state);
}

std::optional<MemoryOrderFrontierId>
linearizePlainMemoryAction(const ReadyMemoryAction &ready,
                           SimulatorState &state) {
  return issueMemoryAction(ready.action, ready.ctrlFrontier, state);
}

void consumeMemoryIssueInputs(const ReadyMemoryAction &ready,
                              const MemoryActorExecutionPlan &plan,
                              SimulatorState &state) {
  consumeMemoryView(state, plan.memoryOperandOrdinal);
  (void)popInputToken(state, plan.addressOperandOrdinal);
  if (plan.dataOperandOrdinal)
    (void)popInputToken(state, *plan.dataOperandOrdinal);
  if (plan.desiredOperandOrdinal)
    (void)popInputToken(state, *plan.desiredOperandOrdinal);
  (void)popInputToken(state, plan.controlOperandOrdinal);
  if (ready.maskOperandOrdinal)
    (void)popInputToken(state, *ready.maskOperandOrdinal);
}

struct ProjectedMemoryFiring {
  MemoryView view;
  ProjectedDataflowMemoryAccess access;
  std::optional<unsigned> maskOperandOrdinal;
};

static std::optional<ProjectedMemoryFiring>
projectAddressedMemoryFiring(mlir::Operation *operation, SimulatorState &state,
                             llvm::SmallVectorImpl<std::string> &diagnostics) {
  assert(state.currentActorPlan && state.currentActorPlan->memory &&
         state.currentActorPlan->operation == operation &&
         "admitted addressed memory actor has no matching execution plan");
  const MemoryActorExecutionPlan &plan = *state.currentActorPlan->memory;
  if (!hasInputToken(state, plan.addressOperandOrdinal) ||
      (plan.dataOperandOrdinal &&
       !hasInputToken(state, *plan.dataOperandOrdinal)) ||
      (plan.desiredOperandOrdinal &&
       !hasInputToken(state, *plan.desiredOperandOrdinal)) ||
      !hasInputToken(state, plan.controlOperandOrdinal) ||
      (plan.maskOperandOrdinal &&
       !hasInputToken(state, *plan.maskOperandOrdinal)))
    return std::nullopt;
  std::optional<MemoryView> view =
      peekMemoryView(state, operation->getOperand(plan.memoryOperandOrdinal),
                     plan.memoryOperandOrdinal, diagnostics);
  if (!view)
    return std::nullopt;
  Token addr = peekInputToken(state, plan.addressOperandOrdinal);
  std::optional<Token> mask;
  if (plan.maskOperandOrdinal)
    mask = peekInputToken(state, *plan.maskOperandOrdinal);
  mlir::Type maskType =
      plan.maskOperandOrdinal
          ? operation->getOperand(*plan.maskOperandOrdinal).getType()
          : mlir::Type{};
  auto access = projectDataflowMemoryAccess(
      *view, addr, plan, mask ? &*mask : nullptr, maskType, operation,
      diagnostics, state, operation->getName().getStringRef());
  if (!access)
    return std::nullopt;
  return ProjectedMemoryFiring{std::move(*view), std::move(*access),
                               plan.maskOperandOrdinal};
}

MemoryActionProjection projectReadyMemoryAction(mlir::Operation *operation,
                                                SimulatorState &state) {
  MemoryActionProjection result;
  if (!state.currentActorPlan || !state.currentActorPlan->memory)
    return result;
  auto projected =
      projectAddressedMemoryFiring(operation, state, result.diagnostics);
  if (!projected)
    return result;
  const auto &plan = *state.currentActorPlan->memory;
  const auto *contract = std::get_if<dataflow::MemoryContractPayload>(
      &state.currentActorPlan->projection.payload);
  const bool isAtomic =
      contract &&
      !std::holds_alternative<dataflow::PlainAccessProjection>(*contract);
  MemoryActionRecord action = projectMemoryAction(
      projected->view, projected->access.slots, plan.elementLayout.byteCount,
      /*isWrite=*/!mlir::isa<dataflow::LoadOp>(operation), isAtomic);
  result.ready = ReadyMemoryAction{
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
  ReadyMemoryAction &ready = admitted->second;
  const auto &plan = *state.currentActorPlan->memory;
  auto read = prepareMemoryRead(ready, plan, state);
  if (!read)
    return false;
  auto publication = linearizePlainMemoryAction(ready, state);
  if (!publication)
    return false;

  consumeMemoryIssueInputs(ready, plan, state);
  state.admittedPlainMemoryActions.erase(op.getOperation());
  emitResultTokenWithMemoryOrder(state, 0, read->data, MemoryOrderFrontierId());
  emitResultTokenWithMemoryOrder(state, 1, noneToken(), *publication);
  return true;
}

bool fireStore(dataflow::StoreOp op, SimulatorState &state) {
  auto admitted = state.admittedPlainMemoryActions.find(op.getOperation());
  if (admitted == state.admittedPlainMemoryActions.end())
    return false;
  ReadyMemoryAction &ready = admitted->second;
  const auto &plan = *state.currentActorPlan->memory;
  assert(plan.dataOperandOrdinal && "store plan has no data operand");
  Token data = peekInputToken(state, *plan.dataOperandOrdinal);
  auto write = prepareMemoryWrite(data, ready, plan, state);
  if (!write)
    return false;
  auto publication = linearizePlainMemoryAction(ready, state);
  if (!publication)
    return false;
  MemoryView view = ready.view;

  consumeMemoryIssueInputs(ready, plan, state);
  state.admittedPlainMemoryActions.erase(op.getOperation());
  commitDataflowMemoryWrite(view, *write);
  emitResultTokenWithMemoryOrder(state, 0, noneToken(), *publication);
  return true;
}

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

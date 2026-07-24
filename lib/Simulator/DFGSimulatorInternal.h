#ifndef LOOM_LIB_SIMULATOR_DFGSIMULATORINTERNAL_H
#define LOOM_LIB_SIMULATOR_DFGSIMULATORINTERNAL_H

#include "Simulator/DFGSimulator.h"
#include "Simulator/MemorySynchronization.h"

#include "Dataflow/IR/DataflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

inline constexpr std::uint64_t kLoadAddressScore = 1;
inline constexpr std::uint64_t kStoreAddressScore = 2;

struct MemoryValue;

struct MemoryView {
  std::shared_ptr<MemoryValue> memory;
  mlir::Value root;
  std::int64_t byteOffset = 0;
};

enum class TokenKind { None, Integer, Float, Bool, Vector, Pointer };

struct Token {
  TokenKind kind = TokenKind::None;
  // Index storage and the schema 2.2 projection for integers up to 64 bits.
  std::int64_t intValue = 0;
  double floatValue = 0.0;
  bool boolValue = false;
  std::optional<llvm::APInt> bitPattern;
  MemoryView pointer;
  // Memory-order witnesses enter token flow only through canonical done
  // publication. Generic actor firing may propagate them from that explicit
  // path, but plain memory data publication never injects its action effect.
  // This state is execution-local and never serialized.
  llvm::SmallVector<SyncEffectId, 1> memoryOrderFrontier;
};

struct DataflowMemoryRead {
  Token data;
  bool accessedMemory = false;
};

// The complete element update one store commits, prepared before any element
// changes so a rejected access leaves memory untouched.
struct DataflowMemoryWrite {
  llvm::SmallVector<std::pair<std::size_t, Token>> elements;
  bool accessedMemory = false;
};

using ChannelMap = llvm::DenseMap<const mlir::OpOperand *, std::deque<Token>>;
using OutputMap = llvm::DenseMap<mlir::Value, llvm::SmallVector<Token>>;

struct LoopState {
  PhaseSemanticState semanticState = PhaseSemanticState::Initial;
  std::optional<Token> latched;
};

struct ParallelizeState {
  ParallelizeSemanticState semanticState;
  llvm::SmallVector<std::optional<Token>, 8> slots;
  // Memory-order frontiers of scalar phases consumed while assembling the
  // current group. The final firing publishes their union.
  llvm::SmallVector<SyncEffectId, 2> phaseFrontier;
};

struct MemoryValue {
  std::uint64_t logicalRootId = 0;
  mlir::Type elementType;
  llvm::SmallVector<Token> elements;
  llvm::SmallBitVector initialized;
};

struct MemoryFixture {
  std::string values;
  std::int64_t byteOffset = 0;
};

// The execution-local footprint of one ordinary access: the logical object it
// touches and the byte ranges its active lanes cover. In the conflict cache,
// ranges already superseded by a later effect are removed from this record.
// MemorySynchronization remains the authority for every causal comparison.
struct MemoryActionRecord {
  std::uint64_t rootId = 0;
  // Half-open byte ranges of the active lanes, relative to the logical root.
  llvm::SmallVector<std::pair<std::int64_t, std::int64_t>, 1> byteRanges;
  bool isWrite = false;
};

inline void canonicalizeMemoryActionRanges(
    llvm::SmallVectorImpl<std::pair<std::int64_t, std::int64_t>> &ranges) {
  llvm::sort(ranges);
  llvm::SmallVector<std::pair<std::int64_t, std::int64_t>> merged;
  for (const auto &range : ranges) {
    if (range.first >= range.second)
      continue;
    if (merged.empty() || merged.back().second < range.first) {
      merged.push_back(range);
      continue;
    }
    merged.back().second = std::max(merged.back().second, range.second);
  }
  ranges.assign(merged.begin(), merged.end());
}

struct ReadyPlainMemoryAction {
  MemoryActionRecord action;
  llvm::SmallVector<SyncEffectId, 2> ctrlFrontier;
};

struct ReadyPlainMemoryConflictScan {
  bool hasConflict = false;
  std::uint64_t inspectedRanges = 0;
};

inline ReadyPlainMemoryConflictScan
scanReadyPlainMemoryConflicts(llvm::ArrayRef<ReadyPlainMemoryAction> actions) {
  struct Range {
    std::uint64_t rootId;
    std::int64_t begin;
    std::int64_t end;
    bool isWrite;
  };

  llvm::SmallVector<Range> ranges;
  for (const ReadyPlainMemoryAction &ready : actions) {
    auto actionRanges = ready.action.byteRanges;
    canonicalizeMemoryActionRanges(actionRanges);
    for (const auto &[begin, end] : actionRanges)
      ranges.push_back(
          Range{ready.action.rootId, begin, end, ready.action.isWrite});
  }
  llvm::sort(ranges, [](const Range &lhs, const Range &rhs) {
    if (lhs.rootId != rhs.rootId)
      return lhs.rootId < rhs.rootId;
    if (lhs.begin != rhs.begin)
      return lhs.begin < rhs.begin;
    if (lhs.end != rhs.end)
      return lhs.end < rhs.end;
    return lhs.isWrite < rhs.isWrite;
  });

  ReadyPlainMemoryConflictScan result;
  std::optional<std::uint64_t> rootId;
  std::int64_t maximalEnd = 0;
  std::int64_t maximalWriteEnd = 0;
  bool hasRange = false;
  bool hasWrite = false;
  for (const Range &range : ranges) {
    ++result.inspectedRanges;
    if (!rootId || *rootId != range.rootId) {
      rootId = range.rootId;
      maximalEnd = range.end;
      maximalWriteEnd = range.end;
      hasRange = true;
      hasWrite = range.isWrite;
      continue;
    }
    if ((range.isWrite && hasRange && maximalEnd > range.begin) ||
        (!range.isWrite && hasWrite && maximalWriteEnd > range.begin)) {
      result.hasConflict = true;
      return result;
    }
    maximalEnd = std::max(maximalEnd, range.end);
    if (range.isWrite) {
      maximalWriteEnd =
          hasWrite ? std::max(maximalWriteEnd, range.end) : range.end;
      hasWrite = true;
    }
  }
  return result;
}

struct PlainMemoryConflictQuery {
  llvm::SmallVector<SyncEffectId, 2> effects;
  std::uint64_t inspectedIntervals = 0;
};

// Exact byte-interval cache of the maximal issued hazards. It stores effect
// handles but no order relation; MemorySynchronization alone decides whether
// one effect covers another and reduces read frontiers.
class PlainMemoryConflictIndex {
public:
  PlainMemoryConflictQuery query(MemoryActionRecord action) const {
    PlainMemoryConflictQuery result;
    canonicalizeMemoryActionRanges(action.byteRanges);
    auto root = intervals_.find(action.rootId);
    if (root == intervals_.end())
      return result;

    const IntervalMap &intervals = root->second->intervals;
    for (const auto &[begin, end] : action.byteRanges) {
      auto interval = intervals.find(begin);
      while (interval.valid() && interval.start() < end) {
        ++result.inspectedIntervals;
        const Hazards &hazards = interval.value();
        if (hazards.write)
          result.effects.push_back(*hazards.write);
        if (action.isWrite)
          result.effects.append(hazards.reads.begin(), hazards.reads.end());
        ++interval;
      }
    }
    llvm::sort(result.effects);
    result.effects.erase(
        std::unique(result.effects.begin(), result.effects.end()),
        result.effects.end());
    return result;
  }

  void retain(MemoryActionRecord action, SyncEffectId effect,
              MemorySynchronization &synchronization) {
    canonicalizeMemoryActionRanges(action.byteRanges);
    if (action.byteRanges.empty())
      return;
    std::unique_ptr<RootIntervals> &root = intervals_[action.rootId];
    if (!root)
      root = std::make_unique<RootIntervals>();
    for (const auto &[begin, end] : action.byteRanges)
      updateRange(*root, begin, end, action.isWrite, effect, synchronization);
  }

  bool empty() const { return intervals_.empty(); }

  std::size_t intervalCount(std::uint64_t rootId) const {
    auto root = intervals_.find(rootId);
    if (root == intervals_.end())
      return 0;
    return std::distance(root->second->intervals.begin(),
                         root->second->intervals.end());
  }

private:
  struct Hazards {
    std::optional<SyncEffectId> write;
    llvm::SmallVector<SyncEffectId, 2> reads;

    friend bool operator==(const Hazards &lhs, const Hazards &rhs) {
      return lhs.write == rhs.write && lhs.reads == rhs.reads;
    }
    friend bool operator!=(const Hazards &lhs, const Hazards &rhs) {
      return !(lhs == rhs);
    }
  };
  using IntervalMap =
      llvm::IntervalMap<std::int64_t, Hazards, 3,
                        llvm::IntervalMapHalfOpenInfo<std::int64_t>>;

  struct RootIntervals {
    IntervalMap::Allocator allocator;
    IntervalMap intervals;

    RootIntervals() : intervals(allocator) {}
  };

  struct IntervalReplacement {
    std::int64_t begin;
    std::int64_t end;
    Hazards hazards;
  };

  static void applyAccess(Hazards &hazards, bool isWrite, SyncEffectId effect,
                          MemorySynchronization &synchronization) {
    if (isWrite) {
      hazards.write = effect;
      hazards.reads.clear();
      return;
    }
    hazards.reads.push_back(effect);
    if (hazards.reads.size() > 1)
      hazards.reads = llvm::cantFail(
          synchronization.maximalHappensBeforeFrontier(hazards.reads));
  }

  static Hazards makeHazards(bool isWrite, SyncEffectId effect,
                             MemorySynchronization &synchronization) {
    Hazards hazards;
    applyAccess(hazards, isWrite, effect, synchronization);
    return hazards;
  }

  static void updateRange(RootIntervals &root, std::int64_t begin,
                          std::int64_t end, bool isWrite, SyncEffectId effect,
                          MemorySynchronization &synchronization) {
    llvm::SmallVector<IntervalReplacement, 4> existing;
    auto interval = root.intervals.find(begin);
    while (interval.valid() && interval.start() < end) {
      existing.push_back(IntervalReplacement{interval.start(), interval.stop(),
                                             interval.value()});
      interval.erase();
    }

    llvm::SmallVector<IntervalReplacement, 6> replacements;
    std::int64_t cursor = begin;
    for (const IntervalReplacement &prior : existing) {
      if (prior.begin < begin)
        replacements.push_back(
            IntervalReplacement{prior.begin, begin, prior.hazards});

      const std::int64_t overlapBegin = std::max(begin, prior.begin);
      if (cursor < overlapBegin)
        replacements.push_back(
            IntervalReplacement{cursor, overlapBegin,
                                makeHazards(isWrite, effect, synchronization)});

      const std::int64_t overlapEnd = std::min(end, prior.end);
      Hazards updated = prior.hazards;
      applyAccess(updated, isWrite, effect, synchronization);
      replacements.push_back(
          IntervalReplacement{overlapBegin, overlapEnd, std::move(updated)});
      cursor = std::max(cursor, overlapEnd);

      if (prior.end > end) {
        replacements.push_back(
            IntervalReplacement{end, prior.end, prior.hazards});
        cursor = end;
      }
    }
    if (cursor < end)
      replacements.push_back(IntervalReplacement{
          cursor, end, makeHazards(isWrite, effect, synchronization)});

    for (IntervalReplacement &replacement : replacements)
      root.intervals.insert(replacement.begin, replacement.end,
                            std::move(replacement.hazards));
  }

  llvm::DenseMap<std::uint64_t, std::unique_ptr<RootIntervals>> intervals_;
};

struct PlainMemoryActionProjection {
  std::optional<ReadyPlainMemoryAction> ready;
  llvm::SmallVector<std::string, 1> diagnostics;
  bool unsupported = false;
};

struct SimulatorState {
  ChannelMap channels;
  ChannelMap pendingChannels;
  OutputMap observedOutputs;
  OutputMap pendingObservedOutputs;
  llvm::DenseMap<mlir::Value, std::shared_ptr<MemoryValue>> memories;
  llvm::DenseMap<mlir::Value, std::uint64_t> memoryRootIds;
  llvm::DenseMap<mlir::Value, MemoryFixture> rawMemoryFixtures;
  llvm::DenseMap<mlir::Operation *, StreamSemanticState> streamStates;
  llvm::DenseSet<mlir::Operation *> failedStreamOps;
  llvm::DenseMap<mlir::Operation *, std::uint64_t> streamTrueEmissionCounts;
  llvm::DenseMap<mlir::Operation *, LoopState> carryStates;
  llvm::DenseMap<mlir::Operation *, LoopState> invariantStates;
  llvm::DenseMap<mlir::Operation *, ParallelizeState> parallelizeStates;
  llvm::DenseSet<mlir::Operation *> gateContinueStates;
  // Canonical memory order retained by a stateful actor for one activation.
  // This simulator-only state is separate from the actor's semantic state.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<SyncEffectId, 2>>
      activationMemoryOrderFrontiers;
  llvm::DenseSet<mlir::Operation *> oneShotOps;
  llvm::DenseSet<mlir::Operation *> terminalPrimitiveOps;
  llvm::DenseMap<mlir::Value, std::uint64_t> seededTokenCounts;
  llvm::SmallVector<std::string> diagnostics;
  std::map<std::string, std::uint64_t> operationFireCounts;
  std::map<std::string, std::uint64_t> modeledLibraryCalls;
  std::uint64_t nextMemoryRootId = 0;
  std::uint64_t modeledLibraryScore = 0;
  std::uint64_t eventCount = 0;
  std::uint64_t memoryAddressScore = 0;
  std::uint64_t actorMutationEpoch = 0;
  // The one graph this run simulates. Every `index` token in it resolves its
  // width against this scope, including the elements of a memory fixture.
  mlir::Operation *graphScope = nullptr;
  // A capability whose absence only the runtime values expose, such as a plain
  // conflicting access that carries no explicit causal order. The run reports
  // an unsupported capability instead of an arbitrary result or a deadlock
  // witness. Ordinary execution diagnostics never set this.
  bool runtimeUnsupportedCapability = false;
  // The causality engines this run projects its plain accesses onto. They are
  // owned indirectly so the bound reference inside MemorySynchronization stays
  // valid however this state itself is stored, and they are created only once
  // an access needs them.
  std::unique_ptr<MemoryAtomicOrder> memoryOrder;
  std::unique_ptr<MemorySynchronization> memorySync;
  PlainMemoryConflictIndex memoryActions;
  // Static operation ordinals and the subset whose token queues changed or
  // may still contain another firing. This limits pure admission projection to
  // memory actors whose readiness can have changed.
  llvm::DenseMap<mlir::Operation *, std::uint64_t> plainMemoryOperationOrder;
  std::map<std::uint64_t, mlir::Operation *> plainMemoryCandidates;
  // Execution-local cache of the plain actions and ctrl-derived order
  // frontiers admitted for the current scheduler decision. The scheduler
  // clears and derives it again before every wave.
  llvm::DenseMap<mlir::Operation *, ReadyPlainMemoryAction>
      admittedPlainMemoryActions;
  // The memory-order frontier of the firing in progress. Generic actors
  // propagate it, while memory actors publish only their admitted ctrl/action
  // frontier. This is cleared before each actor attempt.
  llvm::SmallVector<SyncEffectId, 2> firingMemoryOrderFrontier;
};

struct UnsupportedOperation {
  std::string label;
  std::string reason;
};

Token noneToken();
Token integerValueToken(std::int64_t value);
Token floatValueToken(double value);
Token boolValueToken(bool value);
llvm::Expected<unsigned> tokenTypeBitWidth(mlir::Type type);
llvm::Expected<llvm::APInt> tokenBitPattern(const Token &token,
                                            mlir::Type type);
llvm::Expected<Token> tokenFromBitPattern(const llvm::APInt &bits,
                                          mlir::Type type);
llvm::Expected<Token> parseRuntimeToken(llvm::StringRef raw, mlir::Type type,
                                        mlir::Operation *scope);
llvm::Expected<std::string> tokenToString(const Token &token, mlir::Type type,
                                          mlir::Operation *scope);
Token pointerToken(mlir::Value root, std::shared_ptr<MemoryValue> memory = {},
                   std::int64_t byteOffset = 0);
llvm::Expected<Token> tokenFromTypedAttr(mlir::TypedAttr attr);
llvm::Expected<Token> zeroToken(mlir::Type type);
llvm::Expected<Token> ensurePointerMemory(SimulatorState &state, Token token,
                                          mlir::Type elementType);
llvm::Expected<std::int64_t> gepByteOffset(mlir::LLVM::GEPOp op,
                                           llvm::ArrayRef<Token> dynamicTokens);

void mergeMemoryOrderFrontier(llvm::SmallVectorImpl<SyncEffectId> &into,
                              SyncEffectId effect);
void mergeMemoryOrderFrontier(llvm::SmallVectorImpl<SyncEffectId> &into,
                              llvm::ArrayRef<SyncEffectId> effects);

inline void
reduceMemoryOrderFrontier(SimulatorState &state,
                          llvm::SmallVectorImpl<SyncEffectId> &frontier) {
  if (frontier.size() < 2)
    return;
  assert(state.memorySync && "a memory-order witness requires its authority");
  llvm::SmallVector<SyncEffectId> reduced =
      llvm::cantFail(state.memorySync->maximalHappensBeforeFrontier(frontier));
  frontier.assign(reduced.begin(), reduced.end());
}

inline void
mergeAndReduceMemoryOrderFrontier(SimulatorState &state,
                                  llvm::SmallVectorImpl<SyncEffectId> &into,
                                  llvm::ArrayRef<SyncEffectId> effects) {
  mergeMemoryOrderFrontier(into, effects);
  reduceMemoryOrderFrontier(state, into);
}

inline void retainAndPublishActivationMemoryOrder(SimulatorState &state,
                                                  mlir::Operation *actor) {
  auto &activation = state.activationMemoryOrderFrontiers[actor];
  mergeAndReduceMemoryOrderFrontier(state, activation,
                                    state.firingMemoryOrderFrontier);
  state.firingMemoryOrderFrontier.assign(activation.begin(), activation.end());
}

inline void retireActivationMemoryOrder(SimulatorState &state,
                                        mlir::Operation *actor, bool retire) {
  if (retire)
    state.activationMemoryOrderFrontiers.erase(actor);
}

bool hasToken(ChannelMap &channels, mlir::OpOperand &operand);
Token popToken(SimulatorState &state, mlir::OpOperand &operand);
Token peekToken(ChannelMap &channels, mlir::OpOperand &operand);
void emitToken(SimulatorState &state, mlir::Value value, Token token);
void emitTokenWithMemoryOrder(SimulatorState &state, mlir::Value value,
                              Token token,
                              llvm::ArrayRef<SyncEffectId> memoryOrder);
bool recordEvent(SimulatorState &state, llvm::StringRef opName);
bool hasComputedAddress(mlir::Value value);
std::int64_t integerToken(const Token &token);
bool boolToken(const Token &token);
llvm::Expected<llvm::APInt> vectorIndexTokenBitPattern(const Token &token,
                                                       mlir::VectorType type,
                                                       mlir::Operation *scope);
// The exact value one scalar `index` token carries at the resolved width. An
// index has no width in its MLIR type, so it is normalized here instead of
// through `tokenBitPattern`.
llvm::Expected<llvm::APInt> indexTokenBitPattern(const Token &token,
                                                 unsigned width);
Token indexToken(const llvm::APInt &value);
llvm::Expected<std::int64_t> byteSizeOfType(mlir::Type type,
                                            mlir::Operation *scope);

// The host element slot one semantic address names. `address` is exact at its
// own width and becomes a host index only after the sign and range checks.
std::optional<std::size_t> resolveElementIndex(const MemoryView &view,
                                               const llvm::APInt &address,
                                               SimulatorState &state,
                                               mlir::Operation *scope,
                                               llvm::StringRef opName);
std::optional<std::size_t> resolveElementIndex(const MemoryView &view,
                                               const Token &addr,
                                               SimulatorState &state,
                                               mlir::Operation *scope,
                                               llvm::StringRef opName);
std::optional<Token> readMemoryElement(const MemoryView &view,
                                       std::size_t index, SimulatorState &state,
                                       llvm::StringRef opName);
void writeMemoryElement(const MemoryView &view, std::size_t index, Token value);
void commitDataflowMemoryWrite(const MemoryView &view,
                               const DataflowMemoryWrite &write);

PlainMemoryActionProjection
projectReadyPlainMemoryAction(mlir::Operation *op, SimulatorState &state);
bool isSupportedLLVMCall(mlir::LLVM::CallOp op);
bool executeCmsisNNVecMatMultTS8(mlir::LLVM::CallOp op, SimulatorState &state,
                                 llvm::ArrayRef<Token> operands, Token &result);
bool isSupportedPointerICmp(mlir::LLVM::ICmpOp op);
llvm::Expected<Token> evaluatePointerICmp(mlir::LLVM::ICmpOp op,
                                          const Token &lhs, const Token &rhs);
llvm::Expected<PrimitiveValue> primitiveValueFromToken(const Token &token,
                                                       mlir::Type type);
llvm::Expected<Token> tokenFromPrimitiveValue(const PrimitiveValue &value,
                                              mlir::Type type);
std::string primitivePredicate(mlir::Operation *op);
std::string primitiveOperationName(mlir::Operation *op);
llvm::Expected<PrimitiveOperationDescriptor>
primitiveDescriptor(mlir::Operation *op, llvm::StringRef predicate,
                    mlir::Value result);
llvm::Expected<PrimitiveOperationDescriptor>
primitiveDescriptor(mlir::Operation *op, llvm::StringRef predicate,
                    mlir::Type resultType, mlir::Type operandType);
llvm::Error validatePrimitiveTokenTypes(mlir::Operation *op,
                                        mlir::Value result);
llvm::Expected<Token> evaluatePrimitiveToken(mlir::Operation *op,
                                             mlir::Value result,
                                             llvm::ArrayRef<Token> inputTokens);

bool executeLLVMMemcpy(mlir::LLVM::MemcpyOp op, SimulatorState &state,
                       const Token &dst, const Token &src, const Token &len);
bool isPointerSelect(mlir::LLVM::SelectOp op);
std::optional<Token> evaluatePointerSelect(mlir::LLVM::SelectOp op,
                                           const Token &condition,
                                           const Token &trueValue,
                                           const Token &falseValue,
                                           SimulatorState &state);
bool fireActorOperation(mlir::Operation *op, SimulatorState &state);
std::optional<UnsupportedOperation>
unsupportedActorOperation(mlir::Operation *op);

std::string unsupportedOperationLabel(mlir::Operation *op);

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

#endif // LOOM_LIB_SIMULATOR_DFGSIMULATORINTERNAL_H

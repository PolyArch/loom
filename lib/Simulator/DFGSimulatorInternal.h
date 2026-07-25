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
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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

/// Dense execution-local handle to one immutable memory-order frontier owned
/// by the run's arena. A default handle is the empty frontier, so a token that
/// never observed memory order costs one word and no allocation. The handle
/// names a set of effects and no order between them; MemorySynchronization
/// remains the only authority that relates two effects.
class MemoryOrderFrontierId {
public:
  constexpr MemoryOrderFrontierId() = default;
  explicit constexpr MemoryOrderFrontierId(std::uint32_t value)
      : value_(value) {}

  constexpr std::uint32_t value() const { return value_; }
  constexpr bool empty() const { return value_ == 0; }

  friend constexpr bool operator==(MemoryOrderFrontierId lhs,
                                   MemoryOrderFrontierId rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(MemoryOrderFrontierId lhs,
                                   MemoryOrderFrontierId rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t value_ = 0;
};

/// Execution-local owner of every memory-order frontier a run publishes.
///
/// One frontier is stored once, in canonical shape (ascending identity, no
/// repeats), and every token that observes it holds only its handle. Equal
/// content always interns to one handle, so a firing that publishes several
/// results shares a single stored frontier instead of one copy per result.
///
/// Elements live in bump-allocated chunks that are never reallocated, so an
/// ArrayRef handed to MemorySynchronization stays valid while the arena keeps
/// growing. The arena stores sets of effect handles and no relation between
/// them: it is a representation, never a second happens-before authority.
class MemoryOrderFrontierArena {
public:
  MemoryOrderFrontierArena() { entries_.push_back(Entry{nullptr, 0}); }

  // Entries point into this arena's own chunks, so a copied arena would keep
  // referencing the original's storage. The run owns its arena in place for
  // its whole lifetime, so no move is defined either.
  MemoryOrderFrontierArena(const MemoryOrderFrontierArena &) = delete;
  MemoryOrderFrontierArena &
  operator=(const MemoryOrderFrontierArena &) = delete;

  llvm::ArrayRef<SyncEffectId> elements(MemoryOrderFrontierId id) const {
    const Entry &entry = entries_[id.value()];
    return llvm::ArrayRef<SyncEffectId>(entry.data, entry.size);
  }

  /// Interns the frontier of one effect.
  MemoryOrderFrontierId internCanonical(SyncEffectId effect) {
    return internCanonical(llvm::ArrayRef<SyncEffectId>(effect));
  }

  /// Interns one frontier that is already ascending and free of repeats.
  MemoryOrderFrontierId internCanonical(llvm::ArrayRef<SyncEffectId> elements) {
    assert(std::adjacent_find(elements.begin(), elements.end(),
                              [](SyncEffectId lhs, SyncEffectId rhs) {
                                return !(lhs < rhs);
                              }) == elements.end() &&
           "a canonical frontier is ascending and free of repeats");
    if (elements.empty())
      return MemoryOrderFrontierId();
    const std::uint64_t key = hashOf(elements);
    llvm::SmallVector<MemoryOrderFrontierId, 1> &bucket = interned_[key];
    for (MemoryOrderFrontierId candidate : bucket)
      if (this->elements(candidate) == elements)
        return candidate;

    if (entries_.size() >= std::numeric_limits<std::uint32_t>::max())
      llvm::report_fatal_error(
          "the simulator retained more than 2^32 distinct memory-order "
          "frontiers in one run; the frontier handle space is exhausted");
    const MemoryOrderFrontierId id(static_cast<std::uint32_t>(entries_.size()));
    entries_.push_back(Entry{store(elements), elements.size()});
    bucket.push_back(id);
    return id;
  }

private:
  struct Entry {
    const SyncEffectId *data;
    std::size_t size;
  };

  // Bump-allocated storage keeps one frontier contiguous and never moves it:
  // a chunk is allocated once at a fixed capacity and only ever filled, so a
  // stored frontier keeps its address for the lifetime of the arena.
  static constexpr std::size_t kChunkElements = 1024;

  // A chunk reserves its capacity once and is only ever appended to, so it
  // never reallocates and the addresses it hands out stay valid.
  using Chunk = std::vector<SyncEffectId>;

  // Deterministic incremental hash over the effect identities themselves, so a
  // frontier of any width is keyed without a temporary copy. The constants are
  // the 64-bit FNV-1a basis and prime, mixed per effect.
  static std::uint64_t hashOf(llvm::ArrayRef<SyncEffectId> elements) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (SyncEffectId effect : elements) {
      std::uint64_t value = effect.value();
      for (unsigned byte = 0; byte < sizeof(value); ++byte) {
        hash ^= value & 0xffULL;
        hash *= 0x100000001b3ULL;
        value >>= 8;
      }
    }
    // Clearing the top bits keeps the key away from the DenseMap empty and
    // tombstone identities without weakening the hash in any realistic range.
    return hash >> 2;
  }

  const SyncEffectId *store(llvm::ArrayRef<SyncEffectId> elements) {
    if (chunks_.empty() ||
        chunks_.back().size() + elements.size() > chunks_.back().capacity()) {
      chunks_.emplace_back();
      chunks_.back().reserve(std::max(kChunkElements, elements.size()));
    }
    Chunk &chunk = chunks_.back();
    const SyncEffectId *begin = chunk.data() + chunk.size();
    chunk.insert(chunk.end(), elements.begin(), elements.end());
    assert(chunk.size() <= chunk.capacity() &&
           "a frontier chunk reallocated and invalidated stored frontiers");
    return begin;
  }

  std::vector<Entry> entries_;
  std::vector<Chunk> chunks_;
  llvm::DenseMap<std::uint64_t, llvm::SmallVector<MemoryOrderFrontierId, 1>>
      interned_;
};

/// Memory order that is still being accumulated and has not been published.
///
/// This is transient mutable state, never an arena entry: only a frontier that
/// a token actually carries is interned. Accumulating in place keeps a partial
/// union out of the arena, so a group that absorbs k effects one firing at a
/// time costs one growing buffer instead of k retained frontiers.
///
/// A frontier is a set of effects, so the elements hold each one once and
/// `members_` indexes exactly them. Merging content the accumulator already
/// holds therefore costs a lookup instead of a copy, and reconverging one
/// frontier through k inputs retains it once rather than k times.
///
/// The reduced flag and the optional published handle are memos of exactly
/// this content, both dropped whenever the elements grow, so they can never
/// disagree with the elements. None of the memos here relates two effects and
/// none is an authority: MemorySynchronization alone reduces a frontier.
class MemoryOrderAccumulator {
public:
  llvm::ArrayRef<SyncEffectId> elements() const { return elements_; }
  bool empty() const { return elements_.empty(); }

  void clear() {
    elements_.clear();
    members_.clear();
    absorbed_.clear();
    reduced_ = false;
    published_.reset();
  }

  /// Merges effects, keeping the elements a set. Content this accumulator
  /// already holds changes nothing, so its memos survive an append that adds
  /// no member.
  void append(llvm::ArrayRef<SyncEffectId> effects) {
    bool grew = false;
    for (SyncEffectId effect : effects)
      if (members_.insert(effect.value()).second) {
        elements_.push_back(effect);
        grew = true;
      }
    if (!grew)
      return;
    reduced_ = false;
    published_.reset();
  }

  /// Appends one stored frontier and records that this accumulator already
  /// represents it, so a token carrying exactly that frontier needs no merge.
  /// Absorbing the same handle again is the same content, so it resolves
  /// against the memo without reading the frontier at all.
  void absorb(llvm::ArrayRef<SyncEffectId> effects,
              MemoryOrderFrontierId frontier) {
    if (absorbed_.insert(frontier.value()).second)
      append(effects);
  }

  /// True when this accumulator already absorbed `frontier`, so re-merging it
  /// would add nothing. Reduction only drops effects that a retained maximal
  /// member happens-after, so an absorbed frontier stays covered.
  bool hasAbsorbed(MemoryOrderFrontierId frontier) const {
    return frontier.empty() || absorbed_.contains(frontier.value());
  }

  /// Folds another accumulator's elements and absorbed handles into this one.
  /// Only the members this accumulator does not already hold are retained, so
  /// a fully or partially overlapping other costs no duplicate storage, and
  /// one that adds no member leaves this accumulator's memos untouched.
  void absorbAll(const MemoryOrderAccumulator &other) {
    absorbed_.insert(other.absorbed_.begin(), other.absorbed_.end());
    append(other.elements_);
  }

  /// True once the elements are the canonical maximal members the authority
  /// returned, so a further reduction of the same content is redundant.
  bool isReduced() const { return reduced_; }

  /// Replaces the elements with the reduced shape the authority returned.
  void adoptReduced(llvm::ArrayRef<SyncEffectId> effects) {
    elements_.assign(effects.begin(), effects.end());
    members_.clear();
    for (SyncEffectId effect : elements_)
      members_.insert(effect.value());
    reduced_ = true;
  }

  /// Records the handle interned for exactly the current reduced elements.
  void markPublished(MemoryOrderFrontierId frontier) {
    assert(reduced_ && "only a reduced frontier is interned");
    absorbed_.insert(frontier.value());
    published_ = frontier;
  }

  /// The frontier this accumulator already resolved, if it published one.
  std::optional<MemoryOrderFrontierId> published() const { return published_; }

private:
  llvm::SmallVector<SyncEffectId, 4> elements_;
  // Membership index of exactly `elements_`, derived from them and rebuilt
  // with them. It keeps the elements a set and relates nothing.
  llvm::SmallDenseSet<std::uint64_t, 4> members_;
  // Handles this accumulator already merged. A later reduction may drop a
  // dominated effect from `elements_`, so the invariant is coverage rather
  // than literal presence: what the handle contributed is still a member or
  // is happens-before one, which is why re-merging it cannot change the
  // reduced result. A memo of merged content, cleared with the elements, and
  // never a relation of its own.
  llvm::SmallDenseSet<std::uint32_t, 4> absorbed_;
  bool reduced_ = false;
  std::optional<MemoryOrderFrontierId> published_;
};

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
  MemoryOrderFrontierId memoryOrder;
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
  // current group. Only the final firing publishes their union, so this stays
  // an unpublished accumulator and never interns a partial group.
  MemoryOrderAccumulator phaseFrontier;
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

/// Merges the ranges into the ascending, non-touching cover of the same bytes.
void canonicalizeMemoryActionRanges(
    llvm::SmallVectorImpl<std::pair<std::int64_t, std::int64_t>> &ranges);

struct ReadyPlainMemoryAction {
  MemoryActionRecord action;
  llvm::SmallVector<SyncEffectId, 2> ctrlFrontier;
};

// Exact byte-interval cache of the maximal issued hazards. It stores effect
// handles but no order relation; MemorySynchronization alone decides whether
// one effect covers another and reduces read frontiers.
class PlainMemoryConflictIndex {
public:
  /// The maximal issued hazards `action` meets, without deciding whether any
  /// of them is ordered before it.
  llvm::SmallVector<SyncEffectId> query(MemoryActionRecord action) const;

  /// Records one issued access as the new maximal hazard of its byte ranges.
  void retain(MemoryActionRecord action, SyncEffectId effect,
              MemorySynchronization &synchronization);

  bool empty() const { return intervals_.empty(); }

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
                          MemorySynchronization &synchronization);
  static Hazards makeHazards(bool isWrite, SyncEffectId effect,
                             MemorySynchronization &synchronization);
  static void updateRange(RootIntervals &root, std::int64_t begin,
                          std::int64_t end, bool isWrite, SyncEffectId effect,
                          MemorySynchronization &synchronization);

  llvm::DenseMap<std::uint64_t, std::unique_ptr<RootIntervals>> intervals_;
};

struct PlainMemoryActionProjection {
  std::optional<ReadyPlainMemoryAction> ready;
  llvm::SmallVector<std::string, 1> diagnostics;
};

/// The closed set of runtime failures a run can retain. A run holds at most
/// one, recorded once, so the kind is decided in a single place instead of
/// being reconciled from flags. A retained failure overrides the ordinary
/// lifecycle classification the driver would otherwise derive; it does not
/// name that lifecycle, so static invalidity and a run that merely stopped
/// remain the driver's own terminals. Ordinary execution diagnostics leave it
/// `None`.
enum class RunFailure {
  None,
  /// A capability whose absence only runtime values expose, such as a plain
  /// conflicting access that carries no explicit causal order. The exact model
  /// reports the missing capability instead of an arbitrary result or a
  /// deadlock witness.
  UnsupportedCapability,
  /// An invariant this run's own providers or its finalized program guarantee,
  /// broken during execution. It is an internal failure of the simulator or
  /// its input, never a capability the exact model lacks and never static
  /// invalidity, which is rejected before execution state exists.
  ProviderInvariant,
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
  // Memory-order union retained by a stateful actor for one activation. The
  // union rests here between the activation's firings, keeping its reduction
  // and publication memos, and moves into the firing slot for the firing that
  // publishes it. This simulator-only state is separate from the actor's
  // semantic state.
  llvm::DenseMap<mlir::Operation *, MemoryOrderAccumulator>
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
  // The runtime failure this run retained, if any. Execution stops at the
  // failure, so at most one is ever recorded.
  RunFailure failure = RunFailure::None;
  // The causality engines this run projects its plain accesses onto. They are
  // owned indirectly so the bound reference inside MemorySynchronization stays
  // valid however this state itself is stored, and they are created only once
  // an access needs them.
  std::unique_ptr<MemoryAtomicOrder> memoryOrder;
  std::unique_ptr<MemorySynchronization> memorySync;
  PlainMemoryConflictIndex memoryActions;
  // The one owner of every frontier this run publishes. Tokens and retained
  // actor state reference it by handle.
  MemoryOrderFrontierArena memoryOrderFrontiers;
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
  MemoryOrderAccumulator firingMemoryOrderFrontier;
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

/// Canonicalizes `frontier` and reduces it to its maximal members. The
/// authority owns the reduction; this only shapes the request.
void reduceMemoryOrderFrontier(SimulatorState &state,
                               llvm::SmallVectorImpl<SyncEffectId> &frontier);

/// Reduces an accumulator in place, without interning it. Order that no token
/// carries stays out of the arena but still stops growing.
void reduceMemoryOrder(SimulatorState &state,
                       MemoryOrderAccumulator &accumulator);

/// Resolves the frontier an accumulator publishes, reducing and interning it
/// at most once however many tokens go on to carry it.
MemoryOrderFrontierId publishMemoryOrder(SimulatorState &state,
                                         MemoryOrderAccumulator &accumulator);

/// The frontier one emitted token carries: the firing's own published
/// frontier, merged with order the token brought and the firing never
/// consumed.
MemoryOrderFrontierId publishFiredMemoryOrder(SimulatorState &state,
                                              MemoryOrderFrontierId carried);

/// Folds the firing's consumed order into the actor's retained activation
/// union and hands the union to the firing slot, memos included, so the
/// firing's emissions publish it. A firing that consumed nothing leaves the
/// union's memos untouched, and an activation that emits nothing is never
/// reduced or interned.
void retainAndPublishActivationMemoryOrder(SimulatorState &state,
                                           mlir::Operation *actor);

/// Returns the published union to its activation slot for the next firing of
/// the same activation, or erases it when the activation retired. A retired
/// union keeps nothing; its frontier lives on in the arena only if a token
/// carries it.
void releaseActivationMemoryOrder(SimulatorState &state, mlir::Operation *actor,
                                  bool retire);

bool hasToken(ChannelMap &channels, mlir::OpOperand &operand);
Token popToken(SimulatorState &state, mlir::OpOperand &operand);
Token peekToken(ChannelMap &channels, mlir::OpOperand &operand);
void emitToken(SimulatorState &state, mlir::Value value, Token token);
void emitTokenWithMemoryOrder(SimulatorState &state, mlir::Value value,
                              Token token, MemoryOrderFrontierId memoryOrder);
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

/// The plain action one candidate would issue, derived from peeked inputs
/// alone. It answers only what the access covers and what ctrl order it
/// carries; legality the finalized program already owns is not re-derived.
PlainMemoryActionProjection
projectReadyPlainMemoryAction(mlir::Operation *op, SimulatorState &state);

/// Admits every ready plain action of one scheduler decision, or rejects the
/// whole decision. False leaves nothing admitted, so no access of a rejected
/// decision can still fire.
bool admitReadyPlainMemoryActions(SimulatorState &state);

/// Projects a retained RunFailure onto the report status and returns true, so
/// the caller knows the run exports no observation. A run that retained no
/// failure keeps the driver's own lifecycle classification and returns false.
bool applyRunFailureTerminal(const SimulatorState &state,
                             DFGSimulationReport &report);

/// Records the graph memory a finished run may still export.
llvm::Error captureFinalMemoryState(dataflow::GraphOp graph,
                                    SimulatorState &state,
                                    DFGSimulationReport &report);

/// True when a vector group actor still holds lanes it never flushed, which
/// leaves the run's internal state incomplete.
bool hasPendingVectorGroups(SimulatorState &state);

/// Re-encodes one serialized memory observation as the fixture text a further
/// invocation of the same graph consumes.
llvm::Expected<std::string>
memoryFixtureFromSerializedValues(llvm::ArrayRef<std::string> values);

/// The registered cost model's score for one inventory of fired operations.
/// An operation the model does not know contributes a diagnostic instead of a
/// guessed score.
std::uint64_t estimateWeightedOperationScore(
    const std::map<std::string, std::uint64_t> &operationFireCounts,
    llvm::SmallVectorImpl<std::string> &diagnostics);

/// Copies the run's counters, derived cost scores, and distinct execution
/// diagnostics into the report.
void projectRunObservations(SimulatorState &state, DFGSimulationReport &report);

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

bool fireLoad(dataflow::LoadOp op, SimulatorState &state);
bool fireStore(dataflow::StoreOp op, SimulatorState &state);
bool fireLLVMLoad(mlir::LLVM::LoadOp op, SimulatorState &state);
bool fireLLVMStore(mlir::LLVM::StoreOp op, SimulatorState &state);
bool fireLLVMMemcpy(mlir::LLVM::MemcpyOp op, SimulatorState &state);
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

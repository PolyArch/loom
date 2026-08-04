#ifndef LOOM_LIB_SIMULATOR_DFGSIMULATORINTERNAL_H
#define LOOM_LIB_SIMULATOR_DFGSIMULATORINTERNAL_H

#include "Simulator/DFGSimulator.h"
#include "Simulator/MemorySynchronization.h"
#include "Simulator/SimulationArtifacts.h"

#include "Common/PointerLayout.h"
#include "Dataflow/IR/DataflowActorSemantics.h"
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
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::sim {
namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail {

struct ResolvedLaunchContext;
struct MemoryValue;
struct SimulatorState;
struct ActorExecutionPlan;

struct MemoryView {
  std::shared_ptr<MemoryValue> memory;
  mlir::Value root;
  std::int64_t byteOffset = 0;
  mlir::Type elementType;
};

struct PointerValue {
  std::shared_ptr<MemoryValue> memory;
  std::uint64_t objectOrdinal = 0;
  std::uint32_t addressSpace = 0;
  llvm::APInt byteOffset;
  llvm::APInt representation;
};

using ExtendedTokenPayload =
    std::variant<llvm::APInt, MemoryView, PointerValue>;

enum class TokenKind {
  None,
  Integer,
  Float,
  Bool,
  Vector,
  Pointer,
  MemoryCapability,
};

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
/// Tokens share immutable effect leaves and union nodes. A growing all-of
/// frontier therefore retains one new node instead of copying every prior
/// effect into each prefix. Consumers materialize a canonical effect set only
/// when the memory-order authority needs one. The union graph stores no order
/// between effects and is never a second happens-before authority.
class MemoryOrderFrontierArena {
public:
  MemoryOrderFrontierArena() { entries_.push_back(Entry{nullptr, nullptr, 0}); }

  // Entries point into this arena's own chunks, so a copied arena would keep
  // referencing the original's storage. The run owns its arena in place for
  // its whole lifetime, so no move is defined either.
  MemoryOrderFrontierArena(const MemoryOrderFrontierArena &) = delete;
  MemoryOrderFrontierArena &
  operator=(const MemoryOrderFrontierArena &) = delete;

  /// Interns the frontier of one effect.
  MemoryOrderFrontierId internCanonical(SyncEffectId effect) {
    if (effect.value() >= std::numeric_limits<std::uint32_t>::max())
      llvm::report_fatal_error(
          "a memory effect cannot be represented by the simulator frontier "
          "handle domain");
    const std::size_t ordinal = static_cast<std::size_t>(effect.value());
    if (ordinal >= singletonFrontiers_.size())
      singletonFrontiers_.resize(ordinal + 1);
    MemoryOrderFrontierId &cached = singletonFrontiers_[ordinal];
    if (!cached.empty())
      return cached;
    cached = storeEffectFrontier(llvm::ArrayRef<SyncEffectId>(effect));
    return cached;
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
    if (elements.size() == 1)
      return internCanonical(elements.front());
    const std::uint64_t key = hashEffects(elements);
    llvm::SmallVector<MemoryOrderFrontierId, 1> &bucket = effectInterned_[key];
    for (MemoryOrderFrontierId candidate : bucket)
      if (effectElements(candidate) == elements)
        return candidate;

    const MemoryOrderFrontierId id = storeEffectFrontier(elements);
    bucket.push_back(id);
    return id;
  }

  /// Interns an unordered union of already-published frontiers. Empty and
  /// duplicate handles disappear, and one remaining handle is returned
  /// directly. The node shape is a cache representation only; effect identity
  /// is recovered by appendCanonicalEffects.
  MemoryOrderFrontierId
  internUnion(llvm::ArrayRef<MemoryOrderFrontierId> frontiers) {
    llvm::SmallVector<MemoryOrderFrontierId, 4> canonical;
    canonical.reserve(frontiers.size());
    for (MemoryOrderFrontierId frontier : frontiers)
      if (!frontier.empty())
        canonical.push_back(frontier);
    llvm::sort(canonical,
               [](MemoryOrderFrontierId lhs, MemoryOrderFrontierId rhs) {
                 return lhs.value() < rhs.value();
               });
    canonical.erase(std::unique(canonical.begin(), canonical.end()),
                    canonical.end());
    if (canonical.empty())
      return MemoryOrderFrontierId();
    if (canonical.size() == 1)
      return canonical.front();

    const std::uint64_t key = hashFrontiers(canonical);
    llvm::SmallVector<MemoryOrderFrontierId, 1> &bucket = unionInterned_[key];
    for (MemoryOrderFrontierId candidate : bucket)
      if (unionChildren(candidate) ==
          llvm::ArrayRef<MemoryOrderFrontierId>(canonical))
        return candidate;

    const MemoryOrderFrontierId id = storeUnion(canonical);
    bucket.push_back(id);
    return id;
  }

  /// Appends the frontier's ascending, duplicate-free effect set. Traversal is
  /// iterative so a long-running loop cannot consume the host call stack.
  void
  appendCanonicalEffects(MemoryOrderFrontierId frontier,
                         llvm::SmallVectorImpl<SyncEffectId> &effects) const {
    if (frontier.empty())
      return;
    llvm::SmallBitVector visited(entries_.size());
    llvm::SmallVector<MemoryOrderFrontierId, 16> worklist{frontier};
    while (!worklist.empty()) {
      const MemoryOrderFrontierId current = worklist.pop_back_val();
      if (visited.test(current.value()))
        continue;
      visited.set(current.value());
      const Entry &entry = entries_[current.value()];
      if (entry.effects) {
        effects.append(entry.effects, entry.effects + entry.size);
        continue;
      }
      worklist.append(entry.frontiers, entry.frontiers + entry.size);
    }
    llvm::sort(effects);
    effects.erase(std::unique(effects.begin(), effects.end()), effects.end());
  }

  std::size_t retainedEffectReferences() const {
    return retainedEffectReferences_;
  }

private:
  struct Entry {
    const SyncEffectId *effects;
    const MemoryOrderFrontierId *frontiers;
    std::size_t size;
  };

  // Bump-allocated storage keeps one frontier contiguous and never moves it:
  // a chunk is allocated once at a fixed capacity and only ever filled, so a
  // stored frontier keeps its address for the lifetime of the arena.
  static constexpr std::size_t kChunkElements = 1024;

  // A chunk reserves its capacity once and is only ever appended to, so it
  // never reallocates and the addresses it hands out stay valid.
  using EffectChunk = std::vector<SyncEffectId>;
  using FrontierChunk = std::vector<MemoryOrderFrontierId>;

  // Deterministic incremental hash over the effect identities themselves, so a
  // frontier of any width is keyed without a temporary copy. The constants are
  // the 64-bit FNV-1a basis and prime, mixed per effect.
  static std::uint64_t hashEffects(llvm::ArrayRef<SyncEffectId> elements) {
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

  static std::uint64_t
  hashFrontiers(llvm::ArrayRef<MemoryOrderFrontierId> frontiers) {
    std::uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for (MemoryOrderFrontierId frontier : frontiers) {
      std::uint32_t value = frontier.value();
      for (unsigned byte = 0; byte < sizeof(value); ++byte) {
        hash ^= value & 0xffU;
        hash *= 0x100000001b3ULL;
        value >>= 8;
      }
    }
    return hash >> 2;
  }

  llvm::ArrayRef<SyncEffectId> effectElements(MemoryOrderFrontierId id) const {
    const Entry &entry = entries_[id.value()];
    return entry.effects
               ? llvm::ArrayRef<SyncEffectId>(entry.effects, entry.size)
               : llvm::ArrayRef<SyncEffectId>();
  }

  llvm::ArrayRef<MemoryOrderFrontierId>
  unionChildren(MemoryOrderFrontierId id) const {
    const Entry &entry = entries_[id.value()];
    return entry.frontiers ? llvm::ArrayRef<MemoryOrderFrontierId>(
                                 entry.frontiers, entry.size)
                           : llvm::ArrayRef<MemoryOrderFrontierId>();
  }

  const SyncEffectId *storeEffects(llvm::ArrayRef<SyncEffectId> elements) {
    if (effectChunks_.empty() || effectChunks_.back().size() + elements.size() >
                                     effectChunks_.back().capacity()) {
      effectChunks_.emplace_back();
      effectChunks_.back().reserve(std::max(kChunkElements, elements.size()));
    }
    EffectChunk &chunk = effectChunks_.back();
    const SyncEffectId *begin = chunk.data() + chunk.size();
    chunk.insert(chunk.end(), elements.begin(), elements.end());
    assert(chunk.size() <= chunk.capacity() &&
           "a frontier chunk reallocated and invalidated stored frontiers");
    return begin;
  }

  const MemoryOrderFrontierId *
  storeFrontiers(llvm::ArrayRef<MemoryOrderFrontierId> frontiers) {
    if (frontierChunks_.empty() ||
        frontierChunks_.back().size() + frontiers.size() >
            frontierChunks_.back().capacity()) {
      frontierChunks_.emplace_back();
      frontierChunks_.back().reserve(
          std::max(kChunkElements, frontiers.size()));
    }
    FrontierChunk &chunk = frontierChunks_.back();
    const MemoryOrderFrontierId *begin = chunk.data() + chunk.size();
    chunk.insert(chunk.end(), frontiers.begin(), frontiers.end());
    assert(chunk.size() <= chunk.capacity() &&
           "a frontier chunk reallocated and invalidated stored unions");
    return begin;
  }

  MemoryOrderFrontierId nextId() const {
    if (entries_.size() >= std::numeric_limits<std::uint32_t>::max())
      llvm::report_fatal_error(
          "the simulator retained more than 2^32 distinct memory-order "
          "frontiers in one run; the frontier handle space is exhausted");
    return MemoryOrderFrontierId(static_cast<std::uint32_t>(entries_.size()));
  }

  MemoryOrderFrontierId
  storeEffectFrontier(llvm::ArrayRef<SyncEffectId> elements) {
    const MemoryOrderFrontierId id = nextId();
    entries_.push_back(Entry{storeEffects(elements), nullptr, elements.size()});
    retainedEffectReferences_ += elements.size();
    return id;
  }

  MemoryOrderFrontierId
  storeUnion(llvm::ArrayRef<MemoryOrderFrontierId> frontiers) {
    const MemoryOrderFrontierId id = nextId();
    entries_.push_back(
        Entry{nullptr, storeFrontiers(frontiers), frontiers.size()});
    return id;
  }

  std::vector<Entry> entries_;
  std::vector<EffectChunk> effectChunks_;
  std::vector<FrontierChunk> frontierChunks_;
  std::vector<MemoryOrderFrontierId> singletonFrontiers_;
  llvm::DenseMap<std::uint64_t, llvm::SmallVector<MemoryOrderFrontierId, 1>>
      effectInterned_;
  llvm::DenseMap<std::uint64_t, llvm::SmallVector<MemoryOrderFrontierId, 1>>
      unionInterned_;
  std::size_t retainedEffectReferences_ = 0;
};

/// Memory order that is still being accumulated and has not been published.
///
/// This is transient mutable state, never an arena entry. It accumulates
/// immutable frontier handles, not copied effect sets. Publication interns one
/// union node and collapses the accumulator back to that handle. The absorbed
/// memo prevents reconvergent token flow from adding the same handle twice;
/// none of this state relates two effects or competes with
/// MemorySynchronization.
class MemoryOrderAccumulator {
public:
  MemoryOrderAccumulator() = default;

  MemoryOrderAccumulator(const MemoryOrderAccumulator &other)
      : frontiers_(other.frontiers_), absorbed_(other.absorbed_),
        published_(other.published_) {
    rebuildFrontierIndex();
    rebuildAbsorbedIndex();
  }

  MemoryOrderAccumulator &operator=(const MemoryOrderAccumulator &other) {
    if (this == &other)
      return *this;
    frontiers_ = other.frontiers_;
    absorbed_ = other.absorbed_;
    published_ = other.published_;
    rebuildFrontierIndex();
    rebuildAbsorbedIndex();
    return *this;
  }

  MemoryOrderAccumulator(MemoryOrderAccumulator &&) = default;
  MemoryOrderAccumulator &operator=(MemoryOrderAccumulator &&) = default;

  llvm::ArrayRef<MemoryOrderFrontierId> frontiers() const { return frontiers_; }
  bool empty() const { return frontiers_.empty(); }

  void clear() {
    if (frontiers_.empty() && absorbed_.empty() && !published_)
      return;
    frontiers_.clear();
    frontierIndex_.reset();
    absorbed_.clear();
    absorbedIndex_.reset();
    published_.reset();
  }

  /// Adds one immutable frontier handle. Absorbing the same handle again is
  /// the same content and leaves a published union reusable.
  void absorb(MemoryOrderFrontierId frontier) {
    if (frontier.empty())
      return;
    if (frontiers_.empty() && absorbed_.empty()) {
      frontiers_.push_back(frontier);
      absorbed_.push_back(frontier.value());
      published_ = frontier;
      return;
    }
    if (!insertAbsorbed(frontier.value()) || !insertFrontier(frontier))
      return;
    frontiers_.push_back(frontier);
    published_.reset();
  }

  /// True when this accumulator already absorbed `frontier`, so re-merging it
  /// would add nothing. Reduction only drops effects that a retained maximal
  /// member happens-after, so an absorbed frontier stays covered.
  bool hasAbsorbed(MemoryOrderFrontierId frontier) const {
    return frontier.empty() || containsAbsorbed(frontier.value());
  }

  /// Folds another accumulator's elements and absorbed handles into this one.
  /// One empty accumulator adopts the other's publication memo directly.
  void absorbAll(const MemoryOrderAccumulator &other) {
    if (this == &other)
      return;
    if (frontiers_.empty() && absorbed_.empty()) {
      *this = other;
      return;
    }
    for (std::uint32_t frontier : other.absorbed_)
      (void)insertAbsorbed(frontier);
    bool grew = false;
    for (MemoryOrderFrontierId frontier : other.frontiers_)
      if (insertFrontier(frontier)) {
        frontiers_.push_back(frontier);
        grew = true;
      }
    if (grew)
      published_.reset();
  }

  /// Records the immutable union for the current components. Retaining one
  /// handle keeps every subsequent forwarding operation constant-size.
  void markPublished(MemoryOrderFrontierId frontier) {
    assert(!frontier.empty() && "a nonempty accumulator publishes a handle");
    frontiers_.clear();
    frontierIndex_.reset();
    frontiers_.push_back(frontier);
    (void)insertAbsorbed(frontier.value());
    published_ = frontier;
  }

  /// The frontier this accumulator already resolved, if it published one.
  std::optional<MemoryOrderFrontierId> published() const { return published_; }

private:
  static constexpr std::size_t kLinearMembershipLimit = 8;

  bool insertFrontier(MemoryOrderFrontierId frontier) {
    const std::uint32_t value = frontier.value();
    if (frontierIndex_)
      return frontierIndex_->insert(value).second;
    if (std::find(frontiers_.begin(), frontiers_.end(), frontier) !=
        frontiers_.end())
      return false;
    if (frontiers_.size() == kLinearMembershipLimit) {
      frontierIndex_ = std::make_unique<llvm::DenseSet<std::uint32_t>>();
      frontierIndex_->reserve(frontiers_.size() * 2);
      for (MemoryOrderFrontierId existing : frontiers_)
        frontierIndex_->insert(existing.value());
      frontierIndex_->insert(value);
    }
    return true;
  }

  bool insertAbsorbed(std::uint32_t frontier) {
    if (absorbedIndex_)
      return absorbedIndex_->insert(frontier).second;
    if (std::find(absorbed_.begin(), absorbed_.end(), frontier) !=
        absorbed_.end())
      return false;
    if (absorbed_.size() == kLinearMembershipLimit) {
      absorbedIndex_ = std::make_unique<llvm::DenseSet<std::uint32_t>>();
      absorbedIndex_->reserve(absorbed_.size() * 2);
      absorbedIndex_->insert(absorbed_.begin(), absorbed_.end());
      absorbedIndex_->insert(frontier);
    }
    absorbed_.push_back(frontier);
    return true;
  }

  bool containsAbsorbed(std::uint32_t frontier) const {
    if (absorbedIndex_)
      return absorbedIndex_->contains(frontier);
    return std::find(absorbed_.begin(), absorbed_.end(), frontier) !=
           absorbed_.end();
  }

  void rebuildFrontierIndex() {
    frontierIndex_.reset();
    if (frontiers_.size() <= kLinearMembershipLimit)
      return;
    frontierIndex_ = std::make_unique<llvm::DenseSet<std::uint32_t>>();
    frontierIndex_->reserve(frontiers_.size() * 2);
    for (MemoryOrderFrontierId frontier : frontiers_)
      frontierIndex_->insert(frontier.value());
  }

  void rebuildAbsorbedIndex() {
    absorbedIndex_.reset();
    if (absorbed_.size() <= kLinearMembershipLimit)
      return;
    absorbedIndex_ = std::make_unique<llvm::DenseSet<std::uint32_t>>();
    absorbedIndex_->reserve(absorbed_.size() * 2);
    absorbedIndex_->insert(absorbed_.begin(), absorbed_.end());
  }

  llvm::SmallVector<MemoryOrderFrontierId, 4> frontiers_;
  std::unique_ptr<llvm::DenseSet<std::uint32_t>> frontierIndex_;
  // Handles this accumulator already merged. Published union nodes retain
  // their components transitively, so re-merging an observed handle cannot
  // change the represented effect set. This is a memo of merged content,
  // cleared with the components, and never a relation of its own.
  llvm::SmallVector<std::uint32_t, 4> absorbed_;
  std::unique_ptr<llvm::DenseSet<std::uint32_t>> absorbedIndex_;
  std::optional<MemoryOrderFrontierId> published_;
};

struct Token {
  TokenKind kind = TokenKind::None;
  PrimitiveValueState valueState = PrimitiveValueState::Defined;
  // An exact bit pattern up to 64 bits stays inline. A zero width means that
  // the scalar union holds a host value or that an extended payload owns the
  // exact wide pattern. Wide patterns and memory views are mutually exclusive.
  unsigned inlineBitWidth = 0;
  std::uint64_t scalarValue = 0;
  // Memory-order witnesses enter token flow only through canonical done
  // publication. Generic actor firing may propagate them from that explicit
  // path, but plain memory data publication never injects its action effect.
  // This state is execution-local and never serialized.
  MemoryOrderFrontierId memoryOrder;
  std::shared_ptr<const ExtendedTokenPayload> extended;

  bool hasExactBitPattern() const {
    return inlineBitWidth != 0 ||
           (extended && (std::holds_alternative<llvm::APInt>(*extended) ||
                         std::holds_alternative<PointerValue>(*extended)));
  }

  unsigned exactBitWidth() const {
    if (inlineBitWidth != 0)
      return inlineBitWidth;
    const auto *bits =
        extended ? std::get_if<llvm::APInt>(extended.get()) : nullptr;
    if (bits)
      return bits->getBitWidth();
    const auto *pointer =
        extended ? std::get_if<PointerValue>(extended.get()) : nullptr;
    return pointer ? pointer->representation.getBitWidth() : 0;
  }

  llvm::APInt exactBitPattern() const {
    assert(hasExactBitPattern() && "token has no exact bit pattern");
    if (inlineBitWidth != 0)
      return llvm::APInt(inlineBitWidth, scalarValue,
                         /*isSigned=*/false, /*implicitTrunc=*/true);
    if (const auto *bits = std::get_if<llvm::APInt>(extended.get()))
      return *bits;
    return std::get<PointerValue>(*extended).representation;
  }

  void setExactBitPattern(llvm::APInt bits) {
    if (bits.getBitWidth() <= 64) {
      inlineBitWidth = bits.getBitWidth();
      scalarValue = bits.getZExtValue();
      extended.reset();
      return;
    }
    inlineBitWidth = 0;
    extended = std::make_shared<ExtendedTokenPayload>(std::move(bits));
  }

  const MemoryView *memoryView() const {
    return extended ? std::get_if<MemoryView>(extended.get()) : nullptr;
  }

  void setMemoryView(MemoryView view) {
    inlineBitWidth = 0;
    extended = std::make_shared<ExtendedTokenPayload>(std::move(view));
  }

  const PointerValue *pointerValue() const {
    return extended ? std::get_if<PointerValue>(extended.get()) : nullptr;
  }

  void setPointerValue(PointerValue pointer) {
    inlineBitWidth = 0;
    extended = std::make_shared<ExtendedTokenPayload>(std::move(pointer));
  }
};

struct DataflowMemoryRead {
  Token data;
  bool accessedMemory = false;
};

// The complete element update one store commits, prepared before any element
// changes so a rejected access leaves memory untouched.
struct DataflowMemoryWrite {
  struct Element {
    std::size_t byteOffset = 0;
    llvm::SmallVector<SemanticMemoryByte, 8> bytes;
    std::optional<PointerValue> pointer;
  };
  llvm::SmallVector<Element> elements;
  bool accessedMemory = false;
};

/// Ordered token queue for one software edge. Most queues alternate between
/// empty and one token, so contiguous storage keeps the channel table compact
/// and retains its capacity across firings. A backlog advances a head index
/// and is compacted only after enough consumed storage accumulates, preserving
/// amortized linear movement for long streams.
class TokenQueue {
public:
  bool empty() const { return head_ == tokens_.size(); }
  std::size_t size() const { return tokens_.size() - head_; }

  Token &front() {
    assert(!empty() && "front of empty token queue");
    return tokens_[head_];
  }
  const Token &front() const {
    assert(!empty() && "front of empty token queue");
    return tokens_[head_];
  }

  void push_back(const Token &token) { tokens_.push_back(token); }
  void push_back(Token &&token) { tokens_.push_back(std::move(token)); }

  void pop_front() {
    assert(!empty() && "pop from empty token queue");
    ++head_;
    if (head_ == tokens_.size()) {
      tokens_.clear();
      head_ = 0;
      return;
    }
    if (head_ >= 64 && head_ * 2 >= tokens_.size()) {
      std::move(tokens_.begin() + static_cast<std::ptrdiff_t>(head_),
                tokens_.end(), tokens_.begin());
      tokens_.resize(tokens_.size() - head_);
      head_ = 0;
    }
  }

  void clear() {
    tokens_.clear();
    head_ = 0;
  }

  void appendFrom(TokenQueue &source) {
    if (source.empty())
      return;
    if (empty()) {
      tokens_.swap(source.tokens_);
      std::swap(head_, source.head_);
      source.clear();
      return;
    }
    if (source.size() == 1) {
      tokens_.push_back(std::move(source.front()));
      source.clear();
      return;
    }
    tokens_.insert(
        tokens_.end(),
        std::make_move_iterator(source.tokens_.begin() + source.head_),
        std::make_move_iterator(source.tokens_.end()));
    source.clear();
  }

  auto begin() { return tokens_.begin() + static_cast<std::ptrdiff_t>(head_); }
  auto end() { return tokens_.end(); }
  auto begin() const {
    return tokens_.begin() + static_cast<std::ptrdiff_t>(head_);
  }
  auto end() const { return tokens_.end(); }

private:
  std::vector<Token> tokens_;
  std::size_t head_ = 0;
};

using ChannelOrdinal = std::uint32_t;
inline constexpr unsigned InvalidActorOrdinal =
    std::numeric_limits<unsigned>::max();

/// Run-local storage for one canonical software edge. The graph's OpOperand
/// remains the semantic owner; the ordinal and colocated queues are a dense
/// execution cache derived before firing begins.
struct ChannelSlot {
  const mlir::OpOperand *operand = nullptr;
  unsigned ownerActorOrdinal = InvalidActorOrdinal;
  TokenQueue ready;
  TokenQueue pending;
};

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
  llvm::SmallVector<SemanticMemoryByte> bytes;
  // Fresh allocation bytes are not initialized. Runtime-input bytes remain
  // initialized even when their semantic state is Poison or Undef.
  llvm::SmallBitVector initialized;
  // Provenance cannot be reconstructed from pointer representation bits.
  // This execution-local overlay is invalidated by every overlapping byte
  // write and therefore never becomes a competing memory-content authority.
  std::map<std::size_t, PointerValue> pointerValues;
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
  // Canonical half-open byte ranges of the active lanes, relative to the
  // logical root: ascending, non-empty, and neither overlapping nor touching.
  llvm::SmallVector<std::pair<std::int64_t, std::int64_t>, 1> byteRanges;
  bool isWrite = false;
};

enum class MemoryByteOrder { Little, Big };

struct ResolvedMemoryElementLayout {
  std::size_t byteCount = 0;
  unsigned bitWidth = 0;
  MemoryByteOrder byteOrder = MemoryByteOrder::Little;
};

/// Immutable execution projection of one finalized load or store. Every field
/// is derived once from the actor types and the graph DataLayout; dynamic
/// addresses, masks, aliasing, and ordering remain firing-time state.
struct MemoryActorExecutionPlan {
  dataflow::semantics::MemoryAccessType access;
  unsigned memoryOperandOrdinal = 0;
  unsigned addressOperandOrdinal = 0;
  std::optional<unsigned> dataOperandOrdinal;
  unsigned controlOperandOrdinal = 0;
  std::optional<unsigned> maskOperandOrdinal;
  unsigned indexBitWidth = 0;
  unsigned addressBitWidth = 0;
  unsigned dataBitWidth = 0;
  ResolvedMemoryElementLayout elementLayout;
};

/// One typed GEP path component. A dynamic component names its actor operand;
/// otherwise constantIndex carries the exact source integer. scale is already
/// projected to A(AS) bits from the exact LLVM DataLayout.
struct GepOffsetTerm {
  std::optional<unsigned> dynamicOperandOrdinal;
  llvm::APInt constantIndex = llvm::APInt(1, 0);
  llvm::APInt scale = llvm::APInt(1, 0);
};

/// Immutable execution projection of one scalar LLVM GEP. Type walking and
/// DataLayout queries happen once during graph preparation, never per firing.
struct GepExecutionPlan {
  ::loom::PointerLayout pointerLayout;
  mlir::LLVM::GEPNoWrapFlags noWrapFlags = mlir::LLVM::GEPNoWrapFlags::none;
  llvm::SmallVector<GepOffsetTerm, 4> terms;
};

using ActorProvider = bool (*)(mlir::Operation *,
                               const dataflow::CanonicalActorSchemaProjection &,
                               SimulatorState &);

/// The dynamic input/result shape selected by one non-mutating actor probe.
/// OperationSchema remains the authority for valid cases; this value is only
/// matched against the actor's cached canonical case table.
struct ActorTransitionShape {
  llvm::SmallVector<std::uint32_t, 4> consumedInputs;
  llvm::SmallVector<std::uint32_t, 4> activeResults;
};

enum class ActorTransitionProbeKind : std::uint8_t {
  Unavailable,
  AllInputs,
  OneShot,
  Primitive,
  GetElementPtr,
  Stream,
  Carry,
  Invariant,
  Gate,
  Mux,
  Demux,
  Parallelize,
  Serialize,
};

struct ActorRuntimeProvider {
  ActorProvider commit = nullptr;
  ActorTransitionProbeKind probe = ActorTransitionProbeKind::Unavailable;
};

/// Immutable, admission-derived execution cache for one canonical actor.
/// Persistent identity and semantics remain owned by Canonical Dataflow and
/// OperationSchema; this record only removes MLIR pointer-map reconstruction
/// from the firing loop.
struct ActorExecutionPlan {
  struct Output {
    mlir::Value value;
    llvm::SmallVector<ChannelOrdinal, 2> channels;
    bool observed = false;
  };

  mlir::Operation *operation = nullptr;
  dataflow::CanonicalActorSchemaProjection projection;
  ActorProvider provider = nullptr;
  ChannelOrdinal firstInputChannel = 0;
  std::uint32_t inputChannelCount = 0;
  llvm::SmallVector<Output, 2> outputs;
  std::optional<PrimitiveOperationDescriptor> primitive;
  std::optional<MemoryActorExecutionPlan> memory;
  std::optional<GepExecutionPlan> gep;
  llvm::SmallVector<dataflow::semantics::ActorHandshakeCase, 4> handshakeCases;
  ActorTransitionProbeKind transitionProbe =
      ActorTransitionProbeKind::Unavailable;

  bool isPlainMemory() const { return memory.has_value(); }
};

struct GraphReturnObservation {
  llvm::SmallVector<mlir::Value> complete;
  llvm::SmallVector<mlir::Value> values;
  llvm::SmallVector<mlir::Value> streams;
  llvm::SmallVector<mlir::Value> memories;
};

/// Immutable execution cache derived from one finalized dataflow.graph.
///
/// Canonical Dataflow and OperationSchema remain the semantic authorities.
/// This process-local projection only prevents repeated graph validation,
/// provider lookup, and pointer-to-ordinal reconstruction when the same graph
/// is activated many times with different runtime inputs.
struct PreparedGraphExecution {
  struct Channel {
    const mlir::OpOperand *operand = nullptr;
    unsigned ownerActorOrdinal = InvalidActorOrdinal;
  };

  dataflow::GraphOp graph = {};
  unsigned applicationInputCount = 0;
  GraphReturnObservation returnObservation;
  llvm::DenseMap<const mlir::OpOperand *, ChannelOrdinal> channelOrdinals;
  std::vector<Channel> channels;
  std::vector<ActorExecutionPlan> actorPlans;
  llvm::DenseSet<mlir::Value> observedValues;
  llvm::SmallBitVector initialPlainMemoryCandidates;
};

struct GraphPreparationFailure {
  std::string status;
  llvm::SmallVector<std::string> diagnostics;
};

using GraphPreparationResult =
    std::variant<PreparedGraphExecution, GraphPreparationFailure>;

llvm::Expected<GraphPreparationResult>
prepareGraphExecution(mlir::ModuleOp module, dataflow::GraphOp graph);

/// Merges the ranges into the ascending, non-touching cover of the same bytes.
void canonicalizeMemoryActionRanges(
    llvm::SmallVectorImpl<std::pair<std::int64_t, std::int64_t>> &ranges);

struct ReadyPlainMemoryAction {
  MemoryActionRecord action;
  llvm::SmallVector<SyncEffectId, 2> ctrlFrontier;
  MemoryView view;
  llvm::APInt activeLanes = llvm::APInt();
  llvm::SmallVector<std::size_t> slots;
  std::optional<unsigned> maskOperandOrdinal;
};

// Exact byte-interval cache of the maximal issued hazards. It stores effect
// handles but no order relation; MemorySynchronization alone decides whether
// one effect covers another and reduces read frontiers.
class PlainMemoryConflictIndex {
public:
  /// The maximal issued hazards `action` meets, without deciding whether any
  /// of them is ordered before it.
  llvm::SmallVector<SyncEffectId> query(const MemoryActionRecord &action) const;

  /// Records one issued access as the new maximal hazard of its byte ranges.
  void retain(const MemoryActionRecord &action, SyncEffectId effect,
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
  // The immutable graph projection outlives this run. Dynamic state never
  // mutates it, so repeated activations can share one prepared plan.
  const PreparedGraphExecution *execution = nullptr;
  // Candidates whose readiness may have changed for the next wave. Token
  // arrival schedules only the consuming actor, while a firing schedules
  // itself in case another transition is already buffered. The bitset keeps
  // equal-wave evaluation in structural order without rescanning the graph.
  llvm::SmallBitVector nextActorCandidates;
  std::vector<ChannelSlot> channelSlots;
  llvm::SmallVector<ChannelOrdinal, 16> pendingChannelOrdinals;
  const ActorExecutionPlan *currentActorPlan = nullptr;
  // Values whose complete publication sequence is an explicit graph
  // observation. Internal SSA token history is represented by the edge
  // queues alone and is not retained as an implicit trace.
  OutputMap observedOutputs;
  OutputMap pendingObservedOutputs;
  llvm::SmallVector<mlir::Value, 8> pendingObservedValues;
  llvm::DenseMap<mlir::Value, std::shared_ptr<MemoryValue>> memories;
  llvm::DenseMap<mlir::Value, MemoryView> memoryViews;
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
  // OperationSchemaId is a generated dense domain. The execution loop counts
  // directly by ordinal; the report projects nonzero entries into its public
  // ordered map only once after execution.
  std::vector<std::uint64_t> operationFireCounts =
      std::vector<std::uint64_t>(dataflow::operationSchemaCount(), 0);
  std::map<std::string, std::uint64_t> modeledLibraryCalls;
  std::uint64_t nextMemoryRootId = 0;
  std::uint64_t eventCount = 0;
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
  // The structural actor-order mask for plain memory actors and the subset
  // whose token queues changed or may still contain another firing. Admission
  // walks the dense mask in actor order, avoiding an allocated ordered node
  // for every candidate transition.
  llvm::SmallBitVector plainMemoryCandidates;
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

/// The first exact model's representability check. A returned string names a
/// valid workload feature this provider cannot execute; malformed inputs have
/// already been rejected by shared admission.
std::optional<std::string>
unsupportedTypedDfgInput(const CanonicalSimulationWorkload &workload,
                         const CanonicalSimulationRuntimeInput &runtimeInput,
                         const ResolvedLaunchContext &context);

/// Seeds one already-admitted rooted graph directly from the typed workload
/// and runtime-input models. No CLI syntax or simulator-local persistent ID is
/// involved.
llvm::Error
seedTypedDfgInputs(SimulatorState &state, dataflow::GraphOp graph,
                   const CanonicalSimulationWorkload &workload,
                   const CanonicalSimulationRuntimeInput &runtimeInput,
                   const ResolvedLaunchContext &context);

Token noneToken();
Token integerValueToken(std::int64_t value);
Token floatValueToken(double value);
Token boolValueToken(bool value);
llvm::Expected<Token> exceptionalValueToken(PrimitiveValueState state,
                                            mlir::Type type);
llvm::Expected<unsigned> tokenTypeBitWidth(mlir::Type type);
llvm::Expected<unsigned> resolvedTokenTypeBitWidth(mlir::Type type,
                                                   mlir::Operation *scope);
llvm::Expected<llvm::APInt> tokenBitPattern(const Token &token,
                                            mlir::Type type);
llvm::Expected<llvm::APInt> resolvedTokenBitPattern(const Token &token,
                                                    mlir::Type type,
                                                    mlir::Operation *scope);
llvm::Expected<Token> tokenFromBitPattern(const llvm::APInt &bits,
                                          mlir::Type type);
llvm::Expected<Token> tokenFromResolvedBitPattern(const llvm::APInt &bits,
                                                  mlir::Type type,
                                                  mlir::Operation *scope);
llvm::Expected<Token> parseRuntimeToken(llvm::StringRef raw, mlir::Type type,
                                        mlir::Operation *scope);
llvm::Expected<std::string> tokenToString(const Token &token, mlir::Type type,
                                          mlir::Operation *scope);
Token memoryCapabilityToken(mlir::Value root,
                            std::shared_ptr<MemoryValue> memory = {},
                            std::int64_t byteOffset = 0,
                            mlir::Type elementType = {});
llvm::Expected<Token> tokenFromTypedAttr(mlir::TypedAttr attr);
llvm::Expected<Token> zeroToken(mlir::Type type);

/// Resolves the frontier an accumulator publishes, interning its immutable
/// union at most once however many tokens go on to carry it.
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

TokenQueue &channelQueue(SimulatorState &state, mlir::OpOperand &operand);
bool hasToken(const SimulatorState &state, mlir::OpOperand &operand);
bool hasInputToken(const SimulatorState &state, unsigned operandOrdinal);
Token popInputToken(SimulatorState &state, unsigned operandOrdinal);
const Token &peekInputToken(const SimulatorState &state,
                            unsigned operandOrdinal);
void emitResultToken(SimulatorState &state, unsigned resultOrdinal,
                     Token token);
void emitResultTokenWithMemoryOrder(SimulatorState &state,
                                    unsigned resultOrdinal, Token token,
                                    MemoryOrderFrontierId memoryOrder);
bool recordEvent(SimulatorState &state, dataflow::OperationSchemaId schema);
void flushPendingTokens(SimulatorState &state);
void initializeRunState(SimulatorState &state,
                        const PreparedGraphExecution &execution);
void seedBlockArgument(SimulatorState &state, mlir::BlockArgument argument,
                       const Token &token);
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
llvm::Expected<ResolvedMemoryElementLayout>
resolveMemoryElementLayout(mlir::Type type, mlir::Operation *scope);

// The host element slot one semantic address names. `address` is exact at its
// own width and becomes a host index only after the sign and range checks.
std::optional<std::size_t>
resolveElementByteOffset(const MemoryView &view, const llvm::APInt &address,
                         mlir::Type elementType, SimulatorState &state,
                         mlir::Operation *scope,
                         llvm::StringRef diagnosticLabel);
std::optional<std::size_t>
resolveElementByteOffset(const MemoryView &view, const Token &addr,
                         mlir::Type elementType, SimulatorState &state,
                         mlir::Operation *scope,
                         llvm::StringRef diagnosticLabel);
std::optional<Token>
readMemoryElement(const MemoryView &view, std::size_t byteOffset,
                  mlir::Type elementType, SimulatorState &state,
                  mlir::Operation *scope, llvm::StringRef diagnosticLabel);
std::optional<Token> readMemoryElementResolved(
    const MemoryView &view, std::size_t byteOffset, mlir::Type dataType,
    const ResolvedMemoryElementLayout &layout,
    const std::optional<::loom::PointerLayout> &pointerLayout,
    SimulatorState &state, llvm::StringRef diagnosticLabel);
llvm::Expected<llvm::SmallVector<SemanticMemoryByte, 8>>
encodeMemoryElement(const Token &value, mlir::Type elementType,
                    mlir::Operation *scope);
void writeMemoryElement(const MemoryView &view, std::size_t byteOffset,
                        llvm::ArrayRef<SemanticMemoryByte> bytes);
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

/// Resolve the execution view currently associated with one memory SSA value,
/// peeling only the root-preserving cast forms admitted by Canonical Dataflow.
std::optional<MemoryView> resolveMemoryView(SimulatorState &state,
                                            mlir::Value value);

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

/// Copies the run's observations and distinct execution diagnostics into the
/// report. Performance estimates are Evaluation results, not simulator-local
/// report fields.
void projectRunObservations(SimulatorState &state, DFGSimulationReport &report);

llvm::Expected<SpatialFunctionalObservations>
projectRetiredFunctionalObservations(
    dataflow::GraphOp graph, SimulatorState &state,
    const CanonicalSimulationWorkload &workload,
    const CanonicalSimulationRuntimeInput &runtimeInput,
    const ResolvedLaunchContext &context,
    const dataflow::CanonicalDataflowProgramView &program);

llvm::Expected<PrimitiveValue> primitiveValueFromToken(const Token &token,
                                                       mlir::Type type,
                                                       unsigned indexBitWidth);
llvm::Expected<Token> tokenFromPrimitiveValue(const PrimitiveValue &value,
                                              mlir::Type type);
llvm::Expected<PrimitiveOperationDescriptor>
primitiveDescriptor(const dataflow::CanonicalActorSchemaProjection &projection,
                    mlir::Operation *op, mlir::Value result);
llvm::Expected<PrimitiveOperationDescriptor>
primitiveDescriptor(const dataflow::CanonicalActorSchemaProjection &projection,
                    mlir::Operation *op, mlir::Type resultType,
                    mlir::Type operandType);
llvm::Expected<PrimitiveOperationDescriptor> primitiveDescriptorForActor(
    const dataflow::CanonicalActorSchemaProjection &projection,
    mlir::Operation *op);
llvm::Expected<MemoryActorExecutionPlan>
memoryActorExecutionPlan(mlir::Operation *op, mlir::Operation *graphScope);
llvm::Expected<GepExecutionPlan> gepExecutionPlan(mlir::LLVM::GEPOp op,
                                                  mlir::Operation *graphScope);
bool fireGetElementPtr(
    mlir::Operation *op,
    const dataflow::CanonicalActorSchemaProjection &projection,
    SimulatorState &state);
llvm::Error validatePrimitiveTokenTypes(mlir::Operation *op,
                                        mlir::Value result);
llvm::Expected<Token>
evaluatePrimitiveToken(mlir::Operation *op,
                       const PrimitiveOperationDescriptor &descriptor,
                       mlir::Value result, llvm::ArrayRef<Token> inputTokens);

bool fireLoad(dataflow::LoadOp op, SimulatorState &state);
bool fireStore(dataflow::StoreOp op, SimulatorState &state);
std::optional<ActorRuntimeProvider>
actorRuntimeProvider(dataflow::OperationSchemaId schema);
ActorProvider actorProvider(dataflow::OperationSchemaId schema);
llvm::Expected<std::optional<std::uint32_t>>
probeActorTransition(const ActorExecutionPlan &plan,
                     const SimulatorState &state);
enum class ActorTransitionCommitOutcome : std::uint8_t {
  NotReady,
  Committed,
  Failed,
};
ActorTransitionCommitOutcome
commitActorTransition(const ActorExecutionPlan &plan, SimulatorState &state);
bool fireActorOperation(const ActorExecutionPlan &plan, SimulatorState &state);
std::optional<UnsupportedOperation> unsupportedActorProvider(
    mlir::Operation *op,
    const dataflow::CanonicalActorSchemaProjection &projection);

std::string unsupportedOperationLabel(mlir::Operation *op);

} // namespace LLVM_LIBRARY_VISIBILITY_NAMESPACE detail
} // namespace loom::sim

#endif // LOOM_LIB_SIMULATOR_DFGSIMULATORINTERNAL_H

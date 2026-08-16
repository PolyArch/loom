#ifndef FABRIC_IR_RESOURCECONTRACT_H
#define FABRIC_IR_RESOURCECONTRACT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace fabric {

/// Role of an owner-local key. Each role below is a distinct static type, so a
/// key of one role can never be spelled where another is expected.
enum class ResourceKeyRole : std::uint32_t {
  State,
  CapacityDimension,
  Transition,
  UsePattern,
  Requester,
  Eligibility,
  Event,
  TimingContract,
  Claim,
};

/// One owner-defined closed key.
///
/// The owning resource enumerates its states, capacity dimensions, use
/// patterns, requesters, eligibility conditions, acquire/release events, and
/// timing/progress contracts as closed zero-based enums and spells those
/// values in its declaration. A key inventory is closed exactly when it
/// presents every key of its domain once, so a validated key is also the
/// canonical ordinal that `docs/spec-fabric-identity.md` references and there
/// is no second naming authority, free-form key, or property bag. These atoms
/// are in-memory values; the persistent encoding of a reference stays owned by
/// the identity schema.
template <ResourceKeyRole Role> class ResourceKey {
public:
  explicit constexpr ResourceKey(std::uint32_t ordinal) : ordinal_(ordinal) {}

  constexpr std::uint32_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(ResourceKey lhs, ResourceKey rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(ResourceKey lhs, ResourceKey rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint32_t ordinal_;
};

using StateKey = ResourceKey<ResourceKeyRole::State>;
using CapacityDimensionKey = ResourceKey<ResourceKeyRole::CapacityDimension>;
using ResourceTransitionKey = ResourceKey<ResourceKeyRole::Transition>;
using UsePatternKey = ResourceKey<ResourceKeyRole::UsePattern>;
using RequesterKey = ResourceKey<ResourceKeyRole::Requester>;
using EligibilityKey = ResourceKey<ResourceKeyRole::Eligibility>;
using EventKey = ResourceKey<ResourceKeyRole::Event>;
using TimingContractKey = ResourceKey<ResourceKeyRole::TimingContract>;
using ClaimKey = ResourceKey<ResourceKeyRole::Claim>;

/// A typed integer amount in one owner-declared capacity unit. Capacity,
/// canonical initial occupancy, and claim amounts measure the same unit and so
/// are one type. Addition is checked: an unrepresentable sum is reported
/// instead of wrapping.
class CapacityUnits {
public:
  explicit constexpr CapacityUnits(std::uint32_t value) : value_(value) {}

  constexpr std::uint32_t value() const { return value_; }

  static constexpr std::optional<CapacityUnits> checkedAdd(CapacityUnits lhs,
                                                           CapacityUnits rhs) {
    if (lhs.value_ > std::numeric_limits<std::uint32_t>::max() - rhs.value_)
      return std::nullopt;
    return CapacityUnits(lhs.value_ + rhs.value_);
  }

  friend constexpr bool operator==(CapacityUnits lhs, CapacityUnits rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(CapacityUnits lhs, CapacityUnits rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(CapacityUnits lhs, CapacityUnits rhs) {
    return lhs.value_ < rhs.value_;
  }
  friend constexpr bool operator>(CapacityUnits lhs, CapacityUnits rhs) {
    return rhs < lhs;
  }
  friend constexpr bool operator<=(CapacityUnits lhs, CapacityUnits rhs) {
    return !(rhs < lhs);
  }
  friend constexpr bool operator>=(CapacityUnits lhs, CapacityUnits rhs) {
    return !(lhs < rhs);
  }

private:
  std::uint32_t value_;
};

/// One typed integer capacity dimension of a ResourceState.
/// `initialOccupancy` is the single canonical initial and reset declaration.
/// Dynamic occupancy is execution state and never enters the contract.
struct CapacityDimension {
  CapacityUnits capacity;
  CapacityUnits initialOccupancy;
};

/// One validated capacity claim of an atomic use pattern. A claim is a
/// temporary reservation only: the pattern owns the one acquire and the one
/// release of the whole envelope, so a claim carries no release of its own,
/// cannot be split, and never becomes durable resource state.
struct Claim {
  StateKey state;
  CapacityDimensionKey dimension;
  CapacityUnits amount;
};

/// The one owner-defined state transition an accepted use applies, and the
/// exact event at which it applies atomically. The owner determines whether
/// its effect is durable across uses or local to the current claim. A
/// transition is not itself a capacity claim; claim acquisition and release
/// remain separately defined by the use pattern.
struct Commit {
  EventKey event;
  ResourceTransitionKey transition;
};

/// One position in a Fabric-owned ResourceUse value schema. The kind chooses a
/// closed production codec; its fields are the codec's exact semantic
/// parameters rather than an extensible property map.
enum class UsePatternValueKind : std::uint32_t {
  PhysicalTag = 0,
};

struct UsePatternValueSchema {
  UsePatternValueKind kind = UsePatternValueKind::PhysicalTag;
  std::uint32_t bitWidth = 0;

  static constexpr UsePatternValueSchema physicalTag(std::uint32_t bitWidth) {
    return UsePatternValueSchema{UsePatternValueKind::PhysicalTag, bitWidth};
  }

  friend constexpr bool operator==(const UsePatternValueSchema &lhs,
                                   const UsePatternValueSchema &rhs) {
    return lhs.kind == rhs.kind && lhs.bitWidth == rhs.bitWidth;
  }
  friend constexpr bool operator!=(const UsePatternValueSchema &lhs,
                                   const UsePatternValueSchema &rhs) {
    return !(lhs == rhs);
  }
};

/// One validated atomic resource use. Every claim is acquired together at
/// `acquire` and the complete envelope returns together at `release`. When
/// `commit` is present its transition applies atomically at its own event,
/// which the owning timing contract orders within `[acquire, release]`.
/// Eligibility and the timing/progress contract are owner-declared closed keys
/// rather than a predicate or parameter map. Spans read the owning contract's
/// tables.
struct UsePattern {
  RequesterKey requester;
  EligibilityKey eligibility;
  EventKey acquire;
  EventKey release;
  std::optional<Commit> commit;
  TimingContractKey timingAndProgress;
  llvm::ArrayRef<Claim> claims;
  std::uint32_t internalTransactionCount;
  llvm::ArrayRef<UsePatternValueSchema> parameters;
  llvm::ArrayRef<UsePatternValueSchema> sharingAssignments;
};

/// Intrinsic cycle timing mechanically derived from one validated atomic use.
/// `releaseLatencyCycles` is the exact owner-declared acquire-to-release
/// distance. `commitLatencyCycles`, when present, is the acquire-to-commit
/// distance. `minimumInitiationIntervalCycles` is the smallest integer issue
/// interval admitted by this use's own claim envelopes and canonical initial
/// occupancy. It excludes contention with other requesters and dynamic
/// backpressure; those remain execution state rather than hidden timing
/// defaults.
struct UsePatternTiming final {
  std::uint32_t releaseLatencyCycles = 0;
  std::optional<std::uint32_t> commitLatencyCycles;
  std::uint32_t minimumInitiationIntervalCycles = 1;
};

class ResourceContract;

/// Validated fixed-priority order. It has no cursor: the permutation alone
/// decides every grant.
class FixedPriorityView {
public:
  llvm::ArrayRef<RequesterKey> requesterOrder() const { return order_; }

  /// The first eligible requester in the exact permutation. `eligible` is
  /// transient execution state indexed by requester ordinal and sized to the
  /// requester domain.
  std::optional<RequesterKey> grant(llvm::ArrayRef<bool> eligible) const;

private:
  explicit FixedPriorityView(llvm::ArrayRef<RequesterKey> order)
      : order_(order) {}

  llvm::ArrayRef<RequesterKey> order_;

  friend class ResourceContract;
};

/// One round-robin arbitration step. `nextCursor` equals the incoming cursor
/// unless a grant succeeded.
struct RoundRobinGrant {
  std::optional<RequesterKey> granted;
  RequesterKey nextCursor;
};

/// Validated round-robin cycle. The running cursor is caller-owned execution
/// state; only its reset origin is declared.
class RoundRobinView {
public:
  llvm::ArrayRef<RequesterKey> requesterCycle() const { return cycle_; }

  /// The cursor established by reset.
  RequesterKey resetCursor() const { return cycle_[resetPosition_]; }

  /// Scans the exact cycle from `cursor`, which must be `resetCursor()` or a
  /// `nextCursor` of this cycle, and advances only past a granted requester.
  RoundRobinGrant grant(RequesterKey cursor,
                        llvm::ArrayRef<bool> eligible) const;

private:
  RoundRobinView(llvm::ArrayRef<RequesterKey> cycle,
                 std::uint32_t resetPosition)
      : cycle_(cycle), resetPosition_(resetPosition) {}

  llvm::ArrayRef<RequesterKey> cycle_;
  std::uint32_t resetPosition_;

  friend class ResourceContract;
};

/// The closed exact requester-ordering domain, readable only from a validated
/// contract. Declaration, insertion, map, and arrival order never become an
/// ordering.
using GrantPolicyView = std::variant<FixedPriorityView, RoundRobinView>;

/// Typed contract validation failures, listed in validation precedence order.
enum class ResourceContractViolation : std::uint32_t {
  DuplicateStateKey,
  UnknownStateKey,
  DuplicateCapacityDimensionKey,
  UnknownCapacityDimensionKey,
  InitialOccupancyExceedsCapacity,
  DuplicateResourceTransitionKey,
  UnknownResourceTransitionKey,
  DuplicateTimingContractKey,
  UnknownTimingContractKey,
  DuplicateRequesterKey,
  UnknownRequesterKey,
  DuplicateUsePatternKey,
  UnknownUsePatternKey,
  UnknownEligibilityKey,
  UnknownEventKey,
  TimingContractDoesNotOrderUse,
  InvalidPatternValueSchema,
  DuplicateClaimKey,
  UnknownClaimKey,
  UndeclaredClaim,
  DuplicateCapacityClaim,
  CapacityArithmeticOverflow,
  ClaimExceedsCapacity,
  DuplicateRequesterInGrantPolicy,
  RequesterOmittedFromGrantPolicy,
  ContentionWithoutGrantPolicy,
  GrantPolicyWithoutContention,
};

llvm::StringRef
getResourceContractViolationName(ResourceContractViolation violation);

class ResourceContractError final
    : public llvm::ErrorInfo<ResourceContractError> {
public:
  static char ID;

  ResourceContractError(ResourceContractViolation violation,
                        std::string message)
      : violation_(violation), message_(std::move(message)) {}

  ResourceContractViolation violation() const { return violation_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  ResourceContractViolation violation_;
  std::string message_;
};

/// One declared capacity dimension of a state.
struct CapacityDimensionDeclaration {
  CapacityDimensionKey key;
  CapacityUnits capacity;
  CapacityUnits initialOccupancy;
};

/// One declared resource state and its closed capacity dimension inventory.
struct ResourceStateDeclaration {
  StateKey key;
  std::vector<CapacityDimensionDeclaration> capacityDimensions;
};

/// One declared claim of a use pattern. The enclosing pattern owns the one
/// release event for the complete atomic claim envelope.
struct ClaimDeclaration {
  ClaimKey key;
  StateKey state;
  CapacityDimensionKey dimension;
  CapacityUnits amount;
};

/// One declared implementation transaction of an accepted use, such as a
/// service beat or a lane group. It only selects claim keys already declared
/// by the enclosing pattern and carries no requester, eligibility, event,
/// commit transition, or timing of its own, so a decomposition can neither
/// acquire a second claim envelope, apply another resource transition, become a
/// software actor or Mapping use, nor change the single external firing,
/// retirement, ordering, and progress contract. Declaration order is the exact
/// issue order.
struct InternalTransactionDeclaration {
  std::vector<ClaimKey> claims;
};

/// The one optional declared commit of a use pattern.
struct CommitDeclaration {
  EventKey event;
  ResourceTransitionKey transition;
};

/// One declared timing-and-progress contract. `eventRank` is the owner's
/// relative time of every declared event under this contract, indexed by event
/// ordinal; equal ranks denote one atomic event. It is the one place the owner
/// establishes `acquire <= commit <= release`, so no callback, string
/// predicate, or property bag can order a use instead.
struct TimingContractDeclaration {
  TimingContractKey key;
  std::vector<std::uint32_t> eventRank;
};

/// One declared atomic use pattern.
struct UsePatternDeclaration {
  UsePatternKey key;
  RequesterKey requester;
  EligibilityKey eligibility;
  EventKey acquire;
  EventKey release;
  std::optional<CommitDeclaration> commit;
  TimingContractKey timingAndProgress;
  std::vector<ClaimDeclaration> claims;
  std::vector<InternalTransactionDeclaration> internalTransactions;
  std::vector<UsePatternValueSchema> parameters = {};
  std::vector<UsePatternValueSchema> sharingAssignments = {};
};

struct FixedPriorityDeclaration {
  std::vector<RequesterKey> requesterOrder;
};

struct RoundRobinDeclaration {
  std::vector<RequesterKey> requesterCycle;
  RequesterKey resetCursor;
};

using GrantPolicyDeclaration =
    std::variant<FixedPriorityDeclaration, RoundRobinDeclaration>;

/// What one owning resource declares. Each inventory carries its owner-defined
/// keys and must present every key of its closed domain exactly once;
/// validation normalizes accepted records into key order, so declaration order
/// carries no meaning. `eligibilityCount` and `eventCount` close the two
/// reference-only key domains, which declare no record and therefore have no
/// inventory to duplicate.
///
/// `grantPolicy` is present exactly when arbitration is observable, which is
/// exactly when some capacity dimension is claimed by use patterns of two
/// different requesters.
struct ResourceContractDeclaration {
  std::vector<ResourceStateDeclaration> states;
  std::vector<ResourceTransitionKey> resourceTransitions;
  std::vector<TimingContractDeclaration> timingContracts;
  std::vector<UsePatternDeclaration> usePatterns;
  std::vector<RequesterKey> requesters;
  std::uint32_t eligibilityCount = 0;
  std::uint32_t eventCount = 0;
  std::optional<GrantPolicyDeclaration> grantPolicy;
};

/// One complete validated resource contract. A concrete Fabric resource embeds
/// one of these; it is not an artifact, an independently addressable entity,
/// or an extension registry. Records live in flat key-ordered tables read
/// through spans, so a consumer can cache them directly; only `create`
/// produces one, so every publicly readable record and policy view has already
/// been validated.
class ResourceContract {
public:
  /// Validates one declaration and either returns the contract or the first
  /// violation under this exact precedence, which is a property of the check
  /// class rather than of where an offending record was declared:
  ///
  ///   1. state key inventory;
  ///   2. capacity dimension key inventory of each state;
  ///   3. canonical initial state of every capacity dimension;
  ///   4. resource transition key inventory;
  ///   5. timing contract key inventory and its event-rank shape;
  ///   6. requester key inventory;
  ///   7. use pattern key inventory;
  ///   8. use pattern requester, eligibility, acquire, release, commit, and
  ///      timing keys;
  ///   9. timing-contract ordering of acquire, optional commit, and release;
  ///  10. positional parameter and sharing-assignment schemas;
  ///  11. claim key inventory of each pattern;
  ///  12. claim state and capacity dimension keys;
  ///  13. one atomic envelope per capacity dimension;
  ///  14. claim feasibility against the canonical initial state;
  ///  15. internal transaction claim selection;
  ///  16. grant policy permutation and reset cursor; and
  ///  17. agreement between reachable contention and the declared ordering.
  ///
  /// Every class scans keys in ascending order once its inventory is
  /// validated, and contention is reported at the lowest contended capacity
  /// dimension, so the result never depends on declaration or iteration order.
  /// A key inventory is itself unordered: a repeated key outranks a key
  /// outside the closed domain, and the reported key is the lowest offending
  /// one. The requester permutation of a grant policy is an ordered sequence
  /// rather than an inventory, so it is reported at its first offending
  /// position.
  static llvm::Expected<ResourceContract>
  create(const ResourceContractDeclaration &declaration);

  /// Reconstructs the unique key-ordered declaration of this validated
  /// contract. The result is a derived editing form, not a second authority.
  ResourceContractDeclaration declaration() const;

  /// Exact semantic equality of two validated, normalized contracts. This is
  /// an in-memory query; persistent identity remains owned by the artifact
  /// codec and is not reconstructed to answer it.
  friend bool operator==(const ResourceContract &lhs,
                         const ResourceContract &rhs);
  friend bool operator!=(const ResourceContract &lhs,
                         const ResourceContract &rhs) {
    return !(lhs == rhs);
  }

  std::uint32_t stateCount() const {
    return static_cast<std::uint32_t>(states_.size());
  }
  llvm::ArrayRef<CapacityDimension> capacityDimensions(StateKey key) const {
    assert(key.ordinal() < states_.size() && "undeclared state");
    const Span span = states_[key.ordinal()];
    return llvm::ArrayRef<CapacityDimension>(capacityDimensions_)
        .slice(span.first, span.count);
  }

  std::uint32_t resourceTransitionCount() const {
    return resourceTransitionCount_;
  }

  std::uint32_t usePatternCount() const {
    return static_cast<std::uint32_t>(patterns_.size());
  }
  UsePattern usePattern(UsePatternKey key) const;
  UsePatternTiming usePatternTiming(UsePatternKey key) const;
  llvm::ArrayRef<ClaimKey> internalTransaction(UsePatternKey key,
                                               std::uint32_t transaction) const;

  std::uint32_t requesterCount() const { return requesterCount_; }
  std::uint32_t eligibilityCount() const { return eligibilityCount_; }
  std::uint32_t eventCount() const { return eventCount_; }
  std::uint32_t timingContractCount() const { return timingContractCount_; }

  /// The relative time this timing contract declares for every event, indexed
  /// by event ordinal.
  llvm::ArrayRef<std::uint32_t> eventOrder(TimingContractKey key) const;

  std::optional<GrantPolicyView> grantPolicy() const;

private:
  enum class GrantPolicyKind : std::uint32_t { FixedPriority, RoundRobin };

  struct Span {
    std::uint32_t first = 0;
    std::uint32_t count = 0;
  };

  struct PatternRecord {
    RequesterKey requester;
    EligibilityKey eligibility;
    EventKey acquire;
    EventKey release;
    std::optional<Commit> commit;
    TimingContractKey timingAndProgress;
    Span claims;
    Span internalTransactions;
    Span parameters;
    Span sharingAssignments;
  };

  ResourceContract() = default;

  std::vector<CapacityDimension> capacityDimensions_;
  std::vector<Span> states_;
  std::vector<std::uint32_t> eventRanks_;
  std::vector<Claim> claims_;
  std::vector<ClaimKey> transactionClaims_;
  std::vector<Span> internalTransactions_;
  std::vector<UsePatternValueSchema> valueSchemas_;
  std::vector<PatternRecord> patterns_;
  std::vector<RequesterKey> requesterOrder_;
  std::uint32_t resourceTransitionCount_ = 0;
  std::uint32_t requesterCount_ = 0;
  std::uint32_t eligibilityCount_ = 0;
  std::uint32_t eventCount_ = 0;
  std::uint32_t timingContractCount_ = 0;
  std::optional<GrantPolicyKind> grantPolicyKind_;
  std::uint32_t resetCursorPosition_ = 0;
};

} // namespace fabric

#endif // FABRIC_IR_RESOURCECONTRACT_H

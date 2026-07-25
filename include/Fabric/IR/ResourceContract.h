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
/// timing/progress contracts as closed enums whose zero-based values are
/// exactly these ordinals; the declaration this key indexes is the closure.
/// The key is therefore also the canonical ordinal that
/// `docs/spec-fabric-identity.md` references, so there is no second naming
/// authority, free-form string key, or property bag. These atoms are in-memory
/// values; the persistent encoding of a reference stays owned by the identity
/// schema.
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

/// One resource state. Its key is its declared ordinal and its canonical
/// initial value is the declared initial occupancy of its dimensions.
struct ResourceState {
  std::vector<CapacityDimension> capacityDimensions;
};

/// One atomic capacity claim of a use pattern. `release` names the
/// owner-declared event at which the claimed units return. A pattern claims a
/// capacity dimension at most once, which is what makes both the atomic
/// envelope and the release of every claimed capacity exact.
struct Claim {
  StateKey state;
  CapacityDimensionKey dimension;
  CapacityUnits amount;
  EventKey release;
};

/// One implementation transaction of an accepted use, such as a service beat
/// or a lane group. It only selects claims already declared by the enclosing
/// pattern and carries no requester, eligibility, event, or timing of its own,
/// so a decomposition can neither become a software actor or Mapping use nor
/// change the single external firing, retirement, ordering, and progress
/// contract. Declaration order is the exact issue order.
struct InternalTransaction {
  std::vector<ClaimKey> claims;
};

/// One atomic resource use. Every claim is acquired together at `acquire` and
/// returns at its own declared release event. Eligibility and the
/// timing/progress contract are owner-declared closed keys rather than a
/// predicate or parameter map. Mapping selects a declared pattern; it cannot
/// split these claims.
struct UsePattern {
  RequesterKey requester;
  EligibilityKey eligibility;
  EventKey acquire;
  TimingContractKey timingAndProgress;
  std::vector<Claim> claims;
  std::vector<InternalTransaction> internalTransactions;
};

/// Grants the first eligible requester in the exact permutation.
struct FixedPriority {
  std::vector<RequesterKey> requesterOrder;
};

/// Scans the exact cycle from the current cursor and advances only past a
/// successfully granted requester. `resetCursor` is the scan origin
/// established by reset; the running cursor is execution state held by the
/// consumer and never by the contract.
struct RoundRobin {
  std::vector<RequesterKey> requesterCycle;
  RequesterKey resetCursor;
};

/// The closed exact requester-ordering domain. A contract without a policy has
/// no arbiter at all: declaration, insertion, map, and arrival order never
/// become an ordering.
using GrantPolicy = std::variant<FixedPriority, RoundRobin>;

/// One arbitration step. `nextCursor` is unchanged by a failed or absent
/// grant.
struct GrantDecision {
  std::optional<RequesterKey> granted;
  RequesterKey nextCursor;
};

/// The cursor established by reset: the declared RoundRobin origin, or the
/// front of a FixedPriority permutation. The permutation must be non-empty; a
/// resource with no requester never arbitrates.
RequesterKey resetGrantCursor(const GrantPolicy &policy);

/// Resolves one grant. `eligible` is transient execution state indexed by
/// requester ordinal and sized to the requester domain. FixedPriority always
/// scans from the front of its permutation and keeps that origin as the next
/// cursor; RoundRobin scans its exact cycle from `cursor` and advances only
/// past a successfully granted requester.
GrantDecision arbitrate(const GrantPolicy &policy, RequesterKey cursor,
                        llvm::ArrayRef<bool> eligible);

/// Typed contract validation failures.
enum class ResourceContractViolation : std::uint32_t {
  InitialOccupancyExceedsCapacity,
  UnknownRequesterKey,
  UnknownEligibilityKey,
  UnknownEventKey,
  UnknownTimingContractKey,
  UnknownStateKey,
  UndeclaredClaim,
  DuplicateClaim,
  AmbiguousRelease,
  ClaimExceedsCapacity,
  UnknownClaimKey,
  DuplicateRequesterInGrantPolicy,
  RequesterOmittedFromGrantPolicy,
  ContentionWithoutGrantPolicy,
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

/// What one owning resource declares. Positions are keys: `states[i]` is
/// StateKey `i`, `usePatterns[i]` is UsePatternKey `i`, and each count closes
/// the owner domain of the matching key role.
///
/// `grantPolicy` may be omitted only when the declaration itself proves
/// contention impossible, which is exactly when no capacity dimension is
/// claimed by use patterns of two different requesters.
struct ResourceContractDeclaration {
  std::vector<ResourceState> states;
  std::vector<UsePattern> usePatterns;
  std::uint32_t requesterCount = 0;
  std::uint32_t eligibilityCount = 0;
  std::uint32_t eventCount = 0;
  std::uint32_t timingContractCount = 0;
  std::optional<GrantPolicy> grantPolicy;
};

/// One complete resource contract. A concrete Fabric resource embeds one of
/// these; it is not an artifact, an independently addressable entity, or an
/// extension registry. Only `create` produces one, so a publicly consumable
/// contract has already been validated.
class ResourceContract {
public:
  /// Validates one declaration and either returns the contract or the first
  /// violation under this exact precedence, which is a property of the check
  /// class rather than of where an offending record was declared:
  ///
  ///   1. canonical initial state of every capacity dimension;
  ///   2. use-pattern requester, eligibility, acquire, and timing keys;
  ///   3. claim state, capacity dimension, and release keys;
  ///   4. atomicity of each claim envelope (duplicate or split release);
  ///   5. claim feasibility against the canonical initial state;
  ///   6. internal transaction claim selection;
  ///   7. grant policy permutation and reset cursor; and
  ///   8. contention that no exact grant policy resolves.
  ///
  /// Every class scans states, patterns, claims, and requesters in ascending
  /// declared order, and contention is reported at the lowest contended
  /// capacity dimension, so the result never depends on iteration order.
  static llvm::Expected<ResourceContract>
  create(ResourceContractDeclaration declaration);

  std::uint32_t stateCount() const {
    return static_cast<std::uint32_t>(declaration_.states.size());
  }
  const ResourceState &state(StateKey key) const {
    assert(key.ordinal() < declaration_.states.size() && "undeclared state");
    return declaration_.states[key.ordinal()];
  }

  std::uint32_t usePatternCount() const {
    return static_cast<std::uint32_t>(declaration_.usePatterns.size());
  }
  const UsePattern &usePattern(UsePatternKey key) const {
    assert(key.ordinal() < declaration_.usePatterns.size() &&
           "undeclared use pattern");
    return declaration_.usePatterns[key.ordinal()];
  }

  std::uint32_t requesterCount() const { return declaration_.requesterCount; }
  std::uint32_t eligibilityCount() const {
    return declaration_.eligibilityCount;
  }
  std::uint32_t eventCount() const { return declaration_.eventCount; }
  std::uint32_t timingContractCount() const {
    return declaration_.timingContractCount;
  }

  const std::optional<GrantPolicy> &grantPolicy() const {
    return declaration_.grantPolicy;
  }

private:
  explicit ResourceContract(ResourceContractDeclaration declaration)
      : declaration_(std::move(declaration)) {}

  ResourceContractDeclaration declaration_;
};

} // namespace fabric

#endif // FABRIC_IR_RESOURCECONTRACT_H

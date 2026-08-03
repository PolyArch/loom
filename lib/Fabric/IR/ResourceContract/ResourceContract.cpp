#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace fabric {

namespace {

constexpr std::uint32_t absentPosition =
    std::numeric_limits<std::uint32_t>::max();

llvm::Error rejected(ResourceContractViolation violation,
                     const llvm::Twine &site) {
  return llvm::make_error<ResourceContractError>(
      violation,
      (llvm::Twine(getResourceContractViolationName(violation)) + " at " + site)
          .str());
}

std::string dimensionSite(std::size_t state, std::size_t dimension) {
  return ("state " + llvm::Twine(state) + " capacity dimension " +
          llvm::Twine(dimension))
      .str();
}

std::string patternSite(std::size_t pattern) {
  return ("use pattern " + llvm::Twine(pattern)).str();
}

std::string claimSite(std::size_t pattern, std::size_t claim) {
  return ("use pattern " + llvm::Twine(pattern) + " claim " +
          llvm::Twine(claim))
      .str();
}

std::string transactionSite(std::size_t pattern, std::size_t transaction) {
  return ("use pattern " + llvm::Twine(pattern) + " internal transaction " +
          llvm::Twine(transaction))
      .str();
}

std::string valueSchemaSite(std::size_t pattern, llvm::StringRef field,
                            std::size_t ordinal) {
  return ("use pattern " + llvm::Twine(pattern) + " " + field + " " +
          llvm::Twine(ordinal))
      .str();
}

llvm::Error checkValueSchemas(std::size_t pattern, llvm::StringRef field,
                              llvm::ArrayRef<UsePatternValueSchema> schemas) {
  for (auto [ordinal, schema] : llvm::enumerate(schemas)) {
    switch (schema.kind) {
    case UsePatternValueKind::PhysicalTag:
      if (schema.bitWidth != 0)
        continue;
      break;
    }
    return rejected(ResourceContractViolation::InvalidPatternValueSchema,
                    valueSchemaSite(pattern, field, ordinal));
  }
  return llvm::Error::success();
}

// The one report of a malformed unordered key set: a repeated key outranks a
// key outside the closed domain, and the reported key is the lowest offending
// one, so the outcome depends only on which keys the set holds.
llvm::Error rejectedKeySet(std::optional<std::uint32_t> repeated,
                           std::optional<std::uint32_t> foreign,
                           ResourceContractViolation duplicate,
                           ResourceContractViolation unknown,
                           const llvm::Twine &site) {
  if (repeated)
    return rejected(duplicate, site + " key " + llvm::Twine(*repeated));
  if (foreign)
    return rejected(unknown, site + " key " + llvm::Twine(*foreign));
  return llvm::Error::success();
}

void observeKey(std::uint32_t key, bool inDomain, bool alreadyHeld,
                std::optional<std::uint32_t> &repeated,
                std::optional<std::uint32_t> &foreign) {
  if (!inDomain)
    foreign = foreign ? std::min(*foreign, key) : key;
  else if (alreadyHeld)
    repeated = repeated ? std::min(*repeated, key) : key;
}

// One declaration reached only through validated keys. Every check after an
// inventory is validated, and the built tables, read records by ascending key,
// so declaration order carries no meaning.
struct NormalizedDeclaration {
  explicit NormalizedDeclaration(const ResourceContractDeclaration &declaration)
      : declaration(declaration) {}

  std::size_t stateCount() const { return declaration.states.size(); }
  const ResourceStateDeclaration &state(std::size_t key) const {
    return declaration.states[statePositions[key]];
  }

  std::size_t capacityDimensionCount(std::size_t stateKey) const {
    return capacityDimensionPositions[stateKey].size();
  }
  const CapacityDimensionDeclaration &
  capacityDimension(std::size_t stateKey, std::size_t dimensionKey) const {
    return state(stateKey)
        .capacityDimensions[capacityDimensionPositions[stateKey][dimensionKey]];
  }

  std::size_t timingContractCount() const {
    return declaration.timingContracts.size();
  }
  const TimingContractDeclaration &timingContract(std::size_t key) const {
    return declaration.timingContracts[timingContractPositions[key]];
  }

  std::size_t usePatternCount() const { return declaration.usePatterns.size(); }
  const UsePatternDeclaration &usePattern(std::size_t key) const {
    return declaration.usePatterns[usePatternPositions[key]];
  }

  std::size_t claimCount(std::size_t patternKey) const {
    return claimPositions[patternKey].size();
  }
  const ClaimDeclaration &claim(std::size_t patternKey,
                                std::size_t claimKey) const {
    return usePattern(patternKey).claims[claimPositions[patternKey][claimKey]];
  }

  const ResourceContractDeclaration &declaration;
  std::vector<std::uint32_t> statePositions;
  std::vector<std::vector<std::uint32_t>> capacityDimensionPositions;
  std::vector<std::uint32_t> timingContractPositions;
  std::vector<std::uint32_t> usePatternPositions;
  std::vector<std::vector<std::uint32_t>> claimPositions;
};

// One closed key inventory presents every key of its domain exactly once. It is
// unordered, so the whole of it is scanned before it is reported.
template <typename KeyAt>
llvm::Error indexInventory(std::size_t size, KeyAt keyAt,
                           ResourceContractViolation duplicate,
                           ResourceContractViolation unknown,
                           const llvm::Twine &site,
                           std::vector<std::uint32_t> &positionByKey) {
  positionByKey.assign(size, absentPosition);
  std::optional<std::uint32_t> repeated;
  std::optional<std::uint32_t> foreign;

  for (std::size_t position = 0; position < size; ++position) {
    const std::uint32_t key = keyAt(position);
    const bool inDomain = static_cast<std::size_t>(key) < size;
    const bool alreadyHeld = inDomain && positionByKey[key] != absentPosition;
    observeKey(key, inDomain, alreadyHeld, repeated, foreign);
    if (inDomain && !alreadyHeld)
      positionByKey[key] = static_cast<std::uint32_t>(position);
  }

  return rejectedKeySet(repeated, foreign, duplicate, unknown, site);
}

llvm::ArrayRef<RequesterKey>
declaredPermutation(const GrantPolicyDeclaration &policy) {
  if (const auto *fixed = std::get_if<FixedPriorityDeclaration>(&policy))
    return fixed->requesterOrder;
  return std::get<RoundRobinDeclaration>(policy).requesterCycle;
}

llvm::Error checkGrantPolicy(const GrantPolicyDeclaration &policy,
                             std::size_t requesterCount) {
  const llvm::ArrayRef<RequesterKey> permutation = declaredPermutation(policy);
  std::vector<bool> ordered(requesterCount, false);

  for (std::size_t position = 0; position < permutation.size(); ++position) {
    const RequesterKey requester = permutation[position];
    if (static_cast<std::size_t>(requester.ordinal()) >= requesterCount)
      return rejected(ResourceContractViolation::UnknownRequesterKey,
                      "grant policy position " + llvm::Twine(position));
    if (ordered[requester.ordinal()])
      return rejected(
          ResourceContractViolation::DuplicateRequesterInGrantPolicy,
          "grant policy position " + llvm::Twine(position));
    ordered[requester.ordinal()] = true;
  }

  for (std::size_t requester = 0; requester < requesterCount; ++requester)
    if (!ordered[requester])
      return rejected(
          ResourceContractViolation::RequesterOmittedFromGrantPolicy,
          "requester " + llvm::Twine(requester));

  if (const auto *roundRobin = std::get_if<RoundRobinDeclaration>(&policy))
    if (static_cast<std::size_t>(roundRobin->resetCursor.ordinal()) >=
        requesterCount)
      return rejected(ResourceContractViolation::UnknownRequesterKey,
                      "grant policy reset cursor");

  return llvm::Error::success();
}

// A requester order is observable exactly when two requesters can claim one
// capacity dimension. That one proof both requires an exact policy and forbids
// an unnecessary one.
llvm::Error checkOrderingAgreement(const NormalizedDeclaration &normalized) {
  std::vector<std::size_t> offsets(normalized.stateCount() + 1, 0);
  for (std::size_t state = 0; state < normalized.stateCount(); ++state)
    offsets[state + 1] =
        offsets[state] + normalized.capacityDimensionCount(state);

  std::vector<std::optional<RequesterKey>> claimedBy(offsets.back());
  std::vector<bool> contended(offsets.back(), false);

  for (std::size_t pattern = 0; pattern < normalized.usePatternCount();
       ++pattern) {
    const RequesterKey requester = normalized.usePattern(pattern).requester;
    for (std::size_t claim = 0; claim < normalized.claimCount(pattern);
         ++claim) {
      const ClaimDeclaration &declared = normalized.claim(pattern, claim);
      const std::size_t dimension =
          offsets[declared.state.ordinal()] + declared.dimension.ordinal();
      if (!claimedBy[dimension])
        claimedBy[dimension] = requester;
      else if (*claimedBy[dimension] != requester)
        contended[dimension] = true;
    }
  }

  for (std::size_t state = 0; state < normalized.stateCount(); ++state)
    for (std::size_t dimension = 0;
         dimension < normalized.capacityDimensionCount(state); ++dimension) {
      if (!contended[offsets[state] + dimension])
        continue;
      if (!normalized.declaration.grantPolicy)
        return rejected(ResourceContractViolation::ContentionWithoutGrantPolicy,
                        dimensionSite(state, dimension));
      return llvm::Error::success();
    }

  if (normalized.declaration.grantPolicy)
    return rejected(ResourceContractViolation::GrantPolicyWithoutContention,
                    "grant policy");

  return llvm::Error::success();
}

// The precedence documented on ResourceContract::create is exactly the order of
// the classes below.
llvm::Error validate(NormalizedDeclaration &normalized) {
  const ResourceContractDeclaration &declaration = normalized.declaration;

  if (llvm::Error invalid = indexInventory(
          declaration.states.size(),
          [&](std::size_t position) {
            return declaration.states[position].key.ordinal();
          },
          ResourceContractViolation::DuplicateStateKey,
          ResourceContractViolation::UnknownStateKey, "state declaration",
          normalized.statePositions))
    return invalid;

  normalized.capacityDimensionPositions.resize(normalized.stateCount());
  for (std::size_t state = 0; state < normalized.stateCount(); ++state) {
    const ResourceStateDeclaration &declared = normalized.state(state);
    if (llvm::Error invalid = indexInventory(
            declared.capacityDimensions.size(),
            [&](std::size_t position) {
              return declared.capacityDimensions[position].key.ordinal();
            },
            ResourceContractViolation::DuplicateCapacityDimensionKey,
            ResourceContractViolation::UnknownCapacityDimensionKey,
            "state " + llvm::Twine(state) + " capacity dimension declaration",
            normalized.capacityDimensionPositions[state]))
      return invalid;
  }

  for (std::size_t state = 0; state < normalized.stateCount(); ++state)
    for (std::size_t dimension = 0;
         dimension < normalized.capacityDimensionCount(state); ++dimension) {
      const CapacityDimensionDeclaration &declared =
          normalized.capacityDimension(state, dimension);
      if (declared.initialOccupancy > declared.capacity)
        return rejected(
            ResourceContractViolation::InitialOccupancyExceedsCapacity,
            dimensionSite(state, dimension));
    }

  std::vector<std::uint32_t> transitionPositions;
  if (llvm::Error invalid = indexInventory(
          declaration.resourceTransitions.size(),
          [&](std::size_t position) {
            return declaration.resourceTransitions[position].ordinal();
          },
          ResourceContractViolation::DuplicateResourceTransitionKey,
          ResourceContractViolation::UnknownResourceTransitionKey,
          "resource transition declaration", transitionPositions))
    return invalid;

  if (llvm::Error invalid = indexInventory(
          declaration.timingContracts.size(),
          [&](std::size_t position) {
            return declaration.timingContracts[position].key.ordinal();
          },
          ResourceContractViolation::DuplicateTimingContractKey,
          ResourceContractViolation::UnknownTimingContractKey,
          "timing contract declaration", normalized.timingContractPositions))
    return invalid;

  // A timing contract orders a use only when it ranks every declared event.
  for (std::size_t contract = 0; contract < normalized.timingContractCount();
       ++contract)
    if (normalized.timingContract(contract).eventRank.size() !=
        declaration.eventCount)
      return rejected(ResourceContractViolation::TimingContractDoesNotOrderUse,
                      "timing contract " + llvm::Twine(contract));

  const std::size_t requesterCount = declaration.requesters.size();
  std::vector<std::uint32_t> requesterPositions;
  if (llvm::Error invalid = indexInventory(
          requesterCount,
          [&](std::size_t position) {
            return declaration.requesters[position].ordinal();
          },
          ResourceContractViolation::DuplicateRequesterKey,
          ResourceContractViolation::UnknownRequesterKey,
          "requester declaration", requesterPositions))
    return invalid;

  if (llvm::Error invalid = indexInventory(
          declaration.usePatterns.size(),
          [&](std::size_t position) {
            return declaration.usePatterns[position].key.ordinal();
          },
          ResourceContractViolation::DuplicateUsePatternKey,
          ResourceContractViolation::UnknownUsePatternKey,
          "use pattern declaration", normalized.usePatternPositions))
    return invalid;

  for (std::size_t pattern = 0; pattern < normalized.usePatternCount();
       ++pattern) {
    const UsePatternDeclaration &declared = normalized.usePattern(pattern);
    if (static_cast<std::size_t>(declared.requester.ordinal()) >=
        requesterCount)
      return rejected(ResourceContractViolation::UnknownRequesterKey,
                      patternSite(pattern));
    if (declared.eligibility.ordinal() >= declaration.eligibilityCount)
      return rejected(ResourceContractViolation::UnknownEligibilityKey,
                      patternSite(pattern));
    if (declared.acquire.ordinal() >= declaration.eventCount ||
        declared.release.ordinal() >= declaration.eventCount)
      return rejected(ResourceContractViolation::UnknownEventKey,
                      patternSite(pattern));
    if (declared.commit) {
      if (static_cast<std::size_t>(declared.commit->transition.ordinal()) >=
          declaration.resourceTransitions.size())
        return rejected(ResourceContractViolation::UnknownResourceTransitionKey,
                        patternSite(pattern));
      if (declared.commit->event.ordinal() >= declaration.eventCount)
        return rejected(ResourceContractViolation::UnknownEventKey,
                        patternSite(pattern));
    }
    if (static_cast<std::size_t>(declared.timingAndProgress.ordinal()) >=
        normalized.timingContractCount())
      return rejected(ResourceContractViolation::UnknownTimingContractKey,
                      patternSite(pattern));
    if (!declared.parameters.empty())
      return rejected(ResourceContractViolation::InvalidPatternValueSchema,
                      valueSchemaSite(pattern, "parameter schema", 0));
    if (llvm::Error invalid = checkValueSchemas(
            pattern, "sharing-assignment schema", declared.sharingAssignments))
      return invalid;
  }

  // The owning timing contract must place acquisition no later than the
  // optional commit and the commit no later than release, so an accepted use
  // always reaches its durable transition and then returns its whole claim
  // envelope.
  for (std::size_t pattern = 0; pattern < normalized.usePatternCount();
       ++pattern) {
    const UsePatternDeclaration &declared = normalized.usePattern(pattern);
    const llvm::ArrayRef<std::uint32_t> rank =
        normalized.timingContract(declared.timingAndProgress.ordinal())
            .eventRank;
    const std::uint32_t acquire = rank[declared.acquire.ordinal()];
    const std::uint32_t release = rank[declared.release.ordinal()];
    const bool ordered =
        declared.commit ? acquire <= rank[declared.commit->event.ordinal()] &&
                              rank[declared.commit->event.ordinal()] <= release
                        : acquire <= release;
    if (!ordered)
      return rejected(ResourceContractViolation::TimingContractDoesNotOrderUse,
                      patternSite(pattern));
  }

  normalized.claimPositions.resize(normalized.usePatternCount());
  for (std::size_t pattern = 0; pattern < normalized.usePatternCount();
       ++pattern) {
    const UsePatternDeclaration &declared = normalized.usePattern(pattern);
    if (llvm::Error invalid = indexInventory(
            declared.claims.size(),
            [&](std::size_t position) {
              return declared.claims[position].key.ordinal();
            },
            ResourceContractViolation::DuplicateClaimKey,
            ResourceContractViolation::UnknownClaimKey,
            "use pattern " + llvm::Twine(pattern) + " claim declaration",
            normalized.claimPositions[pattern]))
      return invalid;
  }

  for (std::size_t pattern = 0; pattern < normalized.usePatternCount();
       ++pattern)
    for (std::size_t claim = 0; claim < normalized.claimCount(pattern);
         ++claim) {
      const ClaimDeclaration &declared = normalized.claim(pattern, claim);
      if (static_cast<std::size_t>(declared.state.ordinal()) >=
          normalized.stateCount())
        return rejected(ResourceContractViolation::UnknownStateKey,
                        claimSite(pattern, claim));
      if (static_cast<std::size_t>(declared.dimension.ordinal()) >=
          normalized.capacityDimensionCount(declared.state.ordinal()))
        return rejected(ResourceContractViolation::UndeclaredClaim,
                        claimSite(pattern, claim));
    }

  // One atomic envelope claims a capacity dimension once.
  for (std::size_t pattern = 0; pattern < normalized.usePatternCount();
       ++pattern)
    for (std::size_t claim = 0; claim < normalized.claimCount(pattern);
         ++claim) {
      const ClaimDeclaration &declared = normalized.claim(pattern, claim);
      for (std::size_t earlier = 0; earlier < claim; ++earlier) {
        const ClaimDeclaration &previous = normalized.claim(pattern, earlier);
        if (previous.state == declared.state &&
            previous.dimension == declared.dimension)
          return rejected(ResourceContractViolation::DuplicateCapacityClaim,
                          claimSite(pattern, claim));
      }
    }

  for (std::size_t pattern = 0; pattern < normalized.usePatternCount();
       ++pattern)
    for (std::size_t claim = 0; claim < normalized.claimCount(pattern);
         ++claim) {
      const ClaimDeclaration &declared = normalized.claim(pattern, claim);
      const CapacityDimensionDeclaration &dimension =
          normalized.capacityDimension(declared.state.ordinal(),
                                       declared.dimension.ordinal());
      const std::optional<CapacityUnits> occupancy = CapacityUnits::checkedAdd(
          dimension.initialOccupancy, declared.amount);
      if (!occupancy)
        return rejected(ResourceContractViolation::CapacityArithmeticOverflow,
                        claimSite(pattern, claim));
      if (*occupancy > dimension.capacity)
        return rejected(ResourceContractViolation::ClaimExceedsCapacity,
                        claimSite(pattern, claim));
    }

  // A transaction selects an unordered subset of its pattern's claim envelope,
  // and the accepted selection is stored in ascending claim key order, so the
  // whole selection is scanned before it is reported.
  std::vector<bool> selected;
  for (std::size_t pattern = 0; pattern < normalized.usePatternCount();
       ++pattern) {
    const UsePatternDeclaration &declared = normalized.usePattern(pattern);
    for (std::size_t transaction = 0;
         transaction < declared.internalTransactions.size(); ++transaction) {
      std::optional<std::uint32_t> repeated;
      std::optional<std::uint32_t> foreign;
      selected.assign(declared.claims.size(), false);

      for (const ClaimKey entry :
           declared.internalTransactions[transaction].claims) {
        const std::uint32_t key = entry.ordinal();
        const bool inDomain =
            static_cast<std::size_t>(key) < declared.claims.size();
        const bool alreadyHeld = inDomain && selected[key];
        observeKey(key, inDomain, alreadyHeld, repeated, foreign);
        if (inDomain)
          selected[key] = true;
      }

      if (llvm::Error invalid = rejectedKeySet(
              repeated, foreign, ResourceContractViolation::DuplicateClaimKey,
              ResourceContractViolation::UnknownClaimKey,
              transactionSite(pattern, transaction)))
        return invalid;
    }
  }

  if (declaration.grantPolicy)
    if (llvm::Error invalid =
            checkGrantPolicy(*declaration.grantPolicy, requesterCount))
      return invalid;

  return checkOrderingAgreement(normalized);
}

} // namespace

llvm::StringRef
getResourceContractViolationName(ResourceContractViolation violation) {
  switch (violation) {
  case ResourceContractViolation::DuplicateStateKey:
    return "duplicate_state_key";
  case ResourceContractViolation::UnknownStateKey:
    return "unknown_state_key";
  case ResourceContractViolation::DuplicateCapacityDimensionKey:
    return "duplicate_capacity_dimension_key";
  case ResourceContractViolation::UnknownCapacityDimensionKey:
    return "unknown_capacity_dimension_key";
  case ResourceContractViolation::DuplicateResourceTransitionKey:
    return "duplicate_resource_transition_key";
  case ResourceContractViolation::UnknownResourceTransitionKey:
    return "unknown_resource_transition_key";
  case ResourceContractViolation::DuplicateTimingContractKey:
    return "duplicate_timing_contract_key";
  case ResourceContractViolation::TimingContractDoesNotOrderUse:
    return "timing_contract_does_not_order_use";
  case ResourceContractViolation::InitialOccupancyExceedsCapacity:
    return "initial_occupancy_exceeds_capacity";
  case ResourceContractViolation::DuplicateRequesterKey:
    return "duplicate_requester_key";
  case ResourceContractViolation::UnknownRequesterKey:
    return "unknown_requester_key";
  case ResourceContractViolation::DuplicateUsePatternKey:
    return "duplicate_use_pattern_key";
  case ResourceContractViolation::UnknownUsePatternKey:
    return "unknown_use_pattern_key";
  case ResourceContractViolation::UnknownEligibilityKey:
    return "unknown_eligibility_key";
  case ResourceContractViolation::UnknownEventKey:
    return "unknown_event_key";
  case ResourceContractViolation::UnknownTimingContractKey:
    return "unknown_timing_contract_key";
  case ResourceContractViolation::InvalidPatternValueSchema:
    return "invalid_pattern_value_schema";
  case ResourceContractViolation::DuplicateClaimKey:
    return "duplicate_claim_key";
  case ResourceContractViolation::UnknownClaimKey:
    return "unknown_claim_key";
  case ResourceContractViolation::UndeclaredClaim:
    return "undeclared_claim";
  case ResourceContractViolation::DuplicateCapacityClaim:
    return "duplicate_capacity_claim";
  case ResourceContractViolation::CapacityArithmeticOverflow:
    return "capacity_arithmetic_overflow";
  case ResourceContractViolation::ClaimExceedsCapacity:
    return "claim_exceeds_capacity";
  case ResourceContractViolation::DuplicateRequesterInGrantPolicy:
    return "duplicate_requester_in_grant_policy";
  case ResourceContractViolation::RequesterOmittedFromGrantPolicy:
    return "requester_omitted_from_grant_policy";
  case ResourceContractViolation::ContentionWithoutGrantPolicy:
    return "contention_without_grant_policy";
  case ResourceContractViolation::GrantPolicyWithoutContention:
    return "grant_policy_without_contention";
  }
  llvm_unreachable("unhandled resource contract violation");
}

char ResourceContractError::ID = 0;

void ResourceContractError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code ResourceContractError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

std::optional<RequesterKey>
FixedPriorityView::grant(llvm::ArrayRef<bool> eligible) const {
  assert(eligible.size() == order_.size() &&
         "eligibility must cover the exact requester domain");
  for (RequesterKey requester : order_)
    if (eligible[requester.ordinal()])
      return requester;
  return std::nullopt;
}

RoundRobinGrant RoundRobinView::grant(RequesterKey cursor,
                                      llvm::ArrayRef<bool> eligible) const {
  assert(eligible.size() == cycle_.size() &&
         "eligibility must cover the exact requester domain");

  std::size_t origin = 0;
  while (origin < cycle_.size() && cycle_[origin] != cursor)
    ++origin;
  assert(origin < cycle_.size() && "cursor outside the declared cycle");

  for (std::size_t step = 0; step < cycle_.size(); ++step) {
    const std::size_t position = (origin + step) % cycle_.size();
    const RequesterKey requester = cycle_[position];
    if (!eligible[requester.ordinal()])
      continue;
    return RoundRobinGrant{requester, cycle_[(position + 1) % cycle_.size()]};
  }

  return RoundRobinGrant{std::nullopt, cursor};
}

UsePattern ResourceContract::usePattern(UsePatternKey key) const {
  assert(key.ordinal() < patterns_.size() && "undeclared use pattern");
  const PatternRecord &record = patterns_[key.ordinal()];
  return UsePattern{
      record.requester,
      record.eligibility,
      record.acquire,
      record.release,
      record.commit,
      record.timingAndProgress,
      llvm::ArrayRef<Claim>(claims_).slice(record.claims.first,
                                           record.claims.count),
      record.internalTransactions.count,
      llvm::ArrayRef<UsePatternValueSchema>(valueSchemas_)
          .slice(record.parameters.first, record.parameters.count),
      llvm::ArrayRef<UsePatternValueSchema>(valueSchemas_)
          .slice(record.sharingAssignments.first,
                 record.sharingAssignments.count)};
}

llvm::ArrayRef<std::uint32_t>
ResourceContract::eventOrder(TimingContractKey key) const {
  assert(key.ordinal() < timingContractCount_ && "undeclared timing contract");
  return llvm::ArrayRef<std::uint32_t>(eventRanks_)
      .slice(static_cast<std::size_t>(key.ordinal()) * eventCount_,
             eventCount_);
}

llvm::ArrayRef<ClaimKey>
ResourceContract::internalTransaction(UsePatternKey key,
                                      std::uint32_t transaction) const {
  assert(key.ordinal() < patterns_.size() && "undeclared use pattern");
  const PatternRecord &record = patterns_[key.ordinal()];
  assert(transaction < record.internalTransactions.count &&
         "undeclared internal transaction");
  const Span span =
      internalTransactions_[record.internalTransactions.first + transaction];
  return llvm::ArrayRef<ClaimKey>(transactionClaims_)
      .slice(span.first, span.count);
}

std::optional<GrantPolicyView> ResourceContract::grantPolicy() const {
  if (!grantPolicyKind_)
    return std::nullopt;
  if (*grantPolicyKind_ == GrantPolicyKind::FixedPriority)
    return GrantPolicyView(FixedPriorityView(requesterOrder_));
  return GrantPolicyView(RoundRobinView(requesterOrder_, resetCursorPosition_));
}

ResourceContractDeclaration ResourceContract::declaration() const {
  ResourceContractDeclaration result;
  result.states.reserve(stateCount());
  for (std::uint32_t state = 0; state != stateCount(); ++state) {
    ResourceStateDeclaration declared{StateKey(state), {}};
    for (const CapacityDimension &dimension :
         capacityDimensions(StateKey(state)))
      declared.capacityDimensions.push_back(CapacityDimensionDeclaration{
          CapacityDimensionKey(
              static_cast<std::uint32_t>(declared.capacityDimensions.size())),
          dimension.capacity, dimension.initialOccupancy});
    result.states.push_back(std::move(declared));
  }

  result.resourceTransitions.reserve(resourceTransitionCount());
  for (std::uint32_t transition = 0; transition != resourceTransitionCount();
       ++transition)
    result.resourceTransitions.emplace_back(transition);

  result.timingContracts.reserve(timingContractCount());
  for (std::uint32_t timing = 0; timing != timingContractCount(); ++timing) {
    const llvm::ArrayRef<std::uint32_t> ranks =
        eventOrder(TimingContractKey(timing));
    result.timingContracts.push_back(TimingContractDeclaration{
        TimingContractKey(timing),
        std::vector<std::uint32_t>(ranks.begin(), ranks.end())});
  }

  result.usePatterns.reserve(usePatternCount());
  for (std::uint32_t ordinal = 0; ordinal != usePatternCount(); ++ordinal) {
    const UsePattern pattern = usePattern(UsePatternKey(ordinal));
    UsePatternDeclaration declared{
        UsePatternKey(ordinal),
        pattern.requester,
        pattern.eligibility,
        pattern.acquire,
        pattern.release,
        pattern.commit ? std::optional<CommitDeclaration>(CommitDeclaration{
                             pattern.commit->event, pattern.commit->transition})
                       : std::nullopt,
        pattern.timingAndProgress,
        {},
        {},
        std::vector<UsePatternValueSchema>(pattern.parameters.begin(),
                                           pattern.parameters.end()),
        std::vector<UsePatternValueSchema>(pattern.sharingAssignments.begin(),
                                           pattern.sharingAssignments.end())};
    declared.claims.reserve(pattern.claims.size());
    for (const Claim &claim : pattern.claims)
      declared.claims.push_back(ClaimDeclaration{
          ClaimKey(static_cast<std::uint32_t>(declared.claims.size())),
          claim.state, claim.dimension, claim.amount});
    declared.internalTransactions.reserve(pattern.internalTransactionCount);
    for (std::uint32_t transaction = 0;
         transaction != pattern.internalTransactionCount; ++transaction) {
      const llvm::ArrayRef<ClaimKey> claims =
          internalTransaction(UsePatternKey(ordinal), transaction);
      declared.internalTransactions.push_back(
          {std::vector<ClaimKey>(claims.begin(), claims.end())});
    }
    result.usePatterns.push_back(std::move(declared));
  }

  result.requesters.reserve(requesterCount());
  for (std::uint32_t requester = 0; requester != requesterCount(); ++requester)
    result.requesters.emplace_back(requester);
  result.eligibilityCount = eligibilityCount();
  result.eventCount = eventCount();

  if (const std::optional<GrantPolicyView> policy = grantPolicy()) {
    if (const auto *fixed = std::get_if<FixedPriorityView>(&*policy))
      result.grantPolicy = FixedPriorityDeclaration{std::vector<RequesterKey>(
          fixed->requesterOrder().begin(), fixed->requesterOrder().end())};
    else {
      const auto &roundRobin = std::get<RoundRobinView>(*policy);
      result.grantPolicy = RoundRobinDeclaration{
          std::vector<RequesterKey>(roundRobin.requesterCycle().begin(),
                                    roundRobin.requesterCycle().end()),
          roundRobin.resetCursor()};
    }
  }
  return result;
}

llvm::Expected<ResourceContract>
ResourceContract::create(const ResourceContractDeclaration &declaration) {
  NormalizedDeclaration normalized(declaration);
  if (llvm::Error invalid = validate(normalized))
    return std::move(invalid);

  ResourceContract contract;
  contract.resourceTransitionCount_ =
      static_cast<std::uint32_t>(declaration.resourceTransitions.size());
  contract.requesterCount_ =
      static_cast<std::uint32_t>(declaration.requesters.size());
  contract.eligibilityCount_ = declaration.eligibilityCount;
  contract.eventCount_ = declaration.eventCount;
  contract.timingContractCount_ =
      static_cast<std::uint32_t>(declaration.timingContracts.size());

  contract.eventRanks_.reserve(
      static_cast<std::size_t>(contract.timingContractCount_) *
      declaration.eventCount);
  for (std::size_t timing = 0; timing < normalized.timingContractCount();
       ++timing) {
    const llvm::ArrayRef<std::uint32_t> rank =
        normalized.timingContract(timing).eventRank;
    contract.eventRanks_.insert(contract.eventRanks_.end(), rank.begin(),
                                rank.end());
  }

  for (std::size_t state = 0; state < normalized.stateCount(); ++state) {
    const Span span{
        static_cast<std::uint32_t>(contract.capacityDimensions_.size()),
        static_cast<std::uint32_t>(normalized.capacityDimensionCount(state))};
    for (std::size_t dimension = 0; dimension < span.count; ++dimension) {
      const CapacityDimensionDeclaration &declared =
          normalized.capacityDimension(state, dimension);
      contract.capacityDimensions_.push_back(
          CapacityDimension{declared.capacity, declared.initialOccupancy});
    }
    contract.states_.push_back(span);
  }

  for (std::size_t pattern = 0; pattern < normalized.usePatternCount();
       ++pattern) {
    const UsePatternDeclaration &declared = normalized.usePattern(pattern);
    const PatternRecord record{
        declared.requester,
        declared.eligibility,
        declared.acquire,
        declared.release,
        declared.commit
            ? std::optional<Commit>(
                  Commit{declared.commit->event, declared.commit->transition})
            : std::nullopt,
        declared.timingAndProgress,
        Span{static_cast<std::uint32_t>(contract.claims_.size()),
             static_cast<std::uint32_t>(normalized.claimCount(pattern))},
        Span{static_cast<std::uint32_t>(contract.internalTransactions_.size()),
             static_cast<std::uint32_t>(declared.internalTransactions.size())},
        Span{static_cast<std::uint32_t>(contract.valueSchemas_.size()),
             static_cast<std::uint32_t>(declared.parameters.size())},
        Span{static_cast<std::uint32_t>(contract.valueSchemas_.size() +
                                        declared.parameters.size()),
             static_cast<std::uint32_t>(declared.sharingAssignments.size())}};

    contract.valueSchemas_.insert(contract.valueSchemas_.end(),
                                  declared.parameters.begin(),
                                  declared.parameters.end());
    contract.valueSchemas_.insert(contract.valueSchemas_.end(),
                                  declared.sharingAssignments.begin(),
                                  declared.sharingAssignments.end());

    for (std::size_t claim = 0; claim < record.claims.count; ++claim) {
      const ClaimDeclaration &declaredClaim = normalized.claim(pattern, claim);
      contract.claims_.push_back(Claim{
          declaredClaim.state, declaredClaim.dimension, declaredClaim.amount});
    }

    for (const InternalTransactionDeclaration &internal :
         declared.internalTransactions) {
      const Span entries{
          static_cast<std::uint32_t>(contract.transactionClaims_.size()),
          static_cast<std::uint32_t>(internal.claims.size())};
      contract.transactionClaims_.insert(contract.transactionClaims_.end(),
                                         internal.claims.begin(),
                                         internal.claims.end());
      std::sort(contract.transactionClaims_.begin() + entries.first,
                contract.transactionClaims_.end(),
                [](ClaimKey lhs, ClaimKey rhs) {
                  return lhs.ordinal() < rhs.ordinal();
                });
      contract.internalTransactions_.push_back(entries);
    }

    contract.patterns_.push_back(record);
  }

  if (declaration.grantPolicy) {
    const llvm::ArrayRef<RequesterKey> permutation =
        declaredPermutation(*declaration.grantPolicy);
    contract.requesterOrder_.assign(permutation.begin(), permutation.end());
    if (const auto *roundRobin =
            std::get_if<RoundRobinDeclaration>(&*declaration.grantPolicy)) {
      contract.grantPolicyKind_ = GrantPolicyKind::RoundRobin;
      contract.resetCursorPosition_ = static_cast<std::uint32_t>(
          std::find(permutation.begin(), permutation.end(),
                    roundRobin->resetCursor) -
          permutation.begin());
    } else {
      contract.grantPolicyKind_ = GrantPolicyKind::FixedPriority;
    }
  }

  return contract;
}

} // namespace fabric

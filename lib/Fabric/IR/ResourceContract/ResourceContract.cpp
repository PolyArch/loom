#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace fabric {

namespace {

llvm::Error rejected(ResourceContractViolation violation,
                     const llvm::Twine &site) {
  return llvm::make_error<ResourceContractError>(
      violation,
      (llvm::Twine(getResourceContractViolationName(violation)) + " at " + site)
          .str());
}

std::string dimensionSite(std::size_t state, std::size_t dimension) {
  return ("state " + llvm::Twine(state) + " dimension " +
          llvm::Twine(dimension))
      .str();
}

std::string claimSite(std::size_t pattern, std::size_t claim) {
  return ("use pattern " + llvm::Twine(pattern) + " claim " +
          llvm::Twine(claim))
      .str();
}

std::string transactionSite(std::size_t pattern, std::size_t transaction,
                            std::size_t entry) {
  return ("use pattern " + llvm::Twine(pattern) + " internal transaction " +
          llvm::Twine(transaction) + " entry " + llvm::Twine(entry))
      .str();
}

llvm::ArrayRef<RequesterKey> requesterPermutation(const GrantPolicy &policy) {
  if (const auto *fixed = std::get_if<FixedPriority>(&policy))
    return fixed->requesterOrder;
  return std::get<RoundRobin>(policy).requesterCycle;
}

// Flat index of every declared (state, capacity dimension) pair, used to reason
// about contention over one capacity in declared order.
std::vector<std::size_t>
capacityDimensionOffsets(const ResourceContractDeclaration &declaration) {
  std::vector<std::size_t> offsets(declaration.states.size() + 1, 0);
  for (std::size_t state = 0; state < declaration.states.size(); ++state)
    offsets[state + 1] =
        offsets[state] + declaration.states[state].capacityDimensions.size();
  return offsets;
}

// The exact contention proof documented on ResourceContractDeclaration.
llvm::Error
checkUnarbitratedContention(const ResourceContractDeclaration &declaration) {
  const std::vector<std::size_t> offsets =
      capacityDimensionOffsets(declaration);
  std::vector<std::optional<RequesterKey>> claimedBy(offsets.back());
  std::vector<bool> contended(offsets.back(), false);

  for (const UsePattern &pattern : declaration.usePatterns)
    for (const Claim &claim : pattern.claims) {
      const std::size_t dimension =
          offsets[claim.state.ordinal()] + claim.dimension.ordinal();
      if (!claimedBy[dimension])
        claimedBy[dimension] = pattern.requester;
      else if (*claimedBy[dimension] != pattern.requester)
        contended[dimension] = true;
    }

  for (std::size_t state = 0; state < declaration.states.size(); ++state)
    for (std::size_t dimension = 0;
         dimension < declaration.states[state].capacityDimensions.size();
         ++dimension)
      if (contended[offsets[state] + dimension])
        return rejected(ResourceContractViolation::ContentionWithoutGrantPolicy,
                        dimensionSite(state, dimension));

  return llvm::Error::success();
}

llvm::Error checkGrantPolicy(const ResourceContractDeclaration &declaration,
                             const GrantPolicy &policy) {
  const llvm::ArrayRef<RequesterKey> permutation = requesterPermutation(policy);
  std::vector<bool> ordered(declaration.requesterCount, false);

  for (std::size_t position = 0; position < permutation.size(); ++position) {
    const RequesterKey requester = permutation[position];
    if (requester.ordinal() >= declaration.requesterCount)
      return rejected(ResourceContractViolation::UnknownRequesterKey,
                      "grant policy position " + llvm::Twine(position));
    if (ordered[requester.ordinal()])
      return rejected(
          ResourceContractViolation::DuplicateRequesterInGrantPolicy,
          "grant policy position " + llvm::Twine(position));
    ordered[requester.ordinal()] = true;
  }

  for (std::uint32_t requester = 0; requester < declaration.requesterCount;
       ++requester)
    if (!ordered[requester])
      return rejected(
          ResourceContractViolation::RequesterOmittedFromGrantPolicy,
          "requester " + llvm::Twine(requester));

  if (const auto *roundRobin = std::get_if<RoundRobin>(&policy))
    if (roundRobin->resetCursor.ordinal() >= declaration.requesterCount)
      return rejected(ResourceContractViolation::UnknownRequesterKey,
                      "grant policy reset cursor");

  return llvm::Error::success();
}

// The precedence documented on ResourceContract::create is exactly the order of
// the passes below.
llvm::Error
validateDeclaration(const ResourceContractDeclaration &declaration) {
  for (std::size_t state = 0; state < declaration.states.size(); ++state) {
    const ResourceState &declared = declaration.states[state];
    for (std::size_t index = 0; index < declared.capacityDimensions.size();
         ++index)
      if (declared.capacityDimensions[index].initialOccupancy >
          declared.capacityDimensions[index].capacity)
        return rejected(
            ResourceContractViolation::InitialOccupancyExceedsCapacity,
            dimensionSite(state, index));
  }

  for (std::size_t index = 0; index < declaration.usePatterns.size(); ++index) {
    const UsePattern &pattern = declaration.usePatterns[index];
    if (pattern.requester.ordinal() >= declaration.requesterCount)
      return rejected(ResourceContractViolation::UnknownRequesterKey,
                      "use pattern " + llvm::Twine(index));
    if (pattern.eligibility.ordinal() >= declaration.eligibilityCount)
      return rejected(ResourceContractViolation::UnknownEligibilityKey,
                      "use pattern " + llvm::Twine(index));
    if (pattern.acquire.ordinal() >= declaration.eventCount)
      return rejected(ResourceContractViolation::UnknownEventKey,
                      "use pattern " + llvm::Twine(index));
    if (pattern.timingAndProgress.ordinal() >= declaration.timingContractCount)
      return rejected(ResourceContractViolation::UnknownTimingContractKey,
                      "use pattern " + llvm::Twine(index));
  }

  for (std::size_t index = 0; index < declaration.usePatterns.size(); ++index) {
    const UsePattern &pattern = declaration.usePatterns[index];
    for (std::size_t claimIndex = 0; claimIndex < pattern.claims.size();
         ++claimIndex) {
      const Claim &claim = pattern.claims[claimIndex];
      if (claim.state.ordinal() >= declaration.states.size())
        return rejected(ResourceContractViolation::UnknownStateKey,
                        claimSite(index, claimIndex));
      if (claim.dimension.ordinal() >=
          declaration.states[claim.state.ordinal()].capacityDimensions.size())
        return rejected(ResourceContractViolation::UndeclaredClaim,
                        claimSite(index, claimIndex));
      if (claim.release.ordinal() >= declaration.eventCount)
        return rejected(ResourceContractViolation::UnknownEventKey,
                        claimSite(index, claimIndex));
    }
  }

  // One pattern claims a capacity dimension at most once. A repeat with the
  // same release duplicates a claim key; a repeat with another release splits
  // the release of one capacity.
  for (std::size_t index = 0; index < declaration.usePatterns.size(); ++index) {
    const UsePattern &pattern = declaration.usePatterns[index];
    for (std::size_t claimIndex = 0; claimIndex < pattern.claims.size();
         ++claimIndex) {
      const Claim &claim = pattern.claims[claimIndex];
      for (std::size_t earlier = 0; earlier < claimIndex; ++earlier) {
        const Claim &previous = pattern.claims[earlier];
        if (previous.state != claim.state ||
            previous.dimension != claim.dimension)
          continue;
        return rejected(previous.release == claim.release
                            ? ResourceContractViolation::DuplicateClaim
                            : ResourceContractViolation::AmbiguousRelease,
                        claimSite(index, claimIndex));
      }
    }
  }

  for (std::size_t index = 0; index < declaration.usePatterns.size(); ++index) {
    const UsePattern &pattern = declaration.usePatterns[index];
    for (std::size_t claimIndex = 0; claimIndex < pattern.claims.size();
         ++claimIndex) {
      const Claim &claim = pattern.claims[claimIndex];
      const CapacityDimension &dimension =
          declaration.states[claim.state.ordinal()]
              .capacityDimensions[claim.dimension.ordinal()];
      const std::optional<CapacityUnits> occupancy =
          CapacityUnits::checkedAdd(dimension.initialOccupancy, claim.amount);
      if (!occupancy || *occupancy > dimension.capacity)
        return rejected(ResourceContractViolation::ClaimExceedsCapacity,
                        claimSite(index, claimIndex));
    }
  }

  for (std::size_t index = 0; index < declaration.usePatterns.size(); ++index) {
    const UsePattern &pattern = declaration.usePatterns[index];
    for (std::size_t transaction = 0;
         transaction < pattern.internalTransactions.size(); ++transaction) {
      const InternalTransaction &internal =
          pattern.internalTransactions[transaction];
      for (std::size_t entry = 0; entry < internal.claims.size(); ++entry) {
        if (internal.claims[entry].ordinal() >= pattern.claims.size())
          return rejected(ResourceContractViolation::UnknownClaimKey,
                          transactionSite(index, transaction, entry));
        for (std::size_t earlier = 0; earlier < entry; ++earlier)
          if (internal.claims[earlier] == internal.claims[entry])
            return rejected(ResourceContractViolation::DuplicateClaim,
                            transactionSite(index, transaction, entry));
      }
    }
  }

  if (declaration.grantPolicy)
    return checkGrantPolicy(declaration, *declaration.grantPolicy);

  return checkUnarbitratedContention(declaration);
}

} // namespace

llvm::StringRef
getResourceContractViolationName(ResourceContractViolation violation) {
  switch (violation) {
  case ResourceContractViolation::InitialOccupancyExceedsCapacity:
    return "initial_occupancy_exceeds_capacity";
  case ResourceContractViolation::UnknownRequesterKey:
    return "unknown_requester_key";
  case ResourceContractViolation::UnknownEligibilityKey:
    return "unknown_eligibility_key";
  case ResourceContractViolation::UnknownEventKey:
    return "unknown_event_key";
  case ResourceContractViolation::UnknownTimingContractKey:
    return "unknown_timing_contract_key";
  case ResourceContractViolation::UnknownStateKey:
    return "unknown_state_key";
  case ResourceContractViolation::UndeclaredClaim:
    return "undeclared_claim";
  case ResourceContractViolation::DuplicateClaim:
    return "duplicate_claim";
  case ResourceContractViolation::AmbiguousRelease:
    return "ambiguous_release";
  case ResourceContractViolation::ClaimExceedsCapacity:
    return "claim_exceeds_capacity";
  case ResourceContractViolation::UnknownClaimKey:
    return "unknown_claim_key";
  case ResourceContractViolation::DuplicateRequesterInGrantPolicy:
    return "duplicate_requester_in_grant_policy";
  case ResourceContractViolation::RequesterOmittedFromGrantPolicy:
    return "requester_omitted_from_grant_policy";
  case ResourceContractViolation::ContentionWithoutGrantPolicy:
    return "contention_without_grant_policy";
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

RequesterKey resetGrantCursor(const GrantPolicy &policy) {
  if (const auto *roundRobin = std::get_if<RoundRobin>(&policy))
    return roundRobin->resetCursor;
  const llvm::ArrayRef<RequesterKey> order = requesterPermutation(policy);
  assert(!order.empty() && "a policy without a requester never arbitrates");
  return order.front();
}

GrantDecision arbitrate(const GrantPolicy &policy, RequesterKey cursor,
                        llvm::ArrayRef<bool> eligible) {
  const llvm::ArrayRef<RequesterKey> order = requesterPermutation(policy);
  assert(!order.empty() && "a policy without a requester never arbitrates");
  assert(eligible.size() == order.size() &&
         "eligibility must cover the exact requester domain");

  const bool advances = std::holds_alternative<RoundRobin>(policy);
  std::size_t origin = 0;
  if (advances) {
    while (origin < order.size() && order[origin] != cursor)
      ++origin;
    assert(origin < order.size() && "cursor outside the declared cycle");
  }

  for (std::size_t step = 0; step < order.size(); ++step) {
    const std::size_t position = (origin + step) % order.size();
    const RequesterKey requester = order[position];
    if (!eligible[requester.ordinal()])
      continue;
    return GrantDecision{requester, advances
                                        ? order[(position + 1) % order.size()]
                                        : order.front()};
  }

  return GrantDecision{std::nullopt, advances ? cursor : order.front()};
}

llvm::Expected<ResourceContract>
ResourceContract::create(ResourceContractDeclaration declaration) {
  if (llvm::Error invalid = validateDeclaration(declaration))
    return std::move(invalid);
  return ResourceContract(std::move(declaration));
}

} // namespace fabric

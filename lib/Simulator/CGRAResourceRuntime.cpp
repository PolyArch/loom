#include "CGRAResourceRuntime.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <set>
#include <system_error>
#include <tuple>
#include <utility>
#include <variant>

namespace loom::sim::detail {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), message);
}

struct OwnerScratch final {
  const ::fabric::ResourceContract *contract = nullptr;
  std::uint64_t dimensionOffset = 0;
  std::vector<std::uint64_t> stateOffsets;
  std::vector<std::uint64_t> parents;
};

std::uint64_t findRoot(OwnerScratch &owner, std::uint64_t dimension) {
  std::uint64_t root = dimension;
  while (owner.parents[root] != root)
    root = owner.parents[root];
  while (owner.parents[dimension] != dimension) {
    const std::uint64_t next = owner.parents[dimension];
    owner.parents[dimension] = root;
    dimension = next;
  }
  return root;
}

void join(OwnerScratch &owner, std::uint64_t lhs, std::uint64_t rhs) {
  lhs = findRoot(owner, lhs);
  rhs = findRoot(owner, rhs);
  if (lhs == rhs)
    return;
  if (rhs < lhs)
    std::swap(lhs, rhs);
  owner.parents[rhs] = lhs;
}

llvm::Expected<std::uint64_t> localDimension(const OwnerScratch &owner,
                                             const ::fabric::Claim &claim) {
  if (claim.state.ordinal() >= owner.contract->stateCount())
    return invalid("CGRA resource claim has an unknown state");
  const auto dimensions = owner.contract->capacityDimensions(claim.state);
  if (claim.dimension.ordinal() >= dimensions.size())
    return invalid("CGRA resource claim has an unknown capacity dimension");
  return owner.stateOffsets[claim.state.ordinal()] + claim.dimension.ordinal();
}

struct DomainKey final {
  std::uint64_t owner = 0;
  std::uint64_t root = 0;

  friend bool operator<(const DomainKey &lhs, const DomainKey &rhs) {
    return std::tie(lhs.owner, lhs.root) < std::tie(rhs.owner, rhs.root);
  }
};

struct UseRoot final {
  std::optional<DomainKey> domain;
};

struct PolicyProjection final {
  CgraGrantPolicyKind kind = CgraGrantPolicyKind::None;
  std::vector<std::uint32_t> fullOrder;
  std::optional<std::uint32_t> resetRequester;
};

PolicyProjection projectPolicy(const ::fabric::ResourceContract &contract) {
  const auto policy = contract.grantPolicy();
  if (!policy)
    return {};
  if (const auto *fixed = std::get_if<::fabric::FixedPriorityView>(&*policy)) {
    PolicyProjection result;
    result.kind = CgraGrantPolicyKind::FixedPriority;
    for (::fabric::RequesterKey requester : fixed->requesterOrder())
      result.fullOrder.push_back(requester.ordinal());
    return result;
  }
  const auto &roundRobin = std::get<::fabric::RoundRobinView>(*policy);
  PolicyProjection result;
  result.kind = CgraGrantPolicyKind::RoundRobin;
  for (::fabric::RequesterKey requester : roundRobin.requesterCycle())
    result.fullOrder.push_back(requester.ordinal());
  result.resetRequester = roundRobin.resetCursor().ordinal();
  return result;
}

std::uint32_t cyclicDistance(std::uint32_t position,
                             std::uint32_t resetPosition, std::uint32_t size) {
  return position >= resetPosition ? position - resetPosition
                                   : size - resetPosition + position;
}

struct PendingRequest final {
  CgraResourceRequest request;
  std::uint64_t domain = noCgraResourceDomain;
  std::uint32_t requesterPosition = 0;
};

bool pendingLess(const PendingRequest &lhs, const PendingRequest &rhs) {
  return std::tie(lhs.domain, lhs.requesterPosition,
                  lhs.request.selectedUseOrdinal,
                  lhs.request.occurrenceOrdinal) <
         std::tie(rhs.domain, rhs.requesterPosition,
                  rhs.request.selectedUseOrdinal,
                  rhs.request.occurrenceOrdinal);
}

} // namespace

llvm::Expected<CgraResourceRuntimePlan> freezeCgraResourceRuntimePlan(
    llvm::ArrayRef<const ::fabric::ResourceContract *> ownerContracts,
    llvm::ArrayRef<CgraResourcePatternSelection> selectedPatterns,
    llvm::ArrayRef<CgraResourceActivationSelection> activations) {
  CgraResourceRuntimePlan result;
  std::vector<OwnerScratch> owners;
  owners.reserve(ownerContracts.size());

  for (const ::fabric::ResourceContract *contract : ownerContracts) {
    if (!contract)
      return invalid("CGRA resource owner has no ResourceContract");
    OwnerScratch owner;
    owner.contract = contract;
    owner.dimensionOffset = result.dimensions.size();
    owner.stateOffsets.reserve(contract->stateCount());
    for (std::uint32_t state = 0; state != contract->stateCount(); ++state) {
      owner.stateOffsets.push_back(result.dimensions.size() -
                                   owner.dimensionOffset);
      for (const ::fabric::CapacityDimension &dimension :
           contract->capacityDimensions(::fabric::StateKey(state)))
        result.dimensions.push_back(
            {dimension.capacity.value(), dimension.initialOccupancy.value()});
    }
    const std::uint64_t dimensionCount =
        result.dimensions.size() - owner.dimensionOffset;
    owner.parents.resize(dimensionCount);
    for (std::uint64_t dimension = 0; dimension != dimensionCount; ++dimension)
      owner.parents[dimension] = dimension;

    for (std::uint32_t patternOrdinal = 0;
         patternOrdinal != contract->usePatternCount(); ++patternOrdinal) {
      const ::fabric::UsePattern pattern =
          contract->usePattern(::fabric::UsePatternKey(patternOrdinal));
      if (pattern.claims.empty())
        continue;
      auto first = localDimension(owner, pattern.claims.front());
      if (!first)
        return first.takeError();
      for (const ::fabric::Claim &claim : pattern.claims.drop_front()) {
        auto current = localDimension(owner, claim);
        if (!current)
          return current.takeError();
        join(owner, *first, *current);
      }
    }
    owners.push_back(std::move(owner));
  }

  std::vector<UseRoot> useRoots;
  useRoots.reserve(activations.size());
  std::map<DomainKey, std::set<std::uint32_t>> domainRequesters;
  result.selectedUses.reserve(activations.size());
  std::uint64_t expectedPatternOffset = 0;
  for (const CgraResourceActivationSelection &activation : activations) {
    if (activation.patternCount == 0 ||
        activation.patternOffset != expectedPatternOffset ||
        activation.patternOffset > selectedPatterns.size() ||
        activation.patternCount >
            selectedPatterns.size() - activation.patternOffset)
      return invalid("CGRA resource activation pattern slices are not a "
                     "complete canonical partition");
    expectedPatternOffset += activation.patternCount;
    const auto patterns = selectedPatterns.slice(activation.patternOffset,
                                                 activation.patternCount);
    const CgraResourcePatternSelection &first = patterns.front();
    if (first.ownerOrdinal >= owners.size())
      return invalid("CGRA selected use has an unknown resource owner");
    OwnerScratch &owner = owners[first.ownerOrdinal];
    std::optional<std::uint32_t> requester;
    std::map<std::uint64_t, std::uint32_t> claims;
    std::set<std::uint32_t> uniquePatterns;
    for (const CgraResourcePatternSelection &selected : patterns) {
      if (selected.ownerOrdinal != first.ownerOrdinal)
        return invalid("CGRA atomic activation spans resource owners");
      if (selected.pattern.ordinal() >= owner.contract->usePatternCount())
        return invalid("CGRA selected use has an unknown UsePattern");
      if (!uniquePatterns.insert(selected.pattern.ordinal()).second)
        continue;
      const ::fabric::UsePattern pattern =
          owner.contract->usePattern(selected.pattern);
      if (requester && *requester != pattern.requester.ordinal())
        return invalid("CGRA atomic activation spans resource requesters");
      requester = pattern.requester.ordinal();
      for (const ::fabric::Claim &claim : pattern.claims) {
        auto local = localDimension(owner, claim);
        if (!local)
          return local.takeError();
        auto [position, inserted] =
            claims.try_emplace(*local, claim.amount.value());
        if (!inserted && position->second != claim.amount.value())
          return invalid("CGRA atomic activation has inconsistent duplicate "
                         "claims");
      }
    }
    if (!requester)
      return invalid("CGRA atomic activation has no exact UsePattern");
    if (claims.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA selected use claim count exceeds u32");
    CgraResourceUsePlan use;
    use.ownerOrdinal = first.ownerOrdinal;
    use.requesterOrdinal = *requester;
    use.claimOffset = result.claims.size();
    use.claimCount = static_cast<std::uint32_t>(claims.size());
    std::optional<DomainKey> domain;
    for (const auto &[local, amount] : claims) {
      const DomainKey current{first.ownerOrdinal, findRoot(owner, local)};
      if (domain &&
          (domain->owner != current.owner || domain->root != current.root))
        return invalid(
            "CGRA atomic activation claim envelope spans arbitration "
            "components");
      domain = current;
      result.claims.push_back({owner.dimensionOffset + local, amount});
    }
    if (domain)
      domainRequesters[*domain].insert(*requester);
    useRoots.push_back({domain});
    result.selectedUses.push_back(use);
  }
  if (expectedPatternOffset != selectedPatterns.size())
    return invalid("CGRA resource activation leaves unowned UsePatterns");

  std::map<DomainKey, std::uint64_t> domainOrdinals;
  std::map<std::pair<std::uint64_t, std::uint32_t>, std::uint32_t>
      requesterPositions;
  for (const auto &[key, selectedRequesters] : domainRequesters) {
    const PolicyProjection policy = projectPolicy(*owners[key.owner].contract);
    std::vector<std::uint32_t> orderedRequesters;
    if (policy.kind == CgraGrantPolicyKind::None) {
      orderedRequesters.assign(selectedRequesters.begin(),
                               selectedRequesters.end());
    } else {
      for (std::uint32_t requester : policy.fullOrder)
        if (selectedRequesters.count(requester))
          orderedRequesters.push_back(requester);
      if (orderedRequesters.size() != selectedRequesters.size())
        return invalid("CGRA GrantPolicy omits a selected requester");
    }

    if (result.domains.size() == std::numeric_limits<std::uint64_t>::max())
      return invalid("CGRA resource domain count exceeds u64");
    const std::uint64_t domainOrdinal = result.domains.size();
    domainOrdinals.emplace(key, domainOrdinal);
    CgraResourceDomainPlan domain;
    domain.ownerOrdinal = key.owner;
    domain.policy = policy.kind;
    domain.requesterOffset = result.domainRequesters.size();
    if (orderedRequesters.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("CGRA resource requester domain exceeds u32");
    domain.requesterCount =
        static_cast<std::uint32_t>(orderedRequesters.size());
    if (policy.kind == CgraGrantPolicyKind::RoundRobin) {
      const auto reset = llvm::find(policy.fullOrder, *policy.resetRequester);
      if (reset == policy.fullOrder.end())
        return invalid("CGRA round-robin reset requester is absent");
      const std::uint32_t resetPosition =
          static_cast<std::uint32_t>(reset - policy.fullOrder.begin());
      std::uint32_t bestDistance = std::numeric_limits<std::uint32_t>::max();
      for (auto [position, requester] : llvm::enumerate(orderedRequesters)) {
        const auto full = llvm::find(policy.fullOrder, requester);
        const std::uint32_t fullPosition =
            static_cast<std::uint32_t>(full - policy.fullOrder.begin());
        const std::uint32_t distance =
            cyclicDistance(fullPosition, resetPosition,
                           static_cast<std::uint32_t>(policy.fullOrder.size()));
        if (distance < bestDistance) {
          bestDistance = distance;
          domain.resetPosition = static_cast<std::uint32_t>(position);
        }
      }
    }
    for (auto [position, requester] : llvm::enumerate(orderedRequesters)) {
      result.domainRequesters.push_back(requester);
      requesterPositions.emplace(std::make_pair(domainOrdinal, requester),
                                 static_cast<std::uint32_t>(position));
    }
    result.domains.push_back(domain);
  }

  for (std::size_t ordinal = 0; ordinal != result.selectedUses.size();
       ++ordinal) {
    if (!useRoots[ordinal].domain)
      continue;
    auto domain = domainOrdinals.find(*useRoots[ordinal].domain);
    if (domain == domainOrdinals.end())
      return invalid("CGRA selected use has no arbitration domain");
    CgraResourceUsePlan &use = result.selectedUses[ordinal];
    use.domainOrdinal = domain->second;
    auto position = requesterPositions.find(
        std::make_pair(domain->second, use.requesterOrdinal));
    if (position == requesterPositions.end())
      return invalid("CGRA selected use has no requester position");
    use.requesterPosition = position->second;
  }
  return result;
}

llvm::Expected<CgraResourceRuntimePlan> freezeCgraResourceRuntimePlan(
    llvm::ArrayRef<const ::fabric::ResourceContract *> ownerContracts,
    llvm::ArrayRef<CgraResourcePatternSelection> selectedPatterns) {
  std::vector<CgraResourceActivationSelection> activations;
  activations.reserve(selectedPatterns.size());
  for (std::uint64_t ordinal = 0; ordinal != selectedPatterns.size(); ++ordinal)
    activations.push_back({ordinal, 1});
  return freezeCgraResourceRuntimePlan(ownerContracts, selectedPatterns,
                                       activations);
}

llvm::Expected<CgraResourceRuntime>
CgraResourceRuntime::create(const CgraResourceRuntimePlan &plan) {
  CgraResourceRuntime result(plan);
  result.occupancy_.reserve(plan.dimensions.size());
  for (const CgraResourceDimensionPlan &dimension : plan.dimensions) {
    if (dimension.initialOccupancy > dimension.capacity)
      return invalid("CGRA resource initial occupancy exceeds capacity");
    result.occupancy_.push_back(dimension.initialOccupancy);
  }
  result.domainCursors_.reserve(plan.domains.size());
  for (const CgraResourceDomainPlan &domain : plan.domains) {
    if (domain.requesterCount == 0 ||
        domain.requesterOffset + domain.requesterCount >
            plan.domainRequesters.size() ||
        domain.resetPosition >= domain.requesterCount)
      return invalid("CGRA resource arbitration domain is malformed");
    result.domainCursors_.push_back(domain.resetPosition);
  }
  return result;
}

llvm::Expected<std::vector<CgraResourceGrant>>
CgraResourceRuntime::grant(llvm::ArrayRef<CgraResourceRequest> requests) {
  if (!plan_)
    return invalid("CGRA resource runtime has no static plan");
  std::vector<PendingRequest> pending;
  pending.reserve(requests.size());
  for (const CgraResourceRequest &request : requests) {
    if (request.selectedUseOrdinal >= plan_->selectedUses.size())
      return invalid("CGRA resource request selects an unknown use");
    const CgraResourceUsePlan &use =
        plan_->selectedUses[request.selectedUseOrdinal];
    pending.push_back({request, use.domainOrdinal, use.requesterPosition});
  }
  llvm::sort(pending, pendingLess);
  for (std::size_t ordinal = 1; ordinal != pending.size(); ++ordinal)
    if (pending[ordinal - 1].request.selectedUseOrdinal ==
            pending[ordinal].request.selectedUseOrdinal &&
        pending[ordinal - 1].request.occurrenceOrdinal ==
            pending[ordinal].request.occurrenceOrdinal)
      return invalid("CGRA resource request is duplicated");

  std::size_t reusableEnvelopes = 0;
  for (std::uint32_t slot : freeEnvelopes_)
    reusableEnvelopes +=
        envelopes_[slot].generation < std::numeric_limits<std::uint64_t>::max();
  if (requests.size() > reusableEnvelopes +
                            std::numeric_limits<std::uint32_t>::max() -
                            envelopes_.size())
    return invalid("CGRA claim-envelope inventory exceeds u32");

  const auto feasible = [&](std::uint64_t useOrdinal) {
    const CgraResourceUsePlan &use = plan_->selectedUses[useOrdinal];
    for (std::uint32_t offset = 0; offset != use.claimCount; ++offset) {
      const CgraResourceClaimPlan &claim =
          plan_->claims[use.claimOffset + offset];
      const CgraResourceDimensionPlan &dimension =
          plan_->dimensions[claim.dimensionOrdinal];
      if (claim.amount >
          dimension.capacity - occupancy_[claim.dimensionOrdinal])
        return false;
    }
    return true;
  };

  const auto acquire = [&](const PendingRequest &request) {
    const CgraResourceUsePlan &use =
        plan_->selectedUses[request.request.selectedUseOrdinal];
    for (std::uint32_t offset = 0; offset != use.claimCount; ++offset) {
      const CgraResourceClaimPlan &claim =
          plan_->claims[use.claimOffset + offset];
      occupancy_[claim.dimensionOrdinal] += claim.amount;
    }

    std::uint32_t slot = 0;
    while (!freeEnvelopes_.empty()) {
      slot = freeEnvelopes_.back();
      freeEnvelopes_.pop_back();
      if (envelopes_[slot].generation <
          std::numeric_limits<std::uint64_t>::max()) {
        ++envelopes_[slot].generation;
        envelopes_[slot].selectedUseOrdinal =
            request.request.selectedUseOrdinal;
        envelopes_[slot].active = true;
        return CgraResourceGrant{request.request.selectedUseOrdinal,
                                 request.request.occurrenceOrdinal,
                                 {slot, envelopes_[slot].generation}};
      }
    }
    slot = static_cast<std::uint32_t>(envelopes_.size());
    envelopes_.push_back({1, request.request.selectedUseOrdinal, true});
    return CgraResourceGrant{request.request.selectedUseOrdinal,
                             request.request.occurrenceOrdinal,
                             {slot, 1}};
  };

  std::vector<CgraResourceGrant> grants;
  grants.reserve(requests.size());
  std::size_t first = 0;
  while (first != pending.size()) {
    const std::uint64_t domainOrdinal = pending[first].domain;
    std::size_t last = first + 1;
    while (last != pending.size() && pending[last].domain == domainOrdinal)
      ++last;

    if (domainOrdinal == noCgraResourceDomain) {
      for (std::size_t ordinal = first; ordinal != last; ++ordinal)
        grants.push_back(acquire(pending[ordinal]));
      first = last;
      continue;
    }
    if (domainOrdinal >= plan_->domains.size())
      return invalid("CGRA resource request has an unknown domain");
    const CgraResourceDomainPlan &domain = plan_->domains[domainOrdinal];
    std::vector<std::size_t> begins(domain.requesterCount, last);
    std::vector<std::size_t> ends(domain.requesterCount, last);
    for (std::size_t ordinal = first; ordinal != last; ++ordinal) {
      const std::uint32_t requester = pending[ordinal].requesterPosition;
      if (requester >= domain.requesterCount)
        return invalid("CGRA resource request has an unknown requester");
      if (begins[requester] == last)
        begins[requester] = ordinal;
      ends[requester] = ordinal + 1;
    }
    std::vector<std::size_t> current = begins;

    if (domain.policy == CgraGrantPolicyKind::None &&
        domain.requesterCount > 1) {
      std::vector<std::uint64_t> added(plan_->dimensions.size(), 0);
      std::vector<std::uint64_t> touched;
      for (std::size_t ordinal = first; ordinal != last; ++ordinal) {
        const CgraResourceUsePlan &use =
            plan_->selectedUses[pending[ordinal].request.selectedUseOrdinal];
        for (std::uint32_t offset = 0; offset != use.claimCount; ++offset) {
          const CgraResourceClaimPlan &claim =
              plan_->claims[use.claimOffset + offset];
          if (added[claim.dimensionOrdinal] == 0)
            touched.push_back(claim.dimensionOrdinal);
          if (claim.amount > std::numeric_limits<std::uint64_t>::max() -
                                 added[claim.dimensionOrdinal])
            return invalid("CGRA aggregate resource request overflows u64");
          added[claim.dimensionOrdinal] += claim.amount;
        }
      }
      for (std::uint64_t dimensionOrdinal : touched) {
        const auto &dimension = plan_->dimensions[dimensionOrdinal];
        if (added[dimensionOrdinal] >
            dimension.capacity - occupancy_[dimensionOrdinal])
          return invalid("CGRA reached contention without a GrantPolicy");
      }
      for (std::size_t ordinal = first; ordinal != last; ++ordinal)
        grants.push_back(acquire(pending[ordinal]));
    } else if (domain.policy != CgraGrantPolicyKind::RoundRobin) {
      for (std::uint32_t requester = 0; requester != domain.requesterCount;
           ++requester)
        while (current[requester] != ends[requester]) {
          const PendingRequest &request = pending[current[requester]];
          if (!feasible(request.request.selectedUseOrdinal))
            break;
          grants.push_back(acquire(request));
          ++current[requester];
        }
    } else {
      std::uint32_t &cursor = domainCursors_[domainOrdinal];
      std::vector<bool> blocked(domain.requesterCount, false);
      while (true) {
        bool granted = false;
        for (std::uint32_t scanned = 0; scanned != domain.requesterCount;
             ++scanned) {
          const std::uint32_t requester =
              (cursor + scanned) % domain.requesterCount;
          if (blocked[requester] || current[requester] == ends[requester])
            continue;
          const PendingRequest &request = pending[current[requester]];
          if (!feasible(request.request.selectedUseOrdinal)) {
            blocked[requester] = true;
            continue;
          }
          grants.push_back(acquire(request));
          ++current[requester];
          cursor = (requester + 1) % domain.requesterCount;
          granted = true;
          break;
        }
        if (!granted)
          break;
      }
    }
    first = last;
  }
  return grants;
}

llvm::Error CgraResourceRuntime::release(CgraClaimEnvelope envelope) {
  if (!plan_ || envelope.slot >= envelopes_.size())
    return invalid("CGRA release names an unknown claim envelope");
  EnvelopeSlot &slot = envelopes_[envelope.slot];
  if (!slot.active || slot.generation != envelope.generation)
    return invalid("CGRA release names a stale claim envelope");
  const CgraResourceUsePlan &use = plan_->selectedUses[slot.selectedUseOrdinal];
  for (std::uint32_t offset = 0; offset != use.claimCount; ++offset) {
    const CgraResourceClaimPlan &claim =
        plan_->claims[use.claimOffset + offset];
    if (occupancy_[claim.dimensionOrdinal] < claim.amount)
      return invalid("CGRA claim-envelope release underflows occupancy");
  }
  for (std::uint32_t offset = 0; offset != use.claimCount; ++offset) {
    const CgraResourceClaimPlan &claim =
        plan_->claims[use.claimOffset + offset];
    occupancy_[claim.dimensionOrdinal] -= claim.amount;
  }
  slot.active = false;
  if (slot.generation < std::numeric_limits<std::uint64_t>::max())
    freeEnvelopes_.push_back(envelope.slot);
  return llvm::Error::success();
}

std::uint32_t
CgraResourceRuntime::occupancy(std::uint64_t dimensionOrdinal) const {
  assert(dimensionOrdinal < occupancy_.size() &&
         "unknown CGRA resource dimension");
  return occupancy_[dimensionOrdinal];
}

} // namespace loom::sim::detail

#include "PnR/SpatialRouteCostState.h"

#include "Common/MappingDebugLog.h"
#include "SpatialPhysicalTiming.h"
#include "SpatialRouteCostStateInternal.h"
#include "SpatialSwitchRowPacking.h"

#include "Fabric/Identity/FabricTemporalSwitchRoute.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>

using namespace loom;
using namespace loom::pnr;

namespace {

using ::loom::pnr::detail::routeCostStateError;
using ::loom::pnr::detail::saturatedAdd;

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

template <typename T>
std::size_t retainedNestedBytes(const std::vector<std::vector<T>> &values) {
  std::size_t bytes = values.capacity() * sizeof(std::vector<T>);
  for (const auto &value : values)
    bytes += retainedBytes(value);
  return bytes;
}

std::size_t retainedBytes(const SpatialTagAssignmentDelta &delta) {
  return retainedBytes(delta.domains) +
         retainedBytes(delta.domainResidentCounts) +
         retainedBytes(delta.domainConflictCounts) +
         retainedBytes(delta.logicalNets) +
         retainedBytes(delta.netDomainUseOffsets) +
         retainedBytes(delta.netDomainUseDomains) +
         retainedBytes(delta.netDomainMarginalResidentCounts) +
         retainedBytes(delta.netUnassignedCounts) +
         retainedBytes(delta.netTagValueOffsets) +
         retainedBytes(delta.netTagValues);
}

std::uint64_t encodingCapacity(std::uint32_t tagWidthBits) {
  if (tagWidthBits >= 64)
    return std::numeric_limits<std::uint64_t>::max();
  return std::uint64_t{1} << tagWidthBits;
}

} // namespace

SpatialRouteCostState::SpatialRouteCostState(
    SpatialRouteCostState &&) noexcept = default;

SpatialRouteCostState::~SpatialRouteCostState() = default;

llvm::Expected<SpatialRouteCostState>
SpatialRouteCostState::create(const SpatialCandidateState &candidate) {
  const auto *policy = std::get_if<ResolvedPathFinderPolicy>(
      &candidate.problem().config().policy().search.routing.negotiation);
  if (!policy)
    return routeCostStateError(
        "frozen routing policy does not select PathFinder");
  if (llvm::Error error = validateResolvedPathFinderPolicy(*policy))
    return std::move(error);
  if (llvm::Error error = candidate.verify())
    return std::move(error);

  const FrozenSpatialPnrProblem &problem = candidate.problem();
  if (problem.transfers().logicalNets().size() >
          std::numeric_limits<PnrIndex>::max() ||
      problem.routing().routeClaims().size() >
          std::numeric_limits<PnrIndex>::max())
    return routeCostStateError("frozen dimensions exceed PnrIndex");

  SpatialRouteCostState state;
  state.candidate_ = &candidate;
  state.problem_ = &problem;
  state.switchRows_ =
      std::make_unique<detail::SpatialRouteCostSwitchRowState>();
  state.policy_ = *policy;
  state.logicalNetCount_ =
      static_cast<PnrIndex>(problem.transfers().logicalNets().size());
  state.routeClaimCount_ =
      static_cast<PnrIndex>(problem.routing().routeClaims().size());
  state.routeClaimWordCount_ =
      (static_cast<std::size_t>(state.routeClaimCount_) + 63) / 64;
  state.presentPressure_ = policy->presentPressureInitial;

  for (PnrIndex logicalNet = 0; logicalNet < state.logicalNetCount_;
       ++logicalNet) {
    const auto bits = candidate.logicalNetRouteClaimBits(logicalNet);
    if (bits.size() != state.routeClaimWordCount_)
      return routeCostStateError(
          "candidate logical-net claim bitset has the wrong width");
  }

  const std::size_t capacityCount =
      problem.resources().capacityDimensions().size();
  state.workingCapacityUsageRaw_.reserve(capacityCount);
  state.capacityOveruseCosts_.reserve(capacityCount);
  for (PnrIndex capacity = 0; capacity < capacityCount; ++capacity) {
    const std::uint64_t usage = candidate.routeCapacityUsageRaw(capacity);
    state.workingCapacityUsageRaw_.push_back(usage);
    auto overuse = normalizedRouteOveruseCost(
        usage, 0, problem.resources().capacityDimensions()[capacity].capacity);
    if (!overuse)
      return overuse.takeError();
    state.capacityOveruseCosts_.push_back(*overuse);
  }
  state.historyPressure_.assign(capacityCount, 0);
  state.stagedHistoryPressure_.assign(capacityCount, 0);
  state.capacityUpdateEpochs_.assign(capacityCount, 0);
  state.stagedCapacityUsageRaw_.assign(capacityCount, 0);
  state.stagedCapacityOveruseCosts_.assign(capacityCount, 0);
  state.affectedCapacities_.reserve(capacityCount);

  state.currentClaimOveruseCosts_.assign(state.routeClaimCount_, 0);
  state.selectedLogicalNetClaimBits_.assign(state.routeClaimWordCount_, 0);
  state.claimUpdateEpochs_.assign(state.routeClaimCount_, 0);
  state.stagedClaimOveruseCosts_.assign(state.routeClaimCount_, 0);
  state.affectedClaims_.reserve(state.routeClaimCount_);
  for (PnrIndex claim = 0; claim < state.routeClaimCount_; ++claim) {
    const FrozenSpatialRouteClaim &record =
        problem.routing().routeClaims()[claim];
    if (record.capacityDimension >= capacityCount)
      return routeCostStateError("route claim capacity is out of range");
    const std::uint64_t capacity =
        problem.resources()
            .capacityDimensions()[record.capacityDimension]
            .capacity;
    auto overuse = normalizedRouteOveruseCost(
        state.workingCapacityUsageRaw_[record.capacityDimension], record.amount,
        capacity);
    if (!overuse)
      return overuse.takeError();
    state.currentClaimOveruseCosts_[claim] = *overuse;
  }

  const auto matchDomains = problem.routing().tagContinuity().matchDomains();
  const std::size_t tagDomainCount = matchDomains.size();
  state.switchRows_->enabled =
      llvm::any_of(matchDomains, [](const auto &domain) {
        return domain.kind == ::loom::fabric::FabricPhysicalTagMatchDomainKind::
                                  TemporalSwitchTable;
      });
  state.logicalNetTagUses_.resize(state.logicalNetCount_);
  state.logicalNetTagUnassignedCounts_.assign(state.logicalNetCount_, 0);
  state.logicalNetTagValues_.resize(state.logicalNetCount_);
  state.workingTagDomainUsage_.assign(tagDomainCount, 0);
  state.tagDomainConflictCounts_.assign(tagDomainCount, 0);
  state.tagResidentHistoryPressure_.assign(tagDomainCount, 0);
  state.tagEncodingHistoryPressure_.assign(tagDomainCount, 0);
  state.stagedTagResidentHistoryPressure_.assign(tagDomainCount, 0);
  state.stagedTagEncodingHistoryPressure_.assign(tagDomainCount, 0);
  state.tagResidentOveruseCosts_.assign(tagDomainCount, 0);
  state.tagEncodingPressureCosts_.assign(tagDomainCount, 0);
  state.tagDomainUpdateEpochs_.assign(tagDomainCount, 0);
  state.stagedTagDomainUsage_.assign(tagDomainCount, 0);
  state.stagedTagResidentOveruseCosts_.assign(tagDomainCount, 0);
  state.stagedTagEncodingPressureCosts_.assign(tagDomainCount, 0);
  state.affectedTagDomains_.reserve(tagDomainCount);

  std::vector<std::uint64_t> domainArcCounts(tagDomainCount + 1, 0);
  const auto endpointDomains =
      problem.routing().tagContinuity().endpointMatchDomainOrdinals();
  for (auto [arcOrdinal, arc] :
       llvm::enumerate(problem.routing().routingArcs())) {
    if (!problem.activeRouting().arcIsActive(static_cast<PnrIndex>(arcOrdinal)))
      continue;
    if (arc.target >= endpointDomains.size())
      return routeCostStateError("routing arc target is out of range");
    const PnrIndex domain = endpointDomains[arc.target];
    if (domain != getInvalidPnrIndex()) {
      if (domain >= tagDomainCount)
        return routeCostStateError("routing arc tag domain is out of range");
      ++domainArcCounts[domain + 1];
    }
  }
  for (std::size_t domain = 1; domain < domainArcCounts.size(); ++domain)
    domainArcCounts[domain] += domainArcCounts[domain - 1];
  state.tagDomainArcOffsets_.reserve(domainArcCounts.size());
  for (std::uint64_t offset : domainArcCounts) {
    if (offset > std::numeric_limits<PnrIndex>::max())
      return routeCostStateError("tag-domain arc incidence exceeds PnrIndex");
    state.tagDomainArcOffsets_.push_back(static_cast<PnrIndex>(offset));
  }
  state.tagDomainArcs_.resize(domainArcCounts.back());
  std::vector<std::uint64_t> domainArcCursors = domainArcCounts;
  for (auto [arcOrdinal, arc] :
       llvm::enumerate(problem.routing().routingArcs())) {
    if (!problem.activeRouting().arcIsActive(static_cast<PnrIndex>(arcOrdinal)))
      continue;
    const PnrIndex domain = endpointDomains[arc.target];
    if (domain == getInvalidPnrIndex())
      continue;
    state.tagDomainArcs_[domainArcCursors[domain]++] =
        static_cast<PnrIndex>(arcOrdinal);
  }
  state.tagDomainArcs_.resize(domainArcCounts.back());
  if (llvm::Error error = state.rebuildTagProjectionFromCandidate(true))
    return std::move(error);

  const std::size_t traversalCount = problem.routing().traversals().size();
  state.lowerBoundTraversalCosts_.assign(traversalCount, 0);
  state.currentTraversalCosts_.assign(traversalCount, 0);
  state.traversalUpdateEpochs_.assign(traversalCount, 0);
  state.stagedTraversalCosts_.assign(traversalCount, 0);
  state.affectedTraversals_.reserve(traversalCount);
  for (PnrIndex traversal = 0; traversal < traversalCount; ++traversal) {
    if (!problem.activeRouting().traversalIsActive(traversal)) {
      state.lowerBoundTraversalCosts_[traversal] = 0;
      state.currentTraversalCosts_[traversal] = 0;
      continue;
    }
    auto lower = state.computeTraversalCost(traversal, false, false);
    if (!lower)
      return lower.takeError();
    state.lowerBoundTraversalCosts_[traversal] = *lower;
    auto current = state.computeTraversalCost(traversal, true, false);
    if (!current)
      return current.takeError();
    state.currentTraversalCosts_[traversal] = *current;
  }

  const std::size_t arcCount = problem.routing().routingArcs().size();
  state.lowerBoundArcCosts_.reserve(arcCount);
  state.currentArcCosts_.reserve(arcCount);
  state.stagedArcCosts_.assign(arcCount, 0);
  state.arcUpdateEpochs_.assign(arcCount, 0);
  state.affectedTagArcs_.reserve(arcCount);
  for (PnrIndex arcOrdinal = 0; arcOrdinal < arcCount; ++arcOrdinal) {
    const EndpointRoutingArc &arc = problem.routing().routingArcs()[arcOrdinal];
    if (arc.traversal >= traversalCount)
      return routeCostStateError("routing arc traversal is out of range");
    if (!problem.activeRouting().arcIsActive(arcOrdinal)) {
      state.lowerBoundArcCosts_.push_back(0);
      state.currentArcCosts_.push_back(0);
      continue;
    }
    auto lower = state.computeArcCost(arcOrdinal, false, false, false);
    if (!lower)
      return lower.takeError();
    state.lowerBoundArcCosts_.push_back(*lower);
    auto current = state.computeArcCost(arcOrdinal, true, false, false);
    if (!current)
      return current.takeError();
    state.currentArcCosts_.push_back(*current);
  }
  return state;
}

std::uint64_t SpatialRouteCostState::workingCapacityUsageRaw(
    PnrIndex capacityDimension) const {
  assert(capacityDimension < workingCapacityUsageRaw_.size());
  return workingCapacityUsageRaw_[capacityDimension];
}

std::uint64_t
SpatialRouteCostState::historyPressure(PnrIndex capacityDimension) const {
  assert(capacityDimension < historyPressure_.size());
  return historyPressure_[capacityDimension];
}

RouteCost
SpatialRouteCostState::capacityOveruseCost(PnrIndex capacityDimension) const {
  assert(capacityDimension < capacityOveruseCosts_.size());
  return capacityOveruseCosts_[capacityDimension];
}

bool SpatialRouteCostState::hasCapacityOveruse() const {
  return llvm::any_of(capacityOveruseCosts_,
                      [](RouteCost cost) { return cost != 0; });
}

bool SpatialRouteCostState::hasTagPressureViolation() const {
  if (tagUnassignedCount_ != 0)
    return true;
  const auto domains = problem_->routing().tagContinuity().matchDomains();
  for (PnrIndex domain = 0; domain < domains.size(); ++domain)
    if (tagResidentOveruseCosts_[domain] != 0 ||
        tagDomainConflictCounts_[domain] != 0 ||
        workingTagDomainUsage_[domain] >
            encodingCapacity(domains[domain].tagWidthBits))
      return true;
  return false;
}

std::uint64_t
SpatialRouteCostState::workingTagDomainUsage(PnrIndex domain) const {
  assert(domain < workingTagDomainUsage_.size());
  return workingTagDomainUsage_[domain];
}

std::uint64_t
SpatialRouteCostState::tagDomainEncodingCapacity(PnrIndex domain) const {
  const auto domains = problem_->routing().tagContinuity().matchDomains();
  assert(domain < domains.size());
  return encodingCapacity(domains[domain].tagWidthBits);
}

std::optional<std::uint64_t>
SpatialRouteCostState::tagDomainResidentCapacity(PnrIndex domain) const {
  const auto domains = problem_->routing().tagContinuity().matchDomains();
  assert(domain < domains.size());
  return domains[domain].residentEntryCapacity;
}

std::uint64_t
SpatialRouteCostState::tagDomainResidentOveruse(PnrIndex domain) const {
  assert(domain < workingTagDomainUsage_.size());
  const auto capacity = tagDomainResidentCapacity(domain);
  return capacity && workingTagDomainUsage_[domain] > *capacity
             ? workingTagDomainUsage_[domain] - *capacity
             : 0;
}

std::uint64_t
SpatialRouteCostState::tagDomainConflictCount(PnrIndex domain) const {
  assert(domain < tagDomainConflictCounts_.size());
  return tagDomainConflictCounts_[domain];
}

llvm::ArrayRef<SpatialTagDomainUse>
SpatialRouteCostState::logicalNetTagDomainUses(PnrIndex logicalNet) const {
  assert(logicalNet < logicalNetTagUses_.size());
  return logicalNetTagUses_[logicalNet];
}

std::uint64_t
SpatialRouteCostState::logicalNetTagUnassignedCount(PnrIndex logicalNet) const {
  assert(logicalNet < logicalNetTagUnassignedCounts_.size());
  return logicalNetTagUnassignedCounts_[logicalNet];
}

bool SpatialRouteCostState::logicalNetHasTagPressure(
    PnrIndex logicalNet) const {
  if (logicalNetTagUnassignedCount(logicalNet) != 0)
    return true;
  for (const SpatialTagDomainUse &use : logicalNetTagDomainUses(logicalNet)) {
    if (tagDomainResidentOveruse(use.domain) != 0 ||
        tagDomainConflictCounts_[use.domain] != 0 ||
        workingTagDomainUsage_[use.domain] >
            tagDomainEncodingCapacity(use.domain))
      return true;
  }
  return false;
}

RouteCost
SpatialRouteCostState::logicalNetTagPressure(PnrIndex logicalNet) const {
  auto unassigned = normalizedRouteClaimCost(
      logicalNetTagUnassignedCount(logicalNet),
      std::max<std::uint64_t>(1, candidate_->tagSegments(logicalNet).size()));
  if (!unassigned)
    return routeCostInfinity;
  RouteCost pressure = *unassigned;
  for (const SpatialTagDomainUse &use : logicalNetTagDomainUses(logicalNet)) {
    const PnrIndex domain = use.domain;
    if (tagDomainResidentOveruse(domain) == 0 &&
        tagDomainConflictCounts_[domain] == 0 &&
        workingTagDomainUsage_[domain] <= tagDomainEncodingCapacity(domain))
      continue;
    const RouteCost domainPressure = std::max(
        tagResidentOveruseCosts_[domain], tagEncodingPressureCosts_[domain]);
    auto term = normalizedRouteClaimCost(
        use.marginalResidentCount,
        std::max<std::uint64_t>(1, tagDomainEncodingCapacity(domain)));
    if (!term)
      return routeCostInfinity;
    auto weighted = scaledRouteProduct(*term, domainPressure);
    if (!weighted)
      return routeCostInfinity;
    auto accumulated = accumulateRouteCost(pressure, *weighted);
    if (!accumulated)
      return routeCostInfinity;
    pressure = *accumulated;
  }
  return pressure;
}

bool SpatialRouteCostState::arcHasTagPressure(PnrIndex arc) const {
  const auto arcs = problem_->routing().routingArcs();
  const auto endpointDomains =
      problem_->routing().tagContinuity().endpointMatchDomainOrdinals();
  assert(arc < arcs.size() && arcs[arc].target < endpointDomains.size());
  const PnrIndex domain = endpointDomains[arcs[arc].target];
  return domain != getInvalidPnrIndex() &&
         (tagDomainResidentOveruse(domain) != 0 ||
          tagDomainConflictCounts_[domain] != 0 ||
          workingTagDomainUsage_[domain] > tagDomainEncodingCapacity(domain));
}

llvm::ArrayRef<std::uint64_t>
SpatialRouteCostState::logicalNetClaimBits(PnrIndex logicalNet) const {
  assert(logicalNet < logicalNetCount_);
  const auto bits = candidate_->logicalNetRouteClaimBits(logicalNet);
  assert(bits.size() == routeClaimWordCount_);
  return bits;
}

void SpatialRouteCostState::beginUpdate() {
  ++updateEpoch_;
  if (updateEpoch_ == 0) {
    std::fill(capacityUpdateEpochs_.begin(), capacityUpdateEpochs_.end(), 0);
    std::fill(claimUpdateEpochs_.begin(), claimUpdateEpochs_.end(), 0);
    std::fill(traversalUpdateEpochs_.begin(), traversalUpdateEpochs_.end(), 0);
    std::fill(tagDomainUpdateEpochs_.begin(), tagDomainUpdateEpochs_.end(), 0);
    std::fill(arcUpdateEpochs_.begin(), arcUpdateEpochs_.end(), 0);
    updateEpoch_ = 1;
  }
  affectedCapacities_.clear();
  affectedClaims_.clear();
  affectedTraversals_.clear();
  affectedTagDomains_.clear();
  affectedTagArcs_.clear();
}

llvm::Error SpatialRouteCostState::stageClaim(PnrIndex claim, bool restore) {
  if (claim >= problem_->routing().routeClaims().size())
    return routeCostStateError("logical-net claim is out of range");
  const FrozenSpatialRouteClaim &record =
      problem_->routing().routeClaims()[claim];
  const PnrIndex capacity = record.capacityDimension;
  if (capacity >= workingCapacityUsageRaw_.size())
    return routeCostStateError("route claim capacity is out of range");
  if (capacityUpdateEpochs_[capacity] != updateEpoch_) {
    capacityUpdateEpochs_[capacity] = updateEpoch_;
    stagedCapacityUsageRaw_[capacity] = workingCapacityUsageRaw_[capacity];
    affectedCapacities_.push_back(capacity);
  }
  std::uint64_t &usage = stagedCapacityUsageRaw_[capacity];
  if (restore) {
    if (record.amount > std::numeric_limits<std::uint64_t>::max() - usage)
      return routeCostStateError("restored route occupancy overflows u64");
    usage += record.amount;
  } else {
    if (record.amount > usage) {
      loom::mapping_debug::emit(
          loom::mapping_debug::Level::Decision,
          loom::mapping_debug::Stage::SpatialPnr,
          loom::mapping_debug::Event::ArithmeticFailure,
          [&](llvm::json::Object &fields) {
            fields["operation"] = "route_claim_exclusion";
            fields["claim"] = claim;
            fields["capacity_ref"] = capacity;
            fields["claim_amount"] = record.amount;
            fields["working_usage"] = workingCapacityUsageRaw_[capacity];
            fields["staged_usage"] = usage;
            fields["candidate_usage"] =
                candidate_->routeCapacityUsageRaw(capacity);
            fields["capacity"] =
                problem_->resources().capacityDimensions()[capacity].capacity;
            if (selectedLogicalNet_)
              fields["selected_logical_net"] = *selectedLogicalNet_;
            llvm::json::Array logicalNets;
            for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_;
                 ++logicalNet)
              if (candidate_->logicalNetRouteClaimRefcount(logicalNet, claim) !=
                  0)
                logicalNets.push_back(logicalNet);
            fields["candidate_logical_nets"] = std::move(logicalNets);
          });
      return routeCostStateError(
          "claim " + llvm::Twine(claim) + " excludes amount " +
          llvm::Twine(record.amount) + " from capacity " +
          llvm::Twine(capacity) + " with staged usage " + llvm::Twine(usage));
    }
    usage -= record.amount;
  }
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::stageLogicalNet(PnrIndex logicalNet,
                                                   bool restore) {
  if (logicalNet >= logicalNetCount_)
    return routeCostStateError("selected logical net is out of range");
  return stageClaimBits(logicalNetClaimBits(logicalNet), restore);
}

llvm::Error
SpatialRouteCostState::stageClaimBits(llvm::ArrayRef<std::uint64_t> claimBits,
                                      bool restore) {
  if (claimBits.empty())
    return llvm::Error::success();
  if (claimBits.size() != routeClaimWordCount_)
    return routeCostStateError("selected claim bitset has the wrong width");
  for (auto [wordOrdinal, initialWord] : llvm::enumerate(claimBits)) {
    std::uint64_t word = initialWord;
    while (word != 0) {
      const unsigned bit = llvm::countr_zero(word);
      const std::size_t claim = wordOrdinal * 64 + bit;
      if (claim >= routeClaimCount_)
        return routeCostStateError("logical-net claim padding bit is set");
      if (llvm::Error error = stageClaim(static_cast<PnrIndex>(claim), restore))
        return error;
      word &= word - 1;
    }
  }
  return llvm::Error::success();
}

llvm::Error
SpatialRouteCostState::stageTagUses(llvm::ArrayRef<SpatialTagDomainUse> uses,
                                    bool restore) {
  for (const SpatialTagDomainUse &use : uses) {
    if (use.domain >= workingTagDomainUsage_.size() ||
        use.marginalResidentCount == 0)
      return routeCostStateError("logical-net tag use is out of range");
    if (tagDomainUpdateEpochs_[use.domain] != updateEpoch_) {
      tagDomainUpdateEpochs_[use.domain] = updateEpoch_;
      stagedTagDomainUsage_[use.domain] = workingTagDomainUsage_[use.domain];
      affectedTagDomains_.push_back(use.domain);
    }
    std::uint64_t &usage = stagedTagDomainUsage_[use.domain];
    if (restore) {
      if (use.marginalResidentCount >
          std::numeric_limits<std::uint64_t>::max() - usage)
        return routeCostStateError("restored tag-domain usage overflows u64");
      usage += use.marginalResidentCount;
    } else {
      if (use.marginalResidentCount > usage)
        return routeCostStateError("excluded tag-domain usage underflows u64");
      usage -= use.marginalResidentCount;
    }
  }
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::finishUpdate() {
  const FrozenSpatialRoutingGraph &routing = problem_->routing();
  for (PnrIndex capacity : affectedCapacities_) {
    const std::uint64_t capacityValue =
        problem_->resources().capacityDimensions()[capacity].capacity;
    auto aggregateOveruse = normalizedRouteOveruseCost(
        capacityUsageForCost(capacity, true), 0, capacityValue);
    if (!aggregateOveruse)
      return aggregateOveruse.takeError();
    stagedCapacityOveruseCosts_[capacity] = *aggregateOveruse;

    const auto claims = routing.capacityRouteClaims().slice(
        routing.capacityRouteClaimOffsets()[capacity],
        routing.capacityRouteClaimOffsets()[capacity + 1] -
            routing.capacityRouteClaimOffsets()[capacity]);
    for (PnrIndex claim : claims) {
      if (claimUpdateEpochs_[claim] == updateEpoch_)
        continue;
      claimUpdateEpochs_[claim] = updateEpoch_;
      affectedClaims_.push_back(claim);

      const FrozenSpatialRouteClaim &record = routing.routeClaims()[claim];
      const std::uint64_t capacityValue =
          problem_->resources()
              .capacityDimensions()[record.capacityDimension]
              .capacity;
      auto overuse = normalizedRouteOveruseCost(
          capacityUsageForCost(record.capacityDimension, true), record.amount,
          capacityValue);
      if (!overuse)
        return overuse.takeError();
      stagedClaimOveruseCosts_[claim] = *overuse;

      const auto traversals = routing.routeClaimTraversals().slice(
          routing.routeClaimTraversalOffsets()[claim],
          routing.routeClaimTraversalOffsets()[claim + 1] -
              routing.routeClaimTraversalOffsets()[claim]);
      for (PnrIndex traversal : traversals) {
        if (!problem_->activeRouting().traversalIsActive(traversal))
          continue;
        if (traversalUpdateEpochs_[traversal] == updateEpoch_)
          continue;
        traversalUpdateEpochs_[traversal] = updateEpoch_;
        affectedTraversals_.push_back(traversal);
      }
    }
  }
  for (PnrIndex traversal : affectedTraversals_) {
    auto cost = computeTraversalCost(traversal, true, true);
    if (!cost)
      return cost.takeError();
    stagedTraversalCosts_[traversal] = *cost;
  }

  const auto matchDomains = problem_->routing().tagContinuity().matchDomains();
  for (PnrIndex domain : affectedTagDomains_) {
    const std::uint64_t usage = tagUsageForCost(domain, true);
    if (matchDomains[domain].residentEntryCapacity) {
      auto overuse = normalizedRouteOveruseCost(
          usage, 0, *matchDomains[domain].residentEntryCapacity);
      if (!overuse)
        return overuse.takeError();
      stagedTagResidentOveruseCosts_[domain] = *overuse;
    } else {
      stagedTagResidentOveruseCosts_[domain] = 0;
    }
    auto encodingPressure = normalizedRouteClaimCost(
        saturatedAdd(usage, 1),
        encodingCapacity(matchDomains[domain].tagWidthBits));
    if (!encodingPressure)
      return encodingPressure.takeError();
    stagedTagEncodingPressureCosts_[domain] = *encodingPressure;

    const PnrIndex begin = tagDomainArcOffsets_[domain];
    const PnrIndex end = tagDomainArcOffsets_[domain + 1];
    for (PnrIndex incidence = begin; incidence < end; ++incidence) {
      const PnrIndex arc = tagDomainArcs_[incidence];
      if (arcUpdateEpochs_[arc] == updateEpoch_)
        continue;
      arcUpdateEpochs_[arc] = updateEpoch_;
      affectedTagArcs_.push_back(arc);
    }
  }
  for (PnrIndex traversal : affectedTraversals_)
    for (PnrIndex arc : problem_->routing().traversalArcs().slice(
             problem_->routing().traversalArcOffsets()[traversal],
             problem_->routing().traversalArcOffsets()[traversal + 1] -
                 problem_->routing().traversalArcOffsets()[traversal])) {
      if (!problem_->activeRouting().arcIsActive(arc))
        continue;
      if (arcUpdateEpochs_[arc] == updateEpoch_)
        continue;
      arcUpdateEpochs_[arc] = updateEpoch_;
      affectedTagArcs_.push_back(arc);
    }
  for (PnrIndex arc : affectedTagArcs_) {
    auto cost = computeArcCost(arc, true, true, true);
    if (!cost)
      return cost.takeError();
    stagedArcCosts_[arc] = *cost;
  }

  for (PnrIndex capacity : affectedCapacities_)
    workingCapacityUsageRaw_[capacity] = stagedCapacityUsageRaw_[capacity];
  for (PnrIndex capacity : affectedCapacities_)
    capacityOveruseCosts_[capacity] = stagedCapacityOveruseCosts_[capacity];
  for (PnrIndex claim : affectedClaims_)
    currentClaimOveruseCosts_[claim] = stagedClaimOveruseCosts_[claim];
  for (PnrIndex traversal : affectedTraversals_)
    currentTraversalCosts_[traversal] = stagedTraversalCosts_[traversal];
  for (PnrIndex domain : affectedTagDomains_) {
    workingTagDomainUsage_[domain] = stagedTagDomainUsage_[domain];
    tagResidentOveruseCosts_[domain] = stagedTagResidentOveruseCosts_[domain];
    tagEncodingPressureCosts_[domain] = stagedTagEncodingPressureCosts_[domain];
  }
  for (PnrIndex arc : affectedTagArcs_)
    currentArcCosts_[arc] = stagedArcCosts_[arc];
  return llvm::Error::success();
}

std::uint64_t
SpatialRouteCostState::capacityUsageForCost(PnrIndex capacityDimension,
                                            bool stagedUsage) const {
  if (stagedUsage && capacityUpdateEpochs_[capacityDimension] == updateEpoch_)
    return stagedCapacityUsageRaw_[capacityDimension];
  return workingCapacityUsageRaw_[capacityDimension];
}

RouteCost SpatialRouteCostState::claimOveruseForCost(PnrIndex claim,
                                                     bool stagedClaims) const {
  if (stagedClaims && claimUpdateEpochs_[claim] == updateEpoch_)
    return stagedClaimOveruseCosts_[claim];
  return currentClaimOveruseCosts_[claim];
}

std::uint64_t SpatialRouteCostState::tagUsageForCost(PnrIndex domain,
                                                     bool stagedTags) const {
  if (stagedTags && tagDomainUpdateEpochs_[domain] == updateEpoch_)
    return stagedTagDomainUsage_[domain];
  return workingTagDomainUsage_[domain];
}

std::uint64_t
SpatialRouteCostState::encodingPressureRaw(PnrIndex domain,
                                           bool stagedTags) const {
  return tagUsageForCost(domain, stagedTags);
}

llvm::Expected<RouteCost> SpatialRouteCostState::computeTagDomainCost(
    PnrIndex domain, bool resident, bool dynamicCost, bool stagedTags,
    std::uint64_t presentPressure, std::uint64_t historyPressure) const {
  const auto domains = problem_->routing().tagContinuity().matchDomains();
  if (domain >= domains.size())
    return routeCostStateError("tag-domain cost index is out of range");
  const std::uint64_t capacity =
      resident && domains[domain].residentEntryCapacity
          ? *domains[domain].residentEntryCapacity
          : encodingCapacity(domains[domain].tagWidthBits);
  if (capacity == 0)
    return routeCostStateError("tag-domain capacity is zero");
  auto qCost = normalizedRouteClaimCost(1, capacity);
  if (!qCost)
    return qCost.takeError();
  if (!dynamicCost)
    return *qCost;

  const std::uint64_t usage = resident
                                  ? tagUsageForCost(domain, stagedTags)
                                  : encodingPressureRaw(domain, stagedTags);
  llvm::Expected<RouteCost> pressure =
      resident ? normalizedRouteOveruseCost(usage, 1, capacity)
               : normalizedRouteClaimCost(saturatedAdd(usage, 1), capacity);
  if (!pressure)
    return pressure.takeError();
  return pathFinderResourceCost(policy_.priceKernel, *qCost, *pressure,
                                presentPressure, historyPressure);
}

llvm::Expected<RouteCost> SpatialRouteCostState::computeArcCost(
    PnrIndex arc, bool dynamicCost, bool stagedClaims, bool stagedTags) const {
  const std::uint64_t present = presentPressure_;
  const auto routeHistory = llvm::ArrayRef<std::uint64_t>(historyPressure_);
  const auto residentHistory =
      llvm::ArrayRef<std::uint64_t>(tagResidentHistoryPressure_);
  const auto encodingHistory =
      llvm::ArrayRef<std::uint64_t>(tagEncodingHistoryPressure_);
  if (dynamicCost && !stagedClaims && !stagedTags)
    return computeArcCost(arc, present, routeHistory, residentHistory,
                          encodingHistory);
  if (arc >= problem_->routing().routingArcs().size())
    return routeCostStateError("routing arc cost index is out of range");
  const EndpointRoutingArc &record = problem_->routing().routingArcs()[arc];
  auto traversal =
      computeTraversalCost(record.traversal, dynamicCost, stagedClaims);
  if (!traversal)
    return traversal.takeError();
  RouteCost cost = *traversal;
  const auto endpointDomains =
      problem_->routing().tagContinuity().endpointMatchDomainOrdinals();
  const PnrIndex domain = endpointDomains[record.target];
  if (domain == getInvalidPnrIndex())
    return cost;
  const auto matchDomains = problem_->routing().tagContinuity().matchDomains();
  if (matchDomains[domain].residentEntryCapacity) {
    auto term = computeTagDomainCost(domain, true, dynamicCost, stagedTags,
                                     presentPressure_,
                                     tagResidentHistoryPressure_[domain]);
    if (!term)
      return term.takeError();
    auto accumulated = accumulateRouteCost(cost, *term);
    if (!accumulated)
      return accumulated.takeError();
    cost = *accumulated;
  }
  auto term = computeTagDomainCost(domain, false, dynamicCost, stagedTags,
                                   presentPressure_,
                                   tagEncodingHistoryPressure_[domain]);
  if (!term)
    return term.takeError();
  return accumulateRouteCost(cost, *term);
}

llvm::Expected<RouteCost> SpatialRouteCostState::computeArcCost(
    PnrIndex arc, std::uint64_t presentPressure,
    llvm::ArrayRef<std::uint64_t> routeHistoryPressure,
    llvm::ArrayRef<std::uint64_t> residentHistoryPressure,
    llvm::ArrayRef<std::uint64_t> encodingHistoryPressure) const {
  if (arc >= problem_->routing().routingArcs().size())
    return routeCostStateError("routing arc cost index is out of range");
  const auto domains = problem_->routing().tagContinuity().matchDomains();
  if (residentHistoryPressure.size() != domains.size() ||
      encodingHistoryPressure.size() != domains.size())
    return routeCostStateError("tag history-pressure vector has wrong width");
  const EndpointRoutingArc &record = problem_->routing().routingArcs()[arc];
  auto traversal = computeTraversalCost(record.traversal, presentPressure,
                                        routeHistoryPressure);
  if (!traversal)
    return traversal.takeError();
  RouteCost cost = *traversal;
  const auto endpointDomains =
      problem_->routing().tagContinuity().endpointMatchDomainOrdinals();
  const PnrIndex domain = endpointDomains[record.target];
  if (domain == getInvalidPnrIndex())
    return cost;
  if (domains[domain].residentEntryCapacity) {
    auto term = computeTagDomainCost(domain, true, true, false, presentPressure,
                                     residentHistoryPressure[domain]);
    if (!term)
      return term.takeError();
    auto accumulated = accumulateRouteCost(cost, *term);
    if (!accumulated)
      return accumulated.takeError();
    cost = *accumulated;
  }
  auto term = computeTagDomainCost(domain, false, true, false, presentPressure,
                                   encodingHistoryPressure[domain]);
  if (!term)
    return term.takeError();
  return accumulateRouteCost(cost, *term);
}

llvm::Expected<RouteCost> SpatialRouteCostState::computeTraversalCost(
    PnrIndex traversal, bool dynamicCost, bool stagedClaims) const {
  return computeTraversalCostImpl(traversal, dynamicCost, stagedClaims,
                                  presentPressure_, historyPressure_);
}

llvm::Expected<RouteCost> SpatialRouteCostState::computeTraversalCost(
    PnrIndex traversal, std::uint64_t presentPressure,
    llvm::ArrayRef<std::uint64_t> historyPressure) const {
  return computeTraversalCostImpl(traversal, true, false, presentPressure,
                                  historyPressure);
}

llvm::Expected<RouteCost> SpatialRouteCostState::computeTraversalCostImpl(
    PnrIndex traversal, bool dynamicCost, bool stagedClaims,
    std::uint64_t presentPressure,
    llvm::ArrayRef<std::uint64_t> historyPressure) const {
  if (traversal >= problem_->routing().traversals().size())
    return routeCostStateError("traversal cost index is out of range");
  if (historyPressure.size() != historyPressure_.size())
    return routeCostStateError("history-pressure vector has the wrong width");
  RouteCost cost = 0;
  const FrozenSpatialTraversal &record =
      problem_->routing().traversals()[traversal];
  for (PnrIndex claim : problem_->routing().traversalClaimKeys().slice(
           record.routeClaimOffset, record.routeClaimCount)) {
    const FrozenSpatialRouteClaim &claimRecord =
        problem_->routing().routeClaims()[claim];
    if (claimRecord.qCost == 0)
      continue;
    RouteCost term = claimRecord.qCost;
    if (dynamicCost) {
      auto current = pathFinderResourceCost(
          policy_.priceKernel, claimRecord.qCost,
          claimOveruseForCost(claim, stagedClaims), presentPressure,
          historyPressure[claimRecord.capacityDimension]);
      if (!current)
        return current.takeError();
      term = *current;
    }
    auto accumulated = accumulateRouteCost(cost, term);
    if (!accumulated)
      return accumulated.takeError();
    cost = *accumulated;
  }
  return cost;
}

llvm::Error
SpatialRouteCostState::selectLogicalNet(std::optional<PnrIndex> logicalNet) {
  if (logicalNet && *logicalNet >= logicalNetCount_)
    return routeCostStateError("selected logical net is out of range");
  if (logicalNet == selectedLogicalNet_)
    return llvm::Error::success();

  beginUpdate();
  if (selectedLogicalNet_) {
    if (llvm::Error error = stageClaimBits(selectedLogicalNetClaimBits_, false))
      return error;
    if (llvm::Error error = stageTagUses(selectedLogicalNetTagUses_, false))
      return error;
    if (llvm::Error error = stageLogicalNet(*selectedLogicalNet_, true))
      return error;
    if (llvm::Error error =
            stageTagUses(logicalNetTagUses_[*selectedLogicalNet_], true))
      return error;
  }
  if (logicalNet) {
    if (llvm::Error error = stageLogicalNet(*logicalNet, false))
      return error;
    if (llvm::Error error =
            stageTagUses(logicalNetTagUses_[*logicalNet], false))
      return error;
  }
  if (llvm::Error error = finishUpdate())
    return error;

  std::fill(selectedLogicalNetClaimBits_.begin(),
            selectedLogicalNetClaimBits_.end(), 0);
  selectedLogicalNetTagUses_.clear();
  if (switchRows_)
    switchRows_->selectedNetDemands.clear();
  selectedLogicalNet_ = logicalNet;
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::selectLogicalNet(
    PnrIndex logicalNet, llvm::ArrayRef<std::uint64_t> activeClaimBits) {
  if (selectedLogicalNet_)
    return routeCostStateError("another logical net is already selected");
  if (logicalNet >= logicalNetCount_)
    return routeCostStateError("selected logical net is out of range");
  if (!activeClaimBits.empty() &&
      activeClaimBits.size() != routeClaimWordCount_)
    return routeCostStateError("active claim bitset has the wrong width");

  beginUpdate();
  if (llvm::Error error = stageClaimBits(activeClaimBits, false))
    return error;
  if (llvm::Error error = stageTagUses(logicalNetTagUses_[logicalNet], false))
    return error;
  if (llvm::Error error = finishUpdate())
    return error;
  std::fill(selectedLogicalNetClaimBits_.begin(),
            selectedLogicalNetClaimBits_.end(), 0);
  if (switchRows_)
    switchRows_->selectedNetDemands.clear();
  selectedLogicalNet_ = logicalNet;
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::updateSelectedLogicalNetClaims(
    llvm::ArrayRef<std::uint64_t> claimBits) {
  if (!selectedLogicalNet_)
    return routeCostStateError(
        "prospective claims require one selected logical net");
  if (!claimBits.empty() && claimBits.size() != routeClaimWordCount_)
    return routeCostStateError("selected claim bitset has the wrong width");
  const bool unchanged =
      claimBits.empty()
          ? llvm::all_of(selectedLogicalNetClaimBits_,
                         [](std::uint64_t word) { return word == 0; })
          : llvm::equal(claimBits, selectedLogicalNetClaimBits_);
  if (unchanged)
    return llvm::Error::success();

  beginUpdate();
  if (llvm::Error error = stageClaimBits(selectedLogicalNetClaimBits_, false))
    return error;
  if (llvm::Error error = stageClaimBits(claimBits, true))
    return error;
  if (llvm::Error error = finishUpdate())
    return error;

  if (claimBits.empty()) {
    std::fill(selectedLogicalNetClaimBits_.begin(),
              selectedLogicalNetClaimBits_.end(), 0);
  } else {
    llvm::copy(claimBits, selectedLogicalNetClaimBits_.begin());
  }
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::replaceSelectedTagUses(
    llvm::ArrayRef<SpatialTagDomainUse> replacement) {
  if (!selectedLogicalNet_)
    return routeCostStateError(
        "prospective tag uses require one selected logical net");
  if (llvm::equal(
          replacement, selectedLogicalNetTagUses_,
          [](const SpatialTagDomainUse &lhs, const SpatialTagDomainUse &rhs) {
            return lhs.domain == rhs.domain &&
                   lhs.marginalResidentCount == rhs.marginalResidentCount;
          }))
    return llvm::Error::success();
  beginUpdate();
  if (llvm::Error error = stageTagUses(selectedLogicalNetTagUses_, false))
    return error;
  if (llvm::Error error = stageTagUses(replacement, true))
    return error;
  if (llvm::Error error = finishUpdate())
    return error;
  selectedLogicalNetTagUses_.assign(replacement.begin(), replacement.end());
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::updateSelectedLogicalNetTagUses(
    const RouteTreeState &route,
    const SpatialTagContinuityProjection &continuity) {
  if (!selectedLogicalNet_)
    return routeCostStateError(
        "prospective tag uses require one selected logical net");
  if (switchRows_ && !switchRows_->enabled) {
    std::vector<PnrIndex> domains(continuity.segmentDomains().begin(),
                                  continuity.segmentDomains().end());
    llvm::sort(domains);
    std::vector<SpatialTagDomainUse> uses;
    for (std::size_t begin = 0; begin < domains.size();) {
      std::size_t end = begin + 1;
      while (end < domains.size() && domains[end] == domains[begin])
        ++end;
      uses.push_back({domains[begin], end - begin});
      begin = end;
    }
    return replaceSelectedTagUses(uses);
  }
  if (!switchRows_ || switchRows_->netDemands.size() != logicalNetCount_ ||
      switchRows_->netDemandsSettled.size() != logicalNetCount_)
    return routeCostStateError("Temporal switch row projection is unavailable");
  const auto domains = problem_->routing().tagContinuity().matchDomains();
  std::map<PnrIndex, std::uint64_t> marginalRows;
  const auto segmentOffsets = continuity.segmentDomainOffsets();
  const auto segmentDomains = continuity.segmentDomains();
  for (PnrIndex segment = 0; segment < continuity.segments().size(); ++segment)
    for (PnrIndex incidence = segmentOffsets[segment];
         incidence < segmentOffsets[segment + 1]; ++incidence) {
      const PnrIndex domain = segmentDomains[incidence];
      if (domain >= domains.size())
        return routeCostStateError("prospective tag domain is out of range");
      if (domains[domain].kind !=
          ::loom::fabric::FabricPhysicalTagMatchDomainKind::TemporalSwitchTable)
        ++marginalRows[domain];
    }

  using Demand = detail::SpatialTemporalSwitchSegmentDemand;
  struct CandidateDemand final {
    const Demand *route = nullptr;
    std::optional<llvm::APInt> tag;
  };
  std::vector<std::vector<CandidateDemand>> baseDemands(domains.size());
  if (logicalNetTagValues_.size() != logicalNetCount_)
    return routeCostStateError("settled switch tag snapshot is unavailable");
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
    if (logicalNet == *selectedLogicalNet_)
      continue;
    const auto &values = logicalNetTagValues_[logicalNet];
    for (const Demand &demand : switchRows_->netDemands[logicalNet]) {
      if (demand.domain >= domains.size() ||
          (switchRows_->netDemandsSettled[logicalNet] &&
           demand.segment >= values.size()))
        return routeCostStateError("settled switch row demand is out of range");
      baseDemands[demand.domain].push_back(
          {&demand, switchRows_->netDemandsSettled[logicalNet]
                        ? values[demand.segment]
                        : std::nullopt});
    }
  }

  auto prospective = detail::deriveSpatialTemporalSwitchSegmentDemands(
      *problem_, *selectedLogicalNet_, route, continuity);
  if (!prospective)
    return prospective.takeError();
  std::vector<std::vector<const Demand *>> prospectiveDemands(domains.size());
  for (const Demand &demand : *prospective) {
    if (demand.domain >= prospectiveDemands.size())
      return routeCostStateError("prospective switch demand is out of range");
    prospectiveDemands[demand.domain].push_back(&demand);
  }

  auto projectRows = [&](llvm::ArrayRef<CandidateDemand> candidates,
                         std::uint32_t tagWidthBits)
      -> llvm::Expected<
          std::vector<::loom::fabric::FabricTemporalSwitchCandidateRouteRow>> {
    std::vector<
        std::vector<::loom::fabric::FabricTemporalSwitchRouteSignatureView>>
        signatureStorage;
    signatureStorage.reserve(candidates.size());
    for (const CandidateDemand &candidate : candidates) {
      if (!candidate.route)
        return routeCostStateError("switch row demand has no route");
      signatureStorage.emplace_back();
      auto &signatures = signatureStorage.back();
      signatures.reserve(candidate.route->signatures.size());
      for (const detail::SpatialTemporalSwitchInputSignature &signature :
           candidate.route->signatures)
        signatures.push_back(
            {signature.occurrence, signature.input, signature.outputs});
    }
    std::vector<::loom::fabric::FabricTemporalSwitchCandidateRouteDemandView>
        views;
    views.reserve(candidates.size());
    for (auto [ordinal, signatures] : llvm::enumerate(signatureStorage)) {
      std::optional<llvm::APInt> tag = candidates[ordinal].tag;
      if (tag)
        tag = tag->zextOrTrunc(tagWidthBits);
      views.push_back({{signatures}, std::move(tag)});
    }
    return ::loom::fabric::projectFabricTemporalSwitchCandidateRouteRows(views);
  };

  for (PnrIndex domain = 0; domain < prospectiveDemands.size(); ++domain) {
    if (prospectiveDemands[domain].empty())
      continue;
    auto baseRows =
        projectRows(baseDemands[domain], domains[domain].tagWidthBits);
    if (!baseRows)
      return baseRows.takeError();
    std::vector<CandidateDemand> combined = baseDemands[domain];
    combined.reserve(combined.size() + prospectiveDemands[domain].size());
    for (const Demand *demand : prospectiveDemands[domain])
      combined.push_back({demand, std::nullopt});
    auto combinedRows = projectRows(combined, domains[domain].tagWidthBits);
    if (!combinedRows)
      return combinedRows.takeError();
    if (combinedRows->size() < baseRows->size())
      return routeCostStateError(
          "Fabric switch row projection reduced settled row occupancy");
    marginalRows[domain] += combinedRows->size() - baseRows->size();
  }
  std::vector<SpatialTagDomainUse> uses;
  uses.reserve(marginalRows.size());
  for (const auto &[domain, count] : marginalRows)
    if (count != 0)
      uses.push_back({domain, count});
  if (llvm::Error error = replaceSelectedTagUses(uses))
    return error;
  switchRows_->selectedNetDemands = std::move(*prospective);
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::acceptSelectedLogicalNet() {
  if (!selectedLogicalNet_)
    return routeCostStateError("no selected logical net can be accepted");
  std::fill(selectedLogicalNetClaimBits_.begin(),
            selectedLogicalNetClaimBits_.end(), 0);
  logicalNetTagUses_[*selectedLogicalNet_] = selectedLogicalNetTagUses_;
  if (switchRows_ && switchRows_->enabled)
    switchRows_->netDemands[*selectedLogicalNet_] =
        std::move(switchRows_->selectedNetDemands);
  if (switchRows_ && switchRows_->enabled)
    switchRows_->netDemandsSettled[*selectedLogicalNet_] = 0;
  selectedLogicalNetTagUses_.clear();
  selectedLogicalNet_.reset();
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::synchronizeCandidateTraversals(
    llvm::ArrayRef<PnrIndex> traversals) {
  if (selectedLogicalNet_)
    return routeCostStateError(
        "cannot synchronize while a logical net is selected");
  if (traversals.empty())
    return llvm::Error::success();

  beginUpdate();
  const FrozenSpatialRoutingGraph &routing = problem_->routing();
  for (PnrIndex traversal : traversals) {
    if (traversal >= routing.traversals().size())
      return routeCostStateError("synchronized traversal is out of range");
    const FrozenSpatialTraversal &record = routing.traversals()[traversal];
    for (PnrIndex claim : routing.traversalClaimKeys().slice(
             record.routeClaimOffset, record.routeClaimCount)) {
      if (claim >= routing.routeClaims().size())
        return routeCostStateError(
            "synchronized traversal claim is out of range");
      const PnrIndex capacity = routing.routeClaims()[claim].capacityDimension;
      if (capacity >= workingCapacityUsageRaw_.size())
        return routeCostStateError(
            "synchronized route capacity is out of range");
      if (capacityUpdateEpochs_[capacity] == updateEpoch_)
        continue;
      capacityUpdateEpochs_[capacity] = updateEpoch_;
      stagedCapacityUsageRaw_[capacity] =
          candidate_->routeCapacityUsageRaw(capacity);
      affectedCapacities_.push_back(capacity);
    }
  }
  return finishUpdate();
}

llvm::Error SpatialRouteCostState::synchronizeTagProjection(
    const SpatialTagAssignmentSummary &summary,
    llvm::ArrayRef<PnrIndex> changedLogicalNets) {
  if (selectedLogicalNet_)
    return routeCostStateError(
        "cannot synchronize tags while a logical net is selected");
  if (inverseTagDelta_)
    return routeCostStateError("a tag projection delta is already active");
  const std::size_t domainCount = workingTagDomainUsage_.size();
  if (summary.domainResidentCounts.size() != domainCount ||
      summary.domainConflictCounts.size() != domainCount ||
      summary.netDomainUseOffsets.size() != logicalNetCount_ + 1 ||
      summary.netDomainUseDomains.size() !=
          summary.netDomainMarginalResidentCounts.size() ||
      summary.netUnassignedCounts.size() != logicalNetCount_ ||
      summary.netDomainUseOffsets.back() !=
          summary.netDomainUseDomains.size() ||
      summary.netTagValueOffsets.size() != logicalNetCount_ + 1 ||
      summary.netTagValueOffsets.back() != summary.netTagValues.size())
    return routeCostStateError("tag projection dimensions are inconsistent");

  if (llvm::Error error = synchronizeCandidateSwitchRows(changedLogicalNets))
    return error;
  std::uint64_t unassignedCount = 0;
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
    const std::size_t begin = summary.netTagValueOffsets[logicalNet];
    const std::size_t end = summary.netTagValueOffsets[logicalNet + 1];
    if (begin > end || end > summary.netTagValues.size())
      return routeCostStateError("tag projection value range is invalid");
    const std::uint64_t netUnassigned =
        llvm::count_if(llvm::ArrayRef<std::optional<llvm::APInt>>(
                           summary.netTagValues.data() + begin, end - begin),
                       [](const auto &value) { return !value.has_value(); });
    if (netUnassigned != summary.netUnassignedCounts[logicalNet])
      return routeCostStateError(
          "tag projection unassigned count disagrees with its values");
    unassignedCount = saturatedAdd(unassignedCount, netUnassigned);
    if (switchRows_ && switchRows_->enabled)
      for (const auto &demand : switchRows_->netDemands[logicalNet])
        if (demand.segment >= end - begin)
          return routeCostStateError(
              "switch row demand is outside the synchronized tag snapshot");
  }
  if (unassignedCount != summary.unassignedCount)
    return routeCostStateError(
        "tag projection total unassigned count disagrees with its values");
  if (switchRows_ && switchRows_->enabled) {
    std::fill(switchRows_->netDemandsSettled.begin(),
              switchRows_->netDemandsSettled.end(), 1);
  }
  logicalNetTagValues_.resize(logicalNetCount_);
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
    const std::size_t begin = summary.netTagValueOffsets[logicalNet];
    const std::size_t end = summary.netTagValueOffsets[logicalNet + 1];
    logicalNetTagValues_[logicalNet].assign(summary.netTagValues.begin() +
                                                begin,
                                            summary.netTagValues.begin() + end);
  }
  tagDomainConflictCounts_ = summary.domainConflictCounts;
  logicalNetTagUnassignedCounts_ = summary.netUnassignedCounts;
  const auto updateLogicalNet = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= logicalNetCount_)
      return routeCostStateError("changed logical net is out of range");
    auto &uses = logicalNetTagUses_[logicalNet];
    uses.clear();
    const std::size_t begin = summary.netDomainUseOffsets[logicalNet];
    const std::size_t end = summary.netDomainUseOffsets[logicalNet + 1];
    uses.reserve(end - begin);
    for (std::size_t incidence = begin; incidence < end; ++incidence) {
      const PnrIndex domain = summary.netDomainUseDomains[incidence];
      const std::uint64_t count =
          summary.netDomainMarginalResidentCounts[incidence];
      if (domain >= domainCount || count == 0)
        return routeCostStateError("tag projection use is out of range");
      uses.push_back({domain, count});
    }
    return llvm::Error::success();
  };
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet)
    if (llvm::Error error = updateLogicalNet(logicalNet))
      return error;
  beginUpdate();
  for (PnrIndex domain = 0; domain < domainCount; ++domain) {
    if (workingTagDomainUsage_[domain] == summary.domainResidentCounts[domain])
      continue;
    tagDomainUpdateEpochs_[domain] = updateEpoch_;
    stagedTagDomainUsage_[domain] = summary.domainResidentCounts[domain];
    affectedTagDomains_.push_back(domain);
  }
  if (llvm::Error error = finishUpdate())
    return error;
  tagUnassignedCount_ = summary.unassignedCount;
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::synchronizeCandidateTags() {
  if (selectedLogicalNet_)
    return routeCostStateError(
        "cannot synchronize candidate tags while a logical net is selected");
  return rebuildTagProjectionFromCandidate(false);
}

llvm::Error SpatialRouteCostState::rebuildSwitchRowProjectionFromCandidate() {
  if (!switchRows_)
    return routeCostStateError("Temporal switch row storage is unavailable");
  if (!switchRows_->enabled)
    return llvm::Error::success();
  switchRows_->netDemands.assign(logicalNetCount_, {});
  switchRows_->netDemandsSettled.assign(logicalNetCount_, 0);
  if (llvm::Error error = synchronizeCandidateSwitchRows({}))
    return error;
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
    const auto values = candidate_->tagValues(logicalNet);
    logicalNetTagValues_[logicalNet].assign(values.begin(), values.end());
    for (const auto &demand : switchRows_->netDemands[logicalNet])
      if (demand.segment >= values.size())
        return routeCostStateError(
            "candidate switch row demand is outside its tag snapshot");
  }
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::synchronizeCandidateSwitchRows(
    llvm::ArrayRef<PnrIndex> changedLogicalNets) {
  if (!switchRows_)
    return routeCostStateError("Temporal switch row storage is unavailable");
  switchRows_->selectedNetDemands.clear();
  if (!switchRows_->enabled)
    return llvm::Error::success();
  if (switchRows_->netDemands.size() != logicalNetCount_ ||
      switchRows_->netDemandsSettled.size() != logicalNetCount_)
    return routeCostStateError(
        "Temporal switch row demand domain has the wrong width");

  std::vector<std::pair<
      PnrIndex, std::vector<detail::SpatialTemporalSwitchSegmentDemand>>>
      replacements;
  replacements.reserve(changedLogicalNets.empty() ? logicalNetCount_
                                                  : changedLogicalNets.size());
  SpatialTagContinuityProjection continuity;
  SpatialTagContinuityScratch continuityScratch;
  const auto append = [&](PnrIndex logicalNet) -> llvm::Error {
    if (logicalNet >= logicalNetCount_)
      return routeCostStateError(
          "changed switch row logical net is out of range");
    const RouteTreeState &route = candidate_->routeTree(logicalNet);
    if (llvm::Error error = detail::rebuildSpatialTagContinuityUnchecked(
            route, continuity, continuityScratch))
      return error;
    auto demands = detail::deriveSpatialTemporalSwitchSegmentDemands(
        *problem_, logicalNet, route, continuity);
    if (!demands)
      return demands.takeError();
    replacements.emplace_back(logicalNet, std::move(*demands));
    return llvm::Error::success();
  };
  if (changedLogicalNets.empty()) {
    for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet)
      if (llvm::Error error = append(logicalNet))
        return error;
  } else {
    for (PnrIndex logicalNet : changedLogicalNets)
      if (llvm::Error error = append(logicalNet))
        return error;
  }
  for (auto &replacement : replacements) {
    switchRows_->netDemands[replacement.first] = std::move(replacement.second);
    switchRows_->netDemandsSettled[replacement.first] = 1;
  }
  return llvm::Error::success();
}

llvm::Error
SpatialRouteCostState::rebuildTagProjectionFromCandidate(bool resetHistory) {
  if (llvm::Error error = rebuildSwitchRowProjectionFromCandidate())
    return error;
  logicalNetTagValues_.resize(logicalNetCount_);
  tagUnassignedCount_ = candidate_->tagUnassignedCount();
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
    const auto values = candidate_->tagValues(logicalNet);
    logicalNetTagValues_[logicalNet].assign(values.begin(), values.end());
  }
  const std::size_t domainCount = workingTagDomainUsage_.size();
  std::fill(workingTagDomainUsage_.begin(), workingTagDomainUsage_.end(), 0);
  if (!switchRows_->enabled) {
    for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
      std::vector<PnrIndex> domains;
      const auto segments = candidate_->tagSegments(logicalNet);
      logicalNetTagUnassignedCounts_[logicalNet] =
          llvm::count_if(candidate_->tagValues(logicalNet),
                         [](const auto &value) { return !value.has_value(); });
      for (PnrIndex segment = 0; segment < segments.size(); ++segment) {
        const auto local = candidate_->tagSegmentDomains(logicalNet, segment);
        domains.insert(domains.end(), local.begin(), local.end());
      }
      llvm::sort(domains);
      auto &uses = logicalNetTagUses_[logicalNet];
      uses.clear();
      for (std::size_t begin = 0; begin < domains.size();) {
        std::size_t end = begin + 1;
        while (end < domains.size() && domains[end] == domains[begin])
          ++end;
        if (domains[begin] >= domainCount)
          return routeCostStateError("candidate tag domain is out of range");
        const std::uint64_t count = end - begin;
        uses.push_back({domains[begin], count});
        workingTagDomainUsage_[domains[begin]] += count;
        begin = end;
      }
    }
    for (PnrIndex domain = 0; domain < domainCount; ++domain) {
      if (workingTagDomainUsage_[domain] !=
          candidate_->tagDomainResidentCount(domain))
        return routeCostStateError(
            "candidate tag-domain usage disagrees with segment incidence");
      tagDomainConflictCounts_[domain] =
          candidate_->tagDomainConflictCount(domain);
    }
    return recomputeAllArcCosts(resetHistory);
  }
  std::vector<llvm::DenseMap<llvm::APInt, std::vector<PnrIndex>>> assignedRows(
      domainCount);
  std::vector<std::map<PnrIndex, std::uint64_t>> marginalRows(logicalNetCount_);
  const auto matchDomains = problem_->routing().tagContinuity().matchDomains();
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
    const auto segments = candidate_->tagSegments(logicalNet);
    const auto values = candidate_->tagValues(logicalNet);
    logicalNetTagUnassignedCounts_[logicalNet] = 0;
    for (PnrIndex segment = 0; segment < segments.size(); ++segment) {
      const auto local = candidate_->tagSegmentDomains(logicalNet, segment);
      if (segment >= values.size())
        return routeCostStateError("candidate tag value inventory is short");
      if (!values[segment])
        ++logicalNetTagUnassignedCounts_[logicalNet];
      for (PnrIndex domain : local) {
        if (domain >= domainCount)
          return routeCostStateError("candidate tag domain is out of range");
        const bool packedSwitch =
            matchDomains[domain].kind ==
            ::loom::fabric::FabricPhysicalTagMatchDomainKind::
                TemporalSwitchTable;
        if (values[segment] && packedSwitch) {
          assignedRows[domain][*values[segment]].push_back(logicalNet);
        } else {
          if (workingTagDomainUsage_[domain] ==
              std::numeric_limits<std::uint64_t>::max())
            return routeCostStateError("candidate tag usage overflows u64");
          ++workingTagDomainUsage_[domain];
          ++marginalRows[logicalNet][domain];
        }
      }
    }
  }
  for (PnrIndex domain = 0; domain < domainCount; ++domain)
    for (auto &entry : assignedRows[domain]) {
      if (workingTagDomainUsage_[domain] ==
          std::numeric_limits<std::uint64_t>::max())
        return routeCostStateError("candidate tag usage overflows u64");
      ++workingTagDomainUsage_[domain];
      auto &logicalNets = entry.second;
      llvm::sort(logicalNets);
      logicalNets.erase(std::unique(logicalNets.begin(), logicalNets.end()),
                        logicalNets.end());
      if (logicalNets.size() == 1)
        ++marginalRows[logicalNets.front()][domain];
    }
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
    auto &uses = logicalNetTagUses_[logicalNet];
    uses.clear();
    uses.reserve(marginalRows[logicalNet].size());
    for (const auto &[domain, count] : marginalRows[logicalNet])
      if (count != 0)
        uses.push_back({domain, count});
  }
  for (PnrIndex domain = 0; domain < domainCount; ++domain) {
    if (workingTagDomainUsage_[domain] !=
        candidate_->tagDomainResidentCount(domain))
      return routeCostStateError(
          "candidate tag-domain usage disagrees with segment incidence");
    tagDomainConflictCounts_[domain] =
        candidate_->tagDomainConflictCount(domain);
  }
  return recomputeAllArcCosts(resetHistory);
}

llvm::Error SpatialRouteCostState::recomputeAllArcCosts(bool resetTagHistory) {
  const auto domains = problem_->routing().tagContinuity().matchDomains();
  if (resetTagHistory) {
    std::fill(tagResidentHistoryPressure_.begin(),
              tagResidentHistoryPressure_.end(), 0);
    std::fill(tagEncodingHistoryPressure_.begin(),
              tagEncodingHistoryPressure_.end(), 0);
  }
  for (PnrIndex domain = 0; domain < domains.size(); ++domain) {
    if (domains[domain].residentEntryCapacity) {
      auto resident =
          normalizedRouteOveruseCost(workingTagDomainUsage_[domain], 0,
                                     *domains[domain].residentEntryCapacity);
      if (!resident)
        return resident.takeError();
      tagResidentOveruseCosts_[domain] = *resident;
    } else {
      tagResidentOveruseCosts_[domain] = 0;
    }
    auto encoding = normalizedRouteClaimCost(
        saturatedAdd(encodingPressureRaw(domain, false), 1),
        encodingCapacity(domains[domain].tagWidthBits));
    if (!encoding)
      return encoding.takeError();
    tagEncodingPressureCosts_[domain] = *encoding;
  }
  if (lowerBoundArcCosts_.empty() || currentArcCosts_.empty())
    return llvm::Error::success();
  bool lowerBoundChanged = false;
  for (PnrIndex arc = 0; arc < currentArcCosts_.size(); ++arc) {
    if (!problem_->activeRouting().arcIsActive(arc)) {
      lowerBoundChanged |= lowerBoundArcCosts_[arc] != 0;
      lowerBoundArcCosts_[arc] = 0;
      currentArcCosts_[arc] = 0;
      continue;
    }
    auto lower = computeArcCost(arc, false, false, false);
    if (!lower)
      return lower.takeError();
    lowerBoundChanged |= lowerBoundArcCosts_[arc] != *lower;
    lowerBoundArcCosts_[arc] = *lower;
    auto current = computeArcCost(arc, true, false, false);
    if (!current)
      return current.takeError();
    currentArcCosts_[arc] = *current;
  }
  if (lowerBoundChanged) {
    if (lowerBoundCostRevision_ == std::numeric_limits<std::uint64_t>::max())
      return routeCostStateError("lower-bound cost revision overflows u64");
    ++lowerBoundCostRevision_;
  }
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::resetFromCandidate() {
  if (llvm::Error error = candidate_->verify())
    return error;

  return resetFromVerifiedCandidate();
}

llvm::Error SpatialRouteCostState::resetFromVerifiedCandidate() {
  inverseTagDelta_.reset();
  if (switchRows_)
    switchRows_->demandJournal.clear();
  beginUpdate();
  const auto capacities = problem_->resources().capacityDimensions();
  for (PnrIndex capacity = 0; capacity < capacities.size(); ++capacity) {
    capacityUpdateEpochs_[capacity] = updateEpoch_;
    stagedCapacityUsageRaw_[capacity] =
        candidate_->routeCapacityUsageRaw(capacity);
    auto overuse = normalizedRouteOveruseCost(stagedCapacityUsageRaw_[capacity],
                                              0, capacities[capacity].capacity);
    if (!overuse)
      return overuse.takeError();
    stagedCapacityOveruseCosts_[capacity] = *overuse;
  }
  for (PnrIndex claim = 0; claim < routeClaimCount_; ++claim) {
    claimUpdateEpochs_[claim] = updateEpoch_;
    const FrozenSpatialRouteClaim &record =
        problem_->routing().routeClaims()[claim];
    auto overuse = normalizedRouteOveruseCost(
        stagedCapacityUsageRaw_[record.capacityDimension], record.amount,
        capacities[record.capacityDimension].capacity);
    if (!overuse)
      return overuse.takeError();
    stagedClaimOveruseCosts_[claim] = *overuse;
  }
  std::fill(stagedHistoryPressure_.begin(), stagedHistoryPressure_.end(), 0);
  for (PnrIndex traversal = 0;
       traversal < problem_->routing().traversals().size(); ++traversal) {
    if (!problem_->activeRouting().traversalIsActive(traversal)) {
      stagedTraversalCosts_[traversal] = 0;
      continue;
    }
    auto cost = computeTraversalCostImpl(traversal, true, true,
                                         policy_.presentPressureInitial,
                                         stagedHistoryPressure_);
    if (!cost)
      return cost.takeError();
    stagedTraversalCosts_[traversal] = *cost;
  }

  for (PnrIndex capacity = 0; capacity < capacities.size(); ++capacity) {
    workingCapacityUsageRaw_[capacity] = stagedCapacityUsageRaw_[capacity];
    capacityOveruseCosts_[capacity] = stagedCapacityOveruseCosts_[capacity];
  }
  llvm::copy(stagedClaimOveruseCosts_, currentClaimOveruseCosts_.begin());
  llvm::copy(stagedTraversalCosts_, currentTraversalCosts_.begin());
  for (PnrIndex arc = 0; arc < problem_->routing().routingArcs().size();
       ++arc) {
    currentArcCosts_[arc] =
        problem_->activeRouting().arcIsActive(arc)
            ? currentTraversalCosts_
                  [problem_->routing().routingArcs()[arc].traversal]
            : 0;
  }
  presentPressure_ = policy_.presentPressureInitial;
  std::fill(historyPressure_.begin(), historyPressure_.end(), 0);
  std::fill(selectedLogicalNetClaimBits_.begin(),
            selectedLogicalNetClaimBits_.end(), 0);
  selectedLogicalNetTagUses_.clear();
  selectedLogicalNet_.reset();
  lowerBoundCostRevision_ = 0;
  return rebuildTagProjectionFromCandidate(true);
}

llvm::Error SpatialRouteCostState::advancePathFinderIteration() {
  if (selectedLogicalNet_)
    return routeCostStateError(
        "cannot advance PathFinder while a logical net is selected");
  auto nextPressure =
      ceilMulDiv(presentPressure_, policy_.presentPressureGrowth.numerator,
                 policy_.presentPressureGrowth.denominator);
  if (!nextPressure)
    return nextPressure.takeError();

  beginUpdate();
  const FrozenSpatialRoutingGraph &routing = problem_->routing();
  for (PnrIndex capacity = 0; capacity < historyPressure_.size(); ++capacity) {
    auto nextHistory = pathFinderHistoryUpdate(historyPressure_[capacity],
                                               policy_.historyPressureIncrement,
                                               capacityOveruseCosts_[capacity]);
    if (!nextHistory)
      return nextHistory.takeError();
    stagedHistoryPressure_[capacity] = *nextHistory;

    const auto claims = routing.capacityRouteClaims().slice(
        routing.capacityRouteClaimOffsets()[capacity],
        routing.capacityRouteClaimOffsets()[capacity + 1] -
            routing.capacityRouteClaimOffsets()[capacity]);
    const bool presentCostChanged =
        *nextPressure != presentPressure_ &&
        llvm::any_of(claims, [&](PnrIndex claim) {
          return currentClaimOveruseCosts_[claim] != 0;
        });
    if (!presentCostChanged && *nextHistory == historyPressure_[capacity])
      continue;

    for (PnrIndex claim : claims) {
      for (PnrIndex traversal : routing.routeClaimTraversals().slice(
               routing.routeClaimTraversalOffsets()[claim],
               routing.routeClaimTraversalOffsets()[claim + 1] -
                   routing.routeClaimTraversalOffsets()[claim])) {
        if (!problem_->activeRouting().traversalIsActive(traversal))
          continue;
        if (traversalUpdateEpochs_[traversal] == updateEpoch_)
          continue;
        traversalUpdateEpochs_[traversal] = updateEpoch_;
        affectedTraversals_.push_back(traversal);
      }
    }
  }
  const auto domains = problem_->routing().tagContinuity().matchDomains();
  for (PnrIndex domain = 0; domain < domains.size(); ++domain) {
    auto nextResident = pathFinderHistoryUpdate(
        tagResidentHistoryPressure_[domain], policy_.historyPressureIncrement,
        tagResidentOveruseCosts_[domain]);
    if (!nextResident)
      return nextResident.takeError();
    const std::uint64_t widthOveruse =
        workingTagDomainUsage_[domain] >
                encodingCapacity(domains[domain].tagWidthBits)
            ? workingTagDomainUsage_[domain] -
                  encodingCapacity(domains[domain].tagWidthBits)
            : 0;
    const std::uint64_t encodingOveruse =
        std::max(widthOveruse, tagDomainConflictCounts_[domain]);
    auto nextEncoding = pathFinderHistoryUpdate(
        tagEncodingHistoryPressure_[domain], policy_.historyPressureIncrement,
        encodingOveruse);
    if (!nextEncoding)
      return nextEncoding.takeError();
    const bool changed = *nextResident != tagResidentHistoryPressure_[domain] ||
                         *nextEncoding != tagEncodingHistoryPressure_[domain] ||
                         (*nextPressure != presentPressure_ &&
                          (tagResidentOveruseCosts_[domain] != 0 ||
                           tagEncodingPressureCosts_[domain] != 0));
    stagedTagResidentHistoryPressure_[domain] = *nextResident;
    stagedTagEncodingHistoryPressure_[domain] = *nextEncoding;
    if (!changed)
      continue;
    for (PnrIndex incidence = tagDomainArcOffsets_[domain];
         incidence < tagDomainArcOffsets_[domain + 1]; ++incidence) {
      const PnrIndex arc = tagDomainArcs_[incidence];
      if (arcUpdateEpochs_[arc] == updateEpoch_)
        continue;
      arcUpdateEpochs_[arc] = updateEpoch_;
      affectedTagArcs_.push_back(arc);
    }
  }

  for (PnrIndex traversal : affectedTraversals_)
    for (PnrIndex arc : problem_->routing().traversalArcs().slice(
             problem_->routing().traversalArcOffsets()[traversal],
             problem_->routing().traversalArcOffsets()[traversal + 1] -
                 problem_->routing().traversalArcOffsets()[traversal])) {
      if (!problem_->activeRouting().arcIsActive(arc))
        continue;
      if (arcUpdateEpochs_[arc] == updateEpoch_)
        continue;
      arcUpdateEpochs_[arc] = updateEpoch_;
      affectedTagArcs_.push_back(arc);
    }
  for (PnrIndex traversal : affectedTraversals_) {
    auto cost =
        computeTraversalCost(traversal, *nextPressure, stagedHistoryPressure_);
    if (!cost)
      return cost.takeError();
    stagedTraversalCosts_[traversal] = *cost;
  }
  for (PnrIndex arc : affectedTagArcs_) {
    auto cost = computeArcCost(arc, *nextPressure, stagedHistoryPressure_,
                               stagedTagResidentHistoryPressure_,
                               stagedTagEncodingHistoryPressure_);
    if (!cost)
      return cost.takeError();
    stagedArcCosts_[arc] = *cost;
  }

  presentPressure_ = *nextPressure;
  llvm::copy(stagedHistoryPressure_, historyPressure_.begin());
  llvm::copy(stagedTagResidentHistoryPressure_,
             tagResidentHistoryPressure_.begin());
  llvm::copy(stagedTagEncodingHistoryPressure_,
             tagEncodingHistoryPressure_.begin());
  for (PnrIndex traversal : affectedTraversals_)
    currentTraversalCosts_[traversal] = stagedTraversalCosts_[traversal];
  for (PnrIndex arc : affectedTagArcs_)
    currentArcCosts_[arc] = stagedArcCosts_[arc];
  return llvm::Error::success();
}

std::size_t SpatialRouteCostState::retainedStorageBytes() const {
  return (switchRows_ ? switchRows_->retainedStorageBytes() : 0) +
         (inverseTagDelta_ ? retainedBytes(*inverseTagDelta_) : 0) +
         retainedBytes(workingCapacityUsageRaw_) +
         retainedBytes(historyPressure_) +
         retainedBytes(capacityOveruseCosts_) +
         retainedBytes(currentClaimOveruseCosts_) +
         retainedBytes(lowerBoundTraversalCosts_) +
         retainedBytes(currentTraversalCosts_) +
         retainedBytes(lowerBoundArcCosts_) + retainedBytes(currentArcCosts_) +
         retainedBytes(selectedLogicalNetClaimBits_) +
         retainedNestedBytes(logicalNetTagUses_) +
         retainedBytes(logicalNetTagUnassignedCounts_) +
         retainedNestedBytes(logicalNetTagValues_) +
         retainedBytes(selectedLogicalNetTagUses_) +
         retainedBytes(workingTagDomainUsage_) +
         retainedBytes(tagDomainConflictCounts_) +
         retainedBytes(tagResidentHistoryPressure_) +
         retainedBytes(tagEncodingHistoryPressure_) +
         retainedBytes(tagResidentOveruseCosts_) +
         retainedBytes(tagEncodingPressureCosts_) +
         retainedBytes(tagDomainArcOffsets_) + retainedBytes(tagDomainArcs_) +
         retainedBytes(capacityUpdateEpochs_) +
         retainedBytes(claimUpdateEpochs_) +
         retainedBytes(traversalUpdateEpochs_) +
         retainedBytes(arcUpdateEpochs_) +
         retainedBytes(stagedCapacityUsageRaw_) +
         retainedBytes(stagedHistoryPressure_) +
         retainedBytes(stagedTagResidentHistoryPressure_) +
         retainedBytes(stagedTagEncodingHistoryPressure_) +
         retainedBytes(stagedCapacityOveruseCosts_) +
         retainedBytes(stagedClaimOveruseCosts_) +
         retainedBytes(stagedTraversalCosts_) +
         retainedBytes(tagDomainUpdateEpochs_) +
         retainedBytes(stagedTagDomainUsage_) +
         retainedBytes(stagedTagResidentOveruseCosts_) +
         retainedBytes(stagedTagEncodingPressureCosts_) +
         retainedBytes(stagedArcCosts_) + retainedBytes(affectedCapacities_) +
         retainedBytes(affectedClaims_) + retainedBytes(affectedTraversals_) +
         retainedBytes(affectedTagDomains_) + retainedBytes(affectedTagArcs_);
}

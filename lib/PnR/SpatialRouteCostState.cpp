#include "PnR/SpatialRouteCostState.h"

#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>
#include <utility>

using namespace loom;
using namespace loom::pnr;

namespace {

llvm::Error routeCostStateError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial route cost state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

} // namespace

llvm::Expected<SpatialRouteCostState>
SpatialRouteCostState::create(const SpatialCandidateState &candidate,
                              const ResolvedPathFinderPolicy &policy) {
  if (llvm::Error error = validateResolvedPathFinderPolicy(policy))
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
  state.policy_ = policy;
  state.logicalNetCount_ =
      static_cast<PnrIndex>(problem.transfers().logicalNets().size());
  state.routeClaimCount_ =
      static_cast<PnrIndex>(problem.routing().routeClaims().size());
  state.routeClaimWordCount_ =
      (static_cast<std::size_t>(state.routeClaimCount_) + 63) / 64;
  state.presentPressure_ = policy.presentPressureInitial;

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
  for (PnrIndex capacity = 0; capacity < capacityCount; ++capacity)
    state.workingCapacityUsageRaw_.push_back(
        candidate.routeCapacityUsageRaw(capacity));
  state.historyPressure_.assign(capacityCount, 0);
  state.capacityUpdateEpochs_.assign(capacityCount, 0);
  state.stagedCapacityUsageRaw_.assign(capacityCount, 0);
  state.affectedCapacities_.reserve(capacityCount);

  state.currentClaimOveruseCosts_.assign(state.routeClaimCount_, 0);
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

  const std::size_t traversalCount = problem.routing().traversals().size();
  state.lowerBoundTraversalCosts_.assign(traversalCount, 0);
  state.currentTraversalCosts_.assign(traversalCount, 0);
  state.traversalUpdateEpochs_.assign(traversalCount, 0);
  state.stagedTraversalCosts_.assign(traversalCount, 0);
  state.affectedTraversals_.reserve(traversalCount);
  for (PnrIndex traversal = 0; traversal < traversalCount; ++traversal) {
    auto lower = state.computeTraversalCost(traversal, false, false);
    if (!lower)
      return lower.takeError();
    state.lowerBoundTraversalCosts_[traversal] = *lower;
    auto current = state.computeTraversalCost(traversal, true, false);
    if (!current)
      return current.takeError();
    state.currentTraversalCosts_[traversal] = *current;
  }

  state.lowerBoundArcCosts_.reserve(problem.routing().routingArcs().size());
  state.currentArcCosts_.reserve(problem.routing().routingArcs().size());
  for (const FrozenSpatialRoutingArc &arc : problem.routing().routingArcs()) {
    if (arc.traversal >= traversalCount)
      return routeCostStateError("routing arc traversal is out of range");
    state.lowerBoundArcCosts_.push_back(
        state.lowerBoundTraversalCosts_[arc.traversal]);
    state.currentArcCosts_.push_back(
        state.currentTraversalCosts_[arc.traversal]);
  }
  return state;
}

std::uint64_t SpatialRouteCostState::workingCapacityUsageRaw(
    PnrIndex capacityDimension) const {
  assert(capacityDimension < workingCapacityUsageRaw_.size());
  return workingCapacityUsageRaw_[capacityDimension];
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
    updateEpoch_ = 1;
  }
  affectedCapacities_.clear();
  affectedClaims_.clear();
  affectedTraversals_.clear();
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
    if (record.amount > usage)
      return routeCostStateError("excluded route occupancy underflows u64");
    usage -= record.amount;
  }
  return llvm::Error::success();
}

llvm::Error SpatialRouteCostState::stageLogicalNet(PnrIndex logicalNet,
                                                   bool restore) {
  if (logicalNet >= logicalNetCount_)
    return routeCostStateError("selected logical net is out of range");
  for (auto [wordOrdinal, initialWord] :
       llvm::enumerate(logicalNetClaimBits(logicalNet))) {
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

llvm::Error SpatialRouteCostState::collectAndPriceAffectedClaims() {
  const FrozenSpatialRoutingGraph &routing = problem_->routing();
  for (PnrIndex capacity : affectedCapacities_) {
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
        if (traversalUpdateEpochs_[traversal] == updateEpoch_)
          continue;
        traversalUpdateEpochs_[traversal] = updateEpoch_;
        affectedTraversals_.push_back(traversal);
      }
    }
  }
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

llvm::Expected<RouteCost> SpatialRouteCostState::computeTraversalCost(
    PnrIndex traversal, bool dynamicCost, bool stagedClaims) const {
  if (traversal >= problem_->routing().traversals().size())
    return routeCostStateError("traversal cost index is out of range");
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
          claimOveruseForCost(claim, stagedClaims), presentPressure_,
          historyPressure_[claimRecord.capacityDimension]);
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
  if (selectedLogicalNet_)
    if (llvm::Error error = stageLogicalNet(*selectedLogicalNet_, true))
      return error;
  if (logicalNet)
    if (llvm::Error error = stageLogicalNet(*logicalNet, false))
      return error;

  if (llvm::Error error = collectAndPriceAffectedClaims())
    return error;
  for (PnrIndex traversal : affectedTraversals_) {
    auto cost = computeTraversalCost(traversal, true, true);
    if (!cost)
      return cost.takeError();
    stagedTraversalCosts_[traversal] = *cost;
  }

  for (PnrIndex capacity : affectedCapacities_)
    workingCapacityUsageRaw_[capacity] = stagedCapacityUsageRaw_[capacity];
  for (PnrIndex claim : affectedClaims_)
    currentClaimOveruseCosts_[claim] = stagedClaimOveruseCosts_[claim];
  for (PnrIndex traversal : affectedTraversals_) {
    currentTraversalCosts_[traversal] = stagedTraversalCosts_[traversal];
    for (PnrIndex arc : problem_->routing().traversalArcs().slice(
             problem_->routing().traversalArcOffsets()[traversal],
             problem_->routing().traversalArcOffsets()[traversal + 1] -
                 problem_->routing().traversalArcOffsets()[traversal]))
      currentArcCosts_[arc] = stagedTraversalCosts_[traversal];
  }
  selectedLogicalNet_ = logicalNet;
  return llvm::Error::success();
}

std::size_t SpatialRouteCostState::retainedStorageBytes() const {
  return retainedBytes(workingCapacityUsageRaw_) +
         retainedBytes(historyPressure_) +
         retainedBytes(currentClaimOveruseCosts_) +
         retainedBytes(lowerBoundTraversalCosts_) +
         retainedBytes(currentTraversalCosts_) +
         retainedBytes(lowerBoundArcCosts_) + retainedBytes(currentArcCosts_) +
         retainedBytes(capacityUpdateEpochs_) +
         retainedBytes(claimUpdateEpochs_) +
         retainedBytes(traversalUpdateEpochs_) +
         retainedBytes(stagedCapacityUsageRaw_) +
         retainedBytes(stagedClaimOveruseCosts_) +
         retainedBytes(stagedTraversalCosts_) +
         retainedBytes(affectedCapacities_) + retainedBytes(affectedClaims_) +
         retainedBytes(affectedTraversals_);
}

#include "PnR/SpatialRouteCostState.h"

#include "llvm/ADT/STLExtras.h"
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

  for (PnrIndex capacity : affectedCapacities_)
    workingCapacityUsageRaw_[capacity] = stagedCapacityUsageRaw_[capacity];
  for (PnrIndex capacity : affectedCapacities_)
    capacityOveruseCosts_[capacity] = stagedCapacityOveruseCosts_[capacity];
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
    if (llvm::Error error = stageLogicalNet(*selectedLogicalNet_, true))
      return error;
  }
  if (logicalNet)
    if (llvm::Error error = stageLogicalNet(*logicalNet, false))
      return error;
  if (llvm::Error error = finishUpdate())
    return error;

  std::fill(selectedLogicalNetClaimBits_.begin(),
            selectedLogicalNetClaimBits_.end(), 0);
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
  if (llvm::Error error = finishUpdate())
    return error;
  std::fill(selectedLogicalNetClaimBits_.begin(),
            selectedLogicalNetClaimBits_.end(), 0);
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

llvm::Error SpatialRouteCostState::acceptSelectedLogicalNet() {
  if (!selectedLogicalNet_)
    return routeCostStateError("no selected logical net can be accepted");
  std::fill(selectedLogicalNetClaimBits_.begin(),
            selectedLogicalNetClaimBits_.end(), 0);
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

llvm::Error SpatialRouteCostState::resetFromCandidate() {
  if (llvm::Error error = candidate_->verify())
    return error;

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
  for (PnrIndex arc = 0; arc < problem_->routing().routingArcs().size(); ++arc)
    currentArcCosts_[arc] = currentTraversalCosts_
        [problem_->routing().routingArcs()[arc].traversal];
  presentPressure_ = policy_.presentPressureInitial;
  std::fill(historyPressure_.begin(), historyPressure_.end(), 0);
  std::fill(selectedLogicalNetClaimBits_.begin(),
            selectedLogicalNetClaimBits_.end(), 0);
  selectedLogicalNet_.reset();
  return llvm::Error::success();
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
  for (PnrIndex capacity = 0; capacity < historyPressure_.size(); ++capacity) {
    auto nextHistory = pathFinderHistoryUpdate(historyPressure_[capacity],
                                               policy_.historyPressureIncrement,
                                               capacityOveruseCosts_[capacity]);
    if (!nextHistory)
      return nextHistory.takeError();
    stagedHistoryPressure_[capacity] = *nextHistory;
  }
  for (PnrIndex traversal = 0;
       traversal < problem_->routing().traversals().size(); ++traversal) {
    auto cost =
        computeTraversalCost(traversal, *nextPressure, stagedHistoryPressure_);
    if (!cost)
      return cost.takeError();
    stagedTraversalCosts_[traversal] = *cost;
  }

  presentPressure_ = *nextPressure;
  llvm::copy(stagedHistoryPressure_, historyPressure_.begin());
  llvm::copy(stagedTraversalCosts_, currentTraversalCosts_.begin());
  for (PnrIndex arc = 0; arc < problem_->routing().routingArcs().size(); ++arc)
    currentArcCosts_[arc] = currentTraversalCosts_
        [problem_->routing().routingArcs()[arc].traversal];
  return llvm::Error::success();
}

std::size_t SpatialRouteCostState::retainedStorageBytes() const {
  return retainedBytes(workingCapacityUsageRaw_) +
         retainedBytes(historyPressure_) +
         retainedBytes(capacityOveruseCosts_) +
         retainedBytes(currentClaimOveruseCosts_) +
         retainedBytes(lowerBoundTraversalCosts_) +
         retainedBytes(currentTraversalCosts_) +
         retainedBytes(lowerBoundArcCosts_) + retainedBytes(currentArcCosts_) +
         retainedBytes(selectedLogicalNetClaimBits_) +
         retainedBytes(capacityUpdateEpochs_) +
         retainedBytes(claimUpdateEpochs_) +
         retainedBytes(traversalUpdateEpochs_) +
         retainedBytes(stagedCapacityUsageRaw_) +
         retainedBytes(stagedHistoryPressure_) +
         retainedBytes(stagedCapacityOveruseCosts_) +
         retainedBytes(stagedClaimOveruseCosts_) +
         retainedBytes(stagedTraversalCosts_) +
         retainedBytes(affectedCapacities_) + retainedBytes(affectedClaims_) +
         retainedBytes(affectedTraversals_);
}

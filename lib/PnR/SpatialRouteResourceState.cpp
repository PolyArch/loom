#include "PnR/SpatialRouteResourceState.h"

#include "PnR/PnrIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <system_error>
#include <utility>
#include <vector>

using namespace loom::pnr;

namespace {

constexpr llvm::StringLiteral candidateArtifact = "SpatialCandidateState";
constexpr PnrCapacityContext netCountContext{
    candidateArtifact, "route_resource_state", "logical_nets",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext claimCountContext{
    candidateArtifact, "route_resource_state", "route_claims",
    PnrCapacityMeasure::Count};
constexpr PnrCapacityContext traversalCountContext{
    candidateArtifact, "route_resource_state", "physical_traversals",
    PnrCapacityMeasure::Count};
llvm::Error routeResourceError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial route resource state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

llvm::Error checkedAdd(std::uint64_t &value, std::uint64_t amount,
                       llvm::StringRef subject) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - value)
    return routeResourceError(subject + " overflows u64");
  value += amount;
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t> checkedMultiply(std::uint64_t left,
                                              std::uint64_t right,
                                              llvm::StringRef subject) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
    return routeResourceError(subject + " overflows u64");
  return left * right;
}

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
}

std::size_t retainedSparseRefcountBytes(
    const std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> &values) {
  std::size_t bytes = retainedBytes(values);
  for (const auto &value : values)
    bytes += value.getMemorySize();
  return bytes;
}

bool sparseRefcountsEqual(
    const std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> &lhs,
    const std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    if (lhs[index].size() != rhs[index].size())
      return false;
    for (const auto &[key, value] : lhs[index]) {
      const auto found = rhs[index].find(key);
      if (found == rhs[index].end() || found->second != value)
        return false;
    }
  }
  return true;
}

} // namespace

llvm::Expected<SpatialRouteResourceState>
SpatialRouteResourceState::create(const FrozenSpatialPnrProblem &problem) {
  auto logicalNetCount = checkedPnrIndex(
      netCountContext, problem.transfers().logicalNets().size());
  if (!logicalNetCount)
    return logicalNetCount.takeError();
  auto routeClaimCount = checkedPnrIndex(
      claimCountContext, problem.routing().routeClaims().size());
  if (!routeClaimCount)
    return routeClaimCount.takeError();
  auto traversalCount = checkedPnrIndex(traversalCountContext,
                                        problem.routing().traversals().size());
  if (!traversalCount)
    return traversalCount.takeError();
  const std::size_t routeClaimWordCount =
      (static_cast<std::size_t>(*routeClaimCount) + 63) / 64;
  if (routeClaimWordCount != 0 &&
      static_cast<std::size_t>(*logicalNetCount) >
          std::numeric_limits<std::size_t>::max() / routeClaimWordCount)
    return routeResourceError("net-claim bitset exceeds native size_t");

  std::vector<std::uint32_t> initiationIntervals;
  initiationIntervals.reserve(*traversalCount);
  for (const FrozenSpatialTraversal &traversal :
       problem.routing().traversals()) {
    if (traversal.minimumInitiationIntervalCycles == 0)
      return routeResourceError("traversal has a zero initiation interval");
    initiationIntervals.push_back(traversal.minimumInitiationIntervalCycles);
  }
  llvm::sort(initiationIntervals);
  initiationIntervals.erase(
      std::unique(initiationIntervals.begin(), initiationIntervals.end()),
      initiationIntervals.end());
  std::vector<PnrIndex> traversalIntervalOrdinals;
  traversalIntervalOrdinals.reserve(*traversalCount);
  for (const FrozenSpatialTraversal &traversal :
       problem.routing().traversals()) {
    const auto found = llvm::lower_bound(
        initiationIntervals, traversal.minimumInitiationIntervalCycles);
    traversalIntervalOrdinals.push_back(
        static_cast<PnrIndex>(found - initiationIntervals.begin()));
  }

  std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> netClaimRefcounts(
      *logicalNetCount);
  std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> netTraversalRefcounts(
      *logicalNetCount);
  std::vector<std::uint64_t> netClaimActiveBits(
      static_cast<std::size_t>(*logicalNetCount) * routeClaimWordCount, 0);
  std::vector<PnrIndex> claimSelectionCounts(*routeClaimCount, 0);
  std::vector<std::uint64_t> capacityUsageRaw;
  capacityUsageRaw.reserve(problem.resources().capacityDimensions().size());
  std::uint64_t totalCapacityOveruseRaw = 0;
  for (const FrozenSpatialCapacityDimension &dimension :
       problem.resources().capacityDimensions()) {
    capacityUsageRaw.push_back(dimension.initialOccupancy);
    const std::uint64_t overuse =
        dimension.initialOccupancy > dimension.capacity
            ? dimension.initialOccupancy - dimension.capacity
            : 0;
    if (llvm::Error error = checkedAdd(totalCapacityOveruseRaw, overuse,
                                       "initial raw route capacity overuse"))
      return std::move(error);
  }

  return SpatialRouteResourceState(
      problem, *logicalNetCount, *traversalCount, *routeClaimCount,
      routeClaimWordCount, std::move(initiationIntervals),
      std::move(traversalIntervalOrdinals), std::move(netTraversalRefcounts),
      std::move(netClaimRefcounts), std::move(netClaimActiveBits),
      std::move(claimSelectionCounts), std::move(capacityUsageRaw),
      totalCapacityOveruseRaw);
}

llvm::Expected<SpatialRouteResourceState>
SpatialRouteResourceState::projectVerifiedRoutes(
    const FrozenSpatialPnrProblem &problem,
    llvm::ArrayRef<const RouteTreeState *> routeTrees,
    llvm::ArrayRef<PnrIndex> registerFifoTransfers) {
  auto state = create(problem);
  if (!state)
    return state.takeError();
  if (routeTrees.size() != state->logicalNetCount_ ||
      registerFifoTransfers.size() != state->logicalNetCount_)
    return routeResourceError(
        "route count does not match the frozen logical nets");

  for (PnrIndex logicalNet = 0; logicalNet < state->logicalNetCount_;
       ++logicalNet) {
    const RouteTreeState *route = routeTrees[logicalNet];
    if (!route || &route->routingGraph() != &problem.routing())
      return routeResourceError(
          "RouteTree does not belong to the frozen routing graph");
    const PnrIndex localTransfer = registerFifoTransfers[logicalNet];
    if (localTransfer != getInvalidPnrIndex()) {
      if (localTransfer >= problem.localTransfers().options().size())
        return routeResourceError(
            "register-FIFO transfer option is out of range");
      if (!route->isUnrouted())
        return routeResourceError(
            "register-FIFO transfer also has an external route");
      const auto &option = problem.localTransfers().options()[localTransfer];
      if (option.logicalNet != logicalNet)
        return routeResourceError(
            "register-FIFO transfer belongs to another logical net");
      if (llvm::Error error = state->applyTraversalDelta(
              logicalNet, option.writeTraversal, 0, 1))
        return std::move(error);
      if (llvm::Error error = state->applyTraversalDelta(
              logicalNet, option.readTraversal, 0, 1))
        return std::move(error);
      continue;
    }
    for (const RouteTreeNode &node : route->nodeStorage()) {
      if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
        continue;
      if (node.parentArc >= problem.routing().routingArcs().size())
        return routeResourceError("RouteTree parent arc is out of range");
      const PnrIndex traversal =
          problem.routing().routingArcs()[node.parentArc].traversal;
      if (llvm::Error error =
              state->applyTraversalDelta(logicalNet, traversal, 0, 1))
        return std::move(error);
    }
  }
  return state;
}

PnrIndex
SpatialRouteResourceState::routeClaimSelectionCount(PnrIndex claim) const {
  assert(claim < claimSelectionCounts_.size());
  return claimSelectionCounts_[claim];
}

PnrIndex
SpatialRouteResourceState::logicalNetRouteClaimRefcount(PnrIndex logicalNet,
                                                        PnrIndex claim) const {
  assert(logicalNet < logicalNetCount_ && claim < routeClaimCount_);
  const auto found = netClaimRefcounts_[logicalNet].find(claim);
  return found == netClaimRefcounts_[logicalNet].end() ? 0 : found->second;
}

llvm::ArrayRef<std::uint64_t>
SpatialRouteResourceState::logicalNetRouteClaimBits(PnrIndex logicalNet) const {
  assert(logicalNet < logicalNetCount_);
  return llvm::ArrayRef(netClaimActiveBits_)
      .slice(static_cast<std::size_t>(logicalNet) * routeClaimWordCount_,
             routeClaimWordCount_);
}

std::uint64_t
SpatialRouteResourceState::capacityUsageRaw(PnrIndex capacityDimension) const {
  assert(capacityDimension < capacityUsageRaw_.size());
  return capacityUsageRaw_[capacityDimension];
}

std::uint64_t SpatialRouteResourceState::capacityOveruseRaw(
    PnrIndex capacityDimension) const {
  assert(capacityDimension < capacityUsageRaw_.size());
  const std::uint64_t capacity =
      problem_->resources().capacityDimensions()[capacityDimension].capacity;
  return capacityUsageRaw_[capacityDimension] > capacity
             ? capacityUsageRaw_[capacityDimension] - capacity
             : 0;
}

std::size_t SpatialRouteResourceState::retainedStorageBytes() const {
  return retainedBytes(initiationIntervals_) +
         retainedBytes(traversalIntervalOrdinals_) +
         retainedBytes(activeInitiationIntervalCounts_) +
         retainedSparseRefcountBytes(netTraversalRefcounts_) +
         retainedSparseRefcountBytes(netClaimRefcounts_) +
         retainedBytes(netClaimActiveBits_) +
         retainedBytes(claimSelectionCounts_) +
         retainedBytes(capacityUsageRaw_);
}

llvm::Error SpatialRouteResourceState::applyClaimDelta(PnrIndex logicalNet,
                                                       PnrIndex claim,
                                                       PnrIndex removed,
                                                       PnrIndex added) {
  if (logicalNet >= logicalNetCount_ || claim >= routeClaimCount_)
    return routeResourceError("claim delta index is out of range");
  auto &refcounts = netClaimRefcounts_[logicalNet];
  const auto found = refcounts.find(claim);
  const PnrIndex refcount = found == refcounts.end() ? 0 : found->second;
  if (removed > refcount)
    return routeResourceError("claim refcount removal underflows");
  const PnrIndex remaining = refcount - removed;
  if (added > std::numeric_limits<PnrIndex>::max() - remaining)
    return routeResourceError("claim refcount addition overflows PnrIndex");
  const PnrIndex next = remaining + added;
  const bool activate = refcount == 0 && next != 0;
  const bool deactivate = refcount != 0 && next == 0;
  if (!activate && !deactivate) {
    if (next != 0)
      found->second = next;
    return llvm::Error::success();
  }

  const FrozenSpatialRouteClaim &record =
      problem_->routing().routeClaims()[claim];
  if (record.capacityDimension >= capacityUsageRaw_.size())
    return routeResourceError("claim capacity dimension is out of range");
  PnrIndex &selectionCount = claimSelectionCounts_[claim];
  std::uint64_t &usage = capacityUsageRaw_[record.capacityDimension];
  std::uint64_t &activeWord =
      netClaimActiveBits_[static_cast<std::size_t>(logicalNet) *
                              routeClaimWordCount_ +
                          claim / 64];
  const std::uint64_t activeMask = std::uint64_t{1} << (claim % 64);
  const std::uint64_t capacity =
      problem_->resources()
          .capacityDimensions()[record.capacityDimension]
          .capacity;
  const std::uint64_t oldOveruse = usage > capacity ? usage - capacity : 0;
  std::uint64_t nextUsage = usage;
  if (activate) {
    if (selectionCount == std::numeric_limits<PnrIndex>::max())
      return routeResourceError("claim selection count overflows PnrIndex");
    if (record.amount > std::numeric_limits<std::uint64_t>::max() - usage)
      return routeResourceError("raw route capacity usage overflows u64");
    if (record.qCost > std::numeric_limits<std::uint64_t>::max() -
                           totalSelectedTraversalClaim_)
      return routeResourceError("selected traversal claim cost overflows u64");
    nextUsage += record.amount;
  } else {
    if (selectionCount == 0 || usage < record.amount ||
        totalSelectedTraversalClaim_ < record.qCost)
      return routeResourceError("claim deactivation underflows exact state");
    nextUsage -= record.amount;
  }
  if (oldOveruse > totalCapacityOveruseRaw_)
    return routeResourceError("raw route capacity overuse underflows");
  std::uint64_t nextTotalOveruse = totalCapacityOveruseRaw_ - oldOveruse;
  const std::uint64_t newOveruse =
      nextUsage > capacity ? nextUsage - capacity : 0;
  if (llvm::Error error = checkedAdd(nextTotalOveruse, newOveruse,
                                     "raw route capacity overuse"))
    return error;

  if (activate) {
    ++selectionCount;
    totalSelectedTraversalClaim_ += record.qCost;
    activeWord |= activeMask;
  } else {
    --selectionCount;
    totalSelectedTraversalClaim_ -= record.qCost;
    activeWord &= ~activeMask;
  }
  usage = nextUsage;
  totalCapacityOveruseRaw_ = nextTotalOveruse;
  if (next == 0)
    refcounts.erase(found);
  else
    refcounts.try_emplace(claim, next);
  return llvm::Error::success();
}

llvm::Error SpatialRouteResourceState::applyTraversalDelta(PnrIndex logicalNet,
                                                           PnrIndex traversal,
                                                           PnrIndex removed,
                                                           PnrIndex added) {
  if (logicalNet >= logicalNetCount_ || traversal >= traversalCount_)
    return routeResourceError("traversal delta index is out of range");
  const FrozenSpatialTraversal &record =
      problem_->routing().traversals()[traversal];
  if (record.minimumInitiationIntervalCycles == 0)
    return routeResourceError("traversal has a zero initiation interval");
  auto &refcounts = netTraversalRefcounts_[logicalNet];
  const auto found = refcounts.find(traversal);
  const PnrIndex refcount = found == refcounts.end() ? 0 : found->second;
  if (removed > refcount)
    return routeResourceError("traversal refcount removal underflows");
  const PnrIndex remaining = refcount - removed;
  if (added > std::numeric_limits<PnrIndex>::max() - remaining)
    return routeResourceError("traversal refcount addition overflows PnrIndex");
  const PnrIndex next = remaining + added;
  const bool activate = refcount == 0 && next != 0;
  const bool deactivate = refcount != 0 && next == 0;

  const PnrIndex intervalOrdinal = traversalIntervalOrdinals_[traversal];
  if (intervalOrdinal >= activeInitiationIntervalCounts_.size() ||
      initiationIntervals_[intervalOrdinal] !=
          record.minimumInitiationIntervalCycles)
    return routeResourceError("traversal initiation interval is inconsistent");
  const std::uint64_t intervalCount =
      activeInitiationIntervalCounts_[intervalOrdinal];
  std::uint64_t nextIntervalCount = intervalCount;
  if (activate) {
    if (nextIntervalCount == std::numeric_limits<std::uint64_t>::max())
      return routeResourceError(
          "active initiation interval count overflows u64");
    ++nextIntervalCount;
  } else if (deactivate) {
    if (nextIntervalCount == 0)
      return routeResourceError(
          "active initiation interval count underflows u64");
    --nextIntervalCount;
  }

  std::uint64_t nextRelease = routeReleaseLatencyCycles_;
  std::uint64_t nextInterval = routeMinimumInitiationIntervalCycles_;
  std::uint64_t nextBitCycles = transportBitCycleDemand_;
  auto bitCycles = checkedMultiply(
      problem_->transfers().logicalNets()[logicalNet].payloadWidthBits,
      record.minimumInitiationIntervalCycles, "transport bit-cycle demand");
  if (!bitCycles)
    return bitCycles.takeError();
  if (activate) {
    if (llvm::Error error = checkedAdd(nextRelease, record.releaseLatencyCycles,
                                       "route release latency"))
      return error;
    nextInterval = std::max(
        nextInterval,
        static_cast<std::uint64_t>(record.minimumInitiationIntervalCycles));
    if (llvm::Error error =
            checkedAdd(nextBitCycles, *bitCycles, "transport bit-cycle demand"))
      return error;
  } else if (deactivate) {
    if (nextRelease < record.releaseLatencyCycles || nextBitCycles < *bitCycles)
      return routeResourceError("traversal timing deactivation underflows");
    nextRelease -= record.releaseLatencyCycles;
    nextBitCycles -= *bitCycles;
    if (record.minimumInitiationIntervalCycles == nextInterval &&
        nextIntervalCount == 0) {
      nextInterval = 1;
      for (std::size_t ordinal = intervalOrdinal; ordinal != 0; --ordinal)
        if (activeInitiationIntervalCounts_[ordinal - 1] != 0) {
          nextInterval = initiationIntervals_[ordinal - 1];
          break;
        }
    }
  }
  const auto claims = problem_->routing().traversalClaimKeys().slice(
      record.routeClaimOffset, record.routeClaimCount);
  for (std::size_t index = 0; index < claims.size(); ++index) {
    if (llvm::Error error =
            applyClaimDelta(logicalNet, claims[index], removed, added)) {
      for (std::size_t undo = index; undo != 0; --undo)
        llvm::cantFail(
            applyClaimDelta(logicalNet, claims[undo - 1], added, removed));
      return error;
    }
  }
  if (next == 0)
    refcounts.erase(found);
  else if (found == refcounts.end())
    refcounts.try_emplace(traversal, next);
  else
    found->second = next;
  activeInitiationIntervalCounts_[intervalOrdinal] = nextIntervalCount;
  routeReleaseLatencyCycles_ = nextRelease;
  routeMinimumInitiationIntervalCycles_ = nextInterval;
  transportBitCycleDemand_ = nextBitCycles;
  return llvm::Error::success();
}

void SpatialRouteResourceState::revertTraversalDelta(PnrIndex logicalNet,
                                                     PnrIndex traversal,
                                                     PnrIndex removed,
                                                     PnrIndex added) noexcept {
  llvm::cantFail(applyTraversalDelta(logicalNet, traversal, added, removed));
}

llvm::Error SpatialRouteResourceState::verify(
    llvm::ArrayRef<RouteTreeStateHandle> routeTrees,
    llvm::ArrayRef<PnrIndex> registerFifoTransfers) const {
  if (!problem_ || routeTrees.size() != logicalNetCount_ ||
      traversalCount_ != problem_->routing().traversals().size() ||
      claimSelectionCounts_.size() != routeClaimCount_ ||
      capacityUsageRaw_.size() !=
          problem_->resources().capacityDimensions().size() ||
      netClaimRefcounts_.size() != logicalNetCount_ ||
      netTraversalRefcounts_.size() != logicalNetCount_ ||
      routeClaimWordCount_ !=
          (static_cast<std::size_t>(routeClaimCount_) + 63) / 64 ||
      traversalIntervalOrdinals_.size() != traversalCount_ ||
      activeInitiationIntervalCounts_.size() != initiationIntervals_.size() ||
      netClaimActiveBits_.size() !=
          static_cast<std::size_t>(logicalNetCount_) * routeClaimWordCount_)
    return routeResourceError("state dimensions disagree with the freeze");

  std::vector<const RouteTreeState *> rawRoutes;
  rawRoutes.reserve(routeTrees.size());
  for (const RouteTreeStateHandle &route : routeTrees)
    rawRoutes.push_back(route.get());
  auto expected =
      projectVerifiedRoutes(*problem_, rawRoutes, registerFifoTransfers);
  if (!expected)
    return expected.takeError();

  if (!sparseRefcountsEqual(netTraversalRefcounts_,
                            expected->netTraversalRefcounts_) ||
      initiationIntervals_ != expected->initiationIntervals_ ||
      traversalIntervalOrdinals_ != expected->traversalIntervalOrdinals_ ||
      activeInitiationIntervalCounts_ !=
          expected->activeInitiationIntervalCounts_ ||
      !sparseRefcountsEqual(netClaimRefcounts_, expected->netClaimRefcounts_) ||
      netClaimActiveBits_ != expected->netClaimActiveBits_ ||
      claimSelectionCounts_ != expected->claimSelectionCounts_ ||
      capacityUsageRaw_ != expected->capacityUsageRaw_ ||
      totalCapacityOveruseRaw_ != expected->totalCapacityOveruseRaw_ ||
      totalSelectedTraversalClaim_ != expected->totalSelectedTraversalClaim_ ||
      routeReleaseLatencyCycles_ != expected->routeReleaseLatencyCycles_ ||
      routeMinimumInitiationIntervalCycles_ !=
          expected->routeMinimumInitiationIntervalCycles_ ||
      transportBitCycleDemand_ != expected->transportBitCycleDemand_)
    return routeResourceError(
        "incremental occupancy disagrees with the selected RouteTrees");
  return llvm::Error::success();
}

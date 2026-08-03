#include "PnR/SpatialRouteResourceState.h"

#include "PnR/PnrIndex.h"

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
constexpr PnrCapacityContext cellCountContext{
    candidateArtifact, "route_resource_state", "net_claim_refcounts",
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

template <typename T> std::size_t retainedBytes(const std::vector<T> &values) {
  return values.capacity() * sizeof(T);
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
  auto cellCount = checkedPnrIndexMultiply(cellCountContext, *logicalNetCount,
                                           *routeClaimCount);
  if (!cellCount)
    return cellCount.takeError();
  if (static_cast<std::uint64_t>(*cellCount) >
      std::numeric_limits<std::size_t>::max() / sizeof(PnrIndex))
    return routeResourceError("net-claim matrix exceeds native size_t");
  const std::size_t routeClaimWordCount =
      (static_cast<std::size_t>(*routeClaimCount) + 63) / 64;
  if (routeClaimWordCount != 0 &&
      static_cast<std::size_t>(*logicalNetCount) >
          std::numeric_limits<std::size_t>::max() / routeClaimWordCount)
    return routeResourceError("net-claim bitset exceeds native size_t");

  std::vector<PnrIndex> netClaimRefcounts(static_cast<std::size_t>(*cellCount),
                                          0);
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
      problem, *logicalNetCount, *routeClaimCount, routeClaimWordCount,
      std::move(netClaimRefcounts), std::move(netClaimActiveBits),
      std::move(claimSelectionCounts), std::move(capacityUsageRaw),
      totalCapacityOveruseRaw);
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
  return netClaimRefcounts_[netClaimCell(logicalNet, claim)];
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
  return retainedBytes(netClaimRefcounts_) +
         retainedBytes(netClaimActiveBits_) +
         retainedBytes(claimSelectionCounts_) +
         retainedBytes(capacityUsageRaw_);
}

std::size_t SpatialRouteResourceState::netClaimCell(PnrIndex logicalNet,
                                                    PnrIndex claim) const {
  return static_cast<std::size_t>(logicalNet) * routeClaimCount_ + claim;
}

llvm::Error SpatialRouteResourceState::applyClaimDelta(PnrIndex logicalNet,
                                                       PnrIndex claim,
                                                       PnrIndex removed,
                                                       PnrIndex added) {
  if (logicalNet >= logicalNetCount_ || claim >= routeClaimCount_)
    return routeResourceError("claim delta index is out of range");
  PnrIndex &refcount = netClaimRefcounts_[netClaimCell(logicalNet, claim)];
  if (removed > refcount)
    return routeResourceError("claim refcount removal underflows");
  const PnrIndex remaining = refcount - removed;
  if (added > std::numeric_limits<PnrIndex>::max() - remaining)
    return routeResourceError("claim refcount addition overflows PnrIndex");
  const PnrIndex next = remaining + added;
  const bool activate = refcount == 0 && next != 0;
  const bool deactivate = refcount != 0 && next == 0;
  if (!activate && !deactivate) {
    refcount = next;
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
  refcount = next;
  return llvm::Error::success();
}

llvm::Error SpatialRouteResourceState::applyTraversalDelta(PnrIndex logicalNet,
                                                           PnrIndex traversal,
                                                           PnrIndex removed,
                                                           PnrIndex added) {
  if (logicalNet >= logicalNetCount_ ||
      traversal >= problem_->routing().traversals().size())
    return routeResourceError("traversal delta index is out of range");
  const FrozenSpatialTraversal &record =
      problem_->routing().traversals()[traversal];
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
  return llvm::Error::success();
}

void SpatialRouteResourceState::revertTraversalDelta(PnrIndex logicalNet,
                                                     PnrIndex traversal,
                                                     PnrIndex removed,
                                                     PnrIndex added) noexcept {
  llvm::cantFail(applyTraversalDelta(logicalNet, traversal, added, removed));
}

llvm::Error SpatialRouteResourceState::verify(
    llvm::ArrayRef<RouteTreeStateHandle> routeTrees) const {
  if (!problem_ || routeTrees.size() != logicalNetCount_ ||
      claimSelectionCounts_.size() != routeClaimCount_ ||
      capacityUsageRaw_.size() !=
          problem_->resources().capacityDimensions().size() ||
      netClaimRefcounts_.size() !=
          static_cast<std::size_t>(logicalNetCount_) * routeClaimCount_ ||
      routeClaimWordCount_ !=
          (static_cast<std::size_t>(routeClaimCount_) + 63) / 64 ||
      netClaimActiveBits_.size() !=
          static_cast<std::size_t>(logicalNetCount_) * routeClaimWordCount_)
    return routeResourceError("state dimensions disagree with the freeze");

  std::vector<PnrIndex> expectedRefcounts(netClaimRefcounts_.size(), 0);
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
    if (!routeTrees[logicalNet])
      return routeResourceError("logical net has no RouteTree state");
    for (const RouteTreeNode &node : routeTrees[logicalNet]->nodeStorage()) {
      if (!node.isActive() || node.parentArc == getInvalidPnrIndex())
        continue;
      if (node.parentArc >= problem_->routing().routingArcs().size())
        return routeResourceError("RouteTree parent arc is out of range");
      const PnrIndex traversal =
          problem_->routing().routingArcs()[node.parentArc].traversal;
      if (traversal >= problem_->routing().traversals().size())
        return routeResourceError("RouteTree traversal is out of range");
      const FrozenSpatialTraversal &traversalRecord =
          problem_->routing().traversals()[traversal];
      for (PnrIndex claim : problem_->routing().traversalClaimKeys().slice(
               traversalRecord.routeClaimOffset,
               traversalRecord.routeClaimCount)) {
        PnrIndex &refcount = expectedRefcounts[netClaimCell(logicalNet, claim)];
        if (refcount == std::numeric_limits<PnrIndex>::max())
          return routeResourceError(
              "rebuilt route claim refcount overflows PnrIndex");
        ++refcount;
      }
    }
  }

  std::vector<PnrIndex> expectedSelections(routeClaimCount_, 0);
  std::vector<std::uint64_t> expectedActiveBits(netClaimActiveBits_.size(), 0);
  std::vector<std::uint64_t> expectedUsage;
  expectedUsage.reserve(problem_->resources().capacityDimensions().size());
  for (const FrozenSpatialCapacityDimension &dimension :
       problem_->resources().capacityDimensions())
    expectedUsage.push_back(dimension.initialOccupancy);
  std::uint64_t expectedTotal = 0;
  for (PnrIndex logicalNet = 0; logicalNet < logicalNetCount_; ++logicalNet) {
    for (PnrIndex claim = 0; claim < routeClaimCount_; ++claim) {
      if (expectedRefcounts[netClaimCell(logicalNet, claim)] == 0)
        continue;
      expectedActiveBits[static_cast<std::size_t>(logicalNet) *
                             routeClaimWordCount_ +
                         claim / 64] |= std::uint64_t{1} << (claim % 64);
      if (expectedSelections[claim] == std::numeric_limits<PnrIndex>::max())
        return routeResourceError(
            "rebuilt route claim selection count overflows PnrIndex");
      ++expectedSelections[claim];
      const FrozenSpatialRouteClaim &record =
          problem_->routing().routeClaims()[claim];
      if (llvm::Error error =
              checkedAdd(expectedUsage[record.capacityDimension], record.amount,
                         "rebuilt raw route capacity usage"))
        return error;
      if (llvm::Error error =
              checkedAdd(expectedTotal, record.qCost,
                         "rebuilt selected traversal claim cost"))
        return error;
    }
  }

  std::uint64_t expectedCapacityOveruse = 0;
  const auto dimensions = problem_->resources().capacityDimensions();
  for (PnrIndex capacity = 0; capacity < dimensions.size(); ++capacity) {
    const std::uint64_t overuse =
        expectedUsage[capacity] > dimensions[capacity].capacity
            ? expectedUsage[capacity] - dimensions[capacity].capacity
            : 0;
    if (llvm::Error error = checkedAdd(expectedCapacityOveruse, overuse,
                                       "rebuilt raw route capacity overuse"))
      return error;
  }

  if (netClaimRefcounts_ != expectedRefcounts ||
      netClaimActiveBits_ != expectedActiveBits ||
      claimSelectionCounts_ != expectedSelections ||
      capacityUsageRaw_ != expectedUsage ||
      totalCapacityOveruseRaw_ != expectedCapacityOveruse ||
      totalSelectedTraversalClaim_ != expectedTotal)
    return routeResourceError(
        "incremental occupancy disagrees with the selected RouteTrees");
  return llvm::Error::success();
}

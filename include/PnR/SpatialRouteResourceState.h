#ifndef LOOM_PNR_SPATIALROUTERESOURCESTATE_H
#define LOOM_PNR_SPATIALROUTERESOURCESTATE_H

#include "PnR/RouteTreeState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom::pnr {

class SpatialCandidateState;
class SpatialMoveTransaction;

/// Worker-local exact occupancy induced by selected RouteTrees. One dense
/// net-by-claim refcount matrix normalizes shared prefixes and owner-defined
/// requester groups without persistent-reference work in the move loop. Raw
/// capacity integers and Q-scaled search cost remain separate.
class SpatialRouteResourceState final {
public:
  static llvm::Expected<SpatialRouteResourceState>
  create(const FrozenSpatialPnrProblem &problem);

  std::uint64_t totalSelectedTraversalClaim() const {
    return totalSelectedTraversalClaim_;
  }
  PnrIndex routeClaimSelectionCount(PnrIndex claim) const;
  PnrIndex logicalNetRouteClaimRefcount(PnrIndex logicalNet,
                                        PnrIndex claim) const;
  llvm::ArrayRef<std::uint64_t>
  logicalNetRouteClaimBits(PnrIndex logicalNet) const;
  std::uint64_t capacityUsageRaw(PnrIndex capacityDimension) const;
  std::uint64_t capacityOveruseRaw(PnrIndex capacityDimension) const;
  std::uint64_t totalCapacityOveruseRaw() const {
    return totalCapacityOveruseRaw_;
  }
  std::uint64_t routeReleaseLatencyCycles() const {
    return routeReleaseLatencyCycles_;
  }
  std::uint64_t routeMinimumInitiationIntervalCycles() const {
    return routeMinimumInitiationIntervalCycles_;
  }
  std::uint64_t transportBitCycleDemand() const {
    return transportBitCycleDemand_;
  }
  std::size_t retainedStorageBytes() const;

  llvm::Error verify(llvm::ArrayRef<RouteTreeStateHandle> routeTrees,
                     llvm::ArrayRef<PnrIndex> registerFifoTransfers) const;

private:
  static llvm::Expected<SpatialRouteResourceState>
  projectVerifiedRoutes(const FrozenSpatialPnrProblem &problem,
                        llvm::ArrayRef<const RouteTreeState *> routeTrees,
                        llvm::ArrayRef<PnrIndex> registerFifoTransfers);

  SpatialRouteResourceState(
      const FrozenSpatialPnrProblem &problem, PnrIndex logicalNetCount,
      PnrIndex traversalCount, PnrIndex routeClaimCount,
      std::size_t routeClaimWordCount,
      std::vector<std::uint32_t> initiationIntervals,
      std::vector<PnrIndex> traversalIntervalOrdinals,
      std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> netTraversalRefcounts,
      std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> netClaimRefcounts,
      std::vector<std::uint64_t> netClaimActiveBits,
      std::vector<PnrIndex> claimSelectionCounts,
      std::vector<std::uint64_t> capacityUsageRaw,
      std::uint64_t totalCapacityOveruseRaw)
      : problem_(&problem), logicalNetCount_(logicalNetCount),
        traversalCount_(traversalCount), routeClaimCount_(routeClaimCount),
        routeClaimWordCount_(routeClaimWordCount),
        initiationIntervals_(std::move(initiationIntervals)),
        traversalIntervalOrdinals_(std::move(traversalIntervalOrdinals)),
        activeInitiationIntervalCounts_(initiationIntervals_.size(), 0),
        netTraversalRefcounts_(std::move(netTraversalRefcounts)),
        netClaimRefcounts_(std::move(netClaimRefcounts)),
        netClaimActiveBits_(std::move(netClaimActiveBits)),
        claimSelectionCounts_(std::move(claimSelectionCounts)),
        capacityUsageRaw_(std::move(capacityUsageRaw)),
        totalCapacityOveruseRaw_(totalCapacityOveruseRaw) {}

  llvm::Error applyTraversalDelta(PnrIndex logicalNet, PnrIndex traversal,
                                  PnrIndex removed, PnrIndex added);
  llvm::Error applyClaimDelta(PnrIndex logicalNet, PnrIndex claim,
                              PnrIndex removed, PnrIndex added);
  void revertTraversalDelta(PnrIndex logicalNet, PnrIndex traversal,
                            PnrIndex removed, PnrIndex added) noexcept;

  const FrozenSpatialPnrProblem *problem_ = nullptr;
  PnrIndex logicalNetCount_ = 0;
  PnrIndex traversalCount_ = 0;
  PnrIndex routeClaimCount_ = 0;
  std::size_t routeClaimWordCount_ = 0;
  std::vector<std::uint32_t> initiationIntervals_;
  std::vector<PnrIndex> traversalIntervalOrdinals_;
  std::vector<std::uint64_t> activeInitiationIntervalCounts_;
  std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> netTraversalRefcounts_;
  std::vector<llvm::DenseMap<PnrIndex, PnrIndex>> netClaimRefcounts_;
  std::vector<std::uint64_t> netClaimActiveBits_;
  std::vector<PnrIndex> claimSelectionCounts_;
  std::vector<std::uint64_t> capacityUsageRaw_;
  std::uint64_t totalCapacityOveruseRaw_ = 0;
  std::uint64_t totalSelectedTraversalClaim_ = 0;
  std::uint64_t routeReleaseLatencyCycles_ = 0;
  std::uint64_t routeMinimumInitiationIntervalCycles_ = 1;
  std::uint64_t transportBitCycleDemand_ = 0;

  friend class SpatialCandidateState;
  friend class SpatialMoveTransaction;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALROUTERESOURCESTATE_H

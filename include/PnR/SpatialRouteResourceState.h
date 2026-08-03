#ifndef LOOM_PNR_SPATIALROUTERESOURCESTATE_H
#define LOOM_PNR_SPATIALROUTERESOURCESTATE_H

#include "PnR/RouteTreeState.h"
#include "PnR/SpatialPnrProblem.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom::pnr {

class SpatialMoveTransaction;

/// Worker-local exact occupancy induced by selected RouteTrees. One dense
/// net-by-claim refcount matrix normalizes shared prefixes and owner-defined
/// atomic activation groups without persistent-reference work in the move
/// loop. Raw capacity integers and Q32 search cost remain separate.
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
  std::uint64_t capacityUsageRaw(PnrIndex capacityDimension) const;
  std::uint64_t capacityOveruseRaw(PnrIndex capacityDimension) const;
  std::size_t retainedStorageBytes() const;

  llvm::Error verify(llvm::ArrayRef<RouteTreeStateHandle> routeTrees) const;

private:
  SpatialRouteResourceState(const FrozenSpatialPnrProblem &problem,
                            PnrIndex logicalNetCount, PnrIndex routeClaimCount,
                            std::vector<PnrIndex> netClaimRefcounts,
                            std::vector<PnrIndex> claimSelectionCounts,
                            std::vector<std::uint64_t> capacityUsageRaw)
      : problem_(&problem), logicalNetCount_(logicalNetCount),
        routeClaimCount_(routeClaimCount),
        netClaimRefcounts_(std::move(netClaimRefcounts)),
        claimSelectionCounts_(std::move(claimSelectionCounts)),
        capacityUsageRaw_(std::move(capacityUsageRaw)) {}

  llvm::Error applyTraversalDelta(PnrIndex logicalNet, PnrIndex traversal,
                                  PnrIndex removed, PnrIndex added);
  llvm::Error applyClaimDelta(PnrIndex logicalNet, PnrIndex claim,
                              PnrIndex removed, PnrIndex added);
  void revertTraversalDelta(PnrIndex logicalNet, PnrIndex traversal,
                            PnrIndex removed, PnrIndex added) noexcept;
  std::size_t netClaimCell(PnrIndex logicalNet, PnrIndex claim) const;

  const FrozenSpatialPnrProblem *problem_ = nullptr;
  PnrIndex logicalNetCount_ = 0;
  PnrIndex routeClaimCount_ = 0;
  std::vector<PnrIndex> netClaimRefcounts_;
  std::vector<PnrIndex> claimSelectionCounts_;
  std::vector<std::uint64_t> capacityUsageRaw_;
  std::uint64_t totalSelectedTraversalClaim_ = 0;

  friend class SpatialMoveTransaction;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALROUTERESOURCESTATE_H

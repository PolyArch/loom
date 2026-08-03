#ifndef LOOM_PNR_SPATIALNETROUTER_H
#define LOOM_PNR_SPATIALNETROUTER_H

#include "PnR/EndpointRouter.h"
#include "PnR/SpatialRouteCostState.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace loom::pnr {

/// Worker-local whole-net routing scratch. The selected logical net must
/// already be excluded from the supplied cost state. RouteTree and resource
/// mutations remain owned by the supplied SpatialMoveTransaction. After an
/// error, the caller rolls back that transaction and deselects the logical net
/// to restore the cost overlay.
class SpatialNetRouterScratch final {
public:
  SpatialNetRouterScratch() = default;
  SpatialNetRouterScratch(const SpatialNetRouterScratch &) = delete;
  SpatialNetRouterScratch &operator=(const SpatialNetRouterScratch &) = delete;
  SpatialNetRouterScratch(SpatialNetRouterScratch &&) = delete;
  SpatialNetRouterScratch &operator=(SpatialNetRouterScratch &&) = delete;
  ~SpatialNetRouterScratch() = default;

  llvm::Error prepare(const FrozenSpatialPnrProblem &problem);

  llvm::Expected<RouteCost>
  routeWholeNet(SpatialMoveTransaction &move,
                const SpatialCandidateState &candidate,
                SpatialRouteCostState &costs, PnrIndex logicalNet,
                std::uint64_t endpointExpansionLimit);

  std::size_t retainedStorageBytes() const;

private:
  struct SourceCandidate final {
    PnrIndex endpoint = 0;
    PnrIndex replicationGroup = getInvalidPnrIndex();
  };

  struct TargetCandidate final {
    PnrIndex endpoint = 0;
    PnrIndex sinkObligation = 0;
  };

  llvm::Error collectSourceFrontier(const RouteTreeState &tree,
                                    PnrIndex unroutedSource);
  llvm::Error collectTargetFrontier(const SpatialCandidateState &candidate,
                                    PnrIndex logicalNet, PnrIndex sinkCount);
  llvm::Error addPathClaims(const FrozenSpatialRoutingGraph &routing,
                            llvm::ArrayRef<PnrIndex> forwardArcs);

  EndpointRouteSearchScratch endpointSearch_;
  std::vector<SourceCandidate> sourceCandidates_;
  std::vector<PnrIndex> sourceEndpoints_;
  std::vector<PnrIndex> sourceReplicationGroups_;
  std::vector<TargetCandidate> targetCandidates_;
  std::vector<PnrIndex> targetEndpoints_;
  std::vector<PnrIndex> targetPreferenceRanks_;
  std::vector<PnrIndex> targetObligationByEndpoint_;
  std::vector<std::uint8_t> unresolvedSinks_;
  std::vector<std::uint64_t> prospectiveClaimBits_;
  const FrozenSpatialPnrProblem *preparedProblem_ = nullptr;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALNETROUTER_H

#ifndef LOOM_PNR_SPATIALNETROUTER_H
#define LOOM_PNR_SPATIALNETROUTER_H

#include "PnR/EndpointRouter.h"
#include "PnR/SpatialRouteCostState.h"
#include "PnR/SpatialTagContinuity.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace loom::pnr {

namespace detail {
class SpatialNetRouterPrivate;

enum class SpatialNegotiatedRouteScope : std::uint8_t {
  Preserve,
  SelectedSinks,
  WholeNet,
};

struct SpatialNegotiatedRoutePlan final {
  SpatialNegotiatedRouteScope scope = SpatialNegotiatedRouteScope::WholeNet;
  llvm::ArrayRef<PnrIndex> sinkObligations;
};
} // namespace detail

/// Worker-local whole-net routing scratch. The selected logical net must
/// already be excluded from the supplied cost state. RouteTree and resource
/// mutations remain owned by the supplied SpatialMoveTransaction. After an
/// error, the caller rolls back that transaction and deselects the logical net
/// to restore the cost overlay.
class SpatialNetRouterScratch final {
public:
  SpatialNetRouterScratch();
  SpatialNetRouterScratch(const SpatialNetRouterScratch &) = delete;
  SpatialNetRouterScratch &operator=(const SpatialNetRouterScratch &) = delete;
  SpatialNetRouterScratch(SpatialNetRouterScratch &&) = delete;
  SpatialNetRouterScratch &operator=(SpatialNetRouterScratch &&) = delete;
  ~SpatialNetRouterScratch();

  llvm::Error prepare(const FrozenSpatialPnrProblem &problem,
                      SpatialPnrWorkLedgerView workLedger = {});
  llvm::Error beginConstraintSweep(llvm::ArrayRef<PnrIndex> logicalNets);
  llvm::Error finishConstraintNet(PnrIndex logicalNet);

  llvm::Expected<detail::SpatialNegotiatedRoutePlan>
  planNegotiatedRoute(const SpatialCandidateState &candidate,
                      const SpatialRouteCostState &costs, PnrIndex logicalNet);

  llvm::Expected<RouteCost>
  routeWholeNet(SpatialMoveTransaction &move,
                const SpatialCandidateState &candidate,
                SpatialRouteCostState &costs, PnrIndex logicalNet,
                std::uint64_t endpointExpansionLimit);
  llvm::Expected<RouteCost> routeSingleSink(
      SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
      SpatialRouteCostState &costs, PnrIndex logicalNet,
      PnrIndex sinkObligation, std::uint64_t endpointExpansionLimit);
  llvm::Expected<RouteCost> routeRootedSubtree(
      SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
      SpatialRouteCostState &costs, PnrIndex logicalNet, PnrIndex rootEndpoint,
      std::uint64_t endpointExpansionLimit);
  llvm::Expected<RouteCost>
  routeSinkSet(SpatialMoveTransaction &move,
               const SpatialCandidateState &candidate,
               SpatialRouteCostState &costs, PnrIndex logicalNet,
               llvm::ArrayRef<PnrIndex> sinkObligations,
               std::uint64_t endpointExpansionLimit);

  std::uint64_t endpointExpansionCount() const {
    return endpointSearch_.endpointExpansionCount();
  }
  std::uint64_t heuristicCacheHitCount() const {
    return endpointSearch_.heuristicCacheHitCount();
  }
  std::uint64_t heuristicBuildCount() const {
    return endpointSearch_.heuristicBuildCount();
  }
  std::uint64_t forwardHeuristicQueryCount() const {
    return endpointSearch_.forwardHeuristicQueryCount();
  }
  std::uint64_t forwardHeuristicUnreachableCount() const {
    return endpointSearch_.forwardHeuristicUnreachableCount();
  }
  std::uint64_t heuristicCacheEvictionCount() const {
    return endpointSearch_.heuristicCacheEvictionCount();
  }
  std::size_t heuristicCacheEntryCount() const {
    return endpointSearch_.heuristicCacheEntryCount();
  }
  std::size_t heuristicCacheRetainedBytes() const {
    return endpointSearch_.heuristicCacheRetainedBytes();
  }
  std::size_t retainedStorageBytes() const;

private:
  struct SourceCandidate final {
    PnrIndex endpoint = 0;
    PnrIndex replicationGroup = getInvalidPnrIndex();
  };

  struct TargetCandidate final {
    PnrIndex endpoint = 0;
    PnrIndex sinkObligation = 0;
    bool requiresTraversal = false;
  };

  llvm::Error collectSourceFrontier(const RouteTreeState &tree,
                                    PnrIndex unroutedSource);
  llvm::Error collectTargetFrontier(const SpatialCandidateState &candidate,
                                    PnrIndex logicalNet, PnrIndex sinkCount);
  llvm::Error addPathClaims(const FrozenSpatialRoutingGraph &routing,
                            llvm::ArrayRef<PnrIndex> forwardArcs);
  llvm::Error collectCurrentClaims(const RouteTreeState &tree);
  llvm::Error updateCurrentTagUses(const RouteTreeState &tree,
                                   SpatialRouteCostState &costs);
  llvm::Expected<RouteCost>
  routeSelectedSinks(SpatialMoveTransaction &move,
                     const SpatialCandidateState &candidate,
                     SpatialRouteCostState &costs, PnrIndex logicalNet,
                     std::uint64_t endpointExpansionLimit);
  void beginEndpointMarks();

  EndpointRouteSearchScratch endpointSearch_;
  std::vector<SourceCandidate> sourceCandidates_;
  std::vector<PnrIndex> sourceEndpoints_;
  std::vector<PnrIndex> sourceReplicationGroups_;
  std::vector<std::uint64_t> sourceTimingArrivalQuanta_;
  std::vector<TargetCandidate> targetCandidates_;
  std::vector<PnrIndex> targetEndpoints_;
  std::vector<PnrIndex> targetPreferenceRanks_;
  std::vector<std::uint8_t> targetRequiresTraversal_;
  std::vector<std::uint64_t> targetTimingDelayQuanta_;
  std::vector<PnrIndex> targetObligationByEndpoint_;
  std::vector<std::uint8_t> unresolvedSinks_;
  std::vector<std::uint64_t> prospectiveClaimBits_;
  std::vector<std::uint64_t> bufferedTraversalBits_;
  std::vector<std::uint64_t> arcTimingDelayQuanta_;
  std::vector<std::uint8_t> arcTimingRegisteredDestination_;
  EndpointRouteInputRevisionOwner physicalTimingRevisionOwner_;
  std::vector<std::uint64_t> routeNodeTimingArrivals_;
  std::vector<std::pair<PnrIndex, std::uint64_t>> routeNodeTimingWorklist_;
  std::vector<std::uint64_t> endpointMarks_;
  std::vector<PnrIndex> subtreeWorklist_;
  SpatialTagContinuityProjection tagContinuity_;
  SpatialTagContinuityScratch tagContinuityScratch_;
  std::uint64_t endpointMarkEpoch_ = 0;
  std::unique_ptr<detail::SpatialNetRouterPrivate> private_;
  const FrozenSpatialPnrProblem *preparedProblem_ = nullptr;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALNETROUTER_H

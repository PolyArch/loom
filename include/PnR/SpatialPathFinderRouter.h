#ifndef LOOM_PNR_SPATIALPATHFINDERROUTER_H
#define LOOM_PNR_SPATIALPATHFINDERROUTER_H

#include "PnR/SpatialNetRouter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::dse {
class ObjectiveVector;
}

namespace loom::pnr {

struct SpatialPathFinderRoutingLimits final {
  std::uint64_t endpointExpansionLimit = 0;
  std::uint64_t iterationLimit = 0;
  std::uint64_t noProgressIterationLimit = 0;
  std::uint64_t noProgressTrendWindow = 0;
};

enum class SpatialRoutingClosureRequirement : std::uint8_t {
  PolicyAdmittedTemporary,
  ExactRegional,
  Final,
};

struct SpatialPathFinderClosureResult final {
  std::uint64_t completedIterations = 0;
  bool capacityClosed = false;
};

struct SpatialFixedTerminalCutNet final {
  PnrIndex logicalNet = 0;
  PnrIndex unreachableSink = 0;
};

struct SpatialFixedTerminalCutCertificate final {
  PnrIndex capacity = getInvalidPnrIndex();
  std::vector<SpatialFixedTerminalCutNet> forcedNetCuts;
};

class SpatialPathFinderClosureFailure final
    : public llvm::ErrorInfo<SpatialPathFinderClosureFailure> {
public:
  enum class Kind {
    NonClosure,
    NoProgress,
    RegionalLimit,
    FixedTerminalCapacityCut,
    SelectedCombinationalHandshakeCycle,
  };

  static char ID;

  SpatialPathFinderClosureFailure(
      Kind kind, std::string message,
      SpatialFixedTerminalCutCertificate certificate = {},
      std::uint64_t mandatoryUsage = 0, std::uint64_t physicalCapacity = 0,
      std::uint64_t regionalLogicalNetCount = 0,
      std::uint64_t regionalLogicalNetLimit = 0);

  Kind kind() const { return kind_; }
  const SpatialFixedTerminalCutCertificate &certificate() const {
    return certificate_;
  }
  PnrIndex certificateCapacity() const { return certificate_.capacity; }
  std::uint64_t mandatoryUsage() const { return mandatoryUsage_; }
  std::uint64_t physicalCapacity() const { return physicalCapacity_; }
  llvm::ArrayRef<SpatialFixedTerminalCutNet> forcedNetCuts() const {
    return certificate_.forcedNetCuts;
  }
  std::uint64_t regionalLogicalNetCount() const {
    return regionalLogicalNetCount_;
  }
  std::uint64_t regionalLogicalNetLimit() const {
    return regionalLogicalNetLimit_;
  }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  Kind kind_;
  std::string message_;
  SpatialFixedTerminalCutCertificate certificate_;
  std::uint64_t mandatoryUsage_;
  std::uint64_t physicalCapacity_;
  std::uint64_t regionalLogicalNetCount_;
  std::uint64_t regionalLogicalNetLimit_;
};

/// Worker-local deterministic PathFinder closure scratch. One invocation uses
/// one SpatialMoveTransaction for every routed net and iteration. A failed
/// invocation therefore rolls back the complete route overlay without a
/// second RouteTree snapshot. P/H and canonical net order are frozen within an
/// iteration; only a complete non-closed iteration advances P/H.
class SpatialPathFinderRouterScratch final {
public:
  SpatialPathFinderRouterScratch() = default;
  SpatialPathFinderRouterScratch(const SpatialPathFinderRouterScratch &) =
      delete;
  SpatialPathFinderRouterScratch &
  operator=(const SpatialPathFinderRouterScratch &) = delete;
  SpatialPathFinderRouterScratch(SpatialPathFinderRouterScratch &&) = delete;
  SpatialPathFinderRouterScratch &
  operator=(SpatialPathFinderRouterScratch &&) = delete;
  ~SpatialPathFinderRouterScratch() = default;

  llvm::Error prepare(const FrozenSpatialPnrProblem &problem);

  llvm::Expected<SpatialPathFinderClosureResult>
  routeToClosure(SpatialCandidateState &candidate,
                 SpatialCandidateScratch &candidateScratch,
                 SpatialRouteCostState &costs,
                 SpatialPathFinderRoutingLimits limits,
                 llvm::ArrayRef<RouteCost> evaluationPriorities,
                 SpatialRoutingClosureRequirement closureRequirement =
                     SpatialRoutingClosureRequirement::Final);

  /// Applies negotiated routing inside an already active Mapping move. The
  /// caller owns close, objective evaluation, and commit or rollback. An
  /// empty logical-net region selects the complete frozen net domain.
  llvm::Expected<SpatialPathFinderClosureResult> routeToClosureInMove(
      SpatialMoveTransaction &move, SpatialCandidateState &candidate,
      SpatialRouteCostState &costs, SpatialPathFinderRoutingLimits limits,
      llvm::ArrayRef<PnrIndex> logicalNets,
      llvm::ArrayRef<RouteCost> evaluationPriorities,
      SpatialRoutingClosureRequirement closureRequirement =
          SpatialRoutingClosureRequirement::Final,
      std::uint64_t exactRegionalLogicalNetLimit = 0);

  llvm::Expected<RouteCost>
  routeWholeNetInMove(SpatialMoveTransaction &move,
                      const SpatialCandidateState &candidate,
                      SpatialRouteCostState &costs, PnrIndex logicalNet,
                      std::uint64_t endpointExpansionLimit);
  llvm::Expected<RouteCost> routeSingleSinkInMove(
      SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
      SpatialRouteCostState &costs, PnrIndex logicalNet,
      PnrIndex sinkObligation, std::uint64_t endpointExpansionLimit);
  llvm::Expected<RouteCost> routeRootedSubtreeInMove(
      SpatialMoveTransaction &move, const SpatialCandidateState &candidate,
      SpatialRouteCostState &costs, PnrIndex logicalNet, PnrIndex rootEndpoint,
      std::uint64_t endpointExpansionLimit);

  llvm::Error beginConstraintSweep(llvm::ArrayRef<PnrIndex> logicalNets);
  llvm::Error finishConstraintNet(PnrIndex logicalNet);

  std::uint64_t endpointExpansionCount() const {
    return netRouter_.endpointExpansionCount();
  }
  std::uint64_t heuristicCacheHitCount() const {
    return netRouter_.heuristicCacheHitCount();
  }
  std::uint64_t heuristicBuildCount() const {
    return netRouter_.heuristicBuildCount();
  }
  std::uint64_t heuristicCacheEvictionCount() const {
    return netRouter_.heuristicCacheEvictionCount();
  }
  std::size_t heuristicCacheEntryCount() const {
    return netRouter_.heuristicCacheEntryCount();
  }
  std::size_t heuristicCacheRetainedBytes() const {
    return netRouter_.heuristicCacheRetainedBytes();
  }
  std::uint64_t negotiationIterationCount() const {
    return negotiationIterationCount_;
  }
  std::uint64_t regionalLogicalNetCount() const {
    return routingRegionNets_.size();
  }
  llvm::ArrayRef<PnrIndex> regionalLogicalNets() const {
    return routingRegionNets_;
  }
  std::size_t retainedStorageBytes() const;

private:
  struct NetOrderEntry final {
    std::uint8_t routeStateRank = 0;
    RouteCost conflictPressure = 0;
    std::uint64_t physicalNegativeSlack = 0;
    std::uint64_t physicalCriticalDelay = 0;
    RouteCost evaluationPriority = 0;
    PnrIndex logicalNet = 0;
  };

  struct NetProjection final {
    std::uint8_t routeStateRank = 0;
    RouteCost conflictPressure = 0;
    std::uint64_t physicalNegativeSlack = 0;
    std::uint64_t physicalCriticalDelay = 0;
  };

  struct CapacityConflictAnalysis final {
    std::uint64_t conflictCount = 0;
    std::uint64_t diagnosticConflictCount = 0;
    PnrIndex certificateCapacity = getInvalidPnrIndex();
    std::uint64_t mandatoryUsage = 0;
    std::uint64_t physicalCapacity = 0;

    bool hasCertificate() const {
      return certificateCapacity != getInvalidPnrIndex();
    }
  };

  struct RoutingRegionState final {
    std::uint64_t unroutedObligationCount = 0;
    std::uint64_t routeCapacityOveruse = 0;
    std::uint64_t tagResidentCapacityOveruse = 0;
    std::uint64_t tagUnassignedCount = 0;
    std::uint64_t tagConflictCount = 0;
  };

  llvm::Expected<NetProjection>
  projectLogicalNet(const SpatialCandidateState &candidate,
                    const SpatialRouteCostState &costs, PnrIndex logicalNet);
  llvm::Error
  buildCanonicalNetOrder(const SpatialCandidateState &candidate,
                         const SpatialRouteCostState &costs,
                         llvm::ArrayRef<PnrIndex> logicalNets,
                         llvm::ArrayRef<RouteCost> evaluationPriorities);
  llvm::Error captureCurrentRoutes(const SpatialCandidateState &candidate,
                                   llvm::ArrayRef<PnrIndex> logicalNets);
  llvm::Error restoreCapturedRoutes(
      SpatialMoveTransaction &move, SpatialCandidateState &candidate,
      SpatialRouteCostState &costs, llvm::ArrayRef<PnrIndex> logicalNets,
      const dse::ObjectiveVector &expectedObjective,
      const SpatialCandidateRouteProjection &expectedProjection);
  llvm::Expected<CapacityConflictAnalysis>
  analyzeCapacityConflicts(const SpatialCandidateState &candidate,
                           const SpatialRouteCostState &costs,
                           std::uint64_t iteration,
                           std::uint64_t sessionIteration);
  llvm::Expected<RoutingRegionState>
  projectRoutingRegion(const SpatialCandidateState &candidate,
                       const SpatialRouteCostState &costs,
                       llvm::ArrayRef<PnrIndex> logicalNets);
  llvm::Expected<bool>
  expandExactRegionalConflictClosure(const SpatialCandidateState &candidate,
                                     const SpatialRouteCostState &costs,
                                     std::uint64_t logicalNetLimit);
  llvm::Expected<bool>
  expandRoutingRelationClosure(std::uint64_t logicalNetLimit);
  void beginProjection();

  SpatialNetRouterScratch netRouter_;
  std::vector<NetOrderEntry> netOrder_;
  std::vector<std::uint64_t> activeClaimBits_;
  std::vector<std::uint64_t> claimEpochs_;
  std::vector<std::uint64_t> capacityEpochs_;
  std::vector<RouteCost> capacityNetQCosts_;
  std::vector<PnrIndex> touchedCapacities_;
  std::vector<PnrIndex> conflictCapacities_;
  std::vector<std::uint8_t> regionalCapacityMarks_;
  std::vector<std::uint8_t> regionalTagDomainMarks_;
  std::vector<std::uint8_t> routingRegionNetMarks_;
  std::vector<PnrIndex> routingRegionNets_;
  std::vector<PnrIndex> constraintSweepNets_;
  std::vector<std::size_t> capturedSinkPathOffsets_;
  std::vector<PnrIndex> capturedForwardArcs_;
  std::vector<PnrIndex> reversePath_;
  std::vector<std::uint8_t> cutBlockedTraversals_;
  std::vector<std::uint8_t> cutReachableEndpoints_;
  std::vector<std::uint8_t> cutSeenTraversals_;
  std::vector<std::uint8_t> cutSeenEndpoints_;
  std::vector<PnrIndex> cutWorklist_;
  std::vector<PnrIndex> cutContributingNets_;
  std::vector<SpatialFixedTerminalCutNet> cutForcedNetCuts_;
  std::vector<SpatialFixedTerminalCutNet> cutCertificateForcedNetCuts_;
  std::vector<std::uint32_t> cutPayloadWidths_;
  std::vector<std::uint64_t> cutMinimumClaims_;
  std::vector<std::uint8_t> rankTrendTransitions_;
  std::vector<PnrIndex> cutTouchedClaims_;
  std::vector<PnrIndex> cutNetClaimRefcounts_;
  std::vector<PnrIndex> cutClaimSelectionCounts_;
  std::vector<std::uint64_t> cutClaimTraversalRefcounts_;
  std::vector<std::uint64_t> timingRouteNodeArrivals_;
  std::vector<std::pair<PnrIndex, std::uint64_t>> timingRouteNodeWorklist_;
  std::uint64_t projectionEpoch_ = 0;
  std::uint64_t negotiationIterationCount_ = 0;
  const FrozenSpatialPnrProblem *preparedProblem_ = nullptr;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALPATHFINDERROUTER_H

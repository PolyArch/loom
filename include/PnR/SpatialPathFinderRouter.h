#ifndef LOOM_PNR_SPATIALPATHFINDERROUTER_H
#define LOOM_PNR_SPATIALPATHFINDERROUTER_H

#include "PnR/SpatialNetRouter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

namespace loom::pnr {

struct SpatialPathFinderRoutingLimits final {
  std::uint64_t endpointExpansionLimit = 0;
  std::uint64_t iterationLimit = 0;
  std::uint64_t noProgressIterationLimit = 0;
  std::uint64_t noProgressTrendWindow = 0;
};

enum class SpatialRoutingClosureRequirement : std::uint8_t {
  PolicyAdmittedTemporary,
  Final,
};

struct SpatialPathFinderClosureResult final {
  std::uint64_t completedIterations = 0;
  bool capacityClosed = false;
};

class SpatialPathFinderClosureFailure final
    : public llvm::ErrorInfo<SpatialPathFinderClosureFailure> {
public:
  enum class Kind {
    NonClosure,
    NoProgress,
    FixedTerminalCapacityCut,
    SelectedCombinationalHandshakeCycle,
  };

  static char ID;

  SpatialPathFinderClosureFailure(
      Kind kind, std::string message,
      PnrIndex certificateCapacity = getInvalidPnrIndex(),
      std::uint64_t mandatoryUsage = 0, std::uint64_t physicalCapacity = 0,
      std::vector<PnrIndex> forcedLogicalNets = {});

  Kind kind() const { return kind_; }
  PnrIndex certificateCapacity() const { return certificateCapacity_; }
  std::uint64_t mandatoryUsage() const { return mandatoryUsage_; }
  std::uint64_t physicalCapacity() const { return physicalCapacity_; }
  llvm::ArrayRef<PnrIndex> forcedLogicalNets() const {
    return forcedLogicalNets_;
  }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  Kind kind_;
  std::string message_;
  PnrIndex certificateCapacity_;
  std::uint64_t mandatoryUsage_;
  std::uint64_t physicalCapacity_;
  std::vector<PnrIndex> forcedLogicalNets_;
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
          SpatialRoutingClosureRequirement::Final);

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
  std::uint64_t negotiationIterationCount() const {
    return negotiationIterationCount_;
  }
  std::size_t retainedStorageBytes() const;

private:
  struct NetOrderEntry final {
    std::uint8_t routeStateRank = 0;
    RouteCost conflictPressure = 0;
    RouteCost evaluationPriority = 0;
    PnrIndex logicalNet = 0;
  };

  struct NetProjection final {
    std::uint8_t routeStateRank = 0;
    RouteCost conflictPressure = 0;
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

  llvm::Expected<NetProjection>
  projectLogicalNet(const SpatialCandidateState &candidate,
                    const SpatialRouteCostState &costs, PnrIndex logicalNet);
  llvm::Error
  buildCanonicalNetOrder(const SpatialCandidateState &candidate,
                         const SpatialRouteCostState &costs,
                         llvm::ArrayRef<PnrIndex> logicalNets,
                         llvm::ArrayRef<RouteCost> evaluationPriorities);
  llvm::Error captureCurrentRoutes(const SpatialCandidateState &candidate);
  llvm::Error restoreCapturedRoutes(SpatialMoveTransaction &move,
                                    SpatialCandidateState &candidate,
                                    SpatialRouteCostState &costs);
  llvm::Expected<CapacityConflictAnalysis>
  analyzeCapacityConflicts(const SpatialCandidateState &candidate,
                           const SpatialRouteCostState &costs,
                           std::uint64_t iteration,
                           std::uint64_t sessionIteration);
  void beginProjection();

  SpatialNetRouterScratch netRouter_;
  std::vector<NetOrderEntry> netOrder_;
  std::vector<std::uint64_t> activeClaimBits_;
  std::vector<std::uint64_t> claimEpochs_;
  std::vector<std::uint64_t> capacityEpochs_;
  std::vector<RouteCost> capacityNetQCosts_;
  std::vector<PnrIndex> touchedCapacities_;
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
  std::vector<PnrIndex> cutForcedNets_;
  std::vector<PnrIndex> cutCertificateForcedNets_;
  std::vector<std::uint32_t> cutPayloadWidths_;
  std::vector<std::uint64_t> cutMinimumClaims_;
  std::vector<std::uint8_t> rankTrendTransitions_;
  std::vector<PnrIndex> cutTouchedClaims_;
  std::vector<PnrIndex> cutNetClaimRefcounts_;
  std::vector<PnrIndex> cutClaimSelectionCounts_;
  std::vector<std::uint64_t> cutClaimTraversalRefcounts_;
  std::uint64_t projectionEpoch_ = 0;
  std::uint64_t negotiationIterationCount_ = 0;
  const FrozenSpatialPnrProblem *preparedProblem_ = nullptr;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALPATHFINDERROUTER_H

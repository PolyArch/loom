#ifndef LOOM_MAPPER_MAPPER_H
#define LOOM_MAPPER_MAPPER_H

#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/MapperOptions.h"
#include "loom/Mapper/MapperTiming.h"
#include "loom/Mapper/TechMapper.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>

namespace mlir {
class ModuleOp;
}

namespace loom {

struct CongestionEstimator;
struct CongestionState;
class RelaxedRoutingState;
class LocalRepairDriver;
class TopologyModel;

struct MapperSearchSummary {
  unsigned techMapFeedbackAttempts = 0;
  unsigned techMapFeedbackAcceptedReconfigurations = 0;
  unsigned placementSeedLaneCount = 0;
  unsigned successfulPlacementSeedCount = 0;
  unsigned routedLaneCount = 0;
  unsigned localRepairAttempts = 0;
  unsigned localRepairSuccesses = 0;
  unsigned routeAwareRefinementPasses = 0;
  unsigned routeAwareCheckpointRescorePasses = 0;
  unsigned routeAwareCheckpointRestoreCount = 0;
  unsigned routeAwareNeighborhoodAttempts = 0;
  unsigned routeAwareNeighborhoodAcceptedMoves = 0;
  unsigned routeAwareCoarseFallbackMoves = 0;
  unsigned fifoBufferizationAcceptedToggles = 0;
  unsigned outerJointAcceptedRounds = 0;
};

class Mapper {
public:
  using Options = MapperOptions;
  using SnapshotCallback = std::function<void(
      const MappingState &, llvm::ArrayRef<TechMappedEdgeKind>,
      llvm::ArrayRef<FUConfigSelection>, llvm::StringRef, unsigned)>;

  struct Result {
    struct RoutedAlternative {
      unsigned laneIndex = 0;
      MappingState state;
      std::vector<TechMappedEdgeKind> edgeKinds;
      llvm::SmallVector<FUConfigSelection, 4> fuConfigs;
      MapperTimingSummary timingSummary;
      MapperSearchSummary searchSummary;
      size_t totalPathLen = 0;
      double placementCost = 0.0;
      double throughputCost = 0.0;
      double estimatedClockPeriod = 0.0;
    };

    bool success = false;
    unsigned selectedLaneIndex = 0;
    MappingState state;
    std::vector<TechMappedEdgeKind> edgeKinds;
    llvm::SmallVector<FUConfigSelection, 4> fuConfigs;
    TechMapper::Plan techMapPlan;
    TechMapper::PlanMetrics techMapMetrics;
    std::string techMapDiagnostics;
    MapperTimingSummary timingSummary;
    MapperSearchSummary searchSummary;
    std::vector<RoutedAlternative> routedAlternatives;
    std::string diagnostics;
  };

  /// Run the full PnR pipeline.
  Result run(const Graph &dfg, const Graph &adg, const ADGFlattener &flattener,
             mlir::ModuleOp adgModule, const Options &opts);

  void setSnapshotCallback(SnapshotCallback callback) {
    snapshotCallback_ = std::move(callback);
  }

private:
  friend class LocalRepairDriver;

  // Tech-mapping: match DFG ops to ADG FU types.
  // Returns a candidate map: DFG node ID -> list of compatible ADG FU node IDs.
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>>
  buildCandidates(const Graph &dfg, const Graph &adg);

  // Initial greedy placement.
  bool runPlacement(
      MappingState &state, const Graph &dfg, const Graph &adg,
      const ADGFlattener &flattener,
      const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
      const Options &opts);

  // Simulated annealing refinement.
  bool runRefinement(
      MappingState &state, const Graph &dfg, const Graph &adg,
      const ADGFlattener &flattener,
      const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
      const Options &opts,
      std::vector<TechMappedEdgeKind> *edgeKinds = nullptr);

  bool runLocalRepair(
      MappingState &state, const MappingState::Checkpoint &baseCheckpoint,
      llvm::ArrayRef<IdIndex> failedEdges, const Graph &dfg, const Graph &adg,
      const ADGFlattener &flattener,
      const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
      std::vector<TechMappedEdgeKind> &edgeKinds, const Options &opts,
      const CongestionState *congestion = nullptr,
      unsigned recursionDepth = 0);

  bool runExactRoutingRepair(MappingState &state,
                             llvm::ArrayRef<IdIndex> failedEdges,
                             const Graph &dfg, const Graph &adg,
                             const ADGFlattener &flattener,
                             llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                             const Options &opts,
                             const CongestionState *congestion = nullptr,
                             llvm::ArrayRef<IdIndex> priorityEdges =
                                 llvm::ArrayRef<IdIndex>());

  bool runInterleavedPlaceRoute(
      MappingState &state, const Graph &dfg, const Graph &adg,
      const ADGFlattener &flattener,
      const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
      std::vector<TechMappedEdgeKind> &edgeKinds, const Options &opts);

  /// Route-aware SA main optimization loop. Requires routing to have been
  /// performed before entry. Returns true if all edges are routed.
  bool runRouteAwareSA(
      MappingState &state, const Graph &dfg, const Graph &adg,
      const ADGFlattener &flattener,
      const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
      std::vector<TechMappedEdgeKind> &edgeKinds, const Options &opts);

  bool runCPSatGlobalPlacement(
      MappingState &state, const Graph &dfg, const Graph &adg,
      const ADGFlattener &flattener,
      const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
      const Options &opts);

  bool runCPSatNeighborhoodRepair(
      MappingState &state, const MappingState::Checkpoint &baseCheckpoint,
      llvm::ArrayRef<IdIndex> failedEdges, const Graph &dfg, const Graph &adg,
      const ADGFlattener &flattener,
      const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
      std::vector<TechMappedEdgeKind> &edgeKinds, const Options &opts);

  // Routing.
  bool runRouting(MappingState &state, const Graph &dfg, const Graph &adg,
                  llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                  const Options &opts);

  bool runNegotiatedRouting(MappingState &state, const Graph &dfg,
                            const Graph &adg,
                            llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                            const Options &opts);

  // Sentinel binding: map DFG boundary nodes to ADG boundary nodes.
  bool bindSentinels(MappingState &state, const Graph &dfg, const Graph &adg);

  // Memref sentinel binding: pre-route memref sentinel -> extmemory edges.
  // Must be called after placement so extmemory DFG nodes are already mapped.
  bool bindMemrefSentinels(MappingState &state, const Graph &dfg,
                           const Graph &adg);

  // Rebind scalar input sentinels after placement using a placement-aware
  // assignment to ADG boundary inputs.
  bool rebindScalarInputSentinels(MappingState &state, const Graph &dfg,
                                  const Graph &adg,
                                  const ADGFlattener &flattener);

  // Validation.
  bool runValidation(const MappingState &state, const Graph &dfg,
                     const Graph &adg, const ADGFlattener &flattener,
                     llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                     std::string &diagnostics);

  MapperTimingSummary runPostRouteFifoBufferization(
      MappingState &state, const Graph &dfg, const Graph &adg,
      llvm::ArrayRef<TechMappedEdgeKind> edgeKinds, const Options &opts);

  Result runWithTechMapPlan(const Graph &dfg, const Graph &adg,
                            const ADGFlattener &flattener,
                            mlir::ModuleOp adgModule, const Options &opts,
                            TechMapper &techMapper, TechMapper::Plan techPlan,
                            unsigned techFeedbackAttempt);

  // Routing helpers.
  llvm::SmallVector<IdIndex, 8>
  findPath(IdIndex srcHwPort, IdIndex dstHwPort, IdIndex swEdgeId,
           const MappingState &state, const Graph &dfg, const Graph &adg,
           const llvm::DenseMap<IdIndex, double> &routingOutputHistory,
           IdIndex forcedFirstHop = INVALID_ID,
           const CongestionState *congestion = nullptr,
           const RelaxedRoutingState *relaxedRouting = nullptr);
  bool isEdgeLegal(IdIndex srcPort, IdIndex dstPort, IdIndex swEdgeId,
                   llvm::ArrayRef<IdIndex> candidatePath,
                   const MappingState &state, const Graph &dfg,
                   const Graph &adg);
  bool isEdgeRoutable(IdIndex edgeId, const MappingState &state,
                      const Graph &dfg);
  bool routeOnePass(MappingState &state, const Graph &dfg, const Graph &adg,
                    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                    const std::vector<IdIndex> &edgeOrder,
                    const llvm::DenseMap<IdIndex, double> &routingOutputHistory,
                    unsigned &routed, unsigned &total, const Options &opts,
                    CongestionState *congestion = nullptr,
                    RelaxedRoutingState *relaxedRouting = nullptr);
  bool hasTaggedPathConflict(IdIndex swEdgeId,
                             llvm::ArrayRef<IdIndex> candidatePath,
                             const MappingState &state, const Graph &dfg,
                             const Graph &adg);
  bool validateTaggedPathConflicts(const MappingState &state, const Graph &dfg,
                                   const Graph &adg,
                                   llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                   std::string &diagnostics);

  // Placement scoring.
  double scorePlacement(
      IdIndex swNode, IdIndex hwNode, const MappingState &state,
      const Graph &dfg, const Graph &adg, const ADGFlattener &flattener,
      const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates);

  // Compute total placement cost (sum of edge distances).
  double computeTotalCost(const MappingState &state, const Graph &dfg,
                          const Graph &adg, const ADGFlattener &flattener);

  // Rebind ports after placing a node, including memory-interface ports.
  bool bindMappedNodePorts(IdIndex swNode, MappingState &state,
                           const Graph &dfg, const Graph &adg);

  // Topological order for greedy placement.
  std::vector<IdIndex> computePlacementOrder(const Graph &dfg);

  void resetRunControls(const Options &opts);
  bool shouldStopForBudget(llvm::StringRef stage);
  double remainingBudgetSeconds() const;
  double clampToRemainingBudget(double requestedSeconds) const;
  double clampDeadlineMsToRemainingBudget(double requestedMs) const;
  bool maybeEmitProgressSnapshot(const MappingState &state,
                                 llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                 llvm::StringRef trigger,
                                 const Options &opts);

  ConnectivityMatrix connectivity;
  const ADGFlattener *activeFlattener = nullptr;
  std::shared_ptr<const TopologyModel> activeTopologyModel_;
  double activeHeuristicWeight = 1.5;
  CongestionEstimator *activeCongestionEstimator = nullptr;
  double activeCongestionPlacementWeight = 0.0;
  double activeMemorySharingPenalty = 0.0;
  unsigned activeUnroutedDiagnosticLimit = 8;
  MapperRelaxedRoutingOptions activeRelaxedRoutingOpts_;
  SnapshotCallback snapshotCallback_;
  std::function<void(const MappingState &, llvm::ArrayRef<TechMappedEdgeKind>,
                     llvm::StringRef, unsigned)>
      activeSnapshotEmitter_;
  std::chrono::steady_clock::time_point activeRunStartTime_;
  std::chrono::steady_clock::time_point activeRunDeadline_;
  bool activeBudgetEnabled_ = false;
  bool activeBudgetExceeded_ = false;
  std::string activeBudgetExceededStage_;
  unsigned snapshotSequence_ = 0;
  unsigned snapshotTickCount_ = 0;
  double nextSnapshotAtSeconds_ = -1.0;
  MapperSearchSummary activeSearchSummary_;
};

} // namespace loom

#endif // LOOM_MAPPER_MAPPER_H

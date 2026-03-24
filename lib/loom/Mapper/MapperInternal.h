#ifndef LOOM_MAPPER_MAPPER_INTERNAL_H
#define LOOM_MAPPER_MAPPER_INTERNAL_H

#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/BridgeBinding.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/MapperOptions.h"
#include "loom/Mapper/TechMapper.h"
#include "loom/Mapper/TopologyModel.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <random>
#include <string>
#include <vector>

namespace loom {
namespace mapper_detail {

using CandidateMap = llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>>;
using CandidateSetMap = llvm::DenseMap<IdIndex, llvm::DenseSet<IdIndex>>;

// Type checking helpers.
bool isMemrefType(mlir::Type type);
bool isNoneType(mlir::Type type);
bool isTemporalPENode(const Node *hwNode);
bool isSpatialPENode(const Node *hwNode);
bool isRoutingResourceNode(const Node *hwNode);

// PE utilities.
const PEContainment *findPEContainmentByName(const ADGFlattener &flattener,
                                             llvm::StringRef peName);
bool isSpatialPEName(const ADGFlattener &flattener, llvm::StringRef peName);
bool isSpatialPEOccupied(const MappingState &state, const Graph &adg,
                         const ADGFlattener &flattener, llvm::StringRef peName,
                         IdIndex ignoreHwNode = INVALID_ID);

// Config comparison.
bool sameConfigFields(llvm::ArrayRef<FUConfigField> lhs,
                      llvm::ArrayRef<FUConfigField> rhs);

// Tech mapping conflict detection.
bool detectForcedTemporalConfigConflict(const TechMapper::Plan &plan,
                                        const llvm::DenseMap<
                                            IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
                                        const Graph &dfg,
                                        const Graph &adg,
                                        std::string &diagnostics);

// Edge classification.
void classifyTemporalRegisterEdges(const MappingState &state, const Graph &dfg,
                                   const Graph &adg,
                                   const ADGFlattener &flattener,
                                   std::vector<TechMappedEdgeKind> &edgeKinds);

// Graph navigation.
IdIndex findDownstreamNode(const Graph &graph, IdIndex sentinelNodeId);

// Operation matching.
llvm::StringRef getCompatibleOp(llvm::StringRef dfgOpName);
bool opMatchesFU(llvm::StringRef dfgOpName, const Node *fuNode);

// Memory operation helpers.
bool isMemoryOp(const Node *node);
bool isSoftwareMemoryInterfaceOp(llvm::StringRef opName);
IdIndex getExpandedMemoryInputPort(const Node *hwNode, const Graph &adg,
                                   bool isExtMem, BridgePortCategory cat,
                                   unsigned lane);
IdIndex getExpandedMemoryOutputPort(const Node *hwNode, const Graph &adg,
                                    BridgePortCategory cat, unsigned lane);
IdIndex findBridgePortForCategoryLane(const BridgeInfo &bridge, bool isInput,
                                      BridgePortCategory cat, unsigned lane);
std::optional<std::pair<double, double>>
getPortPlacementPos(IdIndex portId, const Graph &adg,
                    const ADGFlattener &flattener);
std::optional<std::pair<double, double>>
estimateSoftwarePortPlacementPos(IdIndex swNode, IdIndex swPort,
                                 IdIndex hwNode, bool isInput,
                                 const Graph &dfg, const Graph &adg,
                                 const ADGFlattener &flattener);

// Placement heuristics.
double classifyEdgePlacementWeight(const Graph &dfg, IdIndex edgeId);
std::vector<double> buildEdgePlacementWeightCache(const Graph &dfg);
double computeNodePriorityWeight(IdIndex swNode, const Graph &dfg);
std::optional<std::pair<double, double>>
estimateNodePlacementPos(IdIndex swNode, const MappingState &state,
                         const Graph &dfg, const ADGFlattener &flattener,
                         const CandidateMap &candidates);
double computeLocalSpreadPenalty(IdIndex hwNode, const MappingState &state,
                                 const Graph &adg,
                                 const ADGFlattener &flattener);
CandidateSetMap buildCandidateSetMap(const CandidateMap &candidates);

// Refinement/repair shared helpers.
void setActiveTopologyModel(const TopologyModel *model);
const TopologyModel *getActiveTopologyModel();
void setActiveTimingOptions(const MapperTimingOptions *opts);
const MapperTimingOptions *getActiveTimingOptions();
int placementDistance(IdIndex lhsHwNode, IdIndex rhsHwNode,
                      const ADGFlattener &flattener);
bool isWithinMoveRadius(IdIndex lhsHwNode, IdIndex rhsHwNode,
                        const ADGFlattener &flattener, unsigned radius);
double computeNodeTimingPenalty(IdIndex swNode, IdIndex hwNode,
                                const Graph &dfg, const Graph &adg);
bool canRelocateNode(
    IdIndex swNode, IdIndex newHwNode, IdIndex oldHwNode,
    const MappingState &state, const Graph &adg, const ADGFlattener &flattener,
    const CandidateMap &candidates,
    const CandidateSetMap *candidateSets = nullptr);
bool canSwapNodes(
    IdIndex swA, IdIndex swB, IdIndex hwA, IdIndex hwB,
    const MappingState &state, const Graph &adg, const ADGFlattener &flattener,
    const CandidateMap &candidates,
    const CandidateSetMap *candidateSets = nullptr);
double computeUnroutedPenalty(const MappingState &state, const Graph &dfg,
                              llvm::ArrayRef<TechMappedEdgeKind> edgeKinds);

// Edge collection and counting.
struct RoutingEdgeStats {
  unsigned overallEdges = 0;
  unsigned fixedInternalEdges = 0;
  unsigned directBindingEdges = 0;
  unsigned routerEdges = 0;
  unsigned routedOverallEdges = 0;
  unsigned routedRouterEdges = 0;
  unsigned unroutedRouterEdges = 0;
};

RoutingEdgeStats computeRoutingEdgeStats(
    const MappingState &state, const Graph &dfg,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds);

std::vector<IdIndex>
collectUnroutedEdges(const MappingState &state, const Graph &dfg,
                     llvm::ArrayRef<TechMappedEdgeKind> edgeKinds);
unsigned countRoutedEdges(const MappingState &state, const Graph &dfg,
                          llvm::ArrayRef<TechMappedEdgeKind> edgeKinds);
size_t computeTotalMappedPathLen(const MappingState &state);

// --- Shared SA helpers (used by MapperRefinement and MapperRouteAwareSA) ---

constexpr double kCutLoadQuadraticWeight = 0.006;

struct SACostState {
  double totalCost = 0.0;
  bool enableGridCutLoad = false;
  std::vector<double> rowCutLoad;
  std::vector<double> colCutLoad;
  struct UndoRecord {
    bool isRow = false;
    int index = -1;
    double oldValue = 0.0;
  };
  struct Savepoint {
    size_t undoMarker = 0;
    double totalCost = 0.0;
  };
  std::vector<UndoRecord> undoLog;
  std::vector<size_t> savepointMarkers;
};

struct SAAdaptiveState {
  unsigned windowIterations = 0;
  unsigned windowAccepted = 0;
  unsigned windowBestImprovements = 0;
  unsigned iterationsSinceBestImprovement = 0;
};

struct SAMoveResult {
  bool moveOk = false;
  llvm::SmallVector<IdIndex, 2> movedNodes;
  llvm::DenseMap<IdIndex, IdIndex> oldHwBySwNode;
};

// SACostState savepoint management.
SACostState::Savepoint beginCostSavepoint(SACostState &costState);
void rollbackCostSavepoint(SACostState &costState,
                           SACostState::Savepoint savepoint);
void commitCostSavepoint(SACostState &costState,
                         SACostState::Savepoint savepoint);

// Internal cost helpers.
void recordLoadUndo(SACostState &costState, bool isRow, int index,
                    double oldValue);
void adjustQuadraticLoad(std::vector<double> &loads, int index, double delta,
                         double &totalCost, SACostState &costState,
                         bool isRow);
void applyEdgePlacementContribution(double sign, double edgeWeight,
                                    IdIndex srcHw, IdIndex dstHw,
                                    const Graph &adg,
                                    const ADGFlattener &flattener,
                                    SACostState &costState);

// SA cost initialization and incremental update.
SACostState initializeSACostState(const MappingState &state, const Graph &dfg,
                                  const Graph &adg,
                                  const ADGFlattener &flattener,
                                  llvm::ArrayRef<double> edgeWeights);
void applyPlacementDeltaForMovedNodes(
    const llvm::SmallVectorImpl<IdIndex> &movedNodes,
    const llvm::DenseMap<IdIndex, IdIndex> &oldHwBySwNode,
    const MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener, llvm::ArrayRef<double> edgeWeights,
    SACostState &costState);

// Adaptive cooling.
void applyAdaptiveCoolingWindow(double &temperature,
                                const MapperOptions &opts,
                                SAAdaptiveState &adaptiveState);

// Edge collection for SA moves.
llvm::SmallVector<IdIndex, 32>
collectPlacementDeltaEdges(llvm::ArrayRef<IdIndex> movedNodes,
                           const Graph &dfg);

// Route-aware checkpoint cost.
double computeRouteAwareCheckpointCost(
    const MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
    const MapperOptions &opts, double placementCost);

} // namespace mapper_detail
} // namespace loom

#endif // LOOM_MAPPER_MAPPER_INTERNAL_H

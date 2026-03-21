#ifndef FCC_MAPPER_MAPPER_INTERNAL_H
#define FCC_MAPPER_MAPPER_INTERNAL_H

#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/BridgeBinding.h"
#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"
#include "fcc/Mapper/MapperOptions.h"
#include "fcc/Mapper/TechMapper.h"
#include "fcc/Mapper/TopologyModel.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <vector>

namespace fcc {
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

} // namespace mapper_detail
} // namespace fcc

#endif // FCC_MAPPER_MAPPER_INTERNAL_H

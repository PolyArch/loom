#ifndef FCC_MAPPER_MAPPER_H
#define FCC_MAPPER_MAPPER_H

#include "fcc/Mapper/ADGFlattener.h"
#include "fcc/Mapper/Graph.h"
#include "fcc/Mapper/MappingState.h"
#include "fcc/Mapper/TechMapper.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir {
class ModuleOp;
}

namespace fcc {

struct CongestionEstimator;
struct CongestionState;

class Mapper {
public:
  struct Options {
    double budgetSeconds = 60.0;
    int seed = 0;
    std::string profile = "balanced";
    unsigned lanes = 0;
    unsigned interleavedRounds = 4;
    unsigned selectiveRipupPasses = 3;
    unsigned placementMoveRadius = 3;
    unsigned cpSatGlobalNodeLimit = 24;
    unsigned cpSatNeighborhoodNodeLimit = 8;
    double cpSatTimeLimitSeconds = 0.75;
    bool enableCPSat = true;
    bool verbose = false;
    double routingHeuristicWeight = 1.5;
    unsigned negotiatedRoutingPasses = 12;
    double congestionHistoryFactor = 1.0;
    double congestionHistoryScale = 1.5;
    double congestionPresentFactor = 1.0;
    double congestionPlacementWeight = 0.3;
  };

  struct Result {
    bool success = false;
    MappingState state;
    std::vector<TechMappedEdgeKind> edgeKinds;
    llvm::SmallVector<FUConfigSelection, 4> fuConfigs;
    std::string diagnostics;
  };

  /// Run the full PnR pipeline.
  Result run(const Graph &dfg, const Graph &adg, const ADGFlattener &flattener,
             mlir::ModuleOp adgModule, const Options &opts);

private:
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
      const Options &opts);

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

  // Routing helpers.
  llvm::SmallVector<IdIndex, 8>
  findPath(IdIndex srcHwPort, IdIndex dstHwPort, IdIndex swEdgeId,
           const MappingState &state, const Graph &dfg, const Graph &adg,
           const llvm::DenseMap<IdIndex, double> &routingOutputHistory,
           IdIndex forcedFirstHop = INVALID_ID,
           const CongestionState *congestion = nullptr);
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
                    unsigned &routed, unsigned &total,
                    const CongestionState *congestion = nullptr);
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

  ConnectivityMatrix connectivity;
  const ADGFlattener *activeFlattener = nullptr;
  double activeHeuristicWeight = 1.5;
  CongestionEstimator *activeCongestionEstimator = nullptr;
  double activeCongestionPlacementWeight = 0.0;
};

} // namespace fcc

#endif // FCC_MAPPER_MAPPER_H

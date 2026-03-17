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

class Mapper {
public:
  struct Options {
    double budgetSeconds = 60.0;
    int seed = 0;
    std::string profile = "balanced";
    bool verbose = false;
  };

  struct Result {
    bool success = false;
    MappingState state;
    std::vector<TechMappedEdgeKind> edgeKinds;
    llvm::SmallVector<FUConfigSelection, 4> fuConfigs;
    std::string diagnostics;
  };

  /// Run the full PnR pipeline.
  Result run(const Graph &dfg, const Graph &adg,
             const ADGFlattener &flattener, mlir::ModuleOp adgModule,
             const Options &opts);

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

  // BFS routing.
  bool runRouting(MappingState &state, const Graph &dfg, const Graph &adg,
                  int seed = -1);

  // Sentinel binding: map DFG boundary nodes to ADG boundary nodes.
  bool bindSentinels(MappingState &state, const Graph &dfg, const Graph &adg);

  // Memref sentinel binding: pre-route memref sentinel -> extmemory edges.
  // Must be called after placement so extmemory DFG nodes are already mapped.
  bool bindMemrefSentinels(MappingState &state, const Graph &dfg,
                           const Graph &adg);

  // Validation.
  bool runValidation(const MappingState &state, const Graph &dfg,
                     const Graph &adg,
                     llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                     std::string &diagnostics);

  // Routing helpers.
  llvm::SmallVector<IdIndex, 8> findPath(IdIndex srcHwPort, IdIndex dstHwPort,
                                          const MappingState &state,
                                          const Graph &adg);
  bool isEdgeLegal(IdIndex srcPort, IdIndex dstPort,
                   const MappingState &state, const Graph &adg);
  bool isEdgeRoutable(IdIndex edgeId, const MappingState &state,
                      const Graph &dfg);
  bool routeOnePass(MappingState &state, const Graph &dfg, const Graph &adg,
                    const std::vector<IdIndex> &edgeOrder,
                    unsigned &routed, unsigned &total);

  // Placement scoring.
  double scorePlacement(IdIndex swNode, IdIndex hwNode,
                        const MappingState &state, const Graph &dfg,
                        const Graph &adg, const ADGFlattener &flattener);

  // Compute total placement cost (sum of edge distances).
  double computeTotalCost(const MappingState &state, const Graph &dfg,
                          const Graph &adg, const ADGFlattener &flattener);

  // Topological order for greedy placement.
  std::vector<IdIndex> computePlacementOrder(const Graph &dfg);

  ConnectivityMatrix connectivity;
};

} // namespace fcc

#endif // FCC_MAPPER_MAPPER_H

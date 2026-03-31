#ifndef FCC_MAPPER_MAPPINGSTATE_H
#define FCC_MAPPER_MAPPINGSTATE_H

#include "fcc/Mapper/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <vector>

namespace fcc {

class Graph;

enum class ActionResult {
  Success,
  FailedHardConstraint,
  FailedResourceUnavailable,
  FailedInternalError,
};

class MappingState {
public:
  MappingState() = default;

  /// Initialize state vectors for the given DFG and ADG sizes.
  void init(const Graph &dfg, const Graph &adg);

  // Forward mappings: SW entity -> HW entity.
  std::vector<IdIndex> swNodeToHwNode;
  std::vector<IdIndex> swPortToHwPort;
  std::vector<llvm::SmallVector<IdIndex, 8>> swEdgeToHwPaths;

  // Reverse mappings: HW entity -> SW entities.
  std::vector<llvm::SmallVector<IdIndex, 2>> hwNodeToSwNodes;
  std::vector<llvm::SmallVector<IdIndex, 2>> hwPortToSwPorts;
  std::vector<llvm::SmallVector<IdIndex, 4>> hwEdgeToSwEdges;

  // Cost metrics.
  double totalCost = 0.0;

  // Action primitives.
  ActionResult mapNode(IdIndex swNode, IdIndex hwNode,
                       const Graph &dfg, const Graph &adg);
  ActionResult unmapNode(IdIndex swNode,
                         const Graph &dfg, const Graph &adg);
  ActionResult mapPort(IdIndex swPort, IdIndex hwPort,
                       const Graph &dfg, const Graph &adg);
  ActionResult mapEdge(IdIndex swEdge, llvm::ArrayRef<IdIndex> path,
                       const Graph &dfg, const Graph &adg);

  /// Checkpoint/restore for SA.
  struct Checkpoint {
    std::vector<IdIndex> swNodeToHwNode;
    std::vector<IdIndex> swPortToHwPort;
    std::vector<llvm::SmallVector<IdIndex, 8>> swEdgeToHwPaths;
    std::vector<llvm::SmallVector<IdIndex, 2>> hwNodeToSwNodes;
    std::vector<llvm::SmallVector<IdIndex, 2>> hwPortToSwPorts;
    std::vector<llvm::SmallVector<IdIndex, 4>> hwEdgeToSwEdges;
    double totalCost;
  };

  Checkpoint save() const;
  void restore(const Checkpoint &cp);
};

} // namespace fcc

#endif // FCC_MAPPER_MAPPINGSTATE_H

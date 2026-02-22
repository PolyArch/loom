//===-- TechMapper.h - Technology mapping for DFG to ADG ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Technology mapping: determines which ADG nodes can execute each DFG operation.
// Performs PE body pattern extraction, single-operation matching, and
// multi-operation subgraph matching.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_TECHMAPPER_H
#define LOOM_MAPPER_TECHMAPPER_H

#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <vector>

namespace loom {

/// A candidate represents a possible mapping of a DFG node (or group) to
/// an ADG node. For multi-op groups, all DFG nodes in the group map to the
/// same ADG node.
struct Candidate {
  IdIndex hwNodeId = INVALID_ID;
  /// For multi-op groups: the set of DFG nodes that form this group.
  /// For single-op: contains just the one DFG node.
  llvm::SmallVector<IdIndex, 1> swNodeIds;
};

/// CandidateSet maps each DFG node to its list of compatible ADG candidates.
using CandidateSet = llvm::DenseMap<IdIndex, std::vector<Candidate>>;

class TechMapper {
public:
  /// Run technology mapping: find all compatible ADG nodes for each DFG node.
  /// Returns a CandidateSet mapping DFG node IDs to candidate lists.
  /// Returns empty candidate set for a node if no compatible hardware exists.
  CandidateSet map(const Graph &dfg, const Graph &adg);

private:
  /// Extract the operation pattern (body ops) from an ADG PE node.
  std::vector<std::string> extractPEPattern(const Graph &adg, IdIndex nodeId);

  /// Check single-operation compatibility between a DFG op and an ADG node.
  bool isSingleOpCompatible(const Graph &dfg, IdIndex swNode,
                            const Graph &adg, IdIndex hwNode);

  /// Check type compatibility between a DFG port and an ADG port.
  bool isTypeCompatible(const Graph &dfg, IdIndex swPort,
                        const Graph &adg, IdIndex hwPort);
};

} // namespace loom

#endif // LOOM_MAPPER_TECHMAPPER_H

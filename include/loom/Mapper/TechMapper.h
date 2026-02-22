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
  /// True if this is a multi-op group candidate (higher priority).
  bool isGroup = false;
};

/// CandidateSet maps each DFG node to its list of compatible ADG candidates.
using CandidateSet = llvm::DenseMap<IdIndex, std::vector<Candidate>>;

/// A PE body pattern extracted from an ADG PE node.
struct PEBodyPattern {
  IdIndex hwNodeId = INVALID_ID;
  /// Operation names in the PE body (in order).
  std::vector<std::string> opNames;
  /// Internal edge connectivity: pairs of (src_op_idx, dst_op_idx).
  std::vector<std::pair<unsigned, unsigned>> internalEdges;
  /// Hash for fast pattern deduplication.
  uint64_t patternHash = 0;
};

class TechMapper {
public:
  /// Run technology mapping: find all compatible ADG nodes for each DFG node.
  /// Returns a CandidateSet mapping DFG node IDs to candidate lists.
  /// Returns empty candidate set for a node if no compatible hardware exists.
  CandidateSet map(const Graph &dfg, const Graph &adg);

private:
  /// Extract the operation body pattern from an ADG PE node.
  /// Returns the list of ops in the PE body (from body_ops attribute).
  PEBodyPattern extractPEPattern(const Graph &adg, IdIndex nodeId);

  /// Check single-operation compatibility between a DFG op and an ADG node.
  bool isSingleOpCompatible(const Graph &dfg, IdIndex swNode,
                            const Graph &adg, IdIndex hwNode);

  /// Check if a single DFG operation name is compatible with a PE body op.
  bool isOpNameCompatible(llvm::StringRef swOp, llvm::StringRef hwOp);

  /// Find multi-op group candidates: DFG subgraphs matching multi-op PE bodies.
  void findGroupCandidates(const Graph &dfg, const Graph &adg,
                           const std::vector<PEBodyPattern> &patterns,
                           CandidateSet &candidates);

  /// Merge group candidates with single-op candidates, enforcing exclusivity.
  /// Group candidates have priority when they cover more DFG nodes.
  void mergeCandidates(CandidateSet &candidates);

  /// Check type compatibility between a DFG port and an ADG port.
  bool isTypeCompatible(const Graph &dfg, IdIndex swPort,
                        const Graph &adg, IdIndex hwPort);
};

} // namespace loom

#endif // LOOM_MAPPER_TECHMAPPER_H

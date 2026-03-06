//===-- MappingState.h - Canonical mapping state ------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// MappingState holds the complete state of a mapping attempt: forward/reverse
// mappings between software and hardware entities, temporal assignments, cost
// metrics, and action logging for RL compatibility.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_MAPPINGSTATE_H
#define LOOM_MAPPER_MAPPINGSTATE_H

#include "loom/Mapper/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <vector>

namespace loom {

class Graph;

/// Result of an action primitive.
enum class ActionResult {
  Success,
  FailedHardConstraint,
  FailedResourceUnavailable,
  FailedInternalError,
};

/// Temporal PE assignment for a mapped software node.
struct TemporalPEAssignment {
  IdIndex slot = INVALID_ID;
  IdIndex tag = INVALID_ID;
  IdIndex opcode = INVALID_ID;
};

/// Temporal switch assignment for a route segment.
struct TemporalSWAssignment {
  IdIndex slot = INVALID_ID;
  IdIndex tag = INVALID_ID;
  uint64_t routeMask = 0;
};

/// Action record for RL training log.
struct ActionRecord {
  enum Type {
    MAP_NODE,
    UNMAP_NODE,
    MAP_PORT,
    UNMAP_PORT,
    MAP_EDGE,
    UNMAP_EDGE,
  };
  Type type;
  IdIndex arg0 = INVALID_ID;
  IdIndex arg1 = INVALID_ID;
  llvm::SmallVector<IdIndex, 8> pathArgs;
  double costDelta = 0.0;
  ActionResult constraintResult = ActionResult::Success;
};

class MappingState {
public:
  MappingState() = default;

  /// Initialize state vectors for the given DFG and ADG sizes.
  void init(const Graph &dfg, const Graph &adg);

  // --- Forward mappings ---
  std::vector<IdIndex> swNodeToHwNode;
  std::vector<IdIndex> swPortToHwPort;
  std::vector<llvm::SmallVector<IdIndex, 8>> swEdgeToHwPaths;

  // --- Reverse mappings ---
  std::vector<llvm::SmallVector<IdIndex, 2>> hwNodeToSwNodes;
  std::vector<llvm::SmallVector<IdIndex, 2>> hwPortToSwPorts;
  std::vector<llvm::SmallVector<IdIndex, 4>> hwEdgeToSwEdges;

  // --- Temporal assignments ---
  std::vector<TemporalPEAssignment> temporalPEAssignments;
  std::vector<llvm::SmallVector<TemporalSWAssignment, 4>> temporalSWAssignments;
  std::vector<IdIndex> registerAssignments;

  // --- Group placement bindings ---
  // Maps HW node ID to the set of SW nodes that form a valid multi-op group.
  // Populated during group placement; used by C4 validation to allow
  // multiple SW nodes on the same non-temporal PE when they form a group.
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> groupBindings;

  // --- Cost metrics ---
  double totalCost = 0.0;
  double placementPressure = 0.0;
  double routingCost = 0.0;
  double temporalCost = 0.0;
  double perfProxyCost = 0.0;
  double criticalPathEst = 0.0;
  double iiPressure = 0.0;
  double queuePressure = 0.0;
  double configFootprint = 0.0;
  uint32_t nonDefaultWords = 0;
  uint32_t totalConfigWords = 0;

  // --- Action primitives ---
  ActionResult mapNode(IdIndex swNode, IdIndex hwNode,
                       const Graph &dfg, const Graph &adg);
  ActionResult unmapNode(IdIndex swNode,
                         const Graph &dfg, const Graph &adg);
  ActionResult mapPort(IdIndex swPort, IdIndex hwPort,
                       const Graph &dfg, const Graph &adg);
  ActionResult unmapPort(IdIndex swPort,
                         const Graph &dfg, const Graph &adg);
  ActionResult mapEdge(IdIndex swEdge, llvm::ArrayRef<IdIndex> path,
                       const Graph &dfg, const Graph &adg);
  ActionResult unmapEdge(IdIndex swEdge,
                         const Graph &dfg, const Graph &adg);

  /// Check if the overall state is valid.
  bool isValid() const;

  // --- Action log ---
  std::vector<ActionRecord> actionLog;

  // --- Checkpoint/restore ---
  struct Checkpoint {
    std::vector<IdIndex> swNodeToHwNode;
    std::vector<IdIndex> swPortToHwPort;
    std::vector<llvm::SmallVector<IdIndex, 8>> swEdgeToHwPaths;
    std::vector<llvm::SmallVector<IdIndex, 2>> hwNodeToSwNodes;
    std::vector<llvm::SmallVector<IdIndex, 2>> hwPortToSwPorts;
    std::vector<llvm::SmallVector<IdIndex, 4>> hwEdgeToSwEdges;
    std::vector<TemporalPEAssignment> temporalPEAssignments;
    std::vector<llvm::SmallVector<TemporalSWAssignment, 4>> temporalSWAssignments;
    std::vector<IdIndex> registerAssignments;
    llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> groupBindings;
    double totalCost;
    size_t actionLogSize;
  };

  Checkpoint save() const;
  void restore(const Checkpoint &cp);
};

} // namespace loom

#endif // LOOM_MAPPER_MAPPINGSTATE_H

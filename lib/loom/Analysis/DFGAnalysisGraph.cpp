//===-- DFGAnalysisGraph.cpp - Level B: Graph-level DFG analysis --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Level B analysis operates on the extracted DFG Graph:
//   1. Recurrence detection via carry-cycle BFS
//   2. Critical path estimation via topological sort + DP
//   3. Composite temporal candidacy score
//
//===----------------------------------------------------------------------===//

#include "loom/Analysis/DFGAnalysis.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

namespace loom {
namespace analysis {

namespace {

//===----------------------------------------------------------------------===//
// Node attribute helpers (local to this file)
//===----------------------------------------------------------------------===//

llvm::StringRef getOpName(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "op_name") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

int64_t getIntAttr(const Node *node, llvm::StringRef name, int64_t dflt = 0) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return intAttr.getInt();
    }
  }
  return dflt;
}

bool getBoolAttr(const Node *node, llvm::StringRef name, bool dflt = false) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        return boolAttr.getValue();
    }
  }
  return dflt;
}

double getFloatAttr(const Node *node, llvm::StringRef name, double dflt = 0.0) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(attr.getValue()))
        return floatAttr.getValueAsDouble();
    }
  }
  return dflt;
}

void setNodeAttr(Node *node, mlir::NamedAttribute namedAttr) {
  // Update existing or append.
  for (auto &attr : node->attributes) {
    if (attr.getName() == namedAttr.getName()) {
      attr = namedAttr;
      return;
    }
  }
  node->attributes.push_back(namedAttr);
}

/// Get the set of successor node IDs for a given node.
llvm::SmallVector<IdIndex, 4> getSuccessors(const Graph &graph, IdIndex nodeId) {
  llvm::SmallVector<IdIndex, 4> succs;
  const Node *node = graph.getNode(nodeId);
  if (!node)
    return succs;
  for (IdIndex outPortId : node->outputPorts) {
    const Port *outPort = graph.getPort(outPortId);
    if (!outPort)
      continue;
    for (IdIndex edgeId : outPort->connectedEdges) {
      const Edge *edge = graph.getEdge(edgeId);
      if (!edge || edge->srcPort != outPortId)
        continue;
      const Port *dstPort = graph.getPort(edge->dstPort);
      if (dstPort && dstPort->parentNode != INVALID_ID)
        succs.push_back(dstPort->parentNode);
    }
  }
  return succs;
}

/// Get the set of predecessor node IDs for a given node.
llvm::SmallVector<IdIndex, 4> getPredecessors(const Graph &graph,
                                               IdIndex nodeId) {
  llvm::SmallVector<IdIndex, 4> preds;
  const Node *node = graph.getNode(nodeId);
  if (!node)
    return preds;
  for (IdIndex inPortId : node->inputPorts) {
    const Port *inPort = graph.getPort(inPortId);
    if (!inPort)
      continue;
    for (IdIndex edgeId : inPort->connectedEdges) {
      const Edge *edge = graph.getEdge(edgeId);
      if (!edge || edge->dstPort != inPortId)
        continue;
      const Port *srcPort = graph.getPort(edge->srcPort);
      if (srcPort && srcPort->parentNode != INVALID_ID)
        preds.push_back(srcPort->parentNode);
    }
  }
  return preds;
}

//===----------------------------------------------------------------------===//
// Recurrence detection
//===----------------------------------------------------------------------===//

/// Detect recurrence cycles through dataflow.carry ops.
/// A recurrence exists when a path from carry's output reaches carry's
/// input port b (operand index 2).
struct RecurrenceResult {
  /// For each node, whether it's on a recurrence cycle.
  llvm::DenseMap<IdIndex, bool> onRecurrence;
  /// For each node on a recurrence, its cycle ID.
  llvm::DenseMap<IdIndex, int32_t> recurrenceId;
  /// Set of back-edge pairs (src, dst) that form carry loops.
  llvm::DenseSet<std::pair<IdIndex, IdIndex>> backEdges;
};

/// Find the node that feeds carry's input port b (operand index 2).
/// Returns INVALID_ID if not found.
IdIndex findCarryBackEdgeSource(const Graph &graph, IdIndex carryId) {
  const Node *carryNode = graph.getNode(carryId);
  if (!carryNode || carryNode->inputPorts.size() <= 2)
    return INVALID_ID;

  IdIndex bPortId = carryNode->inputPorts[2];
  const Port *bPort = graph.getPort(bPortId);
  if (!bPort)
    return INVALID_ID;

  for (IdIndex edgeId : bPort->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (!edge || edge->dstPort != bPortId)
      continue;
    const Port *srcPort = graph.getPort(edge->srcPort);
    if (srcPort && srcPort->parentNode != INVALID_ID)
      return srcPort->parentNode;
  }
  return INVALID_ID;
}

RecurrenceResult detectRecurrences(const Graph &graph) {
  RecurrenceResult result;

  // Find all carry nodes.
  llvm::SmallVector<IdIndex, 4> carryNodes;
  for (IdIndex i = 0; i < static_cast<IdIndex>(graph.nodes.size()); ++i) {
    const Node *node = graph.getNode(i);
    if (!node || node->kind != Node::OperationNode)
      continue;
    if (getOpName(node) == "dataflow.carry")
      carryNodes.push_back(i);
  }

  int32_t nextCycleId = 0;

  for (IdIndex carryId : carryNodes) {
    const Node *carryNode = graph.getNode(carryId);
    if (!carryNode)
      continue;

    // Find the node that feeds carry's input port b (the back-edge source).
    IdIndex backEdgeSrc = findCarryBackEdgeSource(graph, carryId);
    if (backEdgeSrc == INVALID_ID)
      continue;

    // Record back-edge.
    result.backEdges.insert({backEdgeSrc, carryId});

    // Use forward+backward reachability to find ALL nodes on any cycle
    // through this carry. A node is on the recurrence iff it is both:
    //   (a) reachable forward from carry's output, AND
    //   (b) can reach carry's input port b (backward-reachable from backEdgeSrc)

    // Build set of OTHER carry nodes (for boundary checks during marking).
    llvm::DenseSet<IdIndex> otherCarries;
    for (IdIndex c : carryNodes) {
      if (c != carryId)
        otherCarries.insert(c);
    }

    // Unrestricted forward BFS for cycle DETECTION (allows traversal
    // through other carry nodes, since the recurrence path may cross them).
    llvm::DenseSet<IdIndex> forwardFull;
    {
      std::queue<IdIndex> q;
      forwardFull.insert(carryId);
      for (IdIndex s : getSuccessors(graph, carryId)) {
        if (!forwardFull.count(s)) {
          forwardFull.insert(s);
          q.push(s);
        }
      }
      while (!q.empty()) {
        IdIndex cur = q.front();
        q.pop();
        for (IdIndex s : getSuccessors(graph, cur)) {
          if (!forwardFull.count(s)) {
            forwardFull.insert(s);
            q.push(s);
          }
        }
      }
    }

    bool hasCycle = forwardFull.count(backEdgeSrc);
    if (!hasCycle)
      continue;

    // Restricted forward BFS for MARKING: stop at other carry nodes to
    // prevent inner-cycle nodes from being attributed to outer cycles.
    llvm::DenseSet<IdIndex> forwardRestricted;
    {
      std::queue<IdIndex> q;
      forwardRestricted.insert(carryId);
      for (IdIndex s : getSuccessors(graph, carryId)) {
        if (!forwardRestricted.count(s)) {
          forwardRestricted.insert(s);
          q.push(s);
        }
      }
      while (!q.empty()) {
        IdIndex cur = q.front();
        q.pop();
        if (otherCarries.count(cur))
          continue;
        for (IdIndex s : getSuccessors(graph, cur)) {
          if (!forwardRestricted.count(s)) {
            forwardRestricted.insert(s);
            q.push(s);
          }
        }
      }
    }

    // Backward reachability from backEdgeSrc, stopping at carry itself
    // and at other carry nodes.
    llvm::DenseSet<IdIndex> backwardReachable;
    {
      std::queue<IdIndex> q;
      backwardReachable.insert(backEdgeSrc);
      q.push(backEdgeSrc);
      while (!q.empty()) {
        IdIndex cur = q.front();
        q.pop();
        if (cur != backEdgeSrc && cur == carryId)
          continue;
        if (otherCarries.count(cur))
          continue;
        for (IdIndex p : getPredecessors(graph, cur)) {
          if (!backwardReachable.count(p)) {
            backwardReachable.insert(p);
            q.push(p);
          }
        }
      }
    }

    // Mark nodes at intersection of restricted-forward and backward sets.
    int32_t cycleId = nextCycleId++;
    for (IdIndex nodeId : forwardRestricted) {
      if (backwardReachable.count(nodeId)) {
        result.onRecurrence[nodeId] = true;
        result.recurrenceId[nodeId] = cycleId;
      }
    }
    // Always include carry itself.
    result.onRecurrence[carryId] = true;
    result.recurrenceId[carryId] = cycleId;
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Critical path estimation
//===----------------------------------------------------------------------===//

/// Assign latency weights based on operation type.
unsigned getLatencyWeight(llvm::StringRef opName) {
  if (opName.contains("load") || opName.contains("store") ||
      opName.contains("memory"))
    return 3;
  // Constants fold to immediates and have no runtime execution cost.
  if (opName == "arith.constant")
    return 0;
  if (opName.starts_with("arith.") || opName.starts_with("math."))
    return 1;
  // Control/dataflow ops have zero latency (don't contribute to data path).
  return 0;
}

struct CriticalPathResult {
  llvm::DenseSet<IdIndex> onCriticalPath;
  double criticalPathLength = 0.0;
};

CriticalPathResult estimateCriticalPath(
    const Graph &graph,
    const llvm::DenseSet<std::pair<IdIndex, IdIndex>> &backEdges) {
  CriticalPathResult result;

  // Collect all operation nodes.
  llvm::SmallVector<IdIndex, 32> opNodes;
  for (IdIndex i = 0; i < static_cast<IdIndex>(graph.nodes.size()); ++i) {
    const Node *node = graph.getNode(i);
    if (node && node->kind == Node::OperationNode)
      opNodes.push_back(i);
  }

  if (opNodes.empty())
    return result;

  // Build adjacency list, excluding back-edges.
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> adj;
  llvm::DenseMap<IdIndex, unsigned> inDegree;

  for (IdIndex nodeId : opNodes) {
    inDegree[nodeId] = 0;
    adj[nodeId] = {};
  }

  for (IdIndex nodeId : opNodes) {
    auto succs = getSuccessors(graph, nodeId);
    for (IdIndex s : succs) {
      if (backEdges.count({nodeId, s}))
        continue; // Skip carry back-edges.
      if (!inDegree.count(s))
        continue; // Not an operation node.
      adj[nodeId].push_back(s);
      inDegree[s]++;
    }
  }

  // Topological sort (Kahn's algorithm).
  std::queue<IdIndex> topoQueue;
  for (auto &[nodeId, deg] : inDegree) {
    if (deg == 0)
      topoQueue.push(nodeId);
  }

  llvm::SmallVector<IdIndex, 32> topoOrder;
  while (!topoQueue.empty()) {
    IdIndex cur = topoQueue.front();
    topoQueue.pop();
    topoOrder.push_back(cur);
    for (IdIndex s : adj[cur]) {
      if (--inDegree[s] == 0)
        topoQueue.push(s);
    }
  }

  // DP: longest weighted path.
  llvm::DenseMap<IdIndex, double> dist;
  llvm::DenseMap<IdIndex, IdIndex> predecessor;

  for (IdIndex nodeId : topoOrder) {
    const Node *node = graph.getNode(nodeId);
    unsigned weight = node ? getLatencyWeight(getOpName(node)) : 0;
    dist[nodeId] = std::max(dist[nodeId], static_cast<double>(weight));
  }

  for (IdIndex nodeId : topoOrder) {
    double curDist = dist[nodeId];
    for (IdIndex s : adj[nodeId]) {
      const Node *sNode = graph.getNode(s);
      unsigned sWeight = sNode ? getLatencyWeight(getOpName(sNode)) : 0;
      double newDist = curDist + sWeight;
      if (newDist > dist[s]) {
        dist[s] = newDist;
        predecessor[s] = nodeId;
      }
    }
  }

  // Find the node with maximum distance = end of critical path.
  IdIndex maxNode = INVALID_ID;
  double maxDist = 0.0;
  for (auto &[nodeId, d] : dist) {
    if (d > maxDist) {
      maxDist = d;
      maxNode = nodeId;
    }
  }

  result.criticalPathLength = maxDist;

  // Trace back the critical path.
  if (maxNode != INVALID_ID) {
    IdIndex cur = maxNode;
    while (cur != INVALID_ID) {
      result.onCriticalPath.insert(cur);
      auto it = predecessor.find(cur);
      if (it == predecessor.end())
        break;
      cur = it->second;
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Temporal candidacy score
//===----------------------------------------------------------------------===//

double computeTemporalScore(const Node *node, int32_t maxLoopDepth,
                            int64_t maxExecFreq,
                            const DFGAnalysisConfig &config) {
  llvm::StringRef opName = getOpName(node);

  // Forced spatial ops always get score 0.0.
  if (isForcedSpatialOp(opName))
    return 0.0;

  int32_t loopDepth = static_cast<int32_t>(
      getIntAttr(node, "loom.loop_depth", 0));
  int64_t execFreq = getIntAttr(node, "loom.exec_freq", 1);
  bool onRec = getBoolAttr(node, "loom.on_recurrence", false);
  bool onCP = getBoolAttr(node, "loom.on_critical_path", false);

  // Normalize exec_freq to [0, 1].
  double normFreq = 0.0;
  if (maxExecFreq > 1)
    normFreq = static_cast<double>(execFreq - 1) /
               static_cast<double>(maxExecFreq - 1);

  double score = config.w1 * (1.0 - normFreq) +
                 config.w2 * (onCP ? 0.0 : 1.0) +
                 config.w3 * (loopDepth == 0 ? 1.0 : 0.0) +
                 config.w4 * (onRec ? 0.0 : 0.5);

  // Clamp to [0.0, 1.0].
  return std::max(0.0, std::min(1.0, score));
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Level B public API
//===----------------------------------------------------------------------===//

void analyzeGraph(Graph &graph, const DFGAnalysisConfig &config) {
  if (!graph.context)
    return;

  mlir::Builder builder(graph.context);

  // Run recurrence detection.
  auto recResult = detectRecurrences(graph);

  // Run critical path estimation.
  auto cpResult = estimateCriticalPath(graph, recResult.backEdges);

  // Update node attributes with recurrence and critical path results.
  int32_t maxLoopDepth = 0;
  int64_t maxExecFreq = 1;

  for (auto *node : graph.nodeRange()) {
    if (!node || node->kind != Node::OperationNode)
      continue;

    int32_t depth = static_cast<int32_t>(
        getIntAttr(node, "loom.loop_depth", 0));
    int64_t freq = getIntAttr(node, "loom.exec_freq", 1);
    maxLoopDepth = std::max(maxLoopDepth, depth);
    maxExecFreq = std::max(maxExecFreq, freq);
  }

  // Determine node IDs for setting attributes (need index).
  for (IdIndex i = 0; i < static_cast<IdIndex>(graph.nodes.size()); ++i) {
    Node *node = graph.getNode(i);
    if (!node || node->kind != Node::OperationNode)
      continue;

    // Recurrence attributes.
    bool onRec = recResult.onRecurrence.count(i) &&
                 recResult.onRecurrence[i];
    int32_t recId = recResult.recurrenceId.count(i)
                        ? recResult.recurrenceId[i]
                        : -1;

    setNodeAttr(node, builder.getNamedAttr(
                          "loom.on_recurrence", builder.getBoolAttr(onRec)));
    setNodeAttr(node, builder.getNamedAttr(
                          "loom.recurrence_id",
                          builder.getI32IntegerAttr(recId)));

    // Critical path attributes.
    bool onCP = cpResult.onCriticalPath.count(i);
    setNodeAttr(node, builder.getNamedAttr(
                          "loom.on_critical_path", builder.getBoolAttr(onCP)));
  }

  // Compute temporal candidacy score (requires all other attrs set first).
  for (IdIndex i = 0; i < static_cast<IdIndex>(graph.nodes.size()); ++i) {
    Node *node = graph.getNode(i);
    if (!node || node->kind != Node::OperationNode)
      continue;

    double tscore = computeTemporalScore(node, maxLoopDepth, maxExecFreq,
                                         config);
    setNodeAttr(node, builder.getNamedAttr(
                          "loom.temporal_score",
                          builder.getF64FloatAttr(tscore)));
  }
}

void writeBackToMLIR(const Graph &graph, circt::handshake::FuncOp funcOp) {
  auto &block = funcOp.getBody().front();
  mlir::Builder builder(funcOp.getContext());

  // Collect OperationNodes in graph index order. DFGBuilder creates
  // OperationNodes in block iteration order (non-terminator ops), so
  // positional matching is correct when no intermediate graph mutations
  // occur. We verify with an op_name check for safety.
  llvm::SmallVector<const Node *, 32> opNodes;
  for (auto *node : graph.nodeRange()) {
    if (node && node->kind == Node::OperationNode)
      opNodes.push_back(node);
  }

  unsigned nodeIdx = 0;
  for (auto &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;
    if (nodeIdx >= opNodes.size())
      break;

    const Node *node = opNodes[nodeIdx++];
    if (!node)
      continue;

    // Verify op_name matches to catch graph mutation issues.
    llvm::StringRef graphOpName = getOpName(node);
    if (!graphOpName.empty() &&
        graphOpName != op.getName().getStringRef())
      continue;

    // Read Level B results from graph node attributes.
    bool onRec = getBoolAttr(node, "loom.on_recurrence", false);
    int32_t recId = static_cast<int32_t>(
        getIntAttr(node, "loom.recurrence_id", -1));
    bool onCP = getBoolAttr(node, "loom.on_critical_path", false);
    double tscore = getFloatAttr(node, "loom.temporal_score", 0.0);

    // Read existing loom.analysis dict and update the relevant fields.
    auto existingDict =
        op.getAttrOfType<mlir::DictionaryAttr>("loom.analysis");
    if (!existingDict)
      continue;

    // Rebuild the dict with updated Level B values.
    llvm::SmallVector<mlir::NamedAttribute, 6> entries;
    for (auto entry : existingDict) {
      auto name = entry.getName();
      if (name == "on_recurrence") {
        entries.push_back(
            builder.getNamedAttr(name, builder.getBoolAttr(onRec)));
      } else if (name == "recurrence_id") {
        entries.push_back(
            builder.getNamedAttr(name, builder.getI32IntegerAttr(recId)));
      } else if (name == "on_critical_path") {
        entries.push_back(
            builder.getNamedAttr(name, builder.getBoolAttr(onCP)));
      } else if (name == "temporal_score") {
        entries.push_back(
            builder.getNamedAttr(name, builder.getF64FloatAttr(tscore)));
      } else {
        entries.push_back(entry);
      }
    }
    op.setAttr("loom.analysis", builder.getDictionaryAttr(entries));
  }
}

} // namespace analysis
} // namespace loom

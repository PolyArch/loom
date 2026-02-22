//===-- Mapper.cpp - PnR mapper pipeline orchestration -------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/CandidateBuilder.h"
#include "loom/Mapper/CPSATSolver.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <queue>
#include <random>

namespace loom {

namespace {

/// Get the "op_name" attribute from a node, or empty string.
llvm::StringRef getNodeOpName(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "op_name") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Get the "resource_class" attribute from a node, or empty string.
llvm::StringRef getNodeResourceClass(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "resource_class") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Check if a node is a memory operation.
bool isMemoryOp(const Node *node) {
  llvm::StringRef opName = getNodeOpName(node);
  return opName.contains("load") || opName.contains("store") ||
         opName.contains("memory");
}

} // namespace

// --- Pipeline orchestration ---

Mapper::Result Mapper::run(const Graph &dfg, const Graph &adg,
                           const Options &opts) {
  Result result;

  // Preprocessing: build connectivity matrix and min-hop costs.
  preprocess(adg);

  // Tech-mapping: build candidate sets.
  CandidateBuilder candidateBuilder;
  auto candidateResult = candidateBuilder.build(dfg, adg);
  if (!candidateResult.success) {
    result.success = false;
    result.diagnostics = candidateResult.diagnostics;
    return result;
  }

  // Initialize mapping state.
  result.state.init(dfg, adg);

  // Placement.
  if (!runPlacement(result.state, dfg, adg, candidateResult.candidates, opts)) {
    result.success = false;
    result.diagnostics = "Placement failed";
    return result;
  }

  // Routing.
  if (!runRouting(result.state, dfg, adg)) {
    // Try refinement/repair.
    if (!runRefinement(result.state, dfg, adg, candidateResult.candidates,
                       opts)) {
      result.success = false;
      result.diagnostics = "Routing failed after refinement";
      return result;
    }
  }

  // Temporal assignment (for temporal PEs).
  if (!runTemporalAssignment(result.state, dfg, adg)) {
    result.success = false;
    result.diagnostics = "Temporal assignment failed";
    return result;
  }

  // Validation of heuristic result.
  std::string validationDiag;
  bool heuristicValid =
      runValidation(result.state, dfg, adg, validationDiag);

  // Compute cost for heuristic result.
  computeCost(result.state, dfg, adg, opts);

  // CP-SAT solver integration: run if available and profile requests it.
  if (CPSATSolver::isAvailable()) {
    auto cpsatMode = CPSATSolver::selectMode(dfg, opts.profile);

    if (cpsatMode != CPSATSolver::Mode::DISABLED) {
      CPSATSolver solver;
      CPSATSolver::Options cpsatOpts;
      cpsatOpts.timeLimitSeconds = opts.budgetSeconds * 0.5;
      cpsatOpts.mode = cpsatMode;
      cpsatOpts.subProblemMaxNodes = 50;

      CPSATSolver::Result cpsatResult;

      if (cpsatMode == CPSATSolver::Mode::FULL_PROBLEM) {
        // Warm-start from heuristic solution.
        cpsatResult = solver.solveFullProblem(
            dfg, adg, candidateResult.candidates, connectivity,
            &result.state, cpsatOpts);
      } else {
        // Sub-problem mode: extract conflicting region.
        // Seed from unrouted edges AND congestion/cost hotspots.
        llvm::SmallVector<IdIndex, 8> conflictNodes;

        // Collect nodes from unrouted edges.
        for (IdIndex edgeId = 0;
             edgeId < static_cast<IdIndex>(dfg.edges.size()); ++edgeId) {
          if (edgeId >= result.state.swEdgeToHwPaths.size() ||
              result.state.swEdgeToHwPaths[edgeId].empty()) {
            const Edge *edge = dfg.getEdge(edgeId);
            if (edge) {
              const Port *sp = dfg.getPort(edge->srcPort);
              const Port *dp = dfg.getPort(edge->dstPort);
              if (sp && sp->parentNode != INVALID_ID)
                conflictNodes.push_back(sp->parentNode);
              if (dp && dp->parentNode != INVALID_ID)
                conflictNodes.push_back(dp->parentNode);
            }
          }
        }

        // Also trigger on congestion: if routing cost or placement
        // pressure is high, select highest-utilization HW nodes and
        // include their mapped SW nodes as conflict seeds.
        if (conflictNodes.empty() &&
            (result.state.routingCost > 0.5 ||
             result.state.placementPressure > 0.5)) {
          // Find the most congested HW nodes (those with the most
          // mapped SW nodes) as refinement targets.
          IdIndex bestHwNode = INVALID_ID;
          size_t maxLoad = 1;
          for (IdIndex hwId = 0;
               hwId < static_cast<IdIndex>(
                          result.state.hwNodeToSwNodes.size());
               ++hwId) {
            size_t load = result.state.hwNodeToSwNodes[hwId].size();
            if (load > maxLoad) {
              maxLoad = load;
              bestHwNode = hwId;
            }
          }
          if (bestHwNode != INVALID_ID) {
            for (IdIndex swId :
                 result.state.hwNodeToSwNodes[bestHwNode]) {
              conflictNodes.push_back(swId);
            }
          }
        }

        // Enforce group atomicity: if any conflict node is part of a
        // group, include all group members in the sub-problem.
        llvm::DenseSet<IdIndex> conflictSet(conflictNodes.begin(),
                                            conflictNodes.end());
        for (auto &[hwId, groupMembers] :
             result.state.groupBindings) {
          bool anyInConflict = false;
          for (IdIndex swId : groupMembers) {
            if (conflictSet.count(swId)) {
              anyInConflict = true;
              break;
            }
          }
          if (anyInConflict) {
            for (IdIndex swId : groupMembers) {
              if (conflictSet.insert(swId).second)
                conflictNodes.push_back(swId);
            }
          }
        }

        if (!conflictNodes.empty()) {
          auto subProblem = CPSATSolver::extractSubProblem(
              dfg, conflictNodes, cpsatOpts.subProblemMaxNodes);
          cpsatResult = solver.solveSubProblem(
              dfg, adg, subProblem, result.state,
              candidateResult.candidates, connectivity, cpsatOpts);
        }
      }

      // If CP-SAT found a valid solution, compare costs.
      if (cpsatResult.success) {
        // Post-validate deferred constraints on CP-SAT result.
        std::string cpsatValidDiag;
        bool cpsatValid =
            runValidation(cpsatResult.state, dfg, adg, cpsatValidDiag);

        if (cpsatValid) {
          computeCost(cpsatResult.state, dfg, adg, opts);

          // Accept CP-SAT result if it has lower total cost.
          if (!heuristicValid ||
              cpsatResult.state.totalCost < result.state.totalCost) {
            result.state = std::move(cpsatResult.state);
            heuristicValid = true;
            validationDiag.clear();
          }
        }
      }
    }
  }

  if (!heuristicValid) {
    result.success = false;
    result.diagnostics = "Validation failed: " + validationDiag;
    return result;
  }

  result.success = true;
  return result;
}

// --- Preprocessing ---

void Mapper::preprocess(const Graph &adg) {
  // Build connectivity matrix from ADG edges.
  connectivity = ConnectivityMatrix();

  for (auto *edge : adg.edgeRange()) {
    if (!edge)
      continue;
    connectivity.outToIn[edge->srcPort] = edge->dstPort;
  }

  // Build internal routing (input port -> output ports) for routing nodes.
  // Use connectivity_table when available; otherwise fall back to full crossbar.
  for (auto *node : adg.nodeRange()) {
    if (!node || node->kind != Node::OperationNode)
      continue;

    llvm::StringRef resourceClass = getNodeResourceClass(node);
    if (resourceClass != "routing")
      continue;

    unsigned numIn = node->inputPorts.size();
    unsigned numOut = node->outputPorts.size();

    // Check for connectivity_table attribute (DenseI8ArrayAttr).
    mlir::DenseI8ArrayAttr connTable;
    for (auto &attr : node->attributes) {
      if (attr.getName() == "connectivity_table") {
        connTable = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attr.getValue());
        break;
      }
    }

    if (connTable && static_cast<unsigned>(connTable.size()) == numOut * numIn) {
      // Use connectivity_table: row-major [output_idx][input_idx].
      // connectivity_table[o * numIn + i] == 1 means output o can receive
      // from input i.
      for (unsigned o = 0; o < numOut; ++o) {
        for (unsigned i = 0; i < numIn; ++i) {
          if (connTable[o * numIn + i] != 0) {
            connectivity.inToOut[node->inputPorts[i]].push_back(
                node->outputPorts[o]);
          }
        }
      }
    } else {
      // No connectivity_table or size mismatch: full crossbar fallback.
      for (IdIndex inPortId : node->inputPorts) {
        for (IdIndex outPortId : node->outputPorts) {
          connectivity.inToOut[inPortId].push_back(outPortId);
        }
      }
    }
  }

  // Precompute min-hop costs using BFS from each ADG node.
  minHopCosts.clear();
  for (IdIndex nodeId = 0; nodeId < static_cast<IdIndex>(adg.nodes.size());
       ++nodeId) {
    const Node *node = adg.getNode(nodeId);
    if (!node)
      continue;

    auto &costs = minHopCosts[nodeId];

    // BFS from this node.
    std::queue<std::pair<IdIndex, unsigned>> bfsQueue;
    llvm::DenseSet<IdIndex> visited;

    bfsQueue.push({nodeId, 0});
    visited.insert(nodeId);

    while (!bfsQueue.empty()) {
      auto [curNode, hops] = bfsQueue.front();
      bfsQueue.pop();

      costs[curNode] = hops;

      const Node *cur = adg.getNode(curNode);
      if (!cur)
        continue;

      // Follow output ports -> edges -> destination port -> destination node.
      for (IdIndex outPortId : cur->outputPorts) {
        auto it = connectivity.outToIn.find(outPortId);
        if (it == connectivity.outToIn.end())
          continue;

        IdIndex dstPortId = it->second;
        const Port *dstPort = adg.getPort(dstPortId);
        if (!dstPort)
          continue;

        IdIndex nextNode = dstPort->parentNode;
        if (nextNode == INVALID_ID || visited.count(nextNode))
          continue;

        visited.insert(nextNode);
        bfsQueue.push({nextNode, hops + 1});
      }
    }
  }
}

// --- Placement ---

std::vector<IdIndex> Mapper::computePlacementOrder(const Graph &dfg) {
  std::vector<IdIndex> order;

  // Separate memory ops and compute ops.
  std::vector<IdIndex> memOps;
  std::vector<IdIndex> computeOps;

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node || node->kind != Node::OperationNode)
      continue;

    if (isMemoryOp(node))
      memOps.push_back(i);
    else
      computeOps.push_back(i);
  }

  // Memory ops first (most constrained).
  order.insert(order.end(), memOps.begin(), memOps.end());

  // Then compute ops in topological order (approximated by ID order since
  // DFG nodes are created in program order).
  order.insert(order.end(), computeOps.begin(), computeOps.end());

  return order;
}

/// Helper: find the first downstream node connected to a sentinel's output
/// port in the given graph. Returns INVALID_ID if no edge found.
static IdIndex findDownstreamNode(const Graph &graph, IdIndex sentinelNodeId) {
  const Node *sn = graph.getNode(sentinelNodeId);
  if (!sn || sn->outputPorts.empty())
    return INVALID_ID;

  IdIndex outPortId = sn->outputPorts[0];
  const Port *outPort = graph.getPort(outPortId);
  if (!outPort)
    return INVALID_ID;

  for (IdIndex edgeId : outPort->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (!edge || edge->srcPort != outPortId)
      continue;
    const Port *dstPort = graph.getPort(edge->dstPort);
    if (dstPort)
      return dstPort->parentNode;
  }
  return INVALID_ID;
}

/// Helper: check if a type is a memref type.
static bool isMemrefType(mlir::Type type) {
  return type && mlir::isa<mlir::MemRefType>(type);
}

void Mapper::bindSentinelPorts(MappingState &state, const Graph &dfg,
                               const Graph &adg) {
  // Collect DFG and ADG sentinels.
  std::vector<IdIndex> dfgInputSentinels;
  std::vector<IdIndex> dfgOutputSentinels;
  std::vector<IdIndex> adgInputSentinels;
  std::vector<IdIndex> adgOutputSentinels;

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode)
      dfgInputSentinels.push_back(i);
    else if (node->kind == Node::ModuleOutputNode)
      dfgOutputSentinels.push_back(i);
  }

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode)
      adgInputSentinels.push_back(i);
    else if (node->kind == Node::ModuleOutputNode)
      adgOutputSentinels.push_back(i);
  }

  // Build memref sentinel -> downstream extmemory maps for both DFG and ADG.
  // For memref sentinels, we need to bind them to ADG sentinels that connect
  // to compatible extmemory nodes, not just by type order.
  llvm::DenseMap<IdIndex, IdIndex> dfgSentinelToExtmem; // DFG sentinel -> DFG extmemory
  llvm::DenseMap<IdIndex, IdIndex> adgSentinelToExtmem; // ADG sentinel -> ADG extmemory

  for (size_t di = 0; di < dfgInputSentinels.size(); ++di) {
    const Node *dfgNode = dfg.getNode(dfgInputSentinels[di]);
    if (!dfgNode || dfgNode->outputPorts.empty())
      continue;
    mlir::Type dfgType = dfg.getPort(dfgNode->outputPorts[0])->type;
    if (isMemrefType(dfgType)) {
      IdIndex downstream = findDownstreamNode(dfg, dfgInputSentinels[di]);
      if (downstream != INVALID_ID)
        dfgSentinelToExtmem[dfgInputSentinels[di]] = downstream;
    }
  }

  for (size_t ai = 0; ai < adgInputSentinels.size(); ++ai) {
    const Node *adgNode = adg.getNode(adgInputSentinels[ai]);
    if (!adgNode || adgNode->outputPorts.empty())
      continue;
    mlir::Type adgType = adg.getPort(adgNode->outputPorts[0])->type;
    if (isMemrefType(adgType)) {
      IdIndex downstream = findDownstreamNode(adg, adgInputSentinels[ai]);
      if (downstream != INVALID_ID)
        adgSentinelToExtmem[adgInputSentinels[ai]] = downstream;
    }
  }

  // Bind input sentinels: memref sentinels use extmemory-aware binding,
  // non-memref sentinels use simple type matching.
  llvm::DenseSet<size_t> usedAdgIn;
  llvm::DenseSet<IdIndex> preBoundExtmem; // DFG extmemory nodes pre-bound here

  // First pass: bind memref sentinels by matching their downstream extmemory
  // node's port signature with ADG extmemory node's port signature.
  for (size_t di = 0; di < dfgInputSentinels.size(); ++di) {
    const Node *dfgNode = dfg.getNode(dfgInputSentinels[di]);
    if (!dfgNode || dfgNode->outputPorts.empty())
      continue;
    mlir::Type dfgType = dfg.getPort(dfgNode->outputPorts[0])->type;
    if (!isMemrefType(dfgType))
      continue;

    auto dfgExtIt = dfgSentinelToExtmem.find(dfgInputSentinels[di]);
    if (dfgExtIt == dfgSentinelToExtmem.end())
      continue;
    const Node *dfgExtmem = dfg.getNode(dfgExtIt->second);
    if (!dfgExtmem)
      continue;

    // Find compatible ADG sentinel: one whose downstream extmemory has
    // matching input/output port counts.
    for (size_t ai = 0; ai < adgInputSentinels.size(); ++ai) {
      if (usedAdgIn.count(ai))
        continue;
      const Node *adgNode = adg.getNode(adgInputSentinels[ai]);
      if (!adgNode || adgNode->outputPorts.empty())
        continue;
      mlir::Type adgType = adg.getPort(adgNode->outputPorts[0])->type;
      if (dfgType != adgType)
        continue;

      auto adgExtIt = adgSentinelToExtmem.find(adgInputSentinels[ai]);
      if (adgExtIt == adgSentinelToExtmem.end())
        continue;
      const Node *adgExtmem = adg.getNode(adgExtIt->second);
      if (!adgExtmem)
        continue;

      // Check port compatibility: DFG extmemory ports must fit in ADG
      // extmemory ports.
      if (dfgExtmem->inputPorts.size() > adgExtmem->inputPorts.size())
        continue;
      if (dfgExtmem->outputPorts.size() > adgExtmem->outputPorts.size())
        continue;

      // Bind the sentinel.
      usedAdgIn.insert(ai);
      state.mapNode(dfgInputSentinels[di], adgInputSentinels[ai], dfg, adg);
      for (size_t j = 0;
           j < dfgNode->outputPorts.size() &&
           j < adgNode->outputPorts.size();
           ++j) {
        state.mapPort(dfgNode->outputPorts[j], adgNode->outputPorts[j], dfg,
                      adg);
      }

      // Also pre-bind the extmemory node so placement is consistent.
      state.mapNode(dfgExtIt->second, adgExtIt->second, dfg, adg);
      // Map extmemory ports using type-aware matching (same as placement).
      llvm::SmallVector<bool> hwUsed(adgExtmem->inputPorts.size(), false);
      for (size_t si = 0; si < dfgExtmem->inputPorts.size(); ++si) {
        const Port *sp = dfg.getPort(dfgExtmem->inputPorts[si]);
        if (!sp)
          continue;
        for (size_t hi = 0; hi < adgExtmem->inputPorts.size(); ++hi) {
          if (hwUsed[hi])
            continue;
          const Port *hp = adg.getPort(adgExtmem->inputPorts[hi]);
          if (hp && sp->type == hp->type) {
            state.mapPort(dfgExtmem->inputPorts[si],
                          adgExtmem->inputPorts[hi], dfg, adg);
            hwUsed[hi] = true;
            break;
          }
        }
      }
      {
        llvm::SmallVector<bool> hwOutUsed(adgExtmem->outputPorts.size(),
                                          false);
        for (size_t si = 0; si < dfgExtmem->outputPorts.size(); ++si) {
          const Port *sp = dfg.getPort(dfgExtmem->outputPorts[si]);
          if (!sp)
            continue;
          for (size_t hi = 0; hi < adgExtmem->outputPorts.size(); ++hi) {
            if (hwOutUsed[hi])
              continue;
            const Port *hp = adg.getPort(adgExtmem->outputPorts[hi]);
            if (hp && sp->type == hp->type) {
              state.mapPort(dfgExtmem->outputPorts[si],
                            adgExtmem->outputPorts[hi], dfg, adg);
              hwOutUsed[hi] = true;
              break;
            }
          }
        }
      }
      preBoundExtmem.insert(dfgExtIt->second);
      break;
    }
  }

  // Second pass: bind non-memref sentinels by type matching.
  for (size_t di = 0; di < dfgInputSentinels.size(); ++di) {
    const Node *dfgNode = dfg.getNode(dfgInputSentinels[di]);
    if (!dfgNode || dfgNode->outputPorts.empty())
      continue;
    mlir::Type dfgType = dfg.getPort(dfgNode->outputPorts[0])->type;
    if (isMemrefType(dfgType))
      continue; // Already handled above.

    for (size_t ai = 0; ai < adgInputSentinels.size(); ++ai) {
      if (usedAdgIn.count(ai))
        continue;
      const Node *adgNode = adg.getNode(adgInputSentinels[ai]);
      if (!adgNode || adgNode->outputPorts.empty())
        continue;
      mlir::Type adgType = adg.getPort(adgNode->outputPorts[0])->type;

      if (dfgType == adgType) {
        usedAdgIn.insert(ai);
        state.mapNode(dfgInputSentinels[di], adgInputSentinels[ai], dfg, adg);
        for (size_t j = 0;
             j < dfgNode->outputPorts.size() &&
             j < adgNode->outputPorts.size();
             ++j) {
          state.mapPort(dfgNode->outputPorts[j], adgNode->outputPorts[j], dfg,
                        adg);
        }
        break;
      }
    }
  }

  // Output sentinel binding: type-aware matching.
  llvm::DenseSet<size_t> usedAdgOut;
  for (size_t di = 0; di < dfgOutputSentinels.size(); ++di) {
    const Node *dfgNode = dfg.getNode(dfgOutputSentinels[di]);
    if (!dfgNode || dfgNode->inputPorts.empty())
      continue;
    mlir::Type dfgType = dfg.getPort(dfgNode->inputPorts[0])->type;

    for (size_t ai = 0; ai < adgOutputSentinels.size(); ++ai) {
      if (usedAdgOut.count(ai))
        continue;
      const Node *adgNode = adg.getNode(adgOutputSentinels[ai]);
      if (!adgNode || adgNode->inputPorts.empty())
        continue;
      mlir::Type adgType = adg.getPort(adgNode->inputPorts[0])->type;

      if (dfgType == adgType) {
        usedAdgOut.insert(ai);
        state.mapNode(dfgOutputSentinels[di], adgOutputSentinels[ai], dfg,
                      adg);
        for (size_t j = 0;
             j < dfgNode->inputPorts.size() &&
             j < adgNode->inputPorts.size();
             ++j) {
          state.mapPort(dfgNode->inputPorts[j], adgNode->inputPorts[j], dfg,
                        adg);
        }
        break;
      }
    }
  }
}

double Mapper::scorePlacement(IdIndex swNode, IdIndex hwNode,
                              const MappingState &state, const Graph &dfg,
                              const Graph &adg) {
  double score = 0.0;

  // Proximity: reward placements close to already-mapped neighbors.
  const Node *sw = dfg.getNode(swNode);
  if (!sw)
    return -1e9;

  for (IdIndex inPortId : sw->inputPorts) {
    const Port *inPort = dfg.getPort(inPortId);
    if (!inPort)
      continue;

    for (IdIndex edgeId : inPort->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;

      // Find the source node.
      const Port *srcPort = dfg.getPort(edge->srcPort);
      if (!srcPort)
        continue;

      IdIndex srcSwNode = srcPort->parentNode;
      if (srcSwNode == INVALID_ID || srcSwNode >= state.swNodeToHwNode.size())
        continue;

      IdIndex srcHwNode = state.swNodeToHwNode[srcSwNode];
      if (srcHwNode == INVALID_ID)
        continue;

      // Proximity bonus: closer is better.
      auto hopsIt = minHopCosts.find(srcHwNode);
      if (hopsIt != minHopCosts.end()) {
        auto hopIt = hopsIt->second.find(hwNode);
        if (hopIt != hopsIt->second.end()) {
          unsigned hops = hopIt->second;
          score += 10.0 / (1.0 + hops); // Closer = higher score.
        }
      }
    }
  }

  // Utilization penalty: prefer less-used nodes.
  if (hwNode < state.hwNodeToSwNodes.size()) {
    score -= 5.0 * state.hwNodeToSwNodes[hwNode].size();
  }

  return score;
}

bool Mapper::runPlacement(MappingState &state, const Graph &dfg,
                          const Graph &adg, const CandidateSet &candidates,
                          const Options &opts) {
  // Bind sentinels first.
  bindSentinelPorts(state, dfg, adg);

  // Compute placement order.
  auto order = computePlacementOrder(dfg);

  std::mt19937 rng(opts.seed);

  // Track SW nodes already placed as part of a group.
  llvm::DenseSet<IdIndex> placedInGroup;

  for (IdIndex swNode : order) {
    // Skip nodes already placed atomically as part of a group.
    if (placedInGroup.count(swNode))
      continue;

    // Skip nodes pre-bound during sentinel binding (e.g., extmemory nodes
    // bound together with their memref sentinels).
    if (swNode < state.swNodeToHwNode.size() &&
        state.swNodeToHwNode[swNode] != INVALID_ID)
      continue;

    auto it = candidates.find(swNode);
    if (it == candidates.end() || it->second.empty())
      return false;

    // Score each candidate, including group candidates.
    double bestScore = -1e18;
    const Candidate *bestCand = nullptr;

    for (const auto &candidate : it->second) {
      if (candidate.hwNodeId >= state.hwNodeToSwNodes.size())
        continue;

      // Skip PEs already occupied by another non-group node to prevent
      // port collisions (exclusive PE ports can only serve one operation).
      if (!candidate.isGroup &&
          !state.hwNodeToSwNodes[candidate.hwNodeId].empty())
        continue;

      // For group candidates, verify all members are still unplaced.
      if (candidate.isGroup) {
        bool allAvailable = true;
        for (IdIndex memberId : candidate.swNodeIds) {
          if (placedInGroup.count(memberId) ||
              (memberId < state.swNodeToHwNode.size() &&
               state.swNodeToHwNode[memberId] != INVALID_ID)) {
            allAvailable = false;
            break;
          }
        }
        if (!allAvailable)
          continue;
      }

      double score = scorePlacement(swNode, candidate.hwNodeId, state,
                                    dfg, adg);

      // Add small random perturbation for tie-breaking.
      std::uniform_real_distribution<double> dist(0.0, 0.01);
      score += dist(rng);

      if (score > bestScore) {
        bestScore = score;
        bestCand = &candidate;
      }
    }

    if (!bestCand || bestCand->hwNodeId == INVALID_ID) {
      return false;
    }

    IdIndex bestHw = bestCand->hwNodeId;

    if (bestCand->isGroup && bestCand->swNodeIds.size() > 1) {
      // Group placement: bind ALL SW nodes in the group to the same
      // HW PE atomically.
      unsigned inPortOffset = 0;
      unsigned outPortOffset = 0;

      for (IdIndex groupSwNode : bestCand->swNodeIds) {
        auto mapResult = state.mapNode(groupSwNode, bestHw, dfg, adg);
        if (mapResult != ActionResult::Success)
          return false;

        // Map ports with offsets so group members share the PE ports.
        const Node *sw = dfg.getNode(groupSwNode);
        const Node *hw = adg.getNode(bestHw);
        if (sw && hw) {
          for (size_t i = 0; i < sw->inputPorts.size() &&
                             (inPortOffset + i) < hw->inputPorts.size();
               ++i) {
            state.mapPort(sw->inputPorts[i],
                          hw->inputPorts[inPortOffset + i], dfg, adg);
          }
          inPortOffset += sw->inputPorts.size();

          for (size_t i = 0; i < sw->outputPorts.size() &&
                             (outPortOffset + i) < hw->outputPorts.size();
               ++i) {
            state.mapPort(sw->outputPorts[i],
                          hw->outputPorts[outPortOffset + i], dfg, adg);
          }
          outPortOffset += sw->outputPorts.size();
        }

        placedInGroup.insert(groupSwNode);
      }

      // Record group binding for C4 validation.
      state.groupBindings[bestHw] = llvm::SmallVector<IdIndex, 4>(
          bestCand->swNodeIds.begin(), bestCand->swNodeIds.end());
    } else {
      // Single-op placement.
      auto mapResult = state.mapNode(swNode, bestHw, dfg, adg);
      if (mapResult != ActionResult::Success)
        return false;

      const Node *sw = dfg.getNode(swNode);
      const Node *hw = adg.getNode(bestHw);
      if (sw && hw) {
        // Type-aware port mapping: match SW ports to HW ports by type
        // to ensure edges stay within their type plane for routing.
        // Also skip HW ports already used by other SW nodes on the same PE.
        if (sw->inputPorts.size() <= hw->inputPorts.size()) {
          for (size_t si = 0; si < sw->inputPorts.size(); ++si) {
            const Port *sp = dfg.getPort(sw->inputPorts[si]);
            if (!sp) continue;
            for (size_t hi = 0; hi < hw->inputPorts.size(); ++hi) {
              IdIndex hwPid = hw->inputPorts[hi];
              if (!state.hwPortToSwPorts[hwPid].empty()) continue;
              const Port *hp = adg.getPort(hwPid);
              if (hp && sp->type == hp->type) {
                state.mapPort(sw->inputPorts[si], hwPid, dfg, adg);
                break;
              }
            }
          }
        } else {
          for (size_t i = 0;
               i < sw->inputPorts.size() && i < hw->inputPorts.size(); ++i) {
            state.mapPort(sw->inputPorts[i], hw->inputPorts[i], dfg, adg);
          }
        }
        if (sw->outputPorts.size() <= hw->outputPorts.size()) {
          for (size_t si = 0; si < sw->outputPorts.size(); ++si) {
            const Port *sp = dfg.getPort(sw->outputPorts[si]);
            if (!sp) continue;
            for (size_t hi = 0; hi < hw->outputPorts.size(); ++hi) {
              IdIndex hwPid = hw->outputPorts[hi];
              if (!state.hwPortToSwPorts[hwPid].empty()) continue;
              const Port *hp = adg.getPort(hwPid);
              if (hp && sp->type == hp->type) {
                state.mapPort(sw->outputPorts[si], hwPid, dfg, adg);
                break;
              }
            }
          }
        } else {
          for (size_t i = 0;
               i < sw->outputPorts.size() && i < hw->outputPorts.size(); ++i) {
            state.mapPort(sw->outputPorts[i], hw->outputPorts[i], dfg, adg);
          }
        }
      }
    }
  }

  return true;
}

} // namespace loom

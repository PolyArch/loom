//===-- Mapper.cpp - PnR mapper pipeline orchestration -------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/CandidateBuilder.h"
#include "loom/Mapper/CPSATSolver.h"

#include "mlir/IR/BuiltinAttributes.h"

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
        // Find failed edges or high-cost nodes for sub-problem extraction.
        llvm::SmallVector<IdIndex, 8> conflictNodes;
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

void Mapper::bindSentinelPorts(MappingState &state, const Graph &dfg,
                               const Graph &adg) {
  // Bind DFG sentinels to ADG sentinels by positional matching.
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

  // Bind input sentinels: DFG arg i -> ADG arg i.
  for (size_t i = 0;
       i < dfgInputSentinels.size() && i < adgInputSentinels.size(); ++i) {
    state.mapNode(dfgInputSentinels[i], adgInputSentinels[i], dfg, adg);

    // Bind ports positionally.
    const Node *dfgNode = dfg.getNode(dfgInputSentinels[i]);
    const Node *adgNode = adg.getNode(adgInputSentinels[i]);
    if (dfgNode && adgNode) {
      for (size_t j = 0;
           j < dfgNode->outputPorts.size() && j < adgNode->outputPorts.size();
           ++j) {
        state.mapPort(dfgNode->outputPorts[j], adgNode->outputPorts[j],
                      dfg, adg);
      }
    }
  }

  // Bind output sentinels: DFG ret i -> ADG ret i.
  for (size_t i = 0;
       i < dfgOutputSentinels.size() && i < adgOutputSentinels.size(); ++i) {
    state.mapNode(dfgOutputSentinels[i], adgOutputSentinels[i], dfg, adg);

    const Node *dfgNode = dfg.getNode(dfgOutputSentinels[i]);
    const Node *adgNode = adg.getNode(adgOutputSentinels[i]);
    if (dfgNode && adgNode) {
      for (size_t j = 0;
           j < dfgNode->inputPorts.size() && j < adgNode->inputPorts.size();
           ++j) {
        state.mapPort(dfgNode->inputPorts[j], adgNode->inputPorts[j],
                      dfg, adg);
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

  for (IdIndex swNode : order) {
    auto it = candidates.find(swNode);
    if (it == candidates.end() || it->second.empty())
      return false;

    // Score each candidate.
    double bestScore = -1e18;
    IdIndex bestHw = INVALID_ID;

    for (const auto &candidate : it->second) {
      // Check if hardware node is available (not exclusively occupied for
      // non-temporal nodes).
      if (candidate.hwNodeId >= state.hwNodeToSwNodes.size())
        continue;

      double score = scorePlacement(swNode, candidate.hwNodeId, state,
                                    dfg, adg);

      // Add small random perturbation for tie-breaking.
      std::uniform_real_distribution<double> dist(0.0, 0.01);
      score += dist(rng);

      if (score > bestScore) {
        bestScore = score;
        bestHw = candidate.hwNodeId;
      }
    }

    if (bestHw == INVALID_ID)
      return false;

    auto result = state.mapNode(swNode, bestHw, dfg, adg);
    if (result != ActionResult::Success)
      return false;

    // Map ports positionally.
    const Node *sw = dfg.getNode(swNode);
    const Node *hw = adg.getNode(bestHw);
    if (sw && hw) {
      for (size_t i = 0;
           i < sw->inputPorts.size() && i < hw->inputPorts.size(); ++i) {
        state.mapPort(sw->inputPorts[i], hw->inputPorts[i], dfg, adg);
      }
      for (size_t i = 0;
           i < sw->outputPorts.size() && i < hw->outputPorts.size(); ++i) {
        state.mapPort(sw->outputPorts[i], hw->outputPorts[i], dfg, adg);
      }
    }
  }

  return true;
}

} // namespace loom

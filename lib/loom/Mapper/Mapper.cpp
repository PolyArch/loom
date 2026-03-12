//===-- Mapper.cpp - PnR mapper pipeline orchestration -------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"
#include "loom/Analysis/DFGAnalysis.h"
#include "loom/Mapper/CandidateBuilder.h"
#include "loom/Mapper/CPSATSolver.h"
#include "loom/Mapper/TypeCompat.h"

#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Hardware/Common/FabricConstants.h"
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

/// Get an integer attribute from a node, or default value.
int64_t getNodeInt(const Node *node, llvm::StringRef name, int64_t dflt = -1) {
  for (auto &attr : node->attributes)
    if (attr.getName() == name)
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return ia.getInt();
  return dflt;
}

} // namespace

// --- Pipeline orchestration ---

Mapper::Result Mapper::run(const Graph &dfg, const Graph &adg,
                           const Options &opts) {
  Result result;
  result.log.setEnabled(opts.verbose);
  log = &result.log;

  // Preprocessing: build connectivity matrix and min-hop costs.
  log->beginStage("Preprocessing");
  preprocess(adg);
  {
    unsigned hwNodes = 0, hwEdges = 0;
    for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i)
      if (adg.getNode(i)) ++hwNodes;
    for (IdIndex i = 0; i < static_cast<IdIndex>(adg.edges.size()); ++i)
      if (adg.getEdge(i)) ++hwEdges;
    unsigned swNodes = 0, swEdges = 0;
    for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i)
      if (dfg.getNode(i)) ++swNodes;
    for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i)
      if (dfg.getEdge(i)) ++swEdges;
    log->info("ADG: " + std::to_string(hwNodes) + " nodes, " +
              std::to_string(hwEdges) + " edges");
    log->info("DFG: " + std::to_string(swNodes) + " nodes, " +
              std::to_string(swEdges) + " edges");
    log->info("Connectivity inToOut entries: " +
              std::to_string(connectivity.inToOut.size()));
    log->info("Min-hop BFS entries: " +
              std::to_string(minHopCosts.size()));
  }
  log->endStage();

  // Tech-mapping: build candidate sets.
  log->beginStage("Candidate Building");
  CandidateBuilder candidateBuilder;
  auto candidateResult = candidateBuilder.build(dfg, adg);
  if (!candidateResult.success) {
    log->info("FAILED: " + candidateResult.diagnostics);
    log->endStage();
    result.success = false;
    result.diagnostics = candidateResult.diagnostics;
    return result;
  }
  {
    unsigned totalCandidates = 0;
    for (auto &[swId, cands] : candidateResult.candidates)
      totalCandidates += cands.size();
    log->info("Total candidates: " + std::to_string(totalCandidates) +
              " across " +
              std::to_string(candidateResult.candidates.size()) +
              " SW nodes");
  }
  log->endStage();

  // Initialize mapping state.
  result.state.init(dfg, adg);

  // Placement.
  log->beginStage("Placement");
  if (!runPlacement(result.state, dfg, adg, candidateResult.candidates, opts)) {
    log->info("FAILED: placement could not find valid assignment");
    log->logStateSummary(result.state, dfg, adg);
    log->endStage();
    result.success = false;
    result.diagnostics = "Placement failed";
    return result;
  }
  log->logStateSummary(result.state, dfg, adg);
  log->endStage();

  // Routing.
  log->beginStage("Routing");
  bool routingOk = runRouting(result.state, dfg, adg);
  log->logStateSummary(result.state, dfg, adg);
  if (!routingOk) {
    log->info("Initial routing incomplete, entering refinement");
    log->endStage();

    // Try refinement/repair.
    log->beginStage("Refinement");
    routingOk = runRefinement(result.state, dfg, adg,
                              candidateResult.candidates, opts);
    log->logStateSummary(result.state, dfg, adg);
    log->endStage();
  } else {
    log->endStage();
  }

  if (!routingOk) {
    // If CP-SAT sub-problem mode can attempt recovery from unrouted edges,
    // defer the failure and let the CP-SAT section collect conflict seeds.
    bool deferToSubProblem = false;
    if (CPSATSolver::isAvailable()) {
      auto mode = CPSATSolver::selectMode(dfg, opts.profile);
      if (mode == CPSATSolver::Mode::SUB_PROBLEM)
        deferToSubProblem = true;
    }
    if (!deferToSubProblem) {
      result.success = false;
      result.diagnostics = "Routing failed after refinement";
      return result;
    }
  }

  // Temporal assignment (skip when routing failed and CP-SAT recovery pending).
  log->beginStage("Temporal Assignment");
  if (routingOk && !runTemporalAssignment(result.state, dfg, adg)) {
    log->info("FAILED: temporal assignment error");
    log->endStage();
    result.success = false;
    result.diagnostics = "Temporal assignment failed";
    return result;
  }
  log->endStage();

  // Validation and cost for heuristic result (only meaningful when routing OK).
  log->beginStage("Validation & Cost");
  std::string validationDiag;
  bool heuristicValid = false;
  if (routingOk) {
    heuristicValid = runValidation(result.state, dfg, adg, validationDiag);
    log->logValidation("heuristic", heuristicValid, validationDiag);
    computeCost(result.state, dfg, adg, opts);
    log->logCost(result.state.totalCost, result.state.placementPressure,
                 result.state.routingCost, result.state.temporalCost,
                 result.state.perfProxyCost, result.state.configFootprint);
  }
  log->endStage();

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
          ++result.cpsatSubProblemCalls;
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
    if (!routingOk && validationDiag.empty())
      result.diagnostics = "Routing failed after refinement and CP-SAT recovery";
    else
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

  // Separate compute ops into critical-path and non-critical-path.
  // Critical-path ops are placed first for maximum placement freedom.
  std::vector<IdIndex> cpOps;
  std::vector<IdIndex> nonCpOps;
  for (IdIndex id : computeOps) {
    const Node *node = dfg.getNode(id);
    if (node && analysis::getAnalysisBoolAttr(node, "loom.on_critical_path"))
      cpOps.push_back(id);
    else
      nonCpOps.push_back(id);
  }

  order.insert(order.end(), cpOps.begin(), cpOps.end());
  order.insert(order.end(), nonCpOps.begin(), nonCpOps.end());

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
  // Sort memref sentinel indices by descending extmemory port count so the
  // most constrained DFG extmemory nodes bind first (prevents small DFG
  // extmemory from stealing large ADG extmemory slots).
  llvm::SmallVector<size_t> memrefOrder;
  for (size_t di = 0; di < dfgInputSentinels.size(); ++di) {
    const Node *dfgNode = dfg.getNode(dfgInputSentinels[di]);
    if (!dfgNode || dfgNode->outputPorts.empty())
      continue;
    mlir::Type dfgType = dfg.getPort(dfgNode->outputPorts[0])->type;
    if (!isMemrefType(dfgType))
      continue;
    memrefOrder.push_back(di);
  }
  llvm::sort(memrefOrder, [&](size_t a, size_t b) {
    auto getExtmemPortCount = [&](size_t idx) -> size_t {
      auto it = dfgSentinelToExtmem.find(dfgInputSentinels[idx]);
      if (it == dfgSentinelToExtmem.end())
        return 0;
      const Node *ext = dfg.getNode(it->second);
      if (!ext)
        return 0;
      return ext->inputPorts.size() + ext->outputPorts.size();
    };
    return getExtmemPortCount(a) > getExtmemPortCount(b);
  });
  for (size_t di : memrefOrder) {
    const Node *dfgNode = dfg.getNode(dfgInputSentinels[di]);
    if (!dfgNode || dfgNode->outputPorts.empty())
      continue;
    mlir::Type dfgType = dfg.getPort(dfgNode->outputPorts[0])->type;

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

      // Check for bridge-boundary metadata on ADG extmemory (multi-port).
      mlir::DenseI32ArrayAttr bridgeInPorts;
      mlir::DenseI32ArrayAttr bridgeOutPorts;
      int32_t sentStoreInCount = -1, sentLdDataOutCount = -1;
      for (auto &attr : adgExtmem->attributes) {
        if (attr.getName() == "bridge_input_ports")
          bridgeInPorts =
              mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
        else if (attr.getName() == "bridge_output_ports")
          bridgeOutPorts =
              mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
        else if (attr.getName() == "bridge_store_input_count") {
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
            sentStoreInCount = ia.getInt();
        } else if (attr.getName() == "bridge_ld_data_output_count") {
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
            sentLdDataOutCount = ia.getInt();
        }
      }

      if (bridgeInPorts || bridgeOutPorts) {
        // Multi-port extmemory with bridge. DFG port 0 is memref (direct),
        // remaining DFG ports map to bridge boundary ports.
        unsigned dfgNonMemrefIns = dfgExtmem->inputPorts.size() > 0
                                       ? dfgExtmem->inputPorts.size() - 1
                                       : 0;
        unsigned bridgeInCount = bridgeInPorts ? bridgeInPorts.size() : 0;
        unsigned bridgeOutCount = bridgeOutPorts ? bridgeOutPorts.size() : 0;

        if (dfgNonMemrefIns > bridgeInCount)
          continue;
        if (dfgExtmem->outputPorts.size() > bridgeOutCount)
          continue;

        // Per-port type compatibility via category-aware greedy matching.
        // Store inputs occupy [0, sentStoreInCount), load inputs occupy
        // [sentStoreInCount, bridgeInCount). DFG store inputs are detected
        // by (data, addr) pair pattern matching.
        bool typeMismatch = false;
        llvm::SmallVector<int, 8> inMatch(dfgExtmem->inputPorts.size(), -1);
        {
          unsigned storeInBound =
              (sentStoreInCount > 0)
                  ? static_cast<unsigned>(sentStoreInCount)
                  : 0;

          // DFG store input count from DFG node's stCount attr.
          int64_t dfgStCnt = getNodeInt(dfgExtmem, "stCount", 0);
          unsigned dfgStoreIns = static_cast<unsigned>(dfgStCnt) * 2;

          llvm::SmallVector<bool, 8> used(bridgeInCount, false);
          for (unsigned si = 1; si < dfgExtmem->inputPorts.size(); ++si) {
            const Port *sp = dfg.getPort(dfgExtmem->inputPorts[si]);
            if (!sp)
              continue;
            unsigned relIdx = si - 1;
            bool isStore = (relIdx < dfgStoreIns);
            int lo = isStore ? 0 : static_cast<int>(storeInBound);
            int hi = isStore ? static_cast<int>(storeInBound)
                             : static_cast<int>(bridgeInCount);
            bool found = false;
            for (int bi = lo; bi < hi; ++bi) {
              if (used[bi])
                continue;
              const Port *bp =
                  adg.getPort(static_cast<IdIndex>(bridgeInPorts[bi]));
              if (bp && isTypeWidthCompatible(sp->type, bp->type)) {
                used[bi] = true;
                inMatch[si] = bi;
                found = true;
                break;
              }
            }
            // Fallback: search full range.
            if (!found) {
              for (unsigned bi = 0; bi < bridgeInCount; ++bi) {
                if (used[bi])
                  continue;
                const Port *bp =
                    adg.getPort(static_cast<IdIndex>(bridgeInPorts[bi]));
                if (bp && isTypeWidthCompatible(sp->type, bp->type)) {
                  used[bi] = true;
                  inMatch[si] = static_cast<int>(bi);
                  found = true;
                  break;
                }
              }
            }
            if (!found) {
              typeMismatch = true;
              break;
            }
          }
        }
        llvm::SmallVector<int, 8> outMatch(
            dfgExtmem->outputPorts.size(), -1);
        if (!typeMismatch) {
          // Three-way output category split:
          // ld_data [0, ldOut), ld_done [ldOut, 2*ldOut), st_done [2*ldOut, end)
          int ldOut = (sentLdDataOutCount > 0) ? sentLdDataOutCount : 0;
          int ldDoneStart = ldOut;
          int stDoneStart = ldOut * 2;
          int total = static_cast<int>(bridgeOutCount);

          // Count DFG ld_data outputs (data-width type, sequential from start).
          unsigned dfgLdData = 0;
          const Port *ldDataRef =
              (ldOut > 0)
                  ? adg.getPort(static_cast<IdIndex>(bridgeOutPorts[0]))
                  : nullptr;
          for (size_t i = 0; i < dfgExtmem->outputPorts.size(); ++i) {
            const Port *sp = dfg.getPort(dfgExtmem->outputPorts[i]);
            if (sp && ldDataRef &&
                isTypeWidthCompatible(sp->type, ldDataRef->type))
              ++dfgLdData;
            else
              break;
          }

          llvm::SmallVector<bool, 8> used(bridgeOutCount, false);
          for (size_t i = 0; i < dfgExtmem->outputPorts.size(); ++i) {
            const Port *sp = dfg.getPort(dfgExtmem->outputPorts[i]);
            if (!sp)
              continue;
            int lo, hi;
            if (i < dfgLdData) {
              lo = 0; hi = ldDoneStart;         // ld_data range
            } else if (i < dfgLdData * 2) {
              lo = ldDoneStart; hi = stDoneStart; // ld_done range
            } else {
              lo = stDoneStart; hi = total;       // st_done range
            }
            bool found = false;
            for (int bi = lo; bi < hi; ++bi) {
              if (used[bi])
                continue;
              const Port *bp =
                  adg.getPort(static_cast<IdIndex>(bridgeOutPorts[bi]));
              if (bp && isTypeWidthCompatible(sp->type, bp->type)) {
                used[bi] = true;
                outMatch[i] = bi;
                found = true;
                break;
              }
            }
            // Fallback: search full range.
            if (!found) {
              for (int bi = 0; bi < total; ++bi) {
                if (used[bi])
                  continue;
                const Port *bp =
                    adg.getPort(static_cast<IdIndex>(bridgeOutPorts[bi]));
                if (bp && isTypeWidthCompatible(sp->type, bp->type)) {
                  used[bi] = true;
                  outMatch[i] = bi;
                  found = true;
                  break;
                }
              }
            }
            if (!found) {
              typeMismatch = true;
              break;
            }
          }
        }
        if (typeMismatch)
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

        // Pre-bind extmemory node on real ADG extmemory.
        state.mapNode(dfgExtIt->second, adgExtIt->second, dfg, adg);

        // Bind memref port (DFG input 0 -> ADG input 0).
        if (!dfgExtmem->inputPorts.empty() &&
            !adgExtmem->inputPorts.empty()) {
          state.mapPort(dfgExtmem->inputPorts[0],
                        adgExtmem->inputPorts[0], dfg, adg);
        }

        // Bind non-memref DFG input ports using greedy match results.
        for (size_t si = 1; si < dfgExtmem->inputPorts.size(); ++si) {
          if (inMatch[si] >= 0)
            state.mapPort(dfgExtmem->inputPorts[si],
                          static_cast<IdIndex>(bridgeInPorts[inMatch[si]]),
                          dfg, adg);
        }

        // Bind DFG output ports using greedy match results.
        for (size_t i = 0; i < dfgExtmem->outputPorts.size(); ++i) {
          if (outMatch[i] >= 0)
            state.mapPort(dfgExtmem->outputPorts[i],
                          static_cast<IdIndex>(bridgeOutPorts[outMatch[i]]),
                          dfg, adg);
        }

        preBoundExtmem.insert(dfgExtIt->second);
        break;
      }

      // Single-port extmemory: original logic.
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
          if (hp && isTypeWidthCompatible(sp->type, hp->type)) {
            state.mapPort(dfgExtmem->inputPorts[si],
                          adgExtmem->inputPorts[hi], dfg, adg);
            hwUsed[hi] = true;
            break;
          }
        }
      }
      // Extmemory outputs: positional mapping.
      for (size_t i = 0;
           i < dfgExtmem->outputPorts.size() &&
           i < adgExtmem->outputPorts.size();
           ++i) {
        state.mapPort(dfgExtmem->outputPorts[i],
                      adgExtmem->outputPorts[i], dfg, adg);
      }
      preBoundExtmem.insert(dfgExtIt->second);
      break;
    }
  }

  // Second pass: bind non-memref sentinels by width-compatible type matching.
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

      if (isTypeWidthCompatible(dfgType, adgType)) {
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

  // Output sentinel binding: width-compatible type matching.
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

      if (isTypeWidthCompatible(dfgType, adgType)) {
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

  // Successor-side scoring: for outgoing edges to already-mapped consumers,
  // reward proximity and penalize unreachability.
  for (IdIndex outPortId : sw->outputPorts) {
    const Port *outPort = dfg.getPort(outPortId);
    if (!outPort)
      continue;

    for (IdIndex edgeId : outPort->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge || edge->srcPort != outPortId)
        continue;

      const Port *dstPort = dfg.getPort(edge->dstPort);
      if (!dstPort)
        continue;

      IdIndex dstSwNode = dstPort->parentNode;
      if (dstSwNode == INVALID_ID || dstSwNode >= state.swNodeToHwNode.size())
        continue;

      IdIndex dstHwNode = state.swNodeToHwNode[dstSwNode];
      if (dstHwNode == INVALID_ID)
        continue;

      // Check if hwNode can reach dstHwNode.
      auto hopsIt = minHopCosts.find(hwNode);
      if (hopsIt != minHopCosts.end()) {
        auto hopIt = hopsIt->second.find(dstHwNode);
        if (hopIt != hopsIt->second.end()) {
          unsigned hops = hopIt->second;
          score += 10.0 / (1.0 + hops);
        } else {
          // Destination unreachable from this HW node: strong penalty.
          score -= 1000.0;
        }
      } else {
        score -= 1000.0;
      }
    }
  }

  // Utilization penalty: prefer less-used nodes.
  if (hwNode < state.hwNodeToSwNodes.size()) {
    score -= 5.0 * state.hwNodeToSwNodes[hwNode].size();
  }

  // Analysis-aware affinity: if DFG node has loom.temporal_score, add a
  // bonus for matching spatial/temporal placement.
  if (analysis::hasAnalysisAttr(sw, "loom.temporal_score")) {
    double temporalScore =
        analysis::getAnalysisFloatAttr(sw, "loom.temporal_score", 0.5);
    const Node *hw = adg.getNode(hwNode);
    if (hw) {
      bool hwIsTemporal = isTemporalPEFU(hw);
      if (hwIsTemporal) {
        // Temporal PE: bonus for high temporal_score ops.
        score += 5.0 * temporalScore;
      } else {
        // Spatial PE: bonus for low temporal_score ops.
        score += 5.0 * (1.0 - temporalScore);
      }
    }
  }

  return score;
}

bool Mapper::runPlacement(MappingState &state, const Graph &dfg,
                          const Graph &adg, const CandidateSet &candidates,
                          const Options &opts) {
  // Bind sentinels first.
  bindSentinelPorts(state, dfg, adg);

  // Compute placement order, then refine by candidate scarcity (MRV).
  auto order = computePlacementOrder(dfg);

  // Refine order with MRV (most constrained variable first) within each
  // priority tier. Nodes with fewer candidates are placed first, preventing
  // more-flexible nodes from stealing scarce PE slots.
  auto getTier = [&](IdIndex id) -> int {
    const Node *n = dfg.getNode(id);
    if (n && isMemoryOp(n))
      return 0;
    if (n && analysis::getAnalysisBoolAttr(n, "loom.on_critical_path"))
      return 1;
    return 2;
  };
  llvm::stable_sort(order, [&](IdIndex a, IdIndex b) {
    int tA = getTier(a), tB = getTier(b);
    if (tA != tB)
      return tA < tB;
    auto itA = candidates.find(a);
    auto itB = candidates.find(b);
    size_t cA = itA != candidates.end() ? itA->second.size() : 0;
    size_t cB = itB != candidates.end() ? itB->second.size() : 0;
    return cA < cB;
  });

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
      // Temporal PEs can host multiple SW nodes in different time slots.
      // Memory nodes allow up to numRegion SW nodes (capacity sharing).
      if (!candidate.isGroup &&
          !state.hwNodeToSwNodes[candidate.hwNodeId].empty()) {
        const Node *hwNode = adg.getNode(candidate.hwNodeId);
        if (!hwNode)
          continue;
        llvm::StringRef resClass = getNodeResourceClass(hwNode);
        if (resClass == "temporal") {
          // Virtual temporal PE node (defensive, never a placement target).
        } else if (isTemporalPEFU(hwNode)) {
          // Temporal FU sub-node: allow multi-slot sharing up to
          // num_instruction from the parent TPE.
          int64_t parentTPEId = -1;
          for (auto &attr : hwNode->attributes) {
            if (attr.getName() == "parent_temporal_pe") {
              if (auto intAttr =
                      mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
                parentTPEId = intAttr.getInt();
            }
          }
          if (parentTPEId >= 0) {
            const Node *tpeNode = adg.getNode(static_cast<IdIndex>(parentTPEId));
            int64_t numInstruction = 1;
            if (tpeNode) {
              for (auto &attr : tpeNode->attributes) {
                if (attr.getName() == "num_instruction") {
                  if (auto intAttr =
                          mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
                    numInstruction = intAttr.getInt();
                }
              }
            }
            if (static_cast<int64_t>(
                    state.hwNodeToSwNodes[candidate.hwNodeId].size()) >=
                numInstruction)
              continue; // Capacity exceeded.
          }
        } else if (resClass == "memory") {
          // Memory nodes allow up to numRegion SW nodes. Bridge-backed
          // memories are capped at 1 since tags encode lane indices.
          int64_t numRegion = 1;
          bool isBridgeMem = false;
          for (auto &attr : hwNode->attributes) {
            if (attr.getName() == "numRegion") {
              if (auto intAttr =
                      mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
                numRegion = intAttr.getInt();
            } else if (attr.getName() == "bridge_input_ports" ||
                       attr.getName() == "bridge_output_ports") {
              isBridgeMem = true;
            }
          }
          int64_t cap = isBridgeMem ? 1 : numRegion;
          if (static_cast<int64_t>(
                  state.hwNodeToSwNodes[candidate.hwNodeId].size()) >=
              cap)
            continue;
        } else {
          continue; // Non-temporal, non-memory PEs enforce exclusivity.
        }
      }

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
      log->info("No candidate for SW N" + std::to_string(swNode) +
                " (" + getNodeOpName(dfg.getNode(swNode)).str() + ")");
      return false;
    }

    IdIndex bestHw = bestCand->hwNodeId;
    {
      const Node *swN = dfg.getNode(swNode);
      const Node *hwN = adg.getNode(bestHw);
      llvm::StringRef swName = swN ? getNodeOpName(swN) : "";
      llvm::StringRef hwName = hwN ? getNodeOpName(hwN) : "";
      log->logPlacement(swNode, bestHw, swName, hwName, bestScore);
    }

    if (bestCand->isGroup && bestCand->swNodeIds.size() > 1) {
      // Group placement: bind ALL SW nodes in the group to the same
      // HW PE atomically. Only external ports (those NOT connected to
      // another group member) are bound to PE physical ports.
      llvm::DenseSet<IdIndex> groupMemberSet(bestCand->swNodeIds.begin(),
                                              bestCand->swNodeIds.end());

      // Map all group member nodes first.
      for (IdIndex groupSwNode : bestCand->swNodeIds) {
        auto mapResult = state.mapNode(groupSwNode, bestHw, dfg, adg);
        if (mapResult != ActionResult::Success)
          return false;
        placedInGroup.insert(groupSwNode);
      }

      // Bind only external ports to PE physical ports.
      unsigned inPortOffset = 0;
      unsigned outPortOffset = 0;
      const Node *hw = adg.getNode(bestHw);

      for (IdIndex groupSwNode : bestCand->swNodeIds) {
        const Node *sw = dfg.getNode(groupSwNode);
        if (!sw || !hw)
          continue;

        // Bind external input ports: skip inputs driven by group members.
        for (size_t i = 0; i < sw->inputPorts.size(); ++i) {
          const Port *inPort = dfg.getPort(sw->inputPorts[i]);
          if (!inPort)
            continue;
          bool isInternal = false;
          for (IdIndex edgeId : inPort->connectedEdges) {
            const Edge *edge = dfg.getEdge(edgeId);
            if (!edge || edge->dstPort != sw->inputPorts[i])
              continue;
            const Port *srcPort = dfg.getPort(edge->srcPort);
            if (srcPort && groupMemberSet.count(srcPort->parentNode)) {
              isInternal = true;
              break;
            }
          }
          if (isInternal)
            continue;
          if (inPortOffset < hw->inputPorts.size()) {
            state.mapPort(sw->inputPorts[i],
                          hw->inputPorts[inPortOffset], dfg, adg);
            ++inPortOffset;
          }
        }

        // Bind external output ports: skip outputs consumed only by
        // group members.
        for (size_t i = 0; i < sw->outputPorts.size(); ++i) {
          const Port *outPort = dfg.getPort(sw->outputPorts[i]);
          if (!outPort)
            continue;
          bool allConsumersInternal = true;
          bool hasConsumers = false;
          for (IdIndex edgeId : outPort->connectedEdges) {
            const Edge *edge = dfg.getEdge(edgeId);
            if (!edge || edge->srcPort != sw->outputPorts[i])
              continue;
            hasConsumers = true;
            const Port *dstPort = dfg.getPort(edge->dstPort);
            if (!dstPort || !groupMemberSet.count(dstPort->parentNode)) {
              allConsumersInternal = false;
              break;
            }
          }
          if (hasConsumers && allConsumersInternal)
            continue;
          if (outPortOffset < hw->outputPorts.size()) {
            state.mapPort(sw->outputPorts[i],
                          hw->outputPorts[outPortOffset], dfg, adg);
            ++outPortOffset;
          }
        }
      }

      // Record group binding for C4 validation.
      state.groupBindings[bestHw] = llvm::SmallVector<IdIndex, 4>(
          bestCand->swNodeIds.begin(), bestCand->swNodeIds.end());

      // Record tag assignment for tagged non-temporal PEs (group placement).
      if (hw && !isTemporalPEFU(hw)) {
        for (auto &attr : hw->attributes) {
          if (attr.getName().getValue() == "output_tag") {
            state.taggedPEOutputTags[bestHw] = 0;
            break;
          }
        }
      }
    } else {
      // Single-op placement.
      auto mapResult = state.mapNode(swNode, bestHw, dfg, adg);
      if (mapResult != ActionResult::Success)
        return false;

      const Node *sw = dfg.getNode(swNode);
      const Node *hw = adg.getNode(bestHw);
      if (sw && hw) {
        bool isMemory = (getNodeResourceClass(hw) == "memory");

        if (isMemory) {
          // Check for bridge-boundary metadata (multi-port memory).
          mlir::DenseI32ArrayAttr bridgeInPorts;
          mlir::DenseI32ArrayAttr bridgeOutPorts;
          int32_t storeInCount = -1;
          for (auto &attr : hw->attributes) {
            if (attr.getName() == "bridge_input_ports")
              bridgeInPorts =
                  mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
            else if (attr.getName() == "bridge_output_ports")
              bridgeOutPorts =
                  mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
            else if (attr.getName() == "bridge_store_input_count") {
              if (auto ia =
                      mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
                storeInCount = ia.getInt();
            }
          }

          if (bridgeInPorts || bridgeOutPorts) {
            // Multi-port memory with bridge: bind DFG ports to bridge
            // boundary ports (add_tag inputs / del_tag outputs).
            bool isExtMem =
                (getNodeResourceClass(hw) == "memory" &&
                 getNodeOpName(hw) == "fabric.extmemory");
            unsigned swInSkip = isExtMem ? 1 : 0;

            // Bind memref port directly (extmemory only).
            if (isExtMem && !sw->inputPorts.empty() &&
                !hw->inputPorts.empty()) {
              state.mapPort(sw->inputPorts[0], hw->inputPorts[0], dfg, adg);
            }

            // Bind non-memref inputs to bridge boundary input ports.
            // Bridge inputs: [st0_data, st0_addr, ..., ld0_addr, ...].
            // storeInCount separates store range [0, storeInCount) from load
            // range [storeInCount, end). Store inputs come in (data, addr)
            // pairs; detect them by checking if the first port of each pair
            // matches a bridge store data port (unique data-width type that
            // differs from the addr-width type shared by st_addr and ld_addr).
            if (bridgeInPorts) {
              auto bindInRange = [&](size_t si, int lo, int hi) -> bool {
                const Port *sp = dfg.getPort(sw->inputPorts[si]);
                if (!sp) return false;
                for (int bi = lo; bi < hi; ++bi) {
                  auto hwPid = static_cast<IdIndex>(bridgeInPorts[bi]);
                  if (!state.hwPortToSwPorts[hwPid].empty())
                    continue;
                  const Port *hp = adg.getPort(hwPid);
                  if (hp && isTypeWidthCompatible(sp->type, hp->type)) {
                    state.mapPort(sw->inputPorts[si], hwPid, dfg, adg);
                    return true;
                  }
                }
                return false;
              };
              // Determine DFG store input count from DFG node's stCount attr
              // (each store has data + addr pair = 2 ports).
              int64_t dfgStCount = getNodeInt(sw, "stCount", 0);
              unsigned dfgStoreIns =
                  static_cast<unsigned>(dfgStCount) * 2;
              for (size_t si = swInSkip; si < sw->inputPorts.size(); ++si) {
                unsigned relIdx = si - swInSkip;
                bool isStore = (relIdx < dfgStoreIns);
                int lo = isStore ? 0 : storeInCount;
                int hi = isStore ? storeInCount
                                 : static_cast<int>(bridgeInPorts.size());
                if (storeInCount < 0 || !bindInRange(si, lo, hi)) {
                  bindInRange(si, 0, static_cast<int>(bridgeInPorts.size()));
                }
              }
            }

            // Bind outputs to bridge boundary output ports.
            // Output layout: [ld_data * ldCount, ld_done * ldCount,
            //                 st_done * stCount].
            // Three categories with distinct bridge ranges:
            //   ld_data: [0, ldDataOutCount)
            //   ld_done: [ldDataOutCount, 2*ldDataOutCount)
            //   st_done: [2*ldDataOutCount, end)
            // DFG outputs follow the same ordering.
            if (bridgeOutPorts) {
              int32_t ldDataOutCount = 0;
              for (auto &attr : hw->attributes) {
                if (attr.getName() == "bridge_ld_data_output_count") {
                  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(
                          attr.getValue()))
                    ldDataOutCount = ia.getInt();
                }
              }
              int ldDoneStart = ldDataOutCount;
              int stDoneStart = ldDataOutCount * 2;
              int total = static_cast<int>(bridgeOutPorts.size());

              // Count DFG ld_data outputs (data-width type, comes first).
              unsigned dfgLdData = 0;
              const Port *ldDataRef =
                  (ldDataOutCount > 0)
                      ? adg.getPort(
                            static_cast<IdIndex>(bridgeOutPorts[0]))
                      : nullptr;
              for (size_t i = 0; i < sw->outputPorts.size(); ++i) {
                const Port *sp = dfg.getPort(sw->outputPorts[i]);
                if (sp && ldDataRef &&
                    isTypeWidthCompatible(sp->type, ldDataRef->type))
                  ++dfgLdData;
                else
                  break;
              }

              auto bindOutRange = [&](size_t oi, int lo,
                                      int hi) -> bool {
                const Port *sp = dfg.getPort(sw->outputPorts[oi]);
                if (!sp) return false;
                for (int bi = lo; bi < hi; ++bi) {
                  auto hwPid =
                      static_cast<IdIndex>(bridgeOutPorts[bi]);
                  if (!state.hwPortToSwPorts[hwPid].empty())
                    continue;
                  const Port *hp = adg.getPort(hwPid);
                  if (hp &&
                      isTypeWidthCompatible(sp->type, hp->type)) {
                    state.mapPort(sw->outputPorts[oi], hwPid, dfg,
                                  adg);
                    return true;
                  }
                }
                return false;
              };

              for (size_t i = 0; i < sw->outputPorts.size(); ++i) {
                int lo, hi;
                if (i < dfgLdData) {
                  lo = 0; hi = ldDoneStart;       // ld_data range
                } else if (i < dfgLdData * 2) {
                  lo = ldDoneStart; hi = stDoneStart; // ld_done range
                } else {
                  lo = stDoneStart; hi = total;       // st_done range
                }
                if (!bindOutRange(i, lo, hi))
                  bindOutRange(i, 0, total); // Fallback: full range.
              }
            }
          } else {
            // Single-port memory: original greedy type-based input binding.
            llvm::DenseSet<IdIndex> usedInNode;
            for (size_t si = 0; si < sw->inputPorts.size(); ++si) {
              const Port *sp = dfg.getPort(sw->inputPorts[si]);
              if (!sp) continue;
              for (size_t hi = 0; hi < hw->inputPorts.size(); ++hi) {
                IdIndex hwPid = hw->inputPorts[hi];
                if (!state.hwPortToSwPorts[hwPid].empty())
                  continue;
                if (usedInNode.count(hwPid))
                  continue;
                const Port *hp = adg.getPort(hwPid);
                if (hp && isTypeWidthCompatible(sp->type, hp->type)) {
                  state.mapPort(sw->inputPorts[si], hwPid, dfg, adg);
                  usedInNode.insert(hwPid);
                  break;
                }
              }
            }
            // Memory outputs: positional mapping.
            for (size_t i = 0;
                 i < sw->outputPorts.size() && i < hw->outputPorts.size();
                 ++i) {
              state.mapPort(sw->outputPorts[i], hw->outputPorts[i], dfg, adg);
            }
          }
        } else {
          // PE / temporal PE FU: positional mapping for both inputs and
          // outputs. Port positions are semantically fixed (no internal
          // mux), and TechMapper validates compatibility positionally.
          for (size_t i = 0;
               i < sw->inputPorts.size() && i < hw->inputPorts.size();
               ++i) {
            state.mapPort(sw->inputPorts[i], hw->inputPorts[i], dfg, adg);
          }
          for (size_t i = 0;
               i < sw->outputPorts.size() && i < hw->outputPorts.size();
               ++i) {
            state.mapPort(sw->outputPorts[i], hw->outputPorts[i], dfg, adg);
          }
        }

        // Record tag assignment for tagged non-temporal PEs.
        if (!isMemory && !isTemporalPEFU(hw)) {
          for (auto &attr : hw->attributes) {
            if (attr.getName().getValue() == "output_tag") {
              state.taggedPEOutputTags[bestHw] = 0;
              break;
            }
          }
        }
      }
    }
  }

  return true;
}

} // namespace loom

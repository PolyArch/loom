//===-- CPSATSolver.cpp - CP-SAT solver for mapper -----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/CPSATSolver.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>

#ifdef LOOM_HAS_ORTOOLS
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"
#endif

namespace loom {

namespace {

/// Get a string attribute from a node.
llvm::StringRef getStrAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Get an integer attribute from a node.
int64_t getIntAttr(const Node *node, llvm::StringRef name,
                   int64_t dflt = -1) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return intAttr.getInt();
    }
  }
  return dflt;
}

/// Check if a node has a specific attribute.
bool hasAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes)
    if (attr.getName() == name)
      return true;
  return false;
}

/// Collect DFG neighbor nodes via edges.
llvm::SmallVector<IdIndex, 8> getNeighbors(const Graph &dfg, IdIndex swNode) {
  llvm::SmallVector<IdIndex, 8> neighbors;
  const Node *node = dfg.getNode(swNode);
  if (!node)
    return neighbors;

  // Follow output edges.
  for (IdIndex outPortId : node->outputPorts) {
    const Port *outPort = dfg.getPort(outPortId);
    if (!outPort)
      continue;
    for (IdIndex edgeId : outPort->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      const Port *dstPort = dfg.getPort(edge->dstPort);
      if (dstPort && dstPort->parentNode != swNode)
        neighbors.push_back(dstPort->parentNode);
    }
  }

  // Follow input edges.
  for (IdIndex inPortId : node->inputPorts) {
    const Port *inPort = dfg.getPort(inPortId);
    if (!inPort)
      continue;
    for (IdIndex edgeId : inPort->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      if (srcPort && srcPort->parentNode != swNode)
        neighbors.push_back(srcPort->parentNode);
    }
  }

  return neighbors;
}

} // namespace

bool CPSATSolver::isAvailable() {
#ifdef LOOM_HAS_ORTOOLS
  return true;
#else
  return false;
#endif
}

CPSATSolver::Mode CPSATSolver::selectMode(const Graph &dfg,
                                            const std::string &profile,
                                            int subProblemMaxNodes) {
  if (profile == "cpsat_full")
    return Mode::FULL_PROBLEM;
  if (profile == "heuristic_only")
    return Mode::DISABLED;

  size_t nodeCount = dfg.countNodes();
  if (static_cast<int>(nodeCount) <= subProblemMaxNodes)
    return Mode::FULL_PROBLEM;
  return Mode::SUB_PROBLEM;
}

llvm::SmallVector<IdIndex, 16>
CPSATSolver::extractSubProblem(const Graph &dfg,
                               llvm::ArrayRef<IdIndex> conflictNodes,
                               int maxNodes) {
  llvm::DenseSet<IdIndex> selected;
  for (IdIndex n : conflictNodes)
    selected.insert(n);

  // Expand to include immediate neighbors.
  for (IdIndex n : conflictNodes) {
    if (static_cast<int>(selected.size()) >= maxNodes)
      break;
    auto neighbors = getNeighbors(dfg, n);
    for (IdIndex nb : neighbors) {
      if (static_cast<int>(selected.size()) >= maxNodes)
        break;
      selected.insert(nb);
    }
  }

  llvm::SmallVector<IdIndex, 16> result;
  result.reserve(selected.size());
  for (IdIndex n : selected)
    result.push_back(n);

  // Sort for determinism.
  llvm::sort(result);
  return result;
}

#ifdef LOOM_HAS_ORTOOLS

CPSATSolver::Result CPSATSolver::solveFullProblem(
    const Graph &dfg, const Graph &adg, const CandidateSet &candidates,
    const ConnectivityMatrix &connectivity, const MappingState *warmStart,
    const Options &opts) {
  using namespace operations_research::sat;

  Result result;
  CpModelBuilder model;

  // Collect DFG operation nodes.
  llvm::SmallVector<IdIndex, 32> swNodes;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (node && node->kind == Node::OperationNode)
      swNodes.push_back(i);
  }

  // Decision variables: x[swNode][hwNode] = 1 iff placed there.
  llvm::DenseMap<IdIndex, llvm::DenseMap<IdIndex, BoolVar>> placementVars;

  for (IdIndex sw : swNodes) {
    auto candIt = candidates.find(sw);
    if (candIt == candidates.end() || candIt->second.empty()) {
      result.diagnostics = "No candidates for sw node " + std::to_string(sw);
      return result;
    }

    auto &swVars = placementVars[sw];
    for (const Candidate &cand : candIt->second) {
      swVars[cand.hwNodeId] = model.NewBoolVar();
    }

    // C1: Each sw node placed exactly once.
    std::vector<BoolVar> allVars;
    allVars.reserve(swVars.size());
    for (auto &[hw, var] : swVars)
      allVars.push_back(var);
    model.AddExactlyOne(allVars);
  }

  // C_group: Group atomicity - all members of a multi-op group must be
  // placed together on the group's target HW node, or none of them.
  {
    llvm::DenseSet<uint64_t> processedGroups;
    for (const auto &[sw, candList] : candidates) {
      for (const Candidate &cand : candList) {
        if (!cand.isGroup || cand.swNodeIds.size() <= 1)
          continue;

        // Deduplicate groups by (hwNodeId, smallest swNodeId).
        IdIndex minSw = cand.swNodeIds[0];
        for (IdIndex s : cand.swNodeIds)
          minSw = std::min(minSw, s);
        uint64_t groupKey =
            (static_cast<uint64_t>(cand.hwNodeId) << 32) | minSw;
        if (!processedGroups.insert(groupKey).second)
          continue;

        // Enforce equality: x[sw0][hw] == x[sw1][hw] == ... for all members.
        IdIndex refSw = cand.swNodeIds[0];
        auto refIt = placementVars.find(refSw);
        if (refIt == placementVars.end())
          continue;
        auto refHwIt = refIt->second.find(cand.hwNodeId);
        if (refHwIt == refIt->second.end())
          continue;

        for (size_t k = 1; k < cand.swNodeIds.size(); ++k) {
          IdIndex otherSw = cand.swNodeIds[k];
          auto otherIt = placementVars.find(otherSw);
          if (otherIt == placementVars.end())
            continue;
          auto otherHwIt = otherIt->second.find(cand.hwNodeId);
          if (otherHwIt == otherIt->second.end())
            continue;

          model.AddEquality(refHwIt->second, otherHwIt->second);
        }
      }
    }
  }

  // C4: Capacity constraints - at most one sw node per exclusive hw node.
  llvm::DenseMap<IdIndex, std::vector<BoolVar>> hwNodeVars;
  for (auto &[sw, swVars] : placementVars) {
    for (auto &[hw, var] : swVars)
      hwNodeVars[hw].push_back(var);
  }

  for (auto &[hwId, vars] : hwNodeVars) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;

    llvm::StringRef resClass = getStrAttr(hwNode, "resource_class");
    if (resClass == "functional") {
      bool isTemporal =
          hasAttr(hwNode, "parent_temporal_pe") || hasAttr(hwNode, "is_virtual");
      if (!isTemporal) {
        // Exclusive: at most 1.
        model.AddAtMostOne(vars);
      } else {
        // Temporal: capacity = num_instruction.
        int64_t numInst = getIntAttr(hwNode, "num_instruction", 1);
        LinearExpr sum;
        for (auto &v : vars)
          sum += v;
        model.AddLessOrEqual(sum, numInst);
      }
    } else if (resClass == "memory") {
      int64_t numRegion = getIntAttr(hwNode, "numRegion", 1);
      LinearExpr sum;
      for (auto &v : vars)
        sum += v;
      model.AddLessOrEqual(sum, numRegion);
    }
  }

  // Objective: minimize placement pressure (sum of squared occupancies
  // approximated as sum of occupancies for linearity).
  LinearExpr objective;
  for (auto &[hwId, vars] : hwNodeVars) {
    for (auto &v : vars)
      objective += v;
  }
  model.Minimize(objective);

  // Warm start hints.
  if (warmStart) {
    for (auto &[sw, swVars] : placementVars) {
      for (auto &[hw, var] : swVars) {
        if (sw < warmStart->swNodeToHwNode.size() &&
            warmStart->swNodeToHwNode[sw] == hw) {
          model.AddHint(var, 1);
        } else {
          model.AddHint(var, 0);
        }
      }
    }
  }

  // Solver parameters.
  SatParameters params;
  params.set_max_time_in_seconds(opts.timeLimitSeconds);
  params.set_num_workers(1);

  CpSolverResponse response = SolveWithParameters(model.Build(), params);

  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    result.diagnostics = "CP-SAT: no feasible solution found";
    return result;
  }

  // Extract solution into MappingState.
  result.state.init(dfg, adg);

  for (auto &[sw, swVars] : placementVars) {
    for (auto &[hw, var] : swVars) {
      if (SolutionBoolValue(response, var)) {
        result.state.mapNode(sw, hw, dfg, adg);

        // Map ports positionally.
        const Node *swNode = dfg.getNode(sw);
        const Node *hwNode = adg.getNode(hw);
        if (swNode && hwNode) {
          for (size_t p = 0;
               p < swNode->inputPorts.size() && p < hwNode->inputPorts.size();
               ++p) {
            result.state.mapPort(swNode->inputPorts[p],
                                 hwNode->inputPorts[p], dfg, adg);
          }
          for (size_t p = 0; p < swNode->outputPorts.size() &&
                             p < hwNode->outputPorts.size();
               ++p) {
            result.state.mapPort(swNode->outputPorts[p],
                                 hwNode->outputPorts[p], dfg, adg);
          }
        }
        break;
      }
    }
  }

  // Route edges using BFS through connectivity.
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;

    IdIndex srcSwPort = edge->srcPort;
    IdIndex dstSwPort = edge->dstPort;

    if (srcSwPort >= result.state.swPortToHwPort.size() ||
        dstSwPort >= result.state.swPortToHwPort.size())
      continue;

    IdIndex srcHwPort = result.state.swPortToHwPort[srcSwPort];
    IdIndex dstHwPort = result.state.swPortToHwPort[dstSwPort];

    if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
      continue;

    // Simple direct connection attempt.
    auto directIt = connectivity.outToIn.find(srcHwPort);
    if (directIt != connectivity.outToIn.end() &&
        directIt->second == dstHwPort) {
      llvm::SmallVector<IdIndex, 8> path = {srcHwPort, dstHwPort};
      result.state.mapEdge(edgeId, path, dfg, adg);
    }
  }

  result.success = true;
  return result;
}

CPSATSolver::Result CPSATSolver::solveSubProblem(
    const Graph &dfg, const Graph &adg,
    llvm::ArrayRef<IdIndex> subgraphSwNodes,
    const MappingState &currentState, const CandidateSet &candidates,
    const ConnectivityMatrix &connectivity, const Options &opts) {
  using namespace operations_research::sat;

  Result result;
  CpModelBuilder model;

  llvm::DenseSet<IdIndex> subNodes;
  for (IdIndex n : subgraphSwNodes)
    subNodes.insert(n);

  // Variables only for sub-problem nodes.
  llvm::DenseMap<IdIndex, llvm::DenseMap<IdIndex, BoolVar>> placementVars;

  for (IdIndex sw : subgraphSwNodes) {
    auto candIt = candidates.find(sw);
    if (candIt == candidates.end() || candIt->second.empty())
      continue;

    auto &swVars = placementVars[sw];
    for (const Candidate &cand : candIt->second)
      swVars[cand.hwNodeId] = model.NewBoolVar();

    std::vector<BoolVar> allVars;
    for (auto &[hw, var] : swVars)
      allVars.push_back(var);
    model.AddExactlyOne(allVars);
  }

  // C_group: Group atomicity for sub-problem members.
  {
    llvm::DenseSet<uint64_t> processedGroups;
    for (IdIndex sw : subgraphSwNodes) {
      auto candIt = candidates.find(sw);
      if (candIt == candidates.end())
        continue;
      for (const Candidate &cand : candIt->second) {
        if (!cand.isGroup || cand.swNodeIds.size() <= 1)
          continue;

        IdIndex minSw = cand.swNodeIds[0];
        for (IdIndex s : cand.swNodeIds)
          minSw = std::min(minSw, s);
        uint64_t groupKey =
            (static_cast<uint64_t>(cand.hwNodeId) << 32) | minSw;
        if (!processedGroups.insert(groupKey).second)
          continue;

        // Only enforce equality for members that are in the sub-problem.
        llvm::SmallVector<IdIndex, 4> activeMembers;
        for (IdIndex s : cand.swNodeIds) {
          if (subNodes.count(s) && placementVars.count(s))
            activeMembers.push_back(s);
        }
        if (activeMembers.size() <= 1)
          continue;

        IdIndex refSw = activeMembers[0];
        auto refHwIt = placementVars[refSw].find(cand.hwNodeId);
        if (refHwIt == placementVars[refSw].end())
          continue;

        for (size_t k = 1; k < activeMembers.size(); ++k) {
          IdIndex otherSw = activeMembers[k];
          auto otherHwIt = placementVars[otherSw].find(cand.hwNodeId);
          if (otherHwIt == placementVars[otherSw].end())
            continue;
          model.AddEquality(refHwIt->second, otherHwIt->second);
        }
      }
    }
  }

  // Capacity constraints including fixed nodes.
  llvm::DenseMap<IdIndex, int> fixedOccupancy;
  for (IdIndex i = 0; i < static_cast<IdIndex>(currentState.swNodeToHwNode.size());
       ++i) {
    if (subNodes.count(i))
      continue;
    IdIndex hw = currentState.swNodeToHwNode[i];
    if (hw != INVALID_ID)
      fixedOccupancy[hw]++;
  }

  llvm::DenseMap<IdIndex, std::vector<BoolVar>> hwNodeVars;
  for (auto &[sw, swVars] : placementVars) {
    for (auto &[hw, var] : swVars)
      hwNodeVars[hw].push_back(var);
  }

  for (auto &[hwId, vars] : hwNodeVars) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;

    llvm::StringRef resClass = getStrAttr(hwNode, "resource_class");
    int fixedCount = 0;
    auto fixIt = fixedOccupancy.find(hwId);
    if (fixIt != fixedOccupancy.end())
      fixedCount = fixIt->second;

    if (resClass == "functional") {
      bool isTemporal =
          hasAttr(hwNode, "parent_temporal_pe") || hasAttr(hwNode, "is_virtual");
      if (!isTemporal) {
        int remaining = 1 - fixedCount;
        if (remaining <= 0) {
          // Already full: no sub-problem node can go here.
          for (auto &v : vars)
            model.AddEquality(v, 0);
        } else {
          model.AddAtMostOne(vars);
        }
      } else {
        int64_t numInst = getIntAttr(hwNode, "num_instruction", 1);
        int remaining = static_cast<int>(numInst) - fixedCount;
        LinearExpr sum;
        for (auto &v : vars)
          sum += v;
        model.AddLessOrEqual(sum, std::max(0, remaining));
      }
    } else if (resClass == "memory") {
      int64_t numRegion = getIntAttr(hwNode, "numRegion", 1);
      int remaining = static_cast<int>(numRegion) - fixedCount;
      LinearExpr sum;
      for (auto &v : vars)
        sum += v;
      model.AddLessOrEqual(sum, std::max(0, remaining));
    }
  }

  // Objective: minimize number of occupied HW nodes (proxy).
  LinearExpr objective;
  for (auto &[hwId, vars] : hwNodeVars) {
    for (auto &v : vars)
      objective += v;
  }
  model.Minimize(objective);

  // Warm start from current state.
  for (auto &[sw, swVars] : placementVars) {
    for (auto &[hw, var] : swVars) {
      if (sw < currentState.swNodeToHwNode.size() &&
          currentState.swNodeToHwNode[sw] == hw) {
        model.AddHint(var, 1);
      } else {
        model.AddHint(var, 0);
      }
    }
  }

  SatParameters params;
  params.set_max_time_in_seconds(opts.timeLimitSeconds);
  params.set_num_workers(1);

  CpSolverResponse response = SolveWithParameters(model.Build(), params);

  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    result.diagnostics = "CP-SAT sub-problem: no feasible solution found";
    return result;
  }

  // Start from current state and re-map sub-problem nodes.
  result.state = currentState;

  // Unmap sub-problem nodes.
  for (IdIndex sw : subgraphSwNodes) {
    if (sw < result.state.swNodeToHwNode.size() &&
        result.state.swNodeToHwNode[sw] != INVALID_ID) {
      result.state.unmapNode(sw, dfg, adg);
    }
  }

  // Apply new placements.
  for (auto &[sw, swVars] : placementVars) {
    for (auto &[hw, var] : swVars) {
      if (SolutionBoolValue(response, var)) {
        result.state.mapNode(sw, hw, dfg, adg);

        const Node *swNode = dfg.getNode(sw);
        const Node *hwNode = adg.getNode(hw);
        if (swNode && hwNode) {
          for (size_t p = 0;
               p < swNode->inputPorts.size() && p < hwNode->inputPorts.size();
               ++p) {
            if (result.state.swPortToHwPort[swNode->inputPorts[p]] == INVALID_ID)
              result.state.mapPort(swNode->inputPorts[p],
                                   hwNode->inputPorts[p], dfg, adg);
          }
          for (size_t p = 0; p < swNode->outputPorts.size() &&
                             p < hwNode->outputPorts.size();
               ++p) {
            if (result.state.swPortToHwPort[swNode->outputPorts[p]] == INVALID_ID)
              result.state.mapPort(swNode->outputPorts[p],
                                   hwNode->outputPorts[p], dfg, adg);
          }
        }
        break;
      }
    }
  }

  result.success = true;
  return result;
}

#else // !LOOM_HAS_ORTOOLS

CPSATSolver::Result CPSATSolver::solveFullProblem(
    const Graph &dfg, const Graph &adg, const CandidateSet &candidates,
    const ConnectivityMatrix &connectivity, const MappingState *warmStart,
    const Options &opts) {
  Result result;
  result.diagnostics = "CP-SAT solver not available (OR-Tools not linked)";
  return result;
}

CPSATSolver::Result CPSATSolver::solveSubProblem(
    const Graph &dfg, const Graph &adg,
    llvm::ArrayRef<IdIndex> subgraphSwNodes,
    const MappingState &currentState, const CandidateSet &candidates,
    const ConnectivityMatrix &connectivity, const Options &opts) {
  Result result;
  result.diagnostics = "CP-SAT solver not available (OR-Tools not linked)";
  return result;
}

#endif // LOOM_HAS_ORTOOLS

} // namespace loom

#include "loom/Mapper/Mapper.h"

#include "MapperInternal.h"
#include "loom/Mapper/TopologyModel.h"
#include "loom/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace loom {

using namespace mapper_detail;

bool Mapper::bindSentinels(MappingState &state, const Graph &dfg,
                           const Graph &adg) {
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

  llvm::outs() << "  DFG sentinels: " << dfgInputSentinels.size()
               << " inputs, " << dfgOutputSentinels.size() << " outputs\n";
  llvm::outs() << "  ADG sentinels: " << adgInputSentinels.size()
               << " inputs, " << adgOutputSentinels.size() << " outputs\n";

  std::vector<IdIndex> dfgMemrefSentinels;
  std::vector<IdIndex> dfgScalarSentinels;

  for (IdIndex sid : dfgInputSentinels) {
    const Node *node = dfg.getNode(sid);
    if (!node || node->outputPorts.empty())
      continue;
    mlir::Type portType = dfg.getPort(node->outputPorts[0])->type;
    if (isMemrefType(portType))
      dfgMemrefSentinels.push_back(sid);
    else
      dfgScalarSentinels.push_back(sid);
  }

  llvm::outs() << "    DFG memref inputs: " << dfgMemrefSentinels.size()
               << ", scalar inputs: " << dfgScalarSentinels.size() << "\n";

  llvm::DenseSet<size_t> usedAdgIn;
  for (size_t di = 0; di < dfgScalarSentinels.size(); ++di) {
    IdIndex dfgSid = dfgScalarSentinels[di];

    bool bound = false;
    for (size_t ai = 0; ai < adgInputSentinels.size(); ++ai) {
      if (usedAdgIn.count(ai))
        continue;

      IdIndex adgSid = adgInputSentinels[ai];
      const Node *dfgNode = dfg.getNode(dfgSid);
      const Node *adgNode = adg.getNode(adgSid);
      if (!dfgNode || !adgNode || dfgNode->outputPorts.empty() ||
          adgNode->outputPorts.empty())
        continue;
      const Port *swPort = dfg.getPort(dfgNode->outputPorts[0]);
      const Port *hwPort = adg.getPort(adgNode->outputPorts[0]);
      if (!swPort || !hwPort ||
          !canMapSoftwareTypeToHardware(swPort->type, hwPort->type))
        continue;

      auto result = state.mapNode(dfgSid, adgSid, dfg, adg);
      if (result == ActionResult::Success) {
        usedAdgIn.insert(ai);
        bound = true;
        llvm::outs() << "    Bound DFG input sentinel " << dfgSid
                     << " -> ADG input sentinel " << adgSid << "\n";
        break;
      }
    }

    if (!bound) {
      llvm::errs() << "Mapper: failed to bind DFG input sentinel " << dfgSid
                   << "\n";
    }
  }

  llvm::DenseSet<size_t> usedAdgOut;
  for (size_t di = 0; di < dfgOutputSentinels.size(); ++di) {
    IdIndex dfgSid = dfgOutputSentinels[di];
    const Node *dfgNode = dfg.getNode(dfgSid);
    if (!dfgNode || dfgNode->inputPorts.empty())
      continue;
    const Port *swPort = dfg.getPort(dfgNode->inputPorts[0]);

    bool bound = false;
    for (size_t ai = 0; ai < adgOutputSentinels.size(); ++ai) {
      if (usedAdgOut.count(ai))
        continue;

      IdIndex adgSid = adgOutputSentinels[ai];
      const Node *adgNode = adg.getNode(adgSid);
      if (!dfgNode || !adgNode || dfgNode->inputPorts.empty() ||
          adgNode->inputPorts.empty())
        continue;
      const Port *hwPort = adg.getPort(adgNode->inputPorts[0]);
      if (!swPort || !hwPort ||
          !canMapSoftwareTypeToHardware(swPort->type, hwPort->type))
        continue;

      auto result = state.mapNode(dfgSid, adgSid, dfg, adg);
      if (result == ActionResult::Success) {
        usedAdgOut.insert(ai);
        bound = true;
        llvm::outs() << "    Bound DFG output sentinel " << dfgSid
                     << " -> ADG output sentinel " << adgSid << "\n";
        break;
      }
    }

    if (!bound) {
      llvm::errs() << "Mapper: failed to bind DFG output sentinel " << dfgSid
                   << "\n";
    }
  }

  return true;
}

bool Mapper::bindMemrefSentinels(MappingState &state, const Graph &dfg,
                                 const Graph &adg) {
  for (IdIndex sid = 0; sid < static_cast<IdIndex>(dfg.nodes.size()); ++sid) {
    const Node *sNode = dfg.getNode(sid);
    if (!sNode || sNode->kind != Node::ModuleInputNode)
      continue;
    if (sNode->outputPorts.empty())
      continue;

    const Port *outPort = dfg.getPort(sNode->outputPorts[0]);
    if (!outPort || !mlir::isa<mlir::MemRefType>(outPort->type))
      continue;

    for (IdIndex opId : sNode->outputPorts) {
      const Port *op = dfg.getPort(opId);
      if (!op)
        continue;
      for (IdIndex eid : op->connectedEdges) {
        const Edge *edge = dfg.getEdge(eid);
        if (!edge || edge->srcPort != opId)
          continue;

        const Port *dstPort = dfg.getPort(edge->dstPort);
        if (!dstPort || dstPort->parentNode == INVALID_ID)
          continue;

        IdIndex dstNodeId = dstPort->parentNode;
        const Node *dstNode = dfg.getNode(dstNodeId);
        if (!dstNode)
          continue;

        llvm::StringRef dstOpName = getNodeAttrStr(dstNode, "op_name");
        if (!dstOpName.contains("extmemory"))
          continue;

        if (dstNodeId >= state.swNodeToHwNode.size() ||
            state.swNodeToHwNode[dstNodeId] == INVALID_ID)
          continue;

        IdIndex hwExtMemNodeId = state.swNodeToHwNode[dstNodeId];
        const Node *hwExtMemNode = adg.getNode(hwExtMemNodeId);
        if (!hwExtMemNode)
          continue;

        if (hwExtMemNode->inputPorts.empty())
          continue;

        IdIndex hwMemrefInPort = hwExtMemNode->inputPorts[0];

        llvm::SmallVector<IdIndex, 8> syntheticPath;
        syntheticPath.push_back(hwMemrefInPort);
        syntheticPath.push_back(hwMemrefInPort);

        if (eid < state.swEdgeToHwPaths.size() &&
            state.swEdgeToHwPaths[eid].size() == syntheticPath.size() &&
            std::equal(state.swEdgeToHwPaths[eid].begin(),
                       state.swEdgeToHwPaths[eid].end(),
                       syntheticPath.begin())) {
          continue;
        }

        auto result = state.mapEdge(eid, syntheticPath, dfg, adg);
        if (result == ActionResult::Success) {
          llvm::outs() << "    Pre-routed memref edge " << eid
                       << " (sentinel " << sid << " -> extmem " << dstNodeId
                       << ") as direct binding\n";
        }
      }
    }
  }

  return true;
}

bool Mapper::rebindScalarInputSentinels(MappingState &state, const Graph &dfg,
                                        const Graph &adg,
                                        const ADGFlattener &flattener) {
  std::vector<IdIndex> dfgScalarSentinels;
  std::vector<IdIndex> adgInputSentinels;

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node || node->kind != Node::ModuleInputNode ||
        node->outputPorts.empty())
      continue;
    const Port *swPort = dfg.getPort(node->outputPorts[0]);
    if (!swPort || isMemrefType(swPort->type))
      continue;
    dfgScalarSentinels.push_back(i);
  }

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (node && node->kind == Node::ModuleInputNode)
      adgInputSentinels.push_back(i);
  }

  if (dfgScalarSentinels.empty() || adgInputSentinels.empty())
    return true;

  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> candidateInputs;
  llvm::DenseMap<uint64_t, double> assignmentCostCache;
  llvm::DenseMap<IdIndex, double> emptyRoutingHistory;
  const TopologyModel *topologyModel = getActiveTopologyModel();
  auto estimateAssignmentCost = [&](IdIndex swSentinel, IdIndex adgSentinel) {
    uint64_t cacheKey =
        (static_cast<uint64_t>(static_cast<uint32_t>(swSentinel)) << 32) |
        static_cast<uint32_t>(adgSentinel);
    if (auto it = assignmentCostCache.find(cacheKey);
        it != assignmentCostCache.end()) {
      return it->second;
    }

    const Node *swNode = dfg.getNode(swSentinel);
    const Node *adgNode = adg.getNode(adgSentinel);
    if (!swNode || swNode->outputPorts.empty())
      return std::numeric_limits<double>::infinity();
    if (!adgNode || adgNode->outputPorts.empty())
      return std::numeric_limits<double>::infinity();
    IdIndex srcHwPort = adgNode->outputPorts[0];

    double cost = 0.0;
    IdIndex swOutId = swNode->outputPorts[0];
    const Port *swOut = dfg.getPort(swOutId);
    if (!swOut) {
      assignmentCostCache[cacheKey] = cost;
      return cost;
    }
    for (IdIndex edgeId : swOut->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge || edge->srcPort != swOutId)
        continue;
      const Port *dstPort = dfg.getPort(edge->dstPort);
      if (!dstPort || dstPort->parentNode == INVALID_ID ||
          dstPort->parentNode >= state.swNodeToHwNode.size()) {
        continue;
      }
      IdIndex dstHw = state.swNodeToHwNode[dstPort->parentNode];
      if (dstHw == INVALID_ID)
        continue;
      double weight = classifyEdgePlacementWeight(dfg, edgeId);
      IdIndex dstHwPort =
          edge->dstPort < state.swPortToHwPort.size()
              ? state.swPortToHwPort[edge->dstPort]
              : INVALID_ID;
      if (dstHwPort != INVALID_ID) {
        auto path = findPath(srcHwPort, dstHwPort, edgeId, state, dfg, adg,
                             emptyRoutingHistory);
        if (path.empty()) {
          cost = std::numeric_limits<double>::infinity();
          break;
        }
        cost += weight * static_cast<double>(path.size());
        continue;
      }
      if (topologyModel) {
        cost += weight * static_cast<double>(
                             topologyModel->placementDistance(adgSentinel,
                                                              dstHw));
      } else {
        cost += weight * static_cast<double>(
                             placementDistance(adgSentinel, dstHw, flattener));
      }
    }
    assignmentCostCache[cacheKey] = cost;
    return cost;
  };

  for (IdIndex swSentinel : dfgScalarSentinels) {
    const Node *swNode = dfg.getNode(swSentinel);
    const Port *swPort = swNode && !swNode->outputPorts.empty()
                             ? dfg.getPort(swNode->outputPorts[0])
                             : nullptr;
    if (!swPort)
      continue;
    auto &choices = candidateInputs[swSentinel];
    for (IdIndex adgSentinel : adgInputSentinels) {
      const Node *adgNode = adg.getNode(adgSentinel);
      const Port *hwPort = adgNode && !adgNode->outputPorts.empty()
                               ? adg.getPort(adgNode->outputPorts[0])
                               : nullptr;
      if (!hwPort ||
          !canMapSoftwareTypeToHardware(swPort->type, hwPort->type)) {
        continue;
      }
      choices.push_back(adgSentinel);
    }
    if (choices.empty())
      return false;
    llvm::stable_sort(choices, [&](IdIndex lhs, IdIndex rhs) {
      double lhsCost = estimateAssignmentCost(swSentinel, lhs);
      double rhsCost = estimateAssignmentCost(swSentinel, rhs);
      if (std::abs(lhsCost - rhsCost) > 1e-9)
        return lhsCost < rhsCost;
      return lhs < rhs;
    });
  }

  llvm::stable_sort(dfgScalarSentinels, [&](IdIndex lhs, IdIndex rhs) {
    size_t lhsCount = candidateInputs.lookup(lhs).size();
    size_t rhsCount = candidateInputs.lookup(rhs).size();
    if (lhsCount != rhsCount)
      return lhsCount < rhsCount;
    return lhs < rhs;
  });

  double bestCost = std::numeric_limits<double>::infinity();
  llvm::DenseMap<IdIndex, IdIndex> bestAssignment;
  llvm::DenseMap<IdIndex, IdIndex> currentAssignment;
  llvm::DenseSet<IdIndex> usedInputs;

  std::function<void(unsigned, double)> searchAssignments;
  searchAssignments = [&](unsigned depth, double currentCost) {
    if (currentCost >= bestCost)
      return;
    if (depth >= dfgScalarSentinels.size()) {
      bestCost = currentCost;
      bestAssignment = currentAssignment;
      return;
    }
    IdIndex swSentinel = dfgScalarSentinels[depth];
    for (IdIndex adgSentinel : candidateInputs.lookup(swSentinel)) {
      if (!usedInputs.insert(adgSentinel).second)
        continue;
      currentAssignment[swSentinel] = adgSentinel;
      searchAssignments(depth + 1, currentCost + estimateAssignmentCost(
                                                     swSentinel, adgSentinel));
      currentAssignment.erase(swSentinel);
      usedInputs.erase(adgSentinel);
    }
  };
  searchAssignments(0, 0.0);
  if (bestAssignment.empty())
    return false;

  for (IdIndex swSentinel : dfgScalarSentinels)
    state.unmapNode(swSentinel, dfg, adg);

  for (IdIndex swSentinel : dfgScalarSentinels) {
    IdIndex adgSentinel = bestAssignment.lookup(swSentinel);
    if (adgSentinel == INVALID_ID ||
        state.mapNode(swSentinel, adgSentinel, dfg, adg) !=
            ActionResult::Success) {
      return false;
    }
  }
  return true;
}

} // namespace loom

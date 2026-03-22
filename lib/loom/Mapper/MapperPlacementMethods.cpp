#include "loom/Mapper/Mapper.h"

#include "MapperCongestionEstimator.h"
#include "MapperInternal.h"
#include "loom/Mapper/BridgeBinding.h"
#include "loom/Mapper/TypeCompat.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <vector>

namespace loom {

using namespace mapper_detail;

namespace {

bool functionUnitPortsCompatible(const Node *swNode, const Node *hwNode,
                                 const Graph &dfg, const Graph &adg) {
  if (!swNode || !hwNode)
    return false;
  if (swNode->inputPorts.size() > hwNode->inputPorts.size())
    return false;
  if (swNode->outputPorts.size() > hwNode->outputPorts.size())
    return false;
  for (unsigned i = 0; i < swNode->inputPorts.size(); ++i) {
    const Port *swPort = dfg.getPort(swNode->inputPorts[i]);
    const Port *hwPort = adg.getPort(hwNode->inputPorts[i]);
    if (!swPort || !hwPort ||
        !canMapSoftwareTypeToHardware(swPort->type, hwPort->type))
      return false;
  }
  for (unsigned i = 0; i < swNode->outputPorts.size(); ++i) {
    const Port *swPort = dfg.getPort(swNode->outputPorts[i]);
    const Port *hwPort = adg.getPort(hwNode->outputPorts[i]);
    if (!swPort || !hwPort ||
        !canMapSoftwareTypeToHardware(swPort->type, hwPort->type))
      return false;
  }
  return true;
}

bool canMapSoftwareTypeToDirectMemoryHardware(mlir::Type swType,
                                              mlir::Type hwType) {
  if (mlir::isa<mlir::MemRefType>(swType) || mlir::isa<mlir::MemRefType>(hwType))
    return canMapSoftwareTypeToHardware(swType, hwType);
  return canMapSoftwareTypeToBridgeHardware(swType, hwType);
}

bool directMemoryHardwarePortAllowsSharing(const Port *hwPort) {
  if (!hwPort)
    return false;
  if (mlir::isa<mlir::MemRefType>(hwPort->type))
    return true;
  auto info = detail::getPortTypeInfo(hwPort->type);
  return info && info->isTagged;
}

std::optional<double>
estimateNodeDistanceToHardware(IdIndex swNode, IdIndex anchorHwNode,
                               const MappingState &state, const Graph &dfg,
                               const ADGFlattener &flattener,
                               const CandidateMap &candidates) {
  if (swNode < state.swNodeToHwNode.size()) {
    IdIndex mappedHw = state.swNodeToHwNode[swNode];
    if (mappedHw != INVALID_ID) {
      return static_cast<double>(
          placementDistance(anchorHwNode, mappedHw, flattener));
    }
  }

  auto it = candidates.find(swNode);
  if (it == candidates.end() || it->second.empty())
    return std::nullopt;

  double distanceSum = 0.0;
  unsigned count = 0;
  for (IdIndex candidateHw : it->second) {
    distanceSum += static_cast<double>(
        placementDistance(anchorHwNode, candidateHw, flattener));
    ++count;
  }
  if (count == 0)
    return std::nullopt;
  return distanceSum / static_cast<double>(count);
}

} // namespace

llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>>
Mapper::buildCandidates(const Graph &dfg, const Graph &adg) {
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> candidates;

  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;
    if (swNode->kind != Node::OperationNode)
      continue;

    llvm::StringRef opName = getNodeAttrStr(swNode, "op_name");

    if (isSoftwareMemoryInterfaceOp(opName)) {
      bool isExtMem = (opName == "handshake.extmemory");
      DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);
      for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
           ++hwId) {
        const Node *hwNode = adg.getNode(hwId);
        if (!hwNode)
          continue;
        if (getNodeAttrStr(hwNode, "resource_class") != "memory")
          continue;

        BridgeInfo bridge = BridgeInfo::extract(hwNode);
        if (bridge.hasBridge) {
          if (!isBridgeCompatible(bridge, memInfo, swNode, hwNode, dfg, adg))
            continue;
        } else {
          unsigned hwLdCount = static_cast<unsigned>(
              std::max<int64_t>(0, getNodeAttrInt(hwNode, "ldCount", 0)));
          unsigned hwStCount = static_cast<unsigned>(
              std::max<int64_t>(0, getNodeAttrInt(hwNode, "stCount", 0)));
          if (static_cast<unsigned>(std::max<int64_t>(memInfo.ldCount, 0)) >
                  hwLdCount ||
              static_cast<unsigned>(std::max<int64_t>(memInfo.stCount, 0)) >
                  hwStCount) {
            continue;
          }

          if (memInfo.swInSkip > 0) {
            if (swNode->inputPorts.empty() || hwNode->inputPorts.empty())
              continue;
            const Port *swMemPort = dfg.getPort(swNode->inputPorts[0]);
            const Port *hwMemPort = adg.getPort(hwNode->inputPorts[0]);
            if (!swMemPort || !hwMemPort ||
                !canMapSoftwareTypeToHardware(swMemPort->type,
                                             hwMemPort->type)) {
              continue;
            }
          }

          bool inputTypesOk = true;
          for (unsigned si = memInfo.swInSkip; si < swNode->inputPorts.size();
               ++si) {
            const Port *sp = dfg.getPort(swNode->inputPorts[si]);
            BridgePortCategory cat = memInfo.classifyInput(si - memInfo.swInSkip);
            unsigned lane = memInfo.inputLocalLane(si - memInfo.swInSkip);
            IdIndex hwPid =
                getExpandedMemoryInputPort(hwNode, adg, isExtMem, cat, lane);
            const Port *hp = adg.getPort(hwPid);
            if (!sp || hwPid == INVALID_ID || !hp ||
                !canMapSoftwareTypeToDirectMemoryHardware(sp->type, hp->type)) {
              inputTypesOk = false;
              break;
            }
          }
          if (!inputTypesOk)
            continue;

          bool outputTypesOk = true;
          for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
            const Port *sp = dfg.getPort(swNode->outputPorts[oi]);
            BridgePortCategory cat = memInfo.classifyOutput(oi);
            unsigned lane = memInfo.outputLocalLane(oi);
            IdIndex hwPid = getExpandedMemoryOutputPort(hwNode, adg, cat, lane);
            const Port *hp = adg.getPort(hwPid);
            if (!sp || !hp ||
                !canMapSoftwareTypeToDirectMemoryHardware(sp->type, hp->type)) {
              outputTypesOk = false;
              break;
            }
          }
          if (!outputTypesOk)
            continue;
        }
        candidates[swId].push_back(hwId);
      }
      continue;
    }

    for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
         ++hwId) {
      const Node *hwNode = adg.getNode(hwId);
      if (!hwNode)
        continue;
      if (getNodeAttrStr(hwNode, "resource_class") != "functional")
        continue;
      if (opMatchesFU(opName, hwNode) &&
          functionUnitPortsCompatible(swNode, hwNode, dfg, adg)) {
        candidates[swId].push_back(hwId);
      }
    }
  }

  return candidates;
}

std::vector<IdIndex> Mapper::computePlacementOrder(const Graph &dfg) {
  std::vector<IdIndex> order;
  std::vector<unsigned> inDegree(dfg.nodes.size(), 0);
  std::queue<IdIndex> worklist;

  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *edge = dfg.getEdge(eid);
    if (!edge)
      continue;
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (dstPort && dstPort->parentNode != INVALID_ID &&
        dstPort->parentNode < inDegree.size()) {
      inDegree[dstPort->parentNode]++;
    }
  }

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    if (dfg.getNode(i) && inDegree[i] == 0)
      worklist.push(i);
  }

  llvm::DenseSet<IdIndex> visited;
  while (!worklist.empty()) {
    IdIndex cur = worklist.front();
    worklist.pop();
    if (visited.count(cur))
      continue;
    visited.insert(cur);
    order.push_back(cur);

    const Node *node = dfg.getNode(cur);
    if (!node)
      continue;

    for (IdIndex opId : node->outputPorts) {
      const Port *outPort = dfg.getPort(opId);
      if (!outPort)
        continue;
      for (IdIndex eid : outPort->connectedEdges) {
        const Edge *edge = dfg.getEdge(eid);
        if (!edge)
          continue;
        const Port *dstPort = dfg.getPort(edge->dstPort);
        if (!dstPort || dstPort->parentNode == INVALID_ID)
          continue;
        IdIndex dstNode = dstPort->parentNode;
        if (dstNode < inDegree.size() && !visited.count(dstNode)) {
          inDegree[dstNode]--;
          if (inDegree[dstNode] == 0)
            worklist.push(dstNode);
        }
      }
    }
  }

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    if (dfg.getNode(i) && !visited.count(i))
      order.push_back(i);
  }

  llvm::stable_sort(order, [&](IdIndex lhs, IdIndex rhs) {
    const Node *lhsNode = dfg.getNode(lhs);
    const Node *rhsNode = dfg.getNode(rhs);
    auto nodeRank = [&](IdIndex id, const Node *node) {
      if (!node || node->kind != Node::OperationNode)
        return std::pair<double, double>{-1.0, -1.0};
      double priority = computeNodePriorityWeight(id, dfg);
      double boundary =
          (isMemoryOp(node) ? 1.0 : 0.0) +
          (getNodeAttrStr(node, "op_name") == "handshake.load" ? 1.0 : 0.0) +
          (getNodeAttrStr(node, "op_name") == "handshake.store" ? 1.0 : 0.0);
      return std::pair<double, double>{boundary, priority};
    };
    auto lhsRank = nodeRank(lhs, lhsNode);
    auto rhsRank = nodeRank(rhs, rhsNode);
    if (lhsRank.first != rhsRank.first)
      return lhsRank.first > rhsRank.first;
    if (lhsRank.second != rhsRank.second)
      return lhsRank.second > rhsRank.second;
    return lhs < rhs;
  });

  return order;
}

double Mapper::scorePlacement(
    IdIndex swNode, IdIndex hwNode, const MappingState &state,
    const Graph &dfg, const Graph &adg, const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates) {
  const Node *swN = dfg.getNode(swNode);
  if (!swN)
    return 0.0;
  const Node *hwN = adg.getNode(hwNode);
  if (!hwN)
    return -1.0e18;

  double weightedDist = 0.0;
  double totalWeight = 0.0;
  auto accumulateNeighborAt = [&](IdIndex otherSwNode, IdIndex edgeId) {
    auto estimate = estimateNodeDistanceToHardware(otherSwNode, hwNode, state,
                                                   dfg, flattener, candidates);
    if (!estimate)
      return;
    double edgeWeight = classifyEdgePlacementWeight(dfg, edgeId);
    weightedDist += edgeWeight * *estimate;
    totalWeight += edgeWeight;
  };
  auto accumulateNeighbor = [&](IdIndex otherSwNode, IdIndex edgeId) {
    accumulateNeighborAt(otherSwNode, edgeId);
  };

  bool usedBridgeBoundaryScoring = false;
  if (isSoftwareMemoryInterfaceOp(getNodeAttrStr(swN, "op_name"))) {
    bool isExtMem = (getNodeAttrStr(swN, "op_name") == "handshake.extmemory");
    BridgeInfo bridge = BridgeInfo::extract(hwN);
    if (bridge.hasBridge) {
      DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swN, dfg, isExtMem);
      for (unsigned si = memInfo.swInSkip; si < swN->inputPorts.size(); ++si) {
        IdIndex swPortId = swN->inputPorts[si];
        const Port *swPort = dfg.getPort(swPortId);
        if (!swPort)
          continue;
        BridgePortCategory cat = memInfo.classifyInput(si - memInfo.swInSkip);
        unsigned lane = memInfo.inputLocalLane(si - memInfo.swInSkip);
        IdIndex hwPortId =
            findBridgePortForCategoryLane(bridge, true, cat, lane);
        if (hwPortId == INVALID_ID)
          continue;
        for (IdIndex eid : swPort->connectedEdges) {
          const Edge *edge = dfg.getEdge(eid);
          if (!edge || edge->dstPort != swPortId)
            continue;
          const Port *srcPort = dfg.getPort(edge->srcPort);
          if (!srcPort || srcPort->parentNode == INVALID_ID)
            continue;
          accumulateNeighbor(srcPort->parentNode, eid);
          usedBridgeBoundaryScoring = true;
        }
      }
      for (unsigned oi = 0; oi < swN->outputPorts.size(); ++oi) {
        IdIndex swPortId = swN->outputPorts[oi];
        const Port *swPort = dfg.getPort(swPortId);
        if (!swPort)
          continue;
        BridgePortCategory cat = memInfo.classifyOutput(oi);
        unsigned lane = memInfo.outputLocalLane(oi);
        IdIndex hwPortId =
            findBridgePortForCategoryLane(bridge, false, cat, lane);
        if (hwPortId == INVALID_ID)
          continue;
        for (IdIndex eid : swPort->connectedEdges) {
          const Edge *edge = dfg.getEdge(eid);
          if (!edge || edge->srcPort != swPortId)
            continue;
          const Port *dstPort = dfg.getPort(edge->dstPort);
          if (!dstPort || dstPort->parentNode == INVALID_ID)
            continue;
          accumulateNeighbor(dstPort->parentNode, eid);
          usedBridgeBoundaryScoring = true;
        }
      }
    }
  }

  if (!usedBridgeBoundaryScoring) {
    for (IdIndex ipId : swN->inputPorts) {
      const Port *ip = dfg.getPort(ipId);
      if (!ip)
        continue;
      for (IdIndex eid : ip->connectedEdges) {
        const Edge *edge = dfg.getEdge(eid);
        if (!edge || edge->dstPort != ipId)
          continue;
        const Port *srcPort = dfg.getPort(edge->srcPort);
        if (!srcPort || srcPort->parentNode == INVALID_ID)
          continue;
        accumulateNeighbor(srcPort->parentNode, eid);
      }
    }

    for (IdIndex opId : swN->outputPorts) {
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
        accumulateNeighbor(dstPort->parentNode, eid);
      }
    }
  }

  double cost = 0.0;
  if (totalWeight > 0.0)
    cost += weightedDist / totalWeight;
  else if (const TopologyModel *topologyModel = getActiveTopologyModel())
    cost += topologyModel->averagePlacementDistance(hwNode);
  else if (auto hwPos = flattener.getNodeGridPos(hwNode);
           hwPos.first >= 0 && hwPos.second >= 0)
    cost += 0.25 *
            static_cast<double>(std::abs(hwPos.first) + std::abs(hwPos.second));

  if (activeMemorySharingPenalty > 0.0 &&
      isSoftwareMemoryInterfaceOp(getNodeAttrStr(swN, "op_name")) &&
      getNodeAttrStr(hwN, "resource_class") == "memory") {
    unsigned colocatedMemoryInterfaces = 0;
    if (hwNode < state.hwNodeToSwNodes.size()) {
      for (IdIndex otherSwId : state.hwNodeToSwNodes[hwNode]) {
        if (otherSwId == swNode)
          continue;
        const Node *otherSwNode = dfg.getNode(otherSwId);
        if (!otherSwNode || otherSwNode->kind != Node::OperationNode)
          continue;
        if (isSoftwareMemoryInterfaceOp(getNodeAttrStr(otherSwNode, "op_name")))
          ++colocatedMemoryInterfaces;
      }
    }
    if (colocatedMemoryInterfaces > 0) {
      cost += activeMemorySharingPenalty *
              static_cast<double>(colocatedMemoryInterfaces);
    }
  }

  cost += 0.6 * computeLocalSpreadPenalty(hwNode, state, adg, flattener);

  if (activeCongestionEstimator && activeFlattener &&
      activeCongestionPlacementWeight > 0.0) {
    double congestionPenalty = 0.0;
    for (IdIndex ipId : swN->inputPorts) {
      const Port *ip = dfg.getPort(ipId);
      if (!ip)
        continue;
      for (IdIndex eid : ip->connectedEdges) {
        const Edge *edge = dfg.getEdge(eid);
        if (!edge || edge->dstPort != ipId)
          continue;
        const Port *srcPort = dfg.getPort(edge->srcPort);
        if (!srcPort || srcPort->parentNode == INVALID_ID)
          continue;
        IdIndex srcHw = srcPort->parentNode < state.swNodeToHwNode.size()
                            ? state.swNodeToHwNode[srcPort->parentNode]
                            : INVALID_ID;
        if (srcHw != INVALID_ID) {
          congestionPenalty +=
              activeCongestionEstimator->demandCapacityRatio(
                  srcHw, hwNode, adg, *activeFlattener);
        }
      }
    }
    for (IdIndex opId : swN->outputPorts) {
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
        IdIndex dstHw = dstPort->parentNode < state.swNodeToHwNode.size()
                            ? state.swNodeToHwNode[dstPort->parentNode]
                            : INVALID_ID;
        if (dstHw != INVALID_ID) {
          congestionPenalty +=
              activeCongestionEstimator->demandCapacityRatio(
                  hwNode, dstHw, adg, *activeFlattener);
        }
      }
    }
    cost += activeCongestionPlacementWeight * congestionPenalty;
  }

  cost += computeNodeTimingPenalty(swNode, hwNode, dfg, adg);
  return -cost;
}

bool Mapper::runPlacement(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    const Options &opts) {
  std::mt19937 rng(static_cast<unsigned>(opts.seed));

  auto order = computePlacementOrder(dfg);
  llvm::stable_sort(order, [&](IdIndex lhs, IdIndex rhs) {
    const Node *lhsNode = dfg.getNode(lhs);
    const Node *rhsNode = dfg.getNode(rhs);
    auto candidateCount = [&](IdIndex swId, const Node *node) -> size_t {
      if (!node || node->kind != Node::OperationNode)
        return std::numeric_limits<size_t>::max();
      auto it = candidates.find(swId);
      if (it == candidates.end())
        return std::numeric_limits<size_t>::max();
      return it->second.size();
    };
    size_t lhsCount = candidateCount(lhs, lhsNode);
    size_t rhsCount = candidateCount(rhs, rhsNode);
    if (lhsCount != rhsCount)
      return lhsCount < rhsCount;
    return false;
  });

  for (IdIndex swId : order) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;
    if (swNode->kind != Node::OperationNode)
      continue;
    if (swId < state.swNodeToHwNode.size() &&
        state.swNodeToHwNode[swId] != INVALID_ID)
      continue;

    auto candIt = candidates.find(swId);
    if (candIt == candidates.end() || candIt->second.empty()) {
      llvm::errs() << "Mapper: no candidates for DFG node " << swId
                    << " (" << getNodeAttrStr(swNode, "op_name") << ")\n";
      return false;
    }

    IdIndex bestHw = INVALID_ID;
    double bestScore = -1e18;
    llvm::SmallVector<std::pair<double, IdIndex>, 8> rankedCandidates;

    for (IdIndex hwId : candIt->second) {
      const Node *hwNode = adg.getNode(hwId);
      if (!hwNode)
        continue;

      if (!state.hwNodeToSwNodes[hwId].empty()) {
        if (getNodeAttrStr(hwNode, "resource_class") == "memory") {
          int64_t numRegion = getNodeAttrInt(hwNode, "numRegion", 1);
          if (static_cast<int64_t>(state.hwNodeToSwNodes[hwId].size()) >=
              numRegion)
            continue;
        } else {
          continue;
        }
      }

      llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
      if (isSpatialPEOccupied(state, adg, flattener, peName))
        continue;

      double score =
          scorePlacement(swId, hwId, state, dfg, adg, flattener, candidates);
      rankedCandidates.push_back({-score, hwId});
      if (score > bestScore || bestHw == INVALID_ID) {
        bestScore = score;
        bestHw = hwId;
      }
    }

    if (bestHw == INVALID_ID) {
      llvm::errs() << "Mapper: failed to place DFG node " << swId
                    << " (" << getNodeAttrStr(swNode, "op_name") << ")\n";
      return false;
    }

    llvm::stable_sort(rankedCandidates, [&](const auto &lhs, const auto &rhs) {
      if (lhs.first != rhs.first)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });

    llvm::SmallVector<IdIndex, 4> shortlist;
    for (const auto &candidate : rankedCandidates) {
      double score = -candidate.first;
      if (!shortlist.empty() && bestScore - score > 0.35 &&
          shortlist.size() >= 2)
        break;
      shortlist.push_back(candidate.second);
      if (shortlist.size() >= 4)
        break;
    }
    if (!shortlist.empty()) {
      std::uniform_int_distribution<size_t> dist(0, shortlist.size() - 1);
      bestHw = shortlist[dist(rng)];
    }

    llvm::SmallVector<IdIndex, 8> tryOrder(shortlist.begin(), shortlist.end());
    for (const auto &candidate : rankedCandidates) {
      if (llvm::is_contained(tryOrder, candidate.second))
        continue;
      tryOrder.push_back(candidate.second);
    }

    bool placed = false;
    ActionResult lastResult = ActionResult::FailedInternalError;
    for (IdIndex hwCandidate : tryOrder) {
      auto savepoint = state.beginSavepoint();
      lastResult = state.mapNode(swId, hwCandidate, dfg, adg);
      if (lastResult == ActionResult::Success &&
          bindMappedNodePorts(swId, state, dfg, adg)) {
        bestHw = hwCandidate;
        state.commitSavepoint(savepoint);
        placed = true;
        break;
      }
      state.rollbackSavepoint(savepoint);
    }
    if (!placed) {
      llvm::errs() << "Mapper: port binding failed for " << swId
                   << " after trying " << tryOrder.size() << " candidate(s)\n";
      return false;
    }

    if (opts.verbose) {
      llvm::outs() << "  Placed " << getNodeAttrStr(swNode, "op_name")
                   << " (node " << swId << ") -> HW node " << bestHw << "\n";
    }
  }

  return true;
}

double Mapper::computeTotalCost(const MappingState &state, const Graph &dfg,
                                const Graph &adg,
                                const ADGFlattener &flattener) {
  double cost = 0.0;
  const TopologyModel *topologyModel = getActiveTopologyModel();
  const bool enableGridCutLoad =
      topologyModel && topologyModel->supportsGridCutLoad();
  int maxRow = -1;
  int maxCol = -1;
  if (enableGridCutLoad) {
    for (IdIndex hwId = 0;
         hwId < static_cast<IdIndex>(state.hwNodeToSwNodes.size()); ++hwId) {
      if (state.hwNodeToSwNodes[hwId].empty())
        continue;
      auto [row, col] = flattener.getNodeGridPos(hwId);
      if (row >= 0 && col >= 0) {
        maxRow = std::max(maxRow, row);
        maxCol = std::max(maxCol, col);
      }
    }
  }
  std::vector<double> rowCutLoad(
      enableGridCutLoad && maxRow >= 0 ? static_cast<size_t>(maxRow) + 1 : 0,
      0.0);
  std::vector<double> colCutLoad(
      enableGridCutLoad && maxCol >= 0 ? static_cast<size_t>(maxCol) + 1 : 0,
      0.0);

  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *edge = dfg.getEdge(eid);
    if (!edge)
      continue;
    const Port *sp = dfg.getPort(edge->srcPort);
    const Port *dp = dfg.getPort(edge->dstPort);
    if (!sp || !dp || sp->parentNode == INVALID_ID ||
        dp->parentNode == INVALID_ID)
      continue;
    IdIndex srcSw = sp->parentNode;
    IdIndex dstSw = dp->parentNode;
    if (srcSw >= state.swNodeToHwNode.size() ||
        dstSw >= state.swNodeToHwNode.size())
      continue;
    IdIndex srcHw = state.swNodeToHwNode[srcSw];
    IdIndex dstHw = state.swNodeToHwNode[dstSw];
    if (srcHw == INVALID_ID || dstHw == INVALID_ID)
      continue;
    double edgeWeight = classifyEdgePlacementWeight(dfg, eid);
    cost += edgeWeight *
            static_cast<double>(placementDistance(srcHw, dstHw, flattener));
    if (enableGridCutLoad) {
      auto [sr, sc] = flattener.getNodeGridPos(srcHw);
      auto [dr, dc] = flattener.getNodeGridPos(dstHw);
      if (sr < 0 || sc < 0 || dr < 0 || dc < 0)
        continue;
      for (int row = std::min(sr, dr); row < std::max(sr, dr) &&
                                       row < static_cast<int>(rowCutLoad.size());
           ++row) {
        rowCutLoad[row] += edgeWeight;
      }
      for (int col = std::min(sc, dc); col < std::max(sc, dc) &&
                                       col < static_cast<int>(colCutLoad.size());
           ++col) {
        colCutLoad[col] += edgeWeight;
      }
    }
  }
  for (IdIndex swNode = 0;
       swNode < static_cast<IdIndex>(state.swNodeToHwNode.size()); ++swNode) {
    IdIndex hwNode = state.swNodeToHwNode[swNode];
    if (hwNode == INVALID_ID)
      continue;
    cost += computeNodeTimingPenalty(swNode, hwNode, dfg, adg);
  }
  if (enableGridCutLoad) {
    for (double load : rowCutLoad)
      cost += 0.006 * load * load;
    for (double load : colCutLoad)
      cost += 0.006 * load * load;
  }
  return cost;
}

bool Mapper::bindMappedNodePorts(IdIndex swId, MappingState &state,
                                 const Graph &dfg, const Graph &adg) {
  if (swId >= state.swNodeToHwNode.size())
    return false;

  IdIndex bestHw = state.swNodeToHwNode[swId];
  if (bestHw == INVALID_ID)
    return false;

  const Node *swNode = dfg.getNode(swId);
  const Node *placedHwNode = adg.getNode(bestHw);
  if (!swNode || !placedHwNode)
    return false;

  auto verifyIncidentReachability = [&]() -> bool {
    llvm::StringRef swOpName = getNodeAttrStr(swNode, "op_name");
    if (mapper_detail::isSoftwareMemoryInterfaceOp(swOpName))
      return true;
    llvm::DenseMap<IdIndex, double> emptyHistory;
    llvm::DenseSet<IdIndex> seenEdges;
    auto checkPorts = [&](llvm::ArrayRef<IdIndex> ports) -> bool {
      for (IdIndex swPid : ports) {
        const Port *swPort = dfg.getPort(swPid);
        if (!swPort)
          continue;
        for (IdIndex edgeId : swPort->connectedEdges) {
          if (!seenEdges.insert(edgeId).second)
            continue;
          const Edge *edge = dfg.getEdge(edgeId);
          if (!edge)
            continue;
          IdIndex srcHwPid =
              edge->srcPort < state.swPortToHwPort.size()
                  ? state.swPortToHwPort[edge->srcPort]
                  : INVALID_ID;
          IdIndex dstHwPid =
              edge->dstPort < state.swPortToHwPort.size()
                  ? state.swPortToHwPort[edge->dstPort]
                  : INVALID_ID;
          if (srcHwPid == INVALID_ID || dstHwPid == INVALID_ID)
            continue;
          if (srcHwPid == dstHwPid)
            continue;
          const Port *srcHwPort = adg.getPort(srcHwPid);
          const Port *dstHwPort = adg.getPort(dstHwPid);
          const Node *srcHwNode =
              (srcHwPort && srcHwPort->parentNode != INVALID_ID)
                  ? adg.getNode(srcHwPort->parentNode)
                  : nullptr;
          const Node *dstHwNode =
              (dstHwPort && dstHwPort->parentNode != INVALID_ID)
                  ? adg.getNode(dstHwPort->parentNode)
                  : nullptr;
          if (srcHwNode && dstHwNode) {
            llvm::StringRef srcPe = getNodeAttrStr(srcHwNode, "pe_name");
            llvm::StringRef dstPe = getNodeAttrStr(dstHwNode, "pe_name");
            if (!srcPe.empty() && srcPe == dstPe)
              continue;
          }
          auto path =
              findPath(srcHwPid, dstHwPid, edgeId, state, dfg, adg, emptyHistory);
          if (path.empty())
            return false;
        }
      }
      return true;
    };
    return checkPorts(swNode->inputPorts) && checkPorts(swNode->outputPorts);
  };

  if (getNodeAttrStr(placedHwNode, "resource_class") != "memory")
    return verifyIncidentReachability();

  BridgeInfo bridge = BridgeInfo::extract(placedHwNode);
  bool isExtMem = (getNodeAttrStr(placedHwNode, "op_kind") == "extmemory");
  if (bridge.hasBridge) {
    DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);
    return bindBridgeInputs(bridge, memInfo, swNode, placedHwNode, dfg, adg,
                            state) &&
           bindBridgeOutputs(bridge, memInfo, swNode, placedHwNode, dfg, adg,
                             state) &&
           verifyIncidentReachability();
  }

  llvm::StringRef hwKind = getNodeAttrStr(placedHwNode, "op_kind");
  bool isScalarMemory = (hwKind == "memory" || hwKind == "extmemory");
  if (isScalarMemory) {
    DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);

    if (memInfo.swInSkip > 0 && !swNode->inputPorts.empty() &&
        !placedHwNode->inputPorts.empty()) {
      const Port *swMemPort = dfg.getPort(swNode->inputPorts[0]);
      IdIndex hwMemPid = placedHwNode->inputPorts[0];
      const Port *hwMemPort = adg.getPort(hwMemPid);
      if (!swMemPort || !hwMemPort ||
          !canMapSoftwareTypeToHardware(swMemPort->type, hwMemPort->type)) {
        return false;
      }
      state.mapPort(swNode->inputPorts[0], hwMemPid, dfg, adg);
    }

    for (unsigned si = memInfo.swInSkip; si < swNode->inputPorts.size(); ++si) {
      IdIndex swPid = swNode->inputPorts[si];
      const Port *sp = dfg.getPort(swPid);
      BridgePortCategory cat = memInfo.classifyInput(si - memInfo.swInSkip);
      unsigned lane = memInfo.inputLocalLane(si - memInfo.swInSkip);
      IdIndex hwPid =
          getExpandedMemoryInputPort(placedHwNode, adg, isExtMem, cat, lane);
      const Port *hp = adg.getPort(hwPid);
      if (!sp || hwPid == INVALID_ID || !hp ||
          (!state.hwPortToSwPorts[hwPid].empty() &&
           !directMemoryHardwarePortAllowsSharing(hp)) ||
          !canMapSoftwareTypeToDirectMemoryHardware(sp->type, hp->type)) {
        return false;
      }
      if (state.mapPort(swPid, hwPid, dfg, adg) != ActionResult::Success)
        return false;
    }

    for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
      IdIndex swPid = swNode->outputPorts[oi];
      const Port *sp = dfg.getPort(swPid);
      BridgePortCategory cat = memInfo.classifyOutput(oi);
      unsigned lane = memInfo.outputLocalLane(oi);
      IdIndex hwPid = getExpandedMemoryOutputPort(placedHwNode, adg, cat, lane);
      const Port *hp = adg.getPort(hwPid);
      if (!sp || hwPid == INVALID_ID || !hp ||
          (!state.hwPortToSwPorts[hwPid].empty() &&
           !directMemoryHardwarePortAllowsSharing(hp)) ||
          !canMapSoftwareTypeToDirectMemoryHardware(sp->type, hp->type)) {
        return false;
      }
      if (state.mapPort(swPid, hwPid, dfg, adg) != ActionResult::Success)
        return false;
    }
    return verifyIncidentReachability();
  }

  llvm::DenseMap<IdIndex, double> emptyHistory;
  auto estimateInputBindingCost = [&](IdIndex swPid, IdIndex hwPid) {
    double cost = 0.0;
    bool observed = false;
    const Port *swPort = dfg.getPort(swPid);
    if (!swPort)
      return 0.0;
    for (IdIndex edgeId : swPort->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge || edge->dstPort != swPid)
        continue;
      IdIndex srcHwPid =
          edge->srcPort < state.swPortToHwPort.size()
              ? state.swPortToHwPort[edge->srcPort]
              : INVALID_ID;
      if (srcHwPid == INVALID_ID)
        continue;
      observed = true;
      auto path = findPath(srcHwPid, hwPid, edgeId, state, dfg, adg, emptyHistory);
      cost += path.empty() ? 1.0e6 : static_cast<double>(path.size());
    }
    return observed ? cost : 0.0;
  };
  auto estimateOutputBindingCost = [&](IdIndex swPid, IdIndex hwPid) {
    double cost = 0.0;
    bool observed = false;
    const Port *swPort = dfg.getPort(swPid);
    if (!swPort)
      return 0.0;
    for (IdIndex edgeId : swPort->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge || edge->srcPort != swPid)
        continue;
      IdIndex dstHwPid =
          edge->dstPort < state.swPortToHwPort.size()
              ? state.swPortToHwPort[edge->dstPort]
              : INVALID_ID;
      if (dstHwPid == INVALID_ID)
        continue;
      observed = true;
      auto path = findPath(hwPid, dstHwPid, edgeId, state, dfg, adg, emptyHistory);
      cost += path.empty() ? 1.0e6 : static_cast<double>(path.size());
    }
    return observed ? cost : 0.0;
  };

  llvm::SmallVector<bool, 8> usedIn(placedHwNode->inputPorts.size(), false);
  for (unsigned si = 0; si < swNode->inputPorts.size(); ++si) {
    IdIndex swPid = swNode->inputPorts[si];
    const Port *sp = dfg.getPort(swPid);
    if (!sp)
      continue;

    llvm::SmallVector<std::pair<double, unsigned>, 8> rankedInputs;
    for (unsigned hi = 0; hi < placedHwNode->inputPorts.size(); ++hi) {
      if (usedIn[hi])
        continue;
      IdIndex hwPid = placedHwNode->inputPorts[hi];
      if (!state.hwPortToSwPorts[hwPid].empty())
        continue;
      const Port *hp = adg.getPort(hwPid);
      if (!hp || !canMapSoftwareTypeToHardware(sp->type, hp->type))
        continue;
      rankedInputs.push_back({estimateInputBindingCost(swPid, hwPid), hi});
    }
    if (rankedInputs.empty())
      return false;
    llvm::stable_sort(rankedInputs, [&](const auto &lhs, const auto &rhs) {
      if (std::abs(lhs.first - rhs.first) > 1e-9)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });
    unsigned bestHi = rankedInputs.front().second;
    state.mapPort(swPid, placedHwNode->inputPorts[bestHi], dfg, adg);
    usedIn[bestHi] = true;
  }

  llvm::SmallVector<bool, 8> usedOut(placedHwNode->outputPorts.size(), false);
  for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
    IdIndex swPid = swNode->outputPorts[oi];
    const Port *sp = dfg.getPort(swPid);
    if (!sp)
      continue;

    llvm::SmallVector<std::pair<double, unsigned>, 8> rankedOutputs;
    for (unsigned hi = 0; hi < placedHwNode->outputPorts.size(); ++hi) {
      if (usedOut[hi])
        continue;
      IdIndex hwPid = placedHwNode->outputPorts[hi];
      if (!state.hwPortToSwPorts[hwPid].empty())
        continue;
      const Port *hp = adg.getPort(hwPid);
      if (!hp || !canMapSoftwareTypeToHardware(sp->type, hp->type))
        continue;
      rankedOutputs.push_back({estimateOutputBindingCost(swPid, hwPid), hi});
    }
    if (rankedOutputs.empty())
      return false;
    llvm::stable_sort(rankedOutputs, [&](const auto &lhs, const auto &rhs) {
      if (std::abs(lhs.first - rhs.first) > 1e-9)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });
    unsigned bestHi = rankedOutputs.front().second;
    state.mapPort(swPid, placedHwNode->outputPorts[bestHi], dfg, adg);
    usedOut[bestHi] = true;
  }
  return verifyIncidentReachability();
}

} // namespace loom

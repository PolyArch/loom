#include "MapperLocalRepairInternal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>

namespace loom {

using namespace mapper_detail;

bool LocalRepairDriver::runMemoryExactRepairs() {
  if (!bestAllRouted &&
      bestFailedEdges.size() <= repairOpts.cpSatSmallFailedThreshold) {
    if (mapper.shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
    llvm::DenseSet<IdIndex> clusterNodeSet;
    std::vector<IdIndex> memoryResponseCluster;
    auto maybeAddClusterNode = [&](IdIndex swNodeId) {
      if (swNodeId == INVALID_ID || !clusterNodeSet.insert(swNodeId).second)
        return;
      if (swNodeId >= state.swNodeToHwNode.size() ||
          state.swNodeToHwNode[swNodeId] == INVALID_ID)
        return;
      const Node *swNode = dfg.getNode(swNodeId);
      if (!swNode || swNode->kind != Node::OperationNode)
        return;
      memoryResponseCluster.push_back(swNodeId);
    };

    for (IdIndex edgeId : bestFailedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      const Port *dstPort = dfg.getPort(edge->dstPort);
      const Node *srcNode =
          (srcPort && srcPort->parentNode != INVALID_ID)
              ? dfg.getNode(srcPort->parentNode)
              : nullptr;
      const Node *dstNode =
          (dstPort && dstPort->parentNode != INVALID_ID)
              ? dfg.getNode(dstPort->parentNode)
              : nullptr;
      if (!srcNode || !dstNode || srcNode->kind != Node::OperationNode ||
          dstNode->kind != Node::OperationNode)
        continue;
      if (!isSoftwareMemoryInterfaceOp(getNodeAttrStr(srcNode, "op_name")) ||
          isSoftwareMemoryInterfaceOp(getNodeAttrStr(dstNode, "op_name"))) {
        continue;
      }

      maybeAddClusterNode(srcPort->parentNode);
      maybeAddClusterNode(dstPort->parentNode);

      auto addIncidentOpNeighbors = [&](const Node *node, IdIndex nodeId) {
        auto visitPort = [&](IdIndex portId) {
          const Port *port = dfg.getPort(portId);
          if (!port)
            return;
          for (IdIndex incidentEdgeId : port->connectedEdges) {
            const Edge *incidentEdge = dfg.getEdge(incidentEdgeId);
            if (!incidentEdge)
              continue;
            IdIndex otherPortId = INVALID_ID;
            if (incidentEdge->srcPort == portId)
              otherPortId = incidentEdge->dstPort;
            else if (incidentEdge->dstPort == portId)
              otherPortId = incidentEdge->srcPort;
            const Port *otherPort = dfg.getPort(otherPortId);
            if (!otherPort || otherPort->parentNode == INVALID_ID ||
                otherPort->parentNode == nodeId) {
              continue;
            }
            const Node *otherNode = dfg.getNode(otherPort->parentNode);
            if (!otherNode || otherNode->kind != Node::OperationNode)
              continue;
            llvm::StringRef otherOp = getNodeAttrStr(otherNode, "op_name");
            if (otherOp == "dataflow.gate" || otherOp == "arith.addi" ||
                otherOp == "dataflow.carry" ||
                otherOp == "handshake.cond_br") {
              maybeAddClusterNode(otherPort->parentNode);
            }
          }
        };
        for (IdIndex inPortId : node->inputPorts)
          visitPort(inPortId);
        for (IdIndex outPortId : node->outputPorts)
          visitPort(outPortId);
      };

      addIncidentOpNeighbors(srcNode, srcPort->parentNode);
      addIncidentOpNeighbors(dstNode, dstPort->parentNode);
    }

    if (!memoryResponseCluster.empty()) {
      std::vector<IdIndex> allMemoryNodes;
      for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
           ++swNodeId) {
        if (swNodeId >= state.swNodeToHwNode.size() ||
            state.swNodeToHwNode[swNodeId] == INVALID_ID)
          continue;
        const Node *swNode = dfg.getNode(swNodeId);
        if (!swNode || swNode->kind != Node::OperationNode)
          continue;
        if (!isSoftwareMemoryInterfaceOp(getNodeAttrStr(swNode, "op_name")))
          continue;
        allMemoryNodes.push_back(swNodeId);
      }
      unsigned additionalMemoryNodes = 0;
      for (IdIndex swNodeId : allMemoryNodes) {
        if (!clusterNodeSet.contains(swNodeId))
          ++additionalMemoryNodes;
      }
      if (allMemoryNodes.size() <= repairOpts.memoryFocusNodeLimit &&
          memoryResponseCluster.size() + additionalMemoryNodes <=
              repairOpts.memoryResponseClusterMax) {
        for (IdIndex swNodeId : allMemoryNodes) {
          if (clusterNodeSet.insert(swNodeId).second)
            memoryResponseCluster.push_back(swNodeId);
        }
      }
    }

    if (memoryResponseCluster.size() >= repairOpts.memoryResponseClusterMin &&
        memoryResponseCluster.size() <= repairOpts.memoryResponseClusterMax) {
      state.restore(bestPlacementCheckpoint);
      llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> exactDomains;
      size_t searchSpace = 1;

      llvm::stable_sort(memoryResponseCluster, [&](IdIndex lhs, IdIndex rhs) {
        llvm::StringRef lhsOp = getNodeAttrStr(dfg.getNode(lhs), "op_name");
        llvm::StringRef rhsOp = getNodeAttrStr(dfg.getNode(rhs), "op_name");
        bool lhsMemory = isSoftwareMemoryInterfaceOp(lhsOp);
        bool rhsMemory = isSoftwareMemoryInterfaceOp(rhsOp);
        if (lhsMemory != rhsMemory)
          return lhsMemory;
        return lhs < rhs;
      });

      for (IdIndex swNode : memoryResponseCluster) {
        IdIndex oldHw = state.swNodeToHwNode[swNode];
        auto candIt = candidates.find(swNode);
        if (oldHw == INVALID_ID || candIt == candidates.end() ||
            candIt->second.empty()) {
          exactDomains.clear();
          break;
        }

        llvm::StringRef opName = getNodeAttrStr(dfg.getNode(swNode), "op_name");
        const unsigned candidateLimit =
            isSoftwareMemoryInterfaceOp(opName) ? 3u : 5u;
        const unsigned moveRadius =
            isSoftwareMemoryInterfaceOp(opName) ? 0u : 5u;

        llvm::SmallVector<std::pair<double, IdIndex>, 16> rankedCandidates;
        for (IdIndex candHw : candIt->second) {
          if (moveRadius != 0 && candHw != oldHw &&
              !isWithinMoveRadius(oldHw, candHw, flattener, moveRadius)) {
            continue;
          }
          double candScore = mapper.scorePlacement(swNode, candHw, state, dfg, adg,
                                                   flattener, candidates);
          rankedCandidates.push_back({-candScore, candHw});
        }
        if (rankedCandidates.empty())
          rankedCandidates.push_back({0.0, oldHw});
        llvm::stable_sort(rankedCandidates,
                          [&](const auto &lhs, const auto &rhs) {
                            if (lhs.first != rhs.first)
                              return lhs.first < rhs.first;
                            return lhs.second < rhs.second;
                          });

        auto &domain = exactDomains[swNode];
        domain.push_back(oldHw);
        unsigned limit =
            std::min<unsigned>(rankedCandidates.size(), candidateLimit);
        for (unsigned idx = 0; idx < limit; ++idx) {
          IdIndex candHw = rankedCandidates[idx].second;
          if (std::find(domain.begin(), domain.end(), candHw) == domain.end())
            domain.push_back(candHw);
        }
        searchSpace *= domain.size();
      }

      if (!exactDomains.empty() &&
          searchSpace <= repairOpts.exactDomainSearchSpaceCap) {
        llvm::outs() << "  Exact memory-response cluster: nodes="
                     << memoryResponseCluster.size()
                     << ", searchSpace=" << searchSpace << "\n";

        for (IdIndex swNode : memoryResponseCluster)
          state.unmapNode(swNode, dfg, adg);

        const auto exactDeadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(static_cast<int64_t>(std::max(
                1.0, mapper.clampDeadlineMsToRemainingBudget(8000.0))));
        bool stopExactSearch = false;
        llvm::DenseSet<IdIndex> distinctMemoryCandidates;
        unsigned memoryNodeCount = 0;
        for (IdIndex swNode : memoryResponseCluster) {
          const Node *node = dfg.getNode(swNode);
          if (!node ||
              !isSoftwareMemoryInterfaceOp(getNodeAttrStr(node, "op_name")))
            continue;
          ++memoryNodeCount;
          for (IdIndex candHw : exactDomains.lookup(swNode))
            distinctMemoryCandidates.insert(candHw);
        }
        const bool preferDistinctMemoryAssignments =
            memoryNodeCount >= 2 &&
            distinctMemoryCandidates.size() >= memoryNodeCount;
        llvm::DenseSet<IdIndex> usedDistinctMemoryHws;

        std::function<void(unsigned, bool)> enumerateMemoryResponseCluster;
        enumerateMemoryResponseCluster = [&](unsigned depth,
                                            bool enforceDistinctMemory) {
          if (stopExactSearch || mapper.shouldStopForBudget("local repair") ||
              std::chrono::steady_clock::now() > exactDeadline)
            return;
          if (depth >= memoryResponseCluster.size()) {
            mapper.rebindScalarInputSentinels(state, dfg, adg, flattener);
            mapper.bindMemrefSentinels(state, dfg, adg);
            classifyTemporalRegisterEdges(state, dfg, adg, flattener,
                                          edgeKinds);
            auto responseRepairEdges = buildFocusedRepairNeighborhood(
                bestFailedEdges, repairOpts.earlyCPSatNeighborhoodLimit);
            if (responseRepairEdges.size() < bestFailedEdges.size())
              responseRepairEdges.assign(bestFailedEdges.begin(),
                                         bestFailedEdges.end());
            bool allRouted = mapper.runExactRoutingRepair(
                state, responseRepairEdges, dfg, adg, flattener, edgeKinds,
                opts, congestion, bestFailedEdges);
            if (updateBest(allRouted) && allRouted)
              stopExactSearch = true;
            return;
          }

          IdIndex swNode = memoryResponseCluster[depth];
          const Node *swNodeDef = dfg.getNode(swNode);
          const bool isMemoryNode =
              swNodeDef &&
              isSoftwareMemoryInterfaceOp(getNodeAttrStr(swNodeDef, "op_name"));
          for (IdIndex candHw : exactDomains.lookup(swNode)) {
            auto savepoint = state.beginSavepoint();
            if (enforceDistinctMemory && isMemoryNode &&
                usedDistinctMemoryHws.contains(candHw)) {
              state.rollbackSavepoint(savepoint);
              continue;
            }
            if (enforceDistinctMemory && isMemoryNode)
              usedDistinctMemoryHws.insert(candHw);
            if (state.mapNode(swNode, candHw, dfg, adg) !=
                ActionResult::Success) {
              if (enforceDistinctMemory && isMemoryNode)
                usedDistinctMemoryHws.erase(candHw);
              state.rollbackSavepoint(savepoint);
              continue;
            }
            if (!mapper.bindMappedNodePorts(swNode, state, dfg, adg)) {
              if (enforceDistinctMemory && isMemoryNode)
                usedDistinctMemoryHws.erase(candHw);
              state.rollbackSavepoint(savepoint);
              continue;
            }
            enumerateMemoryResponseCluster(depth + 1, enforceDistinctMemory);
            if (enforceDistinctMemory && isMemoryNode)
              usedDistinctMemoryHws.erase(candHw);
            state.rollbackSavepoint(savepoint);
            if (stopExactSearch)
              return;
          }
        };

        if (preferDistinctMemoryAssignments)
          enumerateMemoryResponseCluster(0, true);
        if (!stopExactSearch)
          enumerateMemoryResponseCluster(0, false);
        llvm::outs() << "  Exact memory-response cluster result: routed "
                     << bestRouted << "/" << dfg.edges.size() << " edges\n";
        if (bestAllRouted) {
          state.restore(bestCheckpoint);
          return true;
        }
      }
    }
  }

  if (!bestAllRouted &&
      bestFailedEdges.size() <= repairOpts.cpSatEscalationFailedEdgeThreshold) {
    if (mapper.shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
    llvm::DenseSet<IdIndex> seenMemoryNodes;
    std::vector<IdIndex> memoryFocusNodes;
    auto maybeAddMemoryNode = [&](IdIndex swNodeId) {
      if (swNodeId == INVALID_ID || !seenMemoryNodes.insert(swNodeId).second)
        return;
      if (swNodeId >= state.swNodeToHwNode.size() ||
          state.swNodeToHwNode[swNodeId] == INVALID_ID)
        return;
      const Node *swNode = dfg.getNode(swNodeId);
      if (!swNode || swNode->kind != Node::OperationNode)
        return;
      if (!isSoftwareMemoryInterfaceOp(getNodeAttrStr(swNode, "op_name")))
        return;
      memoryFocusNodes.push_back(swNodeId);
    };
    auto expandToAllSoftwareMemoryNodes =
        [&](std::vector<IdIndex> &nodes, llvm::DenseSet<IdIndex> &seen) {
          if (nodes.empty())
            return;
          std::vector<IdIndex> allMemoryNodes;
          allMemoryNodes.reserve(nodes.size());
          for (IdIndex swNodeId = 0;
               swNodeId < static_cast<IdIndex>(dfg.nodes.size()); ++swNodeId) {
            if (swNodeId >= state.swNodeToHwNode.size() ||
                state.swNodeToHwNode[swNodeId] == INVALID_ID)
              continue;
            const Node *swNode = dfg.getNode(swNodeId);
            if (!swNode || swNode->kind != Node::OperationNode)
              continue;
            if (!isSoftwareMemoryInterfaceOp(getNodeAttrStr(swNode, "op_name")))
              continue;
            allMemoryNodes.push_back(swNodeId);
          }
          if (allMemoryNodes.size() > repairOpts.memoryFocusNodeLimit)
            return;
          nodes.clear();
          seen.clear();
          for (IdIndex swNodeId : allMemoryNodes) {
            nodes.push_back(swNodeId);
            seen.insert(swNodeId);
          }
        };
    for (IdIndex edgeId : bestFailedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      const Port *dstPort = dfg.getPort(edge->dstPort);
      maybeAddMemoryNode(srcPort && srcPort->parentNode != INVALID_ID
                             ? srcPort->parentNode
                             : INVALID_ID);
      maybeAddMemoryNode(dstPort && dstPort->parentNode != INVALID_ID
                             ? dstPort->parentNode
                             : INVALID_ID);
    }
    expandToAllSoftwareMemoryNodes(memoryFocusNodes, seenMemoryNodes);

    if (!memoryFocusNodes.empty() &&
        memoryFocusNodes.size() <= repairOpts.memoryFocusNodeLimit) {
      state.restore(bestPlacementCheckpoint);
      llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> exactDomains;
      size_t searchSpace = 1;

      for (IdIndex swNode : memoryFocusNodes) {
        IdIndex oldHw = state.swNodeToHwNode[swNode];
        auto candIt = candidates.find(swNode);
        if (oldHw == INVALID_ID || candIt == candidates.end() ||
            candIt->second.empty()) {
          exactDomains.clear();
          break;
        }

        llvm::SmallVector<std::pair<double, IdIndex>, 16> rankedCandidates;
        for (IdIndex candHw : candIt->second) {
          double candScore = mapper.scorePlacement(swNode, candHw, state, dfg,
                                                   adg, flattener, candidates);
          rankedCandidates.push_back({-candScore, candHw});
        }
        llvm::stable_sort(rankedCandidates,
                          [&](const auto &lhs, const auto &rhs) {
                            if (lhs.first != rhs.first)
                              return lhs.first < rhs.first;
                            return lhs.second < rhs.second;
                          });

        auto &domain = exactDomains[swNode];
        domain.push_back(oldHw);
        unsigned limit = std::min<unsigned>(rankedCandidates.size(), 5u);
        for (unsigned idx = 0; idx < limit; ++idx) {
          IdIndex candHw = rankedCandidates[idx].second;
          if (std::find(domain.begin(), domain.end(), candHw) == domain.end())
            domain.push_back(candHw);
        }
        searchSpace *= domain.size();
      }

      if (!exactDomains.empty() &&
          searchSpace <= repairOpts.memoryExactDomainSearchSpaceCap) {
        llvm::stable_sort(memoryFocusNodes, [&](IdIndex lhs, IdIndex rhs) {
          size_t lhsDomain = exactDomains.lookup(lhs).size();
          size_t rhsDomain = exactDomains.lookup(rhs).size();
          if (lhsDomain != rhsDomain)
            return lhsDomain < rhsDomain;
          return lhs < rhs;
        });

        llvm::outs() << "  Exact memory neighborhood: nodes="
                     << memoryFocusNodes.size()
                     << ", searchSpace=" << searchSpace << "\n";

        for (IdIndex swNode : memoryFocusNodes)
          state.unmapNode(swNode, dfg, adg);

        const auto exactDeadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(static_cast<int64_t>(std::max(
                1.0, mapper.clampDeadlineMsToRemainingBudget(6000.0))));
        bool stopExactSearch = false;
        llvm::DenseSet<IdIndex> distinctMemoryCandidates;
        unsigned memoryNodeCount = 0;
        for (IdIndex swNode : memoryFocusNodes) {
          const Node *node = dfg.getNode(swNode);
          if (!node ||
              !isSoftwareMemoryInterfaceOp(getNodeAttrStr(node, "op_name")))
            continue;
          ++memoryNodeCount;
          for (IdIndex candHw : exactDomains.lookup(swNode))
            distinctMemoryCandidates.insert(candHw);
        }
        const bool preferDistinctMemoryAssignments =
            memoryNodeCount >= 2 &&
            distinctMemoryCandidates.size() >= memoryNodeCount;
        llvm::DenseSet<IdIndex> usedDistinctMemoryHws;

        std::function<void(unsigned, bool)> enumerateMemoryNeighborhood;
        enumerateMemoryNeighborhood = [&](unsigned depth,
                                         bool enforceDistinctMemory) {
          if (stopExactSearch || mapper.shouldStopForBudget("local repair") ||
              std::chrono::steady_clock::now() > exactDeadline)
            return;
          if (depth >= memoryFocusNodes.size()) {
            mapper.rebindScalarInputSentinels(state, dfg, adg, flattener);
            mapper.bindMemrefSentinels(state, dfg, adg);
            classifyTemporalRegisterEdges(state, dfg, adg, flattener,
                                          edgeKinds);
            bool allRouted = rerouteRepairState(state);
            if (updateBest(allRouted) && allRouted)
              stopExactSearch = true;
            return;
          }

          IdIndex swNode = memoryFocusNodes[depth];
          const Node *swNodeDef = dfg.getNode(swNode);
          const bool isMemoryNode =
              swNodeDef &&
              isSoftwareMemoryInterfaceOp(getNodeAttrStr(swNodeDef, "op_name"));
          for (IdIndex candHw : exactDomains.lookup(swNode)) {
            auto savepoint = state.beginSavepoint();
            if (enforceDistinctMemory && isMemoryNode &&
                usedDistinctMemoryHws.contains(candHw)) {
              state.rollbackSavepoint(savepoint);
              continue;
            }
            if (enforceDistinctMemory && isMemoryNode)
              usedDistinctMemoryHws.insert(candHw);
            if (state.mapNode(swNode, candHw, dfg, adg) !=
                ActionResult::Success) {
              if (enforceDistinctMemory && isMemoryNode)
                usedDistinctMemoryHws.erase(candHw);
              state.rollbackSavepoint(savepoint);
              continue;
            }
            if (!mapper.bindMappedNodePorts(swNode, state, dfg, adg)) {
              if (enforceDistinctMemory && isMemoryNode)
                usedDistinctMemoryHws.erase(candHw);
              state.rollbackSavepoint(savepoint);
              continue;
            }
            enumerateMemoryNeighborhood(depth + 1, enforceDistinctMemory);
            if (enforceDistinctMemory && isMemoryNode)
              usedDistinctMemoryHws.erase(candHw);
            state.rollbackSavepoint(savepoint);
            if (stopExactSearch)
              return;
          }
        };

        if (preferDistinctMemoryAssignments)
          enumerateMemoryNeighborhood(0, true);
        if (!stopExactSearch)
          enumerateMemoryNeighborhood(0, false);
        llvm::outs() << "  Exact memory neighborhood result: routed "
                     << bestRouted << "/" << dfg.edges.size() << " edges\n";
        if (bestAllRouted) {
          state.restore(bestCheckpoint);
          return true;
        }
      }
    }
  }

  return false;
}

} // namespace loom

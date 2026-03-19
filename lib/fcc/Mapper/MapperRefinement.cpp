#include "MapperInternal.h"
#include "fcc/Mapper/Mapper.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <random>

namespace fcc {

using namespace mapper_detail;

namespace {

int manhattanDistance(IdIndex lhsHwNode, IdIndex rhsHwNode,
                      const ADGFlattener &flattener) {
  auto [lhsRow, lhsCol] = flattener.getNodeGridPos(lhsHwNode);
  auto [rhsRow, rhsCol] = flattener.getNodeGridPos(rhsHwNode);
  if (lhsRow < 0 || lhsCol < 0 || rhsRow < 0 || rhsCol < 0)
    return 0;
  return std::abs(lhsRow - rhsRow) + std::abs(lhsCol - rhsCol);
}

bool isWithinMoveRadius(IdIndex lhsHwNode, IdIndex rhsHwNode,
                        const ADGFlattener &flattener, unsigned radius) {
  if (radius == 0)
    return true;
  return manhattanDistance(lhsHwNode, rhsHwNode, flattener) <=
         static_cast<int>(radius);
}

bool canRelocateNode(
    IdIndex swNode, IdIndex newHwNode, IdIndex oldHwNode,
    const MappingState &state, const Graph &adg, const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates) {
  auto candIt = candidates.find(swNode);
  if (candIt == candidates.end())
    return false;
  if (std::find(candIt->second.begin(), candIt->second.end(), newHwNode) ==
      candIt->second.end())
    return false;
  if (newHwNode != oldHwNode && !state.hwNodeToSwNodes[newHwNode].empty())
    return false;

  const Node *hwNode = adg.getNode(newHwNode);
  if (!hwNode)
    return false;
  llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
  if (!peName.empty() &&
      isSpatialPEOccupied(state, adg, flattener, peName, oldHwNode))
    return false;
  return true;
}

bool canSwapNodes(
    IdIndex swA, IdIndex swB, IdIndex hwA, IdIndex hwB,
    const MappingState &state, const Graph &adg, const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates) {
  auto candItA = candidates.find(swA);
  auto candItB = candidates.find(swB);
  if (candItA == candidates.end() || candItB == candidates.end())
    return false;
  if (std::find(candItA->second.begin(), candItA->second.end(), hwB) ==
      candItA->second.end())
    return false;
  if (std::find(candItB->second.begin(), candItB->second.end(), hwA) ==
      candItB->second.end())
    return false;

  const Node *hwNodeA = adg.getNode(hwA);
  const Node *hwNodeB = adg.getNode(hwB);
  if (!hwNodeA || !hwNodeB)
    return false;

  llvm::StringRef peNameA = getNodeAttrStr(hwNodeA, "pe_name");
  llvm::StringRef peNameB = getNodeAttrStr(hwNodeB, "pe_name");
  if (!peNameA.empty() && isSpatialPEName(flattener, peNameA) &&
      peNameA == peNameB)
    return false;

  if (!peNameA.empty() &&
      isSpatialPEOccupied(state, adg, flattener, peNameA, hwA))
    return false;
  if (!peNameB.empty() &&
      isSpatialPEOccupied(state, adg, flattener, peNameB, hwB))
    return false;

  return true;
}

bool isNonRoutingOutputPort(IdIndex portId, const Graph &adg) {
  const Port *port = adg.getPort(portId);
  if (!port || port->direction != Port::Output ||
      port->parentNode == INVALID_ID)
    return false;
  const Node *owner = adg.getNode(port->parentNode);
  return owner && !isRoutingResourceNode(owner);
}

bool isRoutingCrossbarOutputPortForRepair(IdIndex portId, const Graph &adg) {
  const Port *port = adg.getPort(portId);
  if (!port || port->direction != Port::Output ||
      port->parentNode == INVALID_ID)
    return false;
  const Node *owner = adg.getNode(port->parentNode);
  if (!owner || getNodeAttrStr(owner, "resource_class") != "routing")
    return false;
  return owner->inputPorts.size() > 1 && owner->outputPorts.size() > 1;
}

IdIndex getLockedFirstHopForSource(IdIndex swEdgeId, IdIndex srcHwPort,
                                   const MappingState &state) {
  for (IdIndex otherEdgeId = 0;
       otherEdgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++otherEdgeId) {
    if (otherEdgeId == swEdgeId)
      continue;
    const auto &otherPath = state.swEdgeToHwPaths[otherEdgeId];
    if (otherPath.size() < 2 || otherPath.front() != srcHwPort)
      continue;
    return otherPath[1];
  }
  return INVALID_ID;
}

double estimatePortDistance(IdIndex lhsPortId, IdIndex rhsPortId,
                            const Graph &adg, const ADGFlattener &flattener) {
  const Port *lhsPort = adg.getPort(lhsPortId);
  const Port *rhsPort = adg.getPort(rhsPortId);
  if (!lhsPort || !rhsPort || lhsPort->parentNode == INVALID_ID ||
      rhsPort->parentNode == INVALID_ID)
    return 0.0;
  auto [lhsRow, lhsCol] = flattener.getNodeGridPos(lhsPort->parentNode);
  auto [rhsRow, rhsCol] = flattener.getNodeGridPos(rhsPort->parentNode);
  if (lhsRow < 0 || lhsCol < 0 || rhsRow < 0 || rhsCol < 0)
    return 0.0;
  return static_cast<double>(std::abs(lhsRow - rhsRow) +
                             std::abs(lhsCol - rhsCol));
}

double computeUnroutedPenalty(const MappingState &state, const Graph &dfg,
                              llvm::ArrayRef<TechMappedEdgeKind> edgeKinds) {
  double penalty = 0.0;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (edgeId < state.swEdgeToHwPaths.size() &&
        !state.swEdgeToHwPaths[edgeId].empty())
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;
    bool srcMapped = srcPort->parentNode < state.swNodeToHwNode.size() &&
                     state.swNodeToHwNode[srcPort->parentNode] != INVALID_ID;
    bool dstMapped = dstPort->parentNode < state.swNodeToHwNode.size() &&
                     state.swNodeToHwNode[dstPort->parentNode] != INVALID_ID;
    if (!srcMapped || !dstMapped)
      continue;
    penalty += classifyEdgePlacementWeight(dfg, edgeId);
  }
  return penalty;
}

} // namespace

bool Mapper::runRefinement(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    const Options &opts) {

  std::mt19937 rng(static_cast<unsigned>(opts.seed));

  std::vector<IdIndex> placedNodes;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    if (dfg.getNode(i) && dfg.getNode(i)->kind == Node::OperationNode &&
        state.swNodeToHwNode[i] != INVALID_ID) {
      placedNodes.push_back(i);
    }
  }

  if (placedNodes.size() < 2)
    return true;

  double temperature = 100.0;
  double coolingRate = 0.995;
  int maxIter = static_cast<int>(placedNodes.size()) * 1000;
  if (maxIter > 50000)
    maxIter = 50000;

  double bestCost = computeTotalCost(state, dfg, adg, flattener);
  auto bestCheckpoint = state.save();
  int acceptCount = 0;

  auto startTime = std::chrono::steady_clock::now();

  for (int iter = 0; iter < maxIter; ++iter) {
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    double secs =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() /
        1000.0;
    if (secs > opts.budgetSeconds * 0.4)
      break;

    double oldCost = computeTotalCost(state, dfg, adg, flattener);
    auto checkpoint = state.save();

    bool moveOk = false;
    bool doSwap = std::uniform_int_distribution<int>(0, 1)(rng) == 0;

    if (doSwap) {
      std::uniform_int_distribution<size_t> dist(0, placedNodes.size() - 1);
      IdIndex swA = placedNodes[dist(rng)];
      IdIndex hwA = state.swNodeToHwNode[swA];

      llvm::SmallVector<IdIndex, 16> nearbyNodes;
      for (IdIndex swCandidate : placedNodes) {
        if (swCandidate == swA)
          continue;
        IdIndex hwCandidate = state.swNodeToHwNode[swCandidate];
        if (hwCandidate == INVALID_ID)
          continue;
        if (!isWithinMoveRadius(hwA, hwCandidate, flattener,
                                opts.placementMoveRadius))
          continue;
        nearbyNodes.push_back(swCandidate);
      }
      if (nearbyNodes.empty())
        continue;

      std::uniform_int_distribution<size_t> nearbyDist(0,
                                                       nearbyNodes.size() - 1);
      IdIndex swB = nearbyNodes[nearbyDist(rng)];
      IdIndex hwB = state.swNodeToHwNode[swB];
      if (!canSwapNodes(swA, swB, hwA, hwB, state, adg, flattener, candidates))
        continue;

      state.unmapNode(swA, dfg, adg);
      state.unmapNode(swB, dfg, adg);
      auto r1 = state.mapNode(swA, hwB, dfg, adg);
      auto r2 = state.mapNode(swB, hwA, dfg, adg);
      if (r1 == ActionResult::Success && r2 == ActionResult::Success) {
        moveOk = bindMappedNodePorts(swA, state, dfg, adg) &&
                 bindMappedNodePorts(swB, state, dfg, adg);
      }
    } else {
      std::uniform_int_distribution<size_t> dist(0, placedNodes.size() - 1);
      IdIndex swNode = placedNodes[dist(rng)];
      IdIndex oldHw = state.swNodeToHwNode[swNode];
      auto candIt = candidates.find(swNode);
      if (candIt == candidates.end() || candIt->second.size() < 2)
        continue;

      IdIndex newHw = INVALID_ID;
      double bestCandScore = -1.0e18;
      llvm::SmallVector<IdIndex, 8> topCandidates;
      for (IdIndex candHw : candIt->second) {
        if (candHw == oldHw)
          continue;
        if (!isWithinMoveRadius(oldHw, candHw, flattener,
                                opts.placementMoveRadius))
          continue;
        if (!canRelocateNode(swNode, candHw, oldHw, state, adg, flattener,
                             candidates))
          continue;

        double candScore = scorePlacement(swNode, candHw, state, dfg, adg,
                                          flattener, candidates);
        if (candScore > bestCandScore + 1e-9) {
          bestCandScore = candScore;
          topCandidates.clear();
          topCandidates.push_back(candHw);
        } else if (std::abs(candScore - bestCandScore) <= 1e-9 &&
                   topCandidates.size() < 8) {
          topCandidates.push_back(candHw);
        }
      }

      if (!topCandidates.empty()) {
        std::uniform_int_distribution<size_t> candDist(0, topCandidates.size() -
                                                              1);
        newHw = topCandidates[candDist(rng)];
      }
      if (newHw == INVALID_ID)
        continue;

      state.unmapNode(swNode, dfg, adg);
      auto result = state.mapNode(swNode, newHw, dfg, adg);
      moveOk = (result == ActionResult::Success) &&
               bindMappedNodePorts(swNode, state, dfg, adg);
    }

    if (!moveOk) {
      state.restore(checkpoint);
      continue;
    }

    double newCost = computeTotalCost(state, dfg, adg, flattener);
    double delta = oldCost - newCost;
    bool acceptMove =
        delta > 0 || std::uniform_real_distribution<double>(0.0, 1.0)(rng) <
                         std::exp(delta / temperature);
    if (acceptMove) {
      ++acceptCount;
      if (newCost < bestCost) {
        bestCost = newCost;
        bestCheckpoint = state.save();
      }
    } else {
      state.restore(checkpoint);
    }

    temperature *= coolingRate;
  }

  state.restore(bestCheckpoint);
  if (opts.verbose) {
    llvm::outs() << "  SA: " << acceptCount << " accepted moves, best cost "
                 << bestCost << "\n";
  }
  return true;
}

bool Mapper::runExactRoutingRepair(MappingState &state,
                                   llvm::ArrayRef<IdIndex> failedEdges,
                                   const Graph &dfg, const Graph &adg,
                                   const ADGFlattener &flattener,
                                   llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                   const Options &opts) {
  if (failedEdges.empty())
    return true;

  auto edgeWeight = [&](IdIndex edgeId) {
    return classifyEdgePlacementWeight(dfg, edgeId);
  };
  auto betterResult = [&](unsigned lhsRouted, double lhsPenalty,
                          size_t lhsPathLen, unsigned rhsRouted,
                          double rhsPenalty, size_t rhsPathLen) {
    if (lhsRouted != rhsRouted)
      return lhsRouted > rhsRouted;
    if (std::abs(lhsPenalty - rhsPenalty) > 1e-9)
      return lhsPenalty < rhsPenalty;
    if (lhsPathLen != rhsPathLen)
      return lhsPathLen < rhsPathLen;
    return false;
  };

  auto globalBestCheckpoint = state.save();
  unsigned globalBestRouted = countRoutedEdges(state, dfg, edgeKinds);
  double globalBestPenalty = computeUnroutedPenalty(state, dfg, edgeKinds);
  size_t globalBestPathLen = computeTotalMappedPathLen(state);
  bool globalAllRouted = failedEdges.empty();

  auto edgeOrderLess = [&](IdIndex lhs, IdIndex rhs) {
    double lhsWeight = edgeWeight(lhs);
    double rhsWeight = edgeWeight(rhs);
    if (std::abs(lhsWeight - rhsWeight) > 1e-9)
      return lhsWeight > rhsWeight;
    return lhs < rhs;
  };

  std::vector<IdIndex> pendingFailed(failedEdges.begin(), failedEdges.end());
  llvm::stable_sort(pendingFailed, edgeOrderLess);
  const unsigned maxNeighborhoodPasses =
      std::max(2u, std::min<unsigned>(6u, opts.selectiveRipupPasses + 1u));

  for (unsigned pass = 0; pass < maxNeighborhoodPasses && !globalAllRouted;
       ++pass) {
    auto currentFailed = collectUnroutedEdges(state, dfg, edgeKinds);
    if (currentFailed.empty()) {
      globalAllRouted = true;
      break;
    }
    llvm::stable_sort(currentFailed, edgeOrderLess);

    bool improvedThisPass = false;
    for (IdIndex rootFailedEdge : currentFailed) {
      auto baseCheckpoint = state.save();
      const Edge *rootEdge = dfg.getEdge(rootFailedEdge);
      if (!rootEdge)
        continue;

      const Port *rootSrcSwPort = dfg.getPort(rootEdge->srcPort);
      const Port *rootDstSwPort = dfg.getPort(rootEdge->dstPort);
      if (!rootSrcSwPort || !rootDstSwPort)
        continue;

      IdIndex rootSrcSwNode = rootSrcSwPort->parentNode != INVALID_ID
                                  ? rootSrcSwPort->parentNode
                                  : INVALID_ID;
      IdIndex rootDstSwNode = rootDstSwPort->parentNode != INVALID_ID
                                  ? rootDstSwPort->parentNode
                                  : INVALID_ID;
      IdIndex rootSrcHwPort = rootEdge->srcPort < state.swPortToHwPort.size()
                                  ? state.swPortToHwPort[rootEdge->srcPort]
                                  : INVALID_ID;
      IdIndex rootDstHwPort = rootEdge->dstPort < state.swPortToHwPort.size()
                                  ? state.swPortToHwPort[rootEdge->dstPort]
                                  : INVALID_ID;
      const Port *rootSrcHwPortPtr = adg.getPort(rootSrcHwPort);
      const Port *rootDstHwPortPtr = adg.getPort(rootDstHwPort);
      IdIndex rootSrcHwNode =
          rootSrcHwPortPtr && rootSrcHwPortPtr->parentNode != INVALID_ID
              ? rootSrcHwPortPtr->parentNode
              : INVALID_ID;
      IdIndex rootDstHwNode =
          rootDstHwPortPtr && rootDstHwPortPtr->parentNode != INVALID_ID
              ? rootDstHwPortPtr->parentNode
              : INVALID_ID;
      auto [rootSrcRow, rootSrcCol] =
          rootSrcHwNode != INVALID_ID ? flattener.getNodeGridPos(rootSrcHwNode)
                                      : std::pair<int, int>{-1, -1};
      auto [rootDstRow, rootDstCol] =
          rootDstHwNode != INVALID_ID ? flattener.getNodeGridPos(rootDstHwNode)
                                      : std::pair<int, int>{-1, -1};

      auto touchesEndpointNeighborhood = [&](llvm::ArrayRef<IdIndex> path) {
        for (IdIndex portId : path) {
          const Port *port = adg.getPort(portId);
          if (!port || port->parentNode == INVALID_ID)
            continue;
          auto [row, col] = flattener.getNodeGridPos(port->parentNode);
          if (row < 0 || col < 0)
            continue;
          if (rootSrcRow >= 0 &&
              std::abs(row - rootSrcRow) + std::abs(col - rootSrcCol) <= 1)
            return true;
          if (rootDstRow >= 0 &&
              std::abs(row - rootDstRow) + std::abs(col - rootDstCol) <= 1)
            return true;
        }
        return false;
      };

      llvm::SmallVector<std::pair<int, IdIndex>, 32> rankedEdges;
      for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
           ++edgeId) {
        if (edgeId < edgeKinds.size() &&
            (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
             edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
          continue;
        const Edge *edge = dfg.getEdge(edgeId);
        if (!edge)
          continue;

        const Port *srcSwPort = dfg.getPort(edge->srcPort);
        const Port *dstSwPort = dfg.getPort(edge->dstPort);
        if (!srcSwPort || !dstSwPort)
          continue;
        IdIndex srcSwNode = srcSwPort->parentNode != INVALID_ID
                                ? srcSwPort->parentNode
                                : INVALID_ID;
        IdIndex dstSwNode = dstSwPort->parentNode != INVALID_ID
                                ? dstSwPort->parentNode
                                : INVALID_ID;
        IdIndex srcHwPort = edge->srcPort < state.swPortToHwPort.size()
                                ? state.swPortToHwPort[edge->srcPort]
                                : INVALID_ID;
        IdIndex dstHwPort = edge->dstPort < state.swPortToHwPort.size()
                                ? state.swPortToHwPort[edge->dstPort]
                                : INVALID_ID;
        const Port *srcHwPortPtr = adg.getPort(srcHwPort);
        const Port *dstHwPortPtr = adg.getPort(dstHwPort);
        IdIndex srcHwNode =
            srcHwPortPtr && srcHwPortPtr->parentNode != INVALID_ID
                ? srcHwPortPtr->parentNode
                : INVALID_ID;
        IdIndex dstHwNode =
            dstHwPortPtr && dstHwPortPtr->parentNode != INVALID_ID
                ? dstHwPortPtr->parentNode
                : INVALID_ID;

        int score = 0;
        if (edgeId == rootFailedEdge)
          score += 100;
        if (srcHwPort != INVALID_ID && srcHwPort == rootSrcHwPort)
          score += 60;
        if (dstHwPort != INVALID_ID && dstHwPort == rootDstHwPort)
          score += 40;
        if (srcSwNode != INVALID_ID &&
            (srcSwNode == rootSrcSwNode || srcSwNode == rootDstSwNode))
          score += 20;
        if (dstSwNode != INVALID_ID &&
            (dstSwNode == rootSrcSwNode || dstSwNode == rootDstSwNode))
          score += 20;
        if (srcHwNode != INVALID_ID &&
            (srcHwNode == rootSrcHwNode || srcHwNode == rootDstHwNode))
          score += 16;
        if (dstHwNode != INVALID_ID &&
            (dstHwNode == rootSrcHwNode || dstHwNode == rootDstHwNode))
          score += 16;
        if (edgeId < state.swEdgeToHwPaths.size() &&
            !state.swEdgeToHwPaths[edgeId].empty()) {
          const auto &path = state.swEdgeToHwPaths[edgeId];
          if (rootSrcHwPort != INVALID_ID &&
              std::find(path.begin(), path.end(), rootSrcHwPort) != path.end())
            score += 20;
          if (rootDstHwPort != INVALID_ID &&
              std::find(path.begin(), path.end(), rootDstHwPort) != path.end())
            score += 20;
          if (touchesEndpointNeighborhood(path))
            score += 12;
        }
        if (std::find(currentFailed.begin(), currentFailed.end(), edgeId) !=
            currentFailed.end())
          score += 6;
        if (score > 0)
          rankedEdges.push_back({-score, edgeId});
      }

      llvm::stable_sort(rankedEdges, [&](const auto &lhs, const auto &rhs) {
        if (lhs.first != rhs.first)
          return lhs.first < rhs.first;
        return lhs.second < rhs.second;
      });

      llvm::SmallVector<IdIndex, 16> repairEdges;
      llvm::DenseSet<IdIndex> selectedEdges;
      const unsigned maxRepairEdges =
          rootFailedEdge == currentFailed.front() ? 12u : 10u;
      for (const auto &entry : rankedEdges) {
        if (repairEdges.size() >= maxRepairEdges)
          break;
        if (selectedEdges.insert(entry.second).second)
          repairEdges.push_back(entry.second);
      }
      if (repairEdges.empty()) {
        state.restore(baseCheckpoint);
        continue;
      }

      llvm::stable_sort(repairEdges, [&](IdIndex lhs, IdIndex rhs) {
        double lhsWeight = edgeWeight(lhs);
        double rhsWeight = edgeWeight(rhs);
        bool lhsFailed = std::find(currentFailed.begin(), currentFailed.end(),
                                   lhs) != currentFailed.end();
        bool rhsFailed = std::find(currentFailed.begin(), currentFailed.end(),
                                   rhs) != currentFailed.end();
        if (lhsFailed != rhsFailed)
          return lhsFailed;
        if (std::abs(lhsWeight - rhsWeight) > 1e-9)
          return lhsWeight > rhsWeight;
        return lhs < rhs;
      });

      unsigned baseLocalRouted = 0;
      double totalLocalWeight = 0.0;
      double baseLocalPenalty = 0.0;
      size_t baseLocalPathLen = 0;
      for (IdIndex edgeId : repairEdges) {
        totalLocalWeight += edgeWeight(edgeId);
        if (edgeId < state.swEdgeToHwPaths.size() &&
            !state.swEdgeToHwPaths[edgeId].empty()) {
          ++baseLocalRouted;
          baseLocalPathLen += state.swEdgeToHwPaths[edgeId].size();
        } else {
          baseLocalPenalty += edgeWeight(edgeId);
        }
      }

      if (opts.verbose) {
        llvm::outs() << "  Exact routing repair: root edge " << rootFailedEdge
                     << ", neighborhood=" << repairEdges.size()
                     << ", baseLocalRouted=" << baseLocalRouted << "/"
                     << repairEdges.size() << "\n";
        llvm::outs() << "    edges:";
        for (IdIndex edgeId : repairEdges)
          llvm::outs() << " " << edgeId;
        llvm::outs() << "\n";
      }

      auto bestLocalCheckpoint = baseCheckpoint;
      unsigned bestLocalRouted = baseLocalRouted;
      double bestLocalPenalty = baseLocalPenalty;
      size_t bestLocalPathLen = baseLocalPathLen;

      for (IdIndex edgeId : repairEdges)
        state.unmapEdge(edgeId, adg);

      const auto exactDeadline =
          std::chrono::steady_clock::now() +
          std::chrono::milliseconds(static_cast<int64_t>(
              std::max(1200.0, opts.cpSatTimeLimitSeconds * 2500.0)));

      auto rankFirstHopChoices = [&](IdIndex dstHwPort,
                                     llvm::SmallVectorImpl<IdIndex> &choices) {
        llvm::stable_sort(choices, [&](IdIndex lhs, IdIndex rhs) {
          double lhsScore =
              estimatePortDistance(lhs, dstHwPort, adg, flattener);
          double rhsScore =
              estimatePortDistance(rhs, dstHwPort, adg, flattener);
          if (std::abs(lhsScore - rhsScore) > 1e-9)
            return lhsScore < rhsScore;
          return lhs < rhs;
        });
        if (choices.size() > 4)
          choices.resize(4);
      };

      std::function<void(llvm::SmallVectorImpl<IdIndex> &, unsigned, double,
                         size_t)>
          searchNeighborhood;
      searchNeighborhood = [&](llvm::SmallVectorImpl<IdIndex> &remainingEdges,
                               unsigned currentRouted, double currentPenalty,
                               size_t currentPathLen) {
        if (std::chrono::steady_clock::now() > exactDeadline)
          return;
        if (betterResult(currentRouted, currentPenalty, currentPathLen,
                         bestLocalRouted, bestLocalPenalty, bestLocalPathLen)) {
          bestLocalRouted = currentRouted;
          bestLocalPenalty = currentPenalty;
          bestLocalPathLen = currentPathLen;
          bestLocalCheckpoint = state.save();
        }
        if (remainingEdges.empty())
          return;
        if (currentRouted + remainingEdges.size() < bestLocalRouted)
          return;

        struct EdgeChoiceBundle {
          IdIndex edgeId = INVALID_ID;
          double weight = 0.0;
          llvm::SmallVector<llvm::SmallVector<IdIndex, 8>, 4> paths;
        };

        std::optional<EdgeChoiceBundle> nextBundle;
        size_t nextIndex = 0;
        for (size_t idx = 0; idx < remainingEdges.size(); ++idx) {
          IdIndex edgeId = remainingEdges[idx];
          const Edge *edge = dfg.getEdge(edgeId);
          if (!edge)
            continue;
          IdIndex srcHwPort = edge->srcPort < state.swPortToHwPort.size()
                                  ? state.swPortToHwPort[edge->srcPort]
                                  : INVALID_ID;
          IdIndex dstHwPort = edge->dstPort < state.swPortToHwPort.size()
                                  ? state.swPortToHwPort[edge->dstPort]
                                  : INVALID_ID;

          EdgeChoiceBundle bundle;
          bundle.edgeId = edgeId;
          bundle.weight = edgeWeight(edgeId);
          llvm::SmallVector<IdIndex, 8> firstHopChoices;
          if (isNonRoutingOutputPort(srcHwPort, adg)) {
            IdIndex lockedFirstHop =
                getLockedFirstHopForSource(edgeId, srcHwPort, state);
            if (lockedFirstHop != INVALID_ID) {
              firstHopChoices.push_back(lockedFirstHop);
            } else if (auto it = connectivity.outToIn.find(srcHwPort);
                       it != connectivity.outToIn.end()) {
              firstHopChoices.append(it->second.begin(), it->second.end());
              llvm::sort(firstHopChoices);
              firstHopChoices.erase(
                  std::unique(firstHopChoices.begin(), firstHopChoices.end()),
                  firstHopChoices.end());
              rankFirstHopChoices(dstHwPort, firstHopChoices);
            }
          }
          if (firstHopChoices.empty())
            firstHopChoices.push_back(INVALID_ID);
          llvm::DenseMap<IdIndex, double> localHistory;
          const unsigned maxCandidatePaths =
              std::min<unsigned>(4u, 1u + firstHopChoices.size());
          for (unsigned pathAttempt = 0;
               pathAttempt < maxCandidatePaths &&
               bundle.paths.size() < maxCandidatePaths;
               ++pathAttempt) {
            llvm::SmallVector<IdIndex, 8> bestPath;
            for (IdIndex firstHop : firstHopChoices) {
              auto path = findPath(srcHwPort, dstHwPort, edgeId, state, dfg,
                                   adg, localHistory, firstHop);
              if (path.empty())
                continue;
              bool duplicate = false;
              for (const auto &existing : bundle.paths) {
                if (existing.size() != path.size())
                  continue;
                if (std::equal(existing.begin(), existing.end(),
                               path.begin())) {
                  duplicate = true;
                  break;
                }
              }
              if (duplicate)
                continue;
              if (bestPath.empty() || path.size() < bestPath.size() ||
                  (path.size() == bestPath.size() &&
                   std::lexicographical_compare(path.begin(), path.end(),
                                                bestPath.begin(),
                                                bestPath.end()))) {
                bestPath = std::move(path);
              }
            }
            if (bestPath.empty())
              break;
            bundle.paths.push_back(bestPath);
            for (IdIndex portId : bestPath) {
              if (isRoutingCrossbarOutputPortForRepair(portId, adg))
                localHistory[portId] += 1.5;
            }
          }

          if (!nextBundle.has_value()) {
            nextBundle = std::move(bundle);
            nextIndex = idx;
            continue;
          }

          bool bundleFailed =
              std::find(currentFailed.begin(), currentFailed.end(), edgeId) !=
              currentFailed.end();
          bool nextFailed =
              std::find(currentFailed.begin(), currentFailed.end(),
                        nextBundle->edgeId) != currentFailed.end();
          if (bundle.paths.size() != nextBundle->paths.size()) {
            if (bundle.paths.size() < nextBundle->paths.size()) {
              nextBundle = std::move(bundle);
              nextIndex = idx;
            }
            continue;
          }
          if (bundleFailed != nextFailed) {
            if (bundleFailed) {
              nextBundle = std::move(bundle);
              nextIndex = idx;
            }
            continue;
          }
          if (std::abs(bundle.weight - nextBundle->weight) > 1e-9) {
            if (bundle.weight > nextBundle->weight) {
              nextBundle = std::move(bundle);
              nextIndex = idx;
            }
            continue;
          }
          if (bundle.edgeId < nextBundle->edgeId) {
            nextBundle = std::move(bundle);
            nextIndex = idx;
          }
        }

        if (!nextBundle.has_value())
          return;

        IdIndex edgeId = nextBundle->edgeId;
        double weight = nextBundle->weight;
        IdIndex savedEdgeId = remainingEdges[nextIndex];
        remainingEdges.erase(remainingEdges.begin() + nextIndex);

        for (const auto &path : nextBundle->paths) {
          if (state.mapEdge(edgeId, path, dfg, adg) != ActionResult::Success)
            continue;
          searchNeighborhood(remainingEdges, currentRouted + 1,
                             currentPenalty - weight,
                             currentPathLen + path.size());
          state.unmapEdge(edgeId, adg);
        }

        searchNeighborhood(remainingEdges, currentRouted, currentPenalty,
                           currentPathLen);
        remainingEdges.insert(remainingEdges.begin() + nextIndex, savedEdgeId);
      };

      llvm::SmallVector<IdIndex, 16> remainingEdges(repairEdges.begin(),
                                                    repairEdges.end());
      searchNeighborhood(remainingEdges, 0, totalLocalWeight, 0);

      if (opts.verbose) {
        llvm::outs() << "  Exact routing repair result: root edge "
                     << rootFailedEdge << ", localRouted=" << bestLocalRouted
                     << "/" << repairEdges.size()
                     << ", localPenalty=" << bestLocalPenalty << "\n";
      }

      state.restore(bestLocalCheckpoint);
      unsigned routed = countRoutedEdges(state, dfg, edgeKinds);
      double unroutedPenalty = computeUnroutedPenalty(state, dfg, edgeKinds);
      size_t totalPathLen = computeTotalMappedPathLen(state);
      bool improved =
          betterResult(routed, unroutedPenalty, totalPathLen, globalBestRouted,
                       globalBestPenalty, globalBestPathLen);
      if (improved) {
        if (opts.verbose) {
          llvm::outs() << "  Exact routing repair accepted: routed " << routed
                       << "/" << dfg.edges.size()
                       << ", unroutedPenalty=" << unroutedPenalty << "\n";
        }
        globalBestCheckpoint = state.save();
        globalBestRouted = routed;
        globalBestPenalty = unroutedPenalty;
        globalBestPathLen = totalPathLen;
        globalAllRouted = collectUnroutedEdges(state, dfg, edgeKinds).empty();
        improvedThisPass = true;
        break;
      }

      state.restore(baseCheckpoint);
    }

    if (!improvedThisPass)
      break;
  }

  state.restore(globalBestCheckpoint);
  return globalAllRouted;
}

bool Mapper::runLocalRepair(
    MappingState &state, const MappingState::Checkpoint &baseCheckpoint,
    llvm::ArrayRef<IdIndex> failedEdges, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    std::vector<TechMappedEdgeKind> &edgeKinds, const Options &opts) {
  if (failedEdges.empty())
    return true;

  llvm::DenseMap<IdIndex, double> hotspotWeights;
  llvm::DenseMap<IdIndex, double> failedEdgeWeights;
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 8>> nodeToFailedEdges;
  for (IdIndex edgeId : failedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;
    double weight = classifyEdgePlacementWeight(dfg, edgeId);
    failedEdgeWeights[edgeId] = weight;
    hotspotWeights[srcPort->parentNode] += weight;
    hotspotWeights[dstPort->parentNode] += weight;
    nodeToFailedEdges[srcPort->parentNode].push_back(edgeId);
    nodeToFailedEdges[dstPort->parentNode].push_back(edgeId);
  }

  std::vector<IdIndex> hotspots;
  hotspots.reserve(hotspotWeights.size());
  for (const auto &it : hotspotWeights) {
    const Node *node = dfg.getNode(it.first);
    if (!node || node->kind != Node::OperationNode)
      continue;
    hotspots.push_back(it.first);
  }
  llvm::stable_sort(hotspots, [&](IdIndex lhs, IdIndex rhs) {
    double lhsWeight = hotspotWeights.lookup(lhs);
    double rhsWeight = hotspotWeights.lookup(rhs);
    if (lhsWeight != rhsWeight)
      return lhsWeight > rhsWeight;
    return lhs < rhs;
  });

  auto evaluateFailedEdgeDelta =
      [&](llvm::ArrayRef<std::pair<IdIndex, IdIndex>> moves) -> double {
    llvm::DenseMap<IdIndex, IdIndex> overrides;
    llvm::DenseSet<IdIndex> coveredEdges;
    for (const auto &move : moves)
      overrides[move.first] = move.second;

    double delta = 0.0;
    for (const auto &move : moves) {
      auto it = nodeToFailedEdges.find(move.first);
      if (it == nodeToFailedEdges.end())
        continue;
      for (IdIndex edgeId : it->second) {
        if (!coveredEdges.insert(edgeId).second)
          continue;
        const Edge *edge = dfg.getEdge(edgeId);
        if (!edge)
          continue;
        const Port *srcPort = dfg.getPort(edge->srcPort);
        const Port *dstPort = dfg.getPort(edge->dstPort);
        if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
            dstPort->parentNode == INVALID_ID)
          continue;

        IdIndex srcSwNode = srcPort->parentNode;
        IdIndex dstSwNode = dstPort->parentNode;
        if (srcSwNode >= state.swNodeToHwNode.size() ||
            dstSwNode >= state.swNodeToHwNode.size())
          continue;
        IdIndex oldSrcHw = state.swNodeToHwNode[srcSwNode];
        IdIndex oldDstHw = state.swNodeToHwNode[dstSwNode];
        if (oldSrcHw == INVALID_ID || oldDstHw == INVALID_ID)
          continue;

        auto srcOverrideIt = overrides.find(srcSwNode);
        IdIndex newSrcHw = srcOverrideIt == overrides.end()
                               ? INVALID_ID
                               : srcOverrideIt->second;
        if (newSrcHw == INVALID_ID)
          newSrcHw = oldSrcHw;
        auto dstOverrideIt = overrides.find(dstSwNode);
        IdIndex newDstHw = dstOverrideIt == overrides.end()
                               ? INVALID_ID
                               : dstOverrideIt->second;
        if (newDstHw == INVALID_ID)
          newDstHw = oldDstHw;

        double weight = failedEdgeWeights.lookup(edgeId);
        delta += weight * static_cast<double>(
                              manhattanDistance(oldSrcHw, oldDstHw, flattener) -
                              manhattanDistance(newSrcHw, newDstHw, flattener));
      }
    }
    return delta;
  };

  auto bestCheckpoint = state.save();
  auto bestPlacementCheckpoint = baseCheckpoint;
  unsigned bestRouted = countRoutedEdges(state, dfg, edgeKinds);
  double bestUnroutedPenalty = computeUnroutedPenalty(state, dfg, edgeKinds);
  size_t bestPathLen = computeTotalMappedPathLen(state);
  double bestPlacementCost = computeTotalCost(state, dfg, adg, flattener);
  bool bestAllRouted = false;
  std::vector<IdIndex> bestFailedEdges(failedEdges.begin(), failedEdges.end());
  auto shouldEscalateToCPSat = [&]() {
    return opts.enableCPSat && !bestAllRouted && bestFailedEdges.size() <= 6;
  };

  auto updateBest = [&](bool allRouted) -> bool {
    unsigned routed = countRoutedEdges(state, dfg, edgeKinds);
    double unroutedPenalty = computeUnroutedPenalty(state, dfg, edgeKinds);
    size_t totalPathLen = computeTotalMappedPathLen(state);
    double placementCost = computeTotalCost(state, dfg, adg, flattener);
    bool improved = allRouted || routed > bestRouted ||
                    (routed == bestRouted &&
                     unroutedPenalty + 1e-9 < bestUnroutedPenalty) ||
                    (routed == bestRouted &&
                     std::abs(unroutedPenalty - bestUnroutedPenalty) <= 1e-9 &&
                     totalPathLen < bestPathLen) ||
                    (routed == bestRouted && totalPathLen == bestPathLen &&
                     placementCost + 1e-9 < bestPlacementCost);
    if (!improved)
      return false;

    bestCheckpoint = state.save();
    bestRouted = routed;
    bestUnroutedPenalty = unroutedPenalty;
    bestPathLen = totalPathLen;
    bestPlacementCost = placementCost;
    bestAllRouted = allRouted;
    bestFailedEdges = collectUnroutedEdges(state, dfg, edgeKinds);

    state.clearRoutes(adg);
    bestPlacementCheckpoint = state.save();
    state.restore(bestCheckpoint);
    return true;
  };

  state.restore(baseCheckpoint);
  std::vector<IdIndex> repairNodes;
  for (IdIndex swNode = 0; swNode < static_cast<IdIndex>(dfg.nodes.size());
       ++swNode) {
    const Node *node = dfg.getNode(swNode);
    if (!node || node->kind != Node::OperationNode)
      continue;
    if (swNode < state.swNodeToHwNode.size() &&
        state.swNodeToHwNode[swNode] != INVALID_ID)
      repairNodes.push_back(swNode);
  }

  unsigned maxHotspots = std::min<unsigned>(hotspots.size(), 12);
  bool deferToCPSat = false;
  for (unsigned round = 0; round < std::max(1u, opts.selectiveRipupPasses);
       ++round) {
    bool improvedThisRound = false;
    const unsigned repairRadius =
        opts.placementMoveRadius == 0
            ? 0
            : opts.placementMoveRadius + round * 2u + 1u;

    for (unsigned hotIdx = 0; hotIdx < maxHotspots; ++hotIdx) {
      IdIndex swNode = hotspots[hotIdx];
      auto candIt = candidates.find(swNode);
      if (candIt == candidates.end())
        continue;

      state.restore(bestPlacementCheckpoint);
      IdIndex oldHw = state.swNodeToHwNode[swNode];
      if (oldHw == INVALID_ID)
        continue;

      llvm::SmallVector<std::pair<double, IdIndex>, 16> relocations;
      for (IdIndex candHw : candIt->second) {
        if (candHw == oldHw)
          continue;
        if (!isWithinMoveRadius(oldHw, candHw, flattener, repairRadius))
          continue;
        if (!canRelocateNode(swNode, candHw, oldHw, state, adg, flattener,
                             candidates))
          continue;
        double candScore = scorePlacement(swNode, candHw, state, dfg, adg,
                                          flattener, candidates);
        double failedEdgeDelta =
            evaluateFailedEdgeDelta({std::make_pair(swNode, candHw)});
        double repairScore = failedEdgeDelta * 8.0 + candScore * 0.25;
        relocations.push_back({-repairScore, candHw});
      }
      llvm::stable_sort(relocations, [&](const auto &lhs, const auto &rhs) {
        if (lhs.first != rhs.first)
          return lhs.first < rhs.first;
        return lhs.second < rhs.second;
      });

      unsigned maxRelocations = std::min<unsigned>(relocations.size(), 10);
      for (unsigned moveIdx = 0; moveIdx < maxRelocations; ++moveIdx) {
        state.restore(bestPlacementCheckpoint);
        state.unmapNode(swNode, dfg, adg);
        if (state.mapNode(swNode, relocations[moveIdx].second, dfg, adg) !=
            ActionResult::Success) {
          state.restore(bestPlacementCheckpoint);
          continue;
        }
        if (!bindMappedNodePorts(swNode, state, dfg, adg)) {
          state.restore(bestPlacementCheckpoint);
          continue;
        }
        bindMemrefSentinels(state, dfg, adg);
        classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
        bool allRouted = runRouting(state, dfg, adg, edgeKinds, opts);
        if (updateBest(allRouted)) {
          improvedThisRound = true;
          if (allRouted) {
            state.restore(bestCheckpoint);
            return true;
          }
          if (shouldEscalateToCPSat()) {
            deferToCPSat = true;
            break;
          }
        }
      }
      if (deferToCPSat)
        break;

      llvm::SmallVector<std::pair<double, IdIndex>, 16> swapPartners;
      for (IdIndex otherSw : repairNodes) {
        if (otherSw == swNode)
          continue;
        IdIndex otherHw = state.swNodeToHwNode[otherSw];
        if (otherHw == INVALID_ID)
          continue;
        if (!isWithinMoveRadius(oldHw, otherHw, flattener, repairRadius))
          continue;
        if (!canSwapNodes(swNode, otherSw, oldHw, otherHw, state, adg,
                          flattener, candidates))
          continue;
        double swapScore =
            evaluateFailedEdgeDelta({std::make_pair(swNode, otherHw),
                                     std::make_pair(otherSw, oldHw)}) *
                8.0 +
            scorePlacement(swNode, otherHw, state, dfg, adg, flattener,
                           candidates) *
                0.25 +
            scorePlacement(otherSw, oldHw, state, dfg, adg, flattener,
                           candidates) *
                0.25;
        swapPartners.push_back({-swapScore, otherSw});
      }
      llvm::stable_sort(swapPartners, [&](const auto &lhs, const auto &rhs) {
        if (lhs.first != rhs.first)
          return lhs.first < rhs.first;
        return lhs.second < rhs.second;
      });

      unsigned maxSwaps = std::min<unsigned>(swapPartners.size(), 8);
      for (unsigned swapIdx = 0; swapIdx < maxSwaps; ++swapIdx) {
        state.restore(bestPlacementCheckpoint);
        IdIndex otherSw = swapPartners[swapIdx].second;
        IdIndex otherHw = state.swNodeToHwNode[otherSw];
        state.unmapNode(swNode, dfg, adg);
        state.unmapNode(otherSw, dfg, adg);
        if (state.mapNode(swNode, otherHw, dfg, adg) != ActionResult::Success ||
            state.mapNode(otherSw, oldHw, dfg, adg) != ActionResult::Success) {
          state.restore(bestPlacementCheckpoint);
          continue;
        }
        if (!bindMappedNodePorts(swNode, state, dfg, adg) ||
            !bindMappedNodePorts(otherSw, state, dfg, adg)) {
          state.restore(bestPlacementCheckpoint);
          continue;
        }
        bindMemrefSentinels(state, dfg, adg);
        classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
        bool allRouted = runRouting(state, dfg, adg, edgeKinds, opts);
        if (updateBest(allRouted)) {
          improvedThisRound = true;
          if (allRouted) {
            state.restore(bestCheckpoint);
            return true;
          }
          if (shouldEscalateToCPSat()) {
            deferToCPSat = true;
            break;
          }
        }
      }
      if (deferToCPSat)
        break;
    }

    if (deferToCPSat)
      break;
    if (!improvedThisRound)
      break;
  }

  if (!bestAllRouted && opts.enableCPSat) {
    state.restore(bestPlacementCheckpoint);
    Options cpSatRepairOpts = opts;
    if (bestFailedEdges.size() <= 4) {
      cpSatRepairOpts.cpSatTimeLimitSeconds =
          std::max(cpSatRepairOpts.cpSatTimeLimitSeconds, 8.0);
      cpSatRepairOpts.cpSatNeighborhoodNodeLimit = std::max<unsigned>(
          6u,
          std::min<unsigned>(cpSatRepairOpts.cpSatNeighborhoodNodeLimit, 8u));
    } else if (bestFailedEdges.size() <= 6) {
      cpSatRepairOpts.cpSatTimeLimitSeconds =
          std::max(cpSatRepairOpts.cpSatTimeLimitSeconds, 8.0);
      cpSatRepairOpts.cpSatNeighborhoodNodeLimit = std::max<unsigned>(
          6u,
          std::min<unsigned>(cpSatRepairOpts.cpSatNeighborhoodNodeLimit, 8u));
    }
    bool allRouted = runCPSatNeighborhoodRepair(
        state, bestPlacementCheckpoint, bestFailedEdges, dfg, adg, flattener,
        candidates, edgeKinds, cpSatRepairOpts);
    if (updateBest(allRouted) && allRouted) {
      state.restore(bestCheckpoint);
      return true;
    }
  }

  if (!bestAllRouted && bestFailedEdges.size() <= 6) {
    llvm::DenseMap<IdIndex, double> focusWeights;
    llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> adjacency;
    auto addAdjacency = [&](IdIndex lhs, IdIndex rhs) {
      if (lhs == INVALID_ID || rhs == INVALID_ID || lhs == rhs)
        return;
      auto &lhsAdj = adjacency[lhs];
      if (std::find(lhsAdj.begin(), lhsAdj.end(), rhs) == lhsAdj.end())
        lhsAdj.push_back(rhs);
      auto &rhsAdj = adjacency[rhs];
      if (std::find(rhsAdj.begin(), rhsAdj.end(), lhs) == rhsAdj.end())
        rhsAdj.push_back(lhs);
    };
    auto addIncidentNeighbors = [&](IdIndex swNodeId, double weight) {
      const Node *swNode = dfg.getNode(swNodeId);
      if (!swNode || swNode->kind != Node::OperationNode)
        return;
      auto visitPort = [&](IdIndex portId) {
        const Port *port = dfg.getPort(portId);
        if (!port)
          return;
        for (IdIndex edgeId : port->connectedEdges) {
          const Edge *incidentEdge = dfg.getEdge(edgeId);
          if (!incidentEdge)
            continue;
          IdIndex otherPortId = INVALID_ID;
          if (incidentEdge->srcPort == portId)
            otherPortId = incidentEdge->dstPort;
          else if (incidentEdge->dstPort == portId)
            otherPortId = incidentEdge->srcPort;
          if (otherPortId == INVALID_ID)
            continue;
          const Port *otherPort = dfg.getPort(otherPortId);
          if (!otherPort || otherPort->parentNode == INVALID_ID)
            continue;
          const Node *otherNode = dfg.getNode(otherPort->parentNode);
          if (!otherNode || otherNode->kind != Node::OperationNode)
            continue;
          focusWeights[otherPort->parentNode] += weight * 0.35;
          addAdjacency(swNodeId, otherPort->parentNode);
        }
      };
      for (IdIndex inPortId : swNode->inputPorts)
        visitPort(inPortId);
      for (IdIndex outPortId : swNode->outputPorts)
        visitPort(outPortId);
    };
    for (IdIndex edgeId : bestFailedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      const Port *dstPort = dfg.getPort(edge->dstPort);
      if (!srcPort || !dstPort)
        continue;
      double weight = classifyEdgePlacementWeight(dfg, edgeId);
      IdIndex srcNodeId =
          srcPort->parentNode != INVALID_ID ? srcPort->parentNode : INVALID_ID;
      IdIndex dstNodeId =
          dstPort->parentNode != INVALID_ID ? dstPort->parentNode : INVALID_ID;
      const Node *srcNode =
          srcNodeId != INVALID_ID ? dfg.getNode(srcNodeId) : nullptr;
      const Node *dstNode =
          dstNodeId != INVALID_ID ? dfg.getNode(dstNodeId) : nullptr;
      bool srcIsOp = srcNode && srcNode->kind == Node::OperationNode;
      bool dstIsOp = dstNode && dstNode->kind == Node::OperationNode;
      if (srcIsOp)
        focusWeights[srcNodeId] += weight;
      if (dstIsOp)
        focusWeights[dstNodeId] += weight;
      if (srcIsOp)
        addIncidentNeighbors(srcNodeId, weight);
      if (dstIsOp)
        addIncidentNeighbors(dstNodeId, weight);
      if (srcIsOp && dstIsOp && srcNodeId != dstNodeId)
        addAdjacency(srcNodeId, dstNodeId);
    }

    std::vector<IdIndex> allFocusNodes;
    for (const auto &entry : focusWeights) {
      IdIndex swNode = entry.first;
      if (swNode >= state.swNodeToHwNode.size() ||
          state.swNodeToHwNode[swNode] == INVALID_ID)
        continue;
      allFocusNodes.push_back(swNode);
    }

    llvm::DenseSet<IdIndex> visitedFocus;
    std::vector<std::vector<IdIndex>> focusComponents;
    for (IdIndex seed : allFocusNodes) {
      if (!visitedFocus.insert(seed).second)
        continue;
      std::vector<IdIndex> component;
      std::vector<IdIndex> stack = {seed};
      while (!stack.empty()) {
        IdIndex swNode = stack.back();
        stack.pop_back();
        component.push_back(swNode);
        auto adjIt = adjacency.find(swNode);
        if (adjIt == adjacency.end())
          continue;
        for (IdIndex other : adjIt->second) {
          if (visitedFocus.insert(other).second)
            stack.push_back(other);
        }
      }
      focusComponents.push_back(std::move(component));
    }

    llvm::stable_sort(focusComponents, [&](const std::vector<IdIndex> &lhs,
                                           const std::vector<IdIndex> &rhs) {
      double lhsWeight = 0.0;
      for (IdIndex swNode : lhs)
        lhsWeight += focusWeights.lookup(swNode);
      double rhsWeight = 0.0;
      for (IdIndex swNode : rhs)
        rhsWeight += focusWeights.lookup(swNode);
      if (lhsWeight != rhsWeight)
        return lhsWeight > rhsWeight;
      if (lhs.size() != rhs.size())
        return lhs.size() > rhs.size();
      return lhs < rhs;
    });

    for (unsigned compIdx = 0;
         compIdx < focusComponents.size() && !bestAllRouted; ++compIdx) {
      state.restore(bestPlacementCheckpoint);
      std::vector<IdIndex> focusNodes = focusComponents[compIdx];
      if (focusNodes.empty())
        continue;
      if (focusNodes.size() > 7) {
        llvm::stable_sort(focusNodes, [&](IdIndex lhs, IdIndex rhs) {
          double lhsWeight = focusWeights.lookup(lhs);
          double rhsWeight = focusWeights.lookup(rhs);
          if (lhsWeight != rhsWeight)
            return lhsWeight > rhsWeight;
          return lhs < rhs;
        });
        focusNodes.resize(7);
      }

      const bool largerExactNeighborhood = focusNodes.size() > 4;
      const unsigned exactRadius = std::max(largerExactNeighborhood ? 4u : 4u,
                                            opts.placementMoveRadius + 1u);
      const unsigned candidateLimit = largerExactNeighborhood ? 3u : 4u;
      const size_t searchSpaceLimit = largerExactNeighborhood ? 4096u : 16384u;

      llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> exactDomains;
      size_t searchSpace = 1;

      for (IdIndex swNode : focusNodes) {
        IdIndex oldHw = state.swNodeToHwNode[swNode];
        auto candIt = candidates.find(swNode);
        if (oldHw == INVALID_ID || candIt == candidates.end())
          continue;

        llvm::SmallVector<std::pair<double, IdIndex>, 16> rankedCandidates;
        for (IdIndex candHw : candIt->second) {
          if (candHw != oldHw &&
              !isWithinMoveRadius(oldHw, candHw, flattener, exactRadius)) {
            continue;
          }
          double candScore = scorePlacement(swNode, candHw, state, dfg, adg,
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
        if (domain.empty())
          continue;
        searchSpace *= domain.size();
      }

      llvm::stable_sort(focusNodes, [&](IdIndex lhs, IdIndex rhs) {
        size_t lhsDomain = exactDomains.lookup(lhs).size();
        size_t rhsDomain = exactDomains.lookup(rhs).size();
        if (lhsDomain != rhsDomain)
          return lhsDomain < rhsDomain;
        double lhsWeight = focusWeights.lookup(lhs);
        double rhsWeight = focusWeights.lookup(rhs);
        if (lhsWeight != rhsWeight)
          return lhsWeight > rhsWeight;
        return lhs < rhs;
      });

      llvm::outs() << "  Exact neighborhood " << (compIdx + 1) << "/"
                   << focusComponents.size() << ": nodes=" << focusNodes.size()
                   << ", searchSpace=" << searchSpace << "\n";

      if (searchSpace > searchSpaceLimit) {
        llvm::outs() << "  Exact neighborhood skipped: searchSpace="
                     << searchSpace << "\n";
        continue;
      }

      for (IdIndex swNode : focusNodes)
        state.unmapNode(swNode, dfg, adg);

      const auto exactDeadline =
          std::chrono::steady_clock::now() +
          std::chrono::seconds(largerExactNeighborhood ? 5 : 8);
      bool stopExactSearch = false;

      std::function<void(unsigned)> enumerateNeighborhood;
      enumerateNeighborhood = [&](unsigned depth) {
        if (stopExactSearch || std::chrono::steady_clock::now() > exactDeadline)
          return;
        if (depth >= focusNodes.size()) {
          bindMemrefSentinels(state, dfg, adg);
          classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
          bool allRouted = runRouting(state, dfg, adg, edgeKinds, opts);
          if (updateBest(allRouted) && allRouted)
            stopExactSearch = true;
          return;
        }

        IdIndex swNode = focusNodes[depth];
        auto checkpoint = state.save();
        for (IdIndex candHw : exactDomains.lookup(swNode)) {
          state.restore(checkpoint);
          if (state.mapNode(swNode, candHw, dfg, adg) != ActionResult::Success)
            continue;
          if (!bindMappedNodePorts(swNode, state, dfg, adg))
            continue;
          enumerateNeighborhood(depth + 1);
          if (stopExactSearch)
            return;
        }
        state.restore(checkpoint);
      };

      enumerateNeighborhood(0);
      llvm::outs() << "  Exact neighborhood result: routed " << bestRouted
                   << "/" << dfg.edges.size() << " edges\n";
      if (bestAllRouted) {
        state.restore(bestCheckpoint);
        return true;
      }
    }
  }

  if (!bestAllRouted && bestFailedEdges.size() <= 8) {
    state.restore(bestCheckpoint);
    bool allRouted = runExactRoutingRepair(state, bestFailedEdges, dfg, adg,
                                           flattener, edgeKinds, opts);
    if (updateBest(allRouted) && allRouted) {
      state.restore(bestCheckpoint);
      return true;
    }
  }

  auto buildCpSatRepairOpts = [&]() {
    Options cpSatRepairOpts = opts;
    if (bestFailedEdges.size() <= 4) {
      cpSatRepairOpts.cpSatTimeLimitSeconds =
          std::max(cpSatRepairOpts.cpSatTimeLimitSeconds, 8.0);
      cpSatRepairOpts.cpSatNeighborhoodNodeLimit = std::max<unsigned>(
          6u,
          std::min<unsigned>(cpSatRepairOpts.cpSatNeighborhoodNodeLimit, 8u));
    } else if (bestFailedEdges.size() <= 6) {
      cpSatRepairOpts.cpSatTimeLimitSeconds =
          std::max(cpSatRepairOpts.cpSatTimeLimitSeconds, 8.0);
      cpSatRepairOpts.cpSatNeighborhoodNodeLimit = std::max<unsigned>(
          6u,
          std::min<unsigned>(cpSatRepairOpts.cpSatNeighborhoodNodeLimit, 8u));
    }
    return cpSatRepairOpts;
  };

  for (unsigned polishRound = 0; polishRound < 2 && !bestAllRouted;
       ++polishRound) {
    bool improved = false;

    if (opts.enableCPSat && bestFailedEdges.size() <= 6) {
      state.restore(bestPlacementCheckpoint);
      Options cpSatRepairOpts = buildCpSatRepairOpts();
      bool allRouted = runCPSatNeighborhoodRepair(
          state, bestPlacementCheckpoint, bestFailedEdges, dfg, adg, flattener,
          candidates, edgeKinds, cpSatRepairOpts);
      if (updateBest(allRouted)) {
        improved = true;
        if (allRouted) {
          state.restore(bestCheckpoint);
          return true;
        }
      }
    }

    if (!bestAllRouted && bestFailedEdges.size() <= 8) {
      state.restore(bestCheckpoint);
      bool allRouted = runExactRoutingRepair(state, bestFailedEdges, dfg, adg,
                                             flattener, edgeKinds, opts);
      if (updateBest(allRouted)) {
        improved = true;
        if (allRouted) {
          state.restore(bestCheckpoint);
          return true;
        }
      }
    }

    if (!improved)
      break;
  }

  state.restore(bestCheckpoint);
  return bestAllRouted;
}

} // namespace fcc

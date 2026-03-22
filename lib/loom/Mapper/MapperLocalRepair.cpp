#include "MapperLocalRepairInternal.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace loom {

using namespace mapper_detail;

LocalRepairDriver::LocalRepairDriver(
    Mapper &mapper, MappingState &state,
    const MappingState::Checkpoint &baseCheckpoint,
    llvm::ArrayRef<IdIndex> failedEdges, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    std::vector<TechMappedEdgeKind> &edgeKinds, const Mapper::Options &opts,
    const CongestionState *congestion, unsigned recursionDepth)
    : mapper(mapper), state(state), baseCheckpoint(baseCheckpoint),
      failedEdges(failedEdges), dfg(dfg), adg(adg), flattener(flattener),
      candidates(candidates), edgeKinds(edgeKinds), opts(opts),
      congestion(congestion), recursionDepth(recursionDepth),
      repairOpts(opts.localRepair), edgeWeights(buildEdgePlacementWeightCache(dfg)),
      candidateSets(buildCandidateSetMap(candidates)),
      maxRepairRecursionDepth(
          failedEdges.size() <= repairOpts.focusedTargetEdgeThreshold
              ? repairOpts.microRecursionDepthLimit
              : repairOpts.defaultRecursionDepthLimit),
      bestCheckpoint(state.save()), bestPlacementCheckpoint(baseCheckpoint),
      bestRouted(countRoutedEdges(state, dfg, edgeKinds)),
      bestUnroutedPenalty(computeUnroutedPenalty(state, dfg, edgeKinds)),
      bestPathLen(computeTotalMappedPathLen(state)),
      bestPlacementCost(mapper.computeTotalCost(state, dfg, adg, flattener)),
      bestAllRouted(false),
      bestFailedEdges(failedEdges.begin(), failedEdges.end()) {
  for (IdIndex edgeId : failedEdges)
    repairPriorityEdges.insert(edgeId);
  std::tie(bestPriorityRouted, bestPriorityPenalty) =
      computePriorityMetrics(state);
  bestRepairabilityScore = computeRepairabilityScore(state);

  if (opts.verbose) {
    llvm::outs() << "  Local repair start: depth=" << recursionDepth
                 << ", failedEdges:";
    for (IdIndex edgeId : failedEdges)
      llvm::outs() << " " << edgeId;
    llvm::outs() << "\n";
  }

  for (IdIndex edgeId : failedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;
    double weight = edgeWeight(edgeId);
    failedEdgeWeights[edgeId] = weight;
    hotspotWeights[srcPort->parentNode] += weight;
    hotspotWeights[dstPort->parentNode] += weight;
    nodeToFailedEdges[srcPort->parentNode].push_back(edgeId);
    nodeToFailedEdges[dstPort->parentNode].push_back(edgeId);
  }

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
}

double LocalRepairDriver::edgeWeight(IdIndex edgeId) const {
  return edgeId < static_cast<IdIndex>(edgeWeights.size())
             ? edgeWeights[edgeId]
             : classifyEdgePlacementWeight(dfg, edgeId);
}

std::pair<unsigned, double> LocalRepairDriver::computePriorityMetrics(
    const MappingState &candidateState) const {
  unsigned routed = 0;
  double penalty = 0.0;
  for (IdIndex edgeId : repairPriorityEdges) {
    double weight = edgeWeight(edgeId);
    if (edgeId < candidateState.swEdgeToHwPaths.size() &&
        !candidateState.swEdgeToHwPaths[edgeId].empty()) {
      ++routed;
    } else {
      penalty += weight;
    }
  }
  return {routed, penalty};
}

double LocalRepairDriver::computeRepairabilityScore(
    const MappingState &candidateState) const {
  std::vector<IdIndex> currentFailed =
      collectUnroutedEdges(candidateState, dfg, edgeKinds);
  if (currentFailed.empty())
    return 0.0;

  MappingState probeState = candidateState;
  probeState.clearRoutes(dfg, adg);
  llvm::DenseMap<IdIndex, double> emptyHistory;
  double score = 0.0;
  for (IdIndex edgeId : currentFailed) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    IdIndex srcHwPort =
        edge->srcPort < probeState.swPortToHwPort.size()
            ? probeState.swPortToHwPort[edge->srcPort]
            : INVALID_ID;
    IdIndex dstHwPort =
        edge->dstPort < probeState.swPortToHwPort.size()
            ? probeState.swPortToHwPort[edge->dstPort]
            : INVALID_ID;
    double weight = edgeWeight(edgeId);
    auto freePath = mapper.findPath(srcHwPort, dstHwPort, edgeId, probeState,
                                    dfg, adg, emptyHistory, INVALID_ID,
                                    congestion);
    score += weight *
             static_cast<double>(freePath.empty()
                                     ? repairOpts.freePathMissingPenalty
                                     : freePath.size());
  }
  return score;
}

bool LocalRepairDriver::shouldEscalateToCPSat() const {
  return opts.enableCPSat && !bestAllRouted &&
         bestFailedEdges.size() <=
             repairOpts.cpSatEscalationFailedEdgeThreshold;
}

bool LocalRepairDriver::rerouteRepairState(MappingState &repairState) const {
  if (mapper.shouldStopForBudget("local repair"))
    return false;
  Mapper::Options repairRoutingOpts = opts;
  if (repairRoutingOpts.negotiatedRoutingPasses > 0)
    repairRoutingOpts.negotiatedRoutingPasses =
        std::min<unsigned>(repairRoutingOpts.negotiatedRoutingPasses,
                           repairOpts.repairNegotiatedRoutingPassCap);
  if (repairRoutingOpts.negotiatedRoutingPasses > 0)
    return mapper.runNegotiatedRouting(repairState, dfg, adg, edgeKinds,
                                       repairRoutingOpts);
  return mapper.runRouting(repairState, dfg, adg, edgeKinds, repairRoutingOpts);
}

bool LocalRepairDriver::updateBest(bool allRouted) {
  unsigned routed = countRoutedEdges(state, dfg, edgeKinds);
  double unroutedPenalty = computeUnroutedPenalty(state, dfg, edgeKinds);
  size_t totalPathLen = computeTotalMappedPathLen(state);
  double placementCost = mapper.computeTotalCost(state, dfg, adg, flattener);
  auto [priorityRouted, priorityPenalty] = computePriorityMetrics(state);
  double repairabilityScore = std::numeric_limits<double>::infinity();
  bool improved = allRouted || routed > bestRouted ||
                  (routed == bestRouted &&
                   priorityRouted > bestPriorityRouted) ||
                  (routed == bestRouted &&
                   priorityRouted == bestPriorityRouted &&
                   priorityPenalty + 1e-9 < bestPriorityPenalty) ||
                  (routed == bestRouted &&
                   priorityRouted == bestPriorityRouted &&
                   std::abs(priorityPenalty - bestPriorityPenalty) <= 1e-9 &&
                   unroutedPenalty + 1e-9 < bestUnroutedPenalty) ||
                  (routed == bestRouted &&
                   priorityRouted == bestPriorityRouted &&
                   std::abs(priorityPenalty - bestPriorityPenalty) <= 1e-9 &&
                   std::abs(unroutedPenalty - bestUnroutedPenalty) <= 1e-9 &&
                   totalPathLen < bestPathLen) ||
                  (routed == bestRouted && totalPathLen == bestPathLen &&
                   priorityRouted == bestPriorityRouted &&
                   std::abs(priorityPenalty - bestPriorityPenalty) <= 1e-9 &&
                   placementCost + 1e-9 < bestPlacementCost);
  if (!improved && routed == bestRouted &&
      priorityRouted == bestPriorityRouted &&
      std::abs(priorityPenalty - bestPriorityPenalty) <= 1e-9 &&
      std::abs(unroutedPenalty - bestUnroutedPenalty) <= 1e-9 &&
      totalPathLen == bestPathLen &&
      std::abs(placementCost - bestPlacementCost) <= 1e-9) {
    repairabilityScore = computeRepairabilityScore(state);
    improved = repairabilityScore + 1e-9 < bestRepairabilityScore;
  }
  if (!improved)
    return false;

  bestCheckpoint = state.save();
  bestRouted = routed;
  bestUnroutedPenalty = unroutedPenalty;
  bestPathLen = totalPathLen;
  bestPlacementCost = placementCost;
  bestPriorityRouted = priorityRouted;
  bestPriorityPenalty = priorityPenalty;
  bestAllRouted = allRouted;
  bestFailedEdges = collectUnroutedEdges(state, dfg, edgeKinds);
  if (!std::isfinite(repairabilityScore))
    repairabilityScore = computeRepairabilityScore(state);
  bestRepairabilityScore = repairabilityScore;

  state.clearRoutes(dfg, adg);
  bestPlacementCheckpoint = state.save();
  state.restore(bestCheckpoint);
  mapper.maybeEmitProgressSnapshot(state, edgeKinds, "local-repair-update",
                                   opts);
  return true;
}

double LocalRepairDriver::evaluateFailedEdgeDelta(
    llvm::ArrayRef<std::pair<IdIndex, IdIndex>> moves) const {
  llvm::DenseMap<IdIndex, IdIndex> overrides;
  llvm::DenseSet<IdIndex> coveredEdges;
  for (const auto &move : moves)
    overrides[move.first] = move.second;

  double delta = 0.0;
  for (const auto &move : moves) {
    IdIndex swNode = move.first;
    auto failedIt = nodeToFailedEdges.find(swNode);
    if (failedIt == nodeToFailedEdges.end())
      continue;
    for (IdIndex edgeId : failedIt->second) {
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
      IdIndex newSrcHw =
          srcOverrideIt == overrides.end() ? oldSrcHw : srcOverrideIt->second;
      auto dstOverrideIt = overrides.find(dstSwNode);
      IdIndex newDstHw =
          dstOverrideIt == overrides.end() ? oldDstHw : dstOverrideIt->second;

      double weight = failedEdgeWeights.lookup(edgeId);
      delta += weight *
               static_cast<double>(
                   placementDistance(oldSrcHw, oldDstHw, flattener) -
                   placementDistance(newSrcHw, newDstHw, flattener));
    }
  }
  return delta;
}

llvm::SmallVector<IdIndex, 24> LocalRepairDriver::buildFocusedRepairNeighborhood(
    llvm::ArrayRef<IdIndex> seedEdges, unsigned maxEdges) const {
  llvm::SmallVector<IdIndex, 24> repairEdges;
  llvm::DenseSet<IdIndex> seedSet;
  llvm::DenseSet<IdIndex> directBlockers;
  llvm::DenseSet<IdIndex> selectedEdges;
  llvm::DenseSet<IdIndex> anchorSrcHwPorts;
  llvm::DenseSet<IdIndex> anchorDstHwPorts;
  llvm::DenseSet<IdIndex> anchorSrcSwNodes;
  llvm::DenseSet<IdIndex> anchorDstSwNodes;
  llvm::DenseSet<IdIndex> anchorSrcHwNodes;
  llvm::DenseSet<IdIndex> anchorDstHwNodes;
  llvm::DenseSet<IdIndex> anchorNeighborhoodHwNodes;
  const TopologyModel *topologyModel = getActiveTopologyModel();

  auto recordAnchorNode = [&](IdIndex hwNodeId) {
    if (hwNodeId == INVALID_ID)
      return;
    anchorNeighborhoodHwNodes.insert(hwNodeId);
  };

  for (IdIndex edgeId : seedEdges) {
    if (selectedEdges.insert(edgeId).second)
      repairEdges.push_back(edgeId);
    seedSet.insert(edgeId);
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcSwPort = dfg.getPort(edge->srcPort);
    const Port *dstSwPort = dfg.getPort(edge->dstPort);
    if (srcSwPort && srcSwPort->parentNode != INVALID_ID)
      anchorSrcSwNodes.insert(srcSwPort->parentNode);
    if (dstSwPort && dstSwPort->parentNode != INVALID_ID)
      anchorDstSwNodes.insert(dstSwPort->parentNode);
    IdIndex srcHwPort = edge->srcPort < state.swPortToHwPort.size()
                            ? state.swPortToHwPort[edge->srcPort]
                            : INVALID_ID;
    IdIndex dstHwPort = edge->dstPort < state.swPortToHwPort.size()
                            ? state.swPortToHwPort[edge->dstPort]
                            : INVALID_ID;
    if (srcHwPort != INVALID_ID)
      anchorSrcHwPorts.insert(srcHwPort);
    if (dstHwPort != INVALID_ID)
      anchorDstHwPorts.insert(dstHwPort);
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
    if (srcHwNode != INVALID_ID)
      anchorSrcHwNodes.insert(srcHwNode);
    if (dstHwNode != INVALID_ID)
      anchorDstHwNodes.insert(dstHwNode);
    recordAnchorNode(srcHwNode);
    recordAnchorNode(dstHwNode);
  }

  MappingState probeState = state;
  probeState.clearRoutes(dfg, adg);
  llvm::DenseMap<IdIndex, double> emptyHistory;
  for (IdIndex edgeId : seedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    IdIndex srcHwPort = edge->srcPort < state.swPortToHwPort.size()
                            ? state.swPortToHwPort[edge->srcPort]
                            : INVALID_ID;
    IdIndex dstHwPort = edge->dstPort < state.swPortToHwPort.size()
                            ? state.swPortToHwPort[edge->dstPort]
                            : INVALID_ID;
    if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
      continue;
    auto freeSpacePath = mapper.findPath(srcHwPort, dstHwPort, edgeId, probeState,
                                         dfg, adg, emptyHistory, INVALID_ID,
                                         congestion);
    for (IdIndex portId : freeSpacePath) {
      if (portId >= state.portToUsingEdges.size())
        continue;
      for (IdIndex otherEdgeId : state.portToUsingEdges[portId]) {
        if (selectedEdges.insert(otherEdgeId).second &&
            repairEdges.size() < maxEdges) {
          repairEdges.push_back(otherEdgeId);
        }
        directBlockers.insert(otherEdgeId);
      }
    }
  }

  auto touchesAnchorNeighborhood = [&](llvm::ArrayRef<IdIndex> path) {
    for (IdIndex portId : path) {
      const Port *port = adg.getPort(portId);
      if (!port || port->parentNode == INVALID_ID)
        continue;
      if (anchorNeighborhoodHwNodes.contains(port->parentNode))
        return true;
      for (IdIndex anchorHwNode : anchorNeighborhoodHwNodes) {
        bool nearAnchor =
            topologyModel
                ? topologyModel->isWithinMoveRadius(port->parentNode,
                                                    anchorHwNode, 1)
                : isWithinMoveRadius(port->parentNode, anchorHwNode, flattener,
                                     1);
        if (nearAnchor)
          return true;
      }
    }
    return false;
  };

  llvm::SmallVector<std::pair<int, IdIndex>, 48> rankedEdges;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg)) {
      continue;
    }
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
    if (seedSet.contains(edgeId))
      score += 200;
    if (directBlockers.contains(edgeId))
      score += 120;
    if (srcHwPort != INVALID_ID && anchorSrcHwPorts.contains(srcHwPort))
      score += 60;
    if (dstHwPort != INVALID_ID && anchorDstHwPorts.contains(dstHwPort))
      score += 50;
    if (srcSwNode != INVALID_ID &&
        (anchorSrcSwNodes.contains(srcSwNode) ||
         anchorDstSwNodes.contains(srcSwNode))) {
      score += 24;
    }
    if (dstSwNode != INVALID_ID &&
        (anchorSrcSwNodes.contains(dstSwNode) ||
         anchorDstSwNodes.contains(dstSwNode))) {
      score += 24;
    }
    if (srcHwNode != INVALID_ID &&
        (anchorSrcHwNodes.contains(srcHwNode) ||
         anchorDstHwNodes.contains(srcHwNode))) {
      score += 18;
    }
    if (dstHwNode != INVALID_ID &&
        (anchorSrcHwNodes.contains(dstHwNode) ||
         anchorDstHwNodes.contains(dstHwNode))) {
      score += 18;
    }
    if (edgeId < state.swEdgeToHwPaths.size() &&
        !state.swEdgeToHwPaths[edgeId].empty() &&
        touchesAnchorNeighborhood(state.swEdgeToHwPaths[edgeId])) {
      score += 12;
    }
    if (score > 0)
      rankedEdges.push_back({-score, edgeId});
  }

  llvm::stable_sort(rankedEdges, [&](const auto &lhs, const auto &rhs) {
    if (lhs.first != rhs.first)
      return lhs.first < rhs.first;
    return lhs.second < rhs.second;
  });
  for (const auto &entry : rankedEdges) {
    if (repairEdges.size() >= maxEdges)
      break;
    if (selectedEdges.insert(entry.second).second)
      repairEdges.push_back(entry.second);
  }

  llvm::stable_sort(repairEdges, [&](IdIndex lhs, IdIndex rhs) {
    bool lhsSeed = seedSet.contains(lhs);
    bool rhsSeed = seedSet.contains(rhs);
    if (lhsSeed != rhsSeed)
      return lhsSeed;
    double lhsWeight = edgeWeight(lhs);
    double rhsWeight = edgeWeight(rhs);
    if (std::abs(lhsWeight - rhsWeight) > 1e-9)
      return lhsWeight > rhsWeight;
    return lhs < rhs;
  });
  return repairEdges;
}

llvm::SmallVector<IdIndex, 24> LocalRepairDriver::buildConflictNeighborhood(
    llvm::ArrayRef<IdIndex> seedEdges, unsigned maxEdges) const {
  llvm::SmallVector<IdIndex, 24> repairEdges;
  llvm::DenseSet<IdIndex> seenEdges;
  MappingState probeState = state;
  probeState.clearRoutes(dfg, adg);
  llvm::DenseMap<IdIndex, double> emptyHistory;

  for (IdIndex edgeId : seedEdges) {
    if (!seenEdges.insert(edgeId).second)
      continue;
    repairEdges.push_back(edgeId);
  }

  for (IdIndex edgeId : seedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    IdIndex srcHwPort = edge->srcPort < state.swPortToHwPort.size()
                            ? state.swPortToHwPort[edge->srcPort]
                            : INVALID_ID;
    IdIndex dstHwPort = edge->dstPort < state.swPortToHwPort.size()
                            ? state.swPortToHwPort[edge->dstPort]
                            : INVALID_ID;
    if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
      continue;

    auto freeSpacePath = mapper.findPath(srcHwPort, dstHwPort, edgeId, probeState,
                                         dfg, adg, emptyHistory, INVALID_ID,
                                         congestion);
    if (freeSpacePath.empty())
      continue;

    for (IdIndex portId : freeSpacePath) {
      if (portId >= state.portToUsingEdges.size())
        continue;
      for (IdIndex otherEdgeId : state.portToUsingEdges[portId]) {
        if (seenEdges.insert(otherEdgeId).second) {
          repairEdges.push_back(otherEdgeId);
          if (repairEdges.size() >= maxEdges)
            return repairEdges;
        }
      }
    }
  }

  return repairEdges;
}

void LocalRepairDriver::expandConflictNeighborhood(
    llvm::SmallVectorImpl<IdIndex> &repairEdges, unsigned maxEdges) const {
  llvm::DenseSet<IdIndex> seenEdges;
  for (IdIndex edgeId : repairEdges)
    seenEdges.insert(edgeId);

  for (size_t idx = 0; idx < repairEdges.size() && repairEdges.size() < maxEdges;
       ++idx) {
    IdIndex edgeId = repairEdges[idx];
    if (edgeId >= state.swEdgeToHwPaths.size())
      continue;
    for (IdIndex portId : state.swEdgeToHwPaths[edgeId]) {
      if (portId >= state.portToUsingEdges.size())
        continue;
      for (IdIndex otherEdgeId : state.portToUsingEdges[portId]) {
        if (seenEdges.insert(otherEdgeId).second) {
          repairEdges.push_back(otherEdgeId);
          if (repairEdges.size() >= maxEdges)
            return;
        }
      }
    }
  }
}

bool LocalRepairDriver::run() {
  if (failedEdges.empty())
    return true;
  if (mapper.shouldStopForBudget("local repair"))
    return false;

  if (runHotspotRepairAndEarlyCPSat())
    return true;
  if (runMemoryExactRepairs())
    return true;
  if (runLateRepairStages())
    return true;

  state.restore(bestCheckpoint);
  return bestAllRouted;
}

bool Mapper::runLocalRepair(
    MappingState &state, const MappingState::Checkpoint &baseCheckpoint,
    llvm::ArrayRef<IdIndex> failedEdges, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    std::vector<TechMappedEdgeKind> &edgeKinds, const Mapper::Options &opts,
    const CongestionState *congestion, unsigned recursionDepth) {
  ++activeSearchSummary_.localRepairAttempts;
  LocalRepairDriver driver(*this, state, baseCheckpoint, failedEdges, dfg, adg,
                           flattener, candidates, edgeKinds, opts, congestion,
                           recursionDepth);
  bool repaired = driver.run();
  if (repaired)
    ++activeSearchSummary_.localRepairSuccesses;
  return repaired;
}

} // namespace loom

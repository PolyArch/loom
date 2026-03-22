#include "fcc/Mapper/Mapper.h"

#include "MapperCongestionEstimator.h"
#include "MapperInternal.h"
#include "MapperRoutingCongestion.h"
#include "fcc/Mapper/MapperRelaxedRouting.h"
#include "fcc/Mapper/MapperTiming.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace fcc {

using namespace mapper_detail;

namespace {

bool isBypassableFifoNode(const Node *hwNode) {
  if (!hwNode || getNodeAttrStr(hwNode, "op_kind") != "fifo")
    return false;
  for (const auto &attr : hwNode->attributes) {
    if (attr.getName() == "bypassable") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        return boolAttr.getValue();
    }
  }
  return false;
}

bool getDefaultFifoBypassed(const Node *hwNode) {
  if (!hwNode)
    return false;
  for (const auto &attr : hwNode->attributes) {
    if (attr.getName() == "bypassed") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        return boolAttr.getValue();
    }
  }
  return false;
}

bool isEffectiveFifoBypassed(IdIndex hwNodeId, const MappingState &state,
                             const Graph &adg) {
  const Node *hwNode = adg.getNode(hwNodeId);
  if (!isBypassableFifoNode(hwNode))
    return false;
  if (hwNodeId < state.hwNodeFifoBypassedOverride.size()) {
    int8_t overrideValue = state.hwNodeFifoBypassedOverride[hwNodeId];
    if (overrideValue == 0)
      return false;
    if (overrideValue > 0)
      return true;
  }
  return getDefaultFifoBypassed(hwNode);
}

bool isCarryNextEdge(IdIndex edgeId, const Graph &dfg) {
  const Edge *edge = dfg.getEdge(edgeId);
  if (!edge)
    return false;
  const Port *dstPort = dfg.getPort(edge->dstPort);
  if (!dstPort || dstPort->parentNode == INVALID_ID)
    return false;
  const Node *dstNode = dfg.getNode(dstPort->parentNode);
  if (!dstNode || dstNode->inputPorts.size() < 3)
    return false;
  llvm::StringRef opName = getNodeAttrStr(dstNode, "op_name");
  bool isCarryNode = opName == "dataflow.carry";
  if (!isCarryNode && opName == "techmap_group") {
    llvm::StringRef familySignature =
        getNodeAttrStr(dstNode, "techmap_family_signature");
    isCarryNode = familySignature.contains("dataflow.carry:");
  }
  return isCarryNode && edge->dstPort == dstNode->inputPorts[2];
}

bool isBetterTimingSummary(const MapperTimingSummary &lhs,
                           const MapperTimingSummary &rhs,
                           const MapperBufferizationOptions &opts) {
  if (lhs.estimatedThroughputCost + opts.minThroughputImprovement <
      rhs.estimatedThroughputCost) {
    return true;
  }
  if (rhs.estimatedThroughputCost + opts.minThroughputImprovement <
      lhs.estimatedThroughputCost) {
    return false;
  }
  if (lhs.estimatedClockPeriod + opts.clockTieBreakImprovement <
      rhs.estimatedClockPeriod) {
    return true;
  }
  if (rhs.estimatedClockPeriod + opts.clockTieBreakImprovement <
      lhs.estimatedClockPeriod) {
    return false;
  }
  if (lhs.estimatedInitiationInterval != rhs.estimatedInitiationInterval)
    return lhs.estimatedInitiationInterval < rhs.estimatedInitiationInterval;
  if (lhs.forcedBufferedFifoCount != rhs.forcedBufferedFifoCount)
    return lhs.forcedBufferedFifoCount < rhs.forcedBufferedFifoCount;
  return false;
}

std::vector<IdIndex> collectCriticalBoundaryEdges(const Graph &dfg) {
  std::vector<IdIndex> edges;
  edges.reserve(dfg.edges.size());
  auto getNodeForPort = [&](IdIndex portId) -> const Node * {
    const Port *port = dfg.getPort(portId);
    if (!port || port->parentNode == INVALID_ID)
      return nullptr;
    return dfg.getNode(port->parentNode);
  };
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Node *srcNode = getNodeForPort(edge->srcPort);
    const Node *dstNode = getNodeForPort(edge->dstPort);
    llvm::StringRef srcOp = srcNode ? getNodeAttrStr(srcNode, "op_name") : "";
    llvm::StringRef dstOp = dstNode ? getNodeAttrStr(dstNode, "op_name") : "";
    bool boundaryCritical =
        (srcNode && (srcNode->kind == Node::ModuleInputNode ||
                     srcNode->kind == Node::ModuleOutputNode)) ||
        (dstNode && (dstNode->kind == Node::ModuleInputNode ||
                     dstNode->kind == Node::ModuleOutputNode)) ||
        srcOp == "handshake.extmemory" || srcOp == "handshake.memory" ||
        dstOp == "handshake.extmemory" || dstOp == "handshake.memory";
    if (boundaryCritical)
      edges.push_back(edgeId);
  }
  return edges;
}

} // namespace

bool Mapper::runInterleavedPlaceRoute(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    std::vector<TechMappedEdgeKind> &edgeKinds, const Options &opts) {
  const unsigned rounds = std::max(1u, opts.interleavedRounds);
  auto currentPlacementCheckpoint = state.save();
  auto bestCheckpoint = state.save();
  unsigned bestRouted = 0;
  auto computeUnroutedPenaltyFn = [&](const MappingState &routingState) {
    double penalty = 0.0;
    for (IdIndex edgeId : collectUnroutedEdges(routingState, dfg, edgeKinds))
      penalty += classifyEdgePlacementWeight(dfg, edgeId);
    return penalty;
  };
  auto computePriorityMetrics =
      [&](const MappingState &routingState,
          llvm::ArrayRef<IdIndex> priorityEdges) -> std::pair<unsigned, double> {
    unsigned routed = 0;
    double penalty = 0.0;
    for (IdIndex edgeId : priorityEdges) {
      double weight = classifyEdgePlacementWeight(dfg, edgeId);
      if (edgeId < routingState.swEdgeToHwPaths.size() &&
          !routingState.swEdgeToHwPaths[edgeId].empty()) {
        ++routed;
      } else {
        penalty += weight;
      }
    }
    return {routed, penalty};
  };
  double bestUnroutedPenalty = computeUnroutedPenaltyFn(state);
  size_t bestPathLen = std::numeric_limits<size_t>::max();
  double bestPlacementCost = computeTotalCost(state, dfg, adg, flattener);
  MapperTimingSummary bestTimingSummary =
      analyzeMapperTiming(state, dfg, adg, edgeKinds, opts.timing);
  double bestThroughputCost = bestTimingSummary.estimatedThroughputCost;
  double bestClockPeriod = bestTimingSummary.estimatedClockPeriod;
  bool bestAllRouted = false;
  unsigned bestPriorityRouted = 0;
  double bestPriorityPenalty = 0.0;

  auto updateBest = [&](bool allRouted,
                        llvm::ArrayRef<IdIndex> priorityEdges =
                            llvm::ArrayRef<IdIndex>()) -> bool {
    const bool usePriority = !priorityEdges.empty();
    unsigned routed = countRoutedEdges(state, dfg, edgeKinds);
    double unroutedPenalty = computeUnroutedPenaltyFn(state);
    size_t totalPathLen = computeTotalMappedPathLen(state);
    double placementCost = computeTotalCost(state, dfg, adg, flattener);
    MapperTimingSummary timingSummary =
        analyzeMapperTiming(state, dfg, adg, edgeKinds, opts.timing);
    double throughputCost = timingSummary.estimatedThroughputCost;
    double clockPeriod = timingSummary.estimatedClockPeriod;
    auto [priorityRouted, priorityPenalty] =
        computePriorityMetrics(state, priorityEdges);
    bool improved = allRouted || routed > bestRouted ||
                    (usePriority && routed == bestRouted &&
                     priorityRouted > bestPriorityRouted) ||
                    (usePriority && routed == bestRouted &&
                     priorityRouted == bestPriorityRouted &&
                     priorityPenalty + 1e-9 < bestPriorityPenalty) ||
                    (routed == bestRouted &&
                     (!usePriority ||
                      (priorityRouted == bestPriorityRouted &&
                       std::abs(priorityPenalty - bestPriorityPenalty) <=
                           1e-9)) &&
                     unroutedPenalty + 1e-9 < bestUnroutedPenalty) ||
                    (routed == bestRouted &&
                     (!usePriority ||
                      (priorityRouted == bestPriorityRouted &&
                       std::abs(priorityPenalty - bestPriorityPenalty) <=
                           1e-9)) &&
                     std::abs(unroutedPenalty - bestUnroutedPenalty) <= 1e-9 &&
                     throughputCost + 1e-9 < bestThroughputCost) ||
                    (routed == bestRouted &&
                     (!usePriority ||
                      (priorityRouted == bestPriorityRouted &&
                       std::abs(priorityPenalty - bestPriorityPenalty) <=
                           1e-9)) &&
                     std::abs(unroutedPenalty - bestUnroutedPenalty) <= 1e-9 &&
                     std::abs(throughputCost - bestThroughputCost) <= 1e-9 &&
                     clockPeriod + 1e-9 < bestClockPeriod) ||
                    (routed == bestRouted &&
                     (!usePriority ||
                      (priorityRouted == bestPriorityRouted &&
                       std::abs(priorityPenalty - bestPriorityPenalty) <=
                           1e-9)) &&
                     std::abs(unroutedPenalty - bestUnroutedPenalty) <= 1e-9 &&
                     std::abs(throughputCost - bestThroughputCost) <= 1e-9 &&
                     std::abs(clockPeriod - bestClockPeriod) <= 1e-9 &&
                     totalPathLen < bestPathLen) ||
                    (routed == bestRouted && totalPathLen == bestPathLen &&
                     (!usePriority ||
                      (priorityRouted == bestPriorityRouted &&
                       std::abs(priorityPenalty - bestPriorityPenalty) <=
                           1e-9)) &&
                     std::abs(throughputCost - bestThroughputCost) <= 1e-9 &&
                     std::abs(clockPeriod - bestClockPeriod) <= 1e-9 &&
                     placementCost + 1e-9 < bestPlacementCost);
    if (!improved)
      return false;
    bestCheckpoint = state.save();
    bestRouted = routed;
    bestUnroutedPenalty = unroutedPenalty;
    bestPathLen = totalPathLen;
    bestPlacementCost = placementCost;
    bestTimingSummary = std::move(timingSummary);
    bestThroughputCost = bestTimingSummary.estimatedThroughputCost;
    bestClockPeriod = bestTimingSummary.estimatedClockPeriod;
    bestPriorityRouted = usePriority ? priorityRouted : 0u;
    bestPriorityPenalty = usePriority ? priorityPenalty : 0.0;
    bestAllRouted = allRouted;
    return true;
  };
  auto emitBestSnapshot = [&](llvm::StringRef trigger) {
    auto checkpoint = state.save();
    state.restore(bestCheckpoint);
    maybeEmitProgressSnapshot(state, edgeKinds, trigger, opts);
    state.restore(checkpoint);
  };

  for (unsigned round = 0; round < rounds; ++round) {
    if (shouldStopForBudget("interleaved place-route"))
      break;
    state.restore(currentPlacementCheckpoint);
    rebindScalarInputSentinels(state, dfg, adg, flattener);
    classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);

    CongestionEstimator congEstimator;
    if (activeCongestionPlacementWeight > 0.0 && activeFlattener) {
      congEstimator.estimate(state, dfg, adg, *activeFlattener);
      activeCongestionEstimator = &congEstimator;
    }

    bool allRouted = (opts.negotiatedRoutingPasses > 0)
                         ? runNegotiatedRouting(state, dfg, adg, edgeKinds, opts)
                         : runRouting(state, dfg, adg, edgeKinds, opts);
    activeCongestionEstimator = nullptr;
    updateBest(allRouted);
    emitBestSnapshot("interleaved-round");

    auto edgeStats = computeRoutingEdgeStats(state, dfg, edgeKinds);
    auto failedEdges = collectUnroutedEdges(state, dfg, edgeKinds);
    llvm::outs() << "  Interleaved round " << (round + 1) << "/" << rounds
                 << ": overall " << edgeStats.routedOverallEdges << "/"
                 << edgeStats.overallEdges << ", router "
                 << edgeStats.routedRouterEdges << "/"
                 << edgeStats.routerEdges << ", prebound "
                 << edgeStats.directBindingEdges << ", failed router "
                 << failedEdges.size() << "\n";
    if (allRouted) {
      state.restore(bestCheckpoint);
      return true;
    }
    if (shouldStopForBudget("interleaved place-route"))
      break;
    if (failedEdges.empty())
      break;

    std::optional<CongestionState> repairCongestion;
    const CongestionState *repairCongestionPtr = nullptr;
    if (opts.negotiatedRoutingPasses > 0) {
      repairCongestion.emplace();
      repairCongestion->init(adg);
      repairCongestion->historyIncrement = opts.congestionHistoryFactor;
      repairCongestion->historyScale = opts.congestionHistoryScale;
      repairCongestion->presentFactor = opts.congestionPresentFactor;
      for (const auto &path : state.swEdgeToHwPaths) {
        if (!path.empty() && !(path.size() == 2 && path[0] == path[1]))
          repairCongestion->commitRoute(path, adg);
      }
      repairCongestion->updateHistory();
      repairCongestionPtr = &*repairCongestion;
    }

    state.restore(currentPlacementCheckpoint);
    bool repaired = false;
    if (opts.localRepair.enabled) {
      repaired =
          runLocalRepair(state, currentPlacementCheckpoint, failedEdges, dfg,
                         adg, flattener, candidates, edgeKinds, opts,
                         repairCongestionPtr);
    }
    bool repairImproved = updateBest(repaired, failedEdges);
    if (repairImproved)
      emitBestSnapshot("local-repair");
    if (repaired) {
      state.restore(bestCheckpoint);
      return true;
    }
    if (shouldStopForBudget("local repair"))
      break;

    if (repairImproved) {
      state.clearRoutes(dfg, adg);
      currentPlacementCheckpoint = state.save();
      continue;
    }

    if (round + 1 >= rounds)
      break;

    state.restore(bestCheckpoint);
    Options restartOpts = opts;
    restartOpts.seed = opts.seed +
                       static_cast<int>((round + 1) *
                                        opts.lane.restartSeedStride);
    if (opts.placementMoveRadius != 0)
      restartOpts.placementMoveRadius =
          opts.placementMoveRadius +
          (round + 1) * opts.lane.restartMoveRadiusStep;
    restartOpts.selectiveRipupPasses =
        opts.selectiveRipupPasses +
        std::min<unsigned>(opts.lane.restartRipupBonusCap, round + 1);
    runRefinement(state, dfg, adg, flattener, candidates, restartOpts,
                  &edgeKinds);
    state.clearRoutes(dfg, adg);
    rebindScalarInputSentinels(state, dfg, adg, flattener);
    bindMemrefSentinels(state, dfg, adg);
    classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
    currentPlacementCheckpoint = state.save();
    if (opts.verbose) {
      llvm::outs() << "  Interleaved round " << (round + 1)
                   << ": restarting from placement seed " << restartOpts.seed
                   << "\n";
    }
  }

  state.restore(bestCheckpoint);
  return bestAllRouted;
}

MapperTimingSummary Mapper::runPostRouteFifoBufferization(
    MappingState &state, const Graph &dfg, const Graph &adg,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds, const Options &opts) {
  MapperTimingSummary currentTiming =
      analyzeMapperTiming(state, dfg, adg, edgeKinds, opts.timing);
  if (!opts.bufferization.enabled)
    return currentTiming;

  llvm::DenseSet<IdIndex> recurrenceEdges;
  for (const auto &cycle : currentTiming.recurrenceCycles) {
    for (IdIndex edgeId : cycle.swEdges)
      recurrenceEdges.insert(edgeId);
  }
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (isCarryNextEdge(edgeId, dfg))
      recurrenceEdges.insert(edgeId);
  }
  llvm::DenseSet<IdIndex> criticalEdges(currentTiming.criticalPathEdges.begin(),
                                        currentTiming.criticalPathEdges.end());
  llvm::DenseSet<IdIndex> blockedRecurrenceFifos;
  for (IdIndex edgeId : recurrenceEdges) {
    if (edgeId >= state.swEdgeToHwPaths.size())
      continue;
    for (IdIndex portId : state.swEdgeToHwPaths[edgeId]) {
      const Port *port = adg.getPort(portId);
      if (!port || port->parentNode == INVALID_ID)
        continue;
      IdIndex hwNodeId = port->parentNode;
      const Node *hwNode = adg.getNode(hwNodeId);
      if (!isBypassableFifoNode(hwNode) ||
          !isEffectiveFifoBypassed(hwNodeId, state, adg)) {
        continue;
      }
      blockedRecurrenceFifos.insert(hwNodeId);
    }
  }
  if (opts.verbose && !blockedRecurrenceFifos.empty()) {
    llvm::outs() << "Mapper: excluding " << blockedRecurrenceFifos.size()
                 << " recurrence-sensitive FIFO candidates from bufferization\n";
  }

  for (unsigned iter = 0; iter < opts.bufferization.maxIterations; ++iter) {
    llvm::DenseSet<IdIndex> seenCandidates;
    std::vector<IdIndex> fifoCandidates;
    std::vector<IdIndex> criticalFifoCandidates;
    llvm::DenseMap<IdIndex, bool> candidateTouchesRecurrence;
    llvm::DenseMap<IdIndex, bool> candidateTouchesCriticalPath;
    for (IdIndex edgeId = 0;
         edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size()); ++edgeId) {
      const auto &path = state.swEdgeToHwPaths[edgeId];
      if (path.empty())
        continue;
      for (IdIndex portId : path) {
        const Port *port = adg.getPort(portId);
        if (!port || port->parentNode == INVALID_ID)
          continue;
        IdIndex hwNodeId = port->parentNode;
        const Node *hwNode = adg.getNode(hwNodeId);
        if (!isBypassableFifoNode(hwNode) ||
            !isEffectiveFifoBypassed(hwNodeId, state, adg)) {
          continue;
        }
        if (recurrenceEdges.count(edgeId) || isCarryNextEdge(edgeId, dfg))
          candidateTouchesRecurrence[hwNodeId] = true;
        if (criticalEdges.count(edgeId))
          candidateTouchesCriticalPath[hwNodeId] = true;
        if (seenCandidates.insert(hwNodeId).second) {
          fifoCandidates.push_back(hwNodeId);
          if (candidateTouchesCriticalPath.lookup(hwNodeId))
            criticalFifoCandidates.push_back(hwNodeId);
        }
      }
    }

    if (!criticalFifoCandidates.empty())
      fifoCandidates = criticalFifoCandidates;

    std::sort(fifoCandidates.begin(), fifoCandidates.end());
    IdIndex bestFifo = INVALID_ID;
    MapperTimingSummary bestTiming = currentTiming;

    for (IdIndex hwNodeId : fifoCandidates) {
      if (hwNodeId >= state.hwNodeFifoBypassedOverride.size())
        continue;
      if (blockedRecurrenceFifos.contains(hwNodeId))
        continue;
      int8_t oldOverride = state.hwNodeFifoBypassedOverride[hwNodeId];
      state.hwNodeFifoBypassedOverride[hwNodeId] = 0;
      MapperTimingSummary candidateTiming =
          analyzeMapperTiming(state, dfg, adg, edgeKinds, opts.timing);
      state.hwNodeFifoBypassedOverride[hwNodeId] = oldOverride;

      bool recurrenceSensitive = candidateTouchesRecurrence.lookup(hwNodeId);
      if (recurrenceSensitive)
        continue;
      if (!isBetterTimingSummary(candidateTiming, currentTiming,
                                 opts.bufferization)) {
        continue;
      }
      if (bestFifo == INVALID_ID ||
          isBetterTimingSummary(candidateTiming, bestTiming,
                               opts.bufferization)) {
        bestFifo = hwNodeId;
        bestTiming = std::move(candidateTiming);
      }
    }

    if (bestFifo == INVALID_ID)
      break;

    state.hwNodeFifoBypassedOverride[bestFifo] = 0;
    ++activeSearchSummary_.fifoBufferizationAcceptedToggles;
    currentTiming = std::move(bestTiming);
    if (opts.verbose) {
      llvm::outs() << "Mapper: accepted FIFO timing cut on hw node "
                   << bestFifo << ", throughput="
                   << currentTiming.estimatedThroughputCost
                   << ", clock=" << currentTiming.estimatedClockPeriod
                   << ", ii=" << currentTiming.estimatedInitiationInterval
                   << "\n";
    }
  }

  return currentTiming;
}

} // namespace fcc

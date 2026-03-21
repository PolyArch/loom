#include "MapperInternal.h"
#include "fcc/Mapper/Mapper.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

namespace fcc {

using namespace mapper_detail;

namespace {

struct SACostState {
  double totalCost = 0.0;
  bool enableGridCutLoad = false;
  std::vector<double> rowCutLoad;
  std::vector<double> colCutLoad;
  struct UndoRecord {
    bool isRow = false;
    int index = -1;
    double oldValue = 0.0;
  };
  struct Savepoint {
    size_t undoMarker = 0;
    double totalCost = 0.0;
  };
  std::vector<UndoRecord> undoLog;
  std::vector<size_t> savepointMarkers;
};

struct SAAdaptiveState {
  unsigned windowIterations = 0;
  unsigned windowAccepted = 0;
  unsigned windowBestImprovements = 0;
  unsigned iterationsSinceBestImprovement = 0;
};

double computeRouteAwareCheckpointCost(Mapper &mapper, const MappingState &state,
                                       const Graph &dfg, const Graph &adg,
                                       const ADGFlattener &flattener,
                                       llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                       const Mapper::Options &opts,
                                       double placementCost) {
  const double unroutedPenalty =
      computeUnroutedPenalty(state, dfg, edgeKinds) * 1000000.0;
  const double pathLenPenalty =
      static_cast<double>(computeTotalMappedPathLen(state)) * 0.001;
  const MapperTimingSummary timingSummary =
      analyzeMapperTiming(state, dfg, adg, edgeKinds, opts.timing);
  (void)mapper;
  return placementCost + unroutedPenalty + pathLenPenalty +
         timingSummary.estimatedThroughputCost +
         0.1 * timingSummary.estimatedClockPeriod;
}

constexpr double kCutLoadQuadraticWeight = 0.006;

SACostState::Savepoint beginCostSavepoint(SACostState &costState) {
  SACostState::Savepoint savepoint{costState.undoLog.size(),
                                   costState.totalCost};
  costState.savepointMarkers.push_back(savepoint.undoMarker);
  return savepoint;
}

void rollbackCostSavepoint(SACostState &costState,
                           SACostState::Savepoint savepoint) {
  while (costState.undoLog.size() > savepoint.undoMarker) {
    const auto &record = costState.undoLog.back();
    auto &loads =
        record.isRow ? costState.rowCutLoad : costState.colCutLoad;
    loads[record.index] = record.oldValue;
    costState.undoLog.pop_back();
  }
  costState.totalCost = savepoint.totalCost;
  if (!costState.savepointMarkers.empty())
    costState.savepointMarkers.pop_back();
}

void commitCostSavepoint(SACostState &costState,
                         SACostState::Savepoint savepoint) {
  (void)savepoint;
  if (!costState.savepointMarkers.empty())
    costState.savepointMarkers.pop_back();
}

void recordLoadUndo(SACostState &costState, bool isRow, int index,
                    double oldValue) {
  if (costState.savepointMarkers.empty())
    return;
  costState.undoLog.push_back({isRow, index, oldValue});
}

void adjustQuadraticLoad(std::vector<double> &loads, int index, double delta,
                         double &totalCost, SACostState &costState,
                         bool isRow) {
  if (index < 0 || index >= static_cast<int>(loads.size()) || delta == 0.0)
    return;
  double oldLoad = loads[index];
  recordLoadUndo(costState, isRow, index, oldLoad);
  double newLoad = oldLoad + delta;
  totalCost +=
      kCutLoadQuadraticWeight * (newLoad * newLoad - oldLoad * oldLoad);
  loads[index] = newLoad;
}

void applyEdgePlacementContribution(double sign, double edgeWeight, IdIndex srcHw,
                                    IdIndex dstHw, const Graph &adg,
                                    const ADGFlattener &flattener,
                                    SACostState &costState) {
  (void)adg;
  if (srcHw == INVALID_ID || dstHw == INVALID_ID)
    return;
  int dist = placementDistance(srcHw, dstHw, flattener);
  costState.totalCost += sign * edgeWeight * static_cast<double>(dist);
  if (!costState.enableGridCutLoad)
    return;
  auto [sr, sc] = flattener.getNodeGridPos(srcHw);
  auto [dr, dc] = flattener.getNodeGridPos(dstHw);
  if (sr < 0 || sc < 0 || dr < 0 || dc < 0)
    return;

  for (int row = std::min(sr, dr); row < std::max(sr, dr); ++row)
    adjustQuadraticLoad(costState.rowCutLoad, row, sign * edgeWeight,
                        costState.totalCost, costState, true);
  for (int col = std::min(sc, dc); col < std::max(sc, dc); ++col)
    adjustQuadraticLoad(costState.colCutLoad, col, sign * edgeWeight,
                        costState.totalCost, costState, false);
}

SACostState initializeSACostState(const MappingState &state, const Graph &dfg,
                                  const Graph &adg,
                                  const ADGFlattener &flattener,
                                  llvm::ArrayRef<double> edgeWeights) {
  const TopologyModel *topologyModel = getActiveTopologyModel();
  int maxRow = -1;
  int maxCol = -1;
  SACostState costState;
  costState.enableGridCutLoad =
      topologyModel && topologyModel->supportsGridCutLoad();
  if (costState.enableGridCutLoad) {
    for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
         ++hwId) {
      const Node *hwNode = adg.getNode(hwId);
      if (!hwNode)
        continue;
      auto [row, col] = flattener.getNodeGridPos(hwId);
      if (row >= 0 && col >= 0) {
        maxRow = std::max(maxRow, row);
        maxCol = std::max(maxCol, col);
      }
    }
  }

  costState.rowCutLoad.assign(costState.enableGridCutLoad && maxRow >= 0
                                  ? static_cast<size_t>(maxRow) + 1
                                  : 0,
                              0.0);
  costState.colCutLoad.assign(costState.enableGridCutLoad && maxCol >= 0
                                  ? static_cast<size_t>(maxCol) + 1
                                  : 0,
                              0.0);
  costState.totalCost = 0.0;

  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;
    IdIndex srcHw = state.swNodeToHwNode[srcPort->parentNode];
    IdIndex dstHw = state.swNodeToHwNode[dstPort->parentNode];
    double edgeWeight = edgeId < static_cast<IdIndex>(edgeWeights.size())
                            ? edgeWeights[edgeId]
                            : classifyEdgePlacementWeight(dfg, edgeId);
    applyEdgePlacementContribution(+1.0, edgeWeight, srcHw, dstHw, adg,
                                   flattener, costState);
  }
  for (IdIndex swNode = 0; swNode < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++swNode) {
    IdIndex hwNode = state.swNodeToHwNode[swNode];
    if (hwNode == INVALID_ID)
      continue;
    costState.totalCost += computeNodeTimingPenalty(swNode, hwNode, dfg, adg);
  }
  return costState;
}

llvm::SmallVector<IdIndex, 32>
collectPlacementDeltaEdges(llvm::ArrayRef<IdIndex> movedNodes,
                           const Graph &dfg) {
  llvm::DenseSet<IdIndex> seen;
  llvm::SmallVector<IdIndex, 32> edges;
  for (IdIndex swNode : movedNodes) {
    const Node *node = dfg.getNode(swNode);
    if (!node)
      continue;
    auto collect = [&](llvm::ArrayRef<IdIndex> ports) {
      for (IdIndex portId : ports) {
        const Port *port = dfg.getPort(portId);
        if (!port)
          continue;
        for (IdIndex edgeId : port->connectedEdges) {
          if (seen.insert(edgeId).second)
            edges.push_back(edgeId);
        }
      }
    };
    collect(node->inputPorts);
    collect(node->outputPorts);
  }
  return edges;
}

void applyPlacementDeltaForMovedNodes(
    const llvm::SmallVectorImpl<IdIndex> &movedNodes,
    const llvm::DenseMap<IdIndex, IdIndex> &oldHwBySwNode,
    const MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener, llvm::ArrayRef<double> edgeWeights,
    SACostState &costState) {
  auto affectedEdges = collectPlacementDeltaEdges(movedNodes, dfg);
  for (IdIndex edgeId : affectedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;

    IdIndex srcSw = srcPort->parentNode;
    IdIndex dstSw = dstPort->parentNode;
    IdIndex oldSrcHw = state.swNodeToHwNode[srcSw];
    IdIndex oldDstHw = state.swNodeToHwNode[dstSw];
    if (auto it = oldHwBySwNode.find(srcSw); it != oldHwBySwNode.end())
      oldSrcHw = it->second;
    if (auto it = oldHwBySwNode.find(dstSw); it != oldHwBySwNode.end())
      oldDstHw = it->second;
    IdIndex newSrcHw = state.swNodeToHwNode[srcSw];
    IdIndex newDstHw = state.swNodeToHwNode[dstSw];
    double edgeWeight = edgeId < static_cast<IdIndex>(edgeWeights.size())
                            ? edgeWeights[edgeId]
                            : classifyEdgePlacementWeight(dfg, edgeId);
    applyEdgePlacementContribution(-1.0, edgeWeight, oldSrcHw, oldDstHw, adg,
                                   flattener, costState);
    applyEdgePlacementContribution(+1.0, edgeWeight, newSrcHw, newDstHw, adg,
                                   flattener, costState);
  }

  for (IdIndex swNode : movedNodes) {
    IdIndex oldHw = state.swNodeToHwNode[swNode];
    if (auto it = oldHwBySwNode.find(swNode); it != oldHwBySwNode.end())
      oldHw = it->second;
    IdIndex newHw = state.swNodeToHwNode[swNode];
    if (oldHw != INVALID_ID)
      costState.totalCost -= computeNodeTimingPenalty(swNode, oldHw, dfg, adg);
    if (newHw != INVALID_ID)
      costState.totalCost += computeNodeTimingPenalty(swNode, newHw, dfg, adg);
  }
}

void applyAdaptiveCoolingWindow(double &temperature,
                                const Mapper::Options &opts,
                                SAAdaptiveState &adaptiveState) {
  const auto &refineOpts = opts.refinement;
  if (!refineOpts.adaptiveCoolingEnabled || adaptiveState.windowIterations == 0)
    return;

  const double initialTemperature = refineOpts.initialTemperature;
  const double minTemperature = refineOpts.minTemperature;
  const double maxTemperature =
      initialTemperature * refineOpts.maxTemperatureScale;
  const double acceptanceRatio =
      static_cast<double>(adaptiveState.windowAccepted) /
      static_cast<double>(adaptiveState.windowIterations);

  if (acceptanceRatio < refineOpts.targetAcceptanceLow) {
    temperature = std::min(maxTemperature, temperature *
                                               refineOpts
                                                   .coldAcceptanceReheatMultiplier);
  } else if (acceptanceRatio > refineOpts.targetAcceptanceHigh) {
    temperature = std::max(
        minTemperature,
        temperature * refineOpts.hotAcceptanceCoolingMultiplier);
  }

  if (refineOpts.plateauWindow > 0 &&
      adaptiveState.iterationsSinceBestImprovement >=
          refineOpts.plateauWindow &&
      adaptiveState.windowBestImprovements == 0) {
    temperature = std::min(maxTemperature,
                           std::max(temperature, minTemperature) *
                               refineOpts.plateauReheatMultiplier);
    adaptiveState.iterationsSinceBestImprovement = 0;
  }

  adaptiveState.windowIterations = 0;
  adaptiveState.windowAccepted = 0;
  adaptiveState.windowBestImprovements = 0;
}

} // namespace

bool Mapper::runRefinement(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    const Options &opts, std::vector<TechMappedEdgeKind> *edgeKinds) {

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

  CandidateSetMap candidateSets = buildCandidateSetMap(candidates);
  std::vector<double> edgeWeights = buildEdgePlacementWeightCache(dfg);
  const TopologyModel *topologyModel = getActiveTopologyModel();

  double temperature = opts.refinement.initialTemperature;
  double coolingRate = opts.refinement.coolingRate;
  int maxIter = static_cast<int>(placedNodes.size()) *
                static_cast<int>(opts.refinement.iterationsPerPlacedNode);
  if (maxIter > static_cast<int>(opts.refinement.iterationCap))
    maxIter = static_cast<int>(opts.refinement.iterationCap);

  SACostState costState =
      initializeSACostState(state, dfg, adg, flattener, edgeWeights);
  SAAdaptiveState adaptiveState;
  const bool routeAwareMode =
      edgeKinds && state.routeStatsInitialized &&
      countRoutedEdges(state, dfg, *edgeKinds) > 0;
  if (routeAwareMode)
    ++activeSearchSummary_.routeAwareRefinementPasses;
  double currentObjective =
      routeAwareMode
          ? computeRouteAwareCheckpointCost(*this, state, dfg, adg, flattener,
                                            *edgeKinds, opts,
                                            costState.totalCost)
          : costState.totalCost;
  double bestCost = currentObjective;
  auto bestCheckpoint = state.save();
  auto routeCheckpointState = bestCheckpoint;
  std::vector<TechMappedEdgeKind> routeCheckpointEdgeKinds =
      edgeKinds ? *edgeKinds : std::vector<TechMappedEdgeKind>();
  double routeCheckpointObjective = currentObjective;
  unsigned acceptedSinceRouteCheckpoint = 0;
  int acceptCount = 0;

  auto startTime = std::chrono::steady_clock::now();
  double localBudgetSeconds =
      std::min(opts.budgetSeconds * opts.refinement.budgetFraction,
               remainingBudgetSeconds());

  for (int iter = 0; iter < maxIter; ++iter) {
    if (shouldStopForBudget("placement refinement"))
      break;
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    double secs =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() /
        1000.0;
    if (secs > localBudgetSeconds)
      break;

    double oldCost = currentObjective;
    auto savepoint = state.beginSavepoint();
    auto costSavepoint = beginCostSavepoint(costState);
    bool improvedBestThisIter = false;

    bool moveOk = false;
    bool doSwap = std::uniform_int_distribution<int>(0, 1)(rng) == 0;
    llvm::SmallVector<IdIndex, 2> movedNodes;
    llvm::DenseMap<IdIndex, IdIndex> oldHwBySwNode;

    if (doSwap) {
      std::uniform_int_distribution<size_t> dist(0, placedNodes.size() - 1);
      IdIndex swA = placedNodes[dist(rng)];
      IdIndex hwA = state.swNodeToHwNode[swA];

      llvm::SmallVector<IdIndex, 16> nearbyNodes;
      if (topologyModel) {
        std::vector<IdIndex> nearbyHwNodes =
            topologyModel->placeableNodesWithinRadius(hwA,
                                                      opts.placementMoveRadius);
        for (IdIndex nearbyHwNode : nearbyHwNodes) {
          if (nearbyHwNode >= state.hwNodeToSwNodes.size())
            continue;
          for (IdIndex swCandidate : state.hwNodeToSwNodes[nearbyHwNode]) {
            if (swCandidate == swA)
              continue;
            nearbyNodes.push_back(swCandidate);
          }
        }
      } else {
        llvm::DenseMap<int64_t, llvm::SmallVector<IdIndex, 4>> hwNodesByCell;
        auto cellKey = [](int row, int col) -> int64_t {
          return (static_cast<int64_t>(row) << 32) ^
                 static_cast<uint32_t>(col);
        };
        for (IdIndex hwNode = 0; hwNode < static_cast<IdIndex>(adg.nodes.size());
             ++hwNode) {
          auto [row, col] = flattener.getNodeGridPos(hwNode);
          if (row < 0 || col < 0)
            continue;
          hwNodesByCell[cellKey(row, col)].push_back(hwNode);
        }
        auto [hwARow, hwACol] = flattener.getNodeGridPos(hwA);
        for (int dr = -static_cast<int>(opts.placementMoveRadius);
             dr <= static_cast<int>(opts.placementMoveRadius); ++dr) {
          for (int dc = -static_cast<int>(opts.placementMoveRadius);
               dc <= static_cast<int>(opts.placementMoveRadius); ++dc) {
            int row = hwARow + dr;
            int col = hwACol + dc;
            auto it = hwNodesByCell.find(cellKey(row, col));
            if (it == hwNodesByCell.end())
              continue;
            for (IdIndex nearbyHwNode : it->second) {
              if (!isWithinMoveRadius(hwA, nearbyHwNode, flattener,
                                      opts.placementMoveRadius))
                continue;
              if (nearbyHwNode >= state.hwNodeToSwNodes.size())
                continue;
              for (IdIndex swCandidate : state.hwNodeToSwNodes[nearbyHwNode]) {
                if (swCandidate == swA)
                  continue;
                nearbyNodes.push_back(swCandidate);
              }
            }
          }
        }
      }
      llvm::sort(nearbyNodes);
      nearbyNodes.erase(std::unique(nearbyNodes.begin(), nearbyNodes.end()),
                        nearbyNodes.end());
      if (nearbyNodes.empty())
        goto rollback_move;

      std::uniform_int_distribution<size_t> nearbyDist(0,
                                                       nearbyNodes.size() - 1);
      IdIndex swB = nearbyNodes[nearbyDist(rng)];
      IdIndex hwB = state.swNodeToHwNode[swB];
      if (!canSwapNodes(swA, swB, hwA, hwB, state, adg, flattener, candidates,
                        &candidateSets))
        goto rollback_move;

      movedNodes.push_back(swA);
      movedNodes.push_back(swB);
      oldHwBySwNode[swA] = hwA;
      oldHwBySwNode[swB] = hwB;

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
        goto rollback_move;

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
                             candidates, &candidateSets))
          continue;

        double candScore = scorePlacement(swNode, candHw, state, dfg, adg,
                                          flattener, candidates);
        if (candScore > bestCandScore + 1e-9) {
          bestCandScore = candScore;
          topCandidates.clear();
          topCandidates.push_back(candHw);
        } else if (std::abs(candScore - bestCandScore) <= 1e-9 &&
                   topCandidates.size() <
                       opts.refinement.relocateTopCandidateLimit) {
          topCandidates.push_back(candHw);
        }
      }

      if (!topCandidates.empty()) {
        std::uniform_int_distribution<size_t> candDist(0, topCandidates.size() -
                                                              1);
        newHw = topCandidates[candDist(rng)];
      }
      if (newHw == INVALID_ID)
        goto rollback_move;

      movedNodes.push_back(swNode);
      oldHwBySwNode[swNode] = oldHw;

      state.unmapNode(swNode, dfg, adg);
      auto result = state.mapNode(swNode, newHw, dfg, adg);
      moveOk = (result == ActionResult::Success) &&
               bindMappedNodePorts(swNode, state, dfg, adg);
    }

    if (!moveOk) {
rollback_move:
      state.rollbackSavepoint(savepoint);
      rollbackCostSavepoint(costState, costSavepoint);
      continue;
    }

    applyPlacementDeltaForMovedNodes(movedNodes, oldHwBySwNode, state, dfg, adg,
                                     flattener, edgeWeights, costState);
    bool usedNeighborhoodRepair = false;
    if (routeAwareMode) {
      classifyTemporalRegisterEdges(state, dfg, adg, flattener, *edgeKinds);
      auto repairEdges = collectPlacementDeltaEdges(movedNodes, dfg);
      llvm::DenseSet<IdIndex> seenRepairEdges(repairEdges.begin(),
                                              repairEdges.end());
      for (IdIndex edgeId : collectUnroutedEdges(state, dfg, *edgeKinds)) {
        if (seenRepairEdges.insert(edgeId).second)
          repairEdges.push_back(edgeId);
      }
      if (opts.refinement.routeAwareNeighborhoodEnabled &&
          repairEdges.size() <= opts.refinement.routeAwareNeighborhoodEdgeCap) {
        ++activeSearchSummary_.routeAwareNeighborhoodAttempts;
        usedNeighborhoodRepair = true;
        runExactRoutingRepair(state, repairEdges, dfg, adg, flattener,
                              *edgeKinds, opts);
      } else {
        ++activeSearchSummary_.routeAwareCoarseFallbackMoves;
      }
    }
    double newCost =
        routeAwareMode
            ? computeRouteAwareCheckpointCost(*this, state, dfg, adg, flattener,
                                              *edgeKinds, opts,
                                              costState.totalCost)
            : costState.totalCost;
    double delta = oldCost - newCost;
    bool acceptMove =
        delta > 0 || std::uniform_real_distribution<double>(0.0, 1.0)(rng) <
                         std::exp(delta / temperature);
    if (acceptMove) {
      ++acceptCount;
      ++adaptiveState.windowAccepted;
      if (usedNeighborhoodRepair)
        ++activeSearchSummary_.routeAwareNeighborhoodAcceptedMoves;
      state.commitSavepoint(savepoint);
      commitCostSavepoint(costState, costSavepoint);
      currentObjective = newCost;
      bool allowImmediateBestUpdate =
          !routeAwareMode || usedNeighborhoodRepair ||
          !opts.refinement.routeAwareCheckpointEnabled;
      if (allowImmediateBestUpdate && newCost < bestCost) {
        bestCost = newCost;
        improvedBestThisIter = true;
        adaptiveState.windowBestImprovements++;
        adaptiveState.iterationsSinceBestImprovement = 0;
        bestCheckpoint = state.save();
      }
      if (routeAwareMode) {
        if (usedNeighborhoodRepair) {
          routeCheckpointState = state.save();
          routeCheckpointEdgeKinds = *edgeKinds;
          routeCheckpointObjective = currentObjective;
          acceptedSinceRouteCheckpoint = 0;
        } else {
          ++acceptedSinceRouteCheckpoint;
          if (opts.refinement.routeAwareCheckpointEnabled &&
              acceptedSinceRouteCheckpoint >=
                  opts.refinement.routeAwareCheckpointAcceptedMoveBatch) {
            ++activeSearchSummary_.routeAwareCheckpointRescorePasses;
            bool rerouteSucceeded = runRouting(state, dfg, adg, *edgeKinds, opts);
            if (rerouteSucceeded)
              classifyTemporalRegisterEdges(state, dfg, adg, flattener,
                                            *edgeKinds);
            double reroutedObjective =
                rerouteSucceeded
                    ? computeRouteAwareCheckpointCost(
                          *this, state, dfg, adg, flattener, *edgeKinds, opts,
                          costState.totalCost)
                    : std::numeric_limits<double>::infinity();
            if (!rerouteSucceeded ||
                reroutedObjective > routeCheckpointObjective + 1e-9) {
              ++activeSearchSummary_.routeAwareCheckpointRestoreCount;
              state.restore(routeCheckpointState);
              *edgeKinds = routeCheckpointEdgeKinds;
              costState = initializeSACostState(state, dfg, adg, flattener,
                                                edgeWeights);
              currentObjective = routeCheckpointObjective;
            } else {
              currentObjective = reroutedObjective;
              routeCheckpointState = state.save();
              routeCheckpointEdgeKinds = *edgeKinds;
              routeCheckpointObjective = currentObjective;
              if (currentObjective < bestCost) {
                bestCost = currentObjective;
                improvedBestThisIter = true;
                adaptiveState.windowBestImprovements++;
                adaptiveState.iterationsSinceBestImprovement = 0;
                bestCheckpoint = state.save();
              }
            }
            acceptedSinceRouteCheckpoint = 0;
          }
        }
      }
    } else {
      state.rollbackSavepoint(savepoint);
      rollbackCostSavepoint(costState, costSavepoint);
      if (routeAwareMode)
        classifyTemporalRegisterEdges(state, dfg, adg, flattener, *edgeKinds);
    }

    ++adaptiveState.windowIterations;
    if (!improvedBestThisIter)
      ++adaptiveState.iterationsSinceBestImprovement;
    temperature = std::max(opts.refinement.minTemperature,
                           temperature * coolingRate);
    if (opts.refinement.adaptiveCoolingEnabled &&
        adaptiveState.windowIterations >= opts.refinement.adaptiveWindow) {
      applyAdaptiveCoolingWindow(temperature, opts, adaptiveState);
    }
  }

  state.restore(bestCheckpoint);
  if (edgeKinds)
    classifyTemporalRegisterEdges(state, dfg, adg, flattener, *edgeKinds);
  if (opts.verbose) {
    llvm::outs() << "  SA: " << acceptCount << " accepted moves, best cost "
                 << bestCost << "\n";
  }
  return true;
}

} // namespace fcc

#include "MapperInternal.h"
#include "MapperRoutingCongestion.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MapperTiming.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

namespace loom {

using namespace mapper_detail;

bool Mapper::runRouteAwareSA(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    std::vector<TechMappedEdgeKind> &edgeKinds, const Options &opts) {

  std::mt19937 rng(static_cast<unsigned>(opts.seed));

  // Collect placed operation nodes (same as runRefinement).
  std::vector<IdIndex> placedNodes;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    if (dfg.getNode(i) && dfg.getNode(i)->kind == Node::OperationNode &&
        state.swNodeToHwNode[i] != INVALID_ID) {
      placedNodes.push_back(i);
    }
  }

  // Count total routable edges (overall edges minus IntraFU/TemporalReg).
  unsigned totalRoutableEdges = 0;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (edgeId < static_cast<IdIndex>(edgeKinds.size()) &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    ++totalRoutableEdges;
  }

  if (placedNodes.size() < 2) {
    unsigned routedCount = countRoutedEdges(state, dfg, edgeKinds);
    return routedCount >= totalRoutableEdges;
  }

  CandidateSetMap candidateSets = buildCandidateSetMap(candidates);
  std::vector<double> edgeWeights = buildEdgePlacementWeightCache(dfg);
  const TopologyModel *topologyModel = getActiveTopologyModel();

  // Temperature and cooling from route-aware SA options.
  double temperature = opts.refinement.routeAwareSAInitialTemperature;
  double coolingRate = opts.refinement.routeAwareSACoolingRate;
  const double minTemperature = opts.refinement.routeAwareSAMinTemperature;

  // Initialize cost state.
  SACostState costState =
      initializeSACostState(state, dfg, adg, flattener, edgeWeights);
  SAAdaptiveState adaptiveState;

  // Route-aware cost includes routing quality.
  double currentObjective = computeRouteAwareCheckpointCost(
      state, dfg, adg, flattener, edgeKinds, opts, costState.totalCost);
  double bestCost = currentObjective;
  auto bestCheckpoint = state.save();
  auto routeCheckpointState = bestCheckpoint;
  auto routeCheckpointEdgeKinds = edgeKinds;
  double routeCheckpointObjective = currentObjective;
  unsigned acceptedSinceRouteCheckpoint = 0;
  int acceptCount = 0;
  bool allRoutedReached = false;

  // Time budget.
  auto startTime = std::chrono::steady_clock::now();
  double localBudgetSeconds =
      std::min(opts.budgetSeconds * opts.refinement.routeAwareSABudgetFraction,
               remainingBudgetSeconds());

  // Routing output history for findPath heuristics.
  llvm::DenseMap<IdIndex, double> routingOutputHistory;

  if (opts.verbose) {
    unsigned routedCount = countRoutedEdges(state, dfg, edgeKinds);
    llvm::outs() << "  Route-aware SA: starting with " << placedNodes.size()
                 << " placed nodes, " << routedCount << "/"
                 << totalRoutableEdges << " routed edges\n";
  }

  // ---- MAIN SA LOOP ----
  for (int iter = 0;; ++iter) {
    if (shouldStopForBudget("route-aware SA"))
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

    // 1. Generate move (50% swap, 50% relocate).
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
        for (IdIndex hwNode = 0;
             hwNode < static_cast<IdIndex>(adg.nodes.size()); ++hwNode) {
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
              for (IdIndex swCandidate :
                   state.hwNodeToSwNodes[nearbyHwNode]) {
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

      std::uniform_int_distribution<size_t> nearbyDist(
          0, nearbyNodes.size() - 1);
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
        std::uniform_int_distribution<size_t> candDist(
            0, topCandidates.size() - 1);
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

    // 2. Update placement cost incrementally.
    applyPlacementDeltaForMovedNodes(movedNodes, oldHwBySwNode, state, dfg, adg,
                                     flattener, edgeWeights, costState);

    // 3. Tiered re-routing.
    classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
    auto repairEdges = collectPlacementDeltaEdges(movedNodes, dfg);

    // Merge in currently unrouted edges.
    llvm::DenseSet<IdIndex> seenRepairEdges(repairEdges.begin(),
                                            repairEdges.end());
    for (IdIndex edgeId : collectUnroutedEdges(state, dfg, edgeKinds)) {
      if (seenRepairEdges.insert(edgeId).second)
        repairEdges.push_back(edgeId);
    }

    bool usedExactRepair = false;

    // Tier 1: Cheap re-route via findPath for each affected edge.
    llvm::SmallVector<IdIndex, 16> cheapFailedEdges;
    for (IdIndex edgeId : repairEdges) {
      if (edgeId >= static_cast<IdIndex>(edgeKinds.size()))
        continue;
      if (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
          edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg)
        continue;

      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      const Port *dstPort = dfg.getPort(edge->dstPort);
      if (!srcPort || !dstPort)
        continue;

      IdIndex srcHwPort =
          (edge->srcPort < state.swPortToHwPort.size())
              ? state.swPortToHwPort[edge->srcPort]
              : INVALID_ID;
      IdIndex dstHwPort =
          (edge->dstPort < state.swPortToHwPort.size())
              ? state.swPortToHwPort[edge->dstPort]
              : INVALID_ID;
      if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID) {
        cheapFailedEdges.push_back(edgeId);
        continue;
      }

      // Rip up existing path.
      state.unmapEdge(edgeId, dfg, adg);

      // Try cheap findPath.
      auto path = findPath(srcHwPort, dstHwPort, edgeId, state, dfg, adg,
                           routingOutputHistory);
      if (!path.empty()) {
        state.mapEdge(edgeId, path, dfg, adg);
      } else {
        cheapFailedEdges.push_back(edgeId);
      }
    }

    // Tier 2: Exact repair if cheap re-route left a manageable number of
    // failures.
    if (!cheapFailedEdges.empty() &&
        cheapFailedEdges.size() <=
            opts.refinement.routeAwareSANeighborhoodEdgeCap) {
      usedExactRepair = true;
      ++activeSearchSummary_.routeAwareNeighborhoodAttempts;
      runExactRoutingRepair(state, cheapFailedEdges, dfg, adg, flattener,
                            edgeKinds, opts);
    }
    // Tier 3: Coarse fallback -- do nothing extra, just evaluate cost.

    // 4. Compute new route-aware cost.
    double newCost = computeRouteAwareCheckpointCost(
        state, dfg, adg, flattener, edgeKinds, opts, costState.totalCost);

    // 5. Boltzmann accept/reject.
    double delta = oldCost - newCost;
    bool acceptMove =
        delta > 0 || std::uniform_real_distribution<double>(0.0, 1.0)(rng) <
                         std::exp(delta / temperature);

    if (acceptMove) {
      ++acceptCount;
      ++adaptiveState.windowAccepted;
      if (usedExactRepair)
        ++activeSearchSummary_.routeAwareNeighborhoodAcceptedMoves;
      else
        ++activeSearchSummary_.routeAwareCoarseFallbackMoves;

      state.commitSavepoint(savepoint);
      commitCostSavepoint(costState, costSavepoint);
      currentObjective = newCost;

      if (newCost < bestCost) {
        bestCost = newCost;
        improvedBestThisIter = true;
        adaptiveState.windowBestImprovements++;
        adaptiveState.iterationsSinceBestImprovement = 0;
        bestCheckpoint = state.save();
      }

      // 6. Checkpoint management.
      routeCheckpointState = state.save();
      routeCheckpointEdgeKinds = edgeKinds;
      routeCheckpointObjective = currentObjective;
      ++acceptedSinceRouteCheckpoint;

      // Tier 4: Periodic full re-route after a batch of accepted moves.
      // Adaptive batch sizing: reduce batch for near-success cases.
      unsigned adaptiveBatch = opts.refinement.routeAwareSACheckpointMoveBatch;
      {
        unsigned unroutedNow =
            collectUnroutedEdges(state, dfg, edgeKinds).size();
        unsigned baseBatch = opts.refinement.routeAwareSACheckpointMoveBatch;
        if (unroutedNow <= 3)
          adaptiveBatch = std::max(4u, baseBatch / 4);
        else if (unroutedNow <= 10)
          adaptiveBatch = baseBatch / 2;
        else
          adaptiveBatch = baseBatch;
      }
      if (acceptedSinceRouteCheckpoint >= adaptiveBatch) {
        ++activeSearchSummary_.routeAwareCheckpointRescorePasses;
        bool rerouteSucceeded =
            runRouting(state, dfg, adg, edgeKinds, opts);
        if (rerouteSucceeded)
          classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
        double reroutedObjective =
            rerouteSucceeded
                ? computeRouteAwareCheckpointCost(state, dfg, adg, flattener,
                                                  edgeKinds, opts,
                                                  costState.totalCost)
                : std::numeric_limits<double>::infinity();
        if (!rerouteSucceeded ||
            reroutedObjective > routeCheckpointObjective + 1e-9) {
          ++activeSearchSummary_.routeAwareCheckpointRestoreCount;
          state.restore(routeCheckpointState);
          edgeKinds = routeCheckpointEdgeKinds;
          costState = initializeSACostState(state, dfg, adg, flattener,
                                            edgeWeights);
          currentObjective = routeCheckpointObjective;
        } else {
          currentObjective = reroutedObjective;
          routeCheckpointState = state.save();
          routeCheckpointEdgeKinds = edgeKinds;
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

      // Check if all edges are now routed.
      if (!allRoutedReached) {
        unsigned routedNow = countRoutedEdges(state, dfg, edgeKinds);
        if (routedNow >= totalRoutableEdges) {
          allRoutedReached = true;
          if (opts.verbose) {
            llvm::outs()
                << "  Route-aware SA: all edges routed at iter " << iter
                << ", switching to accelerated cooling\n";
          }
        }
      }
    } else {
      // Reject the move.
      state.rollbackSavepoint(savepoint);
      rollbackCostSavepoint(costState, costSavepoint);
      classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
    }

    // 7. Cooling.
    ++adaptiveState.windowIterations;
    if (!improvedBestThisIter)
      ++adaptiveState.iterationsSinceBestImprovement;

    double effectiveCoolingRate =
        allRoutedReached ? (coolingRate * 0.95) : coolingRate;
    temperature = std::max(minTemperature, temperature * effectiveCoolingRate);

    if (opts.refinement.adaptiveCoolingEnabled &&
        adaptiveState.windowIterations >= opts.refinement.adaptiveWindow) {
      // Periodic logging before adaptive cooling resets counters.
      if (opts.verbose) {
        double acceptanceRatio =
            adaptiveState.windowIterations > 0
                ? static_cast<double>(adaptiveState.windowAccepted) /
                      static_cast<double>(adaptiveState.windowIterations)
                : 0.0;
        unsigned unrouted = collectUnroutedEdges(state, dfg, edgeKinds).size();
        llvm::outs() << "  Route-aware SA: T=" << temperature
                     << " accept=" << acceptanceRatio
                     << " unrouted=" << unrouted << " best=" << bestCost
                     << "\n";
      }
      applyAdaptiveCoolingWindow(temperature, opts, adaptiveState);
    }
  }

  // Restore best checkpoint.
  state.restore(bestCheckpoint);
  classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
  ++activeSearchSummary_.routeAwareRefinementPasses;

  unsigned finalRouted = countRoutedEdges(state, dfg, edgeKinds);
  if (opts.verbose) {
    llvm::outs() << "  Route-aware SA: " << acceptCount
                 << " accepted moves, best cost " << bestCost << ", unrouted "
                 << (totalRoutableEdges - finalRouted) << "\n";
  }

  return finalRouted >= totalRoutableEdges;
}

} // namespace loom

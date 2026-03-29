#include "MapperLocalRepairInternal.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace loom {

using namespace mapper_detail;

bool LocalRepairDriver::runHotspotRepairAndEarlyCPSat() {
  if (opts.enableCPSat &&
      recursionDepth >= repairOpts.earlyCPSatRecursionDepthThreshold &&
      failedEdges.size() <= repairOpts.earlyCPSatFailedEdgeThreshold) {
    state.restore(bestPlacementCheckpoint);
    Mapper::Options earlyCpSatOpts = opts;
    earlyCpSatOpts.cpSatTimeLimitSeconds =
        std::max(earlyCpSatOpts.cpSatTimeLimitSeconds,
                 repairOpts.earlyCPSatMinTime);
    earlyCpSatOpts.cpSatNeighborhoodNodeLimit =
        std::max<unsigned>(earlyCpSatOpts.cpSatNeighborhoodNodeLimit,
                           repairOpts.earlyCPSatNeighborhoodLimit);
    earlyCpSatOpts.placementMoveRadius =
        std::max<unsigned>(earlyCpSatOpts.placementMoveRadius,
                           repairOpts.earlyCPSatMoveRadius);
    bool allRouted = mapper.runCPSatNeighborhoodRepair(
        state, bestPlacementCheckpoint, failedEdges, dfg, adg, flattener,
        candidates, edgeKinds, earlyCpSatOpts);
    if (updateBest(allRouted) && allRouted) {
      state.restore(bestCheckpoint);
      return true;
    }
  }

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

  unsigned maxHotspots =
      std::min<unsigned>(hotspots.size(), repairOpts.hotspotLimit);
  bool deferToCPSat = false;
  for (unsigned round = 0; round < std::max(1u, opts.selectiveRipupPasses);
       ++round) {
    if (mapper.shouldStopForBudget("local repair"))
      break;
    bool improvedThisRound = false;
    const unsigned repairRadius =
        opts.placementMoveRadius == 0
            ? 0
            : opts.placementMoveRadius + round * repairOpts.repairRadiusStep +
                  repairOpts.repairRadiusBias;

    for (unsigned hotIdx = 0; hotIdx < maxHotspots; ++hotIdx) {
      if (mapper.shouldStopForBudget("local repair"))
        break;
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
                             candidates, &candidateSets))
          continue;
        double candScore = mapper.scorePlacement(swNode, candHw, state, dfg, adg,
                                                 flattener, candidates);
        double failedEdgeDelta =
            evaluateFailedEdgeDelta({std::make_pair(swNode, candHw)});
        // Congestion penalty: penalize relocating to congested locations.
        double congestionPenalty = 0.0;
        if (oldHw != INVALID_ID && candHw != INVALID_ID) {
          congestionPenalty = congestionEstimator.demandCapacityRatio(
              oldHw, candHw, adg, flattener);
        }
        double repairScore =
            failedEdgeDelta * repairOpts.failedEdgeDeltaScoreWeight +
            candScore * repairOpts.candidateScoreWeight -
            congestionPenalty * repairOpts.hotspotDistanceScoreWeight;
        relocations.push_back({-repairScore, candHw});
      }
      llvm::stable_sort(relocations, [&](const auto &lhs, const auto &rhs) {
        if (lhs.first != rhs.first)
          return lhs.first < rhs.first;
        return lhs.second < rhs.second;
      });

      unsigned maxRelocations = std::min<unsigned>(
          relocations.size(), repairOpts.relocationCandidateLimit);
      for (unsigned moveIdx = 0; moveIdx < maxRelocations; ++moveIdx) {
        if (mapper.shouldStopForBudget("local repair"))
          break;
        state.restore(bestPlacementCheckpoint);
        state.unmapNode(swNode, dfg, adg);
        if (state.mapNode(swNode, relocations[moveIdx].second, dfg, adg) !=
            ActionResult::Success) {
          state.restore(bestPlacementCheckpoint);
          continue;
        }
        if (!mapper.bindMappedNodePorts(swNode, state, dfg, adg)) {
          state.restore(bestPlacementCheckpoint);
          continue;
        }
        mapper.bindMemrefSentinels(state, dfg, adg);
        classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
        bool allRouted = rerouteRepairState(state);
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
                          flattener, candidates, &candidateSets))
          continue;
        double swapScore =
            evaluateFailedEdgeDelta({std::make_pair(swNode, otherHw),
                                     std::make_pair(otherSw, oldHw)}) *
                repairOpts.failedEdgeDeltaScoreWeight +
            mapper.scorePlacement(swNode, otherHw, state, dfg, adg, flattener,
                                  candidates) *
                repairOpts.candidateScoreWeight +
            mapper.scorePlacement(otherSw, oldHw, state, dfg, adg, flattener,
                                  candidates) *
                repairOpts.candidateScoreWeight;
        swapPartners.push_back({-swapScore, otherSw});
      }
      llvm::stable_sort(swapPartners, [&](const auto &lhs, const auto &rhs) {
        if (lhs.first != rhs.first)
          return lhs.first < rhs.first;
        return lhs.second < rhs.second;
      });

      unsigned maxSwaps = std::min<unsigned>(swapPartners.size(),
                                             repairOpts.swapCandidateLimit);
      for (unsigned swapIdx = 0; swapIdx < maxSwaps; ++swapIdx) {
        if (mapper.shouldStopForBudget("local repair"))
          break;
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
        if (!mapper.bindMappedNodePorts(swNode, state, dfg, adg) ||
            !mapper.bindMappedNodePorts(otherSw, state, dfg, adg)) {
          state.restore(bestPlacementCheckpoint);
          continue;
        }
        mapper.bindMemrefSentinels(state, dfg, adg);
        classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
        bool allRouted = rerouteRepairState(state);
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
    if (mapper.shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
    state.restore(bestPlacementCheckpoint);
    Mapper::Options cpSatRepairOpts = opts;
    if (bestFailedEdges.size() <= repairOpts.cpSatSmallFailedThreshold) {
      cpSatRepairOpts.cpSatTimeLimitSeconds =
          std::max(cpSatRepairOpts.cpSatTimeLimitSeconds,
                   repairOpts.cpSatSmallFailedMinTime);
      cpSatRepairOpts.cpSatNeighborhoodNodeLimit =
          std::max<unsigned>(cpSatRepairOpts.cpSatNeighborhoodNodeLimit,
                             repairOpts.cpSatSmallFailedNodeLimit);
    } else if (bestFailedEdges.size() <=
               repairOpts.cpSatMediumFailedThreshold) {
      cpSatRepairOpts.cpSatTimeLimitSeconds =
          std::max(cpSatRepairOpts.cpSatTimeLimitSeconds,
                   repairOpts.cpSatMediumFailedMinTime);
      cpSatRepairOpts.cpSatNeighborhoodNodeLimit =
          std::max<unsigned>(cpSatRepairOpts.cpSatNeighborhoodNodeLimit,
                             repairOpts.cpSatMediumFailedNodeLimit);
    }
    bool allRouted = mapper.runCPSatNeighborhoodRepair(
        state, bestPlacementCheckpoint, bestFailedEdges, dfg, adg, flattener,
        candidates, edgeKinds, cpSatRepairOpts);
    if (updateBest(allRouted) && allRouted) {
      state.restore(bestCheckpoint);
      return true;
    }
  }

  return false;
}

} // namespace loom

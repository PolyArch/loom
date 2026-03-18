#include "MapperInternal.h"
#include "fcc/Mapper/Mapper.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

namespace fcc {

using namespace mapper_detail;

bool Mapper::runRefinement(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    const Options &opts) {

  // SA refinement: swap + relocate moves with Metropolis acceptance.
  std::mt19937 rng(static_cast<unsigned>(opts.seed));

  // Collect placed operation nodes.
  std::vector<IdIndex> placedNodes;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    if (dfg.getNode(i) && dfg.getNode(i)->kind == Node::OperationNode &&
        state.swNodeToHwNode[i] != INVALID_ID &&
        !isMemoryOp(dfg.getNode(i))) {
      placedNodes.push_back(i);
    }
  }

  if (placedNodes.size() < 2)
    return true;

  double temperature = 100.0;
  double coolingRate = 0.995;
  int maxIter = static_cast<int>(placedNodes.size()) * 1000;
  // Cap at reasonable limit to avoid excessive runtime.
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
      break; // Reserve time for routing.

    double oldCost = computeTotalCost(state, dfg, adg, flattener);
    auto cp = state.save();

    bool moveOk = false;
    // 50% swap moves, 50% relocate moves.
    bool doSwap = std::uniform_int_distribution<int>(0, 1)(rng) == 0;

    if (doSwap) {
      // Swap two placed nodes.
      std::uniform_int_distribution<size_t> dist(0, placedNodes.size() - 1);
      size_t idxA = dist(rng);
      size_t idxB = dist(rng);
      if (idxA == idxB)
        continue;

      IdIndex swA = placedNodes[idxA];
      IdIndex swB = placedNodes[idxB];
      IdIndex hwA = state.swNodeToHwNode[swA];
      IdIndex hwB = state.swNodeToHwNode[swB];

      auto candItA = candidates.find(swA);
      auto candItB = candidates.find(swB);
      if (candItA == candidates.end() || candItB == candidates.end())
        continue;

      bool aCanGoToB = false, bCanGoToA = false;
      for (IdIndex c : candItA->second)
        if (c == hwB) { aCanGoToB = true; break; }
      for (IdIndex c : candItB->second)
        if (c == hwA) { bCanGoToA = true; break; }

      if (!aCanGoToB || !bCanGoToA)
        continue;

      const Node *hwNodeA = adg.getNode(hwA);
      const Node *hwNodeB = adg.getNode(hwB);
      if (!hwNodeA || !hwNodeB)
        continue;
      llvm::StringRef peNameA = getNodeAttrStr(hwNodeA, "pe_name");
      llvm::StringRef peNameB = getNodeAttrStr(hwNodeB, "pe_name");
      if (!peNameA.empty() && peNameA == peNameB &&
          isSpatialPEName(flattener, peNameA))
        continue;

      state.unmapNode(swA, dfg, adg);
      state.unmapNode(swB, dfg, adg);
      auto r1 = state.mapNode(swA, hwB, dfg, adg);
      auto r2 = state.mapNode(swB, hwA, dfg, adg);
      moveOk = (r1 == ActionResult::Success && r2 == ActionResult::Success);
    } else {
      // Relocate: move one node to a random empty candidate.
      std::uniform_int_distribution<size_t> dist(0, placedNodes.size() - 1);
      size_t idx = dist(rng);
      IdIndex swN = placedNodes[idx];
      IdIndex hwOld = state.swNodeToHwNode[swN];

      auto candIt = candidates.find(swN);
      if (candIt == candidates.end() || candIt->second.size() < 2)
        continue;

      auto &candList = candIt->second;
      IdIndex hwNew = INVALID_ID;
      double bestCandScore = -1.0e18;
      llvm::SmallVector<IdIndex, 8> topCandidates;
      for (IdIndex cand : candList) {
        if (cand == hwOld)
          continue;
        if (!state.hwNodeToSwNodes[cand].empty())
          continue;
        // Check PE exclusivity.
        const Node *hwNode = adg.getNode(cand);
        if (!hwNode)
          continue;
        llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
        if (!peName.empty() && isSpatialPEName(flattener, peName)) {
          // Check if any other node is on the same PE.
          bool peConflict = false;
          for (IdIndex other : placedNodes) {
            if (other == swN)
              continue;
            IdIndex otherHw = state.swNodeToHwNode[other];
            if (otherHw == INVALID_ID)
              continue;
            const Node *otherHwNode = adg.getNode(otherHw);
            if (otherHwNode &&
                getNodeAttrStr(otherHwNode, "pe_name") == peName) {
              peConflict = true;
              break;
            }
          }
          if (peConflict)
            continue;
        }
        double candScore =
            scorePlacement(swN, cand, state, dfg, adg, flattener, candidates);
        if (candScore > bestCandScore + 1e-9) {
          bestCandScore = candScore;
          topCandidates.clear();
          topCandidates.push_back(cand);
        } else if (std::abs(candScore - bestCandScore) <= 1e-9 &&
                   topCandidates.size() < 8) {
          topCandidates.push_back(cand);
        }
      }
      if (!topCandidates.empty()) {
        std::uniform_int_distribution<size_t> candDist(0, topCandidates.size() - 1);
        hwNew = topCandidates[candDist(rng)];
      }
      if (hwNew == INVALID_ID)
        continue;

      state.unmapNode(swN, dfg, adg);
      auto r = state.mapNode(swN, hwNew, dfg, adg);
      moveOk = (r == ActionResult::Success);
    }

    if (!moveOk) {
      state.restore(cp);
      continue;
    }

    double newCost = computeTotalCost(state, dfg, adg, flattener);
    double delta = oldCost - newCost; // Positive = improvement.

    if (delta > 0 ||
        std::uniform_real_distribution<double>(0.0, 1.0)(rng) <
            std::exp(delta / temperature)) {
      // Accept move.
      acceptCount++;
      if (newCost < bestCost) {
        bestCost = newCost;
        bestCheckpoint = state.save();
      }
    } else {
      state.restore(cp);
    }

    temperature *= coolingRate;
  }

  // Restore the best placement found.
  state.restore(bestCheckpoint);

  if (opts.verbose) {
    llvm::outs() << "  SA: " << acceptCount << " accepted moves, best cost "
                 << bestCost << "\n";
  }

  return true;
}

bool Mapper::runLocalRepair(
    MappingState &state, const MappingState::Checkpoint &baseCheckpoint,
    llvm::ArrayRef<IdIndex> failedEdges, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds, const Options &opts) {
  if (failedEdges.empty())
    return true;

  llvm::DenseMap<IdIndex, double> hotspotWeights;
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
    hotspotWeights[srcPort->parentNode] += weight;
    hotspotWeights[dstPort->parentNode] += weight;
  }

  std::vector<IdIndex> hotspots;
  hotspots.reserve(hotspotWeights.size());
  for (const auto &it : hotspotWeights) {
    const Node *node = dfg.getNode(it.first);
    if (!node || node->kind != Node::OperationNode || isMemoryOp(node))
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

  auto bestCheckpoint = state.save();
  unsigned bestRouted = countRoutedEdges(state, dfg, edgeKinds);
  size_t bestPathLen = computeTotalMappedPathLen(state);
  bool bestAllRouted = false;

  unsigned maxHotspots = std::min<unsigned>(hotspots.size(), 8);
  for (unsigned hotIdx = 0; hotIdx < maxHotspots; ++hotIdx) {
    state.restore(baseCheckpoint);
    IdIndex swNode = hotspots[hotIdx];
    IdIndex oldHw = state.swNodeToHwNode[swNode];
    auto candIt = candidates.find(swNode);
    if (oldHw == INVALID_ID || candIt == candidates.end())
      continue;

    llvm::SmallVector<std::pair<double, IdIndex>, 16> rankedCandidates;
    for (IdIndex candHw : candIt->second) {
      if (candHw == oldHw)
        continue;
      if (!state.hwNodeToSwNodes[candHw].empty())
        continue;

      const Node *candNode = adg.getNode(candHw);
      if (!candNode)
        continue;
      llvm::StringRef peName = getNodeAttrStr(candNode, "pe_name");
      if (!peName.empty() && isSpatialPEOccupied(state, adg, flattener, peName, candHw))
        continue;

      double candScore =
          scorePlacement(swNode, candHw, state, dfg, adg, flattener, candidates);
      rankedCandidates.push_back({-candScore, candHw});
    }

    llvm::stable_sort(rankedCandidates, [&](const auto &lhs, const auto &rhs) {
      if (lhs.first != rhs.first)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });

    unsigned maxMoves = std::min<unsigned>(rankedCandidates.size(), 6);
    for (unsigned moveIdx = 0; moveIdx < maxMoves; ++moveIdx) {
      state.restore(baseCheckpoint);
      state.unmapNode(swNode, dfg, adg);
      if (state.mapNode(swNode, rankedCandidates[moveIdx].second, dfg, adg) !=
          ActionResult::Success) {
        state.restore(baseCheckpoint);
        continue;
      }

      bool allRouted = runRouting(state, dfg, adg, edgeKinds, opts.seed);
      unsigned routed = countRoutedEdges(state, dfg, edgeKinds);
      size_t totalPathLen = computeTotalMappedPathLen(state);
      if (allRouted || routed > bestRouted ||
          (routed == bestRouted && totalPathLen < bestPathLen)) {
        bestCheckpoint = state.save();
        bestRouted = routed;
        bestPathLen = totalPathLen;
        bestAllRouted = allRouted;
        if (allRouted) {
          state.restore(bestCheckpoint);
          return true;
        }
      }
    }
  }

  state.restore(bestCheckpoint);
  return bestAllRouted;
}

} // namespace fcc

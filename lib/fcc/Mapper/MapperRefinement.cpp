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

} // namespace fcc

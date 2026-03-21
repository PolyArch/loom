#include "MapperLocalRepairInternal.h"

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

namespace fcc {

using namespace mapper_detail;

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

double estimatePortDistance(IdIndex lhsPortId, IdIndex rhsPortId,
                            const Graph &adg,
                            const ADGFlattener &flattener) {
  const Port *lhsPort = adg.getPort(lhsPortId);
  const Port *rhsPort = adg.getPort(rhsPortId);
  if (!lhsPort || !rhsPort || lhsPort->parentNode == INVALID_ID ||
      rhsPort->parentNode == INVALID_ID)
    return 0.0;
  const TopologyModel *topologyModel = getActiveTopologyModel();
  if (topologyModel) {
    return static_cast<double>(topologyModel->placementDistance(
        lhsPort->parentNode, rhsPort->parentNode));
  }
  return static_cast<double>(placementDistance(lhsPort->parentNode,
                                               rhsPort->parentNode,
                                               flattener));
}

bool Mapper::runExactRoutingRepair(
    MappingState &state, llvm::ArrayRef<IdIndex> failedEdges, const Graph &dfg,
    const Graph &adg, const ADGFlattener &flattener,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds, const Mapper::Options &opts,
    const CongestionState *congestion,
    llvm::ArrayRef<IdIndex> forcedPriorityEdges) {
  if (failedEdges.empty())
    return true;
  if (shouldStopForBudget("local repair"))
    return false;
  const auto &repairOpts = opts.localRepair;
  const auto &exactOpts = repairOpts.exact;
  std::vector<double> edgeWeights = buildEdgePlacementWeightCache(dfg);

  auto edgeWeight = [&](IdIndex edgeId) {
    return edgeId < static_cast<IdIndex>(edgeWeights.size())
               ? edgeWeights[edgeId]
               : classifyEdgePlacementWeight(dfg, edgeId);
  };
  llvm::DenseSet<IdIndex> priorityEdges;
  if (!forcedPriorityEdges.empty()) {
    for (IdIndex edgeId : forcedPriorityEdges)
      priorityEdges.insert(edgeId);
  } else {
    for (IdIndex edgeId : failedEdges) {
      if (edgeId < state.swEdgeToHwPaths.size() &&
          state.swEdgeToHwPaths[edgeId].empty()) {
        priorityEdges.insert(edgeId);
      }
    }
  }
  const bool prioritizePriorityBeforePenalty =
      !forcedPriorityEdges.empty() ||
      priorityEdges.size() <= exactOpts.priorityFirstFailedEdgeThreshold;

  auto betterResult = [&](unsigned lhsRouted, double lhsPenalty,
                          unsigned lhsPriorityRouted,
                          double lhsPriorityPenalty, size_t lhsPathLen,
                          unsigned rhsRouted, double rhsPenalty,
                          unsigned rhsPriorityRouted,
                          double rhsPriorityPenalty,
                          size_t rhsPathLen) {
    if (lhsRouted != rhsRouted)
      return lhsRouted > rhsRouted;
    if (prioritizePriorityBeforePenalty) {
      if (lhsPriorityRouted != rhsPriorityRouted)
        return lhsPriorityRouted > rhsPriorityRouted;
      if (std::abs(lhsPriorityPenalty - rhsPriorityPenalty) > 1e-9)
        return lhsPriorityPenalty < rhsPriorityPenalty;
      if (std::abs(lhsPenalty - rhsPenalty) > 1e-9)
        return lhsPenalty < rhsPenalty;
    } else {
      if (std::abs(lhsPenalty - rhsPenalty) > 1e-9)
        return lhsPenalty < rhsPenalty;
      if (lhsPriorityRouted != rhsPriorityRouted)
        return lhsPriorityRouted > rhsPriorityRouted;
      if (std::abs(lhsPriorityPenalty - rhsPriorityPenalty) > 1e-9)
        return lhsPriorityPenalty < rhsPriorityPenalty;
    }
    if (lhsPathLen != rhsPathLen)
      return lhsPathLen < rhsPathLen;
    return false;
  };

  auto globalBestCheckpoint = state.save();
  unsigned globalBestRouted = countRoutedEdges(state, dfg, edgeKinds);
  double globalBestPenalty = computeUnroutedPenalty(state, dfg, edgeKinds);
  unsigned globalBestPriorityRouted = 0;
  double globalBestPriorityPenalty = 0.0;
  for (IdIndex edgeId : priorityEdges) {
    if (edgeId < state.swEdgeToHwPaths.size() &&
        !state.swEdgeToHwPaths[edgeId].empty()) {
      ++globalBestPriorityRouted;
    } else {
      globalBestPriorityPenalty += edgeWeight(edgeId);
    }
  }
  size_t globalBestPathLen = computeTotalMappedPathLen(state);
  bool globalAllRouted = failedEdges.empty();
  auto emitBestSnapshot = [&](llvm::StringRef trigger) {
    auto checkpoint = state.save();
    state.restore(globalBestCheckpoint);
    maybeEmitProgressSnapshot(state, edgeKinds, trigger, opts);
    state.restore(checkpoint);
  };

  auto edgeOrderLess = [&](IdIndex lhs, IdIndex rhs) {
    bool lhsPriority = priorityEdges.contains(lhs);
    bool rhsPriority = priorityEdges.contains(rhs);
    if (lhsPriority != rhsPriority)
      return lhsPriority;
    double lhsWeight = edgeWeight(lhs);
    double rhsWeight = edgeWeight(rhs);
    if (std::abs(lhsWeight - rhsWeight) > 1e-9)
      return lhsWeight > rhsWeight;
    return lhs < rhs;
  };

  std::vector<IdIndex> pendingFailed(failedEdges.begin(), failedEdges.end());
  llvm::stable_sort(pendingFailed, edgeOrderLess);
  const unsigned maxNeighborhoodPasses =
      std::max(exactOpts.neighborhoodPassMin,
               std::min<unsigned>(exactOpts.neighborhoodPassCap,
                                  opts.selectiveRipupPasses + 1u));
  auto expandConflictNeighborhood =
      [&](llvm::SmallVectorImpl<IdIndex> &repairEdges, unsigned maxEdges) {
        llvm::DenseSet<IdIndex> seenEdges;
        for (IdIndex edgeId : repairEdges)
          seenEdges.insert(edgeId);

        for (size_t idx = 0;
             idx < repairEdges.size() && repairEdges.size() < maxEdges;
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
      };

  for (unsigned pass = 0; pass < maxNeighborhoodPasses && !globalAllRouted;
       ++pass) {
    if (shouldStopForBudget("exact routing repair"))
      break;
    auto currentFailed = collectUnroutedEdges(state, dfg, edgeKinds);
    if (currentFailed.empty()) {
      globalAllRouted = true;
      break;
    }
    llvm::stable_sort(currentFailed, edgeOrderLess);

    auto attemptExactNeighborhood =
        [&](llvm::StringRef label, llvm::ArrayRef<IdIndex> repairEdges,
            const MappingState::Checkpoint &baseCheckpoint) -> bool {
      if (repairEdges.empty())
        return false;

      unsigned baseLocalRouted = 0;
      double totalLocalWeight = 0.0;
      double baseLocalPenalty = 0.0;
      unsigned basePriorityRouted = 0;
      double totalPriorityWeight = 0.0;
      double basePriorityPenalty = 0.0;
      size_t baseLocalPathLen = 0;
      for (IdIndex edgeId : repairEdges) {
        double weight = edgeWeight(edgeId);
        totalLocalWeight += weight;
        bool isPriority = priorityEdges.contains(edgeId);
        if (isPriority)
          totalPriorityWeight += weight;
        if (edgeId < state.swEdgeToHwPaths.size() &&
            !state.swEdgeToHwPaths[edgeId].empty()) {
          ++baseLocalRouted;
          if (isPriority)
            ++basePriorityRouted;
          baseLocalPathLen += state.swEdgeToHwPaths[edgeId].size();
        } else {
          baseLocalPenalty += weight;
          if (isPriority)
            basePriorityPenalty += weight;
        }
      }

      if (opts.verbose) {
        llvm::outs() << "  Exact routing repair: " << label
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
      unsigned bestLocalPriorityRouted = basePriorityRouted;
      double bestLocalPriorityPenalty = basePriorityPenalty;
      size_t bestLocalPathLen = baseLocalPathLen;

      for (IdIndex edgeId : repairEdges)
        state.unmapEdge(edgeId, dfg, adg);

      const bool tightEndgame =
          currentFailed.size() <= exactOpts.tightFailedEdgeThreshold ||
          repairEdges.size() <= exactOpts.tightRepairEdgeThreshold;
      const bool microEndgame =
          currentFailed.size() <= exactOpts.microFailedEdgeThreshold &&
          repairEdges.size() <= exactOpts.microRepairEdgeThreshold;
      const double requestedDeadlineMs = std::max(
          microEndgame
              ? exactOpts.microDeadlineMs
              : (tightEndgame
                     ? exactOpts.tightDeadlineMs
                     : (repairEdges.size() <= exactOpts.mediumRepairEdgeThreshold
                            ? exactOpts.mediumDeadlineMs
                            : exactOpts.defaultDeadlineMs)),
          opts.cpSatTimeLimitSeconds *
              (microEndgame ? exactOpts.microDeadlineScale
                            : exactOpts.deadlineScale));
      const auto exactDeadline =
          std::chrono::steady_clock::now() +
          std::chrono::milliseconds(static_cast<int64_t>(std::max(
              1.0, clampDeadlineMsToRemainingBudget(requestedDeadlineMs))));

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
        const unsigned hopLimit =
            microEndgame
                ? exactOpts.microFirstHopLimit
                : (tightEndgame
                       ? exactOpts.tightFirstHopLimit
                       : exactOpts.defaultFirstHopLimit);
        if (choices.size() > hopLimit)
          choices.resize(hopLimit);
      };

      std::function<void(size_t)> searchEdges;
      searchEdges = [&](size_t index) {
        if (shouldStopForBudget("exact routing repair") ||
            std::chrono::steady_clock::now() > exactDeadline)
          return;
        if (index >= repairEdges.size()) {
          unsigned routed = 0;
          double penalty = 0.0;
          unsigned priorityRouted = 0;
          double priorityPenalty = 0.0;
          size_t pathLen = 0;
          for (IdIndex edgeId : repairEdges) {
            double weight = edgeWeight(edgeId);
            bool isPriority = priorityEdges.contains(edgeId);
            if (edgeId < state.swEdgeToHwPaths.size() &&
                !state.swEdgeToHwPaths[edgeId].empty()) {
              ++routed;
              pathLen += state.swEdgeToHwPaths[edgeId].size();
              if (isPriority)
                ++priorityRouted;
            } else {
              penalty += weight;
              if (isPriority)
                priorityPenalty += weight;
            }
          }
          if (betterResult(routed, penalty, priorityRouted, priorityPenalty,
                           pathLen, bestLocalRouted, bestLocalPenalty,
                           bestLocalPriorityRouted, bestLocalPriorityPenalty,
                           bestLocalPathLen)) {
            bestLocalCheckpoint = state.save();
            bestLocalRouted = routed;
            bestLocalPenalty = penalty;
            bestLocalPriorityRouted = priorityRouted;
            bestLocalPriorityPenalty = priorityPenalty;
            bestLocalPathLen = pathLen;
          }
          return;
        }

        IdIndex edgeId = repairEdges[index];
        const Edge *edge = dfg.getEdge(edgeId);
        if (!edge) {
          searchEdges(index + 1);
          return;
        }
        IdIndex srcHwPort =
            edge->srcPort < state.swPortToHwPort.size()
                ? state.swPortToHwPort[edge->srcPort]
                : INVALID_ID;
        IdIndex dstHwPort =
            edge->dstPort < state.swPortToHwPort.size()
                ? state.swPortToHwPort[edge->dstPort]
                : INVALID_ID;
        if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID) {
          searchEdges(index + 1);
          return;
        }

        llvm::DenseMap<IdIndex, double> localHistory;
        llvm::SmallVector<IdIndex, 8> firstHopChoices;
        auto it = connectivity.outToIn.find(srcHwPort);
        if (it != connectivity.outToIn.end())
          firstHopChoices.append(it->second.begin(), it->second.end());
        rankFirstHopChoices(dstHwPort, firstHopChoices);

        auto checkpoint = state.save();
        auto tryPath = [&](IdIndex forcedFirstHop) {
          auto path = findPath(srcHwPort, dstHwPort, edgeId, state, dfg, adg,
                               localHistory, forcedFirstHop, congestion);
          if (path.empty())
            return;
          state.restore(checkpoint);
          if (state.mapEdge(edgeId, path, dfg, adg) != ActionResult::Success)
            return;
          searchEdges(index + 1);
        };

        tryPath(INVALID_ID);
        for (IdIndex firstHop : firstHopChoices) {
          state.restore(checkpoint);
          tryPath(firstHop);
        }
        state.restore(checkpoint);
        searchEdges(index + 1);
      };

      searchEdges(0);
      state.restore(bestLocalCheckpoint);

      bool localAllRouted = bestLocalRouted == repairEdges.size();
      if (betterResult(bestLocalRouted, bestLocalPenalty,
                       bestLocalPriorityRouted, bestLocalPriorityPenalty,
                       bestLocalPathLen, globalBestRouted, globalBestPenalty,
                       globalBestPriorityRouted, globalBestPriorityPenalty,
                       globalBestPathLen)) {
        globalBestCheckpoint = state.save();
        globalBestRouted = bestLocalRouted;
        globalBestPenalty = bestLocalPenalty;
        globalBestPriorityRouted = bestLocalPriorityRouted;
        globalBestPriorityPenalty = bestLocalPriorityPenalty;
        globalBestPathLen = bestLocalPathLen;
        globalAllRouted = localAllRouted;
        emitBestSnapshot("exact-routing-repair");
      }
      if (localAllRouted)
        return true;

      state.restore(baseCheckpoint);
      return false;
    };

    bool improvedThisPass = false;
    for (IdIndex rootFailedEdge : pendingFailed) {
      if (shouldStopForBudget("exact routing repair"))
        break;
      auto currentFailed = collectUnroutedEdges(state, dfg, edgeKinds);
      if (currentFailed.empty()) {
        globalAllRouted = true;
        break;
      }

      llvm::SmallVector<IdIndex, 16> repairEdges;
      llvm::DenseSet<IdIndex> seenRepairEdges;
      if (seenRepairEdges.insert(rootFailedEdge).second)
        repairEdges.push_back(rootFailedEdge);
      const unsigned maxNeighborhoodEdges =
          std::max<unsigned>(repairEdges.size(),
                             repairOpts.focusedBlockerEdgeThreshold);
      expandConflictNeighborhood(repairEdges, maxNeighborhoodEdges);
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

      auto passCheckpoint = state.save();
      if (attemptExactNeighborhood(
              ("root edge " + std::to_string(rootFailedEdge)).c_str(),
              repairEdges, passCheckpoint)) {
        improvedThisPass = true;
        break;
      }
    }

    if (!improvedThisPass)
      break;
  }

  state.restore(globalBestCheckpoint);
  return globalAllRouted;
}

} // namespace fcc

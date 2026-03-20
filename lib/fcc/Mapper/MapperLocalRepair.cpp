#include "MapperInternal.h"
#include "MapperRoutingInternal.h"
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

namespace fcc {

using namespace mapper_detail;

namespace {

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

} // namespace

bool Mapper::runExactRoutingRepair(MappingState &state,
                                   llvm::ArrayRef<IdIndex> failedEdges,
                                   const Graph &dfg, const Graph &adg,
                                   const ADGFlattener &flattener,
                                   llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                   const Options &opts,
                                   const CongestionState *congestion,
                                   llvm::ArrayRef<IdIndex> forcedPriorityEdges) {
  if (failedEdges.empty())
    return true;
  if (shouldStopForBudget("local repair"))
    return false;
  const auto &repairOpts = opts.localRepair;
  const auto &exactOpts = repairOpts.exact;

  auto edgeWeight = [&](IdIndex edgeId) {
    return classifyEdgePlacementWeight(dfg, edgeId);
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
        state.unmapEdge(edgeId, adg);

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
                       : (repairEdges.size() <= exactOpts.mediumRepairEdgeThreshold
                              ? exactOpts.mediumFirstHopLimit
                              : (currentFailed.size() <=
                                         exactOpts.tightFailedEdgeThreshold
                                     ? exactOpts.smallFailedFirstHopLimit
                                     : exactOpts.defaultFirstHopLimit)));
        if (choices.size() > hopLimit)
          choices.resize(hopLimit);
      };

      std::function<void(llvm::SmallVectorImpl<IdIndex> &, unsigned, double,
                         unsigned, double, size_t)>
          searchNeighborhood;
      searchNeighborhood = [&](llvm::SmallVectorImpl<IdIndex> &remainingEdges,
                               unsigned currentRouted, double currentPenalty,
                               unsigned currentPriorityRouted,
                               double currentPriorityPenalty,
                               size_t currentPathLen) {
        if (shouldStopForBudget("exact routing repair"))
          return;
        if (std::chrono::steady_clock::now() > exactDeadline)
          return;
        if (betterResult(currentRouted, currentPenalty, currentPriorityRouted,
                         currentPriorityPenalty, currentPathLen,
                         bestLocalRouted, bestLocalPenalty,
                         bestLocalPriorityRouted, bestLocalPriorityPenalty,
                         bestLocalPathLen)) {
          bestLocalRouted = currentRouted;
          bestLocalPenalty = currentPenalty;
          bestLocalPriorityRouted = currentPriorityRouted;
          bestLocalPriorityPenalty = currentPriorityPenalty;
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
          llvm::SmallVector<llvm::SmallVector<IdIndex, 8>, 6> paths;
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
          if (routing_detail::isNonRoutingOutputPort(srcHwPort, adg)) {
            IdIndex lockedFirstHop =
                routing_detail::getLockedFirstHopForSource(edgeId, srcHwPort,
                                                           state);
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
              microEndgame
                  ? exactOpts.microCandidatePathLimit
                  : (tightEndgame
                         ? exactOpts.tightCandidatePathLimit
                         : (repairEdges.size() <=
                                    exactOpts.mediumRepairEdgeThreshold
                                ? exactOpts.mediumCandidatePathLimit
                                : (currentFailed.size() <=
                                           exactOpts.tightFailedEdgeThreshold
                                       ? exactOpts.smallFailedCandidatePathLimit
                                       : exactOpts.defaultCandidatePathLimit)));
          for (unsigned pathAttempt = 0;
               pathAttempt < maxCandidatePaths &&
               bundle.paths.size() < maxCandidatePaths;
               ++pathAttempt) {
            llvm::SmallVector<IdIndex, 8> bestPath;
            for (IdIndex firstHop : firstHopChoices) {
              auto path = findPath(srcHwPort, dstHwPort, edgeId, state, dfg,
                                   adg, localHistory, firstHop, congestion);
              if (path.empty())
                continue;
              bool duplicate = false;
              for (const auto &existing : bundle.paths) {
                if (existing.size() != path.size())
                  continue;
                if (std::equal(existing.begin(), existing.end(), path.begin())) {
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
                localHistory[portId] += exactOpts.localHistoryBump;
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
          bool bundlePriority = priorityEdges.contains(edgeId);
          bool nextPriority = priorityEdges.contains(nextBundle->edgeId);
          if (bundlePriority != nextPriority) {
            if (bundlePriority) {
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
          bool isPriority = priorityEdges.contains(edgeId);
          searchNeighborhood(
              remainingEdges, currentRouted + 1, currentPenalty - weight,
              currentPriorityRouted + (isPriority ? 1u : 0u),
              currentPriorityPenalty - (isPriority ? weight : 0.0),
              currentPathLen + path.size());
          state.unmapEdge(edgeId, adg);
        }

        searchNeighborhood(remainingEdges, currentRouted, currentPenalty,
                           currentPriorityRouted, currentPriorityPenalty,
                           currentPathLen);
        remainingEdges.insert(remainingEdges.begin() + nextIndex, savedEdgeId);
      };

      llvm::SmallVector<IdIndex, 16> remainingEdges(repairEdges.begin(),
                                                    repairEdges.end());
      searchNeighborhood(remainingEdges, 0, totalLocalWeight, 0,
                         totalPriorityWeight, 0);

      if (opts.verbose) {
        llvm::outs() << "  Exact routing repair result: " << label
                     << ", localRouted=" << bestLocalRouted << "/"
                     << repairEdges.size()
                     << ", localPenalty=" << bestLocalPenalty << "\n";
      }
      if (shouldStopForBudget("exact routing repair")) {
        state.restore(baseCheckpoint);
        return false;
      }

      state.restore(bestLocalCheckpoint);
      unsigned routed = countRoutedEdges(state, dfg, edgeKinds);
      double unroutedPenalty = computeUnroutedPenalty(state, dfg, edgeKinds);
      unsigned priorityRouted = 0;
      double priorityPenalty = 0.0;
      for (IdIndex edgeId : priorityEdges) {
        if (edgeId < state.swEdgeToHwPaths.size() &&
            !state.swEdgeToHwPaths[edgeId].empty()) {
          ++priorityRouted;
        } else {
          priorityPenalty += edgeWeight(edgeId);
        }
      }
      size_t totalPathLen = computeTotalMappedPathLen(state);
      bool improved = betterResult(
          routed, unroutedPenalty, priorityRouted, priorityPenalty,
          totalPathLen, globalBestRouted, globalBestPenalty,
          globalBestPriorityRouted, globalBestPriorityPenalty,
          globalBestPathLen);
      if (improved) {
        if (opts.verbose) {
          llvm::outs() << "  Exact routing repair accepted: routed " << routed
                       << "/" << dfg.edges.size()
                       << ", unroutedPenalty=" << unroutedPenalty << "\n";
        }
        globalBestCheckpoint = state.save();
        globalBestRouted = routed;
        globalBestPenalty = unroutedPenalty;
        globalBestPriorityRouted = priorityRouted;
        globalBestPriorityPenalty = priorityPenalty;
        globalBestPathLen = totalPathLen;
        globalAllRouted = collectUnroutedEdges(state, dfg, edgeKinds).empty();
        emitBestSnapshot("exact-routing-repair");
        return true;
      }

      state.restore(baseCheckpoint);
      return false;
    };

    bool improvedThisPass = false;
    if (shouldStopForBudget("exact routing repair"))
      break;

    if (pass == 0 && !forcedPriorityEdges.empty() &&
        failedEdges.size() > currentFailed.size() &&
        failedEdges.size() <= repairOpts.originalFailedEscalationThreshold) {
      auto forcedCheckpoint = state.save();
      if (attemptExactNeighborhood("priority neighborhood", failedEdges,
                                   forcedCheckpoint)) {
        improvedThisPass = true;
        if (globalAllRouted)
          break;
        state.restore(globalBestCheckpoint);
      }
    }

    if (currentFailed.size() <= repairOpts.smallFailedEdgeThreshold) {
      auto combinedCheckpoint = state.save();
      llvm::DenseSet<IdIndex> anchorSrcHwPorts;
      llvm::DenseSet<IdIndex> anchorDstHwPorts;
      llvm::DenseSet<IdIndex> anchorSrcSwNodes;
      llvm::DenseSet<IdIndex> anchorDstSwNodes;
      llvm::DenseSet<IdIndex> anchorSrcHwNodes;
      llvm::DenseSet<IdIndex> anchorDstHwNodes;
      llvm::SmallVector<std::pair<int, int>, 8> anchorPositions;

      auto recordAnchorNode = [&](IdIndex hwNodeId) {
        if (hwNodeId == INVALID_ID)
          return;
        auto [row, col] = flattener.getNodeGridPos(hwNodeId);
        if (row >= 0 && col >= 0)
          anchorPositions.push_back({row, col});
      };

      for (IdIndex edgeId : currentFailed) {
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

      auto touchesCombinedNeighborhood = [&](llvm::ArrayRef<IdIndex> path) {
        for (IdIndex portId : path) {
          const Port *port = adg.getPort(portId);
          if (!port || port->parentNode == INVALID_ID)
            continue;
          auto [row, col] = flattener.getNodeGridPos(port->parentNode);
          if (row < 0 || col < 0)
            continue;
          for (const auto &pos : anchorPositions) {
            if (std::abs(row - pos.first) + std::abs(col - pos.second) <= 1)
              return true;
          }
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
        IdIndex srcSwNode =
            srcSwPort->parentNode != INVALID_ID ? srcSwPort->parentNode
                                                : INVALID_ID;
        IdIndex dstSwNode =
            dstSwPort->parentNode != INVALID_ID ? dstSwPort->parentNode
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
        if (std::find(currentFailed.begin(), currentFailed.end(), edgeId) !=
            currentFailed.end())
          score += 120;
        if (srcHwPort != INVALID_ID && anchorSrcHwPorts.contains(srcHwPort))
          score += 60;
        if (dstHwPort != INVALID_ID && anchorDstHwPorts.contains(dstHwPort))
          score += 50;
        if (srcSwNode != INVALID_ID &&
            (anchorSrcSwNodes.contains(srcSwNode) ||
             anchorDstSwNodes.contains(srcSwNode)))
          score += 24;
        if (dstSwNode != INVALID_ID &&
            (anchorSrcSwNodes.contains(dstSwNode) ||
             anchorDstSwNodes.contains(dstSwNode)))
          score += 24;
        if (srcHwNode != INVALID_ID &&
            (anchorSrcHwNodes.contains(srcHwNode) ||
             anchorDstHwNodes.contains(srcHwNode)))
          score += 18;
        if (dstHwNode != INVALID_ID &&
            (anchorSrcHwNodes.contains(dstHwNode) ||
             anchorDstHwNodes.contains(dstHwNode)))
          score += 18;
        if (edgeId < state.swEdgeToHwPaths.size() &&
            !state.swEdgeToHwPaths[edgeId].empty() &&
            touchesCombinedNeighborhood(state.swEdgeToHwPaths[edgeId]))
          score += 12;
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
      const unsigned maxRepairEdges = 14u;
      for (const auto &entry : rankedEdges) {
        if (repairEdges.size() >= maxRepairEdges)
          break;
        if (selectedEdges.insert(entry.second).second)
          repairEdges.push_back(entry.second);
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

      if (attemptExactNeighborhood("joint", repairEdges, combinedCheckpoint)) {
        improvedThisPass = true;
        if (globalAllRouted)
          break;
      }
      state.restore(globalBestCheckpoint);
    }

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

      if (attemptExactNeighborhood(
              ("root edge " + std::to_string(rootFailedEdge)).c_str(),
              repairEdges, baseCheckpoint)) {
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

bool Mapper::runLocalRepair(
    MappingState &state, const MappingState::Checkpoint &baseCheckpoint,
    llvm::ArrayRef<IdIndex> failedEdges, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    std::vector<TechMappedEdgeKind> &edgeKinds, const Options &opts,
    const CongestionState *congestion, unsigned recursionDepth) {
  if (failedEdges.empty())
    return true;
  if (shouldStopForBudget("local repair"))
    return false;
  const auto &repairOpts = opts.localRepair;
  const unsigned maxRepairRecursionDepth =
      failedEdges.size() <= repairOpts.focusedTargetEdgeThreshold
          ? repairOpts.microRecursionDepthLimit
          : repairOpts.defaultRecursionDepthLimit;

  if (opts.verbose) {
    llvm::outs() << "  Local repair start: depth=" << recursionDepth
                 << ", failedEdges:";
    for (IdIndex edgeId : failedEdges)
      llvm::outs() << " " << edgeId;
    llvm::outs() << "\n";
  }

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
  llvm::DenseSet<IdIndex> repairPriorityEdges;
  for (IdIndex edgeId : failedEdges)
    repairPriorityEdges.insert(edgeId);
  auto computePriorityMetrics =
      [&](const MappingState &candidateState) -> std::pair<unsigned, double> {
    unsigned routed = 0;
    double penalty = 0.0;
    for (IdIndex edgeId : repairPriorityEdges) {
      double weight = classifyEdgePlacementWeight(dfg, edgeId);
      if (edgeId < candidateState.swEdgeToHwPaths.size() &&
          !candidateState.swEdgeToHwPaths[edgeId].empty()) {
        ++routed;
      } else {
        penalty += weight;
      }
    }
    return {routed, penalty};
  };
  auto [bestPriorityRouted, bestPriorityPenalty] = computePriorityMetrics(state);
  auto computeRepairabilityScore = [&](const MappingState &candidateState) {
    std::vector<IdIndex> currentFailed =
        collectUnroutedEdges(candidateState, dfg, edgeKinds);
    if (currentFailed.empty())
      return 0.0;

    MappingState probeState = candidateState;
    probeState.clearRoutes(adg);
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
      double weight = classifyEdgePlacementWeight(dfg, edgeId);
      auto freePath = findPath(srcHwPort, dstHwPort, edgeId, probeState, dfg,
                               adg, emptyHistory, INVALID_ID, congestion);
      score += weight *
               static_cast<double>(freePath.empty()
                                       ? repairOpts.freePathMissingPenalty
                                       : freePath.size());
    }
    return score;
  };
  double bestRepairabilityScore = computeRepairabilityScore(state);
  auto shouldEscalateToCPSat = [&]() {
    return opts.enableCPSat && !bestAllRouted &&
           bestFailedEdges.size() <= repairOpts.cpSatEscalationFailedEdgeThreshold;
  };
  Options repairRoutingOpts = opts;
  if (repairRoutingOpts.negotiatedRoutingPasses > 0)
    repairRoutingOpts.negotiatedRoutingPasses =
        std::min<unsigned>(repairRoutingOpts.negotiatedRoutingPasses,
                           repairOpts.repairNegotiatedRoutingPassCap);
  auto rerouteRepairState = [&](MappingState &repairState) {
    if (shouldStopForBudget("local repair"))
      return false;
    if (repairRoutingOpts.negotiatedRoutingPasses > 0)
      return runNegotiatedRouting(repairState, dfg, adg, edgeKinds,
                                  repairRoutingOpts);
    return runRouting(repairState, dfg, adg, edgeKinds, repairRoutingOpts);
  };
  auto buildFocusedRepairNeighborhood =
      [&](llvm::ArrayRef<IdIndex> seedEdges,
          unsigned maxEdges) -> llvm::SmallVector<IdIndex, 24> {
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
    llvm::SmallVector<std::pair<int, int>, 8> anchorPositions;

    auto recordAnchorNode = [&](IdIndex hwNodeId) {
      if (hwNodeId == INVALID_ID)
        return;
      auto [row, col] = flattener.getNodeGridPos(hwNodeId);
      if (row >= 0 && col >= 0)
        anchorPositions.push_back({row, col});
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
    probeState.clearRoutes(adg);
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
      auto freeSpacePath = findPath(srcHwPort, dstHwPort, edgeId, probeState,
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
        auto [row, col] = flattener.getNodeGridPos(port->parentNode);
        if (row < 0 || col < 0)
          continue;
        for (const auto &pos : anchorPositions) {
          if (std::abs(row - pos.first) + std::abs(col - pos.second) <= 1)
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

      IdIndex srcSwNode =
          srcSwPort->parentNode != INVALID_ID ? srcSwPort->parentNode
                                              : INVALID_ID;
      IdIndex dstSwNode =
          dstSwPort->parentNode != INVALID_ID ? dstSwPort->parentNode
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
      double lhsWeight = classifyEdgePlacementWeight(dfg, lhs);
      double rhsWeight = classifyEdgePlacementWeight(dfg, rhs);
      if (std::abs(lhsWeight - rhsWeight) > 1e-9)
        return lhsWeight > rhsWeight;
      return lhs < rhs;
    });
    return repairEdges;
  };

  auto updateBest = [&](bool allRouted) -> bool {
    unsigned routed = countRoutedEdges(state, dfg, edgeKinds);
    double unroutedPenalty = computeUnroutedPenalty(state, dfg, edgeKinds);
    size_t totalPathLen = computeTotalMappedPathLen(state);
    double placementCost = computeTotalCost(state, dfg, adg, flattener);
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

    state.clearRoutes(adg);
    bestPlacementCheckpoint = state.save();
    state.restore(bestCheckpoint);
    maybeEmitProgressSnapshot(state, edgeKinds, "local-repair-update", opts);
    return true;
  };

  if (opts.enableCPSat &&
      recursionDepth >= repairOpts.earlyCPSatRecursionDepthThreshold &&
      failedEdges.size() <= repairOpts.earlyCPSatFailedEdgeThreshold) {
    state.restore(bestPlacementCheckpoint);
    Options earlyCpSatOpts = opts;
    earlyCpSatOpts.cpSatTimeLimitSeconds =
        std::max(earlyCpSatOpts.cpSatTimeLimitSeconds,
                 repairOpts.earlyCPSatMinTime);
    earlyCpSatOpts.cpSatNeighborhoodNodeLimit =
        std::max<unsigned>(earlyCpSatOpts.cpSatNeighborhoodNodeLimit,
                           repairOpts.earlyCPSatNeighborhoodLimit);
    earlyCpSatOpts.placementMoveRadius =
        std::max<unsigned>(earlyCpSatOpts.placementMoveRadius,
                           repairOpts.earlyCPSatMoveRadius);
    bool allRouted = runCPSatNeighborhoodRepair(
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
    if (shouldStopForBudget("local repair"))
      break;
    bool improvedThisRound = false;
    const unsigned repairRadius =
        opts.placementMoveRadius == 0
            ? 0
            : opts.placementMoveRadius + round * repairOpts.repairRadiusStep +
                  repairOpts.repairRadiusBias;

    for (unsigned hotIdx = 0; hotIdx < maxHotspots; ++hotIdx) {
      if (shouldStopForBudget("local repair"))
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
                             candidates))
          continue;
        double candScore = scorePlacement(swNode, candHw, state, dfg, adg,
                                          flattener, candidates);
        double failedEdgeDelta =
            evaluateFailedEdgeDelta({std::make_pair(swNode, candHw)});
        double repairScore = failedEdgeDelta * repairOpts.failedEdgeDeltaScoreWeight +
                             candScore * repairOpts.candidateScoreWeight;
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
        if (shouldStopForBudget("local repair"))
          break;
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
                          flattener, candidates))
          continue;
        double swapScore =
            evaluateFailedEdgeDelta({std::make_pair(swNode, otherHw),
                                     std::make_pair(otherSw, oldHw)}) *
                repairOpts.failedEdgeDeltaScoreWeight +
            scorePlacement(swNode, otherHw, state, dfg, adg, flattener,
                           candidates) *
                repairOpts.candidateScoreWeight +
            scorePlacement(otherSw, oldHw, state, dfg, adg, flattener,
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
        if (shouldStopForBudget("local repair"))
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
        if (!bindMappedNodePorts(swNode, state, dfg, adg) ||
            !bindMappedNodePorts(otherSw, state, dfg, adg)) {
          state.restore(bestPlacementCheckpoint);
          continue;
        }
        bindMemrefSentinels(state, dfg, adg);
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
    if (shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
    state.restore(bestPlacementCheckpoint);
    Options cpSatRepairOpts = opts;
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
    bool allRouted = runCPSatNeighborhoodRepair(
        state, bestPlacementCheckpoint, bestFailedEdges, dfg, adg, flattener,
        candidates, edgeKinds, cpSatRepairOpts);
    if (updateBest(allRouted) && allRouted) {
      state.restore(bestCheckpoint);
      return true;
    }
  }

  if (!bestAllRouted &&
      bestFailedEdges.size() <= repairOpts.cpSatSmallFailedThreshold) {
    if (shouldStopForBudget("local repair")) {
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
        llvm::StringRef lhsOp =
            getNodeAttrStr(dfg.getNode(lhs), "op_name");
        llvm::StringRef rhsOp =
            getNodeAttrStr(dfg.getNode(rhs), "op_name");
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
                1.0, clampDeadlineMsToRemainingBudget(8000.0))));
        bool stopExactSearch = false;
        llvm::DenseSet<IdIndex> distinctMemoryCandidates;
        unsigned memoryNodeCount = 0;
        for (IdIndex swNode : memoryResponseCluster) {
          const Node *node = dfg.getNode(swNode);
          if (!node || !isSoftwareMemoryInterfaceOp(getNodeAttrStr(node, "op_name")))
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
          if (stopExactSearch || shouldStopForBudget("local repair") ||
              std::chrono::steady_clock::now() > exactDeadline)
            return;
          if (depth >= memoryResponseCluster.size()) {
            rebindScalarInputSentinels(state, dfg, adg, flattener);
            bindMemrefSentinels(state, dfg, adg);
            classifyTemporalRegisterEdges(state, dfg, adg, flattener,
                                          edgeKinds);
            auto responseRepairEdges = buildFocusedRepairNeighborhood(
                bestFailedEdges, repairOpts.earlyCPSatNeighborhoodLimit);
            if (responseRepairEdges.size() < bestFailedEdges.size()) {
              responseRepairEdges.assign(bestFailedEdges.begin(),
                                         bestFailedEdges.end());
            }
            bool allRouted = runExactRoutingRepair(
                state, responseRepairEdges, dfg, adg, flattener, edgeKinds,
                repairRoutingOpts, congestion, bestFailedEdges);
            if (updateBest(allRouted) && allRouted)
              stopExactSearch = true;
            return;
          }

          IdIndex swNode = memoryResponseCluster[depth];
          const Node *swNodeDef = dfg.getNode(swNode);
          const bool isMemoryNode =
              swNodeDef &&
              isSoftwareMemoryInterfaceOp(getNodeAttrStr(swNodeDef, "op_name"));
          auto checkpoint = state.save();
          for (IdIndex candHw : exactDomains.lookup(swNode)) {
            if (enforceDistinctMemory && isMemoryNode &&
                usedDistinctMemoryHws.contains(candHw)) {
              continue;
            }
            state.restore(checkpoint);
            if (enforceDistinctMemory && isMemoryNode)
              usedDistinctMemoryHws.insert(candHw);
            if (state.mapNode(swNode, candHw, dfg, adg) !=
                ActionResult::Success) {
              if (enforceDistinctMemory && isMemoryNode)
                usedDistinctMemoryHws.erase(candHw);
              continue;
            }
            if (!bindMappedNodePorts(swNode, state, dfg, adg))
            {
              if (enforceDistinctMemory && isMemoryNode)
                usedDistinctMemoryHws.erase(candHw);
              continue;
            }
            enumerateMemoryResponseCluster(depth + 1, enforceDistinctMemory);
            if (enforceDistinctMemory && isMemoryNode)
              usedDistinctMemoryHws.erase(candHw);
            if (stopExactSearch)
              return;
          }
          state.restore(checkpoint);
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
    if (shouldStopForBudget("local repair")) {
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
    auto expandToAllSoftwareMemoryNodes = [&](std::vector<IdIndex> &nodes,
                                              llvm::DenseSet<IdIndex> &seen) {
      if (nodes.empty())
        return;
      std::vector<IdIndex> allMemoryNodes;
      allMemoryNodes.reserve(nodes.size());
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
          double candScore = scorePlacement(swNode, candHw, state, dfg, adg,
                                            flattener, candidates);
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
                1.0, clampDeadlineMsToRemainingBudget(6000.0))));
        bool stopExactSearch = false;
        llvm::DenseSet<IdIndex> distinctMemoryCandidates;
        unsigned memoryNodeCount = 0;
        for (IdIndex swNode : memoryFocusNodes) {
          const Node *node = dfg.getNode(swNode);
          if (!node || !isSoftwareMemoryInterfaceOp(getNodeAttrStr(node, "op_name")))
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
          if (stopExactSearch || shouldStopForBudget("local repair") ||
              std::chrono::steady_clock::now() > exactDeadline)
            return;
          if (depth >= memoryFocusNodes.size()) {
            rebindScalarInputSentinels(state, dfg, adg, flattener);
            bindMemrefSentinels(state, dfg, adg);
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
          auto checkpoint = state.save();
          for (IdIndex candHw : exactDomains.lookup(swNode)) {
            if (enforceDistinctMemory && isMemoryNode &&
                usedDistinctMemoryHws.contains(candHw)) {
              continue;
            }
            state.restore(checkpoint);
            if (enforceDistinctMemory && isMemoryNode)
              usedDistinctMemoryHws.insert(candHw);
            if (state.mapNode(swNode, candHw, dfg, adg) !=
                ActionResult::Success) {
              if (enforceDistinctMemory && isMemoryNode)
                usedDistinctMemoryHws.erase(candHw);
              continue;
            }
            if (!bindMappedNodePorts(swNode, state, dfg, adg))
            {
              if (enforceDistinctMemory && isMemoryNode)
                usedDistinctMemoryHws.erase(candHw);
              continue;
            }
            enumerateMemoryNeighborhood(depth + 1, enforceDistinctMemory);
            if (enforceDistinctMemory && isMemoryNode)
              usedDistinctMemoryHws.erase(candHw);
            if (stopExactSearch)
              return;
          }
          state.restore(checkpoint);
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

  if (!bestAllRouted &&
      bestFailedEdges.size() <= repairOpts.cpSatEscalationFailedEdgeThreshold) {
    if (shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
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
          focusWeights[otherPort->parentNode] +=
              weight * repairOpts.focusNeighborhoodWeightScale;
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
      if (shouldStopForBudget("local repair"))
        break;
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
      const unsigned exactRadius =
          std::max(repairOpts.exactNeighborhoodRadius,
                   opts.placementMoveRadius + repairOpts.repairRadiusBias);
      const unsigned candidateLimit = largerExactNeighborhood ? 3u : 4u;
      const size_t searchSpaceLimit =
          largerExactNeighborhood
              ? (bestFailedEdges.size() <= repairOpts.cpSatSmallFailedThreshold
                     ? repairOpts.exactNeighborhoodSearchSpaceTightCap
                     : repairOpts.exactNeighborhoodSearchSpaceDefaultCap)
              : repairOpts.exactNeighborhoodSearchSpaceTightCap;

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
          std::chrono::milliseconds(static_cast<int64_t>(std::max(
              1.0, clampDeadlineMsToRemainingBudget(
                       largerExactNeighborhood ? 5000.0 : 8000.0))));
      bool stopExactSearch = false;

      std::function<void(unsigned)> enumerateNeighborhood;
      enumerateNeighborhood = [&](unsigned depth) {
        if (stopExactSearch || shouldStopForBudget("local repair") ||
            std::chrono::steady_clock::now() > exactDeadline)
          return;
        if (depth >= focusNodes.size()) {
          rebindScalarInputSentinels(state, dfg, adg, flattener);
          bindMemrefSentinels(state, dfg, adg);
          classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
          bool allRouted = rerouteRepairState(state);
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

  if (!bestAllRouted && bestFailedEdges.size() <= 8 &&
      bestFailedEdges.size() > 3) {
    if (shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
    state.restore(bestCheckpoint);
    bool allRouted = runExactRoutingRepair(state, bestFailedEdges, dfg, adg,
                                           flattener, edgeKinds, opts,
                                           congestion);
    if (updateBest(allRouted) && allRouted) {
      state.restore(bestCheckpoint);
      return true;
    }
  }

  if (!bestAllRouted &&
      bestFailedEdges.size() <= repairOpts.targetFocusedFailedEdgeThreshold) {
    if (shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
    state.restore(bestCheckpoint);

    auto edgeOrderLess = [&](IdIndex lhs, IdIndex rhs) {
      double lhsWeight = classifyEdgePlacementWeight(dfg, lhs);
      double rhsWeight = classifyEdgePlacementWeight(dfg, rhs);
      if (std::abs(lhsWeight - rhsWeight) > 1e-9)
        return lhsWeight > rhsWeight;
      return lhs < rhs;
    };

    std::vector<IdIndex> targetEdges(bestFailedEdges.begin(), bestFailedEdges.end());
    llvm::stable_sort(targetEdges, edgeOrderLess);

    struct ReservedPath {
      IdIndex edgeId = INVALID_ID;
      llvm::SmallVector<IdIndex, 8> path;
    };

    llvm::DenseMap<IdIndex, llvm::SmallVector<llvm::SmallVector<IdIndex, 8>, 8>>
        candidatePathsByEdge;
    bool candidateGenerationFailed = false;
    for (IdIndex edgeId : targetEdges) {
      if (shouldStopForBudget("local repair")) {
        candidateGenerationFailed = true;
        break;
      }
      MappingState probeState = state;
      probeState.clearRoutes(adg);
      llvm::DenseMap<IdIndex, double> localHistory;
      auto &candidatesForEdge = candidatePathsByEdge[edgeId];
      for (unsigned attempt = 0;
           attempt < repairOpts.exact.microCandidatePathLimit &&
           candidatesForEdge.size() < repairOpts.exact.tightFirstHopLimit;
           ++attempt) {
        const Edge *edge = dfg.getEdge(edgeId);
        if (!edge) {
          candidateGenerationFailed = true;
          break;
        }
        IdIndex srcHwPort =
            edge->srcPort < probeState.swPortToHwPort.size()
                ? probeState.swPortToHwPort[edge->srcPort]
                : INVALID_ID;
        IdIndex dstHwPort =
            edge->dstPort < probeState.swPortToHwPort.size()
                ? probeState.swPortToHwPort[edge->dstPort]
                : INVALID_ID;
        auto path = findPath(srcHwPort, dstHwPort, edgeId, probeState, dfg, adg,
                             localHistory, INVALID_ID, nullptr);
        if (path.empty())
          break;
        bool duplicate = false;
        for (const auto &existing : candidatesForEdge) {
          if (existing.size() == path.size() &&
              std::equal(existing.begin(), existing.end(), path.begin())) {
            duplicate = true;
            break;
          }
        }
        if (duplicate)
          break;
        candidatesForEdge.push_back(path);
        for (IdIndex portId : path) {
          if (isRoutingCrossbarOutputPortForRepair(portId, adg))
            localHistory[portId] += repairOpts.focusedLocalHistoryBump;
        }
      }
      if (candidateGenerationFailed || candidatesForEdge.empty()) {
        candidateGenerationFailed = true;
        break;
      }
    }

    llvm::outs() << "  Reserved failed-edge search:";
    for (IdIndex edgeId : targetEdges)
      llvm::outs() << " edge" << edgeId << "="
                   << candidatePathsByEdge.lookup(edgeId).size();
    llvm::outs() << ", generation="
                 << (candidateGenerationFailed ? "failed" : "ok") << "\n";

    if (!candidateGenerationFailed) {
      MappingState reservedState = state;
      reservedState.clearRoutes(adg);
      llvm::SmallVector<ReservedPath, 4> currentReserved;
      struct ReservedCandidateSet {
        llvm::SmallVector<ReservedPath, 4> paths;
        unsigned count = 0;
        size_t totalPathLen = 0;
      };
      std::vector<ReservedCandidateSet> reservedCandidates;

      std::function<void(size_t)> searchReservedPaths;
      searchReservedPaths = [&](size_t index) {
        if (shouldStopForBudget("local repair"))
          return;
        if (index >= targetEdges.size()) {
          ReservedCandidateSet candidate;
          candidate.count = currentReserved.size();
          candidate.paths = currentReserved;
          for (const auto &entry : currentReserved)
            candidate.totalPathLen += entry.path.size();
          reservedCandidates.push_back(std::move(candidate));
          return;
        }

        IdIndex edgeId = targetEdges[index];
        auto checkpoint = reservedState.save();
        for (const auto &path : candidatePathsByEdge.lookup(edgeId)) {
          if (shouldStopForBudget("local repair"))
            return;
          reservedState.restore(checkpoint);
          if (reservedState.mapEdge(edgeId, path, dfg, adg) !=
              ActionResult::Success)
            continue;
          currentReserved.push_back({edgeId, path});
          searchReservedPaths(index + 1);
          currentReserved.pop_back();
        }
        reservedState.restore(checkpoint);
        searchReservedPaths(index + 1);
      };

      searchReservedPaths(0);

      llvm::stable_sort(reservedCandidates,
                        [&](const ReservedCandidateSet &lhs,
                            const ReservedCandidateSet &rhs) {
                          if (lhs.count != rhs.count)
                            return lhs.count > rhs.count;
                          if (lhs.totalPathLen != rhs.totalPathLen)
                            return lhs.totalPathLen < rhs.totalPathLen;
                          if (lhs.paths.size() != rhs.paths.size())
                            return lhs.paths.size() > rhs.paths.size();
                          for (size_t idx = 0;
                               idx < lhs.paths.size() && idx < rhs.paths.size();
                               ++idx) {
                            if (lhs.paths[idx].edgeId != rhs.paths[idx].edgeId)
                              return lhs.paths[idx].edgeId < rhs.paths[idx].edgeId;
                            if (lhs.paths[idx].path.size() !=
                                rhs.paths[idx].path.size()) {
                              return lhs.paths[idx].path.size() <
                                     rhs.paths[idx].path.size();
                            }
                          }
                          return false;
                        });
      if (reservedCandidates.size() >
          repairOpts.exact.mediumCandidatePathLimit)
        reservedCandidates.resize(repairOpts.exact.mediumCandidatePathLimit);

      unsigned bestReservedCount =
          reservedCandidates.empty() ? 0u : reservedCandidates.front().count;
      llvm::outs() << "  Reserved failed-edge compatibility: bestReserved="
                   << bestReservedCount << "\n";

      if (bestReservedCount >= std::min<size_t>(2u, targetEdges.size())) {
        for (size_t candidateIdx = 0; candidateIdx < reservedCandidates.size();
             ++candidateIdx) {
          if (shouldStopForBudget("local repair"))
            break;
          const auto &reservedChoice = reservedCandidates[candidateIdx];
          if (reservedChoice.count < bestReservedCount)
            break;
          state.restore(bestCheckpoint);
          llvm::SmallVector<IdIndex, 16> blockerEdges;
          llvm::DenseSet<IdIndex> seenBlockers;
          for (IdIndex edgeId : targetEdges) {
            seenBlockers.insert(edgeId);
            blockerEdges.push_back(edgeId);
          }
          for (const auto &entry : reservedChoice.paths) {
            for (IdIndex portId : entry.path) {
              if (portId >= state.portToUsingEdges.size())
                continue;
              for (IdIndex otherEdgeId : state.portToUsingEdges[portId]) {
                if (seenBlockers.insert(otherEdgeId).second)
                  blockerEdges.push_back(otherEdgeId);
              }
            }
          }

          if (blockerEdges.size() > repairOpts.singleRipupMaxEdges)
            continue;

          if (targetEdges.size() <= repairOpts.focusedTargetEdgeThreshold &&
              blockerEdges.size() <= repairOpts.focusedBlockerEdgeThreshold) {
            state.restore(bestCheckpoint);
            Options focusedExactOpts = opts;
            focusedExactOpts.cpSatTimeLimitSeconds =
                std::max(focusedExactOpts.cpSatTimeLimitSeconds,
                         repairOpts.focusedTargetMinTime);
            bool allRouted = runExactRoutingRepair(state, blockerEdges, dfg, adg,
                                                   flattener, edgeKinds,
                                                   focusedExactOpts,
                                                   congestion);
            if (updateBest(allRouted) && allRouted) {
              state.restore(bestCheckpoint);
              return true;
            }
          }

          llvm::outs() << "  Reserved failed-edge paths: targets="
                       << targetEdges.size()
                       << ", blockerEdges=" << blockerEdges.size()
                       << ", candidate=" << (candidateIdx + 1) << "/"
                       << reservedCandidates.size() << "\n";
          if (opts.verbose) {
            llvm::outs() << "    reserved targets:";
            for (const auto &entry : reservedChoice.paths)
              llvm::outs() << " edge" << entry.edgeId << "(len="
                           << entry.path.size() << ")";
            llvm::outs() << "\n";
          }
          auto attemptCheckpoint = state.save();
          for (IdIndex edgeId : blockerEdges) {
            if (edgeId < state.swEdgeToHwPaths.size())
              state.unmapEdge(edgeId, adg);
          }
          bool mappingFailed = false;
          llvm::SmallVector<IdIndex, 16> rerouteEdges;
          llvm::DenseSet<IdIndex> seenReroute;
          for (const auto &entry : reservedChoice.paths) {
            if (state.mapEdge(entry.edgeId, entry.path, dfg, adg) !=
                ActionResult::Success) {
              mappingFailed = true;
              break;
            }
          }
          if (!mappingFailed) {
            auto postTargetFailed = collectUnroutedEdges(state, dfg, edgeKinds);
            if (opts.verbose) {
              llvm::outs() << "  Reserved failed-edge after target remap:";
              for (IdIndex edgeId : postTargetFailed)
                llvm::outs() << " " << edgeId;
              llvm::outs() << "\n";
            }
            if (recursionDepth < maxRepairRecursionDepth &&
                !postTargetFailed.empty() &&
                postTargetFailed.size() <= failedEdges.size() &&
                postTargetFailed.size() <=
                    repairOpts.residualRepairFailedEdgeThreshold) {
              bool postTargetChanged =
                  postTargetFailed.size() != failedEdges.size() ||
                  !std::equal(postTargetFailed.begin(), postTargetFailed.end(),
                              failedEdges.begin(), failedEdges.end());
              if (postTargetChanged) {
                auto recursiveRoutedCheckpoint = state.save();
                state.clearRoutes(adg);
                auto recursivePlacementCheckpoint = state.save();
                state.restore(recursiveRoutedCheckpoint);
                bool recursiveAllRouted =
                    runLocalRepair(state, recursivePlacementCheckpoint,
                                   postTargetFailed, dfg, adg, flattener,
                                   candidates, edgeKinds, opts, congestion,
                                   recursionDepth + 1);
                if (updateBest(recursiveAllRouted) && recursiveAllRouted) {
                  state.restore(bestCheckpoint);
                  return true;
                }
                state.restore(recursiveRoutedCheckpoint);
              }
            }
            for (IdIndex edgeId : blockerEdges) {
              if (std::find(targetEdges.begin(), targetEdges.end(), edgeId) !=
                  targetEdges.end())
                continue;
              if (seenReroute.insert(edgeId).second)
                rerouteEdges.push_back(edgeId);
            }
            bool allRouted = rerouteEdges.empty()
                                 ? collectUnroutedEdges(state, dfg, edgeKinds)
                                       .empty()
                                 : runExactRoutingRepair(state, rerouteEdges, dfg,
                                                         adg, flattener,
                                                         edgeKinds, opts,
                                                         congestion);
            unsigned targetStillRouted = 0;
            for (const auto &entry : reservedChoice.paths) {
              if (entry.edgeId < state.swEdgeToHwPaths.size() &&
                  !state.swEdgeToHwPaths[entry.edgeId].empty()) {
                ++targetStillRouted;
              }
            }
            if (!allRouted) {
              auto residualFailed = collectUnroutedEdges(state, dfg, edgeKinds);
              if (opts.verbose) {
                llvm::outs() << "  Reserved failed-edge residual:";
                for (IdIndex edgeId : residualFailed)
                  llvm::outs() << " " << edgeId;
                llvm::outs() << " (reservedTargetsRouted=" << targetStillRouted
                             << "/" << reservedChoice.paths.size() << ")\n";
              }
              if (targetStillRouted == reservedChoice.paths.size() &&
                  !residualFailed.empty() &&
                  residualFailed.size() <=
                      repairOpts.residualRepairFailedEdgeThreshold) {
                allRouted = runExactRoutingRepair(state, residualFailed, dfg, adg,
                                                 flattener, edgeKinds, opts,
                                                 congestion);
                auto nextResidualFailed =
                    allRouted ? std::vector<IdIndex>()
                              : collectUnroutedEdges(state, dfg, edgeKinds);
                bool residualChanged =
                    nextResidualFailed.size() != failedEdges.size() ||
                    !std::equal(nextResidualFailed.begin(),
                                nextResidualFailed.end(), failedEdges.begin(),
                                failedEdges.end());
                if (!allRouted &&
                    failedEdges.size() <=
                        repairOpts.residualJointRepairFailedEdgeThreshold &&
                    nextResidualFailed.size() <=
                        repairOpts.residualJointRepairFailedEdgeThreshold &&
                    residualChanged) {
                  llvm::SmallVector<IdIndex, 8> cycleEdges;
                  llvm::DenseSet<IdIndex> seenCycleEdges;
                  for (IdIndex edgeId : failedEdges) {
                    if (seenCycleEdges.insert(edgeId).second)
                      cycleEdges.push_back(edgeId);
                  }
                  for (IdIndex edgeId : nextResidualFailed) {
                    if (seenCycleEdges.insert(edgeId).second)
                      cycleEdges.push_back(edgeId);
                  }
                  if (cycleEdges.size() >= repairOpts.cycleEdgeClusterMin &&
                      cycleEdges.size() <= repairOpts.cycleEdgeClusterMax) {
                    auto cycleCheckpoint = state.save();
                    auto cycleRepairEdges =
                        buildFocusedRepairNeighborhood(
                            cycleEdges, repairOpts.earlyCPSatNeighborhoodLimit);
                    Options cycleExactOpts = opts;
                    cycleExactOpts.cpSatTimeLimitSeconds =
                        std::max(cycleExactOpts.cpSatTimeLimitSeconds,
                                 repairOpts.cycleExactMinTime);
                    if (opts.verbose) {
                      llvm::outs() << "  Residual cycle exact repair:";
                      for (IdIndex edgeId : cycleEdges)
                        llvm::outs() << " " << edgeId;
                      llvm::outs() << " | neighborhood:";
                      for (IdIndex edgeId : cycleRepairEdges)
                        llvm::outs() << " " << edgeId;
                      llvm::outs() << "\n";
                    }
                    bool cycleAllRouted = runExactRoutingRepair(
                        state, cycleRepairEdges, dfg, adg, flattener,
                        edgeKinds, cycleExactOpts, congestion, cycleEdges);
                    bool cycleImproved = updateBest(cycleAllRouted);
                    if (cycleImproved && cycleAllRouted) {
                      state.restore(bestCheckpoint);
                      return true;
                    }
                    if (cycleImproved &&
                        recursionDepth < maxRepairRecursionDepth) {
                      auto cycleResidualFailed =
                          collectUnroutedEdges(state, dfg, edgeKinds);
                      bool cycleResidualChanged =
                          cycleResidualFailed.size() != nextResidualFailed.size() ||
                          !std::equal(cycleResidualFailed.begin(),
                                      cycleResidualFailed.end(),
                                      nextResidualFailed.begin(),
                                      nextResidualFailed.end());
                      if (!cycleResidualFailed.empty() &&
                          cycleResidualFailed.size() <=
                              repairOpts.residualRepairFailedEdgeThreshold &&
                          cycleResidualChanged) {
                        auto recursiveRoutedCheckpoint = state.save();
                        state.clearRoutes(adg);
                        auto recursivePlacementCheckpoint = state.save();
                        state.restore(recursiveRoutedCheckpoint);
                        bool recursiveAllRouted =
                            runLocalRepair(state, recursivePlacementCheckpoint,
                                           cycleResidualFailed, dfg, adg,
                                           flattener, candidates, edgeKinds,
                                           opts, congestion,
                                           recursionDepth + 1);
                        if (updateBest(recursiveAllRouted) &&
                            recursiveAllRouted) {
                          state.restore(bestCheckpoint);
                          return true;
                        }
                      }
                    }
                    state.restore(cycleCheckpoint);
                  }
                }
                if (!allRouted &&
                    recursionDepth < maxRepairRecursionDepth &&
                    !nextResidualFailed.empty() &&
                    nextResidualFailed.size() <=
                        repairOpts.residualRepairFailedEdgeThreshold &&
                    residualChanged) {
                  auto recursiveRoutedCheckpoint = state.save();
                  state.clearRoutes(adg);
                  auto recursivePlacementCheckpoint = state.save();
                  state.restore(recursiveRoutedCheckpoint);
                  allRouted =
                      runLocalRepair(state, recursivePlacementCheckpoint,
                                     nextResidualFailed, dfg, adg, flattener,
                                     candidates, edgeKinds, opts, congestion,
                                     recursionDepth + 1);
                }
              }
            }
            if (targetStillRouted == reservedChoice.paths.size()) {
              bool improved = updateBest(allRouted);
              if (improved && allRouted) {
                state.restore(bestCheckpoint);
                return true;
              }
            }
          }
          state.restore(attemptCheckpoint);
        }
      }
    }
  }

  if (!bestAllRouted &&
      bestFailedEdges.size() <= repairOpts.cpSatEscalationFailedEdgeThreshold) {
    if (shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
    auto edgeOrderLess = [&](IdIndex lhs, IdIndex rhs) {
      double lhsWeight = classifyEdgePlacementWeight(dfg, lhs);
      double rhsWeight = classifyEdgePlacementWeight(dfg, rhs);
      if (std::abs(lhsWeight - rhsWeight) > 1e-9)
        return lhsWeight > rhsWeight;
      return lhs < rhs;
    };

    std::vector<IdIndex> conflictDrivenEdges(bestFailedEdges.begin(),
                                             bestFailedEdges.end());
    llvm::stable_sort(conflictDrivenEdges, edgeOrderLess);

    auto buildConflictNeighborhood =
        [&](llvm::ArrayRef<IdIndex> seedEdges,
            unsigned maxEdges) -> llvm::SmallVector<IdIndex, 24> {
      llvm::SmallVector<IdIndex, 24> repairEdges;
      llvm::DenseSet<IdIndex> seenEdges;
      MappingState probeState = state;
      probeState.clearRoutes(adg);
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
        IdIndex srcHwPort =
            edge->srcPort < state.swPortToHwPort.size()
                ? state.swPortToHwPort[edge->srcPort]
                : INVALID_ID;
        IdIndex dstHwPort =
            edge->dstPort < state.swPortToHwPort.size()
                ? state.swPortToHwPort[edge->dstPort]
                : INVALID_ID;
        if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
          continue;

        auto freeSpacePath =
            findPath(srcHwPort, dstHwPort, edgeId, probeState, dfg, adg,
                     emptyHistory, INVALID_ID, congestion);
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

      llvm::stable_sort(repairEdges, edgeOrderLess);
      return repairEdges;
    };
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

    bool improvedConflictRepair = false;
    for (unsigned conflictRound = 0;
         conflictRound < repairOpts.targetFocusedFailedEdgeThreshold &&
         !bestAllRouted;
         ++conflictRound) {
      if (shouldStopForBudget("local repair"))
        break;
      bool improvedThisConflictRound = false;
      conflictDrivenEdges.assign(bestFailedEdges.begin(), bestFailedEdges.end());
      llvm::stable_sort(conflictDrivenEdges, edgeOrderLess);

      state.restore(bestCheckpoint);
      auto jointRipupEdges =
          buildConflictNeighborhood(conflictDrivenEdges,
                                    repairOpts.jointRipupMaxEdges);
      if (jointRipupEdges.size() < repairOpts.relocationCandidateLimit)
        expandConflictNeighborhood(jointRipupEdges,
                                   repairOpts.earlyCPSatNeighborhoodLimit);
      llvm::stable_sort(jointRipupEdges, edgeOrderLess);
      if (jointRipupEdges.size() > conflictDrivenEdges.size() &&
          jointRipupEdges.size() <= repairOpts.jointRipupMaxEdges) {
        llvm::outs() << "  Conflict-directed joint repair: failedEdges="
                     << conflictDrivenEdges.size()
                     << ", ripupEdges=" << jointRipupEdges.size() << "\n";
        auto attemptCheckpoint = state.save();
        for (IdIndex edgeId : jointRipupEdges) {
          if (edgeId >= state.swEdgeToHwPaths.size())
            continue;
          state.unmapEdge(edgeId, adg);
        }

        bool allRouted = runExactRoutingRepair(state, jointRipupEdges, dfg, adg,
                                               flattener, edgeKinds, opts,
                                               congestion);
        if (updateBest(allRouted)) {
          improvedConflictRepair = true;
          improvedThisConflictRound = true;
          if (allRouted) {
            state.restore(bestCheckpoint);
            return true;
          }
          continue;
        }
        state.restore(attemptCheckpoint);
      }

      for (IdIndex targetEdge : conflictDrivenEdges) {
        if (shouldStopForBudget("local repair"))
          break;
        state.restore(bestCheckpoint);
        if (targetEdge >= state.swEdgeToHwPaths.size() ||
            !state.swEdgeToHwPaths[targetEdge].empty()) {
          continue;
        }

        const Edge *edge = dfg.getEdge(targetEdge);
        if (!edge)
          continue;
        IdIndex srcHwPort =
            edge->srcPort < state.swPortToHwPort.size()
                ? state.swPortToHwPort[edge->srcPort]
                : INVALID_ID;
        IdIndex dstHwPort =
            edge->dstPort < state.swPortToHwPort.size()
                ? state.swPortToHwPort[edge->dstPort]
                : INVALID_ID;
        if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
          continue;

        MappingState probeState = state;
        probeState.clearRoutes(adg);
        llvm::DenseMap<IdIndex, double> emptyHistory;
        auto freeSpacePath =
            findPath(srcHwPort, dstHwPort, targetEdge, probeState, dfg, adg,
                     emptyHistory, INVALID_ID, congestion);
        if (freeSpacePath.empty())
          continue;

        llvm::SmallVector<IdIndex, 16> ripupEdges;
        llvm::DenseSet<IdIndex> seenRipup;
        seenRipup.insert(targetEdge);
        ripupEdges.push_back(targetEdge);
        for (IdIndex portId : freeSpacePath) {
          if (portId >= state.portToUsingEdges.size())
            continue;
          for (IdIndex otherEdgeId : state.portToUsingEdges[portId]) {
            if (seenRipup.insert(otherEdgeId).second)
              ripupEdges.push_back(otherEdgeId);
          }
        }
        if (ripupEdges.size() < repairOpts.focusedBlockerEdgeThreshold)
          expandConflictNeighborhood(ripupEdges,
                                     repairOpts.relocationCandidateLimit);
        llvm::stable_sort(ripupEdges, edgeOrderLess);

        if (ripupEdges.size() <= repairOpts.singleRipupMinEdges ||
            ripupEdges.size() > repairOpts.singleRipupMaxEdges)
          continue;

        llvm::outs() << "  Conflict-directed repair edge " << targetEdge
                     << ": freePathLen=" << freeSpacePath.size()
                     << ", ripupEdges=" << ripupEdges.size() << "\n";

        auto attemptCheckpoint = state.save();
        for (IdIndex edgeId : ripupEdges) {
          if (edgeId >= state.swEdgeToHwPaths.size())
            continue;
          state.unmapEdge(edgeId, adg);
        }

        bool allRouted = runExactRoutingRepair(state, ripupEdges, dfg, adg,
                                               flattener, edgeKinds, opts,
                                               congestion);
        if (updateBest(allRouted)) {
          improvedConflictRepair = true;
          improvedThisConflictRound = true;
          if (allRouted) {
            state.restore(bestCheckpoint);
            return true;
          }
          break;
        }
        state.restore(attemptCheckpoint);
      }

      if (!improvedThisConflictRound)
        break;
    }

    if (improvedConflictRepair)
      state.restore(bestCheckpoint);
  }

  auto buildCpSatRepairOpts = [&]() {
    Options cpSatRepairOpts = opts;
    if (bestFailedEdges.size() <=
        repairOpts.residualJointRepairFailedEdgeThreshold) {
      cpSatRepairOpts.cpSatTimeLimitSeconds =
          std::max(cpSatRepairOpts.cpSatTimeLimitSeconds,
                   repairOpts.earlyCPSatMinTime);
      cpSatRepairOpts.cpSatNeighborhoodNodeLimit =
          std::max<unsigned>(cpSatRepairOpts.cpSatNeighborhoodNodeLimit,
                             repairOpts.earlyCPSatNeighborhoodLimit);
      cpSatRepairOpts.placementMoveRadius =
          std::max<unsigned>(cpSatRepairOpts.placementMoveRadius,
                             repairOpts.earlyCPSatMoveRadius);
    } else if (bestFailedEdges.size() <= repairOpts.cpSatSmallFailedThreshold) {
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
    return cpSatRepairOpts;
  };

  for (unsigned polishRound = 0;
       polishRound < repairOpts.residualJointRepairFailedEdgeThreshold &&
       !bestAllRouted;
       ++polishRound) {
    if (shouldStopForBudget("local repair"))
      break;
    bool improved = false;

    if (opts.enableCPSat &&
        bestFailedEdges.size() <= repairOpts.cpSatMediumFailedThreshold) {
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

    if (!bestAllRouted &&
        bestFailedEdges.size() <= repairOpts.cpSatFallbackFailedEdgeThreshold) {
      state.restore(bestCheckpoint);
      bool allRouted = runExactRoutingRepair(state, bestFailedEdges, dfg, adg,
                                             flattener, edgeKinds, opts,
                                             congestion);
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

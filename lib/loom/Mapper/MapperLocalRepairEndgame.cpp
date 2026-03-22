#include "MapperLocalRepairInternal.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace loom {

using namespace mapper_detail;

bool LocalRepairDriver::runLateRepairStages() {
  if (!bestAllRouted &&
      bestFailedEdges.size() <= repairOpts.cpSatEscalationFailedEdgeThreshold) {
    if (mapper.shouldStopForBudget("local repair")) {
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
      double weight = edgeWeight(edgeId);
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
      if (mapper.shouldStopForBudget("local repair"))
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
          double candScore = mapper.scorePlacement(swNode, candHw, state, dfg,
                                                   adg, flattener, candidates);
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
              1.0, mapper.clampDeadlineMsToRemainingBudget(
                       largerExactNeighborhood ? 5000.0 : 8000.0))));
      bool stopExactSearch = false;

      std::function<void(unsigned)> enumerateNeighborhood;
      enumerateNeighborhood = [&](unsigned depth) {
        if (stopExactSearch || mapper.shouldStopForBudget("local repair") ||
            std::chrono::steady_clock::now() > exactDeadline)
          return;
        if (depth >= focusNodes.size()) {
          mapper.rebindScalarInputSentinels(state, dfg, adg, flattener);
          mapper.bindMemrefSentinels(state, dfg, adg);
          classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
          bool allRouted = rerouteRepairState(state);
          if (updateBest(allRouted) && allRouted)
            stopExactSearch = true;
          return;
        }

        IdIndex swNode = focusNodes[depth];
        for (IdIndex candHw : exactDomains.lookup(swNode)) {
          auto savepoint = state.beginSavepoint();
          if (state.mapNode(swNode, candHw, dfg, adg) != ActionResult::Success) {
            state.rollbackSavepoint(savepoint);
            continue;
          }
          if (!mapper.bindMappedNodePorts(swNode, state, dfg, adg)) {
            state.rollbackSavepoint(savepoint);
            continue;
          }
          enumerateNeighborhood(depth + 1);
          state.rollbackSavepoint(savepoint);
          if (stopExactSearch)
            return;
        }
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
    if (mapper.shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
    state.restore(bestCheckpoint);
    bool allRouted = mapper.runExactRoutingRepair(state, bestFailedEdges, dfg, adg,
                                                  flattener, edgeKinds, opts,
                                                  congestion);
    if (updateBest(allRouted) && allRouted) {
      state.restore(bestCheckpoint);
      return true;
    }
  }

  if (!bestAllRouted &&
      bestFailedEdges.size() <= repairOpts.targetFocusedFailedEdgeThreshold) {
    if (mapper.shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
    state.restore(bestCheckpoint);

    auto edgeOrderLess = [&](IdIndex lhs, IdIndex rhs) {
      double lhsWeight = edgeWeight(lhs);
      double rhsWeight = edgeWeight(rhs);
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
      if (mapper.shouldStopForBudget("local repair")) {
        candidateGenerationFailed = true;
        break;
      }
      MappingState probeState = state;
      probeState.clearRoutes(dfg, adg);
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
        auto path = mapper.findPath(srcHwPort, dstHwPort, edgeId, probeState, dfg,
                                    adg, localHistory, INVALID_ID, nullptr);
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
      reservedState.clearRoutes(dfg, adg);
      llvm::SmallVector<ReservedPath, 4> currentReserved;
      struct ReservedCandidateSet {
        llvm::SmallVector<ReservedPath, 4> paths;
        unsigned count = 0;
        size_t totalPathLen = 0;
      };
      std::vector<ReservedCandidateSet> reservedCandidates;

      std::function<void(size_t)> searchReservedPaths;
      searchReservedPaths = [&](size_t index) {
        if (mapper.shouldStopForBudget("local repair"))
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
        auto savepoint = reservedState.beginSavepoint();
        for (const auto &path : candidatePathsByEdge.lookup(edgeId)) {
          if (mapper.shouldStopForBudget("local repair"))
            return;
          reservedState.rollbackSavepoint(savepoint);
          savepoint = reservedState.beginSavepoint();
          if (reservedState.mapEdge(edgeId, path, dfg, adg) !=
              ActionResult::Success)
            continue;
          currentReserved.push_back({edgeId, path});
          searchReservedPaths(index + 1);
          currentReserved.pop_back();
        }
        reservedState.rollbackSavepoint(savepoint);
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
          if (mapper.shouldStopForBudget("local repair"))
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
            Mapper::Options focusedExactOpts = opts;
            focusedExactOpts.cpSatTimeLimitSeconds =
                std::max(focusedExactOpts.cpSatTimeLimitSeconds,
                         repairOpts.focusedTargetMinTime);
            bool allRouted =
                mapper.runExactRoutingRepair(state, blockerEdges, dfg, adg,
                                             flattener, edgeKinds,
                                             focusedExactOpts, congestion);
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
          auto attemptSavepoint = state.beginSavepoint();
          for (IdIndex edgeId : blockerEdges) {
            if (edgeId < state.swEdgeToHwPaths.size())
              state.unmapEdge(edgeId, dfg, adg);
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
                state.clearRoutes(dfg, adg);
                auto recursivePlacementCheckpoint = state.save();
                state.restore(recursiveRoutedCheckpoint);
                bool recursiveAllRouted =
                    mapper.runLocalRepair(state, recursivePlacementCheckpoint,
                                          postTargetFailed, dfg, adg, flattener,
                                          candidates, edgeKinds, opts,
                                          congestion, recursionDepth + 1);
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
                                 : mapper.runExactRoutingRepair(state, rerouteEdges,
                                                                dfg, adg,
                                                                flattener,
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
                allRouted = mapper.runExactRoutingRepair(
                    state, residualFailed, dfg, adg, flattener, edgeKinds, opts,
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
                    auto cycleSavepoint = state.beginSavepoint();
                    auto cycleRepairEdges = buildFocusedRepairNeighborhood(
                        cycleEdges, repairOpts.earlyCPSatNeighborhoodLimit);
                    Mapper::Options cycleExactOpts = opts;
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
                    bool cycleAllRouted = mapper.runExactRoutingRepair(
                        state, cycleRepairEdges, dfg, adg, flattener, edgeKinds,
                        cycleExactOpts, congestion, cycleEdges);
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
                        state.clearRoutes(dfg, adg);
                        auto recursivePlacementCheckpoint = state.save();
                        state.restore(recursiveRoutedCheckpoint);
                        bool recursiveAllRouted =
                            mapper.runLocalRepair(state, recursivePlacementCheckpoint,
                                                  cycleResidualFailed, dfg, adg,
                                                  flattener, candidates,
                                                  edgeKinds, opts, congestion,
                                                  recursionDepth + 1);
                        if (updateBest(recursiveAllRouted) &&
                            recursiveAllRouted) {
                          state.restore(bestCheckpoint);
                          return true;
                        }
                      }
                    }
                    state.rollbackSavepoint(cycleSavepoint);
                  }
                }
                if (!allRouted &&
                    recursionDepth < maxRepairRecursionDepth &&
                    !nextResidualFailed.empty() &&
                    nextResidualFailed.size() <=
                        repairOpts.residualRepairFailedEdgeThreshold &&
                    residualChanged) {
                  auto recursiveRoutedCheckpoint = state.save();
                  state.clearRoutes(dfg, adg);
                  auto recursivePlacementCheckpoint = state.save();
                  state.restore(recursiveRoutedCheckpoint);
                  allRouted = mapper.runLocalRepair(
                      state, recursivePlacementCheckpoint, nextResidualFailed,
                      dfg, adg, flattener, candidates, edgeKinds, opts,
                      congestion, recursionDepth + 1);
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
          state.rollbackSavepoint(attemptSavepoint);
        }
      }
    }
  }

  if (!bestAllRouted &&
      bestFailedEdges.size() <= repairOpts.cpSatEscalationFailedEdgeThreshold) {
    if (mapper.shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
    auto edgeOrderLess = [&](IdIndex lhs, IdIndex rhs) {
      double lhsWeight = edgeWeight(lhs);
      double rhsWeight = edgeWeight(rhs);
      if (std::abs(lhsWeight - rhsWeight) > 1e-9)
        return lhsWeight > rhsWeight;
      return lhs < rhs;
    };

    bool improvedConflictRepair = false;
    for (unsigned round = 0;
         round < repairOpts.targetFocusedFailedEdgeThreshold; ++round) {
      if (mapper.shouldStopForBudget("local repair"))
        break;
      auto conflictDrivenEdges = collectUnroutedEdges(state, dfg, edgeKinds);
      llvm::stable_sort(conflictDrivenEdges, edgeOrderLess);
      bool improvedThisConflictRound = false;

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
        auto attemptSavepoint = state.beginSavepoint();
        for (IdIndex edgeId : jointRipupEdges) {
          if (edgeId >= state.swEdgeToHwPaths.size())
            continue;
          state.unmapEdge(edgeId, dfg, adg);
        }

        bool allRouted = mapper.runExactRoutingRepair(state, jointRipupEdges, dfg,
                                                      adg, flattener, edgeKinds,
                                                      opts, congestion);
        if (updateBest(allRouted)) {
          improvedConflictRepair = true;
          improvedThisConflictRound = true;
          if (allRouted) {
            state.restore(bestCheckpoint);
            return true;
          }
          state.rollbackSavepoint(attemptSavepoint);
          continue;
        }
        state.rollbackSavepoint(attemptSavepoint);
      }

      for (IdIndex targetEdge : conflictDrivenEdges) {
        if (mapper.shouldStopForBudget("local repair"))
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
        probeState.clearRoutes(dfg, adg);
        llvm::DenseMap<IdIndex, double> emptyHistory;
        auto freeSpacePath = mapper.findPath(srcHwPort, dstHwPort, targetEdge,
                                             probeState, dfg, adg, emptyHistory,
                                             INVALID_ID, congestion);
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

        auto attemptSavepoint = state.beginSavepoint();
        for (IdIndex edgeId : ripupEdges) {
          if (edgeId >= state.swEdgeToHwPaths.size())
            continue;
          state.unmapEdge(edgeId, dfg, adg);
        }

        bool allRouted = mapper.runExactRoutingRepair(state, ripupEdges, dfg, adg,
                                                      flattener, edgeKinds, opts,
                                                      congestion);
        if (updateBest(allRouted)) {
          improvedConflictRepair = true;
          improvedThisConflictRound = true;
          if (allRouted) {
            state.restore(bestCheckpoint);
            return true;
          }
          state.rollbackSavepoint(attemptSavepoint);
          break;
        }
        state.rollbackSavepoint(attemptSavepoint);
      }

      if (!improvedThisConflictRound)
        break;
    }

    if (improvedConflictRepair)
      state.restore(bestCheckpoint);
  }

  auto buildCpSatRepairOpts = [&]() {
    Mapper::Options cpSatRepairOpts = opts;
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
    }
    return cpSatRepairOpts;
  };

  if (!bestAllRouted && opts.enableCPSat &&
      !bestFailedEdges.empty() &&
      bestFailedEdges.size() <= repairOpts.cpSatEscalationFailedEdgeThreshold) {
    if (mapper.shouldStopForBudget("local repair")) {
      state.restore(bestCheckpoint);
      return bestAllRouted;
    }
    state.restore(bestPlacementCheckpoint);
    Mapper::Options cpSatRepairOpts = buildCpSatRepairOpts();
    bool allRouted = mapper.runCPSatNeighborhoodRepair(
        state, bestPlacementCheckpoint, bestFailedEdges, dfg, adg, flattener,
        candidates, edgeKinds, cpSatRepairOpts);
    if (updateBest(allRouted) && allRouted) {
      state.restore(bestCheckpoint);
      return true;
    }
  }

  state.restore(bestCheckpoint);
  return bestAllRouted;
}

} // namespace loom

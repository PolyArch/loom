#include "fcc/Mapper/Mapper.h"
#include "fcc/Mapper/MapperRelaxedRouting.h"
#include "MapperInternal.h"
#include "MapperRoutingCongestion.h"
#include "MapperRoutingInternal.h"

#include "fcc/Mapper/TechMapper.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fcc {

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers used only by routeOnePass / runRouting
// ---------------------------------------------------------------------------

namespace {

bool isDirectBindingPath(llvm::ArrayRef<IdIndex> path) {
  return path.size() == 2 && path[0] == path[1];
}

void collectUnroutedSiblingEdges(
    IdIndex srcHwPort, const std::vector<IdIndex> &edgeOrder,
    const MappingState &state, const Graph &dfg,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
    llvm::SmallVectorImpl<IdIndex> &siblings) {
  siblings.clear();
  for (IdIndex edgeId : edgeOrder) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (edgeId >= state.swEdgeToHwPaths.size() ||
        !state.swEdgeToHwPaths[edgeId].empty())
      continue;
    if (edge->srcPort >= state.swPortToHwPort.size() ||
        edge->dstPort >= state.swPortToHwPort.size())
      continue;
    if (state.swPortToHwPort[edge->srcPort] != srcHwPort ||
        state.swPortToHwPort[edge->dstPort] == INVALID_ID)
      continue;
    siblings.push_back(edgeId);
  }
}

bool isBoundarySinkNode(const Node *node) {
  if (!node)
    return false;
  if (node->kind == Node::ModuleOutputNode)
    return true;
  if (node->kind != Node::OperationNode)
    return false;
  return routing_detail::isSoftwareMemoryInterfaceOpName(
      getNodeAttrStr(node, "op_name"));
}

void collectUnroutedSinkEdgesForNode(
    IdIndex dstSwNode, const std::vector<IdIndex> &edgeOrder,
    const MappingState &state, const Graph &dfg,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
    llvm::SmallVectorImpl<IdIndex> &groupedEdges) {
  groupedEdges.clear();
  for (IdIndex edgeId : edgeOrder) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (edgeId >= state.swEdgeToHwPaths.size() ||
        !state.swEdgeToHwPaths[edgeId].empty())
      continue;
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!dstPort || dstPort->parentNode == INVALID_ID)
      continue;
    if (dstPort->parentNode != dstSwNode)
      continue;
    groupedEdges.push_back(edgeId);
  }
}

void collectRepairSiblingEdges(IdIndex failedEdgeId,
                               const std::vector<IdIndex> &edgeOrder,
                               const MappingState &state, const Graph &dfg,
                               llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                               llvm::SmallVectorImpl<IdIndex> &siblings) {
  siblings.clear();
  const Edge *failedEdge = dfg.getEdge(failedEdgeId);
  if (!failedEdge)
    return;

  IdIndex failedSrcHwPort =
      failedEdge->srcPort < state.swPortToHwPort.size()
          ? state.swPortToHwPort[failedEdge->srcPort]
          : INVALID_ID;
  const Port *failedSrcPort = dfg.getPort(failedEdge->srcPort);
  const Port *failedDstPort = dfg.getPort(failedEdge->dstPort);
  IdIndex failedSrcSwNode =
      (failedSrcPort && failedSrcPort->parentNode != INVALID_ID)
          ? failedSrcPort->parentNode
          : INVALID_ID;
  IdIndex failedDstSwNode =
      (failedDstPort && failedDstPort->parentNode != INVALID_ID)
          ? failedDstPort->parentNode
          : INVALID_ID;

  llvm::DenseSet<IdIndex> seen;
  for (IdIndex edgeId : edgeOrder) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;
    IdIndex srcHwPort =
        edge->srcPort < state.swPortToHwPort.size()
            ? state.swPortToHwPort[edge->srcPort]
            : INVALID_ID;
    bool sameSourcePort =
        failedSrcHwPort != INVALID_ID && srcHwPort == failedSrcHwPort;
    bool sameSinkNode = dstPort->parentNode == failedDstSwNode;
    bool sameEndpointPair = srcPort->parentNode == failedSrcSwNode &&
                            dstPort->parentNode == failedDstSwNode;
    if (!sameSourcePort && !sameSinkNode && !sameEndpointPair)
      continue;
    if (seen.insert(edgeId).second)
      siblings.push_back(edgeId);
  }
}

unsigned getRoutingPriority(IdIndex edgeId, const Graph &dfg,
                            const Mapper::Options &opts) {
  const Edge *edge = dfg.getEdge(edgeId);
  if (!edge)
    return opts.routing.priority.invalidEdge;

  auto getNode = [&](IdIndex portId) -> const Node * {
    const Port *port = dfg.getPort(portId);
    if (!port || port->parentNode == INVALID_ID)
      return nullptr;
    return dfg.getNode(port->parentNode);
  };
  auto getOpName = [&](IdIndex portId) -> llvm::StringRef {
    const Node *node = getNode(portId);
    if (!node || node->kind != Node::OperationNode)
      return {};
    return getNodeAttrStr(node, "op_name");
  };

  const Node *srcNode = getNode(edge->srcPort);
  const Node *dstNode = getNode(edge->dstPort);
  llvm::StringRef srcOp = getOpName(edge->srcPort);
  llvm::StringRef dstOp = getOpName(edge->dstPort);

  if (routing_detail::isSoftwareMemoryInterfaceOpName(dstOp))
    return opts.routing.priority.memorySink;
  if (dstNode && dstNode->kind == Node::ModuleOutputNode)
    return opts.routing.priority.moduleOutput;
  if (srcOp == "handshake.load" || srcOp == "handshake.store")
    return opts.routing.priority.loadStoreSource;
  if (routing_detail::isSoftwareMemoryInterfaceOpName(srcOp))
    return opts.routing.priority.memorySource;
  if (srcOp == "handshake.load" || dstOp == "handshake.load" ||
      srcOp == "handshake.store" || dstOp == "handshake.store")
    return opts.routing.priority.loadStoreIncident;
  return opts.routing.priority.fallback;
}

void collectFailedRoutingHotspots(
    llvm::ArrayRef<IdIndex> failedEdges, const MappingState &state,
    const Graph &dfg, const Graph &adg,
    llvm::DenseSet<IdIndex> &hotspotPorts,
    llvm::DenseSet<IdIndex> &hotspotNodes) {
  hotspotPorts.clear();
  hotspotNodes.clear();

  auto recordPort = [&](IdIndex hwPortId) {
    if (hwPortId == INVALID_ID)
      return;
    hotspotPorts.insert(hwPortId);
    const Port *hwPort = adg.getPort(hwPortId);
    if (!hwPort || hwPort->parentNode == INVALID_ID)
      return;
    hotspotNodes.insert(hwPort->parentNode);
  };

  for (IdIndex edgeId : failedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (edge->srcPort < state.swPortToHwPort.size())
      recordPort(state.swPortToHwPort[edge->srcPort]);
    if (edge->dstPort < state.swPortToHwPort.size())
      recordPort(state.swPortToHwPort[edge->dstPort]);
  }
}

bool routedEdgeTouchesHotspot(llvm::ArrayRef<IdIndex> path,
                              const llvm::DenseSet<IdIndex> &hotspotPorts,
                              const llvm::DenseSet<IdIndex> &hotspotNodes,
                              const Graph &adg) {
  if (path.empty())
    return false;
  for (IdIndex portId : path) {
    if (hotspotPorts.contains(portId))
      return true;
    const Port *port = adg.getPort(portId);
    if (!port || port->parentNode == INVALID_ID)
      continue;
    if (hotspotNodes.contains(port->parentNode))
      return true;
  }
  return false;
}

void collectSelectiveRipupEdges(
    llvm::ArrayRef<IdIndex> failedEdges, const std::vector<IdIndex> &edgeOrder,
    const MappingState &state, const Graph &dfg, const Graph &adg,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds, const Mapper::Options &opts,
    llvm::SmallVectorImpl<IdIndex> &ripupEdges) {
  ripupEdges.clear();

  llvm::DenseSet<IdIndex> hotspotPorts;
  llvm::DenseSet<IdIndex> hotspotNodes;
  collectFailedRoutingHotspots(failedEdges, state, dfg, adg, hotspotPorts,
                               hotspotNodes);
  const bool tightRipup =
      failedEdges.size() <= opts.routing.tightRipupFailedEdgeThreshold;

  llvm::DenseSet<IdIndex> seen;
  for (IdIndex edgeId : failedEdges) {
    if (seen.insert(edgeId).second)
      ripupEdges.push_back(edgeId);
  }

  if (failedEdges.size() <= opts.routing.siblingExpansionFailedEdgeThreshold) {
    llvm::SmallVector<IdIndex, 16> siblings;
    for (IdIndex edgeId : failedEdges) {
      collectRepairSiblingEdges(edgeId, edgeOrder, state, dfg, edgeKinds,
                                siblings);
      for (IdIndex siblingEdgeId : siblings) {
        if (seen.insert(siblingEdgeId).second)
          ripupEdges.push_back(siblingEdgeId);
      }
    }
  }

  for (IdIndex edgeId : edgeOrder) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    if (edgeId >= state.swEdgeToHwPaths.size())
      continue;
    const auto &path = state.swEdgeToHwPaths[edgeId];
    if (path.empty() || isDirectBindingPath(path))
      continue;
    llvm::DenseSet<IdIndex> emptyNodes;
    if (!routedEdgeTouchesHotspot(path, hotspotPorts,
                                  tightRipup ? emptyNodes : hotspotNodes,
                                  adg))
      continue;
    if (seen.insert(edgeId).second)
      ripupEdges.push_back(edgeId);
  }

  if (tightRipup &&
      ripupEdges.size() < std::max<size_t>(opts.routing.tightRipupMinEdges,
                                           failedEdges.size() * 2)) {
    const size_t targetRipup =
        std::min<size_t>(opts.routing.tightRipupTargetCap, edgeOrder.size());
    for (IdIndex edgeId : edgeOrder) {
      if (ripupEdges.size() >= targetRipup)
        break;
      if (edgeId < edgeKinds.size() &&
          (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
           edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
        continue;
      if (edgeId >= state.swEdgeToHwPaths.size())
        continue;
      const auto &path = state.swEdgeToHwPaths[edgeId];
      if (path.empty() || isDirectBindingPath(path))
        continue;
      if (!routedEdgeTouchesHotspot(path, hotspotPorts, hotspotNodes, adg))
        continue;
      if (seen.insert(edgeId).second)
        ripupEdges.push_back(edgeId);
    }
  }
}

void dumpFailedEdgeDiagnostics(llvm::ArrayRef<IdIndex> failedEdges,
                               const MappingState &state, const Graph &dfg,
                               unsigned limit) {
  unsigned emitted = 0;
  for (IdIndex edgeId : failedEdges) {
    if (emitted >= limit)
      break;
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
    IdIndex srcHwPort =
        edge->srcPort < state.swPortToHwPort.size() ? state.swPortToHwPort[edge->srcPort]
                                                    : INVALID_ID;
    IdIndex dstHwPort =
        edge->dstPort < state.swPortToHwPort.size() ? state.swPortToHwPort[edge->dstPort]
                                                    : INVALID_ID;
    llvm::outs() << "    failed edge " << edgeId << ": "
                 << (srcNode ? getNodeAttrStr(srcNode, "op_name") : "<src>")
                 << " -> "
                 << (dstNode ? getNodeAttrStr(dstNode, "op_name") : "<dst>")
                 << " (hw " << srcHwPort << " -> " << dstHwPort << ")\n";
    ++emitted;
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Mapper methods: single-pass routing and main routing orchestrator
// ---------------------------------------------------------------------------

bool Mapper::routeOnePass(MappingState &state, const Graph &dfg,
                          const Graph &adg,
                          llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                          const std::vector<IdIndex> &edgeOrder,
                          const llvm::DenseMap<IdIndex, double>
                              &routingOutputHistory,
                          unsigned &routed, unsigned &total,
                          const Options &opts,
                          CongestionState *congestion,
                          RelaxedRoutingState *relaxedRouting) {
  bool allRouted = true;
  routed = 0;
  total = 0;
  llvm::DenseSet<IdIndex> processedFanoutSources;
  llvm::DenseSet<IdIndex> processedSinkNodes;
  llvm::DenseSet<IdIndex> processedSinkEdges;
  auto estimatePortDistance = [&](IdIndex lhsPortId, IdIndex rhsPortId) {
    if (!activeFlattener)
      return 0.0;
    const Port *lhsPort = adg.getPort(lhsPortId);
    const Port *rhsPort = adg.getPort(rhsPortId);
    if (!lhsPort || !rhsPort || lhsPort->parentNode == INVALID_ID ||
        rhsPort->parentNode == INVALID_ID)
      return 0.0;
    return static_cast<double>(mapper_detail::placementDistance(
        lhsPort->parentNode, rhsPort->parentNode, *activeFlattener));
  };
  auto rankFirstHopChoices = [&](IdIndex dstHwPort,
                                 llvm::SmallVectorImpl<IdIndex> &choices,
                                 unsigned limit) {
    llvm::stable_sort(choices, [&](IdIndex lhs, IdIndex rhs) {
      double lhsScore = estimatePortDistance(lhs, dstHwPort);
      double rhsScore = estimatePortDistance(rhs, dstHwPort);
      if (std::abs(lhsScore - rhsScore) > 1e-9)
        return lhsScore < rhsScore;
      return lhs < rhs;
    });
    if (limit != 0 && choices.size() > limit)
      choices.resize(limit);
  };
  auto rankSharedFirstHopChoices =
      [&](llvm::ArrayRef<IdIndex> siblingEdges,
          llvm::SmallVectorImpl<IdIndex> &choices, unsigned limit) {
        llvm::stable_sort(choices, [&](IdIndex lhs, IdIndex rhs) {
          auto scoreChoice = [&](IdIndex choice) {
            double totalScore = 0.0;
            for (IdIndex siblingEdgeId : siblingEdges) {
              const Edge *siblingEdge = dfg.getEdge(siblingEdgeId);
              if (!siblingEdge)
                continue;
              IdIndex siblingDstHwPort = siblingEdge->dstPort <
                                                 state.swPortToHwPort.size()
                                             ? state.swPortToHwPort
                                                   [siblingEdge->dstPort]
                                             : INVALID_ID;
              if (siblingDstHwPort == INVALID_ID)
                continue;
              double weight =
                  mapper_detail::classifyEdgePlacementWeight(dfg,
                                                             siblingEdgeId);
              totalScore +=
                  weight * estimatePortDistance(choice, siblingDstHwPort);
            }
            return totalScore;
          };
          double lhsScore = scoreChoice(lhs);
          double rhsScore = scoreChoice(rhs);
          if (std::abs(lhsScore - rhsScore) > 1e-9)
            return lhsScore < rhsScore;
          return lhs < rhs;
        });
        if (limit != 0 && choices.size() > limit)
          choices.resize(limit);
      };
  auto savePresentDemand = [&]() -> std::vector<unsigned> {
    if (!congestion)
      return {};
    return congestion->presentDemand;
  };
  auto restorePresentDemand = [&](const std::vector<unsigned> &saved) {
    if (!congestion)
      return;
    congestion->presentDemand = saved;
  };
  auto saveRelaxedRouting = [&]() -> RelaxedRoutingState::Snapshot {
    if (!relaxedRouting)
      return {};
    return relaxedRouting->save();
  };
  auto restoreRelaxedRouting =
      [&](const RelaxedRoutingState::Snapshot &snapshot) {
        if (!relaxedRouting)
          return;
        relaxedRouting->restore(snapshot);
      };
  auto commitRoutedPath = [&](IdIndex routedEdgeId,
                              llvm::ArrayRef<IdIndex> routedPath) {
    if (routedPath.empty() || isDirectBindingPath(routedPath))
      return;
    if (congestion)
      congestion->commitRoute(routedPath, adg);
    if (relaxedRouting)
      relaxedRouting->commitRoute(routedEdgeId, routedPath.front(), routedPath,
                                  adg);
  };

  for (IdIndex edgeId : edgeOrder) {
    if (shouldStopForBudget("routing"))
      return false;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (processedSinkEdges.contains(edgeId))
      continue;
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;

    // Skip pre-routed edges.
    if (edgeId < state.swEdgeToHwPaths.size() &&
        !state.swEdgeToHwPaths[edgeId].empty())
      continue;

    if (!isEdgeRoutable(edgeId, state, dfg))
      continue;

    IdIndex srcSwPort = edge->srcPort;
    IdIndex dstSwPort = edge->dstPort;
    IdIndex srcHwPort = state.swPortToHwPort[srcSwPort];
    IdIndex dstHwPort = state.swPortToHwPort[dstSwPort];
    const Port *dstSwPortPtr = dfg.getPort(dstSwPort);
    const Node *dstSwNode =
        (dstSwPortPtr && dstSwPortPtr->parentNode != INVALID_ID)
            ? dfg.getNode(dstSwPortPtr->parentNode)
            : nullptr;
    IdIndex dstSwNodeId =
        (dstSwPortPtr && dstSwPortPtr->parentNode != INVALID_ID)
            ? dstSwPortPtr->parentNode
            : INVALID_ID;

    if (dstSwNodeId != INVALID_ID &&
        dstSwNode && dstSwNode->kind == Node::ModuleOutputNode &&
        !processedSinkNodes.contains(dstSwNodeId)) {
      llvm::SmallVector<IdIndex, 8> outputEdges;
      collectUnroutedSinkEdgesForNode(dstSwNodeId, edgeOrder, state, dfg,
                                      edgeKinds, outputEdges);
      if (outputEdges.size() > 1 &&
          outputEdges.size() <= opts.routing.moduleOutputGroupMaxEdges) {
        processedSinkNodes.insert(dstSwNodeId);
        unsigned bestSuccessCount = 0;
        size_t bestTotalPathLen = std::numeric_limits<size_t>::max();
        llvm::SmallVector<IdIndex, 8> bestOrder;
        llvm::SmallVector<IdIndex, 8> currentOrder;
        llvm::SmallVector<IdIndex, 8> currentRemaining(outputEdges.begin(),
                                                       outputEdges.end());
        llvm::sort(currentRemaining);
        llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 8>> firstHopChoices;
        for (IdIndex groupedEdgeId : outputEdges) {
          const Edge *groupedEdge = dfg.getEdge(groupedEdgeId);
          if (!groupedEdge)
            continue;
          IdIndex groupedSrcHwPort = state.swPortToHwPort[groupedEdge->srcPort];
          llvm::SmallVector<IdIndex, 8> choices;
          if (routing_detail::isNonRoutingOutputPort(groupedSrcHwPort, adg)) {
            IdIndex lockedFirstHop =
                routing_detail::getLockedFirstHopForSource(groupedEdgeId, groupedSrcHwPort, state);
            if (lockedFirstHop != INVALID_ID) {
              choices.push_back(lockedFirstHop);
            } else if (auto it = connectivity.outToIn.find(groupedSrcHwPort);
                       it != connectivity.outToIn.end()) {
              choices.append(it->second.begin(), it->second.end());
              llvm::sort(choices);
              choices.erase(std::unique(choices.begin(), choices.end()),
                            choices.end());
            }
          }
          if (choices.empty())
            choices.push_back(INVALID_ID);
          if (choices.front() != INVALID_ID)
            rankFirstHopChoices(state.swPortToHwPort[groupedEdge->dstPort],
                                choices,
                                opts.routing.moduleOutputFirstHopLimit);
          firstHopChoices[groupedEdgeId] = std::move(choices);
        }

        auto boundaryCheckpoint = state.save();
        auto boundaryRelaxedCheckpoint = saveRelaxedRouting();
        std::function<void(llvm::SmallVectorImpl<IdIndex> &, unsigned, size_t)>
            searchBoundaryGroup;
        searchBoundaryGroup =
            [&](llvm::SmallVectorImpl<IdIndex> &remainingEdges,
                unsigned successCount, size_t totalPathLen) {
              if (successCount + remainingEdges.size() < bestSuccessCount)
                return;
              if (remainingEdges.empty()) {
                bool preferLexicographically =
                    bestOrder.empty() ||
                    std::lexicographical_compare(
                        currentOrder.begin(), currentOrder.end(),
                        bestOrder.begin(), bestOrder.end());
                if (successCount > bestSuccessCount ||
                    (successCount == bestSuccessCount &&
                     totalPathLen < bestTotalPathLen) ||
                    (successCount == bestSuccessCount &&
                     totalPathLen == bestTotalPathLen &&
                     preferLexicographically)) {
                  bestSuccessCount = successCount;
                  bestTotalPathLen = totalPathLen;
                  bestOrder.assign(currentOrder.begin(), currentOrder.end());
                }
                return;
              }

              const size_t edgeBranchLimit = std::min<size_t>(
                  remainingEdges.size(), opts.routing.moduleOutputBranchLimit);
              for (size_t idx = 0; idx < edgeBranchLimit; ++idx) {
                IdIndex groupedEdgeId = remainingEdges[idx];
                const Edge *groupedEdge = dfg.getEdge(groupedEdgeId);
                if (!groupedEdge)
                  continue;
                IdIndex groupedSrcHwPort =
                    state.swPortToHwPort[groupedEdge->srcPort];
                IdIndex groupedDstHwPort =
                    state.swPortToHwPort[groupedEdge->dstPort];
                if (groupedSrcHwPort == INVALID_ID ||
                    groupedDstHwPort == INVALID_ID)
                  continue;

                currentOrder.push_back(groupedEdgeId);
                IdIndex savedEdge = remainingEdges[idx];
                remainingEdges.erase(remainingEdges.begin() + idx);
                auto demandCheckpoint = savePresentDemand();
                auto relaxedCheckpoint = saveRelaxedRouting();
                bool mappedThisEdge = false;

                const auto &choices = firstHopChoices[groupedEdgeId];
                for (IdIndex firstHop : choices) {
                  auto savepoint = state.beginSavepoint();
                  auto groupedPath =
                      findPath(groupedSrcHwPort, groupedDstHwPort, groupedEdgeId,
                               state, dfg, adg, routingOutputHistory, firstHop,
                               congestion, relaxedRouting);
                  if (groupedPath.empty()) {
                    state.rollbackSavepoint(savepoint);
                    continue;
                  }
                  if (state.mapEdge(groupedEdgeId, groupedPath, dfg, adg) !=
                      ActionResult::Success) {
                    state.rollbackSavepoint(savepoint);
                    restorePresentDemand(demandCheckpoint);
                    continue;
                  }
                  mappedThisEdge = true;
                  commitRoutedPath(groupedEdgeId, groupedPath);
                  searchBoundaryGroup(remainingEdges, successCount + 1,
                                      totalPathLen + groupedPath.size());
                  state.rollbackSavepoint(savepoint);
                  restorePresentDemand(demandCheckpoint);
                  restoreRelaxedRouting(relaxedCheckpoint);
                }

                if (!mappedThisEdge)
                  searchBoundaryGroup(remainingEdges, successCount, totalPathLen);

                remainingEdges.insert(remainingEdges.begin() + idx, savedEdge);
                currentOrder.pop_back();
                restorePresentDemand(demandCheckpoint);
                restoreRelaxedRouting(relaxedCheckpoint);
              }
            };

        searchBoundaryGroup(currentRemaining, 0, 0);
        state.restore(boundaryCheckpoint);
        restoreRelaxedRouting(boundaryRelaxedCheckpoint);

        if (bestOrder.empty())
          bestOrder.assign(outputEdges.begin(), outputEdges.end());

        for (IdIndex groupedEdgeId : outputEdges)
          processedSinkEdges.insert(groupedEdgeId);

        for (IdIndex groupedEdgeId : bestOrder) {
          ++total;
          const Edge *groupedEdge = dfg.getEdge(groupedEdgeId);
          if (!groupedEdge) {
            allRouted = false;
            continue;
          }
          IdIndex groupedSrcHwPort = state.swPortToHwPort[groupedEdge->srcPort];
          IdIndex groupedDstHwPort = state.swPortToHwPort[groupedEdge->dstPort];
          auto &choices = firstHopChoices[groupedEdgeId];
          llvm::SmallVector<IdIndex, 8> groupedPath;
          for (IdIndex firstHop : choices) {
            groupedPath = findPath(groupedSrcHwPort, groupedDstHwPort,
                                   groupedEdgeId, state, dfg, adg,
                                   routingOutputHistory, firstHop,
                                   congestion, relaxedRouting);
            if (!groupedPath.empty())
              break;
          }
          if (groupedPath.empty()) {
            allRouted = false;
            continue;
          }
          auto mapResult = state.mapEdge(groupedEdgeId, groupedPath, dfg, adg);
          if (mapResult != ActionResult::Success) {
            allRouted = false;
            continue;
          }
          commitRoutedPath(groupedEdgeId, groupedPath);
          ++routed;
        }
        continue;
      }
      processedSinkNodes.insert(dstSwNodeId);
    }

    if (routing_detail::isNonRoutingOutputPort(srcHwPort, adg)) {
      if (processedFanoutSources.contains(srcHwPort))
        continue;

      llvm::SmallVector<IdIndex, 8> siblingEdges;
      collectUnroutedSiblingEdges(srcHwPort, edgeOrder, state, dfg, edgeKinds,
                                  siblingEdges);
      if (siblingEdges.size() > 1) {
        processedFanoutSources.insert(srcHwPort);

        IdIndex lockedFirstHop =
            routing_detail::getLockedFirstHopForSource(edgeId, srcHwPort, state);
        llvm::SmallVector<IdIndex, 8> firstHopCandidates;
        if (lockedFirstHop != INVALID_ID) {
          firstHopCandidates.push_back(lockedFirstHop);
        } else if (auto it = connectivity.outToIn.find(srcHwPort);
                   it != connectivity.outToIn.end()) {
          firstHopCandidates.append(it->second.begin(), it->second.end());
          llvm::sort(firstHopCandidates);
          firstHopCandidates.erase(
              std::unique(firstHopCandidates.begin(), firstHopCandidates.end()),
              firstHopCandidates.end());
        }
        if (!firstHopCandidates.empty() && firstHopCandidates.front() != INVALID_ID)
          rankSharedFirstHopChoices(siblingEdges, firstHopCandidates,
                                    opts.routing.sourceFanoutFirstHopLimit);

        unsigned bestSuccessCount = 0;
        size_t bestTotalPathLen = std::numeric_limits<size_t>::max();
        IdIndex bestFirstHop = INVALID_ID;

        for (IdIndex firstHop : firstHopCandidates) {
          auto demandCheckpoint = savePresentDemand();
          auto relaxedCheckpoint = saveRelaxedRouting();
          unsigned successCount = 0;
          size_t totalPathLen = 0;
          auto savepoint = state.beginSavepoint();
          for (IdIndex siblingEdgeId : siblingEdges) {
            const Edge *siblingEdge = dfg.getEdge(siblingEdgeId);
            if (!siblingEdge)
              continue;
            IdIndex siblingDstHwPort = state.swPortToHwPort[siblingEdge->dstPort];
            if (siblingDstHwPort == INVALID_ID)
              continue;
            auto siblingPath = findPath(srcHwPort, siblingDstHwPort,
                                        siblingEdgeId, state, dfg, adg,
                                        routingOutputHistory, firstHop,
                                        congestion, relaxedRouting);
            if (siblingPath.empty())
              break;
            if (state.mapEdge(siblingEdgeId, siblingPath, dfg, adg) !=
                ActionResult::Success)
              break;
            commitRoutedPath(siblingEdgeId, siblingPath);
            ++successCount;
            totalPathLen += siblingPath.size();
          }
          state.rollbackSavepoint(savepoint);
          restorePresentDemand(demandCheckpoint);
          restoreRelaxedRouting(relaxedCheckpoint);

          if (successCount > bestSuccessCount ||
              (successCount == bestSuccessCount &&
               totalPathLen < bestTotalPathLen) ||
              (successCount == bestSuccessCount &&
               totalPathLen == bestTotalPathLen &&
               (bestFirstHop == INVALID_ID || firstHop < bestFirstHop))) {
            bestSuccessCount = successCount;
            bestTotalPathLen = totalPathLen;
            bestFirstHop = firstHop;
          }
        }

        for (IdIndex siblingEdgeId : siblingEdges) {
          ++total;
          const Edge *siblingEdge = dfg.getEdge(siblingEdgeId);
          if (!siblingEdge) {
            allRouted = false;
            continue;
          }
          IdIndex siblingDstHwPort = state.swPortToHwPort[siblingEdge->dstPort];
          auto siblingPath =
              bestFirstHop == INVALID_ID
                  ? llvm::SmallVector<IdIndex, 8>()
                  : findPath(srcHwPort, siblingDstHwPort, siblingEdgeId, state,
                             dfg, adg, routingOutputHistory, bestFirstHop,
                             congestion, relaxedRouting);
          if (siblingPath.empty()) {
            allRouted = false;
            continue;
          }
          auto mapResult = state.mapEdge(siblingEdgeId, siblingPath, dfg, adg);
          if (mapResult != ActionResult::Success) {
            allRouted = false;
            continue;
          }
          commitRoutedPath(siblingEdgeId, siblingPath);
          ++routed;
        }
        continue;
      }
    }

    total++;

    auto path = findPath(srcHwPort, dstHwPort, edgeId, state, dfg, adg,
                         routingOutputHistory, INVALID_ID, congestion,
                         relaxedRouting);
    if (path.empty()) {
      allRouted = false;
      continue;
    }

    auto mapResult = state.mapEdge(edgeId, path, dfg, adg);
    if (mapResult != ActionResult::Success) {
      allRouted = false;
      continue;
    }
    commitRoutedPath(edgeId, path);
    routed++;
  }

  return allRouted;
}

bool Mapper::runRouting(MappingState &state, const Graph &dfg,
                        const Graph &adg,
                        llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                        const Options &opts) {
  if (shouldStopForBudget("routing"))
    return false;

  // Build edge order: memory edges first.
  std::vector<IdIndex> memEdges, otherEdges;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *e = dfg.getEdge(i);
    if (!e)
      continue;

    bool isMemEdge = false;
    for (IdIndex swPort : {e->srcPort, e->dstPort}) {
      const Port *p = dfg.getPort(swPort);
      if (p && p->parentNode != INVALID_ID) {
        const Node *n = dfg.getNode(p->parentNode);
        if (n && (n->kind == Node::ModuleInputNode ||
                  n->kind == Node::ModuleOutputNode ||
                  getNodeAttrStr(n, "op_name").contains("extmemory") ||
                  getNodeAttrStr(n, "op_name").contains("load") ||
                  getNodeAttrStr(n, "op_name").contains("store")))
          isMemEdge = true;
      }
    }

    if (isMemEdge)
      memEdges.push_back(i);
    else
      otherEdges.push_back(i);
  }

  auto edgeOrderLess = [&](IdIndex lhs, IdIndex rhs) {
    unsigned lhsPriority = getRoutingPriority(lhs, dfg, opts);
    unsigned rhsPriority = getRoutingPriority(rhs, dfg, opts);
    if (lhsPriority != rhsPriority)
      return lhsPriority < rhsPriority;
    return lhs < rhs;
  };
  llvm::stable_sort(memEdges, edgeOrderLess);
  llvm::stable_sort(otherEdges, edgeOrderLess);

  std::vector<IdIndex> edgeOrder;
  edgeOrder.reserve(memEdges.size() + otherEdges.size());
  edgeOrder.insert(edgeOrder.end(), memEdges.begin(), memEdges.end());
  edgeOrder.insert(edgeOrder.end(), otherEdges.begin(), otherEdges.end());

  llvm::DenseMap<IdIndex, double> routingOutputHistory;

  unsigned routed = 0;
  unsigned total = 0;
  bool allRouted = routeOnePass(state, dfg, adg, edgeKinds, edgeOrder,
                                routingOutputHistory, routed, total, opts);
  auto initialStats =
      mapper_detail::computeRoutingEdgeStats(state, dfg, edgeKinds);

  auto computeTotalPathLen = [&](const MappingState &routingState) -> size_t {
    size_t totalPathLen = 0;
    for (const auto &path : routingState.swEdgeToHwPaths)
      totalPathLen += path.size();
    return totalPathLen;
  };

  MappingState::Checkpoint bestCheckpoint = state.save();
  unsigned bestRouted = routed;
  unsigned bestTotal = total;
  size_t bestTotalPathLen = computeTotalPathLen(state);
  bool bestAllRouted = allRouted;

  llvm::outs() << "  Initial routing: router " << routed << "/" << total
               << ", overall " << initialStats.routedOverallEdges << "/"
               << initialStats.overallEdges << ", prebound "
               << initialStats.directBindingEdges << "\n";

  // Failed-edge driven repair: only rip up the neighborhood around failed
  // edges so later place/route rounds can preserve working regions.
  const unsigned maxRipupPasses = std::max(1u, opts.selectiveRipupPasses);
  for (unsigned pass = 0; pass < maxRipupPasses && !allRouted; ++pass) {
    if (shouldStopForBudget("routing"))
      break;
    // Collect failed edge IDs.
    std::vector<IdIndex> failedEdges;
    for (IdIndex edgeId : edgeOrder) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      if (edgeId < edgeKinds.size() &&
          (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
           edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
        continue;
      if (!isEdgeRoutable(edgeId, state, dfg))
        continue;
      // Check if this edge has no route.
      if (edgeId >= state.swEdgeToHwPaths.size() ||
          state.swEdgeToHwPaths[edgeId].empty()) {
        failedEdges.push_back(edgeId);
      }
    }

    if (failedEdges.empty())
      break;

    llvm::stable_sort(failedEdges, edgeOrderLess);
    llvm::SmallVector<IdIndex, 16> ripupEdges;
    collectSelectiveRipupEdges(failedEdges, edgeOrder, state, dfg, adg,
                               edgeKinds, opts, ripupEdges);

    llvm::outs() << "  Repair pass " << (pass + 1) << ": " << failedEdges.size()
                 << " failed edges, ripping up " << ripupEdges.size()
                 << " routes\n";
    if (opts.verbose)
      dumpFailedEdgeDiagnostics(failedEdges, state, dfg,
                                opts.routing.failedEdgeDiagnosticLimit);

    for (IdIndex edgeId : ripupEdges) {
      if (edgeId >= state.swEdgeToHwPaths.size())
        continue;
      const auto &path = state.swEdgeToHwPaths[edgeId];
      if (path.empty())
        continue;
      if (isDirectBindingPath(path))
        continue;
      for (IdIndex portId : path) {
        if (!routing_detail::isRoutingCrossbarOutputPort(portId, adg))
          continue;
        routingOutputHistory[portId] += opts.routing.ripupHistoryBump;
      }
    }

    for (IdIndex edgeId : ripupEdges)
      state.unmapEdge(edgeId, dfg, adg);

    std::vector<IdIndex> remainingEdges;
    llvm::DenseSet<IdIndex> ripupSet;
    for (IdIndex eid : ripupEdges)
      ripupSet.insert(eid);
    for (IdIndex eid : edgeOrder) {
      if (!ripupSet.count(eid))
        remainingEdges.push_back(eid);
    }
    llvm::stable_sort(remainingEdges, edgeOrderLess);

    std::vector<IdIndex> newOrder;
    newOrder.reserve(ripupEdges.size() + remainingEdges.size());
    newOrder.insert(newOrder.end(), ripupEdges.begin(), ripupEdges.end());
    newOrder.insert(newOrder.end(), remainingEdges.begin(),
                    remainingEdges.end());

    allRouted = routeOnePass(state, dfg, adg, edgeKinds, newOrder,
                             routingOutputHistory, routed, total, opts);
    auto repairStats =
        mapper_detail::computeRoutingEdgeStats(state, dfg, edgeKinds);
    llvm::outs() << "  Repair pass " << (pass + 1) << " result: router "
                 << routed << "/" << total << ", overall "
                 << repairStats.routedOverallEdges << "/"
                 << repairStats.overallEdges << ", prebound "
                 << repairStats.directBindingEdges << "\n";

    size_t totalPathLen = computeTotalPathLen(state);
    if (routed > bestRouted ||
        (routed == bestRouted && totalPathLen < bestTotalPathLen)) {
      bestCheckpoint = state.save();
      bestRouted = routed;
      bestTotal = total;
      bestTotalPathLen = totalPathLen;
      bestAllRouted = allRouted;
    }
  }

  state.restore(bestCheckpoint);
  auto bestStats = mapper_detail::computeRoutingEdgeStats(state, dfg, edgeKinds);
  llvm::outs() << "  Final routing: router " << bestRouted << "/"
               << bestTotal << ", overall " << bestStats.routedOverallEdges
               << "/" << bestStats.overallEdges << ", prebound "
               << bestStats.directBindingEdges << "\n";
  return bestAllRouted;
}

} // namespace fcc

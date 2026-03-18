#include "fcc/Mapper/Mapper.h"
#include "MapperRoutingInternal.h"

#include "fcc/Mapper/TechMapper.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>

namespace fcc {

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers used only by routeOnePass / runRouting
// ---------------------------------------------------------------------------

namespace {

bool isNonRoutingOutputPort(IdIndex portId, const Graph &adg) {
  const Port *port = adg.getPort(portId);
  if (!port || port->direction != Port::Output || port->parentNode == INVALID_ID)
    return false;
  const Node *owner = adg.getNode(port->parentNode);
  return owner && !routing_detail::isRoutingNode(owner);
}

IdIndex getLockedFirstHopForSource(IdIndex swEdgeId, IdIndex srcHwPort,
                                   const MappingState &state) {
  for (IdIndex otherEdgeId = 0;
       otherEdgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++otherEdgeId) {
    if (otherEdgeId == swEdgeId)
      continue;
    const auto &otherPath = state.swEdgeToHwPaths[otherEdgeId];
    if (otherPath.size() < 2 || otherPath.front() != srcHwPort)
      continue;
    return otherPath[1];
  }
  return INVALID_ID;
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

void collectUnroutedBoundarySinkEdges(
    const std::vector<IdIndex> &edgeOrder, const MappingState &state,
    const Graph &dfg, llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
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
    if (edge->dstPort >= state.swPortToHwPort.size() ||
        state.swPortToHwPort[edge->dstPort] == INVALID_ID)
      continue;
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!dstPort || dstPort->parentNode == INVALID_ID)
      continue;
    const Node *dstNode = dfg.getNode(dstPort->parentNode);
    if (!isBoundarySinkNode(dstNode))
      continue;
    groupedEdges.push_back(edgeId);
  }
}

unsigned getRoutingPriority(IdIndex edgeId, const Graph &dfg) {
  const Edge *edge = dfg.getEdge(edgeId);
  if (!edge)
    return 100;

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
    return 0;
  if (dstNode && dstNode->kind == Node::ModuleOutputNode)
    return 1;
  if (srcOp == "handshake.load" || srcOp == "handshake.store")
    return 2;
  if (routing_detail::isSoftwareMemoryInterfaceOpName(srcOp))
    return 3;
  if (srcOp == "handshake.load" || dstOp == "handshake.load" ||
      srcOp == "handshake.store" || dstOp == "handshake.store")
    return 4;
  return 5;
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
                          unsigned &routed, unsigned &total) {
  bool allRouted = true;
  routed = 0;
  total = 0;
  llvm::DenseSet<IdIndex> processedFanoutSources;
  bool processedBoundarySinks = false;
  llvm::DenseSet<IdIndex> processedBoundarySinkEdges;

  for (IdIndex edgeId : edgeOrder) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (processedBoundarySinkEdges.contains(edgeId))
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

    if (isBoundarySinkNode(dstSwNode) && !processedBoundarySinks) {
      llvm::SmallVector<IdIndex, 8> outputEdges;
      collectUnroutedBoundarySinkEdges(edgeOrder, state, dfg, edgeKinds,
                                       outputEdges);
      if (outputEdges.size() > 1 && outputEdges.size() <= 6) {
        processedBoundarySinks = true;
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
          if (isNonRoutingOutputPort(groupedSrcHwPort, adg)) {
            IdIndex lockedFirstHop =
                getLockedFirstHopForSource(groupedEdgeId, groupedSrcHwPort, state);
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
          firstHopChoices[groupedEdgeId] = std::move(choices);
        }

        auto boundaryCheckpoint = state.save();
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

              for (size_t idx = 0; idx < remainingEdges.size(); ++idx) {
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
                auto checkpoint = state.save();
                bool mappedThisEdge = false;

                const auto &choices = firstHopChoices[groupedEdgeId];
                for (IdIndex firstHop : choices) {
                  auto groupedPath =
                      findPath(groupedSrcHwPort, groupedDstHwPort, groupedEdgeId,
                               state, dfg, adg, routingOutputHistory, firstHop);
                  if (groupedPath.empty())
                    continue;
                  if (state.mapEdge(groupedEdgeId, groupedPath, dfg, adg) !=
                      ActionResult::Success) {
                    state.restore(checkpoint);
                    continue;
                  }
                  mappedThisEdge = true;
                  searchBoundaryGroup(remainingEdges, successCount + 1,
                                      totalPathLen + groupedPath.size());
                  state.restore(checkpoint);
                }

                if (!mappedThisEdge)
                  searchBoundaryGroup(remainingEdges, successCount, totalPathLen);

                remainingEdges.insert(remainingEdges.begin() + idx, savedEdge);
                currentOrder.pop_back();
                state.restore(checkpoint);
              }
            };

        searchBoundaryGroup(currentRemaining, 0, 0);
        state.restore(boundaryCheckpoint);

        if (bestOrder.empty())
          bestOrder.assign(outputEdges.begin(), outputEdges.end());

        for (IdIndex groupedEdgeId : outputEdges)
          processedBoundarySinkEdges.insert(groupedEdgeId);

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
                                   routingOutputHistory, firstHop);
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
          ++routed;
        }
        continue;
      }
      processedBoundarySinks = true;
    }

    if (isNonRoutingOutputPort(srcHwPort, adg)) {
      if (processedFanoutSources.contains(srcHwPort))
        continue;

      llvm::SmallVector<IdIndex, 8> siblingEdges;
      collectUnroutedSiblingEdges(srcHwPort, edgeOrder, state, dfg, edgeKinds,
                                  siblingEdges);
      if (siblingEdges.size() > 1) {
        processedFanoutSources.insert(srcHwPort);

        IdIndex lockedFirstHop =
            getLockedFirstHopForSource(edgeId, srcHwPort, state);
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

        unsigned bestSuccessCount = 0;
        size_t bestTotalPathLen = std::numeric_limits<size_t>::max();
        IdIndex bestFirstHop = INVALID_ID;

        for (IdIndex firstHop : firstHopCandidates) {
          auto checkpoint = state.save();
          unsigned successCount = 0;
          size_t totalPathLen = 0;
          for (IdIndex siblingEdgeId : siblingEdges) {
            const Edge *siblingEdge = dfg.getEdge(siblingEdgeId);
            if (!siblingEdge)
              continue;
            IdIndex siblingDstHwPort = state.swPortToHwPort[siblingEdge->dstPort];
            if (siblingDstHwPort == INVALID_ID)
              continue;
            auto siblingPath = findPath(srcHwPort, siblingDstHwPort,
                                        siblingEdgeId, state, dfg, adg,
                                        routingOutputHistory, firstHop);
            if (siblingPath.empty())
              break;
            if (state.mapEdge(siblingEdgeId, siblingPath, dfg, adg) !=
                ActionResult::Success)
              break;
            ++successCount;
            totalPathLen += siblingPath.size();
          }
          state.restore(checkpoint);

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
                             dfg, adg, routingOutputHistory, bestFirstHop);
          if (siblingPath.empty()) {
            allRouted = false;
            continue;
          }
          auto mapResult = state.mapEdge(siblingEdgeId, siblingPath, dfg, adg);
          if (mapResult != ActionResult::Success) {
            allRouted = false;
            continue;
          }
          ++routed;
        }
        continue;
      }
    }

    total++;

    auto path = findPath(srcHwPort, dstHwPort, edgeId, state, dfg, adg,
                         routingOutputHistory);
    if (path.empty()) {
      allRouted = false;
      continue;
    }

    auto mapResult = state.mapEdge(edgeId, path, dfg, adg);
    if (mapResult != ActionResult::Success) {
      allRouted = false;
      continue;
    }
    routed++;
  }

  return allRouted;
}

bool Mapper::runRouting(MappingState &state, const Graph &dfg,
                        const Graph &adg,
                        llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                        int seed) {
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
    unsigned lhsPriority = getRoutingPriority(lhs, dfg);
    unsigned rhsPriority = getRoutingPriority(rhs, dfg);
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
                                routingOutputHistory, routed, total);

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

  llvm::outs() << "  Initial routing: " << routed << "/" << total << " edges\n";

  // Rip-up and reroute: if some edges failed, rip up ALL routes and retry
  // with different ordering. Repeat up to maxRipupPasses times.
  const int maxRipupPasses = 10;
  for (int pass = 0; pass < maxRipupPasses && !allRouted; ++pass) {
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

    llvm::outs() << "  Rip-up pass " << (pass + 1) << ": " << failedEdges.size()
                 << " failed edges, ripping up all routes\n";

    for (IdIndex edgeId : edgeOrder) {
      if (edgeId >= state.swEdgeToHwPaths.size())
        continue;
      const auto &path = state.swEdgeToHwPaths[edgeId];
      if (path.empty())
        continue;
      if (path.size() == 2 && path[0] == path[1])
        continue;
      for (IdIndex portId : path) {
        if (!routing_detail::isRoutingCrossbarOutputPort(portId, adg))
          continue;
        routingOutputHistory[portId] += 0.35;
      }
    }

    // Rip up all routed edges (clear paths) but keep pre-routed memref edges.
    for (IdIndex edgeId : edgeOrder) {
      if (edgeId < state.swEdgeToHwPaths.size() &&
          !state.swEdgeToHwPaths[edgeId].empty()) {
        // Check if this is a pre-routed memref binding (path length 2 with
        // identical ports). Keep those intact.
        auto &path = state.swEdgeToHwPaths[edgeId];
        if (path.size() == 2 && path[0] == path[1])
          continue;
        state.swEdgeToHwPaths[edgeId].clear();
      }
    }

    // Rebuild edge order: failed edges first, then others. Within each group,
    // keep a deterministic heuristic order so a fixed seed implies a stable
    // routing result.
    llvm::stable_sort(failedEdges, edgeOrderLess);
    std::vector<IdIndex> remainingEdges;
    llvm::DenseSet<IdIndex> failedSet;
    for (IdIndex eid : failedEdges)
      failedSet.insert(eid);
    for (IdIndex eid : edgeOrder) {
      if (!failedSet.count(eid))
        remainingEdges.push_back(eid);
    }
    llvm::stable_sort(remainingEdges, edgeOrderLess);

    std::vector<IdIndex> newOrder;
    newOrder.reserve(failedEdges.size() + remainingEdges.size());
    newOrder.insert(newOrder.end(), failedEdges.begin(), failedEdges.end());
    newOrder.insert(newOrder.end(), remainingEdges.begin(),
                    remainingEdges.end());

    allRouted = routeOnePass(state, dfg, adg, edgeKinds, newOrder,
                             routingOutputHistory, routed, total);
    llvm::outs() << "  Rip-up pass " << (pass + 1) << " result: " << routed
                 << "/" << total << " edges\n";

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
  llvm::outs() << "  Final routing: " << bestRouted << "/" << bestTotal
               << " edges\n";
  return bestAllRouted;
}

} // namespace fcc

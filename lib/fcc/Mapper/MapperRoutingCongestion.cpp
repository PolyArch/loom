#include "MapperRoutingCongestion.h"
#include "MapperRoutingInternal.h"
#include "MapperInternal.h"
#include "fcc/Mapper/Mapper.h"
#include "fcc/Mapper/TechMapper.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>

namespace fcc {

using namespace mapper_detail;

// ---------------------------------------------------------------------------
// CongestionState method implementations
// ---------------------------------------------------------------------------

void CongestionState::init(const Graph &adg) {
  size_t numPorts = adg.ports.size();
  historicalCongestion.assign(numPorts, 0.0);
  presentDemand.assign(numPorts, 0);
  capacity.assign(numPorts, 0);

  for (IdIndex portId = 0; portId < static_cast<IdIndex>(numPorts); ++portId) {
    if (routing_detail::isRoutingCrossbarOutputPort(portId, adg)) {
      capacity[portId] = 1;
      historicalCongestion[portId] = 0.0;
    }
  }
}

double CongestionState::resourceCost(IdIndex portId) const {
  if (portId >= capacity.size() || capacity[portId] == 0)
    return 0.0;
  double hist = historicalCongestion[portId];
  double present = static_cast<double>(presentDemand[portId]);
  return (1.0 + hist) * (1.0 + present * presentFactor);
}

void CongestionState::commitRoute(llvm::ArrayRef<IdIndex> path,
                                  const Graph &adg) {
  for (IdIndex portId : path) {
    if (routing_detail::isRoutingCrossbarOutputPort(portId, adg))
      ++presentDemand[portId];
  }
}

void CongestionState::uncommitRoute(llvm::ArrayRef<IdIndex> path,
                                    const Graph &adg) {
  for (IdIndex portId : path) {
    if (routing_detail::isRoutingCrossbarOutputPort(portId, adg)) {
      if (presentDemand[portId] > 0)
        --presentDemand[portId];
    }
  }
}

bool CongestionState::hasOveruse() const {
  for (size_t i = 0; i < capacity.size(); ++i) {
    if (capacity[i] > 0 && presentDemand[i] > capacity[i])
      return true;
  }
  return false;
}

void CongestionState::updateHistory() {
  for (size_t i = 0; i < capacity.size(); ++i) {
    if (capacity[i] == 0 || presentDemand[i] == 0)
      continue;
    if (presentDemand[i] > capacity[i]) {
      historicalCongestion[i] +=
          historyIncrement *
          static_cast<double>(presentDemand[i] - capacity[i]);
      continue;
    }
    if (presentDemand[i] == capacity[i]) {
      historicalCongestion[i] += historyIncrement * saturationPenalty;
    }
  }
}

void CongestionState::decayHistory(double factor) {
  if (factor >= 1.0)
    return;
  for (double &hist : historicalCongestion)
    hist *= factor;
}

void CongestionState::resetPresentDemand() {
  std::fill(presentDemand.begin(), presentDemand.end(), 0);
}

// ---------------------------------------------------------------------------
// Mapper::runNegotiatedRouting
// ---------------------------------------------------------------------------

bool Mapper::runNegotiatedRouting(MappingState &state, const Graph &dfg,
                                  const Graph &adg,
                                  llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                  const Options &opts) {
  if (opts.negotiatedRoutingPasses == 0)
    return runRouting(state, dfg, adg, edgeKinds, opts);

  CongestionState congestion;
  congestion.init(adg);
  congestion.historyIncrement = opts.congestionHistoryFactor;
  congestion.historyScale = opts.congestionHistoryScale;
  congestion.presentFactor = opts.congestionPresentFactor;
  congestion.saturationPenalty = opts.congestion.saturationPenalty;

  // Build edge order (same as runRouting).
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

  // Sort by routing priority (use a local lambda similar to runRouting).
  auto getRoutingPriority = [&](IdIndex edgeId) -> unsigned {
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
  };

  auto edgeOrderLess = [&](IdIndex lhs, IdIndex rhs) {
    unsigned lhsPriority = getRoutingPriority(lhs);
    unsigned rhsPriority = getRoutingPriority(rhs);
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

  auto bestCheckpoint = state.save();
  unsigned bestRouted = 0;
  unsigned bestTotal = 0;
  size_t bestPathLen = std::numeric_limits<size_t>::max();
  unsigned stagnantIterations = 0;

  llvm::DenseMap<IdIndex, double> routingOutputHistory;
  auto emitBestSnapshot = [&](llvm::StringRef trigger) {
    auto checkpoint = state.save();
    state.restore(bestCheckpoint);
    maybeEmitProgressSnapshot(state, edgeKinds, trigger, opts);
    state.restore(checkpoint);
  };

  for (unsigned iter = 0; iter < opts.negotiatedRoutingPasses; ++iter) {
    if (shouldStopForBudget("negotiated routing"))
      break;
    state.clearRoutes(dfg, adg);
    congestion.resetPresentDemand();

    unsigned routed = 0;
    unsigned total = 0;
    bool allRouted = routeOnePass(state, dfg, adg, edgeKinds, edgeOrder,
                                  routingOutputHistory, routed, total, opts,
                                  &congestion);
    auto stats = computeRoutingEdgeStats(state, dfg, edgeKinds);

    size_t totalPathLen = 0;
    for (const auto &path : state.swEdgeToHwPaths)
      totalPathLen += path.size();

    if (allRouted) {
      bestCheckpoint = state.save();
      bestRouted = routed;
      bestTotal = total;
      bestPathLen = totalPathLen;
      llvm::outs() << "  Negotiated routing converged at iteration "
                   << (iter + 1) << ": router " << routed << "/" << total
                   << ", overall " << stats.routedOverallEdges << "/"
                   << stats.overallEdges << ", prebound "
                   << stats.directBindingEdges << "\n";
      state.restore(bestCheckpoint);
      return true;
    }

    if (routed > bestRouted ||
        (routed == bestRouted && totalPathLen < bestPathLen)) {
      bestCheckpoint = state.save();
      bestRouted = routed;
      bestTotal = total;
      bestPathLen = totalPathLen;
      stagnantIterations = 0;
    } else {
      ++stagnantIterations;
    }

    llvm::outs() << "  Negotiated routing iteration " << (iter + 1)
                 << ": router " << routed << "/" << total << ", overall "
                 << stats.routedOverallEdges << "/" << stats.overallEdges
                 << ", prebound " << stats.directBindingEdges << "\n";
    emitBestSnapshot("negotiated-iter");

    if (opts.congestion.routingOutputHistoryDecay < 1.0) {
      for (auto &entry : routingOutputHistory)
        entry.second *= opts.congestion.routingOutputHistoryDecay;
    }
    if (opts.congestion.routingOutputHistoryBump > 0.0) {
      for (size_t i = 0; i < congestion.capacity.size(); ++i) {
        if (congestion.capacity[i] == 0)
          continue;
        if (congestion.presentDemand[i] >= congestion.capacity[i])
          routingOutputHistory[static_cast<IdIndex>(i)] +=
              opts.congestion.routingOutputHistoryBump;
      }
    }

    congestion.decayHistory(opts.congestion.historyDecay);
    congestion.updateHistory();
    congestion.historyIncrement =
        std::min(congestion.historyIncrement * congestion.historyScale,
                 opts.congestion.historyIncrementCap);

    if (opts.congestion.earlyTerminationWindow > 0 &&
        stagnantIterations >= opts.congestion.earlyTerminationWindow) {
      llvm::outs() << "  Negotiated routing early stop after "
                   << stagnantIterations << " non-improving iterations\n";
      break;
    }
  }

  state.restore(bestCheckpoint);
  auto bestStats = computeRoutingEdgeStats(state, dfg, edgeKinds);
  llvm::outs() << "  Negotiated routing best: router " << bestRouted << "/"
               << bestTotal << ", overall " << bestStats.routedOverallEdges
               << "/" << bestStats.overallEdges << ", prebound "
               << bestStats.directBindingEdges << "\n";
  return bestRouted == bestTotal && bestTotal != 0;
}

} // namespace fcc

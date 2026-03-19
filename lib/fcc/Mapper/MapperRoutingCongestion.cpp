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
      historicalCongestion[i] += historyIncrement * 0.5;
    }
  }
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

  llvm::DenseMap<IdIndex, double> routingOutputHistory;

  for (unsigned iter = 0; iter < opts.negotiatedRoutingPasses; ++iter) {
    state.clearRoutes(adg);
    congestion.resetPresentDemand();

    unsigned routed = 0;
    unsigned total = 0;
    bool allRouted = routeOnePass(state, dfg, adg, edgeKinds, edgeOrder,
                                  routingOutputHistory, routed, total,
                                  &congestion);

    // Commit routes to congestion state.
    for (const auto &path : state.swEdgeToHwPaths) {
      if (!path.empty() && !(path.size() == 2 && path[0] == path[1]))
        congestion.commitRoute(path, adg);
    }

    size_t totalPathLen = 0;
    for (const auto &path : state.swEdgeToHwPaths)
      totalPathLen += path.size();

    if (allRouted) {
      bestCheckpoint = state.save();
      bestRouted = routed;
      bestTotal = total;
      bestPathLen = totalPathLen;
      llvm::outs() << "  Negotiated routing converged at iteration "
                   << (iter + 1) << ": " << routed << "/" << total
                   << " edges\n";
      state.restore(bestCheckpoint);
      return true;
    }

    if (routed > bestRouted ||
        (routed == bestRouted && totalPathLen < bestPathLen)) {
      bestCheckpoint = state.save();
      bestRouted = routed;
      bestTotal = total;
      bestPathLen = totalPathLen;
    }

    llvm::outs() << "  Negotiated routing iteration " << (iter + 1) << ": "
                 << routed << "/" << total << " edges\n";

    congestion.updateHistory();
    congestion.historyIncrement =
        std::min(congestion.historyIncrement * congestion.historyScale, 8.0);
  }

  state.restore(bestCheckpoint);
  llvm::outs() << "  Negotiated routing best: " << bestRouted << "/"
               << bestTotal << " edges\n";
  return bestRouted == bestTotal && bestTotal != 0;
}

} // namespace fcc

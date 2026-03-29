#include "MapperCongestionEstimator.h"
#include "MapperInternal.h"
#include "MapperRoutingInternal.h"

#include <algorithm>
#include <cmath>

namespace loom {

using namespace mapper_detail;

namespace {

bool isPortInTopologyCorridor(IdIndex srcHwNode, IdIndex dstHwNode,
                              IdIndex corridorHwNode,
                              const TopologyModel *topologyModel) {
  if (!topologyModel)
    return false;
  unsigned endToEnd = topologyModel->placementDistance(srcHwNode, dstHwNode);
  unsigned viaCandidate =
      topologyModel->placementDistance(srcHwNode, corridorHwNode) +
      topologyModel->placementDistance(corridorHwNode, dstHwNode);
  unsigned slack = std::max<unsigned>(2U, endToEnd / 2U);
  return viaCandidate <= endToEnd + slack;
}

} // namespace

void CongestionEstimator::estimate(const MappingState &state, const Graph &dfg,
                                   const Graph &adg,
                                   const ADGFlattener &flattener) {
  switchOutputDemand.clear();
  const TopologyModel *topologyModel = getActiveTopologyModel();

  // Fan-out inflation factor: high-fan-out sources get wider corridors.
  constexpr double fanOutInflationFactor = 0.25;

  // Pre-compute per-node fan-out (number of outgoing edges).
  llvm::DenseMap<IdIndex, unsigned> nodeFanOut;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *e = dfg.getEdge(eid);
    if (!e)
      continue;
    const Port *sp = dfg.getPort(e->srcPort);
    if (sp && sp->parentNode != INVALID_ID)
      nodeFanOut[sp->parentNode]++;
  }

  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
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
    IdIndex srcHwNode = state.swNodeToHwNode[srcSwNode];
    IdIndex dstHwNode = state.swNodeToHwNode[dstSwNode];
    if (srcHwNode == INVALID_ID || dstHwNode == INVALID_ID)
      continue;

    double weight = classifyEdgePlacementWeight(dfg, edgeId);

    // Inflate weight by source fan-out for wider demand corridors.
    unsigned fanOut = nodeFanOut.lookup(srcSwNode);
    if (fanOut > 1)
      weight *= (1.0 + fanOutInflationFactor * static_cast<double>(fanOut - 1));

    llvm::SmallVector<IdIndex, 16> corridorPorts;
    for (IdIndex portId = 0; portId < static_cast<IdIndex>(adg.ports.size());
         ++portId) {
      if (!routing_detail::isRoutingCrossbarOutputPort(portId, adg))
        continue;
      const Port *port = adg.getPort(portId);
      if (!port || port->parentNode == INVALID_ID)
        continue;
      bool includePort = false;
      if (topologyModel) {
        includePort = isPortInTopologyCorridor(srcHwNode, dstHwNode,
                                               port->parentNode,
                                               topologyModel);
      } else {
        auto [srcRow, srcCol] = flattener.getNodeGridPos(srcHwNode);
        auto [dstRow, dstCol] = flattener.getNodeGridPos(dstHwNode);
        auto [row, col] = flattener.getNodeGridPos(port->parentNode);
        if (srcRow >= 0 && dstRow >= 0 && row >= 0) {
          int minRow = std::min(srcRow, dstRow);
          int maxRow = std::max(srcRow, dstRow);
          int minCol = std::min(srcCol, dstCol);
          int maxCol = std::max(srcCol, dstCol);
          includePort =
              row >= minRow && row <= maxRow && col >= minCol && col <= maxCol;
        }
      }
      if (includePort)
        corridorPorts.push_back(portId);
    }

    if (corridorPorts.empty())
      continue;

    double share = weight / static_cast<double>(corridorPorts.size());
    for (IdIndex portId : corridorPorts)
      switchOutputDemand[portId] += share;
  }
}

double CongestionEstimator::demandCapacityRatio(
    IdIndex srcHwNode, IdIndex dstHwNode, const Graph &adg,
    const ADGFlattener &flattener) const {
  const TopologyModel *topologyModel = getActiveTopologyModel();

  double totalRatio = 0.0;
  for (const auto &entry : switchOutputDemand) {
    IdIndex portId = entry.first;
    double demand = entry.second;
    const Port *port = adg.getPort(portId);
    if (!port || port->parentNode == INVALID_ID)
      continue;
    bool includePort = false;
    if (topologyModel) {
      includePort = isPortInTopologyCorridor(srcHwNode, dstHwNode,
                                             port->parentNode, topologyModel);
    } else {
      auto [srcRow, srcCol] = flattener.getNodeGridPos(srcHwNode);
      auto [dstRow, dstCol] = flattener.getNodeGridPos(dstHwNode);
      auto [row, col] = flattener.getNodeGridPos(port->parentNode);
      if (srcRow >= 0 && dstRow >= 0 && row >= 0) {
        int minRow = std::min(srcRow, dstRow);
        int maxRow = std::max(srcRow, dstRow);
        int minCol = std::min(srcCol, dstCol);
        int maxCol = std::max(srcCol, dstCol);
        includePort =
            row >= minRow && row <= maxRow && col >= minCol && col <= maxCol;
      }
    }
    if (includePort)
      totalRatio += demand;
  }
  return totalRatio;
}

double CongestionEstimator::totalDemandExcess() const {
  double excess = 0.0;
  for (const auto &entry : switchOutputDemand) {
    double over = entry.second - 1.0;
    if (over > 0.0)
      excess += over;
  }
  return excess;
}

} // namespace loom

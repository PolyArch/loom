#include "MapperCongestionEstimator.h"
#include "MapperInternal.h"
#include "MapperRoutingInternal.h"

#include <algorithm>
#include <cmath>

namespace fcc {

using namespace mapper_detail;

void CongestionEstimator::estimate(const MappingState &state, const Graph &dfg,
                                   const Graph &adg,
                                   const ADGFlattener &flattener) {
  switchOutputDemand.clear();

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

    auto [srcRow, srcCol] = flattener.getNodeGridPos(srcHwNode);
    auto [dstRow, dstCol] = flattener.getNodeGridPos(dstHwNode);
    if (srcRow < 0 || dstRow < 0)
      continue;

    int minRow = std::min(srcRow, dstRow);
    int maxRow = std::max(srcRow, dstRow);
    int minCol = std::min(srcCol, dstCol);
    int maxCol = std::max(srcCol, dstCol);

    double weight = classifyEdgePlacementWeight(dfg, edgeId);

    // Collect routing crossbar output ports within bounding box.
    llvm::SmallVector<IdIndex, 16> bboxPorts;
    for (IdIndex portId = 0; portId < static_cast<IdIndex>(adg.ports.size());
         ++portId) {
      if (!routing_detail::isRoutingCrossbarOutputPort(portId, adg))
        continue;
      const Port *port = adg.getPort(portId);
      if (!port || port->parentNode == INVALID_ID)
        continue;
      auto [row, col] = flattener.getNodeGridPos(port->parentNode);
      if (row < 0)
        continue;
      if (row >= minRow && row <= maxRow && col >= minCol && col <= maxCol)
        bboxPorts.push_back(portId);
    }

    if (bboxPorts.empty())
      continue;

    double share = weight / static_cast<double>(bboxPorts.size());
    for (IdIndex portId : bboxPorts)
      switchOutputDemand[portId] += share;
  }
}

double CongestionEstimator::demandCapacityRatio(
    IdIndex srcHwNode, IdIndex dstHwNode, const Graph &adg,
    const ADGFlattener &flattener) const {
  auto [srcRow, srcCol] = flattener.getNodeGridPos(srcHwNode);
  auto [dstRow, dstCol] = flattener.getNodeGridPos(dstHwNode);
  if (srcRow < 0 || dstRow < 0)
    return 0.0;

  int minRow = std::min(srcRow, dstRow);
  int maxRow = std::max(srcRow, dstRow);
  int minCol = std::min(srcCol, dstCol);
  int maxCol = std::max(srcCol, dstCol);

  double totalRatio = 0.0;
  for (const auto &entry : switchOutputDemand) {
    IdIndex portId = entry.first;
    double demand = entry.second;
    const Port *port = adg.getPort(portId);
    if (!port || port->parentNode == INVALID_ID)
      continue;
    auto [row, col] = flattener.getNodeGridPos(port->parentNode);
    if (row < 0)
      continue;
    if (row >= minRow && row <= maxRow && col >= minCol && col <= maxCol)
      totalRatio += demand; // capacity is 1 per port
  }
  return totalRatio;
}

} // namespace fcc

#include "fcc/Mapper/MappingState.h"
#include "fcc/Mapper/Graph.h"

#include <algorithm>

namespace fcc {

namespace {

bool isSpatialPEFunctionalNode(const Node *node) {
  return node && getNodeAttrStr(node, "resource_class") == "functional" &&
         getNodeAttrStr(node, "pe_kind") == "spatial_pe";
}

bool hasConflictingSpatialPEOccupant(IdIndex targetHwNode, llvm::StringRef peName,
                                     const std::vector<IdIndex> &swNodeToHwNode,
                                     const Graph &adg) {
  if (peName.empty())
    return false;
  for (IdIndex mappedHwNode : swNodeToHwNode) {
    if (mappedHwNode == INVALID_ID || mappedHwNode == targetHwNode)
      continue;
    const Node *otherHwNode = adg.getNode(mappedHwNode);
    if (!isSpatialPEFunctionalNode(otherHwNode))
      continue;
    if (getNodeAttrStr(otherHwNode, "pe_name") == peName)
      return true;
  }
  return false;
}

} // namespace

void MappingState::init(const Graph &dfg, const Graph &adg) {
  size_t swNodes = dfg.nodes.size();
  size_t swPorts = dfg.ports.size();
  size_t swEdges = dfg.edges.size();
  size_t hwNodes = adg.nodes.size();
  size_t hwPorts = adg.ports.size();
  size_t hwEdges = adg.edges.size();

  swNodeToHwNode.assign(swNodes, INVALID_ID);
  swPortToHwPort.assign(swPorts, INVALID_ID);
  swEdgeToHwPaths.resize(swEdges);

  hwNodeToSwNodes.resize(hwNodes);
  hwPortToSwPorts.resize(hwPorts);
  hwEdgeToSwEdges.resize(hwEdges);

  totalCost = 0.0;
}

ActionResult MappingState::mapNode(IdIndex swNode, IdIndex hwNode,
                                    const Graph &dfg, const Graph &adg) {
  if (swNode >= swNodeToHwNode.size() || hwNode >= hwNodeToSwNodes.size())
    return ActionResult::FailedInternalError;

  // Most hardware nodes are exclusive. Memory nodes may host multiple
  // software memories and are capacity-checked by the mapper before mapNode().
  const Node *hwN = adg.getNode(hwNode);
  bool allowMultiOccupancy =
      hwN && getNodeAttrStr(hwN, "resource_class") == "memory";
  if (!allowMultiOccupancy && !hwNodeToSwNodes[hwNode].empty()) {
    return ActionResult::FailedResourceUnavailable;
  }

  if (isSpatialPEFunctionalNode(hwN) &&
      hasConflictingSpatialPEOccupant(hwNode, getNodeAttrStr(hwN, "pe_name"),
                                      swNodeToHwNode, adg)) {
    return ActionResult::FailedHardConstraint;
  }

  swNodeToHwNode[swNode] = hwNode;
  hwNodeToSwNodes[hwNode].push_back(swNode);

  // Auto-map ports: match by position.
  const Node *swN = dfg.getNode(swNode);
  bool skipAutoPorts = false;
  if (hwN && getNodeAttrStr(hwN, "resource_class") == "memory")
    skipAutoPorts = true;

  if (swN && hwN && !skipAutoPorts) {
    for (unsigned i = 0;
         i < swN->inputPorts.size() && i < hwN->inputPorts.size(); ++i) {
      mapPort(swN->inputPorts[i], hwN->inputPorts[i], dfg, adg);
    }
    for (unsigned i = 0;
         i < swN->outputPorts.size() && i < hwN->outputPorts.size(); ++i) {
      mapPort(swN->outputPorts[i], hwN->outputPorts[i], dfg, adg);
    }
  }

  return ActionResult::Success;
}

ActionResult MappingState::unmapNode(IdIndex swNode,
                                      const Graph &dfg, const Graph &adg) {
  if (swNode >= swNodeToHwNode.size())
    return ActionResult::FailedInternalError;

  IdIndex hwNode = swNodeToHwNode[swNode];
  if (hwNode == INVALID_ID)
    return ActionResult::Success;

  // Remove reverse mapping.
  if (hwNode < hwNodeToSwNodes.size()) {
    auto &vec = hwNodeToSwNodes[hwNode];
    vec.erase(std::remove(vec.begin(), vec.end(), swNode), vec.end());
  }

  // Any routed software edge incident to this node becomes invalid once the
  // node or its ports move, including synthetic direct bindings for extmemory.
  const Node *swN = dfg.getNode(swNode);
  if (swN) {
    llvm::SmallVector<IdIndex, 16> incidentEdges;
    auto collectIncidentEdges = [&](llvm::ArrayRef<IdIndex> ports) {
      for (IdIndex pid : ports) {
        const Port *port = dfg.getPort(pid);
        if (!port)
          continue;
        for (IdIndex edgeId : port->connectedEdges) {
          if (std::find(incidentEdges.begin(), incidentEdges.end(), edgeId) ==
              incidentEdges.end()) {
            incidentEdges.push_back(edgeId);
          }
        }
      }
    };
    collectIncidentEdges(swN->inputPorts);
    collectIncidentEdges(swN->outputPorts);
    for (IdIndex edgeId : incidentEdges)
      unmapEdge(edgeId, adg);
  }

  // Unmap ports.
  if (swN) {
    for (IdIndex pid : swN->inputPorts) {
      if (pid < swPortToHwPort.size()) {
        IdIndex hwPort = swPortToHwPort[pid];
        if (hwPort != INVALID_ID && hwPort < hwPortToSwPorts.size()) {
          auto &vec = hwPortToSwPorts[hwPort];
          vec.erase(std::remove(vec.begin(), vec.end(), pid), vec.end());
        }
        swPortToHwPort[pid] = INVALID_ID;
      }
    }
    for (IdIndex pid : swN->outputPorts) {
      if (pid < swPortToHwPort.size()) {
        IdIndex hwPort = swPortToHwPort[pid];
        if (hwPort != INVALID_ID && hwPort < hwPortToSwPorts.size()) {
          auto &vec = hwPortToSwPorts[hwPort];
          vec.erase(std::remove(vec.begin(), vec.end(), pid), vec.end());
        }
        swPortToHwPort[pid] = INVALID_ID;
      }
    }
  }

  swNodeToHwNode[swNode] = INVALID_ID;
  return ActionResult::Success;
}

ActionResult MappingState::mapPort(IdIndex swPort, IdIndex hwPort,
                                    const Graph &dfg, const Graph &adg) {
  if (swPort >= swPortToHwPort.size() || hwPort >= hwPortToSwPorts.size())
    return ActionResult::FailedInternalError;

  swPortToHwPort[swPort] = hwPort;
  hwPortToSwPorts[hwPort].push_back(swPort);
  return ActionResult::Success;
}

ActionResult MappingState::mapEdge(IdIndex swEdge,
                                    llvm::ArrayRef<IdIndex> path,
                                    const Graph &dfg, const Graph &adg) {
  if (swEdge >= swEdgeToHwPaths.size())
    return ActionResult::FailedInternalError;

  if (!swEdgeToHwPaths[swEdge].empty())
    unmapEdge(swEdge, adg);

  swEdgeToHwPaths[swEdge].assign(path.begin(), path.end());

  // Track HW edge usage.
  for (size_t i = 0; i + 1 < path.size(); i += 2) {
    IdIndex outPort = path[i];
    IdIndex inPort = path[i + 1];
    const Port *op = adg.getPort(outPort);
    if (!op)
      continue;
    for (IdIndex edgeId : op->connectedEdges) {
      const Edge *hwEdge = adg.getEdge(edgeId);
      if (!hwEdge)
        continue;
      if (hwEdge->srcPort == outPort && hwEdge->dstPort == inPort) {
        if (edgeId < hwEdgeToSwEdges.size()) {
          hwEdgeToSwEdges[edgeId].push_back(swEdge);
        }
        break;
      }
    }
  }

  return ActionResult::Success;
}

ActionResult MappingState::unmapEdge(IdIndex swEdge, const Graph &adg) {
  if (swEdge >= swEdgeToHwPaths.size())
    return ActionResult::FailedInternalError;

  auto &path = swEdgeToHwPaths[swEdge];
  if (path.empty())
    return ActionResult::Success;

  for (size_t i = 0; i + 1 < path.size(); i += 2) {
    IdIndex outPort = path[i];
    IdIndex inPort = path[i + 1];
    const Port *op = adg.getPort(outPort);
    if (!op)
      continue;
    for (IdIndex edgeId : op->connectedEdges) {
      const Edge *hwEdge = adg.getEdge(edgeId);
      if (!hwEdge || hwEdge->srcPort != outPort || hwEdge->dstPort != inPort)
        continue;
      if (edgeId < hwEdgeToSwEdges.size()) {
        auto &mappedEdges = hwEdgeToSwEdges[edgeId];
        mappedEdges.erase(
            std::remove(mappedEdges.begin(), mappedEdges.end(), swEdge),
            mappedEdges.end());
      }
      break;
    }
  }

  path.clear();
  return ActionResult::Success;
}

void MappingState::clearRoutes(const Graph &adg, bool preserveDirectBindings) {
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(swEdgeToHwPaths.size());
       ++edgeId) {
    const auto &path = swEdgeToHwPaths[edgeId];
    if (path.empty())
      continue;
    if (preserveDirectBindings && path.size() == 2 && path[0] == path[1])
      continue;
    unmapEdge(edgeId, adg);
  }
}

MappingState::Checkpoint MappingState::save() const {
  Checkpoint cp;
  cp.swNodeToHwNode = swNodeToHwNode;
  cp.swPortToHwPort = swPortToHwPort;
  cp.swEdgeToHwPaths = swEdgeToHwPaths;
  cp.hwNodeToSwNodes = hwNodeToSwNodes;
  cp.hwPortToSwPorts = hwPortToSwPorts;
  cp.hwEdgeToSwEdges = hwEdgeToSwEdges;
  cp.totalCost = totalCost;
  return cp;
}

void MappingState::restore(const Checkpoint &cp) {
  swNodeToHwNode = cp.swNodeToHwNode;
  swPortToHwPort = cp.swPortToHwPort;
  swEdgeToHwPaths = cp.swEdgeToHwPaths;
  hwNodeToSwNodes = cp.hwNodeToSwNodes;
  hwPortToSwPorts = cp.hwPortToSwPorts;
  hwEdgeToSwEdges = cp.hwEdgeToSwEdges;
  totalCost = cp.totalCost;
}

} // namespace fcc

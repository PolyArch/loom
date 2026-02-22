//===-- MappingState.cpp - Canonical mapping state -----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/Graph.h"

#include <algorithm>

namespace loom {

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

  temporalPEAssignments.resize(swNodes);
  temporalSWAssignments.resize(hwNodes);
  registerAssignments.assign(swEdges, INVALID_ID);

  totalCost = 0.0;
  placementPressure = 0.0;
  routingCost = 0.0;
  temporalCost = 0.0;
  perfProxyCost = 0.0;
  criticalPathEst = 0.0;
  iiPressure = 0.0;
  queuePressure = 0.0;
  configFootprint = 0.0;
  nonDefaultWords = 0;
  totalConfigWords = 0;

  actionLog.clear();
}

ActionResult MappingState::mapNode(IdIndex swNode, IdIndex hwNode,
                                   const Graph &dfg, const Graph &adg) {
  if (swNode >= swNodeToHwNode.size() || hwNode >= hwNodeToSwNodes.size())
    return ActionResult::FailedInternalError;

  // Check if already mapped.
  if (swNodeToHwNode[swNode] != INVALID_ID)
    return ActionResult::FailedHardConstraint;

  swNodeToHwNode[swNode] = hwNode;
  hwNodeToSwNodes[hwNode].push_back(swNode);

  // Log action.
  ActionRecord record;
  record.type = ActionRecord::MAP_NODE;
  record.arg0 = swNode;
  record.arg1 = hwNode;
  record.constraintResult = ActionResult::Success;
  actionLog.push_back(record);

  return ActionResult::Success;
}

ActionResult MappingState::unmapNode(IdIndex swNode,
                                     const Graph &dfg, const Graph &adg) {
  if (swNode >= swNodeToHwNode.size())
    return ActionResult::FailedInternalError;

  IdIndex hwNode = swNodeToHwNode[swNode];
  if (hwNode == INVALID_ID)
    return ActionResult::FailedHardConstraint;

  swNodeToHwNode[swNode] = INVALID_ID;

  // Remove from reverse mapping.
  auto &swNodes = hwNodeToSwNodes[hwNode];
  swNodes.erase(std::remove(swNodes.begin(), swNodes.end(), swNode),
                swNodes.end());

  // Log action.
  ActionRecord record;
  record.type = ActionRecord::UNMAP_NODE;
  record.arg0 = swNode;
  record.arg1 = hwNode;
  record.constraintResult = ActionResult::Success;
  actionLog.push_back(record);

  return ActionResult::Success;
}

ActionResult MappingState::mapPort(IdIndex swPort, IdIndex hwPort,
                                   const Graph &dfg, const Graph &adg) {
  if (swPort >= swPortToHwPort.size() || hwPort >= hwPortToSwPorts.size())
    return ActionResult::FailedInternalError;

  if (swPortToHwPort[swPort] != INVALID_ID)
    return ActionResult::FailedHardConstraint;

  swPortToHwPort[swPort] = hwPort;
  hwPortToSwPorts[hwPort].push_back(swPort);

  ActionRecord record;
  record.type = ActionRecord::MAP_PORT;
  record.arg0 = swPort;
  record.arg1 = hwPort;
  record.constraintResult = ActionResult::Success;
  actionLog.push_back(record);

  return ActionResult::Success;
}

ActionResult MappingState::unmapPort(IdIndex swPort,
                                     const Graph &dfg, const Graph &adg) {
  if (swPort >= swPortToHwPort.size())
    return ActionResult::FailedInternalError;

  IdIndex hwPort = swPortToHwPort[swPort];
  if (hwPort == INVALID_ID)
    return ActionResult::FailedHardConstraint;

  swPortToHwPort[swPort] = INVALID_ID;

  auto &swPorts = hwPortToSwPorts[hwPort];
  swPorts.erase(std::remove(swPorts.begin(), swPorts.end(), swPort),
                swPorts.end());

  ActionRecord record;
  record.type = ActionRecord::UNMAP_PORT;
  record.arg0 = swPort;
  record.arg1 = hwPort;
  record.constraintResult = ActionResult::Success;
  actionLog.push_back(record);

  return ActionResult::Success;
}

ActionResult MappingState::mapEdge(IdIndex swEdge,
                                   llvm::ArrayRef<IdIndex> path,
                                   const Graph &dfg, const Graph &adg) {
  if (swEdge >= swEdgeToHwPaths.size())
    return ActionResult::FailedInternalError;

  if (!swEdgeToHwPaths[swEdge].empty())
    return ActionResult::FailedHardConstraint;

  swEdgeToHwPaths[swEdge].assign(path.begin(), path.end());

  // Update hwEdgeToSwEdges reverse mapping for each physical edge hop.
  // Path format: [outPort0, inPort0, outPort1, inPort1, ...]
  // Physical edge hops are (path[0], path[1]), (path[2], path[3]), etc.
  for (size_t i = 0; i + 1 < path.size(); i += 2) {
    IdIndex outPortId = path[i];
    IdIndex inPortId = path[i + 1];

    // Find the ADG edge connecting outPortId -> inPortId.
    const Port *outPort = adg.getPort(outPortId);
    if (!outPort)
      continue;
    for (IdIndex edgeId : outPort->connectedEdges) {
      const Edge *hwEdge = adg.getEdge(edgeId);
      if (hwEdge && hwEdge->srcPort == outPortId &&
          hwEdge->dstPort == inPortId) {
        if (edgeId < hwEdgeToSwEdges.size()) {
          hwEdgeToSwEdges[edgeId].push_back(swEdge);
        }
        break;
      }
    }
  }

  ActionRecord record;
  record.type = ActionRecord::MAP_EDGE;
  record.arg0 = swEdge;
  record.pathArgs.assign(path.begin(), path.end());
  record.constraintResult = ActionResult::Success;
  actionLog.push_back(record);

  return ActionResult::Success;
}

ActionResult MappingState::unmapEdge(IdIndex swEdge,
                                     const Graph &dfg, const Graph &adg) {
  if (swEdge >= swEdgeToHwPaths.size())
    return ActionResult::FailedInternalError;

  if (swEdgeToHwPaths[swEdge].empty())
    return ActionResult::FailedHardConstraint;

  // Clean up hwEdgeToSwEdges reverse mapping for each physical edge hop.
  const auto &path = swEdgeToHwPaths[swEdge];
  for (size_t i = 0; i + 1 < path.size(); i += 2) {
    IdIndex outPortId = path[i];
    IdIndex inPortId = path[i + 1];

    const Port *outPort = adg.getPort(outPortId);
    if (!outPort)
      continue;
    for (IdIndex edgeId : outPort->connectedEdges) {
      const Edge *hwEdge = adg.getEdge(edgeId);
      if (hwEdge && hwEdge->srcPort == outPortId &&
          hwEdge->dstPort == inPortId) {
        if (edgeId < hwEdgeToSwEdges.size()) {
          auto &swEdges = hwEdgeToSwEdges[edgeId];
          swEdges.erase(
              std::remove(swEdges.begin(), swEdges.end(), swEdge),
              swEdges.end());
        }
        break;
      }
    }
  }

  ActionRecord record;
  record.type = ActionRecord::UNMAP_EDGE;
  record.arg0 = swEdge;
  record.pathArgs.assign(swEdgeToHwPaths[swEdge].begin(),
                         swEdgeToHwPaths[swEdge].end());
  record.constraintResult = ActionResult::Success;
  actionLog.push_back(record);

  swEdgeToHwPaths[swEdge].clear();

  return ActionResult::Success;
}

bool MappingState::isValid() const {
  // Basic validity: all forward mappings have valid reverse entries.
  for (size_t i = 0; i < swNodeToHwNode.size(); ++i) {
    if (swNodeToHwNode[i] == INVALID_ID)
      continue;
    IdIndex hwNode = swNodeToHwNode[i];
    if (hwNode >= hwNodeToSwNodes.size())
      return false;
    auto &reverse = hwNodeToSwNodes[hwNode];
    if (std::find(reverse.begin(), reverse.end(), static_cast<IdIndex>(i)) ==
        reverse.end())
      return false;
  }
  return true;
}

MappingState::Checkpoint MappingState::save() const {
  Checkpoint cp;
  cp.swNodeToHwNode = swNodeToHwNode;
  cp.swPortToHwPort = swPortToHwPort;
  cp.swEdgeToHwPaths = swEdgeToHwPaths;
  cp.hwNodeToSwNodes = hwNodeToSwNodes;
  cp.hwPortToSwPorts = hwPortToSwPorts;
  cp.hwEdgeToSwEdges = hwEdgeToSwEdges;
  cp.temporalPEAssignments = temporalPEAssignments;
  cp.temporalSWAssignments = temporalSWAssignments;
  cp.registerAssignments = registerAssignments;
  cp.totalCost = totalCost;
  cp.actionLogSize = actionLog.size();
  return cp;
}

void MappingState::restore(const Checkpoint &cp) {
  swNodeToHwNode = cp.swNodeToHwNode;
  swPortToHwPort = cp.swPortToHwPort;
  swEdgeToHwPaths = cp.swEdgeToHwPaths;
  hwNodeToSwNodes = cp.hwNodeToSwNodes;
  hwPortToSwPorts = cp.hwPortToSwPorts;
  hwEdgeToSwEdges = cp.hwEdgeToSwEdges;
  temporalPEAssignments = cp.temporalPEAssignments;
  temporalSWAssignments = cp.temporalSWAssignments;
  registerAssignments = cp.registerAssignments;
  totalCost = cp.totalCost;
  actionLog.resize(cp.actionLogSize);
}

} // namespace loom

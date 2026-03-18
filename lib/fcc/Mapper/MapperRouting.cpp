#include "fcc/Mapper/Mapper.h"

#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "fcc/Mapper/TagRuntime.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <queue>
#include <random>

namespace fcc {

namespace {

enum class TaggedResourceKind {
  RoutingOutput,
  HardwareEdge,
};

struct TaggedPathObservation {
  TaggedResourceKind kind;
  IdIndex first = INVALID_ID;
  IdIndex second = INVALID_ID;
  uint64_t tag = 0;
};

struct TemporalSwitchTagRouteObservation {
  IdIndex nodeId = INVALID_ID;
  IdIndex inPortId = INVALID_ID;
  IdIndex outPortId = INVALID_ID;
  uint64_t tag = 0;
};

std::optional<uint64_t>
computeObservedTagAtPathIndex(IdIndex swEdgeId, llvm::ArrayRef<IdIndex> path,
                              size_t pathIndex, const MappingState &state,
                              const Graph &dfg, const Graph &adg);
bool isRoutingCrossbarOutputPort(IdIndex portId, const Graph &adg);

bool pathUsesPort(IdIndex portId, const MappingState &state) {
  for (const auto &path : state.swEdgeToHwPaths) {
    for (IdIndex usedPort : path) {
      if (usedPort == portId)
        return true;
    }
  }
  return false;
}

bool isTaggedPort(const Port *port) {
  if (!port)
    return false;
  auto info = detail::getPortTypeInfo(port->type);
  return info && info->isTagged;
}

void appendTemporalSwitchTagRouteObservations(
    IdIndex swEdgeId, llvm::ArrayRef<IdIndex> path, const MappingState &state,
    const Graph &dfg, const Graph &adg,
    llvm::SmallVectorImpl<TemporalSwitchTagRouteObservation> &observations) {
  for (size_t i = 0; i + 1 < path.size(); ++i) {
    IdIndex inPortId = path[i];
    IdIndex outPortId = path[i + 1];
    const Port *inPort = adg.getPort(inPortId);
    const Port *outPort = adg.getPort(outPortId);
    if (!inPort || !outPort)
      continue;
    if (inPort->direction != Port::Input || outPort->direction != Port::Output)
      continue;
    if (inPort->parentNode == INVALID_ID || inPort->parentNode != outPort->parentNode)
      continue;
    const Node *owner = adg.getNode(inPort->parentNode);
    if (!owner || getNodeAttrStr(owner, "op_kind") != "temporal_sw")
      continue;
    auto tag =
        computeObservedTagAtPathIndex(swEdgeId, path, i + 1, state, dfg, adg);
    if (!tag)
      continue;
    observations.push_back({inPort->parentNode, inPortId, outPortId, *tag});
  }
}

void appendTaggedPathObservations(
    IdIndex swEdgeId, llvm::ArrayRef<IdIndex> path, const MappingState &state,
    const Graph &dfg, const Graph &adg,
    llvm::SmallVectorImpl<TaggedPathObservation> &observations) {
  for (size_t i = 0; i < path.size(); ++i) {
    IdIndex portId = path[i];
    const Port *port = adg.getPort(portId);
    if (!port || !isTaggedPort(port))
      continue;
    if (!isRoutingCrossbarOutputPort(portId, adg))
      continue;
    auto tag =
        computeObservedTagAtPathIndex(swEdgeId, path, i, state, dfg, adg);
    if (!tag)
      continue;
    observations.push_back(
        {TaggedResourceKind::RoutingOutput, portId, INVALID_ID, *tag});
  }

  for (size_t i = 0; i + 1 < path.size(); ++i) {
    IdIndex srcPortId = path[i];
    IdIndex dstPortId = path[i + 1];
    const Port *srcPort = adg.getPort(srcPortId);
    const Port *dstPort = adg.getPort(dstPortId);
    if (!srcPort || !dstPort)
      continue;
    if (srcPort->direction != Port::Output || dstPort->direction != Port::Input)
      continue;
    if (!isTaggedPort(srcPort) || !isTaggedPort(dstPort))
      continue;
    auto tag =
        computeObservedTagAtPathIndex(swEdgeId, path, i + 1, state, dfg, adg);
    if (!tag)
      continue;
    observations.push_back(
        {TaggedResourceKind::HardwareEdge, srcPortId, dstPortId, *tag});
  }
}

bool observationsConflict(const TaggedPathObservation &lhs,
                          const TaggedPathObservation &rhs) {
  return lhs.kind == rhs.kind && lhs.first == rhs.first &&
         lhs.second == rhs.second && lhs.tag == rhs.tag;
}

bool temporalSwitchTagRouteConflict(
    const TemporalSwitchTagRouteObservation &lhs,
    const TemporalSwitchTagRouteObservation &rhs) {
  return lhs.nodeId == rhs.nodeId && lhs.tag == rhs.tag &&
         (lhs.inPortId != rhs.inPortId || lhs.outPortId != rhs.outPortId);
}

bool isSoftwareMemoryInterfaceOpName(llvm::StringRef opName) {
  return opName == "handshake.extmemory" || opName == "handshake.memory";
}

IdIndex findFeedingPort(const Graph &graph, IdIndex inputPortId) {
  const Port *port = graph.getPort(inputPortId);
  if (!port)
    return INVALID_ID;
  for (IdIndex edgeId : port->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (edge && edge->dstPort == inputPortId)
      return edge->srcPort;
  }
  return INVALID_ID;
}

llvm::SmallVector<IdIndex, 4> findConsumingPorts(const Graph &graph,
                                                 IdIndex outputPortId) {
  llvm::SmallVector<IdIndex, 4> consumers;
  const Port *port = graph.getPort(outputPortId);
  if (!port)
    return consumers;
  for (IdIndex edgeId : port->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (edge && edge->srcPort == outputPortId)
      consumers.push_back(edge->dstPort);
  }
  return consumers;
}

bool isBridgeTraversalNode(llvm::StringRef opKind) {
  return opKind == "add_tag" || opKind == "del_tag" || opKind == "map_tag" ||
         opKind == "spatial_sw" || opKind == "temporal_sw" ||
         opKind == "fifo";
}

llvm::SmallVector<IdIndex, 8>
findBridgePathForward(const Graph &graph, IdIndex startPortId,
                      IdIndex hwMemoryNodeId) {
  llvm::SmallVector<IdIndex, 8> empty;
  const Port *startPort = graph.getPort(startPortId);
  if (!startPort)
    return empty;
  if (startPort->parentNode == hwMemoryNodeId &&
      startPort->direction == Port::Input)
    return empty;

  llvm::SmallVector<IdIndex, 16> queue;
  llvm::DenseMap<IdIndex, IdIndex> prev;
  queue.push_back(startPortId);
  prev[startPortId] = INVALID_ID;

  IdIndex found = INVALID_ID;
  size_t cursor = 0;
  while (cursor < queue.size()) {
    IdIndex portId = queue[cursor++];
    const Port *port = graph.getPort(portId);
    if (!port)
      continue;

    if (portId != startPortId && port->parentNode == hwMemoryNodeId &&
        port->direction == Port::Input) {
      const Node *owner = graph.getNode(port->parentNode);
      if (owner && getNodeAttrStr(owner, "resource_class") == "memory") {
        found = portId;
        break;
      }
    }

    auto tryPush = [&](IdIndex nextPortId) {
      if (nextPortId == INVALID_ID || prev.count(nextPortId))
        return;
      prev[nextPortId] = portId;
      queue.push_back(nextPortId);
    };

    if (port->direction == Port::Input) {
      const Node *owner = getPortOwnerNode(graph, portId);
      if (!owner)
        continue;
      if (!isBridgeTraversalNode(getNodeAttrStr(owner, "op_kind")))
        continue;
      for (IdIndex outPortId : owner->outputPorts)
        tryPush(outPortId);
      continue;
    }

    for (IdIndex consumerPortId : findConsumingPorts(graph, portId))
      tryPush(consumerPortId);
  }

  if (found == INVALID_ID)
    return empty;

  llvm::SmallVector<IdIndex, 8> path;
  for (IdIndex cur = found; cur != INVALID_ID; cur = prev.lookup(cur))
    path.push_back(cur);
  std::reverse(path.begin(), path.end());
  if (!path.empty())
    path.erase(path.begin());
  return path;
}

llvm::SmallVector<IdIndex, 8>
findBridgePathBackward(const Graph &graph, IdIndex startPortId,
                       IdIndex hwMemoryNodeId) {
  llvm::SmallVector<IdIndex, 8> empty;
  const Port *startPort = graph.getPort(startPortId);
  if (!startPort)
    return empty;
  if (startPort->parentNode == hwMemoryNodeId &&
      startPort->direction == Port::Output)
    return empty;

  llvm::SmallVector<IdIndex, 16> queue;
  llvm::DenseMap<IdIndex, IdIndex> prev;
  queue.push_back(startPortId);
  prev[startPortId] = INVALID_ID;

  IdIndex found = INVALID_ID;
  size_t cursor = 0;
  while (cursor < queue.size()) {
    IdIndex portId = queue[cursor++];
    const Port *port = graph.getPort(portId);
    if (!port)
      continue;

    if (portId != startPortId && port->parentNode == hwMemoryNodeId &&
        port->direction == Port::Output) {
      const Node *owner = graph.getNode(port->parentNode);
      if (owner && getNodeAttrStr(owner, "resource_class") == "memory") {
        found = portId;
        break;
      }
    }

    auto tryPush = [&](IdIndex nextPortId) {
      if (nextPortId == INVALID_ID || prev.count(nextPortId))
        return;
      prev[nextPortId] = portId;
      queue.push_back(nextPortId);
    };

    if (port->direction == Port::Output) {
      const Node *owner = getPortOwnerNode(graph, portId);
      if (!owner)
        continue;
      if (!isBridgeTraversalNode(getNodeAttrStr(owner, "op_kind")))
        continue;
      for (IdIndex inPortId : owner->inputPorts)
        tryPush(inPortId);
      continue;
    }

    tryPush(findFeedingPort(graph, portId));
  }

  if (found == INVALID_ID)
    return empty;

  llvm::SmallVector<IdIndex, 8> path;
  for (IdIndex cur = found; cur != INVALID_ID; cur = prev.lookup(cur))
    path.push_back(cur);
  if (!path.empty() && path.back() == startPortId)
    path.pop_back();
  return path;
}

llvm::SmallVector<IdIndex, 16>
buildExportPathForEdge(IdIndex swEdgeId, llvm::ArrayRef<IdIndex> rawPath,
                       const MappingState &state, const Graph &dfg,
                       const Graph &adg) {
  llvm::SmallVector<IdIndex, 16> path(rawPath.begin(), rawPath.end());
  if (path.size() < 2)
    return path;

  const Edge *edge = dfg.getEdge(swEdgeId);
  if (!edge)
    return path;

  const Port *srcPort = dfg.getPort(edge->srcPort);
  if (srcPort && srcPort->parentNode != INVALID_ID) {
    const Node *srcNode = dfg.getNode(srcPort->parentNode);
    if (srcNode &&
        isSoftwareMemoryInterfaceOpName(getNodeAttrStr(srcNode, "op_name")) &&
        srcPort->parentNode < state.swNodeToHwNode.size()) {
      IdIndex hwNodeId = state.swNodeToHwNode[srcPort->parentNode];
      if (hwNodeId != INVALID_ID) {
        auto prefix = findBridgePathBackward(adg, path.front(), hwNodeId);
        if (!prefix.empty()) {
          prefix.append(path.begin(), path.end());
          path = std::move(prefix);
        }
      }
    }
  }

  const Port *dstPort = dfg.getPort(edge->dstPort);
  if (dstPort && dstPort->parentNode != INVALID_ID) {
    const Node *dstNode = dfg.getNode(dstPort->parentNode);
    if (dstNode &&
        isSoftwareMemoryInterfaceOpName(getNodeAttrStr(dstNode, "op_name")) &&
        dstPort->parentNode < state.swNodeToHwNode.size()) {
      IdIndex hwNodeId = state.swNodeToHwNode[dstPort->parentNode];
      if (hwNodeId != INVALID_ID) {
        auto suffix = findBridgePathForward(adg, path.back(), hwNodeId);
        if (!suffix.empty())
          path.append(suffix.begin(), suffix.end());
      }
    }
  }

  return path;
}

std::optional<size_t> findPortIndexInPath(llvm::ArrayRef<IdIndex> path,
                                          IdIndex portId) {
  for (size_t i = 0; i < path.size(); ++i) {
    if (path[i] == portId)
      return i;
  }
  return std::nullopt;
}

std::optional<size_t> findTransitionDstIndex(llvm::ArrayRef<IdIndex> path,
                                             IdIndex outPortId,
                                             IdIndex inPortId) {
  for (size_t i = 0; i + 1 < path.size(); ++i) {
    if (path[i] == outPortId && path[i + 1] == inPortId)
      return i + 1;
  }
  return std::nullopt;
}

std::optional<uint64_t>
computeObservedTagAtPathIndex(IdIndex swEdgeId, llvm::ArrayRef<IdIndex> path,
                              size_t pathIndex, const MappingState &state,
                              const Graph &dfg, const Graph &adg) {
  return computeRuntimeTagValueAlongMappedPath(swEdgeId, path, pathIndex, state,
                                               dfg, adg);
}

bool hasTaggedRoutingOutputConflict(IdIndex swEdgeId,
                                    llvm::ArrayRef<IdIndex> candidatePath,
                                    IdIndex outputPortId,
                                    const MappingState &state,
                                    const Graph &dfg, const Graph &adg) {
  auto candidateIndex = findPortIndexInPath(candidatePath, outputPortId);
  if (!candidateIndex)
    return false;
  auto candidateTag = computeObservedTagAtPathIndex(swEdgeId, candidatePath,
                                                    *candidateIndex, state,
                                                    dfg, adg);
  if (!candidateTag)
    return false;

  for (IdIndex otherEdgeId = 0;
       otherEdgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++otherEdgeId) {
    if (otherEdgeId == swEdgeId)
      continue;
    const auto &otherPath = state.swEdgeToHwPaths[otherEdgeId];
    if (otherPath.empty())
      continue;
    auto otherIndex = findPortIndexInPath(otherPath, outputPortId);
    if (!otherIndex)
      continue;
    auto otherTag = computeObservedTagAtPathIndex(otherEdgeId, otherPath,
                                                  *otherIndex, state, dfg, adg);
    if (otherTag && *otherTag == *candidateTag)
      return true;
  }

  return false;
}

bool hasTaggedHardwareEdgeConflict(IdIndex swEdgeId,
                                   llvm::ArrayRef<IdIndex> candidatePath,
                                   IdIndex srcPortId, IdIndex dstPortId,
                                   const MappingState &state,
                                   const Graph &dfg, const Graph &adg) {
  auto candidateDstIndex =
      findTransitionDstIndex(candidatePath, srcPortId, dstPortId);
  if (!candidateDstIndex)
    return false;
  auto candidateTag = computeObservedTagAtPathIndex(swEdgeId, candidatePath,
                                                    *candidateDstIndex, state,
                                                    dfg, adg);
  if (!candidateTag)
    return false;

  for (IdIndex otherEdgeId = 0;
       otherEdgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++otherEdgeId) {
    if (otherEdgeId == swEdgeId)
      continue;
    const auto &otherPath = state.swEdgeToHwPaths[otherEdgeId];
    if (otherPath.empty())
      continue;
    auto otherDstIndex =
        findTransitionDstIndex(otherPath, srcPortId, dstPortId);
    if (!otherDstIndex)
      continue;
    auto otherTag = computeObservedTagAtPathIndex(otherEdgeId, otherPath,
                                                  *otherDstIndex, state, dfg,
                                                  adg);
    if (otherTag && *otherTag == *candidateTag)
      return true;
  }

  return false;
}

bool isRoutingCrossbarOutputPort(IdIndex portId, const Graph &adg) {
  const Port *port = adg.getPort(portId);
  if (!port || port->direction != Port::Output ||
      port->parentNode == INVALID_ID)
    return false;

  const Node *node = adg.getNode(port->parentNode);
  if (!node)
    return false;

  if (getNodeAttrStr(node, "resource_class") != "routing")
    return false;

  // Treat multi-port routing nodes as spatial switch-style crossbars.
  // Single-in/single-out routing nodes such as FIFOs do not use mux-style
  // route tables and therefore do not need this exclusivity rule.
  if (node->inputPorts.size() <= 1 || node->outputPorts.size() <= 1)
    return false;
  return true;
}

bool isInternalHopLegal(IdIndex inPortId, IdIndex outPortId,
                        IdIndex swEdgeId,
                        llvm::ArrayRef<IdIndex> candidatePath,
                        const MappingState &state, const Graph &dfg,
                        const Graph &adg) {
  const Port *inPort = adg.getPort(inPortId);
  const Port *outPort = adg.getPort(outPortId);
  if (!inPort || !outPort)
    return false;
  if (inPort->parentNode == INVALID_ID || outPort->parentNode == INVALID_ID)
    return false;
  if (inPort->parentNode != outPort->parentNode)
    return false;

  if (isRoutingCrossbarOutputPort(outPortId, adg)) {
    if (isTaggedPort(outPort)) {
      if (hasTaggedRoutingOutputConflict(swEdgeId, candidatePath, outPortId,
                                         state, dfg, adg))
        return false;
    } else if (pathUsesPort(outPortId, state)) {
      return false;
    }
  }

  return true;
}

} // namespace

bool Mapper::hasTaggedPathConflict(IdIndex swEdgeId,
                                   llvm::ArrayRef<IdIndex> candidatePath,
                                   const MappingState &state,
                                   const Graph &dfg, const Graph &adg) {
  llvm::SmallVector<TaggedPathObservation, 8> candidateObs;
  llvm::SmallVector<TemporalSwitchTagRouteObservation, 8> candidateTemporalObs;
  auto fullCandidatePath =
      buildExportPathForEdge(swEdgeId, candidatePath, state, dfg, adg);
  appendTaggedPathObservations(swEdgeId, fullCandidatePath, state, dfg, adg,
                               candidateObs);
  appendTemporalSwitchTagRouteObservations(swEdgeId, fullCandidatePath, state,
                                           dfg, adg, candidateTemporalObs);
  if (candidateObs.empty() && candidateTemporalObs.empty())
    return false;

  for (IdIndex otherEdgeId = 0;
       otherEdgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++otherEdgeId) {
    if (otherEdgeId == swEdgeId)
      continue;
    const auto &otherPath = state.swEdgeToHwPaths[otherEdgeId];
    if (otherPath.empty())
      continue;
    auto fullOtherPath =
        buildExportPathForEdge(otherEdgeId, otherPath, state, dfg, adg);

    llvm::SmallVector<TaggedPathObservation, 8> otherObs;
    llvm::SmallVector<TemporalSwitchTagRouteObservation, 8> otherTemporalObs;
    appendTaggedPathObservations(otherEdgeId, fullOtherPath, state, dfg, adg,
                                 otherObs);
    appendTemporalSwitchTagRouteObservations(otherEdgeId, fullOtherPath, state,
                                             dfg, adg, otherTemporalObs);
    for (const auto &lhs : candidateObs) {
      for (const auto &rhs : otherObs) {
        if (observationsConflict(lhs, rhs))
          return true;
      }
    }
    for (const auto &lhs : candidateTemporalObs) {
      for (const auto &rhs : otherTemporalObs) {
        if (temporalSwitchTagRouteConflict(lhs, rhs))
          return true;
      }
    }
  }

  return false;
}

bool Mapper::validateTaggedPathConflicts(
    const MappingState &state, const Graph &dfg, const Graph &adg,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds, std::string &diagnostics) {
  bool valid = true;
  llvm::SmallVector<llvm::SmallVector<TaggedPathObservation, 8>, 8>
      observationsByEdge(state.swEdgeToHwPaths.size());
  llvm::SmallVector<llvm::SmallVector<TemporalSwitchTagRouteObservation, 8>, 8>
      temporalObsByEdge(state.swEdgeToHwPaths.size());

  for (IdIndex edgeId = 0;
       edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size()); ++edgeId) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    const auto &path = state.swEdgeToHwPaths[edgeId];
    if (path.empty())
      continue;
    auto fullPath = buildExportPathForEdge(edgeId, path, state, dfg, adg);
    appendTaggedPathObservations(edgeId, fullPath, state, dfg, adg,
                                 observationsByEdge[edgeId]);
    appendTemporalSwitchTagRouteObservations(edgeId, fullPath, state, dfg, adg,
                                             temporalObsByEdge[edgeId]);
  }

  for (IdIndex lhsEdge = 0;
       lhsEdge < static_cast<IdIndex>(observationsByEdge.size()); ++lhsEdge) {
    for (IdIndex rhsEdge = lhsEdge + 1;
         rhsEdge < static_cast<IdIndex>(observationsByEdge.size()); ++rhsEdge) {
      for (const auto &lhsObs : observationsByEdge[lhsEdge]) {
        for (const auto &rhsObs : observationsByEdge[rhsEdge]) {
          if (!observationsConflict(lhsObs, rhsObs))
            continue;
          if (lhsObs.kind == TaggedResourceKind::RoutingOutput) {
            diagnostics +=
                "C4: tagged routing-output conflict on hw_port " +
                std::to_string(lhsObs.first) + " tag " +
                std::to_string(lhsObs.tag) + " between sw_edges " +
                std::to_string(lhsEdge) + " and " +
                std::to_string(rhsEdge) + "\n";
          } else {
            diagnostics += "C4: tagged hardware-edge conflict on transition " +
                           std::to_string(lhsObs.first) + "->" +
                           std::to_string(lhsObs.second) + " tag " +
                           std::to_string(lhsObs.tag) +
                           " between sw_edges " +
                           std::to_string(lhsEdge) + " and " +
                           std::to_string(rhsEdge) + "\n";
          }
          valid = false;
        }
      }
      for (const auto &lhsObs : temporalObsByEdge[lhsEdge]) {
        for (const auto &rhsObs : temporalObsByEdge[rhsEdge]) {
          if (!temporalSwitchTagRouteConflict(lhsObs, rhsObs))
            continue;
          diagnostics +=
              "C4: temporal_sw tag-route conflict on hw_node " +
              std::to_string(lhsObs.nodeId) + " tag " +
              std::to_string(lhsObs.tag) + " between sw_edges " +
              std::to_string(lhsEdge) + " and " +
              std::to_string(rhsEdge) + "\n";
          valid = false;
        }
      }
    }
  }

  return valid;
}

bool Mapper::isEdgeLegal(IdIndex srcPort, IdIndex dstPort, IdIndex swEdgeId,
                         llvm::ArrayRef<IdIndex> candidatePath,
                         const MappingState &state, const Graph &dfg,
                         const Graph &adg) {
  const Port *sp = adg.getPort(srcPort);
  const Port *dp = adg.getPort(dstPort);
  if (!sp || !dp)
    return false;

  if (sp->direction != Port::Output || dp->direction != Port::Input)
    return false;

  // Check physical connectivity exists (multi-destination).
  auto it = connectivity.outToIn.find(srcPort);
  if (it == connectivity.outToIn.end())
    return false;
  bool found = false;
  for (IdIndex destPort : it->second) {
    if (destPort == dstPort) {
      found = true;
      break;
    }
  }
  if (!found)
    return false;

  if (isTaggedPort(sp) && isTaggedPort(dp) &&
      hasTaggedHardwareEdgeConflict(swEdgeId, candidatePath, srcPort, dstPort,
                                    state, dfg, adg)) {
    return false;
  }

  // C4: For the MVP, allow edge sharing in the fully-connected fabric.
  // In a real fabric with dedicated switch routing, edges are not exclusive
  // at this level. Exclusivity is enforced at the switch port level.
  return true;
}

llvm::SmallVector<IdIndex, 8>
Mapper::findPath(IdIndex srcHwPort, IdIndex dstHwPort, IdIndex swEdgeId,
                 const MappingState &state, const Graph &dfg,
                 const Graph &adg) {
  llvm::SmallVector<IdIndex, 8> path;

  if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
    return path;

  // Direct connection check.
  llvm::SmallVector<IdIndex, 8> directPath{srcHwPort, dstHwPort};
  if (isEdgeLegal(srcHwPort, dstHwPort, swEdgeId, directPath, state, dfg,
                  adg) &&
      !hasTaggedPathConflict(swEdgeId, directPath, state, dfg, adg)) {
    path.push_back(srcHwPort);
    path.push_back(dstHwPort);
    return path;
  }

  // BFS through connectivity matrix.
  struct BFSEntry {
    IdIndex portId;
    llvm::SmallVector<IdIndex, 8> pathSoFar;
  };

  std::queue<BFSEntry> bfsQueue;
  llvm::DenseSet<IdIndex> visited;

  BFSEntry start;
  start.portId = srcHwPort;
  start.pathSoFar.push_back(srcHwPort);
  bfsQueue.push(start);
  visited.insert(srcHwPort);

  while (!bfsQueue.empty()) {
    auto current = bfsQueue.front();
    bfsQueue.pop();

    // Follow all physical edges from current output port.
    auto physIt = connectivity.outToIn.find(current.portId);
    if (physIt == connectivity.outToIn.end())
      continue;

    for (IdIndex nextInputPort : physIt->second) {
      if (visited.count(nextInputPort))
        continue;

      auto nextPath = current.pathSoFar;
      nextPath.push_back(nextInputPort);

      if (!isEdgeLegal(current.portId, nextInputPort, swEdgeId, nextPath, state,
                       dfg, adg))
        continue;

      visited.insert(nextInputPort);

      if (nextInputPort == dstHwPort) {
        if (!hasTaggedPathConflict(swEdgeId, nextPath, state, dfg, adg))
          return nextPath;
        continue;
      }

      // Traverse routing node internals.
      auto internalIt = connectivity.inToOut.find(nextInputPort);
      if (internalIt != connectivity.inToOut.end()) {
        for (IdIndex outPortId : internalIt->second) {
          if (visited.count(outPortId))
            continue;
          auto outPath = nextPath;
          outPath.push_back(outPortId);
          if (!isInternalHopLegal(nextInputPort, outPortId, swEdgeId, outPath,
                                  state, dfg, adg))
            continue;
          visited.insert(outPortId);

          BFSEntry next;
          next.portId = outPortId;
          next.pathSoFar = outPath;
          bfsQueue.push(next);
        }
      }
    }
  }

  return path; // Empty = no path found.
}

bool Mapper::isEdgeRoutable(IdIndex edgeId, const MappingState &state,
                            const Graph &dfg) {
  const Edge *edge = dfg.getEdge(edgeId);
  if (!edge)
    return false;

  // Skip edges already routed (e.g., memref-binding edges).
  if (edgeId < state.swEdgeToHwPaths.size() &&
      !state.swEdgeToHwPaths[edgeId].empty())
    return false;

  IdIndex srcSwPort = edge->srcPort;
  IdIndex dstSwPort = edge->dstPort;

  if (srcSwPort >= state.swPortToHwPort.size() ||
      dstSwPort >= state.swPortToHwPort.size())
    return false;

  // Skip edges involving UNMAPPED sentinel nodes.
  const Port *srcSwPortPtr = dfg.getPort(srcSwPort);
  const Port *dstSwPortPtr = dfg.getPort(dstSwPort);
  if (srcSwPortPtr && srcSwPortPtr->parentNode != INVALID_ID) {
    const Node *srcNode = dfg.getNode(srcSwPortPtr->parentNode);
    if (srcNode && srcNode->kind == Node::ModuleInputNode) {
      IdIndex srcNodeId = srcSwPortPtr->parentNode;
      if (srcNodeId >= state.swNodeToHwNode.size() ||
          state.swNodeToHwNode[srcNodeId] == INVALID_ID)
        return false;
    }
  }
  if (dstSwPortPtr && dstSwPortPtr->parentNode != INVALID_ID) {
    const Node *dstNode = dfg.getNode(dstSwPortPtr->parentNode);
    if (dstNode && dstNode->kind == Node::ModuleOutputNode) {
      IdIndex dstNodeId = dstSwPortPtr->parentNode;
      if (dstNodeId >= state.swNodeToHwNode.size() ||
          state.swNodeToHwNode[dstNodeId] == INVALID_ID)
        return false;
    }
  }

  IdIndex srcHwPort = state.swPortToHwPort[srcSwPort];
  IdIndex dstHwPort = state.swPortToHwPort[dstSwPort];

  return srcHwPort != INVALID_ID && dstHwPort != INVALID_ID;
}

bool Mapper::routeOnePass(MappingState &state, const Graph &dfg,
                           const Graph &adg,
                           llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                           const std::vector<IdIndex> &edgeOrder,
                           unsigned &routed, unsigned &total) {
  bool allRouted = true;
  routed = 0;
  total = 0;

  for (IdIndex edgeId : edgeOrder) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
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

    total++;

    auto path = findPath(srcHwPort, dstHwPort, edgeId, state, dfg, adg);
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

  std::mt19937 rng(seed >= 0 ? static_cast<unsigned>(seed) : 0u);

  if (seed >= 0) {
    std::shuffle(memEdges.begin(), memEdges.end(), rng);
    std::shuffle(otherEdges.begin(), otherEdges.end(), rng);
  }

  std::vector<IdIndex> edgeOrder;
  edgeOrder.reserve(memEdges.size() + otherEdges.size());
  edgeOrder.insert(edgeOrder.end(), memEdges.begin(), memEdges.end());
  edgeOrder.insert(edgeOrder.end(), otherEdges.begin(), otherEdges.end());

  unsigned routed = 0;
  unsigned total = 0;
  bool allRouted =
      routeOnePass(state, dfg, adg, edgeKinds, edgeOrder, routed, total);

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

    // Rebuild edge order: failed edges first (priority), then others.
    // Shuffle within each group for variety.
    std::shuffle(failedEdges.begin(), failedEdges.end(), rng);
    std::vector<IdIndex> remainingEdges;
    llvm::DenseSet<IdIndex> failedSet;
    for (IdIndex eid : failedEdges)
      failedSet.insert(eid);
    for (IdIndex eid : edgeOrder) {
      if (!failedSet.count(eid))
        remainingEdges.push_back(eid);
    }
    std::shuffle(remainingEdges.begin(), remainingEdges.end(), rng);

    std::vector<IdIndex> newOrder;
    newOrder.reserve(failedEdges.size() + remainingEdges.size());
    newOrder.insert(newOrder.end(), failedEdges.begin(), failedEdges.end());
    newOrder.insert(newOrder.end(), remainingEdges.begin(),
                    remainingEdges.end());

    allRouted =
        routeOnePass(state, dfg, adg, edgeKinds, newOrder, routed, total);
    llvm::outs() << "  Rip-up pass " << (pass + 1) << " result: " << routed
                 << "/" << total << " edges\n";
  }

  llvm::outs() << "  Final routing: " << routed << "/" << total << " edges\n";
  return allRouted;
}

} // namespace fcc

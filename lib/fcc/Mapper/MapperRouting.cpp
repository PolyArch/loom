#include "fcc/Mapper/Mapper.h"
#include "MapperRoutingInternal.h"

#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "fcc/Mapper/TagRuntime.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <random>

namespace fcc {

// ---------------------------------------------------------------------------
// Shared helper implementations (declared in MapperRoutingInternal.h)
// ---------------------------------------------------------------------------

namespace routing_detail {

bool isRoutingNode(const Node *node) {
  return node && getNodeAttrStr(node, "resource_class") == "routing";
}

bool isSoftwareMemoryInterfaceOpName(llvm::StringRef opName) {
  return opName == "handshake.extmemory" || opName == "handshake.memory";
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

} // namespace routing_detail

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers used only by methods in this file
// ---------------------------------------------------------------------------

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

bool pathUsesPort(IdIndex portId, const MappingState &state) {
  for (const auto &path : state.swEdgeToHwPaths) {
    for (IdIndex usedPort : path) {
      if (usedPort == portId)
        return true;
    }
  }
  return false;
}

bool routingOutputUsedByDifferentSource(IdIndex outPortId, IdIndex sourceHwPort,
                                        const MappingState &state) {
  for (const auto &path : state.swEdgeToHwPaths) {
    if (path.empty())
      continue;
    bool usesPort = false;
    for (IdIndex usedPort : path) {
      if (usedPort == outPortId) {
        usesPort = true;
        break;
      }
    }
    if (!usesPort)
      continue;
    if (path.front() != sourceHwPort)
      return true;
  }
  return false;
}

bool isTaggedPort(const Port *port) {
  if (!port)
    return false;
  auto info = detail::getPortTypeInfo(port->type);
  return info && info->isTagged;
}

const Edge *findEdgeByPorts(const Graph &graph, IdIndex srcPortId,
                            IdIndex dstPortId) {
  const Port *srcPort = graph.getPort(srcPortId);
  if (!srcPort)
    return nullptr;
  for (IdIndex edgeId : srcPort->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (edge && edge->srcPort == srcPortId && edge->dstPort == dstPortId)
      return edge;
  }
  return nullptr;
}

std::optional<uint64_t> getUIntEdgeAttr(const Edge *edge, llvm::StringRef name) {
  if (!edge)
    return std::nullopt;
  for (auto attr : edge->attributes) {
    if (attr.getName() != name)
      continue;
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
      return static_cast<uint64_t>(intAttr.getInt());
  }
  return std::nullopt;
}

bool isNonRoutingBroadcastTransitionConflict(
    IdIndex swEdgeId, llvm::ArrayRef<IdIndex> candidatePath,
    const MappingState &state, const Graph &adg) {
  if (candidatePath.size() < 2)
    return false;
  IdIndex srcPortId = candidatePath.front();
  IdIndex nextInputPortId = candidatePath[1];
  const Port *srcPort = adg.getPort(srcPortId);
  const Port *nextInputPort = adg.getPort(nextInputPortId);
  if (!srcPort || !nextInputPort || srcPort->direction != Port::Output ||
      nextInputPort->direction != Port::Input ||
      srcPort->parentNode == INVALID_ID)
    return false;

  const Node *owner = adg.getNode(srcPort->parentNode);
  if (routing_detail::isRoutingNode(owner))
    return false;

  for (IdIndex otherEdgeId = 0;
       otherEdgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++otherEdgeId) {
    if (otherEdgeId == swEdgeId)
      continue;
    const auto &otherPath = state.swEdgeToHwPaths[otherEdgeId];
    if (otherPath.size() < 2 || otherPath.front() != srcPortId)
      continue;
    if (otherPath[1] != nextInputPortId)
      return true;
  }

  return false;
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
        routing_detail::isSoftwareMemoryInterfaceOpName(
            getNodeAttrStr(srcNode, "op_name")) &&
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
        routing_detail::isSoftwareMemoryInterfaceOpName(
            getNodeAttrStr(dstNode, "op_name")) &&
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
    if (!routing_detail::isRoutingCrossbarOutputPort(portId, adg))
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

  if (routing_detail::isRoutingCrossbarOutputPort(outPortId, adg)) {
    if (isTaggedPort(outPort)) {
      if (hasTaggedRoutingOutputConflict(swEdgeId, candidatePath, outPortId,
                                         state, dfg, adg))
        return false;
    } else if (!candidatePath.empty() &&
               routingOutputUsedByDifferentSource(outPortId,
                                                 candidatePath.front(),
                                                 state)) {
      return false;
    }
  }

  return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Mapper methods: conflict detection, legality, pathfinding, routability
// ---------------------------------------------------------------------------

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

  if (isNonRoutingBroadcastTransitionConflict(swEdgeId, candidatePath, state,
                                              adg)) {
    return false;
  }

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
                 const Graph &adg,
                 const llvm::DenseMap<IdIndex, double> &routingOutputHistory,
                 IdIndex forcedFirstHop) {
  llvm::SmallVector<IdIndex, 8> path;

  if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
    return path;

  // Direct connection check.
  llvm::SmallVector<IdIndex, 8> directPath{srcHwPort, dstHwPort};
  if ((forcedFirstHop == INVALID_ID || forcedFirstHop == dstHwPort) &&
      isEdgeLegal(srcHwPort, dstHwPort, swEdgeId, directPath, state, dfg,
                  adg) &&
      !hasTaggedPathConflict(swEdgeId, directPath, state, dfg, adg)) {
    path.push_back(srcHwPort);
    path.push_back(dstHwPort);
    return path;
  }

  struct SearchEntry {
    double cost = 0.0;
    IdIndex portId;
    llvm::SmallVector<IdIndex, 8> pathSoFar;
  };

struct SearchEntryLess {
    bool operator()(const SearchEntry &lhs, const SearchEntry &rhs) const {
      if (std::abs(lhs.cost - rhs.cost) > 1e-9)
        return lhs.cost > rhs.cost;
      if (lhs.pathSoFar.size() != rhs.pathSoFar.size())
        return lhs.pathSoFar.size() > rhs.pathSoFar.size();
      return lhs.portId > rhs.portId;
    }
  };

  std::priority_queue<SearchEntry, std::vector<SearchEntry>, SearchEntryLess>
      worklist;
  llvm::DenseMap<IdIndex, double> bestCost;

  SearchEntry start;
  start.cost = 0.0;
  start.portId = srcHwPort;
  start.pathSoFar.push_back(srcHwPort);
  worklist.push(start);
  bestCost[srcHwPort] = 0.0;

  while (!worklist.empty()) {
    SearchEntry current = worklist.top();
    worklist.pop();

    auto bestIt = bestCost.find(current.portId);
    if (bestIt != bestCost.end() && current.cost > bestIt->second + 1e-9)
      continue;

    const Port *currentPort = adg.getPort(current.portId);
    if (!currentPort)
      continue;

    if (current.portId == dstHwPort) {
      if (!hasTaggedPathConflict(swEdgeId, current.pathSoFar, state, dfg, adg))
        return current.pathSoFar;
      continue;
    }

    if (currentPort->direction == Port::Output) {
      auto physIt = connectivity.outToIn.find(current.portId);
      if (physIt == connectivity.outToIn.end())
        continue;

      for (IdIndex nextInputPort : physIt->second) {
        if (current.portId == srcHwPort && forcedFirstHop != INVALID_ID &&
            nextInputPort != forcedFirstHop)
          continue;
        auto nextPath = current.pathSoFar;
        nextPath.push_back(nextInputPort);

        if (!isEdgeLegal(current.portId, nextInputPort, swEdgeId, nextPath,
                         state, dfg, adg))
          continue;

        double nextCost = current.cost + 1.0;
        auto nextBestIt = bestCost.find(nextInputPort);
        if (nextBestIt != bestCost.end() &&
            nextCost >= nextBestIt->second - 1e-9)
          continue;

        bestCost[nextInputPort] = nextCost;
        worklist.push({nextCost, nextInputPort, std::move(nextPath)});
      }
      continue;
    }

    auto internalIt = connectivity.inToOut.find(current.portId);
    if (internalIt == connectivity.inToOut.end())
      continue;

    for (IdIndex outPortId : internalIt->second) {
      auto outPath = current.pathSoFar;
      outPath.push_back(outPortId);
      if (!isInternalHopLegal(current.portId, outPortId, swEdgeId, outPath,
                              state, dfg, adg))
        continue;

      double historyPenalty = 0.0;
      if (auto it = routingOutputHistory.find(outPortId);
          it != routingOutputHistory.end())
        historyPenalty = it->second;

      double nextCost = current.cost + 1.0 + historyPenalty;
      auto nextBestIt = bestCost.find(outPortId);
      if (nextBestIt != bestCost.end() &&
          nextCost >= nextBestIt->second - 1e-9)
        continue;

      bestCost[outPortId] = nextCost;
      worklist.push({nextCost, outPortId, std::move(outPath)});
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

} // namespace fcc

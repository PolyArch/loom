#include "fcc/Mapper/Mapper.h"
#include "MapperRoutingCongestion.h"
#include "MapperRoutingInternal.h"

#include "fcc/Dialect/Fabric/FabricTypes.h"
#include "fcc/Mapper/TagRuntime.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
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

bool isNonRoutingOutputPort(IdIndex portId, const Graph &adg) {
  const Port *port = adg.getPort(portId);
  if (!port || port->direction != Port::Output || port->parentNode == INVALID_ID)
    return false;
  const Node *owner = adg.getNode(port->parentNode);
  return owner && !isRoutingNode(owner);
}

IdIndex getLockedFirstHopForSource(IdIndex swEdgeId, IdIndex srcHwPort,
                                   const MappingState &state) {
  // Use portToUsingEdges index for O(1) lookup.
  if (srcHwPort < state.portToUsingEdges.size()) {
    for (IdIndex otherEdgeId : state.portToUsingEdges[srcHwPort]) {
      if (otherEdgeId == swEdgeId)
        continue;
      const auto &otherPath = state.swEdgeToHwPaths[otherEdgeId];
      if (otherPath.size() >= 2 && otherPath.front() == srcHwPort)
        return otherPath[1];
    }
  }
  return INVALID_ID;
}

} // namespace routing_detail

// ---------------------------------------------------------------------------
// Anonymous-namespace helpers used only by methods in this file
// ---------------------------------------------------------------------------

namespace {

struct SearchStateKey {
  IdIndex portId = INVALID_ID;
  bool hasTag = false;
  uint64_t tagValue = 0;

  bool operator==(const SearchStateKey &other) const {
    return portId == other.portId && hasTag == other.hasTag &&
           tagValue == other.tagValue;
  }
};

struct SearchStateKeyInfo {
  static inline SearchStateKey getEmptyKey() {
    return SearchStateKey{INVALID_ID, false, 0};
  }

  static inline SearchStateKey getTombstoneKey() {
    return SearchStateKey{INVALID_ID - 1, false, 0};
  }

  static unsigned getHashValue(const SearchStateKey &key) {
    return llvm::hash_combine(key.portId, key.hasTag, key.tagValue);
  }

  static bool isEqual(const SearchStateKey &lhs, const SearchStateKey &rhs) {
    return lhs == rhs;
  }
};

struct RoutingPathAnalysisCacheKey {
  IdIndex swEdgeId = INVALID_ID;
  unsigned pathSize = 0;
  IdIndex firstPort = INVALID_ID;
  IdIndex lastPort = INVALID_ID;
  llvm::hash_code rawPathHash = 0;

  bool operator==(const RoutingPathAnalysisCacheKey &other) const {
    return swEdgeId == other.swEdgeId && pathSize == other.pathSize &&
           firstPort == other.firstPort && lastPort == other.lastPort &&
           rawPathHash == other.rawPathHash;
  }
};

struct RoutingPathAnalysisCacheKeyInfo {
  static inline RoutingPathAnalysisCacheKey getEmptyKey() {
    return RoutingPathAnalysisCacheKey{INVALID_ID, 0, INVALID_ID, INVALID_ID, 0};
  }

  static inline RoutingPathAnalysisCacheKey getTombstoneKey() {
    return RoutingPathAnalysisCacheKey{INVALID_ID - 1, 0, INVALID_ID, INVALID_ID,
                                       llvm::hash_value("tombstone")};
  }

  static unsigned
  getHashValue(const RoutingPathAnalysisCacheKey &key) {
    return llvm::hash_combine(key.swEdgeId, key.pathSize, key.firstPort,
                              key.lastPort, key.rawPathHash);
  }

  static bool isEqual(const RoutingPathAnalysisCacheKey &lhs,
                      const RoutingPathAnalysisCacheKey &rhs) {
    return lhs == rhs;
  }
};

struct RoutingPathAnalysis {
  bool fullPathReady = false;
  llvm::SmallVector<IdIndex, 16> fullPath;
  bool runtimeTagFailureReady = false;
  std::optional<routing_detail::RuntimeTagPathFailure> runtimeTagFailure;
  bool taggedObservationsReady = false;
  llvm::SmallVector<routing_detail::TaggedPathObservation, 8>
      taggedObservations;
  bool temporalObservationsReady = false;
  llvm::SmallVector<routing_detail::TemporalSwitchTagRouteObservation, 8>
      temporalObservations;
};

struct RoutingPathAnalysisCache {
  llvm::DenseMap<RoutingPathAnalysisCacheKey, RoutingPathAnalysis,
                 RoutingPathAnalysisCacheKeyInfo>
      entries;
};

bool pathUsesPort(IdIndex portId, const MappingState &state) {
  if (portId < state.portToUsingEdges.size())
    return !state.portToUsingEdges[portId].empty();
  return false;
}

bool routingOutputUsedByDifferentSource(IdIndex outPortId, IdIndex sourceHwPort,
                                        const MappingState &state) {
  if (outPortId >= state.portToUsingEdges.size())
    return false;
  for (IdIndex edgeId : state.portToUsingEdges[outPortId]) {
    const auto &path = state.swEdgeToHwPaths[edgeId];
    if (path.empty())
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

RoutingPathAnalysisCacheKey
makeRoutingPathAnalysisCacheKey(IdIndex swEdgeId,
                                llvm::ArrayRef<IdIndex> rawPath) {
  return RoutingPathAnalysisCacheKey{
      swEdgeId,
      static_cast<unsigned>(rawPath.size()),
      rawPath.empty() ? INVALID_ID : rawPath.front(),
      rawPath.empty() ? INVALID_ID : rawPath.back(),
      llvm::hash_combine_range(rawPath.begin(), rawPath.end())};
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

  if (srcPortId >= state.portToUsingEdges.size())
    return false;
  for (IdIndex otherEdgeId : state.portToUsingEdges[srcPortId]) {
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

} // namespace

namespace routing_detail {

namespace {

std::optional<size_t> findPortIndexInPath(llvm::ArrayRef<IdIndex> path,
                                          IdIndex portId) {
  for (size_t i = 0; i < path.size(); ++i) {
    if (path[i] == portId)
      return i;
  }
  return std::nullopt;
}

std::optional<uint64_t>
computeObservedTagAtPathIndex(IdIndex swEdgeId, llvm::ArrayRef<IdIndex> path,
                              size_t pathIndex, const MappingState &state,
                              const Graph &dfg, const Graph &adg) {
  auto info = computeRuntimeTagValueInfoAlongMappedPath(swEdgeId, path,
                                                        pathIndex, state, dfg,
                                                        adg);
  if (!info.representable)
    return std::nullopt;
  return info.tag;
}

} // namespace

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

std::optional<RuntimeTagPathFailure> findRuntimeTagPathFailure(
    IdIndex swEdgeId, llvm::ArrayRef<IdIndex> fullPath, const MappingState &state,
    const Graph &dfg, const Graph &adg) {
  for (size_t i = 0; i < fullPath.size(); ++i) {
    auto info = computeRuntimeTagValueInfoAlongMappedPath(swEdgeId, fullPath, i,
                                                          state, dfg, adg);
    if (info.representable)
      continue;

    const Port *port = adg.getPort(fullPath[i]);
    unsigned tagWidth = 0;
    if (port) {
      if (auto typeInfo = detail::getPortTypeInfo(port->type);
          typeInfo && typeInfo->isTagged) {
        tagWidth = typeInfo->tagWidth;
      }
    }
    return RuntimeTagPathFailure{i, fullPath[i], info.rejectedTag.value_or(0),
                                 tagWidth};
  }
  return std::nullopt;
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
        {MappingState::TaggedObservationKind::RoutingOutput, portId,
         INVALID_ID, *tag});
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
        {MappingState::TaggedObservationKind::HardwareEdge, srcPortId,
         dstPortId, *tag});
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

} // namespace routing_detail

namespace {

RoutingPathAnalysis &
getOrCreateRoutingPathAnalysis(IdIndex swEdgeId,
                               llvm::ArrayRef<IdIndex> candidatePath,
                               RoutingPathAnalysisCache &cache,
                               const MappingState &state, const Graph &dfg,
                               const Graph &adg) {
  RoutingPathAnalysisCacheKey key =
      makeRoutingPathAnalysisCacheKey(swEdgeId, candidatePath);
  auto [it, inserted] = cache.entries.try_emplace(key);
  if (inserted) {
    it->second.fullPath = routing_detail::buildExportPathForEdge(
        swEdgeId, candidatePath, state, dfg, adg);
    it->second.fullPathReady = true;
  }
  return it->second;
}

std::optional<routing_detail::RuntimeTagPathFailure>
getRuntimeTagPathFailure(IdIndex swEdgeId, llvm::ArrayRef<IdIndex> candidatePath,
                         RoutingPathAnalysisCache &cache,
                         const MappingState &state, const Graph &dfg,
                         const Graph &adg) {
  auto &analysis =
      getOrCreateRoutingPathAnalysis(swEdgeId, candidatePath, cache, state, dfg,
                                     adg);
  if (!analysis.runtimeTagFailureReady) {
    analysis.runtimeTagFailure = routing_detail::findRuntimeTagPathFailure(
        swEdgeId, analysis.fullPath, state, dfg, adg);
    analysis.runtimeTagFailureReady = true;
  }
  return analysis.runtimeTagFailure;
}

llvm::ArrayRef<routing_detail::TaggedPathObservation>
getTaggedPathObservations(IdIndex swEdgeId,
                          llvm::ArrayRef<IdIndex> candidatePath,
                          RoutingPathAnalysisCache &cache,
                          const MappingState &state, const Graph &dfg,
                          const Graph &adg) {
  auto &analysis =
      getOrCreateRoutingPathAnalysis(swEdgeId, candidatePath, cache, state, dfg,
                                     adg);
  if (!analysis.taggedObservationsReady) {
    routing_detail::appendTaggedPathObservations(swEdgeId, analysis.fullPath,
                                                 state, dfg, adg,
                                                 analysis.taggedObservations);
    analysis.taggedObservationsReady = true;
  }
  return analysis.taggedObservations;
}

llvm::ArrayRef<routing_detail::TemporalSwitchTagRouteObservation>
getTemporalSwitchTagRouteObservations(
    IdIndex swEdgeId, llvm::ArrayRef<IdIndex> candidatePath,
    RoutingPathAnalysisCache &cache, const MappingState &state,
    const Graph &dfg, const Graph &adg) {
  auto &analysis =
      getOrCreateRoutingPathAnalysis(swEdgeId, candidatePath, cache, state, dfg,
                                     adg);
  if (!analysis.temporalObservationsReady) {
    routing_detail::appendTemporalSwitchTagRouteObservations(
        swEdgeId, analysis.fullPath, state, dfg, adg,
        analysis.temporalObservations);
    analysis.temporalObservationsReady = true;
  }
  return analysis.temporalObservations;
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

bool isRuntimeTagPathRepresentable(IdIndex swEdgeId,
                                   llvm::ArrayRef<IdIndex> candidatePath,
                                   RoutingPathAnalysisCache &cache,
                                   const MappingState &state,
                                   const Graph &dfg, const Graph &adg) {
  if (!state.hasTaggedResources)
    return true;
  return !getRuntimeTagPathFailure(swEdgeId, candidatePath, cache, state, dfg,
                                   adg);
}

bool hasTaggedRoutingOutputConflict(IdIndex swEdgeId,
                                    llvm::ArrayRef<IdIndex> candidatePath,
                                    IdIndex outputPortId,
                                    RoutingPathAnalysisCache &cache,
                                    const MappingState &state,
                                    const Graph &dfg, const Graph &adg) {
  if (!state.hasTaggedResources)
    return false;
  auto candidateObs =
      getTaggedPathObservations(swEdgeId, candidatePath, cache, state, dfg, adg);
  auto it = llvm::find_if(candidateObs, [&](const auto &obs) {
    return obs.kind == MappingState::TaggedObservationKind::RoutingOutput &&
           obs.first == outputPortId;
  });
  if (it == candidateObs.end())
    return false;

  MappingState::TaggedObservationKey key{it->kind, it->first, it->second,
                                         it->tag};
  auto existing = state.taggedObservationIndex.find(key);
  if (existing == state.taggedObservationIndex.end())
    return false;
  return llvm::any_of(existing->second, [&](IdIndex otherEdgeId) {
    return otherEdgeId != swEdgeId;
  });
}

bool hasTaggedHardwareEdgeConflict(IdIndex swEdgeId,
                                   llvm::ArrayRef<IdIndex> candidatePath,
                                   IdIndex srcPortId, IdIndex dstPortId,
                                   RoutingPathAnalysisCache &cache,
                                   const MappingState &state,
                                   const Graph &dfg, const Graph &adg) {
  if (!state.hasTaggedResources)
    return false;
  auto candidateObs =
      getTaggedPathObservations(swEdgeId, candidatePath, cache, state, dfg, adg);
  auto it = llvm::find_if(candidateObs, [&](const auto &obs) {
    return obs.kind == MappingState::TaggedObservationKind::HardwareEdge &&
           obs.first == srcPortId && obs.second == dstPortId;
  });
  if (it == candidateObs.end())
    return false;

  MappingState::TaggedObservationKey key{it->kind, it->first, it->second,
                                         it->tag};
  auto existing = state.taggedObservationIndex.find(key);
  if (existing == state.taggedObservationIndex.end())
    return false;
  return llvm::any_of(existing->second, [&](IdIndex otherEdgeId) {
    return otherEdgeId != swEdgeId;
  });
}

bool isInternalHopLegal(IdIndex inPortId, IdIndex outPortId,
                        IdIndex swEdgeId,
                        llvm::ArrayRef<IdIndex> candidatePath,
                        RoutingPathAnalysisCache &cache,
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

  if (!isRuntimeTagPathRepresentable(swEdgeId, candidatePath, cache, state, dfg,
                                     adg))
    return false;

  if (routing_detail::isRoutingCrossbarOutputPort(outPortId, adg)) {
    if (isTaggedPort(outPort)) {
      if (hasTaggedRoutingOutputConflict(swEdgeId, candidatePath, outPortId,
                                         cache, state, dfg, adg))
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

bool hasTaggedPathConflictCached(IdIndex swEdgeId,
                                 llvm::ArrayRef<IdIndex> candidatePath,
                                 RoutingPathAnalysisCache &cache,
                                 const MappingState &state, const Graph &dfg,
                                 const Graph &adg) {
  if (!state.hasTaggedResources)
    return false;
  if (!isRuntimeTagPathRepresentable(swEdgeId, candidatePath, cache, state, dfg,
                                     adg))
    return true;

  auto candidateObs =
      getTaggedPathObservations(swEdgeId, candidatePath, cache, state, dfg, adg);
  auto candidateTemporalObs = getTemporalSwitchTagRouteObservations(
      swEdgeId, candidatePath, cache, state, dfg, adg);
  if (candidateObs.empty() && candidateTemporalObs.empty())
    return false;

  for (const auto &obs : candidateObs) {
    MappingState::TaggedObservationKey key{obs.kind, obs.first, obs.second,
                                           obs.tag};
    auto it = state.taggedObservationIndex.find(key);
    if (it == state.taggedObservationIndex.end())
      continue;
    if (llvm::any_of(it->second, [&](IdIndex otherEdgeId) {
          return otherEdgeId != swEdgeId;
        })) {
      return true;
    }
  }
  for (const auto &obs : candidateTemporalObs) {
    MappingState::TemporalRouteGroupKey key{obs.nodeId, obs.tag};
    auto it = state.temporalRouteIndex.find(key);
    if (it == state.temporalRouteIndex.end())
      continue;
    if (llvm::any_of(it->second, [&](const MappingState::TemporalRouteUse &use) {
          return use.edgeId != swEdgeId &&
                 (use.inPortId != obs.inPortId ||
                  use.outPortId != obs.outPortId);
        })) {
      return true;
    }
  }

  return false;
}

// ---------------------------------------------------------------------------
// Mapper methods: conflict detection, legality, pathfinding, routability
// ---------------------------------------------------------------------------

bool Mapper::hasTaggedPathConflict(IdIndex swEdgeId,
                                   llvm::ArrayRef<IdIndex> candidatePath,
                                   const MappingState &state,
                                   const Graph &dfg, const Graph &adg) {
  RoutingPathAnalysisCache cache;
  return hasTaggedPathConflictCached(swEdgeId, candidatePath, cache, state, dfg,
                                     adg);
}

bool Mapper::validateTaggedPathConflicts(
    const MappingState &state, const Graph &dfg, const Graph &adg,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds, std::string &diagnostics) {
  bool valid = true;
  if (!state.hasTaggedResources)
    return true;
  llvm::SmallVector<llvm::SmallVector<routing_detail::TaggedPathObservation, 8>,
                    8>
      observationsByEdge(state.swEdgeToHwPaths.size());
  llvm::SmallVector<
      llvm::SmallVector<routing_detail::TemporalSwitchTagRouteObservation, 8>,
      8>
      temporalObsByEdge(state.swEdgeToHwPaths.size());
  RoutingPathAnalysisCache cache;

  for (IdIndex edgeId = 0;
       edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size()); ++edgeId) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    const auto &path = state.swEdgeToHwPaths[edgeId];
    if (path.empty())
      continue;
    if (auto failure =
            getRuntimeTagPathFailure(edgeId, path, cache, state, dfg, adg)) {
      diagnostics += "C4: runtime tag " + std::to_string(failure->tag) +
                     " cannot be represented on hw_port " +
                     std::to_string(failure->portId);
      if (failure->tagWidth != 0)
        diagnostics += " with tag_width=" + std::to_string(failure->tagWidth);
      diagnostics += " for sw_edge " + std::to_string(edgeId) + "\n";
      valid = false;
      continue;
    }
    auto taggedObs =
        getTaggedPathObservations(edgeId, path, cache, state, dfg, adg);
    observationsByEdge[edgeId].append(taggedObs.begin(), taggedObs.end());
    auto temporalObs =
        getTemporalSwitchTagRouteObservations(edgeId, path, cache, state, dfg,
                                              adg);
    temporalObsByEdge[edgeId].append(temporalObs.begin(), temporalObs.end());
  }

  for (IdIndex lhsEdge = 0;
       lhsEdge < static_cast<IdIndex>(observationsByEdge.size()); ++lhsEdge) {
    for (IdIndex rhsEdge = lhsEdge + 1;
         rhsEdge < static_cast<IdIndex>(observationsByEdge.size()); ++rhsEdge) {
      for (const auto &lhsObs : observationsByEdge[lhsEdge]) {
        for (const auto &rhsObs : observationsByEdge[rhsEdge]) {
          if (!routing_detail::observationsConflict(lhsObs, rhsObs))
            continue;
          if (lhsObs.kind == MappingState::TaggedObservationKind::RoutingOutput) {
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
          if (!routing_detail::temporalSwitchTagRouteConflict(lhsObs, rhsObs))
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

bool isEdgeLegalCached(const ConnectivityMatrix &connectivity, IdIndex srcPort,
                       IdIndex dstPort,
                       IdIndex swEdgeId,
                       llvm::ArrayRef<IdIndex> candidatePath,
                       RoutingPathAnalysisCache &cache,
                       const MappingState &state, const Graph &dfg,
                       const Graph &adg) {
  const Port *sp = adg.getPort(srcPort);
  const Port *dp = adg.getPort(dstPort);
  if (!sp || !dp)
    return false;

  if (sp->direction != Port::Output || dp->direction != Port::Input)
    return false;

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

  if (!isRuntimeTagPathRepresentable(swEdgeId, candidatePath, cache, state, dfg,
                                     adg))
    return false;

  if (isNonRoutingBroadcastTransitionConflict(swEdgeId, candidatePath, state,
                                              adg)) {
    return false;
  }

  if (isTaggedPort(sp) && isTaggedPort(dp) &&
      hasTaggedHardwareEdgeConflict(swEdgeId, candidatePath, srcPort, dstPort,
                                    cache, state, dfg, adg)) {
    return false;
  }

  return true;
}

bool Mapper::isEdgeLegal(IdIndex srcPort, IdIndex dstPort, IdIndex swEdgeId,
                         llvm::ArrayRef<IdIndex> candidatePath,
                         const MappingState &state, const Graph &dfg,
                         const Graph &adg) {
  RoutingPathAnalysisCache cache;
  return isEdgeLegalCached(connectivity, srcPort, dstPort, swEdgeId,
                           candidatePath, cache, state, dfg, adg);
}

llvm::SmallVector<IdIndex, 8>
Mapper::findPath(IdIndex srcHwPort, IdIndex dstHwPort, IdIndex swEdgeId,
                 const MappingState &state, const Graph &dfg,
                 const Graph &adg,
                 const llvm::DenseMap<IdIndex, double> &routingOutputHistory,
                 IdIndex forcedFirstHop,
                 const CongestionState *congestion) {
  llvm::SmallVector<IdIndex, 8> path;
  RoutingPathAnalysisCache analysisCache;

  if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
    return path;

  // Direct connection check.
  llvm::SmallVector<IdIndex, 8> directPath{srcHwPort, dstHwPort};
  if ((forcedFirstHop == INVALID_ID || forcedFirstHop == dstHwPort) &&
      isEdgeLegalCached(connectivity, srcHwPort, dstHwPort, swEdgeId, directPath,
                        analysisCache, state, dfg, adg) &&
      !hasTaggedPathConflictCached(swEdgeId, directPath, analysisCache, state,
                                   dfg, adg)) {
    path.push_back(srcHwPort);
    path.push_back(dstHwPort);
    return path;
  }

  struct TrailNode {
    IdIndex portId = INVALID_ID;
    int prevTrail = -1;
    unsigned depth = 0;
  };

  struct SearchEntry {
    double gCost = 0.0;
    double fCost = 0.0;
    IdIndex portId = INVALID_ID;
    int trailIndex = -1;
    unsigned depth = 0;
    std::optional<uint64_t> currentTag;
    SearchStateKey stateKey;
  };

  struct SearchEntryLess {
    bool operator()(const SearchEntry &lhs, const SearchEntry &rhs) const {
      if (std::abs(lhs.fCost - rhs.fCost) > 1e-9)
        return lhs.fCost > rhs.fCost;
      if (std::abs(lhs.gCost - rhs.gCost) > 1e-9)
        return lhs.gCost > rhs.gCost;
      if (lhs.depth != rhs.depth)
        return lhs.depth > rhs.depth;
      return lhs.portId > rhs.portId;
    }
  };

  auto estimateRemainingCost = [&](IdIndex currentPortId) -> double {
    if (!activeFlattener)
      return 0.0;
    const Port *currentPort = adg.getPort(currentPortId);
    const Port *dstPort = adg.getPort(dstHwPort);
    if (!currentPort || !dstPort || currentPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      return 0.0;
    auto [srcRow, srcCol] =
        activeFlattener->getNodeGridPos(currentPort->parentNode);
    auto [dstRow, dstCol] =
        activeFlattener->getNodeGridPos(dstPort->parentNode);
    if (srcRow < 0 || srcCol < 0 || dstRow < 0 || dstCol < 0)
      return 0.0;
    int dr = std::abs(srcRow - dstRow);
    int dc = std::abs(srcCol - dstCol);
    int chebyshev = std::max(dr, dc);
    return static_cast<double>(chebyshev) * activeHeuristicWeight;
  };

  llvm::SmallVector<TrailNode, 64> trails;
  trails.push_back({srcHwPort, -1, 0});

  auto buildPathFromTrail = [&](int trailIndex,
                                IdIndex appendPort = INVALID_ID) {
    llvm::SmallVector<IdIndex, 16> reversed;
    while (trailIndex >= 0) {
      reversed.push_back(trails[trailIndex].portId);
      trailIndex = trails[trailIndex].prevTrail;
    }

    llvm::SmallVector<IdIndex, 8> rebuiltPath;
    rebuiltPath.reserve(reversed.size() + (appendPort == INVALID_ID ? 0 : 1));
    for (auto it = reversed.rbegin(); it != reversed.rend(); ++it)
      rebuiltPath.push_back(*it);
    if (appendPort != INVALID_ID)
      rebuiltPath.push_back(appendPort);
    return rebuiltPath;
  };

  std::priority_queue<SearchEntry, std::vector<SearchEntry>, SearchEntryLess>
      worklist;
  llvm::DenseMap<SearchStateKey, double, SearchStateKeyInfo> bestCost;
  auto externalTagAtPort = [&](IdIndex portId) -> std::optional<uint64_t> {
    return computeExternalRuntimeTagAtMappedPort(swEdgeId, portId, state, dfg,
                                                 adg);
  };
  auto makeSearchKey = [&](IdIndex portId, std::optional<uint64_t> tag) {
    return SearchStateKey{portId, tag.has_value(), tag.value_or(0)};
  };

  std::optional<uint64_t> initialTag;
  if (state.hasTaggedResources) {
    llvm::SmallVector<IdIndex, 8> initialTagPath;
    const Edge *swEdge = dfg.getEdge(swEdgeId);
    const Port *srcSwPort = swEdge ? dfg.getPort(swEdge->srcPort) : nullptr;
    if (srcSwPort && srcSwPort->parentNode != INVALID_ID) {
      const Node *srcSwNode = dfg.getNode(srcSwPort->parentNode);
      if (srcSwNode &&
          routing_detail::isSoftwareMemoryInterfaceOpName(
              getNodeAttrStr(srcSwNode, "op_name")) &&
          srcSwPort->parentNode < state.swNodeToHwNode.size()) {
        IdIndex hwNodeId = state.swNodeToHwNode[srcSwPort->parentNode];
        if (hwNodeId != INVALID_ID) {
          auto prefix = findBridgePathBackward(adg, srcHwPort, hwNodeId);
          initialTagPath.append(prefix.begin(), prefix.end());
        }
      }
    }
    initialTagPath.push_back(srcHwPort);
    auto initialInfo = computeRuntimeTagValueInfoAlongMappedPath(
        swEdgeId, initialTagPath, initialTagPath.size() - 1, state, dfg, adg);
    if (!initialInfo.representable)
      return path;
    initialTag = initialInfo.tag;
  }

  SearchStateKey srcKey = makeSearchKey(srcHwPort, initialTag);
  worklist.push({0.0, estimateRemainingCost(srcHwPort), srcHwPort, 0, 0,
                 initialTag, srcKey});
  bestCost[srcKey] = 0.0;

  while (!worklist.empty()) {
    SearchEntry current = worklist.top();
    worklist.pop();

    auto bestIt = bestCost.find(current.stateKey);
    if (bestIt != bestCost.end() && current.gCost > bestIt->second + 1e-9)
      continue;

    const Port *currentPort = adg.getPort(current.portId);
    if (!currentPort)
      continue;

    if (current.portId == dstHwPort) {
      auto candidatePath = buildPathFromTrail(current.trailIndex);
      if (!hasTaggedPathConflictCached(swEdgeId, candidatePath, analysisCache,
                                       state, dfg, adg))
        return candidatePath;
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
        auto nextPath = buildPathFromTrail(current.trailIndex, nextInputPort);

        if (!isEdgeLegalCached(connectivity, current.portId, nextInputPort,
                               swEdgeId, nextPath, analysisCache, state, dfg,
                               adg))
          continue;
        auto nextTagInfo =
            advanceRuntimeTagValueInfoAtPort(current.currentTag, nextInputPort,
                                             adg, externalTagAtPort);
        if (!nextTagInfo.representable)
          continue;
        SearchStateKey nextKey = makeSearchKey(nextInputPort, nextTagInfo.tag);

        double nextCost = current.gCost + 1.0;
        auto nextBestIt = bestCost.find(nextKey);
        if (nextBestIt != bestCost.end() &&
            nextCost >= nextBestIt->second - 1e-9)
          continue;

        bestCost[nextKey] = nextCost;
        int nextTrailIndex = static_cast<int>(trails.size());
        trails.push_back({nextInputPort, current.trailIndex, current.depth + 1});
        worklist.push({nextCost, nextCost + estimateRemainingCost(nextInputPort),
                       nextInputPort, nextTrailIndex, current.depth + 1,
                       nextTagInfo.tag, nextKey});
      }
      continue;
    }

    auto internalIt = connectivity.inToOut.find(current.portId);
    if (internalIt == connectivity.inToOut.end())
      continue;

    for (IdIndex outPortId : internalIt->second) {
      auto outPath = buildPathFromTrail(current.trailIndex, outPortId);
      if (!isInternalHopLegal(current.portId, outPortId, swEdgeId, outPath,
                              analysisCache, state, dfg, adg))
        continue;

      double historyPenalty = 0.0;
      if (auto it = routingOutputHistory.find(outPortId);
          it != routingOutputHistory.end())
        historyPenalty = it->second;

      double congestionPenalty = 0.0;
      if (congestion)
        congestionPenalty = congestion->resourceCost(outPortId);

      auto nextTagInfo = advanceRuntimeTagValueInfoAtPort(current.currentTag,
                                                          outPortId, adg,
                                                          externalTagAtPort);
      if (!nextTagInfo.representable)
        continue;
      SearchStateKey nextKey = makeSearchKey(outPortId, nextTagInfo.tag);

      double nextCost = current.gCost + 1.0 + historyPenalty + congestionPenalty;
      auto nextBestIt = bestCost.find(nextKey);
      if (nextBestIt != bestCost.end() &&
          nextCost >= nextBestIt->second - 1e-9)
        continue;

      bestCost[nextKey] = nextCost;
      int nextTrailIndex = static_cast<int>(trails.size());
      trails.push_back({outPortId, current.trailIndex, current.depth + 1});
      worklist.push(
          {nextCost, nextCost + estimateRemainingCost(outPortId), outPortId,
           nextTrailIndex, current.depth + 1, nextTagInfo.tag, nextKey});
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

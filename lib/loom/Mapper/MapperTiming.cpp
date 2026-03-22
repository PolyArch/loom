#include "loom/Mapper/MapperTiming.h"

#include "loom/Mapper/Graph.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <queue>

namespace loom {

namespace {

bool isBypassableFifoNode(const Node *hwNode) {
  if (!hwNode || getNodeAttrStr(hwNode, "op_kind") != "fifo")
    return false;
  for (const auto &attr : hwNode->attributes) {
    if (attr.getName().strref() == "bypassable") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        return boolAttr.getValue();
    }
  }
  return false;
}

bool isBypassedFifoNode(IdIndex hwNodeId, const MappingState &state,
                        const Graph &adg) {
  const Node *hwNode = adg.getNode(hwNodeId);
  if (!hwNode || getNodeAttrStr(hwNode, "op_kind") != "fifo")
    return false;
  bool bypassable = false;
  bool bypassed = false;
  for (const auto &attr : hwNode->attributes) {
    if (attr.getName().strref() == "bypassable") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        bypassable = boolAttr.getValue();
    } else if (attr.getName().strref() == "bypassed") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        bypassed = boolAttr.getValue();
    }
  }
  if (hwNodeId < state.hwNodeFifoBypassedOverride.size()) {
    int8_t overrideValue = state.hwNodeFifoBypassedOverride[hwNodeId];
    if (overrideValue == 0)
      bypassed = false;
    else if (overrideValue > 0)
      bypassed = true;
  }
  return bypassable && bypassed;
}

bool isEffectiveBufferedFifoNode(IdIndex hwNodeId, const MappingState &state,
                                 const Graph &adg) {
  const Node *hwNode = adg.getNode(hwNodeId);
  if (!hwNode || getNodeAttrStr(hwNode, "op_kind") != "fifo")
    return false;
  return !isBypassedFifoNode(hwNodeId, state, adg);
}

bool isMapperSelectedBufferedFifoNode(IdIndex hwNodeId, const MappingState &state,
                                      const Graph &adg) {
  const Node *hwNode = adg.getNode(hwNodeId);
  if (!isBypassableFifoNode(hwNode))
    return false;
  if (hwNodeId >= state.hwNodeFifoBypassedOverride.size())
    return false;
  return state.hwNodeFifoBypassedOverride[hwNodeId] == 0 &&
         isEffectiveBufferedFifoNode(hwNodeId, state, adg);
}

unsigned mappedNodeLatencyCycles(IdIndex swNode, const MappingState &state,
                                 const Graph &adg) {
  if (swNode >= state.swNodeToHwNode.size())
    return 0;
  IdIndex hwNodeId = state.swNodeToHwNode[swNode];
  const Node *hwNode = adg.getNode(hwNodeId);
  if (!hwNode)
    return 0;
  return static_cast<unsigned>(
      std::max<int64_t>(0, getNodeAttrInt(hwNode, "latency", 0)));
}

unsigned mappedNodeInterval(IdIndex swNode, const MappingState &state,
                            const Graph &adg) {
  if (swNode >= state.swNodeToHwNode.size())
    return 1;
  IdIndex hwNodeId = state.swNodeToHwNode[swNode];
  const Node *hwNode = adg.getNode(hwNodeId);
  if (!hwNode)
    return 1;
  return static_cast<unsigned>(
      std::max<int64_t>(1, getNodeAttrInt(hwNode, "interval", 1)));
}

unsigned fifoNodeDepth(IdIndex hwNodeId, const Graph &adg) {
  const Node *hwNode = adg.getNode(hwNodeId);
  if (!hwNode || getNodeAttrStr(hwNode, "op_kind") != "fifo")
    return 0;
  return static_cast<unsigned>(
      std::max<int64_t>(0, getNodeAttrInt(hwNode, "depth", 0)));
}

struct EdgeTimingContribution {
  double combinationalDelay = 0.0;
  double trailingCombinationalDelay = 0.0;
  unsigned fifoStageCuts = 0;
  bool bufferized = false;
};

EdgeTimingContribution analyzeEdgeTimingContribution(IdIndex edgeId,
                                                     const MappingState &state,
                                                     const Graph &adg,
                                                     const MapperTimingOptions &opts) {
  EdgeTimingContribution contribution;
  if (edgeId >= state.swEdgeToHwPaths.size())
    return contribution;
  const auto &path = state.swEdgeToHwPaths[edgeId];
  if (path.empty())
    return contribution;

  llvm::DenseSet<IdIndex> seenFifoNodes;
  unsigned routingInteriorHops = 0;
  unsigned trailingInteriorHops = 0;
  for (size_t pathIdx = 1; pathIdx + 1 < path.size(); ++pathIdx) {
    const Port *port = adg.getPort(path[pathIdx]);
    if (!port || port->parentNode == INVALID_ID)
      continue;
    ++routingInteriorHops;
    ++trailingInteriorHops;
    const Node *owner = adg.getNode(port->parentNode);
    if (!owner || getNodeAttrStr(owner, "op_kind") != "fifo")
      continue;
    if (!seenFifoNodes.insert(port->parentNode).second)
      continue;
    contribution.bufferized = true;
    if (isEffectiveBufferedFifoNode(port->parentNode, state, adg)) {
      ++contribution.fifoStageCuts;
      trailingInteriorHops = 0;
    }
  }

  contribution.combinationalDelay =
      static_cast<double>(routingInteriorHops) * opts.routingHopDelay;
  contribution.trailingCombinationalDelay =
      static_cast<double>(trailingInteriorHops) * opts.routingHopDelay;
  return contribution;
}

bool isRecurrenceRelevantNode(const Node *node) {
  return node && node->kind == Node::OperationNode;
}

IdIndex findCarryNextEdge(IdIndex carryNodeId, const Graph &dfg) {
  const Node *carryNode = dfg.getNode(carryNodeId);
  if (!carryNode || getNodeAttrStr(carryNode, "op_name") != "dataflow.carry" ||
      carryNode->inputPorts.size() < 3) {
    return INVALID_ID;
  }
  IdIndex nextPortId = carryNode->inputPorts[2];
  const Port *nextPort = dfg.getPort(nextPortId);
  if (!nextPort)
    return INVALID_ID;
  for (IdIndex edgeId : nextPort->connectedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (edge && edge->dstPort == nextPortId)
      return edgeId;
  }
  return INVALID_ID;
}

bool isOperationEdge(IdIndex edgeId, const Graph &dfg,
                     llvm::ArrayRef<TechMappedEdgeKind> edgeKinds) {
  if (edgeId >= static_cast<IdIndex>(dfg.edges.size()))
    return false;
  if (edgeId < edgeKinds.size() &&
      edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU)
    return false;
  const Edge *edge = dfg.getEdge(edgeId);
  if (!edge)
    return false;
  const Port *srcPort = dfg.getPort(edge->srcPort);
  const Port *dstPort = dfg.getPort(edge->dstPort);
  if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
      dstPort->parentNode == INVALID_ID) {
    return false;
  }
  return isRecurrenceRelevantNode(dfg.getNode(srcPort->parentNode)) &&
         isRecurrenceRelevantNode(dfg.getNode(dstPort->parentNode));
}

} // namespace

MapperTimingSummary analyzeMapperTiming(const MappingState &state,
                                       const Graph &dfg, const Graph &adg,
                                       llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                       const MapperTimingOptions &opts) {
  MapperTimingSummary summary;

  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> outgoingEdgesByNode;
  llvm::DenseMap<IdIndex, unsigned> indegreeByNode;
  llvm::DenseSet<IdIndex> operationNodes;

  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (!isOperationEdge(edgeId, dfg, edgeKinds))
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    IdIndex srcNode = srcPort->parentNode;
    IdIndex dstNode = dstPort->parentNode;
    outgoingEdgesByNode[srcNode].push_back(edgeId);
    ++indegreeByNode[dstNode];
    operationNodes.insert(srcNode);
    operationNodes.insert(dstNode);
  }

  llvm::DenseMap<IdIndex, unsigned> indexByNode;
  llvm::DenseMap<IdIndex, unsigned> lowlinkByNode;
  llvm::DenseSet<IdIndex> onStack;
  llvm::SmallVector<IdIndex, 16> stack;
  std::vector<llvm::SmallVector<IdIndex, 8>> recurrenceSCCs;
  unsigned nextIndex = 0;

  std::function<void(IdIndex)> strongConnect = [&](IdIndex nodeId) {
    indexByNode[nodeId] = nextIndex;
    lowlinkByNode[nodeId] = nextIndex;
    ++nextIndex;
    stack.push_back(nodeId);
    onStack.insert(nodeId);

    for (IdIndex edgeId : outgoingEdgesByNode[nodeId]) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      const Port *dstPort = dfg.getPort(edge->dstPort);
      if (!dstPort || dstPort->parentNode == INVALID_ID)
        continue;
      IdIndex succ = dstPort->parentNode;
      if (!indexByNode.count(succ)) {
        strongConnect(succ);
        lowlinkByNode[nodeId] =
            std::min(lowlinkByNode[nodeId], lowlinkByNode[succ]);
      } else if (onStack.count(succ)) {
        lowlinkByNode[nodeId] =
            std::min(lowlinkByNode[nodeId], indexByNode[succ]);
      }
    }

    if (lowlinkByNode[nodeId] != indexByNode[nodeId])
      return;

    llvm::SmallVector<IdIndex, 8> component;
    while (!stack.empty()) {
      IdIndex member = stack.back();
      stack.pop_back();
      onStack.erase(member);
      component.push_back(member);
      if (member == nodeId)
        break;
    }

    bool hasSelfLoop = false;
    if (component.size() == 1) {
      for (IdIndex edgeId : outgoingEdgesByNode[component.front()]) {
        const Edge *edge = dfg.getEdge(edgeId);
        const Port *dstPort = edge ? dfg.getPort(edge->dstPort) : nullptr;
        if (dstPort && dstPort->parentNode == component.front()) {
          hasSelfLoop = true;
          break;
        }
      }
    }
    if (component.size() > 1 || hasSelfLoop)
      recurrenceSCCs.push_back(component);
  };

  for (IdIndex nodeId : operationNodes) {
    if (!indexByNode.count(nodeId))
      strongConnect(nodeId);
  }

  llvm::DenseSet<IdIndex> recurrenceEdges;
  for (size_t cycleIdx = 0; cycleIdx < recurrenceSCCs.size(); ++cycleIdx) {
    const auto &component = recurrenceSCCs[cycleIdx];
    llvm::DenseSet<IdIndex> componentNodes(component.begin(), component.end());
    MapperRecurrenceCycleSummary cycleSummary;
    cycleSummary.cycleId = static_cast<unsigned>(cycleIdx);
    cycleSummary.swNodes.assign(component.begin(), component.end());

    for (IdIndex nodeId : component) {
      cycleSummary.sequentialLatencyCycles +=
          mappedNodeLatencyCycles(nodeId, state, adg);
      cycleSummary.maxIntervalOnCycle =
          std::max(cycleSummary.maxIntervalOnCycle,
                   mappedNodeInterval(nodeId, state, adg));
      cycleSummary.combinationalDelay += opts.combinationalNodeDelay;

      for (IdIndex edgeId : outgoingEdgesByNode[nodeId]) {
        const Edge *edge = dfg.getEdge(edgeId);
        if (!edge)
          continue;
        const Port *dstPort = dfg.getPort(edge->dstPort);
        if (!dstPort || dstPort->parentNode == INVALID_ID ||
            !componentNodes.count(dstPort->parentNode)) {
          continue;
        }
        cycleSummary.swEdges.push_back(edgeId);
        recurrenceEdges.insert(edgeId);
        EdgeTimingContribution edgeContribution =
            analyzeEdgeTimingContribution(edgeId, state, adg, opts);
        cycleSummary.combinationalDelay += edgeContribution.combinationalDelay;
        cycleSummary.fifoStageCutContribution += edgeContribution.fifoStageCuts;
      }
    }

    cycleSummary.sequentialLatencyCycles +=
        cycleSummary.fifoStageCutContribution;
    cycleSummary.estimatedCycleII = std::max<unsigned>(
        cycleSummary.maxIntervalOnCycle,
        static_cast<unsigned>(std::ceil(
            static_cast<double>(cycleSummary.sequentialLatencyCycles) /
            static_cast<double>(cycleSummary.recurrenceDistance))));
    summary.estimatedInitiationInterval = std::max(
        summary.estimatedInitiationInterval, cycleSummary.estimatedCycleII);
    summary.recurrencePressure += cycleSummary.combinationalDelay;
    summary.recurrenceCycles.push_back(std::move(cycleSummary));
  }

  for (IdIndex nodeId : operationNodes) {
    const Node *node = dfg.getNode(nodeId);
    if (!node || getNodeAttrStr(node, "op_name") != "dataflow.carry")
      continue;
    IdIndex edgeId = findCarryNextEdge(nodeId, dfg);
    if (edgeId == INVALID_ID || recurrenceEdges.count(edgeId))
      continue;

    MapperRecurrenceCycleSummary cycleSummary;
    cycleSummary.cycleId =
        static_cast<unsigned>(summary.recurrenceCycles.size());
    cycleSummary.swNodes.push_back(nodeId);

    const Edge *edge = dfg.getEdge(edgeId);
    const Port *srcPort = edge ? dfg.getPort(edge->srcPort) : nullptr;
    if (srcPort && srcPort->parentNode != INVALID_ID &&
        srcPort->parentNode != nodeId) {
      cycleSummary.swNodes.push_back(srcPort->parentNode);
    }

    for (IdIndex cycleNode : cycleSummary.swNodes) {
      cycleSummary.sequentialLatencyCycles +=
          mappedNodeLatencyCycles(cycleNode, state, adg);
      cycleSummary.maxIntervalOnCycle =
          std::max(cycleSummary.maxIntervalOnCycle,
                   mappedNodeInterval(cycleNode, state, adg));
      cycleSummary.combinationalDelay += opts.combinationalNodeDelay;
    }

    cycleSummary.swEdges.push_back(edgeId);
    recurrenceEdges.insert(edgeId);
    EdgeTimingContribution edgeContribution =
        analyzeEdgeTimingContribution(edgeId, state, adg, opts);
    cycleSummary.combinationalDelay += edgeContribution.combinationalDelay;
    cycleSummary.fifoStageCutContribution += edgeContribution.fifoStageCuts;
    cycleSummary.sequentialLatencyCycles +=
        cycleSummary.fifoStageCutContribution;
    cycleSummary.estimatedCycleII = std::max<unsigned>(
        cycleSummary.maxIntervalOnCycle,
        static_cast<unsigned>(std::ceil(
            static_cast<double>(cycleSummary.sequentialLatencyCycles) /
            static_cast<double>(cycleSummary.recurrenceDistance))));
    summary.estimatedInitiationInterval = std::max(
        summary.estimatedInitiationInterval, cycleSummary.estimatedCycleII);
    summary.recurrencePressure += cycleSummary.combinationalDelay;
    summary.recurrenceCycles.push_back(std::move(cycleSummary));
  }

  llvm::DenseMap<IdIndex, double> longestPathByNode;
  llvm::DenseMap<IdIndex, IdIndex> predecessorEdgeByNode;
  llvm::DenseMap<IdIndex, IdIndex> predecessorNodeByNode;
  std::queue<IdIndex> ready;
  for (IdIndex nodeId : operationNodes) {
    if (indegreeByNode.lookup(nodeId) == 0)
      ready.push(nodeId);
  }

  llvm::DenseMap<IdIndex, unsigned> remainingIndegree = indegreeByNode;
  while (!ready.empty()) {
    IdIndex nodeId = ready.front();
    ready.pop();
    double nodeDelay = opts.combinationalNodeDelay;
    longestPathByNode[nodeId] =
        std::max(longestPathByNode[nodeId], nodeDelay);
    for (IdIndex edgeId : outgoingEdgesByNode[nodeId]) {
      if (recurrenceEdges.count(edgeId))
        continue;
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      const Port *dstPort = dfg.getPort(edge->dstPort);
      if (!dstPort || dstPort->parentNode == INVALID_ID)
        continue;
      IdIndex dstNode = dstPort->parentNode;
      EdgeTimingContribution edgeContribution =
          analyzeEdgeTimingContribution(edgeId, state, adg, opts);
      double candidateDelay = 0.0;
      if (edgeContribution.fifoStageCuts > 0) {
        // Buffered FIFOs on one routed edge break the combinational path in the
        // first production timing model. Use only the downstream segment after
        // the last effective stage cut as the sink-visible edge delay.
        candidateDelay =
            edgeContribution.trailingCombinationalDelay +
            opts.combinationalNodeDelay;
      } else {
        candidateDelay =
            longestPathByNode[nodeId] + edgeContribution.combinationalDelay +
            opts.combinationalNodeDelay;
      }
      if (candidateDelay > longestPathByNode.lookup(dstNode)) {
        longestPathByNode[dstNode] = candidateDelay;
        predecessorEdgeByNode[dstNode] = edgeId;
        predecessorNodeByNode[dstNode] = nodeId;
      }
      if (remainingIndegree.lookup(dstNode) > 0) {
        unsigned updated = --remainingIndegree[dstNode];
        if (updated == 0)
          ready.push(dstNode);
      }
    }
  }

  IdIndex criticalSink = INVALID_ID;
  for (const auto &entry : longestPathByNode)
    if (entry.second > summary.estimatedCriticalPathDelay) {
      summary.estimatedCriticalPathDelay = entry.second;
      criticalSink = entry.first;
    }

  while (criticalSink != INVALID_ID) {
    auto edgeIt = predecessorEdgeByNode.find(criticalSink);
    auto predIt = predecessorNodeByNode.find(criticalSink);
    if (edgeIt == predecessorEdgeByNode.end() ||
        predIt == predecessorNodeByNode.end()) {
      break;
    }
    summary.criticalPathEdges.push_back(edgeIt->second);
    criticalSink = predIt->second;
  }
  std::reverse(summary.criticalPathEdges.begin(), summary.criticalPathEdges.end());

  summary.estimatedClockPeriod =
      std::max(summary.estimatedCriticalPathDelay, summary.recurrencePressure);
  summary.estimatedThroughputCost =
      summary.estimatedClockPeriod *
      static_cast<double>(summary.estimatedInitiationInterval);

  llvm::DenseSet<IdIndex> seenBufferedFifos;
  llvm::DenseSet<IdIndex> seenForcedBufferedFifos;
  llvm::DenseSet<IdIndex> seenMapperSelectedBufferedFifos;
  llvm::DenseSet<IdIndex> seenBufferizedEdges;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++edgeId) {
    EdgeTimingContribution edgeContribution =
        analyzeEdgeTimingContribution(edgeId, state, adg, opts);
    if (edgeContribution.bufferized && seenBufferizedEdges.insert(edgeId).second)
      summary.bufferizedEdges.push_back(edgeId);
    const auto &path = state.swEdgeToHwPaths[edgeId];
    for (IdIndex portId : path) {
      const Port *port = adg.getPort(portId);
      if (!port || port->parentNode == INVALID_ID)
        continue;
      const Node *owner = adg.getNode(port->parentNode);
      if (!owner || getNodeAttrStr(owner, "op_kind") != "fifo")
        continue;
      if (isEffectiveBufferedFifoNode(port->parentNode, state, adg) &&
          seenBufferedFifos.insert(port->parentNode).second)
        ++summary.fifoBufferCount;
      if (isEffectiveBufferedFifoNode(port->parentNode, state, adg) &&
          seenForcedBufferedFifos.insert(port->parentNode).second) {
        ++summary.forcedBufferedFifoCount;
        summary.forcedBufferedFifoNodes.push_back(port->parentNode);
        summary.forcedBufferedFifoDepths.push_back(
            fifoNodeDepth(port->parentNode, adg));
      }
      if (isMapperSelectedBufferedFifoNode(port->parentNode, state, adg) &&
          seenMapperSelectedBufferedFifos.insert(port->parentNode).second) {
        ++summary.mapperSelectedBufferedFifoCount;
        summary.mapperSelectedBufferedFifoNodes.push_back(port->parentNode);
        summary.mapperSelectedBufferedFifoDepths.push_back(
            fifoNodeDepth(port->parentNode, adg));
      }
    }
  }

  return summary;
}

} // namespace loom

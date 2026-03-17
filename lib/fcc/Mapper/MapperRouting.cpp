#include "fcc/Mapper/Mapper.h"

#include "fcc/Dialect/Fabric/FabricTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <queue>
#include <random>

namespace fcc {

namespace {

bool pathUsesPort(IdIndex portId, const MappingState &state) {
  for (const auto &path : state.swEdgeToHwPaths) {
    for (IdIndex usedPort : path) {
      if (usedPort == portId)
        return true;
    }
  }
  return false;
}

bool isSpatialSwitchOutputPort(IdIndex portId, const Graph &adg) {
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

  // Temporal/tagged routing can share an output port by tag. The current
  // spatial_sw path must remain exclusive per output port.
  return !mlir::isa<fcc::fabric::TaggedType>(port->type);
}

bool isInternalHopLegal(IdIndex inPortId, IdIndex outPortId,
                        const MappingState &state, const Graph &adg) {
  const Port *inPort = adg.getPort(inPortId);
  const Port *outPort = adg.getPort(outPortId);
  if (!inPort || !outPort)
    return false;
  if (inPort->parentNode == INVALID_ID || outPort->parentNode == INVALID_ID)
    return false;
  if (inPort->parentNode != outPort->parentNode)
    return false;

  if (isSpatialSwitchOutputPort(outPortId, adg) &&
      pathUsesPort(outPortId, state))
    return false;

  return true;
}

} // namespace

bool Mapper::isEdgeLegal(IdIndex srcPort, IdIndex dstPort,
                          const MappingState &state, const Graph &adg) {
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

  // C4: For the MVP, allow edge sharing in the fully-connected fabric.
  // In a real fabric with dedicated switch routing, edges are not exclusive
  // at this level. Exclusivity is enforced at the switch port level.
  return true;
}

llvm::SmallVector<IdIndex, 8>
Mapper::findPath(IdIndex srcHwPort, IdIndex dstHwPort,
                  const MappingState &state, const Graph &adg) {
  llvm::SmallVector<IdIndex, 8> path;

  if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
    return path;

  // Direct connection check.
  if (isEdgeLegal(srcHwPort, dstHwPort, state, adg)) {
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

      if (!isEdgeLegal(current.portId, nextInputPort, state, adg))
        continue;

      visited.insert(nextInputPort);
      auto nextPath = current.pathSoFar;
      nextPath.push_back(nextInputPort);

      if (nextInputPort == dstHwPort)
        return nextPath;

      // Traverse routing node internals.
      auto internalIt = connectivity.inToOut.find(nextInputPort);
      if (internalIt != connectivity.inToOut.end()) {
        for (IdIndex outPortId : internalIt->second) {
          if (visited.count(outPortId))
            continue;
          if (!isInternalHopLegal(nextInputPort, outPortId, state, adg))
            continue;
          visited.insert(outPortId);

          BFSEntry next;
          next.portId = outPortId;
          next.pathSoFar = nextPath;
          next.pathSoFar.push_back(outPortId);
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
                           const std::vector<IdIndex> &edgeOrder,
                           unsigned &routed, unsigned &total) {
  bool allRouted = true;
  routed = 0;
  total = 0;

  for (IdIndex edgeId : edgeOrder) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
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

    auto path = findPath(srcHwPort, dstHwPort, state, adg);
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
                         const Graph &adg, int seed) {
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
  bool allRouted = routeOnePass(state, dfg, adg, edgeOrder, routed, total);

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

    allRouted = routeOnePass(state, dfg, adg, newOrder, routed, total);
    llvm::outs() << "  Rip-up pass " << (pass + 1) << " result: " << routed
                 << "/" << total << " edges\n";
  }

  llvm::outs() << "  Final routing: " << routed << "/" << total << " edges\n";
  return allRouted;
}

} // namespace fcc

//===-- MapperRouting.cpp - BFS/A* routing for mapper --------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <queue>

namespace loom {

namespace {

/// Get the "resource_class" attribute from a node, or empty string.
llvm::StringRef getResClass(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "resource_class") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

} // namespace

bool Mapper::isEdgeLegal(IdIndex srcPort, IdIndex dstPort,
                         const MappingState &state, const Graph &adg) {
  // Check C2: type compatibility (relaxed for routing nodes).
  const Port *sp = adg.getPort(srcPort);
  const Port *dp = adg.getPort(dstPort);
  if (!sp || !dp)
    return false;

  // Direction check: source must be output, dest must be input.
  if (sp->direction != Port::Output || dp->direction != Port::Input)
    return false;

  // Check physical connectivity exists.
  auto it = connectivity.outToIn.find(srcPort);
  if (it == connectivity.outToIn.end() || it->second != dstPort)
    return false;

  // C3: Check exclusivity - find the ADG edge and verify it's not
  // already exclusively used by another SW edge.
  for (IdIndex edgeId : sp->connectedEdges) {
    const Edge *hwEdge = adg.getEdge(edgeId);
    if (!hwEdge || hwEdge->srcPort != srcPort || hwEdge->dstPort != dstPort)
      continue;

    if (edgeId < state.hwEdgeToSwEdges.size() &&
        !state.hwEdgeToSwEdges[edgeId].empty()) {
      // The edge is already used. Check if the destination node is a
      // routing node (tagged sharing allowed) or exclusive.
      const Node *dstNode = adg.getNode(dp->parentNode);
      if (!dstNode)
        return false;
      llvm::StringRef resClass = getResClass(dstNode);
      if (resClass != "routing") {
        // Exclusive edge already in use.
        return false;
      }
      // For routing nodes, check tag capacity (allow up to 256 tags).
      if (state.hwEdgeToSwEdges[edgeId].size() >= 256)
        return false;
    }
    break;
  }

  return true;
}

IdIndex Mapper::allocateTag(IdIndex hwPort, const MappingState &state,
                            const Graph &adg) {
  // Smallest-available-tag strategy.
  // Scan existing tag allocations on this port's connected edges.
  llvm::DenseSet<IdIndex> usedTags;

  const Port *port = adg.getPort(hwPort);
  if (!port)
    return 0;

  for (IdIndex edgeId : port->connectedEdges) {
    // Check if this edge is already used by any mapped SW edge.
    if (edgeId < state.hwEdgeToSwEdges.size()) {
      for (IdIndex swEdge : state.hwEdgeToSwEdges[edgeId]) {
        // The tag is implicitly the SW edge index for now.
        usedTags.insert(swEdge);
      }
    }
  }

  // Find smallest available tag.
  for (IdIndex tag = 0; tag < 256; ++tag) {
    if (!usedTags.count(tag))
      return tag;
  }

  return INVALID_ID; // Tag exhaustion.
}

llvm::SmallVector<IdIndex, 8>
Mapper::findPath(IdIndex srcHwPort, IdIndex dstHwPort,
                 const MappingState &state, const Graph &adg) {
  llvm::SmallVector<IdIndex, 8> path;

  if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
    return path;

  // Direct connection check with legality.
  auto directIt = connectivity.outToIn.find(srcHwPort);
  if (directIt != connectivity.outToIn.end() &&
      directIt->second == dstHwPort &&
      isEdgeLegal(srcHwPort, dstHwPort, state, adg)) {
    path.push_back(srcHwPort);
    path.push_back(dstHwPort);
    return path;
  }

  // BFS through connectivity matrix with edge legality checks.
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

    // From current output port, follow physical edge to input port.
    auto physIt = connectivity.outToIn.find(current.portId);
    if (physIt == connectivity.outToIn.end())
      continue;

    IdIndex nextInputPort = physIt->second;
    if (visited.count(nextInputPort))
      continue;

    // C2/C3: Check if this physical edge is legal.
    if (!isEdgeLegal(current.portId, nextInputPort, state, adg))
      continue;

    visited.insert(nextInputPort);
    auto nextPath = current.pathSoFar;
    nextPath.push_back(nextInputPort);

    // Check if we've reached the destination.
    if (nextInputPort == dstHwPort) {
      return nextPath;
    }

    // From input port, traverse routing node internals to output ports.
    auto internalIt = connectivity.inToOut.find(nextInputPort);
    if (internalIt != connectivity.inToOut.end()) {
      for (IdIndex outPortId : internalIt->second) {
        if (visited.count(outPortId))
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

  return path; // Empty = no path found.
}

bool Mapper::runRouting(MappingState &state, const Graph &dfg,
                        const Graph &adg) {
  bool allRouted = true;

  // Route each DFG edge.
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;

    // Get the mapped hardware ports for source and destination.
    IdIndex srcSwPort = edge->srcPort;
    IdIndex dstSwPort = edge->dstPort;

    if (srcSwPort >= state.swPortToHwPort.size() ||
        dstSwPort >= state.swPortToHwPort.size())
      continue;

    IdIndex srcHwPort = state.swPortToHwPort[srcSwPort];
    IdIndex dstHwPort = state.swPortToHwPort[dstSwPort];

    if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID) {
      allRouted = false;
      continue;
    }

    // Find a path through the ADG.
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
  }

  return allRouted;
}

} // namespace loom

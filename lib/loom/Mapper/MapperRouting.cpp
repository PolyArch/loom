//===-- MapperRouting.cpp - BFS/A* routing for mapper --------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"

#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <queue>
#include <random>

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

/// Get the "op_name" attribute from a node, or empty string.
llvm::StringRef getOpName(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "op_name") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Check if a DFG node is a memory-related operation.
bool isMemoryOp(const Node *node) {
  auto name = getOpName(node);
  return name.contains("memory") || name.contains("extmemory") ||
         name.contains("load") || name.contains("store");
}

/// Get bit width from an MLIR type (handles integer, float, bits, and index types).
unsigned getTypeBitWidth(mlir::Type type) {
  if (!type)
    return 0;
  if (auto bitsType = mlir::dyn_cast<loom::dataflow::BitsType>(type))
    return bitsType.getWidth();
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type))
    return intTy.getWidth();
  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(type))
    return floatTy.getWidth();
  return 0;
}

/// Check C2 bit-width compatibility for routing node edges.
/// Native types: only bit width must match (i32 ~ f32 both 32-bit OK).
/// Tagged-to-native mixing is never allowed.
bool isRoutingTypeCompatible(mlir::Type srcType, mlir::Type dstType) {
  if (!srcType || !dstType)
    return true; // Untyped ports are compatible.
  if (srcType == dstType)
    return true;

  unsigned srcWidth = getTypeBitWidth(srcType);
  unsigned dstWidth = getTypeBitWidth(dstType);

  // Both must have valid widths for routing compatibility.
  if (srcWidth == 0 || dstWidth == 0)
    return false;

  // Width must match for routing (not just <=, exact match for routing).
  return srcWidth == dstWidth;
}

} // namespace

bool Mapper::isEdgeLegal(IdIndex srcPort, IdIndex dstPort,
                         const MappingState &state, const Graph &adg,
                         const Graph *dfg, IdIndex swEdgeId) {
  const Port *sp = adg.getPort(srcPort);
  const Port *dp = adg.getPort(dstPort);
  if (!sp || !dp) {
    if (log)
      log->logEdgeRejection(srcPort, dstPort, "null port");
    return false;
  }

  // Direction check: source must be output, dest must be input.
  if (sp->direction != Port::Output || dp->direction != Port::Input) {
    if (log)
      log->logEdgeRejection(srcPort, dstPort, "direction mismatch");
    return false;
  }

  // Check physical connectivity exists.
  auto it = connectivity.outToIn.find(srcPort);
  if (it == connectivity.outToIn.end() || it->second != dstPort) {
    if (log)
      log->logEdgeRejection(srcPort, dstPort, "no physical connectivity");
    return false;
  }

  // C2: Bit-width compatibility for routing hops.
  // For routing nodes, enforce that bit widths match along the path.
  const Node *srcNode = adg.getNode(sp->parentNode);
  const Node *dstNode = adg.getNode(dp->parentNode);

  if (srcNode && dstNode) {
    bool srcIsRouting = getResClass(srcNode) == "routing";
    bool dstIsRouting = getResClass(dstNode) == "routing";

    // If either endpoint is a routing node, enforce bit-width compatibility.
    if (srcIsRouting || dstIsRouting) {
      if (sp->type && dp->type) {
        if (!isRoutingTypeCompatible(sp->type, dp->type)) {
          if (log)
            log->logEdgeRejection(srcPort, dstPort,
                                  "C2 bit-width mismatch (routing)");
          return false;
        }
      }
    }
  }

  // C3: Check exclusivity - find the ADG edge and verify it's not
  // already exclusively used by another SW edge.
  // Only edges whose source port carries a tagged type
  // (dataflow.tagged<bits<N>, iM>) support sharing, up to 2^M tags.
  for (IdIndex edgeId : sp->connectedEdges) {
    const Edge *hwEdge = adg.getEdge(edgeId);
    if (!hwEdge || hwEdge->srcPort != srcPort || hwEdge->dstPort != dstPort)
      continue;

    if (edgeId < state.hwEdgeToSwEdges.size() &&
        !state.hwEdgeToSwEdges[edgeId].empty()) {
      // Edge already in use. Check if sharing is legal.
      auto taggedType =
          mlir::dyn_cast<loom::dataflow::TaggedType>(sp->type);
      if (!taggedType) {
        // Non-tagged edge: check for valid fan-out sharing.
        // Fan-out is allowed when all SW edges on this HW edge originate
        // from the same DFG source port (they carry the same data value).
        bool isFanout = false;
        if (dfg && swEdgeId != INVALID_ID) {
          const Edge *swEdge = dfg->getEdge(swEdgeId);
          if (swEdge) {
            isFanout = true;
            for (IdIndex existingSwId : state.hwEdgeToSwEdges[edgeId]) {
              const Edge *existingSw = dfg->getEdge(existingSwId);
              if (!existingSw ||
                  existingSw->srcPort != swEdge->srcPort) {
                isFanout = false;
                break;
              }
            }
          }
        }
        if (!isFanout) {
          if (log)
            log->logEdgeRejection(
                srcPort, dstPort,
                "C3 exclusive (non-tagged, non-fanout, used by " +
                    std::to_string(state.hwEdgeToSwEdges[edgeId].size()) +
                    " SW edges)");
          return false; // Non-tagged, non-fanout: exclusive.
        }
      } else {
        // Tagged edge: capacity is 2^tagWidth.
        unsigned tagWidth = taggedType.getTagType().getWidth();
        unsigned maxTags = 1u << tagWidth;
        if (state.hwEdgeToSwEdges[edgeId].size() >= maxTags) {
          if (log)
            log->logEdgeRejection(
                srcPort, dstPort,
                "C3 tag capacity exhausted (i" +
                    std::to_string(tagWidth) + " max=" +
                    std::to_string(maxTags) + " used=" +
                    std::to_string(state.hwEdgeToSwEdges[edgeId].size()) +
                    ")");
          return false;
        }
      }
    }
    break;
  }

  return true;
}

IdIndex Mapper::allocateTag(IdIndex hwPort, const MappingState &state,
                            const Graph &adg) {
  // Deterministic smallest-available-tag strategy.
  // Scan all edges connected to this port to find tags already in use.
  llvm::DenseSet<IdIndex> usedTags;

  const Port *port = adg.getPort(hwPort);
  if (!port)
    return 0;

  // Derive max tag from the port's tagged type (2^tagWidth).
  unsigned maxTag = 0;
  if (auto taggedType =
          mlir::dyn_cast<loom::dataflow::TaggedType>(port->type)) {
    unsigned tagWidth = taggedType.getTagType().getWidth();
    maxTag = 1u << tagWidth;
  }
  if (maxTag == 0)
    return INVALID_ID; // Non-tagged port cannot allocate tags.

  for (IdIndex edgeId : port->connectedEdges) {
    if (edgeId >= state.hwEdgeToSwEdges.size())
      continue;
    // Count the number of SW edges already sharing this HW edge.
    // Each gets a unique tag value.
    size_t usageCount = state.hwEdgeToSwEdges[edgeId].size();
    for (size_t t = 0; t < usageCount; ++t) {
      usedTags.insert(static_cast<IdIndex>(t));
    }
  }

  // Find smallest available tag within the type-derived capacity.
  for (IdIndex tag = 0; tag < maxTag; ++tag) {
    if (!usedTags.count(tag))
      return tag;
  }

  return INVALID_ID; // Tag exhaustion.
}

llvm::SmallVector<IdIndex, 8>
Mapper::findPath(IdIndex srcHwPort, IdIndex dstHwPort,
                 const MappingState &state, const Graph &adg,
                 const Graph *dfg, IdIndex swEdgeId) {
  llvm::SmallVector<IdIndex, 8> path;

  if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
    return path;

  // Direct connection check with legality.
  auto directIt = connectivity.outToIn.find(srcHwPort);
  if (directIt != connectivity.outToIn.end() &&
      directIt->second == dstHwPort &&
      isEdgeLegal(srcHwPort, dstHwPort, state, adg, dfg, swEdgeId)) {
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
    if (!isEdgeLegal(current.portId, nextInputPort, state, adg,
                     dfg, swEdgeId))
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

llvm::SmallVector<IdIndex, 8>
Mapper::findPathRelaxed(IdIndex srcHwPort, IdIndex dstHwPort,
                        const MappingState &state, const Graph &adg,
                        llvm::SmallVector<IdIndex, 8> &blockingEdges) {
  llvm::SmallVector<IdIndex, 8> path;
  blockingEdges.clear();

  if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID)
    return path;

  // BFS ignoring C3 exclusivity but enforcing C2 bit-width compatibility.
  // Track which HW edges along the path have SW edges assigned (blockers).
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

    auto physIt = connectivity.outToIn.find(current.portId);
    if (physIt == connectivity.outToIn.end())
      continue;

    IdIndex nextInputPort = physIt->second;
    if (visited.count(nextInputPort))
      continue;

    // C2: check bit-width compatibility (skip C3 exclusivity).
    const Port *sp = adg.getPort(current.portId);
    const Port *dp = adg.getPort(nextInputPort);
    if (sp && dp && sp->type && dp->type) {
      if (!isRoutingTypeCompatible(sp->type, dp->type))
        continue;
    }

    visited.insert(nextInputPort);
    auto nextPath = current.pathSoFar;
    nextPath.push_back(nextInputPort);

    if (nextInputPort == dstHwPort) {
      // Found path. Collect blocking SW edges along this path.
      for (size_t i = 0; i + 1 < nextPath.size(); i += 2) {
        IdIndex outPort = nextPath[i];
        IdIndex inPort = nextPath[i + 1];
        const Port *op = adg.getPort(outPort);
        if (!op)
          continue;
        for (IdIndex hwEdgeId : op->connectedEdges) {
          const Edge *hwE = adg.getEdge(hwEdgeId);
          if (!hwE || hwE->srcPort != outPort || hwE->dstPort != inPort)
            continue;
          if (hwEdgeId < state.hwEdgeToSwEdges.size()) {
            for (IdIndex swEdge : state.hwEdgeToSwEdges[hwEdgeId])
              blockingEdges.push_back(swEdge);
          }
        }
      }
      return nextPath;
    }

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

  return path; // Empty = no relaxed path found.
}

bool Mapper::runRouting(MappingState &state, const Graph &dfg,
                        const Graph &adg, int seed) {
  bool allRouted = true;

  // Build edge order: prioritize memory-connected edges (most constrained
  // destinations) so they get first pick of routing channels, then shuffle
  // the rest for deadlock avoidance.
  std::vector<IdIndex> memEdges, otherEdges;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *e = dfg.getEdge(i);
    if (!e)
      continue;
    // Check if either endpoint maps to a memory/bridge HW node.
    bool isMemEdge = false;
    for (IdIndex swPort : {e->srcPort, e->dstPort}) {
      if (swPort >= state.swPortToHwPort.size())
        continue;
      IdIndex hwPort = state.swPortToHwPort[swPort];
      if (hwPort == INVALID_ID)
        continue;
      const Port *hp = adg.getPort(hwPort);
      if (!hp || hp->parentNode == INVALID_ID)
        continue;
      const Node *hwNode = adg.getNode(hp->parentNode);
      if (hwNode) {
        auto rc = getResClass(hwNode);
        if (rc == "memory" || rc == "bridge")
          isMemEdge = true;
      }
    }
    // Also prioritize edges to/from DFG sentinel nodes and memory ops.
    const Port *sp = dfg.getPort(e->srcPort);
    const Port *dp = dfg.getPort(e->dstPort);
    if (sp && sp->parentNode != INVALID_ID) {
      const Node *n = dfg.getNode(sp->parentNode);
      if (n && (n->kind == Node::ModuleInputNode ||
                n->kind == Node::ModuleOutputNode ||
                isMemoryOp(n)))
        isMemEdge = true;
    }
    if (dp && dp->parentNode != INVALID_ID) {
      const Node *n = dfg.getNode(dp->parentNode);
      if (n && (n->kind == Node::ModuleInputNode ||
                n->kind == Node::ModuleOutputNode ||
                isMemoryOp(n)))
        isMemEdge = true;
    }
    if (isMemEdge)
      memEdges.push_back(i);
    else
      otherEdges.push_back(i);
  }
  if (seed >= 0) {
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::shuffle(memEdges.begin(), memEdges.end(), rng);
    std::shuffle(otherEdges.begin(), otherEdges.end(), rng);
  }
  std::vector<IdIndex> edgeOrder;
  edgeOrder.reserve(memEdges.size() + otherEdges.size());
  edgeOrder.insert(edgeOrder.end(), memEdges.begin(), memEdges.end());
  edgeOrder.insert(edgeOrder.end(), otherEdges.begin(), otherEdges.end());

  // Route each DFG edge.
  for (IdIndex edgeId : edgeOrder) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;

    // Get the mapped hardware ports for source and destination.
    IdIndex srcSwPort = edge->srcPort;
    IdIndex dstSwPort = edge->dstPort;

    if (srcSwPort >= state.swPortToHwPort.size() ||
        dstSwPort >= state.swPortToHwPort.size())
      continue;

    // Skip internal group edges: if both endpoints map to the same HW
    // node AND both are members of a tech-mapped group on that node,
    // the connection is handled inside the PE (FU body) and does not
    // need switch-fabric routing. Inter-slot edges on temporal FU
    // sub-nodes are NOT group-internal and must be routed.
    const Port *srcSwPortPtr = dfg.getPort(srcSwPort);
    const Port *dstSwPortPtr = dfg.getPort(dstSwPort);
    if (srcSwPortPtr && dstSwPortPtr) {
      IdIndex srcSwNode = srcSwPortPtr->parentNode;
      IdIndex dstSwNode = dstSwPortPtr->parentNode;
      if (srcSwNode != INVALID_ID && dstSwNode != INVALID_ID &&
          srcSwNode < state.swNodeToHwNode.size() &&
          dstSwNode < state.swNodeToHwNode.size()) {
        IdIndex srcHwNode = state.swNodeToHwNode[srcSwNode];
        IdIndex dstHwNode = state.swNodeToHwNode[dstSwNode];
        if (srcHwNode != INVALID_ID && srcHwNode == dstHwNode) {
          // Only skip if both nodes are in the same group binding.
          bool isGroupInternal = false;
          auto groupIt = state.groupBindings.find(srcHwNode);
          if (groupIt != state.groupBindings.end()) {
            bool srcInGroup = false, dstInGroup = false;
            for (IdIndex gid : groupIt->second) {
              if (gid == srcSwNode) srcInGroup = true;
              if (gid == dstSwNode) dstInGroup = true;
            }
            isGroupInternal = srcInGroup && dstInGroup;
          }
          if (isGroupInternal)
            continue; // Truly internal group edge, skip routing.
        }
      }
    }

    IdIndex srcHwPort = state.swPortToHwPort[srcSwPort];
    IdIndex dstHwPort = state.swPortToHwPort[dstSwPort];

    if (srcHwPort == INVALID_ID || dstHwPort == INVALID_ID) {
      allRouted = false;
      continue;
    }

    // Find a path through the ADG (pass DFG + edge ID for fan-out sharing).
    auto path = findPath(srcHwPort, dstHwPort, state, adg, &dfg, edgeId);
    if (path.empty()) {
      if (log)
        log->logRouteAttempt(edgeId, srcHwPort, dstHwPort, false, 0);
      allRouted = false;
      continue;
    }

    auto mapResult = state.mapEdge(edgeId, path, dfg, adg);
    if (mapResult != ActionResult::Success) {
      if (log)
        log->logRouteAttempt(edgeId, srcHwPort, dstHwPort, false, 0);
      allRouted = false;
      continue;
    }

    if (log)
      log->logRouteAttempt(edgeId, srcHwPort, dstHwPort, true,
                           static_cast<unsigned>(path.size() / 2));
  }

  return allRouted;
}

} // namespace loom

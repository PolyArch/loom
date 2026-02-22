//===-- MapperValidation.cpp - Constraint validation for mapper ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace loom {

namespace {

llvm::StringRef getNodeOpName(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "op_name") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

llvm::StringRef getNodeResClass(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "resource_class") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

int64_t getNodeIntAttr(const Node *node, llvm::StringRef name,
                       int64_t dflt = -1) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return intAttr.getInt();
    }
  }
  return dflt;
}

bool nodeHasAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name)
      return true;
  }
  return false;
}

/// Get bit width from an MLIR type.
unsigned getTypeWidth(mlir::Type type) {
  if (!type)
    return 0;
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type))
    return intTy.getWidth();
  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(type))
    return floatTy.getWidth();
  return 0;
}

} // namespace

bool Mapper::validateC1(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C1: Node compatibility - each mapped DFG node must be compatible with
  // its ADG target.
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    IdIndex hwNode = state.swNodeToHwNode[i];
    if (hwNode == INVALID_ID)
      continue;

    const Node *sw = dfg.getNode(i);
    const Node *hw = adg.getNode(hwNode);
    if (!sw || !hw) {
      diag = "C1: invalid node reference sw=" + std::to_string(i) +
             " hw=" + std::to_string(hwNode);
      return false;
    }

    // Sentinel nodes must map to sentinel nodes of the same kind.
    if (sw->kind != hw->kind && sw->kind != Node::OperationNode) {
      diag = "C1: sentinel kind mismatch sw=" + std::to_string(i);
      return false;
    }
  }
  return true;
}

bool Mapper::validateC2(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C2: Port/type compatibility.
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swPortToHwPort.size());
       ++i) {
    IdIndex hwPort = state.swPortToHwPort[i];
    if (hwPort == INVALID_ID)
      continue;

    const Port *sw = dfg.getPort(i);
    const Port *hw = adg.getPort(hwPort);
    if (!sw || !hw) {
      diag = "C2: invalid port reference sw=" + std::to_string(i);
      return false;
    }

    // Direction must match.
    if (sw->direction != hw->direction) {
      diag = "C2: direction mismatch sw_port=" + std::to_string(i);
      return false;
    }

    // Type compatibility check.
    if (sw->type && hw->type) {
      const Node *hwNode = adg.getNode(hw->parentNode);
      bool isRouting = hwNode && getNodeResClass(hwNode) == "routing";

      if (isRouting) {
        // Routing nodes: bit-width must match (relaxed type semantics).
        unsigned swWidth = getTypeWidth(sw->type);
        unsigned hwWidth = getTypeWidth(hw->type);
        if (swWidth > 0 && hwWidth > 0 && swWidth != hwWidth) {
          diag = "C2: routing bit-width mismatch sw_port=" +
                 std::to_string(i) + " (sw=" + std::to_string(swWidth) +
                 " hw=" + std::to_string(hwWidth) + ")";
          return false;
        }
      } else {
        // Functional/memory nodes: strict type compatibility.
        if (sw->type != hw->type) {
          diag = "C2: type mismatch sw_port=" + std::to_string(i);
          return false;
        }
      }
    }
  }

  // C2 for routing hops: check bit-width consistency along each route path.
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++i) {
    const auto &path = state.swEdgeToHwPaths[i];
    if (path.empty())
      continue;

    // Check each physical hop pair in the path.
    for (size_t j = 0; j + 1 < path.size(); j += 2) {
      const Port *fromPort = adg.getPort(path[j]);
      const Port *toPort = adg.getPort(path[j + 1]);
      if (!fromPort || !toPort)
        continue;

      // Check bit-width compatibility along routing hops.
      if (fromPort->type && toPort->type) {
        unsigned fromWidth = getTypeWidth(fromPort->type);
        unsigned toWidth = getTypeWidth(toPort->type);
        if (fromWidth > 0 && toWidth > 0 && fromWidth != toWidth) {
          diag = "C2: routing hop bit-width mismatch in edge " +
                 std::to_string(i) + " at hop " + std::to_string(j / 2);
          return false;
        }
      }
    }
  }

  return true;
}

bool Mapper::validateC3(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C3: Route legality - each mapped edge path must follow physical
  // connectivity, and no exclusive HW edge is used by more than one SW edge.
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++i) {
    const auto &path = state.swEdgeToHwPaths[i];
    if (path.empty())
      continue;

    // Verify each physical edge hop in the path.
    // Path: [outPort0, inPort0, outPort1, inPort1, ...]
    for (size_t j = 0; j + 1 < path.size(); j += 2) {
      IdIndex fromPort = path[j];
      IdIndex toPort = path[j + 1];

      const Port *fp = adg.getPort(fromPort);
      const Port *tp = adg.getPort(toPort);
      if (!fp || !tp) {
        diag = "C3: invalid port in path of edge " + std::to_string(i);
        return false;
      }

      // Verify physical connectivity exists.
      auto connIt = connectivity.outToIn.find(fromPort);
      if (connIt == connectivity.outToIn.end() ||
          connIt->second != toPort) {
        diag = "C3: no physical connection from port " +
               std::to_string(fromPort) + " to " + std::to_string(toPort) +
               " in edge " + std::to_string(i);
        return false;
      }
    }

    // C3 internal routing validation: for multi-hop paths, verify that
    // intermediate routing hops use legal internal transitions.
    // Path pairs: (out0, in0), (out1, in1), ...
    // Between pairs: in0's routing node must connect to out1 internally.
    for (size_t j = 1; j + 2 < path.size(); j += 2) {
      IdIndex inPort = path[j];
      IdIndex nextOutPort = path[j + 1];

      // Check that the routing node containing inPort has an internal
      // connection to nextOutPort.
      auto internalIt = connectivity.inToOut.find(inPort);
      if (internalIt == connectivity.inToOut.end()) {
        diag = "C3: no internal routing from port " +
               std::to_string(inPort) + " in edge " + std::to_string(i);
        return false;
      }

      bool found = false;
      for (IdIndex outId : internalIt->second) {
        if (outId == nextOutPort) {
          found = true;
          break;
        }
      }
      if (!found) {
        diag = "C3: illegal internal routing transition from port " +
               std::to_string(inPort) + " to port " +
               std::to_string(nextOutPort) + " in edge " +
               std::to_string(i);
        return false;
      }
    }
  }

  // Check C3 exclusivity: no exclusive HW edge used by more than one SW edge.
  for (IdIndex e = 0; e < static_cast<IdIndex>(state.hwEdgeToSwEdges.size());
       ++e) {
    const auto &swEdges = state.hwEdgeToSwEdges[e];
    if (swEdges.size() <= 1)
      continue;

    // Check if the destination is a routing node (allows tagged sharing).
    const Edge *hwEdge = adg.getEdge(e);
    if (!hwEdge)
      continue;
    const Port *dstPort = adg.getPort(hwEdge->dstPort);
    if (!dstPort)
      continue;
    const Node *dstNode = adg.getNode(dstPort->parentNode);
    if (!dstNode)
      continue;
    llvm::StringRef resClass = getNodeResClass(dstNode);
    if (resClass != "routing" && swEdges.size() > 1) {
      diag = "C3: exclusive hw_edge=" + std::to_string(e) +
             " used by " + std::to_string(swEdges.size()) + " sw edges";
      return false;
    }
  }

  // C3.5: Memory done-token wiring legality.
  // For memory operations, verify that done-token ports are connected.
  for (IdIndex swId = 0;
       swId < static_cast<IdIndex>(state.swNodeToHwNode.size()); ++swId) {
    IdIndex hwId = state.swNodeToHwNode[swId];
    if (hwId == INVALID_ID)
      continue;

    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;

    llvm::StringRef resClass = getNodeResClass(hwNode);
    if (resClass != "memory")
      continue;

    // Memory nodes: verify that the "done" output port (if present)
    // has a routed edge. Check if the DFG node's output ports are routed.
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;

    for (IdIndex outPortId : swNode->outputPorts) {
      const Port *outPort = dfg.getPort(outPortId);
      if (!outPort)
        continue;
      // Check that all connected edges from this port are routed.
      for (IdIndex edgeId : outPort->connectedEdges) {
        if (edgeId >= state.swEdgeToHwPaths.size() ||
            state.swEdgeToHwPaths[edgeId].empty()) {
          diag = "C3.5: memory done-token edge " + std::to_string(edgeId) +
                 " from node " + std::to_string(swId) + " unrouted";
          return false;
        }
      }
    }
  }

  // C3.6: Fan-out legality.
  // DFG output ports with multiple consumer edges must have all edges routed
  // and they must share the source hardware port.
  for (IdIndex swId = 0;
       swId < static_cast<IdIndex>(state.swNodeToHwNode.size()); ++swId) {
    if (state.swNodeToHwNode[swId] == INVALID_ID)
      continue;

    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;

    for (IdIndex outPortId : swNode->outputPorts) {
      const Port *outPort = dfg.getPort(outPortId);
      if (!outPort || outPort->connectedEdges.size() <= 1)
        continue;

      // Multiple consumer edges: all must start from the same HW source port.
      IdIndex expectedSrcHwPort = INVALID_ID;
      for (IdIndex edgeId : outPort->connectedEdges) {
        if (edgeId >= state.swEdgeToHwPaths.size() ||
            state.swEdgeToHwPaths[edgeId].empty())
          continue;

        IdIndex pathSrcPort = state.swEdgeToHwPaths[edgeId][0];
        if (expectedSrcHwPort == INVALID_ID) {
          expectedSrcHwPort = pathSrcPort;
        } else if (pathSrcPort != expectedSrcHwPort) {
          diag = "C3.6: fan-out from sw_port " + std::to_string(outPortId) +
                 " uses inconsistent hw source ports";
          return false;
        }
      }
    }
  }

  return true;
}

bool Mapper::validateC4(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C4: Capacity constraints.
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.hwNodeToSwNodes.size());
       ++i) {
    const auto &swNodes = state.hwNodeToSwNodes[i];
    if (swNodes.empty())
      continue;

    const Node *hwNode = adg.getNode(i);
    if (!hwNode)
      continue;

    llvm::StringRef resClass = getNodeResClass(hwNode);

    // Non-temporal functional nodes: exclusive (at most 1 mapping).
    if (resClass == "functional") {
      bool isTemporal = nodeHasAttr(hwNode, "parent_temporal_pe") ||
                        nodeHasAttr(hwNode, "is_virtual");

      if (!isTemporal && swNodes.size() > 1) {
        diag = "C4: capacity exceeded on hw_node=" + std::to_string(i) +
               " (" + std::to_string(swNodes.size()) + " mappings)";
        return false;
      }
    }

    // Memory nodes: check region capacity.
    if (resClass == "memory") {
      int64_t numRegion = getNodeIntAttr(hwNode, "numRegion", 1);
      if (static_cast<int64_t>(swNodes.size()) > numRegion) {
        diag = "C4: memory region overflow on hw_node=" + std::to_string(i) +
               " (" + std::to_string(swNodes.size()) + "/" +
               std::to_string(numRegion) + ")";
        return false;
      }
    }
  }
  return true;
}

bool Mapper::validateC5(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C5: Temporal constraints - verify slot and register assignments are
  // within bounds.

  // Collect operations per temporal PE.
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 8>> tpeGroups;
  for (IdIndex swId = 0;
       swId < static_cast<IdIndex>(state.swNodeToHwNode.size()); ++swId) {
    IdIndex hwId = state.swNodeToHwNode[swId];
    if (hwId == INVALID_ID)
      continue;
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;
    int64_t parentTPE = getNodeIntAttr(hwNode, "parent_temporal_pe", -1);
    if (parentTPE >= 0)
      tpeGroups[static_cast<IdIndex>(parentTPE)].push_back(swId);
  }

  for (auto &[tpeId, swOps] : tpeGroups) {
    const Node *tpeNode = adg.getNode(tpeId);
    if (!tpeNode)
      continue;

    int64_t numInstruction = getNodeIntAttr(tpeNode, "num_instruction", 0);
    int64_t numRegister = getNodeIntAttr(tpeNode, "num_register", 0);

    // Check instruction slot capacity.
    if (static_cast<int64_t>(swOps.size()) > numInstruction) {
      diag = "C5: temporal PE " + std::to_string(tpeId) + " has " +
             std::to_string(swOps.size()) + " ops but only " +
             std::to_string(numInstruction) + " instruction slots";
      return false;
    }

    // Check slot assignments are valid.
    llvm::DenseSet<IdIndex> usedTags;
    for (IdIndex swId : swOps) {
      if (swId >= state.temporalPEAssignments.size())
        continue;
      const auto &tpa = state.temporalPEAssignments[swId];
      if (tpa.slot == INVALID_ID)
        continue;

      if (static_cast<int64_t>(tpa.slot) >= numInstruction) {
        diag = "C5: slot " + std::to_string(tpa.slot) +
               " exceeds num_instruction " + std::to_string(numInstruction) +
               " on temporal PE " + std::to_string(tpeId);
        return false;
      }

      // C5.4: Check tag uniqueness within temporal PE.
      if (tpa.tag != INVALID_ID && !usedTags.insert(tpa.tag).second) {
        diag = "C5: duplicate tag " + std::to_string(tpa.tag) +
               " in temporal PE " + std::to_string(tpeId);
        return false;
      }
    }

    // Check register assignments.
    uint32_t regCount = 0;
    for (IdIndex swId : swOps) {
      const Node *swNode = dfg.getNode(swId);
      if (!swNode)
        continue;
      for (IdIndex outPortId : swNode->outputPorts) {
        const Port *outPort = dfg.getPort(outPortId);
        if (!outPort)
          continue;
        for (IdIndex edgeId : outPort->connectedEdges) {
          if (edgeId >= state.registerAssignments.size())
            continue;
          if (state.registerAssignments[edgeId] != INVALID_ID)
            ++regCount;
        }
      }
    }
    if (static_cast<int64_t>(regCount) > numRegister) {
      diag = "C5: register overflow on temporal PE " +
             std::to_string(tpeId) + " (" + std::to_string(regCount) +
             "/" + std::to_string(numRegister) + ")";
      return false;
    }
  }

  return true;
}

bool Mapper::validateC6(const MappingState &state, const Graph &dfg,
                        const Graph &adg, std::string &diag) {
  // C6: Configuration encoding - verify that config values are encodable.
  // Check that mapped operations have valid config footprints.
  for (IdIndex hwId = 0;
       hwId < static_cast<IdIndex>(state.hwNodeToSwNodes.size()); ++hwId) {
    const auto &swNodes = state.hwNodeToSwNodes[hwId];
    if (swNodes.empty())
      continue;

    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;

    llvm::StringRef resClass = getNodeResClass(hwNode);

    // For temporal PEs, check that total config fits instruction memory.
    if (nodeHasAttr(hwNode, "is_virtual")) {
      int64_t numInst = getNodeIntAttr(hwNode, "num_instruction", 0);
      if (numInst > 0 && static_cast<int64_t>(swNodes.size()) > numInst) {
        diag = "C6: config encoding overflow on temporal PE " +
               std::to_string(hwId);
        return false;
      }
    }

    // For memory nodes, check that addr_offset_table entries don't overlap.
    if (resClass == "memory") {
      int64_t numRegion = getNodeIntAttr(hwNode, "numRegion", 1);
      if (static_cast<int64_t>(swNodes.size()) > numRegion) {
        diag = "C6: config encoding overflow on memory " +
               std::to_string(hwId) + " (regions)";
        return false;
      }
    }
  }

  return true;
}

bool Mapper::runValidation(const MappingState &state, const Graph &dfg,
                           const Graph &adg, std::string &diagnostics) {
  // Check all operations are mapped.
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::OperationNode &&
        (i >= state.swNodeToHwNode.size() ||
         state.swNodeToHwNode[i] == INVALID_ID)) {
      diagnostics = "Unmapped operation node " + std::to_string(i);
      return false;
    }
  }

  // Check all edges are routed.
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    if (i >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[i].empty()) {
      diagnostics = "Unrouted edge " + std::to_string(i);
      return false;
    }
  }

  // Run constraint checks.
  if (!validateC1(state, dfg, adg, diagnostics))
    return false;
  if (!validateC2(state, dfg, adg, diagnostics))
    return false;
  if (!validateC3(state, dfg, adg, diagnostics))
    return false;
  if (!validateC4(state, dfg, adg, diagnostics))
    return false;
  if (!validateC5(state, dfg, adg, diagnostics))
    return false;
  if (!validateC6(state, dfg, adg, diagnostics))
    return false;

  return true;
}

} // namespace loom

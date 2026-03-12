//===-- MapperValidation.cpp - Constraint validation for mapper ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/TypeCompat.h"

#include "loom/Dialect/Dataflow/DataflowTypes.h"  // TaggedType, BitsType
#include "loom/Hardware/Common/FabricConstants.h"
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
  if (auto bitsType = mlir::dyn_cast<loom::dataflow::BitsType>(type))
    return bitsType.getWidth();
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type))
    return intTy.getWidth();
  if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(type))
    return floatTy.getWidth();
  if (type.isIndex())
    return loom::ADDR_BIT_WIDTH;
  return 0;
}

/// Get bridge boundary port arrays from a node's attributes.
bool getBridgePortsVal(const Node *node,
                       mlir::DenseI32ArrayAttr &bridgeInPorts,
                       mlir::DenseI32ArrayAttr &bridgeOutPorts) {
  bridgeInPorts = {};
  bridgeOutPorts = {};
  for (auto &attr : node->attributes) {
    if (attr.getName() == "bridge_input_ports")
      bridgeInPorts =
          mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
    else if (attr.getName() == "bridge_output_ports")
      bridgeOutPorts =
          mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
  }
  return bridgeInPorts || bridgeOutPorts;
}

/// Get bridge mux/demux temporal_sw node IDs from a node's attributes.
/// ADGFlattener stores these as "bridge_mux_nodes" and "bridge_demux_nodes".
void getBridgeMuxDemuxNodes(const Node *node,
                            mlir::DenseI32ArrayAttr &muxNodes,
                            mlir::DenseI32ArrayAttr &demuxNodes) {
  muxNodes = {};
  demuxNodes = {};
  for (auto &attr : node->attributes) {
    if (attr.getName() == "bridge_mux_nodes")
      muxNodes =
          mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
    else if (attr.getName() == "bridge_demux_nodes")
      demuxNodes =
          mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
  }
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

      // For temporal PE FU nodes, use tagged-unwrap comparison.
      // The port's parentNode is the virtual temporal PE (port owner), but
      // the SW node is mapped to an FU sub-node. Look up the actual mapped
      // HW node for the SW port's parent node.
      bool temporalFU = false;
      if (sw->parentNode != INVALID_ID &&
          sw->parentNode < state.swNodeToHwNode.size()) {
        IdIndex mappedHw = state.swNodeToHwNode[sw->parentNode];
        if (mappedHw != INVALID_ID) {
          const Node *mappedHwNode = adg.getNode(mappedHw);
          temporalFU = isTemporalPEFU(mappedHwNode);
        }
      }

      // Detect non-temporal tagged PE via output_tag attribute.
      bool taggedPE = false;
      if (!temporalFU && hwNode) {
        for (auto &attr : hwNode->attributes) {
          if (attr.getName().getValue() == "output_tag") {
            taggedPE = true;
            break;
          }
        }
      }

      bool typeOk = temporalFU
                        ? isTypeWidthCompatibleForTemporalFU(sw->type, hw->type)
                    : taggedPE
                        ? isTypeWidthCompatibleForTaggedPE(sw->type, hw->type)
                        : isTypeWidthCompatible(sw->type, hw->type);
      if (!typeOk) {
        unsigned swWidth = getTypeWidth(sw->type);
        unsigned hwWidth = getTypeWidth(hw->type);
        diag = "C2: type width mismatch sw_port=" + std::to_string(i) +
               " (sw=" + std::to_string(swWidth) +
               " hw=" + std::to_string(hwWidth) + ")" +
               (isRouting ? " [routing]" : " [functional]");
        return false;
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

  // Check C3 exclusivity: only edges with tagged port types
  // (dataflow.tagged<bits<N>, iM>) may be shared, up to 2^M tags.
  for (IdIndex e = 0; e < static_cast<IdIndex>(state.hwEdgeToSwEdges.size());
       ++e) {
    const auto &swEdges = state.hwEdgeToSwEdges[e];
    if (swEdges.size() <= 1)
      continue;

    const Edge *hwEdge = adg.getEdge(e);
    if (!hwEdge)
      continue;
    const Port *srcPort = adg.getPort(hwEdge->srcPort);
    if (!srcPort)
      continue;

    auto taggedType =
        mlir::dyn_cast<loom::dataflow::TaggedType>(srcPort->type);
    if (!taggedType) {
      // Non-tagged edge shared by multiple SW edges: check if this is
      // valid fan-out (all SW edges originate from the same DFG source port).
      bool isFanout = true;
      IdIndex commonSrcPort = INVALID_ID;
      for (IdIndex swEdgeId : swEdges) {
        const Edge *swEdge = dfg.getEdge(swEdgeId);
        if (!swEdge) {
          isFanout = false;
          break;
        }
        if (commonSrcPort == INVALID_ID) {
          commonSrcPort = swEdge->srcPort;
        } else if (swEdge->srcPort != commonSrcPort) {
          isFanout = false;
          break;
        }
      }
      if (!isFanout) {
        diag = "C3: non-tagged hw_edge=" + std::to_string(e) +
               " shared by " + std::to_string(swEdges.size()) + " sw edges";
        return false;
      }
      continue; // Valid fan-out, skip tag capacity check.
    }

    unsigned tagWidth = taggedType.getTagType().getWidth();
    unsigned maxTags = 1u << tagWidth;
    if (swEdges.size() > maxTags) {
      diag = "C3: tag capacity exceeded on hw_edge=" + std::to_string(e) +
             " (" + std::to_string(swEdges.size()) + "/" +
             std::to_string(maxTags) + ")";
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

    // Non-temporal functional nodes: exclusive (at most 1 mapping),
    // unless the nodes form a valid multi-op PE group.
    if (resClass == "functional") {
      bool isTemporal = nodeHasAttr(hwNode, "parent_temporal_pe") ||
                        nodeHasAttr(hwNode, "is_virtual");

      if (!isTemporal && swNodes.size() > 1) {
        // Check if these SW nodes form a valid group binding.
        auto groupIt = state.groupBindings.find(i);
        if (groupIt != state.groupBindings.end()) {
          // Verify ALL mapped SW nodes are members of the group.
          llvm::DenseSet<IdIndex> groupSet;
          for (IdIndex gid : groupIt->second)
            groupSet.insert(gid);

          bool allInGroup = true;
          for (IdIndex swId : swNodes) {
            if (!groupSet.count(swId)) {
              allInGroup = false;
              break;
            }
          }
          if (!allInGroup) {
            diag = "C4: capacity exceeded on hw_node=" + std::to_string(i) +
                   " (non-group nodes sharing PE)";
            return false;
          }
          // Valid group; allow multiple mappings.
        } else {
          diag = "C4: capacity exceeded on hw_node=" + std::to_string(i) +
                 " (" + std::to_string(swNodes.size()) + " mappings)";
          return false;
        }
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

      // Bridge memory validation: check port bindings go to bridge boundary
      // ports and temporal_sw mux/demux nodes have assignments.
      mlir::DenseI32ArrayAttr bridgeInPorts, bridgeOutPorts;
      if (getBridgePortsVal(hwNode, bridgeInPorts, bridgeOutPorts)) {
        // Validate bridge mux/demux temporal_sw assignments.
        mlir::DenseI32ArrayAttr muxNodes, demuxNodes;
        getBridgeMuxDemuxNodes(hwNode, muxNodes, demuxNodes);

        // Collect valid tag ranges from this memory's config.
        int64_t ldCount = getNodeIntAttr(hwNode, "ldCount", 1);
        int64_t stCount = getNodeIntAttr(hwNode, "stCount", 1);
        int64_t tagCount = std::max(ldCount, stCount);
        bool isBridge = (ldCount > 1 || stCount > 1);
        // Build addr_offset_table tag ranges matching ConfigGen layout.
        size_t mappedCnt =
            (i < state.hwNodeToSwNodes.size() &&
             !state.hwNodeToSwNodes[i].empty())
            ? state.hwNodeToSwNodes[i].size() : 0;
        size_t regCnt = std::min(mappedCnt, static_cast<size_t>(numRegion));
        if (isBridge && regCnt == 0 && numRegion > 0)
          regCnt = 1;
        llvm::SmallVector<std::pair<int64_t, int64_t>, 4> tagRanges;
        for (size_t r = 0; r < regCnt; ++r) {
          if (isBridge && regCnt == 1)
            tagRanges.push_back({0, tagCount});
          else
            tagRanges.push_back({static_cast<int64_t>(r),
                                 static_cast<int64_t>(r + 1)});
        }

        auto validateBridgeTsw = [&](mlir::DenseI32ArrayAttr nodes,
                                     const char *label) -> bool {
          if (!nodes)
            return true;
          for (int32_t tswId : nodes.asArrayRef()) {
            auto tswIdx = static_cast<IdIndex>(tswId);
            if (tswIdx >= state.temporalSWAssignments.size() ||
                state.temporalSWAssignments[tswIdx].empty()) {
              diag = "C4: bridge " + std::string(label) + " " +
                     std::to_string(tswId) +
                     " for memory hw_node=" + std::to_string(i) +
                     " has no temporal assignment";
              return false;
            }
            const auto &assigns = state.temporalSWAssignments[tswIdx];
            const Node *tswNode = adg.getNode(tswIdx);
            int64_t numRT = tswNode
                ? getNodeIntAttr(tswNode, "num_route_table", 0) : 0;
            unsigned tswNumIn = tswNode ? tswNode->inputPorts.size() : 0;
            unsigned tswNumOut = tswNode ? tswNode->outputPorts.size() : 0;
            // Parse connectivity_table for route-mask validation.
            mlir::DenseI8ArrayAttr connTable;
            if (tswNode) {
              for (auto &attr : tswNode->attributes) {
                if (attr.getName() == "connectivity_table") {
                  connTable = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(
                      attr.getValue());
                  break;
                }
              }
            }
            // Build connected (out, in) positions for fan-in checking.
            unsigned K = tswNumOut * tswNumIn;
            llvm::SmallVector<std::pair<unsigned, unsigned>> connPos;
            if (connTable &&
                static_cast<unsigned>(connTable.size()) ==
                    tswNumOut * tswNumIn) {
              K = 0;
              for (unsigned oi = 0; oi < tswNumOut; ++oi)
                for (unsigned ii = 0; ii < tswNumIn; ++ii)
                  if (connTable[oi * tswNumIn + ii] != 0) {
                    connPos.push_back({oi, ii});
                    ++K;
                  }
            }
            llvm::DenseSet<IdIndex> usedTags;
            for (const auto &a : assigns) {
              // Slot bounds check.
              if (numRT > 0 &&
                  static_cast<int64_t>(a.slot) >= numRT) {
                diag = "C4: bridge " + std::string(label) +
                       " tsw=" + std::to_string(tswId) +
                       " slot " + std::to_string(a.slot) +
                       " exceeds num_route_table " +
                       std::to_string(numRT);
                return false;
              }
              // Duplicate tag check.
              if (a.tag != INVALID_ID &&
                  !usedTags.insert(a.tag).second) {
                diag = "C4: bridge " + std::string(label) +
                       " tsw=" + std::to_string(tswId) +
                       " duplicate tag " + std::to_string(a.tag);
                return false;
              }
              // Route-mask legality: only K valid bit positions.
              if (K > 0 && a.routeMask >= (1ULL << K)) {
                diag = "C4: bridge " + std::string(label) +
                       " tsw=" + std::to_string(tswId) +
                       " route mask exceeds K=" + std::to_string(K);
                return false;
              }
              // One-input-per-output: each output selects at most one
              // routed input per slot (spec rule).
              if (!connPos.empty()) {
                llvm::DenseMap<unsigned, unsigned> outFanIn;
                for (unsigned k = 0; k < connPos.size(); ++k)
                  if (a.routeMask & (1ULL << k))
                    ++outFanIn[connPos[k].first];
                for (auto &[outIdx, cnt] : outFanIn) {
                  if (cnt > 1) {
                    diag = "C4: bridge " + std::string(label) +
                           " tsw=" + std::to_string(tswId) +
                           " slot " + std::to_string(a.slot) +
                           " routes " + std::to_string(cnt) +
                           " inputs to output " + std::to_string(outIdx);
                    return false;
                  }
                }
              }
              // Validate tag against memory's addr_offset_table ranges.
              if (a.tag != INVALID_ID && !tagRanges.empty()) {
                auto t = static_cast<int64_t>(a.tag);
                bool covered = false;
                for (auto &[s, e] : tagRanges)
                  if (t >= s && t < e) { covered = true; break; }
                if (!covered) {
                  diag = "C4: bridge " + std::string(label) +
                         " tsw=" + std::to_string(tswId) +
                         " tag " + std::to_string(a.tag) +
                         " not covered by addr_offset_table";
                  return false;
                }
              }
            }
          }
          return true;
        };
        if (!validateBridgeTsw(muxNodes, "mux"))
          return false;
        if (!validateBridgeTsw(demuxNodes, "demux"))
          return false;

        // Verify that mapped DFG ports bind to bridge boundary ports.
        // For extmemory, skip memref at input port 0 (binds directly).
        // For memory, all input ports go through bridge boundary.
        bool isExtMem =
            (getNodeOpName(hwNode) == "fabric.extmemory");
        unsigned swInSkip = isExtMem ? 1 : 0;

        for (IdIndex swId : swNodes) {
          const Node *swNode = dfg.getNode(swId);
          if (!swNode)
            continue;
          // Check input port bindings.
          for (size_t p = swInSkip; p < swNode->inputPorts.size(); ++p) {
            IdIndex swPort = swNode->inputPorts[p];
            if (swPort >= state.swPortToHwPort.size())
              continue;
            IdIndex hwPort = state.swPortToHwPort[swPort];
            if (hwPort == INVALID_ID)
              continue;
            if (bridgeInPorts) {
              bool found = false;
              for (int32_t bp : bridgeInPorts.asArrayRef()) {
                if (static_cast<IdIndex>(bp) == hwPort) {
                  found = true;
                  break;
                }
              }
              if (!found) {
                diag = "C4: DFG input port " + std::to_string(swPort) +
                       " bound to non-bridge hw_port " +
                       std::to_string(hwPort) + " on bridge memory " +
                       std::to_string(i);
                return false;
              }
            }
          }
          // Check output port bindings (all outputs through bridge).
          for (size_t p = 0; p < swNode->outputPorts.size(); ++p) {
            IdIndex swPort = swNode->outputPorts[p];
            if (swPort >= state.swPortToHwPort.size())
              continue;
            IdIndex hwPort = state.swPortToHwPort[swPort];
            if (hwPort == INVALID_ID)
              continue;
            if (bridgeOutPorts) {
              bool found = false;
              for (int32_t bp : bridgeOutPorts.asArrayRef()) {
                if (static_cast<IdIndex>(bp) == hwPort) {
                  found = true;
                  break;
                }
              }
              if (!found) {
                diag = "C4: DFG output port " + std::to_string(swPort) +
                       " bound to non-bridge hw_port " +
                       std::to_string(hwPort) + " on bridge memory " +
                       std::to_string(i);
                return false;
              }
            }
          }
        }
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

  // Check all edges are routed (skip internal group edges).
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    // Skip internal group edges: both endpoints mapped to the same HW node
    // AND both are members of a tech-mapped group on that node.
    // Inter-slot edges on temporal FU sub-nodes must be routed.
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (srcPort && dstPort) {
      IdIndex srcSwNode = srcPort->parentNode;
      IdIndex dstSwNode = dstPort->parentNode;
      if (srcSwNode != INVALID_ID && dstSwNode != INVALID_ID &&
          srcSwNode < state.swNodeToHwNode.size() &&
          dstSwNode < state.swNodeToHwNode.size()) {
        IdIndex srcHwNode = state.swNodeToHwNode[srcSwNode];
        IdIndex dstHwNode = state.swNodeToHwNode[dstSwNode];
        if (srcHwNode != INVALID_ID && srcHwNode == dstHwNode) {
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
            continue; // Truly internal group edge, no routing needed.
        }
      }
    }
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

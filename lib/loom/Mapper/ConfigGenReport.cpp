//===-- ConfigGenReport.cpp - Mapping report output (JSON + text) --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/ConfigGenUtil.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace loom {

using namespace configgen;

// ===========================================================================
// writeMapJson
// ===========================================================================

bool ConfigGen::writeMapJson(const MappingState &state, const Graph &dfg,
                                 const Graph &adg, const std::string &path,
                                 const std::string &profile, int seed) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return false;

  llvm::json::OStream json(out, 2);
  json.objectBegin();

  json.attribute("version", 1);
  json.attribute("status", "success");
  json.attribute("profile", profile);
  json.attribute("seed", seed);

  // Placement.
  json.attributeBegin("placement");
  json.objectBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    IdIndex hwNode = state.swNodeToHwNode[i];
    if (hwNode == INVALID_ID)
      continue;
    const Node *sw = dfg.getNode(i);
    if (!sw || sw->kind != Node::OperationNode)
      continue;

    json.attributeBegin(std::to_string(i));
    json.objectBegin();
    json.attribute("hwNode", static_cast<int64_t>(hwNode));
    const Node *hw = adg.getNode(hwNode);
    if (hw) {
      json.attribute("hwNodeName", getNodeName(hw));
      json.attribute("swOp", getNodeOpName(sw));
    }
    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Port binding.
  json.attributeBegin("portBinding");
  json.objectBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swPortToHwPort.size());
       ++i) {
    IdIndex hwPort = state.swPortToHwPort[i];
    if (hwPort == INVALID_ID)
      continue;
    json.attribute(std::to_string(i), static_cast<int64_t>(hwPort));
  }
  json.objectEnd();
  json.attributeEnd();

  // Routes (with tag field per spec).
  json.attributeBegin("routes");
  json.objectBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++i) {
    const auto &pathVec = state.swEdgeToHwPaths[i];
    if (pathVec.empty())
      continue;

    json.attributeBegin(std::to_string(i));
    json.objectBegin();

    const Edge *edge = dfg.getEdge(i);
    if (edge) {
      json.attribute("srcSwPort", static_cast<int64_t>(edge->srcPort));
      json.attribute("dstSwPort", static_cast<int64_t>(edge->dstPort));
    }

    json.attributeBegin("hwPath");
    json.arrayBegin();
    for (size_t j = 0; j + 1 < pathVec.size(); j += 2) {
      json.objectBegin();
      json.attribute("src", static_cast<int64_t>(pathVec[j]));
      json.attribute("dst", static_cast<int64_t>(pathVec[j + 1]));
      json.objectEnd();
    }
    json.arrayEnd();
    json.attributeEnd();

    // Tag field: assigned tag for tagged edge sharing (null for exclusive).
    // Check if the first hop destination is a routing node with shared edges.
    bool hasTag = false;
    if (pathVec.size() >= 2) {
      const Port *dstPort = adg.getPort(pathVec[1]);
      if (dstPort) {
        const Node *dstNode = adg.getNode(dstPort->parentNode);
        if (dstNode && getNodeResClass(dstNode) == "routing") {
          // Check if multiple SW edges share this HW edge.
          for (IdIndex edgeId : adg.getPort(pathVec[0])->connectedEdges) {
            if (edgeId < state.hwEdgeToSwEdges.size() &&
                state.hwEdgeToSwEdges[edgeId].size() > 1) {
              hasTag = true;
              break;
            }
          }
        }
      }
    }

    if (hasTag) {
      // Use actual assigned per-edge tag from MappingState temporal
      // routing state. Fall back to source node's temporal PE tag.
      int64_t tagVal = 0;
      if (i < state.temporalSWAssignments.size() &&
          !state.temporalSWAssignments[i].empty()) {
        const auto &tswa = state.temporalSWAssignments[i];
        if (tswa[0].tag != INVALID_ID)
          tagVal = static_cast<int64_t>(tswa[0].tag);
      } else if (edge) {
        const Port *srcPort = dfg.getPort(edge->srcPort);
        if (srcPort && srcPort->parentNode != INVALID_ID &&
            srcPort->parentNode < state.temporalPEAssignments.size()) {
          const auto &tpa =
              state.temporalPEAssignments[srcPort->parentNode];
          if (tpa.tag != INVALID_ID)
            tagVal = static_cast<int64_t>(tpa.tag);
        }
      }
      json.attribute("tag", tagVal);
    } else {
      json.attributeBegin("tag");
      json.rawValue("null");
      json.attributeEnd();
    }

    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Temporal assignments.
  json.attributeBegin("temporal");
  json.objectBegin();
  for (IdIndex i = 0;
       i < static_cast<IdIndex>(state.temporalPEAssignments.size()); ++i) {
    const auto &tpa = state.temporalPEAssignments[i];
    if (tpa.slot == INVALID_ID)
      continue;

    json.attributeBegin(std::to_string(i));
    json.objectBegin();
    json.attribute("slot", static_cast<int64_t>(tpa.slot));
    json.attribute("tag", static_cast<int64_t>(tpa.tag));
    json.attribute("opcode", static_cast<int64_t>(tpa.opcode));

    // Include temporal PE virtual node ID.
    if (i < state.swNodeToHwNode.size()) {
      IdIndex hwId = state.swNodeToHwNode[i];
      const Node *hwNode = adg.getNode(hwId);
      if (hwNode) {
        int64_t parentTPE = getNodeIntAttr(hwNode, "parent_temporal_pe", -1);
        if (parentTPE >= 0)
          json.attribute("temporalPE", parentTPE);
      }
    }

    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Register assignments.
  json.attributeBegin("registers");
  json.objectBegin();
  for (IdIndex i = 0;
       i < static_cast<IdIndex>(state.registerAssignments.size()); ++i) {
    if (state.registerAssignments[i] == INVALID_ID)
      continue;

    json.attributeBegin(std::to_string(i));
    json.objectBegin();

    // temporalPE: the HW virtual node ID of the temporal PE where the
    // register is located (per spec-mapper-output.md).
    const Edge *regEdge = dfg.getEdge(i);
    if (regEdge) {
      const Port *srcPort = dfg.getPort(regEdge->srcPort);
      if (srcPort && srcPort->parentNode != INVALID_ID &&
          srcPort->parentNode < state.swNodeToHwNode.size()) {
        IdIndex hwId = state.swNodeToHwNode[srcPort->parentNode];
        const Node *hwNode = adg.getNode(hwId);
        if (hwNode) {
          int64_t parentTPE =
              getNodeIntAttr(hwNode, "parent_temporal_pe", -1);
          if (parentTPE >= 0)
            json.attribute("temporalPE", parentTPE);
        }
      }
    }

    json.attribute("registerIndex",
                   static_cast<int64_t>(state.registerAssignments[i]));
    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Cost.
  json.attributeBegin("cost");
  json.objectBegin();
  json.attribute("total", state.totalCost);
  json.attribute("placementPressure", state.placementPressure);
  json.attribute("routingCost", state.routingCost);
  json.attribute("temporalCost", state.temporalCost);
  json.attribute("perfProxy", state.perfProxyCost);
  json.attribute("configFootprint", state.configFootprint);
  json.objectEnd();
  json.attributeEnd();

  // Diagnostics section per spec-mapper-output.md.
  json.attributeBegin("diagnostics");
  json.objectBegin();

  // Unmapped nodes.
  json.attributeBegin("unmappedNodes");
  json.arrayBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node || node->kind != Node::OperationNode)
      continue;
    if (i >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[i] == INVALID_ID) {
      json.value(std::to_string(i));
    }
  }
  json.arrayEnd();
  json.attributeEnd();

  // Failed edges.
  json.attributeBegin("failedEdges");
  json.arrayBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    if (i >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[i].empty()) {
      json.value(std::to_string(i));
    }
  }
  json.arrayEnd();
  json.attributeEnd();

  json.attributeBegin("firstViolatedConstraint");
  json.rawValue("null");
  json.attributeEnd();

  json.attributeBegin("conflictingResources");
  json.arrayBegin();
  json.arrayEnd();
  json.attributeEnd();

  json.objectEnd();
  json.attributeEnd();

  json.objectEnd();
  return true;
}

// ===========================================================================
// writeMapText helpers
// ===========================================================================

namespace {

/// Get a display name for a DFG (software) node.
std::string swNodeDisplayName(const Node *node, IdIndex id) {
  if (node->kind == Node::ModuleInputNode) {
    int64_t idx = getNodeIntAttr(node, "arg_index", -1);
    return "ModuleInput(arg" + std::to_string(idx >= 0 ? idx : id) + ")";
  }
  if (node->kind == Node::ModuleOutputNode) {
    int64_t idx = getNodeIntAttr(node, "ret_index", -1);
    return "ModuleOutput(ret" + std::to_string(idx >= 0 ? idx : id) + ")";
  }
  llvm::StringRef opName = getNodeOpName(node);
  return opName.empty() ? ("node_" + std::to_string(id)) : opName.str();
}

/// Get a display name for an ADG (hardware) node.
std::string hwNodeDisplayName(const Node *node, IdIndex id) {
  if (node->kind == Node::ModuleInputNode) {
    int64_t idx = getNodeIntAttr(node, "arg_index", -1);
    return "in" + std::to_string(idx >= 0 ? idx : id);
  }
  if (node->kind == Node::ModuleOutputNode) {
    int64_t idx = getNodeIntAttr(node, "ret_index", -1);
    return "out" + std::to_string(idx >= 0 ? idx : id);
  }
  llvm::StringRef symName = getNodeName(node);
  if (!symName.empty())
    return symName.str();
  return "node_" + std::to_string(id);
}

/// Get a description string for an ADG node (op_name, resource_class).
std::string hwNodeDesc(const Node *node) {
  if (node->kind == Node::ModuleInputNode)
    return "ModuleInput";
  if (node->kind == Node::ModuleOutputNode)
    return "ModuleOutput";
  llvm::StringRef opName = getNodeOpName(node);
  llvm::StringRef resClass = getNodeResClass(node);
  std::string desc;
  if (!opName.empty())
    desc = opName.str();
  if (!resClass.empty()) {
    if (!desc.empty())
      desc += ", ";
    desc += resClass.str();
  }
  return desc;
}

/// Get the "loc" string attribute from a node, or empty string.
llvm::StringRef getNodeLocStr(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "loc") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Format a port reference as "nodeName:in0" or "nodeName:out0".
std::string formatPortRef(const Graph &graph, IdIndex portId,
                          bool isHwGraph) {
  const Port *port = graph.getPort(portId);
  if (!port)
    return "port_" + std::to_string(portId);

  const Node *node = graph.getNode(port->parentNode);
  if (!node)
    return "port_" + std::to_string(portId);

  std::string nodeName = isHwGraph
      ? hwNodeDisplayName(node, port->parentNode)
      : swNodeDisplayName(node, port->parentNode);

  // Find port index within the node's input or output port list.
  const auto &portList = (port->direction == Port::Input)
      ? node->inputPorts : node->outputPorts;
  unsigned portIdx = 0;
  for (unsigned i = 0; i < portList.size(); ++i) {
    if (portList[i] == portId) {
      portIdx = i;
      break;
    }
  }

  std::string dir = (port->direction == Port::Input) ? "in" : "out";
  return nodeName + ":" + dir + std::to_string(portIdx);
}

/// Check if a DFG edge is internal to an operation group (both endpoints
/// mapped to the same HW node AND both are members of a tech-mapped group).
/// Such edges are handled inside the PE and do not need switch-fabric
/// routing. Inter-slot edges on temporal FU sub-nodes are NOT group-internal.
bool isInternalGroupEdge(const Edge *edge, const Graph &dfg,
                         const MappingState &state) {
  const Port *srcPort = dfg.getPort(edge->srcPort);
  const Port *dstPort = dfg.getPort(edge->dstPort);
  if (!srcPort || !dstPort)
    return false;
  IdIndex srcSwNode = srcPort->parentNode;
  IdIndex dstSwNode = dstPort->parentNode;
  if (srcSwNode == INVALID_ID || dstSwNode == INVALID_ID)
    return false;
  if (srcSwNode >= state.swNodeToHwNode.size() ||
      dstSwNode >= state.swNodeToHwNode.size())
    return false;
  IdIndex srcHwNode = state.swNodeToHwNode[srcSwNode];
  IdIndex dstHwNode = state.swNodeToHwNode[dstSwNode];
  if (srcHwNode == INVALID_ID || srcHwNode != dstHwNode)
    return false;
  // Both mapped to same HW node: only truly internal if both are in
  // the same group binding (tech-mapped group).
  auto groupIt = state.groupBindings.find(srcHwNode);
  if (groupIt == state.groupBindings.end())
    return false;
  bool srcInGroup = false, dstInGroup = false;
  for (IdIndex gid : groupIt->second) {
    if (gid == srcSwNode) srcInGroup = true;
    if (gid == dstSwNode) dstInGroup = true;
  }
  return srcInGroup && dstInGroup;
}

} // namespace

// ===========================================================================
// writeMapText
// ===========================================================================

bool ConfigGen::writeMapText(const MappingState &state, const Graph &dfg,
                              const Graph &adg, const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return false;

  // --- Header ---
  out << "=== Mapping Report ===\n\n";

  // --- SW Node List ---
  unsigned swNodeCount = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    if (dfg.getNode(i))
      ++swNodeCount;
  }
  out << "--- SW Node List (" << swNodeCount << " nodes) ---\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;
    out << "  [N" << i << "] " << swNodeDisplayName(node, i);
    llvm::StringRef loc = getNodeLocStr(node);
    if (!loc.empty())
      out << "  " << loc;
    out << "\n";
  }
  out << "\n";

  // --- SW Edge List ---
  unsigned swEdgeCount = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    if (dfg.getEdge(i))
      ++swEdgeCount;
  }
  out << "--- SW Edge List (" << swEdgeCount << " edges) ---\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    out << "  [E" << i << "] "
        << formatPortRef(dfg, edge->srcPort, false)
        << " -> "
        << formatPortRef(dfg, edge->dstPort, false);
    // Use destination node's location for the edge.
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (dstPort) {
      const Node *dstNode = dfg.getNode(dstPort->parentNode);
      if (dstNode) {
        llvm::StringRef loc = getNodeLocStr(dstNode);
        if (!loc.empty())
          out << "  " << loc;
      }
    }
    out << "\n";
  }
  out << "\n";

  // --- HW Node List ---
  unsigned hwNodeCount = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    if (adg.getNode(i))
      ++hwNodeCount;
  }
  out << "--- HW Node List (" << hwNodeCount << " nodes) ---\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;
    out << "  [H" << i << "] " << hwNodeDisplayName(node, i);
    std::string desc = hwNodeDesc(node);
    if (!desc.empty())
      out << " (" << desc << ")";
    llvm::StringRef loc = getNodeLocStr(node);
    if (!loc.empty())
      out << "  " << loc;
    out << "\n";
  }
  out << "\n";

  // --- Node Mapping ---
  unsigned mappedNodeCount = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    if (state.swNodeToHwNode[i] != INVALID_ID && dfg.getNode(i))
      ++mappedNodeCount;
  }
  out << "--- Node Mapping (" << mappedNodeCount << " mapped) ---\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    IdIndex hwId = state.swNodeToHwNode[i];
    const Node *swNode = dfg.getNode(i);
    if (!swNode)
      continue;
    if (hwId == INVALID_ID) {
      out << "  [N" << i << "] " << swNodeDisplayName(swNode, i)
          << " -> UNMAPPED\n";
      continue;
    }
    const Node *hwNode = adg.getNode(hwId);
    out << "  [N" << i << "] " << swNodeDisplayName(swNode, i)
        << " -> [H" << hwId << "] "
        << (hwNode ? hwNodeDisplayName(hwNode, hwId) : "???") << "\n";
  }
  out << "\n";

  // --- Edge Routing ---
  unsigned routedEdgeCount = 0;
  unsigned internalEdgeCount = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    if (isInternalGroupEdge(edge, dfg, state))
      ++internalEdgeCount;
    else if (i < state.swEdgeToHwPaths.size() &&
             !state.swEdgeToHwPaths[i].empty())
      ++routedEdgeCount;
  }
  out << "--- Edge Routing (" << routedEdgeCount << " routed";
  if (internalEdgeCount > 0)
    out << ", " << internalEdgeCount << " internal";
  out << ") ---\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;

    out << "  [E" << i << "] "
        << formatPortRef(dfg, edge->srcPort, false)
        << " -> "
        << formatPortRef(dfg, edge->dstPort, false) << "\n";

    // Internal group edges are handled inside the PE, no routing needed.
    if (isInternalGroupEdge(edge, dfg, state)) {
      out << "    Route: INTERNAL (within PE group)\n\n";
      continue;
    }

    if (i >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[i].empty()) {
      out << "    Route: UNROUTED\n\n";
      continue;
    }

    const auto &hwPath = state.swEdgeToHwPaths[i];
    out << "    Route: ";
    for (size_t j = 0; j < hwPath.size(); ++j) {
      if (j > 0)
        out << " -> ";
      out << formatPortRef(adg, hwPath[j], true);
    }
    out << "\n";
    out << "    Hops: " << (hwPath.size() / 2) << "\n";
    out << "\n";
  }

  // --- Unmapped ---
  out << "--- Unmapped ---\n";
  bool hasUnmapped = false;

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;
    if (i >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[i] == INVALID_ID) {
      if (!hasUnmapped) {
        out << "  Nodes:\n";
        hasUnmapped = true;
      }
      out << "    [N" << i << "] " << swNodeDisplayName(node, i) << "\n";
    }
  }

  bool hasUnroutedEdges = false;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    // Skip internal group edges (handled inside PE, not a routing failure).
    if (isInternalGroupEdge(edge, dfg, state))
      continue;
    if (i >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[i].empty()) {
      if (!hasUnroutedEdges) {
        out << "  Edges:\n";
        hasUnroutedEdges = true;
        hasUnmapped = true;
      }
      out << "    [E" << i << "] "
          << formatPortRef(dfg, edge->srcPort, false) << " -> "
          << formatPortRef(dfg, edge->dstPort, false) << "\n";
    }
  }

  if (!hasUnmapped)
    out << "  (none)\n";
  out << "\n";

  // --- Cost ---
  out << "--- Cost ---\n";
  out << llvm::format("  total           = %.4f\n", state.totalCost);
  out << llvm::format("  placement       = %.4f\n", state.placementPressure);
  out << llvm::format("  routing         = %.4f\n", state.routingCost);
  out << llvm::format("  temporal        = %.4f\n", state.temporalCost);
  out << llvm::format("  perfProxy       = %.4f\n", state.perfProxyCost);
  out << llvm::format("  configFootprint = %.4f\n", state.configFootprint);

  return true;
}

} // namespace loom

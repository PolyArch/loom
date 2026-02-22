//===-- DOTExporterMapped.cpp - Mapped visualization DOT export ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Exports mapped visualization (DFG placed on ADG) as DOT strings for both
// overlay and side-by-side modes, per docs/spec-viz-mapped.md.
//
//===----------------------------------------------------------------------===//

#include "loom/Viz/DOTExporter.h"

#include "VizUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"
#include "loom/Mapper/Types.h"

#include "mlir/IR/Attributes.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace loom {
namespace viz {

namespace {

// Build overlay label for a mapped HW node.
std::string buildOverlayMappedLabel(const Node *hwNode, IdIndex hwId,
                                    const Graph &dfg,
                                    const MappingState &state) {
  std::string hwName = getNodeStrAttr(hwNode, "sym_name");
  std::string opName = getNodeOpName(hwNode);

  std::ostringstream label;

  // HW name.
  if (!hwName.empty()) {
    label << dotEscape(hwName);
  } else {
    std::string display = opName;
    if (display.substr(0, 7) == "fabric.")
      display = display.substr(7);
    label << dotEscape(display) << "_" << hwId;
  }

  // Check what SW operations are mapped to this HW node.
  if (hwId < state.hwNodeToSwNodes.size()) {
    const auto &swNodes = state.hwNodeToSwNodes[hwId];
    bool isTemporal = (opName == "fabric.temporal_pe");

    for (IdIndex swId : swNodes) {
      if (swId == INVALID_ID)
        continue;
      const Node *swNode = dfg.getNode(swId);
      if (!swNode)
        continue;

      std::string swOpName = getNodeOpName(swNode);
      if (isTemporal && swId < state.temporalPEAssignments.size()) {
        const auto &tpa = state.temporalPEAssignments[swId];
        label << "\\n<- " << dotEscape(swOpName) << " [slot "
              << tpa.slot << ", tag=" << tpa.tag << "]";
      } else {
        label << "\\n<- " << dotEscape(swOpName);
      }
    }

    // Register usage for temporal PEs.
    if (isTemporal && !swNodes.empty()) {
      int64_t numReg = getNodeIntAttr(hwNode, "num_register", -1);
      if (numReg >= 0) {
        // Count used registers across all mapped SW nodes.
        size_t usedRegs = 0;
        for (IdIndex swId : swNodes) {
          if (swId != INVALID_ID &&
              swId < state.registerAssignments.size() &&
              state.registerAssignments[swId] != INVALID_ID) {
            ++usedRegs;
          }
        }
        label << "\\nReg: " << usedRegs << "/" << numReg;
      }
    }
  }

  return label.str();
}

// Get the primary dialect color for a mapped HW node based on its SW ops.
const char *getMappedNodeColor(IdIndex hwId, const Graph &dfg,
                               const MappingState &state) {
  if (hwId >= state.hwNodeToSwNodes.size())
    return "white";

  const auto &swNodes = state.hwNodeToSwNodes[hwId];
  if (swNodes.empty())
    return "white";

  // Use the first valid mapped SW node's dialect color.
  for (IdIndex swId : swNodes) {
    if (swId == INVALID_ID)
      continue;
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;
    return getMappedDialectColor(getNodeOpName(swNode));
  }
  return "white";
}

// Check if an HW node has any mapped SW operations.
bool isMapped(IdIndex hwId, const MappingState &state) {
  if (hwId >= state.hwNodeToSwNodes.size())
    return false;
  for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
    if (swId != INVALID_ID)
      return true;
  }
  return false;
}

// Collect mapped edge route segments for color overlay.
struct RouteSegment {
  IdIndex srcHwNode;
  IdIndex dstHwNode;
  size_t colorIndex;
};

std::vector<RouteSegment>
collectRouteSegments(const Graph &dfg, const Graph &adg,
                     const MappingState &state) {
  std::vector<RouteSegment> segments;

  // Collect SW edges sorted by source node for deterministic color assignment.
  struct SwEdgeInfo {
    IdIndex swEdgeId;
    IdIndex srcNodeId;
  };
  std::vector<SwEdgeInfo> swEdges;

  IdIndex eidx = 0;
  for (const auto &edgePtr : dfg.edges) {
    IdIndex eid = eidx++;
    if (!edgePtr)
      continue;
    const Port *sp = dfg.getPort(edgePtr->srcPort);
    if (sp)
      swEdges.push_back({eid, sp->parentNode});
  }

  std::sort(swEdges.begin(), swEdges.end(),
            [](const SwEdgeInfo &a, const SwEdgeInfo &b) {
              return a.srcNodeId < b.srcNodeId;
            });

  // Assign colors and collect route segments.
  size_t colorIdx = 0;
  for (const auto &info : swEdges) {
    if (info.swEdgeId >= state.swEdgeToHwPaths.size())
      continue;
    const auto &hwPath = state.swEdgeToHwPaths[info.swEdgeId];
    if (hwPath.empty())
      continue;

    // hwPath is a port-sequence: [outPort0, inPort0, outPort1, inPort1, ...]
    // Each pair (outPort, inPort) represents a physical link hop.
    for (size_t pi = 0; pi + 1 < hwPath.size(); pi += 2) {
      IdIndex outPortId = hwPath[pi];
      IdIndex inPortId = hwPath[pi + 1];
      const Port *sp = adg.getPort(outPortId);
      const Port *dp = adg.getPort(inPortId);
      if (!sp || !dp)
        continue;
      segments.push_back({sp->parentNode, dp->parentNode, colorIdx});
    }
    ++colorIdx;
  }

  return segments;
}

} // namespace

std::string exportMappedOverlayDot(const Graph &dfg, const Graph &adg,
                                   const MappingState &state,
                                   const DOTOptions &opts) {
  std::ostringstream dot;

  dot << "digraph MappedOverlay {\n";
  dot << "  rankdir=LR;\n";
  dot << "  node [style=filled, fontname=\"Helvetica\"];\n";
  dot << "  edge [fontname=\"Helvetica\"];\n";
  dot << "\n";

  // Emit HW nodes with mapping coloring.
  IdIndex nodeIdx = 0;
  for (const auto &nodePtr : adg.nodes) {
    IdIndex nid = nodeIdx++;
    if (!nodePtr)
      continue;

    const Node *node = nodePtr.get();
    std::string opName = getNodeOpName(node);
    ADGNodeStyle baseStyle = getADGNodeStyle(opName, node->kind);

    bool mapped = isMapped(nid, state);
    const char *fillColor =
        mapped ? getMappedNodeColor(nid, dfg, state) : "white";
    const char *borderStyle = mapped ? "filled" : "filled,dashed";

    std::string label =
        mapped ? buildOverlayMappedLabel(node, nid, dfg, state)
               : (getNodeStrAttr(node, "sym_name").empty()
                      ? std::string(baseStyle.fillColor)
                      : getNodeStrAttr(node, "sym_name"));

    // For unmapped nodes, use the structure label.
    if (!mapped) {
      std::string symName = getNodeStrAttr(node, "sym_name");
      std::string display = opName;
      if (display.substr(0, 7) == "fabric.")
        display = display.substr(7);
      label = dotEscape(display);
      if (!symName.empty())
        label += "\\n" + dotEscape(symName);
    }

    dot << "  hw_" << nid << " [";
    dot << "id=\"hw_" << nid << "\", ";
    dot << "label=\"" << label << "\", ";
    dot << "shape=" << baseStyle.shape << ", ";
    dot << "fillcolor=\"" << fillColor << "\", ";
    dot << "fontcolor=\"" << baseStyle.fontColor << "\", ";
    dot << "style=\"" << borderStyle << "\"";

    if (opName == "fabric.temporal_pe")
      dot << ", width=1.5, height=1.2";

    dot << "];\n";
  }

  dot << "\n";

  // Emit base ADG edges (thin, gray).
  IdIndex edgeIdx = 0;
  for (const auto &edgePtr : adg.edges) {
    IdIndex eid = edgeIdx++;
    if (!edgePtr)
      continue;

    const Edge *edge = edgePtr.get();
    const Port *srcPort = adg.getPort(edge->srcPort);
    const Port *dstPort = adg.getPort(edge->dstPort);
    if (!srcPort || !dstPort)
      continue;

    IdIndex srcNode = srcPort->parentNode;
    IdIndex dstNode = dstPort->parentNode;
    if (!adg.getNode(srcNode) || !adg.getNode(dstNode))
      continue;

    dot << "  hw_" << srcNode << " -> hw_" << dstNode << " [";
    dot << "style=solid, color=\"#cccccc\", penwidth=1.0";
    dot << "];\n";
  }

  // Emit routing path overlays as colored edges.
  auto segments = collectRouteSegments(dfg, adg, state);
  for (const auto &seg : segments) {
    dot << "  hw_" << seg.srcHwNode << " -> hw_" << seg.dstHwNode << " [";
    dot << "color=\"" << routeColor(seg.colorIndex) << "\", ";
    dot << "penwidth=3.0, ";
    dot << "style=bold";
    dot << "];\n";
  }

  dot << "}\n";
  return dot.str();
}

std::string exportMappedDFGDot(const Graph &dfg, const MappingState &state,
                               const DOTOptions &opts) {
  std::ostringstream dot;

  dot << "digraph MappedDFG {\n";
  dot << "  rankdir=TB;\n";
  dot << "  node [style=filled, fontname=\"Helvetica\"];\n";
  dot << "  edge [fontname=\"Helvetica\"];\n";
  dot << "\n";

  // Emit DFG nodes with cross-link IDs.
  IdIndex nodeIdx = 0;
  for (const auto &nodePtr : dfg.nodes) {
    IdIndex nid = nodeIdx++;
    if (!nodePtr)
      continue;

    const Node *node = nodePtr.get();
    DFGNodeStyle style = getDFGNodeStyle(getNodeOpName(node), node->kind);

    std::string opName = getNodeOpName(node);
    std::string label = dotEscape(opName);
    if (node->kind == Node::ModuleInputNode)
      label = "In";
    else if (node->kind == Node::ModuleOutputNode)
      label = "Out";

    dot << "  sw_" << nid << " [";
    dot << "id=\"sw_" << nid << "\", ";
    dot << "label=\"" << label << "\", ";
    dot << "shape=" << style.shape << ", ";
    dot << "fillcolor=\"" << style.fillColor << "\", ";
    dot << "fontcolor=\"" << style.fontColor << "\"";
    dot << "];\n";
  }

  dot << "\n";

  // Emit edges.
  IdIndex edgeIdx = 0;
  for (const auto &edgePtr : dfg.edges) {
    IdIndex eid = edgeIdx++;
    if (!edgePtr)
      continue;

    const Edge *edge = edgePtr.get();
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort)
      continue;

    IdIndex srcNode = srcPort->parentNode;
    IdIndex dstNode = dstPort->parentNode;
    if (!dfg.getNode(srcNode) || !dfg.getNode(dstNode))
      continue;

    dot << "  sw_" << srcNode << " -> sw_" << dstNode << " [";
    dot << "style=solid, color=black, penwidth=2.0";
    dot << "];\n";
  }

  dot << "}\n";
  return dot.str();
}

std::string exportMappedADGDot(const Graph &dfg, const Graph &adg,
                               const MappingState &state,
                               const DOTOptions &opts) {
  std::ostringstream dot;

  dot << "digraph MappedADG {\n";
  dot << "  rankdir=LR;\n";
  dot << "  node [style=filled, fontname=\"Helvetica\"];\n";
  dot << "  edge [fontname=\"Helvetica\"];\n";
  dot << "\n";

  // Emit ADG nodes with mapping annotations.
  IdIndex nodeIdx = 0;
  for (const auto &nodePtr : adg.nodes) {
    IdIndex nid = nodeIdx++;
    if (!nodePtr)
      continue;

    const Node *node = nodePtr.get();
    std::string opName = getNodeOpName(node);
    ADGNodeStyle baseStyle = getADGNodeStyle(opName, node->kind);

    bool mapped = isMapped(nid, state);
    const char *fillColor =
        mapped ? getMappedNodeColor(nid, dfg, state) : "white";
    const char *borderStyle = mapped ? "filled" : "filled,dashed";

    // Build label.
    std::ostringstream label;
    std::string symName = getNodeStrAttr(node, "sym_name");
    std::string display = opName;
    if (display.substr(0, 7) == "fabric.")
      display = display.substr(7);
    label << dotEscape(display);
    if (!symName.empty())
      label << "\\n" << dotEscape(symName);

    // Add mapped SW op names.
    if (mapped && nid < state.hwNodeToSwNodes.size()) {
      for (IdIndex swId : state.hwNodeToSwNodes[nid]) {
        if (swId == INVALID_ID)
          continue;
        const Node *swNode = dfg.getNode(swId);
        if (!swNode)
          continue;
        label << "\\n<- " << dotEscape(getNodeOpName(swNode));
      }
    }

    dot << "  hw_" << nid << " [";
    dot << "id=\"hw_" << nid << "\", ";
    dot << "label=\"" << label.str() << "\", ";
    dot << "shape=" << baseStyle.shape << ", ";
    dot << "fillcolor=\"" << fillColor << "\", ";
    dot << "fontcolor=\"" << baseStyle.fontColor << "\", ";
    dot << "style=\"" << borderStyle << "\"";

    if (opName == "fabric.temporal_pe")
      dot << ", width=1.5, height=1.2";

    dot << "];\n";
  }

  dot << "\n";

  // Emit edges.
  IdIndex edgeIdx = 0;
  for (const auto &edgePtr : adg.edges) {
    IdIndex eid = edgeIdx++;
    if (!edgePtr)
      continue;

    const Edge *edge = edgePtr.get();
    const Port *srcPort = adg.getPort(edge->srcPort);
    const Port *dstPort = adg.getPort(edge->dstPort);
    if (!srcPort || !dstPort)
      continue;

    IdIndex srcNode = srcPort->parentNode;
    IdIndex dstNode = dstPort->parentNode;
    if (!adg.getNode(srcNode) || !adg.getNode(dstNode))
      continue;

    dot << "  hw_" << srcNode << " -> hw_" << dstNode << " [";
    dot << "style=solid, color=\"#cccccc\", penwidth=1.0";
    dot << "];\n";
  }

  dot << "}\n";
  return dot.str();
}

} // namespace viz
} // namespace loom

//===-- DOTExporterDFG.cpp - DFG to DOT export --------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Exports a software dataflow graph (DFG) as a DOT string, following the
// conventions in docs/spec-viz-dfg.md.
//
//===----------------------------------------------------------------------===//

#include "loom/Viz/DOTExporter.h"

#include "VizUtil.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Types.h"

#include "mlir/IR/Attributes.h"

#include <sstream>
#include <string>

namespace loom {
namespace viz {

namespace {

// Build the label for a DFG node.
std::string buildDFGNodeLabel(const Node *node, IdIndex nodeId,
                              const DOTOptions &opts) {
  std::string opName = getNodeOpName(node);
  std::string typeSummary = getNodeStrAttr(node, "type_summary");
  std::string srcLoc = getNodeStrAttr(node, "src_loc");
  std::string symName = getNodeStrAttr(node, "sym_name");

  std::ostringstream label;

  if (node->kind == Node::ModuleInputNode) {
    label << "In";
    if (!symName.empty())
      label << " " << dotEscape(symName);
    return label.str();
  }
  if (node->kind == Node::ModuleOutputNode) {
    label << "Out";
    if (!symName.empty())
      label << " " << dotEscape(symName);
    return label.str();
  }

  // Operation name.
  label << dotEscape(opName);

  // Operation-specific additions.
  std::string constVal = getNodeStrAttr(node, "constant_value");
  if (!constVal.empty())
    label << "\\n= " << dotEscape(constVal);

  std::string predicate = getNodeStrAttr(node, "predicate");
  if (!predicate.empty())
    label << "\\n[" << dotEscape(predicate) << "]";

  std::string stepOp = getNodeStrAttr(node, "step_op");
  if (!stepOp.empty())
    label << "\\nstep: " << dotEscape(stepOp);

  std::string ldCount = getNodeStrAttr(node, "ldCount");
  std::string stCount = getNodeStrAttr(node, "stCount");
  if (!ldCount.empty() || !stCount.empty()) {
    label << "\\nld:" << (ldCount.empty() ? "0" : ldCount)
          << " st:" << (stCount.empty() ? "0" : stCount);
  }

  // Type summary.
  if (!typeSummary.empty())
    label << "\\n" << dotEscape(typeSummary);

  // Source location.
  if (!srcLoc.empty())
    label << "\\n" << dotEscape(srcLoc);

  return label.str();
}

// Determine edge style: data (solid, black) vs control (dashed, gray).
struct DFGEdgeStyle {
  const char *style;
  const char *color;
  const char *penwidth;
};

DFGEdgeStyle getDFGEdgeStyle(const Edge *edge, const Graph &dfg) {
  // Check if the edge's source port has a "none" type (control token).
  IdIndex srcPortId = edge->srcPort;
  const Port *srcPort = dfg.getPort(srcPortId);
  if (srcPort) {
    std::string portType;
    for (const auto &attr : srcPort->attributes) {
      if (attr.getName().str() == "port_type") {
        if (auto strAttr = llvm::dyn_cast<mlir::StringAttr>(attr.getValue()))
          portType = strAttr.getValue().str();
      }
    }
    if (portType == "none" || portType == "control")
      return {"dashed", "gray", "1.0"};
  }

  // Check edge attributes for control type.
  for (const auto &attr : edge->attributes) {
    if (attr.getName().str() == "edge_type") {
      if (auto strAttr = llvm::dyn_cast<mlir::StringAttr>(attr.getValue())) {
        if (strAttr.getValue() == "control")
          return {"dashed", "gray", "1.0"};
      }
    }
  }

  return {"solid", "black", "2.0"};
}

} // namespace

std::string exportDFGDot(const Graph &dfg, const DOTOptions &opts) {
  std::ostringstream dot;

  dot << "digraph DFG {\n";
  dot << "  rankdir=TB;\n";
  dot << "  node [style=filled, fontname=\"Helvetica\"];\n";
  dot << "  edge [fontname=\"Helvetica\"];\n";
  dot << "\n";

  // Emit nodes.
  IdIndex nodeIdx = 0;
  for (const auto &nodePtr : dfg.nodes) {
    IdIndex nid = nodeIdx++;
    if (!nodePtr)
      continue;

    const Node *node = nodePtr.get();
    DFGNodeStyle style = getDFGNodeStyle(getNodeOpName(node), node->kind);
    std::string label = buildDFGNodeLabel(node, nid, opts);

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

    DFGEdgeStyle edgeStyle = getDFGEdgeStyle(edge, dfg);

    dot << "  sw_" << srcNode << " -> sw_" << dstNode << " [";
    dot << "style=" << edgeStyle.style << ", ";
    dot << "color=\"" << edgeStyle.color << "\", ";
    dot << "penwidth=" << edgeStyle.penwidth;

    // Add edge label for disambiguation if needed.
    std::string edgeLabel = getNodeStrAttr(
        dfg.getNode(srcNode), "edge_label_" + std::to_string(eid));
    if (!edgeLabel.empty())
      dot << ", label=\"" << dotEscape(edgeLabel) << "\"";

    dot << "];\n";
  }

  dot << "}\n";
  return dot.str();
}

} // namespace viz
} // namespace loom

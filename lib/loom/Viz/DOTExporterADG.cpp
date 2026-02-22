//===-- DOTExporterADG.cpp - ADG to DOT export --------------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Exports a hardware architecture description graph (ADG) as a DOT string,
// following the conventions in docs/spec-viz-adg.md.
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

// Build the label for an ADG node in Structure mode.
std::string buildADGStructureLabel(const Node *node, IdIndex nodeId) {
  std::string opName = getNodeOpName(node);
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

  // Kind name (e.g., "PE", "Switch").
  if (!opName.empty()) {
    // Strip "fabric." prefix for display.
    std::string display = opName;
    if (display.substr(0, 7) == "fabric.")
      display = display.substr(7);
    label << dotEscape(display);
  }

  // Symbol or instance name on second line.
  if (!symName.empty())
    label << "\\n" << dotEscape(symName);

  return label.str();
}

// Build the label for an ADG node in Detailed mode.
std::string buildADGDetailedLabel(const Node *node, IdIndex nodeId) {
  std::string opName = getNodeOpName(node);
  std::string symName = getNodeStrAttr(node, "sym_name");
  std::string typeSummary = getNodeStrAttr(node, "type_summary");

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

  // Kind name.
  if (!opName.empty()) {
    std::string display = opName;
    if (display.substr(0, 7) == "fabric.")
      display = display.substr(7);
    label << dotEscape(display);
  }

  if (!symName.empty())
    label << "\\n" << dotEscape(symName);

  if (!typeSummary.empty())
    label << "\\n" << dotEscape(typeSummary);

  // Key parameters.
  int64_t numInstr = getNodeIntAttr(node, "num_instruction", -1);
  if (numInstr >= 0)
    label << "\\ninsn: " << numInstr;

  int64_t numReg = getNodeIntAttr(node, "num_register", -1);
  if (numReg >= 0)
    label << "\\nreg: " << numReg;

  std::string ldCount = getNodeStrAttr(node, "ldCount");
  std::string stCount = getNodeStrAttr(node, "stCount");
  if (!ldCount.empty() || !stCount.empty()) {
    label << "\\nld:" << (ldCount.empty() ? "0" : ldCount)
          << " st:" << (stCount.empty() ? "0" : stCount);
  }

  int64_t tableSize = getNodeIntAttr(node, "table_size", -1);
  if (tableSize >= 0)
    label << "\\ntable: " << tableSize;

  return label.str();
}

// Determine ADG edge style based on attributes.
struct ADGEdgeStyle {
  const char *style;
  const char *color;
  const char *penwidth;
};

ADGEdgeStyle getADGEdgeStyle(const Edge *edge, const Graph &adg) {
  for (const auto &attr : edge->attributes) {
    if (attr.getName().str() == "edge_type") {
      if (auto strAttr = llvm::dyn_cast<mlir::StringAttr>(attr.getValue())) {
        llvm::StringRef val = strAttr.getValue();
        if (val == "tagged")
          return {"dashed", "purple", "2.0"};
        if (val == "memref")
          return {"dotted", "blue", "2.0"};
        if (val == "control" || val == "none")
          return {"dashed", "gray", "1.0"};
      }
    }
  }
  return {"solid", "black", "2.0"};
}

} // namespace

std::string exportADGDot(const Graph &adg, const DOTOptions &opts) {
  bool isDetailed = (opts.mode == DOTMode::Detailed);
  std::ostringstream dot;

  dot << "digraph ADG {\n";
  dot << "  rankdir=" << (isDetailed ? "TB" : "LR") << ";\n";
  dot << "  node [style=filled, fontname=\"Helvetica\"];\n";
  dot << "  edge [fontname=\"Helvetica\"];\n";
  dot << "\n";

  // Emit nodes.
  IdIndex nodeIdx = 0;
  for (const auto &nodePtr : adg.nodes) {
    IdIndex nid = nodeIdx++;
    if (!nodePtr)
      continue;

    const Node *node = nodePtr.get();
    std::string opName = getNodeOpName(node);
    ADGNodeStyle style = getADGNodeStyle(opName, node->kind);

    std::string label = isDetailed ? buildADGDetailedLabel(node, nid)
                                   : buildADGStructureLabel(node, nid);

    dot << "  hw_" << nid << " [";
    dot << "id=\"hw_" << nid << "\", ";
    dot << "label=\"" << label << "\", ";
    dot << "shape=" << style.shape << ", ";
    dot << "fillcolor=\"" << style.fillColor << "\", ";
    dot << "fontcolor=\"" << style.fontColor << "\"";

    // Larger size for temporal PE.
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

    ADGEdgeStyle edgeStyle = getADGEdgeStyle(edge, adg);

    dot << "  hw_" << srcNode << " -> hw_" << dstNode << " [";
    dot << "style=" << edgeStyle.style << ", ";
    dot << "color=\"" << edgeStyle.color << "\", ";
    dot << "penwidth=" << edgeStyle.penwidth;

    // Edge labels in detailed mode.
    if (isDetailed) {
      // Show port indices if available.
      dot << ", label=\"e" << eid << "\"";
    }

    dot << "];\n";
  }

  dot << "}\n";
  return dot.str();
}

} // namespace viz
} // namespace loom

//===-- VizHTMLExporter.cpp - Self-contained HTML visualization ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Viz/VizHTMLExporter.h"
#include "loom/Viz/VizHTMLHelpers.h"

#include "loom/Dialect/Fabric/FabricOps.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

// Embedded asset declarations (generated at build time).
#include "asset_viz_standalone_js.h"
#include "asset_d3_min_js.h"
#include "asset_renderer_js.h"
#include "asset_renderer_css.h"

namespace loom {

// ---- FU-to-container map (temporal PE aggregation) ----

llvm::DenseMap<IdIndex, IdIndex> buildFUToContainerMap(const Graph &adg) {
  llvm::DenseMap<IdIndex, IdIndex> fuToContainer;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node) continue;
    int64_t parent = getNodeIntAttr(node, "parent_temporal_pe", -1);
    if (parent >= 0)
      fuToContainer[i] = static_cast<IdIndex>(parent);
  }
  return fuToContainer;
}

llvm::DenseMap<IdIndex, int>
buildFULocalIndexMap(const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer) {
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> containerFUs;
  for (const auto &entry : fuToContainer)
    containerFUs[entry.second].push_back(entry.first);
  for (auto &entry : containerFUs)
    std::sort(entry.second.begin(), entry.second.end());

  llvm::DenseMap<IdIndex, int> localIndex;
  for (const auto &entry : containerFUs)
    for (int idx = 0; idx < static_cast<int>(entry.second.size()); ++idx)
      localIndex[entry.second[idx]] = idx;
  return localIndex;
}

// ---- DFG node-to-MLIR-op correlation map ----

DFGOpMap buildDFGNodeToOpMap(mlir::ModuleOp dfgModule) {
  DFGOpMap result;
  if (!dfgModule)
    return result;

  circt::handshake::FuncOp funcOp;
  dfgModule.walk([&](circt::handshake::FuncOp op) {
    if (!funcOp ||
        (!op.getName().ends_with("_esi") &&
         funcOp.getName().ends_with("_esi")))
      funcOp = op;
  });
  if (!funcOp)
    return result;

  auto &block = funcOp.getBody().front();

  IdIndex nodeId = 0;
  for (auto &op : block) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;
    result[nodeId] = &op;
    nodeId++;
  }
  return result;
}

// ---- PE body ops map from fabric MLIR module ----

static llvm::SmallVector<std::string, 4>
extractPEBodyOps(mlir::Operation *peOp) {
  llvm::SmallVector<std::string, 4> ops;
  if (!peOp || peOp->getNumRegions() == 0)
    return ops;
  for (auto &innerOp : peOp->getRegion(0).front()) {
    if (innerOp.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;
    ops.push_back(innerOp.getName().getStringRef().str());
  }
  return ops;
}

BodyOpsMap buildPEBodyOpsMap(mlir::Operation *fabricModule) {
  BodyOpsMap result;
  if (!fabricModule)
    return result;

  BodyOpsMap defBodyOps;

  fabricModule->walk([&](fabric::PEOp peOp) {
    auto sym = peOp->getAttrOfType<mlir::StringAttr>("sym_name");
    if (!sym)
      return;
    defBodyOps[sym.getValue()] = extractPEBodyOps(peOp.getOperation());
  });

  fabricModule->walk([&](fabric::TemporalPEOp tpeOp) {
    auto sym = tpeOp->getAttrOfType<mlir::StringAttr>("sym_name");
    if (!sym)
      return;
    llvm::SmallVector<std::string, 4> combined;
    for (auto &innerOp : tpeOp.getBody().front()) {
      if (innerOp.hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      auto innerBody = extractPEBodyOps(&innerOp);
      combined.append(innerBody.begin(), innerBody.end());
    }
    defBodyOps[sym.getValue()] = std::move(combined);
  });

  fabricModule->walk([&](fabric::InstanceOp instOp) {
    auto instSym = instOp->getAttrOfType<mlir::StringAttr>("sym_name");
    if (!instSym)
      return;
    llvm::StringRef targetName = instOp.getModule();
    auto it = defBodyOps.find(targetName);
    if (it != defBodyOps.end())
      result[instSym.getValue()] = it->second;
  });

  return result;
}

// ---- Body ops lookup/write helpers ----

llvm::SmallVector<std::string, 4>
lookupBodyOps(const Node *node, llvm::StringRef symName,
              const BodyOpsMap &bodyOps) {
  auto it = bodyOps.find(symName);
  if (it != bodyOps.end())
    return it->second;
  return getNodeBodyOps(node);
}

void writeBodyOpsJSON(llvm::json::OStream &json, const Node *node,
                      llvm::StringRef symName, const BodyOpsMap &bodyOps) {
  json.attributeBegin("body_ops");
  json.arrayBegin();
  for (auto &op : lookupBodyOps(node, symName, bodyOps))
    json.value(op);
  json.arrayEnd();
  json.attributeEnd();
}

// ---- Area heuristics ----

AreaInfo computeArea(const Node *node, const Graph &adg,
                     const BodyOpsMap &bodyOps) {
  AreaInfo a;
  std::string type = nodeTypeStr(node);

  if (type == "fabric.temporal_pe") {
    a.w = 2; a.h = 2; a.cost = 4.0;
  } else if (type == "fabric.memory" || type == "fabric.extmemory") {
    a.w = 1; a.h = 2; a.cost = 2.0;
  } else if (type == "fabric.fifo" || type == "fabric.add_tag" ||
             type == "fabric.map_tag" || type == "fabric.del_tag") {
    a.w = 1; a.h = 1; a.cost = 0.25;
  } else if (type == "fabric.switch" || type == "fabric.temporal_sw") {
    a.w = 1; a.h = 1; a.cost = 1.0;
  } else {
    llvm::StringRef symName = getNodeStrAttr(node, "sym_name");
    auto ops = lookupBodyOps(node, symName, bodyOps);
    int opCount = std::max(1, static_cast<int>(ops.size()));
    bool hasFloat = false;
    bool has64bit = false;
    for (auto &op : ops) {
      if (op.find("addf") != std::string::npos ||
          op.find("mulf") != std::string::npos ||
          op.find("divf") != std::string::npos ||
          op.find("math.") != std::string::npos)
        hasFloat = true;
    }
    if (!node->outputPorts.empty()) {
      const Port *p = adg.getPort(node->outputPorts[0]);
      if (p && p->type) {
        if (auto intT = mlir::dyn_cast<mlir::IntegerType>(p->type))
          has64bit = intT.getWidth() > 32;
        if (mlir::isa<mlir::Float64Type>(p->type))
          has64bit = true;
      }
    }
    double c = static_cast<double>(opCount);
    if (hasFloat) c *= 1.5;
    if (has64bit) c *= 1.5;
    a.cost = std::max(1.0, c);
    a.w = 1; a.h = 1;
  }
  return a;
}

// ---- Build the DOT string for the DFG ----

static std::string dotLabelEscape(llvm::StringRef s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s) {
    if (c == '"')
      result += "\\\"";
    else
      result += c;
  }
  return result;
}

std::string buildDFGDot(const Graph &dfg, const DFGOpMap &dfgOpMap) {
  std::string dot;
  llvm::raw_string_ostream os(dot);

  os << "digraph DFG {\n";
  os << "  rankdir=TB;\n";
  os << "  node [style=filled, fontsize=10];\n";
  os << "  edge [color=\"#333333\"];\n\n";

  std::string sourceGroup, sinkGroup;

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node) continue;

    std::string nodeId = swId(i);
    llvm::StringRef opName = getNodeStrAttr(node, "op_name");
    auto style = dfgNodeStyle(opName, node->kind);

    std::string label;
    if (node->kind == Node::ModuleInputNode) {
      int64_t argIdx = getNodeIntAttr(node, "arg_index", -1);
      label = "arg" + std::to_string(argIdx >= 0 ? argIdx : static_cast<int64_t>(i));
    } else if (node->kind == Node::ModuleOutputNode) {
      int64_t retIdx = getNodeIntAttr(node, "ret_index", -1);
      label = "ret" + std::to_string(retIdx >= 0 ? retIdx : static_cast<int64_t>(i));
    } else {
      label = opName.str();

      auto opIt = dfgOpMap.find(i);
      if (opIt != dfgOpMap.end())
        label += getMLIRLabelSuffix(opIt->second);

      if (!node->outputPorts.empty()) {
        const Port *port = dfg.getPort(node->outputPorts[0]);
        if (port && port->type)
          label += "\\n" + printType(port->type);
      }
    }

    os << "  \"" << nodeId << "\" ["
       << "id=\"" << nodeId << "\", "
       << "label=\"" << dotLabelEscape(label) << "\", "
       << "shape=" << style.shape << ", "
       << "fillcolor=\"" << style.fillColor << "\""
       << "];\n";

    if (node->kind == Node::ModuleInputNode) {
      sourceGroup += "\"" + nodeId + "\"; ";
    } else if (node->kind == Node::ModuleOutputNode) {
      sinkGroup += "\"" + nodeId + "\"; ";
    }
  }

  if (!sourceGroup.empty())
    os << "\n  { rank=source; " << sourceGroup << "}\n";
  if (!sinkGroup.empty())
    os << "  { rank=sink; " << sinkGroup << "}\n";

  os << "\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge) continue;

    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort) continue;

    std::string srcNodeId = swId(srcPort->parentNode);
    std::string dstNodeId = swId(dstPort->parentNode);
    std::string edgeLabel = swEdgeId(i);

    bool isControl = srcPort->type && mlir::isa<mlir::NoneType>(srcPort->type);

    os << "  \"" << srcNodeId << "\" -> \"" << dstNodeId << "\" ["
       << "id=\"" << edgeLabel << "\"";
    if (isControl)
      os << ", style=dashed, color=\"#999999\", penwidth=1.0";
    else
      os << ", penwidth=2.0";
    os << "];\n";
  }

  os << "}\n";
  return dot;
}

// ---- Public interface ----

bool VizHTMLExporter::emitHTML(const Graph &adg, const Graph &dfg,
                               const MappingState &state,
                               mlir::ModuleOp dfgModule,
                               mlir::Operation *fabricModule,
                               const std::string &basePath,
                               bool vizNeato) {
  std::string outPath = basePath + ".viz.html";
  std::error_code ec;
  llvm::raw_fd_ostream out(outPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: cannot open " << outPath << ": " << ec.message()
                 << "\n";
    return false;
  }

  auto fuToContainer = buildFUToContainerMap(adg);
  auto fuLocalIndex = buildFULocalIndexMap(fuToContainer);
  auto dfgOpMap = buildDFGNodeToOpMap(dfgModule);
  auto bodyOps = buildPEBodyOpsMap(fabricModule);

  std::string dfgDotStr = buildDFGDot(dfg, dfgOpMap);

  std::string adgJsonStr;
  {
    llvm::raw_string_ostream ss(adgJsonStr);
    writeADGGraphJSON(ss, adg, state, fuToContainer, fuLocalIndex, bodyOps);
  }

  std::string mappingJsonStr;
  {
    llvm::raw_string_ostream ss(mappingJsonStr);
    writeMappingDataJSON(ss, adg, dfg, state, fuToContainer, fuLocalIndex);
  }

  std::string swMetaStr;
  {
    llvm::raw_string_ostream ss(swMetaStr);
    writeSWMetadataJSON(ss, dfg, state, fuToContainer, dfgOpMap);
  }

  std::string hwMetaStr;
  {
    llvm::raw_string_ostream ss(hwMetaStr);
    writeHWMetadataJSON(ss, adg, state, fuToContainer, bodyOps);
  }

  std::string title = "Loom Visualization";
  llvm::StringRef baseFile = llvm::sys::path::filename(basePath);
  if (!baseFile.empty())
    title = "Loom: " + baseFile.str();

  // Emit HTML
  out << "<!DOCTYPE html>\n<html>\n<head>\n"
      << "  <meta charset=\"UTF-8\">\n"
      << "  <title>" << htmlEscape(title) << "</title>\n"
      << "  <style>\n"
      << reinterpret_cast<const char *>(loom_viz_renderer_css)
      << "\n  </style>\n"
      << "</head>\n<body>\n\n";

  out << "<div id=\"toolbar\">\n"
      << "  <span id=\"title\">" << htmlEscape(title) << "</span>\n"
      << "  <div id=\"mode-buttons\">\n"
      << "    <button id=\"btn-sidebyside\" class=\"active\">Side-by-Side</button>\n"
      << "    <button id=\"btn-overlay\">Overlay</button>\n"
      << "  </div>\n"
      << "  <button id=\"btn-fit\">Fit</button>\n"
      << "  <button id=\"btn-restore\">Restore</button>\n"
      << "  <span id=\"status-bar\">Loading...</span>\n"
      << "</div>\n\n";

  out << "<div id=\"graph-area\">\n"
      << "  <div id=\"panel-adg\">\n"
      << "    <div class=\"panel-header\">Hardware (ADG)</div>\n"
      << "    <svg id=\"svg-adg\"></svg>\n"
      << "  </div>\n"
      << "  <div id=\"panel-divider\"></div>\n"
      << "  <div id=\"panel-dfg\">\n"
      << "    <div class=\"panel-header\">Software (DFG)</div>\n"
      << "    <svg id=\"svg-dfg\"></svg>\n"
      << "  </div>\n"
      << "</div>\n\n";

  out << "<div id=\"detail-panel\">\n"
      << "  <div id=\"detail-content\"></div>\n"
      << "  <button id=\"detail-close\">Close</button>\n"
      << "</div>\n\n";

  out << "<script>\n"
      << "const vizConfig = {\"neato\": " << (vizNeato ? "true" : "false")
      << "};\n\n"
      << "const adgGraph = " << scriptSafe(adgJsonStr) << ";\n\n"
      << "const dfgDot = \"" << scriptSafe(jsonEscape(dfgDotStr)) << "\";\n\n"
      << "const mappingData = " << scriptSafe(mappingJsonStr) << ";\n\n"
      << "const swNodeMetadata = " << scriptSafe(swMetaStr) << ";\n\n"
      << "const hwNodeMetadata = " << scriptSafe(hwMetaStr) << ";\n"
      << "</script>\n\n";

  out << "<script>\n"
      << reinterpret_cast<const char *>(loom_viz_viz_standalone_js)
      << "\n</script>\n\n";

  out << "<script>\n"
      << reinterpret_cast<const char *>(loom_viz_d3_min_js)
      << "\n</script>\n\n";

  out << "<script>\n"
      << reinterpret_cast<const char *>(loom_viz_renderer_js)
      << "\n</script>\n\n";

  out << "</body>\n</html>\n";

  return true;
}

} // namespace loom

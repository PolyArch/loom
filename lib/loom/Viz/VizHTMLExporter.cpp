//===-- VizHTMLExporter.cpp - Self-contained HTML visualization ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Viz/VizHTMLExporter.h"

#include "loom/Dialect/Fabric/FabricOps.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <regex>

// Embedded asset declarations (generated at build time).
#include "asset_viz_standalone_js.h"
#include "asset_d3_min_js.h"
#include "asset_renderer_js.h"
#include "asset_renderer_css.h"

namespace loom {

namespace {

// ---- Attribute helpers ----

llvm::StringRef getNodeStrAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes)
    if (attr.getName() == name)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return s.getValue();
  return "";
}

int64_t getNodeIntAttr(const Node *node, llvm::StringRef name,
                       int64_t dflt = 0) {
  for (auto &attr : node->attributes)
    if (attr.getName() == name)
      if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return i.getInt();
  return dflt;
}

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

// ---- Grid coordinate extraction from node names ----

struct GridCoord {
  int col = -1;
  int row = -1;
  bool valid = false;
};

GridCoord extractGridFromName(llvm::StringRef name) {
  GridCoord gc;
  std::string nameStr = name.str();

  // Pattern: sw_w<W>_<R>_<C> -> (C*2, R*2)
  static const std::regex reSW_W("sw_w(\\d+)_(\\d+)_(\\d+)");
  // Pattern: l<L>_sw_<R>_<C> -> (C*2, R*2)
  static const std::regex reL_SW("l(\\d+)_sw_(\\d+)_(\\d+)");
  // Pattern: sw_<R>_<C> -> (C*2, R*2)
  static const std::regex reSW("^sw_(\\d+)_(\\d+)$");
  // Pattern: <type>_r<R>_c<C> -> (C*2+1, R*2+1)
  static const std::regex rePE_RC("_r(\\d+)_c(\\d+)$");
  // Pattern: pe_<R>_<C> -> (C*2+1, R*2+1)
  static const std::regex rePE("^pe_(\\d+)_(\\d+)$");
  // Pattern: tsw_<R>_<C> -> (C*2, R*2) in temporal band
  static const std::regex reTSW("^tsw_(\\d+)_(\\d+)$");
  // Pattern: sw_w<W>_<D>_<R>_<C> -> (C*2, R*2) per depth layer
  static const std::regex reSW_WD("sw_w(\\d+)_(\\d+)_(\\d+)_(\\d+)");

  std::smatch m;

  if (std::regex_search(nameStr, m, reSW_WD)) {
    gc.col = std::stoi(m[4]) * 2;
    gc.row = std::stoi(m[3]) * 2;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, reSW_W)) {
    gc.col = std::stoi(m[3]) * 2;
    gc.row = std::stoi(m[2]) * 2;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, reL_SW)) {
    gc.col = std::stoi(m[3]) * 2;
    gc.row = std::stoi(m[2]) * 2;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, reSW)) {
    gc.col = std::stoi(m[2]) * 2;
    gc.row = std::stoi(m[1]) * 2;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, reTSW)) {
    gc.col = std::stoi(m[2]) * 2;
    gc.row = std::stoi(m[1]) * 2;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, rePE_RC)) {
    gc.col = std::stoi(m[2]) * 2 + 1;
    gc.row = std::stoi(m[1]) * 2 + 1;
    gc.valid = true;
  } else if (std::regex_search(nameStr, m, rePE)) {
    gc.col = std::stoi(m[2]) * 2 + 1;
    gc.row = std::stoi(m[1]) * 2 + 1;
    gc.valid = true;
  }

  return gc;
}

// ---- Node type classification ----

std::string nodeTypeStr(const Node *node) {
  llvm::StringRef opName = getNodeStrAttr(node, "op_name");
  if (opName.starts_with("fabric."))
    return opName.str();
  llvm::StringRef resClass = getNodeStrAttr(node, "resource_class");
  if (resClass == "routing") return "fabric.switch";
  if (resClass == "memory") return "fabric.memory";
  if (resClass == "functional") return "fabric.pe";
  return "fabric.pe";
}

// ---- Area heuristics ----

struct AreaInfo {
  int w = 1;
  int h = 1;
  double cost = 1.0;
};

AreaInfo computeArea(const Node *node, const Graph &adg) {
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
    // PE: scale by body complexity
    int64_t bodyOps = getNodeIntAttr(node, "body_op_count", 1);
    a.cost = std::max(1.0, static_cast<double>(bodyOps));
    a.w = 1; a.h = 1;
  }
  return a;
}

// ---- Type-to-string helper ----

std::string printType(mlir::Type type) {
  std::string str;
  llvm::raw_string_ostream os(str);
  type.print(os);
  return str;
}

// ---- Edge type classification ----

std::string edgeTypeStr(const Edge *edge, const Graph &g) {
  const Port *srcPort = g.getPort(edge->srcPort);
  if (!srcPort) return "native";

  mlir::Type type = srcPort->type;
  if (!type) return "native";

  std::string typeStr = printType(type);
  if (typeStr.find("tagged") != std::string::npos) return "tagged";
  if (typeStr.find("memref") != std::string::npos) return "memref";
  if (typeStr.find("none") != std::string::npos) return "control";
  return "native";
}

// ---- Bit width extraction ----

int bitWidthFromType(mlir::Type type) {
  if (!type) return 32;
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type))
    return intType.getWidth();
  if (mlir::isa<mlir::Float32Type>(type)) return 32;
  if (mlir::isa<mlir::Float64Type>(type)) return 64;
  if (mlir::isa<mlir::Float16Type>(type)) return 16;
  if (type.isIndex()) return 64;
  if (mlir::isa<mlir::NoneType>(type)) return 1;
  return 32;
}

// ---- String escaping ----

std::string htmlEscape(llvm::StringRef s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '&':  result += "&amp;"; break;
    case '<':  result += "&lt;"; break;
    case '>':  result += "&gt;"; break;
    case '"':  result += "&quot;"; break;
    default:   result += c;
    }
  }
  return result;
}

std::string jsonEscape(llvm::StringRef s) {
  std::string result;
  result.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
    case '"':  result += "\\\""; break;
    case '\\': result += "\\\\"; break;
    case '\n': result += "\\n"; break;
    case '\r': result += "\\r"; break;
    case '\t': result += "\\t"; break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buf[8];
        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
        result += buf;
      } else {
        result += c;
      }
    }
  }
  return result;
}

// ---- DFG node styles for DOT ----

struct DFGNodeStyle {
  const char *shape;
  const char *fillColor;
};

DFGNodeStyle dfgNodeStyle(llvm::StringRef opName, Node::Kind kind) {
  if (kind == Node::ModuleInputNode)
    return {"invhouse", "#ffb6c1"};
  if (kind == Node::ModuleOutputNode)
    return {"house", "#f08080"};

  if (opName.starts_with("arith."))
    return {"box", "#add8e6"};
  if (opName == "handshake.constant")
    return {"ellipse", "#ffd700"};
  if (opName == "handshake.cond_br")
    return {"diamond", "#ffffe0"};
  if (opName == "handshake.mux")
    return {"invtriangle", "#ffffe0"};
  if (opName == "handshake.join")
    return {"triangle", "#ffffe0"};
  if (opName == "handshake.load")
    return {"box", "#87ceeb"};
  if (opName == "handshake.store")
    return {"box", "#ffa07a"};
  if (opName == "handshake.memory")
    return {"cylinder", "#87ceeb"};
  if (opName == "handshake.extmemory")
    return {"hexagon", "#ffd700"};
  if (opName == "handshake.sink")
    return {"point", "#999999"};
  if (opName == "dataflow.carry")
    return {"octagon", "#90ee90"};
  if (opName == "dataflow.gate")
    return {"octagon", "#98fb98"};
  if (opName == "dataflow.invariant")
    return {"octagon", "#f5fffa"};
  if (opName == "dataflow.stream")
    return {"doubleoctagon", "#90ee90"};
  if (opName.starts_with("math."))
    return {"box", "#dda0dd"};

  return {"star", "#ff0000"};
}

// ---- Build the DOT string for the DFG ----

std::string buildDFGDot(const Graph &dfg) {
  std::string dot;
  llvm::raw_string_ostream os(dot);

  os << "digraph DFG {\n";
  os << "  rankdir=TB;\n";
  os << "  node [style=filled, fontsize=10];\n";
  os << "  edge [color=\"#333333\"];\n\n";

  // Track source/sink rank groups
  std::string sourceGroup, sinkGroup;

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node) continue;

    std::string nodeId = "sw_" + std::to_string(i);
    llvm::StringRef opName = getNodeStrAttr(node, "op_name");
    auto style = dfgNodeStyle(opName, node->kind);

    // Build label
    std::string label;
    if (node->kind == Node::ModuleInputNode) {
      int64_t argIdx = getNodeIntAttr(node, "arg_index", -1);
      label = "arg" + std::to_string(argIdx >= 0 ? argIdx : static_cast<int64_t>(i));
    } else if (node->kind == Node::ModuleOutputNode) {
      int64_t retIdx = getNodeIntAttr(node, "ret_index", -1);
      label = "ret" + std::to_string(retIdx >= 0 ? retIdx : static_cast<int64_t>(i));
    } else {
      label = opName.str();
      // Add type summary from first output port
      if (!node->outputPorts.empty()) {
        const Port *port = dfg.getPort(node->outputPorts[0]);
        if (port && port->type)
          label += "\\n" + printType(port->type);
      }
    }

    os << "  \"" << nodeId << "\" ["
       << "id=\"" << nodeId << "\", "
       << "label=\"" << jsonEscape(label) << "\", "
       << "shape=" << style.shape << ", "
       << "fillcolor=\"" << style.fillColor << "\""
       << "];\n";

    if (node->kind == Node::ModuleInputNode) {
      sourceGroup += "\"" + nodeId + "\"; ";
    } else if (node->kind == Node::ModuleOutputNode) {
      sinkGroup += "\"" + nodeId + "\"; ";
    }
  }

  // Rank constraints
  if (!sourceGroup.empty())
    os << "\n  { rank=source; " << sourceGroup << "}\n";
  if (!sinkGroup.empty())
    os << "  { rank=sink; " << sinkGroup << "}\n";

  // Edges
  os << "\n";
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge) continue;

    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort) continue;

    std::string srcId = "sw_" + std::to_string(srcPort->parentNode);
    std::string dstId = "sw_" + std::to_string(dstPort->parentNode);
    std::string edgeId = "swedge_" + std::to_string(i);

    // Check if control edge
    bool isControl = false;
    if (srcPort->type && mlir::isa<mlir::NoneType>(srcPort->type))
      isControl = true;

    os << "  \"" << srcId << "\" -> \"" << dstId << "\" ["
       << "id=\"" << edgeId << "\"";
    if (isControl)
      os << ", style=dashed, color=\"#999999\", penwidth=1.0";
    else
      os << ", penwidth=2.0";
    os << "];\n";
  }

  os << "}\n";
  return dot;
}

// ---- Build adgGraph JSON ----

void writeADGGraphJSON(llvm::raw_ostream &os, const Graph &adg,
                       const MappingState &state,
                       const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer) {
  llvm::json::OStream json(os, 2);
  json.objectBegin();

  // Nodes
  json.attributeBegin("nodes");
  json.arrayBegin();

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node) continue;

    // Skip FU sub-nodes (they are aggregated into container)
    if (fuToContainer.count(i))
      continue;

    std::string id = "hw_" + std::to_string(i);
    llvm::StringRef symName = getNodeStrAttr(node, "sym_name");
    std::string name = symName.empty() ? ("node_" + std::to_string(i)) : symName.str();
    std::string type = nodeTypeStr(node);
    llvm::StringRef resClass = getNodeStrAttr(node, "resource_class");
    std::string classStr = resClass.empty() ? "functional" : resClass.str();

    // Grid coordinates
    GridCoord gc;
    // Priority 1: viz_row/viz_col attributes
    int64_t vizCol = getNodeIntAttr(node, "viz_col", -1);
    int64_t vizRow = getNodeIntAttr(node, "viz_row", -1);
    if (vizCol >= 0 && vizRow >= 0) {
      gc.col = static_cast<int>(vizCol);
      gc.row = static_cast<int>(vizRow);
      gc.valid = true;
    }
    // Priority 2: name extraction
    if (!gc.valid)
      gc = extractGridFromName(name);

    // Area
    AreaInfo area = computeArea(node, adg);

    // Primary bit width (from first output port)
    int bw = 32;
    if (!node->outputPorts.empty()) {
      const Port *port = adg.getPort(node->outputPorts[0]);
      if (port && port->type)
        bw = bitWidthFromType(port->type);
    }

    json.objectBegin();
    json.attribute("id", id);
    json.attribute("name", name);
    json.attribute("type", type);
    json.attribute("class", classStr);

    if (gc.valid) {
      json.attribute("gridCol", gc.col);
      json.attribute("gridRow", gc.row);
    } else {
      json.attributeBegin("gridCol"); json.rawValue("null"); json.attributeEnd();
      json.attributeBegin("gridRow"); json.rawValue("null"); json.attributeEnd();
    }

    json.attribute("areaCost", area.cost);
    json.attribute("areaW", area.w);
    json.attribute("areaH", area.h);
    json.attribute("bitWidth", bw);

    // Params
    json.attributeBegin("params");
    json.objectBegin();

    // Body ops for PEs
    if (type == "fabric.pe") {
      llvm::StringRef bodyOp = getNodeStrAttr(node, "op_name");
      if (!bodyOp.empty()) {
        json.attributeBegin("body_ops");
        json.arrayBegin();
        json.value(bodyOp);
        json.arrayEnd();
        json.attributeEnd();
      }
    }

    // Temporal PE: collect FU info
    if (type == "fabric.temporal_pe") {
      int64_t numInst = getNodeIntAttr(node, "num_instruction", 0);
      int64_t numReg = getNodeIntAttr(node, "num_register", 0);
      json.attribute("num_instruction", numInst);
      json.attribute("num_register", numReg);

      json.attributeBegin("fuNodes");
      json.arrayBegin();
      for (IdIndex j = 0; j < static_cast<IdIndex>(adg.nodes.size()); ++j) {
        const Node *fuNode = adg.getNode(j);
        if (!fuNode) continue;
        int64_t parent = getNodeIntAttr(fuNode, "parent_temporal_pe", -1);
        if (parent == static_cast<int64_t>(i)) {
          json.objectBegin();
          json.attribute("id", "hw_" + std::to_string(j));
          json.attribute("name", name + "/fu_" + std::to_string(j));
          llvm::StringRef fuOp = getNodeStrAttr(fuNode, "op_name");
          if (!fuOp.empty())
            json.attribute("op", fuOp);
          json.objectEnd();
        }
      }
      json.arrayEnd();
      json.attributeEnd();
    }

    json.objectEnd();
    json.attributeEnd();

    json.objectEnd();
  }

  json.arrayEnd();
  json.attributeEnd();

  // Edges
  json.attributeBegin("edges");
  json.arrayBegin();

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.edges.size()); ++i) {
    const Edge *edge = adg.getEdge(i);
    if (!edge) continue;

    const Port *srcPort = adg.getPort(edge->srcPort);
    const Port *dstPort = adg.getPort(edge->dstPort);
    if (!srcPort || !dstPort) continue;

    // Resolve parent nodes (skip edges between FU sub-nodes and container)
    IdIndex srcNode = srcPort->parentNode;
    IdIndex dstNode = dstPort->parentNode;

    // Resolve FU -> container
    if (fuToContainer.count(srcNode))
      srcNode = fuToContainer.lookup(srcNode);
    if (fuToContainer.count(dstNode))
      dstNode = fuToContainer.lookup(dstNode);

    std::string eType = edgeTypeStr(edge, adg);
    int totalBw = 32;
    int valueBw = 32;
    int tagBw = 0;

    if (srcPort->type) {
      valueBw = bitWidthFromType(srcPort->type);
      totalBw = valueBw;

      // Check for tagged type
      if (printType(srcPort->type).find("tagged") != std::string::npos) {
        tagBw = getNodeIntAttr(adg.getNode(srcNode), "tag_width", 4);
        totalBw = valueBw + tagBw;
      }
    }

    json.objectBegin();
    json.attribute("id", "hwedge_" + std::to_string(i));
    json.attribute("srcNode", "hw_" + std::to_string(srcNode));
    json.attribute("dstNode", "hw_" + std::to_string(dstNode));
    json.attribute("srcPort", std::to_string(edge->srcPort));
    json.attribute("dstPort", std::to_string(edge->dstPort));
    json.attribute("edgeType", eType);
    json.attribute("bitWidth", totalBw);
    json.attribute("valueBitWidth", valueBw);
    json.attribute("tagBitWidth", tagBw);
    json.objectEnd();
  }

  json.arrayEnd();
  json.attributeEnd();

  json.objectEnd();
}

// ---- Build mappingData JSON ----

void writeMappingDataJSON(llvm::raw_ostream &os, const Graph &adg,
                          const Graph &dfg, const MappingState &state,
                          const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer) {
  // Route color palette
  static const char *palette[] = {
    "#e6194b","#3cb44b","#4363d8","#f58231",
    "#911eb4","#42d4f4","#f032e6","#bfef45",
    "#fabed4","#469990","#dcbeff","#9A6324"
  };

  llvm::json::OStream json(os, 2);
  json.objectBegin();

  // swToHw
  json.attributeBegin("swToHw");
  json.objectBegin();
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    IdIndex hwId = state.swNodeToHwNode[i];
    if (hwId == INVALID_ID) continue;
    // Resolve FU to container
    if (fuToContainer.count(hwId))
      hwId = fuToContainer.lookup(hwId);
    json.attribute("sw_" + std::to_string(i), "hw_" + std::to_string(hwId));
  }
  json.objectEnd();
  json.attributeEnd();

  // hwToSw (aggregated by container)
  json.attributeBegin("hwToSw");
  json.objectBegin();
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> hwToSwAgg;
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    IdIndex hwId = state.swNodeToHwNode[i];
    if (hwId == INVALID_ID) continue;
    if (fuToContainer.count(hwId))
      hwId = fuToContainer.lookup(hwId);
    hwToSwAgg[hwId].push_back(i);
  }
  for (auto &kv : hwToSwAgg) {
    json.attributeBegin("hw_" + std::to_string(kv.first));
    json.arrayBegin();
    for (IdIndex swId : kv.second)
      json.value("sw_" + std::to_string(swId));
    json.arrayEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Routes
  json.attributeBegin("routes");
  json.objectBegin();
  int colorIdx = 0;
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++i) {
    const auto &pathVec = state.swEdgeToHwPaths[i];
    if (pathVec.empty()) continue;

    std::string key = "swedge_" + std::to_string(i);
    json.attributeBegin(key);
    json.objectBegin();

    json.attributeBegin("hwPath");
    json.arrayBegin();
    for (size_t j = 0; j + 1 < pathVec.size(); j += 2) {
      json.objectBegin();
      json.attribute("src", std::to_string(pathVec[j]));
      json.attribute("dst", std::to_string(pathVec[j + 1]));

      // Resolve hwEdgeId by finding the ADG edge connecting these ports
      std::string hwEdgeId;
      const Port *srcP = adg.getPort(pathVec[j]);
      if (srcP) {
        for (IdIndex eid : srcP->connectedEdges) {
          const Edge *e = adg.getEdge(eid);
          if (e && e->srcPort == pathVec[j] && e->dstPort == pathVec[j + 1]) {
            hwEdgeId = "hwedge_" + std::to_string(eid);
            break;
          }
        }
      }
      if (!hwEdgeId.empty())
        json.attribute("hwEdgeId", hwEdgeId);

      json.objectEnd();
    }
    json.arrayEnd();
    json.attributeEnd();

    json.attribute("color", palette[colorIdx % 12]);
    colorIdx++;

    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Temporal
  json.attributeBegin("temporal");
  json.objectBegin();
  for (IdIndex i = 0;
       i < static_cast<IdIndex>(state.temporalPEAssignments.size()); ++i) {
    const auto &tpa = state.temporalPEAssignments[i];
    if (tpa.slot == INVALID_ID) continue;

    json.attributeBegin("sw_" + std::to_string(i));
    json.objectBegin();

    // Find container
    IdIndex hwId = INVALID_ID;
    if (i < state.swNodeToHwNode.size())
      hwId = state.swNodeToHwNode[i];
    IdIndex containerId = INVALID_ID;
    if (hwId != INVALID_ID && fuToContainer.count(hwId))
      containerId = fuToContainer.lookup(hwId);

    if (containerId != INVALID_ID) {
      json.attribute("container", "hw_" + std::to_string(containerId));
      llvm::StringRef containerName = getNodeStrAttr(adg.getNode(containerId), "sym_name");
      json.attribute("fuName",
                     (containerName.empty() ? "tpe_" + std::to_string(containerId)
                                            : containerName.str()) +
                         "/fu_" + std::to_string(hwId));
    }

    json.attribute("slot", static_cast<int64_t>(tpa.slot));
    json.attribute("tag", static_cast<int64_t>(tpa.tag));

    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  // Registers
  json.attributeBegin("registers");
  json.objectBegin();
  for (IdIndex i = 0;
       i < static_cast<IdIndex>(state.registerAssignments.size()); ++i) {
    if (state.registerAssignments[i] == INVALID_ID) continue;

    json.attributeBegin("swedge_" + std::to_string(i));
    json.objectBegin();
    json.attribute("registerIndex",
                   static_cast<int64_t>(state.registerAssignments[i]));
    json.objectEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  json.objectEnd();
}

// ---- Build swNodeMetadata JSON ----

void writeSWMetadataJSON(llvm::raw_ostream &os, const Graph &dfg,
                         const MappingState &state,
                         const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer) {
  llvm::json::OStream json(os, 2);
  json.objectBegin();

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node) continue;

    std::string key = "sw_" + std::to_string(i);
    json.attributeBegin(key);
    json.objectBegin();

    llvm::StringRef opName = getNodeStrAttr(node, "op_name");
    json.attribute("op", opName.empty() ? "unknown" : opName);

    // Type summary from first output port
    if (!node->outputPorts.empty()) {
      const Port *port = dfg.getPort(node->outputPorts[0]);
      if (port && port->type)
        json.attribute("types", printType(port->type));
    }

    // Loc
    llvm::StringRef loc = getNodeStrAttr(node, "loc");
    if (!loc.empty())
      json.attribute("loc", loc);

    // HW target
    if (i < state.swNodeToHwNode.size() &&
        state.swNodeToHwNode[i] != INVALID_ID) {
      IdIndex hwId = state.swNodeToHwNode[i];
      if (fuToContainer.count(hwId))
        hwId = fuToContainer.lookup(hwId);
      json.attribute("hwTarget", "hw_" + std::to_string(hwId));
    }

    // Attrs (selected interesting ones)
    json.attributeBegin("attrs");
    json.objectBegin();
    for (auto &attr : node->attributes) {
      llvm::StringRef name = attr.getName();
      if (name == "op_name" || name == "loc" || name == "sym_name")
        continue;
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        json.attribute(name, strAttr.getValue());
      else if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        json.attribute(name, intAttr.getInt());
    }
    json.objectEnd();
    json.attributeEnd();

    json.objectEnd();
    json.attributeEnd();
  }

  json.objectEnd();
}

// ---- Build hwNodeMetadata JSON ----

void writeHWMetadataJSON(llvm::raw_ostream &os, const Graph &adg,
                         const MappingState &state,
                         const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer) {
  llvm::json::OStream json(os, 2);
  json.objectBegin();

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node) continue;
    // Skip FU sub-nodes
    if (fuToContainer.count(i)) continue;

    std::string key = "hw_" + std::to_string(i);
    json.attributeBegin(key);
    json.objectBegin();

    llvm::StringRef symName = getNodeStrAttr(node, "sym_name");
    json.attribute("name", symName.empty() ? ("node_" + std::to_string(i))
                                           : symName);
    json.attribute("type", nodeTypeStr(node));

    // Body ops
    llvm::StringRef bodyOp = getNodeStrAttr(node, "op_name");
    json.attributeBegin("body_ops");
    json.arrayBegin();
    if (!bodyOp.empty())
      json.value(bodyOp);
    json.arrayEnd();
    json.attributeEnd();

    // Port counts
    json.attributeBegin("ports");
    json.objectBegin();
    json.attribute("in", static_cast<int64_t>(node->inputPorts.size()));
    json.attribute("out", static_cast<int64_t>(node->outputPorts.size()));
    json.objectEnd();
    json.attributeEnd();

    // Mapped SW nodes
    json.attributeBegin("mappedSw");
    json.arrayBegin();
    if (i < state.hwNodeToSwNodes.size()) {
      for (IdIndex swId : state.hwNodeToSwNodes[i])
        json.value("sw_" + std::to_string(swId));
    }
    json.arrayEnd();
    json.attributeEnd();

    json.objectEnd();
    json.attributeEnd();
  }

  json.objectEnd();
}

} // anonymous namespace

// ---- Public interface ----

bool VizHTMLExporter::emitHTML(const Graph &adg, const Graph &dfg,
                               const MappingState &state,
                               mlir::ModuleOp mlirModule,
                               const std::string &basePath) {
  (void)mlirModule; // Reserved for future use.

  std::string outPath = basePath + ".viz.html";
  std::error_code ec;
  llvm::raw_fd_ostream out(outPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: cannot open " << outPath << ": " << ec.message()
                 << "\n";
    return false;
  }

  auto fuToContainer = buildFUToContainerMap(adg);

  // Build DFG DOT string
  std::string dfgDotStr = buildDFGDot(dfg);

  // Build JSON data strings
  std::string adgJsonStr;
  {
    llvm::raw_string_ostream ss(adgJsonStr);
    writeADGGraphJSON(ss, adg, state, fuToContainer);
  }

  std::string mappingJsonStr;
  {
    llvm::raw_string_ostream ss(mappingJsonStr);
    writeMappingDataJSON(ss, adg, dfg, state, fuToContainer);
  }

  std::string swMetaStr;
  {
    llvm::raw_string_ostream ss(swMetaStr);
    writeSWMetadataJSON(ss, dfg, state, fuToContainer);
  }

  std::string hwMetaStr;
  {
    llvm::raw_string_ostream ss(hwMetaStr);
    writeHWMetadataJSON(ss, adg, state, fuToContainer);
  }

  // Derive title
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

  // Toolbar
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

  // Graph area
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

  // Detail panel
  out << "<div id=\"detail-panel\">\n"
      << "  <div id=\"detail-content\"></div>\n"
      << "  <button id=\"detail-close\">Close</button>\n"
      << "</div>\n\n";

  // Embedded data
  out << "<script>\n"
      << "const adgGraph = " << adgJsonStr << ";\n\n"
      << "const dfgDot = \"" << jsonEscape(dfgDotStr) << "\";\n\n"
      << "const mappingData = " << mappingJsonStr << ";\n\n"
      << "const swNodeMetadata = " << swMetaStr << ";\n\n"
      << "const hwNodeMetadata = " << hwMetaStr << ";\n"
      << "</script>\n\n";

  // Vendored assets
  out << "<script>\n"
      << reinterpret_cast<const char *>(loom_viz_viz_standalone_js)
      << "\n</script>\n\n";

  out << "<script>\n"
      << reinterpret_cast<const char *>(loom_viz_d3_min_js)
      << "\n</script>\n\n";

  // Renderer
  out << "<script>\n"
      << reinterpret_cast<const char *>(loom_viz_renderer_js)
      << "\n</script>\n\n";

  out << "</body>\n</html>\n";

  return true;
}

} // namespace loom

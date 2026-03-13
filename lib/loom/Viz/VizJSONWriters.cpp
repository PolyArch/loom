//===-- VizJSONWriters.cpp - JSON data builders for visualization --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// JSON serialization for ADG graph, mapping data, and metadata used by the
// HTML visualization renderer. Also includes string escaping helpers and
// MLIR attribute extraction for DFG label enrichment.
//
//===----------------------------------------------------------------------===//

#include "loom/Viz/VizHTMLHelpers.h"
#include "loom/Simulator/SimTypes.h"

#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <map>
#include <regex>

namespace loom {

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

std::string scriptSafe(const std::string &s) {
  std::string result;
  result.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '<' && i + 1 < s.size() && s[i + 1] == '/') {
      result += "<\\/";
      ++i;
    } else {
      result += s[i];
    }
  }
  return result;
}

// ---- MLIR attribute extraction helpers ----

static const char *arithCmpIPredicateName(int64_t pred) {
  static const char *names[] = {
    "eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"
  };
  if (pred >= 0 && pred < 10)
    return names[pred];
  return "?";
}

static const char *arithCmpFPredicateName(int64_t pred) {
  static const char *names[] = {
    "false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord",
    "ueq", "ugt", "uge", "ult", "ule", "une", "uno", "true"
  };
  if (pred >= 0 && pred < 16)
    return names[pred];
  return "?";
}

std::string getMLIRLabelSuffix(mlir::Operation *op) {
  if (!op)
    return "";
  llvm::StringRef opName = op->getName().getStringRef();

  if (opName == "handshake.constant") {
    if (auto val = op->getAttr("value")) {
      std::string str;
      llvm::raw_string_ostream os(str);
      if (auto intA = mlir::dyn_cast<mlir::IntegerAttr>(val))
        os << intA.getInt();
      else if (auto fpA = mlir::dyn_cast<mlir::FloatAttr>(val))
        os << fpA.getValueAsDouble();
      else
        val.print(os);
      return " = " + str;
    }
  }

  if (opName == "arith.cmpi") {
    if (auto predAttr = op->getAttrOfType<mlir::IntegerAttr>("predicate"))
      return std::string(" ") + arithCmpIPredicateName(predAttr.getInt());
  }
  if (opName == "arith.cmpf") {
    if (auto predAttr = op->getAttrOfType<mlir::IntegerAttr>("predicate"))
      return std::string(" ") + arithCmpFPredicateName(predAttr.getInt());
  }

  if (opName == "dataflow.stream") {
    if (auto stepOp = op->getAttrOfType<mlir::StringAttr>("step_op"))
      return " [" + stepOp.getValue().str() + "]";
  }

  if (opName == "handshake.extmemory" || opName == "handshake.memory") {
    std::string suffix;
    if (auto ld = op->getAttrOfType<mlir::IntegerAttr>("ldCount"))
      suffix += " ld=" + std::to_string(ld.getInt());
    if (auto st = op->getAttrOfType<mlir::IntegerAttr>("stCount"))
      suffix += " st=" + std::to_string(st.getInt());
    return suffix;
  }

  return "";
}

void writeMLIRAttrs(llvm::json::OStream &json, mlir::Operation *op) {
  if (!op)
    return;
  llvm::StringRef opName = op->getName().getStringRef();

  if (opName == "handshake.constant") {
    if (auto val = op->getAttr("value")) {
      std::string str;
      llvm::raw_string_ostream os(str);
      val.print(os);
      json.attribute("value", str);
    }
  }

  if (opName == "arith.cmpi") {
    if (auto predAttr = op->getAttrOfType<mlir::IntegerAttr>("predicate"))
      json.attribute("predicate", arithCmpIPredicateName(predAttr.getInt()));
  }
  if (opName == "arith.cmpf") {
    if (auto predAttr = op->getAttrOfType<mlir::IntegerAttr>("predicate"))
      json.attribute("predicate", arithCmpFPredicateName(predAttr.getInt()));
  }

  if (opName == "dataflow.stream") {
    if (auto stepOp = op->getAttrOfType<mlir::StringAttr>("step_op"))
      json.attribute("step_op", stepOp.getValue());
  }

  if (opName == "handshake.extmemory" || opName == "handshake.memory") {
    if (auto ld = op->getAttrOfType<mlir::IntegerAttr>("ldCount"))
      json.attribute("ldCount", ld.getInt());
    if (auto st = op->getAttrOfType<mlir::IntegerAttr>("stCount"))
      json.attribute("stCount", st.getInt());
  }
}

// ---- DFG node styles ----

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

// ---- Build adgGraph JSON ----

void writeADGGraphJSON(llvm::raw_ostream &os, const Graph &adg,
                       const MappingState &state,
                       const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
                       const llvm::DenseMap<IdIndex, int> &fuLocalIndex,
                       const BodyOpsMap &bodyOps) {
  llvm::json::OStream json(os, 2);
  json.objectBegin();

  // Scan node names to compute layout parameters.
  int meshBandSize = 10;
  int meshBaseOffset = 0;
  int temporalRowOffset = 0;
  int planeBandSize = 0;
  {
    static const std::regex rePeMesh2D("^(?:pe|tpe)_[a-z]_(\\d+)_(\\d+)$");
    static const std::regex rePeMesh1D("^(?:pe|tpe)_[a-z]_(\\d+)$");
    static const std::regex reSpatialSW("^sw_(\\d+)_(\\d+)$");
    static const std::regex reSpatialPE("^pe_(\\d+)_(\\d+)$");
    int maxMeshIdx = 0;
    int maxSpatialRow = 0;
    int maxSpatialCol = 0;
    bool hasBaseGrid = false;
    for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
      const Node *node = adg.getNode(i);
      if (!node || fuToContainer.count(i))
        continue;
      llvm::StringRef sn = getNodeStrAttr(node, "sym_name");
      if (sn.empty()) continue;
      std::string nameStr = sn.str();
      std::smatch m;
      if (sn.starts_with("pe_") || sn.starts_with("tpe_")) {
        if (std::regex_match(nameStr, m, rePeMesh2D))
          maxMeshIdx = std::max(maxMeshIdx, std::stoi(m[2]));
        else if (std::regex_match(nameStr, m, rePeMesh1D))
          maxMeshIdx = std::max(maxMeshIdx, std::stoi(m[1]));
      }
      if (std::regex_match(nameStr, m, reSpatialSW)) {
        maxSpatialRow = std::max(maxSpatialRow, std::stoi(m[1]));
        maxSpatialCol = std::max(maxSpatialCol, std::stoi(m[2]));
        hasBaseGrid = true;
      } else if (std::regex_match(nameStr, m, reSpatialPE)) {
        maxSpatialRow = std::max(maxSpatialRow, std::stoi(m[1]));
        maxSpatialCol = std::max(maxSpatialCol, std::stoi(m[2]));
        hasBaseGrid = true;
      }
    }
    meshBandSize = std::max(10, (maxMeshIdx + 1) * 2 + 2);
    if (hasBaseGrid)
      meshBaseOffset = (maxSpatialCol + 1) * 2 + 2;
    temporalRowOffset = (maxSpatialRow + 1) * 2 + 2;
    planeBandSize = std::max(10, (maxSpatialCol + 1) * 2 + 2);
  }

  // Pre-compute grid coordinates.
  llvm::DenseMap<IdIndex, GridCoord> nodeCoords;
  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node || fuToContainer.count(i))
      continue;

    GridCoord gc;
    int64_t vizCol = getNodeIntAttr(node, "viz_col", -1);
    int64_t vizRow = getNodeIntAttr(node, "viz_row", -1);
    if (vizCol >= 0 && vizRow >= 0) {
      gc.col = static_cast<int>(vizCol);
      gc.row = static_cast<int>(vizRow);
      gc.valid = true;
    }
    if (!gc.valid) {
      gc = extractGridFromName(nodeName(node, i), meshBandSize,
                               temporalRowOffset, planeBandSize,
                               meshBaseOffset);
    }
    if (gc.valid)
      nodeCoords[i] = gc;
  }

  // Detect width planes and separate overlapping sub-tiles.
  llvm::DenseMap<int, std::string> planeLabels;
  auto nodePlane = detectAndSeparateWidthPlanes(adg, nodeCoords, fuToContainer,
                                                planeLabels);

  // Fallback inference for remaining unplaced nodes.
  if (!nodePlane.empty()) {
    int crossIdx = -1;
    for (auto &kv : planeLabels)
      if (kv.second == "cross") { crossIdx = kv.first; break; }
    inferMissingCoordsPlaneAware(adg, nodeCoords, fuToContainer, nodePlane,
                                 crossIdx);
  } else {
    inferMissingCoords(adg, nodeCoords, fuToContainer);
  }

  // Nodes
  json.attributeBegin("nodes");
  json.arrayBegin();

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node) continue;
    if (fuToContainer.count(i))
      continue;

    std::string id = hwId(i);
    llvm::StringRef symName = getNodeStrAttr(node, "sym_name");
    std::string name = nodeName(node, i);
    std::string type = nodeTypeStr(node);
    llvm::StringRef resClass = getNodeStrAttr(node, "resource_class");
    std::string classStr;
    if (node->kind == Node::ModuleInputNode ||
        node->kind == Node::ModuleOutputNode)
      classStr = "boundary";
    else
      classStr = resClass.empty() ? "functional" : resClass.str();

    auto coordIt = nodeCoords.find(i);
    bool hasCoord = coordIt != nodeCoords.end() && coordIt->second.valid;

    AreaInfo area = computeArea(node, adg, bodyOps);

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

    if (hasCoord) {
      json.attribute("gridCol", coordIt->second.col);
      json.attribute("gridRow", coordIt->second.row);
      json.attribute("coordExplicit", !coordIt->second.inferred);
    } else {
      json.attributeBegin("gridCol"); json.rawValue("null"); json.attributeEnd();
      json.attributeBegin("gridRow"); json.rawValue("null"); json.attributeEnd();
      json.attribute("coordExplicit", false);
    }

    json.attribute("areaCost", area.cost);
    json.attribute("areaW", area.w);
    json.attribute("areaH", area.h);
    json.attribute("bitWidth", bw);

    auto planeIt = nodePlane.find(i);
    if (planeIt != nodePlane.end()) {
      json.attribute("widthPlane", planeIt->second);
      auto labelIt = planeLabels.find(planeIt->second);
      if (labelIt != planeLabels.end())
        json.attribute("widthPlaneLabel", labelIt->second);
    }

    // Params
    json.attributeBegin("params");
    json.objectBegin();

    if (type == "fabric.pe") {
      writeBodyOpsJSON(json, node, symName, bodyOps);
    }

    if (type == "fabric.temporal_pe") {
      json.attribute("num_instruction",
                     getNodeIntAttr(node, "num_instruction", 0));
      json.attribute("num_register",
                     getNodeIntAttr(node, "num_register", 0));

      auto ops = lookupBodyOps(node, symName, bodyOps);
      if (!ops.empty()) {
        json.attributeBegin("body_ops");
        json.arrayBegin();
        for (auto &op : ops)
          json.value(op);
        json.arrayEnd();
        json.attributeEnd();
      }

      json.attributeBegin("fuNodes");
      json.arrayBegin();
      for (IdIndex j = 0; j < static_cast<IdIndex>(adg.nodes.size()); ++j) {
        const Node *fuNode = adg.getNode(j);
        if (!fuNode) continue;
        int64_t parent = getNodeIntAttr(fuNode, "parent_temporal_pe", -1);
        if (parent == static_cast<int64_t>(i)) {
          auto idxIt = fuLocalIndex.find(j);
          int localIdx = idxIt != fuLocalIndex.end() ? idxIt->second : 0;
          json.objectBegin();
          json.attribute("id", hwId(j));
          json.attribute("name",
                         name + "/fu_" + std::to_string(localIdx));
          auto fuBodyOps = getNodeBodyOps(fuNode);
          if (!fuBodyOps.empty()) {
            std::string opStr;
            for (size_t k = 0; k < fuBodyOps.size(); ++k) {
              if (k > 0) opStr += "+";
              opStr += fuBodyOps[k];
            }
            json.attribute("op", opStr);
          } else {
            llvm::StringRef fuOp = getNodeStrAttr(fuNode, "op_name");
            if (!fuOp.empty())
              json.attribute("op", fuOp);
          }
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

    IdIndex srcNode = resolveContainer(srcPort->parentNode, fuToContainer);
    IdIndex dstNode = resolveContainer(dstPort->parentNode, fuToContainer);

    std::string eType = edgeTypeStr(edge, adg);
    int totalBw = 32;
    int valueBw = 32;
    int tagBw = 0;

    if (srcPort->type) {
      if (auto taggedType =
              mlir::dyn_cast<dataflow::TaggedType>(srcPort->type)) {
        auto valW = fabric::getNativeBitWidth(taggedType.getValueType());
        valueBw = valW ? static_cast<int>(*valW) : 32;
        tagBw = taggedType.getTagType().getWidth();
        totalBw = valueBw + tagBw;
      } else {
        valueBw = bitWidthFromType(srcPort->type);
        totalBw = valueBw;
      }
    }

    json.objectBegin();
    json.attribute("id", hwEdgeId(i));
    json.attribute("srcNode", hwId(srcNode));
    json.attribute("dstNode", hwId(dstNode));
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
                          const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
                          const llvm::DenseMap<IdIndex, int> &fuLocalIndex) {
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
    IdIndex hw = resolveContainer(state.swNodeToHwNode[i], fuToContainer);
    if (hw == INVALID_ID) continue;
    json.attribute(swId(i), hwId(hw));
  }
  json.objectEnd();
  json.attributeEnd();

  // hwToSw
  json.attributeBegin("hwToSw");
  json.objectBegin();
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> hwToSwAgg;
  for (IdIndex i = 0; i < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++i) {
    IdIndex hw = resolveContainer(state.swNodeToHwNode[i], fuToContainer);
    if (hw == INVALID_ID) continue;
    hwToSwAgg[hw].push_back(i);
  }
  for (auto &kv : hwToSwAgg) {
    json.attributeBegin(hwId(kv.first));
    json.arrayBegin();
    for (IdIndex sw : kv.second)
      json.value(swId(sw));
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

    json.attributeBegin(swEdgeId(i));
    json.objectBegin();

    json.attributeBegin("hwPath");
    json.arrayBegin();
    for (size_t j = 0; j + 1 < pathVec.size(); j += 2) {
      json.objectBegin();
      json.attribute("src", std::to_string(pathVec[j]));
      json.attribute("dst", std::to_string(pathVec[j + 1]));

      std::string resolvedEdgeId;
      const Port *srcP = adg.getPort(pathVec[j]);
      if (srcP) {
        for (IdIndex eid : srcP->connectedEdges) {
          const Edge *e = adg.getEdge(eid);
          if (e && e->srcPort == pathVec[j] && e->dstPort == pathVec[j + 1]) {
            resolvedEdgeId = hwEdgeId(eid);
            break;
          }
        }
      }
      if (!resolvedEdgeId.empty())
        json.attribute("hwEdgeId", resolvedEdgeId);

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

    json.attributeBegin(swId(i));
    json.objectBegin();

    IdIndex hw = (i < state.swNodeToHwNode.size())
                     ? state.swNodeToHwNode[i] : INVALID_ID;
    if (hw != INVALID_ID && fuToContainer.count(hw)) {
      IdIndex container = fuToContainer.lookup(hw);
      json.attribute("container", hwId(container));
      std::string cName = nodeName(adg.getNode(container), container);
      auto idxIt = fuLocalIndex.find(hw);
      int localIdx = idxIt != fuLocalIndex.end() ? idxIt->second : 0;
      json.attribute("fuName", cName + "/fu_" + std::to_string(localIdx));
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

    json.attributeBegin(swEdgeId(i));
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
                         const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
                         const DFGOpMap &dfgOpMap) {
  llvm::json::OStream json(os, 2);
  json.objectBegin();

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node) continue;

    json.attributeBegin(swId(i));
    json.objectBegin();

    llvm::StringRef opName = getNodeStrAttr(node, "op_name");
    json.attribute("op", opName.empty() ? "unknown" : opName);

    if (!node->outputPorts.empty()) {
      const Port *port = dfg.getPort(node->outputPorts[0]);
      if (port && port->type)
        json.attribute("types", printType(port->type));
    }

    llvm::StringRef loc = getNodeStrAttr(node, "loc");
    if (!loc.empty())
      json.attribute("loc", loc);

    if (i < state.swNodeToHwNode.size() &&
        state.swNodeToHwNode[i] != INVALID_ID) {
      IdIndex hw = resolveContainer(state.swNodeToHwNode[i], fuToContainer);
      json.attribute("hwTarget", hwId(hw));
    }

    // Attrs
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
    auto opIt = dfgOpMap.find(i);
    if (opIt != dfgOpMap.end())
      writeMLIRAttrs(json, opIt->second);
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
                         const llvm::DenseMap<IdIndex, IdIndex> &fuToContainer,
                         const BodyOpsMap &bodyOps) {
  llvm::json::OStream json(os, 2);
  json.objectBegin();

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node) continue;
    if (fuToContainer.count(i)) continue;

    json.attributeBegin(hwId(i));
    json.objectBegin();

    llvm::StringRef symName = getNodeStrAttr(node, "sym_name");
    json.attribute("name", nodeName(node, i));
    std::string type = nodeTypeStr(node);
    json.attribute("type", type);

    writeBodyOpsJSON(json, node, symName, bodyOps);

    json.attributeBegin("ports");
    json.objectBegin();
    json.attribute("in", static_cast<int64_t>(node->inputPorts.size()));
    json.attribute("out", static_cast<int64_t>(node->outputPorts.size()));
    json.objectEnd();
    json.attributeEnd();

    // Mapped SW nodes
    json.attributeBegin("mappedSw");
    json.arrayBegin();
    if (type == "fabric.temporal_pe") {
      llvm::SmallVector<IdIndex, 4> aggregated;
      for (auto &kv : fuToContainer) {
        if (kv.second != i)
          continue;
        if (kv.first < state.hwNodeToSwNodes.size()) {
          for (IdIndex sw : state.hwNodeToSwNodes[kv.first])
            aggregated.push_back(sw);
        }
      }
      if (i < state.hwNodeToSwNodes.size()) {
        for (IdIndex sw : state.hwNodeToSwNodes[i])
          aggregated.push_back(sw);
      }
      for (IdIndex sw : aggregated)
        json.value(swId(sw));
    } else {
      if (i < state.hwNodeToSwNodes.size()) {
        for (IdIndex sw : state.hwNodeToSwNodes[i])
          json.value(swId(sw));
      }
    }
    json.arrayEnd();
    json.attributeEnd();

    json.objectEnd();
    json.attributeEnd();
  }

  json.objectEnd();
}

// ---- Trace data JSON (for simulation playback) ----

void writeTraceDataJSON(llvm::raw_ostream &os,
                        const std::vector<sim::TraceEvent> &events,
                        uint64_t totalCycles, uint64_t configCycles) {
  // Index events by cycle: cycle -> [(hwNodeId, eventKind), ...]
  std::map<uint64_t, std::vector<std::pair<uint32_t, uint8_t>>> byCycle;
  for (const auto &ev : events) {
    if (ev.eventKind == sim::EV_INVOCATION_START ||
        ev.eventKind == sim::EV_INVOCATION_DONE ||
        ev.eventKind == sim::EV_DEVICE_ERROR)
      continue;
    byCycle[ev.cycle].emplace_back(ev.hwNodeId,
                                   static_cast<uint8_t>(ev.eventKind));
  }

  llvm::json::OStream json(os, /*IndentSize=*/0);
  json.objectBegin();

  json.attributeBegin("totalCycles");
  json.value(static_cast<int64_t>(totalCycles));
  json.attributeEnd();

  json.attributeBegin("configCycles");
  json.value(static_cast<int64_t>(configCycles));
  json.attributeEnd();

  // Emit per-cycle event arrays as compact [hwNodeId, eventKind] pairs.
  json.attributeBegin("cycleEvents");
  json.objectBegin();
  for (const auto &entry : byCycle) {
    json.attributeBegin(std::to_string(entry.first));
    json.arrayBegin();
    for (const auto &pair : entry.second) {
      json.arrayBegin();
      json.value(static_cast<int64_t>(pair.first));
      json.value(static_cast<int64_t>(pair.second));
      json.arrayEnd();
    }
    json.arrayEnd();
    json.attributeEnd();
  }
  json.objectEnd();
  json.attributeEnd();

  json.objectEnd();
}

} // namespace loom

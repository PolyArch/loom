#include "fcc/Viz/VizExporter.h"

#include "VizAssets.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/StringSet.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <regex>

namespace fcc {

// ---- String escaping ----

static std::string htmlEscape(llvm::StringRef s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '&':
      result += "&amp;";
      break;
    case '<':
      result += "&lt;";
      break;
    case '>':
      result += "&gt;";
      break;
    case '"':
      result += "&quot;";
      break;
    default:
      result += c;
    }
  }
  return result;
}

static std::string jsonEscapeStr(llvm::StringRef s) {
  std::string result;
  result.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
    case '"':
      result += "\\\"";
      break;
    case '\\':
      result += "\\\\";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    case '\t':
      result += "\\t";
      break;
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

/// Prevent `</script>` from appearing inside script blocks.
static std::string scriptSafe(const std::string &s) {
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

// ---- JSON builders ----

/// Parse grid position from a node name like "fifo_2_3" or "extmem_a".
/// Returns {row, col} or {-1, -1} if no position found.
static std::pair<int, int> parseGridPosFromName(llvm::StringRef name) {
  std::string str = name.str();
  std::regex re("_(\\d+)_(\\d+)$");
  std::smatch m;
  if (std::regex_search(str, m, re))
    return {std::stoi(m[1].str()), std::stoi(m[2].str())};
  return {-1, -1};
}

/// Write ADG graph as JSON with port-level detail (like loom).
/// Nodes include gridCol/gridRow using loom's interleaved convention:
///   PEs at odd positions (row*2+1, col*2+1)
///   SWs at even positions (row*2, col*2)
/// This produces a checkerboard layout where PEs and SWs alternate.
static void writeADGJSON(llvm::raw_ostream &os, const Graph &adg,
                         const ADGFlattener &flattener) {
  // Build PE containment map: FU node ID -> PE name and representative ID.
  // All FU nodes belonging to the same PE are aggregated into one visual node.
  auto &peContainment = flattener.getPEContainment();
  llvm::DenseMap<IdIndex, llvm::StringRef> fuToPEName;
  llvm::DenseMap<IdIndex, IdIndex> fuToRepresentative;
  // PE name -> representative FU node ID (first FU in each PE)
  llvm::StringMap<IdIndex> peRepresentative;
  // PE name -> list of FU details (name + ops)
  struct FUDetail {
    std::string fuName;
    llvm::SmallVector<std::string, 4> ops;
  };
  llvm::StringMap<llvm::SmallVector<FUDetail, 8>> peFUDetails;
  // Backward compatible: PE name -> list of FU op_name strings
  llvm::StringMap<llvm::SmallVector<std::string, 8>> peFUOps;

  for (const auto &pe : peContainment) {
    if (pe.fuNodeIds.empty())
      continue;
    IdIndex repId = pe.fuNodeIds[0];
    peRepresentative[pe.peName] = repId;
    for (IdIndex fuId : pe.fuNodeIds) {
      fuToPEName[fuId] = llvm::StringRef(pe.peName);
      fuToRepresentative[fuId] = repId;
      // Collect FU details
      const Node *fuNode = adg.getNode(fuId);
      if (fuNode) {
        llvm::StringRef opName = getNodeAttrStr(fuNode, "op_name");
        if (!opName.empty())
          peFUOps[pe.peName].push_back(opName.str());

        FUDetail detail;
        detail.fuName = opName.str();
        // Extract ops from the "ops" attribute
        for (const auto &attr : fuNode->attributes) {
          if (attr.getName().str() == "ops") {
            if (auto arrAttr =
                    mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue())) {
              for (auto elem : arrAttr) {
                if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elem))
                  detail.ops.push_back(strAttr.getValue().str());
              }
            }
            break;
          }
        }
        peFUDetails[pe.peName].push_back(std::move(detail));
      }
    }
  }

  os << "{\n  \"nodes\": [\n";
  bool firstNode = true;

  // Track which PE names have been emitted (to avoid duplicates)
  llvm::StringSet<llvm::MallocAllocator> emittedPEs;

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;

    llvm::StringRef opName = getNodeAttrStr(node, "op_name");
    llvm::StringRef resClass = getNodeAttrStr(node, "resource_class");
    llvm::StringRef peName = getNodeAttrStr(node, "pe_name");

    // Determine node type string (matching loom convention)
    std::string type;
    std::string classStr;
    if (node->kind == Node::ModuleInputNode) {
      type = "input";
      classStr = "boundary";
    } else if (node->kind == Node::ModuleOutputNode) {
      type = "output";
      classStr = "boundary";
    } else if (resClass == "routing") {
      if (opName.contains("fifo"))
        type = "fabric.fifo";
      else
        type = "fabric.spatial_sw";
      classStr = "routing";
    } else if (resClass == "memory") {
      type = "fabric.memory";
      classStr = "memory";
    } else if (resClass == "extmemory") {
      type = "fabric.extmemory";
      classStr = "extmemory";
    } else if (resClass == "fifo") {
      type = "fabric.fifo";
      classStr = "routing";
    } else {
      type = "fabric.spatial_pe";
      classStr = "functional";
    }

    // For FU nodes: only emit the representative (first FU per PE).
    // Skip non-representative FU nodes but include their info in the
    // representative's aggregate.
    if (type == "fabric.spatial_pe" && fuToRepresentative.count(i)) {
      if (fuToRepresentative[i] != i)
        continue; // Skip non-representative FU nodes

      // Emit as PE aggregate node
      auto peNameIt = fuToPEName.find(i);
      std::string peNameStr =
          peNameIt != fuToPEName.end() ? peNameIt->second.str() : opName.str();
      if (emittedPEs.count(peNameStr))
        continue;
      emittedPEs.insert(peNameStr);
    }

    auto gridPos = flattener.getNodeGridPos(i);

    // If the flattener has no position, try parsing from the node name
    if (gridPos.first < 0)
      gridPos = parseGridPosFromName(opName);

    // Compute interleaved grid coordinates (loom convention)
    int gridCol = -1, gridRow = -1;
    if (gridPos.first >= 0) {
      if (type == "fabric.spatial_pe") {
        gridRow = gridPos.first * 2 + 1;
        gridCol = gridPos.second * 2 + 1;
      } else if (type == "fabric.spatial_sw") {
        gridRow = gridPos.first * 2;
        gridCol = gridPos.second * 2;
      } else if (type == "fabric.fifo") {
        gridRow = gridPos.first * 2 + 1;
        gridCol = gridPos.second * 2;
      } else if (type == "fabric.extmemory" || type == "fabric.memory") {
        gridRow = gridPos.first * 2;
        gridCol = gridPos.second * 2;
      } else {
        gridRow = gridPos.first * 2;
        gridCol = gridPos.second * 2;
      }
    }

    if (!firstNode)
      os << ",\n";
    firstNode = false;

    // For PE nodes, use the PE name as the display name
    std::string displayName;
    if (type == "fabric.spatial_pe") {
      auto peNameIt = fuToPEName.find(i);
      displayName =
          peNameIt != fuToPEName.end() ? peNameIt->second.str() : opName.str();
    } else {
      displayName = opName.str();
    }

    os << "    {\"id\": \"hw_" << i << "\"";
    os << ", \"name\": \"" << jsonEscapeStr(displayName) << "\"";
    os << ", \"type\": \"" << type << "\"";
    os << ", \"class\": \"" << classStr << "\"";

    if (gridCol >= 0) {
      os << ", \"gridCol\": " << gridCol;
      os << ", \"gridRow\": " << gridRow;
    } else {
      os << ", \"gridCol\": null, \"gridRow\": null";
    }

    // Port counts
    os << ", \"ports\": {\"in\": " << node->inputPorts.size()
       << ", \"out\": " << node->outputPorts.size() << "}";

    // Port details array
    os << ", \"portDetails\": [";
    bool firstPort = true;
    for (size_t j = 0; j < node->inputPorts.size(); ++j) {
      if (!firstPort)
        os << ", ";
      firstPort = false;
      os << "{\"id\": " << node->inputPorts[j] << ", \"dir\": \"in\", \"idx\": " << j << "}";
    }
    for (size_t j = 0; j < node->outputPorts.size(); ++j) {
      if (!firstPort)
        os << ", ";
      firstPort = false;
      os << "{\"id\": " << node->outputPorts[j] << ", \"dir\": \"out\", \"idx\": " << j << "}";
    }
    os << "]";

    // For PE aggregate nodes: list all FU ops and detailed FU info
    if (type == "fabric.spatial_pe") {
      auto peNameIt = fuToPEName.find(i);
      if (peNameIt != fuToPEName.end()) {
        // Simple list for backward compatibility
        auto opsIt = peFUOps.find(peNameIt->second);
        if (opsIt != peFUOps.end() && !opsIt->second.empty()) {
          os << ", \"fus\": [";
          for (size_t j = 0; j < opsIt->second.size(); ++j) {
            if (j > 0)
              os << ", ";
            os << "\"" << jsonEscapeStr(opsIt->second[j]) << "\"";
          }
          os << "]";
        }
        // Structured FU details for nested PE rendering
        auto detailIt = peFUDetails.find(peNameIt->second);
        if (detailIt != peFUDetails.end() && !detailIt->second.empty()) {
          os << ", \"fuDetails\": [";
          for (size_t j = 0; j < detailIt->second.size(); ++j) {
            if (j > 0)
              os << ", ";
            const auto &fd = detailIt->second[j];
            os << "{\"name\": \"" << jsonEscapeStr(fd.fuName) << "\"";
            if (!fd.ops.empty()) {
              os << ", \"ops\": [";
              for (size_t k = 0; k < fd.ops.size(); ++k) {
                if (k > 0)
                  os << ", ";
                os << "\"" << jsonEscapeStr(fd.ops[k]) << "\"";
              }
              os << "]";
            }
            os << "}";
          }
          os << "]";
        }
      }
    } else if (nodeHasAttr(node, "ops")) {
      for (const auto &attr : node->attributes) {
        if (attr.getName().str() == "ops") {
          if (auto arrAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue())) {
            os << ", \"fus\": [";
            bool firstOp = true;
            for (auto elem : arrAttr) {
              if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elem)) {
                if (!firstOp)
                  os << ", ";
                firstOp = false;
                os << "\"" << jsonEscapeStr(strAttr.getValue()) << "\"";
              }
            }
            os << "]";
          }
          break;
        }
      }
    }

    os << "}";
  }

  os << "\n  ],\n  \"edges\": [\n";

  // Build edges with port IDs and edge type info.
  // Redirect FU node references to their representative PE node.
  // Also deduplicate edges that become identical after FU aggregation.
  llvm::DenseSet<std::pair<IdIndex, IdIndex>> emittedEdgePairs;
  bool firstEdge = true;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(adg.edges.size()); ++eid) {
    const Edge *edge = adg.getEdge(eid);
    if (!edge)
      continue;

    const Port *srcPort = adg.getPort(edge->srcPort);
    const Port *dstPort = adg.getPort(edge->dstPort);
    if (!srcPort || !dstPort)
      continue;

    IdIndex srcNode = srcPort->parentNode;
    IdIndex dstNode = dstPort->parentNode;
    if (srcNode == INVALID_ID || dstNode == INVALID_ID)
      continue;

    // Resolve FU nodes to their representative PE node
    auto srcRep = fuToRepresentative.find(srcNode);
    if (srcRep != fuToRepresentative.end())
      srcNode = srcRep->second;
    auto dstRep = fuToRepresentative.find(dstNode);
    if (dstRep != fuToRepresentative.end())
      dstNode = dstRep->second;

    // Skip self-loops (internal edges within same PE or same node)
    if (srcNode == dstNode)
      continue;

    // Deduplicate edges between same node pair
    auto edgePair = std::make_pair(srcNode, dstNode);
    if (emittedEdgePairs.count(edgePair))
      continue;
    emittedEdgePairs.insert(edgePair);

    if (!firstEdge)
      os << ",\n";
    firstEdge = false;

    // Determine edge type
    std::string edgeType = "data";
    if (srcPort->type) {
      if (mlir::isa<mlir::NoneType>(srcPort->type))
        edgeType = "control";
      else if (mlir::isa<mlir::MemRefType>(srcPort->type))
        edgeType = "memref";
    }

    os << "    {\"id\": \"hwedge_" << eid << "\"";
    os << ", \"srcNode\": \"hw_" << srcNode << "\"";
    os << ", \"dstNode\": \"hw_" << dstNode << "\"";
    os << ", \"srcPort\": \"" << edge->srcPort << "\"";
    os << ", \"dstPort\": \"" << edge->dstPort << "\"";
    os << ", \"edgeType\": \"" << edgeType << "\"";
    os << "}";
  }

  os << "\n  ]\n}";
}

/// Write DFG data as JSON (includes DOT string for Graphviz).
static void writeDFGJSON(llvm::raw_ostream &os, const Graph &dfg) {
  // Build DOT string
  std::string dot;
  {
    llvm::raw_string_ostream ds(dot);
    ds << "digraph DFG {\\n";
    ds << "  rankdir=TB;\\n";
    ds << "  node [style=filled, fontsize=10];\\n";
    ds << "  edge [color=\\\"#333333\\\"];\\n\\n";

    std::string sourceGroup, sinkGroup;

    for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
      const Node *node = dfg.getNode(i);
      if (!node)
        continue;

      llvm::StringRef opName = getNodeAttrStr(node, "op_name");
      std::string nodeId = "sw_" + std::to_string(i);
      std::string label;
      std::string shape = "box";
      std::string fillColor = "#dddddd";

      if (node->kind == Node::ModuleInputNode) {
        int64_t argIdx = getNodeAttrInt(node, "arg_index", -1);
        label = "arg" + std::to_string(argIdx >= 0 ? argIdx
                                                    : static_cast<int64_t>(i));
        shape = "invtriangle";
        fillColor = "#ffb6c1";
      } else if (node->kind == Node::ModuleOutputNode) {
        int64_t retIdx = getNodeAttrInt(node, "result_index", -1);
        label = "ret" + std::to_string(retIdx >= 0 ? retIdx
                                                    : static_cast<int64_t>(i));
        shape = "triangle";
        fillColor = "#f08080";
      } else {
        label = opName.str();
        shape = "ellipse";
        // Color by dialect
        std::string prefix = opName.split(".").first.str();
        if (prefix == "arith")
          fillColor = "#add8e6";
        else if (prefix == "dataflow")
          fillColor = "#90ee90";
        else if (prefix == "handshake")
          fillColor = "#ffffe0";
        else if (prefix == "math")
          fillColor = "#dda0dd";
      }

      ds << "  \\\"" << nodeId << "\\\" [";
      ds << "id=\\\"" << nodeId << "\\\", ";
      ds << "label=\\\"" << dotLabelEscape(label) << "\\\", ";
      ds << "shape=" << shape << ", ";
      ds << "fillcolor=\\\"" << fillColor << "\\\"";
      ds << "];\\n";

      if (node->kind == Node::ModuleInputNode) {
        sourceGroup += "\\\"" + nodeId + "\\\"; ";
      } else if (node->kind == Node::ModuleOutputNode) {
        sinkGroup += "\\\"" + nodeId + "\\\"; ";
      }
    }

    if (!sourceGroup.empty())
      ds << "\\n  { rank=source; " << sourceGroup << "}\\n";
    if (!sinkGroup.empty())
      ds << "  { rank=sink; " << sinkGroup << "}\\n";

    ds << "\\n";
    for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
      const Edge *edge = dfg.getEdge(eid);
      if (!edge)
        continue;

      const Port *srcPort = dfg.getPort(edge->srcPort);
      const Port *dstPort = dfg.getPort(edge->dstPort);
      if (!srcPort || !dstPort)
        continue;

      std::string srcNodeId = "sw_" + std::to_string(srcPort->parentNode);
      std::string dstNodeId = "sw_" + std::to_string(dstPort->parentNode);

      bool isControl =
          srcPort->type && mlir::isa<mlir::NoneType>(srcPort->type);

      ds << "  \\\"" << srcNodeId << "\\\" -> \\\"" << dstNodeId << "\\\" [";
      ds << "id=\\\"e_" << eid << "\\\"";
      if (isControl)
        ds << ", style=dashed, color=\\\"#999999\\\", penwidth=1.0";
      else
        ds << ", penwidth=2.0";
      ds << "];\\n";
    }
    ds << "}\\n";
  }

  // Write JSON with nodes, edges, and DOT string
  os << "{\n  \"dot\": \"" << dot << "\",\n";

  os << "  \"nodes\": [\n";
  bool first = true;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;

    if (!first)
      os << ",\n";
    first = false;

    llvm::StringRef opName = getNodeAttrStr(node, "op_name");
    os << "    {\"id\": " << i;
    os << ", \"opName\": \"" << jsonEscapeStr(opName) << "\"";
    if (node->kind == Node::ModuleInputNode)
      os << ", \"kind\": \"input\"";
    else if (node->kind == Node::ModuleOutputNode)
      os << ", \"kind\": \"output\"";
    else
      os << ", \"kind\": \"op\"";
    os << "}";
  }
  os << "\n  ],\n";

  os << "  \"edges\": [\n";
  first = true;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *edge = dfg.getEdge(eid);
    if (!edge)
      continue;

    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort)
      continue;

    if (!first)
      os << ",\n";
    first = false;

    os << "    {\"id\": " << eid;
    os << ", \"srcNode\": " << srcPort->parentNode;
    os << ", \"dstNode\": " << dstPort->parentNode;
    os << "}";
  }
  os << "\n  ]\n}";
}

/// Write mapping data as JSON.
/// Uses loom-style swToHw/hwToSw maps plus routes with hw edge references.
static void writeMappingJSON(llvm::raw_ostream &os, const Graph &dfg,
                             const Graph &adg, const MappingState &state,
                             const ADGFlattener &flattener) {
  // Build FU-to-representative map for redirecting mapping targets
  auto &peContainment = flattener.getPEContainment();
  llvm::DenseMap<IdIndex, IdIndex> fuToRep;
  for (const auto &pe : peContainment) {
    if (pe.fuNodeIds.empty())
      continue;
    IdIndex repId = pe.fuNodeIds[0];
    for (IdIndex fuId : pe.fuNodeIds)
      fuToRep[fuId] = repId;
  }

  auto resolveHw = [&](IdIndex hwId) -> IdIndex {
    auto it = fuToRep.find(hwId);
    return it != fuToRep.end() ? it->second : hwId;
  };

  os << "{\n";

  // swToHw map (resolve FU -> PE representative)
  os << "  \"swToHw\": {";
  bool first = true;
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    if (swId >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[swId] == INVALID_ID)
      continue;
    if (!first)
      os << ",";
    first = false;
    os << "\n    \"sw_" << swId << "\": \"hw_"
       << resolveHw(state.swNodeToHwNode[swId]) << "\"";
  }
  os << "\n  },\n";

  // hwToSw map (aggregate by PE representative)
  os << "  \"hwToSw\": {";
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> hwToSwMap;
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    if (swId >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[swId] == INVALID_ID)
      continue;
    hwToSwMap[resolveHw(state.swNodeToHwNode[swId])].push_back(swId);
  }
  first = true;
  for (auto &kv : hwToSwMap) {
    if (!first)
      os << ",";
    first = false;
    os << "\n    \"hw_" << kv.first << "\": [";
    for (size_t j = 0; j < kv.second.size(); ++j) {
      if (j > 0)
        os << ", ";
      os << "\"sw_" << kv.second[j] << "\"";
    }
    os << "]";
  }
  os << "\n  },\n";

  // placements (for backward compatibility)
  os << "  \"placements\": [\n";
  first = true;
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;
    if (swId >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[swId] == INVALID_ID)
      continue;

    if (!first)
      os << ",\n";
    first = false;

    IdIndex rawHwId = state.swNodeToHwNode[swId];
    IdIndex hwId = resolveHw(rawHwId);
    const Node *hwNode = adg.getNode(hwId);
    // Use the ORIGINAL mapped FU node (before resolving to PE representative)
    // to get the correct FU name for highlighting
    const Node *rawHwNode = adg.getNode(rawHwId);

    os << "    {\"dfgNode\": " << swId << ", \"adgNode\": \"hw_" << hwId
       << "\"";
    os << ", \"swOp\": \"" << jsonEscapeStr(getNodeAttrStr(swNode, "op_name"))
       << "\"";
    if (rawHwNode) {
      // hwName = the actual FU name (e.g., "fu_mux"), not the PE representative
      os << ", \"hwName\": \""
         << jsonEscapeStr(getNodeAttrStr(rawHwNode, "op_name")) << "\"";
    }
    if (hwNode) {
      llvm::StringRef pe = getNodeAttrStr(hwNode, "pe_name");
      if (!pe.empty())
        os << ", \"peName\": \"" << jsonEscapeStr(pe) << "\"";
    }
    os << "}";
  }

  os << "\n  ],\n  \"routes\": [\n";

  // Edge routing paths. Convert port-level paths to node-level paths.
  first = true;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *edge = dfg.getEdge(eid);
    if (!edge)
      continue;
    if (eid >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[eid].empty())
      continue;

    auto &portPath = state.swEdgeToHwPaths[eid];
    // Skip synthetic direct-binding paths
    if (portPath.size() == 2 && portPath[0] == portPath[1])
      continue;

    if (!first)
      os << ",\n";
    first = false;

    // Convert port IDs to node IDs (dedup consecutive same-node entries)
    os << "    {\"dfgEdge\": " << eid << ", \"path\": [";
    IdIndex prevNode = INVALID_ID;
    bool firstPath = true;
    for (IdIndex portId : portPath) {
      const Port *port = adg.getPort(portId);
      if (!port)
        continue;
      IdIndex nodeId = port->parentNode;
      if (nodeId == INVALID_ID || nodeId == prevNode)
        continue;
      if (!firstPath)
        os << ", ";
      firstPath = false;
      os << "\"hw_" << nodeId << "\"";
      prevNode = nodeId;
    }
    os << "]}";
  }

  os << "\n  ],\n";

  // SW internal routing connections: for each routing node, record which
  // input port index connects to which output port index.
  llvm::DenseMap<IdIndex, llvm::SmallVector<std::pair<int, int>, 8>>
      swInternalRoutes;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++eid) {
    auto &portPath = state.swEdgeToHwPaths[eid];
    for (size_t pi = 0; pi + 1 < portPath.size(); pi++) {
      const Port *p1 = adg.getPort(portPath[pi]);
      const Port *p2 = adg.getPort(portPath[pi + 1]);
      if (!p1 || !p2)
        continue;
      if (p1->parentNode != p2->parentNode)
        continue;
      // Same node - this is an internal routing connection
      IdIndex nodeId = p1->parentNode;
      const Node *node = adg.getNode(nodeId);
      if (!node)
        continue;
      llvm::StringRef resClass = getNodeAttrStr(node, "resource_class");
      if (resClass != "routing")
        continue;
      // Find port indices within the node
      int inIdx = -1, outIdx = -1;
      for (size_t j = 0; j < node->inputPorts.size(); j++) {
        if (node->inputPorts[j] == portPath[pi])
          inIdx = j;
        if (node->inputPorts[j] == portPath[pi + 1])
          inIdx = j;
      }
      for (size_t j = 0; j < node->outputPorts.size(); j++) {
        if (node->outputPorts[j] == portPath[pi])
          outIdx = j;
        if (node->outputPorts[j] == portPath[pi + 1])
          outIdx = j;
      }
      IdIndex resolvedId = resolveHw(nodeId);
      if (inIdx >= 0 && outIdx >= 0)
        swInternalRoutes[resolvedId].push_back({inIdx, outIdx});
    }
  }

  os << "  \"swRoutes\": {";
  first = true;
  for (auto &kv : swInternalRoutes) {
    if (!first)
      os << ",";
    first = false;
    os << "\n    \"hw_" << kv.first << "\": [";
    for (size_t j = 0; j < kv.second.size(); ++j) {
      if (j > 0)
        os << ", ";
      os << "[" << kv.second[j].first << ", " << kv.second[j].second << "]";
    }
    os << "]";
  }
  os << "\n  }\n}";
}

/// Write trace data as JSON.
static void writeTraceJSON(llvm::raw_ostream &os,
                           const std::vector<TraceEvent> &trace) {
  if (trace.empty()) {
    os << "null";
    return;
  }

  uint64_t maxCycle = 0;
  for (auto &ev : trace) {
    if (ev.cycle > maxCycle)
      maxCycle = ev.cycle;
  }

  os << "{\n  \"totalCycles\": " << maxCycle << ",\n";
  os << "  \"events\": [\n";
  bool first = true;
  for (auto &ev : trace) {
    if (!first)
      os << ",\n";
    first = false;
    os << "    {\"cycle\": " << ev.cycle;
    os << ", \"hwNode\": \"hw_" << ev.hwNodeId << "\"";
    os << ", \"kind\": " << static_cast<int>(ev.kind) << "}";
  }
  os << "\n  ]\n}";
}

// ---- Public interface ----

mlir::LogicalResult
exportVisualization(const std::string &outputPath, const Graph &adg,
                    const Graph &dfg, const MappingState &mapping,
                    const ADGFlattener &flattener,
                    const std::vector<TraceEvent> &trace) {
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "fcc viz: cannot open " << outputPath << ": "
                 << ec.message() << "\n";
    return mlir::failure();
  }

  llvm::StringRef baseFile = llvm::sys::path::filename(outputPath);
  std::string title = "fcc: " + baseFile.str();

  // Build JSON data strings
  std::string adgJson;
  {
    llvm::raw_string_ostream ss(adgJson);
    writeADGJSON(ss, adg, flattener);
  }

  std::string dfgJson;
  {
    llvm::raw_string_ostream ss(dfgJson);
    writeDFGJSON(ss, dfg);
  }

  std::string mappingJson;
  {
    llvm::raw_string_ostream ss(mappingJson);
    writeMappingJSON(ss, dfg, adg, mapping, flattener);
  }

  std::string traceJson;
  {
    llvm::raw_string_ostream ss(traceJson);
    writeTraceJSON(ss, trace);
  }

  bool hasTrace = !trace.empty();

  // Emit HTML
  out << "<!DOCTYPE html>\n<html>\n<head>\n"
      << "  <meta charset=\"UTF-8\">\n"
      << "  <title>" << htmlEscape(title) << "</title>\n"
      << "  <style>\n"
      << viz::RENDERER_CSS << "\n"
      << "  </style>\n"
      << "</head>\n<body>\n\n";

  // Toolbar
  out << "<div id=\"toolbar\">\n"
      << "  <span id=\"title\">" << htmlEscape(title) << "</span>\n"
      << "  <div id=\"mode-buttons\">\n"
      << "    <button id=\"btn-sidebyside\" class=\"active\">Side-by-Side"
         "</button>\n"
      << "    <button id=\"btn-overlay\">Overlay</button>\n"
      << "  </div>\n"
      << "  <button id=\"btn-fit\">Fit</button>\n"
      << "  <button id=\"btn-restore\" style=\"display:none\">Restore"
         "</button>\n"
      << "  <span id=\"status-bar\">Loading...</span>\n"
      << "</div>\n\n";

  // Graph panels
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

  // Trace toolbar (shown when trace data is available)
  if (hasTrace) {
    out << "<div id=\"trace-toolbar\">\n"
        << "  <button id=\"trace-step-back\" title=\"Step back\">&lt;</button>\n"
        << "  <button id=\"trace-play\" title=\"Play/Pause\">Play</button>\n"
        << "  <button id=\"trace-step-fwd\" title=\"Step "
           "forward\">&gt;</button>\n"
        << "  <input id=\"trace-slider\" type=\"range\" min=\"0\" max=\"0\" "
           "value=\"0\">\n"
        << "  <span id=\"trace-cycle\">Cycle: 0 / 0</span>\n"
        << "  <label>Speed: <select id=\"trace-speed\">\n"
        << "    <option value=\"1\">1x</option>\n"
        << "    <option value=\"5\" selected>5x</option>\n"
        << "    <option value=\"20\">20x</option>\n"
        << "    <option value=\"100\">100x</option>\n"
        << "  </select></label>\n"
        << "</div>\n\n";
  }

  // Embedded data
  out << "<script>\n"
      << "const adgGraph = " << scriptSafe(adgJson) << ";\n\n"
      << "const DFG_DATA = " << scriptSafe(dfgJson) << ";\n\n"
      << "const mappingData = " << scriptSafe(mappingJson) << ";\n\n"
      << "const TRACE = " << scriptSafe(traceJson) << ";\n"
      << "</script>\n\n";

  // D3.js from CDN
  out << "<script "
         "src=\"https://d3js.org/d3.v7.min.js\"></script>\n";

  // Graphviz WASM from CDN
  out << "<script "
         "src=\"https://unpkg.com/@viz-js/"
         "viz@3.2.4/lib/viz-standalone.js\"></script>\n\n";

  // Renderer JS (embedded)
  out << "<script>\n" << viz::RENDERER_JS << "\n</script>\n\n";

  out << "</body>\n</html>\n";

  return mlir::success();
}

} // namespace fcc

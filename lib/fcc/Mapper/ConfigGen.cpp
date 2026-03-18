// ConfigGen.cpp -- ConfigGen::writeMapJson and ConfigGen::writeMapText.

#include "ConfigGenInternal.h"

#include "fcc/Mapper/ConfigGen.h"
#include "fcc/Mapper/BridgeBinding.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace fcc {

// ===========================================================================
// ConfigGen::writeMapJson
// ===========================================================================

bool ConfigGen::writeMapJson(const MappingState &state, const Graph &dfg,
                             const Graph &adg, const ADGFlattener &flattener,
                             llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                             llvm::ArrayRef<FUConfigSelection> fuConfigs,
                             const std::string &path, int seed) {
  using namespace configgen_detail;

  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "ConfigGen: cannot open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }

  out << "{\n";
  out << "  \"seed\": " << seed << ",\n";
  out << "  \"node_mappings\": [\n";

  bool first = true;
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;
    if (swId >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[swId] == INVALID_ID)
      continue;

    if (!first)
      out << ",\n";
    first = false;

    IdIndex hwId = state.swNodeToHwNode[swId];
    const Node *hwNode = adg.getNode(hwId);

    out << "    {\"sw_node\": " << swId << ", \"hw_node\": " << hwId;
    out << ", \"sw_op\": \"" << getNodeAttrStr(swNode, "op_name") << "\"";
    if (hwNode) {
      out << ", \"hw_name\": \"" << getNodeAttrStr(hwNode, "op_name") << "\"";
      out << ", \"pe_name\": \"" << getNodeAttrStr(hwNode, "pe_name") << "\"";
    }
    out << "}";
  }

  out << "\n  ],\n";

  // Edge routing.
  out << "  \"edge_routings\": [\n";
  first = true;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *edge = dfg.getEdge(eid);
    if (!edge)
      continue;

    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"sw_edge\": " << eid;
    if (eid < edgeKinds.size() && edgeKinds[eid] == TechMappedEdgeKind::IntraFU) {
      IdIndex hwNodeId = INVALID_ID;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      if (srcPort && srcPort->parentNode != INVALID_ID &&
          srcPort->parentNode < state.swNodeToHwNode.size()) {
        hwNodeId = state.swNodeToHwNode[srcPort->parentNode];
      }
      out << ", \"kind\": \"intra_fu\"";
      if (hwNodeId != INVALID_ID)
        out << ", \"hw_node\": " << hwNodeId;
      out << ", \"path\": []}";
      continue;
    }
    if (eid < edgeKinds.size() &&
        edgeKinds[eid] == TechMappedEdgeKind::TemporalReg) {
      IdIndex hwNodeId = INVALID_ID;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      if (srcPort && srcPort->parentNode != INVALID_ID &&
          srcPort->parentNode < state.swNodeToHwNode.size()) {
        hwNodeId = state.swNodeToHwNode[srcPort->parentNode];
      }
      out << ", \"kind\": \"temporal_reg\"";
      if (hwNodeId != INVALID_ID)
        out << ", \"hw_node\": " << hwNodeId;
      out << ", \"path\": []}";
      continue;
    }

    if (eid >= state.swEdgeToHwPaths.size() || state.swEdgeToHwPaths[eid].empty()) {
      out << ", \"kind\": \"unrouted\", \"path\": []}";
      continue;
    }

    auto exportPath = buildExportPathForEdge(eid, state, dfg, adg);
    out << ", \"kind\": \"routed\", \"path\": [";
    for (size_t i = 0; i < exportPath.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << exportPath[i];
    }
    out << "]}";
  }

  out << "\n  ],\n";

  // Port table: maps flat port IDs to viz-compatible component names.
  struct PortVizInfo {
    bool valid = false;
    std::string kind;
    std::string component;
    std::string pe;
    std::string fu;
    int index = -1;
    std::string dir;
  };

  std::vector<PortVizInfo> portInfo(adg.ports.size());
  out << "  \"port_table\": [\n";
  first = true;
  for (IdIndex pid = 0; pid < static_cast<IdIndex>(adg.ports.size()); ++pid) {
    const Port *p = adg.getPort(pid);
    if (!p)
      continue;
    const Node *pn = adg.getNode(p->parentNode);
    if (!pn)
      continue;

    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"id\": " << pid;
    portInfo[pid].valid = true;
    portInfo[pid].dir =
        (p->direction == Port::Input ? std::string("in") : std::string("out"));

    if (pn->kind == Node::ModuleInputNode) {
      int argIdx = getNodeAttrInt(pn, "arg_index");
      portInfo[pid].kind = "module_in";
      portInfo[pid].component = "module_in";
      portInfo[pid].index = argIdx;
      out << ", \"kind\": \"module_in\", \"index\": " << argIdx;
    } else if (pn->kind == Node::ModuleOutputNode) {
      int resIdx = getNodeAttrInt(pn, "result_index");
      portInfo[pid].kind = "module_out";
      portInfo[pid].component = "module_out";
      portInfo[pid].index = resIdx;
      out << ", \"kind\": \"module_out\", \"index\": " << resIdx;
    } else {
      llvm::StringRef resClass = getNodeAttrStr(pn, "resource_class");
      if (resClass == "routing") {
        llvm::StringRef opName = getNodeAttrStr(pn, "op_name");
        llvm::StringRef opKind = getNodeAttrStr(pn, "op_kind");
        bool isSwitchLike =
            pn->inputPorts.size() > 1 || pn->outputPorts.size() > 1;
        if (opKind == "temporal_sw")
          portInfo[pid].kind = "temporal_sw";
        else if (opKind == "spatial_sw")
          portInfo[pid].kind = "sw";
        else if (opKind == "add_tag" || opKind == "del_tag" ||
                 opKind == "map_tag")
          portInfo[pid].kind = opKind.str();
        else
          portInfo[pid].kind = isSwitchLike ? "sw" : "fifo";
        portInfo[pid].component = opName.str();
        out << ", \"kind\": \"" << portInfo[pid].kind << "\", \"name\": \""
            << opName << "\"";
      } else if (resClass == "memory") {
        llvm::StringRef opName = getNodeAttrStr(pn, "op_name");
        portInfo[pid].kind = "memory";
        portInfo[pid].component = opName.str();
        out << ", \"kind\": \"memory\", \"name\": \"" << opName << "\"";
      } else {
        llvm::StringRef peName = getNodeAttrStr(pn, "pe_name");
        llvm::StringRef opName = getNodeAttrStr(pn, "op_name");
        portInfo[pid].kind = "fu";
        portInfo[pid].component = peName.str();
        portInfo[pid].pe = peName.str();
        portInfo[pid].fu = opName.str();
        out << ", \"kind\": \"fu\", \"pe\": \"" << peName << "\", \"fu\": \""
            << opName << "\"";
      }
      // Port index within the node.
      int portIdx = -1;
      if (p->direction == Port::Input) {
        for (unsigned i = 0; i < pn->inputPorts.size(); ++i) {
          if (pn->inputPorts[i] == pid) {
            portIdx = static_cast<int>(i);
            break;
          }
        }
      } else {
        for (unsigned i = 0; i < pn->outputPorts.size(); ++i) {
          if (pn->outputPorts[i] == pid) {
            portIdx = static_cast<int>(i);
            break;
          }
        }
      }
      portInfo[pid].index = portIdx;
      out << ", \"index\": " << portIdx;
    }

    out << ", \"dir\": \""
        << (p->direction == Port::Input ? "in" : "out") << "\"}";
  }
  out << "\n  ],\n";

  out << "  \"switch_routes\": [\n";
  using SwitchRouteKey = std::tuple<std::string, int, int>;
  struct SwitchRouteEntry {
    std::string component;
    IdIndex inputPortId = INVALID_ID;
    IdIndex outputPortId = INVALID_ID;
    int inputPort = -1;
    int outputPort = -1;
    std::vector<IdIndex> swEdges;
  };
  std::map<SwitchRouteKey, SwitchRouteEntry> switchRoutes;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++eid) {
    auto hwPath = buildExportPathForEdge(eid, state, dfg, adg);
    if (hwPath.size() < 3)
      continue;
    for (size_t i = 1; i + 1 < hwPath.size(); i += 2) {
      IdIndex inPortId = hwPath[i];
      IdIndex outPortId = hwPath[i + 1];
      if (inPortId >= portInfo.size() || outPortId >= portInfo.size())
        continue;
      const auto &inInfo = portInfo[inPortId];
      const auto &outInfo = portInfo[outPortId];
      if (!inInfo.valid || !outInfo.valid)
        continue;
      bool inSwitch =
          (inInfo.kind == "sw" || inInfo.kind == "temporal_sw");
      bool outSwitch =
          (outInfo.kind == "sw" || outInfo.kind == "temporal_sw");
      if (!inSwitch || !outSwitch)
        continue;
      if (inInfo.component != outInfo.component)
        continue;

      auto key =
          std::make_tuple(inInfo.component, inInfo.index, outInfo.index);
      auto &entry = switchRoutes[key];
      if (entry.component.empty()) {
        entry.component = inInfo.component;
        entry.inputPortId = inPortId;
        entry.outputPortId = outPortId;
        entry.inputPort = inInfo.index;
        entry.outputPort = outInfo.index;
      }
      entry.swEdges.push_back(eid);
    }
  }

  first = true;
  for (const auto &it : switchRoutes) {
    const auto &entry = it.second;
    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"component\": \"" << entry.component << "\"";
    out << ", \"input_port_id\": " << entry.inputPortId;
    out << ", \"output_port_id\": " << entry.outputPortId;
    out << ", \"input_port\": " << entry.inputPort;
    out << ", \"output_port\": " << entry.outputPort;
    out << ", \"sw_edges\": [";
    for (size_t i = 0; i < entry.swEdges.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << entry.swEdges[i];
    }
    out << "]}";
  }

  out << "\n  ],\n";

  out << "  \"fu_configs\": [\n";
  first = true;
  for (const auto &selection : fuConfigs) {
    if (!first)
      out << ",\n";
    first = false;
    out << "    {\"hw_node\": " << selection.hwNodeId;
    out << ", \"hw_name\": \"" << selection.hwName << "\"";
    out << ", \"pe_name\": \"" << selection.peName << "\"";
    out << ", \"sw_nodes\": [";
    for (size_t i = 0; i < selection.swNodeIds.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << selection.swNodeIds[i];
    }
    out << "], \"fields\": [";
    for (size_t i = 0; i < selection.fields.size(); ++i) {
      if (i > 0)
        out << ", ";
      const auto &field = selection.fields[i];
      out << "{\"op_index\": " << field.opIndex;
      out << ", \"op_name\": \"" << field.opName << "\"";
      out << ", \"kind\": \"" << configFieldKindName(field.kind) << "\"";
      out << ", \"bit_width\": " << field.bitWidth;
      out << ", \"value\": " << field.value;
      out << ", \"display\": \"" << formatConfigFieldValue(field) << "\"";
      if (field.kind == FUConfigFieldKind::Mux) {
        out << ", \"sel\": " << field.sel;
        out << ", \"discard\": " << (field.discard ? "true" : "false");
        out << ", \"disconnect\": "
            << (field.disconnect ? "true" : "false");
      }
      out << "}";
    }
    out << "]}";
  }
  out << "\n  ],\n";

  out << "  \"tag_configs\": [\n";
  first = true;
  for (const auto &slice : configSlices_) {
    if (slice.kind != "add_tag" && slice.kind != "map_tag")
      continue;
    const Node *hwNode =
        slice.hwNode == INVALID_ID ? nullptr : adg.getNode(slice.hwNode);
    if (!first)
      out << ",\n";
    first = false;
    out << "    {\"name\": \"" << slice.name << "\", \"kind\": \""
        << slice.kind << "\", \"complete\": "
        << (slice.complete ? "true" : "false");
    if (slice.kind == "add_tag") {
      out << ", \"tag\": " << getNodeAttrInt(hwNode, "tag", 0);
    } else {
      out << ", \"table_size\": " << getNodeAttrInt(hwNode, "table_size", 0);
      auto [inTagWidth, outTagWidth] = getMapTagTagWidths(hwNode, adg);
      out << ", \"input_tag_width\": " << inTagWidth;
      out << ", \"output_tag_width\": " << outTagWidth;
      out << ", \"table\": [";
      bool firstElem = true;
      for (const auto &entry : getMapTagTableEntries(hwNode)) {
        if (!firstElem)
          out << ", ";
        firstElem = false;
        out << "{\"valid\": " << (entry.valid ? "true" : "false")
            << ", \"src_tag\": " << entry.srcTag
            << ", \"dst_tag\": " << entry.dstTag << "}";
      }
      out << "]";
    }
    out << "}";
  }
  out << "\n  ],\n";

  out << "  \"fifo_configs\": [\n";
  first = true;
  for (const auto &slice : configSlices_) {
    if (slice.kind != "fifo")
      continue;
    const Node *hwNode =
        slice.hwNode == INVALID_ID ? nullptr : adg.getNode(slice.hwNode);
    bool bypassable = false;
    bool bypassed = false;
    for (const auto &attr : hwNode->attributes) {
      if (attr.getName() == "bypassable") {
        if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
          bypassable = boolAttr.getValue();
      } else if (attr.getName() == "bypassed") {
        if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
          bypassed = boolAttr.getValue();
      }
    }
    if (!first)
      out << ",\n";
    first = false;
    out << "    {\"name\": \"" << slice.name << "\", \"complete\": "
        << (slice.complete ? "true" : "false");
    out << ", \"bypassable\": " << (bypassable ? "true" : "false");
    out << ", \"bypassed\": " << (bypassed ? "true" : "false");
    out << "}";
  }
  out << "\n  ],\n";

  out << "  \"temporal_registers\": [\n";
  first = true;
  for (const auto &pe : flattener.getPEContainment()) {
    if (pe.peKind != "temporal_pe")
      continue;
    auto temporalPlan = buildTemporalConfigPlan(pe, state, dfg, adg, edgeKinds);
    for (const auto &binding : temporalPlan.registerBindings) {
      if (!first)
        out << ",\n";
      first = false;
      out << "    {\"pe_name\": \"" << binding.peName << "\"";
      out << ", \"sw_edge\": " << binding.swEdgeId;
      out << ", \"register_index\": " << binding.registerIndex;
      out << ", \"writer_sw_node\": " << binding.writerSwNode;
      out << ", \"reader_sw_node\": " << binding.readerSwNode;
      out << ", \"writer_hw_node\": " << binding.writerHwNode;
      out << ", \"reader_hw_node\": " << binding.readerHwNode;
      out << ", \"writer_output_index\": " << binding.writerOutputIndex;
      out << ", \"reader_input_index\": " << binding.readerInputIndex << "}";
    }
  }
  out << "\n  ],\n";

  out << "  \"memory_regions\": [\n";
  first = true;
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;

    if (!first)
      out << ",\n";
    first = false;

    auto regions = collectMemoryRegionsForNode(hwId, state, dfg, adg);
    auto addrTable = buildAddrOffsetTable(hwNode, hwId, state, dfg, adg);

    out << "    {\"hw_node\": " << hwId;
    out << ", \"hw_name\": \"" << getNodeAttrStr(hwNode, "op_name") << "\"";
    out << ", \"memory_kind\": \"" << getNodeAttrStr(hwNode, "op_kind")
        << "\"";
    out << ", \"num_region\": " << getNodeAttrInt(hwNode, "numRegion", 1);
    out << ", \"regions\": [";
    for (size_t i = 0; i < regions.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << "{\"region_index\": " << i;
      out << ", \"sw_node\": " << regions[i].swNode;
      out << ", \"memref_arg_index\": " << regions[i].memrefArgIndex;
      out << ", \"start_lane\": " << regions[i].startLane;
      out << ", \"end_lane\": " << regions[i].endLane;
      out << ", \"ld_count\": " << regions[i].ldCount;
      out << ", \"st_count\": " << regions[i].stCount;
      out << ", \"elem_size_log2\": " << regions[i].elemSizeLog2 << "}";
    }
    out << "], \"addr_offset_table\": [";
    for (size_t i = 0; i < addrTable.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << addrTable[i];
    }
    out << "]}";
  }

  out << "\n  ]\n";
  out << "}\n";

  return true;
}

// ===========================================================================
// ConfigGen::writeMapText
// ===========================================================================

bool ConfigGen::writeMapText(const MappingState &state, const Graph &dfg,
                             const Graph &adg, const ADGFlattener &flattener,
                             llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                             const std::string &path) {
  using namespace configgen_detail;

  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "ConfigGen: cannot open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }

  out << "=== Mapping Report ===\n\n";

  // Node placements.
  out << "--- Node Placements ---\n";
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;

    llvm::StringRef opName = getNodeAttrStr(swNode, "op_name");
    out << "DFG[" << swId << "] " << opName;

    if (swNode->kind == Node::ModuleInputNode) {
      out << " (input sentinel)";
    } else if (swNode->kind == Node::ModuleOutputNode) {
      out << " (output sentinel)";
    }

    if (swId < state.swNodeToHwNode.size() &&
        state.swNodeToHwNode[swId] != INVALID_ID) {
      IdIndex hwId = state.swNodeToHwNode[swId];
      const Node *hwNode = adg.getNode(hwId);
      out << " -> ADG[" << hwId << "]";
      if (hwNode) {
        out << " " << getNodeAttrStr(hwNode, "op_name");
        llvm::StringRef pe = getNodeAttrStr(hwNode, "pe_name");
        if (!pe.empty())
          out << " (PE: " << pe << ")";
      }
    } else {
      // Check if this is a memref sentinel (direct binding to extmemory,
      // not mapped to an ADG sentinel node).
      bool isMemrefSentinel = false;
      if (swNode->kind == Node::ModuleInputNode &&
          !swNode->outputPorts.empty()) {
        const Port *p = dfg.getPort(swNode->outputPorts[0]);
        if (p && mlir::isa<mlir::MemRefType>(p->type))
          isMemrefSentinel = true;
      }
      if (isMemrefSentinel)
        out << " -> MEMORY_INTERFACE_BINDING";
      else
        out << " -> UNMAPPED";
    }
    out << "\n";
  }

  // Edge routings.
  out << "\n--- Edge Routings ---\n";
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *edge = dfg.getEdge(eid);
    if (!edge)
      continue;

    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);

    out << "Edge[" << eid << "] ";
    if (srcPort && srcPort->parentNode != INVALID_ID) {
      out << "node " << srcPort->parentNode << " -> ";
    }
    if (dstPort && dstPort->parentNode != INVALID_ID) {
      out << "node " << dstPort->parentNode;
    }

    if (eid < edgeKinds.size() && edgeKinds[eid] == TechMappedEdgeKind::IntraFU) {
      IdIndex hwNodeId = INVALID_ID;
      if (srcPort && srcPort->parentNode != INVALID_ID &&
          srcPort->parentNode < state.swNodeToHwNode.size())
        hwNodeId = state.swNodeToHwNode[srcPort->parentNode];
      out << " : INTRA_FU";
      if (hwNodeId != INVALID_ID)
        out << " (hw_node " << hwNodeId << ")";
    } else if (eid < edgeKinds.size() &&
               edgeKinds[eid] == TechMappedEdgeKind::TemporalReg) {
      out << " : TEMPORAL_REG";
    } else if (eid < state.swEdgeToHwPaths.size() &&
               !state.swEdgeToHwPaths[eid].empty()) {
      auto path = buildExportPathForEdge(eid, state, dfg, adg);
      // Detect synthetic memref-binding path (same port for src and dst).
      if (path.size() == 2 && path[0] == path[1]) {
        out << " : MEMORY_INTERFACE_BINDING";
      } else {
        out << " : [";
        for (size_t i = 0; i < path.size(); ++i) {
          if (i > 0)
            out << ", ";
          out << path[i];
        }
        out << "]";
      }
    } else {
      out << " : UNROUTED";
    }
    out << "\n";
  }

  out << "\n--- Temporal Registers ---\n";
  bool emittedTemporalReg = false;
  for (const auto &pe : flattener.getPEContainment()) {
    if (pe.peKind != "temporal_pe")
      continue;
    auto temporalPlan = buildTemporalConfigPlan(pe, state, dfg, adg, edgeKinds);
    for (const auto &binding : temporalPlan.registerBindings) {
      emittedTemporalReg = true;
      out << pe.peName << " reg[" << binding.registerIndex << "] <- edge["
          << binding.swEdgeId << "] writer_sw=" << binding.writerSwNode
          << " reader_sw=" << binding.readerSwNode << " writer_out="
          << binding.writerOutputIndex << " reader_in="
          << binding.readerInputIndex << "\n";
    }
  }
  if (!emittedTemporalReg)
    out << "(none)\n";

  // PE utilization summary.
  out << "\n--- PE Utilization ---\n";
  auto &peContainment = flattener.getPEContainment();
  for (auto &pe : peContainment) {
    bool used = false;
    for (IdIndex fuId : pe.fuNodeIds) {
      if (fuId < state.hwNodeToSwNodes.size() &&
          !state.hwNodeToSwNodes[fuId].empty()) {
        used = true;
        out << pe.peName << " (" << pe.row << "," << pe.col << "): "
            << getNodeAttrStr(adg.getNode(fuId), "op_name");
        out << " <- [";
        for (size_t i = 0; i < state.hwNodeToSwNodes[fuId].size(); ++i) {
          if (i > 0)
            out << ", ";
          const Node *swNode = dfg.getNode(state.hwNodeToSwNodes[fuId][i]);
          if (swNode)
            out << getNodeAttrStr(swNode, "op_name");
          else
            out << state.hwNodeToSwNodes[fuId][i];
        }
        out << "]";
        out << "\n";
      }
    }
    if (!used) {
      out << pe.peName << " (" << pe.row << "," << pe.col << "): unused\n";
    }
  }

  out << "\n--- Memory Regions ---\n";
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;

    out << "ADG[" << hwId << "] " << getNodeAttrStr(hwNode, "op_name")
        << " (" << getNodeAttrStr(hwNode, "op_kind") << ")";
    out << " numRegion=" << getNodeAttrInt(hwNode, "numRegion", 1) << "\n";

    auto regions = collectMemoryRegionsForNode(hwId, state, dfg, adg);
    auto addrTable = buildAddrOffsetTable(hwNode, hwId, state, dfg, adg);
    for (size_t i = 0; i < regions.size(); ++i) {
      out << "  region[" << i << "] <- DFG[" << regions[i].swNode << "]";
      out << " memref_arg=" << regions[i].memrefArgIndex;
      out << " lane_range=[" << regions[i].startLane << ", "
          << regions[i].endLane << ")";
      out << " ld=" << regions[i].ldCount;
      out << " st=" << regions[i].stCount;
      out << " elem_size_log2=" << regions[i].elemSizeLog2 << "\n";
    }
    out << "  addr_offset_table = [";
    for (size_t i = 0; i < addrTable.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << addrTable[i];
    }
    out << "]\n";
  }

  return true;
}

} // namespace fcc

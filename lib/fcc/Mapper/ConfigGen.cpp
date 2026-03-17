#include "fcc/Mapper/ConfigGen.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <tuple>

namespace fcc {

bool ConfigGen::generate(const MappingState &state, const Graph &dfg,
                          const Graph &adg, const ADGFlattener &flattener,
                          const std::string &basePath, int seed) {
  if (!writeMapJson(state, dfg, adg, flattener, basePath + ".map.json", seed))
    return false;
  if (!writeMapText(state, dfg, adg, flattener, basePath + ".map.txt"))
    return false;
  return true;
}

bool ConfigGen::writeMapJson(const MappingState &state, const Graph &dfg,
                              const Graph &adg, const ADGFlattener &flattener,
                              const std::string &path, int seed) {
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
    if (eid >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[eid].empty())
      continue;

    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"sw_edge\": " << eid << ", \"path\": [";
    for (size_t i = 0; i < state.swEdgeToHwPaths[eid].size(); ++i) {
      if (i > 0)
        out << ", ";
      out << state.swEdgeToHwPaths[eid][i];
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
        bool isSwitchLike = pn->inputPorts.size() > 1 || pn->outputPorts.size() > 1;
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
    const auto &hwPath = state.swEdgeToHwPaths[eid];
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
      if (inInfo.kind != "sw" || outInfo.kind != "sw")
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
  out << "\n  ]\n";
  out << "}\n";

  return true;
}

bool ConfigGen::writeMapText(const MappingState &state, const Graph &dfg,
                              const Graph &adg, const ADGFlattener &flattener,
                              const std::string &path) {
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
        out << " -> DIRECT_BINDING (memref -> extmemory)";
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

    if (eid < state.swEdgeToHwPaths.size() &&
        !state.swEdgeToHwPaths[eid].empty()) {
      auto &path = state.swEdgeToHwPaths[eid];
      // Detect synthetic memref-binding path (same port for src and dst).
      if (path.size() == 2 && path[0] == path[1]) {
        out << " : DIRECT_BINDING";
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

  // PE utilization summary.
  out << "\n--- PE Utilization ---\n";
  auto &peContainment = flattener.getPEContainment();
  for (auto &pe : peContainment) {
    bool used = false;
    for (IdIndex fuId : pe.fuNodeIds) {
      if (fuId < state.hwNodeToSwNodes.size() &&
          !state.hwNodeToSwNodes[fuId].empty()) {
        used = true;
        const Node *swNode = dfg.getNode(state.hwNodeToSwNodes[fuId][0]);
        out << pe.peName << " (" << pe.row << "," << pe.col << "): "
            << getNodeAttrStr(adg.getNode(fuId), "op_name");
        if (swNode)
          out << " <- " << getNodeAttrStr(swNode, "op_name");
        out << "\n";
      }
    }
    if (!used) {
      out << pe.peName << " (" << pe.row << "," << pe.col << "): unused\n";
    }
  }

  return true;
}

} // namespace fcc

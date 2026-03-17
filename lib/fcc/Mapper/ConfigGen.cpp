#include "fcc/Mapper/ConfigGen.h"
#include "fcc/Mapper/BridgeBinding.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <tuple>

namespace fcc {

namespace {

struct MemoryRegionEntry {
  IdIndex swNode = INVALID_ID;
  int64_t memrefArgIndex = -1;
  int64_t lane = -1;
  int64_t ldCount = 0;
  int64_t stCount = 0;
  int64_t elemSizeLog2 = 0;
};

mlir::DenseI64ArrayAttr getDenseI64NodeAttr(const Node *node,
                                            llvm::StringRef name) {
  if (!node)
    return {};
  for (const auto &attr : node->attributes) {
    if (attr.getName() == name)
      return mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr.getValue());
  }
  return {};
}

mlir::Type getTypeNodeAttr(const Node *node, llvm::StringRef name) {
  if (!node)
    return {};
  for (const auto &attr : node->attributes) {
    if (attr.getName() != name)
      continue;
    if (auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr.getValue()))
      return typeAttr.getValue();
  }
  return {};
}

const Node *getPortOwnerNode(const Graph &graph, IdIndex portId) {
  const Port *port = graph.getPort(portId);
  if (!port || port->parentNode == INVALID_ID)
    return nullptr;
  return graph.getNode(port->parentNode);
}

IdIndex findFeedingPort(const Graph &graph, IdIndex inputPortId) {
  const Port *port = graph.getPort(inputPortId);
  if (!port)
    return INVALID_ID;
  for (IdIndex edgeId : port->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (edge && edge->dstPort == inputPortId)
      return edge->srcPort;
  }
  return INVALID_ID;
}

llvm::SmallVector<IdIndex, 4> findConsumingPorts(const Graph &graph,
                                                 IdIndex outputPortId) {
  llvm::SmallVector<IdIndex, 4> consumers;
  const Port *port = graph.getPort(outputPortId);
  if (!port)
    return consumers;
  for (IdIndex edgeId : port->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (edge && edge->srcPort == outputPortId)
      consumers.push_back(edge->dstPort);
  }
  return consumers;
}

IdIndex findMemoryInputFromSwitchOutput(const Graph &graph, IdIndex switchOutPort,
                                        IdIndex hwMemoryNodeId) {
  for (IdIndex dstPortId : findConsumingPorts(graph, switchOutPort)) {
    const Port *dstPort = graph.getPort(dstPortId);
    if (!dstPort || dstPort->parentNode != hwMemoryNodeId)
      continue;
    const Node *dstNode = graph.getNode(dstPort->parentNode);
    if (dstNode && getNodeAttrStr(dstNode, "resource_class") == "memory")
      return dstPortId;
  }
  return INVALID_ID;
}

IdIndex findSwitchInputFromMemory(const Graph &graph, const Node *switchNode,
                                  IdIndex hwMemoryNodeId) {
  if (!switchNode)
    return INVALID_ID;
  for (IdIndex inPortId : switchNode->inputPorts) {
    IdIndex srcPortId = findFeedingPort(graph, inPortId);
    const Port *srcPort = graph.getPort(srcPortId);
    if (!srcPort || srcPort->parentNode != hwMemoryNodeId)
      continue;
    const Node *srcNode = graph.getNode(srcPort->parentNode);
    if (srcNode && getNodeAttrStr(srcNode, "resource_class") == "memory")
      return inPortId;
  }
  return INVALID_ID;
}

int64_t findDfgMemrefArgIndex(const Node *swNode, const Graph &dfg) {
  if (!swNode || swNode->inputPorts.empty())
    return -1;
  IdIndex memrefPort = swNode->inputPorts[0];
  const Port *inPort = dfg.getPort(memrefPort);
  if (!inPort)
    return -1;
  for (IdIndex edgeId : inPort->connectedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge || edge->dstPort != memrefPort)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    if (!srcPort || srcPort->parentNode == INVALID_ID)
      continue;
    const Node *srcNode = dfg.getNode(srcPort->parentNode);
    if (!srcNode || srcNode->kind != Node::ModuleInputNode)
      continue;
    return getNodeAttrInt(srcNode, "arg_index", -1);
  }
  return -1;
}

std::optional<int64_t> getRegionElemSizeLog2(const Node *swNode,
                                             const Node *hwNode,
                                             const Graph &dfg,
                                             const Graph &adg) {
  if (swNode && !swNode->inputPorts.empty()) {
    const Port *swMemrefPort = dfg.getPort(swNode->inputPorts[0]);
    if (swMemrefPort) {
      if (auto log2 = detail::getMemRefElementByteWidthLog2(swMemrefPort->type))
        return static_cast<int64_t>(*log2);
    }
  }

  if (hwNode) {
    if (mlir::Type hwMemRefType = getTypeNodeAttr(hwNode, "memref_type")) {
      if (auto log2 = detail::getMemRefElementByteWidthLog2(hwMemRefType))
        return static_cast<int64_t>(*log2);
    }
    if (!hwNode->inputPorts.empty()) {
      if (const Port *hwMemrefPort = adg.getPort(hwNode->inputPorts[0])) {
        if (auto log2 =
                detail::getMemRefElementByteWidthLog2(hwMemrefPort->type)) {
          return static_cast<int64_t>(*log2);
        }
      }
    }
  }

  return std::nullopt;
}

std::vector<MemoryRegionEntry> collectMemoryRegionsForNode(
    IdIndex hwId, const MappingState &state, const Graph &dfg,
    const Graph &adg) {
  std::vector<MemoryRegionEntry> entries;
  const Node *hwNode = adg.getNode(hwId);
  if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "memory" ||
      hwId >= state.hwNodeToSwNodes.size()) {
    return entries;
  }

  BridgeInfo bridge = BridgeInfo::extract(hwNode);
  bool isExtMem = (getNodeAttrStr(hwNode, "op_kind") == "extmemory");

  for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;

    DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);
    MemoryRegionEntry entry;
    entry.swNode = swId;
    entry.memrefArgIndex = findDfgMemrefArgIndex(swNode, dfg);
    entry.ldCount = memInfo.ldCount;
    entry.stCount = memInfo.stCount;
    if (auto elemSize = getRegionElemSizeLog2(swNode, hwNode, dfg, adg))
      entry.elemSizeLog2 = *elemSize;
    if (bridge.hasBridge) {
      auto lane = inferBridgeLane(bridge, memInfo, swNode, state);
      entry.lane = lane ? static_cast<int64_t>(*lane) : -1;
    } else {
      entry.lane = static_cast<int64_t>(entries.size());
    }
    entries.push_back(entry);
  }

  std::sort(entries.begin(), entries.end(),
            [](const MemoryRegionEntry &a, const MemoryRegionEntry &b) {
    if (a.lane != b.lane)
      return a.lane < b.lane;
    return a.swNode < b.swNode;
  });
  return entries;
}

llvm::SmallVector<int64_t, 16> buildAddrOffsetTable(const Node *hwNode,
                                                    IdIndex hwId,
                                                    const MappingState &state,
                                                    const Graph &dfg,
                                                    const Graph &adg) {
  llvm::SmallVector<int64_t, 16> table;
  constexpr int64_t kDefaultRegionStride = 4096;
  constexpr int64_t kRegionFieldCount = 5;
  int64_t numRegion = getNodeAttrInt(hwNode, "numRegion", 1);
  if (numRegion < 1)
    return table;

  auto regions = collectMemoryRegionsForNode(hwId, state, dfg, adg);
  auto templateTable = getDenseI64NodeAttr(hwNode, "addrOffsetTable");
  llvm::ArrayRef<int64_t> templateVals;
  if (templateTable &&
      static_cast<int64_t>(templateTable.size()) == numRegion * kRegionFieldCount) {
    templateVals = templateTable.asArrayRef();
  }

  table.reserve(static_cast<size_t>(numRegion) * kRegionFieldCount);
  for (int64_t slot = 0; slot < numRegion; ++slot) {
    if (static_cast<size_t>(slot) < regions.size()) {
      const auto &region = regions[slot];
      int64_t lane = regions[slot].lane >= 0 ? regions[slot].lane : slot;
      int64_t base = 0;
      int64_t elemSizeLog2 = region.elemSizeLog2;
      if (!templateVals.empty()) {
        base = templateVals[slot * kRegionFieldCount + 3];
        elemSizeLog2 = templateVals[slot * kRegionFieldCount + 4];
      } else if (region.memrefArgIndex >= 0) {
        base = region.memrefArgIndex * kDefaultRegionStride;
      } else {
        base = slot * kDefaultRegionStride;
      }
      table.push_back(1);
      table.push_back(lane);
      table.push_back(lane + 1);
      table.push_back(base);
      table.push_back(elemSizeLog2);
    } else {
      table.append({0, 0, 0, 0, 0});
    }
  }
  return table;
}

llvm::SmallVector<IdIndex, 8> buildBridgeInputSuffix(const Graph &adg,
                                                     IdIndex boundaryInPortId,
                                                     IdIndex hwMemoryNodeId) {
  llvm::SmallVector<IdIndex, 8> suffix;
  const Node *boundaryOwner = getPortOwnerNode(adg, boundaryInPortId);
  if (!boundaryOwner)
    return suffix;

  if (getNodeAttrStr(boundaryOwner, "resource_class") == "memory")
    return suffix;

  llvm::StringRef ownerKind = getNodeAttrStr(boundaryOwner, "op_kind");
  if (ownerKind == "add_tag") {
    if (boundaryOwner->outputPorts.empty())
      return suffix;
    IdIndex addTagOutPort = boundaryOwner->outputPorts[0];
    for (IdIndex nextInPortId : findConsumingPorts(adg, addTagOutPort)) {
      const Node *nextNode = getPortOwnerNode(adg, nextInPortId);
      if (!nextNode)
        continue;

      if (getNodeAttrStr(nextNode, "op_kind") == "temporal_sw") {
        for (IdIndex tswOutPort : nextNode->outputPorts) {
          IdIndex memInPortId =
              findMemoryInputFromSwitchOutput(adg, tswOutPort, hwMemoryNodeId);
          if (memInPortId == INVALID_ID)
            continue;
          suffix.push_back(addTagOutPort);
          suffix.push_back(nextInPortId);
          suffix.push_back(tswOutPort);
          suffix.push_back(memInPortId);
          return suffix;
        }
      }

      if (getNodeAttrStr(nextNode, "resource_class") == "memory" &&
          nextInPortId != boundaryInPortId) {
        suffix.push_back(addTagOutPort);
        suffix.push_back(nextInPortId);
        return suffix;
      }
    }
    return suffix;
  }

  if (ownerKind == "temporal_sw") {
    for (IdIndex tswOutPort : boundaryOwner->outputPorts) {
      IdIndex memInPortId =
          findMemoryInputFromSwitchOutput(adg, tswOutPort, hwMemoryNodeId);
      if (memInPortId == INVALID_ID)
        continue;
      suffix.push_back(tswOutPort);
      suffix.push_back(memInPortId);
      return suffix;
    }
  }

  return suffix;
}

llvm::SmallVector<IdIndex, 8> buildBridgeOutputPrefix(const Graph &adg,
                                                      IdIndex boundaryOutPortId,
                                                      IdIndex hwMemoryNodeId) {
  llvm::SmallVector<IdIndex, 8> prefix;
  const Node *boundaryOwner = getPortOwnerNode(adg, boundaryOutPortId);
  if (!boundaryOwner)
    return prefix;

  if (getNodeAttrStr(boundaryOwner, "resource_class") == "memory")
    return prefix;

  llvm::StringRef ownerKind = getNodeAttrStr(boundaryOwner, "op_kind");
  if (ownerKind == "del_tag") {
    if (boundaryOwner->inputPorts.empty())
      return prefix;
    IdIndex delTagInPort = boundaryOwner->inputPorts[0];
    IdIndex prevOutPort = findFeedingPort(adg, delTagInPort);
    const Node *prevNode = getPortOwnerNode(adg, prevOutPort);
    if (!prevNode)
      return prefix;

    if (getNodeAttrStr(prevNode, "op_kind") == "temporal_sw") {
      IdIndex tswInPort = findSwitchInputFromMemory(adg, prevNode, hwMemoryNodeId);
      if (tswInPort != INVALID_ID) {
        prefix.push_back(findFeedingPort(adg, tswInPort));
        prefix.push_back(tswInPort);
        prefix.push_back(prevOutPort);
        prefix.push_back(delTagInPort);
      }
      return prefix;
    }

    if (getNodeAttrStr(prevNode, "resource_class") == "memory") {
      prefix.push_back(prevOutPort);
      prefix.push_back(delTagInPort);
    }
    return prefix;
  }

  if (ownerKind == "temporal_sw") {
    IdIndex tswInPort = findSwitchInputFromMemory(adg, boundaryOwner, hwMemoryNodeId);
    if (tswInPort != INVALID_ID) {
      prefix.push_back(findFeedingPort(adg, tswInPort));
      prefix.push_back(tswInPort);
    }
  }

  return prefix;
}

llvm::SmallVector<IdIndex, 16>
buildExportPathForEdge(IdIndex edgeId, const MappingState &state,
                       const Graph &dfg, const Graph &adg) {
  llvm::SmallVector<IdIndex, 16> path;
  if (edgeId >= state.swEdgeToHwPaths.size())
    return path;
  path.append(state.swEdgeToHwPaths[edgeId].begin(),
              state.swEdgeToHwPaths[edgeId].end());
  if (path.size() < 2)
    return path;

  const Edge *edge = dfg.getEdge(edgeId);
  if (!edge)
    return path;

  const Port *srcPort = dfg.getPort(edge->srcPort);
  if (srcPort && srcPort->parentNode != INVALID_ID) {
    const Node *srcNode = dfg.getNode(srcPort->parentNode);
    if (srcNode && getNodeAttrStr(srcNode, "op_name") == "handshake.extmemory" &&
        srcPort->parentNode < state.swNodeToHwNode.size()) {
      IdIndex hwNodeId = state.swNodeToHwNode[srcPort->parentNode];
      if (hwNodeId != INVALID_ID) {
        auto prefix = buildBridgeOutputPrefix(adg, path.front(), hwNodeId);
        if (!prefix.empty()) {
          prefix.append(path.begin(), path.end());
          path = std::move(prefix);
        }
      }
    }
  }

  const Port *dstPort = dfg.getPort(edge->dstPort);
  if (dstPort && dstPort->parentNode != INVALID_ID) {
    const Node *dstNode = dfg.getNode(dstPort->parentNode);
    if (dstNode && getNodeAttrStr(dstNode, "op_name") == "handshake.extmemory" &&
        dstPort->parentNode < state.swNodeToHwNode.size()) {
      IdIndex hwNodeId = state.swNodeToHwNode[dstPort->parentNode];
      if (hwNodeId != INVALID_ID) {
        auto suffix = buildBridgeInputSuffix(adg, path.back(), hwNodeId);
        if (!suffix.empty())
          path.append(suffix.begin(), suffix.end());
      }
    }
  }

  return path;
}

} // namespace

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

    auto exportPath = buildExportPathForEdge(eid, state, dfg, adg);
    if (exportPath.empty())
      continue;

    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"sw_edge\": " << eid << ", \"path\": [";
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
      out << ", \"lane\": " << regions[i].lane;
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

    if (eid < state.swEdgeToHwPaths.size() &&
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
      out << " lane=" << regions[i].lane;
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

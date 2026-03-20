#include "fcc/Mapper/TagRuntime.h"
#include "fcc/Mapper/BridgeBinding.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace fcc {

namespace {

struct MapTagTableEntry {
  bool valid = false;
  uint64_t srcTag = 0;
  uint64_t dstTag = 0;
};

std::optional<uint64_t> getUIntNodeAttr(const Node *node, llvm::StringRef name) {
  if (!node)
    return std::nullopt;
  for (const auto &attr : node->attributes) {
    if (attr.getName() != name)
      continue;
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
      return static_cast<uint64_t>(intAttr.getInt());
  }
  return std::nullopt;
}

llvm::SmallVector<MapTagTableEntry, 8> getMapTagTableEntries(const Node *node) {
  llvm::SmallVector<MapTagTableEntry, 8> entries;
  if (!node)
    return entries;
  for (const auto &attr : node->attributes) {
    if (attr.getName() != "table")
      continue;
    auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue());
    if (!arrayAttr)
      return entries;
    for (size_t idx = 0; idx < arrayAttr.size(); ++idx) {
      mlir::Attribute elem = arrayAttr[idx];
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(elem)) {
        MapTagTableEntry entry;
        entry.valid = true;
        entry.srcTag = static_cast<uint64_t>(idx);
        entry.dstTag = static_cast<uint64_t>(intAttr.getInt());
        entries.push_back(entry);
        continue;
      }
      auto tupleAttr = mlir::dyn_cast<mlir::ArrayAttr>(elem);
      if (!tupleAttr || tupleAttr.size() != 3)
        continue;
      auto validAttr = mlir::dyn_cast<mlir::IntegerAttr>(tupleAttr[0]);
      auto srcAttr = mlir::dyn_cast<mlir::IntegerAttr>(tupleAttr[1]);
      auto dstAttr = mlir::dyn_cast<mlir::IntegerAttr>(tupleAttr[2]);
      if (!validAttr || !srcAttr || !dstAttr)
        continue;
      MapTagTableEntry entry;
      entry.valid = validAttr.getInt() != 0;
      entry.srcTag = static_cast<uint64_t>(srcAttr.getInt());
      entry.dstTag = static_cast<uint64_t>(dstAttr.getInt());
      entries.push_back(entry);
    }
    return entries;
  }
  return entries;
}

bool isSoftwareMemoryInterfaceOpName(llvm::StringRef opName) {
  return opName == "handshake.extmemory" || opName == "handshake.memory";
}

std::optional<uint64_t>
findMemoryRegionStartLane(IdIndex swNodeId, IdIndex hwNodeId,
                          const MappingState &state, const Graph &dfg,
                          const Graph &adg) {
  const Node *hwNode = adg.getNode(hwNodeId);
  if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "memory" ||
      hwNodeId >= state.hwNodeToSwNodes.size()) {
    return std::nullopt;
  }

  BridgeInfo bridge = BridgeInfo::extract(hwNode);
  bool isExtMem = (getNodeAttrStr(hwNode, "op_kind") == "extmemory");

  if (bridge.hasBridge) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode)
      return std::nullopt;
    DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);
    auto laneRange = inferBridgeLaneRange(bridge, memInfo, swNode, state);
    if (!laneRange)
      return std::nullopt;
    return laneRange->start;
  }

  llvm::SmallVector<IdIndex, 4> swNodes(state.hwNodeToSwNodes[hwNodeId].begin(),
                                        state.hwNodeToSwNodes[hwNodeId].end());
  llvm::sort(swNodes);
  for (unsigned idx = 0; idx < swNodes.size(); ++idx) {
    if (swNodes[idx] == swNodeId)
      return static_cast<uint64_t>(idx);
  }
  return std::nullopt;
}

std::optional<uint64_t>
computeSoftwareMemoryPortLane(IdIndex swNodeId, IdIndex swPortId, bool isOutput,
                              IdIndex hwNodeId, const MappingState &state,
                              const Graph &dfg, const Graph &adg) {
  if (swNodeId == INVALID_ID || swPortId == INVALID_ID ||
      swNodeId >= state.swNodeToHwNode.size() ||
      state.swNodeToHwNode[swNodeId] != hwNodeId)
    return std::nullopt;

  const Node *swNode = dfg.getNode(swNodeId);
  if (!swNode)
    return std::nullopt;
  llvm::StringRef opName = getNodeAttrStr(swNode, "op_name");
  if (!isSoftwareMemoryInterfaceOpName(opName))
    return std::nullopt;

  bool isExtMem = opName == "handshake.extmemory";
  DfgMemoryInfo mem = DfgMemoryInfo::extract(swNode, dfg, isExtMem);
  auto baseLane = findMemoryRegionStartLane(swNodeId, hwNodeId, state, dfg, adg);
  if (!baseLane)
    return std::nullopt;

  if (isOutput) {
    for (unsigned idx = 0; idx < swNode->outputPorts.size(); ++idx) {
      if (swNode->outputPorts[idx] != swPortId)
        continue;
      return *baseLane + mem.outputLocalLane(idx);
    }
    return std::nullopt;
  }

  for (unsigned idx = mem.swInSkip; idx < swNode->inputPorts.size(); ++idx) {
    if (swNode->inputPorts[idx] != swPortId)
      continue;
    return *baseLane + mem.inputLocalLane(idx - mem.swInSkip);
  }
  return std::nullopt;
}

} // namespace

const Node *getPortOwnerNode(const Graph &graph, IdIndex portId) {
  const Port *port = graph.getPort(portId);
  if (!port || port->parentNode == INVALID_ID)
    return nullptr;
  return graph.getNode(port->parentNode);
}

std::optional<uint64_t> applyMapTagTableValue(const Node *mapTagNode,
                                              std::optional<uint64_t> tag) {
  if (!mapTagNode || !tag)
    return std::nullopt;

  unsigned tableSize =
      static_cast<unsigned>(getNodeAttrInt(mapTagNode, "table_size", 0));
  if (tableSize == 0)
    return std::nullopt;
  auto entries = getMapTagTableEntries(mapTagNode);
  if (entries.size() != tableSize)
    return std::nullopt;
  std::optional<uint64_t> mapped;
  for (const auto &entry : entries) {
    if (!entry.valid || entry.srcTag != *tag)
      continue;
    if (mapped)
      return std::nullopt;
    mapped = entry.dstTag;
  }
  if (mapped)
    return mapped;
  return std::nullopt;
}

bool runtimeTagValueFitsType(uint64_t tag, mlir::Type type) {
  auto info = detail::getPortTypeInfo(type);
  if (!info || !info->isTagged)
    return false;
  if (info->tagWidth >= 64)
    return true;
  return (tag >> info->tagWidth) == 0;
}

RuntimeTagValueInfo
projectRuntimeTagValueToTypeInfo(std::optional<uint64_t> tag, mlir::Type type) {
  if (!tag)
    return RuntimeTagValueInfo{true, std::nullopt, std::nullopt};

  auto info = detail::getPortTypeInfo(type);
  if (!info || !info->isTagged)
    return RuntimeTagValueInfo{true, std::nullopt, std::nullopt};

  if (!runtimeTagValueFitsType(*tag, type))
    return RuntimeTagValueInfo{false, std::nullopt, *tag};

  return RuntimeTagValueInfo{true, *tag, std::nullopt};
}

std::optional<uint64_t> projectRuntimeTagValueToType(std::optional<uint64_t> tag,
                                                     mlir::Type type) {
  auto info = projectRuntimeTagValueToTypeInfo(tag, type);
  if (!info.representable)
    return std::nullopt;
  return info.tag;
}

RuntimeTagValueInfo advanceRuntimeTagValueInfoAtPort(
    std::optional<uint64_t> currentTag, IdIndex portId, const Graph &adg,
    llvm::function_ref<std::optional<uint64_t>(IdIndex)> externalTagAtPort) {
  std::optional<uint64_t> tag = currentTag;
  const Port *port = adg.getPort(portId);
  const Node *owner = getPortOwnerNode(adg, portId);
  if (!port)
    return RuntimeTagValueInfo{true, tag, std::nullopt};

  if (externalTagAtPort) {
    if (!tag) {
      if (auto extTag = externalTagAtPort(portId))
        tag = *extTag;
    }
  }

  if (owner && port->direction == Port::Output) {
    llvm::StringRef opKind = getNodeAttrStr(owner, "op_kind");
    if (opKind == "add_tag") {
      if (auto edgeTag = getUIntNodeAttr(owner, "tag"))
        tag = *edgeTag;
    } else if (opKind == "map_tag") {
      tag = applyMapTagTableValue(owner, tag);
    } else if (opKind == "del_tag") {
      tag = std::nullopt;
    }
  }

  auto projected = projectRuntimeTagValueToTypeInfo(tag, port->type);
  if (!projected.representable)
    return projected;
  return RuntimeTagValueInfo{true, projected.tag, std::nullopt};
}

RuntimeTagValueInfo computeRuntimeTagValueInfoAlongPath(
    llvm::ArrayRef<IdIndex> hwPath, size_t uptoIndex, const Graph &adg,
    llvm::function_ref<std::optional<uint64_t>(IdIndex)> externalTagAtPort) {
  if (hwPath.empty())
    return RuntimeTagValueInfo{true, std::nullopt, std::nullopt};

  std::optional<uint64_t> tag;
  for (size_t idx = 0; idx <= uptoIndex && idx < hwPath.size(); ++idx) {
    auto advanced =
        advanceRuntimeTagValueInfoAtPort(tag, hwPath[idx], adg, externalTagAtPort);
    if (!advanced.representable)
      return advanced;
    tag = advanced.tag;
  }

  return RuntimeTagValueInfo{true, tag, std::nullopt};
}

std::optional<uint64_t> computeRuntimeTagValueAlongPath(
    llvm::ArrayRef<IdIndex> hwPath, size_t uptoIndex, const Graph &adg,
    llvm::function_ref<std::optional<uint64_t>(IdIndex)> externalTagAtPort) {
  auto info =
      computeRuntimeTagValueInfoAlongPath(hwPath, uptoIndex, adg, externalTagAtPort);
  if (!info.representable)
    return std::nullopt;
  return info.tag;
}

std::optional<uint64_t> computeExternalRuntimeTagAtMappedPort(
    IdIndex swEdgeId, IdIndex portId, const MappingState &state,
    const Graph &dfg, const Graph &adg) {
  const Edge *swEdge = dfg.getEdge(swEdgeId);
  if (!swEdge)
    return std::nullopt;

  const Port *port = adg.getPort(portId);
  const Node *owner = getPortOwnerNode(adg, portId);
  if (!port || !owner ||
      getNodeAttrStr(owner, "resource_class") != "memory") {
    return std::nullopt;
  }

  const Port *swSrcPort = dfg.getPort(swEdge->srcPort);
  const Port *swDstPort = dfg.getPort(swEdge->dstPort);
  IdIndex swSrcNodeId =
      swSrcPort ? swSrcPort->parentNode : static_cast<IdIndex>(INVALID_ID);
  IdIndex swDstNodeId =
      swDstPort ? swDstPort->parentNode : static_cast<IdIndex>(INVALID_ID);
  IdIndex hwNodeId = port->parentNode;

  if (auto lane = computeSoftwareMemoryPortLane(swSrcNodeId, swEdge->srcPort,
                                                true, hwNodeId, state, dfg,
                                                adg)) {
    return *lane;
  }
  if (auto lane = computeSoftwareMemoryPortLane(swDstNodeId, swEdge->dstPort,
                                                false, hwNodeId, state, dfg,
                                                adg)) {
    return *lane;
  }
  return std::nullopt;
}

RuntimeTagValueInfo computeRuntimeTagValueInfoAlongMappedPath(
    IdIndex swEdgeId, llvm::ArrayRef<IdIndex> hwPath, size_t uptoIndex,
    const MappingState &state, const Graph &dfg, const Graph &adg) {
  if (hwPath.empty())
    return RuntimeTagValueInfo{true, std::nullopt, std::nullopt};

  return computeRuntimeTagValueInfoAlongPath(
      hwPath, uptoIndex, adg, [&](IdIndex portId) -> std::optional<uint64_t> {
        return computeExternalRuntimeTagAtMappedPort(swEdgeId, portId, state,
                                                     dfg, adg);
      });
}

std::optional<uint64_t> computeRuntimeTagValueAlongMappedPath(
    IdIndex swEdgeId, llvm::ArrayRef<IdIndex> hwPath, size_t uptoIndex,
    const MappingState &state, const Graph &dfg, const Graph &adg) {
  auto info = computeRuntimeTagValueInfoAlongMappedPath(swEdgeId, hwPath,
                                                        uptoIndex, state, dfg,
                                                        adg);
  if (!info.representable)
    return std::nullopt;
  return info.tag;
}

} // namespace fcc

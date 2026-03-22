// ConfigGenHelpers.cpp -- Helper functions for ConfigGen.
// Bit packing, port/graph traversal, memory, tag, and export path helpers.
// This file is compiled as part of the ConfigGen translation unit.

#include "ConfigGenInternal.h"

#include <cctype>
#include <sstream>
#include <tuple>

namespace loom {
namespace configgen_detail {

// ---------------------------------------------------------------------------
// Bit packing
// ---------------------------------------------------------------------------

void packBits(std::vector<uint32_t> &words, uint32_t &bitPos, uint64_t value,
              unsigned width) {
  if (width == 0)
    return;
  uint32_t wordIndex = bitPos / kConfigWordBits;
  uint32_t bitIndex = bitPos % kConfigWordBits;
  uint32_t bitsRemaining = width;
  uint64_t remainingValue = value;

  while (bitsRemaining > 0) {
    if (words.size() <= wordIndex)
      words.resize(wordIndex + 1, 0);
    uint32_t chunkWidth =
        std::min<uint32_t>(bitsRemaining, kConfigWordBits - bitIndex);
    uint32_t chunkMask =
        chunkWidth == kConfigWordBits
            ? 0xffffffffu
            : ((uint32_t{1} << chunkWidth) - 1u);
    words[wordIndex] |=
        static_cast<uint32_t>(remainingValue & chunkMask) << bitIndex;
    remainingValue >>= chunkWidth;
    bitsRemaining -= chunkWidth;
    bitIndex = 0;
    ++wordIndex;
  }

  bitPos += width;
}

std::string buildHeaderGuard(llvm::StringRef pathStem) {
  std::string guard = "LOOM_";
  for (char ch : pathStem) {
    if (std::isalnum(static_cast<unsigned char>(ch)))
      guard.push_back(static_cast<char>(std::toupper(ch)));
    else
      guard.push_back('_');
  }
  guard += "_CONFIG_H";
  return guard;
}

std::string getConfigHeaderFilename(llvm::StringRef basePath) {
  std::string result = basePath.str();
  result += ".config.h";
  return result;
}

unsigned bitWidthForChoices(unsigned count) {
  return count > 1 ? llvm::Log2_32_Ceil(count) : 0;
}

llvm::StringRef configFieldKindName(FUConfigFieldKind kind) {
  switch (kind) {
  case FUConfigFieldKind::Mux:
    return "mux";
  case FUConfigFieldKind::ConstantValue:
    return "constant_value";
  case FUConfigFieldKind::CmpIPredicate:
    return "cmpi_predicate";
  case FUConfigFieldKind::CmpFPredicate:
    return "cmpf_predicate";
  case FUConfigFieldKind::StreamContCond:
    return "stream_cont_cond";
  case FUConfigFieldKind::JoinMask:
    return "join_mask";
  }
  return "unknown";
}

std::string formatConfigFieldValue(const FUConfigField &field) {
  switch (field.kind) {
  case FUConfigFieldKind::Mux:
    return ("sel=" + std::to_string(field.sel));
  case FUConfigFieldKind::ConstantValue:
    return ("value=" + std::to_string(field.value));
  case FUConfigFieldKind::CmpIPredicate:
  case FUConfigFieldKind::CmpFPredicate:
    return ("predicate=" + std::to_string(field.value));
  case FUConfigFieldKind::StreamContCond:
    switch (field.value) {
    case 1u << 0:
      return "cont_cond=<";
    case 1u << 1:
      return "cont_cond=<=";
    case 1u << 2:
      return "cont_cond=>";
    case 1u << 3:
      return "cont_cond=>=";
    case 1u << 4:
      return "cont_cond=!=";
    default:
      return ("cont_cond=" + std::to_string(field.value));
    }
  case FUConfigFieldKind::JoinMask: {
    std::string bits;
    bits.reserve(field.bitWidth);
    for (unsigned bit = 0; bit < field.bitWidth; ++bit)
      bits.push_back(((field.value >> (field.bitWidth - 1 - bit)) & 1u) ? '1'
                                                                         : '0');
    return "join_mask=0b" + bits;
  }
  }
  return std::to_string(field.value);
}

void packMuxField(std::vector<uint32_t> &words, uint32_t &bitPos,
                  unsigned selBits, uint64_t sel, bool discard,
                  bool disconnect) {
  packBits(words, bitPos, sel, selBits);
  packBits(words, bitPos, discard ? 1u : 0u, 1);
  packBits(words, bitPos, disconnect ? 1u : 0u, 1);
}

// ---------------------------------------------------------------------------
// Port / graph traversal
// ---------------------------------------------------------------------------

bool isConnectedPosition(const std::vector<std::string> &rows, unsigned outIdx,
                         unsigned inIdx, unsigned numIn) {
  if (rows.empty())
    return inIdx < numIn;
  if (outIdx >= rows.size())
    return false;
  const std::string &row = rows[outIdx];
  if (inIdx >= row.size())
    return false;
  return row[inIdx] == '1';
}

unsigned countConnectedPositions(const std::vector<std::string> &rows,
                                 unsigned numIn, unsigned numOut) {
  if (rows.empty())
    return numIn * numOut;
  unsigned count = 0;
  for (unsigned outIdx = 0; outIdx < numOut; ++outIdx) {
    for (unsigned inIdx = 0; inIdx < numIn; ++inIdx) {
      if (isConnectedPosition(rows, outIdx, inIdx, numIn))
        ++count;
    }
  }
  return count;
}

unsigned connectedPositionOrdinal(const std::vector<std::string> &rows,
                                  unsigned numIn, unsigned numOut,
                                  unsigned inputIdx, unsigned outputIdx) {
  unsigned ordinal = 0;
  for (unsigned outIdx = 0; outIdx < numOut; ++outIdx) {
    for (unsigned inIdx = 0; inIdx < numIn; ++inIdx) {
      if (!isConnectedPosition(rows, outIdx, inIdx, numIn))
        continue;
      if (inIdx == inputIdx && outIdx == outputIdx)
        return ordinal;
      ++ordinal;
    }
  }
  return ordinal;
}

int findNodeInputIndex(const Node *node, IdIndex portId) {
  if (!node)
    return -1;
  for (unsigned i = 0; i < node->inputPorts.size(); ++i) {
    if (node->inputPorts[i] == portId)
      return static_cast<int>(i);
  }
  return -1;
}

int findNodeOutputIndex(const Node *node, IdIndex portId) {
  if (!node)
    return -1;
  for (unsigned i = 0; i < node->outputPorts.size(); ++i) {
    if (node->outputPorts[i] == portId)
      return static_cast<int>(i);
  }
  return -1;
}

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

std::optional<uint64_t> getUIntEdgeAttr(const Edge *edge,
                                        llvm::StringRef name) {
  if (!edge)
    return std::nullopt;
  for (const auto &attr : edge->attributes) {
    if (attr.getName() != name)
      continue;
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
      return static_cast<uint64_t>(intAttr.getInt());
  }
  return std::nullopt;
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

IdIndex findMemoryInputFromSwitchOutput(const Graph &graph,
                                        IdIndex switchOutPort,
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

bool isBridgeTraversalNode(llvm::StringRef opKind) {
  return opKind == "add_tag" || opKind == "del_tag" || opKind == "map_tag" ||
         opKind == "spatial_sw" || opKind == "temporal_sw" ||
         opKind == "fifo";
}

llvm::SmallVector<IdIndex, 8>
findBridgePathForward(const Graph &graph, IdIndex startPortId,
                      IdIndex hwMemoryNodeId) {
  llvm::SmallVector<IdIndex, 8> empty;
  const Port *startPort = graph.getPort(startPortId);
  if (!startPort)
    return empty;
  if (startPort->parentNode == hwMemoryNodeId &&
      startPort->direction == Port::Input)
    return empty;

  llvm::SmallVector<IdIndex, 16> queue;
  llvm::DenseMap<IdIndex, IdIndex> prev;
  queue.push_back(startPortId);
  prev[startPortId] = INVALID_ID;

  IdIndex found = INVALID_ID;
  size_t cursor = 0;
  while (cursor < queue.size()) {
    IdIndex portId = queue[cursor++];
    const Port *port = graph.getPort(portId);
    if (!port)
      continue;

    if (portId != startPortId && port->parentNode == hwMemoryNodeId &&
        port->direction == Port::Input) {
      const Node *owner = graph.getNode(port->parentNode);
      if (owner && getNodeAttrStr(owner, "resource_class") == "memory") {
        found = portId;
        break;
      }
    }

    auto tryPush = [&](IdIndex nextPortId) {
      if (nextPortId == INVALID_ID || prev.count(nextPortId))
        return;
      prev[nextPortId] = portId;
      queue.push_back(nextPortId);
    };

    if (port->direction == Port::Input) {
      const Node *owner = getPortOwnerNode(graph, portId);
      if (!owner)
        continue;
      if (!isBridgeTraversalNode(getNodeAttrStr(owner, "op_kind")))
        continue;
      for (IdIndex outPortId : owner->outputPorts)
        tryPush(outPortId);
      continue;
    }

    for (IdIndex consumerPortId : findConsumingPorts(graph, portId))
      tryPush(consumerPortId);
  }

  if (found == INVALID_ID)
    return empty;

  llvm::SmallVector<IdIndex, 8> path;
  for (IdIndex cur = found; cur != INVALID_ID; cur = prev.lookup(cur))
    path.push_back(cur);
  std::reverse(path.begin(), path.end());
  if (!path.empty())
    path.erase(path.begin());
  return path;
}

llvm::SmallVector<IdIndex, 8>
findBridgePathBackward(const Graph &graph, IdIndex startPortId,
                       IdIndex hwMemoryNodeId) {
  llvm::SmallVector<IdIndex, 8> empty;
  const Port *startPort = graph.getPort(startPortId);
  if (!startPort)
    return empty;
  if (startPort->parentNode == hwMemoryNodeId &&
      startPort->direction == Port::Output)
    return empty;

  llvm::SmallVector<IdIndex, 16> queue;
  llvm::DenseMap<IdIndex, IdIndex> prev;
  queue.push_back(startPortId);
  prev[startPortId] = INVALID_ID;

  IdIndex found = INVALID_ID;
  size_t cursor = 0;
  while (cursor < queue.size()) {
    IdIndex portId = queue[cursor++];
    const Port *port = graph.getPort(portId);
    if (!port)
      continue;

    if (portId != startPortId && port->parentNode == hwMemoryNodeId &&
        port->direction == Port::Output) {
      const Node *owner = graph.getNode(port->parentNode);
      if (owner && getNodeAttrStr(owner, "resource_class") == "memory") {
        found = portId;
        break;
      }
    }

    auto tryPush = [&](IdIndex nextPortId) {
      if (nextPortId == INVALID_ID || prev.count(nextPortId))
        return;
      prev[nextPortId] = portId;
      queue.push_back(nextPortId);
    };

    if (port->direction == Port::Output) {
      const Node *owner = getPortOwnerNode(graph, portId);
      if (!owner)
        continue;
      if (!isBridgeTraversalNode(getNodeAttrStr(owner, "op_kind")))
        continue;
      for (IdIndex inPortId : owner->inputPorts)
        tryPush(inPortId);
      continue;
    }

    tryPush(findFeedingPort(graph, portId));
  }

  if (found == INVALID_ID)
    return empty;

  llvm::SmallVector<IdIndex, 8> path;
  for (IdIndex cur = found; cur != INVALID_ID; cur = prev.lookup(cur))
    path.push_back(cur);
  if (!path.empty() && path.back() == startPortId)
    path.pop_back();
  return path;
}

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------

bool isSoftwareMemoryInterfaceOp(llvm::StringRef opName) {
  return opName == "handshake.extmemory" || opName == "handshake.memory";
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

std::vector<MemoryRegionEntry>
collectMemoryRegionsForNode(IdIndex hwId, const MappingState &state,
                            const Graph &dfg, const Graph &adg) {
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
      auto laneRange =
          inferBridgeLaneRange(bridge, memInfo, swNode, dfg, state);
      if (laneRange) {
        entry.startLane = static_cast<int64_t>(laneRange->start);
        entry.endLane = static_cast<int64_t>(laneRange->end);
      }
    } else {
      entry.startLane = static_cast<int64_t>(entries.size());
      entry.endLane = entry.startLane + 1;
    }
    entries.push_back(entry);
  }

  std::sort(entries.begin(), entries.end(),
            [](const MemoryRegionEntry &a, const MemoryRegionEntry &b) {
    if (a.startLane != b.startLane)
      return a.startLane < b.startLane;
    return a.swNode < b.swNode;
  });
  return entries;
}

llvm::SmallVector<int64_t, 16>
buildAddrOffsetTable(const Node *hwNode, IdIndex hwId,
                     const MappingState &state, const Graph &dfg,
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
      int64_t startLane =
          region.startLane >= 0 ? region.startLane : static_cast<int64_t>(slot);
      int64_t endLane =
          region.endLane > startLane ? region.endLane : startLane + 1;
      int64_t base = 0;
      int64_t elemSizeLog2 = region.elemSizeLog2;
      bool isExtMemory = getNodeAttrStr(hwNode, "op_kind") == "extmemory";
      if (!templateVals.empty()) {
        if (!isExtMemory)
          base = templateVals[slot * kRegionFieldCount + 3];
        elemSizeLog2 = templateVals[slot * kRegionFieldCount + 4];
      } else if (!isExtMemory && region.memrefArgIndex >= 0) {
        base = region.memrefArgIndex * kDefaultRegionStride;
      } else if (!isExtMemory) {
        base = slot * kDefaultRegionStride;
      }
      table.push_back(1);
      table.push_back(startLane);
      table.push_back(endLane);
      table.push_back(base);
      table.push_back(elemSizeLog2);
    } else {
      table.append({0, 0, 0, 0, 0});
    }
  }
  return table;
}

std::optional<uint64_t>
findMemoryRegionStartLane(IdIndex swNodeId, IdIndex hwNodeId,
                          const MappingState &state, const Graph &dfg,
                          const Graph &adg) {
  auto regions = collectMemoryRegionsForNode(hwNodeId, state, dfg, adg);
  for (const auto &region : regions) {
    if (region.swNode == swNodeId && region.startLane >= 0)
      return static_cast<uint64_t>(region.startLane);
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
  if (!isSoftwareMemoryInterfaceOp(opName))
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

// ---------------------------------------------------------------------------
// Tag helpers
// ---------------------------------------------------------------------------

std::optional<uint64_t> getUIntNodeAttr(const Node *node,
                                        llvm::StringRef name) {
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

std::pair<unsigned, unsigned> getMapTagTagWidths(const Node *node,
                                                 const Graph &adg) {
  unsigned inWidth = 0;
  unsigned outWidth = 0;
  if (!node)
    return {inWidth, outWidth};
  if (!node->inputPorts.empty()) {
    if (const Port *port = adg.getPort(node->inputPorts.front())) {
      if (auto info = detail::getPortTypeInfo(port->type); info && info->isTagged)
        inWidth = info->tagWidth;
    }
  }
  if (!node->outputPorts.empty()) {
    if (const Port *port = adg.getPort(node->outputPorts.front())) {
      if (auto info = detail::getPortTypeInfo(port->type); info && info->isTagged)
        outWidth = info->tagWidth;
    }
  }
  return {inWidth, outWidth};
}

std::vector<std::string> getBinaryRowsNodeAttr(const Node *node,
                                               llvm::StringRef name) {
  std::vector<std::string> rows;
  if (!node)
    return rows;
  for (const auto &attr : node->attributes) {
    if (attr.getName() != name)
      continue;
    auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue());
    if (!arrayAttr)
      return rows;
    for (auto elem : arrayAttr) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elem))
        rows.push_back(strAttr.getValue().str());
    }
    return rows;
  }
  return rows;
}

std::optional<uint64_t>
computeRuntimeTagValueAlongPath(IdIndex swEdgeId,
                                llvm::ArrayRef<IdIndex> hwPath,
                                size_t uptoIndex, const MappingState &state,
                                const Graph &dfg, const Graph &adg) {
  return loom::computeRuntimeTagValueAlongMappedPath(swEdgeId, hwPath, uptoIndex,
                                                    state, dfg, adg);
}

std::optional<size_t> findLastTaggedPortIndex(llvm::ArrayRef<IdIndex> hwPath,
                                              const Graph &adg) {
  for (size_t i = hwPath.size(); i > 0; --i) {
    const Port *port = adg.getPort(hwPath[i - 1]);
    if (!port)
      continue;
    if (auto info = detail::getPortTypeInfo(port->type)) {
      if (info->isTagged)
        return i - 1;
    }
  }
  return std::nullopt;
}

std::optional<uint64_t>
computeTemporalNodeTagValue(IdIndex swNodeId, const MappingState &state,
                            const Graph &dfg, const Graph &adg) {
  if (swNodeId == INVALID_ID)
    return std::nullopt;
  const Node *swNode = dfg.getNode(swNodeId);
  if (!swNode)
    return std::nullopt;

  std::optional<uint64_t> tag;
  auto considerEdge = [&](IdIndex swEdgeId) -> bool {
    auto hwPath = buildExportPathForEdge(swEdgeId, state, dfg, adg);
    if (hwPath.empty())
      return true;
    size_t observeIndex = hwPath.size() - 1;
    if (auto taggedIndex = findLastTaggedPortIndex(hwPath, adg))
      observeIndex = *taggedIndex;
    auto edgeTag = computeRuntimeTagValueAlongPath(swEdgeId, hwPath,
                                                   observeIndex, state, dfg,
                                                   adg);
    if (!edgeTag)
      return true;
    if (tag && *tag != *edgeTag)
        return false;
    tag = *edgeTag;
    return true;
  };

  for (IdIndex portId : swNode->inputPorts) {
    const Port *port = dfg.getPort(portId);
    if (!port)
      continue;
    for (IdIndex edgeId : port->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge || edge->dstPort != portId)
        continue;
      if (!considerEdge(edgeId))
        return std::nullopt;
    }
  }

  if (tag)
    return tag;

  for (IdIndex portId : swNode->outputPorts) {
    const Port *port = dfg.getPort(portId);
    if (!port)
      continue;
    for (IdIndex edgeId : port->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge || edge->srcPort != portId)
        continue;
      if (!considerEdge(edgeId))
        return std::nullopt;
    }
  }

  return tag;
}

std::optional<uint64_t>
computeTemporalNodeIngressTagValue(IdIndex swNodeId,
                                   const MappingState &state,
                                   const Graph &dfg, const Graph &adg) {
  if (swNodeId == INVALID_ID)
    return std::nullopt;
  const Node *swNode = dfg.getNode(swNodeId);
  if (!swNode)
    return std::nullopt;

  std::optional<uint64_t> tag;
  for (IdIndex portId : swNode->inputPorts) {
    const Port *port = dfg.getPort(portId);
    if (!port)
      continue;
    for (IdIndex edgeId : port->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge || edge->dstPort != portId)
        continue;
      auto hwPath = buildExportPathForEdge(edgeId, state, dfg, adg);
      if (hwPath.empty())
        continue;
      size_t observeIndex = hwPath.size() - 1;
      if (auto taggedIndex = findLastTaggedPortIndex(hwPath, adg))
        observeIndex = *taggedIndex;
      auto edgeTag = computeRuntimeTagValueAlongPath(edgeId, hwPath,
                                                     observeIndex, state, dfg,
                                                     adg);
      if (!edgeTag)
        continue;
      if (tag && *tag != *edgeTag)
        return std::nullopt;
      tag = *edgeTag;
    }
  }

  return tag;
}

std::optional<uint64_t>
computeTemporalRouteTagValue(IdIndex swEdgeId,
                             llvm::ArrayRef<IdIndex> hwPath,
                             size_t transitionIndex,
                             const MappingState &state, const Graph &dfg,
                             const Graph &adg) {
  if (auto tag = computeRuntimeTagValueAlongPath(swEdgeId, hwPath,
                                                 transitionIndex, state, dfg,
                                                 adg)) {
    return tag;
  }

  const Edge *swEdge = dfg.getEdge(swEdgeId);
  if (!swEdge)
    return std::nullopt;
  const Port *swSrcPort = dfg.getPort(swEdge->srcPort);
  const Port *swDstPort = dfg.getPort(swEdge->dstPort);
  IdIndex swSrcNodeId =
      swSrcPort ? swSrcPort->parentNode : static_cast<IdIndex>(INVALID_ID);
  IdIndex swDstNodeId =
      swDstPort ? swDstPort->parentNode : static_cast<IdIndex>(INVALID_ID);
  if (swSrcNodeId != INVALID_ID) {
    if (auto tag =
            computeTemporalNodeIngressTagValue(swSrcNodeId, state, dfg, adg)) {
      return tag;
    }
    if (auto tag = computeTemporalNodeTagValue(swSrcNodeId, state, dfg, adg))
      return tag;
  }
  if (swDstNodeId != INVALID_ID) {
    if (auto tag =
            computeTemporalNodeIngressTagValue(swDstNodeId, state, dfg, adg)) {
      return tag;
    }
    if (auto tag = computeTemporalNodeTagValue(swDstNodeId, state, dfg, adg))
      return tag;
  }

  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Export path helpers
// ---------------------------------------------------------------------------

llvm::SmallVector<IdIndex, 8>
buildBridgeInputSuffix(const Graph &adg, IdIndex boundaryInPortId,
                       IdIndex hwMemoryNodeId) {
  return findBridgePathForward(adg, boundaryInPortId, hwMemoryNodeId);
}

llvm::SmallVector<IdIndex, 8>
buildBridgeOutputPrefix(const Graph &adg, IdIndex boundaryOutPortId,
                        IdIndex hwMemoryNodeId) {
  return findBridgePathBackward(adg, boundaryOutPortId, hwMemoryNodeId);
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
    if (srcNode &&
        isSoftwareMemoryInterfaceOp(getNodeAttrStr(srcNode, "op_name")) &&
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
    if (dstNode &&
        isSoftwareMemoryInterfaceOp(getNodeAttrStr(dstNode, "op_name")) &&
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

} // namespace configgen_detail
} // namespace loom

#include "fcc/Mapper/ConfigGen.h"
#include "fcc/Mapper/BridgeBinding.h"
#include "fcc/Mapper/TagRuntime.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>

namespace fcc {

namespace {

constexpr uint32_t kConfigWordBits = 32;

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
  std::string guard = "FCC_";
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

struct MemoryRegionEntry {
  IdIndex swNode = INVALID_ID;
  int64_t memrefArgIndex = -1;
  int64_t startLane = -1;
  int64_t endLane = -1;
  int64_t ldCount = 0;
  int64_t stCount = 0;
  int64_t elemSizeLog2 = 0;
};

bool isSoftwareMemoryInterfaceOp(llvm::StringRef opName) {
  return opName == "handshake.extmemory" || opName == "handshake.memory";
}

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

struct MapTagTableEntry {
  bool valid = false;
  uint64_t srcTag = 0;
  uint64_t dstTag = 0;
};

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

std::optional<uint64_t> getUIntEdgeAttr(const Edge *edge, llvm::StringRef name) {
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
      auto laneRange = inferBridgeLaneRange(bridge, memInfo, swNode, state);
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

llvm::SmallVector<IdIndex, 8> buildBridgeInputSuffix(const Graph &adg,
                                                     IdIndex boundaryInPortId,
                                                     IdIndex hwMemoryNodeId) {
  return findBridgePathForward(adg, boundaryInPortId, hwMemoryNodeId);
}

llvm::SmallVector<IdIndex, 8> buildBridgeOutputPrefix(const Graph &adg,
                                                      IdIndex boundaryOutPortId,
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

struct GeneratedNodeConfig {
  std::vector<uint32_t> words;
  bool complete = true;
};

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

// Compute the runtime tag value associated with one software node.
// This is not a hardware tag-width inference; hardware tag shape comes from the
// tagged port types declared by the ADG.
std::optional<uint64_t>
computeRuntimeTagValueAlongPath(IdIndex swEdgeId, llvm::ArrayRef<IdIndex> hwPath,
                                size_t uptoIndex, const MappingState &state,
                                const Graph &dfg, const Graph &adg) {
  return fcc::computeRuntimeTagValueAlongMappedPath(swEdgeId, hwPath, uptoIndex,
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

std::optional<uint64_t> computeTemporalNodeTagValue(IdIndex swNodeId,
                                                    const MappingState &state,
                                                    const Graph &dfg,
                                                    const Graph &adg) {
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
computeTemporalNodeIngressTagValue(IdIndex swNodeId, const MappingState &state,
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

// Compute the runtime tag value seen by one routed software edge at a temporal
// routing/config boundary. This does not infer any hardware tag parameter.
std::optional<uint64_t>
computeTemporalRouteTagValue(IdIndex swEdgeId, llvm::ArrayRef<IdIndex> hwPath,
                             size_t transitionIndex, const MappingState &state,
                             const Graph &dfg, const Graph &adg) {
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

GeneratedNodeConfig buildSpatialSwitchConfig(const Node *hwNode, IdIndex hwId,
                                             const MappingState &state,
                                             const Graph &dfg,
                                             const Graph &adg) {
  GeneratedNodeConfig cfg;
  if (!hwNode)
    return cfg;
  unsigned numIn = hwNode->inputPorts.size();
  unsigned numOut = hwNode->outputPorts.size();
  if (numIn == 0 || numOut == 0)
    return cfg;

  auto rows = getBinaryRowsNodeAttr(hwNode, "connectivity_table");
  uint32_t bitPos = 0;
  std::set<std::pair<int, int>> activeTransitions;

  for (IdIndex edgeId = 0;
       edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size()); ++edgeId) {
    auto hwPath = buildExportPathForEdge(edgeId, state, dfg, adg);
    if (hwPath.size() < 3)
      continue;
    for (size_t i = 1; i + 1 < hwPath.size(); i += 2) {
      IdIndex inPortId = hwPath[i];
      IdIndex outPortId = hwPath[i + 1];
      const Port *inPort = adg.getPort(inPortId);
      const Port *outPort = adg.getPort(outPortId);
      if (!inPort || !outPort || inPort->parentNode != hwId ||
          outPort->parentNode != hwId)
        continue;
      int inputIdx = findNodeInputIndex(hwNode, inPortId);
      int outputIdx = findNodeOutputIndex(hwNode, outPortId);
      if (inputIdx >= 0 && outputIdx >= 0)
        activeTransitions.insert({inputIdx, outputIdx});
    }
  }

  for (unsigned outIdx = 0; outIdx < numOut; ++outIdx) {
    for (unsigned inIdx = 0; inIdx < numIn; ++inIdx) {
      if (!isConnectedPosition(rows, outIdx, inIdx, numIn))
        continue;
      bool enabled =
          activeTransitions.count({static_cast<int>(inIdx),
                                   static_cast<int>(outIdx)}) > 0;
      packBits(cfg.words, bitPos, enabled ? 1u : 0u, 1);
    }
  }

  return cfg;
}

GeneratedNodeConfig buildTemporalSwitchConfig(const Node *hwNode, IdIndex hwId,
                                              const MappingState &state,
                                              const Graph &dfg,
                                              const Graph &adg) {
  GeneratedNodeConfig cfg;
  if (!hwNode)
    return cfg;

  unsigned numIn = hwNode->inputPorts.size();
  unsigned numOut = hwNode->outputPorts.size();
  unsigned slotCount =
      static_cast<unsigned>(std::max<int64_t>(getNodeAttrInt(hwNode, "num_route_table", 0), 0));
  if (numIn == 0 || numOut == 0 || slotCount == 0)
    return cfg;

  auto rows = getBinaryRowsNodeAttr(hwNode, "connectivity_table");
  unsigned routeBits = countConnectedPositions(rows, numIn, numOut);
  // Hardware tag width is a native parameter of the switch interface and comes
  // directly from the tagged port type, not from any runtime tag-value scan.
  unsigned tagWidth = 0;
  for (IdIndex portId : hwNode->inputPorts) {
    const Port *port = adg.getPort(portId);
    if (!port)
      continue;
    if (auto info = detail::getPortTypeInfo(port->type)) {
      if (info->isTagged) {
        tagWidth = info->tagWidth;
        break;
      }
    }
  }
  if (tagWidth == 0)
    tagWidth = 1;

  std::map<uint64_t, std::vector<unsigned>> routesByTag;
  for (IdIndex edgeId = 0;
       edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size()); ++edgeId) {
    auto hwPath = buildExportPathForEdge(edgeId, state, dfg, adg);
    if (hwPath.size() < 3)
      continue;
    for (size_t i = 1; i + 1 < hwPath.size(); i += 2) {
      IdIndex inPortId = hwPath[i];
      IdIndex outPortId = hwPath[i + 1];
      const Port *inPort = adg.getPort(inPortId);
      const Port *outPort = adg.getPort(outPortId);
      if (!inPort || !outPort || inPort->parentNode != hwId ||
          outPort->parentNode != hwId)
        continue;

      int inputIdx = findNodeInputIndex(hwNode, inPortId);
      int outputIdx = findNodeOutputIndex(hwNode, outPortId);
      if (inputIdx < 0 || outputIdx < 0)
        continue;

      auto tag =
          computeTemporalRouteTagValue(edgeId, hwPath, i, state, dfg, adg);
      if (!tag) {
        cfg.complete = false;
        continue;
      }

      unsigned ordinal =
          connectedPositionOrdinal(rows, numIn, numOut, inputIdx, outputIdx);
      auto &routeBitsForTag = routesByTag[*tag];
      if (std::find(routeBitsForTag.begin(), routeBitsForTag.end(), ordinal) ==
          routeBitsForTag.end()) {
        routeBitsForTag.push_back(ordinal);
      }
    }
  }

  uint32_t bitPos = 0;
  unsigned emittedSlots = 0;
  for (const auto &entry : routesByTag) {
    if (emittedSlots >= slotCount) {
      cfg.complete = false;
      break;
    }
    packBits(cfg.words, bitPos, 1u, 1);
    packBits(cfg.words, bitPos, entry.first, tagWidth);
    for (unsigned ordinal = 0; ordinal < routeBits; ++ordinal) {
      bool enabled = std::find(entry.second.begin(), entry.second.end(),
                               ordinal) != entry.second.end();
      packBits(cfg.words, bitPos, enabled ? 1u : 0u, 1);
    }
    ++emittedSlots;
  }
  for (; emittedSlots < slotCount; ++emittedSlots) {
    packBits(cfg.words, bitPos, 0u, 1);
    packBits(cfg.words, bitPos, 0u, tagWidth);
    for (unsigned ordinal = 0; ordinal < routeBits; ++ordinal)
      packBits(cfg.words, bitPos, 0u, 1);
  }

  return cfg;
}

GeneratedNodeConfig buildAddTagConfig(const Node *hwNode) {
  GeneratedNodeConfig cfg;
  if (auto tag = getUIntNodeAttr(hwNode, "tag"))
    cfg.words.push_back(static_cast<uint32_t>(*tag));
  return cfg;
}

GeneratedNodeConfig buildMapTagConfig(const Node *hwNode, const Graph &adg) {
  GeneratedNodeConfig cfg;
  if (!hwNode)
    return cfg;
  unsigned tableSize =
      static_cast<unsigned>(getNodeAttrInt(hwNode, "table_size", 0));
  if (tableSize == 0)
    return cfg;

  auto [inputTagWidth, outputTagWidth] = getMapTagTagWidths(hwNode, adg);
  std::vector<uint32_t> words;
  uint32_t bitPos = 0;
  auto entries = getMapTagTableEntries(hwNode);
  if (entries.size() != tableSize)
    cfg.complete = false;
  for (unsigned idx = 0; idx < tableSize; ++idx) {
    MapTagTableEntry entry;
    if (idx < entries.size())
      entry = entries[idx];
    packBits(words, bitPos, entry.valid ? 1u : 0u, 1);
    packBits(words, bitPos, entry.srcTag, inputTagWidth);
    packBits(words, bitPos, entry.dstTag, outputTagWidth);
  }
  cfg.words = std::move(words);
  return cfg;
}

GeneratedNodeConfig buildMemoryConfig(const Node *hwNode, IdIndex hwId,
                                      const MappingState &state,
                                      const Graph &dfg, const Graph &adg) {
  GeneratedNodeConfig cfg;
  if (!hwNode)
    return cfg;
  auto addrTable = buildAddrOffsetTable(hwNode, hwId, state, dfg, adg);
  cfg.words.reserve(addrTable.size());
  for (int64_t value : addrTable)
    cfg.words.push_back(static_cast<uint32_t>(value));
  return cfg;
}

GeneratedNodeConfig buildFifoConfig(const Node *hwNode) {
  GeneratedNodeConfig cfg;
  if (!hwNode)
    return cfg;
  bool bypassable = false;
  bool bypassed = false;
  for (const auto &attr : hwNode->attributes) {
    if (attr.getName() == "bypassable") {
      if (mlir::isa<mlir::BoolAttr>(attr.getValue()))
        bypassable = mlir::cast<mlir::BoolAttr>(attr.getValue()).getValue();
    } else if (attr.getName() == "bypassed") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        bypassed = boolAttr.getValue();
    }
  }
  if (!bypassable)
    return cfg;
  cfg.words.push_back(bypassed ? 1u : 0u);
  return cfg;
}

const FUConfigSelection *
findFUConfigSelection(llvm::ArrayRef<FUConfigSelection> fuConfigs,
                      IdIndex hwNodeId) {
  for (const auto &selection : fuConfigs) {
    if (selection.hwNodeId == hwNodeId)
      return &selection;
  }
  return nullptr;
}

const Edge *findEdgeByPorts(const Graph &graph, IdIndex srcPortId,
                            IdIndex dstPortId) {
  const Port *srcPort = graph.getPort(srcPortId);
  if (!srcPort)
    return nullptr;
  for (IdIndex edgeId : srcPort->connectedEdges) {
    const Edge *edge = graph.getEdge(edgeId);
    if (edge && edge->srcPort == srcPortId && edge->dstPort == dstPortId)
      return edge;
  }
  return nullptr;
}

llvm::SmallVector<unsigned, 4> getFUConfigFieldWidths(const Node *hwNode) {
  llvm::SmallVector<unsigned, 4> widths;
  auto attr = getDenseI64NodeAttr(hwNode, "fu_config_field_widths");
  if (!attr)
    return widths;
  for (int64_t width : attr.asArrayRef()) {
    if (width > 0)
      widths.push_back(static_cast<unsigned>(width));
  }
  return widths;
}

unsigned getFUConfigBitWidth(const Node *hwNode) {
  return static_cast<unsigned>(
      std::max<int64_t>(0, getNodeAttrInt(hwNode, "fu_config_bits", 0)));
}

void packFUConfigBits(std::vector<uint32_t> &words, uint32_t &bitPos,
                      const Node *hwNode,
                      const FUConfigSelection *selection, bool &complete) {
  auto widths = getFUConfigFieldWidths(hwNode);
  if (widths.empty())
    return;

  size_t fieldCount = widths.size();
  size_t selectedCount = selection ? selection->fields.size() : 0;
  if (selection && selectedCount != fieldCount)
    complete = false;

  for (size_t i = 0; i < fieldCount; ++i) {
    unsigned width = widths[i];
    if (selection && i < selectedCount) {
      const auto &field = selection->fields[i];
      if (field.kind == FUConfigFieldKind::Mux) {
        unsigned selBits = width >= 2 ? width - 2 : 0;
        packMuxField(words, bitPos, selBits, field.sel, field.discard,
                     field.disconnect);
      } else {
        packBits(words, bitPos, field.value, width);
      }
      continue;
    }
    packBits(words, bitPos, 0u, width);
  }
}

struct PERouteSummary {
  llvm::DenseMap<IdIndex, llvm::SmallVector<int, 4>> inputPortSelects;
  llvm::DenseMap<IdIndex, llvm::SmallVector<int, 4>> outputPortSelects;
  llvm::DenseMap<IdIndex, llvm::SmallVector<uint64_t, 4>> tagsByFU;
  bool complete = true;
};

struct TemporalRegisterBinding {
  IdIndex swEdgeId = INVALID_ID;
  IdIndex writerSwNode = INVALID_ID;
  IdIndex readerSwNode = INVALID_ID;
  IdIndex writerHwNode = INVALID_ID;
  IdIndex readerHwNode = INVALID_ID;
  IdIndex srcSwPort = INVALID_ID;
  unsigned writerOutputIndex = 0;
  unsigned readerInputIndex = 0;
  unsigned registerIndex = 0;
  std::string peName;
};

struct TemporalConfigPlan {
  llvm::SmallVector<std::pair<unsigned, IdIndex>, 8> usedFUs;
  llvm::DenseMap<IdIndex, unsigned> slotByFU;
  llvm::DenseMap<IdIndex, llvm::SmallVector<std::optional<unsigned>, 4>>
      operandRegsByFU;
  llvm::DenseMap<IdIndex, llvm::SmallVector<std::optional<unsigned>, 4>>
      resultRegsByFU;
  llvm::SmallVector<TemporalRegisterBinding, 8> registerBindings;
  bool complete = true;
};

TemporalConfigPlan buildTemporalConfigPlan(
    const PEContainment &pe, const MappingState &state, const Graph &dfg,
    const Graph &adg, llvm::ArrayRef<TechMappedEdgeKind> edgeKinds) {
  TemporalConfigPlan plan;

  for (unsigned ordinal = 0; ordinal < pe.fuNodeIds.size(); ++ordinal) {
    IdIndex fuId = pe.fuNodeIds[ordinal];
    const Node *fuNode = adg.getNode(fuId);
    if (!fuNode)
      continue;
    plan.operandRegsByFU[fuId].assign(fuNode->inputPorts.size(), std::nullopt);
    plan.resultRegsByFU[fuId].assign(fuNode->outputPorts.size(), std::nullopt);
    if (fuId < state.hwNodeToSwNodes.size() && !state.hwNodeToSwNodes[fuId].empty()) {
      plan.slotByFU[fuId] = static_cast<unsigned>(plan.usedFUs.size());
      plan.usedFUs.push_back({ordinal, fuId});
    }
  }

  llvm::DenseMap<IdIndex, unsigned> regBySrcPort;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeId >= edgeKinds.size() ||
        edgeKinds[edgeId] != TechMappedEdgeKind::TemporalReg)
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;

    IdIndex writerSwNode = srcPort->parentNode;
    IdIndex readerSwNode = dstPort->parentNode;
    if (writerSwNode >= state.swNodeToHwNode.size() ||
        readerSwNode >= state.swNodeToHwNode.size())
      continue;
    IdIndex writerHwNode = state.swNodeToHwNode[writerSwNode];
    IdIndex readerHwNode = state.swNodeToHwNode[readerSwNode];
    if (writerHwNode == INVALID_ID || readerHwNode == INVALID_ID)
      continue;
    const Node *writerHw = adg.getNode(writerHwNode);
    const Node *readerHw = adg.getNode(readerHwNode);
    if (!writerHw || !readerHw)
      continue;
    llvm::StringRef writerPE = getNodeAttrStr(writerHw, "pe_name");
    if (writerPE.empty() || writerPE != pe.peName ||
        writerPE != getNodeAttrStr(readerHw, "pe_name"))
      continue;

    int writerOutputIdx = findNodeOutputIndex(dfg.getNode(writerSwNode), edge->srcPort);
    int readerInputIdx = findNodeInputIndex(dfg.getNode(readerSwNode), edge->dstPort);
    if (writerOutputIdx < 0 || readerInputIdx < 0) {
      plan.complete = false;
      continue;
    }

    unsigned regIdx = 0;
    auto foundReg = regBySrcPort.find(edge->srcPort);
    if (foundReg == regBySrcPort.end()) {
      regIdx = static_cast<unsigned>(regBySrcPort.size());
      regBySrcPort[edge->srcPort] = regIdx;
    } else {
      regIdx = foundReg->second;
    }

    if (regIdx >= pe.numRegister) {
      plan.complete = false;
      continue;
    }

    auto &resultRegs = plan.resultRegsByFU[writerHwNode];
    auto &operandRegs = plan.operandRegsByFU[readerHwNode];
    if (static_cast<size_t>(writerOutputIdx) >= resultRegs.size() ||
        static_cast<size_t>(readerInputIdx) >= operandRegs.size()) {
      plan.complete = false;
      continue;
    }
    if (resultRegs[writerOutputIdx].has_value() &&
        *resultRegs[writerOutputIdx] != regIdx)
      plan.complete = false;
    if (operandRegs[readerInputIdx].has_value() &&
        *operandRegs[readerInputIdx] != regIdx)
      plan.complete = false;

    resultRegs[writerOutputIdx] = regIdx;
    operandRegs[readerInputIdx] = regIdx;

    TemporalRegisterBinding binding;
    binding.swEdgeId = edgeId;
    binding.writerSwNode = writerSwNode;
    binding.readerSwNode = readerSwNode;
    binding.writerHwNode = writerHwNode;
    binding.readerHwNode = readerHwNode;
    binding.srcSwPort = edge->srcPort;
    binding.writerOutputIndex = static_cast<unsigned>(writerOutputIdx);
    binding.readerInputIndex = static_cast<unsigned>(readerInputIdx);
    binding.registerIndex = regIdx;
    binding.peName = pe.peName;
    plan.registerBindings.push_back(std::move(binding));
  }

  return plan;
}

PERouteSummary collectPERouteSummary(const PEContainment &pe,
                                     const MappingState &state,
                                     const Graph &dfg, const Graph &adg) {
  PERouteSummary summary;
  llvm::DenseSet<IdIndex> fuSet(pe.fuNodeIds.begin(), pe.fuNodeIds.end());

  for (IdIndex fuId : pe.fuNodeIds) {
    const Node *fuNode = adg.getNode(fuId);
    if (!fuNode)
      continue;
    summary.inputPortSelects[fuId].assign(fuNode->inputPorts.size(), -1);
    summary.outputPortSelects[fuId].assign(fuNode->outputPorts.size(), -1);
  }

  for (IdIndex swEdgeId = 0;
       swEdgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++swEdgeId) {
    auto hwPath = buildExportPathForEdge(swEdgeId, state, dfg, adg);
    if (hwPath.size() < 2)
      continue;

    for (size_t i = 0; i + 1 < hwPath.size(); ++i) {
      IdIndex srcPortId = hwPath[i];
      IdIndex dstPortId = hwPath[i + 1];
      const Port *srcPort = adg.getPort(srcPortId);
      const Port *dstPort = adg.getPort(dstPortId);
      if (!srcPort || !dstPort)
        continue;

      const Edge *flatEdge = findEdgeByPorts(adg, srcPortId, dstPortId);
      if (!flatEdge)
        continue;

      if (dstPort->parentNode != INVALID_ID && fuSet.contains(dstPort->parentNode)) {
        auto peInputIndex = getUIntEdgeAttr(flatEdge, "pe_input_index");
        if (peInputIndex) {
          const Node *fuNode = adg.getNode(dstPort->parentNode);
          int inputIdx = findNodeInputIndex(fuNode, dstPortId);
          if (fuNode && inputIdx >= 0) {
            auto &slots = summary.inputPortSelects[dstPort->parentNode];
            if (static_cast<size_t>(inputIdx) < slots.size()) {
              if (slots[inputIdx] >= 0 &&
                  slots[inputIdx] != static_cast<int>(*peInputIndex))
                summary.complete = false;
              slots[inputIdx] = static_cast<int>(*peInputIndex);
            }
            if (pe.peKind == "temporal_pe") {
              if (auto tag =
                      computeTemporalRouteTagValue(swEdgeId, hwPath, i, state,
                                                   dfg, adg)) {
                auto &tags = summary.tagsByFU[dstPort->parentNode];
                if (std::find(tags.begin(), tags.end(), *tag) == tags.end())
                  tags.push_back(*tag);
              }
            }
          }
        }
      }

      if (srcPort->parentNode != INVALID_ID && fuSet.contains(srcPort->parentNode)) {
        auto peOutputIndex = getUIntEdgeAttr(flatEdge, "pe_output_index");
        if (peOutputIndex) {
          const Node *fuNode = adg.getNode(srcPort->parentNode);
          int outputIdx = findNodeOutputIndex(fuNode, srcPortId);
          if (fuNode && outputIdx >= 0) {
            auto &slots = summary.outputPortSelects[srcPort->parentNode];
            if (static_cast<size_t>(outputIdx) < slots.size()) {
              if (slots[outputIdx] >= 0 &&
                  slots[outputIdx] != static_cast<int>(*peOutputIndex))
                summary.complete = false;
              slots[outputIdx] = static_cast<int>(*peOutputIndex);
            }
            if (pe.peKind == "temporal_pe") {
              if (auto tag = computeTemporalRouteTagValue(swEdgeId, hwPath,
                                                          i + 1, state, dfg,
                                                          adg)) {
                auto &tags = summary.tagsByFU[srcPort->parentNode];
                if (std::find(tags.begin(), tags.end(), *tag) == tags.end())
                  tags.push_back(*tag);
              }
            }
          }
        }
      }
    }
  }

  return summary;
}

GeneratedNodeConfig
buildFunctionUnitConfig(llvm::ArrayRef<FUConfigSelection> fuConfigs,
                        IdIndex hwId) {
  GeneratedNodeConfig cfg;
  const FUConfigSelection *selection = findFUConfigSelection(fuConfigs, hwId);
  if (!selection)
    return cfg;
  for (const auto &field : selection->fields) {
    uint32_t word = 0;
    if (field.kind == FUConfigFieldKind::Mux) {
      word = static_cast<uint32_t>(field.sel & 0xffffu);
      if (field.discard)
        word |= (1u << 16);
      if (field.disconnect)
        word |= (1u << 17);
    } else {
      word = static_cast<uint32_t>(field.value & 0xffffffffu);
    }
    cfg.words.push_back(word);
  }
  return cfg;
}

GeneratedNodeConfig buildSpatialPEConfig(const PEContainment &pe,
                                         const MappingState &state,
                                         const Graph &dfg, const Graph &adg,
                                         llvm::ArrayRef<FUConfigSelection> fuConfigs,
                                         bool &globalComplete) {
  GeneratedNodeConfig cfg;
  bool complete = true;
  unsigned fuCount = pe.fuNodeIds.size();
  if (fuCount == 0)
    return cfg;

  unsigned opcodeBits = bitWidthForChoices(fuCount);
  unsigned peInputSelBits = bitWidthForChoices(pe.numInputPorts);
  unsigned peOutputSelBits = bitWidthForChoices(pe.numOutputPorts);

  unsigned maxFuInputs = 0;
  unsigned maxFuOutputs = 0;
  unsigned maxFuConfigBits = 0;
  IdIndex activeFuId = INVALID_ID;
  unsigned activeOpcode = 0;
  unsigned usedFuCount = 0;
  for (unsigned ordinal = 0; ordinal < fuCount; ++ordinal) {
    IdIndex fuId = pe.fuNodeIds[ordinal];
    const Node *fuNode = adg.getNode(fuId);
    if (!fuNode)
      continue;
    maxFuInputs = std::max<unsigned>(maxFuInputs, fuNode->inputPorts.size());
    maxFuOutputs = std::max<unsigned>(maxFuOutputs, fuNode->outputPorts.size());
    maxFuConfigBits = std::max<unsigned>(maxFuConfigBits, getFUConfigBitWidth(fuNode));
    if (fuId < state.hwNodeToSwNodes.size() && !state.hwNodeToSwNodes[fuId].empty()) {
      ++usedFuCount;
      if (activeFuId == INVALID_ID) {
        activeFuId = fuId;
        activeOpcode = ordinal;
      }
    }
  }

  if (usedFuCount > 1)
    complete = false;

  PERouteSummary routes;
  if (activeFuId != INVALID_ID)
    routes = collectPERouteSummary(pe, state, dfg, adg);

  uint32_t bitPos = 0;
  bool enabled = activeFuId != INVALID_ID;
  packBits(cfg.words, bitPos, enabled ? 1u : 0u, 1);
  packBits(cfg.words, bitPos, activeOpcode, opcodeBits);

  const Node *activeFu = enabled ? adg.getNode(activeFuId) : nullptr;
  auto inputSelects =
      (enabled && routes.inputPortSelects.count(activeFuId))
          ? routes.inputPortSelects.lookup(activeFuId)
          : llvm::SmallVector<int, 4>();
  auto outputSelects =
      (enabled && routes.outputPortSelects.count(activeFuId))
          ? routes.outputPortSelects.lookup(activeFuId)
          : llvm::SmallVector<int, 4>();

  for (unsigned inputIdx = 0; inputIdx < maxFuInputs; ++inputIdx) {
    bool disconnect = true;
    bool discard = false;
    uint64_t sel = 0;
    if (enabled && activeFu && inputIdx < activeFu->inputPorts.size()) {
      if (inputIdx < inputSelects.size() && inputSelects[inputIdx] >= 0) {
        sel = static_cast<uint64_t>(inputSelects[inputIdx]);
        disconnect = false;
      }
    }
    packMuxField(cfg.words, bitPos, peInputSelBits, sel, discard, disconnect);
  }

  for (unsigned outputIdx = 0; outputIdx < maxFuOutputs; ++outputIdx) {
    bool disconnect = true;
    bool discard = false;
    uint64_t sel = 0;
    if (enabled && activeFu && outputIdx < activeFu->outputPorts.size()) {
      if (outputIdx < outputSelects.size() && outputSelects[outputIdx] >= 0) {
        sel = static_cast<uint64_t>(outputSelects[outputIdx]);
        disconnect = false;
      } else {
        disconnect = false;
        discard = true;
      }
    }
    packMuxField(cfg.words, bitPos, peOutputSelBits, sel, discard, disconnect);
  }

  if (maxFuConfigBits > 0) {
    if (enabled && activeFu) {
      const FUConfigSelection *selection =
          findFUConfigSelection(fuConfigs, activeFuId);
      uint32_t configStart = bitPos;
      packFUConfigBits(cfg.words, bitPos, activeFu, selection, complete);
      unsigned actualBits = bitPos - configStart;
      if (actualBits < maxFuConfigBits)
        packBits(cfg.words, bitPos, 0u, maxFuConfigBits - actualBits);
      else if (actualBits > maxFuConfigBits)
        complete = false;
    } else {
      packBits(cfg.words, bitPos, 0u, maxFuConfigBits);
    }
  }

  cfg.complete = complete;
  globalComplete = globalComplete && complete;
  return cfg;
}

GeneratedNodeConfig buildTemporalPEConfig(const PEContainment &pe,
                                          const MappingState &state,
                                          const Graph &dfg, const Graph &adg,
                                          llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                          llvm::ArrayRef<FUConfigSelection> fuConfigs,
                                          bool &globalComplete) {
  GeneratedNodeConfig cfg;
  bool complete = true;
  unsigned fuCount = pe.fuNodeIds.size();
  if (fuCount == 0 || pe.numInstruction == 0)
    return cfg;

  unsigned opcodeBits = bitWidthForChoices(fuCount);
  unsigned regIdxBits = bitWidthForChoices(pe.numRegister);
  unsigned operandCfgWidth = pe.numRegister > 0 ? (1 + regIdxBits) : 0;
  unsigned inputMuxWidth = bitWidthForChoices(pe.numInputPorts) + 2;
  unsigned outputMuxWidth = bitWidthForChoices(pe.numOutputPorts) + 2;
  unsigned resultCfgWidth =
      pe.tagWidth + (pe.numRegister > 0 ? (1 + regIdxBits) : 0);

  unsigned maxFuInputs = 0;
  unsigned maxFuOutputs = 0;
  for (IdIndex fuId : pe.fuNodeIds) {
    const Node *fuNode = adg.getNode(fuId);
    if (!fuNode)
      continue;
    maxFuInputs = std::max<unsigned>(maxFuInputs, fuNode->inputPorts.size());
    maxFuOutputs = std::max<unsigned>(maxFuOutputs, fuNode->outputPorts.size());
  }

  PERouteSummary routes = collectPERouteSummary(pe, state, dfg, adg);
  complete = complete && routes.complete;
  TemporalConfigPlan temporalPlan =
      buildTemporalConfigPlan(pe, state, dfg, adg, edgeKinds);
  complete = complete && temporalPlan.complete;
  auto &usedFUs = temporalPlan.usedFUs;
  if (usedFUs.size() > pe.numInstruction)
    complete = false;

  uint32_t bitPos = 0;
  for (unsigned slot = 0; slot < pe.numInstruction; ++slot) {
    bool valid = slot < usedFUs.size();
    uint64_t tag = 0;
    unsigned opcode = 0;
    const Node *fuNode = nullptr;
    llvm::SmallVector<int, 4> inputSelects;
    llvm::SmallVector<int, 4> outputSelects;

    if (valid) {
      opcode = usedFUs[slot].first;
      IdIndex fuId = usedFUs[slot].second;
      fuNode = adg.getNode(fuId);
      if (routes.inputPortSelects.count(fuId))
        inputSelects = routes.inputPortSelects.lookup(fuId);
      if (routes.outputPortSelects.count(fuId))
        outputSelects = routes.outputPortSelects.lookup(fuId);
      auto tags = routes.tagsByFU.lookup(fuId);
      if (tags.empty()) {
        complete = false;
      } else {
        tag = tags.front();
        if (tags.size() > 1)
          complete = false;
      }
    }

    packBits(cfg.words, bitPos, valid ? 1u : 0u, 1);
    packBits(cfg.words, bitPos, tag, pe.tagWidth);
    packBits(cfg.words, bitPos, opcode, opcodeBits);

    for (unsigned operandIdx = 0; operandIdx < maxFuInputs; ++operandIdx) {
      if (operandCfgWidth == 0)
        continue;
      uint64_t regIdx = 0;
      bool isReg = false;
      if (valid && fuNode) {
        auto found = temporalPlan.operandRegsByFU.find(usedFUs[slot].second);
        if (found != temporalPlan.operandRegsByFU.end() &&
            operandIdx < found->second.size() &&
            found->second[operandIdx].has_value()) {
          regIdx = *found->second[operandIdx];
          isReg = true;
        }
      }
      packBits(cfg.words, bitPos, regIdx, regIdxBits);
      packBits(cfg.words, bitPos, isReg ? 1u : 0u, 1);
    }

    for (unsigned inputIdx = 0; inputIdx < maxFuInputs; ++inputIdx) {
      bool disconnect = true;
      bool discard = false;
      uint64_t sel = 0;
      bool isReg = false;
      if (valid && fuNode) {
        auto found = temporalPlan.operandRegsByFU.find(usedFUs[slot].second);
        if (found != temporalPlan.operandRegsByFU.end() &&
            inputIdx < found->second.size() &&
            found->second[inputIdx].has_value()) {
          isReg = true;
        }
      }
      if (!isReg && valid && fuNode && inputIdx < fuNode->inputPorts.size()) {
        if (inputIdx < inputSelects.size() && inputSelects[inputIdx] >= 0) {
          sel = static_cast<uint64_t>(inputSelects[inputIdx]);
          disconnect = false;
        }
      }
      packMuxField(cfg.words, bitPos, inputMuxWidth - 2, sel, discard,
                   disconnect);
    }

    for (unsigned outputIdx = 0; outputIdx < maxFuOutputs; ++outputIdx) {
      bool disconnect = true;
      bool discard = false;
      uint64_t sel = 0;
      if (valid && fuNode && outputIdx < fuNode->outputPorts.size()) {
        if (outputIdx < outputSelects.size() && outputSelects[outputIdx] >= 0) {
          sel = static_cast<uint64_t>(outputSelects[outputIdx]);
          disconnect = false;
        } else {
          disconnect = false;
          discard = true;
        }
      }
      packMuxField(cfg.words, bitPos, outputMuxWidth - 2, sel, discard,
                   disconnect);
    }

    for (unsigned resultIdx = 0; resultIdx < maxFuOutputs; ++resultIdx) {
      if (resultCfgWidth == 0)
        continue;
      uint64_t resultTag = 0;
      uint64_t regIdx = 0;
      bool isReg = false;
      if (valid && fuNode) {
        auto found = temporalPlan.resultRegsByFU.find(usedFUs[slot].second);
        if (found != temporalPlan.resultRegsByFU.end() &&
            resultIdx < found->second.size() &&
            found->second[resultIdx].has_value()) {
          regIdx = *found->second[resultIdx];
          isReg = true;
        }
      }
      if (isReg) {
        resultTag = 0;
        if (resultIdx < outputSelects.size() && outputSelects[resultIdx] >= 0)
          complete = false;
      } else if (valid && fuNode && resultIdx < fuNode->outputPorts.size() &&
                 resultIdx < outputSelects.size() &&
                 outputSelects[resultIdx] >= 0) {
        resultTag = tag;
      }
      packBits(cfg.words, bitPos, resultTag, pe.tagWidth);
      if (pe.numRegister > 0) {
        packBits(cfg.words, bitPos, regIdx, regIdxBits);
        packBits(cfg.words, bitPos, isReg ? 1u : 0u, 1);
      }
    }
  }

  for (IdIndex fuId : pe.fuNodeIds) {
    const Node *fuNode = adg.getNode(fuId);
    const FUConfigSelection *selection = findFUConfigSelection(fuConfigs, fuId);
    packFUConfigBits(cfg.words, bitPos, fuNode, selection, complete);
  }

  cfg.complete = complete;
  globalComplete = globalComplete && complete;
  return cfg;
}

} // namespace

bool ConfigGen::buildConfigArtifacts(const MappingState &state,
                                     const Graph &dfg, const Graph &adg,
                                     const ADGFlattener &flattener,
                                     llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                     llvm::ArrayRef<FUConfigSelection> fuConfigs) {
  nodeConfigs_.clear();
  configSlices_.clear();
  configWords_.clear();
  configBlob_.clear();
  configComplete_ = true;

  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size()); ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || hwNode->kind != Node::OperationNode)
      continue;

    GeneratedNodeConfig generated;
    llvm::StringRef resourceClass = getNodeAttrStr(hwNode, "resource_class");
    llvm::StringRef opKind = getNodeAttrStr(hwNode, "op_kind");

    if (resourceClass == "routing") {
      if (opKind == "spatial_sw")
        generated = buildSpatialSwitchConfig(hwNode, hwId, state, dfg, adg);
      else if (opKind == "temporal_sw")
        generated = buildTemporalSwitchConfig(hwNode, hwId, state, dfg, adg);
      else if (opKind == "add_tag")
        generated = buildAddTagConfig(hwNode);
      else if (opKind == "map_tag")
        generated = buildMapTagConfig(hwNode, adg);
      else if (opKind == "fifo")
        generated = buildFifoConfig(hwNode);
      else
        continue;
    } else if (resourceClass == "memory") {
      generated = buildMemoryConfig(hwNode, hwId, state, dfg, adg);
    } else {
      continue;
    }

    if (generated.words.empty())
      continue;

    NodeConfig nodeCfg;
    nodeCfg.name = getNodeAttrStr(hwNode, "op_name").str();
    nodeCfg.kind = opKind.str();
    nodeCfg.hwNode = hwId;
    nodeCfg.complete = generated.complete;
    nodeCfg.words = std::move(generated.words);
    nodeConfigs_.push_back(nodeCfg);

    ConfigSlice slice;
    slice.name = nodeCfg.name;
    slice.kind = nodeCfg.kind;
    slice.hwNode = hwId;
    slice.wordOffset = static_cast<uint32_t>(configWords_.size());
    slice.wordCount = static_cast<uint32_t>(nodeCfg.words.size());
    slice.complete = nodeCfg.complete;
    configSlices_.push_back(slice);

    configWords_.insert(configWords_.end(), nodeCfg.words.begin(),
                        nodeCfg.words.end());
    configComplete_ = configComplete_ && nodeCfg.complete;
  }

  for (const auto &pe : flattener.getPEContainment()) {
    GeneratedNodeConfig generated;
    if (pe.peKind == "spatial_pe") {
      generated = buildSpatialPEConfig(pe, state, dfg, adg, fuConfigs,
                                       configComplete_);
    } else if (pe.peKind == "temporal_pe") {
      generated = buildTemporalPEConfig(pe, state, dfg, adg, edgeKinds,
                                        fuConfigs,
                                        configComplete_);
    } else {
      continue;
    }

    if (generated.words.empty())
      continue;

    NodeConfig nodeCfg;
    nodeCfg.name = pe.peName;
    nodeCfg.kind = pe.peKind;
    nodeCfg.hwNode = INVALID_ID;
    nodeCfg.complete = generated.complete;
    nodeCfg.words = std::move(generated.words);
    nodeConfigs_.push_back(nodeCfg);

    ConfigSlice slice;
    slice.name = nodeCfg.name;
    slice.kind = nodeCfg.kind;
    slice.hwNode = INVALID_ID;
    slice.wordOffset = static_cast<uint32_t>(configWords_.size());
    slice.wordCount = static_cast<uint32_t>(nodeCfg.words.size());
    slice.complete = nodeCfg.complete;
    configSlices_.push_back(slice);

    configWords_.insert(configWords_.end(), nodeCfg.words.begin(),
                        nodeCfg.words.end());
  }

  configBlob_.reserve(configWords_.size() * sizeof(uint32_t));
  for (uint32_t word : configWords_) {
    configBlob_.push_back(static_cast<uint8_t>(word & 0xffu));
    configBlob_.push_back(static_cast<uint8_t>((word >> 8) & 0xffu));
    configBlob_.push_back(static_cast<uint8_t>((word >> 16) & 0xffu));
    configBlob_.push_back(static_cast<uint8_t>((word >> 24) & 0xffu));
  }

  return true;
}

bool ConfigGen::writeConfigBinary(const std::string &path) const {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "ConfigGen: cannot open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }
  if (!configBlob_.empty())
    out.write(reinterpret_cast<const char *>(configBlob_.data()),
              configBlob_.size());
  return true;
}

bool ConfigGen::writeConfigJson(const std::string &path) const {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "ConfigGen: cannot open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }

  out << "{\n";
  out << "  \"word_width_bits\": 32,\n";
  out << "  \"word_count\": " << configWords_.size() << ",\n";
  out << "  \"byte_size\": " << configBlob_.size() << ",\n";
  out << "  \"complete\": " << (configComplete_ ? "true" : "false") << ",\n";
  out << "  \"coverage_note\": "
      << (configComplete_
              ? "\"all currently modeled configurable slices serialized\""
              : "\"routing, tag, memory, spatial_pe, and temporal_pe slices serialized; some slice contents remain partially modeled\"")
      << ",\n";
  out << "  \"words\": [";
  for (size_t i = 0; i < configWords_.size(); ++i) {
    if (i > 0)
      out << ", ";
    out << configWords_[i];
  }
  out << "],\n";
  out << "  \"slices\": [\n";
  for (size_t i = 0; i < configSlices_.size(); ++i) {
    const auto &slice = configSlices_[i];
    if (i > 0)
      out << ",\n";
    out << "    {\"name\": \"" << slice.name << "\", \"kind\": \""
        << slice.kind << "\", \"hw_node\": ";
    if (slice.hwNode == INVALID_ID)
      out << "null";
    else
      out << slice.hwNode;
    out
        << ", \"word_offset\": " << slice.wordOffset
        << ", \"word_count\": " << slice.wordCount
        << ", \"complete\": " << (slice.complete ? "true" : "false") << "}";
  }
  out << "\n  ]\n";
  out << "}\n";
  return true;
}

bool ConfigGen::writeConfigHeader(const std::string &path) const {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "ConfigGen: cannot open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }

  std::string guard = buildHeaderGuard(path);

  out << "#ifndef " << guard << "\n";
  out << "#define " << guard << "\n\n";
  out << "#include <stdint.h>\n\n";
  out << "static const uint32_t fcc_accel_config_words[] = {";
  if (configWords_.empty()) {
    out << "0";
  } else {
    for (size_t i = 0; i < configWords_.size(); ++i) {
      if (i == 0)
        out << "\n    ";
      else if ((i % 6) == 0)
        out << ",\n    ";
      else
        out << ", ";
      out << "0x";
      out.write_hex(configWords_[i]);
    }
    out << "\n";
  }
  out << "};\n";
  out << "static const unsigned fcc_accel_config_word_count = "
      << configWords_.size() << ";\n";
  out << "static const int fcc_accel_config_complete = "
      << (configComplete_ ? 1 : 0) << ";\n\n";
  out << "#endif\n";
  return true;
}

bool ConfigGen::generate(const MappingState &state, const Graph &dfg,
                         const Graph &adg, const ADGFlattener &flattener,
                         llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                         llvm::ArrayRef<FUConfigSelection> fuConfigs,
                         const std::string &basePath, int seed) {
  if (!buildConfigArtifacts(state, dfg, adg, flattener, edgeKinds, fuConfigs))
    return false;
  if (!writeConfigBinary(basePath + ".config.bin"))
    return false;
  if (!writeConfigJson(basePath + ".config.json"))
    return false;
  if (!writeConfigHeader(getConfigHeaderFilename(basePath)))
    return false;
  if (!writeMapJson(state, dfg, adg, flattener, edgeKinds, fuConfigs,
                    basePath + ".map.json", seed))
    return false;
  if (!writeMapText(state, dfg, adg, flattener, edgeKinds,
                    basePath + ".map.txt"))
    return false;
  return true;
}

bool ConfigGen::writeMapJson(const MappingState &state, const Graph &dfg,
                             const Graph &adg, const ADGFlattener &flattener,
                             llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                             llvm::ArrayRef<FUConfigSelection> fuConfigs,
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

bool ConfigGen::writeMapText(const MappingState &state, const Graph &dfg,
                             const Graph &adg, const ADGFlattener &flattener,
                             llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
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

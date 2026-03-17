#include "fcc/Mapper/BridgeBinding.h"

#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace fcc {

namespace {

mlir::DenseI32ArrayAttr getDenseI32ArrayAttr(const Node *node,
                                             llvm::StringRef name) {
  if (!node)
    return {};
  for (auto &attr : node->attributes) {
    if (attr.getName() == name)
      return mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
  }
  return {};
}

void appendPortArray(mlir::DenseI32ArrayAttr attr,
                     llvm::SmallVectorImpl<IdIndex> &dst) {
  if (!attr)
    return;
  for (int32_t v : attr.asArrayRef())
    dst.push_back(static_cast<IdIndex>(v));
}

void appendCategoryArray(mlir::DenseI32ArrayAttr attr,
                         llvm::SmallVectorImpl<BridgePortCategory> &dst) {
  if (!attr)
    return;
  for (int32_t v : attr.asArrayRef())
    dst.push_back(static_cast<BridgePortCategory>(v));
}

void appendLaneArray(mlir::DenseI32ArrayAttr attr,
                     llvm::SmallVectorImpl<unsigned> &dst) {
  if (!attr)
    return;
  for (int32_t v : attr.asArrayRef())
    dst.push_back(static_cast<unsigned>(v));
}

std::optional<unsigned>
findLaneForPort(IdIndex hwPort, llvm::ArrayRef<IdIndex> ports,
                llvm::ArrayRef<unsigned> lanes) {
  for (size_t i = 0; i < ports.size() && i < lanes.size(); ++i) {
    if (ports[i] == hwPort)
      return lanes[i];
  }
  return std::nullopt;
}

IdIndex inferNodeId(const Node *node, const Graph &graph) {
  if (!node)
    return INVALID_ID;
  for (IdIndex portId : node->inputPorts) {
    const Port *port = graph.getPort(portId);
    if (port && port->parentNode != INVALID_ID)
      return port->parentNode;
  }
  for (IdIndex portId : node->outputPorts) {
    const Port *port = graph.getPort(portId);
    if (port && port->parentNode != INVALID_ID)
      return port->parentNode;
  }
  return INVALID_ID;
}

unsigned getMaxBridgeLane(const BridgeInfo &bridge) {
  unsigned maxLane = 0;
  for (unsigned lane : bridge.inputLanes)
    maxLane = std::max(maxLane, lane);
  for (unsigned lane : bridge.outputLanes)
    maxLane = std::max(maxLane, lane);
  return maxLane;
}

bool rangesOverlap(unsigned lhsStart, unsigned lhsEnd, unsigned rhsStart,
                   unsigned rhsEnd) {
  return lhsStart < rhsEnd && rhsStart < lhsEnd;
}

std::optional<IdIndex> findCompatibleBridgePort(
    llvm::ArrayRef<IdIndex> ports,
    llvm::ArrayRef<BridgePortCategory> categories,
    llvm::ArrayRef<unsigned> lanes, BridgePortCategory targetCategory,
    unsigned targetLane, mlir::Type swType, const Graph &adg,
    const MappingState &state) {
  for (size_t i = 0; i < ports.size() && i < categories.size() &&
                     i < lanes.size();
       ++i) {
    if (categories[i] != targetCategory || lanes[i] != targetLane)
      continue;
    IdIndex hwPid = ports[i];
    if (!state.hwPortToSwPorts[hwPid].empty())
      continue;
    const Port *hp = adg.getPort(hwPid);
    if (hp && canMapSoftwareTypeToHardware(swType, hp->type))
      return hwPid;
  }
  return std::nullopt;
}

bool isMappedPortCompatible(IdIndex mappedHwPort, llvm::ArrayRef<IdIndex> ports,
                            llvm::ArrayRef<BridgePortCategory> categories,
                            llvm::ArrayRef<unsigned> lanes,
                            BridgePortCategory targetCategory,
                            unsigned targetLane, mlir::Type swType,
                            const Graph &adg) {
  for (size_t i = 0; i < ports.size() && i < categories.size() &&
                     i < lanes.size();
       ++i) {
    if (ports[i] != mappedHwPort)
      continue;
    if (categories[i] != targetCategory || lanes[i] != targetLane)
      return false;
    const Port *hp = adg.getPort(mappedHwPort);
    return hp && canMapSoftwareTypeToHardware(swType, hp->type);
  }
  return false;
}

bool canBindInputsAtBase(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                         const Node *swNode, unsigned baseLane,
                         const Graph &dfg, const Graph &adg,
                         const MappingState &state) {
  for (unsigned si = mem.swInSkip; si < swNode->inputPorts.size(); ++si) {
    IdIndex swPortId = swNode->inputPorts[si];
    const Port *sp = dfg.getPort(swPortId);
    if (!sp)
      continue;
    BridgePortCategory cat = mem.classifyInput(si - mem.swInSkip);
    unsigned lane = baseLane + mem.inputLocalLane(si - mem.swInSkip);
    IdIndex mappedHwPort =
        swPortId < state.swPortToHwPort.size() ? state.swPortToHwPort[swPortId]
                                               : INVALID_ID;
    if (mappedHwPort != INVALID_ID) {
      if (!isMappedPortCompatible(mappedHwPort, bridge.inputPorts,
                                  bridge.inputCategories, bridge.inputLanes,
                                  cat, lane, sp->type, adg))
        return false;
      continue;
    }
    if (!findCompatibleBridgePort(bridge.inputPorts, bridge.inputCategories,
                                  bridge.inputLanes, cat, lane, sp->type, adg,
                                  state)) {
      return false;
    }
  }
  return true;
}

bool canBindOutputsAtBase(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                          const Node *swNode, unsigned baseLane,
                          const Graph &dfg, const Graph &adg,
                          const MappingState &state) {
  for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
    IdIndex swPortId = swNode->outputPorts[oi];
    const Port *sp = dfg.getPort(swPortId);
    if (!sp)
      continue;
    BridgePortCategory cat = mem.classifyOutput(oi);
    unsigned lane = baseLane + mem.outputLocalLane(oi);
    IdIndex mappedHwPort =
        swPortId < state.swPortToHwPort.size() ? state.swPortToHwPort[swPortId]
                                               : INVALID_ID;
    if (mappedHwPort != INVALID_ID) {
      if (!isMappedPortCompatible(mappedHwPort, bridge.outputPorts,
                                  bridge.outputCategories, bridge.outputLanes,
                                  cat, lane, sp->type, adg))
        return false;
      continue;
    }
    if (!findCompatibleBridgePort(bridge.outputPorts, bridge.outputCategories,
                                  bridge.outputLanes, cat, lane, sp->type, adg,
                                  state)) {
      return false;
    }
  }
  return true;
}

bool overlapsExistingLaneRanges(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                                const Node *swNode, IdIndex hwNodeId,
                                unsigned baseLane, const Graph &dfg,
                                const MappingState &state) {
  if (hwNodeId == INVALID_ID || hwNodeId >= state.hwNodeToSwNodes.size())
    return false;

  unsigned endLane = baseLane + mem.laneSpan();
  for (IdIndex otherSwId : state.hwNodeToSwNodes[hwNodeId]) {
    const Node *otherSwNode = dfg.getNode(otherSwId);
    if (!otherSwNode || otherSwNode == swNode)
      continue;

    bool isExtMem =
        (getNodeAttrStr(otherSwNode, "op_name") == "handshake.extmemory");
    DfgMemoryInfo otherMem = DfgMemoryInfo::extract(otherSwNode, dfg, isExtMem);
    auto otherRange = inferBridgeLaneRange(bridge, otherMem, otherSwNode, state);
    if (!otherRange)
      continue;
    if (rangesOverlap(baseLane, endLane, otherRange->start, otherRange->end))
      return true;
  }
  return false;
}

std::optional<unsigned> chooseBridgeBaseLane(const BridgeInfo &bridge,
                                             const DfgMemoryInfo &mem,
                                             const Node *swNode,
                                             const Node *hwNode,
                                             const Graph &dfg,
                                             const Graph &adg,
                                             const MappingState &state) {
  if (!swNode || !hwNode || !bridge.hasBridge)
    return std::nullopt;

  if (auto existing = inferBridgeLaneRange(bridge, mem, swNode, state))
    return existing->start;

  unsigned span = mem.laneSpan();
  unsigned maxLane = getMaxBridgeLane(bridge);
  if (span == 0 || maxLane + 1 < span)
    return std::nullopt;

  IdIndex hwNodeId = inferNodeId(hwNode, adg);
  auto canUseBaseLane = [&](unsigned baseLane) {
    if (overlapsExistingLaneRanges(bridge, mem, swNode, hwNodeId, baseLane, dfg,
                                   state))
      return false;
    if (!canBindInputsAtBase(bridge, mem, swNode, baseLane, dfg, adg, state))
      return false;
    if (!canBindOutputsAtBase(bridge, mem, swNode, baseLane, dfg, adg, state))
      return false;
    return true;
  };

  int64_t preferredBase = getNodeAttrInt(swNode, "id", -1);
  if (preferredBase >= 0 &&
      static_cast<unsigned>(preferredBase) + span - 1 <= maxLane &&
      canUseBaseLane(static_cast<unsigned>(preferredBase))) {
    return static_cast<unsigned>(preferredBase);
  }

  for (unsigned baseLane = 0; baseLane + span - 1 <= maxLane; ++baseLane) {
    if (!canUseBaseLane(baseLane))
      continue;
    return baseLane;
  }
  return std::nullopt;
}

} // namespace

BridgeInfo BridgeInfo::extract(const Node *hwNode) {
  BridgeInfo info;
  if (!hwNode)
    return info;

  auto inPorts = getDenseI32ArrayAttr(hwNode, "bridge_input_ports");
  auto inCats = getDenseI32ArrayAttr(hwNode, "bridge_input_categories");
  auto inLanes = getDenseI32ArrayAttr(hwNode, "bridge_input_lanes");
  auto outPorts = getDenseI32ArrayAttr(hwNode, "bridge_output_ports");
  auto outCats = getDenseI32ArrayAttr(hwNode, "bridge_output_categories");
  auto outLanes = getDenseI32ArrayAttr(hwNode, "bridge_output_lanes");
  auto muxNodes = getDenseI32ArrayAttr(hwNode, "bridge_mux_nodes");
  auto demuxNodes = getDenseI32ArrayAttr(hwNode, "bridge_demux_nodes");

  if (!inPorts && !outPorts)
    return info;

  info.hasBridge = true;
  appendPortArray(inPorts, info.inputPorts);
  appendCategoryArray(inCats, info.inputCategories);
  appendLaneArray(inLanes, info.inputLanes);
  appendPortArray(outPorts, info.outputPorts);
  appendCategoryArray(outCats, info.outputCategories);
  appendLaneArray(outLanes, info.outputLanes);
  appendPortArray(muxNodes, info.muxNodes);
  appendPortArray(demuxNodes, info.demuxNodes);
  return info;
}

BridgePortCategory DfgMemoryInfo::classifyInput(unsigned relIdx) const {
  unsigned storeInputs = static_cast<unsigned>(stCount) * 2;
  if (relIdx < storeInputs)
    return (relIdx % 2 == 0) ? BridgePortCategory::StData
                             : BridgePortCategory::StAddr;
  return BridgePortCategory::LdAddr;
}

BridgePortCategory DfgMemoryInfo::classifyOutput(unsigned idx) const {
  unsigned ld = static_cast<unsigned>(ldCount);
  if (idx < ld)
    return BridgePortCategory::LdData;
  if (idx < ld * 2)
    return BridgePortCategory::LdDone;
  return BridgePortCategory::StDone;
}

unsigned DfgMemoryInfo::inputLocalLane(unsigned relIdx) const {
  unsigned storeInputs = static_cast<unsigned>(stCount) * 2;
  if (relIdx < storeInputs)
    return relIdx / 2;
  return relIdx - storeInputs;
}

unsigned DfgMemoryInfo::outputLocalLane(unsigned idx) const {
  unsigned ld = static_cast<unsigned>(ldCount);
  if (idx < ld)
    return idx;
  if (idx < ld * 2)
    return idx - ld;
  return idx - ld * 2;
}

unsigned DfgMemoryInfo::laneSpan() const {
  return std::max<unsigned>(
      1, std::max(static_cast<unsigned>(ldCount),
                  static_cast<unsigned>(stCount)));
}

DfgMemoryInfo DfgMemoryInfo::extract(const Node *swNode, const Graph &dfg,
                                     bool isExtMem) {
  DfgMemoryInfo info;
  if (!swNode)
    return info;

  info.swInSkip = isExtMem ? 1 : 0;
  info.stCount = getNodeAttrInt(swNode, "stCount", 0);
  info.ldCount = getNodeAttrInt(swNode, "ldCount", 0);
  if (info.ldCount == 0 && info.stCount == 0) {
    unsigned nonMemInputs =
        static_cast<unsigned>(swNode->inputPorts.size()) - info.swInSkip;
    if (!swNode->outputPorts.empty())
      info.ldCount = 1;
    if (nonMemInputs > 1)
      info.stCount = 1;
  }
  (void)dfg;
  return info;
}

bool isBridgeCompatible(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                        const Node *swNode, const Node *hwNode,
                        const Graph &dfg, const Graph &adg) {
  if (!bridge.hasBridge || !swNode)
    return false;

  if (mem.swInSkip > 0) {
    if (swNode->inputPorts.empty() || !hwNode || hwNode->inputPorts.empty())
      return false;
    const Port *swMemPort = dfg.getPort(swNode->inputPorts[0]);
    const Port *hwMemPort = adg.getPort(hwNode->inputPorts[0]);
    if (!swMemPort || !hwMemPort ||
        !canMapSoftwareTypeToHardware(swMemPort->type, hwMemPort->type))
      return false;
  }

  MappingState emptyState;
  emptyState.init(dfg, adg);
  return chooseBridgeBaseLane(bridge, mem, swNode, hwNode, dfg, adg,
                              emptyState)
      .has_value();
}

bool bindBridgeInputs(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                      const Node *swNode, const Node *hwNode,
                      const Graph &dfg, const Graph &adg,
                      MappingState &state) {
  if (!swNode || !hwNode)
    return false;

  if (mem.swInSkip > 0 && !swNode->inputPorts.empty() &&
      !hwNode->inputPorts.empty()) {
    const Port *swMemPort = dfg.getPort(swNode->inputPorts[0]);
    const Port *hwMemPort = adg.getPort(hwNode->inputPorts[0]);
    if (!swMemPort || !hwMemPort ||
        !canMapSoftwareTypeToHardware(swMemPort->type, hwMemPort->type))
      return false;
    state.mapPort(swNode->inputPorts[0], hwNode->inputPorts[0], dfg, adg);
  }

  auto baseLane =
      chooseBridgeBaseLane(bridge, mem, swNode, hwNode, dfg, adg, state);
  if (!baseLane)
    return false;

  for (unsigned si = mem.swInSkip; si < swNode->inputPorts.size(); ++si) {
    const Port *sp = dfg.getPort(swNode->inputPorts[si]);
    if (!sp)
      continue;
    BridgePortCategory cat = mem.classifyInput(si - mem.swInSkip);
    unsigned lane = *baseLane + mem.inputLocalLane(si - mem.swInSkip);
    IdIndex swPid = swNode->inputPorts[si];
    if (swPid < state.swPortToHwPort.size() &&
        state.swPortToHwPort[swPid] != INVALID_ID)
      continue;
    auto hwPid = findCompatibleBridgePort(bridge.inputPorts,
                                          bridge.inputCategories,
                                          bridge.inputLanes, cat, lane,
                                          sp->type, adg, state);
    if (!hwPid)
      return false;
    state.mapPort(swPid, *hwPid, dfg, adg);
  }
  return true;
}

bool bindBridgeOutputs(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                       const Node *swNode, const Node *hwNode,
                       const Graph &dfg, const Graph &adg,
                       MappingState &state) {
  if (!swNode || !hwNode)
    return false;

  auto baseLane =
      chooseBridgeBaseLane(bridge, mem, swNode, hwNode, dfg, adg, state);
  if (!baseLane)
    return false;

  for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
    const Port *sp = dfg.getPort(swNode->outputPorts[oi]);
    if (!sp)
      continue;
    BridgePortCategory cat = mem.classifyOutput(oi);
    unsigned lane = *baseLane + mem.outputLocalLane(oi);
    IdIndex swPid = swNode->outputPorts[oi];
    if (swPid < state.swPortToHwPort.size() &&
        state.swPortToHwPort[swPid] != INVALID_ID)
      continue;
    auto hwPid = findCompatibleBridgePort(bridge.outputPorts,
                                          bridge.outputCategories,
                                          bridge.outputLanes, cat, lane,
                                          sp->type, adg, state);
    if (!hwPid)
      return false;
    state.mapPort(swPid, *hwPid, dfg, adg);
  }
  return true;
}

std::optional<unsigned> inferBridgeLane(const BridgeInfo &bridge,
                                        const DfgMemoryInfo &mem,
                                        const Node *swNode,
                                        const MappingState &state) {
  auto range = inferBridgeLaneRange(bridge, mem, swNode, state);
  if (!range)
    return std::nullopt;
  return range->start;
}

std::optional<BridgeLaneRange>
inferBridgeLaneRange(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                     const Node *swNode, const MappingState &state) {
  if (!bridge.hasBridge || !swNode)
    return std::nullopt;

  bool found = false;
  unsigned baseLane = 0;

  auto considerPort = [&](IdIndex swPort, unsigned localLane,
                          llvm::ArrayRef<IdIndex> ports,
                          llvm::ArrayRef<unsigned> lanes) -> bool {
    if (swPort >= state.swPortToHwPort.size())
      return true;
    IdIndex hwPort = state.swPortToHwPort[swPort];
    if (hwPort == INVALID_ID)
      return true;
    auto lane = findLaneForPort(hwPort, ports, lanes);
    if (!lane || *lane < localLane)
      return false;
    unsigned candidateBase = *lane - localLane;
    if (!found) {
      baseLane = candidateBase;
      found = true;
      return true;
    }
    return baseLane == candidateBase;
  };

  for (unsigned si = mem.swInSkip; si < swNode->inputPorts.size(); ++si) {
    if (!considerPort(swNode->inputPorts[si],
                      mem.inputLocalLane(si - mem.swInSkip), bridge.inputPorts,
                      bridge.inputLanes))
      return std::nullopt;
  }

  for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
    if (!considerPort(swNode->outputPorts[oi], mem.outputLocalLane(oi),
                      bridge.outputPorts, bridge.outputLanes))
      return std::nullopt;
  }

  if (!found)
    return std::nullopt;
  return BridgeLaneRange{baseLane, baseLane + mem.laneSpan()};
}

} // namespace fcc

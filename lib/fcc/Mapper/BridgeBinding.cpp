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

  unsigned swInCount =
      static_cast<unsigned>(swNode->inputPorts.size()) - mem.swInSkip;
  if (swInCount > bridge.inputPorts.size())
    return false;
  if (swNode->outputPorts.size() > bridge.outputPorts.size())
    return false;

  llvm::SmallVector<bool, 8> usedIn(bridge.inputPorts.size(), false);
  for (unsigned si = mem.swInSkip; si < swNode->inputPorts.size(); ++si) {
    const Port *sp = dfg.getPort(swNode->inputPorts[si]);
    if (!sp || !sp->type)
      continue;
    BridgePortCategory cat = mem.classifyInput(si - mem.swInSkip);
    bool found = false;
    for (unsigned bi = 0; bi < bridge.inputPorts.size(); ++bi) {
      if (usedIn[bi] || bridge.inputCategories[bi] != cat)
        continue;
      const Port *hp = adg.getPort(bridge.inputPorts[bi]);
      if (hp && canMapSoftwareTypeToHardware(sp->type, hp->type)) {
        usedIn[bi] = true;
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }

  llvm::SmallVector<bool, 8> usedOut(bridge.outputPorts.size(), false);
  for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
    const Port *sp = dfg.getPort(swNode->outputPorts[oi]);
    if (!sp || !sp->type)
      continue;
    BridgePortCategory cat = mem.classifyOutput(oi);
    bool found = false;
    for (unsigned bi = 0; bi < bridge.outputPorts.size(); ++bi) {
      if (usedOut[bi] || bridge.outputCategories[bi] != cat)
        continue;
      const Port *hp = adg.getPort(bridge.outputPorts[bi]);
      if (hp && canMapSoftwareTypeToHardware(sp->type, hp->type)) {
        usedOut[bi] = true;
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }

  return true;
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

  for (unsigned si = mem.swInSkip; si < swNode->inputPorts.size(); ++si) {
    const Port *sp = dfg.getPort(swNode->inputPorts[si]);
    if (!sp)
      continue;
    BridgePortCategory cat = mem.classifyInput(si - mem.swInSkip);
    bool found = false;
    for (unsigned bi = 0; bi < bridge.inputPorts.size(); ++bi) {
      if (bridge.inputCategories[bi] != cat)
        continue;
      IdIndex hwPid = bridge.inputPorts[bi];
      if (!state.hwPortToSwPorts[hwPid].empty())
        continue;
      const Port *hp = adg.getPort(hwPid);
      if (hp && canMapSoftwareTypeToHardware(sp->type, hp->type)) {
        state.mapPort(swNode->inputPorts[si], hwPid, dfg, adg);
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

bool bindBridgeOutputs(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                       const Node *swNode, const Node *hwNode,
                       const Graph &dfg, const Graph &adg,
                       MappingState &state) {
  if (!swNode || !hwNode)
    return false;

  for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
    const Port *sp = dfg.getPort(swNode->outputPorts[oi]);
    if (!sp)
      continue;
    BridgePortCategory cat = mem.classifyOutput(oi);
    bool found = false;
    for (unsigned bi = 0; bi < bridge.outputPorts.size(); ++bi) {
      if (bridge.outputCategories[bi] != cat)
        continue;
      IdIndex hwPid = bridge.outputPorts[bi];
      if (!state.hwPortToSwPorts[hwPid].empty())
        continue;
      const Port *hp = adg.getPort(hwPid);
      if (hp && canMapSoftwareTypeToHardware(sp->type, hp->type)) {
        state.mapPort(swNode->outputPorts[oi], hwPid, dfg, adg);
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

std::optional<unsigned> inferBridgeLane(const BridgeInfo &bridge,
                                        const DfgMemoryInfo &mem,
                                        const Node *swNode,
                                        const MappingState &state) {
  if (!bridge.hasBridge || !swNode)
    return std::nullopt;

  for (unsigned si = mem.swInSkip; si < swNode->inputPorts.size(); ++si) {
    IdIndex swPort = swNode->inputPorts[si];
    if (swPort >= state.swPortToHwPort.size())
      continue;
    IdIndex hwPort = state.swPortToHwPort[swPort];
    if (hwPort == INVALID_ID)
      continue;
    if (auto lane =
            findLaneForPort(hwPort, bridge.inputPorts, bridge.inputLanes)) {
      return lane;
    }
  }

  for (IdIndex swPort : swNode->outputPorts) {
    if (swPort >= state.swPortToHwPort.size())
      continue;
    IdIndex hwPort = state.swPortToHwPort[swPort];
    if (hwPort == INVALID_ID)
      continue;
    if (auto lane =
            findLaneForPort(hwPort, bridge.outputPorts, bridge.outputLanes)) {
      return lane;
    }
  }

  return std::nullopt;
}

} // namespace fcc

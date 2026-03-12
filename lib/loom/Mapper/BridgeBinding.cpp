//===-- BridgeBinding.cpp - Bridge port binding utilities -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/BridgeBinding.h"
#include "loom/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinTypes.h"

namespace loom {

namespace {

/// Get a string attribute from a node, or empty string.
llvm::StringRef getStrAttrBB(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes)
    if (attr.getName() == name)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return s.getValue();
  return "";
}

/// Get an integer attribute from a node, or default.
int64_t getIntAttrBB(const Node *node, llvm::StringRef name,
                     int64_t dflt = -1) {
  for (auto &attr : node->attributes)
    if (attr.getName() == name)
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return ia.getInt();
  return dflt;
}

} // namespace

// --- BridgeInfo ---

BridgeInfo BridgeInfo::extract(const Node *hwNode) {
  BridgeInfo info;
  if (!hwNode)
    return info;

  mlir::DenseI32ArrayAttr bridgeInPorts, bridgeOutPorts;
  mlir::DenseI32ArrayAttr bridgeInCats, bridgeOutCats;
  mlir::DenseI32ArrayAttr muxNodesAttr, demuxNodesAttr;

  for (auto &attr : hwNode->attributes) {
    llvm::StringRef name = attr.getName();
    if (name == "bridge_input_ports")
      bridgeInPorts = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
    else if (name == "bridge_output_ports")
      bridgeOutPorts =
          mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
    else if (name == "bridge_input_categories")
      bridgeInCats = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
    else if (name == "bridge_output_categories")
      bridgeOutCats = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
    else if (name == "bridge_mux_nodes")
      muxNodesAttr = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
    else if (name == "bridge_demux_nodes")
      demuxNodesAttr =
          mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr.getValue());
  }

  if (!bridgeInPorts && !bridgeOutPorts)
    return info;

  info.hasBridge = true;

  // Populate port ID vectors.
  if (bridgeInPorts)
    for (int32_t v : bridgeInPorts.asArrayRef())
      info.inputPorts.push_back(static_cast<IdIndex>(v));
  if (bridgeOutPorts)
    for (int32_t v : bridgeOutPorts.asArrayRef())
      info.outputPorts.push_back(static_cast<IdIndex>(v));

  // Populate mux/demux nodes.
  if (muxNodesAttr)
    for (int32_t v : muxNodesAttr.asArrayRef())
      info.muxNodes.push_back(static_cast<IdIndex>(v));
  if (demuxNodesAttr)
    for (int32_t v : demuxNodesAttr.asArrayRef())
      info.demuxNodes.push_back(static_cast<IdIndex>(v));

  // Populate categories: prefer explicit arrays, else reconstruct from legacy.
  if (bridgeInCats &&
      static_cast<size_t>(bridgeInCats.size()) == info.inputPorts.size()) {
    for (int32_t v : bridgeInCats.asArrayRef())
      info.inputCategories.push_back(static_cast<BridgePortCategory>(v));
  } else if (!info.inputPorts.empty()) {
    // Legacy fallback: reconstruct from bridge_store_input_count.
    // Input ordering: [st0_data, st0_addr, ..., stN_data, stN_addr,
    //                  ld0_addr, ..., ldN_addr]
    int32_t storeInCount = -1;
    for (auto &attr : hwNode->attributes)
      if (attr.getName() == "bridge_store_input_count")
        if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
          storeInCount = ia.getInt();
    unsigned stBound =
        (storeInCount > 0) ? static_cast<unsigned>(storeInCount) : 0;
    for (unsigned i = 0; i < info.inputPorts.size(); ++i) {
      if (i < stBound) {
        // Store inputs interleaved: even=data, odd=addr.
        info.inputCategories.push_back((i % 2 == 0)
                                           ? BridgePortCategory::StData
                                           : BridgePortCategory::StAddr);
      } else {
        info.inputCategories.push_back(BridgePortCategory::LdAddr);
      }
    }
  }

  if (bridgeOutCats &&
      static_cast<size_t>(bridgeOutCats.size()) == info.outputPorts.size()) {
    for (int32_t v : bridgeOutCats.asArrayRef())
      info.outputCategories.push_back(static_cast<BridgePortCategory>(v));
  } else if (!info.outputPorts.empty()) {
    // Legacy fallback: reconstruct from bridge_ld_data_output_count.
    // Output ordering: [ld_data * ldCount, ld_done * ldCount, st_done * stCount]
    int32_t ldDataOutCount = -1;
    for (auto &attr : hwNode->attributes)
      if (attr.getName() == "bridge_ld_data_output_count")
        if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
          ldDataOutCount = ia.getInt();
    unsigned ldOut =
        (ldDataOutCount > 0) ? static_cast<unsigned>(ldDataOutCount) : 0;
    unsigned ldDoneStart = ldOut;
    unsigned stDoneStart = ldOut * 2;
    for (unsigned i = 0; i < info.outputPorts.size(); ++i) {
      if (i < ldDoneStart)
        info.outputCategories.push_back(BridgePortCategory::LdData);
      else if (i < stDoneStart)
        info.outputCategories.push_back(BridgePortCategory::LdDone);
      else
        info.outputCategories.push_back(BridgePortCategory::StDone);
    }
  }

  return info;
}

// --- DfgMemoryInfo ---

BridgePortCategory DfgMemoryInfo::classifyInput(unsigned relIdx) const {
  // Store inputs occupy [0, stCount*2) with even=data, odd=addr.
  unsigned storeInputs = static_cast<unsigned>(stCount) * 2;
  if (relIdx < storeInputs)
    return (relIdx % 2 == 0) ? BridgePortCategory::StData
                             : BridgePortCategory::StAddr;
  return BridgePortCategory::LdAddr;
}

BridgePortCategory DfgMemoryInfo::classifyOutput(unsigned idx) const {
  // [0, ldCount) = ld_data, [ldCount, 2*ldCount) = ld_done, rest = st_done.
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
  info.stCount = getIntAttrBB(swNode, "stCount", 0);
  info.ldCount = getIntAttrBB(swNode, "ldCount", 0);
  // If ldCount not explicitly set, infer from output count.
  // Outputs: ldCount * ld_data + ldCount * ld_done + stCount * st_done.
  // If stCount > 0 and ldCount == 0 and there are outputs, they're all st_done.
  // If both are 0, fall back to port count inference.
  if (info.ldCount == 0 && info.stCount == 0) {
    // Single-port memory: default counts.
    unsigned nonMemIns = swNode->inputPorts.size() -
                         (isExtMem ? 1 : 0);
    // data+addr per store, addr per load.
    // With 1 store + 1 load: ins = 3 (data, addr, ld_addr)
    // Simple heuristic: if we have outputs, assume at least 1 load.
    if (!swNode->outputPorts.empty())
      info.ldCount = 1;
    if (nonMemIns > 1)
      info.stCount = 1;
  }
  return info;
}

// --- Compatibility check ---

bool isBridgeCompatible(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                        const Node *swNode, const Graph &dfg,
                        const Graph &adg) {
  if (!bridge.hasBridge || !swNode)
    return false;

  unsigned swInCount = swNode->inputPorts.size() - mem.swInSkip;
  if (swInCount > bridge.inputPorts.size())
    return false;
  if (swNode->outputPorts.size() > bridge.outputPorts.size())
    return false;

  // Input type matching: category-aware greedy.
  {
    llvm::SmallVector<bool, 8> used(bridge.inputPorts.size(), false);
    for (unsigned si = mem.swInSkip; si < swNode->inputPorts.size(); ++si) {
      const Port *sp = dfg.getPort(swNode->inputPorts[si]);
      if (!sp || !sp->type)
        continue;
      unsigned relIdx = si - mem.swInSkip;
      BridgePortCategory cat = mem.classifyInput(relIdx);
      bool found = false;
      for (unsigned bi = 0; bi < bridge.inputPorts.size(); ++bi) {
        if (used[bi])
          continue;
        if (bridge.inputCategories[bi] != cat)
          continue;
        const Port *hp = adg.getPort(bridge.inputPorts[bi]);
        if (hp && isTypeWidthCompatible(sp->type, hp->type)) {
          used[bi] = true;
          found = true;
          break;
        }
      }
      if (!found)
        return false;
    }
  }

  // Output type matching: category-aware greedy.
  {
    llvm::SmallVector<bool, 8> used(bridge.outputPorts.size(), false);
    for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
      const Port *sp = dfg.getPort(swNode->outputPorts[oi]);
      if (!sp || !sp->type)
        continue;
      BridgePortCategory cat = mem.classifyOutput(oi);
      bool found = false;
      for (unsigned bi = 0; bi < bridge.outputPorts.size(); ++bi) {
        if (used[bi])
          continue;
        if (bridge.outputCategories[bi] != cat)
          continue;
        const Port *hp = adg.getPort(bridge.outputPorts[bi]);
        if (hp && isTypeWidthCompatible(sp->type, hp->type)) {
          used[bi] = true;
          found = true;
          break;
        }
      }
      if (!found)
        return false;
    }
  }

  return true;
}

// --- Input binding ---

bool bindBridgeInputs(const BridgeInfo &bridge, const DfgMemoryInfo &mem,
                      const Node *swNode, const Node *hwNode,
                      const Graph &dfg, const Graph &adg,
                      MappingState &state) {
  if (!swNode || !hwNode)
    return false;

  // Bind memref port directly (extmemory only).
  if (mem.swInSkip > 0 && !swNode->inputPorts.empty() &&
      !hwNode->inputPorts.empty()) {
    state.mapPort(swNode->inputPorts[0], hwNode->inputPorts[0], dfg, adg);
  }

  for (unsigned si = mem.swInSkip; si < swNode->inputPorts.size(); ++si) {
    const Port *sp = dfg.getPort(swNode->inputPorts[si]);
    if (!sp)
      continue;
    unsigned relIdx = si - mem.swInSkip;
    BridgePortCategory cat = mem.classifyInput(relIdx);

    bool found = false;
    for (unsigned bi = 0; bi < bridge.inputPorts.size(); ++bi) {
      if (bridge.inputCategories[bi] != cat)
        continue;
      IdIndex hwPid = bridge.inputPorts[bi];
      if (!state.hwPortToSwPorts[hwPid].empty())
        continue;
      const Port *hp = adg.getPort(hwPid);
      if (hp && isTypeWidthCompatible(sp->type, hp->type)) {
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

// --- Output binding ---

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
      if (hp && isTypeWidthCompatible(sp->type, hp->type)) {
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

} // namespace loom

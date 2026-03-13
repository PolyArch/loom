//===-- ConfigGenFabric.cpp - Configured fabric MLIR output -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/ConfigGenUtil.h"
#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "loom/Dialect/Fabric/FabricOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace loom {

using namespace configgen;

namespace {

/// Build instruction_mem ArrayAttr for a temporal PE instance.
/// Returns empty ArrayAttr if no entries are generated.
mlir::ArrayAttr buildInstructionMem(
    const Node *hwNode, IdIndex hwId, const MappingState &state,
    const Graph &dfg, const Graph &adg,
    fabric::InstanceOp instanceOp, mlir::Builder &builder) {
  int64_t numInst = getNodeIntAttr(hwNode, "num_instruction", 0);
  int64_t numReg = getNodeIntAttr(hwNode, "num_register", 0);
  if (numInst <= 0)
    return {};

  (void)numReg; // Register encoding handled inline below.

  // Collect FU sub-node IDs in body order (ascending node ID == body order,
  // since ADGFlattener creates FU nodes sequentially in body order).
  llvm::SmallVector<IdIndex, 4> fuNodeIds;
  for (IdIndex nodeId = 0;
       nodeId < static_cast<IdIndex>(adg.nodes.size()); ++nodeId) {
    const Node *node = adg.getNode(nodeId);
    if (!node || node->kind != Node::OperationNode)
      continue;
    int64_t parentTPE = getNodeIntAttr(node, "parent_temporal_pe", -1);
    if (parentTPE == static_cast<int64_t>(hwId))
      fuNodeIds.push_back(nodeId);
  }

  // Build fuNodeId -> fuIndex map.
  llvm::DenseMap<IdIndex, unsigned> fuNodeToIndex;
  for (unsigned i = 0; i < fuNodeIds.size(); ++i)
    fuNodeToIndex[fuNodeIds[i]] = i;

  // Get FU names from the TemporalPEOp definition body.
  llvm::SmallVector<std::string, 4> fuNames;
  mlir::Operation *target =
      lookupSymbolInModule(instanceOp.getOperation(), instanceOp.getModule());
  if (auto tpeOp = mlir::dyn_cast_or_null<fabric::TemporalPEOp>(target)) {
    for (auto &innerOp : tpeOp.getBody().front()) {
      if (innerOp.hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      if (auto symName =
              innerOp.getAttrOfType<mlir::StringAttr>("sym_name"))
        fuNames.push_back(symName.getValue().str());
      else
        fuNames.push_back("fu_" + std::to_string(fuNames.size()));
    }
  }

  // Get port counts from the virtual node.
  unsigned numOutputs = hwNode->outputPorts.size();
  unsigned numInputs = hwNode->inputPorts.size();

  // Build instruction entries (sparse format: only populated slots).
  llvm::SmallVector<mlir::Attribute, 4> entries;
  for (int64_t slot = 0; slot < numInst; ++slot) {
    // Find SW node assigned to this slot on a FU of this TPE.
    for (IdIndex swId = 0;
         swId < static_cast<IdIndex>(state.temporalPEAssignments.size());
         ++swId) {
      const auto &tpa = state.temporalPEAssignments[swId];
      if (tpa.slot != static_cast<IdIndex>(slot))
        continue;

      // Verify this SW node is mapped to a FU of this temporal PE.
      if (swId >= state.swNodeToHwNode.size())
        continue;
      IdIndex mappedFuId = state.swNodeToHwNode[swId];
      auto fuIdxIt = fuNodeToIndex.find(mappedFuId);
      if (fuIdxIt == fuNodeToIndex.end())
        continue;

      // Derive correct opcode from FU body position (not tpa.opcode).
      unsigned opcode = fuIdxIt->second;
      int64_t tag = tpa.tag != INVALID_ID
          ? static_cast<int64_t>(tpa.tag) : 0;

      std::string fuName = opcode < fuNames.size()
          ? fuNames[opcode]
          : "fu_" + std::to_string(opcode);

      // Format: inst[S]: when(tag=T) dest, ... = fuName(O) src, ...
      // dest: out(idx) or reg(idx), src: in(idx) or reg(idx)
      std::string entry = "inst[" + std::to_string(slot) + "]: when(tag=" +
                          std::to_string(tag) + ") ";

      // Get DFG node for register-aware encoding.
      const Node *swNode = dfg.getNode(swId);

      // Destinations: check if result edges use registers.
      bool firstDest = true;
      if (swNode) {
        for (unsigned o = 0; o < swNode->outputPorts.size(); ++o) {
          IdIndex swPortId = swNode->outputPorts[o];
          const Port *swPort = dfg.getPort(swPortId);
          if (!swPort)
            continue;

          bool hasRegDest = false;
          bool hasOutDest = false;
          IdIndex regIdx = INVALID_ID;

          for (IdIndex edgeId : swPort->connectedEdges) {
            const Edge *edge = dfg.getEdge(edgeId);
            if (!edge || edge->srcPort != swPortId)
              continue;
            if (edgeId < state.registerAssignments.size() &&
                state.registerAssignments[edgeId] != INVALID_ID) {
              hasRegDest = true;
              regIdx = state.registerAssignments[edgeId];
            } else {
              hasOutDest = true;
            }
          }

          if (hasRegDest) {
            if (!firstDest)
              entry += ", ";
            entry += "reg(" + std::to_string(regIdx) + ")";
            firstDest = false;
          }
          if (hasOutDest) {
            if (!firstDest)
              entry += ", ";
            entry += "out(" + std::to_string(o) + ")";
            firstDest = false;
          }
          if (!hasRegDest && !hasOutDest) {
            if (!firstDest)
              entry += ", ";
            entry += "out(" + std::to_string(o) + ")";
            firstDest = false;
          }
        }
      } else {
        for (unsigned o = 0; o < numOutputs; ++o) {
          if (!firstDest)
            entry += ", ";
          entry += "out(" + std::to_string(o) + ")";
          firstDest = false;
        }
      }

      entry += " = " + fuName + "(" + std::to_string(opcode) + ") ";

      // Sources: check if operand edges use registers.
      bool firstSrc = true;
      if (swNode) {
        for (unsigned i = 0; i < swNode->inputPorts.size(); ++i) {
          IdIndex swPortId = swNode->inputPorts[i];
          const Port *swPort = dfg.getPort(swPortId);
          if (!swPort)
            continue;

          bool isReg = false;
          IdIndex regIdx = INVALID_ID;

          for (IdIndex edgeId : swPort->connectedEdges) {
            const Edge *edge = dfg.getEdge(edgeId);
            if (!edge || edge->dstPort != swPortId)
              continue;
            if (edgeId < state.registerAssignments.size() &&
                state.registerAssignments[edgeId] != INVALID_ID) {
              isReg = true;
              regIdx = state.registerAssignments[edgeId];
              break;
            }
          }

          if (!firstSrc)
            entry += ", ";
          if (isReg) {
            entry += "reg(" + std::to_string(regIdx) + ")";
          } else {
            // Find HW input port index via port mapping.
            unsigned hwInIdx = i;
            if (swPortId < state.swPortToHwPort.size()) {
              IdIndex hwPortId = state.swPortToHwPort[swPortId];
              for (unsigned hi = 0; hi < hwNode->inputPorts.size(); ++hi) {
                if (hwNode->inputPorts[hi] == hwPortId) {
                  hwInIdx = hi;
                  break;
                }
              }
            }
            entry += "in(" + std::to_string(hwInIdx) + ")";
          }
          firstSrc = false;
        }
      } else {
        for (unsigned i = 0; i < numInputs; ++i) {
          if (!firstSrc)
            entry += ", ";
          entry += "in(" + std::to_string(i) + ")";
          firstSrc = false;
        }
      }

      entries.push_back(builder.getStringAttr(entry));
      break;
    }
  }

  if (entries.empty())
    return {};
  return builder.getArrayAttr(entries);
}

/// Set output_tag attribute on a tagged PE instance.
/// For non-temporal tagged PEs, reads from taggedPEOutputTags (populated
/// during placement). For temporal PEs, derives from temporalPEAssignments.
void buildPEOutputTag(const Node *hwNode, IdIndex hwId,
                      const MappingState &state,
                      fabric::InstanceOp instanceOp,
                      mlir::Builder &builder) {
  if (!nodeHasAttr(hwNode, "output_tag"))
    return;

  unsigned numOutputs = hwNode->outputPorts.size();
  if (numOutputs == 0)
    return;

  // Non-temporal tagged PE: read from explicit tag assignment.
  uint64_t tagVal = 0;
  bool hasAssignment = false;
  auto tagIt = state.taggedPEOutputTags.find(hwId);
  if (tagIt != state.taggedPEOutputTags.end()) {
    tagVal = tagIt->second;
    hasAssignment = true;
  } else if (hwId < state.hwNodeToSwNodes.size() &&
             !state.hwNodeToSwNodes[hwId].empty()) {
    // Temporal PE: derive from temporal assignment.
    IdIndex swId = state.hwNodeToSwNodes[hwId][0];
    if (swId < state.temporalPEAssignments.size()) {
      const auto &tpa = state.temporalPEAssignments[swId];
      if (tpa.tag != INVALID_ID) {
        tagVal = tpa.tag;
        hasAssignment = true;
      }
    }
  }

  // Only emit output_tag on PEs that have a recorded assignment.
  if (!hasAssignment)
    return;

  llvm::SmallVector<mlir::Attribute, 4> tagAttrs;
  auto i64Type = builder.getIntegerType(64);
  for (unsigned o = 0; o < numOutputs; ++o)
    tagAttrs.push_back(builder.getIntegerAttr(i64Type, tagVal));

  instanceOp->setAttr("output_tag", builder.getArrayAttr(tagAttrs));
}

/// Set compare_predicate attribute on a PE instance when the mapped SW
/// node is a comparison operation.
void buildComparePredicate(const Node *hwNode, IdIndex hwId,
                           const MappingState &state, const Graph &dfg,
                           fabric::InstanceOp instanceOp,
                           mlir::Builder &builder) {
  if (hwId >= state.hwNodeToSwNodes.size())
    return;

  for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;
    llvm::StringRef swOp = getNodeOpName(swNode);
    if (swOp.starts_with("arith.cmp")) {
      for (auto &a : swNode->attributes) {
        if (a.getName() == "predicate") {
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a.getValue())) {
            instanceOp->setAttr("compare_predicate", ia);
            return;
          }
        }
      }
    }
  }
}

} // namespace

bool ConfigGen::writeConfiguredFabric(
    const MappingState &state, const Graph &dfg, const Graph &adg,
    const llvm::DenseMap<mlir::Operation *, IdIndex> &opMap,
    mlir::Operation *adgModule, const std::string &path) {
  mlir::Builder builder(adgModule->getContext());

  // Walk switch ops and set route_table.
  adgModule->walk([&](fabric::SwitchOp switchOp) {
    mlir::Operation *op = switchOp.getOperation();
    auto it = opMap.find(op);
    if (it == opMap.end())
      return;
    IdIndex hwId = it->second;

    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      return;

    unsigned numIn = hwNode->inputPorts.size();
    unsigned numOut = hwNode->outputPorts.size();
    if (numIn == 0 || numOut == 0)
      return;

    // Build active (output, input) transitions from routed paths.
    llvm::DenseSet<uint64_t> activeTransitions;
    for (const auto &pathVec : state.swEdgeToHwPaths) {
      if (pathVec.empty())
        continue;
      for (size_t j = 1; j + 1 < pathVec.size(); j += 2) {
        IdIndex inPortId = pathVec[j];
        IdIndex outPortId = pathVec[j + 1];
        const Port *inPort = adg.getPort(inPortId);
        if (!inPort || inPort->parentNode != hwId)
          continue;
        for (unsigned i = 0; i < numIn; ++i) {
          if (hwNode->inputPorts[i] == inPortId) {
            for (unsigned o = 0; o < numOut; ++o) {
              if (hwNode->outputPorts[o] == outPortId) {
                uint64_t key = (static_cast<uint64_t>(o) << 32) | i;
                activeTransitions.insert(key);
              }
            }
          }
        }
      }
    }

    // Get connectivity_table from the MLIR op attribute.
    auto connTable = switchOp.getConnectivityTableAttr();

    // Build compressed route_table: only entries where connectivity == 1.
    llvm::SmallVector<int8_t> routeTable;
    for (unsigned o = 0; o < numOut; ++o) {
      for (unsigned i = 0; i < numIn; ++i) {
        unsigned idx = o * numIn + i;
        bool connected = true;
        if (connTable && !connTable.empty()) {
          connected =
              idx < static_cast<unsigned>(connTable.size()) && connTable[idx];
        }
        if (!connected)
          continue;
        uint64_t key = (static_cast<uint64_t>(o) << 32) | i;
        routeTable.push_back(activeTransitions.count(key) ? 1 : 0);
      }
    }

    switchOp.setRouteTableAttr(
        mlir::DenseI8ArrayAttr::get(builder.getContext(), routeTable));
  });

  // Walk temporal switch ops and set route_table from temporal SW assignments.
  adgModule->walk([&](fabric::TemporalSwOp tswOp) {
    auto it = opMap.find(tswOp.getOperation());
    if (it == opMap.end())
      return;
    IdIndex hwId = it->second;

    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      return;

    int64_t numRouteTable = getNodeIntAttr(hwNode, "num_route_table", 0);
    if (numRouteTable <= 0)
      return;

    unsigned numIn = hwNode->inputPorts.size();
    unsigned numOut = hwNode->outputPorts.size();
    if (numOut == 0)
      return;

    // Look up temporal SW assignments for this node.
    const llvm::SmallVector<TemporalSWAssignment, 4> *assignments = nullptr;
    if (hwId < state.temporalSWAssignments.size() &&
        !state.temporalSWAssignments[hwId].empty()) {
      assignments = &state.temporalSWAssignments[hwId];
    }

    // Build human-readable route_table entries.
    llvm::SmallVector<mlir::Attribute, 4> routeEntries;
    for (int64_t rt = 0; rt < numRouteTable; ++rt) {
      uint64_t routeMask = 0;
      int64_t tagVal = -1;
      if (assignments) {
        for (const auto &tswa : *assignments) {
          if (tswa.slot == static_cast<IdIndex>(rt)) {
            routeMask = tswa.routeMask;
            if (tswa.tag != INVALID_ID)
              tagVal = static_cast<int64_t>(tswa.tag);
            break;
          }
        }
      }

      // Format: route_table[N]: when(tag=T) O[o]<-I[i], ...
      std::string entry = "route_table[" + std::to_string(rt) + "]: ";
      if (tagVal < 0 && routeMask == 0) {
        entry += "invalid";
      } else {
        if (tagVal >= 0)
          entry += "when(tag=" + std::to_string(tagVal) + ") ";

        // Decode routeMask (connected-position bits) back to (output, input).
        // Get connectivity_table for proper decoding.
        auto connTable = tswOp.getConnectivityTableAttr();
        bool firstRoute = true;
        unsigned posIdx = 0;
        for (unsigned o = 0; o < numOut; ++o) {
          for (unsigned i = 0; i < numIn; ++i) {
            bool connected = true;
            unsigned flatIdx = o * numIn + i;
            if (connTable && !connTable.empty()) {
              connected = flatIdx < static_cast<unsigned>(connTable.size()) &&
                          connTable[flatIdx];
            }
            if (!connected)
              continue;
            if (routeMask & (1ULL << posIdx)) {
              if (!firstRoute)
                entry += ", ";
              entry += "O[" + std::to_string(o) + "]<-I[" +
                       std::to_string(i) + "]";
              firstRoute = false;
            }
            ++posIdx;
          }
        }
        if (firstRoute)
          entry += "(idle)";
      }

      routeEntries.push_back(builder.getStringAttr(entry));
    }

    if (!routeEntries.empty())
      tswOp->setAttr("route_table", builder.getArrayAttr(routeEntries));
  });

  // Walk instance ops and set runtime config attributes.
  adgModule->walk([&](fabric::InstanceOp instanceOp) {
    auto it = opMap.find(instanceOp.getOperation());
    if (it == opMap.end())
      return;
    IdIndex hwId = it->second;
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      return;

    // Temporal PE instance: set instruction_mem.
    if (nodeHasAttr(hwNode, "is_virtual")) {
      auto instrMem = buildInstructionMem(hwNode, hwId, state, dfg, adg,
                                          instanceOp, builder);
      if (instrMem && !instrMem.empty())
        instanceOp->setAttr("instruction_mem", instrMem);
      return;
    }

    buildPEOutputTag(hwNode, hwId, state, instanceOp, builder);
    buildComparePredicate(hwNode, hwId, state, dfg, instanceOp, builder);
  });

  // Walk memory ops and set addr_offset_table MLIR attribute.
  auto setMemoryAddrOffset = [&](mlir::Operation *op) {
    auto it = opMap.find(op);
    if (it == opMap.end())
      return;
    IdIndex hwId = it->second;
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      return;

    bool hasMapped = (hwId < state.hwNodeToSwNodes.size() &&
                      !state.hwNodeToSwNodes[hwId].empty());
    int64_t numRegion = getNodeIntAttr(hwNode, "numRegion", 1);
    size_t mappedCount = hasMapped ? state.hwNodeToSwNodes[hwId].size() : 0;
    size_t regionCount =
        std::min(mappedCount, static_cast<size_t>(numRegion));
    int64_t ldCount = getNodeIntAttr(hwNode, "ldCount", 1);
    int64_t stCount = getNodeIntAttr(hwNode, "stCount", 1);
    bool isBridgeMemory = (ldCount > 1 || stCount > 1);
    int64_t tagCount = std::max(ldCount, stCount);

    // Bridge: each active region uses [0, tagCount); non-bridge: per-mapping.
    size_t effectiveRegionCount =
        isBridgeMemory ? 1 : regionCount;

    llvm::SmallVector<int64_t> table; // [valid, start, end, base] * numRegion
    for (size_t r = 0; r < static_cast<size_t>(numRegion); ++r) {
      if (r < effectiveRegionCount) {
        table.push_back(1);
        if (isBridgeMemory) {
          // Bridge lane tags span [0, tagCount) for every active region.
          table.push_back(0); table.push_back(tagCount);
        } else {
          table.push_back(static_cast<int64_t>(r));
          table.push_back(static_cast<int64_t>(r + 1));
        }
        table.push_back(0);
      } else {
        table.append({0, 0, 0, 0});
      }
    }

    op->setAttr("addrOffsetTable", builder.getDenseI64ArrayAttr(table));
    // Emit config node ID so external tools can correlate with _addr.h.
    op->setAttr("loom.configNodeId",
                builder.getI64IntegerAttr(static_cast<int64_t>(hwId)));
  };
  adgModule->walk(
      [&](fabric::MemoryOp memOp) { setMemoryAddrOffset(memOp); });
  adgModule->walk(
      [&](fabric::ExtMemoryOp memOp) { setMemoryAddrOffset(memOp); });

  // Walk add_tag ops and set mapper-assigned tag values.
  adgModule->walk([&](fabric::AddTagOp addTagOp) {
    auto it = opMap.find(addTagOp.getOperation());
    if (it == opMap.end())
      return;
    IdIndex hwId = it->second;
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      return;

    // Check if this is a bridge add_tag with a direct lane index.
    for (auto &attr : hwNode->attributes) {
      if (attr.getName() == "bridge_lane_index") {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue())) {
          auto resultType = mlir::dyn_cast<loom::dataflow::TaggedType>(
              addTagOp.getResult().getType());
          if (resultType) {
            auto tagType = resultType.getTagType();
            addTagOp.setTagAttr(
                mlir::IntegerAttr::get(tagType, intAttr.getInt()));
          }
          return; // Bridge add_tag handled.
        }
      }
    }

    // Collect all port IDs owned by this add_tag node.
    llvm::DenseSet<IdIndex> nodePortIds;
    for (IdIndex pid : hwNode->inputPorts)
      nodePortIds.insert(pid);
    for (IdIndex pid : hwNode->outputPorts)
      nodePortIds.insert(pid);

    // Scan routed SW edges to find one whose HW path goes through this node.
    for (IdIndex edgeId = 0;
         edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
         ++edgeId) {
      const auto &pathVec = state.swEdgeToHwPaths[edgeId];
      bool usesNode = false;
      for (IdIndex portId : pathVec) {
        if (nodePortIds.count(portId)) {
          usesNode = true;
          break;
        }
      }
      if (!usesNode)
        continue;

      // Found an edge routed through this add_tag.
      // Get the destination SW node's temporal PE assignment tag.
      const Edge *swEdge = dfg.getEdge(edgeId);
      if (!swEdge)
        continue;
      const Port *dstPort = dfg.getPort(swEdge->dstPort);
      if (!dstPort || dstPort->parentNode == INVALID_ID)
        continue;
      IdIndex dstSwNode = dstPort->parentNode;

      if (dstSwNode < state.temporalPEAssignments.size()) {
        const auto &tpa = state.temporalPEAssignments[dstSwNode];
        if (tpa.tag != INVALID_ID) {
          auto resultType = mlir::dyn_cast<loom::dataflow::TaggedType>(
              addTagOp.getResult().getType());
          if (resultType) {
            auto tagType = resultType.getTagType();
            addTagOp.setTagAttr(
                mlir::IntegerAttr::get(tagType, tpa.tag));
          }
          break;
        }
      }
    }
  });

  // Write the modified module to disk.
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return false;
  adgModule->print(out);
  out << "\n";
  return true;
}

} // namespace loom

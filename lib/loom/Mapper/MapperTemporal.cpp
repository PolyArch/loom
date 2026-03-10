//===-- MapperTemporal.cpp - Temporal assignment for mapper --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace loom {

namespace {

/// Get an integer attribute by name from a node.
int64_t getIntAttr(const Node *node, llvm::StringRef name, int64_t dflt = -1) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return intAttr.getInt();
    }
  }
  return dflt;
}

/// Check if a node has a unit attribute.
bool hasAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name)
      return true;
  }
  return false;
}

} // namespace

bool Mapper::runTemporalAssignment(MappingState &state, const Graph &dfg,
                                   const Graph &adg) {
  // For each mapped SW node, check if it's mapped to a temporal PE FU node.
  // If so, assign instruction slot, tag, and opcode.

  // Collect temporal PE virtual nodes and their mapped operations.
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 8>> temporalPEOps;

  for (IdIndex swId = 0; swId < static_cast<IdIndex>(state.swNodeToHwNode.size());
       ++swId) {
    IdIndex hwId = state.swNodeToHwNode[swId];
    if (hwId == INVALID_ID)
      continue;

    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode)
      continue;

    // Check if this HW node has a parent_temporal_pe attribute.
    int64_t parentTPE = getIntAttr(hwNode, "parent_temporal_pe", -1);
    if (parentTPE >= 0) {
      temporalPEOps[static_cast<IdIndex>(parentTPE)].push_back(swId);
    }
  }

  // Assign slots, tags, and opcodes for each temporal PE group.
  for (auto &[tpeId, swOps] : temporalPEOps) {
    const Node *tpeNode = adg.getNode(tpeId);
    if (!tpeNode)
      continue;

    int64_t numInstruction = getIntAttr(tpeNode, "num_instruction", 0);
    int64_t numRegister = getIntAttr(tpeNode, "num_register", 0);

    // Fail-fast: check capacity.
    if (static_cast<int64_t>(swOps.size()) > numInstruction)
      return false;

    // Build FU node ID -> FU body index map for correct opcode derivation.
    // The opcode is the FU's position among the parent TPE's fu_node attributes.
    llvm::DenseMap<IdIndex, IdIndex> fuNodeToIndex;
    {
      IdIndex fuIdx = 0;
      for (auto &attr : tpeNode->attributes) {
        if (attr.getName() == "fu_node") {
          if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
            fuNodeToIndex[static_cast<IdIndex>(intAttr.getInt())] = fuIdx++;
        }
      }
    }

    // Assign sequential slots and tags, with correct FU-body-position opcodes.
    for (size_t i = 0; i < swOps.size(); ++i) {
      IdIndex swNode = swOps[i];
      if (swNode >= state.temporalPEAssignments.size())
        continue;

      state.temporalPEAssignments[swNode].slot = static_cast<IdIndex>(i);
      state.temporalPEAssignments[swNode].tag = static_cast<IdIndex>(i);

      // Derive opcode from FU body position, not slot index.
      IdIndex hwNodeId = state.swNodeToHwNode[swNode];
      auto fuIt = fuNodeToIndex.find(hwNodeId);
      state.temporalPEAssignments[swNode].opcode =
          (fuIt != fuNodeToIndex.end()) ? fuIt->second
                                        : static_cast<IdIndex>(i);
    }

    // Assign registers for intra-PE edges.
    uint32_t regCount = 0;
    for (IdIndex swId : swOps) {
      const Node *swNode = dfg.getNode(swId);
      if (!swNode)
        continue;

      for (IdIndex outPortId : swNode->outputPorts) {
        const Port *outPort = dfg.getPort(outPortId);
        if (!outPort)
          continue;

        for (IdIndex edgeId : outPort->connectedEdges) {
          const Edge *edge = dfg.getEdge(edgeId);
          if (!edge)
            continue;

          // Check if destination is also in the same temporal PE.
          const Port *dstPort = dfg.getPort(edge->dstPort);
          if (!dstPort)
            continue;

          IdIndex dstSwNode = dstPort->parentNode;
          bool sameTPE = false;
          for (IdIndex op : swOps) {
            if (op == dstSwNode) {
              sameTPE = true;
              break;
            }
          }

          if (sameTPE) {
            // Skip inter-slot edges that were routed through the switch
            // fabric (they don't use internal temporal registers).
            if (edgeId < state.swEdgeToHwPaths.size() &&
                !state.swEdgeToHwPaths[edgeId].empty())
              continue;

            // Skip group-internal edges (FU body handles the connection).
            IdIndex srcHwNode = state.swNodeToHwNode[swId];
            auto groupIt = state.groupBindings.find(srcHwNode);
            if (groupIt != state.groupBindings.end()) {
              bool srcInGroup = false, dstInGroup = false;
              for (IdIndex gid : groupIt->second) {
                if (gid == swId) srcInGroup = true;
                if (gid == dstSwNode) dstInGroup = true;
              }
              if (srcInGroup && dstInGroup)
                continue;
            }

            // Allocate temporal register for true intra-TPE register-routed edge.
            if (edgeId < state.registerAssignments.size()) {
              state.registerAssignments[edgeId] = regCount++;
              if (static_cast<int64_t>(regCount) > numRegister)
                return false;
            }
          }
        }
      }
    }
  }

  return true;
}

} // namespace loom

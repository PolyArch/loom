//===-- MapperTemporal.cpp - Temporal assignment for mapper --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/Mapper.h"

#include "loom/Dialect/Dataflow/DataflowTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>

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

/// Get a string attribute by name from a node.
llvm::StringRef getStrAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Check if a node has a unit attribute.
bool hasAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name)
      return true;
  }
  return false;
}

/// Get the connectivity_table for a node, or empty array.
mlir::DenseI8ArrayAttr getConnTable(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "connectivity_table") {
      return mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attr.getValue());
    }
  }
  return {};
}

/// Count number of connected positions (1s) in a connectivity_table.
unsigned countConnectedPositions(mlir::DenseI8ArrayAttr connTable) {
  if (!connTable)
    return 0;
  unsigned count = 0;
  for (int64_t i = 0; i < connTable.size(); ++i) {
    if (connTable[i] != 0)
      ++count;
  }
  return count;
}

/// Get the connected-position index for (output, input) in a temporal_sw
/// connectivity_table. Returns INVALID_ID if the position is not connected.
/// The position index is the count of 1s in the table before [oi * numIn + ii].
IdIndex connectedPosIdx(mlir::DenseI8ArrayAttr connTable, unsigned numIn,
                        unsigned oi, unsigned ii) {
  unsigned flatIdx = oi * numIn + ii;
  if (!connTable || static_cast<int64_t>(flatIdx) >= connTable.size())
    return INVALID_ID;
  if (connTable[flatIdx] == 0)
    return INVALID_ID;
  unsigned pos = 0;
  for (unsigned k = 0; k < flatIdx; ++k) {
    if (connTable[k] != 0)
      ++pos;
  }
  return static_cast<IdIndex>(pos);
}

/// Get tag width from the first input port's tagged type.
unsigned getTagWidth(const Node *node, const Graph &adg) {
  for (IdIndex portId : node->inputPorts) {
    const Port *port = adg.getPort(portId);
    if (!port)
      continue;
    if (auto taggedType =
            mlir::dyn_cast<loom::dataflow::TaggedType>(port->type)) {
      return taggedType.getTagType().getWidth();
    }
  }
  return 4; // Default
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

  // Assign slots, tags, and opcodes for each temporal PE group (per-TPE first).
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

      if (log)
        log->logTemporalAssignment(
            swNode, tpeId, static_cast<IdIndex>(i),
            static_cast<IdIndex>(i),
            state.temporalPEAssignments[swNode].opcode);
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

  // --- Cross-TPE tag coordination (sub-task 2b) ---
  //
  // Find temporal_sw HW nodes and build "tag domains": groups of TPEs
  // connected through shared temporal_sw nodes. Within each domain,
  // assign globally unique tags.

  // Collect all temporal_sw HW node IDs.
  llvm::SmallVector<IdIndex, 8> temporalSWNodes;
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size()); ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || hwNode->kind != Node::OperationNode)
      continue;
    if (getStrAttr(hwNode, "op_name") == "fabric.temporal_sw")
      temporalSWNodes.push_back(hwId);
  }

  if (!temporalSWNodes.empty()) {
    // For each temporal_sw, find which TPE instances are connected by
    // scanning routed paths.
    // tswToTPEs: temporal_sw HW node -> set of TPE virtual node IDs
    llvm::DenseMap<IdIndex, llvm::DenseSet<IdIndex>> tswToTPEs;

    for (const auto &pathVec : state.swEdgeToHwPaths) {
      if (pathVec.empty())
        continue;

      // Walk path: [outPort0, inPort0, outPort1, inPort1, ...]
      // Internal transitions: inPort[k] -> outPort[k+1] at node level
      for (size_t j = 1; j + 1 < pathVec.size(); j += 2) {
        IdIndex inPortId = pathVec[j];
        const Port *inPort = adg.getPort(inPortId);
        if (!inPort)
          continue;
        IdIndex nodeId = inPort->parentNode;
        const Node *node = adg.getNode(nodeId);
        if (!node)
          continue;
        if (getStrAttr(node, "op_name") != "fabric.temporal_sw")
          continue;

        // This path goes through a temporal_sw. Find which TPE virtual
        // nodes are the source/destination of the DFG edge.
        // We need the DFG edge's source and dest SW nodes' parent TPEs.
        // We'll scan all DFG edges to find the one using this path below.
        // For now just record the temporal_sw.
      }
    }

    // More precise approach: for each DFG edge, if its HW path crosses a
    // temporal_sw, record the source/dest TPE virtual IDs.
    for (IdIndex edgeId = 0;
         edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
         ++edgeId) {
      const auto &pathVec = state.swEdgeToHwPaths[edgeId];
      if (pathVec.empty())
        continue;

      const Edge *swEdge = dfg.getEdge(edgeId);
      if (!swEdge)
        continue;

      // Find source and dest SW nodes' parent TPEs.
      const Port *srcPort = dfg.getPort(swEdge->srcPort);
      const Port *dstPort = dfg.getPort(swEdge->dstPort);
      if (!srcPort || !dstPort)
        continue;

      IdIndex srcSwNode = srcPort->parentNode;
      IdIndex dstSwNode = dstPort->parentNode;

      auto getParentTPE = [&](IdIndex swNodeId) -> IdIndex {
        if (swNodeId == INVALID_ID ||
            swNodeId >= state.swNodeToHwNode.size())
          return INVALID_ID;
        IdIndex hwId = state.swNodeToHwNode[swNodeId];
        if (hwId == INVALID_ID)
          return INVALID_ID;
        const Node *hwNode = adg.getNode(hwId);
        if (!hwNode)
          return INVALID_ID;
        int64_t parentTPE = getIntAttr(hwNode, "parent_temporal_pe", -1);
        return parentTPE >= 0 ? static_cast<IdIndex>(parentTPE) : INVALID_ID;
      };

      IdIndex srcTPE = getParentTPE(srcSwNode);
      IdIndex dstTPE = getParentTPE(dstSwNode);

      // Check if path crosses any temporal_sw.
      for (size_t j = 1; j + 1 < pathVec.size(); j += 2) {
        IdIndex inPortId = pathVec[j];
        const Port *inPort = adg.getPort(inPortId);
        if (!inPort)
          continue;
        IdIndex nodeId = inPort->parentNode;
        const Node *node = adg.getNode(nodeId);
        if (!node)
          continue;
        if (getStrAttr(node, "op_name") != "fabric.temporal_sw")
          continue;

        // Record TPEs connected through this temporal_sw.
        if (srcTPE != INVALID_ID)
          tswToTPEs[nodeId].insert(srcTPE);
        if (dstTPE != INVALID_ID)
          tswToTPEs[nodeId].insert(dstTPE);
      }
    }

    // Build connected components (tag domains): TPEs sharing any temporal_sw
    // form a domain. Chained temporal_sw nodes collapse into one domain.
    // Use union-find over TPE IDs.
    llvm::DenseMap<IdIndex, IdIndex> parent; // union-find parent

    auto findRoot = [&](IdIndex id) -> IdIndex {
      while (parent.count(id) && parent[id] != id)
        id = parent[id];
      return id;
    };

    auto unite = [&](IdIndex a, IdIndex b) {
      a = findRoot(a);
      b = findRoot(b);
      if (a != b) {
        // Deterministic: smaller ID becomes root.
        if (a > b)
          std::swap(a, b);
        parent[b] = a;
      }
    };

    // Initialize each TPE as its own root.
    for (auto &[tpeId, _] : temporalPEOps) {
      parent[tpeId] = tpeId;
    }

    // Unite TPEs that share a temporal_sw.
    for (auto &[tswId, tpeSet] : tswToTPEs) {
      IdIndex firstTPE = INVALID_ID;
      for (IdIndex tpeId : tpeSet) {
        if (firstTPE == INVALID_ID)
          firstTPE = tpeId;
        else
          unite(firstTPE, tpeId);
      }
    }

    // Group TPEs by domain root.
    llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> domains;
    for (auto &[tpeId, _] : temporalPEOps) {
      IdIndex root = findRoot(tpeId);
      domains[root].push_back(tpeId);
    }

    // For each multi-TPE domain, re-assign tags globally.
    for (auto &[domainRoot, tpeIds] : domains) {
      if (tpeIds.size() <= 1)
        continue; // Single-TPE domain: per-TPE assignment is fine.

      // Sort TPEs by HW node ID for deterministic ordering.
      llvm::sort(tpeIds);

      // Get tag width from any TPE in the domain.
      unsigned tagWidth = 4;
      for (IdIndex tpeId : tpeIds) {
        const Node *tpeNode = adg.getNode(tpeId);
        if (tpeNode) {
          tagWidth = getTagWidth(tpeNode, adg);
          break;
        }
      }
      unsigned maxTags = 1u << tagWidth;

      // Re-assign tags with a global counter.
      IdIndex domainCounter = 0;
      for (IdIndex tpeId : tpeIds) {
        auto it = temporalPEOps.find(tpeId);
        if (it == temporalPEOps.end())
          continue;

        for (IdIndex swNode : it->second) {
          if (swNode >= state.temporalPEAssignments.size())
            continue;
          state.temporalPEAssignments[swNode].tag = domainCounter;
          if (log)
            log->info("Cross-TPE tag: SW N" + std::to_string(swNode) +
                      " on TPE H" + std::to_string(tpeId) +
                      " -> tag=" + std::to_string(domainCounter));
          ++domainCounter;
        }
      }

      // Check capacity overflow.
      if (domainCounter > maxTags)
        return false;
    }

    // --- Populate temporalSWAssignments (sub-task 2c) ---
    //
    // Scan routed paths to build temporal_sw route table entries.
    for (IdIndex edgeId = 0;
         edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
         ++edgeId) {
      const auto &pathVec = state.swEdgeToHwPaths[edgeId];
      if (pathVec.empty())
        continue;

      const Edge *swEdge = dfg.getEdge(edgeId);
      if (!swEdge)
        continue;

      // Find the destination DFG node's tag.
      const Port *dstDfgPort = dfg.getPort(swEdge->dstPort);
      if (!dstDfgPort)
        continue;
      IdIndex dstSwNode = dstDfgPort->parentNode;
      IdIndex edgeTag = INVALID_ID;
      if (dstSwNode != INVALID_ID &&
          dstSwNode < state.temporalPEAssignments.size()) {
        edgeTag = state.temporalPEAssignments[dstSwNode].tag;
      }

      // If the destination doesn't have a temporal assignment, try source.
      if (edgeTag == INVALID_ID) {
        const Port *srcDfgPort = dfg.getPort(swEdge->srcPort);
        if (srcDfgPort && srcDfgPort->parentNode != INVALID_ID &&
            srcDfgPort->parentNode < state.temporalPEAssignments.size()) {
          edgeTag = state.temporalPEAssignments[srcDfgPort->parentNode].tag;
        }
      }

      if (edgeTag == INVALID_ID)
        continue;

      // Walk the HW path looking for temporal_sw transitions.
      for (size_t j = 1; j + 1 < pathVec.size(); j += 2) {
        IdIndex inPortId = pathVec[j];
        IdIndex outPortId = pathVec[j + 1];

        const Port *inPort = adg.getPort(inPortId);
        if (!inPort)
          continue;
        IdIndex tswNodeId = inPort->parentNode;
        const Node *tswNode = adg.getNode(tswNodeId);
        if (!tswNode)
          continue;
        if (getStrAttr(tswNode, "op_name") != "fabric.temporal_sw")
          continue;

        unsigned numIn = tswNode->inputPorts.size();

        // Find input and output port indices.
        unsigned ii = 0, oi = 0;
        for (unsigned k = 0; k < tswNode->inputPorts.size(); ++k) {
          if (tswNode->inputPorts[k] == inPortId) {
            ii = k;
            break;
          }
        }
        for (unsigned k = 0; k < tswNode->outputPorts.size(); ++k) {
          if (tswNode->outputPorts[k] == outPortId) {
            oi = k;
            break;
          }
        }

        // Compute connected-position index from connectivity_table.
        auto connTable = getConnTable(tswNode);
        IdIndex posIdx = connectedPosIdx(connTable, numIn, oi, ii);
        if (posIdx == INVALID_ID)
          continue;

        // Find or create an assignment slot for this tag.
        auto &assignments = state.temporalSWAssignments[tswNodeId];
        TemporalSWAssignment *target = nullptr;
        for (auto &tswa : assignments) {
          if (tswa.tag == edgeTag) {
            target = &tswa;
            break;
          }
        }
        if (!target) {
          // Allocate a new slot.
          IdIndex newSlot = 0;
          for (const auto &tswa : assignments) {
            if (tswa.slot >= newSlot)
              newSlot = tswa.slot + 1;
          }
          assignments.push_back({newSlot, edgeTag, 0});
          target = &assignments.back();
        }

        // OR the position bit into the routeMask.
        target->routeMask |= (1ULL << posIdx);

        if (log)
          log->logTemporalSWEntry(tswNodeId, target->slot,
                                  target->tag, target->routeMask);
      }
    }
  }

  return true;
}

} // namespace loom

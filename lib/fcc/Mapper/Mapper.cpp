#include "fcc/Mapper/Mapper.h"
#include "fcc/Mapper/BridgeBinding.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <queue>
#include <random>
#include <set>

namespace fcc {

namespace {

/// Check if a type is a memref type.
bool isMemrefType(mlir::Type type) {
  return mlir::isa<mlir::MemRefType>(type);
}

bool isNoneType(mlir::Type type) {
  return mlir::isa<mlir::NoneType>(type);
}

/// Find the first downstream operation node connected to a sentinel's output.
IdIndex findDownstreamNode(const Graph &graph, IdIndex sentinelNodeId) {
  const Node *sn = graph.getNode(sentinelNodeId);
  if (!sn)
    return INVALID_ID;
  for (IdIndex opId : sn->outputPorts) {
    const Port *outPort = graph.getPort(opId);
    if (!outPort)
      continue;
    for (IdIndex eid : outPort->connectedEdges) {
      const Edge *edge = graph.getEdge(eid);
      if (!edge || edge->srcPort != opId)
        continue;
      const Port *dstPort = graph.getPort(edge->dstPort);
      if (!dstPort || dstPort->parentNode == INVALID_ID)
        continue;
      const Node *dstNode = graph.getNode(dstPort->parentNode);
      if (dstNode && dstNode->kind == Node::OperationNode)
        return dstPort->parentNode;
    }
  }
  return INVALID_ID;
}

/// Known equivalences for ops that may not be explicitly in FU ops lists
/// but can be mapped to a compatible FU.
/// NOTE: arith.select and handshake.mux are NOT equivalent!
/// - handshake.mux: partial-consume (only selected input consumed)
/// - arith.select: full-consume (all inputs consumed)
/// They require separate FUs.
llvm::StringRef getCompatibleOp(llvm::StringRef dfgOpName) {
  if (dfgOpName == "dataflow.invariant")
    return "dataflow.gate";
  // No other equivalences — arith.select needs its own fu_select
  return "";
}

/// Check if a DFG op name matches an ADG FU's ops list.
bool opMatchesFU(llvm::StringRef dfgOpName, const Node *fuNode) {
  for (auto &attr : fuNode->attributes) {
    if (attr.getName() != "ops")
      continue;
    auto arrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue());
    if (!arrayAttr)
      continue;
    for (auto elem : arrayAttr) {
      auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elem);
      if (!strAttr)
        continue;
      if (strAttr.getValue() == dfgOpName)
        return true;
      // Check equivalence.
      llvm::StringRef compat = getCompatibleOp(dfgOpName);
      if (!compat.empty() && strAttr.getValue() == compat)
        return true;
    }
  }
  return false;
}

/// Check if a DFG node is a memory-related operation.
bool isMemoryOp(const Node *node) {
  auto name = getNodeAttrStr(node, "op_name");
  return name.contains("extmemory") || name.contains("memory") ||
         name.contains("load") ||
         name.contains("store");
}

bool isSoftwareMemoryInterfaceOp(llvm::StringRef opName) {
  return opName == "handshake.extmemory" || opName == "handshake.memory";
}

IdIndex getExpandedMemoryInputPort(const Node *hwNode, bool isExtMem,
                                   BridgePortCategory cat, unsigned lane) {
  if (!hwNode)
    return INVALID_ID;
  if (lane != 0)
    return INVALID_ID;
  unsigned ldCount =
      static_cast<unsigned>(std::max<int64_t>(0, getNodeAttrInt(hwNode, "ldCount", 0)));
  unsigned stCount =
      static_cast<unsigned>(std::max<int64_t>(0, getNodeAttrInt(hwNode, "stCount", 0)));
  unsigned portIdx = isExtMem ? 1u : 0u;

  if (ldCount > 0) {
    if (cat == BridgePortCategory::LdAddr) {
      if (portIdx >= hwNode->inputPorts.size())
        return INVALID_ID;
      return hwNode->inputPorts[portIdx];
    }
    ++portIdx;
  }

  if (stCount > 0) {
    if (cat == BridgePortCategory::StAddr) {
      if (portIdx >= hwNode->inputPorts.size())
        return INVALID_ID;
      return hwNode->inputPorts[portIdx];
    }
    ++portIdx;

    if (cat == BridgePortCategory::StData) {
      if (portIdx >= hwNode->inputPorts.size())
        return INVALID_ID;
      return hwNode->inputPorts[portIdx];
    }
  }

  return INVALID_ID;
}

IdIndex getExpandedMemoryOutputPort(const Node *hwNode,
                                    BridgePortCategory cat, unsigned lane) {
  if (!hwNode)
    return INVALID_ID;
  if (lane != 0)
    return INVALID_ID;
  unsigned ldCount =
      static_cast<unsigned>(std::max<int64_t>(0, getNodeAttrInt(hwNode, "ldCount", 0)));
  unsigned stCount =
      static_cast<unsigned>(std::max<int64_t>(0, getNodeAttrInt(hwNode, "stCount", 0)));
  unsigned portIdx = 0;

  if (ldCount > 0) {
    if (cat == BridgePortCategory::LdData) {
      if (portIdx >= hwNode->outputPorts.size())
        return INVALID_ID;
      return hwNode->outputPorts[portIdx];
    }
    ++portIdx;

    if (cat == BridgePortCategory::LdDone) {
      if (portIdx >= hwNode->outputPorts.size())
        return INVALID_ID;
      return hwNode->outputPorts[portIdx];
    }
    ++portIdx;
  }

  if (stCount > 0 && cat == BridgePortCategory::StDone) {
    if (portIdx >= hwNode->outputPorts.size())
      return INVALID_ID;
    return hwNode->outputPorts[portIdx];
  }

  return INVALID_ID;
}

} // namespace

llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>>
Mapper::buildCandidates(const Graph &dfg, const Graph &adg) {
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> candidates;

  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;

    // Sentinel nodes (module input/output) don't need FU candidates.
    if (swNode->kind != Node::OperationNode)
      continue;

    llvm::StringRef opName = getNodeAttrStr(swNode, "op_name");

    // Memory interface ops map to hardware memory nodes in the ADG.
    if (isSoftwareMemoryInterfaceOp(opName)) {
      bool isExtMem = (opName == "handshake.extmemory");
      DfgMemoryInfo memInfo =
          DfgMemoryInfo::extract(swNode, dfg, isExtMem);
      for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
           ++hwId) {
        const Node *hwNode = adg.getNode(hwId);
        if (!hwNode)
          continue;
        if (getNodeAttrStr(hwNode, "resource_class") != "memory")
          continue;

        BridgeInfo bridge = BridgeInfo::extract(hwNode);
        if (bridge.hasBridge) {
          if (!isBridgeCompatible(bridge, memInfo, swNode, hwNode, dfg, adg))
            continue;
        } else {
          if (swNode->inputPorts.size() > hwNode->inputPorts.size())
            continue;
          if (swNode->outputPorts.size() > hwNode->outputPorts.size())
            continue;

          if (memInfo.swInSkip > 0) {
            if (swNode->inputPorts.empty() || hwNode->inputPorts.empty())
              continue;
            const Port *swMemPort = dfg.getPort(swNode->inputPorts[0]);
            const Port *hwMemPort = adg.getPort(hwNode->inputPorts[0]);
            if (!swMemPort || !hwMemPort ||
                !canMapSoftwareTypeToHardware(swMemPort->type,
                                             hwMemPort->type))
              continue;
          }

          bool inputTypesOk = true;
          for (unsigned si = memInfo.swInSkip; si < swNode->inputPorts.size();
               ++si) {
            const Port *sp = dfg.getPort(swNode->inputPorts[si]);
            BridgePortCategory cat =
                memInfo.classifyInput(si - memInfo.swInSkip);
            unsigned lane = memInfo.inputLocalLane(si - memInfo.swInSkip);
            IdIndex hwPid = getExpandedMemoryInputPort(hwNode, isExtMem, cat, lane);
            const Port *hp = adg.getPort(hwPid);
            if (!sp || hwPid == INVALID_ID || !hp ||
                !canMapSoftwareTypeToHardware(sp->type, hp->type)) {
              inputTypesOk = false;
              break;
            }
          }
          if (!inputTypesOk)
            continue;

          bool outputTypesOk = true;
          for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
            const Port *sp = dfg.getPort(swNode->outputPorts[oi]);
            BridgePortCategory cat = memInfo.classifyOutput(oi);
            unsigned lane = memInfo.outputLocalLane(oi);
            IdIndex hwPid = getExpandedMemoryOutputPort(hwNode, cat, lane);
            const Port *hp = adg.getPort(hwPid);
            if (!sp || !hp ||
                !canMapSoftwareTypeToHardware(sp->type, hp->type)) {
              outputTypesOk = false;
              break;
            }
          }
          if (!outputTypesOk)
            continue;
        }
        candidates[swId].push_back(hwId);
      }
      continue;
    }

    // For regular ops, match against FU nodes.
    for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
         ++hwId) {
      const Node *hwNode = adg.getNode(hwId);
      if (!hwNode)
        continue;
      if (getNodeAttrStr(hwNode, "resource_class") != "functional")
        continue;
      if (opMatchesFU(opName, hwNode)) {
        candidates[swId].push_back(hwId);
      }
    }
  }

  return candidates;
}

std::vector<IdIndex> Mapper::computePlacementOrder(const Graph &dfg) {
  // Topological order: BFS from input sentinels.
  std::vector<IdIndex> order;
  std::vector<unsigned> inDegree(dfg.nodes.size(), 0);
  std::queue<IdIndex> worklist;

  // Compute in-degrees.
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *edge = dfg.getEdge(eid);
    if (!edge)
      continue;
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (dstPort && dstPort->parentNode != INVALID_ID &&
        dstPort->parentNode < inDegree.size()) {
      inDegree[dstPort->parentNode]++;
    }
  }

  // Seed with zero in-degree nodes.
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    if (dfg.getNode(i) && inDegree[i] == 0)
      worklist.push(i);
  }

  llvm::DenseSet<IdIndex> visited;
  while (!worklist.empty()) {
    IdIndex cur = worklist.front();
    worklist.pop();
    if (visited.count(cur))
      continue;
    visited.insert(cur);
    order.push_back(cur);

    const Node *node = dfg.getNode(cur);
    if (!node)
      continue;

    for (IdIndex opId : node->outputPorts) {
      const Port *outPort = dfg.getPort(opId);
      if (!outPort)
        continue;
      for (IdIndex eid : outPort->connectedEdges) {
        const Edge *edge = dfg.getEdge(eid);
        if (!edge)
          continue;
        const Port *dstPort = dfg.getPort(edge->dstPort);
        if (!dstPort || dstPort->parentNode == INVALID_ID)
          continue;
        IdIndex dstNode = dstPort->parentNode;
        if (dstNode < inDegree.size() && !visited.count(dstNode)) {
          inDegree[dstNode]--;
          if (inDegree[dstNode] == 0)
            worklist.push(dstNode);
        }
      }
    }
  }

  // Add any remaining unvisited nodes (cycles).
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    if (dfg.getNode(i) && !visited.count(i))
      order.push_back(i);
  }

  return order;
}

double Mapper::scorePlacement(IdIndex swNode, IdIndex hwNode,
                               const MappingState &state, const Graph &dfg,
                               const Graph &adg,
                               const ADGFlattener &flattener) {
  double score = 0.0;

  // Proximity to already-placed neighbors.
  auto [hwRow, hwCol] = flattener.getNodeGridPos(hwNode);

  const Node *swN = dfg.getNode(swNode);
  if (!swN)
    return score;

  int neighborCount = 0;
  double totalDist = 0.0;

  // Check input edges: are sources already placed?
  for (IdIndex ipId : swN->inputPorts) {
    const Port *ip = dfg.getPort(ipId);
    if (!ip)
      continue;
    for (IdIndex eid : ip->connectedEdges) {
      const Edge *edge = dfg.getEdge(eid);
      if (!edge || edge->dstPort != ipId)
        continue;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      if (!srcPort || srcPort->parentNode == INVALID_ID)
        continue;
      IdIndex srcSwNode = srcPort->parentNode;
      if (srcSwNode < state.swNodeToHwNode.size()) {
        IdIndex srcHwNode = state.swNodeToHwNode[srcSwNode];
        if (srcHwNode != INVALID_ID) {
          auto [sRow, sCol] = flattener.getNodeGridPos(srcHwNode);
          if (hwRow >= 0 && sRow >= 0) {
            totalDist += std::abs(hwRow - sRow) + std::abs(hwCol - sCol);
          }
          neighborCount++;
        }
      }
    }
  }

  // Check output edges: are destinations already placed?
  for (IdIndex opId : swN->outputPorts) {
    const Port *op = dfg.getPort(opId);
    if (!op)
      continue;
    for (IdIndex eid : op->connectedEdges) {
      const Edge *edge = dfg.getEdge(eid);
      if (!edge || edge->srcPort != opId)
        continue;
      const Port *dstPort = dfg.getPort(edge->dstPort);
      if (!dstPort || dstPort->parentNode == INVALID_ID)
        continue;
      IdIndex dstSwNode = dstPort->parentNode;
      if (dstSwNode < state.swNodeToHwNode.size()) {
        IdIndex dstHwNode = state.swNodeToHwNode[dstSwNode];
        if (dstHwNode != INVALID_ID) {
          auto [dRow, dCol] = flattener.getNodeGridPos(dstHwNode);
          if (hwRow >= 0 && dRow >= 0) {
            totalDist += std::abs(hwRow - dRow) + std::abs(hwCol - dCol);
          }
          neighborCount++;
        }
      }
    }
  }

  if (neighborCount > 0)
    score = -totalDist / neighborCount; // Negative because lower distance is
                                         // better.

  return score;
}

bool Mapper::runPlacement(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    const Options &opts) {

  auto order = computePlacementOrder(dfg);

  // Track which PE has been used (C8: at most one FU per spatial_pe).
  std::set<std::string> usedPEs;

  for (IdIndex swId : order) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;

    // Sentinel nodes don't need placement through the normal path.
    if (swNode->kind != Node::OperationNode)
      continue;

    // Skip nodes already placed (e.g., by sentinel binding or extmemory
    // pre-binding).
    if (swId < state.swNodeToHwNode.size() &&
        state.swNodeToHwNode[swId] != INVALID_ID)
      continue;

    auto candIt = candidates.find(swId);
    if (candIt == candidates.end() || candIt->second.empty()) {
      llvm::errs() << "Mapper: no candidates for DFG node " << swId
                    << " (" << getNodeAttrStr(swNode, "op_name") << ")\n";
      return false;
    }

    bool hasUnusedPECandidate = false;
    for (IdIndex hwId : candIt->second) {
      const Node *hwNode = adg.getNode(hwId);
      if (!hwNode)
        continue;
      if (!state.hwNodeToSwNodes[hwId].empty()) {
        if (getNodeAttrStr(hwNode, "resource_class") == "memory") {
          int64_t numRegion = getNodeAttrInt(hwNode, "numRegion", 1);
          if (static_cast<int64_t>(state.hwNodeToSwNodes[hwId].size()) >=
              numRegion)
            continue;
        } else {
          continue;
        }
      }
      llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
      if (!peName.empty() && !usedPEs.count(peName.str())) {
        hasUnusedPECandidate = true;
        break;
      }
    }

    // Score each candidate and pick the best.
    IdIndex bestHw = INVALID_ID;
    double bestScore = -1e18;

    for (IdIndex hwId : candIt->second) {
      // Check C8: PE exclusivity.
      const Node *hwNode = adg.getNode(hwId);
      if (!hwNode)
        continue;

      // Memory nodes can host multiple software memories up to numRegion.
      if (!state.hwNodeToSwNodes[hwId].empty()) {
        if (getNodeAttrStr(hwNode, "resource_class") == "memory") {
          int64_t numRegion = getNodeAttrInt(hwNode, "numRegion", 1);
          if (static_cast<int64_t>(state.hwNodeToSwNodes[hwId].size()) >=
              numRegion)
            continue;
        } else {
          continue;
        }
      }

      // Prefer unused PEs, but allow reuse if necessary (soft C8).
      llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
      if (hasUnusedPECandidate && !peName.empty() && usedPEs.count(peName.str()))
        continue;

      double score = scorePlacement(swId, hwId, state, dfg, adg, flattener);

      // Prefer unused PEs (soft C8 penalty).
      if (!peName.empty() && usedPEs.count(peName.str()))
        score -= 100.0;

      if (score > bestScore || bestHw == INVALID_ID) {
        bestScore = score;
        bestHw = hwId;
      }
    }

    if (bestHw == INVALID_ID) {
      llvm::errs() << "Mapper: failed to place DFG node " << swId
                    << " (" << getNodeAttrStr(swNode, "op_name") << ")\n";
      return false;
    }

    auto result = state.mapNode(swId, bestHw, dfg, adg);
    if (result != ActionResult::Success) {
      llvm::errs() << "Mapper: mapNode failed for " << swId << " -> "
                    << bestHw << "\n";
      return false;
    }

    const Node *placedHwNode = adg.getNode(bestHw);
    if (swNode && placedHwNode &&
        getNodeAttrStr(placedHwNode, "resource_class") == "memory") {
      BridgeInfo bridge = BridgeInfo::extract(placedHwNode);
      bool isExtMem = (getNodeAttrStr(placedHwNode, "op_kind") == "extmemory");
      if (bridge.hasBridge) {
        DfgMemoryInfo memInfo =
            DfgMemoryInfo::extract(swNode, dfg, isExtMem);
        if (!bindBridgeInputs(bridge, memInfo, swNode, placedHwNode, dfg, adg,
                              state) ||
            !bindBridgeOutputs(bridge, memInfo, swNode, placedHwNode, dfg, adg,
                               state)) {
          llvm::errs() << "Mapper: bridge port binding failed for " << swId
                       << " -> " << bestHw << "\n";
          return false;
        }
        } else {
          llvm::StringRef hwKind = getNodeAttrStr(placedHwNode, "op_kind");
          bool isScalarMemory = (hwKind == "memory" || hwKind == "extmemory");
          if (isScalarMemory) {
          DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);

          if (memInfo.swInSkip > 0 && !swNode->inputPorts.empty() &&
              !placedHwNode->inputPorts.empty()) {
            const Port *swMemPort = dfg.getPort(swNode->inputPorts[0]);
            IdIndex hwMemPid = placedHwNode->inputPorts[0];
            const Port *hwMemPort = adg.getPort(hwMemPid);
            if (!swMemPort || !hwMemPort ||
                !canMapSoftwareTypeToHardware(swMemPort->type,
                                             hwMemPort->type)) {
              llvm::errs() << "Mapper: memory-interface port mismatch for "
                           << swId << " -> " << bestHw << "\n";
              return false;
            }
            state.mapPort(swNode->inputPorts[0], hwMemPid, dfg, adg);
          }

          for (unsigned si = memInfo.swInSkip; si < swNode->inputPorts.size();
               ++si) {
            IdIndex swPid = swNode->inputPorts[si];
            const Port *sp = dfg.getPort(swPid);
            BridgePortCategory cat = memInfo.classifyInput(si - memInfo.swInSkip);
            unsigned lane = memInfo.inputLocalLane(si - memInfo.swInSkip);
            IdIndex hwPid =
                getExpandedMemoryInputPort(placedHwNode, isExtMem, cat, lane);
            const Port *hp = adg.getPort(hwPid);
            if (!sp || hwPid == INVALID_ID || !hp ||
                !state.hwPortToSwPorts[hwPid].empty() ||
                !canMapSoftwareTypeToHardware(sp->type, hp->type)) {
              llvm::errs() << "Mapper: scalar memory input binding failed for "
                           << swId << " -> " << bestHw << "\n";
              return false;
            }
            state.mapPort(swPid, hwPid, dfg, adg);
          }

          for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
            IdIndex swPid = swNode->outputPorts[oi];
            const Port *sp = dfg.getPort(swPid);
            BridgePortCategory cat = memInfo.classifyOutput(oi);
            unsigned lane = memInfo.outputLocalLane(oi);
            IdIndex hwPid = getExpandedMemoryOutputPort(placedHwNode, cat, lane);
            const Port *hp = adg.getPort(hwPid);
            if (!sp || hwPid == INVALID_ID || !hp ||
                !state.hwPortToSwPorts[hwPid].empty() ||
                !canMapSoftwareTypeToHardware(sp->type, hp->type)) {
              llvm::errs() << "Mapper: scalar memory output binding failed for "
                           << swId << " -> " << bestHw << "\n";
              return false;
            }
            state.mapPort(swPid, hwPid, dfg, adg);
          }
        } else {
          llvm::SmallVector<bool, 8> usedIn(placedHwNode->inputPorts.size(),
                                            false);
          for (unsigned si = 0; si < swNode->inputPorts.size(); ++si) {
            const Port *sp = dfg.getPort(swNode->inputPorts[si]);
            if (!sp)
              continue;
            for (unsigned hi = 0; hi < placedHwNode->inputPorts.size(); ++hi) {
              if (usedIn[hi])
                continue;
              IdIndex hwPid = placedHwNode->inputPorts[hi];
              if (!state.hwPortToSwPorts[hwPid].empty())
                continue;
              const Port *hp = adg.getPort(hwPid);
              if (hp && canMapSoftwareTypeToHardware(sp->type, hp->type)) {
                state.mapPort(swNode->inputPorts[si], hwPid, dfg, adg);
                usedIn[hi] = true;
                break;
              }
            }
          }
          for (unsigned oi = 0;
               oi < swNode->outputPorts.size() &&
               oi < placedHwNode->outputPorts.size();
               ++oi) {
            const Port *sp = dfg.getPort(swNode->outputPorts[oi]);
            IdIndex hwPid = placedHwNode->outputPorts[oi];
            const Port *hp = adg.getPort(hwPid);
            if (!state.hwPortToSwPorts[hwPid].empty())
              continue;
            if (!sp || !hp ||
                !canMapSoftwareTypeToHardware(sp->type, hp->type)) {
              llvm::errs() << "Mapper: output port type mismatch for " << swId
                           << " -> " << bestHw << "\n";
              return false;
            }
            state.mapPort(swNode->outputPorts[oi], hwPid, dfg, adg);
          }
        }
      }
    }

    // Mark PE as used.
    const Node *hwNode = adg.getNode(bestHw);
    if (hwNode) {
      llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
      if (!peName.empty())
        usedPEs.insert(peName.str());
    }

    if (opts.verbose) {
      llvm::outs() << "  Placed " << getNodeAttrStr(swNode, "op_name")
                    << " (node " << swId << ") -> HW node " << bestHw << "\n";
    }
  }

  return true;
}

double Mapper::computeTotalCost(const MappingState &state, const Graph &dfg,
                                const Graph &adg,
                                const ADGFlattener &flattener) {
  double cost = 0.0;
  // Sum Manhattan distances of all placed neighbor pairs across DFG edges.
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *edge = dfg.getEdge(eid);
    if (!edge)
      continue;
    const Port *sp = dfg.getPort(edge->srcPort);
    const Port *dp = dfg.getPort(edge->dstPort);
    if (!sp || !dp || sp->parentNode == INVALID_ID ||
        dp->parentNode == INVALID_ID)
      continue;
    IdIndex srcSw = sp->parentNode;
    IdIndex dstSw = dp->parentNode;
    if (srcSw >= state.swNodeToHwNode.size() ||
        dstSw >= state.swNodeToHwNode.size())
      continue;
    IdIndex srcHw = state.swNodeToHwNode[srcSw];
    IdIndex dstHw = state.swNodeToHwNode[dstSw];
    if (srcHw == INVALID_ID || dstHw == INVALID_ID)
      continue;
    auto [sr, sc] = flattener.getNodeGridPos(srcHw);
    auto [dr, dc] = flattener.getNodeGridPos(dstHw);
    if (sr >= 0 && dr >= 0)
      cost += std::abs(sr - dr) + std::abs(sc - dc);
  }
  return cost;
}

bool Mapper::runRefinement(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    const Options &opts) {

  // SA refinement: swap + relocate moves with Metropolis acceptance.
  std::mt19937 rng(static_cast<unsigned>(opts.seed));

  // Collect placed operation nodes.
  std::vector<IdIndex> placedNodes;
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    if (dfg.getNode(i) && dfg.getNode(i)->kind == Node::OperationNode &&
        state.swNodeToHwNode[i] != INVALID_ID &&
        !isMemoryOp(dfg.getNode(i))) {
      placedNodes.push_back(i);
    }
  }

  if (placedNodes.size() < 2)
    return true;

  double temperature = 100.0;
  double coolingRate = 0.995;
  int maxIter = static_cast<int>(placedNodes.size()) * 1000;
  // Cap at reasonable limit to avoid excessive runtime.
  if (maxIter > 50000)
    maxIter = 50000;

  double bestCost = computeTotalCost(state, dfg, adg, flattener);
  auto bestCheckpoint = state.save();
  int acceptCount = 0;

  auto startTime = std::chrono::steady_clock::now();

  for (int iter = 0; iter < maxIter; ++iter) {
    auto elapsed = std::chrono::steady_clock::now() - startTime;
    double secs =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() /
        1000.0;
    if (secs > opts.budgetSeconds * 0.4)
      break; // Reserve time for routing.

    double oldCost = computeTotalCost(state, dfg, adg, flattener);
    auto cp = state.save();

    bool moveOk = false;
    // 50% swap moves, 50% relocate moves.
    bool doSwap = std::uniform_int_distribution<int>(0, 1)(rng) == 0;

    if (doSwap) {
      // Swap two placed nodes.
      std::uniform_int_distribution<size_t> dist(0, placedNodes.size() - 1);
      size_t idxA = dist(rng);
      size_t idxB = dist(rng);
      if (idxA == idxB)
        continue;

      IdIndex swA = placedNodes[idxA];
      IdIndex swB = placedNodes[idxB];
      IdIndex hwA = state.swNodeToHwNode[swA];
      IdIndex hwB = state.swNodeToHwNode[swB];

      auto candItA = candidates.find(swA);
      auto candItB = candidates.find(swB);
      if (candItA == candidates.end() || candItB == candidates.end())
        continue;

      bool aCanGoToB = false, bCanGoToA = false;
      for (IdIndex c : candItA->second)
        if (c == hwB) { aCanGoToB = true; break; }
      for (IdIndex c : candItB->second)
        if (c == hwA) { bCanGoToA = true; break; }

      if (!aCanGoToB || !bCanGoToA)
        continue;

      const Node *hwNodeA = adg.getNode(hwA);
      const Node *hwNodeB = adg.getNode(hwB);
      if (!hwNodeA || !hwNodeB)
        continue;
      llvm::StringRef peNameA = getNodeAttrStr(hwNodeA, "pe_name");
      llvm::StringRef peNameB = getNodeAttrStr(hwNodeB, "pe_name");
      if (!peNameA.empty() && !peNameB.empty() && peNameA == peNameB)
        continue;

      state.unmapNode(swA, dfg, adg);
      state.unmapNode(swB, dfg, adg);
      auto r1 = state.mapNode(swA, hwB, dfg, adg);
      auto r2 = state.mapNode(swB, hwA, dfg, adg);
      moveOk = (r1 == ActionResult::Success && r2 == ActionResult::Success);
    } else {
      // Relocate: move one node to a random empty candidate.
      std::uniform_int_distribution<size_t> dist(0, placedNodes.size() - 1);
      size_t idx = dist(rng);
      IdIndex swN = placedNodes[idx];
      IdIndex hwOld = state.swNodeToHwNode[swN];

      auto candIt = candidates.find(swN);
      if (candIt == candidates.end() || candIt->second.size() < 2)
        continue;

      // Pick a random candidate that is not occupied.
      auto &candList = candIt->second;
      std::uniform_int_distribution<size_t> candDist(0, candList.size() - 1);
      IdIndex hwNew = INVALID_ID;
      // Try a few random candidates.
      for (int attempt = 0; attempt < 8; ++attempt) {
        IdIndex cand = candList[candDist(rng)];
        if (cand == hwOld)
          continue;
        if (!state.hwNodeToSwNodes[cand].empty())
          continue;
        // Check PE exclusivity.
        const Node *hwNode = adg.getNode(cand);
        if (!hwNode)
          continue;
        llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
        if (!peName.empty()) {
          // Check if any other node is on the same PE.
          bool peConflict = false;
          for (IdIndex other : placedNodes) {
            if (other == swN)
              continue;
            IdIndex otherHw = state.swNodeToHwNode[other];
            if (otherHw == INVALID_ID)
              continue;
            const Node *otherHwNode = adg.getNode(otherHw);
            if (otherHwNode &&
                getNodeAttrStr(otherHwNode, "pe_name") == peName) {
              peConflict = true;
              break;
            }
          }
          if (peConflict)
            continue;
        }
        hwNew = cand;
        break;
      }
      if (hwNew == INVALID_ID)
        continue;

      state.unmapNode(swN, dfg, adg);
      auto r = state.mapNode(swN, hwNew, dfg, adg);
      moveOk = (r == ActionResult::Success);
    }

    if (!moveOk) {
      state.restore(cp);
      continue;
    }

    double newCost = computeTotalCost(state, dfg, adg, flattener);
    double delta = oldCost - newCost; // Positive = improvement.

    if (delta > 0 ||
        std::uniform_real_distribution<double>(0.0, 1.0)(rng) <
            std::exp(delta / temperature)) {
      // Accept move.
      acceptCount++;
      if (newCost < bestCost) {
        bestCost = newCost;
        bestCheckpoint = state.save();
      }
    } else {
      state.restore(cp);
    }

    temperature *= coolingRate;
  }

  // Restore the best placement found.
  state.restore(bestCheckpoint);

  if (opts.verbose) {
    llvm::outs() << "  SA: " << acceptCount << " accepted moves, best cost "
                 << bestCost << "\n";
  }

  return true;
}

bool Mapper::bindSentinels(MappingState &state, const Graph &dfg,
                           const Graph &adg) {
  // Collect DFG and ADG sentinels.
  std::vector<IdIndex> dfgInputSentinels;
  std::vector<IdIndex> dfgOutputSentinels;
  std::vector<IdIndex> adgInputSentinels;
  std::vector<IdIndex> adgOutputSentinels;

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode)
      dfgInputSentinels.push_back(i);
    else if (node->kind == Node::ModuleOutputNode)
      dfgOutputSentinels.push_back(i);
  }

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node)
      continue;
    if (node->kind == Node::ModuleInputNode)
      adgInputSentinels.push_back(i);
    else if (node->kind == Node::ModuleOutputNode)
      adgOutputSentinels.push_back(i);
  }

  llvm::outs() << "  DFG sentinels: " << dfgInputSentinels.size()
               << " inputs, " << dfgOutputSentinels.size() << " outputs\n";
  llvm::outs() << "  ADG sentinels: " << adgInputSentinels.size()
               << " inputs, " << adgOutputSentinels.size() << " outputs\n";

  // Separate DFG input sentinels into memref and non-memref.
  std::vector<IdIndex> dfgMemrefSentinels;
  std::vector<IdIndex> dfgScalarSentinels;

  for (IdIndex sid : dfgInputSentinels) {
    const Node *node = dfg.getNode(sid);
    if (!node || node->outputPorts.empty())
      continue;
    mlir::Type portType = dfg.getPort(node->outputPorts[0])->type;
    if (isMemrefType(portType))
      dfgMemrefSentinels.push_back(sid);
    else
      dfgScalarSentinels.push_back(sid);
  }

  llvm::outs() << "    DFG memref inputs: " << dfgMemrefSentinels.size()
               << ", scalar inputs: " << dfgScalarSentinels.size() << "\n";

  // For memref sentinels: these are NOT mapped to ADG sentinels (the ADG
  // doesn't have memref sentinels since memrefs bind directly to extmemory).
  // Instead, map memref sentinel -> first available memory node in ADG.
  // The mapper already handles extmemory binding through buildCandidates.
  // We leave memref sentinels unmapped here; they are handled by the
  // extmemory matching in buildCandidates.

  // For scalar sentinels: bind DFG scalar inputs to ADG input sentinels.
  llvm::DenseSet<size_t> usedAdgIn;
  for (size_t di = 0; di < dfgScalarSentinels.size(); ++di) {
    IdIndex dfgSid = dfgScalarSentinels[di];

    // Find a matching ADG input sentinel (by index order).
    bool bound = false;
    for (size_t ai = 0; ai < adgInputSentinels.size(); ++ai) {
      if (usedAdgIn.count(ai))
        continue;

      IdIndex adgSid = adgInputSentinels[ai];
      const Node *dfgNode = dfg.getNode(dfgSid);
      const Node *adgNode = adg.getNode(adgSid);
      if (!dfgNode || !adgNode || dfgNode->outputPorts.empty() ||
          adgNode->outputPorts.empty())
        continue;
      const Port *swPort = dfg.getPort(dfgNode->outputPorts[0]);
      const Port *hwPort = adg.getPort(adgNode->outputPorts[0]);
      if (!swPort || !hwPort ||
          !canMapSoftwareTypeToHardware(swPort->type, hwPort->type))
        continue;

      auto result = state.mapNode(dfgSid, adgSid, dfg, adg);
      if (result == ActionResult::Success) {
        usedAdgIn.insert(ai);
        bound = true;
        llvm::outs() << "    Bound DFG input sentinel " << dfgSid
                      << " -> ADG input sentinel " << adgSid << "\n";
        break;
      }
    }

    if (!bound) {
      llvm::errs() << "Mapper: failed to bind DFG input sentinel "
                    << dfgSid << "\n";
    }
  }

  // For output sentinels: bind DFG output sentinels to ADG output sentinels.
  llvm::DenseSet<size_t> usedAdgOut;
  for (size_t di = 0; di < dfgOutputSentinels.size(); ++di) {
    IdIndex dfgSid = dfgOutputSentinels[di];
    const Node *dfgNode = dfg.getNode(dfgSid);
    if (!dfgNode || dfgNode->inputPorts.empty())
      continue;
    const Port *swPort = dfg.getPort(dfgNode->inputPorts[0]);

    bool bound = false;
    for (size_t ai = 0; ai < adgOutputSentinels.size(); ++ai) {
      if (usedAdgOut.count(ai))
        continue;

      IdIndex adgSid = adgOutputSentinels[ai];
      const Node *adgNode = adg.getNode(adgSid);
      if (!dfgNode || !adgNode || dfgNode->inputPorts.empty() ||
          adgNode->inputPorts.empty())
        continue;
      const Port *hwPort = adg.getPort(adgNode->inputPorts[0]);
      if (!swPort || !hwPort ||
          !canMapSoftwareTypeToHardware(swPort->type, hwPort->type))
        continue;

      auto result = state.mapNode(dfgSid, adgSid, dfg, adg);
      if (result == ActionResult::Success) {
        usedAdgOut.insert(ai);
        bound = true;
        llvm::outs() << "    Bound DFG output sentinel " << dfgSid
                      << " -> ADG output sentinel " << adgSid << "\n";
        break;
      }
    }

    if (!bound) {
      llvm::errs() << "Mapper: failed to bind DFG output sentinel "
                    << dfgSid << "\n";
    }
  }

  return true;
}

bool Mapper::bindMemrefSentinels(MappingState &state, const Graph &dfg,
                                  const Graph &adg) {
  // For each DFG memref input sentinel, find the edge to its downstream
  // extmemory node, and pre-route it as a direct binding. The memref sentinel
  // itself stays unmapped (it has no ADG counterpart), but its edge to the
  // extmemory node is marked as routed with a synthetic path.

  for (IdIndex sid = 0; sid < static_cast<IdIndex>(dfg.nodes.size()); ++sid) {
    const Node *sNode = dfg.getNode(sid);
    if (!sNode || sNode->kind != Node::ModuleInputNode)
      continue;
    if (sNode->outputPorts.empty())
      continue;

    // Check if this is a memref sentinel.
    const Port *outPort = dfg.getPort(sNode->outputPorts[0]);
    if (!outPort || !mlir::isa<mlir::MemRefType>(outPort->type))
      continue;

    // Find the edge(s) from this memref sentinel to extmemory nodes.
    for (IdIndex opId : sNode->outputPorts) {
      const Port *op = dfg.getPort(opId);
      if (!op)
        continue;
      for (IdIndex eid : op->connectedEdges) {
        const Edge *edge = dfg.getEdge(eid);
        if (!edge || edge->srcPort != opId)
          continue;

        // Get the destination node.
        const Port *dstPort = dfg.getPort(edge->dstPort);
        if (!dstPort || dstPort->parentNode == INVALID_ID)
          continue;

        IdIndex dstNodeId = dstPort->parentNode;
        const Node *dstNode = dfg.getNode(dstNodeId);
        if (!dstNode)
          continue;

        // Verify destination is an extmemory node.
        llvm::StringRef dstOpName = getNodeAttrStr(dstNode, "op_name");
        if (!dstOpName.contains("extmemory"))
          continue;

        // The extmemory DFG node should already be placed on an ADG node.
        if (dstNodeId >= state.swNodeToHwNode.size() ||
            state.swNodeToHwNode[dstNodeId] == INVALID_ID)
          continue;

        IdIndex hwExtMemNodeId = state.swNodeToHwNode[dstNodeId];
        const Node *hwExtMemNode = adg.getNode(hwExtMemNodeId);
        if (!hwExtMemNode)
          continue;

        // Get the memref input port on the ADG extmemory node (port index 0,
        // the memref port from the function type).
        if (hwExtMemNode->inputPorts.empty())
          continue;

        IdIndex hwMemrefInPort = hwExtMemNode->inputPorts[0];

        // Create a synthetic output port mapping for the memref sentinel.
        // We use the memref input port of the ADG extmemory as both
        // source and destination since this is a direct binding.
        // The path just contains [hwMemrefInPort, hwMemrefInPort] as a
        // sentinel marker for "direct memref binding".
        llvm::SmallVector<IdIndex, 8> syntheticPath;
        syntheticPath.push_back(hwMemrefInPort);
        syntheticPath.push_back(hwMemrefInPort);

        auto result = state.mapEdge(eid, syntheticPath, dfg, adg);
        if (result == ActionResult::Success) {
          llvm::outs() << "    Pre-routed memref edge " << eid
                        << " (sentinel " << sid << " -> extmem " << dstNodeId
                        << ") as direct binding\n";
        }
      }
    }
  }

  return true;
}

Mapper::Result Mapper::run(const Graph &dfg, const Graph &adg,
                           const ADGFlattener &flattener,
                           mlir::ModuleOp adgModule, const Options &opts) {
  Result result;
  TechMapper techMapper;
  TechMapper::Plan techPlan;
  if (!techMapper.buildPlan(dfg, adgModule, adg, techPlan)) {
    result.diagnostics = "Tech-mapping failed";
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  MappingState contractedState;
  contractedState.init(techPlan.contractedDFG, adg);
  result.edgeKinds = techPlan.originalEdgeKinds;

  // Copy connectivity from flattener.
  connectivity = flattener.getConnectivity();

  // Bind sentinels (DFG boundary nodes -> ADG boundary nodes).
  llvm::outs() << "Mapper: binding sentinels...\n";
  bindSentinels(contractedState, techPlan.contractedDFG, adg);

  llvm::outs() << "Mapper: building candidates...\n";
  auto candidates = buildCandidates(techPlan.contractedDFG, adg);
  for (const auto &entry : techPlan.contractedCandidates)
    candidates[entry.first] = entry.second;

  // Check that all operation nodes have candidates.
  for (IdIndex i = 0;
       i < static_cast<IdIndex>(techPlan.contractedDFG.nodes.size()); ++i) {
    const Node *node = techPlan.contractedDFG.getNode(i);
    if (!node || node->kind != Node::OperationNode)
      continue;
    if (candidates.find(i) == candidates.end() || candidates[i].empty()) {
      result.diagnostics = "No hardware candidates for DFG node " +
                           std::to_string(i) + " (" +
                           getNodeAttrStr(node, "op_name").str() + ")";
      llvm::errs() << "Mapper: " << result.diagnostics << "\n";
      return result;
    }
    if (opts.verbose) {
      llvm::outs() << "  Node " << i << " ("
                    << getNodeAttrStr(node, "op_name")
                    << "): " << candidates[i].size() << " candidates\n";
    }
  }

  llvm::outs() << "Mapper: placing...\n";
  if (!runPlacement(contractedState, techPlan.contractedDFG, adg, flattener,
                    candidates, opts)) {
    result.diagnostics = "Placement failed";
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  llvm::outs() << "Mapper: refining placement...\n";
  runRefinement(contractedState, techPlan.contractedDFG, adg, flattener,
                candidates, opts);

  llvm::outs() << "Mapper: binding memref sentinels...\n";
  bindMemrefSentinels(contractedState, techPlan.contractedDFG, adg);

  llvm::outs() << "Mapper: routing...\n";
  bool routingSucceeded =
      runRouting(contractedState, techPlan.contractedDFG, adg, opts.seed);
  if (!routingSucceeded) {
    result.diagnostics = "Routing failed";
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    // Continue anyway to produce partial output.
  }

  if (!techMapper.expandPlanMapping(dfg, adg, techPlan, contractedState,
                                    result.state, result.fuConfigs)) {
    result.diagnostics = "Tech-mapping expansion failed";
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  llvm::outs() << "Mapper: validating...\n";
  bool validationSucceeded = runValidation(result.state, dfg, adg,
                                          result.edgeKinds, result.diagnostics);
  if (!validationSucceeded) {
    llvm::errs() << "Mapper: validation issues: " << result.diagnostics
                 << "\n";
    // Proceed with partial result.
  }

  result.success = routingSucceeded && validationSucceeded;
  llvm::outs() << "Mapper: done.\n";
  return result;
}

bool Mapper::runValidation(const MappingState &state, const Graph &dfg,
                           const Graph &adg,
                           llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                           std::string &diagnostics) {
  bool valid = true;

  // C1: All operation nodes are placed. Memref sentinels are exempt
  // (they bind directly to extmemory, not through the ADG).
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node)
      continue;
    // Skip memref sentinels (they're expected to be unmapped).
    if (node->kind == Node::ModuleInputNode) {
      if (!node->outputPorts.empty()) {
        const Port *p = dfg.getPort(node->outputPorts[0]);
        if (p && mlir::isa<mlir::MemRefType>(p->type))
          continue;
      }
    }
    if (node->kind == Node::ModuleOutputNode) {
      if (!node->inputPorts.empty()) {
        const Port *p = dfg.getPort(node->inputPorts[0]);
      }
    }
    if (node->kind != Node::OperationNode &&
        node->kind != Node::ModuleInputNode &&
        node->kind != Node::ModuleOutputNode)
      continue;
    if (i >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[i] == INVALID_ID) {
      diagnostics += "C1: unmapped node " + std::to_string(i) + "\n";
      valid = false;
    }
  }

  // C3: All edges are routed. Only warn if both endpoints are placed
  // in the ADG (memref sentinel edges are exempt since memref sentinels
  // bind directly to extmemory without routing).
  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.edges.size()); ++i) {
    const Edge *edge = dfg.getEdge(i);
    if (!edge)
      continue;
    if (i < edgeKinds.size() && edgeKinds[i] == TechMappedEdgeKind::IntraFU)
      continue;
    if (i >= state.swEdgeToHwPaths.size() ||
        state.swEdgeToHwPaths[i].empty()) {
      const Port *sp = dfg.getPort(edge->srcPort);
      const Port *dp = dfg.getPort(edge->dstPort);
      if (sp && dp && sp->parentNode != INVALID_ID &&
          dp->parentNode != INVALID_ID) {
        IdIndex srcNodeId = sp->parentNode;
        IdIndex dstNodeId = dp->parentNode;
        // Check both endpoints are actually mapped in the ADG.
        bool srcMapped = srcNodeId < state.swNodeToHwNode.size() &&
                         state.swNodeToHwNode[srcNodeId] != INVALID_ID;
        bool dstMapped = dstNodeId < state.swNodeToHwNode.size() &&
                         state.swNodeToHwNode[dstNodeId] != INVALID_ID;
        if (srcMapped && dstMapped) {
          diagnostics +=
              "C3: unrouted edge " + std::to_string(i) + "\n";
          valid = false;
        }
      }
    }
  }

  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size()); ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;

    int64_t numRegion = getNodeAttrInt(hwNode, "numRegion", 1);
    if (hwId < state.hwNodeToSwNodes.size() &&
        static_cast<int64_t>(state.hwNodeToSwNodes[hwId].size()) > numRegion) {
      diagnostics += "C4: memory region overflow on hw_node " +
                     std::to_string(hwId) + "\n";
      valid = false;
    }

    BridgeInfo bridge = BridgeInfo::extract(hwNode);
    if (!bridge.hasBridge || hwId >= state.hwNodeToSwNodes.size())
      continue;

    bool isExtMem = (getNodeAttrStr(hwNode, "op_kind") == "extmemory");
    llvm::SmallVector<BridgeLaneRange, 4> usedLaneRanges;
    for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
      const Node *swNode = dfg.getNode(swId);
      if (!swNode)
        continue;
      DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);
      auto laneRange = inferBridgeLaneRange(bridge, memInfo, swNode, state);
      if (!laneRange) {
        diagnostics += "C4: missing bridge lane range for sw_node " +
                       std::to_string(swId) + " on hw_node " +
                       std::to_string(hwId) + "\n";
        valid = false;
        continue;
      }
      for (const auto &usedRange : usedLaneRanges) {
        if (laneRange->start < usedRange.end &&
            usedRange.start < laneRange->end) {
          diagnostics += "C4: overlapping bridge lane range [" +
                         std::to_string(laneRange->start) + ", " +
                         std::to_string(laneRange->end) + ") on hw_node " +
                         std::to_string(hwId) + "\n";
          valid = false;
          break;
        }
      }
      usedLaneRanges.push_back(*laneRange);
    }
  }

  return valid;
}

} // namespace fcc

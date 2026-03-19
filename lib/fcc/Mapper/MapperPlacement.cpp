#include "MapperInternal.h"
#include "fcc/Mapper/Mapper.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>
#include <random>

namespace fcc {

// ---------------------------------------------------------------------------
// Helper function implementations (mapper_detail namespace)
// ---------------------------------------------------------------------------

namespace mapper_detail {

bool isMemrefType(mlir::Type type) {
  return mlir::isa<mlir::MemRefType>(type);
}

bool isNoneType(mlir::Type type) {
  return mlir::isa<mlir::NoneType>(type);
}

bool isTemporalPENode(const Node *hwNode) {
  return hwNode && getNodeAttrStr(hwNode, "resource_class") == "functional" &&
         getNodeAttrStr(hwNode, "pe_kind") == "temporal_pe";
}

bool isSpatialPENode(const Node *hwNode) {
  return hwNode && getNodeAttrStr(hwNode, "resource_class") == "functional" &&
         getNodeAttrStr(hwNode, "pe_kind") == "spatial_pe";
}

bool isRoutingResourceNode(const Node *hwNode) {
  return hwNode && getNodeAttrStr(hwNode, "resource_class") == "routing";
}

const PEContainment *findPEContainmentByName(const ADGFlattener &flattener,
                                             llvm::StringRef peName) {
  for (const auto &pe : flattener.getPEContainment()) {
    if (pe.peName == peName)
      return &pe;
  }
  return nullptr;
}

bool isSpatialPEName(const ADGFlattener &flattener, llvm::StringRef peName) {
  const PEContainment *pe = findPEContainmentByName(flattener, peName);
  return pe && pe->peKind == "spatial_pe";
}

bool isSpatialPEOccupied(const MappingState &state, const Graph &adg,
                         const ADGFlattener &flattener, llvm::StringRef peName,
                         IdIndex ignoreHwNode) {
  if (peName.empty())
    return false;
  if (!isSpatialPEName(flattener, peName))
    return false;
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(state.hwNodeToSwNodes.size());
       ++hwId) {
    if (hwId == ignoreHwNode || state.hwNodeToSwNodes[hwId].empty())
      continue;
    const Node *hwNode = adg.getNode(hwId);
    if (hwNode && getNodeAttrStr(hwNode, "pe_name") == peName)
      return true;
  }
  return false;
}

bool sameConfigFields(llvm::ArrayRef<FUConfigField> lhs,
                      llvm::ArrayRef<FUConfigField> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].kind != rhs[i].kind || lhs[i].opIndex != rhs[i].opIndex ||
        lhs[i].opName != rhs[i].opName ||
        lhs[i].bitWidth != rhs[i].bitWidth ||
        lhs[i].value != rhs[i].value || lhs[i].sel != rhs[i].sel ||
        lhs[i].discard != rhs[i].discard ||
        lhs[i].disconnect != rhs[i].disconnect) {
      return false;
    }
  }
  return true;
}

bool detectForcedTemporalConfigConflict(
    const TechMapper::Plan &plan, const Graph &adg, std::string &diagnostics) {
  llvm::DenseMap<IdIndex, llvm::SmallVector<const TechMapper::Unit *, 4>>
      forcedByHwNode;

  for (const auto &unit : plan.units) {
    if (unit.contractedNodeId == INVALID_ID || unit.candidates.empty())
      continue;
    if (unit.candidates.size() != 1)
      continue;
    IdIndex hwNodeId = unit.candidates.front().hwNodeId;
    const Node *hwNode = adg.getNode(hwNodeId);
    if (!isTemporalPENode(hwNode))
      continue;
    forcedByHwNode[hwNodeId].push_back(&unit);
  }

  for (const auto &it : forcedByHwNode) {
    IdIndex hwNodeId = it.first;
    const auto &units = it.second;
    if (units.size() < 2)
      continue;

    llvm::ArrayRef<FUConfigField> firstConfig = units.front()->candidates.front().configFields;
    for (size_t i = 1; i < units.size(); ++i) {
      llvm::ArrayRef<FUConfigField> otherConfig =
          units[i]->candidates.front().configFields;
      if (sameConfigFields(firstConfig, otherConfig))
        continue;

      const Node *hwNode = adg.getNode(hwNodeId);
      diagnostics = "Temporal function_unit config conflict on hw node " +
                    std::to_string(hwNodeId);
      if (hwNode) {
        llvm::StringRef hwName = getNodeAttrStr(hwNode, "op_name");
        llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
        if (!hwName.empty())
          diagnostics += " (" + hwName.str() + ")";
        if (!peName.empty())
          diagnostics += " in " + peName.str();
      }
      return true;
    }
  }

  return false;
}

void classifyTemporalRegisterEdges(const MappingState &state, const Graph &dfg,
                                   const Graph &adg,
                                   const ADGFlattener &flattener,
                                   std::vector<TechMappedEdgeKind> &edgeKinds) {
  if (edgeKinds.size() < dfg.edges.size())
    edgeKinds.resize(dfg.edges.size(), TechMappedEdgeKind::Routed);
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU)
      continue;
    edgeKinds[edgeId] = TechMappedEdgeKind::Routed;
  }

  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeKinds[edgeId] != TechMappedEdgeKind::Routed)
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;

    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;

    IdIndex srcNodeId = srcPort->parentNode;
    IdIndex dstNodeId = dstPort->parentNode;
    if (srcNodeId >= state.swNodeToHwNode.size() ||
        dstNodeId >= state.swNodeToHwNode.size())
      continue;
    IdIndex srcHwNodeId = state.swNodeToHwNode[srcNodeId];
    IdIndex dstHwNodeId = state.swNodeToHwNode[dstNodeId];
    if (srcHwNodeId == INVALID_ID || dstHwNodeId == INVALID_ID ||
        srcHwNodeId == dstHwNodeId)
      continue;

    const Node *srcHwNode = adg.getNode(srcHwNodeId);
    const Node *dstHwNode = adg.getNode(dstHwNodeId);
    if (!isTemporalPENode(srcHwNode) || !isTemporalPENode(dstHwNode))
      continue;
    llvm::StringRef srcPE = getNodeAttrStr(srcHwNode, "pe_name");
    llvm::StringRef dstPE = getNodeAttrStr(dstHwNode, "pe_name");
    if (srcPE.empty() || srcPE != dstPE)
      continue;

    const PEContainment *pe = findPEContainmentByName(flattener, srcPE);
    if (!pe || pe->numRegister == 0)
      continue;

    edgeKinds[edgeId] = TechMappedEdgeKind::TemporalReg;
  }
}

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

llvm::StringRef getCompatibleOp(llvm::StringRef dfgOpName) {
  // No other equivalences -- arith.select needs its own fu_select
  return "";
}

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

double classifyEdgePlacementWeight(const Graph &dfg, IdIndex edgeId) {
  const Edge *edge = dfg.getEdge(edgeId);
  if (!edge)
    return 1.0;
  const Port *srcPort = dfg.getPort(edge->srcPort);
  const Port *dstPort = dfg.getPort(edge->dstPort);
  const Node *srcNode =
      (srcPort && srcPort->parentNode != INVALID_ID)
          ? dfg.getNode(srcPort->parentNode)
          : nullptr;
  const Node *dstNode =
      (dstPort && dstPort->parentNode != INVALID_ID)
          ? dfg.getNode(dstPort->parentNode)
          : nullptr;
  llvm::StringRef srcOp = srcNode ? getNodeAttrStr(srcNode, "op_name") : "";
  llvm::StringRef dstOp = dstNode ? getNodeAttrStr(dstNode, "op_name") : "";

  double weight = 1.0;
  if ((srcPort && isNoneType(srcPort->type)) || (dstPort && isNoneType(dstPort->type)))
    weight += 0.35;
  if ((srcNode && (srcNode->kind == Node::ModuleInputNode ||
                   srcNode->kind == Node::ModuleOutputNode)) ||
      (dstNode && (dstNode->kind == Node::ModuleInputNode ||
                   dstNode->kind == Node::ModuleOutputNode)))
    weight += 1.75;
  if (isMemoryOp(srcNode) || isMemoryOp(dstNode))
    weight += 1.50;

  auto isControlHub = [](const Node *node) {
    if (!node)
      return false;
    llvm::StringRef opName = getNodeAttrStr(node, "op_name");
    return opName == "dataflow.carry" || opName == "dataflow.gate" ||
           opName == "handshake.cond_br" || opName == "handshake.join" ||
           opName == "handshake.mux";
  };
  auto isMemoryBoundaryOp = [](llvm::StringRef opName) {
    return opName == "handshake.extmemory" || opName == "handshake.memory";
  };
  auto isMemoryDataOp = [](llvm::StringRef opName) {
    return opName == "handshake.load" || opName == "handshake.store";
  };
  if (isControlHub(srcNode) || isControlHub(dstNode))
    weight += 0.40;
  if (isMemoryBoundaryOp(srcOp) || isMemoryBoundaryOp(dstOp))
    weight += 2.25;
  if ((isMemoryDataOp(srcOp) && isMemoryBoundaryOp(dstOp)) ||
      (isMemoryBoundaryOp(srcOp) && isMemoryDataOp(dstOp)))
    weight += 2.75;
  if ((isMemoryBoundaryOp(srcOp) && isControlHub(dstNode)) ||
      (isMemoryBoundaryOp(dstOp) && isControlHub(srcNode)))
    weight += 2.50;
  if (((srcPort && isNoneType(srcPort->type)) ||
       (dstPort && isNoneType(dstPort->type))) &&
      (isMemoryBoundaryOp(srcOp) || isMemoryBoundaryOp(dstOp)))
    weight += 1.50;
  return weight;
}

double computeNodePriorityWeight(IdIndex swNode, const Graph &dfg) {
  const Node *node = dfg.getNode(swNode);
  if (!node)
    return 0.0;
  double weight = 0.0;
  for (IdIndex portId : node->inputPorts) {
    const Port *port = dfg.getPort(portId);
    if (!port)
      continue;
    for (IdIndex edgeId : port->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (edge && edge->dstPort == portId)
        weight += classifyEdgePlacementWeight(dfg, edgeId);
    }
  }
  for (IdIndex portId : node->outputPorts) {
    const Port *port = dfg.getPort(portId);
    if (!port)
      continue;
    for (IdIndex edgeId : port->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (edge && edge->srcPort == portId)
        weight += classifyEdgePlacementWeight(dfg, edgeId);
    }
  }
  if (isMemoryOp(node))
    weight += 4.0;
  llvm::StringRef opName = getNodeAttrStr(node, "op_name");
  if (opName == "handshake.extmemory" || opName == "handshake.memory")
    weight += 6.0;
  if (opName == "handshake.load" || opName == "handshake.store")
    weight += 3.0;
  if (opName == "handshake.cond_br" || opName == "dataflow.carry" ||
      opName == "dataflow.gate")
    weight += 2.5;
  return weight;
}

std::optional<std::pair<double, double>>
estimateNodePlacementPos(IdIndex swNode, const MappingState &state,
                         const Graph &dfg, const ADGFlattener &flattener,
                         const llvm::DenseMap<IdIndex,
                                              llvm::SmallVector<IdIndex, 4>>
                             &candidates) {
  if (swNode < state.swNodeToHwNode.size()) {
    IdIndex mappedHw = state.swNodeToHwNode[swNode];
    if (mappedHw != INVALID_ID) {
      auto [row, col] = flattener.getNodeGridPos(mappedHw);
      if (row >= 0 && col >= 0)
        return std::make_pair(static_cast<double>(row),
                              static_cast<double>(col));
    }
  }

  auto it = candidates.find(swNode);
  if (it == candidates.end() || it->second.empty())
    return std::nullopt;
  if (it->second.size() > 4)
    return std::nullopt;

  double rowSum = 0.0;
  double colSum = 0.0;
  unsigned count = 0;
  for (IdIndex hwNode : it->second) {
    auto [row, col] = flattener.getNodeGridPos(hwNode);
    if (row < 0 || col < 0)
      continue;
    rowSum += static_cast<double>(row);
    colSum += static_cast<double>(col);
    ++count;
  }
  if (count == 0)
    return std::nullopt;
  return std::make_pair(rowSum / count, colSum / count);
}

double computeLocalSpreadPenalty(IdIndex hwNode, const MappingState &state,
                                 const Graph &adg,
                                 const ADGFlattener &flattener) {
  auto [row, col] = flattener.getNodeGridPos(hwNode);
  if (row < 0 || col < 0)
    return 0.0;

  unsigned sameRow = 0;
  unsigned sameCol = 0;
  unsigned nearby = 0;
  for (IdIndex otherHw = 0;
       otherHw < static_cast<IdIndex>(state.hwNodeToSwNodes.size()); ++otherHw) {
    if (otherHw == hwNode || state.hwNodeToSwNodes[otherHw].empty())
      continue;
    const Node *otherNode = adg.getNode(otherHw);
    if (!otherNode || getNodeAttrStr(otherNode, "resource_class") != "functional")
      continue;
    auto [otherRow, otherCol] = flattener.getNodeGridPos(otherHw);
    if (otherRow < 0 || otherCol < 0)
      continue;
    if (otherRow == row)
      ++sameRow;
    if (otherCol == col)
      ++sameCol;
    if (std::abs(otherRow - row) + std::abs(otherCol - col) <= 2)
      ++nearby;
  }

  return 0.12 * static_cast<double>(sameRow + sameCol) +
         0.25 * static_cast<double>(nearby);
}

std::vector<IdIndex>
collectUnroutedEdges(const MappingState &state, const Graph &dfg,
                     llvm::ArrayRef<TechMappedEdgeKind> edgeKinds) {
  std::vector<IdIndex> failedEdges;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    if (edgeId >= state.swEdgeToHwPaths.size() || state.swEdgeToHwPaths[edgeId].empty())
      failedEdges.push_back(edgeId);
  }
  return failedEdges;
}

unsigned countRoutedEdges(const MappingState &state, const Graph &dfg,
                          llvm::ArrayRef<TechMappedEdgeKind> edgeKinds) {
  unsigned routed = 0;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg)) {
      ++routed;
      continue;
    }
    if (edgeId < state.swEdgeToHwPaths.size() && !state.swEdgeToHwPaths[edgeId].empty())
      ++routed;
  }
  return routed;
}

size_t computeTotalMappedPathLen(const MappingState &state) {
  size_t totalPathLen = 0;
  for (const auto &path : state.swEdgeToHwPaths)
    totalPathLen += path.size();
  return totalPathLen;
}

} // namespace mapper_detail

// ---------------------------------------------------------------------------
// Mapper placement methods
// ---------------------------------------------------------------------------

using namespace mapper_detail;

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
          bool bridgeOk =
              isBridgeCompatible(bridge, memInfo, swNode, hwNode, dfg, adg);
          if (!bridgeOk)
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

  llvm::stable_sort(order, [&](IdIndex lhs, IdIndex rhs) {
    const Node *lhsNode = dfg.getNode(lhs);
    const Node *rhsNode = dfg.getNode(rhs);
    auto nodeRank = [&](IdIndex id, const Node *node) {
      if (!node || node->kind != Node::OperationNode)
        return std::pair<double, double>{-1.0, -1.0};
      double priority = computeNodePriorityWeight(id, dfg);
      double boundary =
          (isMemoryOp(node) ? 1.0 : 0.0) +
          (getNodeAttrStr(node, "op_name") == "handshake.load" ? 1.0 : 0.0) +
          (getNodeAttrStr(node, "op_name") == "handshake.store" ? 1.0 : 0.0);
      return std::pair<double, double>{boundary, priority};
    };
    auto lhsRank = nodeRank(lhs, lhsNode);
    auto rhsRank = nodeRank(rhs, rhsNode);
    if (lhsRank.first != rhsRank.first)
      return lhsRank.first > rhsRank.first;
    if (lhsRank.second != rhsRank.second)
      return lhsRank.second > rhsRank.second;
    return lhs < rhs;
  });

  return order;
}

double Mapper::scorePlacement(IdIndex swNode, IdIndex hwNode,
                               const MappingState &state, const Graph &dfg,
                               const Graph &adg,
                               const ADGFlattener &flattener,
                               const llvm::DenseMap<IdIndex,
                                                    llvm::SmallVector<IdIndex, 4>>
                                   &candidates) {
  auto hwPos = flattener.getNodeGridPos(hwNode);
  int hwRow = hwPos.first;
  int hwCol = hwPos.second;
  if (hwRow < 0 || hwCol < 0)
    return -1.0e18;

  const Node *swN = dfg.getNode(swNode);
  if (!swN)
    return 0.0;

  double weightedDist = 0.0;
  double totalWeight = 0.0;
  auto accumulateNeighbor = [&](IdIndex otherSwNode, IdIndex edgeId) {
    auto estimate =
        estimateNodePlacementPos(otherSwNode, state, dfg, flattener, candidates);
    if (!estimate)
      return;
    double edgeWeight = classifyEdgePlacementWeight(dfg, edgeId);
    weightedDist += edgeWeight *
                    (std::abs(static_cast<double>(hwRow) - estimate->first) +
                     std::abs(static_cast<double>(hwCol) - estimate->second));
    totalWeight += edgeWeight;
  };

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
      accumulateNeighbor(srcPort->parentNode, eid);
    }
  }

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
      accumulateNeighbor(dstPort->parentNode, eid);
    }
  }

  double cost = 0.0;
  if (totalWeight > 0.0)
    cost += weightedDist / totalWeight;
  else
    cost += 0.25 * (std::abs(hwRow) + std::abs(hwCol));

  cost += 0.6 * computeLocalSpreadPenalty(hwNode, state, adg, flattener);
  return -cost;
}

bool Mapper::runPlacement(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    const Options &opts) {
  std::mt19937 rng(static_cast<unsigned>(opts.seed));

  auto order = computePlacementOrder(dfg);
  llvm::stable_sort(order, [&](IdIndex lhs, IdIndex rhs) {
    const Node *lhsNode = dfg.getNode(lhs);
    const Node *rhsNode = dfg.getNode(rhs);
    auto candidateCount = [&](IdIndex swId, const Node *node) -> size_t {
      if (!node || node->kind != Node::OperationNode)
        return std::numeric_limits<size_t>::max();
      auto it = candidates.find(swId);
      if (it == candidates.end())
        return std::numeric_limits<size_t>::max();
      return it->second.size();
    };
    size_t lhsCount = candidateCount(lhs, lhsNode);
    size_t rhsCount = candidateCount(rhs, rhsNode);
    if (lhsCount != rhsCount)
      return lhsCount < rhsCount;
    return false;
  });

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

    // Score each candidate and pick the best.
    IdIndex bestHw = INVALID_ID;
    double bestScore = -1e18;
    llvm::SmallVector<std::pair<double, IdIndex>, 8> rankedCandidates;

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

      llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
      if (isSpatialPEOccupied(state, adg, flattener, peName))
        continue;

      double score =
          scorePlacement(swId, hwId, state, dfg, adg, flattener, candidates);
      rankedCandidates.push_back({-score, hwId});
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

    llvm::stable_sort(rankedCandidates, [&](const auto &lhs, const auto &rhs) {
      if (lhs.first != rhs.first)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });

    llvm::SmallVector<IdIndex, 4> shortlist;
    for (const auto &candidate : rankedCandidates) {
      double score = -candidate.first;
      if (!shortlist.empty() && bestScore - score > 0.35 &&
          shortlist.size() >= 2)
        break;
      shortlist.push_back(candidate.second);
      if (shortlist.size() >= 4)
        break;
    }
    if (!shortlist.empty()) {
      std::uniform_int_distribution<size_t> dist(0, shortlist.size() - 1);
      bestHw = shortlist[dist(rng)];
    }

    auto result = state.mapNode(swId, bestHw, dfg, adg);
    if (result != ActionResult::Success) {
      llvm::errs() << "Mapper: mapNode failed for " << swId << " -> "
                    << bestHw << "\n";
      return false;
    }
    if (!bindMappedNodePorts(swId, state, dfg, adg)) {
      llvm::errs() << "Mapper: port binding failed for " << swId << " -> "
                   << bestHw << "\n";
      return false;
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
  int maxRow = -1;
  int maxCol = -1;
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(state.hwNodeToSwNodes.size());
       ++hwId) {
    if (state.hwNodeToSwNodes[hwId].empty())
      continue;
    auto [row, col] = flattener.getNodeGridPos(hwId);
    if (row >= 0 && col >= 0) {
      maxRow = std::max(maxRow, row);
      maxCol = std::max(maxCol, col);
    }
  }
  std::vector<double> rowCutLoad(
      maxRow >= 0 ? static_cast<size_t>(maxRow) + 1 : 0, 0.0);
  std::vector<double> colCutLoad(
      maxCol >= 0 ? static_cast<size_t>(maxCol) + 1 : 0, 0.0);

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
    if (sr >= 0 && dr >= 0) {
      double edgeWeight = classifyEdgePlacementWeight(dfg, eid);
      int dist = std::abs(sr - dr) + std::abs(sc - dc);
      cost += edgeWeight * static_cast<double>(dist);
      for (int row = std::min(sr, dr); row < std::max(sr, dr) &&
                                       row < static_cast<int>(rowCutLoad.size());
           ++row)
        rowCutLoad[row] += edgeWeight;
      for (int col = std::min(sc, dc); col < std::max(sc, dc) &&
                                       col < static_cast<int>(colCutLoad.size());
           ++col)
        colCutLoad[col] += edgeWeight;
    }
  }
  for (double load : rowCutLoad)
    cost += 0.015 * load * load;
  for (double load : colCutLoad)
    cost += 0.015 * load * load;
  return cost;
}

bool Mapper::bindMappedNodePorts(IdIndex swId, MappingState &state,
                                 const Graph &dfg, const Graph &adg) {
  if (swId >= state.swNodeToHwNode.size())
    return false;

  IdIndex bestHw = state.swNodeToHwNode[swId];
  if (bestHw == INVALID_ID)
    return false;

  const Node *swNode = dfg.getNode(swId);
  const Node *placedHwNode = adg.getNode(bestHw);
  if (!swNode || !placedHwNode)
    return false;

  if (getNodeAttrStr(placedHwNode, "resource_class") != "memory")
    return true;

  BridgeInfo bridge = BridgeInfo::extract(placedHwNode);
  bool isExtMem = (getNodeAttrStr(placedHwNode, "op_kind") == "extmemory");
  if (bridge.hasBridge) {
    DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);
    return bindBridgeInputs(bridge, memInfo, swNode, placedHwNode, dfg, adg,
                            state) &&
           bindBridgeOutputs(bridge, memInfo, swNode, placedHwNode, dfg, adg,
                             state);
  }

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
          !canMapSoftwareTypeToHardware(swMemPort->type, hwMemPort->type)) {
        return false;
      }
      state.mapPort(swNode->inputPorts[0], hwMemPid, dfg, adg);
    }

    for (unsigned si = memInfo.swInSkip; si < swNode->inputPorts.size(); ++si) {
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
        return false;
      }
      state.mapPort(swPid, hwPid, dfg, adg);
    }
    return true;
  }

  llvm::DenseMap<IdIndex, double> emptyHistory;
  auto estimateInputBindingCost = [&](IdIndex swPid, IdIndex hwPid) {
    double cost = 0.0;
    bool observed = false;
    const Port *swPort = dfg.getPort(swPid);
    if (!swPort)
      return 0.0;
    for (IdIndex edgeId : swPort->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge || edge->dstPort != swPid)
        continue;
      IdIndex srcHwPid =
          edge->srcPort < state.swPortToHwPort.size()
              ? state.swPortToHwPort[edge->srcPort]
              : INVALID_ID;
      if (srcHwPid == INVALID_ID)
        continue;
      observed = true;
      auto path =
          findPath(srcHwPid, hwPid, edgeId, state, dfg, adg, emptyHistory);
      cost += path.empty() ? 1.0e6 : static_cast<double>(path.size());
    }
    return observed ? cost : 0.0;
  };
  auto estimateOutputBindingCost = [&](IdIndex swPid, IdIndex hwPid) {
    double cost = 0.0;
    bool observed = false;
    const Port *swPort = dfg.getPort(swPid);
    if (!swPort)
      return 0.0;
    for (IdIndex edgeId : swPort->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge || edge->srcPort != swPid)
        continue;
      IdIndex dstHwPid =
          edge->dstPort < state.swPortToHwPort.size()
              ? state.swPortToHwPort[edge->dstPort]
              : INVALID_ID;
      if (dstHwPid == INVALID_ID)
        continue;
      observed = true;
      auto path =
          findPath(hwPid, dstHwPid, edgeId, state, dfg, adg, emptyHistory);
      cost += path.empty() ? 1.0e6 : static_cast<double>(path.size());
    }
    return observed ? cost : 0.0;
  };

  llvm::SmallVector<bool, 8> usedIn(placedHwNode->inputPorts.size(), false);
  for (unsigned si = 0; si < swNode->inputPorts.size(); ++si) {
    IdIndex swPid = swNode->inputPorts[si];
    const Port *sp = dfg.getPort(swPid);
    if (!sp)
      continue;

    llvm::SmallVector<std::pair<double, unsigned>, 8> rankedInputs;
    for (unsigned hi = 0; hi < placedHwNode->inputPorts.size(); ++hi) {
      if (usedIn[hi])
        continue;
      IdIndex hwPid = placedHwNode->inputPorts[hi];
      if (!state.hwPortToSwPorts[hwPid].empty())
        continue;
      const Port *hp = adg.getPort(hwPid);
      if (!hp || !canMapSoftwareTypeToHardware(sp->type, hp->type))
        continue;
      rankedInputs.push_back({estimateInputBindingCost(swPid, hwPid), hi});
    }
    if (rankedInputs.empty())
      return false;
    llvm::stable_sort(rankedInputs, [&](const auto &lhs, const auto &rhs) {
      if (std::abs(lhs.first - rhs.first) > 1e-9)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });
    unsigned bestHi = rankedInputs.front().second;
    state.mapPort(swPid, placedHwNode->inputPorts[bestHi], dfg, adg);
    usedIn[bestHi] = true;
  }

  llvm::SmallVector<bool, 8> usedOut(placedHwNode->outputPorts.size(), false);
  for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
    IdIndex swPid = swNode->outputPorts[oi];
    const Port *sp = dfg.getPort(swPid);
    if (!sp)
      continue;

    llvm::SmallVector<std::pair<double, unsigned>, 8> rankedOutputs;
    for (unsigned hi = 0; hi < placedHwNode->outputPorts.size(); ++hi) {
      if (usedOut[hi])
        continue;
      IdIndex hwPid = placedHwNode->outputPorts[hi];
      if (!state.hwPortToSwPorts[hwPid].empty())
        continue;
      const Port *hp = adg.getPort(hwPid);
      if (!hp || !canMapSoftwareTypeToHardware(sp->type, hp->type))
        continue;
      rankedOutputs.push_back({estimateOutputBindingCost(swPid, hwPid), hi});
    }
    if (rankedOutputs.empty())
      return false;
    llvm::stable_sort(rankedOutputs, [&](const auto &lhs, const auto &rhs) {
      if (std::abs(lhs.first - rhs.first) > 1e-9)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });
    unsigned bestHi = rankedOutputs.front().second;
    state.mapPort(swPid, placedHwNode->outputPorts[bestHi], dfg, adg);
    usedOut[bestHi] = true;
  }
  return true;
}

} // namespace fcc

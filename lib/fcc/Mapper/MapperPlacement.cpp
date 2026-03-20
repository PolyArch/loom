#include "MapperCongestionEstimator.h"
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
  return state.isSpatialPEOccupied(peName, adg, ignoreHwNode);
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

  std::vector<uint8_t> fixedInternalMask(dfg.edges.size(), 0);
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeId < static_cast<IdIndex>(edgeKinds.size()) &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg)) {
      fixedInternalMask[edgeId] = 1;
    }
  }
  const_cast<MappingState &>(state).initializeRouteStats(dfg, fixedInternalMask);
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

bool functionUnitPortsCompatible(const Node *swNode, const Node *hwNode,
                                 const Graph &dfg, const Graph &adg) {
  if (!swNode || !hwNode)
    return false;
  if (swNode->inputPorts.size() > hwNode->inputPorts.size())
    return false;
  if (swNode->outputPorts.size() > hwNode->outputPorts.size())
    return false;
  for (unsigned i = 0; i < swNode->inputPorts.size(); ++i) {
    const Port *swPort = dfg.getPort(swNode->inputPorts[i]);
    const Port *hwPort = adg.getPort(hwNode->inputPorts[i]);
    if (!swPort || !hwPort ||
        !canMapSoftwareTypeToHardware(swPort->type, hwPort->type))
      return false;
  }
  for (unsigned i = 0; i < swNode->outputPorts.size(); ++i) {
    const Port *swPort = dfg.getPort(swNode->outputPorts[i]);
    const Port *hwPort = adg.getPort(hwNode->outputPorts[i]);
    if (!swPort || !hwPort ||
        !canMapSoftwareTypeToHardware(swPort->type, hwPort->type))
      return false;
  }
  return true;
}

bool canMapSoftwareTypeToDirectMemoryHardware(mlir::Type swType,
                                              mlir::Type hwType) {
  if (mlir::isa<mlir::MemRefType>(swType) || mlir::isa<mlir::MemRefType>(hwType))
    return canMapSoftwareTypeToHardware(swType, hwType);
  return canMapSoftwareTypeToBridgeHardware(swType, hwType);
}

bool directMemoryHardwarePortAllowsSharing(const Port *hwPort) {
  if (!hwPort)
    return false;
  if (mlir::isa<mlir::MemRefType>(hwPort->type))
    return true;
  auto info = detail::getPortTypeInfo(hwPort->type);
  return info && info->isTagged;
}

IdIndex findBridgePortForCategoryLane(const BridgeInfo &bridge, bool isInput,
                                      BridgePortCategory cat, unsigned lane) {
  llvm::ArrayRef<IdIndex> ports =
      isInput ? llvm::ArrayRef<IdIndex>(bridge.inputPorts)
              : llvm::ArrayRef<IdIndex>(bridge.outputPorts);
  llvm::ArrayRef<BridgePortCategory> categories =
      isInput ? llvm::ArrayRef<BridgePortCategory>(bridge.inputCategories)
              : llvm::ArrayRef<BridgePortCategory>(bridge.outputCategories);
  llvm::ArrayRef<unsigned> lanes =
      isInput ? llvm::ArrayRef<unsigned>(bridge.inputLanes)
              : llvm::ArrayRef<unsigned>(bridge.outputLanes);
  for (size_t idx = 0; idx < ports.size() && idx < categories.size() &&
                       idx < lanes.size();
       ++idx) {
    if (categories[idx] == cat && lanes[idx] == lane)
      return ports[idx];
  }
  return INVALID_ID;
}

std::optional<std::pair<double, double>>
getPortPlacementPos(IdIndex portId, const Graph &adg,
                    const ADGFlattener &flattener) {
  const Port *port = adg.getPort(portId);
  if (!port || port->parentNode == INVALID_ID)
    return std::nullopt;
  auto [row, col] = flattener.getNodeGridPos(port->parentNode);
  if (row < 0 || col < 0)
    return std::nullopt;
  return std::make_pair(static_cast<double>(row), static_cast<double>(col));
}

std::optional<std::pair<double, double>>
estimateSoftwarePortPlacementPos(IdIndex swNode, IdIndex swPort,
                                 IdIndex hwNode, bool isInput,
                                 const Graph &dfg, const Graph &adg,
                                 const ADGFlattener &flattener) {
  const Node *swN = dfg.getNode(swNode);
  const Node *hwN = adg.getNode(hwNode);
  if (!swN || !hwN)
    return std::nullopt;

  if (isSoftwareMemoryInterfaceOp(getNodeAttrStr(swN, "op_name"))) {
    bool isExtMem = (getNodeAttrStr(swN, "op_name") == "handshake.extmemory");
    BridgeInfo bridge = BridgeInfo::extract(hwN);
    DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swN, dfg, isExtMem);

    if (isInput) {
      auto it = std::find(swN->inputPorts.begin(), swN->inputPorts.end(),
                          swPort);
      if (it != swN->inputPorts.end()) {
        unsigned absIdx = static_cast<unsigned>(std::distance(swN->inputPorts.begin(), it));
        if (absIdx >= memInfo.swInSkip) {
          unsigned relIdx = absIdx - memInfo.swInSkip;
          BridgePortCategory cat = memInfo.classifyInput(relIdx);
          unsigned lane = memInfo.inputLocalLane(relIdx);
          if (bridge.hasBridge) {
            IdIndex hwPortId =
                findBridgePortForCategoryLane(bridge, true, cat, lane);
            if (auto pos = getPortPlacementPos(hwPortId, adg, flattener))
              return pos;
          }
          if (IdIndex hwPortId =
                  getExpandedMemoryInputPort(hwN, adg, isExtMem, cat, lane);
              hwPortId != INVALID_ID) {
            if (auto pos = getPortPlacementPos(hwPortId, adg, flattener))
              return pos;
          }
        }
      }
    } else {
      auto it = std::find(swN->outputPorts.begin(), swN->outputPorts.end(),
                          swPort);
      if (it != swN->outputPorts.end()) {
        unsigned outIdx =
            static_cast<unsigned>(std::distance(swN->outputPorts.begin(), it));
        BridgePortCategory cat = memInfo.classifyOutput(outIdx);
        unsigned lane = memInfo.outputLocalLane(outIdx);
        if (bridge.hasBridge) {
          IdIndex hwPortId =
              findBridgePortForCategoryLane(bridge, false, cat, lane);
          if (auto pos = getPortPlacementPos(hwPortId, adg, flattener))
            return pos;
        }
        if (IdIndex hwPortId = getExpandedMemoryOutputPort(hwN, adg, cat, lane);
            hwPortId != INVALID_ID) {
          if (auto pos = getPortPlacementPos(hwPortId, adg, flattener))
            return pos;
        }
      }
    }
  }

  auto [row, col] = flattener.getNodeGridPos(hwNode);
  if (row < 0 || col < 0)
    return std::nullopt;
  return std::make_pair(static_cast<double>(row), static_cast<double>(col));
}

IdIndex getExpandedMemoryInputPort(const Node *hwNode, const Graph &adg,
                                   bool isExtMem, BridgePortCategory cat,
                                   unsigned lane) {
  if (!hwNode)
    return INVALID_ID;
  unsigned ldCount =
      static_cast<unsigned>(std::max<int64_t>(0, getNodeAttrInt(hwNode, "ldCount", 0)));
  unsigned stCount =
      static_cast<unsigned>(std::max<int64_t>(0, getNodeAttrInt(hwNode, "stCount", 0)));
  unsigned portIdx = isExtMem ? 1u : 0u;
  auto finalizePort = [&](unsigned idx) -> IdIndex {
    if (idx >= hwNode->inputPorts.size())
      return INVALID_ID;
    IdIndex hwPortId = hwNode->inputPorts[idx];
    if (lane == 0)
      return hwPortId;
    const Port *hwPort = adg.getPort(hwPortId);
    return directMemoryHardwarePortAllowsSharing(hwPort) ? hwPortId
                                                         : INVALID_ID;
  };

  if (ldCount > 0) {
    if (cat == BridgePortCategory::LdAddr) {
      return finalizePort(portIdx);
    }
    ++portIdx;
  }

  if (stCount > 0) {
    if (cat == BridgePortCategory::StAddr) {
      return finalizePort(portIdx);
    }
    ++portIdx;

    if (cat == BridgePortCategory::StData) {
      return finalizePort(portIdx);
    }
  }

  return INVALID_ID;
}

IdIndex getExpandedMemoryOutputPort(const Node *hwNode, const Graph &adg,
                                    BridgePortCategory cat, unsigned lane) {
  if (!hwNode)
    return INVALID_ID;
  unsigned ldCount =
      static_cast<unsigned>(std::max<int64_t>(0, getNodeAttrInt(hwNode, "ldCount", 0)));
  unsigned stCount =
      static_cast<unsigned>(std::max<int64_t>(0, getNodeAttrInt(hwNode, "stCount", 0)));
  unsigned portIdx = 0;
  auto finalizePort = [&](unsigned idx) -> IdIndex {
    if (idx >= hwNode->outputPorts.size())
      return INVALID_ID;
    IdIndex hwPortId = hwNode->outputPorts[idx];
    if (lane == 0)
      return hwPortId;
    const Port *hwPort = adg.getPort(hwPortId);
    return directMemoryHardwarePortAllowsSharing(hwPort) ? hwPortId
                                                         : INVALID_ID;
  };

  if (ldCount > 0) {
    if (cat == BridgePortCategory::LdData) {
      return finalizePort(portIdx);
    }
    ++portIdx;

    if (cat == BridgePortCategory::LdDone) {
      return finalizePort(portIdx);
    }
    ++portIdx;
  }

  if (stCount > 0 && cat == BridgePortCategory::StDone) {
    return finalizePort(portIdx);
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

std::vector<double> buildEdgePlacementWeightCache(const Graph &dfg) {
  std::vector<double> edgeWeights(dfg.edges.size(), 1.0);
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId)
    edgeWeights[edgeId] = classifyEdgePlacementWeight(dfg, edgeId);
  return edgeWeights;
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
                         const CandidateMap &candidates) {
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

CandidateSetMap buildCandidateSetMap(const CandidateMap &candidates) {
  CandidateSetMap candidateSets;
  for (const auto &entry : candidates) {
    auto &set = candidateSets[entry.first];
    for (IdIndex hwNode : entry.second)
      set.insert(hwNode);
  }
  return candidateSets;
}

double computeLocalSpreadPenalty(IdIndex hwNode, const MappingState &state,
                                 const Graph &adg,
                                 const ADGFlattener &flattener) {
  auto [row, col] = flattener.getNodeGridPos(hwNode);
  if (row < 0 || col < 0)
    return 0.0;

  const Node *hwNodePtr = adg.getNode(hwNode);
  bool isFunctionalCandidate =
      hwNodePtr && getNodeAttrStr(hwNodePtr, "resource_class") == "functional";
  unsigned selfOccupancy =
      (isFunctionalCandidate && hwNode < state.hwNodeToSwNodes.size() &&
       !state.hwNodeToSwNodes[hwNode].empty())
          ? static_cast<unsigned>(state.hwNodeToSwNodes[hwNode].size())
          : 0u;

  unsigned sameRow = state.getFunctionalRowOccupancy(row);
  unsigned sameCol = state.getFunctionalColOccupancy(col);
  unsigned nearby = 0;
  for (int dr = -2; dr <= 2; ++dr) {
    for (int dc = -2; dc <= 2; ++dc) {
      if (std::abs(dr) + std::abs(dc) > 2)
        continue;
      nearby += state.getFunctionalCellOccupancy(row + dr, col + dc);
    }
  }
  sameRow = (sameRow >= selfOccupancy) ? (sameRow - selfOccupancy) : 0;
  sameCol = (sameCol >= selfOccupancy) ? (sameCol - selfOccupancy) : 0;
  nearby = (nearby >= selfOccupancy) ? (nearby - selfOccupancy) : 0;

  return 0.12 * static_cast<double>(sameRow + sameCol) +
         0.25 * static_cast<double>(nearby);
}

std::vector<IdIndex>
collectUnroutedEdges(const MappingState &state, const Graph &dfg,
                     llvm::ArrayRef<TechMappedEdgeKind> edgeKinds) {
  if (state.routeStatsInitialized) {
    return std::vector<IdIndex>(state.routeStatsUnroutedEdgeSet.begin(),
                                state.routeStatsUnroutedEdgeSet.end());
  }
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

RoutingEdgeStats computeRoutingEdgeStats(const MappingState &state,
                                         const Graph &dfg,
                                         llvm::ArrayRef<TechMappedEdgeKind>
                                             edgeKinds) {
  if (state.routeStatsInitialized) {
    RoutingEdgeStats stats;
    stats.overallEdges = state.routeStatsCounters.overallEdges;
    stats.fixedInternalEdges = state.routeStatsCounters.fixedInternalEdges;
    stats.directBindingEdges = state.routeStatsCounters.directBindingEdges;
    stats.routerEdges = state.routeStatsCounters.routerEdges;
    stats.routedOverallEdges = state.routeStatsCounters.routedOverallEdges;
    stats.routedRouterEdges = state.routeStatsCounters.routedRouterEdges;
    stats.unroutedRouterEdges = state.routeStatsCounters.unroutedRouterEdges;
    return stats;
  }
  RoutingEdgeStats stats;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    ++stats.overallEdges;

    bool fixedInternal =
        edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg);
    if (fixedInternal) {
      ++stats.fixedInternalEdges;
      ++stats.routedOverallEdges;
      continue;
    }

    bool hasPath = edgeId < state.swEdgeToHwPaths.size() &&
                   !state.swEdgeToHwPaths[edgeId].empty();
    bool directBinding =
        hasPath && state.swEdgeToHwPaths[edgeId].size() == 2 &&
        state.swEdgeToHwPaths[edgeId][0] == state.swEdgeToHwPaths[edgeId][1];
    if (directBinding) {
      ++stats.directBindingEdges;
      ++stats.routedOverallEdges;
      continue;
    }

    ++stats.routerEdges;
    if (hasPath) {
      ++stats.routedOverallEdges;
      ++stats.routedRouterEdges;
    } else {
      ++stats.unroutedRouterEdges;
    }
  }
  return stats;
}

unsigned countRoutedEdges(const MappingState &state, const Graph &dfg,
                          llvm::ArrayRef<TechMappedEdgeKind> edgeKinds) {
  if (state.routeStatsInitialized)
    return state.routeStatsCounters.routedOverallEdges;
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

int manhattanDistance(IdIndex lhsHwNode, IdIndex rhsHwNode,
                      const ADGFlattener &flattener) {
  auto [lhsRow, lhsCol] = flattener.getNodeGridPos(lhsHwNode);
  auto [rhsRow, rhsCol] = flattener.getNodeGridPos(rhsHwNode);
  if (lhsRow < 0 || lhsCol < 0 || rhsRow < 0 || rhsCol < 0)
    return 0;
  return std::abs(lhsRow - rhsRow) + std::abs(lhsCol - rhsCol);
}

bool isWithinMoveRadius(IdIndex lhsHwNode, IdIndex rhsHwNode,
                        const ADGFlattener &flattener, unsigned radius) {
  if (radius == 0)
    return true;
  return manhattanDistance(lhsHwNode, rhsHwNode, flattener) <=
         static_cast<int>(radius);
}

bool canRelocateNode(
    IdIndex swNode, IdIndex newHwNode, IdIndex oldHwNode,
    const MappingState &state, const Graph &adg, const ADGFlattener &flattener,
    const CandidateMap &candidates, const CandidateSetMap *candidateSets) {
  auto candIt = candidates.find(swNode);
  if (candIt == candidates.end())
    return false;
  if (candidateSets) {
    auto setIt = candidateSets->find(swNode);
    if (setIt == candidateSets->end() || !setIt->second.contains(newHwNode))
      return false;
  } else if (std::find(candIt->second.begin(), candIt->second.end(), newHwNode) ==
             candIt->second.end()) {
    return false;
  }
  if (newHwNode != oldHwNode && !state.hwNodeToSwNodes[newHwNode].empty())
    return false;

  const Node *hwNode = adg.getNode(newHwNode);
  if (!hwNode)
    return false;
  llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
  if (!peName.empty() &&
      isSpatialPEOccupied(state, adg, flattener, peName, oldHwNode))
    return false;
  return true;
}

bool canSwapNodes(
    IdIndex swA, IdIndex swB, IdIndex hwA, IdIndex hwB,
    const MappingState &state, const Graph &adg, const ADGFlattener &flattener,
    const CandidateMap &candidates, const CandidateSetMap *candidateSets) {
  auto candItA = candidates.find(swA);
  auto candItB = candidates.find(swB);
  if (candItA == candidates.end() || candItB == candidates.end())
    return false;
  if (candidateSets) {
    auto setItA = candidateSets->find(swA);
    auto setItB = candidateSets->find(swB);
    if (setItA == candidateSets->end() || setItB == candidateSets->end())
      return false;
    if (!setItA->second.contains(hwB) || !setItB->second.contains(hwA))
      return false;
  } else {
    if (std::find(candItA->second.begin(), candItA->second.end(), hwB) ==
        candItA->second.end())
      return false;
    if (std::find(candItB->second.begin(), candItB->second.end(), hwA) ==
        candItB->second.end())
      return false;
  }

  const Node *hwNodeA = adg.getNode(hwA);
  const Node *hwNodeB = adg.getNode(hwB);
  if (!hwNodeA || !hwNodeB)
    return false;

  llvm::StringRef peNameA = getNodeAttrStr(hwNodeA, "pe_name");
  llvm::StringRef peNameB = getNodeAttrStr(hwNodeB, "pe_name");
  if (!peNameA.empty() && isSpatialPEName(flattener, peNameA) &&
      peNameA == peNameB)
    return false;

  if (!peNameA.empty() &&
      isSpatialPEOccupied(state, adg, flattener, peNameA, hwA))
    return false;
  if (!peNameB.empty() &&
      isSpatialPEOccupied(state, adg, flattener, peNameB, hwB))
    return false;

  return true;
}

double computeUnroutedPenalty(const MappingState &state, const Graph &dfg,
                              llvm::ArrayRef<TechMappedEdgeKind> edgeKinds) {
  if (state.routeStatsInitialized)
    return state.routeStatsUnroutedPenalty;
  double penalty = 0.0;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    if (edgeId < state.swEdgeToHwPaths.size() &&
        !state.swEdgeToHwPaths[edgeId].empty())
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;
    bool srcMapped = srcPort->parentNode < state.swNodeToHwNode.size() &&
                     state.swNodeToHwNode[srcPort->parentNode] != INVALID_ID;
    bool dstMapped = dstPort->parentNode < state.swNodeToHwNode.size() &&
                     state.swNodeToHwNode[dstPort->parentNode] != INVALID_ID;
    if (!srcMapped || !dstMapped)
      continue;
    penalty += classifyEdgePlacementWeight(dfg, edgeId);
  }
  return penalty;
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
          unsigned hwLdCount = static_cast<unsigned>(
              std::max<int64_t>(0, getNodeAttrInt(hwNode, "ldCount", 0)));
          unsigned hwStCount = static_cast<unsigned>(
              std::max<int64_t>(0, getNodeAttrInt(hwNode, "stCount", 0)));
          if (static_cast<unsigned>(std::max<int64_t>(memInfo.ldCount, 0)) >
                  hwLdCount ||
              static_cast<unsigned>(std::max<int64_t>(memInfo.stCount, 0)) >
                  hwStCount) {
            continue;
          }

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
            IdIndex hwPid =
                getExpandedMemoryInputPort(hwNode, adg, isExtMem, cat, lane);
            const Port *hp = adg.getPort(hwPid);
            if (!sp || hwPid == INVALID_ID || !hp ||
                !canMapSoftwareTypeToDirectMemoryHardware(sp->type,
                                                          hp->type)) {
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
            IdIndex hwPid = getExpandedMemoryOutputPort(hwNode, adg, cat, lane);
            const Port *hp = adg.getPort(hwPid);
            if (!sp || !hp ||
                !canMapSoftwareTypeToDirectMemoryHardware(sp->type,
                                                          hp->type)) {
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
      if (opMatchesFU(opName, hwNode) &&
          functionUnitPortsCompatible(swNode, hwNode, dfg, adg)) {
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
  const Node *hwN = adg.getNode(hwNode);
  if (!hwN)
    return -1.0e18;

  double weightedDist = 0.0;
  double totalWeight = 0.0;
  auto accumulateNeighborAt = [&](double anchorRow, double anchorCol,
                                  IdIndex otherSwNode, IdIndex edgeId) {
    auto estimate =
        estimateNodePlacementPos(otherSwNode, state, dfg, flattener, candidates);
    if (!estimate)
      return;
    double edgeWeight = classifyEdgePlacementWeight(dfg, edgeId);
    weightedDist +=
        edgeWeight * (std::abs(anchorRow - estimate->first) +
                      std::abs(anchorCol - estimate->second));
    totalWeight += edgeWeight;
  };
  auto accumulateNeighbor = [&](IdIndex otherSwNode, IdIndex edgeId) {
    accumulateNeighborAt(static_cast<double>(hwRow), static_cast<double>(hwCol),
                         otherSwNode, edgeId);
  };

  bool usedBridgeBoundaryScoring = false;
  if (isSoftwareMemoryInterfaceOp(getNodeAttrStr(swN, "op_name"))) {
    bool isExtMem = (getNodeAttrStr(swN, "op_name") == "handshake.extmemory");
    BridgeInfo bridge = BridgeInfo::extract(hwN);
    if (bridge.hasBridge) {
      DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swN, dfg, isExtMem);
      for (unsigned si = memInfo.swInSkip; si < swN->inputPorts.size(); ++si) {
        IdIndex swPortId = swN->inputPorts[si];
        const Port *swPort = dfg.getPort(swPortId);
        if (!swPort)
          continue;
        BridgePortCategory cat =
            memInfo.classifyInput(si - memInfo.swInSkip);
        unsigned lane = memInfo.inputLocalLane(si - memInfo.swInSkip);
        IdIndex hwPortId = findBridgePortForCategoryLane(bridge, true, cat, lane);
        auto hwPortPos = getPortPlacementPos(hwPortId, adg, flattener);
        if (!hwPortPos)
          continue;
        for (IdIndex eid : swPort->connectedEdges) {
          const Edge *edge = dfg.getEdge(eid);
          if (!edge || edge->dstPort != swPortId)
            continue;
          const Port *srcPort = dfg.getPort(edge->srcPort);
          if (!srcPort || srcPort->parentNode == INVALID_ID)
            continue;
          accumulateNeighborAt(hwPortPos->first, hwPortPos->second,
                               srcPort->parentNode, eid);
          usedBridgeBoundaryScoring = true;
        }
      }
      for (unsigned oi = 0; oi < swN->outputPorts.size(); ++oi) {
        IdIndex swPortId = swN->outputPorts[oi];
        const Port *swPort = dfg.getPort(swPortId);
        if (!swPort)
          continue;
        BridgePortCategory cat = memInfo.classifyOutput(oi);
        unsigned lane = memInfo.outputLocalLane(oi);
        IdIndex hwPortId =
            findBridgePortForCategoryLane(bridge, false, cat, lane);
        auto hwPortPos = getPortPlacementPos(hwPortId, adg, flattener);
        if (!hwPortPos)
          continue;
        for (IdIndex eid : swPort->connectedEdges) {
          const Edge *edge = dfg.getEdge(eid);
          if (!edge || edge->srcPort != swPortId)
            continue;
          const Port *dstPort = dfg.getPort(edge->dstPort);
          if (!dstPort || dstPort->parentNode == INVALID_ID)
            continue;
          accumulateNeighborAt(hwPortPos->first, hwPortPos->second,
                               dstPort->parentNode, eid);
          usedBridgeBoundaryScoring = true;
        }
      }
    }
  }

  if (!usedBridgeBoundaryScoring) {
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
  }

  double cost = 0.0;
  if (totalWeight > 0.0)
    cost += weightedDist / totalWeight;
  else
    cost += 0.25 * (std::abs(hwRow) + std::abs(hwCol));

  if (activeMemorySharingPenalty > 0.0 &&
      isSoftwareMemoryInterfaceOp(getNodeAttrStr(swN, "op_name")) &&
      getNodeAttrStr(hwN, "resource_class") == "memory") {
    unsigned colocatedMemoryInterfaces = 0;
    if (hwNode < state.hwNodeToSwNodes.size()) {
      for (IdIndex otherSwId : state.hwNodeToSwNodes[hwNode]) {
        if (otherSwId == swNode)
          continue;
        const Node *otherSwNode = dfg.getNode(otherSwId);
        if (!otherSwNode || otherSwNode->kind != Node::OperationNode)
          continue;
        if (isSoftwareMemoryInterfaceOp(getNodeAttrStr(otherSwNode, "op_name")))
          ++colocatedMemoryInterfaces;
      }
    }
    if (colocatedMemoryInterfaces > 0) {
      cost += activeMemorySharingPenalty *
              static_cast<double>(colocatedMemoryInterfaces);
    }
  }

  cost += 0.6 * computeLocalSpreadPenalty(hwNode, state, adg, flattener);

  if (activeCongestionEstimator && activeFlattener &&
      activeCongestionPlacementWeight > 0.0) {
    double congestionPenalty = 0.0;
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
        IdIndex srcHw = srcPort->parentNode < state.swNodeToHwNode.size()
                            ? state.swNodeToHwNode[srcPort->parentNode]
                            : INVALID_ID;
        if (srcHw != INVALID_ID) {
          congestionPenalty +=
              activeCongestionEstimator->demandCapacityRatio(
                  srcHw, hwNode, adg, *activeFlattener);
        }
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
        IdIndex dstHw = dstPort->parentNode < state.swNodeToHwNode.size()
                            ? state.swNodeToHwNode[dstPort->parentNode]
                            : INVALID_ID;
        if (dstHw != INVALID_ID) {
          congestionPenalty +=
              activeCongestionEstimator->demandCapacityRatio(
                  hwNode, dstHw, adg, *activeFlattener);
        }
      }
    }
    cost += activeCongestionPlacementWeight * congestionPenalty;
  }

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

    llvm::SmallVector<IdIndex, 8> tryOrder(shortlist.begin(), shortlist.end());
    for (const auto &candidate : rankedCandidates) {
      if (llvm::is_contained(tryOrder, candidate.second))
        continue;
      tryOrder.push_back(candidate.second);
    }

    bool placed = false;
    ActionResult lastResult = ActionResult::FailedInternalError;
    for (IdIndex hwCandidate : tryOrder) {
      auto savepoint = state.beginSavepoint();
      lastResult = state.mapNode(swId, hwCandidate, dfg, adg);
      if (lastResult == ActionResult::Success &&
          bindMappedNodePorts(swId, state, dfg, adg)) {
        bestHw = hwCandidate;
        state.commitSavepoint(savepoint);
        placed = true;
        break;
      }
      state.rollbackSavepoint(savepoint);
    }
    if (!placed) {
      llvm::errs() << "Mapper: port binding failed for " << swId
                   << " after trying " << tryOrder.size() << " candidate(s)\n";
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
    cost += 0.006 * load * load;
  for (double load : colCutLoad)
    cost += 0.006 * load * load;
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

  auto verifyIncidentReachability = [&]() -> bool {
    llvm::StringRef swOpName = getNodeAttrStr(swNode, "op_name");
    if (mapper_detail::isSoftwareMemoryInterfaceOp(swOpName))
      return true;
    llvm::DenseMap<IdIndex, double> emptyHistory;
    llvm::DenseSet<IdIndex> seenEdges;
    auto checkPorts = [&](llvm::ArrayRef<IdIndex> ports) -> bool {
      for (IdIndex swPid : ports) {
        const Port *swPort = dfg.getPort(swPid);
        if (!swPort)
          continue;
        for (IdIndex edgeId : swPort->connectedEdges) {
          if (!seenEdges.insert(edgeId).second)
            continue;
          const Edge *edge = dfg.getEdge(edgeId);
          if (!edge)
            continue;
          IdIndex srcHwPid =
              edge->srcPort < state.swPortToHwPort.size()
                  ? state.swPortToHwPort[edge->srcPort]
                  : INVALID_ID;
          IdIndex dstHwPid =
              edge->dstPort < state.swPortToHwPort.size()
                  ? state.swPortToHwPort[edge->dstPort]
                  : INVALID_ID;
          if (srcHwPid == INVALID_ID || dstHwPid == INVALID_ID)
            continue;
          if (srcHwPid == dstHwPid)
            continue;
          const Port *srcHwPort = adg.getPort(srcHwPid);
          const Port *dstHwPort = adg.getPort(dstHwPid);
          const Node *srcHwNode =
              (srcHwPort && srcHwPort->parentNode != INVALID_ID)
                  ? adg.getNode(srcHwPort->parentNode)
                  : nullptr;
          const Node *dstHwNode =
              (dstHwPort && dstHwPort->parentNode != INVALID_ID)
                  ? adg.getNode(dstHwPort->parentNode)
                  : nullptr;
          if (srcHwNode && dstHwNode) {
            llvm::StringRef srcPe = getNodeAttrStr(srcHwNode, "pe_name");
            llvm::StringRef dstPe = getNodeAttrStr(dstHwNode, "pe_name");
            if (!srcPe.empty() && srcPe == dstPe)
              continue;
          }
          auto path =
              findPath(srcHwPid, dstHwPid, edgeId, state, dfg, adg, emptyHistory);
          if (path.empty())
            return false;
        }
      }
      return true;
    };
    return checkPorts(swNode->inputPorts) && checkPorts(swNode->outputPorts);
  };

  if (getNodeAttrStr(placedHwNode, "resource_class") != "memory")
    return verifyIncidentReachability();

  BridgeInfo bridge = BridgeInfo::extract(placedHwNode);
  bool isExtMem = (getNodeAttrStr(placedHwNode, "op_kind") == "extmemory");
  if (bridge.hasBridge) {
    DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);
    return bindBridgeInputs(bridge, memInfo, swNode, placedHwNode, dfg, adg,
                            state) &&
           bindBridgeOutputs(bridge, memInfo, swNode, placedHwNode, dfg, adg,
                             state) &&
           verifyIncidentReachability();
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
          getExpandedMemoryInputPort(placedHwNode, adg, isExtMem, cat, lane);
      const Port *hp = adg.getPort(hwPid);
      if (!sp || hwPid == INVALID_ID || !hp ||
          (!state.hwPortToSwPorts[hwPid].empty() &&
           !directMemoryHardwarePortAllowsSharing(hp)) ||
          !canMapSoftwareTypeToDirectMemoryHardware(sp->type, hp->type)) {
        return false;
      }
      if (state.mapPort(swPid, hwPid, dfg, adg) != ActionResult::Success)
        return false;
    }

    for (unsigned oi = 0; oi < swNode->outputPorts.size(); ++oi) {
      IdIndex swPid = swNode->outputPorts[oi];
      const Port *sp = dfg.getPort(swPid);
      BridgePortCategory cat = memInfo.classifyOutput(oi);
      unsigned lane = memInfo.outputLocalLane(oi);
      IdIndex hwPid = getExpandedMemoryOutputPort(placedHwNode, adg, cat, lane);
      const Port *hp = adg.getPort(hwPid);
      if (!sp || hwPid == INVALID_ID || !hp ||
          (!state.hwPortToSwPorts[hwPid].empty() &&
           !directMemoryHardwarePortAllowsSharing(hp)) ||
          !canMapSoftwareTypeToDirectMemoryHardware(sp->type, hp->type)) {
        return false;
      }
      if (state.mapPort(swPid, hwPid, dfg, adg) != ActionResult::Success)
        return false;
    }
    return verifyIncidentReachability();
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
  return verifyIncidentReachability();
}

} // namespace fcc

#include "MapperCongestionEstimator.h"
#include "MapperInternal.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/OpCompat.h"
#include "loom/Mapper/TopologyModel.h"
#include "loom/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <random>

namespace loom {

// ---------------------------------------------------------------------------
// Helper function implementations (mapper_detail namespace)
// ---------------------------------------------------------------------------

namespace mapper_detail {

namespace {

thread_local const TopologyModel *activeTopologyModel = nullptr;
thread_local const MapperTimingOptions *activeTimingOptions = nullptr;

struct RecurrenceProxyCache {
  const Graph *dfg = nullptr;
  std::vector<uint8_t> recurrenceNodeMask;
  std::vector<uint8_t> recurrenceEdgeMask;
};

thread_local RecurrenceProxyCache activeRecurrenceProxyCache;

const MapperTimingOptions &getTimingProxyOptions() {
  static const MapperTimingOptions defaults;
  return activeTimingOptions ? *activeTimingOptions : defaults;
}

bool isRecurrenceProxyOperationNode(const Node *node) {
  return node && node->kind == Node::OperationNode;
}

IdIndex findCarryNextEdge(IdIndex carryNodeId, const Graph &dfg) {
  const Node *carryNode = dfg.getNode(carryNodeId);
  if (!carryNode || getNodeAttrStr(carryNode, "op_name") != "dataflow.carry" ||
      carryNode->inputPorts.size() < 3) {
    return INVALID_ID;
  }
  IdIndex nextPortId = carryNode->inputPorts[2];
  const Port *nextPort = dfg.getPort(nextPortId);
  if (!nextPort)
    return INVALID_ID;
  for (IdIndex edgeId : nextPort->connectedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (edge && edge->dstPort == nextPortId)
      return edgeId;
  }
  return INVALID_ID;
}

void rebuildRecurrenceProxyCache(const Graph &dfg) {
  activeRecurrenceProxyCache = {};
  activeRecurrenceProxyCache.dfg = &dfg;
  activeRecurrenceProxyCache.recurrenceNodeMask.assign(dfg.nodes.size(), 0);
  activeRecurrenceProxyCache.recurrenceEdgeMask.assign(dfg.edges.size(), 0);

  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> outgoingEdgesByNode;
  llvm::DenseSet<IdIndex> operationNodes;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID) {
      continue;
    }
    const Node *srcNode = dfg.getNode(srcPort->parentNode);
    const Node *dstNode = dfg.getNode(dstPort->parentNode);
    if (!isRecurrenceProxyOperationNode(srcNode) ||
        !isRecurrenceProxyOperationNode(dstNode)) {
      continue;
    }
    outgoingEdgesByNode[srcPort->parentNode].push_back(edgeId);
    operationNodes.insert(srcPort->parentNode);
    operationNodes.insert(dstPort->parentNode);
  }

  llvm::DenseMap<IdIndex, unsigned> indexByNode;
  llvm::DenseMap<IdIndex, unsigned> lowlinkByNode;
  llvm::DenseSet<IdIndex> onStack;
  llvm::SmallVector<IdIndex, 16> stack;
  unsigned nextIndex = 0;

  std::function<void(IdIndex)> strongConnect = [&](IdIndex nodeId) {
    indexByNode[nodeId] = nextIndex;
    lowlinkByNode[nodeId] = nextIndex;
    ++nextIndex;
    stack.push_back(nodeId);
    onStack.insert(nodeId);

    for (IdIndex edgeId : outgoingEdgesByNode.lookup(nodeId)) {
      const Edge *edge = dfg.getEdge(edgeId);
      const Port *dstPort = edge ? dfg.getPort(edge->dstPort) : nullptr;
      if (!dstPort || dstPort->parentNode == INVALID_ID)
        continue;
      IdIndex succ = dstPort->parentNode;
      if (!indexByNode.count(succ)) {
        strongConnect(succ);
        lowlinkByNode[nodeId] =
            std::min(lowlinkByNode[nodeId], lowlinkByNode[succ]);
      } else if (onStack.count(succ)) {
        lowlinkByNode[nodeId] =
            std::min(lowlinkByNode[nodeId], indexByNode[succ]);
      }
    }

    if (lowlinkByNode[nodeId] != indexByNode[nodeId])
      return;

    llvm::SmallVector<IdIndex, 8> component;
    while (!stack.empty()) {
      IdIndex member = stack.back();
      stack.pop_back();
      onStack.erase(member);
      component.push_back(member);
      if (member == nodeId)
        break;
    }

    bool hasSelfLoop = false;
    if (component.size() == 1) {
      for (IdIndex edgeId : outgoingEdgesByNode.lookup(component.front())) {
        const Edge *edge = dfg.getEdge(edgeId);
        const Port *dstPort = edge ? dfg.getPort(edge->dstPort) : nullptr;
        if (dstPort && dstPort->parentNode == component.front()) {
          hasSelfLoop = true;
          break;
        }
      }
    }
    if (component.size() == 1 && !hasSelfLoop)
      return;

    llvm::DenseSet<IdIndex> componentNodes(component.begin(), component.end());
    for (IdIndex nodeIdInCycle : component) {
      if (nodeIdInCycle >=
          static_cast<IdIndex>(activeRecurrenceProxyCache.recurrenceNodeMask.size())) {
        continue;
      }
      activeRecurrenceProxyCache.recurrenceNodeMask[nodeIdInCycle] = 1;
      for (IdIndex edgeId : outgoingEdgesByNode.lookup(nodeIdInCycle)) {
        const Edge *edge = dfg.getEdge(edgeId);
        const Port *dstPort = edge ? dfg.getPort(edge->dstPort) : nullptr;
        if (!dstPort || dstPort->parentNode == INVALID_ID ||
            !componentNodes.count(dstPort->parentNode)) {
          continue;
        }
        if (edgeId <
            static_cast<IdIndex>(activeRecurrenceProxyCache.recurrenceEdgeMask.size()))
          activeRecurrenceProxyCache.recurrenceEdgeMask[edgeId] = 1;
      }
    }
  };

  for (IdIndex nodeId : operationNodes) {
    if (!indexByNode.count(nodeId))
      strongConnect(nodeId);
  }

  for (IdIndex nodeId : operationNodes) {
    const Node *node = dfg.getNode(nodeId);
    if (!node || getNodeAttrStr(node, "op_name") != "dataflow.carry")
      continue;
    IdIndex edgeId = findCarryNextEdge(nodeId, dfg);
    if (edgeId == INVALID_ID ||
        edgeId >=
            static_cast<IdIndex>(
                activeRecurrenceProxyCache.recurrenceEdgeMask.size()) ||
        activeRecurrenceProxyCache.recurrenceEdgeMask[edgeId] != 0) {
      continue;
    }

    activeRecurrenceProxyCache.recurrenceNodeMask[nodeId] = 1;
    activeRecurrenceProxyCache.recurrenceEdgeMask[edgeId] = 1;

    const Edge *edge = dfg.getEdge(edgeId);
    const Port *srcPort = edge ? dfg.getPort(edge->srcPort) : nullptr;
    if (!srcPort || srcPort->parentNode == INVALID_ID ||
        srcPort->parentNode == nodeId ||
        srcPort->parentNode >=
            static_cast<IdIndex>(
                activeRecurrenceProxyCache.recurrenceNodeMask.size())) {
      continue;
    }
    activeRecurrenceProxyCache.recurrenceNodeMask[srcPort->parentNode] = 1;
  }
}

void ensureRecurrenceProxyCache(const Graph &dfg) {
  if (activeRecurrenceProxyCache.dfg == &dfg)
    return;
  rebuildRecurrenceProxyCache(dfg);
}

bool isRecurrenceProxyNode(IdIndex swNode, const Graph &dfg) {
  ensureRecurrenceProxyCache(dfg);
  return swNode >= 0 &&
         swNode < static_cast<IdIndex>(
                      activeRecurrenceProxyCache.recurrenceNodeMask.size()) &&
         activeRecurrenceProxyCache.recurrenceNodeMask[swNode] != 0;
}

} // namespace

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
    const TechMapper::Plan &plan,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    const Graph &dfg, const Graph &adg, std::string &diagnostics) {
  llvm::DenseMap<IdIndex, llvm::SmallVector<const TechMapper::Unit *, 4>>
      forcedByHwNode;

  for (const auto &unit : TechMapper::allUnits(plan)) {
    if (unit.contractedNodeId == INVALID_ID || unit.candidates.empty())
      continue;
    auto forcedHwNodeId = TechMapper::findForcedTemporalHwNodeId(unit, adg);
    if (!forcedHwNodeId)
      continue;
    forcedByHwNode[*forcedHwNodeId].push_back(&unit);
  }

  for (const auto &it : forcedByHwNode) {
    IdIndex hwNodeId = it.first;
    const auto &units = it.second;
    if (units.size() < 2)
      continue;

    const auto *firstPreferredCandidate =
        TechMapper::findPreferredUnitCandidate(*units.front());
    const auto *firstConfigInfo =
        TechMapper::findSelectedUnitConfigClass(plan, *units.front());
    unsigned firstConfigClass =
        firstConfigInfo ? firstConfigInfo->id
                        : std::numeric_limits<unsigned>::max();
    llvm::ArrayRef<FUConfigField> firstConfig =
        firstPreferredCandidate ? llvm::ArrayRef<FUConfigField>(
                                      firstPreferredCandidate->configFields)
                                : llvm::ArrayRef<FUConfigField>();
    for (size_t i = 1; i < units.size(); ++i) {
      const auto *otherPreferredCandidate =
          TechMapper::findPreferredUnitCandidate(*units[i]);
      const auto *otherConfigInfo =
          TechMapper::findSelectedUnitConfigClass(plan, *units[i]);
      unsigned otherConfigClass =
          otherConfigInfo ? otherConfigInfo->id
                          : std::numeric_limits<unsigned>::max();
      llvm::ArrayRef<FUConfigField> otherConfig =
          otherPreferredCandidate ? llvm::ArrayRef<FUConfigField>(
                                        otherPreferredCandidate->configFields)
                                  : llvm::ArrayRef<FUConfigField>();
      if ((firstConfigClass != std::numeric_limits<unsigned>::max() ||
           otherConfigClass != std::numeric_limits<unsigned>::max()) &&
          TechMapper::areConfigClassesCompatible(plan, firstConfigClass,
                                                 otherConfigClass))
        continue;
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
      diagnostics += " between config classes " +
                     std::to_string(firstConfigClass) + " and " +
                     std::to_string(otherConfigClass);
      if (const auto *firstInfo =
              TechMapper::findConfigClass(plan, firstConfigClass)) {
        diagnostics += " [" + firstInfo->key + "]";
      }
      if (const auto *otherInfo =
              TechMapper::findConfigClass(plan, otherConfigClass)) {
        diagnostics += " [" + otherInfo->key + "]";
      }
      if (const auto *incompat = TechMapper::findTemporalIncompatibility(
              plan, firstConfigClass, otherConfigClass)) {
        diagnostics += ": " + incompat->reason;
      }
      diagnostics += " (contracted nodes " +
                     std::to_string(units.front()->contractedNodeId) + " and " +
                     std::to_string(units[i]->contractedNodeId) + ")";
      return true;
    }
  }

  struct ForcedTemporalCandidateInfo {
    IdIndex swNodeId = INVALID_ID;
    llvm::SmallVector<unsigned, 4> configClassIds;
  };

  llvm::DenseMap<IdIndex, llvm::SmallVector<ForcedTemporalCandidateInfo, 4>>
      forcedCandidatesByHwNode;
  for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++swNodeId) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;

    auto candidateIt = candidates.find(swNodeId);
    if (candidateIt == candidates.end() || candidateIt->second.empty())
      continue;

    IdIndex forcedHwNodeId = candidateIt->second.front();
    bool allSameHwNode = true;
    for (IdIndex hwNodeId : candidateIt->second) {
      if (hwNodeId != forcedHwNodeId) {
        allSameHwNode = false;
        break;
      }
    }
    if (!allSameHwNode)
      continue;

    llvm::ArrayRef<unsigned> supportClassIds;
    if (const auto *contractedSupportClassIds =
            TechMapper::findContractedCandidateSupportClasses(plan, swNodeId)) {
      supportClassIds = *contractedSupportClassIds;
    }
    llvm::SmallVector<unsigned, 8> fallbackSupportClassIds;
    if (supportClassIds.empty()) {
      if (const auto *nodeInfo = TechMapper::findNodeTechInfo(plan, swNodeId)) {
        fallbackSupportClassIds.assign(nodeInfo->supportClassIds.begin(),
                                       nodeInfo->supportClassIds.end());
        supportClassIds = fallbackSupportClassIds;
      }
    }
    bool temporalSupportOnly = !supportClassIds.empty();
    if (temporalSupportOnly) {
      for (unsigned supportClassId : supportClassIds) {
        if (!TechMapper::isTemporalSupportClass(plan, supportClassId)) {
          temporalSupportOnly = false;
          break;
        }
      }
    }
    if (!temporalSupportOnly)
      continue;

    ForcedTemporalCandidateInfo info;
    info.swNodeId = swNodeId;
    llvm::ArrayRef<unsigned> configClassIds;
    if (const auto *contractedConfigClassIds =
            TechMapper::findContractedCandidateConfigClasses(plan, swNodeId)) {
      configClassIds = *contractedConfigClassIds;
    }
    llvm::SmallVector<unsigned, 8> fallbackConfigClassIds;
    if (configClassIds.empty()) {
      if (const auto *nodeInfo = TechMapper::findNodeTechInfo(plan, swNodeId)) {
        fallbackConfigClassIds.assign(nodeInfo->configClassIds.begin(),
                                      nodeInfo->configClassIds.end());
        configClassIds = fallbackConfigClassIds;
      }
    }
    if (!configClassIds.empty()) {
      info.configClassIds.assign(configClassIds.begin(), configClassIds.end());
      std::sort(info.configClassIds.begin(), info.configClassIds.end());
      info.configClassIds.erase(
          std::unique(info.configClassIds.begin(), info.configClassIds.end()),
          info.configClassIds.end());
    }
    forcedCandidatesByHwNode[forcedHwNodeId].push_back(std::move(info));
  }

  for (const auto &it : forcedCandidatesByHwNode) {
    IdIndex hwNodeId = it.first;
    const auto &infos = it.second;
    if (infos.size() < 2)
      continue;

    for (size_t lhsIdx = 0; lhsIdx < infos.size(); ++lhsIdx) {
      for (size_t rhsIdx = lhsIdx + 1; rhsIdx < infos.size(); ++rhsIdx) {
        bool hasCompatibleConfigPair = false;
        if (!infos[lhsIdx].configClassIds.empty() &&
            !infos[rhsIdx].configClassIds.empty()) {
          for (unsigned lhsConfigClassId : infos[lhsIdx].configClassIds) {
            for (unsigned rhsConfigClassId : infos[rhsIdx].configClassIds) {
              if (TechMapper::areConfigClassesCompatible(
                      plan, lhsConfigClassId, rhsConfigClassId)) {
                hasCompatibleConfigPair = true;
                break;
              }
            }
            if (hasCompatibleConfigPair)
              break;
          }
        }
        if (hasCompatibleConfigPair)
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
        diagnostics += " between contracted nodes " +
                       std::to_string(infos[lhsIdx].swNodeId) + " and " +
                       std::to_string(infos[rhsIdx].swNodeId);
        return true;
      }
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
  return opcompat::getCompatibleOp(dfgOpName);
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
  ensureRecurrenceProxyCache(dfg);
  if (edgeId >= 0 &&
      edgeId < static_cast<IdIndex>(
                   activeRecurrenceProxyCache.recurrenceEdgeMask.size()) &&
      activeRecurrenceProxyCache.recurrenceEdgeMask[edgeId] != 0) {
    weight *= getTimingProxyOptions().recurrenceEdgeWeightMultiplier;
  }
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
  if (isRecurrenceProxyNode(swNode, dfg))
    weight += getTimingProxyOptions().recurrenceEdgeWeightMultiplier;
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
  const Node *hwNodePtr = adg.getNode(hwNode);
  bool isFunctionalCandidate =
      hwNodePtr && getNodeAttrStr(hwNodePtr, "resource_class") == "functional";
  unsigned selfOccupancy =
      (isFunctionalCandidate && hwNode < state.hwNodeToSwNodes.size() &&
       !state.hwNodeToSwNodes[hwNode].empty())
          ? static_cast<unsigned>(state.hwNodeToSwNodes[hwNode].size())
          : 0u;

  if (const TopologyModel *topologyModel = getActiveTopologyModel()) {
    unsigned nearbyRadius1 = 0;
    unsigned nearbyRadius2 = 0;
    for (IdIndex nearbyHwNode :
         topologyModel->placeableNodesWithinRadius(hwNode, 2)) {
      if (nearbyHwNode >= state.hwNodeToSwNodes.size())
        continue;
      const Node *nearbyNodePtr = adg.getNode(nearbyHwNode);
      if (!nearbyNodePtr ||
          getNodeAttrStr(nearbyNodePtr, "resource_class") != "functional") {
        continue;
      }
      unsigned occupancy =
          static_cast<unsigned>(state.hwNodeToSwNodes[nearbyHwNode].size());
      if (occupancy == 0)
        continue;
      unsigned radius =
          topologyModel->undirectedNodeDistance(hwNode, nearbyHwNode);
      if (radius <= 1)
        nearbyRadius1 += occupancy;
      nearbyRadius2 += occupancy;
    }
    nearbyRadius1 =
        (nearbyRadius1 >= selfOccupancy) ? (nearbyRadius1 - selfOccupancy) : 0;
    nearbyRadius2 =
        (nearbyRadius2 >= selfOccupancy) ? (nearbyRadius2 - selfOccupancy) : 0;
    return 0.37 * static_cast<double>(nearbyRadius1) +
           0.18 * static_cast<double>(nearbyRadius2);
  }

  auto [row, col] = flattener.getNodeGridPos(hwNode);
  if (row < 0 || col < 0)
    return 0.0;

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

void setActiveTopologyModel(const TopologyModel *model) {
  activeTopologyModel = model;
}

const TopologyModel *getActiveTopologyModel() {
  return activeTopologyModel;
}

void setActiveTimingOptions(const MapperTimingOptions *opts) {
  activeTimingOptions = opts;
}

const MapperTimingOptions *getActiveTimingOptions() {
  return activeTimingOptions;
}

int placementDistance(IdIndex lhsHwNode, IdIndex rhsHwNode,
                      const ADGFlattener &flattener) {
  if (activeTopologyModel) {
    return static_cast<int>(
        activeTopologyModel->placementDistance(lhsHwNode, rhsHwNode));
  }
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
  if (activeTopologyModel)
    return activeTopologyModel->isWithinMoveRadius(lhsHwNode, rhsHwNode,
                                                   radius);
  return placementDistance(lhsHwNode, rhsHwNode, flattener) <=
         static_cast<int>(radius);
}

double computeNodeTimingPenalty(IdIndex swNode, IdIndex hwNode,
                                const Graph &dfg, const Graph &adg) {
  if (!isRecurrenceProxyNode(swNode, dfg))
    return 0.0;
  const Node *hwNodePtr = adg.getNode(hwNode);
  if (!hwNodePtr)
    return 0.0;
  const auto &timingOpts = getTimingProxyOptions();
  const unsigned latencyCycles = static_cast<unsigned>(
      std::max<int64_t>(0, getNodeAttrInt(hwNodePtr, "latency", 0)));
  const unsigned intervalCycles = static_cast<unsigned>(
      std::max<int64_t>(1, getNodeAttrInt(hwNodePtr, "interval", 1)));
  const double latencyPenalty =
      timingOpts.recurrenceNodeLatencyWeight *
      static_cast<double>(latencyCycles > 0 ? latencyCycles - 1 : 0);
  const double intervalPenalty =
      timingOpts.recurrenceNodeIntervalWeight *
      static_cast<double>(intervalCycles > 0 ? intervalCycles - 1 : 0);
  return latencyPenalty + intervalPenalty;
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

} // namespace loom

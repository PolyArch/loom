#include "fcc/Mapper/Mapper.h"
#include "fcc/Mapper/MapperRelaxedRouting.h"
#include "fcc/Mapper/MapperTiming.h"
#include "ConfigGenInternal.h"
#include "MapperCongestionEstimator.h"
#include "MapperInternal.h"
#include "MapperRoutingCongestion.h"
#include "fcc/Mapper/BridgeBinding.h"
#include "fcc/Mapper/TypeCompat.h"
#include "fcc/Mapper/TopologyModel.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace fcc {

using namespace mapper_detail;

namespace {

unsigned selectLaneCount(const Mapper::Options &opts, const Graph &dfg) {
  unsigned configuredLanes = opts.lanes;
  if (configuredLanes != 0)
    return configuredLanes;
  // Tiny unit-style graphs do not benefit from speculative parallel lanes,
  // and serializing them avoids cross-lane diagnostic races.
  if (dfg.nodes.size() <= opts.lane.autoSerialNodeThreshold)
    return 1;
  unsigned concurrency = std::thread::hardware_concurrency();
  if (concurrency == 0)
    concurrency = 1;
  return std::max(1u, std::min(opts.lane.autoLaneCap, concurrency));
}

struct LaneAttempt {
  bool placementSucceeded = false;
  bool routingSucceeded = false;
  bool usedCPSatGlobalPlacement = false;
  bool budgetExceeded = false;
  unsigned laneIndex = 0;
  unsigned routedEdges = 0;
  size_t totalPathLen = std::numeric_limits<size_t>::max();
  double placementCost = std::numeric_limits<double>::infinity();
  double throughputCost = std::numeric_limits<double>::infinity();
  double estimatedClockPeriod = std::numeric_limits<double>::infinity();
  MapperSearchSummary searchSummary;
  std::string budgetExceededStage;
  MappingState state;
  std::vector<TechMappedEdgeKind> edgeKinds;
};

bool isBetterLaneResult(const LaneAttempt &lhs, const LaneAttempt &rhs) {
  if (lhs.routedEdges != rhs.routedEdges)
    return lhs.routedEdges > rhs.routedEdges;
  if (lhs.routingSucceeded != rhs.routingSucceeded)
    return lhs.routingSucceeded;
  if (lhs.budgetExceeded != rhs.budgetExceeded)
    return !lhs.budgetExceeded;
  if (std::abs(lhs.throughputCost - rhs.throughputCost) > 1e-9)
    return lhs.throughputCost < rhs.throughputCost;
  if (std::abs(lhs.estimatedClockPeriod - rhs.estimatedClockPeriod) > 1e-9)
    return lhs.estimatedClockPeriod < rhs.estimatedClockPeriod;
  if (lhs.totalPathLen != rhs.totalPathLen)
    return lhs.totalPathLen < rhs.totalPathLen;
  if (std::abs(lhs.placementCost - rhs.placementCost) > 1e-9)
    return lhs.placementCost < rhs.placementCost;
  return lhs.laneIndex < rhs.laneIndex;
}

bool isBetterPlacementSeed(const LaneAttempt &lhs, const LaneAttempt &rhs) {
  if (lhs.placementSucceeded != rhs.placementSucceeded)
    return lhs.placementSucceeded;
  if (lhs.budgetExceeded != rhs.budgetExceeded)
    return !lhs.budgetExceeded;
  if (std::abs(lhs.throughputCost - rhs.throughputCost) > 1e-9)
    return lhs.throughputCost < rhs.throughputCost;
  if (std::abs(lhs.estimatedClockPeriod - rhs.estimatedClockPeriod) > 1e-9)
    return lhs.estimatedClockPeriod < rhs.estimatedClockPeriod;
  if (std::abs(lhs.placementCost - rhs.placementCost) > 1e-9)
    return lhs.placementCost < rhs.placementCost;
  return lhs.laneIndex < rhs.laneIndex;
}

bool isBypassableFifoNode(const Node *hwNode) {
  if (!hwNode || getNodeAttrStr(hwNode, "op_kind") != "fifo")
    return false;
  for (const auto &attr : hwNode->attributes) {
    if (attr.getName() == "bypassable") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        return boolAttr.getValue();
    }
  }
  return false;
}

bool getDefaultFifoBypassed(const Node *hwNode) {
  if (!hwNode)
    return false;
  for (const auto &attr : hwNode->attributes) {
    if (attr.getName() == "bypassed") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        return boolAttr.getValue();
    }
  }
  return false;
}

bool isEffectiveFifoBypassed(IdIndex hwNodeId, const MappingState &state,
                             const Graph &adg) {
  const Node *hwNode = adg.getNode(hwNodeId);
  if (!isBypassableFifoNode(hwNode))
    return false;
  if (hwNodeId < state.hwNodeFifoBypassedOverride.size()) {
    int8_t overrideValue = state.hwNodeFifoBypassedOverride[hwNodeId];
    if (overrideValue == 0)
      return false;
    if (overrideValue > 0)
      return true;
  }
  return getDefaultFifoBypassed(hwNode);
}

bool pathTouchesHwNode(llvm::ArrayRef<IdIndex> path, IdIndex hwNodeId,
                       const Graph &adg) {
  for (IdIndex portId : path) {
    const Port *port = adg.getPort(portId);
    if (port && port->parentNode == hwNodeId)
      return true;
  }
  return false;
}

bool isCarryNextEdge(IdIndex edgeId, const Graph &dfg) {
  const Edge *edge = dfg.getEdge(edgeId);
  if (!edge)
    return false;
  const Port *dstPort = dfg.getPort(edge->dstPort);
  if (!dstPort || dstPort->parentNode == INVALID_ID)
    return false;
  const Node *dstNode = dfg.getNode(dstPort->parentNode);
  if (!dstNode || dstNode->inputPorts.size() < 3)
    return false;
  llvm::StringRef opName = getNodeAttrStr(dstNode, "op_name");
  bool isCarryNode = opName == "dataflow.carry";
  if (!isCarryNode && opName == "techmap_group") {
    llvm::StringRef familySignature =
        getNodeAttrStr(dstNode, "techmap_family_signature");
    isCarryNode = familySignature.contains("dataflow.carry:");
  }
  return isCarryNode && edge->dstPort == dstNode->inputPorts[2];
}

bool sameConfigFields(llvm::ArrayRef<FUConfigField> lhs,
                      llvm::ArrayRef<FUConfigField> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t idx = 0; idx < lhs.size(); ++idx) {
    if (lhs[idx].kind != rhs[idx].kind ||
        lhs[idx].opIndex != rhs[idx].opIndex ||
        lhs[idx].templateOpIndex != rhs[idx].templateOpIndex ||
        lhs[idx].opName != rhs[idx].opName ||
        lhs[idx].bitWidth != rhs[idx].bitWidth ||
        lhs[idx].value != rhs[idx].value || lhs[idx].sel != rhs[idx].sel ||
        lhs[idx].discard != rhs[idx].discard ||
        lhs[idx].disconnect != rhs[idx].disconnect) {
      return false;
    }
  }
  return true;
}

bool isBetterTimingSummary(const MapperTimingSummary &lhs,
                           const MapperTimingSummary &rhs,
                           const MapperBufferizationOptions &opts) {
  if (lhs.estimatedThroughputCost + opts.minThroughputImprovement <
      rhs.estimatedThroughputCost) {
    return true;
  }
  if (rhs.estimatedThroughputCost + opts.minThroughputImprovement <
      lhs.estimatedThroughputCost) {
    return false;
  }
  if (lhs.estimatedClockPeriod + opts.clockTieBreakImprovement <
      rhs.estimatedClockPeriod) {
    return true;
  }
  if (rhs.estimatedClockPeriod + opts.clockTieBreakImprovement <
      lhs.estimatedClockPeriod) {
    return false;
  }
  if (lhs.estimatedInitiationInterval != rhs.estimatedInitiationInterval)
    return lhs.estimatedInitiationInterval < rhs.estimatedInitiationInterval;
  if (lhs.forcedBufferedFifoCount != rhs.forcedBufferedFifoCount)
    return lhs.forcedBufferedFifoCount < rhs.forcedBufferedFifoCount;
  return false;
}

std::vector<IdIndex> collectCriticalBoundaryEdges(const Graph &dfg) {
  std::vector<IdIndex> edges;
  edges.reserve(dfg.edges.size());
  auto getNodeForPort = [&](IdIndex portId) -> const Node * {
    const Port *port = dfg.getPort(portId);
    if (!port || port->parentNode == INVALID_ID)
      return nullptr;
    return dfg.getNode(port->parentNode);
  };
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Node *srcNode = getNodeForPort(edge->srcPort);
    const Node *dstNode = getNodeForPort(edge->dstPort);
    llvm::StringRef srcOp = srcNode ? getNodeAttrStr(srcNode, "op_name") : "";
    llvm::StringRef dstOp = dstNode ? getNodeAttrStr(dstNode, "op_name") : "";
    bool boundaryCritical =
        (srcNode && (srcNode->kind == Node::ModuleInputNode ||
                     srcNode->kind == Node::ModuleOutputNode)) ||
        (dstNode && (dstNode->kind == Node::ModuleInputNode ||
                     dstNode->kind == Node::ModuleOutputNode)) ||
        srcOp == "handshake.extmemory" || srcOp == "handshake.memory" ||
        dstOp == "handshake.extmemory" || dstOp == "handshake.memory";
    if (boundaryCritical)
      edges.push_back(edgeId);
  }
  return edges;
}

bool addUnitTechMapFeedback(const TechMapper::Unit &unit,
                            TechMapper::Feedback &feedback,
                            llvm::DenseSet<unsigned> &seenSplitCandidates,
                            llvm::DenseSet<unsigned> &seenBannedCandidates) {
  if (unit.selectedCandidateId == std::numeric_limits<unsigned>::max())
    return false;
  if (unit.swNodes.size() > 1) {
    if (seenSplitCandidates.insert(unit.selectedCandidateId).second)
      feedback.splitCandidateIds.push_back(unit.selectedCandidateId);
    return true;
  }
  if (unit.candidates.size() > 1) {
    if (seenBannedCandidates.insert(unit.selectedCandidateId).second)
      feedback.bannedCandidateIds.push_back(unit.selectedCandidateId);
    return true;
  }
  return false;
}

bool buildForcedTemporalConflictFeedback(
    const TechMapper::Plan &plan,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    const Graph &dfg, const Graph &adg,
    TechMapper::Feedback &feedback) {
  llvm::DenseSet<unsigned> seenSplitCandidates;
  llvm::DenseSet<unsigned> seenBannedCandidates;
  llvm::DenseMap<IdIndex, llvm::SmallVector<const TechMapper::Unit *, 4>>
      forcedByHwNode;

  for (const auto &unit : TechMapper::allUnits(plan)) {
    auto forcedHwNodeId = TechMapper::findForcedTemporalHwNodeId(unit, adg);
    if (!forcedHwNodeId)
      continue;
    forcedByHwNode[*forcedHwNodeId].push_back(&unit);
  }

  for (const auto &it : forcedByHwNode) {
    const auto &units = it.second;
    if (units.size() < 2)
      continue;
    for (size_t lhsIdx = 0; lhsIdx < units.size(); ++lhsIdx) {
      const auto *lhsConfigInfo =
          TechMapper::findSelectedUnitConfigClass(plan, *units[lhsIdx]);
      const auto *lhsCandidate =
          TechMapper::findPreferredUnitCandidate(*units[lhsIdx]);
      for (size_t rhsIdx = lhsIdx + 1; rhsIdx < units.size(); ++rhsIdx) {
        const auto *rhsConfigInfo =
            TechMapper::findSelectedUnitConfigClass(plan, *units[rhsIdx]);
        const auto *rhsCandidate =
            TechMapper::findPreferredUnitCandidate(*units[rhsIdx]);
        unsigned lhsConfigClass =
            lhsConfigInfo ? lhsConfigInfo->id
                          : std::numeric_limits<unsigned>::max();
        unsigned rhsConfigClass =
            rhsConfigInfo ? rhsConfigInfo->id
                          : std::numeric_limits<unsigned>::max();
        bool compatible =
            (lhsConfigClass != std::numeric_limits<unsigned>::max() ||
             rhsConfigClass != std::numeric_limits<unsigned>::max()) &&
            TechMapper::areConfigClassesCompatible(plan, lhsConfigClass,
                                                   rhsConfigClass);
        bool sameConfig =
            lhsCandidate && rhsCandidate &&
            sameConfigFields(lhsCandidate->configFields,
                             rhsCandidate->configFields);
        if (compatible || sameConfig)
          continue;
        bool changed = false;
        changed |= addUnitTechMapFeedback(*units[lhsIdx], feedback,
                                          seenSplitCandidates,
                                          seenBannedCandidates);
        changed |= addUnitTechMapFeedback(*units[rhsIdx], feedback,
                                          seenSplitCandidates,
                                          seenBannedCandidates);
        if (changed)
          return true;
      }
    }
  }

  struct ForcedTemporalCandidateInfo {
    IdIndex contractedNodeId = INVALID_ID;
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
    info.contractedNodeId = swNodeId;
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
    info.configClassIds.assign(configClassIds.begin(), configClassIds.end());
    std::sort(info.configClassIds.begin(), info.configClassIds.end());
    info.configClassIds.erase(
        std::unique(info.configClassIds.begin(), info.configClassIds.end()),
        info.configClassIds.end());
    forcedCandidatesByHwNode[forcedHwNodeId].push_back(std::move(info));
  }

  for (const auto &it : forcedCandidatesByHwNode) {
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
        bool changed = false;
        if (const auto *lhsUnit = TechMapper::findUnitForContractedNode(
                plan, infos[lhsIdx].contractedNodeId)) {
          changed |= addUnitTechMapFeedback(*lhsUnit, feedback,
                                            seenSplitCandidates,
                                            seenBannedCandidates);
        }
        if (const auto *rhsUnit = TechMapper::findUnitForContractedNode(
                plan, infos[rhsIdx].contractedNodeId)) {
          changed |= addUnitTechMapFeedback(*rhsUnit, feedback,
                                            seenSplitCandidates,
                                            seenBannedCandidates);
        }
        if (changed)
          return true;
      }
    }
  }

  return false;
}

bool buildRoutingFailureFeedback(const TechMapper::Plan &plan,
                                 const MappingState &state, const Graph &dfg,
                                 llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                                 const MapperTechFeedbackOptions &opts,
                                 TechMapper::Feedback &feedback) {
  auto failedEdges = collectUnroutedEdges(state, dfg, edgeKinds);
  if (failedEdges.empty())
    return false;

  llvm::DenseMap<IdIndex, unsigned> failedEdgeIncidence;
  for (IdIndex edgeId : failedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    auto accumulatePortNode = [&](IdIndex portId) {
      const Port *port = dfg.getPort(portId);
      if (!port || port->parentNode == INVALID_ID)
        return;
      ++failedEdgeIncidence[port->parentNode];
    };
    accumulatePortNode(edge->srcPort);
    accumulatePortNode(edge->dstPort);
  }

  std::vector<std::pair<IdIndex, unsigned>> rankedNodes(
      failedEdgeIncidence.begin(), failedEdgeIncidence.end());
  llvm::stable_sort(rankedNodes,
                    [](const auto &lhs, const auto &rhs) {
                      if (lhs.second != rhs.second)
                        return lhs.second > rhs.second;
                      return lhs.first < rhs.first;
                    });

  llvm::DenseSet<unsigned> seenSplitCandidates;
  llvm::DenseSet<unsigned> seenBannedCandidates;
  unsigned targets = 0;
  for (const auto &entry : rankedNodes) {
    if (targets >= opts.maxTargetsPerRetry)
      break;
    const auto *unit =
        TechMapper::findUnitForContractedNode(plan, entry.first);
    if (!unit)
      continue;
    if (!addUnitTechMapFeedback(*unit, feedback, seenSplitCandidates,
                                seenBannedCandidates)) {
      continue;
    }
    ++targets;
  }
  return !feedback.splitCandidateIds.empty() ||
         !feedback.bannedCandidateIds.empty();
}

} // namespace

void Mapper::resetRunControls(const Options &opts) {
  activeRunStartTime_ = std::chrono::steady_clock::now();
  activeRunDeadline_ = activeRunStartTime_ +
                       std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                           std::chrono::duration<double>(opts.budgetSeconds));
  activeBudgetEnabled_ = opts.budgetSeconds > 0.0;
  activeBudgetExceeded_ = false;
  activeBudgetExceededStage_.clear();
  snapshotSequence_ = 0;
  snapshotTickCount_ = 0;
  nextSnapshotAtSeconds_ =
      opts.snapshotIntervalSeconds > 0.0 ? opts.snapshotIntervalSeconds : -1.0;
}

double Mapper::remainingBudgetSeconds() const {
  if (!activeBudgetEnabled_)
    return std::numeric_limits<double>::infinity();
  auto remaining = activeRunDeadline_ - std::chrono::steady_clock::now();
  double seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(remaining).count() /
      1000.0;
  return std::max(0.0, seconds);
}

double Mapper::clampToRemainingBudget(double requestedSeconds) const {
  if (requestedSeconds <= 0.0)
    return requestedSeconds;
  return std::min(requestedSeconds, remainingBudgetSeconds());
}

double Mapper::clampDeadlineMsToRemainingBudget(double requestedMs) const {
  if (requestedMs <= 0.0)
    return requestedMs;
  double remainingMs = remainingBudgetSeconds() * 1000.0;
  if (!std::isfinite(remainingMs))
    return requestedMs;
  return std::min(requestedMs, remainingMs);
}

bool Mapper::shouldStopForBudget(llvm::StringRef stage) {
  if (!activeBudgetEnabled_ || activeBudgetExceeded_)
    return activeBudgetExceeded_;
  if (std::chrono::steady_clock::now() <= activeRunDeadline_)
    return false;
  activeBudgetExceeded_ = true;
  activeBudgetExceededStage_ = stage.str();
  llvm::errs() << "Mapper: budget exhausted";
  if (!stage.empty())
    llvm::errs() << " during " << stage;
  llvm::errs() << "\n";
  return true;
}

bool Mapper::maybeEmitProgressSnapshot(
    const MappingState &state, llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
    llvm::StringRef trigger, const Options &opts) {
  if (!activeSnapshotEmitter_)
    return false;

  bool due = false;
  if (opts.snapshotIntervalRounds > 0) {
    ++snapshotTickCount_;
    due = (snapshotTickCount_ % static_cast<unsigned>(opts.snapshotIntervalRounds) ==
           0);
  } else if (opts.snapshotIntervalSeconds > 0.0) {
    auto elapsed = std::chrono::steady_clock::now() - activeRunStartTime_;
    double elapsedSeconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() /
        1000.0;
    if (nextSnapshotAtSeconds_ > 0.0 &&
        elapsedSeconds + 1e-9 >= nextSnapshotAtSeconds_) {
      due = true;
      do {
        nextSnapshotAtSeconds_ += opts.snapshotIntervalSeconds;
      } while (nextSnapshotAtSeconds_ > 0.0 &&
               elapsedSeconds + 1e-9 >= nextSnapshotAtSeconds_);
    }
  }

  if (!due)
    return false;

  ++snapshotSequence_;
  activeSnapshotEmitter_(state, edgeKinds, trigger, snapshotSequence_);
  return true;
}

// ---------------------------------------------------------------------------
// Mapper::bindSentinels
// ---------------------------------------------------------------------------

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

  llvm::outs() << "  DFG sentinels: " << dfgInputSentinels.size() << " inputs, "
               << dfgOutputSentinels.size() << " outputs\n";
  llvm::outs() << "  ADG sentinels: " << adgInputSentinels.size() << " inputs, "
               << adgOutputSentinels.size() << " outputs\n";

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
      llvm::errs() << "Mapper: failed to bind DFG input sentinel " << dfgSid
                   << "\n";
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
      llvm::errs() << "Mapper: failed to bind DFG output sentinel " << dfgSid
                   << "\n";
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// Mapper::bindMemrefSentinels
// ---------------------------------------------------------------------------

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

        if (eid < state.swEdgeToHwPaths.size() &&
            state.swEdgeToHwPaths[eid].size() == syntheticPath.size() &&
            std::equal(state.swEdgeToHwPaths[eid].begin(),
                       state.swEdgeToHwPaths[eid].end(),
                       syntheticPath.begin())) {
          continue;
        }

        auto result = state.mapEdge(eid, syntheticPath, dfg, adg);
        if (result == ActionResult::Success) {
          llvm::outs() << "    Pre-routed memref edge " << eid << " (sentinel "
                       << sid << " -> extmem " << dstNodeId
                       << ") as direct binding\n";
        }
      }
    }
  }

  return true;
}

bool Mapper::rebindScalarInputSentinels(MappingState &state, const Graph &dfg,
                                        const Graph &adg,
                                        const ADGFlattener &flattener) {
  std::vector<IdIndex> dfgScalarSentinels;
  std::vector<IdIndex> adgInputSentinels;

  for (IdIndex i = 0; i < static_cast<IdIndex>(dfg.nodes.size()); ++i) {
    const Node *node = dfg.getNode(i);
    if (!node || node->kind != Node::ModuleInputNode ||
        node->outputPorts.empty())
      continue;
    const Port *swPort = dfg.getPort(node->outputPorts[0]);
    if (!swPort || isMemrefType(swPort->type))
      continue;
    dfgScalarSentinels.push_back(i);
  }

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (node && node->kind == Node::ModuleInputNode)
      adgInputSentinels.push_back(i);
  }

  if (dfgScalarSentinels.empty() || adgInputSentinels.empty())
    return true;

  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> candidateInputs;
  llvm::DenseMap<uint64_t, double> assignmentCostCache;
  llvm::DenseMap<IdIndex, double> emptyRoutingHistory;
  const TopologyModel *topologyModel = getActiveTopologyModel();
  auto estimateAssignmentCost = [&](IdIndex swSentinel, IdIndex adgSentinel) {
    uint64_t cacheKey = (static_cast<uint64_t>(static_cast<uint32_t>(swSentinel))
                         << 32) |
                        static_cast<uint32_t>(adgSentinel);
    if (auto it = assignmentCostCache.find(cacheKey);
        it != assignmentCostCache.end()) {
      return it->second;
    }

    const Node *swNode = dfg.getNode(swSentinel);
    const Node *adgNode = adg.getNode(adgSentinel);
    if (!swNode || swNode->outputPorts.empty())
      return std::numeric_limits<double>::infinity();
    if (!adgNode || adgNode->outputPorts.empty())
      return std::numeric_limits<double>::infinity();
    IdIndex srcHwPort = adgNode->outputPorts[0];

    double cost = 0.0;
    IdIndex swOutId = swNode->outputPorts[0];
    const Port *swOut = dfg.getPort(swOutId);
    if (!swOut) {
      assignmentCostCache[cacheKey] = cost;
      return cost;
    }
    for (IdIndex edgeId : swOut->connectedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge || edge->srcPort != swOutId)
        continue;
      const Port *dstPort = dfg.getPort(edge->dstPort);
      if (!dstPort || dstPort->parentNode == INVALID_ID ||
          dstPort->parentNode >= state.swNodeToHwNode.size())
        continue;
      IdIndex dstHw = state.swNodeToHwNode[dstPort->parentNode];
      if (dstHw == INVALID_ID)
        continue;
      double weight = classifyEdgePlacementWeight(dfg, edgeId);
      IdIndex dstHwPort =
          edge->dstPort < state.swPortToHwPort.size()
              ? state.swPortToHwPort[edge->dstPort]
              : INVALID_ID;
      if (dstHwPort != INVALID_ID) {
        auto path = findPath(srcHwPort, dstHwPort, edgeId, state, dfg, adg,
                             emptyRoutingHistory);
        if (path.empty()) {
          cost = std::numeric_limits<double>::infinity();
          break;
        }
        cost += weight * static_cast<double>(path.size());
        continue;
      }
      if (topologyModel) {
        cost += weight * static_cast<double>(
                             topologyModel->placementDistance(adgSentinel,
                                                              dstHw));
      } else {
        cost += weight * static_cast<double>(
                             placementDistance(adgSentinel, dstHw, flattener));
      }
    }
    assignmentCostCache[cacheKey] = cost;
    return cost;
  };

  for (IdIndex swSentinel : dfgScalarSentinels) {
    const Node *swNode = dfg.getNode(swSentinel);
    const Port *swPort = swNode && !swNode->outputPorts.empty()
                             ? dfg.getPort(swNode->outputPorts[0])
                             : nullptr;
    if (!swPort)
      continue;
    auto &choices = candidateInputs[swSentinel];
    for (IdIndex adgSentinel : adgInputSentinels) {
      const Node *adgNode = adg.getNode(adgSentinel);
      const Port *hwPort = adgNode && !adgNode->outputPorts.empty()
                               ? adg.getPort(adgNode->outputPorts[0])
                               : nullptr;
      if (!hwPort ||
          !canMapSoftwareTypeToHardware(swPort->type, hwPort->type)) {
        continue;
      }
      choices.push_back(adgSentinel);
    }
    if (choices.empty())
      return false;
    llvm::stable_sort(choices, [&](IdIndex lhs, IdIndex rhs) {
      double lhsCost = estimateAssignmentCost(swSentinel, lhs);
      double rhsCost = estimateAssignmentCost(swSentinel, rhs);
      if (std::abs(lhsCost - rhsCost) > 1e-9)
        return lhsCost < rhsCost;
      return lhs < rhs;
    });
  }

  llvm::stable_sort(dfgScalarSentinels, [&](IdIndex lhs, IdIndex rhs) {
    size_t lhsCount = candidateInputs.lookup(lhs).size();
    size_t rhsCount = candidateInputs.lookup(rhs).size();
    if (lhsCount != rhsCount)
      return lhsCount < rhsCount;
    return lhs < rhs;
  });

  double bestCost = std::numeric_limits<double>::infinity();
  llvm::DenseMap<IdIndex, IdIndex> bestAssignment;
  llvm::DenseMap<IdIndex, IdIndex> currentAssignment;
  llvm::DenseSet<IdIndex> usedInputs;

  std::function<void(unsigned, double)> searchAssignments;
  searchAssignments = [&](unsigned depth, double currentCost) {
    if (currentCost >= bestCost)
      return;
    if (depth >= dfgScalarSentinels.size()) {
      bestCost = currentCost;
      bestAssignment = currentAssignment;
      return;
    }
    IdIndex swSentinel = dfgScalarSentinels[depth];
    for (IdIndex adgSentinel : candidateInputs.lookup(swSentinel)) {
      if (!usedInputs.insert(adgSentinel).second)
        continue;
      currentAssignment[swSentinel] = adgSentinel;
      searchAssignments(depth + 1, currentCost + estimateAssignmentCost(
                                                     swSentinel, adgSentinel));
      currentAssignment.erase(swSentinel);
      usedInputs.erase(adgSentinel);
    }
  };
  searchAssignments(0, 0.0);
  if (bestAssignment.empty())
    return false;

  for (IdIndex swSentinel : dfgScalarSentinels)
    state.unmapNode(swSentinel, dfg, adg);

  for (IdIndex swSentinel : dfgScalarSentinels) {
    IdIndex adgSentinel = bestAssignment.lookup(swSentinel);
    if (adgSentinel == INVALID_ID ||
        state.mapNode(swSentinel, adgSentinel, dfg, adg) !=
            ActionResult::Success) {
      return false;
    }
  }
  return true;
}

bool Mapper::runInterleavedPlaceRoute(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    std::vector<TechMappedEdgeKind> &edgeKinds, const Options &opts) {
  const unsigned rounds = std::max(1u, opts.interleavedRounds);
  auto currentPlacementCheckpoint = state.save();
  auto bestCheckpoint = state.save();
  unsigned bestRouted = 0;
  auto computeUnroutedPenalty = [&](const MappingState &routingState) {
    double penalty = 0.0;
    for (IdIndex edgeId : collectUnroutedEdges(routingState, dfg, edgeKinds))
      penalty += classifyEdgePlacementWeight(dfg, edgeId);
    return penalty;
  };
  auto computePriorityMetrics =
      [&](const MappingState &routingState,
          llvm::ArrayRef<IdIndex> priorityEdges) -> std::pair<unsigned, double> {
    unsigned routed = 0;
    double penalty = 0.0;
    for (IdIndex edgeId : priorityEdges) {
      double weight = classifyEdgePlacementWeight(dfg, edgeId);
      if (edgeId < routingState.swEdgeToHwPaths.size() &&
          !routingState.swEdgeToHwPaths[edgeId].empty()) {
        ++routed;
      } else {
        penalty += weight;
      }
    }
    return {routed, penalty};
  };
  double bestUnroutedPenalty = computeUnroutedPenalty(state);
  size_t bestPathLen = std::numeric_limits<size_t>::max();
  double bestPlacementCost = computeTotalCost(state, dfg, adg, flattener);
  MapperTimingSummary bestTimingSummary =
      analyzeMapperTiming(state, dfg, adg, edgeKinds, opts.timing);
  double bestThroughputCost = bestTimingSummary.estimatedThroughputCost;
  double bestClockPeriod = bestTimingSummary.estimatedClockPeriod;
  bool bestAllRouted = false;
  unsigned bestPriorityRouted = 0;
  double bestPriorityPenalty = 0.0;

  auto updateBest = [&](bool allRouted,
                        llvm::ArrayRef<IdIndex> priorityEdges =
                            llvm::ArrayRef<IdIndex>()) -> bool {
    const bool usePriority = !priorityEdges.empty();
    unsigned routed = countRoutedEdges(state, dfg, edgeKinds);
    double unroutedPenalty = computeUnroutedPenalty(state);
    size_t totalPathLen = computeTotalMappedPathLen(state);
    double placementCost = computeTotalCost(state, dfg, adg, flattener);
    MapperTimingSummary timingSummary =
        analyzeMapperTiming(state, dfg, adg, edgeKinds, opts.timing);
    double throughputCost = timingSummary.estimatedThroughputCost;
    double clockPeriod = timingSummary.estimatedClockPeriod;
    auto [priorityRouted, priorityPenalty] =
        computePriorityMetrics(state, priorityEdges);
    bool improved = allRouted || routed > bestRouted ||
                    (usePriority && routed == bestRouted &&
                     priorityRouted > bestPriorityRouted) ||
                    (usePriority && routed == bestRouted &&
                     priorityRouted == bestPriorityRouted &&
                     priorityPenalty + 1e-9 < bestPriorityPenalty) ||
                    (routed == bestRouted &&
                     (!usePriority ||
                     (priorityRouted == bestPriorityRouted &&
                      std::abs(priorityPenalty - bestPriorityPenalty) <=
                          1e-9)) &&
                     unroutedPenalty + 1e-9 < bestUnroutedPenalty) ||
                    (routed == bestRouted &&
                     (!usePriority ||
                      (priorityRouted == bestPriorityRouted &&
                       std::abs(priorityPenalty - bestPriorityPenalty) <=
                           1e-9)) &&
                     std::abs(unroutedPenalty - bestUnroutedPenalty) <= 1e-9 &&
                     throughputCost + 1e-9 < bestThroughputCost) ||
                    (routed == bestRouted &&
                     (!usePriority ||
                      (priorityRouted == bestPriorityRouted &&
                       std::abs(priorityPenalty - bestPriorityPenalty) <=
                           1e-9)) &&
                     std::abs(unroutedPenalty - bestUnroutedPenalty) <= 1e-9 &&
                     std::abs(throughputCost - bestThroughputCost) <= 1e-9 &&
                     clockPeriod + 1e-9 < bestClockPeriod) ||
                    (routed == bestRouted &&
                     (!usePriority ||
                      (priorityRouted == bestPriorityRouted &&
                       std::abs(priorityPenalty - bestPriorityPenalty) <=
                           1e-9)) &&
                     std::abs(unroutedPenalty - bestUnroutedPenalty) <= 1e-9 &&
                     std::abs(throughputCost - bestThroughputCost) <= 1e-9 &&
                     std::abs(clockPeriod - bestClockPeriod) <= 1e-9 &&
                     totalPathLen < bestPathLen) ||
                    (routed == bestRouted && totalPathLen == bestPathLen &&
                     (!usePriority ||
                      (priorityRouted == bestPriorityRouted &&
                       std::abs(priorityPenalty - bestPriorityPenalty) <=
                           1e-9)) &&
                     std::abs(throughputCost - bestThroughputCost) <= 1e-9 &&
                     std::abs(clockPeriod - bestClockPeriod) <= 1e-9 &&
                     placementCost + 1e-9 < bestPlacementCost);
    if (!improved)
      return false;
    bestCheckpoint = state.save();
    bestRouted = routed;
    bestUnroutedPenalty = unroutedPenalty;
    bestPathLen = totalPathLen;
    bestPlacementCost = placementCost;
    bestTimingSummary = std::move(timingSummary);
    bestThroughputCost = bestTimingSummary.estimatedThroughputCost;
    bestClockPeriod = bestTimingSummary.estimatedClockPeriod;
    bestPriorityRouted = usePriority ? priorityRouted : 0u;
    bestPriorityPenalty = usePriority ? priorityPenalty : 0.0;
    bestAllRouted = allRouted;
    return true;
  };
  auto emitBestSnapshot = [&](llvm::StringRef trigger) {
    auto checkpoint = state.save();
    state.restore(bestCheckpoint);
    maybeEmitProgressSnapshot(state, edgeKinds, trigger, opts);
    state.restore(checkpoint);
  };

  for (unsigned round = 0; round < rounds; ++round) {
    if (shouldStopForBudget("interleaved place-route"))
      break;
    state.restore(currentPlacementCheckpoint);
    rebindScalarInputSentinels(state, dfg, adg, flattener);
    classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);

    CongestionEstimator congEstimator;
    if (activeCongestionPlacementWeight > 0.0 && activeFlattener) {
      congEstimator.estimate(state, dfg, adg, *activeFlattener);
      activeCongestionEstimator = &congEstimator;
    }

    bool allRouted = (opts.negotiatedRoutingPasses > 0)
                         ? runNegotiatedRouting(state, dfg, adg, edgeKinds, opts)
                         : runRouting(state, dfg, adg, edgeKinds, opts);
    activeCongestionEstimator = nullptr;
    updateBest(allRouted);
    emitBestSnapshot("interleaved-round");

    auto edgeStats = computeRoutingEdgeStats(state, dfg, edgeKinds);
    auto failedEdges = collectUnroutedEdges(state, dfg, edgeKinds);
    llvm::outs() << "  Interleaved round " << (round + 1) << "/" << rounds
                 << ": overall " << edgeStats.routedOverallEdges << "/"
                 << edgeStats.overallEdges << ", router "
                 << edgeStats.routedRouterEdges << "/"
                 << edgeStats.routerEdges << ", prebound "
                 << edgeStats.directBindingEdges << ", failed router "
                 << failedEdges.size() << "\n";
    if (allRouted) {
      state.restore(bestCheckpoint);
      return true;
    }
    if (shouldStopForBudget("interleaved place-route"))
      break;
    if (failedEdges.empty())
      break;

    std::optional<CongestionState> repairCongestion;
    const CongestionState *repairCongestionPtr = nullptr;
    if (opts.negotiatedRoutingPasses > 0) {
      repairCongestion.emplace();
      repairCongestion->init(adg);
      repairCongestion->historyIncrement = opts.congestionHistoryFactor;
      repairCongestion->historyScale = opts.congestionHistoryScale;
      repairCongestion->presentFactor = opts.congestionPresentFactor;
      for (const auto &path : state.swEdgeToHwPaths) {
        if (!path.empty() && !(path.size() == 2 && path[0] == path[1]))
          repairCongestion->commitRoute(path, adg);
      }
      repairCongestion->updateHistory();
      repairCongestionPtr = &*repairCongestion;
    }

    state.restore(currentPlacementCheckpoint);
    bool repaired = false;
    if (opts.localRepair.enabled) {
      repaired =
          runLocalRepair(state, currentPlacementCheckpoint, failedEdges, dfg,
                         adg, flattener, candidates, edgeKinds, opts,
                         repairCongestionPtr);
    }
    bool repairImproved = updateBest(repaired, failedEdges);
    if (repairImproved)
      emitBestSnapshot("local-repair");
    if (repaired) {
      state.restore(bestCheckpoint);
      return true;
    }
    if (shouldStopForBudget("local repair"))
      break;

    if (repairImproved) {
      state.clearRoutes(dfg, adg);
      currentPlacementCheckpoint = state.save();
      continue;
    }

    if (round + 1 >= rounds)
      break;

    state.restore(bestCheckpoint);
    Options restartOpts = opts;
    restartOpts.seed = opts.seed +
                       static_cast<int>((round + 1) *
                                        opts.lane.restartSeedStride);
    if (opts.placementMoveRadius != 0)
      restartOpts.placementMoveRadius =
          opts.placementMoveRadius +
          (round + 1) * opts.lane.restartMoveRadiusStep;
    restartOpts.selectiveRipupPasses =
        opts.selectiveRipupPasses +
        std::min<unsigned>(opts.lane.restartRipupBonusCap, round + 1);
    runRefinement(state, dfg, adg, flattener, candidates, restartOpts,
                  &edgeKinds);
    state.clearRoutes(dfg, adg);
    rebindScalarInputSentinels(state, dfg, adg, flattener);
    bindMemrefSentinels(state, dfg, adg);
    classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
    currentPlacementCheckpoint = state.save();
    if (opts.verbose) {
      llvm::outs() << "  Interleaved round " << (round + 1)
                   << ": restarting from placement seed " << restartOpts.seed
                   << "\n";
    }
  }

  state.restore(bestCheckpoint);
  return bestAllRouted;
}

MapperTimingSummary Mapper::runPostRouteFifoBufferization(
    MappingState &state, const Graph &dfg, const Graph &adg,
    llvm::ArrayRef<TechMappedEdgeKind> edgeKinds, const Options &opts) {
  MapperTimingSummary currentTiming =
      analyzeMapperTiming(state, dfg, adg, edgeKinds, opts.timing);
  if (!opts.bufferization.enabled)
    return currentTiming;

  llvm::DenseSet<IdIndex> recurrenceEdges;
  for (const auto &cycle : currentTiming.recurrenceCycles) {
    for (IdIndex edgeId : cycle.swEdges)
      recurrenceEdges.insert(edgeId);
  }
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (isCarryNextEdge(edgeId, dfg))
      recurrenceEdges.insert(edgeId);
  }
  llvm::DenseSet<IdIndex> criticalEdges(currentTiming.criticalPathEdges.begin(),
                                        currentTiming.criticalPathEdges.end());
  llvm::DenseSet<IdIndex> blockedRecurrenceFifos;
  for (IdIndex edgeId : recurrenceEdges) {
    if (edgeId >= state.swEdgeToHwPaths.size())
      continue;
    for (IdIndex portId : state.swEdgeToHwPaths[edgeId]) {
      const Port *port = adg.getPort(portId);
      if (!port || port->parentNode == INVALID_ID)
        continue;
      IdIndex hwNodeId = port->parentNode;
      const Node *hwNode = adg.getNode(hwNodeId);
      if (!isBypassableFifoNode(hwNode) ||
          !isEffectiveFifoBypassed(hwNodeId, state, adg)) {
        continue;
      }
      blockedRecurrenceFifos.insert(hwNodeId);
    }
  }
  if (opts.verbose && !blockedRecurrenceFifos.empty()) {
    llvm::outs() << "Mapper: excluding " << blockedRecurrenceFifos.size()
                 << " recurrence-sensitive FIFO candidates from bufferization\n";
  }

  for (unsigned iter = 0; iter < opts.bufferization.maxIterations; ++iter) {
    llvm::DenseSet<IdIndex> seenCandidates;
    std::vector<IdIndex> fifoCandidates;
    std::vector<IdIndex> criticalFifoCandidates;
    llvm::DenseMap<IdIndex, bool> candidateTouchesRecurrence;
    llvm::DenseMap<IdIndex, bool> candidateTouchesCriticalPath;
    for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
         ++edgeId) {
      const auto &path = state.swEdgeToHwPaths[edgeId];
      if (path.empty())
        continue;
      for (IdIndex portId : path) {
        const Port *port = adg.getPort(portId);
        if (!port || port->parentNode == INVALID_ID)
          continue;
        IdIndex hwNodeId = port->parentNode;
        const Node *hwNode = adg.getNode(hwNodeId);
        if (!isBypassableFifoNode(hwNode) ||
            !isEffectiveFifoBypassed(hwNodeId, state, adg)) {
          continue;
        }
        if (recurrenceEdges.count(edgeId) || isCarryNextEdge(edgeId, dfg))
          candidateTouchesRecurrence[hwNodeId] = true;
        if (criticalEdges.count(edgeId))
          candidateTouchesCriticalPath[hwNodeId] = true;
        if (seenCandidates.insert(hwNodeId).second) {
          fifoCandidates.push_back(hwNodeId);
          if (candidateTouchesCriticalPath.lookup(hwNodeId))
            criticalFifoCandidates.push_back(hwNodeId);
        }
      }
    }

    if (!criticalFifoCandidates.empty())
      fifoCandidates = criticalFifoCandidates;

    std::sort(fifoCandidates.begin(), fifoCandidates.end());
    IdIndex bestFifo = INVALID_ID;
    MapperTimingSummary bestTiming = currentTiming;

    for (IdIndex hwNodeId : fifoCandidates) {
      if (hwNodeId >= state.hwNodeFifoBypassedOverride.size())
        continue;
      if (blockedRecurrenceFifos.contains(hwNodeId))
        continue;
      int8_t oldOverride = state.hwNodeFifoBypassedOverride[hwNodeId];
      state.hwNodeFifoBypassedOverride[hwNodeId] = 0;
      MapperTimingSummary candidateTiming =
          analyzeMapperTiming(state, dfg, adg, edgeKinds, opts.timing);
      state.hwNodeFifoBypassedOverride[hwNodeId] = oldOverride;

      bool recurrenceSensitive = candidateTouchesRecurrence.lookup(hwNodeId);
      if (recurrenceSensitive)
        continue;
      if (!isBetterTimingSummary(candidateTiming, currentTiming,
                                 opts.bufferization)) {
        continue;
      }
      if (bestFifo == INVALID_ID ||
          isBetterTimingSummary(candidateTiming, bestTiming,
                               opts.bufferization)) {
        bestFifo = hwNodeId;
        bestTiming = std::move(candidateTiming);
      }
    }

    if (bestFifo == INVALID_ID)
      break;

    state.hwNodeFifoBypassedOverride[bestFifo] = 0;
    ++activeSearchSummary_.fifoBufferizationAcceptedToggles;
    currentTiming = std::move(bestTiming);
    if (opts.verbose) {
      llvm::outs() << "Mapper: accepted FIFO timing cut on hw node "
                   << bestFifo << ", throughput="
                   << currentTiming.estimatedThroughputCost
                   << ", clock=" << currentTiming.estimatedClockPeriod
                   << ", ii=" << currentTiming.estimatedInitiationInterval
                   << "\n";
    }
  }

  return currentTiming;
}

// ---------------------------------------------------------------------------
// Mapper::run
// ---------------------------------------------------------------------------

Mapper::Result Mapper::run(const Graph &dfg, const Graph &adg,
                           const ADGFlattener &flattener,
                           mlir::ModuleOp adgModule, const Options &opts) {
  Result result;
  if (!validateMapperOptions(opts, result.diagnostics)) {
    result.searchSummary = activeSearchSummary_;
    llvm::errs() << "Mapper: invalid options: " << result.diagnostics << "\n";
    return result;
  }
  resetRunControls(opts);
  activeSearchSummary_ = {};
  activeSnapshotEmitter_ = nullptr;
  TechMapper techMapper;
  TechMapper::Plan techPlan;
  if (!techMapper.buildPlan(dfg, adgModule, adg, techPlan)) {
    result.diagnostics = techPlan.diagnostics.empty()
                             ? "Tech-mapping failed"
                             : "Tech-mapping failed: " + techPlan.diagnostics;
    result.searchSummary = activeSearchSummary_;
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }
  return runWithTechMapPlan(dfg, adg, flattener, adgModule, opts, techMapper,
                            std::move(techPlan), 0);
}

Mapper::Result Mapper::runWithTechMapPlan(
    const Graph &dfg, const Graph &adg, const ADGFlattener &flattener,
    mlir::ModuleOp adgModule, const Options &opts, TechMapper &techMapper,
    TechMapper::Plan techPlan, unsigned techFeedbackAttempt) {
  Result result;
  result.techMapPlan = techPlan;
  result.techMapMetrics = techPlan.metrics;
  result.techMapDiagnostics = techPlan.diagnostics;
  if (opts.verbose && !techPlan.diagnostics.empty())
    llvm::outs() << "Mapper: techmap note: " << techPlan.diagnostics << "\n";

  MappingState contractedState;
  contractedState.init(techPlan.contractedDFG, adg, &flattener);
  result.edgeKinds = techPlan.originalEdgeKinds;
  const auto originalEdgeKinds = result.edgeKinds;
  std::vector<TechMappedEdgeKind> initialContractedEdgeKinds(
      techPlan.contractedDFG.edges.size(), TechMappedEdgeKind::Routed);

  // Copy connectivity from flattener.
  connectivity = flattener.getConnectivity();
  activeFlattener = &flattener;
  activeTopologyModel_ = std::make_shared<TopologyModel>(adg, flattener);
  mapper_detail::setActiveTopologyModel(activeTopologyModel_.get());
  mapper_detail::setActiveTimingOptions(&opts.timing);
  activeHeuristicWeight = opts.routingHeuristicWeight;
  activeCongestionPlacementWeight = opts.congestionPlacementWeight;
  activeMemorySharingPenalty = opts.memorySharingPenalty;
  activeUnroutedDiagnosticLimit = opts.lane.unroutedDiagnosticLimit;
  activeRelaxedRoutingOpts_ = opts.relaxedRouting;
  activeSnapshotEmitter_ = nullptr;
  if (snapshotCallback_ &&
      (opts.snapshotIntervalSeconds > 0.0 || opts.snapshotIntervalRounds > 0)) {
    activeSnapshotEmitter_ =
        [&](const MappingState &contractedSnapshot,
            llvm::ArrayRef<TechMappedEdgeKind> contractedSnapshotEdgeKinds,
            llvm::StringRef trigger, unsigned ordinal) {
          MappingState expandedState;
          llvm::SmallVector<FUConfigSelection, 4> expandedFuConfigs;
          if (!techMapper.expandPlanMapping(dfg, adg, techPlan,
                                            contractedSnapshot, expandedState,
                                            expandedFuConfigs)) {
            llvm::errs() << "Mapper: snapshot expansion failed for trigger "
                         << trigger << "\n";
            return;
          }
          std::vector<TechMappedEdgeKind> expandedEdgeKinds(
              originalEdgeKinds.begin(), originalEdgeKinds.end());
          for (IdIndex edgeId = 0;
               edgeId < static_cast<IdIndex>(dfg.edges.size()); ++edgeId) {
            if (edgeId >= expandedEdgeKinds.size() ||
                expandedEdgeKinds[edgeId] == TechMappedEdgeKind::IntraFU) {
              continue;
            }
            if (edgeId >= techPlan.originalEdgeToContractedEdge.size())
              continue;
            IdIndex contractedEdgeId =
                techPlan.originalEdgeToContractedEdge[edgeId];
            if (contractedEdgeId == INVALID_ID ||
                contractedEdgeId >= contractedSnapshotEdgeKinds.size()) {
              continue;
            }
            if (contractedSnapshotEdgeKinds[contractedEdgeId] ==
                TechMappedEdgeKind::TemporalReg) {
              expandedEdgeKinds[edgeId] = TechMappedEdgeKind::TemporalReg;
            }
          }
          snapshotCallback_(expandedState, expandedEdgeKinds, expandedFuConfigs,
                            trigger, ordinal);
        };
  }

  // Bind sentinels (DFG boundary nodes -> ADG boundary nodes).
  llvm::outs() << "Mapper: binding sentinels...\n";
  bindSentinels(contractedState, techPlan.contractedDFG, adg);

  llvm::outs() << "Mapper: building candidates...\n";
  auto candidates = buildCandidates(techPlan.contractedDFG, adg);
  for (IdIndex swNodeId = 0;
       swNodeId < static_cast<IdIndex>(techPlan.contractedDFG.nodes.size());
       ++swNodeId) {
    const auto *contractedCandidates =
        TechMapper::findContractedCandidates(techPlan, swNodeId);
    if (!contractedCandidates)
      continue;
    candidates[swNodeId] = *contractedCandidates;
  }

  if (detectForcedTemporalConfigConflict(techPlan, candidates,
                                         techPlan.contractedDFG, adg,
                                         result.diagnostics)) {
    if (opts.techFeedback.enabled &&
        techFeedbackAttempt < opts.techFeedback.maxRetries) {
      TechMapper::Feedback feedback;
      if (buildForcedTemporalConflictFeedback(
              techPlan, candidates, techPlan.contractedDFG, adg, feedback)) {
        ++activeSearchSummary_.techMapFeedbackAttempts;
        TechMapper::Plan feedbackPlan;
        if (techMapper.applyFeedback(dfg, adg, techPlan, feedback,
                                     feedbackPlan)) {
          ++activeSearchSummary_.techMapFeedbackAcceptedReconfigurations;
          if (opts.verbose) {
            llvm::outs() << "Mapper: retrying techmap after temporal config "
                            "conflict with "
                         << feedback.splitCandidateIds.size()
                         << " split requests and "
                         << feedback.bannedCandidateIds.size()
                         << " banned candidates\n";
          }
          return runWithTechMapPlan(dfg, adg, flattener, adgModule, opts,
                                    techMapper, std::move(feedbackPlan),
                                    techFeedbackAttempt + 1);
        }
      }
    }
    result.searchSummary = activeSearchSummary_;
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

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
      result.searchSummary = activeSearchSummary_;
      llvm::errs() << "Mapper: " << result.diagnostics << "\n";
      return result;
    }
    if (opts.verbose) {
      llvm::outs() << "  Node " << i << " (" << getNodeAttrStr(node, "op_name")
                   << "): " << candidates[i].size() << " candidates\n";
    }
  }

  const unsigned laneCount = selectLaneCount(opts, dfg);
  if (opts.verbose && laneCount > 1)
    llvm::outs() << "Mapper: launching " << laneCount << " parallel lanes\n";

  auto configureLaneOptions = [&](unsigned laneIndex, Mapper &laneMapper,
                                  bool resetBudgetWindow) {
    Options laneOpts = opts;
    laneOpts.seed =
        opts.seed + static_cast<int>(laneIndex * opts.lane.laneSeedStride);
    laneOpts.verbose = (laneCount == 1) ? opts.verbose : false;
    if (resetBudgetWindow && laneOpts.budgetSeconds > 0.0 &&
        laneOpts.lane.finalPolishReserveFraction > 0.0) {
      double laneBudgetSeconds =
          laneOpts.budgetSeconds *
          std::max(0.0, 1.0 - laneOpts.lane.finalPolishReserveFraction);
      laneOpts.budgetSeconds = std::max(1.0, laneBudgetSeconds);
      laneMapper.resetRunControls(laneOpts);
    }
    return laneOpts;
  };

  auto runLanePlacementSeed = [&](unsigned laneIndex) -> LaneAttempt {
    LaneAttempt attempt;
    attempt.laneIndex = laneIndex;
    attempt.state = contractedState;
    attempt.edgeKinds = initialContractedEdgeKinds;
    Mapper laneMapper = *this;
    mapper_detail::setActiveTopologyModel(activeTopologyModel_.get());
    mapper_detail::setActiveTimingOptions(&opts.timing);

    Options laneOpts = configureLaneOptions(laneIndex, laneMapper, true);
    const bool preferCPSatGlobal =
        laneOpts.enableCPSat && (laneCount == 1 || laneIndex == 0);
    if (preferCPSatGlobal) {
      laneOpts.cpSatTimeLimitSeconds =
          std::max(laneOpts.cpSatTimeLimitSeconds,
                   laneCount == 1 ? opts.lane.globalCPSatMinTimeSingleLane
                                  : opts.lane.globalCPSatMinTimeMultiLane);
    }

    if (opts.verbose && laneCount == 1)
      llvm::outs() << "Mapper: placing...\n";

    if (preferCPSatGlobal &&
        laneMapper.runCPSatGlobalPlacement(attempt.state, techPlan.contractedDFG,
                                           adg, flattener, candidates,
                                           laneOpts)) {
      attempt.usedCPSatGlobalPlacement = true;
    } else if (!laneMapper.runPlacement(attempt.state, techPlan.contractedDFG,
                                        adg, flattener, candidates,
                                        laneOpts)) {
      return attempt;
    }

    attempt.placementSucceeded = true;

    if (laneOpts.verbose)
      llvm::outs() << "Mapper: refining placement...\n";
    laneMapper.runRefinement(attempt.state, techPlan.contractedDFG, adg,
                             flattener, candidates, laneOpts,
                             &attempt.edgeKinds);
    laneMapper.rebindScalarInputSentinels(attempt.state, techPlan.contractedDFG,
                                          adg, flattener);

    classifyTemporalRegisterEdges(attempt.state, techPlan.contractedDFG, adg,
                                  flattener, attempt.edgeKinds);

    if (laneOpts.verbose)
      llvm::outs() << "Mapper: binding memref sentinels...\n";
    laneMapper.bindMemrefSentinels(attempt.state, techPlan.contractedDFG, adg);

    attempt.budgetExceeded = laneMapper.activeBudgetExceeded_;
    attempt.budgetExceededStage = laneMapper.activeBudgetExceededStage_;
    attempt.placementCost =
        computeTotalCost(attempt.state, techPlan.contractedDFG, adg, flattener);
    MapperTimingSummary attemptTiming = analyzeMapperTiming(
        attempt.state, techPlan.contractedDFG, adg, attempt.edgeKinds,
        laneOpts.timing);
    attempt.throughputCost = attemptTiming.estimatedThroughputCost;
    attempt.estimatedClockPeriod = attemptTiming.estimatedClockPeriod;
    attempt.searchSummary = laneMapper.activeSearchSummary_;
    return attempt;
  };

  auto runLaneRouting = [&](LaneAttempt attempt) -> LaneAttempt {
    if (!attempt.placementSucceeded)
      return attempt;
    Mapper laneMapper = *this;
    laneMapper.activeSearchSummary_ = attempt.searchSummary;
    mapper_detail::setActiveTopologyModel(activeTopologyModel_.get());
    mapper_detail::setActiveTimingOptions(&opts.timing);
    Options laneOpts =
        configureLaneOptions(attempt.laneIndex, laneMapper, false);

    if (laneOpts.enableCPSat) {
      auto criticalBoundaryEdges =
          collectCriticalBoundaryEdges(techPlan.contractedDFG);
      if (!criticalBoundaryEdges.empty()) {
        auto cpSatCheckpoint = attempt.state.save();
        Options criticalBoundaryOpts = laneOpts;
        criticalBoundaryOpts.cpSatTimeLimitSeconds =
            std::max(criticalBoundaryOpts.cpSatTimeLimitSeconds,
                     laneCount == 1
                         ? opts.lane.boundaryCPSatMinTimeSingleLane
                         : opts.lane.boundaryCPSatMinTimeMultiLane);
        criticalBoundaryOpts.cpSatNeighborhoodNodeLimit = std::min<unsigned>(
            criticalBoundaryOpts.cpSatNeighborhoodNodeLimit,
            opts.lane.boundaryNeighborhoodCap);
        laneMapper.runCPSatNeighborhoodRepair(
            attempt.state, cpSatCheckpoint, criticalBoundaryEdges,
            techPlan.contractedDFG, adg, flattener, candidates,
            attempt.edgeKinds, criticalBoundaryOpts);
        attempt.state.clearRoutes(dfg, adg);
        laneMapper.rebindScalarInputSentinels(attempt.state,
                                              techPlan.contractedDFG, adg,
                                              flattener);
        classifyTemporalRegisterEdges(attempt.state, techPlan.contractedDFG,
                                      adg, flattener, attempt.edgeKinds);
        laneMapper.bindMemrefSentinels(attempt.state, techPlan.contractedDFG,
                                       adg);
      }
    }

    if (laneOpts.verbose)
      llvm::outs() << "Mapper: routing...\n";
    attempt.routingSucceeded = laneMapper.runInterleavedPlaceRoute(
        attempt.state, techPlan.contractedDFG, adg, flattener, candidates,
        attempt.edgeKinds, laneOpts);
    attempt.budgetExceeded = laneMapper.activeBudgetExceeded_;
    attempt.budgetExceededStage = laneMapper.activeBudgetExceededStage_;
    attempt.routedEdges = countRoutedEdges(
        attempt.state, techPlan.contractedDFG, attempt.edgeKinds);
    attempt.totalPathLen = computeTotalMappedPathLen(attempt.state);
    attempt.placementCost =
        computeTotalCost(attempt.state, techPlan.contractedDFG, adg, flattener);
    MapperTimingSummary attemptTiming = analyzeMapperTiming(
        attempt.state, techPlan.contractedDFG, adg, attempt.edgeKinds,
        laneOpts.timing);
    attempt.throughputCost = attemptTiming.estimatedThroughputCost;
    attempt.estimatedClockPeriod = attemptTiming.estimatedClockPeriod;
    attempt.searchSummary = laneMapper.activeSearchSummary_;
    return attempt;
  };

  std::vector<LaneAttempt> placementSeeds;
  placementSeeds.reserve(laneCount);
  if (laneCount == 1) {
    placementSeeds.push_back(runLanePlacementSeed(0));
  } else {
    std::vector<std::future<LaneAttempt>> futures;
    futures.reserve(laneCount);
    for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
      futures.push_back(
          std::async(std::launch::async, runLanePlacementSeed, laneIndex));
    }
    for (auto &future : futures)
      placementSeeds.push_back(future.get());
  }

  llvm::SmallVector<LaneAttempt, 4> routingSeeds;
  for (const auto &attempt : placementSeeds) {
    if (attempt.placementSucceeded)
      routingSeeds.push_back(attempt);
  }
  llvm::stable_sort(routingSeeds, isBetterPlacementSeed);

  activeSearchSummary_.placementSeedLaneCount =
      static_cast<unsigned>(placementSeeds.size());
  activeSearchSummary_.successfulPlacementSeedCount =
      static_cast<unsigned>(routingSeeds.size());

  unsigned routingBeamWidth =
      opts.lane.routingBeamWidth == 0
          ? static_cast<unsigned>(routingSeeds.size())
          : std::min(opts.lane.routingBeamWidth,
                     static_cast<unsigned>(routingSeeds.size()));
  if (routingBeamWidth < routingSeeds.size())
    routingSeeds.resize(routingBeamWidth);
  activeSearchSummary_.routedLaneCount =
      static_cast<unsigned>(routingSeeds.size());

  for (auto &attempt : routingSeeds) {
    attempt.searchSummary.placementSeedLaneCount =
        activeSearchSummary_.placementSeedLaneCount;
    attempt.searchSummary.successfulPlacementSeedCount =
        activeSearchSummary_.successfulPlacementSeedCount;
    attempt.searchSummary.routedLaneCount =
        activeSearchSummary_.routedLaneCount;
  }

  if (opts.verbose && laneCount > 1 &&
      routingSeeds.size() < placementSeeds.size()) {
    llvm::outs() << "Mapper: narrowed " << placementSeeds.size()
                 << " placement lanes to " << routingSeeds.size()
                 << " routing lanes\n";
  }

  std::vector<LaneAttempt> attempts;
  attempts.reserve(routingSeeds.size());
  if (routingSeeds.size() <= 1) {
    for (auto &attempt : routingSeeds)
      attempts.push_back(runLaneRouting(std::move(attempt)));
  } else {
    std::vector<std::future<LaneAttempt>> futures;
    futures.reserve(routingSeeds.size());
    for (auto &attempt : routingSeeds) {
      futures.push_back(
          std::async(std::launch::async, runLaneRouting, std::move(attempt)));
    }
    for (auto &future : futures)
      attempts.push_back(future.get());
  }

  for (const auto &attempt : attempts) {
    if (!attempt.placementSucceeded)
      continue;
    if (opts.verbose && laneCount > 1) {
      llvm::outs() << "  Lane " << attempt.laneIndex << ": routed overall "
                   << attempt.routedEdges << "/"
                   << techPlan.contractedDFG.edges.size()
                   << " edges, pathLen=" << attempt.totalPathLen
                   << ", throughput=" << attempt.throughputCost
                   << ", clock=" << attempt.estimatedClockPeriod
                   << ", cost=" << attempt.placementCost << ", cp-sat-global="
                   << (attempt.usedCPSatGlobalPlacement ? "yes" : "no") << "\n";
    }
  }

  auto bestIt = attempts.end();
  for (auto it = attempts.begin(); it != attempts.end(); ++it) {
    if (bestIt == attempts.end() || isBetterLaneResult(*it, *bestIt))
      bestIt = it;
  }
  if (bestIt == attempts.end() || !bestIt->placementSucceeded) {
    result.diagnostics = "Placement failed";
    result.searchSummary = activeSearchSummary_;
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  contractedState = bestIt->state;
  auto contractedEdgeKinds = bestIt->edgeKinds;
  bool routingSucceeded = bestIt->routingSucceeded;
  bool selectedBudgetExceeded = bestIt->budgetExceeded;
  std::string selectedBudgetExceededStage = bestIt->budgetExceededStage;
  activeSearchSummary_ = bestIt->searchSummary;
  result.selectedLaneIndex = bestIt->laneIndex;
  if (opts.verbose && laneCount > 1) {
    llvm::outs() << "Mapper: selected lane " << bestIt->laneIndex
                 << " (routed overall " << bestIt->routedEdges << "/"
                 << techPlan.contractedDFG.edges.size()
                 << ", throughput=" << bestIt->throughputCost
                 << ", clock=" << bestIt->estimatedClockPeriod << ")\n";
  }

  if (!routingSucceeded && remainingBudgetSeconds() > 1.0) {
    auto polishFailed =
        collectUnroutedEdges(contractedState, techPlan.contractedDFG,
                             contractedEdgeKinds);
    if (opts.localRepair.enabled && !polishFailed.empty() &&
        polishFailed.size() <= opts.localRepair.cpSatFallbackFailedEdgeThreshold) {
      if (opts.verbose)
        llvm::outs() << "Mapper: final polish on selected lane...\n";

      auto routedCheckpoint = contractedState.save();
      contractedState.clearRoutes(techPlan.contractedDFG, adg);
      auto placementCheckpoint = contractedState.save();
      contractedState.restore(routedCheckpoint);

      Options polishOpts = opts;
      polishOpts.verbose = opts.verbose;
      polishOpts.cpSatTimeLimitSeconds =
          std::max(polishOpts.cpSatTimeLimitSeconds,
                   std::min(remainingBudgetSeconds(), 6.0));
      polishOpts.localRepair.focusedTargetMinTime =
          std::max(polishOpts.localRepair.focusedTargetMinTime,
                   std::min(remainingBudgetSeconds(), 4.0));

      activeBudgetExceeded_ = false;
      activeBudgetExceededStage_.clear();
      bool polishAllRouted =
          runLocalRepair(contractedState, placementCheckpoint, polishFailed,
                         techPlan.contractedDFG, adg, flattener, candidates,
                         contractedEdgeKinds, polishOpts);
      unsigned polishedRouted = countRoutedEdges(contractedState,
                                                 techPlan.contractedDFG,
                                                 contractedEdgeKinds);
      if (polishAllRouted || polishedRouted > bestIt->routedEdges)
        routingSucceeded = polishAllRouted;
      selectedBudgetExceeded = activeBudgetExceeded_;
      selectedBudgetExceededStage = activeBudgetExceededStage_;
    }
  }

  activeBudgetExceeded_ = selectedBudgetExceeded;
  activeBudgetExceededStage_ = selectedBudgetExceededStage;
  if (!routingSucceeded && !activeBudgetExceeded_ && opts.techFeedback.enabled &&
      techFeedbackAttempt < opts.techFeedback.maxRetries) {
    TechMapper::Feedback feedback;
    if (buildRoutingFailureFeedback(techPlan, contractedState,
                                    techPlan.contractedDFG, contractedEdgeKinds,
                                    opts.techFeedback, feedback)) {
      ++activeSearchSummary_.techMapFeedbackAttempts;
      TechMapper::Plan feedbackPlan;
      if (techMapper.applyFeedback(dfg, adg, techPlan, feedback,
                                   feedbackPlan)) {
        ++activeSearchSummary_.techMapFeedbackAcceptedReconfigurations;
        if (opts.verbose) {
          llvm::outs() << "Mapper: retrying techmap after routing failure with "
                       << feedback.splitCandidateIds.size()
                       << " split requests and "
                       << feedback.bannedCandidateIds.size()
                       << " banned candidates\n";
        }
        return runWithTechMapPlan(dfg, adg, flattener, adgModule, opts,
                                  techMapper, std::move(feedbackPlan),
                                  techFeedbackAttempt + 1);
      }
    }
  }
  if (!routingSucceeded) {
    result.diagnostics = activeBudgetExceeded_
                             ? ("Mapper budget exceeded" +
                                (activeBudgetExceededStage_.empty()
                                     ? std::string()
                                     : (" during " + activeBudgetExceededStage_)))
                             : "Routing failed";
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
  }

  MapperTimingSummary contractedTimingSummary =
      analyzeMapperTiming(contractedState, techPlan.contractedDFG, adg,
                          contractedEdgeKinds, opts.timing);
  if (routingSucceeded && opts.bufferization.enabled) {
    for (unsigned outerIter = 0;
         outerIter < opts.bufferization.outerJointIterations; ++outerIter) {
      auto beforeBufferizationCheckpoint = contractedState.save();
      auto beforeBufferizationOverrides =
          contractedState.hwNodeFifoBypassedOverride;
      auto beforeBufferizationEdgeKinds = contractedEdgeKinds;
      MapperTimingSummary bufferizedTiming = runPostRouteFifoBufferization(
          contractedState, techPlan.contractedDFG, adg, contractedEdgeKinds,
          opts);
      if (beforeBufferizationOverrides ==
          contractedState.hwNodeFifoBypassedOverride) {
        contractedState.restore(beforeBufferizationCheckpoint);
        contractedEdgeKinds = beforeBufferizationEdgeKinds;
        break;
      }
      if (!isBetterTimingSummary(bufferizedTiming, contractedTimingSummary,
                                 opts.bufferization)) {
        contractedState.restore(beforeBufferizationCheckpoint);
        contractedEdgeKinds = beforeBufferizationEdgeKinds;
        break;
      }

      contractedTimingSummary = bufferizedTiming;
      auto acceptedBufferizedCheckpoint = contractedState.save();
      auto acceptedBufferizedEdgeKinds = contractedEdgeKinds;
      if (opts.verbose) {
        llvm::outs() << "Mapper: accepted bufferization round "
                     << (outerIter + 1) << "/"
                     << opts.bufferization.outerJointIterations
                     << ", throughput="
                     << contractedTimingSummary.estimatedThroughputCost
                     << ", clock="
                     << contractedTimingSummary.estimatedClockPeriod
                     << ", ii="
                     << contractedTimingSummary.estimatedInitiationInterval
                     << "\n";
      }

      if (outerIter + 1 >= opts.bufferization.outerJointIterations)
        break;

      Options outerOpts = opts;
      outerOpts.seed =
          opts.seed +
          static_cast<int>((outerIter + 1) * opts.lane.restartSeedStride);
      runRefinement(contractedState, techPlan.contractedDFG, adg, flattener,
                    candidates, outerOpts, &contractedEdgeKinds);
      rebindScalarInputSentinels(contractedState, techPlan.contractedDFG, adg,
                                flattener);
      bindMemrefSentinels(contractedState, techPlan.contractedDFG, adg);
      bool outerRoutingSucceeded = runInterleavedPlaceRoute(
          contractedState, techPlan.contractedDFG, adg, flattener, candidates,
          contractedEdgeKinds, outerOpts);
      if (!outerRoutingSucceeded) {
        contractedState.restore(acceptedBufferizedCheckpoint);
        contractedEdgeKinds = acceptedBufferizedEdgeKinds;
        break;
      }
      MapperTimingSummary reroutedTiming =
          analyzeMapperTiming(contractedState, techPlan.contractedDFG, adg,
                              contractedEdgeKinds, opts.timing);
      if (!isBetterTimingSummary(reroutedTiming, contractedTimingSummary,
                                 opts.bufferization)) {
        contractedState.restore(acceptedBufferizedCheckpoint);
        contractedEdgeKinds = acceptedBufferizedEdgeKinds;
        break;
      }
      contractedTimingSummary = reroutedTiming;
      if (opts.verbose) {
        llvm::outs() << "Mapper: accepted outer joint PnR round "
                     << (outerIter + 1) << ", throughput="
                     << contractedTimingSummary.estimatedThroughputCost
                     << ", clock="
                     << contractedTimingSummary.estimatedClockPeriod
                     << ", ii="
                     << contractedTimingSummary.estimatedInitiationInterval
                     << "\n";
      }
      ++activeSearchSummary_.outerJointAcceptedRounds;
    }
  }

  auto expandContractedMapping =
      [&](const MappingState &contractedMapping,
          llvm::ArrayRef<TechMappedEdgeKind> contractedKinds,
          MappingState &expandedState,
          llvm::SmallVector<FUConfigSelection, 4> &expandedFuConfigs,
          std::vector<TechMappedEdgeKind> &expandedEdgeKinds,
          std::string &expandDiagnostics) -> bool {
    if (!techMapper.expandPlanMapping(dfg, adg, techPlan, contractedMapping,
                                      expandedState, expandedFuConfigs)) {
      expandDiagnostics = "Tech-mapping expansion failed";
      return false;
    }

    for (IdIndex swNode = 0; swNode < static_cast<IdIndex>(dfg.nodes.size());
         ++swNode) {
      const Node *swNodeObj = dfg.getNode(swNode);
      if (!swNodeObj || swNodeObj->kind != Node::OperationNode)
        continue;
      if (!mapper_detail::isSoftwareMemoryInterfaceOp(
              getNodeAttrStr(swNodeObj, "op_name"))) {
        continue;
      }
      if (swNode >= expandedState.swNodeToHwNode.size() ||
          expandedState.swNodeToHwNode[swNode] == INVALID_ID) {
        continue;
      }

      bool needsRebind = false;
      for (IdIndex swPort : swNodeObj->inputPorts) {
        if (swPort >= expandedState.swPortToHwPort.size() ||
            expandedState.swPortToHwPort[swPort] == INVALID_ID) {
          needsRebind = true;
          break;
        }
      }
      if (!needsRebind) {
        for (IdIndex swPort : swNodeObj->outputPorts) {
          if (swPort >= expandedState.swPortToHwPort.size() ||
              expandedState.swPortToHwPort[swPort] == INVALID_ID) {
            needsRebind = true;
            break;
          }
        }
      }
      if (!needsRebind)
        continue;
      if (!bindMappedNodePorts(swNode, expandedState, dfg, adg)) {
        expandDiagnostics = "Port rebinding failed after tech-mapping expansion";
        return false;
      }
    }

    expandedEdgeKinds.assign(originalEdgeKinds.begin(), originalEdgeKinds.end());
    for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
         ++edgeId) {
      if (edgeId >= expandedEdgeKinds.size() ||
          expandedEdgeKinds[edgeId] == TechMappedEdgeKind::IntraFU) {
        continue;
      }
      if (edgeId >= techPlan.originalEdgeToContractedEdge.size())
        continue;
      IdIndex contractedEdgeId = techPlan.originalEdgeToContractedEdge[edgeId];
      if (contractedEdgeId == INVALID_ID ||
          contractedEdgeId >= contractedKinds.size()) {
        continue;
      }
      if (contractedKinds[contractedEdgeId] == TechMappedEdgeKind::TemporalReg)
        expandedEdgeKinds[edgeId] = TechMappedEdgeKind::TemporalReg;
    }
    return true;
  };

  if (!expandContractedMapping(contractedState, contractedEdgeKinds, result.state,
                               result.fuConfigs, result.edgeKinds,
                               result.diagnostics)) {
    result.searchSummary = activeSearchSummary_;
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  llvm::outs() << "Mapper: validating...\n";
  bool validationSucceeded = runValidation(
      result.state, dfg, adg, flattener, result.edgeKinds, result.diagnostics);
  if (!validationSucceeded) {
    llvm::errs() << "Mapper: validation issues: " << result.diagnostics << "\n";
    // Proceed with partial result.
  }

  result.timingSummary =
      analyzeMapperTiming(result.state, dfg, adg, result.edgeKinds,
                          opts.timing);
  result.searchSummary = activeSearchSummary_;

  std::vector<LaneAttempt> sortedAttempts = attempts;
  llvm::stable_sort(sortedAttempts, isBetterLaneResult);
  for (const auto &attempt : sortedAttempts) {
    if (!attempt.placementSucceeded || !attempt.routingSucceeded)
      continue;
    if (attempt.laneIndex == result.selectedLaneIndex)
      continue;
    Result::RoutedAlternative alternative;
    alternative.laneIndex = attempt.laneIndex;
    alternative.totalPathLen = attempt.totalPathLen;
    alternative.placementCost = attempt.placementCost;
    alternative.throughputCost = attempt.throughputCost;
    alternative.estimatedClockPeriod = attempt.estimatedClockPeriod;
    alternative.searchSummary = attempt.searchSummary;
    std::string expandDiagnostics;
    if (!expandContractedMapping(attempt.state, attempt.edgeKinds,
                                 alternative.state, alternative.fuConfigs,
                                 alternative.edgeKinds, expandDiagnostics)) {
      if (opts.verbose) {
        llvm::outs() << "Mapper: skipping lane " << attempt.laneIndex
                     << " fallback export: " << expandDiagnostics << "\n";
      }
      continue;
    }
    alternative.timingSummary = analyzeMapperTiming(
        alternative.state, dfg, adg, alternative.edgeKinds, opts.timing);
    result.routedAlternatives.push_back(std::move(alternative));
  }

  result.success = routingSucceeded && validationSucceeded;
  activeSnapshotEmitter_ = nullptr;
  llvm::outs() << "Mapper: done.\n";
  return result;
}

// ---------------------------------------------------------------------------
// Mapper::runValidation
// ---------------------------------------------------------------------------

bool Mapper::runValidation(const MappingState &state, const Graph &dfg,
                           const Graph &adg, const ADGFlattener &flattener,
                           llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                           std::string &diagnostics) {
  bool valid = true;
  unsigned unroutedDiagnosticsEmitted = 0;
  auto emitUnroutedEdgeDebug = [&](IdIndex edgeId) {
    if (unroutedDiagnosticsEmitted >= activeUnroutedDiagnosticLimit)
      return;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      return;
    IdIndex srcHwPort =
        edge->srcPort < state.swPortToHwPort.size()
            ? state.swPortToHwPort[edge->srcPort]
            : INVALID_ID;
    IdIndex dstHwPort =
        edge->dstPort < state.swPortToHwPort.size()
            ? state.swPortToHwPort[edge->dstPort]
            : INVALID_ID;
    const Port *srcSwPort = dfg.getPort(edge->srcPort);
    const Port *dstSwPort = dfg.getPort(edge->dstPort);
    const Node *srcNode =
        (srcSwPort && srcSwPort->parentNode != INVALID_ID)
            ? dfg.getNode(srcSwPort->parentNode)
            : nullptr;
    const Node *dstNode =
        (dstSwPort && dstSwPort->parentNode != INVALID_ID)
            ? dfg.getNode(dstSwPort->parentNode)
            : nullptr;

    MappingState probeState = state;
    probeState.clearRoutes(dfg, adg);
    llvm::DenseMap<IdIndex, double> emptyHistory;
    auto freeSpacePath =
        findPath(srcHwPort, dstHwPort, edgeId, probeState, dfg, adg,
                 emptyHistory);

    llvm::outs() << "  Unrouted edge debug " << edgeId << ": "
                 << (srcNode ? getNodeAttrStr(srcNode, "op_name") : "<src>")
                 << " -> "
                 << (dstNode ? getNodeAttrStr(dstNode, "op_name") : "<dst>")
                 << ", hw " << srcHwPort << " -> " << dstHwPort
                 << ", free-space-path="
                 << (freeSpacePath.empty() ? "none" : "yes");
    if (!freeSpacePath.empty())
      llvm::outs() << " len=" << freeSpacePath.size();
    llvm::outs() << "\n";
    ++unroutedDiagnosticsEmitted;
  };

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
    if (i < edgeKinds.size() &&
        (edgeKinds[i] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[i] == TechMappedEdgeKind::TemporalReg))
      continue;
    if (i >= state.swEdgeToHwPaths.size() || state.swEdgeToHwPaths[i].empty()) {
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
          diagnostics += "C3: unrouted edge " + std::to_string(i) + "\n";
          emitUnroutedEdgeDebug(i);
          valid = false;
        }
      }
    }
  }

  for (IdIndex edgeId = 0;
       edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size()); ++edgeId) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    const auto &path = state.swEdgeToHwPaths[edgeId];
    if (path.size() < 3)
      continue;

    for (size_t pathIdx = 1; pathIdx + 1 < path.size(); ++pathIdx) {
      const Port *port = adg.getPort(path[pathIdx]);
      if (!port || port->parentNode == INVALID_ID)
        continue;
      const Node *owner = adg.getNode(port->parentNode);
      if (!owner)
        continue;
      if (getNodeAttrStr(owner, "resource_class") != "functional")
        continue;
      diagnostics += "C9: routed edge " + std::to_string(edgeId) +
                     " illegally traverses functional node " +
                     getNodeAttrStr(owner, "op_name").str() + "\n";
      valid = false;
      break;
    }
  }

  llvm::DenseMap<IdIndex, IdIndex> firstHopByNonRoutingSource;
  for (IdIndex edgeId = 0;
       edgeId < static_cast<IdIndex>(state.swEdgeToHwPaths.size()); ++edgeId) {
    if (edgeId < edgeKinds.size() &&
        (edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU ||
         edgeKinds[edgeId] == TechMappedEdgeKind::TemporalReg))
      continue;
    const auto &path = state.swEdgeToHwPaths[edgeId];
    if (path.size() < 2)
      continue;

    IdIndex srcPortId = path.front();
    IdIndex firstHopInputId = path[1];
    const Port *srcPort = adg.getPort(srcPortId);
    const Port *firstHopInput = adg.getPort(firstHopInputId);
    if (!srcPort || !firstHopInput || srcPort->direction != Port::Output ||
        firstHopInput->direction != Port::Input ||
        srcPort->parentNode == INVALID_ID)
      continue;

    const Node *owner = adg.getNode(srcPort->parentNode);
    if (isRoutingResourceNode(owner))
      continue;

    auto it = firstHopByNonRoutingSource.find(srcPortId);
    if (it == firstHopByNonRoutingSource.end()) {
      firstHopByNonRoutingSource[srcPortId] = firstHopInputId;
      continue;
    }
    if (it->second == firstHopInputId)
      continue;

    diagnostics += "C10: non-routing source port " + std::to_string(srcPortId) +
                   " fans out to multiple next hops (" +
                   std::to_string(it->second) + " and " +
                   std::to_string(firstHopInputId) + ")\n";
    valid = false;
  }

  for (IdIndex outPortId = 0; outPortId < static_cast<IdIndex>(adg.ports.size());
       ++outPortId) {
    if (!isRelaxableRoutingOutput(outPortId, adg))
      continue;
    unsigned distinctSources =
        countDistinctLogicalSourcesForOutput(outPortId, state);
    if (distinctSources <= 1)
      continue;
    diagnostics += "C4: non-tagged routing output overuse on hw_port " +
                   std::to_string(outPortId) + " with " +
                   std::to_string(distinctSources) + " logical sources\n";
    valid = false;
  }

  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
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
    llvm::SmallVector<std::pair<IdIndex, BridgeLaneUsage>, 4> usedLaneUsages;
    for (IdIndex swId : state.hwNodeToSwNodes[hwId]) {
      const Node *swNode = dfg.getNode(swId);
      if (!swNode)
        continue;
      DfgMemoryInfo memInfo = DfgMemoryInfo::extract(swNode, dfg, isExtMem);
      auto laneRange =
          inferBridgeLaneRange(bridge, memInfo, swNode, dfg, state);
      if (!laneRange) {
        diagnostics += "C4: missing bridge lane range for sw_node " +
                       std::to_string(swId) + " on hw_node " +
                       std::to_string(hwId) + "\n";
        valid = false;
        continue;
      }
      BridgeLaneUsage laneUsage =
          computeBridgeLaneUsage(memInfo, laneRange->start);
      for (const auto &usedEntry : usedLaneUsages) {
        if (!bridgeLaneUsageConflicts(laneUsage, usedEntry.second))
          continue;
        diagnostics += "C4: conflicting bridge family lanes on hw_node " +
                       std::to_string(hwId) + " between sw_node " +
                       std::to_string(swId) + " and sw_node " +
                       std::to_string(usedEntry.first) + "\n";
          valid = false;
          break;
      }
      usedLaneUsages.push_back({swId, laneUsage});
    }
  }

  llvm::StringMap<llvm::DenseSet<IdIndex>> activeSpatialFUsByPE;
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    if (hwId >= state.hwNodeToSwNodes.size() ||
        state.hwNodeToSwNodes[hwId].empty())
      continue;
    const Node *hwNode = adg.getNode(hwId);
    llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
    if (peName.empty() || !isSpatialPEName(flattener, peName))
      continue;
    activeSpatialFUsByPE[peName].insert(hwId);
  }
  for (const auto &entry : activeSpatialFUsByPE) {
    if (entry.getValue().size() > 1) {
      diagnostics +=
          "C8: multiple active function_unit instances in spatial_pe " +
          entry.getKey().str() + "\n";
      valid = false;
    }
  }

  for (const auto &entry : activeSpatialFUsByPE) {
    const PEContainment *pe = findPEContainmentByName(flattener, entry.getKey());
    if (!pe)
      continue;
    auto routes =
        configgen_detail::collectPERouteSummary(*pe, state, dfg, adg);
    llvm::DenseMap<unsigned, IdIndex> usedOutputs;
    for (IdIndex fuId : entry.getValue()) {
      auto it = routes.outputPortSelects.find(fuId);
      if (it == routes.outputPortSelects.end())
        continue;
      const auto &selects = it->second;
      for (size_t outputIdx = 0; outputIdx < selects.size(); ++outputIdx) {
        int peOutputIndex = selects[outputIdx];
        if (peOutputIndex < 0)
          continue;
        auto existing = usedOutputs.find(static_cast<unsigned>(peOutputIndex));
        if (existing != usedOutputs.end() &&
            existing->second != static_cast<IdIndex>(outputIdx)) {
          diagnostics += "C8: spatial_pe " + entry.getKey().str() +
                         " output index " + std::to_string(peOutputIndex) +
                         " used by multiple function_unit results\n";
          valid = false;
          continue;
        }
        usedOutputs[static_cast<unsigned>(peOutputIndex)] =
            static_cast<IdIndex>(outputIdx);
      }
    }
  }

  llvm::StringMap<llvm::DenseSet<IdIndex>> temporalRegsByPE;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeId >= edgeKinds.size() ||
        edgeKinds[edgeId] != TechMappedEdgeKind::TemporalReg)
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
    IdIndex srcHwId = state.swNodeToHwNode[srcNodeId];
    IdIndex dstHwId = state.swNodeToHwNode[dstNodeId];
    const Node *srcHwNode = adg.getNode(srcHwId);
    const Node *dstHwNode = adg.getNode(dstHwId);
    if (!isTemporalPENode(srcHwNode) || !isTemporalPENode(dstHwNode))
      continue;
    llvm::StringRef peName = getNodeAttrStr(srcHwNode, "pe_name");
    if (peName.empty() || peName != getNodeAttrStr(dstHwNode, "pe_name"))
      continue;
    temporalRegsByPE[peName].insert(edge->srcPort);
  }

  for (const auto &entry : temporalRegsByPE) {
    const PEContainment *pe =
        findPEContainmentByName(flattener, entry.getKey());
    if (!pe)
      continue;
    if (entry.getValue().size() > pe->numRegister) {
      diagnostics +=
          "C5.2: temporal register overflow on " + entry.getKey().str() + "\n";
      valid = false;
    }
  }

  if (!validateTaggedPathConflicts(state, dfg, adg, edgeKinds, diagnostics))
    valid = false;

  return valid;
}

} // namespace fcc

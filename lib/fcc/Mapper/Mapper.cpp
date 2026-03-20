#include "fcc/Mapper/Mapper.h"
#include "MapperCongestionEstimator.h"
#include "MapperInternal.h"
#include "MapperRoutingCongestion.h"
#include "fcc/Mapper/BridgeBinding.h"
#include "fcc/Mapper/TypeCompat.h"

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
  if (lhs.totalPathLen != rhs.totalPathLen)
    return lhs.totalPathLen < rhs.totalPathLen;
  if (std::abs(lhs.placementCost - rhs.placementCost) > 1e-9)
    return lhs.placementCost < rhs.placementCost;
  return lhs.laneIndex < rhs.laneIndex;
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
    auto [srcRow, srcCol] = flattener.getNodeGridPos(adgSentinel);
    if (srcRow < 0 || srcCol < 0)
      return std::numeric_limits<double>::infinity();

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
      auto [dstRow, dstCol] = flattener.getNodeGridPos(dstHw);
      if (dstRow < 0 || dstCol < 0)
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
      cost += weight * static_cast<double>(std::abs(srcRow - dstRow) +
                                           std::abs(srcCol - dstCol));
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
                     totalPathLen < bestPathLen) ||
                    (routed == bestRouted && totalPathLen == bestPathLen &&
                     (!usePriority ||
                      (priorityRouted == bestPriorityRouted &&
                       std::abs(priorityPenalty - bestPriorityPenalty) <=
                           1e-9)) &&
                     placementCost + 1e-9 < bestPlacementCost);
    if (!improved)
      return false;
    bestCheckpoint = state.save();
    bestRouted = routed;
    bestUnroutedPenalty = unroutedPenalty;
    bestPathLen = totalPathLen;
    bestPlacementCost = placementCost;
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
    bool repaired =
        runLocalRepair(state, currentPlacementCheckpoint, failedEdges, dfg, adg,
                       flattener, candidates, edgeKinds, opts,
                       repairCongestionPtr);
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
    state.clearRoutes(dfg, adg);
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
    runRefinement(state, dfg, adg, flattener, candidates, restartOpts);
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

// ---------------------------------------------------------------------------
// Mapper::run
// ---------------------------------------------------------------------------

Mapper::Result Mapper::run(const Graph &dfg, const Graph &adg,
                           const ADGFlattener &flattener,
                           mlir::ModuleOp adgModule, const Options &opts) {
  Result result;
  if (!validateMapperOptions(opts, result.diagnostics)) {
    llvm::errs() << "Mapper: invalid options: " << result.diagnostics << "\n";
    return result;
  }
  resetRunControls(opts);
  activeSnapshotEmitter_ = nullptr;
  TechMapper techMapper;
  TechMapper::Plan techPlan;
  if (!techMapper.buildPlan(dfg, adgModule, adg, techPlan)) {
    result.diagnostics = "Tech-mapping failed";
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  MappingState contractedState;
  contractedState.init(techPlan.contractedDFG, adg, &flattener);
  result.edgeKinds = techPlan.originalEdgeKinds;
  const auto originalEdgeKinds = result.edgeKinds;
  std::vector<TechMappedEdgeKind> initialContractedEdgeKinds(
      techPlan.contractedDFG.edges.size(), TechMappedEdgeKind::Routed);

  // Copy connectivity from flattener.
  connectivity = flattener.getConnectivity();
  activeFlattener = &flattener;
  activeHeuristicWeight = opts.routingHeuristicWeight;
  activeCongestionPlacementWeight = opts.congestionPlacementWeight;
  activeMemorySharingPenalty = opts.memorySharingPenalty;
  activeUnroutedDiagnosticLimit = opts.lane.unroutedDiagnosticLimit;
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
  for (const auto &entry : techPlan.contractedCandidates)
    candidates[entry.first] = entry.second;

  if (detectForcedTemporalConfigConflict(techPlan, adg, result.diagnostics)) {
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

  auto runLane = [&](unsigned laneIndex) -> LaneAttempt {
    LaneAttempt attempt;
    attempt.laneIndex = laneIndex;
    attempt.state = contractedState;
    attempt.edgeKinds = initialContractedEdgeKinds;
    Mapper laneMapper = *this;

    Options laneOpts = opts;
    laneOpts.seed =
        opts.seed + static_cast<int>(laneIndex * opts.lane.laneSeedStride);
    laneOpts.verbose = (laneCount == 1) ? opts.verbose : false;
    if (laneOpts.budgetSeconds > 0.0 &&
        laneOpts.lane.finalPolishReserveFraction > 0.0) {
      double laneBudgetSeconds =
          laneOpts.budgetSeconds *
          std::max(0.0, 1.0 - laneOpts.lane.finalPolishReserveFraction);
      laneOpts.budgetSeconds = std::max(1.0, laneBudgetSeconds);
      laneMapper.resetRunControls(laneOpts);
    }
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
                             flattener, candidates, laneOpts);
    laneMapper.rebindScalarInputSentinels(attempt.state, techPlan.contractedDFG,
                                          adg, flattener);

    classifyTemporalRegisterEdges(attempt.state, techPlan.contractedDFG, adg,
                                  flattener, attempt.edgeKinds);

    if (laneOpts.verbose)
      llvm::outs() << "Mapper: binding memref sentinels...\n";
    laneMapper.bindMemrefSentinels(attempt.state, techPlan.contractedDFG, adg);

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
    return attempt;
  };

  std::vector<LaneAttempt> attempts;
  attempts.reserve(laneCount);
  if (laneCount == 1) {
    attempts.push_back(runLane(0));
  } else {
    std::vector<std::future<LaneAttempt>> futures;
    futures.reserve(laneCount);
    for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
      futures.push_back(std::async(std::launch::async, runLane, laneIndex));
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
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  contractedState = bestIt->state;
  auto contractedEdgeKinds = bestIt->edgeKinds;
  bool routingSucceeded = bestIt->routingSucceeded;
  bool selectedBudgetExceeded = bestIt->budgetExceeded;
  std::string selectedBudgetExceededStage = bestIt->budgetExceededStage;
  if (opts.verbose && laneCount > 1) {
    llvm::outs() << "Mapper: selected lane " << bestIt->laneIndex
                 << " (routed overall " << bestIt->routedEdges << "/"
                 << techPlan.contractedDFG.edges.size() << ")\n";
  }

  if (!routingSucceeded && remainingBudgetSeconds() > 1.0) {
    auto polishFailed =
        collectUnroutedEdges(contractedState, techPlan.contractedDFG,
                             contractedEdgeKinds);
    if (!polishFailed.empty() &&
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
  if (!routingSucceeded) {
    result.diagnostics = activeBudgetExceeded_
                             ? ("Mapper budget exceeded" +
                                (activeBudgetExceededStage_.empty()
                                     ? std::string()
                                     : (" during " + activeBudgetExceededStage_)))
                             : "Routing failed";
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
  }

  if (!techMapper.expandPlanMapping(dfg, adg, techPlan, contractedState,
                                    result.state, result.fuConfigs)) {
    result.diagnostics = "Tech-mapping expansion failed";
    llvm::errs() << "Mapper: " << result.diagnostics << "\n";
    return result;
  }

  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (edgeId >= result.edgeKinds.size() ||
        result.edgeKinds[edgeId] == TechMappedEdgeKind::IntraFU)
      continue;
    if (edgeId >= techPlan.originalEdgeToContractedEdge.size())
      continue;
    IdIndex contractedEdgeId = techPlan.originalEdgeToContractedEdge[edgeId];
    if (contractedEdgeId == INVALID_ID ||
        contractedEdgeId >= contractedEdgeKinds.size())
      continue;
    if (contractedEdgeKinds[contractedEdgeId] ==
        TechMappedEdgeKind::TemporalReg)
      result.edgeKinds[edgeId] = TechMappedEdgeKind::TemporalReg;
  }

  llvm::outs() << "Mapper: validating...\n";
  bool validationSucceeded = runValidation(
      result.state, dfg, adg, flattener, result.edgeKinds, result.diagnostics);
  if (!validationSucceeded) {
    llvm::errs() << "Mapper: validation issues: " << result.diagnostics << "\n";
    // Proceed with partial result.
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
      auto laneRange = inferBridgeLaneRange(bridge, memInfo, swNode, state);
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

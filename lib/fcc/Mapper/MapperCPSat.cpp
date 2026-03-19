#include "MapperInternal.h"
#include "fcc/Mapper/Mapper.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

#ifdef FCC_HAVE_ORTOOLS
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/model.h"
#endif

namespace fcc {

using namespace mapper_detail;

#ifdef FCC_HAVE_ORTOOLS
using namespace operations_research::sat;

namespace {

struct PreparedDomain {
  IdIndex swNode = INVALID_ID;
  IdIndex hintedHw = INVALID_ID;
  std::vector<IdIndex> hwCandidates;
};

int64_t scaledPlacementCost(double weight, int distance) {
  return static_cast<int64_t>(std::llround(weight * 100.0)) *
         static_cast<int64_t>(distance);
}

bool isPlacedOperationNode(IdIndex swNode, const MappingState &state,
                           const Graph &dfg) {
  const Node *node = dfg.getNode(swNode);
  if (!node || node->kind != Node::OperationNode)
    return false;
  return swNode < state.swNodeToHwNode.size() &&
         state.swNodeToHwNode[swNode] != INVALID_ID;
}

bool solvePlacementSubset(
    MappingState &state, const MappingState::Checkpoint &baseCheckpoint,
    llvm::ArrayRef<PreparedDomain> domains, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener, const Mapper::Options &opts,
    const llvm::DenseMap<IdIndex, double> *edgeWeightOverrides = nullptr) {
  if (domains.empty())
    return false;

  state.restore(baseCheckpoint);

  llvm::DenseSet<IdIndex> variableNodes;
  for (const auto &domain : domains)
    variableNodes.insert(domain.swNode);

  std::vector<int64_t> remainingCapacity(adg.nodes.size(), 0);
  llvm::StringMap<int64_t> remainingSpatialPECapacity;
  for (IdIndex hwNode = 0; hwNode < static_cast<IdIndex>(adg.nodes.size());
       ++hwNode) {
    const Node *node = adg.getNode(hwNode);
    if (!node)
      continue;
    int64_t capacity = 1;
    if (getNodeAttrStr(node, "resource_class") == "memory")
      capacity = std::max<int64_t>(1, getNodeAttrInt(node, "numRegion", 1));
    int64_t fixedOccupants = 0;
    if (hwNode < state.hwNodeToSwNodes.size()) {
      for (IdIndex swNode : state.hwNodeToSwNodes[hwNode]) {
        if (!variableNodes.contains(swNode))
          ++fixedOccupants;
      }
    }
    remainingCapacity[hwNode] = capacity - fixedOccupants;

    llvm::StringRef peName = getNodeAttrStr(node, "pe_name");
    if (!peName.empty() && isSpatialPENode(node) &&
        !remainingSpatialPECapacity.count(peName)) {
      int64_t fixedPEOccupants = 0;
      for (IdIndex peHwNode = 0;
           peHwNode < static_cast<IdIndex>(adg.nodes.size()); ++peHwNode) {
        const Node *peNode = adg.getNode(peHwNode);
        if (!peNode || getNodeAttrStr(peNode, "pe_name") != peName)
          continue;
        if (peHwNode >= state.hwNodeToSwNodes.size())
          continue;
        for (IdIndex swNode : state.hwNodeToSwNodes[peHwNode]) {
          if (!variableNodes.contains(swNode)) {
            ++fixedPEOccupants;
            break;
          }
        }
      }
      remainingSpatialPECapacity[peName] =
          std::max<int64_t>(0, 1 - fixedPEOccupants);
    }
  }

  CpModelBuilder cpModel;
  LinearExpr objective;
  std::vector<std::vector<BoolVar>> assignmentVars(domains.size());
  std::vector<std::vector<IdIndex>> activeCandidates(domains.size());
  llvm::DenseMap<IdIndex, unsigned> nodeToDomainIndex;
  for (unsigned domainIndex = 0; domainIndex < domains.size(); ++domainIndex) {
    nodeToDomainIndex[domains[domainIndex].swNode] = domainIndex;
  }

  llvm::DenseMap<IdIndex, std::vector<BoolVar>> varsByHwNode;
  llvm::StringMap<std::vector<BoolVar>> varsBySpatialPE;

  auto estimatedPlacementDistance = [&](IdIndex srcSwNode, IdIndex srcSwPort,
                                        IdIndex srcHwNode, IdIndex dstSwNode,
                                        IdIndex dstSwPort,
                                        IdIndex dstHwNode) -> int {
    auto srcPos = estimateSoftwarePortPlacementPos(
        srcSwNode, srcSwPort, srcHwNode, /*isInput=*/false, dfg, adg,
        flattener);
    auto dstPos = estimateSoftwarePortPlacementPos(
        dstSwNode, dstSwPort, dstHwNode, /*isInput=*/true, dfg, adg,
        flattener);
    if (srcPos && dstPos) {
      double dist = std::abs(srcPos->first - dstPos->first) +
                    std::abs(srcPos->second - dstPos->second);
      return static_cast<int>(std::lround(dist));
    }
    return manhattanDistance(srcHwNode, dstHwNode, flattener);
  };

  for (unsigned domainIndex = 0; domainIndex < domains.size(); ++domainIndex) {
    const auto &domain = domains[domainIndex];
    assignmentVars[domainIndex].reserve(domain.hwCandidates.size());
    activeCandidates[domainIndex].reserve(domain.hwCandidates.size());
    for (IdIndex hwCandidate : domain.hwCandidates) {
      if (hwCandidate == INVALID_ID || hwCandidate >= remainingCapacity.size())
        continue;
      if (remainingCapacity[hwCandidate] <= 0 &&
          hwCandidate != domain.hintedHw) {
        continue;
      }
      const Node *hwNode = adg.getNode(hwCandidate);
      llvm::StringRef peName =
          hwNode ? getNodeAttrStr(hwNode, "pe_name") : llvm::StringRef();
      if (!peName.empty() && hwNode && isSpatialPENode(hwNode)) {
        auto capIt = remainingSpatialPECapacity.find(peName);
        if (capIt != remainingSpatialPECapacity.end() && capIt->second <= 0 &&
            hwCandidate != domain.hintedHw) {
          continue;
        }
      }
      BoolVar var = cpModel.NewBoolVar();
      assignmentVars[domainIndex].push_back(var);
      activeCandidates[domainIndex].push_back(hwCandidate);
      varsByHwNode[hwCandidate].push_back(var);
      if (!peName.empty() && hwNode && isSpatialPENode(hwNode))
        varsBySpatialPE[peName].push_back(var);
      if (hwCandidate == domain.hintedHw)
        cpModel.AddHint(var, true);
      else
        cpModel.AddHint(var, false);
      if (domain.hintedHw != INVALID_ID && hwCandidate != domain.hintedHw)
        objective += 3 * var;
    }
    if (assignmentVars[domainIndex].empty()) {
      if (opts.verbose) {
        llvm::outs() << "  CP-SAT domain exhausted for sw "
                     << domain.swNode << " hint=" << domain.hintedHw
                     << " raw=" << domain.hwCandidates.size() << "\n";
      }
      return false;
    }
    cpModel.AddExactlyOne(assignmentVars[domainIndex]);
  }

  for (const auto &entry : varsByHwNode) {
    IdIndex hwNode = entry.first;
    if (hwNode >= remainingCapacity.size())
      continue;
    int64_t capacity = remainingCapacity[hwNode];
    if (capacity <= 0)
      continue;
    if (capacity == 1) {
      cpModel.AddAtMostOne(entry.second);
      continue;
    }
    LinearExpr occupancy;
    for (const BoolVar &var : entry.second)
      occupancy += var;
    cpModel.AddLessOrEqual(occupancy, capacity);
  }

  for (const auto &entry : varsBySpatialPE) {
    auto capIt = remainingSpatialPECapacity.find(entry.getKey());
    int64_t capacity = capIt == remainingSpatialPECapacity.end() ? 1
                                                                 : capIt->second;
    if (capacity <= 0) {
      for (const BoolVar &var : entry.getValue())
        cpModel.AddEquality(var, 0);
      continue;
    }
    if (capacity == 1) {
      cpModel.AddAtMostOne(entry.getValue());
      continue;
    }
    LinearExpr occupancy;
    for (const BoolVar &var : entry.getValue())
      occupancy += var;
    cpModel.AddLessOrEqual(occupancy, capacity);
  }

  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;

    bool srcVariable = nodeToDomainIndex.count(srcPort->parentNode);
    bool dstVariable = nodeToDomainIndex.count(dstPort->parentNode);
    if (!srcVariable && !dstVariable)
      continue;

    double edgeWeightBase =
        edgeWeightOverrides && edgeWeightOverrides->count(edgeId)
            ? edgeWeightOverrides->lookup(edgeId)
            : classifyEdgePlacementWeight(dfg, edgeId);
    int64_t edgeWeight = scaledPlacementCost(edgeWeightBase, 1);
    if (edgeWeight <= 0)
      continue;

    if (srcVariable && dstVariable) {
      unsigned srcIndex = nodeToDomainIndex.lookup(srcPort->parentNode);
      unsigned dstIndex = nodeToDomainIndex.lookup(dstPort->parentNode);
      for (unsigned srcCandIdx = 0;
           srcCandIdx < assignmentVars[srcIndex].size(); ++srcCandIdx) {
        for (unsigned dstCandIdx = 0;
             dstCandIdx < assignmentVars[dstIndex].size(); ++dstCandIdx) {
          IdIndex srcHw = activeCandidates[srcIndex][srcCandIdx];
          IdIndex dstHw = activeCandidates[dstIndex][dstCandIdx];
          int64_t distanceCost = edgeWeight * estimatedPlacementDistance(
                                                  srcPort->parentNode,
                                                  edge->srcPort, srcHw,
                                                  dstPort->parentNode,
                                                  edge->dstPort, dstHw);
          if (distanceCost == 0)
            continue;
          BoolVar pairActive = cpModel.NewBoolVar();
          cpModel.AddImplication(pairActive,
                                 assignmentVars[srcIndex][srcCandIdx]);
          cpModel.AddImplication(pairActive,
                                 assignmentVars[dstIndex][dstCandIdx]);
          cpModel.AddBoolOr({pairActive,
                             assignmentVars[srcIndex][srcCandIdx].Not(),
                             assignmentVars[dstIndex][dstCandIdx].Not()});
          objective += distanceCost * pairActive;
        }
      }
      continue;
    }

    IdIndex fixedNode = srcVariable ? dstPort->parentNode : srcPort->parentNode;
    if (fixedNode >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[fixedNode] == INVALID_ID)
      continue;
    IdIndex fixedHw = state.swNodeToHwNode[fixedNode];
    unsigned varIndex = srcVariable
                            ? nodeToDomainIndex.lookup(srcPort->parentNode)
                            : nodeToDomainIndex.lookup(dstPort->parentNode);
    for (unsigned candIdx = 0; candIdx < assignmentVars[varIndex].size();
         ++candIdx) {
      IdIndex candHw = activeCandidates[varIndex][candIdx];
      int64_t distanceCost =
          srcVariable
              ? edgeWeight * estimatedPlacementDistance(
                                 srcPort->parentNode, edge->srcPort, candHw,
                                 dstPort->parentNode, edge->dstPort, fixedHw)
              : edgeWeight * estimatedPlacementDistance(
                                 srcPort->parentNode, edge->srcPort, fixedHw,
                                 dstPort->parentNode, edge->dstPort, candHw);
      if (distanceCost == 0)
        continue;
      objective += distanceCost * assignmentVars[varIndex][candIdx];
    }
  }
  cpModel.Minimize(objective);

  Model model;
  SatParameters params;
  params.set_max_time_in_seconds(opts.cpSatTimeLimitSeconds);
  unsigned hwConcurrency = std::thread::hardware_concurrency();
  if (hwConcurrency == 0)
    hwConcurrency = 4;
  unsigned laneBudget = opts.lanes == 0 ? 1u : opts.lanes;
  unsigned cpSatWorkers =
      std::max(1u, std::min(8u, hwConcurrency / laneBudget));
  params.set_num_search_workers(cpSatWorkers);
  model.Add(NewSatParameters(params));

  const CpSolverResponse response = SolveCpModel(cpModel.Build(), &model);
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    if (opts.verbose) {
      llvm::outs() << "  CP-SAT solve failed with status "
                   << static_cast<int>(response.status()) << " for";
      for (const auto &domain : domains) {
        llvm::outs() << " [sw " << domain.swNode << " hint=" << domain.hintedHw
                     << " cand=" << domain.hwCandidates.size() << "]";
      }
      llvm::outs() << "\n";
    }
    return false;
  }

  state.restore(baseCheckpoint);
  for (const auto &domain : domains) {
    if (domain.swNode < state.swNodeToHwNode.size() &&
        state.swNodeToHwNode[domain.swNode] != INVALID_ID) {
      state.unmapNode(domain.swNode, dfg, adg);
    }
  }

  for (unsigned domainIndex = 0; domainIndex < domains.size(); ++domainIndex) {
    IdIndex chosenHw = INVALID_ID;
    for (unsigned candIdx = 0; candIdx < assignmentVars[domainIndex].size();
         ++candIdx) {
      if (SolutionBooleanValue(response,
                               assignmentVars[domainIndex][candIdx])) {
        chosenHw = activeCandidates[domainIndex][candIdx];
        break;
      }
    }
    if (chosenHw == INVALID_ID) {
      if (opts.verbose) {
        llvm::outs() << "  CP-SAT chose no hw node for sw "
                     << domains[domainIndex].swNode << "\n";
      }
      return false;
    }
    if (state.mapNode(domains[domainIndex].swNode, chosenHw, dfg, adg) !=
        ActionResult::Success) {
      if (opts.verbose) {
        llvm::outs() << "  CP-SAT mapNode failed for sw "
                     << domains[domainIndex].swNode << " -> hw " << chosenHw
                     << "\n";
      }
      state.restore(baseCheckpoint);
      return false;
    }
  }

  return true;
}

} // namespace
#endif

bool Mapper::runCPSatGlobalPlacement(
    MappingState &state, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    const Options &opts) {
#ifndef FCC_HAVE_ORTOOLS
  (void)state;
  (void)dfg;
  (void)adg;
  (void)flattener;
  (void)candidates;
  (void)opts;
  return false;
#else
  if (!opts.enableCPSat)
    return false;

  std::vector<IdIndex> unplacedNodes;
  for (IdIndex swNode = 0; swNode < static_cast<IdIndex>(dfg.nodes.size());
       ++swNode) {
    const Node *node = dfg.getNode(swNode);
    if (!node || node->kind != Node::OperationNode)
      continue;
    if (swNode < state.swNodeToHwNode.size() &&
        state.swNodeToHwNode[swNode] != INVALID_ID)
      continue;
    unplacedNodes.push_back(swNode);
  }

  if (unplacedNodes.empty() || unplacedNodes.size() > opts.cpSatGlobalNodeLimit)
    return false;

  auto checkpoint = state.save();
  std::vector<PreparedDomain> domains;
  domains.reserve(unplacedNodes.size());
  for (IdIndex swNode : unplacedNodes) {
    auto candIt = candidates.find(swNode);
    if (candIt == candidates.end() || candIt->second.empty()) {
      state.restore(checkpoint);
      return false;
    }

    llvm::SmallVector<std::pair<double, IdIndex>, 16> rankedCandidates;
    for (IdIndex hwNode : candIt->second) {
      double score = scorePlacement(swNode, hwNode, state, dfg, adg, flattener,
                                    candidates);
      rankedCandidates.push_back({-score, hwNode});
    }
    llvm::stable_sort(rankedCandidates, [&](const auto &lhs, const auto &rhs) {
      if (lhs.first != rhs.first)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });

    PreparedDomain domain;
    domain.swNode = swNode;
    unsigned limit = std::min<unsigned>(rankedCandidates.size(), 8);
    for (unsigned idx = 0; idx < limit; ++idx)
      domain.hwCandidates.push_back(rankedCandidates[idx].second);
    if (domain.hwCandidates.empty()) {
      state.restore(checkpoint);
      return false;
    }
    domains.push_back(std::move(domain));
  }

  bool solved = solvePlacementSubset(state, checkpoint, domains, dfg, adg,
                                     flattener, opts);
  if (solved) {
    for (const auto &domain : domains) {
      if (!bindMappedNodePorts(domain.swNode, state, dfg, adg)) {
        state.restore(checkpoint);
        solved = false;
        break;
      }
    }
  }
  if (solved && opts.verbose)
    llvm::outs() << "  CP-SAT global placement solved " << domains.size()
                 << " nodes\n";
  if (!solved && opts.verbose)
    llvm::outs() << "  CP-SAT global placement did not solve " << domains.size()
                 << " nodes\n";
  if (!solved)
    state.restore(checkpoint);
  return solved;
#endif
}

bool Mapper::runCPSatNeighborhoodRepair(
    MappingState &state, const MappingState::Checkpoint &baseCheckpoint,
    llvm::ArrayRef<IdIndex> failedEdges, const Graph &dfg, const Graph &adg,
    const ADGFlattener &flattener,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> &candidates,
    std::vector<TechMappedEdgeKind> &edgeKinds, const Options &opts) {
#ifndef FCC_HAVE_ORTOOLS
  (void)state;
  (void)baseCheckpoint;
  (void)failedEdges;
  (void)dfg;
  (void)adg;
  (void)flattener;
  (void)candidates;
  (void)edgeKinds;
  (void)opts;
  return false;
#else
  if (!opts.enableCPSat || failedEdges.empty())
    return false;

  state.restore(baseCheckpoint);

  llvm::DenseMap<IdIndex, double> hotspotWeights;
  llvm::DenseSet<IdIndex> selectedNodes;
  for (IdIndex edgeId : failedEdges) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;
    double weight = classifyEdgePlacementWeight(dfg, edgeId);
    hotspotWeights[srcPort->parentNode] += weight;
    hotspotWeights[dstPort->parentNode] += weight;
  }

  std::vector<IdIndex> rankedNodes;
  for (const auto &entry : hotspotWeights) {
    if (!isPlacedOperationNode(entry.first, state, dfg))
      continue;
    rankedNodes.push_back(entry.first);
  }
  llvm::stable_sort(rankedNodes, [&](IdIndex lhs, IdIndex rhs) {
    double lhsWeight = hotspotWeights.lookup(lhs);
    double rhsWeight = hotspotWeights.lookup(rhs);
    if (lhsWeight != rhsWeight)
      return lhsWeight > rhsWeight;
    return lhs < rhs;
  });

  const bool tightEndgame = failedEdges.size() <= 6;
  const bool veryTightEndgame = failedEdges.size() <= 2;
  const unsigned targetNeighborhood =
      tightEndgame ? opts.cpSatNeighborhoodNodeLimit
                   : std::min<unsigned>(12u, failedEdges.size() * 2u + 2u);
  const unsigned neighborhoodLimit =
      std::max(1u, std::min(opts.cpSatNeighborhoodNodeLimit, targetNeighborhood));

  std::vector<IdIndex> neighborhood;
  auto maybeAddNeighborhoodNode = [&](IdIndex swNode) {
    if (neighborhood.size() >= neighborhoodLimit)
      return;
    if (!isPlacedOperationNode(swNode, state, dfg))
      return;
    if (selectedNodes.insert(swNode).second)
      neighborhood.push_back(swNode);
  };

  if (tightEndgame) {
    llvm::DenseMap<IdIndex, double> endpointWeights;
    for (IdIndex edgeId : failedEdges) {
      const Edge *edge = dfg.getEdge(edgeId);
      if (!edge)
        continue;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      const Port *dstPort = dfg.getPort(edge->dstPort);
      double weight = classifyEdgePlacementWeight(dfg, edgeId);
      if (srcPort && srcPort->parentNode != INVALID_ID)
        endpointWeights[srcPort->parentNode] += weight;
      if (dstPort && dstPort->parentNode != INVALID_ID)
        endpointWeights[dstPort->parentNode] += weight;
    }

    std::vector<IdIndex> rankedEndpoints;
    rankedEndpoints.reserve(endpointWeights.size());
    for (const auto &entry : endpointWeights) {
      if (isPlacedOperationNode(entry.first, state, dfg))
        rankedEndpoints.push_back(entry.first);
    }
    llvm::stable_sort(rankedEndpoints, [&](IdIndex lhs, IdIndex rhs) {
      double lhsWeight = endpointWeights.lookup(lhs);
      double rhsWeight = endpointWeights.lookup(rhs);
      if (lhsWeight != rhsWeight)
        return lhsWeight > rhsWeight;
      return lhs < rhs;
    });

    for (IdIndex swNode : rankedEndpoints)
      maybeAddNeighborhoodNode(swNode);
  }

  for (IdIndex swNode : rankedNodes) {
    maybeAddNeighborhoodNode(swNode);
    if (neighborhood.size() >= neighborhoodLimit)
      break;
  }

  for (size_t cursor = 0;
       cursor < neighborhood.size() && neighborhood.size() < neighborhoodLimit;
       ++cursor) {
    IdIndex swNode = neighborhood[cursor];
    const Node *node = dfg.getNode(swNode);
    if (!node)
      continue;
    auto maybeAddNeighbor = [&](IdIndex otherSwNode) {
      maybeAddNeighborhoodNode(otherSwNode);
    };
    for (IdIndex inPortId : node->inputPorts) {
      const Port *inPort = dfg.getPort(inPortId);
      if (!inPort)
        continue;
      for (IdIndex edgeId : inPort->connectedEdges) {
        const Edge *edge = dfg.getEdge(edgeId);
        if (!edge || edge->dstPort != inPortId)
          continue;
        const Port *srcPort = dfg.getPort(edge->srcPort);
        if (srcPort && srcPort->parentNode != INVALID_ID)
          maybeAddNeighbor(srcPort->parentNode);
      }
    }
    for (IdIndex outPortId : node->outputPorts) {
      const Port *outPort = dfg.getPort(outPortId);
      if (!outPort)
        continue;
      for (IdIndex edgeId : outPort->connectedEdges) {
        const Edge *edge = dfg.getEdge(edgeId);
        if (!edge || edge->srcPort != outPortId)
          continue;
        const Port *dstPort = dfg.getPort(edge->dstPort);
        if (dstPort && dstPort->parentNode != INVALID_ID)
          maybeAddNeighbor(dstPort->parentNode);
      }
    }
  }

  if (neighborhood.empty())
    return false;

  if (opts.verbose) {
    llvm::outs() << "  CP-SAT neighborhood nodes:";
    for (IdIndex swNode : neighborhood)
      llvm::outs() << " " << swNode;
    llvm::outs() << "\n";
  }

  llvm::DenseSet<IdIndex> neighborhoodSet;
  for (IdIndex swNode : neighborhood)
    neighborhoodSet.insert(swNode);
  llvm::DenseMap<IdIndex, double> focusEdgeWeights;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);
    if (!srcPort || !dstPort || srcPort->parentNode == INVALID_ID ||
        dstPort->parentNode == INVALID_ID)
      continue;
    if (!neighborhoodSet.contains(srcPort->parentNode) &&
        !neighborhoodSet.contains(dstPort->parentNode))
      continue;
    focusEdgeWeights[edgeId] = classifyEdgePlacementWeight(dfg, edgeId);
  }
  for (IdIndex edgeId : failedEdges)
    focusEdgeWeights[edgeId] =
        veryTightEndgame
            ? std::max(80.0, focusEdgeWeights.lookup(edgeId) * 28.0)
            : std::max(40.0, focusEdgeWeights.lookup(edgeId) * 18.0);

  std::vector<PreparedDomain> domains;
  domains.reserve(neighborhood.size());
  for (IdIndex swNode : neighborhood) {
    auto candIt = candidates.find(swNode);
    if (candIt == candidates.end() || candIt->second.empty())
      return false;

    IdIndex oldHw = state.swNodeToHwNode[swNode];
    llvm::SmallVector<std::pair<double, IdIndex>, 16> rankedCandidates;
    const unsigned moveRadius =
        veryTightEndgame
            ? 0u
            : (tightEndgame
            ? std::max(veryTightEndgame ? 5u : 3u,
                       opts.placementMoveRadius + (veryTightEndgame ? 2u : 1u))
            : 0u);
    for (IdIndex hwNode : candIt->second) {
      if (oldHw != INVALID_ID && moveRadius != 0 &&
          manhattanDistance(oldHw, hwNode, flattener) >
              static_cast<int>(moveRadius)) {
        continue;
      }
      double score = scorePlacement(swNode, hwNode, state, dfg, adg, flattener,
                                    candidates);
      rankedCandidates.push_back({-score, hwNode});
    }
    if (rankedCandidates.empty()) {
      for (IdIndex hwNode : candIt->second) {
        double score = scorePlacement(swNode, hwNode, state, dfg, adg,
                                      flattener, candidates);
        rankedCandidates.push_back({-score, hwNode});
      }
    }
    llvm::stable_sort(rankedCandidates, [&](const auto &lhs, const auto &rhs) {
      if (lhs.first != rhs.first)
        return lhs.first < rhs.first;
      return lhs.second < rhs.second;
    });

    PreparedDomain domain;
    domain.swNode = swNode;
    domain.hintedHw = oldHw;
    if (oldHw != INVALID_ID)
      domain.hwCandidates.push_back(oldHw);
    unsigned limit = std::min<unsigned>(
        rankedCandidates.size(),
        veryTightEndgame ? 12u : (tightEndgame ? 5u : 8u));
    for (unsigned idx = 0; idx < limit; ++idx) {
      IdIndex hwNode = rankedCandidates[idx].second;
      if (std::find(domain.hwCandidates.begin(), domain.hwCandidates.end(),
                    hwNode) == domain.hwCandidates.end()) {
        domain.hwCandidates.push_back(hwNode);
      }
    }
    if (domain.hwCandidates.empty()) {
      if (opts.verbose) {
        llvm::outs() << "  CP-SAT empty domain for sw " << swNode
                     << " oldHw=" << oldHw
                     << " ranked=" << rankedCandidates.size() << "\n";
      }
      return false;
    }
    domains.push_back(std::move(domain));
  }

  bool solved = solvePlacementSubset(state, baseCheckpoint, domains, dfg, adg,
                                     flattener, opts, &focusEdgeWeights);
  if (!solved) {
    if (opts.verbose) {
      llvm::outs() << "  CP-SAT neighborhood could not solve " << domains.size()
                   << " nodes\n";
    }
    return false;
  }

  for (const auto &domain : domains) {
    if (!bindMappedNodePorts(domain.swNode, state, dfg, adg)) {
      if (opts.verbose) {
        llvm::outs() << "  CP-SAT neighborhood port rebinding failed for node "
                     << domain.swNode << "\n";
      }
      state.restore(baseCheckpoint);
      return false;
    }
  }
  rebindScalarInputSentinels(state, dfg, adg, flattener);
  bindMemrefSentinels(state, dfg, adg);
  classifyTemporalRegisterEdges(state, dfg, adg, flattener, edgeKinds);
  bool allRouted =
      opts.negotiatedRoutingPasses > 0
          ? runNegotiatedRouting(state, dfg, adg, edgeKinds, opts)
          : runRouting(state, dfg, adg, edgeKinds, opts);
  if (opts.verbose) {
    llvm::outs() << "  CP-SAT neighborhood repaired " << domains.size()
                 << " nodes, routed " << countRoutedEdges(state, dfg, edgeKinds)
                 << "/" << dfg.edges.size() << " edges\n";
  }
  return allRouted;
#endif
}

} // namespace fcc

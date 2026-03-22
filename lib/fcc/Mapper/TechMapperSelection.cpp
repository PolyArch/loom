#include "TechMapperInternal.h"
#include "fcc/Mapper/TypeCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifdef FCC_HAVE_ORTOOLS
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/model.h"
#endif

namespace fcc {

#ifdef FCC_HAVE_ORTOOLS
using namespace operations_research::sat;
#endif

namespace {

std::string serializeConfigFields(llvm::ArrayRef<FUConfigField> fields) {
  llvm::SmallVector<std::string, 4> tokens;
  tokens.reserve(fields.size());
  for (const auto &field : fields) {
    std::string token;
    llvm::raw_string_ostream os(token);
    os << static_cast<unsigned>(field.kind) << ":" << field.opIndex << ":"
       << field.templateOpIndex << ":" << field.opName << ":" << field.bitWidth
       << ":" << field.value << ":" << field.sel << ":" << field.discard << ":"
       << field.disconnect;
    tokens.push_back(os.str());
  }
  std::sort(tokens.begin(), tokens.end());
  std::string text;
  llvm::raw_string_ostream joined(text);
  for (size_t idx = 0; idx < tokens.size(); ++idx) {
    if (idx)
      joined << ";";
    joined << tokens[idx];
  }
  return text;
}

} // namespace

bool sameCandidateDetails(const TechMapper::Candidate &lhs,
                          const TechMapper::Candidate &rhs) {
  return lhs.hwNodeId == rhs.hwNodeId &&
         lhs.supportClassId == rhs.supportClassId &&
         lhs.configClassId == rhs.configClassId &&
         serializeConfigFields(lhs.configFields) ==
             serializeConfigFields(rhs.configFields);
}

void populateFallbackNodeSummary(
    const TechMapper::Plan &plan, IdIndex swNodeId,
    const llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> &nodeInfoBySwNode,
    TechMapper::FallbackNodeInfo &fallbackInfo) {
  fallbackInfo.swNodeId = swNodeId;
  if (const auto *hwNodes =
          TechMapper::findConservativeFallbackCandidates(plan, swNodeId)) {
    fallbackInfo.candidateHwNodeCount = hwNodes->size();
  }
  if (const auto *supportClasses =
          TechMapper::findConservativeFallbackCandidateSupportClasses(
              plan, swNodeId)) {
    fallbackInfo.supportClassCount = supportClasses->size();
    fallbackInfo.supportClassIds.assign(supportClasses->begin(),
                                        supportClasses->end());
    std::sort(fallbackInfo.supportClassIds.begin(),
              fallbackInfo.supportClassIds.end());
  }
  if (const auto *configClasses =
          TechMapper::findConservativeFallbackCandidateConfigClasses(
              plan, swNodeId)) {
    fallbackInfo.configClassCount = configClasses->size();
    fallbackInfo.configClassIds.assign(configClasses->begin(),
                                       configClasses->end());
    std::sort(fallbackInfo.configClassIds.begin(),
              fallbackInfo.configClassIds.end());
  }
  auto nodeInfoIt = nodeInfoBySwNode.find(swNodeId);
  if (nodeInfoIt == nodeInfoBySwNode.end())
    return;
  fallbackInfo.selectionComponentId = nodeInfoIt->second.selectionComponentId;
  fallbackInfo.candidateIds.assign(nodeInfoIt->second.candidateIds.begin(),
                                   nodeInfoIt->second.candidateIds.end());
}

void markNodeAsConservativeFallback(TechMapper::NodeTechInfo &info,
                                    llvm::StringRef status) {
  info.selected = false;
  info.selectedAsFusion = false;
  info.conservativeFallback = true;
  info.status = status.str();
}

void accumulateNodeTechCandidateCoverage(
    llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> &nodeInfoBySwNode,
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        &nodeSupportClassesBySwNode,
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        &nodeConfigClassesBySwNode,
    const AggregatedMatch &aggregated, unsigned candidateId,
    std::optional<unsigned> selectionComponentId) {
  for (IdIndex swNodeId : aggregated.swNodesByOp) {
    auto &info = nodeInfoBySwNode[swNodeId];
    info.swNodeId = swNodeId;
    if (selectionComponentId)
      info.selectionComponentId = *selectionComponentId;
    ++info.candidateCount;
    info.candidateIds.push_back(candidateId);
    info.maxFusionSize =
        std::max<unsigned>(info.maxFusionSize, aggregated.swNodesByOp.size());
    auto &supportClasses = nodeSupportClassesBySwNode[swNodeId];
    if (std::find(supportClasses.begin(), supportClasses.end(),
                  aggregated.supportClassId) == supportClasses.end()) {
      supportClasses.push_back(aggregated.supportClassId);
    }
    auto &configClasses = nodeConfigClassesBySwNode[swNodeId];
    if (std::find(configClasses.begin(), configClasses.end(),
                  aggregated.configClassId) == configClasses.end()) {
      configClasses.push_back(aggregated.configClassId);
    }
  }
}

void markSelectedNodeTechInfo(
    llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> &nodeInfoBySwNode,
    llvm::ArrayRef<IdIndex> swNodeIds, unsigned selectedUnitIndex,
    unsigned selectedCandidateId, bool legacyFallbackOrigin, bool demandOrigin,
    bool mixedOrigin) {
  for (IdIndex swNodeId : swNodeIds) {
    auto infoIt = nodeInfoBySwNode.find(swNodeId);
    if (infoIt == nodeInfoBySwNode.end())
      continue;
    infoIt->second.selectedUnitIndex = selectedUnitIndex;
    infoIt->second.selectedCandidateId = selectedCandidateId;
    infoIt->second.selected = true;
    infoIt->second.selectedAsFusion = swNodeIds.size() > 1;
    infoIt->second.conservativeFallback = false;
    infoIt->second.selectedFromLegacyFallback = legacyFallbackOrigin;
    infoIt->second.selectedFromDemand = demandOrigin;
    infoIt->second.selectedFromMixedOrigin = mixedOrigin;
    infoIt->second.status =
        swNodeIds.size() > 1 ? "selected_fusion" : "selected_single";
  }
}

void finalizeNodeTechCoverageSummaries(
    llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> &nodeInfoBySwNode,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        &nodeSupportClassesBySwNode,
    const llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        &nodeConfigClassesBySwNode) {
  for (auto &entry : nodeInfoBySwNode) {
    IdIndex swNodeId = entry.first;
    auto &info = entry.second;
    std::sort(info.candidateIds.begin(), info.candidateIds.end());
    auto supportIt = nodeSupportClassesBySwNode.find(swNodeId);
    if (supportIt != nodeSupportClassesBySwNode.end()) {
      info.supportClassCount = supportIt->second.size();
      info.supportClassIds.assign(supportIt->second.begin(),
                                  supportIt->second.end());
      std::sort(info.supportClassIds.begin(), info.supportClassIds.end());
    }
    auto configIt = nodeConfigClassesBySwNode.find(swNodeId);
    if (configIt != nodeConfigClassesBySwNode.end()) {
      info.configClassCount = configIt->second.size();
      info.configClassIds.assign(configIt->second.begin(),
                                 configIt->second.end());
      std::sort(info.configClassIds.begin(), info.configClassIds.end());
    }
  }
}

void collectCoveredNodes(const std::vector<AggregatedMatch> &matches,
                         llvm::ArrayRef<unsigned> selectedIndices,
                         llvm::DenseSet<IdIndex> &candidateCoveredNodes,
                         llvm::DenseSet<IdIndex> &selectedCoveredNodes) {
  candidateCoveredNodes.clear();
  selectedCoveredNodes.clear();
  llvm::DenseSet<unsigned> selectedIndexSet;
  for (unsigned selectedIdx : selectedIndices)
    selectedIndexSet.insert(selectedIdx);
  for (unsigned matchIdx = 0; matchIdx < matches.size(); ++matchIdx) {
    for (IdIndex swNodeId : matches[matchIdx].swNodesByOp) {
      candidateCoveredNodes.insert(swNodeId);
      if (selectedIndexSet.contains(matchIdx))
        selectedCoveredNodes.insert(swNodeId);
    }
  }
}

void sortSelectedUnitIndices(TechMapper::Plan &plan) {
  for (auto &componentInfo : TechMapper::allSelectionComponents(plan)) {
    std::sort(componentInfo.selectedUnitIndices.begin(),
              componentInfo.selectedUnitIndices.end());
  }
}

TechMapper::Unit buildSelectedUnitFromAggregatedMatch(
    const AggregatedMatch &aggregated, unsigned selectedCandidateId,
    std::optional<unsigned> selectionComponentId) {
  TechMapper::Unit unit;
  unit.familyIndex = aggregated.familyIndex;
  unit.selectedCandidateId = selectedCandidateId;
  if (selectionComponentId)
    unit.selectionComponentId = *selectionComponentId;
  unit.swNodes = aggregated.swNodesByOp;
  unit.inputBindings = aggregated.inputBindings;
  unit.outputBindings = aggregated.outputBindings;
  unit.internalEdges = aggregated.internalEdges;
  unit.configurable = aggregated.configurable;
  unit.configClassId = aggregated.configClassId;
  unit.selectionScore = aggregated.selectionScore;
  unit.conservativeFallback = false;
  unit.demandOrigin = aggregated.hasDemandOrigin;
  unit.legacyFallbackOrigin =
      aggregated.hasLegacyOrigin && !aggregated.hasDemandOrigin;
  unit.mixedOrigin = aggregated.hasDemandOrigin && aggregated.hasLegacyOrigin;
  for (IdIndex hwNodeId : aggregated.hwNodeIds) {
    TechMapper::Candidate candidate;
    candidate.hwNodeId = hwNodeId;
    candidate.supportClassId = aggregated.supportClassId;
    candidate.configClassId = aggregated.configClassId;
    candidate.temporal = aggregated.temporal;
    candidate.configFields.assign(aggregated.configFields.begin(),
                                  aggregated.configFields.end());
    unit.candidates.push_back(std::move(candidate));
  }
  unit.preferredCandidateIndex = 0;
  return unit;
}

void registerSelectedUnit(
    TechMapper::Plan &plan, const AggregatedMatch &aggregated,
    const TechMapper::Unit &unit, unsigned unitIndex,
    unsigned selectedCandidateId, llvm::DenseSet<unsigned> &selectedConfigClasses,
    llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> &nodeInfoBySwNode) {
  if (auto *candidateSummary =
          TechMapper::findCandidateSummary(plan, selectedCandidateId)) {
    candidateSummary->selectedUnitIndex = unitIndex;
  }
  if (auto *selectionComponent =
          TechMapper::findSelectionComponent(plan, unit.selectionComponentId)) {
    selectionComponent->selectedUnitIndices.push_back(unitIndex);
  }
  if (aggregated.temporal)
    ++plan.metrics.temporalRiskCount;
  plan.metrics.selectedFusedOpCount += aggregated.swNodesByOp.size();
  plan.metrics.selectedInternalEdgeCount += aggregated.internalEdges.size();
  plan.metrics.selectedCandidateChoiceCount += aggregated.hwNodeIds.size();
  selectedConfigClasses.insert(aggregated.configClassId);
  if (unit.legacyFallbackOrigin)
    ++plan.metrics.selectedLegacyFallbackCount;
  if (unit.mixedOrigin)
    ++plan.metrics.selectedMixedOriginCount;
  if (unit.legacyFallbackOrigin || unit.mixedOrigin)
    ++plan.metrics.selectedLegacyDerivedCount;
  if (auto *familyInfo =
          TechMapper::findFamilyTechInfo(plan, aggregated.familyIndex)) {
    ++familyInfo->selectedCount;
  }
  markSelectedNodeTechInfo(nodeInfoBySwNode, aggregated.swNodesByOp, unitIndex,
                           selectedCandidateId, unit.legacyFallbackOrigin,
                           unit.demandOrigin, unit.mixedOrigin);
}

void accumulateLegacyDerivedCandidateMetrics(
    TechMapper::Plan &plan, bool legacyFallbackOrigin, bool mixedOrigin,
    llvm::ArrayRef<IdIndex> hwNodeIds,
    llvm::DenseSet<IdIndex> &legacyDerivedHwNodeIds) {
  if (legacyFallbackOrigin)
    ++plan.metrics.legacyFallbackCandidateCount;
  if (legacyFallbackOrigin || mixedOrigin) {
    ++plan.metrics.legacyContaminatedCandidateCount;
    for (IdIndex hwNodeId : hwNodeIds)
      legacyDerivedHwNodeIds.insert(hwNodeId);
  }
}

void accumulateRejectedCandidateMetrics(TechMapper::Plan &plan,
                                        llvm::StringRef status) {
  if (status == "rejected_overlap")
    ++plan.metrics.rejectedOverlapCandidateCount;
  else if (status == "rejected_temporal_conflict")
    ++plan.metrics.rejectedTemporalCandidateCount;
  else if (status == "rejected_support_capacity")
    ++plan.metrics.rejectedSupportCapacityCandidateCount;
  else if (status == "rejected_spatial_pool")
    ++plan.metrics.rejectedSpatialPoolCandidateCount;
  else if (status == "not_selected_by_objective")
    ++plan.metrics.objectiveDroppedCandidateCount;
}

void applyCandidateSelectionOutcome(TechMapper::Plan &plan,
                                    TechMapper::CandidateSummaryInfo &summary,
                                    bool selected, llvm::StringRef status) {
  summary.selected = selected;
  summary.status = status.str();
  accumulateRejectedCandidateMetrics(plan, status);
}

void markFeedbackFilteredCandidate(TechMapper::Plan &plan,
                                   TechMapper::CandidateSummaryInfo &summary,
                                   llvm::StringRef status) {
  summary.selected = false;
  summary.status = status.str();
  ++plan.metrics.feedbackFilteredCandidateCount;
}

bool preferFallbackCandidate(const TechMapper::Candidate &lhs,
                             const TechMapper::Candidate &rhs,
                             llvm::ArrayRef<TechMapper::SupportClassInfo>
                                 supportClasses);

void accumulateConservativeFallbackCandidate(TechMapper::Plan &plan,
                                             IdIndex swNodeId,
                                             const AggregatedMatch &aggregated) {
  auto &hwNodes = plan.conservativeFallbackCandidates[swNodeId];
  auto &supportClasses =
      plan.conservativeFallbackCandidateSupportClasses[swNodeId];
  auto &configClasses =
      plan.conservativeFallbackCandidateConfigClasses[swNodeId];
  auto &candidateDetails =
      plan.conservativeFallbackCandidateDetails[swNodeId];
  for (IdIndex hwNodeId : aggregated.hwNodeIds) {
    if (std::find(hwNodes.begin(), hwNodes.end(), hwNodeId) == hwNodes.end())
      hwNodes.push_back(hwNodeId);
    TechMapper::Candidate candidate;
    candidate.hwNodeId = hwNodeId;
    candidate.supportClassId = aggregated.supportClassId;
    candidate.configClassId = aggregated.configClassId;
    candidate.temporal = aggregated.temporal;
    candidate.configFields.assign(aggregated.configFields.begin(),
                                  aggregated.configFields.end());
    if (std::find_if(candidateDetails.begin(), candidateDetails.end(),
                     [&](const TechMapper::Candidate &existing) {
                       return sameCandidateDetails(existing, candidate);
                     }) == candidateDetails.end()) {
      candidateDetails.push_back(std::move(candidate));
    }
  }
  if (std::find(supportClasses.begin(), supportClasses.end(),
                aggregated.supportClassId) == supportClasses.end()) {
    supportClasses.push_back(aggregated.supportClassId);
  }
  if (std::find(configClasses.begin(), configClasses.end(),
                aggregated.configClassId) == configClasses.end()) {
    configClasses.push_back(aggregated.configClassId);
  }
}

void rebuildPreferredConservativeFallbackCandidates(TechMapper::Plan &plan) {
  plan.conservativeFallbackPreferredCandidate.clear();
  for (auto &entry : plan.conservativeFallbackCandidateDetails) {
    if (entry.second.empty())
      continue;
    TechMapper::Candidate preferred = entry.second.front();
    for (size_t idx = 1; idx < entry.second.size(); ++idx) {
      if (preferFallbackCandidate(entry.second[idx], preferred,
                                  TechMapper::allSupportClasses(plan))) {
        preferred = entry.second[idx];
      }
    }
    plan.conservativeFallbackPreferredCandidate[entry.first] =
        std::move(preferred);
  }
}

bool preferFallbackCandidate(const TechMapper::Candidate &lhs,
                             const TechMapper::Candidate &rhs,
                             llvm::ArrayRef<TechMapper::SupportClassInfo>
                                 supportClasses) {
  if (lhs.temporal != rhs.temporal)
    return !lhs.temporal;
  auto capacityFor = [&](unsigned supportClassId) -> unsigned {
    if (supportClassId >= supportClasses.size())
      return 0;
    return supportClasses[supportClassId].capacity;
  };
  unsigned lhsCapacity = capacityFor(lhs.supportClassId);
  unsigned rhsCapacity = capacityFor(rhs.supportClassId);
  if (lhsCapacity != rhsCapacity)
    return lhsCapacity > rhsCapacity;
  if (lhs.configFields.size() != rhs.configFields.size())
    return lhs.configFields.size() < rhs.configFields.size();
  if (lhs.configClassId != rhs.configClassId)
    return lhs.configClassId < rhs.configClassId;
  return lhs.hwNodeId < rhs.hwNodeId;
}

std::optional<IdIndex> findForcedTemporalHwNodeId(
    const AggregatedMatch &match) {
  if (!match.temporal || match.hwNodeIds.size() != 1)
    return std::nullopt;
  return match.hwNodeIds.front();
}

bool temporalConflictForced(const AggregatedMatch &lhs, const AggregatedMatch &rhs) {
  auto lhsHwNodeId = findForcedTemporalHwNodeId(lhs);
  auto rhsHwNodeId = findForcedTemporalHwNodeId(rhs);
  if (!lhsHwNodeId || !rhsHwNodeId)
    return false;
  if (*lhsHwNodeId != *rhsHwNodeId)
    return false;
  return lhs.configClassId != rhs.configClassId;
}

bool shareSoftwareCoverage(const AggregatedMatch &lhs, const AggregatedMatch &rhs) {
  for (IdIndex lhsNodeId : lhs.swNodesByOp) {
    for (IdIndex rhsNodeId : rhs.swNodesByOp) {
      if (lhsNodeId == rhsNodeId)
        return true;
    }
  }
  return false;
}

bool shareSpatialHardwarePool(const AggregatedMatch &lhs,
                              const AggregatedMatch &rhs, const Graph &adg) {
  llvm::StringSet<> lhsSpatialPEs;
  llvm::DenseSet<IdIndex> lhsSpatialHwNodes;
  for (IdIndex hwNodeId : lhs.hwNodeIds) {
    const Node *hwNode = adg.getNode(hwNodeId);
    if (!hwNode || getNodeAttrStr(hwNode, "pe_kind") != "spatial_pe")
      continue;
    lhsSpatialHwNodes.insert(hwNodeId);
    llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
    if (!peName.empty())
      lhsSpatialPEs.insert(peName);
  }

  for (IdIndex hwNodeId : rhs.hwNodeIds) {
    const Node *hwNode = adg.getNode(hwNodeId);
    if (!hwNode || getNodeAttrStr(hwNode, "pe_kind") != "spatial_pe")
      continue;
    if (lhsSpatialHwNodes.contains(hwNodeId))
      return true;
    llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
    if (!peName.empty() && lhsSpatialPEs.contains(peName))
      return true;
  }
  return false;
}

bool shareSupportClassConstraint(const AggregatedMatch &lhs,
                                 const AggregatedMatch &rhs) {
  if (lhs.temporal || rhs.temporal)
    return false;
  return lhs.supportClassId != std::numeric_limits<unsigned>::max() &&
         lhs.supportClassId == rhs.supportClassId;
}

unsigned supportClassCapacityForMatches(
    llvm::ArrayRef<AggregatedMatch> matches, unsigned supportClassId) {
  unsigned capacity = 0;
  for (const auto &match : matches) {
    if (match.temporal ||
        match.supportClassId != supportClassId) {
      continue;
    }
    unsigned matchCapacity =
        match.supportClassCapacity != 0 ? match.supportClassCapacity
                                        : match.hwNodeIds.size();
    capacity = std::max<unsigned>(capacity, matchCapacity);
  }
  return capacity;
}

bool supportClassCapacityBlocked(
    unsigned candidateIdx, llvm::ArrayRef<AggregatedMatch> matches,
    llvm::ArrayRef<unsigned> selectedMatches,
    llvm::ArrayRef<unsigned> matchComponentIds) {
  if (candidateIdx >= matches.size())
    return false;
  const auto &candidate = matches[candidateIdx];
  if (candidate.temporal ||
      candidate.supportClassId == std::numeric_limits<unsigned>::max()) {
    return false;
  }
  unsigned capacity =
      supportClassCapacityForMatches(matches, candidate.supportClassId);
  if (capacity == 0)
    return false;

  unsigned selectedCount = 0;
  for (unsigned selectedIdx : selectedMatches) {
    if (selectedIdx >= matches.size())
      continue;
    if (!matchComponentIds.empty() && selectedIdx < matchComponentIds.size() &&
        candidateIdx < matchComponentIds.size() &&
        matchComponentIds[candidateIdx] != matchComponentIds[selectedIdx]) {
      continue;
    }
    if (matches[selectedIdx].temporal)
      continue;
    if (matches[selectedIdx].supportClassId != candidate.supportClassId)
      continue;
    ++selectedCount;
  }
  return selectedCount >= capacity;
}

bool matchesCoupledForSelection(const AggregatedMatch &lhs,
                                const AggregatedMatch &rhs,
                                const Graph &adg) {
  return shareSoftwareCoverage(lhs, rhs) || temporalConflictForced(lhs, rhs) ||
         shareSupportClassConstraint(lhs, rhs) ||
         shareSpatialHardwarePool(lhs, rhs, adg);
}

bool coversSoftwareNode(const AggregatedMatch &match, IdIndex swNodeId) {
  return std::find(match.swNodesByOp.begin(), match.swNodesByOp.end(), swNodeId) !=
         match.swNodesByOp.end();
}

std::string inferRejectedReason(
    IdIndex swNodeId, const std::vector<AggregatedMatch> &matches,
    llvm::ArrayRef<unsigned> selectedMatches,
    llvm::ArrayRef<unsigned> matchComponentIds, const Graph &adg) {
  bool overlap = false;
  bool temporalConflict = false;
  bool supportCapacity = false;
  bool spatialPool = false;
  for (unsigned matchIdx = 0; matchIdx < matches.size(); ++matchIdx) {
    const auto &candidate = matches[matchIdx];
    if (!coversSoftwareNode(candidate, swNodeId))
      continue;
    for (unsigned selectedIdx : selectedMatches) {
      const auto &selected = matches[selectedIdx];
      if (matchIdx < matchComponentIds.size() && selectedIdx < matchComponentIds.size() &&
          matchComponentIds[matchIdx] != matchComponentIds[selectedIdx]) {
        continue;
      }
      if (shareSoftwareCoverage(candidate, selected))
        overlap = true;
      if (temporalConflictForced(candidate, selected))
        temporalConflict = true;
      if (shareSpatialHardwarePool(candidate, selected, adg))
        spatialPool = true;
    }
    if (supportClassCapacityBlocked(matchIdx, matches, selectedMatches,
                                    matchComponentIds)) {
      supportCapacity = true;
    }
  }
  if (overlap)
    return "rejected_overlap";
  if (temporalConflict)
    return "rejected_temporal_conflict";
  if (supportCapacity)
    return "rejected_support_capacity";
  if (spatialPool)
    return "rejected_spatial_pool";
  return "selection_rejected";
}

std::string inferCandidateStatus(unsigned candidateIdx,
                                 const std::vector<AggregatedMatch> &matches,
                                 llvm::ArrayRef<unsigned> selectedMatches,
                                 llvm::ArrayRef<unsigned> matchComponentIds,
                                 const Graph &adg) {
  for (unsigned selectedIdx : selectedMatches) {
    if (candidateIdx == selectedIdx)
      return "selected";
  }
  bool overlap = false;
  bool temporalConflict = false;
  bool supportCapacity = false;
  bool spatialPool = false;
  for (unsigned selectedIdx : selectedMatches) {
    if (candidateIdx < matchComponentIds.size() &&
        selectedIdx < matchComponentIds.size() &&
        matchComponentIds[candidateIdx] != matchComponentIds[selectedIdx]) {
      continue;
    }
    if (shareSoftwareCoverage(matches[candidateIdx], matches[selectedIdx]))
      overlap = true;
    if (temporalConflictForced(matches[candidateIdx], matches[selectedIdx]))
      temporalConflict = true;
    if (shareSpatialHardwarePool(matches[candidateIdx], matches[selectedIdx],
                                 adg)) {
      spatialPool = true;
    }
  }
  if (supportClassCapacityBlocked(candidateIdx, matches, selectedMatches,
                                  matchComponentIds)) {
    supportCapacity = true;
  }
  if (overlap)
    return "rejected_overlap";
  if (temporalConflict)
    return "rejected_temporal_conflict";
  if (supportCapacity)
    return "rejected_support_capacity";
  if (spatialPool)
    return "rejected_spatial_pool";
  return "not_selected_by_objective";
}

std::vector<std::vector<unsigned>>
buildSelectionComponents(const std::vector<AggregatedMatch> &matches,
                         const Graph &adg) {
  std::vector<std::vector<unsigned>> adjacency(matches.size());
  for (unsigned lhs = 0; lhs < matches.size(); ++lhs) {
    for (unsigned rhs = lhs + 1; rhs < matches.size(); ++rhs) {
      if (!matchesCoupledForSelection(matches[lhs], matches[rhs], adg))
        continue;
      adjacency[lhs].push_back(rhs);
      adjacency[rhs].push_back(lhs);
    }
  }

  std::vector<std::vector<unsigned>> components;
  std::vector<uint8_t> visited(matches.size(), 0);
  for (unsigned start = 0; start < matches.size(); ++start) {
    if (visited[start])
      continue;
    std::vector<unsigned> component;
    llvm::SmallVector<unsigned, 16> stack;
    stack.push_back(start);
    visited[start] = 1;
    while (!stack.empty()) {
      unsigned current = stack.pop_back_val();
      component.push_back(current);
      for (unsigned neighbor : adjacency[current]) {
        if (visited[neighbor])
          continue;
        visited[neighbor] = 1;
        stack.push_back(neighbor);
      }
    }
    components.push_back(std::move(component));
  }
  return components;
}

bool greedySelectMatches(const std::vector<AggregatedMatch> &matches,
                         const Graph &adg,
                         std::vector<unsigned> &selectedIndices) {
  selectedIndices.clear();
  std::vector<unsigned> order(matches.size());
  for (unsigned idx = 0; idx < matches.size(); ++idx)
    order[idx] = idx;

  std::sort(order.begin(), order.end(), [&](unsigned lhsIdx, unsigned rhsIdx) {
    const auto &lhs = matches[lhsIdx];
    const auto &rhs = matches[rhsIdx];
    if (lhs.selectionScore != rhs.selectionScore)
      return lhs.selectionScore > rhs.selectionScore;
    if (lhs.swNodesByOp.size() != rhs.swNodesByOp.size())
      return lhs.swNodesByOp.size() > rhs.swNodesByOp.size();
    if (lhs.hwNodeIds.size() != rhs.hwNodeIds.size())
      return lhs.hwNodeIds.size() > rhs.hwNodeIds.size();
    return lhsIdx < rhsIdx;
  });

  llvm::DenseSet<IdIndex> usedSwNodes;
  llvm::DenseSet<IdIndex> usedSpatialHwNodes;
  llvm::StringSet<> usedSpatialPEs;
  llvm::DenseMap<unsigned, unsigned> supportClassUseCount;
  llvm::DenseMap<unsigned, unsigned> supportClassCapacity;

  for (const auto &match : matches) {
    if (match.temporal ||
        match.supportClassId == std::numeric_limits<unsigned>::max()) {
      continue;
    }
    unsigned matchCapacity =
        match.supportClassCapacity != 0 ? match.supportClassCapacity
                                        : match.hwNodeIds.size();
    supportClassCapacity[match.supportClassId] =
        std::max<unsigned>(supportClassCapacity.lookup(match.supportClassId),
                           matchCapacity);
  }

  for (unsigned idx : order) {
    const auto &match = matches[idx];
    bool overlaps = false;
    for (IdIndex swNodeId : match.swNodesByOp) {
      if (usedSwNodes.contains(swNodeId)) {
        overlaps = true;
        break;
      }
    }
    if (overlaps)
      continue;

    bool temporalConflict = false;
    for (unsigned selectedIdx : selectedIndices) {
      if (temporalConflictForced(match, matches[selectedIdx])) {
        temporalConflict = true;
        break;
      }
    }
    if (temporalConflict)
      continue;

    if (!match.temporal &&
        match.supportClassId != std::numeric_limits<unsigned>::max() &&
        supportClassUseCount[match.supportClassId] >=
            supportClassCapacity.lookup(match.supportClassId)) {
      continue;
    }

    bool spatialLegal = true;
    if (!match.temporal) {
      bool foundHw = false;
      for (IdIndex hwNodeId : match.hwNodeIds) {
        const Node *hwNode = adg.getNode(hwNodeId);
        if (!hwNode || getNodeAttrStr(hwNode, "pe_kind") != "spatial_pe")
          continue;
        llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
        if (usedSpatialHwNodes.contains(hwNodeId) ||
            (!peName.empty() && usedSpatialPEs.contains(peName))) {
          continue;
        }
        foundHw = true;
        break;
      }
      spatialLegal = foundHw;
    }
    if (!spatialLegal)
      continue;

    selectedIndices.push_back(idx);
    for (IdIndex swNodeId : match.swNodesByOp)
      usedSwNodes.insert(swNodeId);
    if (!match.temporal &&
        match.supportClassId != std::numeric_limits<unsigned>::max())
      ++supportClassUseCount[match.supportClassId];
    if (!match.temporal) {
      for (IdIndex hwNodeId : match.hwNodeIds) {
        const Node *hwNode = adg.getNode(hwNodeId);
        if (!hwNode || getNodeAttrStr(hwNode, "pe_kind") != "spatial_pe")
          continue;
        llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
        if (usedSpatialHwNodes.contains(hwNodeId) ||
            (!peName.empty() && usedSpatialPEs.contains(peName))) {
          continue;
        }
        usedSpatialHwNodes.insert(hwNodeId);
        if (!peName.empty())
          usedSpatialPEs.insert(peName);
        break;
      }
    }
  }

  return true;
}

bool isBetterExactSelection(const std::vector<unsigned> &candidateSelection,
                            int64_t candidateScore,
                            const std::vector<unsigned> &bestSelection,
                            int64_t bestScore,
                            const std::vector<AggregatedMatch> &matches) {
  if (candidateScore != bestScore)
    return candidateScore > bestScore;

  auto fusedOpCount = [&](const std::vector<unsigned> &selection) -> unsigned {
    unsigned total = 0;
    for (unsigned idx : selection)
      total += matches[idx].swNodesByOp.size();
    return total;
  };
  unsigned candidateFusedOps = fusedOpCount(candidateSelection);
  unsigned bestFusedOps = fusedOpCount(bestSelection);
  if (candidateFusedOps != bestFusedOps)
    return candidateFusedOps > bestFusedOps;
  if (candidateSelection.size() != bestSelection.size())
    return candidateSelection.size() > bestSelection.size();

  std::vector<unsigned> lhs = candidateSelection;
  std::vector<unsigned> rhs = bestSelection;
  std::sort(lhs.begin(), lhs.end());
  std::sort(rhs.begin(), rhs.end());
  return lhs < rhs;
}

bool solveMatchesExactly(const std::vector<AggregatedMatch> &matches,
                         const Graph &adg,
                         std::vector<unsigned> &selectedIndices) {
  selectedIndices.clear();
  if (matches.empty())
    return true;

  struct SpatialChoice {
    IdIndex hwNodeId = INVALID_ID;
    std::string peName;
  };

  std::vector<unsigned> order(matches.size());
  for (unsigned idx = 0; idx < matches.size(); ++idx)
    order[idx] = idx;
  std::sort(order.begin(), order.end(), [&](unsigned lhsIdx, unsigned rhsIdx) {
    const auto &lhs = matches[lhsIdx];
    const auto &rhs = matches[rhsIdx];
    if (lhs.selectionScore != rhs.selectionScore)
      return lhs.selectionScore > rhs.selectionScore;
    if (lhs.swNodesByOp.size() != rhs.swNodesByOp.size())
      return lhs.swNodesByOp.size() > rhs.swNodesByOp.size();
    if (lhs.hwNodeIds.size() != rhs.hwNodeIds.size())
      return lhs.hwNodeIds.size() < rhs.hwNodeIds.size();
    return lhsIdx < rhsIdx;
  });

  std::vector<int64_t> optimisticSuffix(order.size() + 1, 0);
  for (int idx = static_cast<int>(order.size()) - 1; idx >= 0; --idx) {
    int64_t optimistic = matches[order[idx]].selectionScore;
    if (optimistic < 0)
      optimistic = 0;
    optimisticSuffix[idx] = optimisticSuffix[idx + 1] + optimistic;
  }

  llvm::DenseMap<unsigned, unsigned> supportClassCapacity;
  std::vector<std::vector<SpatialChoice>> spatialChoices(matches.size());
  for (unsigned idx = 0; idx < matches.size(); ++idx) {
    const auto &match = matches[idx];
    if (!match.temporal &&
        match.supportClassId != std::numeric_limits<unsigned>::max()) {
      unsigned matchCapacity =
          match.supportClassCapacity != 0 ? match.supportClassCapacity
                                          : match.hwNodeIds.size();
      supportClassCapacity[match.supportClassId] =
          std::max<unsigned>(supportClassCapacity.lookup(match.supportClassId),
                             matchCapacity);
    }
    if (match.temporal)
      continue;
    for (IdIndex hwNodeId : match.hwNodeIds) {
      const Node *hwNode = adg.getNode(hwNodeId);
      if (!hwNode || getNodeAttrStr(hwNode, "pe_kind") != "spatial_pe")
        continue;
      SpatialChoice choice;
      choice.hwNodeId = hwNodeId;
      choice.peName = getNodeAttrStr(hwNode, "pe_name").str();
      spatialChoices[idx].push_back(std::move(choice));
    }
    std::sort(spatialChoices[idx].begin(), spatialChoices[idx].end(),
              [](const SpatialChoice &lhs, const SpatialChoice &rhs) {
                if (lhs.peName != rhs.peName)
                  return lhs.peName < rhs.peName;
                return lhs.hwNodeId < rhs.hwNodeId;
              });
  }

  llvm::DenseSet<IdIndex> usedSwNodes;
  llvm::DenseSet<IdIndex> usedSpatialHwNodes;
  llvm::StringSet<> usedSpatialPEs;
  llvm::DenseMap<unsigned, unsigned> supportClassUseCount;
  llvm::DenseMap<IdIndex, unsigned> temporalConfigByHwNode;
  std::vector<unsigned> currentSelection;
  std::vector<unsigned> bestSelection;
  int64_t currentScore = 0;
  int64_t bestScore = std::numeric_limits<int64_t>::min();

  std::function<void(size_t)> search = [&](size_t orderIdx) {
    if (orderIdx == order.size()) {
      if (bestSelection.empty() ||
          isBetterExactSelection(currentSelection, currentScore, bestSelection,
                                 bestScore, matches)) {
        bestSelection = currentSelection;
        bestScore = currentScore;
      }
      return;
    }

    if (bestScore != std::numeric_limits<int64_t>::min() &&
        currentScore + optimisticSuffix[orderIdx] < bestScore) {
      return;
    }

    const unsigned matchIdx = order[orderIdx];
    const auto &match = matches[matchIdx];

    bool swOverlap = false;
    for (IdIndex swNodeId : match.swNodesByOp) {
      if (usedSwNodes.contains(swNodeId)) {
        swOverlap = true;
        break;
      }
    }

    bool temporalConflict = false;
    auto forcedTemporalHwNodeId = findForcedTemporalHwNodeId(match);
    if (!swOverlap && forcedTemporalHwNodeId) {
      IdIndex hwNodeId = *forcedTemporalHwNodeId;
      auto found = temporalConfigByHwNode.find(hwNodeId);
      if (found != temporalConfigByHwNode.end() &&
          found->second != match.configClassId) {
        temporalConflict = true;
      }
    }

    unsigned capacity = 0;
    bool supportCapacityConflict = false;
    if (!swOverlap && !temporalConflict && !match.temporal &&
        match.supportClassId != std::numeric_limits<unsigned>::max()) {
      capacity = supportClassCapacity.lookup(match.supportClassId);
      if (capacity == 0) {
        capacity = match.supportClassCapacity != 0 ? match.supportClassCapacity
                                                   : match.hwNodeIds.size();
      }
      if (supportClassUseCount[match.supportClassId] >= capacity)
        supportCapacityConflict = true;
    }

    if (!swOverlap && !temporalConflict && !supportCapacityConflict) {
      if (match.temporal) {
        for (IdIndex swNodeId : match.swNodesByOp)
          usedSwNodes.insert(swNodeId);
        bool insertedTemporalBinding = false;
        if (forcedTemporalHwNodeId) {
          temporalConfigByHwNode[*forcedTemporalHwNodeId] = match.configClassId;
          insertedTemporalBinding = true;
        }
        currentSelection.push_back(matchIdx);
        currentScore += match.selectionScore;
        search(orderIdx + 1);
        currentScore -= match.selectionScore;
        currentSelection.pop_back();
        if (insertedTemporalBinding)
          temporalConfigByHwNode.erase(*forcedTemporalHwNodeId);
        for (IdIndex swNodeId : match.swNodesByOp)
          usedSwNodes.erase(swNodeId);
      } else {
        for (const auto &choice : spatialChoices[matchIdx]) {
          if (usedSpatialHwNodes.contains(choice.hwNodeId))
            continue;
          if (!choice.peName.empty() && usedSpatialPEs.contains(choice.peName))
            continue;

          for (IdIndex swNodeId : match.swNodesByOp)
            usedSwNodes.insert(swNodeId);
          usedSpatialHwNodes.insert(choice.hwNodeId);
          if (!choice.peName.empty())
            usedSpatialPEs.insert(choice.peName);
          if (match.supportClassId != std::numeric_limits<unsigned>::max())
            ++supportClassUseCount[match.supportClassId];
          currentSelection.push_back(matchIdx);
          currentScore += match.selectionScore;
          search(orderIdx + 1);
          currentScore -= match.selectionScore;
          currentSelection.pop_back();
          if (match.supportClassId != std::numeric_limits<unsigned>::max()) {
            auto found = supportClassUseCount.find(match.supportClassId);
            if (found != supportClassUseCount.end()) {
              if (found->second > 1)
                --found->second;
              else
                supportClassUseCount.erase(found);
            }
          }
          if (!choice.peName.empty())
            usedSpatialPEs.erase(choice.peName);
          usedSpatialHwNodes.erase(choice.hwNodeId);
          for (IdIndex swNodeId : match.swNodesByOp)
            usedSwNodes.erase(swNodeId);
        }
      }
    }

    search(orderIdx + 1);
  };

  search(0);
  selectedIndices = bestSelection;
  std::sort(selectedIndices.begin(), selectedIndices.end());
  return true;
}

#ifdef FCC_HAVE_ORTOOLS
bool solveMatchesWithCpSat(const std::vector<AggregatedMatch> &matches,
                           const Graph &adg,
                           std::vector<unsigned> &selectedIndices);
#endif

bool selectMatchesByComponent(const std::vector<AggregatedMatch> &matches,
                              const Graph &adg,
                              llvm::SmallVectorImpl<TechMapper::SelectionComponentInfo>
                                  *componentInfos,
                              std::vector<unsigned> *matchComponentIds,
                              TechMapper::PlanMetrics *metrics,
                              std::vector<unsigned> &selectedIndices) {
  selectedIndices.clear();
  auto components = buildSelectionComponents(matches, adg);
  if (metrics)
    metrics->selectionComponentCount = components.size();
  if (componentInfos)
    componentInfos->clear();
  if (matchComponentIds)
    matchComponentIds->assign(matches.size(), std::numeric_limits<unsigned>::max());
  for (size_t componentIdx = 0; componentIdx < components.size(); ++componentIdx) {
    const auto &component = components[componentIdx];
    std::vector<AggregatedMatch> localMatches;
    localMatches.reserve(component.size());
    for (unsigned globalIdx : component)
      localMatches.push_back(matches[globalIdx]);

    std::vector<unsigned> localSelected;
    std::string solver = "greedy";
    constexpr size_t kMaxTechmapExactCandidates = 24;
    bool usedExact = false;
    if (localMatches.size() <= kMaxTechmapExactCandidates)
      usedExact = solveMatchesExactly(localMatches, adg, localSelected);
#ifdef FCC_HAVE_ORTOOLS
    constexpr size_t kMaxTechmapCpSatCandidates = 256;
    bool usedCpSat = false;
    if (!usedExact && localMatches.size() <= kMaxTechmapCpSatCandidates)
      usedCpSat = solveMatchesWithCpSat(localMatches, adg, localSelected);
    if (usedExact) {
      if (metrics)
        ++metrics->exactComponentCount;
      solver = "exact";
    } else if (usedCpSat) {
      if (metrics)
        ++metrics->cpSatComponentCount;
      solver = "cpsat";
    } else {
      greedySelectMatches(localMatches, adg, localSelected);
      if (metrics)
        ++metrics->greedyComponentCount;
    }
#else
    if (usedExact) {
      if (metrics)
        ++metrics->exactComponentCount;
      solver = "exact";
    } else {
      greedySelectMatches(localMatches, adg, localSelected);
      if (metrics)
        ++metrics->greedyComponentCount;
    }
#endif
    if (componentInfos) {
      llvm::DenseSet<IdIndex> swNodes;
      bool containsTemporalCandidate = false;
      int64_t baseMaxCandidateScore = std::numeric_limits<int64_t>::min();
      int64_t maxCandidateScore = std::numeric_limits<int64_t>::min();
      int64_t baseSelectedScoreSum = 0;
      int64_t selectedScoreSum = 0;
      for (const auto &match : localMatches) {
        containsTemporalCandidate =
            containsTemporalCandidate || match.temporal;
        baseMaxCandidateScore =
            std::max<int64_t>(baseMaxCandidateScore, match.selectionScore);
        maxCandidateScore =
            std::max<int64_t>(maxCandidateScore, match.selectionScore);
        for (IdIndex swNodeId : match.swNodesByOp)
          swNodes.insert(swNodeId);
      }
      TechMapper::SelectionComponentInfo info;
      info.id = componentIdx;
      info.candidateCount = localMatches.size();
      info.swNodeCount = swNodes.size();
      info.selectedCount = localSelected.size();
      info.containsTemporalCandidate = containsTemporalCandidate;
      info.baseMaxCandidateScore =
          baseMaxCandidateScore == std::numeric_limits<int64_t>::min()
              ? 0
              : baseMaxCandidateScore;
      info.maxCandidateScore =
          maxCandidateScore == std::numeric_limits<int64_t>::min()
              ? 0
              : maxCandidateScore;
      info.solver = solver;
      info.candidateIds.assign(component.begin(), component.end());
      std::sort(info.candidateIds.begin(), info.candidateIds.end());
      for (IdIndex swNodeId : swNodes)
        info.swNodeIds.push_back(swNodeId);
      std::sort(info.swNodeIds.begin(), info.swNodeIds.end());
      for (unsigned localIdx : localSelected) {
        info.selectedCandidateIds.push_back(component[localIdx]);
        baseSelectedScoreSum += localMatches[localIdx].selectionScore;
        selectedScoreSum += localMatches[localIdx].selectionScore;
      }
      std::sort(info.selectedCandidateIds.begin(),
                info.selectedCandidateIds.end());
      info.baseSelectedScoreSum = baseSelectedScoreSum;
      info.selectedScoreSum = selectedScoreSum;
      componentInfos->push_back(std::move(info));
    }
    unsigned componentId = componentIdx;
    if (matchComponentIds) {
      for (unsigned globalIdx : component)
        (*matchComponentIds)[globalIdx] = componentId;
    }
    for (unsigned localIdx : localSelected)
      selectedIndices.push_back(component[localIdx]);
  }
  std::sort(selectedIndices.begin(), selectedIndices.end());
  return true;
}

bool selectMatchesByCachedComponents(
    const std::vector<AggregatedMatch> &matches,
    llvm::ArrayRef<unsigned> candidateIds,
    llvm::ArrayRef<unsigned> cachedComponentIds, unsigned baseComponentCount,
    const Graph &adg,
    llvm::SmallVectorImpl<TechMapper::SelectionComponentInfo> *componentInfos,
    std::vector<unsigned> *matchComponentIds, TechMapper::PlanMetrics *metrics,
    std::vector<unsigned> &selectedIndices) {
  if (matches.size() != candidateIds.size() ||
      matches.size() != cachedComponentIds.size()) {
    return selectMatchesByComponent(matches, adg, componentInfos,
                                    matchComponentIds, metrics,
                                    selectedIndices);
  }

  selectedIndices.clear();
  if (matchComponentIds)
    matchComponentIds->assign(matches.size(), std::numeric_limits<unsigned>::max());

  std::map<unsigned, std::vector<unsigned>> componentMembers;
  unsigned nextSyntheticComponentId = baseComponentCount;
  for (unsigned matchIdx = 0; matchIdx < matches.size(); ++matchIdx) {
    unsigned componentId = cachedComponentIds[matchIdx];
    if (componentId == std::numeric_limits<unsigned>::max())
      componentId = nextSyntheticComponentId++;
    componentMembers[componentId].push_back(matchIdx);
    if (matchComponentIds)
      (*matchComponentIds)[matchIdx] = componentId;
  }

  if (metrics)
    metrics->selectionComponentCount = nextSyntheticComponentId;
  if (componentInfos) {
    componentInfos->clear();
    componentInfos->resize(nextSyntheticComponentId);
    for (unsigned componentId = 0; componentId < nextSyntheticComponentId;
         ++componentId) {
      auto &info = (*componentInfos)[componentId];
      info.id = componentId;
      info.solver = "filtered_empty";
      info.baseMaxCandidateScore = 0;
      info.baseSelectedScoreSum = 0;
      info.filteredCandidateIds.clear();
    }
  }

  for (unsigned componentId = 0; componentId < nextSyntheticComponentId;
       ++componentId) {
    auto memberIt = componentMembers.find(componentId);
    if (memberIt == componentMembers.end())
      continue;
    const auto &memberIndices = memberIt->second;
    std::vector<AggregatedMatch> localMatches;
    localMatches.reserve(memberIndices.size());
    for (unsigned matchIdx : memberIndices)
      localMatches.push_back(matches[matchIdx]);

    std::vector<unsigned> localSelected;
    std::string solver = "greedy";
    constexpr size_t kMaxTechmapExactCandidates = 24;
    bool usedExact = false;
    if (localMatches.size() <= kMaxTechmapExactCandidates)
      usedExact = solveMatchesExactly(localMatches, adg, localSelected);
#ifdef FCC_HAVE_ORTOOLS
    constexpr size_t kMaxTechmapCpSatCandidates = 256;
    bool usedCpSat = false;
    if (!usedExact && localMatches.size() <= kMaxTechmapCpSatCandidates)
      usedCpSat = solveMatchesWithCpSat(localMatches, adg, localSelected);
    if (usedExact) {
      if (metrics)
        ++metrics->exactComponentCount;
      solver = "exact";
    } else if (usedCpSat) {
      if (metrics)
        ++metrics->cpSatComponentCount;
      solver = "cpsat";
    } else {
      greedySelectMatches(localMatches, adg, localSelected);
      if (metrics)
        ++metrics->greedyComponentCount;
    }
#else
    if (usedExact) {
      if (metrics)
        ++metrics->exactComponentCount;
      solver = "exact";
    } else {
      greedySelectMatches(localMatches, adg, localSelected);
      if (metrics)
        ++metrics->greedyComponentCount;
    }
#endif

    if (componentInfos) {
      llvm::DenseSet<IdIndex> swNodes;
      bool containsTemporalCandidate = false;
      int64_t maxCandidateScore = std::numeric_limits<int64_t>::min();
      int64_t selectedScoreSum = 0;
      auto &info = (*componentInfos)[componentId];
      info.candidateCount = localMatches.size();
      info.selectedCount = localSelected.size();
      info.solver = solver;
      info.candidateIds.clear();
      info.filteredCandidateIds.clear();
      info.selectedCandidateIds.clear();
      info.selectedUnitIndices.clear();
      info.swNodeIds.clear();
      for (unsigned localIdx = 0; localIdx < memberIndices.size(); ++localIdx) {
        unsigned globalIdx = memberIndices[localIdx];
        info.candidateIds.push_back(candidateIds[globalIdx]);
        const auto &match = localMatches[localIdx];
        containsTemporalCandidate =
            containsTemporalCandidate || match.temporal;
        maxCandidateScore =
            std::max<int64_t>(maxCandidateScore, match.selectionScore);
        for (IdIndex swNodeId : match.swNodesByOp)
          swNodes.insert(swNodeId);
      }
      for (IdIndex swNodeId : swNodes)
        info.swNodeIds.push_back(swNodeId);
      std::sort(info.candidateIds.begin(), info.candidateIds.end());
      std::sort(info.swNodeIds.begin(), info.swNodeIds.end());
      for (unsigned localIdx : localSelected) {
        unsigned globalIdx = memberIndices[localIdx];
        info.selectedCandidateIds.push_back(candidateIds[globalIdx]);
        selectedScoreSum += localMatches[localIdx].selectionScore;
      }
      std::sort(info.selectedCandidateIds.begin(),
                info.selectedCandidateIds.end());
      info.swNodeCount = swNodes.size();
      info.containsTemporalCandidate = containsTemporalCandidate;
      info.baseMaxCandidateScore = 0;
      info.maxCandidateScore =
          maxCandidateScore == std::numeric_limits<int64_t>::min()
              ? 0
              : maxCandidateScore;
      info.baseSelectedScoreSum = 0;
      info.selectedScoreSum = selectedScoreSum;
    }

    for (unsigned localIdx : localSelected)
      selectedIndices.push_back(memberIndices[localIdx]);
  }

  std::sort(selectedIndices.begin(), selectedIndices.end());
  return true;
}

#ifdef FCC_HAVE_ORTOOLS
bool solveMatchesWithCpSat(const std::vector<AggregatedMatch> &matches,
                          const Graph &adg,
                          std::vector<unsigned> &selectedIndices) {
  selectedIndices.clear();
  if (matches.empty())
    return true;

  CpModelBuilder cpModel;
  std::vector<BoolVar> useVars;
  useVars.reserve(matches.size());
  for (size_t idx = 0; idx < matches.size(); ++idx)
    useVars.push_back(cpModel.NewBoolVar());

  llvm::DenseMap<IdIndex, std::vector<int>> candidateIndicesBySwNode;
  for (unsigned idx = 0; idx < matches.size(); ++idx) {
    for (IdIndex swNodeId : matches[idx].swNodesByOp)
      candidateIndicesBySwNode[swNodeId].push_back(idx);
  }
  for (const auto &entry : candidateIndicesBySwNode) {
    LinearExpr expr;
    for (int idx : entry.second)
      expr += useVars[idx];
    cpModel.AddLessOrEqual(expr, 1);
  }

  llvm::DenseMap<IdIndex, std::vector<BoolVar>> varsBySpatialHwNode;
  llvm::StringMap<std::vector<BoolVar>> varsBySpatialPE;
  llvm::DenseMap<unsigned, std::vector<int>> candidateIndicesBySupportClass;
  llvm::DenseMap<unsigned, int64_t> supportClassCapacity;
  for (unsigned idx = 0; idx < matches.size(); ++idx) {
    const auto &match = matches[idx];
    if (!match.temporal &&
        match.supportClassId != std::numeric_limits<unsigned>::max()) {
      candidateIndicesBySupportClass[match.supportClassId].push_back(idx);
      int64_t matchCapacity =
          match.supportClassCapacity != 0
              ? static_cast<int64_t>(match.supportClassCapacity)
              : static_cast<int64_t>(match.hwNodeIds.size());
      supportClassCapacity[match.supportClassId] =
          std::max<int64_t>(supportClassCapacity.lookup(match.supportClassId),
                            matchCapacity);
    }
    if (match.temporal)
      continue;
    std::vector<BoolVar> assignVars;
    for (IdIndex hwNodeId : match.hwNodeIds) {
      const Node *hwNode = adg.getNode(hwNodeId);
      if (!hwNode || getNodeAttrStr(hwNode, "pe_kind") != "spatial_pe")
        continue;
      BoolVar assignVar = cpModel.NewBoolVar();
      assignVars.push_back(assignVar);
      varsBySpatialHwNode[hwNodeId].push_back(assignVar);
      llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
      if (!peName.empty())
        varsBySpatialPE[peName].push_back(assignVar);
    }
    if (assignVars.empty()) {
      cpModel.AddEquality(useVars[idx], 0);
      continue;
    }
    LinearExpr sumAssign;
    for (BoolVar assignVar : assignVars)
      sumAssign += assignVar;
    cpModel.AddEquality(sumAssign, useVars[idx]);
  }

  for (const auto &entry : varsBySpatialHwNode) {
    LinearExpr expr;
    for (BoolVar var : entry.second)
      expr += var;
    cpModel.AddLessOrEqual(expr, 1);
  }
  for (const auto &entry : varsBySpatialPE) {
    LinearExpr expr;
    for (BoolVar var : entry.second)
      expr += var;
    cpModel.AddLessOrEqual(expr, 1);
  }
  for (const auto &entry : candidateIndicesBySupportClass) {
    LinearExpr expr;
    for (int idx : entry.second)
      expr += useVars[idx];
    cpModel.AddLessOrEqual(expr, supportClassCapacity.lookup(entry.first));
  }

  for (unsigned lhs = 0; lhs < matches.size(); ++lhs) {
    for (unsigned rhs = lhs + 1; rhs < matches.size(); ++rhs) {
      if (!temporalConflictForced(matches[lhs], matches[rhs]))
        continue;
      cpModel.AddLessOrEqual(useVars[lhs] + useVars[rhs], 1);
    }
  }

  LinearExpr objective;
  for (unsigned idx = 0; idx < matches.size(); ++idx)
    objective += useVars[idx] * matches[idx].selectionScore;
  cpModel.Maximize(objective);

  Model model;
  const CpSolverResponse response = SolveCpModel(cpModel.Build(), &model);
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE) {
    return false;
  }

  for (unsigned idx = 0; idx < matches.size(); ++idx) {
    if (SolutionIntegerValue(response, useVars[idx]) != 0)
      selectedIndices.push_back(idx);
  }
  return true;
}
#endif

AggregatedMatch buildAggregatedMatchFromSummary(
    const TechMapper::CandidateSummaryInfo &summary) {
  AggregatedMatch match;
  match.familyIndex = summary.familyIndex;
  match.swNodesByOp.assign(summary.swNodeIds.begin(), summary.swNodeIds.end());
  match.inputBindings.assign(summary.inputBindings.begin(),
                             summary.inputBindings.end());
  match.outputBindings.assign(summary.outputBindings.begin(),
                              summary.outputBindings.end());
  match.internalEdges.assign(summary.internalEdgeIds.begin(),
                             summary.internalEdgeIds.end());
  match.configFields.assign(summary.configFields.begin(),
                            summary.configFields.end());
  match.hwNodeIds.assign(summary.hwNodeIds.begin(), summary.hwNodeIds.end());
  match.configurable = summary.configurable;
  match.temporal = summary.temporal;
  match.supportClassId = summary.supportClassId;
  match.supportClassCapacity = summary.supportClassCapacity;
  match.configClassId = summary.configClassId;
  match.selectionScore = summary.selectionScore;
  match.hasDemandOrigin = summary.demandOrigin || summary.mixedOrigin;
  match.hasLegacyOrigin =
      summary.legacyFallbackOrigin || summary.mixedOrigin;
  return match;
}

int64_t lookupPenalty(llvm::ArrayRef<TechMapper::WeightedIdPenalty> penalties,
                      unsigned id) {
  int64_t totalPenalty = 0;
  for (const auto &entry : penalties) {
    if (entry.id == id)
      totalPenalty += entry.penalty;
  }
  return totalPenalty;
}

void resetSelectionMetrics(TechMapper::PlanMetrics &metrics) {
  metrics.coverageScore = 1.0;
  metrics.candidateGenerationTimeMicros = 0;
  metrics.selectionTimeMicros = 0;
  metrics.totalLayer2TimeMicros = 0;
  metrics.opAliasPairCount = 0;
  metrics.demandCandidateCount = 0;
  metrics.structuralStateCount = 0;
  metrics.structuralStateCacheHitCount = 0;
  metrics.structuralStateCacheMissCount = 0;
  metrics.selectedCandidateCount = 0;
  metrics.rejectedOverlapCandidateCount = 0;
  metrics.rejectedTemporalCandidateCount = 0;
  metrics.rejectedSupportCapacityCandidateCount = 0;
  metrics.rejectedSpatialPoolCandidateCount = 0;
  metrics.objectiveDroppedCandidateCount = 0;
  metrics.conservativeFallbackCount = 0;
  metrics.overlapEdgeCount = 0;
  metrics.supportClassCount = 0;
  metrics.configClassCount = 0;
  metrics.temporalRiskCount = 0;
  metrics.selectedFusedOpCount = 0;
  metrics.selectedInternalEdgeCount = 0;
  metrics.selectedCandidateChoiceCount = 0;
  metrics.selectedConfigDiversityCount = 0;
  metrics.selectedLegacyFallbackCount = 0;
  metrics.selectedMixedOriginCount = 0;
  metrics.selectedLegacyDerivedCount = 0;
  metrics.selectionComponentCount = 0;
  metrics.exactComponentCount = 0;
  metrics.cpSatComponentCount = 0;
  metrics.greedyComponentCount = 0;
  metrics.fallbackNoCandidateCount = 0;
  metrics.fallbackRejectedCount = 0;
  metrics.conservativeFallbackCoveredCount = 0;
  metrics.conservativeFallbackMissingCount = 0;
  metrics.legacyOracleEnabled = false;
  metrics.legacyOracleRequired = false;
  metrics.legacyOracleCheckCount = 0;
  metrics.legacyOracleCandidateCount = 0;
  metrics.legacyOracleMissingCount = 0;
  metrics.legacyFallbackCount = 0;
  metrics.legacyFallbackCandidateCount = 0;
  metrics.legacyContaminatedCandidateCount = 0;
  metrics.legacyDerivedSourceCount = 0;
  metrics.feedbackReselectionCount = 0;
  metrics.feedbackFilteredCandidateCount = 0;
  metrics.feedbackPenaltyCount = 0;
  metrics.feedbackUnknownCandidateRefCount = 0;
  metrics.feedbackUnknownFamilyRefCount = 0;
  metrics.feedbackUnknownConfigClassRefCount = 0;
}

void restoreReselectionBaselineMetrics(TechMapper::PlanMetrics &metrics,
                                       const TechMapper::PlanMetrics &seed) {
  metrics.candidateGenerationTimeMicros = seed.candidateGenerationTimeMicros;
  metrics.opAliasPairCount = seed.opAliasPairCount;
  metrics.demandCandidateCount = seed.demandCandidateCount;
  metrics.structuralStateCount = seed.structuralStateCount;
  metrics.structuralStateCacheHitCount = seed.structuralStateCacheHitCount;
  metrics.structuralStateCacheMissCount = seed.structuralStateCacheMissCount;
  metrics.overlapEdgeCount = seed.overlapEdgeCount;
  metrics.supportClassCount = seed.supportClassCount;
  metrics.configClassCount = seed.configClassCount;
  metrics.legacyOracleEnabled = seed.legacyOracleEnabled;
  metrics.legacyOracleRequired = seed.legacyOracleRequired;
  metrics.legacyOracleCheckCount = seed.legacyOracleCheckCount;
  metrics.legacyOracleCandidateCount = seed.legacyOracleCandidateCount;
  metrics.legacyOracleMissingCount = seed.legacyOracleMissingCount;
}

} // namespace fcc

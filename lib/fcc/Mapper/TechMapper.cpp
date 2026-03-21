#include "TechMapperInternal.h"
#include "fcc/Mapper/OpCompat.h"
#include "fcc/Mapper/TypeCompat.h"

#include "fcc/Dialect/Fabric/FabricOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

using techmapper_detail::FamilyMatch;
using techmapper_detail::findFunctionUnitNode;
using techmapper_detail::collectVariantsForFU;
using techmapper_detail::findDemandDrivenMatchesForFU;
using techmapper_detail::findMatchesForFamily;
using techmapper_detail::Match;
using techmapper_detail::VariantFamily;

#ifdef FCC_HAVE_ORTOOLS
using namespace operations_research::sat;
#endif

namespace {

void addNodeAttr(Node *node, llvm::StringRef key, mlir::Attribute value,
                 mlir::MLIRContext *ctx) {
  node->attributes.push_back(
      mlir::NamedAttribute(mlir::StringAttr::get(ctx, key), value));
}

void addUIntNodeAttr(Node *node, llvm::StringRef key, uint64_t value,
                     mlir::MLIRContext *ctx) {
  addNodeAttr(node, key,
              mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), value),
              ctx);
}

void addBoolNodeAttr(Node *node, llvm::StringRef key, bool value,
                     mlir::MLIRContext *ctx) {
  addNodeAttr(node, key, mlir::BoolAttr::get(ctx, value), ctx);
}

std::unique_ptr<Node> cloneNodeShell(const Node *src, mlir::MLIRContext *ctx) {
  auto node = std::make_unique<Node>();
  node->kind = src->kind;
  node->attributes = src->attributes;
  (void)ctx;
  return node;
}

std::unique_ptr<Port> clonePort(const Port *src) {
  auto port = std::make_unique<Port>();
  port->direction = src->direction;
  port->type = src->type;
  port->attributes = src->attributes;
  return port;
}

struct AggregatedMatch {
  unsigned familyIndex = 0;
  llvm::SmallVector<IdIndex, 4> swNodesByOp;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 4> operandOrderByOp;
  llvm::SmallVector<TechMapper::PortBinding, 4> inputBindings;
  llvm::SmallVector<TechMapper::PortBinding, 4> outputBindings;
  llvm::SmallVector<IdIndex, 4> internalEdges;
  llvm::SmallVector<FUConfigField, 4> configFields;
  llvm::SmallVector<IdIndex, 4> hwNodeIds;
  bool configurable = false;
  bool temporal = false;
  unsigned supportClassId = std::numeric_limits<unsigned>::max();
  unsigned supportClassCapacity = 0;
  unsigned configClassId = std::numeric_limits<unsigned>::max();
  int64_t selectionScore = 0;
  unsigned commutativeSwapCount = 0;
  bool hasDemandOrigin = false;
  bool hasLegacyOrigin = false;
};

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

llvm::SmallVector<mlir::Attribute, 4>
buildConfigFieldSummaryAttrs(llvm::ArrayRef<FUConfigField> fields,
                             mlir::MLIRContext *ctx) {
  llvm::SmallVector<std::string, 4> tokens;
  tokens.reserve(fields.size());
  for (const auto &field : fields) {
    std::string summary;
    llvm::raw_string_ostream os(summary);
    os << static_cast<unsigned>(field.kind) << ":" << field.opIndex << ":"
       << field.templateOpIndex << ":" << field.opName << ":" << field.bitWidth
       << ":" << field.value << ":" << field.sel << ":" << field.discard << ":"
       << field.disconnect;
    tokens.push_back(os.str());
  }
  std::sort(tokens.begin(), tokens.end());
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  for (const auto &token : tokens) {
    attrs.push_back(mlir::StringAttr::get(ctx, token));
  }
  return attrs;
}

std::string buildAggregatedMatchKey(unsigned familyIndex, const Match &match) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << familyIndex << "|nodes(";
  for (size_t idx = 0; idx < match.swNodesByOp.size(); ++idx) {
    if (idx)
      os << ",";
    os << match.swNodesByOp[idx];
  }
  os << ")|in(";
  for (size_t idx = 0; idx < match.inputBindings.size(); ++idx) {
    if (idx)
      os << ",";
    os << match.inputBindings[idx].swPortId << "->"
       << match.inputBindings[idx].hwPortIndex;
  }
  os << ")|out(";
  for (size_t idx = 0; idx < match.outputBindings.size(); ++idx) {
    if (idx)
      os << ",";
    os << match.outputBindings[idx].swPortId << "->"
       << match.outputBindings[idx].hwPortIndex;
  }
  os << ")|edge(";
  for (size_t idx = 0; idx < match.internalEdges.size(); ++idx) {
    if (idx)
      os << ",";
    os << match.internalEdges[idx];
  }
  os << ")|cfg(" << serializeConfigFields(match.configFields) << ")";
  return text;
}

bool isTemporalCandidateList(llvm::ArrayRef<IdIndex> hwNodeIds, const Graph &adg) {
  if (hwNodeIds.empty())
    return false;
  for (IdIndex hwNodeId : hwNodeIds) {
    const Node *hwNode = adg.getNode(hwNodeId);
    if (!hwNode || getNodeAttrStr(hwNode, "pe_kind") != "temporal_pe")
      return false;
  }
  return true;
}

int64_t computeFamilyScarcityPenalty(size_t hwSupportCount) {
  if (hwSupportCount >= 4)
    return 0;
  return static_cast<int64_t>(4 - hwSupportCount) * 48;
}

unsigned countCommutativeSwaps(
    llvm::ArrayRef<llvm::SmallVector<unsigned, 4>> operandOrderByOp) {
  unsigned swaps = 0;
  for (const auto &operandOrder : operandOrderByOp) {
    for (unsigned operandIdx = 0; operandIdx < operandOrder.size();
         ++operandIdx) {
      if (operandOrder[operandIdx] != operandIdx) {
        ++swaps;
        break;
      }
    }
  }
  return swaps;
}

int64_t scoreAggregatedMatch(const AggregatedMatch &match,
                             size_t familyHwSupportCount) {
  int64_t score = 0;
  score += static_cast<int64_t>(match.swNodesByOp.size()) * 1024;
  score += static_cast<int64_t>(match.internalEdges.size()) * 192;
  score += static_cast<int64_t>(match.hwNodeIds.size()) * 32;
  score -= static_cast<int64_t>(match.inputBindings.size() + match.outputBindings.size()) *
           48;
  score -= static_cast<int64_t>(match.configFields.size()) * 12;
  score -= computeFamilyScarcityPenalty(familyHwSupportCount);
  score -= static_cast<int64_t>(match.commutativeSwapCount) * 16;
  if (match.temporal)
    score -= 64;
  return score;
}

int64_t scoreAggregatedMatch(const AggregatedMatch &match) {
  return scoreAggregatedMatch(match, match.hwNodeIds.size());
}

std::string buildTechmapDiagnostics(const TechMapper::Plan &plan) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "coverage=" << plan.coverageScore;
  os << ", total_layer2_us=" << plan.metrics.totalLayer2TimeMicros;
  os << ", candidate_gen_us=" << plan.metrics.candidateGenerationTimeMicros;
  os << ", selection_us=" << plan.metrics.selectionTimeMicros;
  os << ", op_alias_pairs=" << plan.metrics.opAliasPairCount;
  os << ", demand_candidates=" << plan.metrics.demandCandidateCount;
  os << ", structural_states=" << plan.metrics.structuralStateCount;
  os << ", structural_cache_hits="
     << plan.metrics.structuralStateCacheHitCount;
  os << ", structural_cache_misses="
     << plan.metrics.structuralStateCacheMissCount;
  os << ", selected_candidates=" << plan.metrics.selectedCandidateCount;
  if (plan.metrics.rejectedOverlapCandidateCount != 0)
    os << ", rejected_overlap_candidates="
       << plan.metrics.rejectedOverlapCandidateCount;
  if (plan.metrics.rejectedTemporalCandidateCount != 0)
    os << ", rejected_temporal_candidates="
       << plan.metrics.rejectedTemporalCandidateCount;
  if (plan.metrics.rejectedSupportCapacityCandidateCount != 0)
    os << ", rejected_support_capacity_candidates="
       << plan.metrics.rejectedSupportCapacityCandidateCount;
  if (plan.metrics.rejectedSpatialPoolCandidateCount != 0)
    os << ", rejected_spatial_pool_candidates="
       << plan.metrics.rejectedSpatialPoolCandidateCount;
  if (plan.metrics.objectiveDroppedCandidateCount != 0)
    os << ", objective_dropped_candidates="
       << plan.metrics.objectiveDroppedCandidateCount;
  os << ", fallback_nodes=" << plan.metrics.conservativeFallbackCount;
  os << ", support_classes=" << plan.metrics.supportClassCount;
  os << ", config_classes=" << plan.metrics.configClassCount;
  os << ", fused_ops=" << plan.metrics.selectedFusedOpCount;
  os << ", internal_edges=" << plan.metrics.selectedInternalEdgeCount;
  os << ", candidate_choices=" << plan.metrics.selectedCandidateChoiceCount;
  os << ", selected_config_diversity="
     << plan.metrics.selectedConfigDiversityCount;
  if (plan.metrics.selectedLegacyFallbackCount != 0)
    os << ", selected_legacy_fallback="
       << plan.metrics.selectedLegacyFallbackCount;
  if (plan.metrics.selectedMixedOriginCount != 0)
    os << ", selected_mixed_origin=" << plan.metrics.selectedMixedOriginCount;
  if (plan.metrics.selectedLegacyDerivedCount != 0)
    os << ", selected_legacy_derived="
       << plan.metrics.selectedLegacyDerivedCount;
  os << ", selection_components=" << plan.metrics.selectionComponentCount;
  if (plan.metrics.cpSatComponentCount != 0)
    os << ", cpsat_components=" << plan.metrics.cpSatComponentCount;
  if (plan.metrics.exactComponentCount != 0)
    os << ", exact_components=" << plan.metrics.exactComponentCount;
  if (plan.metrics.greedyComponentCount != 0)
    os << ", greedy_components=" << plan.metrics.greedyComponentCount;
  if (plan.metrics.temporalRiskCount != 0)
    os << ", temporal_risk=" << plan.metrics.temporalRiskCount;
  if (plan.metrics.legacyOracleCandidateCount != 0 ||
      plan.metrics.legacyOracleMissingCount != 0) {
    os << ", legacy_oracle_candidates="
       << plan.metrics.legacyOracleCandidateCount;
    os << ", legacy_oracle_missing=" << plan.metrics.legacyOracleMissingCount;
  }
  if (plan.metrics.legacyOracleEnabled)
    os << ", legacy_oracle_checks=" << plan.metrics.legacyOracleCheckCount;
  if (plan.metrics.legacyOracleRequired)
    os << ", legacy_oracle_required=1";
  if (plan.metrics.legacyFallbackCount != 0)
    os << ", legacy_fallback_sources=" << plan.metrics.legacyFallbackCount;
  if (plan.metrics.legacyFallbackCandidateCount != 0)
    os << ", legacy_fallback_candidates="
       << plan.metrics.legacyFallbackCandidateCount;
  if (plan.metrics.legacyContaminatedCandidateCount != 0)
    os << ", legacy_contaminated_candidates="
       << plan.metrics.legacyContaminatedCandidateCount;
  if (plan.metrics.legacyDerivedSourceCount != 0)
    os << ", legacy_derived_sources="
       << plan.metrics.legacyDerivedSourceCount;
  if (plan.metrics.feedbackReselectionCount != 0)
    os << ", feedback_reselection=" << plan.metrics.feedbackReselectionCount;
  if (plan.metrics.feedbackFilteredCandidateCount != 0)
    os << ", feedback_filtered_candidates="
       << plan.metrics.feedbackFilteredCandidateCount;
  if (plan.metrics.feedbackPenaltyCount != 0)
    os << ", feedback_penalty_terms=" << plan.metrics.feedbackPenaltyCount;
  if (TechMapper::feedbackUnknownCandidateRefCount(plan) != 0)
    os << ", feedback_unknown_candidate_refs="
       << TechMapper::feedbackUnknownCandidateRefCount(plan);
  if (TechMapper::feedbackUnknownFamilyRefCount(plan) != 0)
    os << ", feedback_unknown_family_refs="
       << TechMapper::feedbackUnknownFamilyRefCount(plan);
  if (TechMapper::feedbackUnknownConfigClassRefCount(plan) != 0)
    os << ", feedback_unknown_config_class_refs="
       << TechMapper::feedbackUnknownConfigClassRefCount(plan);
  if (plan.metrics.fallbackNoCandidateCount != 0 ||
      plan.metrics.fallbackRejectedCount != 0) {
    os << ", fallback_no_candidate=" << plan.metrics.fallbackNoCandidateCount;
    os << ", fallback_rejected=" << plan.metrics.fallbackRejectedCount;
  }
  os << ", conservative_fallback_covered="
     << plan.metrics.conservativeFallbackCoveredCount;
  if (plan.metrics.conservativeFallbackMissingCount != 0)
    os << ", conservative_fallback_missing="
       << plan.metrics.conservativeFallbackMissingCount;
  return text;
}

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

bool finalizePlanGraphs(const Graph &dfg, const Graph &adg,
                        TechMapper::Plan &plan) {
  auto &fallbackSwNodes = TechMapper::conservativeFallbackSwNodes(plan);
  fallbackSwNodes.clear();
  for (IdIndex swNodeId = 0;
       swNodeId < static_cast<IdIndex>(plan.conservativeFallbackDFG.nodes.size());
       ++swNodeId) {
    Node *fallbackNode = plan.conservativeFallbackDFG.getNode(swNodeId);
    if (!fallbackNode || fallbackNode->kind != Node::OperationNode)
      continue;
    addBoolNodeAttr(fallbackNode, "techmap_conservative_fallback_plan", true,
                    dfg.context);
    const auto *hwNodes =
        TechMapper::findConservativeFallbackCandidates(plan, swNodeId);
    const auto *supportClasses =
        TechMapper::findConservativeFallbackCandidateSupportClasses(plan,
                                                                   swNodeId);
    const auto *configClasses =
        TechMapper::findConservativeFallbackCandidateConfigClasses(plan,
                                                                  swNodeId);
    const auto *candidateDetails =
        TechMapper::findConservativeFallbackCandidateDetails(plan, swNodeId);
    const auto *preferredCandidate =
        TechMapper::findConservativeFallbackPreferredCandidate(plan, swNodeId);
    llvm::SmallVector<mlir::Attribute, 4> hwAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportAttrs;
    llvm::SmallVector<mlir::Attribute, 4> configAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportKeyAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportKindAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportTemporalAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportHardCapacityAttrs;
    llvm::SmallVector<mlir::Attribute, 4> supportCapacityAttrs;
    llvm::SmallVector<mlir::Attribute, 4> configKeyAttrs;
    llvm::SmallVector<mlir::Attribute, 4> configReasonAttrs;
    llvm::SmallVector<mlir::Attribute, 4> configTemporalAttrs;
    llvm::SmallVector<mlir::Attribute, 4> configFieldSetAttrs;
    if (hwNodes) {
      for (IdIndex hwNodeId : *hwNodes) {
        hwAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64), hwNodeId));
      }
    }
    if (supportClasses) {
      for (unsigned supportClassId : *supportClasses) {
        const auto *supportClassInfo =
            TechMapper::findSupportClass(plan, supportClassId);
        supportAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64), supportClassId));
        supportKeyAttrs.push_back(mlir::StringAttr::get(
            dfg.context,
            supportClassInfo ? llvm::StringRef(supportClassInfo->key)
                             : llvm::StringRef()));
        supportKindAttrs.push_back(mlir::StringAttr::get(
            dfg.context,
            supportClassInfo ? llvm::StringRef(supportClassInfo->kind)
                             : llvm::StringRef()));
        supportTemporalAttrs.push_back(mlir::BoolAttr::get(
            dfg.context, supportClassInfo && supportClassInfo->temporal));
        supportHardCapacityAttrs.push_back(mlir::BoolAttr::get(
            dfg.context,
            TechMapper::supportClassEnforcesHardCapacity(plan, supportClassId)));
        supportCapacityAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64),
            TechMapper::supportClassCapacity(plan, supportClassId)));
      }
    }
    if (configClasses) {
      for (unsigned configClassId : *configClasses) {
        const auto *configClassInfo =
            TechMapper::findConfigClass(plan, configClassId);
        configAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64), configClassId));
        configKeyAttrs.push_back(mlir::StringAttr::get(
            dfg.context,
            configClassInfo ? llvm::StringRef(configClassInfo->key)
                            : llvm::StringRef()));
        configReasonAttrs.push_back(mlir::StringAttr::get(
            dfg.context,
            configClassInfo ? llvm::StringRef(configClassInfo->reason)
                            : llvm::StringRef()));
        configTemporalAttrs.push_back(mlir::BoolAttr::get(
            dfg.context, TechMapper::isTemporalConfigClass(plan, configClassId)));
      }
    }
    if (candidateDetails) {
      for (const auto &candidate : *candidateDetails) {
        configFieldSetAttrs.push_back(mlir::StringAttr::get(
            dfg.context, serializeConfigFields(candidate.configFields)));
      }
    }
    addNodeAttr(fallbackNode, "techmap_candidate_hw_nodes",
                mlir::ArrayAttr::get(dfg.context, hwAttrs), dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_support_classes",
                mlir::ArrayAttr::get(dfg.context, supportAttrs), dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_support_class_keys",
                mlir::ArrayAttr::get(dfg.context, supportKeyAttrs),
                dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_support_class_kinds",
                mlir::ArrayAttr::get(dfg.context, supportKindAttrs),
                dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_support_class_temporal",
                mlir::ArrayAttr::get(dfg.context, supportTemporalAttrs),
                dfg.context);
    addNodeAttr(
        fallbackNode, "techmap_candidate_support_class_enforce_hard_capacity",
        mlir::ArrayAttr::get(dfg.context, supportHardCapacityAttrs),
        dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_support_class_capacities",
                mlir::ArrayAttr::get(dfg.context, supportCapacityAttrs),
                dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_config_classes",
                mlir::ArrayAttr::get(dfg.context, configAttrs), dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_config_class_keys",
                mlir::ArrayAttr::get(dfg.context, configKeyAttrs), dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_config_class_reasons",
                mlir::ArrayAttr::get(dfg.context, configReasonAttrs),
                dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_config_class_temporal",
                mlir::ArrayAttr::get(dfg.context, configTemporalAttrs),
                dfg.context);
    addNodeAttr(fallbackNode, "techmap_candidate_config_field_sets",
                mlir::ArrayAttr::get(dfg.context, configFieldSetAttrs),
                dfg.context);
    addBoolNodeAttr(fallbackNode, "techmap_conservative_fallback_covered",
                    !hwAttrs.empty(), dfg.context);
    addUIntNodeAttr(fallbackNode, "techmap_conservative_fallback_candidate_count",
                    configFieldSetAttrs.size(), dfg.context);
    addUIntNodeAttr(
        fallbackNode, "techmap_conservative_fallback_support_class_count",
        supportAttrs.size(), dfg.context);
    addUIntNodeAttr(
        fallbackNode, "techmap_conservative_fallback_config_class_count",
        configAttrs.size(), dfg.context);
    if (const auto *fallbackInfo =
            TechMapper::findFallbackNodeInfo(plan, swNodeId)) {
      addNodeAttr(fallbackNode, "techmap_fallback_reason",
                  mlir::StringAttr::get(dfg.context, fallbackInfo->reason),
                  dfg.context);
      llvm::SmallVector<mlir::Attribute, 8> candidateIdAttrs;
      for (unsigned candidateId : fallbackInfo->candidateIds) {
        candidateIdAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64), candidateId));
      }
      addNodeAttr(fallbackNode, "techmap_fallback_candidate_ids",
                  mlir::ArrayAttr::get(dfg.context, candidateIdAttrs),
                  dfg.context);
    }
    if (preferredCandidate) {
      const auto *preferredSupportClass =
          TechMapper::findSupportClass(plan, preferredCandidate->supportClassId);
      const auto *preferredConfigClass =
          TechMapper::findConfigClass(plan, preferredCandidate->configClassId);
      addUIntNodeAttr(fallbackNode, "techmap_preferred_hw_node",
                      preferredCandidate->hwNodeId, dfg.context);
      addUIntNodeAttr(fallbackNode, "techmap_preferred_support_class_id",
                      preferredCandidate->supportClassId, dfg.context);
      addNodeAttr(fallbackNode, "techmap_preferred_support_class_key",
                  mlir::StringAttr::get(
                      dfg.context, preferredSupportClass
                                       ? llvm::StringRef(preferredSupportClass->key)
                                       : llvm::StringRef()),
                  dfg.context);
      addNodeAttr(fallbackNode, "techmap_preferred_support_class_kind",
                  mlir::StringAttr::get(
                      dfg.context, preferredSupportClass
                                       ? llvm::StringRef(preferredSupportClass->kind)
                                       : llvm::StringRef()),
                  dfg.context);
      addBoolNodeAttr(
          fallbackNode, "techmap_preferred_support_class_temporal",
          preferredSupportClass && preferredSupportClass->temporal,
          dfg.context);
      addBoolNodeAttr(
          fallbackNode, "techmap_preferred_support_class_enforce_hard_capacity",
          TechMapper::supportClassEnforcesHardCapacity(
              plan, preferredCandidate->supportClassId),
          dfg.context);
      addUIntNodeAttr(fallbackNode, "techmap_preferred_support_class_capacity",
                      TechMapper::supportClassCapacity(
                          plan, preferredCandidate->supportClassId),
                      dfg.context);
      addUIntNodeAttr(fallbackNode, "techmap_preferred_config_class_id",
                      preferredCandidate->configClassId, dfg.context);
      addNodeAttr(fallbackNode, "techmap_preferred_config_class_key",
                  mlir::StringAttr::get(
                      dfg.context, preferredConfigClass
                                       ? llvm::StringRef(preferredConfigClass->key)
                                       : llvm::StringRef()),
                  dfg.context);
      addNodeAttr(fallbackNode, "techmap_preferred_config_class_reason",
                  mlir::StringAttr::get(
                      dfg.context, preferredConfigClass
                                       ? llvm::StringRef(preferredConfigClass->reason)
                                       : llvm::StringRef()),
                  dfg.context);
      addBoolNodeAttr(
          fallbackNode, "techmap_preferred_config_class_temporal",
          TechMapper::isTemporalConfigClass(plan,
                                            preferredCandidate->configClassId),
          dfg.context);
      if (TechMapper::findConfigClass(plan, preferredCandidate->configClassId)) {
        llvm::SmallVector<mlir::Attribute, 4> compatIdAttrs;
        llvm::SmallVector<mlir::Attribute, 4> compatKeyAttrs;
        for (unsigned compatId :
             TechMapper::compatibleConfigClasses(
                 plan, preferredCandidate->configClassId)) {
          const auto *compatConfigClass =
              TechMapper::findConfigClass(plan, compatId);
          compatIdAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), compatId));
          compatKeyAttrs.push_back(mlir::StringAttr::get(
              dfg.context, compatConfigClass
                               ? llvm::StringRef(compatConfigClass->key)
                               : llvm::StringRef()));
        }
        addNodeAttr(fallbackNode, "techmap_preferred_config_class_compatible_with",
                    mlir::ArrayAttr::get(dfg.context, compatIdAttrs), dfg.context);
        addNodeAttr(
            fallbackNode, "techmap_preferred_config_class_compatible_with_keys",
            mlir::ArrayAttr::get(dfg.context, compatKeyAttrs), dfg.context);
      }
      addBoolNodeAttr(fallbackNode, "techmap_preferred_temporal",
                      preferredCandidate->temporal, dfg.context);
      addNodeAttr(fallbackNode, "techmap_preferred_config_fields",
                  mlir::ArrayAttr::get(
                      dfg.context,
                      buildConfigFieldSummaryAttrs(preferredCandidate->configFields,
                                                  dfg.context)),
                  dfg.context);
    }
    if (const auto *nodeInfo = TechMapper::findNodeTechInfo(plan, swNodeId);
        nodeInfo && nodeInfo->selectionComponentId !=
                        std::numeric_limits<unsigned>::max()) {
      addUIntNodeAttr(fallbackNode, "techmap_selection_component_id",
                      nodeInfo->selectionComponentId, dfg.context);
    }
    if (!hwAttrs.empty())
      ++plan.metrics.conservativeFallbackCoveredCount;
    else
      ++plan.metrics.conservativeFallbackMissingCount;
  }

  std::vector<int> nodeToUnit(dfg.nodes.size(), -1);
  for (unsigned unitIndex = 0; unitIndex < TechMapper::allUnits(plan).size();
       ++unitIndex) {
    const auto *unit = TechMapper::findUnit(plan, unitIndex);
    if (!unit)
      continue;
    for (IdIndex swNodeId : unit->swNodes) {
      if (swNodeId < nodeToUnit.size())
        nodeToUnit[swNodeId] = unitIndex;
    }
  }

  plan.contractedDFG = Graph(dfg.context);
  auto &contracted = plan.contractedDFG;
  contracted.reserve(dfg.countNodes(), dfg.countPorts(), dfg.countEdges());
  plan.contractedCandidates.clear();
  plan.contractedCandidateSupportClasses.clear();
  plan.contractedCandidateConfigClasses.clear();

  std::vector<bool> unitCreated(TechMapper::allUnits(plan).size(), false);
  for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++swNodeId) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode)
      continue;

    int unitIndex = swNodeId < nodeToUnit.size() ? nodeToUnit[swNodeId] : -1;
    if (unitIndex >= 0) {
      auto *unit = TechMapper::findUnit(plan, static_cast<unsigned>(unitIndex));
      if (!unit)
        return false;
      if (unitCreated[unitIndex]) {
        plan.originalNodeToContractedNode[swNodeId] =
            unit->contractedNodeId;
        continue;
      }

      const auto *preferredCandidate =
          TechMapper::findPreferredUnitCandidate(plan, unitIndex);
      if (!preferredCandidate)
        return false;
      const auto *selectionComponent = TechMapper::findSelectionComponent(plan, *unit);
      const auto *familyInfo = TechMapper::findFamilyTechInfo(plan, *unit);
      const auto *selectedCandidateSummary =
          TechMapper::findSelectedCandidateSummary(plan, *unit);
      const auto *selectedConfigClass = TechMapper::findSelectedUnitConfigClass(plan, *unit);
      const auto *preferredSupportClass =
          TechMapper::findPreferredUnitSupportClass(plan, unitIndex);
      const auto *preferredConfigClass =
          TechMapper::findPreferredUnitConfigClass(plan, unitIndex);
      const Node *hwNode = adg.getNode(preferredCandidate->hwNodeId);
      if (!hwNode)
        return false;

      auto node = std::make_unique<Node>();
      node->kind = Node::OperationNode;
      addNodeAttr(node.get(), "op_name",
                  mlir::StringAttr::get(dfg.context, "techmap_group"),
                  dfg.context);
      addNodeAttr(node.get(), "tech_group_size",
                  mlir::IntegerAttr::get(mlir::IntegerType::get(dfg.context, 32),
                                         unit->swNodes.size()),
                  dfg.context);
      addUIntNodeAttr(node.get(), "techmap_preferred_candidate_index",
                      unit->preferredCandidateIndex,
                      dfg.context);
      addUIntNodeAttr(node.get(), "techmap_selected_unit_index", unitIndex,
                      dfg.context);
      addUIntNodeAttr(node.get(), "techmap_selected_candidate_id",
                      unit->selectedCandidateId, dfg.context);
      if (unit->selectionComponentId !=
          std::numeric_limits<unsigned>::max()) {
        addUIntNodeAttr(node.get(), "techmap_selection_component_id",
                        unit->selectionComponentId, dfg.context);
        if (selectionComponent) {
          addNodeAttr(
              node.get(), "techmap_selection_solver",
              mlir::StringAttr::get(
                  dfg.context, selectionComponent->solver),
              dfg.context);
        }
      }
      addUIntNodeAttr(node.get(), "techmap_family_index",
                      unit->familyIndex, dfg.context);
      if (familyInfo) {
        addNodeAttr(
            node.get(), "techmap_family_signature",
            mlir::StringAttr::get(dfg.context, familyInfo->signature),
            dfg.context);
      }
      addUIntNodeAttr(node.get(), "techmap_config_class_id",
                      unit->configClassId, dfg.context);
      addNodeAttr(node.get(), "techmap_config_class_key",
                  mlir::StringAttr::get(dfg.context, selectedConfigClass
                                                         ? selectedConfigClass
                                                               ->key
                                                         : llvm::StringRef()),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_config_class_reason",
                  mlir::StringAttr::get(dfg.context, selectedConfigClass
                                                         ? selectedConfigClass
                                                               ->reason
                                                         : llvm::StringRef()),
                  dfg.context);
      addBoolNodeAttr(node.get(), "techmap_config_class_temporal",
                      TechMapper::isTemporalConfigClass(plan, unit->configClassId),
                      dfg.context);
      if (TechMapper::findConfigClass(plan, unit->configClassId)) {
        llvm::SmallVector<mlir::Attribute, 4> compatIdAttrs;
        llvm::SmallVector<mlir::Attribute, 4> compatKeyAttrs;
        for (unsigned compatId :
             TechMapper::compatibleConfigClasses(plan, unit->configClassId)) {
          const auto *compatConfigClass =
              TechMapper::findConfigClass(plan, compatId);
          compatIdAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), compatId));
          compatKeyAttrs.push_back(mlir::StringAttr::get(
              dfg.context, compatConfigClass
                               ? llvm::StringRef(compatConfigClass->key)
                               : llvm::StringRef()));
        }
        addNodeAttr(node.get(), "techmap_config_class_compatible_with",
                    mlir::ArrayAttr::get(dfg.context, compatIdAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_config_class_compatible_with_keys",
                    mlir::ArrayAttr::get(dfg.context, compatKeyAttrs),
                    dfg.context);
      }
      addUIntNodeAttr(node.get(), "techmap_support_class_id",
                      preferredCandidate->supportClassId, dfg.context);
      addNodeAttr(node.get(), "techmap_support_class_key",
                  mlir::StringAttr::get(dfg.context, preferredSupportClass
                                                         ? preferredSupportClass
                                                               ->key
                                                         : llvm::StringRef()),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_support_class_kind",
                  mlir::StringAttr::get(dfg.context, preferredSupportClass
                                                         ? preferredSupportClass
                                                               ->kind
                                                         : llvm::StringRef()),
                  dfg.context);
      addBoolNodeAttr(node.get(), "techmap_support_class_temporal",
                      preferredSupportClass && preferredSupportClass->temporal,
                      dfg.context);
      addBoolNodeAttr(node.get(), "techmap_support_class_enforce_hard_capacity",
                      TechMapper::supportClassEnforcesHardCapacity(
                          plan, preferredCandidate->supportClassId),
                      dfg.context);
      addUIntNodeAttr(node.get(), "techmap_support_class_capacity",
                      TechMapper::supportClassCapacity(plan,
                                                       preferredCandidate
                                                           ->supportClassId),
                      dfg.context);
      if (!unit->swNodes.empty() &&
          unit->selectionComponentId == std::numeric_limits<unsigned>::max()) {
        IdIndex reprSwNodeId = unit->swNodes.front();
        if (const auto *nodeInfo =
                TechMapper::findNodeTechInfo(plan, reprSwNodeId);
            nodeInfo && nodeInfo->selectionComponentId !=
                            std::numeric_limits<unsigned>::max()) {
          addUIntNodeAttr(node.get(), "techmap_selection_component_id",
                          nodeInfo->selectionComponentId, dfg.context);
        }
      }
      addNodeAttr(node.get(), "techmap_selected_config_fields",
                  mlir::ArrayAttr::get(
                      dfg.context,
                      buildConfigFieldSummaryAttrs(preferredCandidate->configFields,
                                                  dfg.context)),
                  dfg.context);
      if (selectedCandidateSummary) {
        addUIntNodeAttr(
            node.get(), "techmap_base_selection_score",
            static_cast<uint64_t>(std::max<int64_t>(
                0, selectedCandidateSummary->baseSelectionScore)),
            dfg.context);
        addUIntNodeAttr(node.get(), "techmap_candidate_penalty",
                        static_cast<uint64_t>(
                            std::max<int64_t>(0,
                                              selectedCandidateSummary
                                                  ->candidatePenalty)),
                        dfg.context);
        addUIntNodeAttr(node.get(), "techmap_family_penalty",
                        static_cast<uint64_t>(
                            std::max<int64_t>(0,
                                              selectedCandidateSummary
                                                  ->familyPenalty)),
                        dfg.context);
        addUIntNodeAttr(
            node.get(), "techmap_config_class_penalty",
            static_cast<uint64_t>(
                std::max<int64_t>(0,
                                  selectedCandidateSummary
                                      ->configClassPenalty)),
            dfg.context);
      }
      addUIntNodeAttr(node.get(), "techmap_selection_score",
                      static_cast<uint64_t>(
                          std::max<int64_t>(0, unit->selectionScore)),
                      dfg.context);
      addUIntNodeAttr(node.get(), "techmap_candidate_count",
                      unit->candidates.size(), dfg.context);
      addBoolNodeAttr(node.get(), "techmap_conservative_fallback",
                      unit->conservativeFallback, dfg.context);
      addBoolNodeAttr(node.get(), "techmap_legacy_fallback_origin",
                      unit->legacyFallbackOrigin, dfg.context);
      addNodeAttr(node.get(), "techmap_origin_kind",
                  mlir::StringAttr::get(
                      dfg.context,
                      TechMapper::originKind(unit->demandOrigin,
                                             unit->legacyFallbackOrigin,
                                             unit->mixedOrigin)),
                  dfg.context);
      llvm::SmallVector<mlir::Attribute, 4> candidateClassAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateSupportAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateClassKeyAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateClassReasonAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateClassTemporalAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateSupportKeyAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateSupportKindAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateSupportTemporalAttrs;
      llvm::SmallVector<mlir::Attribute, 4>
          candidateSupportHardCapacityAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateSupportCapacityAttrs;
      llvm::SmallVector<mlir::Attribute, 4> candidateHwAttrs;
      for (const auto &unitCandidate : unit->candidates) {
        const auto *candidateSupportClass =
            TechMapper::findSupportClass(plan, unitCandidate);
        const auto *candidateConfigClass =
            TechMapper::findConfigClass(plan, unitCandidate);
        candidateHwAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64), unitCandidate.hwNodeId));
        candidateSupportAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64),
            unitCandidate.supportClassId));
        candidateSupportKeyAttrs.push_back(mlir::StringAttr::get(
            dfg.context, candidateSupportClass ? candidateSupportClass->key
                                               : llvm::StringRef()));
        candidateSupportKindAttrs.push_back(mlir::StringAttr::get(
            dfg.context, candidateSupportClass ? candidateSupportClass->kind
                                               : llvm::StringRef()));
        candidateSupportTemporalAttrs.push_back(mlir::BoolAttr::get(
            dfg.context, candidateSupportClass
                             ? candidateSupportClass->temporal
                             : false));
        candidateSupportHardCapacityAttrs.push_back(mlir::BoolAttr::get(
            dfg.context,
            candidateSupportClass
                ? candidateSupportClass->enforceHardCapacity
                : false));
        candidateSupportCapacityAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64),
            TechMapper::supportClassCapacity(plan,
                                             unitCandidate.supportClassId)));
        candidateClassAttrs.push_back(mlir::IntegerAttr::get(
            mlir::IntegerType::get(dfg.context, 64),
            unitCandidate.configClassId));
        candidateClassKeyAttrs.push_back(mlir::StringAttr::get(
            dfg.context, candidateConfigClass ? candidateConfigClass->key
                                              : llvm::StringRef()));
        candidateClassReasonAttrs.push_back(mlir::StringAttr::get(
            dfg.context, candidateConfigClass ? candidateConfigClass->reason
                                              : llvm::StringRef()));
        candidateClassTemporalAttrs.push_back(mlir::BoolAttr::get(
            dfg.context, candidateConfigClass ? candidateConfigClass->temporal
                                              : false));
      }
      addNodeAttr(node.get(), "techmap_candidate_hw_nodes",
                  mlir::ArrayAttr::get(dfg.context, candidateHwAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_support_classes",
                  mlir::ArrayAttr::get(dfg.context, candidateSupportAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_support_class_keys",
                  mlir::ArrayAttr::get(dfg.context, candidateSupportKeyAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_support_class_kinds",
                  mlir::ArrayAttr::get(dfg.context, candidateSupportKindAttrs),
                  dfg.context);
      addNodeAttr(
          node.get(), "techmap_candidate_support_class_temporal",
          mlir::ArrayAttr::get(dfg.context, candidateSupportTemporalAttrs),
          dfg.context);
      addNodeAttr(
          node.get(), "techmap_candidate_support_class_enforce_hard_capacity",
          mlir::ArrayAttr::get(dfg.context, candidateSupportHardCapacityAttrs),
          dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_support_class_capacities",
                  mlir::ArrayAttr::get(dfg.context, candidateSupportCapacityAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_config_classes",
                  mlir::ArrayAttr::get(dfg.context, candidateClassAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_config_class_keys",
                  mlir::ArrayAttr::get(dfg.context, candidateClassKeyAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_config_class_reasons",
                  mlir::ArrayAttr::get(dfg.context, candidateClassReasonAttrs),
                  dfg.context);
      addNodeAttr(node.get(), "techmap_candidate_config_class_temporal",
                  mlir::ArrayAttr::get(dfg.context, candidateClassTemporalAttrs),
                  dfg.context);
      IdIndex contractedNodeId = contracted.addNode(std::move(node));
      unit->contractedNodeId = contractedNodeId;
      if (auto *selectedCandidateSummary = TechMapper::findCandidateSummary(
              plan, unit->selectedCandidateId)) {
        selectedCandidateSummary->contractedNodeId = contractedNodeId;
      }
      for (IdIndex member : unit->swNodes) {
        plan.originalNodeToContractedNode[member] = contractedNodeId;
        if (auto *nodeInfo = TechMapper::findNodeTechInfo(plan, member))
          nodeInfo->contractedNodeId = contractedNodeId;
      }

      for (IdIndex hwPortId : hwNode->inputPorts) {
        const Port *hwPort = adg.getPort(hwPortId);
        auto port = std::make_unique<Port>();
        port->direction = Port::Input;
        port->type = hwPort ? hwPort->type : mlir::Type();
        IdIndex portId = contracted.addPort(std::move(port));
        contracted.ports[portId]->parentNode = contractedNodeId;
        contracted.nodes[contractedNodeId]->inputPorts.push_back(portId);
        unit->contractedInputPorts.push_back(portId);
      }
      for (IdIndex hwPortId : hwNode->outputPorts) {
        const Port *hwPort = adg.getPort(hwPortId);
        auto port = std::make_unique<Port>();
        port->direction = Port::Output;
        port->type = hwPort ? hwPort->type : mlir::Type();
        IdIndex portId = contracted.addPort(std::move(port));
        contracted.ports[portId]->parentNode = contractedNodeId;
        contracted.nodes[contractedNodeId]->outputPorts.push_back(portId);
        unit->contractedOutputPorts.push_back(portId);
      }
      for (const auto &unitCandidate : unit->candidates)
        plan.contractedCandidates[contractedNodeId].push_back(
            unitCandidate.hwNodeId);
      for (const auto &unitCandidate : unit->candidates)
        plan.contractedCandidateSupportClasses[contractedNodeId].push_back(
            unitCandidate.supportClassId);
      for (const auto &unitCandidate : unit->candidates)
        plan.contractedCandidateConfigClasses[contractedNodeId].push_back(
            unitCandidate.configClassId);
      unitCreated[unitIndex] = true;
      continue;
    }

    auto node = cloneNodeShell(swNode, dfg.context);
    if (swNode->kind == Node::OperationNode) {
      addBoolNodeAttr(node.get(), "techmap_conservative_fallback", true,
                      dfg.context);
      if (const auto *nodeInfo = TechMapper::findNodeTechInfo(plan, swNodeId)) {
        if (nodeInfo->selectionComponentId !=
            std::numeric_limits<unsigned>::max()) {
          addUIntNodeAttr(node.get(), "techmap_selection_component_id",
                          nodeInfo->selectionComponentId, dfg.context);
        }
        addUIntNodeAttr(node.get(), "techmap_candidate_count",
                        nodeInfo->candidateCount, dfg.context);
        addUIntNodeAttr(node.get(), "techmap_support_class_count",
                        nodeInfo->supportClassCount, dfg.context);
        addUIntNodeAttr(node.get(), "techmap_config_class_count",
                        nodeInfo->configClassCount, dfg.context);
        addUIntNodeAttr(node.get(), "techmap_max_fusion_size",
                        nodeInfo->maxFusionSize, dfg.context);
        addNodeAttr(node.get(), "techmap_status",
                    mlir::StringAttr::get(dfg.context, nodeInfo->status),
                    dfg.context);
        llvm::SmallVector<mlir::Attribute, 8> candidateIdAttrs;
        for (unsigned candidateId : nodeInfo->candidateIds) {
          candidateIdAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), candidateId));
        }
        addNodeAttr(node.get(), "techmap_candidate_ids",
                    mlir::ArrayAttr::get(dfg.context, candidateIdAttrs),
                    dfg.context);
        llvm::SmallVector<mlir::Attribute, 8> supportClassAttrs;
        llvm::SmallVector<mlir::Attribute, 8> supportClassKeyAttrs;
        llvm::SmallVector<mlir::Attribute, 8> supportClassKindAttrs;
        llvm::SmallVector<mlir::Attribute, 8> supportClassTemporalAttrs;
        llvm::SmallVector<mlir::Attribute, 8> supportClassHardCapacityAttrs;
        llvm::SmallVector<mlir::Attribute, 8> supportClassCapacityAttrs;
        for (unsigned supportClassId : nodeInfo->supportClassIds) {
          const auto *supportClassInfo =
              TechMapper::findSupportClass(plan, supportClassId);
          supportClassAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), supportClassId));
          supportClassKeyAttrs.push_back(mlir::StringAttr::get(
              dfg.context,
              supportClassInfo ? llvm::StringRef(supportClassInfo->key)
                               : llvm::StringRef()));
          supportClassKindAttrs.push_back(mlir::StringAttr::get(
              dfg.context,
              supportClassInfo ? llvm::StringRef(supportClassInfo->kind)
                               : llvm::StringRef()));
          supportClassTemporalAttrs.push_back(mlir::BoolAttr::get(
              dfg.context, supportClassInfo && supportClassInfo->temporal));
          supportClassHardCapacityAttrs.push_back(mlir::BoolAttr::get(
              dfg.context,
              TechMapper::supportClassEnforcesHardCapacity(plan, supportClassId)));
          supportClassCapacityAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64),
              TechMapper::supportClassCapacity(plan, supportClassId)));
        }
        addNodeAttr(node.get(), "techmap_support_class_ids",
                    mlir::ArrayAttr::get(dfg.context, supportClassAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_support_class_keys",
                    mlir::ArrayAttr::get(dfg.context, supportClassKeyAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_support_class_kinds",
                    mlir::ArrayAttr::get(dfg.context, supportClassKindAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_support_class_temporal",
                    mlir::ArrayAttr::get(dfg.context, supportClassTemporalAttrs),
                    dfg.context);
        addNodeAttr(
            node.get(), "techmap_support_class_enforce_hard_capacity",
            mlir::ArrayAttr::get(dfg.context, supportClassHardCapacityAttrs),
            dfg.context);
        addNodeAttr(node.get(), "techmap_support_class_capacities",
                    mlir::ArrayAttr::get(dfg.context, supportClassCapacityAttrs),
                    dfg.context);
        llvm::SmallVector<mlir::Attribute, 8> configClassAttrs;
        llvm::SmallVector<mlir::Attribute, 8> configClassKeyAttrs;
        llvm::SmallVector<mlir::Attribute, 8> configClassReasonAttrs;
        llvm::SmallVector<mlir::Attribute, 8> configClassTemporalAttrs;
        for (unsigned configClassId : nodeInfo->configClassIds) {
          const auto *configClassInfo =
              TechMapper::findConfigClass(plan, configClassId);
          configClassAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), configClassId));
          configClassKeyAttrs.push_back(mlir::StringAttr::get(
              dfg.context,
              configClassInfo ? llvm::StringRef(configClassInfo->key)
                              : llvm::StringRef()));
          configClassReasonAttrs.push_back(mlir::StringAttr::get(
              dfg.context,
              configClassInfo ? llvm::StringRef(configClassInfo->reason)
                              : llvm::StringRef()));
          configClassTemporalAttrs.push_back(mlir::BoolAttr::get(
              dfg.context,
              TechMapper::isTemporalConfigClass(plan, configClassId)));
        }
        addNodeAttr(node.get(), "techmap_config_class_ids",
                    mlir::ArrayAttr::get(dfg.context, configClassAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_config_class_keys",
                    mlir::ArrayAttr::get(dfg.context, configClassKeyAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_config_class_reasons",
                    mlir::ArrayAttr::get(dfg.context, configClassReasonAttrs),
                    dfg.context);
        addNodeAttr(node.get(), "techmap_config_class_temporal",
                    mlir::ArrayAttr::get(dfg.context, configClassTemporalAttrs),
                    dfg.context);
      }
      if (const auto *fallbackInfo =
              TechMapper::findFallbackNodeInfo(plan, swNodeId)) {
        addNodeAttr(node.get(), "techmap_fallback_reason",
                    mlir::StringAttr::get(dfg.context, fallbackInfo->reason),
                    dfg.context);
        llvm::SmallVector<mlir::Attribute, 8> fallbackCandidateIdAttrs;
        for (unsigned candidateId : fallbackInfo->candidateIds) {
          fallbackCandidateIdAttrs.push_back(mlir::IntegerAttr::get(
              mlir::IntegerType::get(dfg.context, 64), candidateId));
        }
        addNodeAttr(node.get(), "techmap_fallback_candidate_ids",
                    mlir::ArrayAttr::get(dfg.context, fallbackCandidateIdAttrs),
                    dfg.context);
      }
      fallbackSwNodes.push_back(swNodeId);
    }
    IdIndex contractedNodeId = contracted.addNode(std::move(node));
    plan.originalNodeToContractedNode[swNodeId] = contractedNodeId;
    if (auto *nodeInfo = TechMapper::findNodeTechInfo(plan, swNodeId))
      nodeInfo->contractedNodeId = contractedNodeId;
    if (auto *fallbackInfo = TechMapper::findFallbackNodeInfo(plan, swNodeId))
      fallbackInfo->contractedNodeId = contractedNodeId;

    for (IdIndex swPortId : swNode->inputPorts) {
      const Port *swPort = dfg.getPort(swPortId);
      auto port = clonePort(swPort);
      IdIndex contractedPortId = contracted.addPort(std::move(port));
      contracted.ports[contractedPortId]->parentNode = contractedNodeId;
      contracted.nodes[contractedNodeId]->inputPorts.push_back(contractedPortId);
      plan.originalPortToContractedPort[swPortId] = contractedPortId;
    }
    for (IdIndex swPortId : swNode->outputPorts) {
      const Port *swPort = dfg.getPort(swPortId);
      auto port = clonePort(swPort);
      IdIndex contractedPortId = contracted.addPort(std::move(port));
      contracted.ports[contractedPortId]->parentNode = contractedNodeId;
      contracted.nodes[contractedNodeId]->outputPorts.push_back(contractedPortId);
      plan.originalPortToContractedPort[swPortId] = contractedPortId;
    }
  }

  for (auto &unit : TechMapper::allUnits(plan)) {
    for (const auto &binding : unit.inputBindings) {
      if (binding.swPortId >= plan.originalPortToContractedPort.size() ||
          binding.hwPortIndex >= unit.contractedInputPorts.size())
        return false;
      plan.originalPortToContractedPort[binding.swPortId] =
          unit.contractedInputPorts[binding.hwPortIndex];
    }
    for (const auto &binding : unit.outputBindings) {
      if (binding.swPortId >= plan.originalPortToContractedPort.size() ||
          binding.hwPortIndex >= unit.contractedOutputPorts.size())
        return false;
      plan.originalPortToContractedPort[binding.swPortId] =
          unit.contractedOutputPorts[binding.hwPortIndex];
    }
    for (IdIndex edgeId : unit.internalEdges) {
      if (edgeId < plan.originalEdgeKinds.size())
        plan.originalEdgeKinds[edgeId] = TechMappedEdgeKind::IntraFU;
    }
  }

  std::map<std::string, IdIndex> dedupEdges;
  for (IdIndex edgeId = 0; edgeId < static_cast<IdIndex>(dfg.edges.size());
       ++edgeId) {
    if (plan.originalEdgeKinds[edgeId] == TechMappedEdgeKind::IntraFU)
      continue;
    const Edge *edge = dfg.getEdge(edgeId);
    if (!edge)
      continue;
    IdIndex srcPort = plan.originalPortToContractedPort[edge->srcPort];
    IdIndex dstPort = plan.originalPortToContractedPort[edge->dstPort];
    if (srcPort == INVALID_ID || dstPort == INVALID_ID) {
      plan.diagnostics = "tech-mapping lost an external port binding";
      return false;
    }

    std::string key = std::to_string(srcPort) + ":" + std::to_string(dstPort);
    auto found = dedupEdges.find(key);
    if (found != dedupEdges.end()) {
      plan.originalEdgeToContractedEdge[edgeId] = found->second;
      continue;
    }

    auto newEdge = std::make_unique<Edge>();
    newEdge->srcPort = srcPort;
    newEdge->dstPort = dstPort;
    newEdge->attributes = edge->attributes;
    IdIndex contractedEdgeId = contracted.addEdge(std::move(newEdge));
    contracted.ports[srcPort]->connectedEdges.push_back(contractedEdgeId);
    contracted.ports[dstPort]->connectedEdges.push_back(contractedEdgeId);
    dedupEdges[key] = contractedEdgeId;
    plan.originalEdgeToContractedEdge[edgeId] = contractedEdgeId;
  }

  return true;
}

} // namespace

bool TechMapper::buildPlan(const Graph &dfg, mlir::ModuleOp adgModule,
                           const Graph &adg, Plan &plan) {
  const auto layer2StartTime = std::chrono::steady_clock::now();
  auto candidateGenerationStartTime = layer2StartTime;
  bool candidateGenerationTimed = false;
  bool selectionTimed = false;
  Plan freshPlan;
  plan = std::move(freshPlan);
  plan.originalNodeToContractedNode.assign(dfg.nodes.size(), INVALID_ID);
  plan.originalPortToContractedPort.assign(dfg.ports.size(), INVALID_ID);
  plan.originalEdgeToContractedEdge.assign(dfg.edges.size(), INVALID_ID);
  plan.originalEdgeKinds.assign(dfg.edges.size(), TechMappedEdgeKind::Routed);
  plan.contractedDFG = Graph(dfg.context);
  plan.conservativeFallbackDFG = dfg.clone();
  plan.metrics.opAliasPairCount = opcompat::getAliasPairs().size();
  plan.coverageScore = 1.0;
  unsigned totalOpCount = 0;
  for (const Node *swNode : dfg.nodeRange()) {
    if (swNode && swNode->kind == Node::OperationNode)
      ++totalOpCount;
  }

  llvm::SmallVector<VariantFamily, 16> familyList;

  llvm::DenseMap<mlir::Block *, llvm::DenseSet<llvm::StringRef>>
      referencedTargetsByBlock;
  adgModule.walk([&](fcc::fabric::InstanceOp instOp) {
    referencedTargetsByBlock[instOp->getBlock()].insert(instOp.getModule());
  });

  auto isDefinitionOp = [&](mlir::Operation *op,
                            llvm::StringRef name) -> bool {
    if (mlir::isa<fcc::fabric::FunctionUnitOp>(op))
      return true;
    if (!mlir::isa<fcc::fabric::SpatialPEOp, fcc::fabric::TemporalPEOp>(op))
      return false;
    return !op->hasAttr("inline_instantiation");
  };

  llvm::StringMap<fcc::fabric::SpatialPEOp> peDefs;
  llvm::StringMap<fcc::fabric::TemporalPEOp> temporalPeDefs;
  llvm::StringMap<fcc::fabric::FunctionUnitOp> functionUnitDefs;
  adgModule->walk([&](fcc::fabric::SpatialPEOp peOp) {
    if (auto symAttr = peOp.getSymNameAttr();
        symAttr && isDefinitionOp(peOp.getOperation(), symAttr.getValue()))
      peDefs[symAttr.getValue()] = peOp;
  });
  adgModule->walk([&](fcc::fabric::TemporalPEOp peOp) {
    if (auto symAttr = peOp.getSymNameAttr();
        symAttr && isDefinitionOp(peOp.getOperation(), symAttr.getValue()))
      temporalPeDefs[symAttr.getValue()] = peOp;
  });
  adgModule->walk([&](fcc::fabric::FunctionUnitOp fuOp) {
    auto symName = fuOp.getSymNameAttr().getValue();
    if (isDefinitionOp(fuOp.getOperation(), symName))
      functionUnitDefs[symName] = fuOp;
  });

  bool builtTechUnits = false;
  if (auto fabricMod = [&]() -> fcc::fabric::ModuleOp {
        fcc::fabric::ModuleOp found;
        adgModule->walk([&](fcc::fabric::ModuleOp op) {
          if (!found)
            found = op;
        });
        return found;
      }()) {
    auto visitPEFunctionUnits =
        [&](auto peOp, llvm::StringRef peName,
            auto &&visitor) {
          auto &peBody = peOp.getBody().front();
          auto referencedIt = referencedTargetsByBlock.find(&peBody);
          const llvm::DenseSet<llvm::StringRef> *referencedTargets =
              referencedIt != referencedTargetsByBlock.end()
                  ? &referencedIt->second
                  : nullptr;
          for (mlir::Operation &bodyOp : peBody.getOperations()) {
            if (auto fuOp = mlir::dyn_cast<fcc::fabric::FunctionUnitOp>(bodyOp)) {
              llvm::StringRef symName = fuOp.getSymNameAttr().getValue();
              if (!symName.empty() && referencedTargets &&
                  referencedTargets->contains(symName))
                continue;
              visitor(fuOp, symName.str());
              continue;
            }
            auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(bodyOp);
            if (!instOp)
              continue;
            auto fuIt = functionUnitDefs.find(instOp.getModule());
            if (fuIt == functionUnitDefs.end())
              continue;
            visitor(fuIt->second,
                    instOp.getSymName().value_or(instOp.getModule()).str());
          }
        };

    std::map<std::string, unsigned> familyIndexBySignature;
    std::map<std::string, unsigned> demandMaterializedStateCountBySignature;
    std::map<std::string, unsigned> legacyMaterializedStateCountBySignature;
    std::map<std::string, unsigned> aggregatedIndexByKey;
    std::vector<AggregatedMatch> aggregatedMatches;
    llvm::SmallVector<TechMapper::LegacyOracleSampleInfo, 4>
        legacyOracleMissingSampleBuffer;
    const bool runLegacyOracle = []() {
      const char *env = std::getenv("FCC_TECHMAP_RUN_LEGACY_ORACLE");
      return env && std::string(env) == "1";
    }();
    plan.metrics.legacyOracleEnabled = runLegacyOracle;
    const bool requireLegacyOracleSuperset = []() {
      const char *env = std::getenv("FCC_TECHMAP_REQUIRE_LEGACY_ORACLE_SUPERSET");
      return env && std::string(env) == "1";
    }();
    plan.metrics.legacyOracleRequired = requireLegacyOracleSuperset;
    const bool allowLegacyFallback = []() {
      const char *env = std::getenv("FCC_TECHMAP_ENABLE_LEGACY_FALLBACK");
      return env && std::string(env) == "1";
    }();

    auto registerFamily = [&](VariantFamily &&family, IdIndex hwNodeId) -> unsigned {
      auto found = familyIndexBySignature.find(family.signature);
      if (found != familyIndexBySignature.end()) {
        auto &existing = familyList[found->second];
        existing.hwNodeIds.push_back(hwNodeId);
        return found->second;
      }
      family.hwNodeIds.clear();
      family.hwNodeIds.push_back(hwNodeId);
      unsigned index = familyList.size();
      familyIndexBySignature[family.signature] = index;
      familyList.push_back(std::move(family));
      return index;
    };

    auto absorbMatch = [&](unsigned familyIndex, const Match &match,
                           IdIndex hwNodeId, bool fromLegacyFallback) {
      std::string key = buildAggregatedMatchKey(familyIndex, match);
      auto found = aggregatedIndexByKey.find(key);
      if (found == aggregatedIndexByKey.end()) {
        AggregatedMatch aggregated;
        aggregated.familyIndex = familyIndex;
        aggregated.swNodesByOp = match.swNodesByOp;
        aggregated.operandOrderByOp = match.operandOrderByOp;
        aggregated.inputBindings = match.inputBindings;
        aggregated.outputBindings = match.outputBindings;
        aggregated.internalEdges = match.internalEdges;
        aggregated.configFields = match.configFields;
        aggregated.configurable = familyList[familyIndex].configurable;
        aggregated.commutativeSwapCount =
            countCommutativeSwaps(match.operandOrderByOp);
        aggregated.hasDemandOrigin = !fromLegacyFallback;
        aggregated.hasLegacyOrigin = fromLegacyFallback;
        aggregated.hwNodeIds.push_back(hwNodeId);
        aggregatedMatches.push_back(std::move(aggregated));
        aggregatedIndexByKey[key] = aggregatedMatches.size() - 1;
        return;
      }

      auto &aggregated = aggregatedMatches[found->second];
      aggregated.hasDemandOrigin =
          aggregated.hasDemandOrigin || !fromLegacyFallback;
      aggregated.hasLegacyOrigin =
          aggregated.hasLegacyOrigin || fromLegacyFallback;
      if (std::find(aggregated.hwNodeIds.begin(), aggregated.hwNodeIds.end(),
                    hwNodeId) == aggregated.hwNodeIds.end()) {
        aggregated.hwNodeIds.push_back(hwNodeId);
      }
    };

    auto recordMatchesForPE = [&](llvm::StringRef peName,
                                  fcc::fabric::FunctionUnitOp fuOp,
                                  const std::string &fuName) {
      IdIndex hwNodeId = findFunctionUnitNode(adg, peName, fuName);
      if (hwNodeId == INVALID_ID)
        return;
      const Node *hwNode = adg.getNode(hwNodeId);
      if (!hwNode)
        return;

      techmapper_detail::DemandMatchStats demandStats;
      auto demandMatches =
          findDemandDrivenMatchesForFU(dfg, fuOp, hwNode, &demandStats);
      llvm::StringSet<> demandKeysForPE;
      llvm::StringSet<> demandFamilySignaturesForPE;
      llvm::StringSet<> legacyFamilySignaturesForPE;
      plan.metrics.demandCandidateCount += demandMatches.size();
      plan.metrics.structuralStateCount += demandStats.structuralStateCount;
      plan.metrics.structuralStateCacheHitCount +=
          demandStats.structuralStateCacheHitCount;
      plan.metrics.structuralStateCacheMissCount +=
          demandStats.structuralStateCacheMissCount;
      for (auto &familyMatch : demandMatches) {
        if (demandFamilySignaturesForPE.insert(familyMatch.family.signature)
                .second) {
          ++demandMaterializedStateCountBySignature[familyMatch.family.signature];
        }
        unsigned familyIndex =
            registerFamily(std::move(familyMatch.family), hwNodeId);
        familyMatch.match.familyIndex = familyIndex;
        demandKeysForPE.insert(
            buildAggregatedMatchKey(familyIndex, familyMatch.match));
        absorbMatch(familyIndex, familyMatch.match, hwNodeId, false);
      }

      bool usedLegacyFallbackForPE = false;
      if (allowLegacyFallback) {
        llvm::SmallVector<VariantFamily, 8> legacyVariants;
        collectVariantsForFU(fuOp, hwNode, legacyVariants);
        for (auto &variant : legacyVariants) {
          if (!variant.isTechFamily())
            continue;
          if (legacyFamilySignaturesForPE.insert(variant.signature).second) {
            ++legacyMaterializedStateCountBySignature[variant.signature];
          }
          unsigned familyIndex = registerFamily(std::move(variant), hwNodeId);
          auto legacyMatches =
              findMatchesForFamily(dfg, familyList[familyIndex], familyIndex);
          for (const auto &legacyMatch : legacyMatches) {
            std::string legacyKey =
                buildAggregatedMatchKey(familyIndex, legacyMatch);
            if (demandKeysForPE.contains(legacyKey))
              continue;
            absorbMatch(familyIndex, legacyMatch, hwNodeId, true);
            usedLegacyFallbackForPE = true;
          }
        }
        if (usedLegacyFallbackForPE)
          ++plan.metrics.legacyFallbackCount;
      }

      if (!runLegacyOracle)
        return;
      ++plan.metrics.legacyOracleCheckCount;

      llvm::SmallVector<VariantFamily, 8> variants;
      collectVariantsForFU(fuOp, hwNode, variants);
      for (auto &variant : variants) {
        if (!variant.isTechFamily())
          continue;
        if (legacyFamilySignaturesForPE.insert(variant.signature).second) {
          ++legacyMaterializedStateCountBySignature[variant.signature];
        }
        unsigned familyIndex = registerFamily(std::move(variant), hwNodeId);
        auto legacyMatches =
            findMatchesForFamily(dfg, familyList[familyIndex], familyIndex);
        plan.metrics.legacyOracleCandidateCount += legacyMatches.size();
        for (const auto &legacyMatch : legacyMatches) {
          std::string legacyKey = buildAggregatedMatchKey(familyIndex, legacyMatch);
          if (!demandKeysForPE.contains(legacyKey)) {
            ++plan.metrics.legacyOracleMissingCount;
            if (legacyOracleMissingSampleBuffer.size() < 4) {
              TechMapper::LegacyOracleSampleInfo sample;
              sample.key = legacyKey;
              sample.familyIndex = familyIndex;
              if (familyIndex < familyList.size())
                sample.familySignature = familyList[familyIndex].signature;
              sample.hwNodeId = hwNodeId;
              sample.peName = peName.str();
              sample.hwName = fuName;
              legacyOracleMissingSampleBuffer.push_back(std::move(sample));
            }
          }
        }
      }
    };

    for (mlir::Operation &op : fabricMod.getBody().front().getOperations()) {
      if (auto instOp = mlir::dyn_cast<fcc::fabric::InstanceOp>(op)) {
        std::string peName =
            instOp.getSymName().value_or(instOp.getModule()).str();
        auto peIt = peDefs.find(instOp.getModule());
        if (peIt != peDefs.end()) {
          visitPEFunctionUnits(
              peIt->second, peName,
              [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
                recordMatchesForPE(peName, fuOp, fuName);
              });
          continue;
        }
        auto temporalPeIt = temporalPeDefs.find(instOp.getModule());
        if (temporalPeIt != temporalPeDefs.end()) {
          visitPEFunctionUnits(
              temporalPeIt->second, peName,
              [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
                recordMatchesForPE(peName, fuOp, fuName);
              });
        }
        continue;
      }

      if (auto peOp = mlir::dyn_cast<fcc::fabric::SpatialPEOp>(op)) {
        llvm::StringRef peName = peOp.getSymName().value_or("");
        visitPEFunctionUnits(
            peOp, peName,
            [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
              recordMatchesForPE(peName, fuOp, fuName);
            });
        continue;
      }

      if (auto peOp = mlir::dyn_cast<fcc::fabric::TemporalPEOp>(op)) {
        llvm::StringRef peName = peOp.getSymName().value_or("");
        visitPEFunctionUnits(
            peOp, peName,
            [&](fcc::fabric::FunctionUnitOp fuOp, const std::string &fuName) {
              recordMatchesForPE(peName, fuOp, fuName);
            });
      }
    }

    for (auto &family : familyList) {
      std::sort(family.hwNodeIds.begin(), family.hwNodeIds.end());
      family.hwNodeIds.erase(
          std::unique(family.hwNodeIds.begin(), family.hwNodeIds.end()),
          family.hwNodeIds.end());
    }

    std::map<std::string, unsigned> supportClassIds;
    std::map<std::string, unsigned> configClassIds;
    for (auto &aggregated : aggregatedMatches) {
      std::sort(aggregated.hwNodeIds.begin(), aggregated.hwNodeIds.end());
      aggregated.hwNodeIds.erase(
          std::unique(aggregated.hwNodeIds.begin(), aggregated.hwNodeIds.end()),
          aggregated.hwNodeIds.end());
      aggregated.temporal = isTemporalCandidateList(aggregated.hwNodeIds, adg);

      std::string supportKey = aggregated.temporal ? "temporal|" : "spatial|";
      for (IdIndex hwNodeId : aggregated.hwNodeIds) {
        if (supportKey.back() != '|')
          supportKey += ",";
        supportKey += std::to_string(hwNodeId);
      }
      auto supportInserted =
          supportClassIds.emplace(supportKey, supportClassIds.size());
      aggregated.supportClassId = supportInserted.first->second;

      std::string configKey =
          familyList[aggregated.familyIndex].signature + "|" +
          serializeConfigFields(aggregated.configFields);
      auto configInserted =
          configClassIds.emplace(configKey, configClassIds.size());
      aggregated.configClassId = configInserted.first->second;
      aggregated.selectionScore = scoreAggregatedMatch(
          aggregated, familyList[aggregated.familyIndex].hwNodeIds.size());
    }
    plan.metrics.supportClassCount = supportClassIds.size();
    plan.metrics.configClassCount = configClassIds.size();
    auto &supportClasses = TechMapper::allSupportClasses(plan);
    supportClasses.resize(supportClassIds.size());
    for (const auto &entry : supportClassIds) {
      if (entry.second >= supportClasses.size())
        continue;
      auto &info = supportClasses[entry.second];
      info.id = entry.second;
      info.key = entry.first;
    }
    for (const auto &aggregated : aggregatedMatches) {
      if (aggregated.supportClassId >= supportClasses.size())
        continue;
      auto &info = supportClasses[aggregated.supportClassId];
      info.temporal = aggregated.temporal;
      info.kind = aggregated.temporal ? "temporal" : "spatial";
      info.hwNodeIds.assign(aggregated.hwNodeIds.begin(), aggregated.hwNodeIds.end());
      info.peNames.clear();
      for (IdIndex hwNodeId : info.hwNodeIds) {
        const Node *hwNode = adg.getNode(hwNodeId);
        if (!hwNode)
          continue;
        llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
        if (peName.empty())
          continue;
        if (std::find(info.peNames.begin(), info.peNames.end(), peName.str()) ==
            info.peNames.end()) {
          info.peNames.push_back(peName.str());
        }
      }
      std::sort(info.peNames.begin(), info.peNames.end());
      info.capacity = info.hwNodeIds.size();
      info.enforceHardCapacity = !aggregated.temporal;
    }
    for (auto &aggregated : aggregatedMatches) {
      if (aggregated.supportClassId >= supportClasses.size())
        continue;
      aggregated.supportClassCapacity =
          supportClasses[aggregated.supportClassId].capacity;
    }
    auto &configClasses = TechMapper::allConfigClasses(plan);
    configClasses.resize(configClassIds.size());
    for (const auto &entry : configClassIds) {
      if (entry.second >= configClasses.size())
        continue;
      auto &info = configClasses[entry.second];
      info.id = entry.second;
      info.key = entry.first;
      info.reason = "same FU signature and exact config-field binding: " + entry.first;
      info.temporal = false;
      info.compatibleConfigClassIds.clear();
      info.compatibleConfigClassIds.push_back(entry.second);
    }
    for (const auto &aggregated : aggregatedMatches) {
      if (!aggregated.temporal || aggregated.configClassId >= configClasses.size())
        continue;
      configClasses[aggregated.configClassId].temporal = true;
    }
    for (size_t lhs = 0; lhs < configClasses.size(); ++lhs) {
      auto &lhsInfo = configClasses[lhs];
      lhsInfo.compatibleConfigClassIds.clear();
      lhsInfo.compatibleConfigClassIds.push_back(lhs);
      if (lhsInfo.temporal)
        continue;
      for (size_t rhs = 0; rhs < configClasses.size(); ++rhs) {
        if (rhs == lhs)
          continue;
        if (configClasses[rhs].temporal)
          continue;
        lhsInfo.compatibleConfigClassIds.push_back(rhs);
      }
      std::sort(lhsInfo.compatibleConfigClassIds.begin(),
                lhsInfo.compatibleConfigClassIds.end());
    }
    for (size_t lhs = 0; lhs < configClasses.size(); ++lhs) {
      if (!configClasses[lhs].temporal)
        continue;
      for (size_t rhs = lhs + 1; rhs < configClasses.size(); ++rhs) {
        if (!configClasses[rhs].temporal)
          continue;
        TechMapper::TemporalIncompatibilityInfo info;
        info.lhsConfigClassId = lhs;
        info.rhsConfigClassId = rhs;
        info.reason =
            "temporal reuse currently requires identical config classes; " +
            std::to_string(lhs) + " and " + std::to_string(rhs) +
            " differ";
        TechMapper::temporalIncompatibilities(plan).push_back(std::move(info));
      }
    }

    llvm::DenseSet<IdIndex> overlapNodes;
    for (const auto &aggregated : aggregatedMatches) {
      for (IdIndex swNodeId : aggregated.swNodesByOp) {
        if (!overlapNodes.insert(swNodeId).second)
          ++plan.metrics.overlapEdgeCount;
      }
    }
    plan.metrics.candidateGenerationTimeMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - candidateGenerationStartTime)
            .count();
    candidateGenerationTimed = true;

    std::vector<unsigned> selectedMatches;
    std::vector<unsigned> matchComponentIds;
    const auto selectionStartTime = std::chrono::steady_clock::now();
    selectMatchesByComponent(aggregatedMatches, adg,
                             &TechMapper::allSelectionComponents(plan),
                             &matchComponentIds,
                             &plan.metrics,
                             selectedMatches);
    plan.metrics.selectionTimeMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - selectionStartTime)
            .count();
    selectionTimed = true;
    llvm::DenseSet<unsigned> selectedMatchSet;
    for (unsigned selectedIdx : selectedMatches)
      selectedMatchSet.insert(selectedIdx);
    auto &familyTechInfos = TechMapper::allFamilyTechInfos(plan);
    familyTechInfos.clear();
    familyTechInfos.resize(familyList.size());
    for (unsigned familyIndex = 0; familyIndex < familyList.size();
         ++familyIndex) {
      auto &info = familyTechInfos[familyIndex];
      info.familyIndex = familyIndex;
      info.signature = familyList[familyIndex].signature;
      info.hwSupportCount = familyList[familyIndex].hwNodeIds.size();
      info.materializedStateCount =
          demandMaterializedStateCountBySignature[info.signature];
      info.legacyMaterializedStateCount =
          legacyMaterializedStateCountBySignature[info.signature];
      info.opCount = familyList[familyIndex].ops.size();
      info.configurable = familyList[familyIndex].configurable;
    }
    for (const auto &aggregated : aggregatedMatches) {
      auto *info = TechMapper::findFamilyTechInfo(plan, aggregated.familyIndex);
      if (!info)
        continue;
      ++info->matchCount;
      info->maxFusionSize =
          std::max<unsigned>(info->maxFusionSize, aggregated.swNodesByOp.size());
    }
    auto &candidateSummaries = TechMapper::allCandidateSummaries(plan);
    candidateSummaries.clear();
    candidateSummaries.reserve(aggregatedMatches.size());
    llvm::DenseSet<IdIndex> legacyDerivedHwNodeIds;
    for (unsigned matchIdx = 0; matchIdx < aggregatedMatches.size(); ++matchIdx) {
      const auto &aggregated = aggregatedMatches[matchIdx];
      TechMapper::CandidateSummaryInfo summary;
      summary.id = matchIdx;
      summary.familyIndex = aggregated.familyIndex;
      if (aggregated.familyIndex < familyList.size())
        summary.familySignature = familyList[aggregated.familyIndex].signature;
      if (matchIdx < matchComponentIds.size())
        summary.selectionComponentId = matchComponentIds[matchIdx];
      summary.swNodeIds.assign(aggregated.swNodesByOp.begin(),
                               aggregated.swNodesByOp.end());
      summary.internalEdgeIds.assign(aggregated.internalEdges.begin(),
                                     aggregated.internalEdges.end());
      summary.inputBindings.assign(aggregated.inputBindings.begin(),
                                   aggregated.inputBindings.end());
      summary.outputBindings.assign(aggregated.outputBindings.begin(),
                                    aggregated.outputBindings.end());
      summary.hwNodeIds.assign(aggregated.hwNodeIds.begin(),
                               aggregated.hwNodeIds.end());
      for (IdIndex hwNodeId : summary.hwNodeIds) {
        const Node *hwNode = adg.getNode(hwNodeId);
        if (!hwNode)
          continue;
        llvm::StringRef peName = getNodeAttrStr(hwNode, "pe_name");
        if (peName.empty())
          continue;
        if (std::find(summary.peNames.begin(), summary.peNames.end(),
                      peName.str()) == summary.peNames.end()) {
          summary.peNames.push_back(peName.str());
        }
      }
      std::sort(summary.peNames.begin(), summary.peNames.end());
      summary.supportClassId = aggregated.supportClassId;
      summary.supportClassCapacity = aggregated.supportClassCapacity;
      if (const auto *supportClassInfo =
              TechMapper::findSupportClass(plan, aggregated.supportClassId)) {
        summary.supportClassKey = supportClassInfo->key;
      }
      summary.configClassId = aggregated.configClassId;
      if (const auto *configClassInfo =
              TechMapper::findConfigClass(plan, aggregated.configClassId)) {
        summary.configClassKey = configClassInfo->key;
        summary.configClassReason = configClassInfo->reason;
      }
      summary.temporal = aggregated.temporal;
      summary.configurable = aggregated.configurable;
      summary.baseSelectionScore = aggregated.selectionScore;
      summary.candidatePenalty = 0;
      summary.familyPenalty = 0;
      summary.configClassPenalty = 0;
      summary.selectionScore = aggregated.selectionScore;
      applyCandidateSelectionOutcome(
          plan, summary, selectedMatchSet.contains(matchIdx),
          inferCandidateStatus(matchIdx, aggregatedMatches, selectedMatches,
                               matchComponentIds, adg));
      summary.demandOrigin = aggregated.hasDemandOrigin;
      summary.legacyFallbackOrigin =
          aggregated.hasLegacyOrigin && !aggregated.hasDemandOrigin;
      summary.mixedOrigin =
          aggregated.hasDemandOrigin && aggregated.hasLegacyOrigin;
      accumulateLegacyDerivedCandidateMetrics(
          plan, summary.legacyFallbackOrigin, summary.mixedOrigin,
          aggregated.hwNodeIds, legacyDerivedHwNodeIds);
      summary.configFields.assign(aggregated.configFields.begin(),
                                  aggregated.configFields.end());
      candidateSummaries.push_back(std::move(summary));
    }
    plan.metrics.legacyDerivedSourceCount = legacyDerivedHwNodeIds.size();

    llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> nodeInfoBySwNode;
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        nodeSupportClassesBySwNode;
    llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
        nodeConfigClassesBySwNode;
    for (unsigned matchIdx = 0; matchIdx < aggregatedMatches.size(); ++matchIdx) {
      const auto &aggregated = aggregatedMatches[matchIdx];
      accumulateNodeTechCandidateCoverage(
          nodeInfoBySwNode, nodeSupportClassesBySwNode,
          nodeConfigClassesBySwNode, aggregated, matchIdx,
          matchIdx < matchComponentIds.size()
              ? std::optional<unsigned>(matchComponentIds[matchIdx])
              : std::nullopt);
    }

    for (const auto &aggregated : aggregatedMatches) {
      if (aggregated.swNodesByOp.size() != 1)
        continue;
      IdIndex swNodeId = aggregated.swNodesByOp.front();
      accumulateConservativeFallbackCandidate(plan, swNodeId, aggregated);
    }
    rebuildPreferredConservativeFallbackCandidates(plan);
    finalizeNodeTechCoverageSummaries(nodeInfoBySwNode,
                                      nodeSupportClassesBySwNode,
                                      nodeConfigClassesBySwNode);

    std::vector<int> nodeToUnit(dfg.nodes.size(), -1);
    unsigned techNodeCount = 0;
    llvm::DenseSet<unsigned> selectedConfigClasses;
    for (unsigned selectedIdx : selectedMatches) {
      const auto &aggregated = aggregatedMatches[selectedIdx];
      TechMapper::Unit unit = buildSelectedUnitFromAggregatedMatch(
          aggregated, selectedIdx,
          selectedIdx < matchComponentIds.size()
              ? std::optional<unsigned>(matchComponentIds[selectedIdx])
              : std::nullopt);
      int unitIndex = static_cast<int>(TechMapper::allUnits(plan).size());
      for (IdIndex swNodeId : unit.swNodes) {
        if (swNodeId < nodeToUnit.size()) {
          nodeToUnit[swNodeId] = unitIndex;
          ++techNodeCount;
        }
      }
      registerSelectedUnit(plan, aggregated, unit, unitIndex, selectedIdx,
                           selectedConfigClasses, nodeInfoBySwNode);
      TechMapper::allUnits(plan).push_back(std::move(unit));
    }
    plan.metrics.selectedConfigDiversityCount = selectedConfigClasses.size();
    plan.metrics.selectedCandidateCount = selectedMatches.size();
    sortSelectedUnitIndices(plan);
    if (totalOpCount > techNodeCount)
      plan.metrics.conservativeFallbackCount = totalOpCount - techNodeCount;

    llvm::DenseSet<IdIndex> candidateCoveredNodes;
    llvm::DenseSet<IdIndex> selectedCoveredNodes;
    collectCoveredNodes(aggregatedMatches, selectedMatches,
                        candidateCoveredNodes, selectedCoveredNodes);
    for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
         ++swNodeId) {
      const Node *swNode = dfg.getNode(swNodeId);
      if (!swNode || swNode->kind != Node::OperationNode)
        continue;
      if (selectedCoveredNodes.contains(swNodeId))
        continue;
      TechMapper::FallbackNodeInfo fallbackInfo;
      populateFallbackNodeSummary(plan, swNodeId, nodeInfoBySwNode,
                                  fallbackInfo);
      if (candidateCoveredNodes.contains(swNodeId)) {
        ++plan.metrics.fallbackRejectedCount;
        fallbackInfo.reason =
            inferRejectedReason(swNodeId, aggregatedMatches, selectedMatches,
                                matchComponentIds, adg);
      } else {
        ++plan.metrics.fallbackNoCandidateCount;
        fallbackInfo.reason = "no_candidate";
      }
      TechMapper::allFallbackNodes(plan).push_back(std::move(fallbackInfo));
      auto infoIt = nodeInfoBySwNode.find(swNodeId);
      if (infoIt != nodeInfoBySwNode.end()) {
        markNodeAsConservativeFallback(
            infoIt->second,
            candidateCoveredNodes.contains(swNodeId)
                ? inferRejectedReason(swNodeId, aggregatedMatches,
                                      selectedMatches, matchComponentIds, adg)
                : llvm::StringRef("fallback_no_candidate"));
      }
    }
    auto &nodeTechInfos = TechMapper::allNodeTechInfos(plan);
    for (auto &entry : nodeInfoBySwNode)
      nodeTechInfos.push_back(entry.second);
    std::sort(nodeTechInfos.begin(), nodeTechInfos.end(),
              [](const TechMapper::NodeTechInfo &lhs,
                 const TechMapper::NodeTechInfo &rhs) {
                return lhs.swNodeId < rhs.swNodeId;
              });

    if (totalOpCount > 0)
      plan.coverageScore = static_cast<double>(techNodeCount) /
                           static_cast<double>(totalOpCount);
    plan.metrics.coverageScore = plan.coverageScore;

    if (runLegacyOracle && plan.metrics.legacyOracleMissingCount != 0) {
      auto &oracleMissingSamples = TechMapper::legacyOracleMissingSamples(plan);
      oracleMissingSamples.assign(legacyOracleMissingSampleBuffer.begin(),
                                  legacyOracleMissingSampleBuffer.end());
      plan.diagnostics =
          "techmap legacy oracle found missing demand-driven candidates: " +
          std::to_string(plan.metrics.legacyOracleMissingCount);
      if (!oracleMissingSamples.empty()) {
        plan.diagnostics += " samples=";
        for (size_t idx = 0; idx < oracleMissingSamples.size(); ++idx) {
          if (idx != 0)
            plan.diagnostics += " || ";
          plan.diagnostics += oracleMissingSamples[idx].key;
        }
      }
      if (requireLegacyOracleSuperset)
        return false;
    }
    builtTechUnits = true;
  }
  if (!builtTechUnits) {
    unsigned totalOpCount = 0;
    for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
         ++swNodeId) {
      const Node *swNode = dfg.getNode(swNodeId);
      if (!swNode || swNode->kind != Node::OperationNode)
        continue;
      ++totalOpCount;
      TechMapper::NodeTechInfo info;
      info.swNodeId = swNodeId;
      info.candidateCount = 0;
      info.supportClassCount = 0;
      info.configClassCount = 0;
      info.maxFusionSize = 0;
      info.selected = false;
      info.selectedAsFusion = false;
      info.conservativeFallback = true;
      info.status = "fallback_no_candidate";
      TechMapper::allNodeTechInfos(plan).push_back(std::move(info));

      TechMapper::FallbackNodeInfo fallbackInfo;
      fallbackInfo.swNodeId = swNodeId;
      fallbackInfo.reason = "no_candidate";
      TechMapper::allFallbackNodes(plan).push_back(std::move(fallbackInfo));
      ++plan.metrics.fallbackNoCandidateCount;
    }
    plan.metrics.conservativeFallbackCount = totalOpCount;
    if (totalOpCount > 0)
      plan.coverageScore = 0.0;
    auto &nodeTechInfos = TechMapper::allNodeTechInfos(plan);
    std::sort(nodeTechInfos.begin(), nodeTechInfos.end(),
              [](const TechMapper::NodeTechInfo &lhs,
                 const TechMapper::NodeTechInfo &rhs) {
                return lhs.swNodeId < rhs.swNodeId;
              });
  }
  plan.metrics.coverageScore = plan.coverageScore;
  if (!candidateGenerationTimed) {
    plan.metrics.candidateGenerationTimeMicros =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - candidateGenerationStartTime)
            .count();
  }
  if (!selectionTimed)
    plan.metrics.selectionTimeMicros = 0;
  plan.metrics.totalLayer2TimeMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - layer2StartTime)
          .count();
  if (plan.diagnostics.empty())
    plan.diagnostics = buildTechmapDiagnostics(plan);
  else
    plan.diagnostics += "; " + buildTechmapDiagnostics(plan);
  return finalizePlanGraphs(dfg, adg, plan);
}

bool TechMapper::applyFeedback(const Graph &dfg, const Graph &adg,
                               const Plan &seedPlan,
                               const Feedback &feedback, Plan &plan) {
  const auto reselectionStartTime = std::chrono::steady_clock::now();
  plan = seedPlan;
  plan.contractedDFG = Graph(dfg.context);
  plan.conservativeFallbackDFG = dfg.clone();
  plan.contractedCandidates.clear();
  plan.contractedCandidateSupportClasses.clear();
  plan.contractedCandidateConfigClasses.clear();
  plan.conservativeFallbackCandidates.clear();
  plan.conservativeFallbackCandidateSupportClasses.clear();
  plan.conservativeFallbackCandidateConfigClasses.clear();
  plan.conservativeFallbackCandidateDetails.clear();
  plan.conservativeFallbackPreferredCandidate.clear();
  TechMapper::allFallbackNodes(plan).clear();
  TechMapper::allNodeTechInfos(plan).clear();
  TechMapper::allSelectionComponents(plan).clear();
  TechMapper::temporalIncompatibilities(plan).assign(
      TechMapper::temporalIncompatibilities(seedPlan).begin(),
      TechMapper::temporalIncompatibilities(seedPlan).end());
  TechMapper::conservativeFallbackSwNodes(plan).clear();
  TechMapper::allUnits(plan).clear();
  plan.originalNodeToContractedNode.assign(dfg.nodes.size(), INVALID_ID);
  plan.originalPortToContractedPort.assign(dfg.ports.size(), INVALID_ID);
  plan.originalEdgeToContractedEdge.assign(dfg.edges.size(), INVALID_ID);
  plan.originalEdgeKinds.assign(dfg.edges.size(), TechMappedEdgeKind::Routed);
  TechMapper::markFeedbackApplied(plan, feedback);
  auto &feedbackResolution = TechMapper::feedbackResolution(plan);
  resetSelectionMetrics(plan.metrics);
  restoreReselectionBaselineMetrics(plan.metrics, seedPlan.metrics);
  plan.metrics.feedbackReselectionCount =
      seedPlan.metrics.feedbackReselectionCount + 1;
  plan.metrics.feedbackPenaltyCount = 0;
  for (auto &familyInfo : TechMapper::allFamilyTechInfos(plan))
    familyInfo.selectedCount = 0;
  for (auto &summary : TechMapper::allCandidateSummaries(plan)) {
    summary.selected = false;
    summary.selectionComponentId = std::numeric_limits<unsigned>::max();
    summary.selectedUnitIndex = std::numeric_limits<unsigned>::max();
    summary.contractedNodeId = INVALID_ID;
    summary.status.clear();
    summary.candidatePenalty = 0;
    summary.familyPenalty = 0;
    summary.configClassPenalty = 0;
    summary.selectionScore = summary.baseSelectionScore;
  }

  llvm::DenseSet<unsigned> bannedCandidateIds;
  llvm::DenseSet<unsigned> bannedFamilyIds;
  llvm::DenseSet<unsigned> bannedConfigClassIds;
  llvm::DenseSet<unsigned> splitCandidateIds;
  llvm::SmallVector<TechMapper::WeightedIdPenalty, 8> validCandidatePenalties;
  llvm::SmallVector<TechMapper::WeightedIdPenalty, 8> validFamilyPenalties;
  llvm::SmallVector<TechMapper::WeightedIdPenalty, 8>
      validConfigClassPenalties;
  llvm::ArrayRef<TechMapper::CandidateSummaryInfo> seedCandidateSummaries =
      TechMapper::allCandidateSummaries(seedPlan);
  llvm::ArrayRef<TechMapper::FamilyTechInfo> seedFamilyTechInfos =
      TechMapper::allFamilyTechInfos(seedPlan);
  llvm::ArrayRef<TechMapper::ConfigClassInfo> seedConfigClasses =
      TechMapper::allConfigClasses(seedPlan);
  for (unsigned id : feedback.bannedCandidateIds) {
    if (id < seedCandidateSummaries.size()) {
      bannedCandidateIds.insert(id);
      continue;
    }
    feedbackResolution.unknownBannedCandidateIds.push_back(id);
    ++plan.metrics.feedbackUnknownCandidateRefCount;
  }
  for (unsigned id : feedback.bannedFamilyIds) {
    if (id < seedFamilyTechInfos.size()) {
      bannedFamilyIds.insert(id);
      continue;
    }
    feedbackResolution.unknownBannedFamilyIds.push_back(id);
    ++plan.metrics.feedbackUnknownFamilyRefCount;
  }
  for (unsigned id : feedback.bannedConfigClassIds) {
    if (id < seedConfigClasses.size()) {
      bannedConfigClassIds.insert(id);
      continue;
    }
    feedbackResolution.unknownBannedConfigClassIds.push_back(id);
    ++plan.metrics.feedbackUnknownConfigClassRefCount;
  }
  for (unsigned id : feedback.splitCandidateIds) {
    if (id < seedCandidateSummaries.size()) {
      splitCandidateIds.insert(id);
      continue;
    }
    feedbackResolution.unknownSplitCandidateIds.push_back(id);
    ++plan.metrics.feedbackUnknownCandidateRefCount;
  }
  for (const auto &penalty : feedback.candidatePenalties) {
    if (penalty.id < seedCandidateSummaries.size()) {
      validCandidatePenalties.push_back(penalty);
      continue;
    }
    feedbackResolution.unknownCandidatePenalties.push_back(penalty);
    ++plan.metrics.feedbackUnknownCandidateRefCount;
  }
  for (const auto &penalty : feedback.familyPenalties) {
    if (penalty.id < seedFamilyTechInfos.size()) {
      validFamilyPenalties.push_back(penalty);
      continue;
    }
    feedbackResolution.unknownFamilyPenalties.push_back(penalty);
    ++plan.metrics.feedbackUnknownFamilyRefCount;
  }
  for (const auto &penalty : feedback.configClassPenalties) {
    if (penalty.id < seedConfigClasses.size()) {
      validConfigClassPenalties.push_back(penalty);
      continue;
    }
    feedbackResolution.unknownConfigClassPenalties.push_back(penalty);
    ++plan.metrics.feedbackUnknownConfigClassRefCount;
  }
  plan.metrics.feedbackPenaltyCount = validCandidatePenalties.size() +
                                      validFamilyPenalties.size() +
                                      validConfigClassPenalties.size();

  std::vector<AggregatedMatch> filteredMatches;
  std::vector<unsigned> filteredCandidateIds;
  std::vector<unsigned> cachedComponentIds;
  llvm::DenseSet<IdIndex> legacyFallbackHwNodeIds;
  llvm::DenseSet<IdIndex> legacyDerivedHwNodeIds;
  filteredMatches.reserve(seedCandidateSummaries.size());
  filteredCandidateIds.reserve(seedCandidateSummaries.size());
  cachedComponentIds.reserve(seedCandidateSummaries.size());

  for (const auto &seedSummary : seedCandidateSummaries) {
    auto *summary = TechMapper::findCandidateSummary(plan, seedSummary.id);
    if (!summary)
      continue;
    if (bannedCandidateIds.contains(seedSummary.id)) {
      markFeedbackFilteredCandidate(plan, *summary, "feedback_banned_candidate");
      continue;
    }
    if (splitCandidateIds.contains(seedSummary.id)) {
      markFeedbackFilteredCandidate(plan, *summary, "feedback_split_requested");
      continue;
    }
    if (bannedFamilyIds.contains(seedSummary.familyIndex)) {
      markFeedbackFilteredCandidate(plan, *summary, "feedback_banned_family");
      continue;
    }
    if (bannedConfigClassIds.contains(seedSummary.configClassId)) {
      markFeedbackFilteredCandidate(plan, *summary,
                                    "feedback_banned_config_class");
      continue;
    }

    AggregatedMatch match = buildAggregatedMatchFromSummary(seedSummary);
    int64_t candidatePenalty =
        lookupPenalty(validCandidatePenalties, seedSummary.id);
    int64_t familyPenalty =
        lookupPenalty(validFamilyPenalties, seedSummary.familyIndex);
    int64_t configClassPenalty =
        lookupPenalty(validConfigClassPenalties, seedSummary.configClassId);
    match.selectionScore -= candidatePenalty;
    match.selectionScore -= familyPenalty;
    match.selectionScore -= configClassPenalty;
    summary->baseSelectionScore = seedSummary.baseSelectionScore;
    summary->candidatePenalty = candidatePenalty;
    summary->familyPenalty = familyPenalty;
    summary->configClassPenalty = configClassPenalty;
    summary->selectionScore = match.selectionScore;
    accumulateLegacyDerivedCandidateMetrics(
        plan, seedSummary.legacyFallbackOrigin, seedSummary.mixedOrigin,
        match.hwNodeIds, legacyDerivedHwNodeIds);
    if (seedSummary.legacyFallbackOrigin) {
      for (IdIndex hwNodeId : match.hwNodeIds)
        legacyFallbackHwNodeIds.insert(hwNodeId);
    }
    filteredMatches.push_back(std::move(match));
    filteredCandidateIds.push_back(seedSummary.id);
    cachedComponentIds.push_back(seedSummary.selectionComponentId);
  }
  plan.metrics.legacyFallbackCount = legacyFallbackHwNodeIds.size();
  plan.metrics.legacyDerivedSourceCount = legacyDerivedHwNodeIds.size();

  std::vector<unsigned> selectedFilteredMatches;
  std::vector<unsigned> filteredMatchComponentIds;
  const auto selectionStartTime = std::chrono::steady_clock::now();
  selectMatchesByCachedComponents(filteredMatches, filteredCandidateIds,
                                  cachedComponentIds,
                                  TechMapper::allSelectionComponents(seedPlan).size(),
                                  adg,
                                  &TechMapper::allSelectionComponents(plan),
                                  &filteredMatchComponentIds, &plan.metrics,
                                  selectedFilteredMatches);
  llvm::ArrayRef<TechMapper::SelectionComponentInfo> seedSelectionComponents =
      TechMapper::allSelectionComponents(seedPlan);
  if (!seedSelectionComponents.empty()) {
    for (auto &componentInfo : TechMapper::allSelectionComponents(plan)) {
      if (componentInfo.id >= seedSelectionComponents.size())
        continue;
      componentInfo.baseMaxCandidateScore =
          seedSelectionComponents[componentInfo.id].maxCandidateScore;
      componentInfo.baseSelectedScoreSum =
          seedSelectionComponents[componentInfo.id].selectedScoreSum;
      llvm::DenseSet<unsigned> retainedCandidateIds;
      for (unsigned candidateId : componentInfo.candidateIds)
        retainedCandidateIds.insert(candidateId);
      for (unsigned candidateId :
           seedSelectionComponents[componentInfo.id].candidateIds) {
        if (!retainedCandidateIds.contains(candidateId))
          componentInfo.filteredCandidateIds.push_back(candidateId);
      }
      std::sort(componentInfo.filteredCandidateIds.begin(),
                componentInfo.filteredCandidateIds.end());
    }
  }
  plan.metrics.selectionTimeMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - selectionStartTime)
          .count();

  llvm::DenseSet<unsigned> selectedFilteredSet;
  for (unsigned selectedIdx : selectedFilteredMatches)
    selectedFilteredSet.insert(selectedIdx);
  for (unsigned filteredIdx = 0; filteredIdx < filteredMatches.size();
       ++filteredIdx) {
    unsigned candidateId = filteredCandidateIds[filteredIdx];
    auto *summary = TechMapper::findCandidateSummary(plan, candidateId);
    if (!summary)
      continue;
    if (filteredIdx < filteredMatchComponentIds.size())
      summary->selectionComponentId = filteredMatchComponentIds[filteredIdx];
    applyCandidateSelectionOutcome(
        plan, *summary, selectedFilteredSet.contains(filteredIdx),
        selectedFilteredSet.contains(filteredIdx)
            ? llvm::StringRef("selected")
            : llvm::StringRef(inferCandidateStatus(
                  filteredIdx, filteredMatches, selectedFilteredMatches,
                  filteredMatchComponentIds, adg)));
  }

  llvm::DenseMap<IdIndex, TechMapper::NodeTechInfo> nodeInfoBySwNode;
  llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
      nodeSupportClassesBySwNode;
  llvm::DenseMap<IdIndex, llvm::SmallVector<unsigned, 4>>
      nodeConfigClassesBySwNode;
  for (const auto &seedInfo : TechMapper::allNodeTechInfos(seedPlan)) {
    auto &info = nodeInfoBySwNode[seedInfo.swNodeId];
    info.swNodeId = seedInfo.swNodeId;
    info.contractedNodeId = INVALID_ID;
    info.selectionComponentId = std::numeric_limits<unsigned>::max();
    info.selectedUnitIndex = std::numeric_limits<unsigned>::max();
    info.selectedCandidateId = std::numeric_limits<unsigned>::max();
    info.candidateCount = 0;
    info.supportClassCount = 0;
    info.configClassCount = 0;
    info.maxFusionSize = 0;
    info.candidateIds.clear();
    info.supportClassIds.clear();
    info.configClassIds.clear();
    info.selected = false;
    info.selectedAsFusion = false;
    info.conservativeFallback = true;
    info.selectedFromLegacyFallback = false;
    info.selectedFromDemand = false;
    info.selectedFromMixedOrigin = false;
    info.status = "feedback_filtered_out";
  }

  auto hadSeedCandidates = [&](IdIndex swNodeId) {
    const auto *seedInfo = TechMapper::findNodeTechInfo(seedPlan, swNodeId);
    return seedInfo && seedInfo->candidateCount != 0;
  };

  for (unsigned filteredIdx = 0; filteredIdx < filteredMatches.size();
       ++filteredIdx) {
    const auto &aggregated = filteredMatches[filteredIdx];
    unsigned candidateId = filteredCandidateIds[filteredIdx];
    accumulateNodeTechCandidateCoverage(
        nodeInfoBySwNode, nodeSupportClassesBySwNode,
        nodeConfigClassesBySwNode, aggregated, candidateId,
        filteredIdx < filteredMatchComponentIds.size()
            ? std::optional<unsigned>(filteredMatchComponentIds[filteredIdx])
            : std::nullopt);
  }

  for (const auto &aggregated : filteredMatches) {
    if (aggregated.swNodesByOp.size() != 1)
      continue;
    IdIndex swNodeId = aggregated.swNodesByOp.front();
    accumulateConservativeFallbackCandidate(plan, swNodeId, aggregated);
  }
  rebuildPreferredConservativeFallbackCandidates(plan);
  finalizeNodeTechCoverageSummaries(nodeInfoBySwNode,
                                    nodeSupportClassesBySwNode,
                                    nodeConfigClassesBySwNode);

  unsigned techNodeCount = 0;
  llvm::DenseSet<unsigned> selectedConfigClasses;
  for (unsigned selectedFilteredIdx : selectedFilteredMatches) {
    const auto &aggregated = filteredMatches[selectedFilteredIdx];
    unsigned selectedCandidateId = filteredCandidateIds[selectedFilteredIdx];
    TechMapper::Unit unit = buildSelectedUnitFromAggregatedMatch(
        aggregated, selectedCandidateId,
        selectedFilteredIdx < filteredMatchComponentIds.size()
            ? std::optional<unsigned>(filteredMatchComponentIds[selectedFilteredIdx])
            : std::nullopt);
    int unitIndex = static_cast<int>(TechMapper::allUnits(plan).size());
    techNodeCount += unit.swNodes.size();
    registerSelectedUnit(plan, aggregated, unit, unitIndex,
                         selectedCandidateId, selectedConfigClasses,
                         nodeInfoBySwNode);
    TechMapper::allUnits(plan).push_back(std::move(unit));
  }

  plan.metrics.selectedConfigDiversityCount = selectedConfigClasses.size();
  plan.metrics.selectedCandidateCount = selectedFilteredMatches.size();
  sortSelectedUnitIndices(plan);

  llvm::DenseSet<IdIndex> candidateCoveredNodes;
  llvm::DenseSet<IdIndex> selectedCoveredNodes;
  collectCoveredNodes(filteredMatches, selectedFilteredMatches,
                      candidateCoveredNodes, selectedCoveredNodes);

  unsigned totalOpCount = 0;
  for (IdIndex swNodeId = 0; swNodeId < static_cast<IdIndex>(dfg.nodes.size());
       ++swNodeId) {
    const Node *swNode = dfg.getNode(swNodeId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;
    ++totalOpCount;
    if (selectedCoveredNodes.contains(swNodeId))
      continue;
    TechMapper::FallbackNodeInfo fallbackInfo;
    populateFallbackNodeSummary(plan, swNodeId, nodeInfoBySwNode,
                                fallbackInfo);
    if (candidateCoveredNodes.contains(swNodeId)) {
      ++plan.metrics.fallbackRejectedCount;
      fallbackInfo.reason =
          inferRejectedReason(swNodeId, filteredMatches, selectedFilteredMatches,
                              filteredMatchComponentIds, adg);
    } else {
      ++plan.metrics.fallbackNoCandidateCount;
      fallbackInfo.reason =
          hadSeedCandidates(swNodeId) ? "feedback_filtered_out" : "no_candidate";
    }
    TechMapper::allFallbackNodes(plan).push_back(std::move(fallbackInfo));
    auto infoIt = nodeInfoBySwNode.find(swNodeId);
    if (infoIt != nodeInfoBySwNode.end() && !infoIt->second.selected) {
      markNodeAsConservativeFallback(
          infoIt->second,
          candidateCoveredNodes.contains(swNodeId)
              ? inferRejectedReason(swNodeId, filteredMatches,
                                    selectedFilteredMatches,
                                    filteredMatchComponentIds, adg)
              : (hadSeedCandidates(swNodeId)
                     ? llvm::StringRef("feedback_filtered_out")
                     : llvm::StringRef("fallback_no_candidate")));
    }
  }
  if (totalOpCount > techNodeCount)
    plan.metrics.conservativeFallbackCount = totalOpCount - techNodeCount;

  for (auto &entry : nodeInfoBySwNode)
    TechMapper::allNodeTechInfos(plan).push_back(entry.second);
  auto &nodeTechInfos = TechMapper::allNodeTechInfos(plan);
  std::sort(nodeTechInfos.begin(), nodeTechInfos.end(),
            [](const TechMapper::NodeTechInfo &lhs,
               const TechMapper::NodeTechInfo &rhs) {
              return lhs.swNodeId < rhs.swNodeId;
            });

  plan.coverageScore =
      totalOpCount == 0 ? 1.0
                        : static_cast<double>(techNodeCount) /
                              static_cast<double>(totalOpCount);
  plan.metrics.coverageScore = plan.coverageScore;
  plan.metrics.totalLayer2TimeMicros =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - reselectionStartTime)
          .count();
  std::string reselectionDiagnostics =
      "techmap cached reselection applied: filtered_candidates=" +
      std::to_string(plan.metrics.feedbackFilteredCandidateCount) +
      ", penalty_terms=" + std::to_string(plan.metrics.feedbackPenaltyCount);
  if (TechMapper::feedbackUnknownCandidateRefCount(plan) != 0 ||
      TechMapper::feedbackUnknownFamilyRefCount(plan) != 0 ||
      TechMapper::feedbackUnknownConfigClassRefCount(plan) != 0) {
    reselectionDiagnostics +=
        ", unresolved(candidate=" +
        std::to_string(TechMapper::feedbackUnknownCandidateRefCount(plan)) +
        ", family=" +
        std::to_string(TechMapper::feedbackUnknownFamilyRefCount(plan)) +
        ", config=" +
        std::to_string(TechMapper::feedbackUnknownConfigClassRefCount(plan)) +
        ")";
  }
  plan.diagnostics = reselectionDiagnostics;
  plan.diagnostics += "; " + buildTechmapDiagnostics(plan);
  return finalizePlanGraphs(dfg, adg, plan);
}

bool TechMapper::expandPlanMapping(
    const Graph &originalDfg, const Graph &adg, const Plan &plan,
    const MappingState &contractedState, MappingState &expandedState,
    llvm::SmallVectorImpl<FUConfigSelection> &fuConfigs) {
  expandedState.init(originalDfg, adg);
  expandedState.hwNodeFifoBypassedOverride =
      contractedState.hwNodeFifoBypassedOverride;
  fuConfigs.clear();

  for (IdIndex swNodeId = 0;
       swNodeId < static_cast<IdIndex>(plan.originalNodeToContractedNode.size());
       ++swNodeId) {
    IdIndex contractedNodeId = plan.originalNodeToContractedNode[swNodeId];
    if (contractedNodeId == INVALID_ID ||
        contractedNodeId >= contractedState.swNodeToHwNode.size())
      continue;
    IdIndex hwNodeId = contractedState.swNodeToHwNode[contractedNodeId];
    if (hwNodeId == INVALID_ID)
      continue;
    expandedState.swNodeToHwNode[swNodeId] = hwNodeId;
    expandedState.hwNodeToSwNodes[hwNodeId].push_back(swNodeId);
  }

  for (IdIndex swPortId = 0;
       swPortId < static_cast<IdIndex>(plan.originalPortToContractedPort.size());
       ++swPortId) {
    IdIndex contractedPortId = plan.originalPortToContractedPort[swPortId];
    if (contractedPortId == INVALID_ID ||
        contractedPortId >= contractedState.swPortToHwPort.size())
      continue;
    IdIndex hwPortId = contractedState.swPortToHwPort[contractedPortId];
    if (hwPortId == INVALID_ID)
      continue;
    expandedState.swPortToHwPort[swPortId] = hwPortId;
    expandedState.hwPortToSwPorts[hwPortId].push_back(swPortId);
  }

  for (IdIndex swEdgeId = 0;
       swEdgeId < static_cast<IdIndex>(plan.originalEdgeToContractedEdge.size());
       ++swEdgeId) {
    if (plan.originalEdgeKinds[swEdgeId] == TechMappedEdgeKind::IntraFU)
      continue;
    IdIndex contractedEdgeId = plan.originalEdgeToContractedEdge[swEdgeId];
    if (contractedEdgeId == INVALID_ID ||
        contractedEdgeId >= contractedState.swEdgeToHwPaths.size())
      continue;
    expandedState.swEdgeToHwPaths[swEdgeId] =
        contractedState.swEdgeToHwPaths[contractedEdgeId];
    llvm::ArrayRef<IdIndex> path = expandedState.swEdgeToHwPaths[swEdgeId];
    for (size_t i = 0; i + 1 < path.size(); i += 2) {
      IdIndex outPort = path[i];
      IdIndex inPort = path[i + 1];
      const Port *hwOut = adg.getPort(outPort);
      if (!hwOut)
        continue;
      for (IdIndex hwEdgeId : hwOut->connectedEdges) {
        const Edge *hwEdge = adg.getEdge(hwEdgeId);
        if (!hwEdge)
          continue;
        if (hwEdge->srcPort == outPort && hwEdge->dstPort == inPort) {
          expandedState.hwEdgeToSwEdges[hwEdgeId].push_back(swEdgeId);
          break;
        }
      }
    }
  }

  for (const Unit &unit : TechMapper::allUnits(plan)) {
    if (unit.contractedNodeId == INVALID_ID ||
        unit.contractedNodeId >= contractedState.swNodeToHwNode.size())
      continue;
    IdIndex hwNodeId = contractedState.swNodeToHwNode[unit.contractedNodeId];
    if (hwNodeId == INVALID_ID)
      continue;
    for (const Candidate &candidate : unit.candidates) {
      if (candidate.hwNodeId != hwNodeId || candidate.configFields.empty())
        continue;
      FUConfigSelection selection;
      selection.hwNodeId = hwNodeId;
      selection.supportClassId = candidate.supportClassId;
      selection.configClassId = candidate.configClassId;
      if (const Node *hwNode = adg.getNode(hwNodeId)) {
        selection.hwName = getNodeAttrStr(hwNode, "op_name").str();
        selection.peName = getNodeAttrStr(hwNode, "pe_name").str();
      }
      selection.swNodeIds.append(unit.swNodes.begin(), unit.swNodes.end());
      selection.fields.append(candidate.configFields.begin(),
                              candidate.configFields.end());
      fuConfigs.push_back(std::move(selection));
      break;
    }
  }

  return true;
}

} // namespace fcc

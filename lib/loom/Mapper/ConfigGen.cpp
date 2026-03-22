// ConfigGen.cpp -- ConfigGen::writeMapJson and ConfigGen::writeMapText.

#include "ConfigGenInternal.h"

#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/BridgeBinding.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/OpCompat.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace loom {

namespace {

} // namespace

// ===========================================================================
// ConfigGen::writeMapJson
// ===========================================================================

bool ConfigGen::writeMapJson(const MappingState &state, const Graph &dfg,
                             const Graph &adg, const ADGFlattener &flattener,
                             llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                             llvm::ArrayRef<FUConfigSelection> fuConfigs,
                             const std::string &path, int seed,
                             const TechMapper::Plan *techMapPlan,
                             const TechMapper::PlanMetrics *techMapMetrics,
                             const MapperTimingSummary *timingSummary,
                             const MapperSearchSummary *searchSummary,
                             llvm::StringRef techMapDiagnostics) {
  using namespace configgen_detail;

  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "ConfigGen: cannot open " << path << ": " << ec.message()
                 << "\n";
    return false;
  }

  out << "{\n";
  out << "  \"seed\": " << seed << ",\n";
  out << "  \"techmap\": {\n";
  const std::string stableTechMapDiagnostics =
      sanitizeTechMapDiagnosticsForArtifact(techMapDiagnostics);
  out << "    \"contract_version\": 1";
  out << ",\n    \"selected_plan_kind\": "
      << "\"single_best_plus_conservative_fallback\"";
  out << ",\n    \"coverage_score\": "
      << (techMapMetrics ? techMapMetrics->coverageScore : 1.0);
  if (techMapMetrics) {
    llvm::StringRef legacyOracleStatus = "disabled";
    if (techMapMetrics->legacyOracleEnabled) {
      legacyOracleStatus =
          techMapMetrics->legacyOracleMissingCount == 0 ? "passed"
                                                        : "missing_candidates";
      if (techMapMetrics->legacyOracleRequired &&
          techMapMetrics->legacyOracleMissingCount != 0) {
        legacyOracleStatus = "required_missing_candidates";
      }
    }
    llvm::StringRef layer2HandoffStatus = "not_available";
    llvm::SmallVector<llvm::StringRef, 4> layer2HandoffBlockers;
    bool selectedUsesLegacyDerivedSupport = false;
    bool candidatePoolUsesLegacyDerivedSupport = false;
    bool sourcePoolUsesLegacyDerivedSupport = false;
    if (techMapPlan) {
      layer2HandoffStatus = TechMapper::mapperHandoffStatus(*techMapPlan);
      TechMapper::collectMapperHandoffBlockers(*techMapPlan,
                                               layer2HandoffBlockers);
      selectedUsesLegacyDerivedSupport =
          TechMapper::selectedPlanUsesLegacyDerivedSupport(*techMapPlan);
      candidatePoolUsesLegacyDerivedSupport =
          TechMapper::candidatePoolUsesLegacyDerivedSupport(*techMapPlan);
      sourcePoolUsesLegacyDerivedSupport =
          TechMapper::sourcePoolUsesLegacyDerivedSupport(*techMapPlan);
    }
    out << ",\n    \"total_layer2_time_us\": 0";
    out << ",\n    \"candidate_generation_time_us\": 0";
    out << ",\n    \"selection_time_us\": 0";
    out << ",\n    \"op_alias_pair_count\": "
        << techMapMetrics->opAliasPairCount;
    out << ",\n    \"demand_candidate_count\": "
        << techMapMetrics->demandCandidateCount;
    out << ",\n    \"structural_state_count\": "
        << techMapMetrics->structuralStateCount;
    out << ",\n    \"structural_state_cache_hit_count\": "
        << techMapMetrics->structuralStateCacheHitCount;
    out << ",\n    \"structural_state_cache_miss_count\": "
        << techMapMetrics->structuralStateCacheMissCount;
    out << ",\n    \"selected_candidate_count\": "
        << techMapMetrics->selectedCandidateCount;
    out << ",\n    \"rejected_overlap_candidate_count\": "
        << techMapMetrics->rejectedOverlapCandidateCount;
    out << ",\n    \"rejected_temporal_candidate_count\": "
        << techMapMetrics->rejectedTemporalCandidateCount;
    out << ",\n    \"rejected_support_capacity_candidate_count\": "
        << techMapMetrics->rejectedSupportCapacityCandidateCount;
    out << ",\n    \"rejected_spatial_pool_candidate_count\": "
        << techMapMetrics->rejectedSpatialPoolCandidateCount;
    out << ",\n    \"objective_dropped_candidate_count\": "
        << techMapMetrics->objectiveDroppedCandidateCount;
    out << ",\n    \"conservative_fallback_count\": "
        << techMapMetrics->conservativeFallbackCount;
    out << ",\n    \"overlap_count\": " << techMapMetrics->overlapEdgeCount;
    out << ",\n    \"support_class_count\": "
        << techMapMetrics->supportClassCount;
    out << ",\n    \"config_class_count\": "
        << techMapMetrics->configClassCount;
    out << ",\n    \"temporal_risk_count\": "
        << techMapMetrics->temporalRiskCount;
    out << ",\n    \"selected_fused_op_count\": "
        << techMapMetrics->selectedFusedOpCount;
    out << ",\n    \"selected_internal_edge_count\": "
        << techMapMetrics->selectedInternalEdgeCount;
    out << ",\n    \"selected_candidate_choice_count\": "
        << techMapMetrics->selectedCandidateChoiceCount;
    out << ",\n    \"selected_config_diversity_count\": "
        << techMapMetrics->selectedConfigDiversityCount;
    out << ",\n    \"selected_legacy_fallback_count\": "
        << techMapMetrics->selectedLegacyFallbackCount;
    out << ",\n    \"selected_mixed_origin_count\": "
        << techMapMetrics->selectedMixedOriginCount;
    out << ",\n    \"selected_legacy_derived_count\": "
        << techMapMetrics->selectedLegacyDerivedCount;
    out << ",\n    \"selection_component_count\": "
        << techMapMetrics->selectionComponentCount;
    out << ",\n    \"exact_component_count\": "
        << techMapMetrics->exactComponentCount;
    out << ",\n    \"cpsat_component_count\": "
        << techMapMetrics->cpSatComponentCount;
    out << ",\n    \"greedy_component_count\": "
        << techMapMetrics->greedyComponentCount;
    out << ",\n    \"fallback_no_candidate_count\": "
        << techMapMetrics->fallbackNoCandidateCount;
    out << ",\n    \"fallback_rejected_count\": "
        << techMapMetrics->fallbackRejectedCount;
    out << ",\n    \"conservative_fallback_covered_count\": "
        << techMapMetrics->conservativeFallbackCoveredCount;
    out << ",\n    \"conservative_fallback_missing_count\": "
        << techMapMetrics->conservativeFallbackMissingCount;
    out << ",\n    \"legacy_oracle_enabled\": "
        << (techMapMetrics->legacyOracleEnabled ? "true" : "false");
    out << ",\n    \"legacy_oracle_required\": "
        << (techMapMetrics->legacyOracleRequired ? "true" : "false");
    out << ",\n    \"legacy_oracle_check_count\": "
        << techMapMetrics->legacyOracleCheckCount;
    out << ",\n    \"legacy_oracle_candidate_count\": "
        << techMapMetrics->legacyOracleCandidateCount;
    out << ",\n    \"legacy_oracle_missing_count\": "
        << techMapMetrics->legacyOracleMissingCount;
    out << ",\n    \"legacy_oracle_status\": \""
        << escapeJsonString(legacyOracleStatus) << "\"";
    out << ",\n    \"layer2_handoff_status\": \""
        << escapeJsonString(layer2HandoffStatus) << "\"";
    out << ",\n    \"layer2_handoff_blockers\": [";
    for (size_t idx = 0; idx < layer2HandoffBlockers.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << "\"" << escapeJsonString(layer2HandoffBlockers[idx]) << "\"";
    }
    out << "]";
    out << ",\n    \"demand_driven_primary_plan\": "
        << ((techMapPlan && TechMapper::isDemandDrivenPrimaryPlan(*techMapPlan))
                ? "true"
                : "false");
    out << ",\n    \"selected_uses_legacy_derived_support\": "
        << (selectedUsesLegacyDerivedSupport ? "true" : "false");
    out << ",\n    \"candidate_pool_uses_legacy_derived_support\": "
        << (candidatePoolUsesLegacyDerivedSupport ? "true" : "false");
    out << ",\n    \"source_pool_uses_legacy_derived_support\": "
        << (sourcePoolUsesLegacyDerivedSupport ? "true" : "false");
    out << ",\n    \"legacy_fallback_count\": "
        << techMapMetrics->legacyFallbackCount;
    out << ",\n    \"legacy_fallback_source_count\": "
        << techMapMetrics->legacyFallbackCount;
    out << ",\n    \"legacy_fallback_candidate_count\": "
        << techMapMetrics->legacyFallbackCandidateCount;
    out << ",\n    \"legacy_contaminated_candidate_count\": "
        << techMapMetrics->legacyContaminatedCandidateCount;
    out << ",\n    \"legacy_derived_source_count\": "
        << techMapMetrics->legacyDerivedSourceCount;
    out << ",\n    \"feedback_reselection_count\": "
        << techMapMetrics->feedbackReselectionCount;
    out << ",\n    \"feedback_filtered_candidate_count\": "
        << techMapMetrics->feedbackFilteredCandidateCount;
    out << ",\n    \"feedback_penalty_count\": "
        << techMapMetrics->feedbackPenaltyCount;
    out << ",\n    \"feedback_unknown_candidate_ref_count\": "
        << (techMapPlan ? TechMapper::feedbackUnknownCandidateRefCount(*techMapPlan)
                        : techMapMetrics->feedbackUnknownCandidateRefCount);
    out << ",\n    \"feedback_unknown_family_ref_count\": "
        << (techMapPlan ? TechMapper::feedbackUnknownFamilyRefCount(*techMapPlan)
                        : techMapMetrics->feedbackUnknownFamilyRefCount);
    out << ",\n    \"feedback_unknown_config_class_ref_count\": "
        << (techMapPlan
                ? TechMapper::feedbackUnknownConfigClassRefCount(*techMapPlan)
                : techMapMetrics->feedbackUnknownConfigClassRefCount);
  }
  if (!stableTechMapDiagnostics.empty())
    out << ",\n    \"diagnostics\": \""
        << escapeJsonString(stableTechMapDiagnostics) << "\"";
  if (techMapPlan && TechMapper::hasAppliedFeedback(*techMapPlan)) {
    const auto &feedback = TechMapper::appliedFeedback(*techMapPlan);
    const auto &resolution = TechMapper::feedbackResolution(*techMapPlan);
    out << ",\n    \"feedback_request\": {";
    out << "\"unknown_candidate_ref_count\": "
        << TechMapper::feedbackUnknownCandidateRefCount(*techMapPlan);
    out << ", \"unknown_family_ref_count\": "
        << TechMapper::feedbackUnknownFamilyRefCount(*techMapPlan);
    out << ", \"unknown_config_class_ref_count\": "
        << TechMapper::feedbackUnknownConfigClassRefCount(*techMapPlan);
    out << ", \"banned_candidate_ids\": [";
    for (size_t idx = 0; idx < feedback.bannedCandidateIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << feedback.bannedCandidateIds[idx];
    }
    out << "], \"banned_family_ids\": [";
    for (size_t idx = 0; idx < feedback.bannedFamilyIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << feedback.bannedFamilyIds[idx];
    }
    out << "], \"banned_config_class_ids\": [";
    for (size_t idx = 0; idx < feedback.bannedConfigClassIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << feedback.bannedConfigClassIds[idx];
    }
    out << "], \"split_candidate_ids\": [";
    for (size_t idx = 0; idx < feedback.splitCandidateIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << feedback.splitCandidateIds[idx];
    }
    out << "], \"candidate_penalties\": [";
    for (size_t idx = 0; idx < feedback.candidatePenalties.size(); ++idx) {
      const auto &penalty = feedback.candidatePenalties[idx];
      if (idx != 0)
        out << ", ";
      writeCandidatePenaltyJson(out, techMapPlan, penalty);
    }
    out << "], \"family_penalties\": [";
    for (size_t idx = 0; idx < feedback.familyPenalties.size(); ++idx) {
      const auto &penalty = feedback.familyPenalties[idx];
      if (idx != 0)
        out << ", ";
      writeFamilyPenaltyJson(out, techMapPlan, penalty);
    }
    out << "], \"config_class_penalties\": [";
    for (size_t idx = 0; idx < feedback.configClassPenalties.size(); ++idx) {
      const auto &penalty = feedback.configClassPenalties[idx];
      if (idx != 0)
        out << ", ";
      writeConfigClassPenaltyJson(out, techMapPlan, penalty);
    }
    out << "], \"banned_candidates\": [";
    for (size_t idx = 0; idx < feedback.bannedCandidateIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      writeCandidateFeedbackRefJson(out, techMapPlan,
                                    feedback.bannedCandidateIds[idx]);
    }
    out << "], \"banned_families\": [";
    for (size_t idx = 0; idx < feedback.bannedFamilyIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      writeFamilyFeedbackRefJson(out, techMapPlan, feedback.bannedFamilyIds[idx]);
    }
    out << "], \"banned_config_classes\": [";
    for (size_t idx = 0; idx < feedback.bannedConfigClassIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      writeConfigClassFeedbackRefJson(out, techMapPlan,
                                      feedback.bannedConfigClassIds[idx]);
    }
    out << "], \"split_candidates\": [";
    for (size_t idx = 0; idx < feedback.splitCandidateIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      writeCandidateFeedbackRefJson(out, techMapPlan,
                                    feedback.splitCandidateIds[idx]);
    }
    out << "], \"unknown_banned_candidate_ids\": [";
    for (size_t idx = 0; idx < resolution.unknownBannedCandidateIds.size();
         ++idx) {
      if (idx != 0)
        out << ", ";
      out << resolution.unknownBannedCandidateIds[idx];
    }
    out << "], \"unknown_banned_family_ids\": [";
    for (size_t idx = 0; idx < resolution.unknownBannedFamilyIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << resolution.unknownBannedFamilyIds[idx];
    }
    out << "], \"unknown_banned_config_class_ids\": [";
    for (size_t idx = 0; idx < resolution.unknownBannedConfigClassIds.size();
         ++idx) {
      if (idx != 0)
        out << ", ";
      out << resolution.unknownBannedConfigClassIds[idx];
    }
    out << "], \"unknown_split_candidate_ids\": [";
    for (size_t idx = 0; idx < resolution.unknownSplitCandidateIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << resolution.unknownSplitCandidateIds[idx];
    }
    out << "], \"unknown_candidate_penalties\": [";
    for (size_t idx = 0; idx < resolution.unknownCandidatePenalties.size();
         ++idx) {
      if (idx != 0)
        out << ", ";
      out << "{\"id\": " << resolution.unknownCandidatePenalties[idx].id
          << ", \"penalty\": "
          << resolution.unknownCandidatePenalties[idx].penalty << "}";
    }
    out << "], \"unknown_family_penalties\": [";
    for (size_t idx = 0; idx < resolution.unknownFamilyPenalties.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << "{\"id\": " << resolution.unknownFamilyPenalties[idx].id
          << ", \"penalty\": "
          << resolution.unknownFamilyPenalties[idx].penalty << "}";
    }
    out << "], \"unknown_config_class_penalties\": [";
    for (size_t idx = 0; idx < resolution.unknownConfigClassPenalties.size();
         ++idx) {
      if (idx != 0)
        out << ", ";
      out << "{\"id\": " << resolution.unknownConfigClassPenalties[idx].id
          << ", \"penalty\": "
          << resolution.unknownConfigClassPenalties[idx].penalty << "}";
    }
    out << "]}";
  }
  if (techMapPlan && !TechMapper::allSupportClasses(*techMapPlan).empty()) {
    out << ",\n    \"support_classes\": [";
    llvm::ArrayRef<TechMapper::SupportClassInfo> supportClasses =
        TechMapper::allSupportClasses(*techMapPlan);
    for (size_t idx = 0; idx < supportClasses.size(); ++idx) {
      const auto &info = supportClasses[idx];
      if (idx != 0)
        out << ", ";
      out << "{\"id\": " << info.id << ", \"key\": \""
          << escapeJsonString(info.key) << "\", \"kind\": \""
          << escapeJsonString(info.kind) << "\", \"temporal\": "
          << (info.temporal ? "true" : "false")
          << ", \"enforce_hard_capacity\": "
          << (info.enforceHardCapacity ? "true" : "false")
          << ", \"hw_nodes\": [";
      for (size_t hwIdx = 0; hwIdx < info.hwNodeIds.size(); ++hwIdx) {
        if (hwIdx != 0)
          out << ", ";
        out << info.hwNodeIds[hwIdx];
      }
      out << "], \"pe_names\": [";
      for (size_t peIdx = 0; peIdx < info.peNames.size(); ++peIdx) {
        if (peIdx != 0)
          out << ", ";
        out << "\"" << escapeJsonString(info.peNames[peIdx]) << "\"";
      }
      out << "], \"capacity\": " << info.capacity << "}";
    }
    out << "]";
  }
  auto aliasPairs = opcompat::getAliasPairs();
  out << ",\n    \"op_alias_pairs\": [";
  for (size_t idx = 0; idx < aliasPairs.size(); ++idx) {
    if (idx != 0)
      out << ", ";
    out << "{\"lhs\": \"" << escapeJsonString(aliasPairs[idx].lhs)
        << "\", \"rhs\": \"" << escapeJsonString(aliasPairs[idx].rhs)
        << "\"}";
  }
  out << "]";
  if (techMapPlan && !TechMapper::allFallbackNodes(*techMapPlan).empty()) {
    out << ",\n    \"fallback_nodes\": [";
    llvm::ArrayRef<TechMapper::FallbackNodeInfo> fallbackNodes =
        TechMapper::allFallbackNodes(*techMapPlan);
    for (size_t idx = 0; idx < fallbackNodes.size(); ++idx) {
      const auto &info = fallbackNodes[idx];
      if (idx != 0)
        out << ", ";
      out << "{\"sw_node\": " << info.swNodeId << ", \"reason\": \""
          << escapeJsonString(info.reason)
          << "\", \"contracted_node\": ";
      if (info.contractedNodeId == INVALID_ID)
        out << "null";
      else
        out << info.contractedNodeId;
      out << ", \"selection_component_id\": ";
      writeOptionalUIntJson(out, info.selectionComponentId);
      out << ", \"candidate_hw_node_count\": " << info.candidateHwNodeCount
          << ", \"support_class_count\": " << info.supportClassCount
          << ", \"config_class_count\": " << info.configClassCount
          << ", \"candidate_ids\": [";
      for (size_t candIdx = 0; candIdx < info.candidateIds.size(); ++candIdx) {
        if (candIdx != 0)
          out << ", ";
        out << info.candidateIds[candIdx];
      }
      out << "], \"support_class_ids\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        if (scIdx != 0)
          out << ", ";
        out << info.supportClassIds[scIdx];
      }
      out << "], \"support_class_keys\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        if (scIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(
                   lookupSupportClassKey(techMapPlan, info.supportClassIds[scIdx]))
            << "\"";
      }
      out << "], \"support_class_kinds\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        const auto *supportClass =
            TechMapper::findSupportClass(*techMapPlan, info.supportClassIds[scIdx]);
        if (scIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(supportClass ? llvm::StringRef(supportClass->kind)
                                             : llvm::StringRef())
            << "\"";
      }
      out << "], \"support_class_temporal\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        const auto *supportClass =
            TechMapper::findSupportClass(*techMapPlan, info.supportClassIds[scIdx]);
        if (scIdx != 0)
          out << ", ";
        out << (supportClass && supportClass->temporal ? "true" : "false");
      }
      out << "], \"support_class_enforce_hard_capacity\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        const auto *supportClass =
            TechMapper::findSupportClass(*techMapPlan, info.supportClassIds[scIdx]);
        if (scIdx != 0)
          out << ", ";
        out << (supportClass && supportClass->enforceHardCapacity ? "true"
                                                                  : "false");
      }
      out << "], \"support_class_capacities\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        const auto *supportClass =
            TechMapper::findSupportClass(*techMapPlan, info.supportClassIds[scIdx]);
        if (scIdx != 0)
          out << ", ";
        out << (supportClass ? supportClass->capacity : 0);
      }
      out << "], \"config_class_ids\": [";
      for (size_t ccIdx = 0; ccIdx < info.configClassIds.size(); ++ccIdx) {
        if (ccIdx != 0)
          out << ", ";
        out << info.configClassIds[ccIdx];
      }
      out << "], \"config_class_keys\": [";
      for (size_t ccIdx = 0; ccIdx < info.configClassIds.size(); ++ccIdx) {
        if (ccIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(
                   lookupConfigClassKey(techMapPlan, info.configClassIds[ccIdx]))
            << "\"";
      }
      out << "], \"config_class_reasons\": [";
      for (size_t ccIdx = 0; ccIdx < info.configClassIds.size(); ++ccIdx) {
        const auto *configClass =
            TechMapper::findConfigClass(*techMapPlan, info.configClassIds[ccIdx]);
        if (ccIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(configClass ? llvm::StringRef(configClass->reason)
                                            : llvm::StringRef())
            << "\"";
      }
      out << "], \"config_class_temporal\": [";
      for (size_t ccIdx = 0; ccIdx < info.configClassIds.size(); ++ccIdx) {
        const auto *configClass =
            TechMapper::findConfigClass(*techMapPlan, info.configClassIds[ccIdx]);
        if (ccIdx != 0)
          out << ", ";
        out << (configClass && configClass->temporal ? "true" : "false");
      }
      out << "]}";
    }
    out << "]";
  }
  if (techMapPlan && !TechMapper::allSelectionComponents(*techMapPlan).empty()) {
    out << ",\n    \"selection_components\": [";
    llvm::ArrayRef<TechMapper::SelectionComponentInfo> selectionComponents =
        TechMapper::allSelectionComponents(*techMapPlan);
    for (size_t idx = 0; idx < selectionComponents.size(); ++idx) {
      const auto &info = selectionComponents[idx];
      if (idx != 0)
        out << ", ";
      out << "{\"id\": " << info.id
          << ", \"candidate_count\": " << info.candidateCount
          << ", \"sw_node_count\": " << info.swNodeCount
          << ", \"selected_count\": " << info.selectedCount
          << ", \"base_max_candidate_score\": " << info.baseMaxCandidateScore
          << ", \"max_candidate_score\": " << info.maxCandidateScore
          << ", \"base_selected_score_sum\": " << info.baseSelectedScoreSum
          << ", \"selected_score_sum\": " << info.selectedScoreSum
          << ", \"contains_temporal_candidate\": "
          << (info.containsTemporalCandidate ? "true" : "false")
          << ", \"sw_nodes\": [";
      for (size_t swIdx = 0; swIdx < info.swNodeIds.size(); ++swIdx) {
        if (swIdx != 0)
          out << ", ";
        out << info.swNodeIds[swIdx];
      }
      out << "], \"candidate_ids\": [";
      for (size_t candIdx = 0; candIdx < info.candidateIds.size(); ++candIdx) {
        if (candIdx != 0)
          out << ", ";
        out << info.candidateIds[candIdx];
      }
      out << "], \"filtered_candidate_ids\": [";
      for (size_t candIdx = 0; candIdx < info.filteredCandidateIds.size();
           ++candIdx) {
        if (candIdx != 0)
          out << ", ";
        out << info.filteredCandidateIds[candIdx];
      }
      out << "], \"selected_candidate_ids\": [";
      for (size_t selIdx = 0; selIdx < info.selectedCandidateIds.size();
           ++selIdx) {
        if (selIdx != 0)
          out << ", ";
        out << info.selectedCandidateIds[selIdx];
      }
      out << "], \"selected_unit_indices\": [";
      for (size_t unitIdx = 0; unitIdx < info.selectedUnitIndices.size();
           ++unitIdx) {
        if (unitIdx != 0)
          out << ", ";
        out << info.selectedUnitIndices[unitIdx];
      }
      out << "], \"solver\": \""
          << escapeJsonString(info.solver) << "\"}";
    }
    out << "]";
  }
  if (techMapPlan && !TechMapper::allFamilyTechInfos(*techMapPlan).empty()) {
    out << ",\n    \"family_stats\": [";
    llvm::ArrayRef<TechMapper::FamilyTechInfo> familyTechInfos =
        TechMapper::allFamilyTechInfos(*techMapPlan);
    for (size_t idx = 0; idx < familyTechInfos.size(); ++idx) {
      const auto &info = familyTechInfos[idx];
      if (idx != 0)
        out << ", ";
      out << "{\"family_index\": " << info.familyIndex
          << ", \"signature\": \"" << escapeJsonString(info.signature) << "\""
          << ", \"hw_support_count\": " << info.hwSupportCount
          << ", \"materialized_state_count\": " << info.materializedStateCount
          << ", \"legacy_materialized_state_count\": "
          << info.legacyMaterializedStateCount
          << ", \"match_count\": " << info.matchCount
          << ", \"selected_count\": " << info.selectedCount
          << ", \"max_fusion_size\": " << info.maxFusionSize
          << ", \"op_count\": " << info.opCount
          << ", \"configurable\": "
          << (info.configurable ? "true" : "false") << "}";
    }
    out << "]";
  }
  if (techMapPlan && !TechMapper::allCandidateSummaries(*techMapPlan).empty()) {
    out << ",\n    \"candidates\": [";
    llvm::ArrayRef<TechMapper::CandidateSummaryInfo> candidateSummaries =
        TechMapper::allCandidateSummaries(*techMapPlan);
    for (size_t idx = 0; idx < candidateSummaries.size(); ++idx) {
      const auto &candidate = candidateSummaries[idx];
      if (idx != 0)
        out << ", ";
      out << "{\"candidate_id\": " << candidate.id
          << ", \"family_index\": ";
      writeOptionalUIntJson(out, candidate.familyIndex);
      out << ", \"family_signature\": \""
          << escapeJsonString(candidate.familySignature) << "\""
          << ", \"selection_component_id\": ";
      writeOptionalUIntJson(out, candidate.selectionComponentId);
      out << ", \"selected_unit_index\": ";
      writeOptionalUIntJson(out, candidate.selectedUnitIndex);
      out << ", \"contracted_node\": ";
      if (candidate.contractedNodeId == INVALID_ID)
        out << "null";
      else
        out << candidate.contractedNodeId;
      out << ", \"sw_nodes\": [";
      for (size_t swIdx = 0; swIdx < candidate.swNodeIds.size(); ++swIdx) {
        if (swIdx != 0)
          out << ", ";
        out << candidate.swNodeIds[swIdx];
      }
      out << "], \"internal_edges\": [";
      for (size_t edgeIdx = 0; edgeIdx < candidate.internalEdgeIds.size();
           ++edgeIdx) {
        if (edgeIdx != 0)
          out << ", ";
        out << candidate.internalEdgeIds[edgeIdx];
      }
      out << "], \"input_bindings\": [";
      for (size_t bindIdx = 0; bindIdx < candidate.inputBindings.size();
           ++bindIdx) {
        if (bindIdx != 0)
          out << ", ";
        out << "{\"sw_port\": " << candidate.inputBindings[bindIdx].swPortId
            << ", \"hw_port_index\": "
            << candidate.inputBindings[bindIdx].hwPortIndex << "}";
      }
      out << "], \"output_bindings\": [";
      for (size_t bindIdx = 0; bindIdx < candidate.outputBindings.size();
           ++bindIdx) {
        if (bindIdx != 0)
          out << ", ";
        out << "{\"sw_port\": " << candidate.outputBindings[bindIdx].swPortId
            << ", \"hw_port_index\": "
            << candidate.outputBindings[bindIdx].hwPortIndex << "}";
      }
      out << "], \"hw_nodes\": [";
      for (size_t hwIdx = 0; hwIdx < candidate.hwNodeIds.size(); ++hwIdx) {
        if (hwIdx != 0)
          out << ", ";
        out << candidate.hwNodeIds[hwIdx];
      }
      out << "], \"pe_names\": [";
      for (size_t peIdx = 0; peIdx < candidate.peNames.size(); ++peIdx) {
        if (peIdx != 0)
          out << ", ";
        out << "\"" << escapeJsonString(candidate.peNames[peIdx]) << "\"";
      }
      out << "]";
      out << ", \"support_class_id\": ";
      writeOptionalUIntJson(out, candidate.supportClassId);
      out << ", \"support_class_capacity\": "
          << candidate.supportClassCapacity;
      out << ", \"support_class_key\": \""
          << escapeJsonString(candidate.supportClassKey) << "\"";
      out << ", \"support_class_kind\": \""
          << escapeJsonString(
                 [&]() -> llvm::StringRef {
                   const auto *supportClass =
                       TechMapper::findSupportClass(*techMapPlan,
                                                    candidate.supportClassId);
                   return supportClass ? llvm::StringRef(supportClass->kind)
                                       : llvm::StringRef();
                 }())
          << "\"";
      out << ", \"support_class_temporal\": "
          << (TechMapper::isTemporalSupportClass(*techMapPlan,
                                                 candidate.supportClassId)
                  ? "true"
                  : "false");
      out << ", \"support_class_enforce_hard_capacity\": "
          << (TechMapper::supportClassEnforcesHardCapacity(
                  *techMapPlan, candidate.supportClassId)
                  ? "true"
                  : "false");
      out << ", \"config_class_id\": ";
      writeOptionalUIntJson(out, candidate.configClassId);
      out << ", \"config_class_key\": \""
          << escapeJsonString(candidate.configClassKey) << "\"";
      out << ", \"config_class_reason\": \""
          << escapeJsonString(candidate.configClassReason) << "\"";
      out << ", \"config_class_temporal\": "
          << (TechMapper::isTemporalConfigClass(*techMapPlan,
                                                candidate.configClassId)
                  ? "true"
                  : "false");
      out << ", \"base_selection_score\": " << candidate.baseSelectionScore;
      out << ", \"candidate_penalty\": " << candidate.candidatePenalty;
      out << ", \"family_penalty\": " << candidate.familyPenalty;
      out << ", \"config_class_penalty\": " << candidate.configClassPenalty;
      out << ", \"temporal\": " << (candidate.temporal ? "true" : "false")
          << ", \"configurable\": "
          << (candidate.configurable ? "true" : "false")
          << ", \"selected\": " << (candidate.selected ? "true" : "false")
          << ", \"status\": \"" << escapeJsonString(candidate.status) << "\""
          << ", \"selection_score\": " << candidate.selectionScore
          << ", \"score_breakdown\": {\"fused_op_bonus\": "
          << static_cast<int64_t>(candidate.swNodeIds.size()) * 1024
          << ", \"internal_edge_bonus\": "
          << static_cast<int64_t>(candidate.internalEdgeIds.size()) * 192
          << ", \"candidate_choice_bonus\": "
          << static_cast<int64_t>(candidate.hwNodeIds.size()) * 32
          << ", \"boundary_penalty\": "
          << static_cast<int64_t>(candidate.inputBindings.size() +
                                  candidate.outputBindings.size()) *
                 48
          << ", \"config_penalty\": "
          << static_cast<int64_t>(candidate.configFields.size()) * 12
          << ", \"family_scarcity_penalty\": "
          << computeFamilyScarcityPenalty(techMapPlan, candidate.familyIndex)
          << ", \"temporal_penalty\": " << (candidate.temporal ? 64 : 0)
          << ", \"candidate_penalty\": " << candidate.candidatePenalty
          << ", \"family_penalty\": " << candidate.familyPenalty
          << ", \"config_class_penalty\": "
          << candidate.configClassPenalty << "}"
          << ", \"demand_origin\": "
          << (candidate.demandOrigin ? "true" : "false")
          << ", \"legacy_fallback_origin\": "
          << (candidate.legacyFallbackOrigin ? "true" : "false")
          << ", \"mixed_origin\": "
          << (candidate.mixedOrigin ? "true" : "false")
          << ", \"origin_kind\": \""
          << escapeJsonString(TechMapper::originKind(
                 candidate.demandOrigin, candidate.legacyFallbackOrigin,
                 candidate.mixedOrigin))
          << "\""
          << ", \"config_fields\": [";
      for (size_t fieldIdx = 0; fieldIdx < candidate.configFields.size();
           ++fieldIdx) {
        if (fieldIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(
                   summarizeConfigField(candidate.configFields[fieldIdx]))
            << "\"";
      }
      out << "]}";
    }
    out << "]";
  }
  if (techMapPlan && !TechMapper::allUnits(*techMapPlan).empty()) {
    llvm::ArrayRef<TechMapper::Unit> units = TechMapper::allUnits(*techMapPlan);
    out << ",\n    \"selected_units\": [";
    for (size_t unitIdx = 0; unitIdx < units.size(); ++unitIdx) {
      const auto &unit = units[unitIdx];
      const auto *selectionComponent =
          TechMapper::findSelectionComponent(*techMapPlan, unit);
      const auto *familyInfo =
          TechMapper::findFamilyTechInfo(*techMapPlan, unit);
      const auto *selectedCandidateSummary =
          TechMapper::findSelectedCandidateSummary(*techMapPlan, unit);
      const auto *selectedConfigClass =
          TechMapper::findSelectedUnitConfigClass(*techMapPlan, unit);
      const auto *preferredCandidate =
          TechMapper::findPreferredUnitCandidate(unit);
      const auto *preferredSupportClass =
          TechMapper::findPreferredUnitSupportClass(*techMapPlan, unit);
      const auto *preferredConfigClass =
          TechMapper::findPreferredUnitConfigClass(*techMapPlan, unit);
      if (unitIdx != 0)
        out << ", ";
      out << "{\"unit_index\": " << unitIdx
          << ", \"candidate_id\": " << unit.selectedCandidateId
          << ", \"family_index\": " << unit.familyIndex;
      if (unit.contractedNodeId != INVALID_ID)
        out << ", \"contracted_node\": " << unit.contractedNodeId;
      out << ", \"selection_component_id\": ";
      writeOptionalUIntJson(out, unit.selectionComponentId);
      if (selectionComponent) {
        out << ", \"selection_solver\": \""
            << escapeJsonString(selectionComponent->solver)
            << "\"";
      }
      if (familyInfo) {
        out << ", \"family_signature\": \""
            << escapeJsonString(familyInfo->signature)
            << "\"";
      }
      out << ", \"sw_nodes\": [";
      for (size_t swIdx = 0; swIdx < unit.swNodes.size(); ++swIdx) {
        if (swIdx != 0)
          out << ", ";
        out << unit.swNodes[swIdx];
      }
      out << "], \"internal_edges\": [";
      for (size_t edgeIdx = 0; edgeIdx < unit.internalEdges.size(); ++edgeIdx) {
        if (edgeIdx != 0)
          out << ", ";
        out << unit.internalEdges[edgeIdx];
      }
      out << "], \"input_bindings\": [";
      for (size_t bindIdx = 0; bindIdx < unit.inputBindings.size(); ++bindIdx) {
        if (bindIdx != 0)
          out << ", ";
        out << "{\"sw_port\": " << unit.inputBindings[bindIdx].swPortId
            << ", \"hw_port_index\": " << unit.inputBindings[bindIdx].hwPortIndex
            << "}";
      }
      out << "], \"output_bindings\": [";
      for (size_t bindIdx = 0; bindIdx < unit.outputBindings.size(); ++bindIdx) {
        if (bindIdx != 0)
          out << ", ";
        out << "{\"sw_port\": " << unit.outputBindings[bindIdx].swPortId
            << ", \"hw_port_index\": " << unit.outputBindings[bindIdx].hwPortIndex
            << "}";
      }
      out << "], \"candidate_count\": " << unit.candidates.size()
          << ", \"candidate_hw_nodes\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        if (candIdx != 0)
          out << ", ";
        out << unit.candidates[candIdx].hwNodeId;
      }
      out << "], \"candidate_pe_names\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        const Node *candidateHwNode = adg.getNode(unit.candidates[candIdx].hwNodeId);
        if (candIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(candidateHwNode
                                    ? getNodeAttrStr(candidateHwNode, "pe_name")
                                    : llvm::StringRef())
            << "\"";
      }
      out << "], \"candidate_support_classes\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        if (candIdx != 0)
          out << ", ";
        out << unit.candidates[candIdx].supportClassId;
      }
      out << "], \"candidate_support_class_keys\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        const auto *candidateSupportClass =
            TechMapper::findSupportClass(*techMapPlan, unit.candidates[candIdx]);
        if (candIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(candidateSupportClass
                                    ? candidateSupportClass->key
                                    : llvm::StringRef())
            << "\"";
      }
      out << "], \"candidate_support_class_kinds\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        const auto *candidateSupportClass =
            TechMapper::findSupportClass(*techMapPlan, unit.candidates[candIdx]);
        if (candIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(candidateSupportClass
                                    ? candidateSupportClass->kind
                                    : llvm::StringRef())
            << "\"";
      }
      out << "], \"candidate_support_class_temporal\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        const auto *candidateSupportClass =
            TechMapper::findSupportClass(*techMapPlan, unit.candidates[candIdx]);
        if (candIdx != 0)
          out << ", ";
        out << (candidateSupportClass && candidateSupportClass->temporal
                    ? "true"
                    : "false");
      }
      out << "], \"candidate_support_class_enforce_hard_capacity\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        const auto *candidateSupportClass =
            TechMapper::findSupportClass(*techMapPlan, unit.candidates[candIdx]);
        if (candIdx != 0)
          out << ", ";
        out << (candidateSupportClass &&
                        candidateSupportClass->enforceHardCapacity
                    ? "true"
                    : "false");
      }
      out << "], \"candidate_support_class_capacities\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        const auto *candidateSupportClass =
            TechMapper::findSupportClass(*techMapPlan, unit.candidates[candIdx]);
        if (candIdx != 0)
          out << ", ";
        out << (candidateSupportClass ? candidateSupportClass->capacity : 0);
      }
      out << "], \"candidate_config_classes\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        if (candIdx != 0)
          out << ", ";
        out << unit.candidates[candIdx].configClassId;
      }
      out << "], \"candidate_config_class_keys\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        const auto *candidateConfigClass =
            TechMapper::findConfigClass(*techMapPlan, unit.candidates[candIdx]);
        if (candIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(candidateConfigClass
                                    ? candidateConfigClass->key
                                    : llvm::StringRef())
            << "\"";
      }
      out << "], \"candidate_config_class_reasons\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        const auto *candidateConfigClass =
            TechMapper::findConfigClass(*techMapPlan, unit.candidates[candIdx]);
        if (candIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(candidateConfigClass
                                    ? candidateConfigClass->reason
                                    : llvm::StringRef())
            << "\"";
      }
      out << "], \"candidate_config_class_temporal\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        const auto *candidateConfigClass =
            TechMapper::findConfigClass(*techMapPlan, unit.candidates[candIdx]);
        if (candIdx != 0)
          out << ", ";
        out << (candidateConfigClass && candidateConfigClass->temporal
                    ? "true"
                    : "false");
      }
      out << "], \"candidate_details\": [";
      for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
        const auto &candidate = unit.candidates[candIdx];
        const auto *candidateSupportClass =
            TechMapper::findSupportClass(*techMapPlan, candidate);
        const auto *candidateConfigClass =
            TechMapper::findConfigClass(*techMapPlan, candidate);
        const Node *candidateHwNode = adg.getNode(candidate.hwNodeId);
        if (candIdx != 0)
          out << ", ";
        out << "{\"candidate_index\": " << candIdx
            << ", \"hw_node\": " << candidate.hwNodeId
            << ", \"pe_name\": \""
            << escapeJsonString(candidateHwNode
                                    ? getNodeAttrStr(candidateHwNode, "pe_name")
                                    : llvm::StringRef())
            << "\""
            << ", \"support_class_id\": " << candidate.supportClassId
            << ", \"support_class_key\": \""
            << escapeJsonString(candidateSupportClass
                                    ? candidateSupportClass->key
                                    : llvm::StringRef())
            << "\""
            << ", \"support_class_kind\": \""
            << escapeJsonString(candidateSupportClass
                                    ? candidateSupportClass->kind
                                    : llvm::StringRef())
            << "\""
            << ", \"support_class_temporal\": "
            << (candidateSupportClass && candidateSupportClass->temporal
                    ? "true"
                    : "false")
            << ", \"support_class_enforce_hard_capacity\": "
            << (candidateSupportClass &&
                        candidateSupportClass->enforceHardCapacity
                    ? "true"
                    : "false")
            << ", \"support_class_capacity\": "
            << (candidateSupportClass ? candidateSupportClass->capacity : 0)
            << ", \"config_class_id\": " << candidate.configClassId
            << ", \"config_class_key\": \""
            << escapeJsonString(candidateConfigClass
                                    ? candidateConfigClass->key
                                    : llvm::StringRef())
            << "\""
            << ", \"config_class_reason\": \""
            << escapeJsonString(candidateConfigClass
                                    ? candidateConfigClass->reason
                                    : llvm::StringRef())
            << "\""
            << ", \"config_class_temporal\": "
            << (candidateConfigClass && candidateConfigClass->temporal
                    ? "true"
                    : "false")
            << ", \"temporal\": "
            << (candidate.temporal ? "true" : "false")
            << ", \"config_fields\": [";
        for (size_t fieldIdx = 0; fieldIdx < candidate.configFields.size();
             ++fieldIdx) {
          if (fieldIdx != 0)
            out << ", ";
          out << "\""
              << escapeJsonString(
                     summarizeConfigField(candidate.configFields[fieldIdx]))
              << "\"";
        }
        out << "]}";
      }
      out << "]";
      int64_t baseSelectionScore = unit.selectionScore;
      int64_t candidatePenalty = 0;
      int64_t familyPenalty = 0;
      int64_t configClassPenalty = 0;
      if (selectedCandidateSummary) {
        baseSelectionScore = selectedCandidateSummary->baseSelectionScore;
        candidatePenalty = selectedCandidateSummary->candidatePenalty;
        familyPenalty = selectedCandidateSummary->familyPenalty;
        configClassPenalty = selectedCandidateSummary->configClassPenalty;
      }
      out << ", \"preferred_candidate_index\": " << unit.preferredCandidateIndex
          << ", \"config_class_id\": " << unit.configClassId
          << ", \"config_class_key\": \""
          << escapeJsonString(selectedConfigClass ? selectedConfigClass->key
                                                  : llvm::StringRef())
          << "\""
          << ", \"config_class_reason\": \""
          << escapeJsonString(selectedConfigClass ? selectedConfigClass->reason
                                                  : llvm::StringRef())
          << "\""
          << ", \"base_selection_score\": " << baseSelectionScore
          << ", \"candidate_penalty\": " << candidatePenalty
          << ", \"family_penalty\": " << familyPenalty
          << ", \"config_class_penalty\": " << configClassPenalty
          << ", \"selection_score\": " << unit.selectionScore
          << ", \"score_breakdown\": {\"fused_op_bonus\": "
          << static_cast<int64_t>(unit.swNodes.size()) * 1024
          << ", \"internal_edge_bonus\": "
          << static_cast<int64_t>(unit.internalEdges.size()) * 192
          << ", \"candidate_choice_bonus\": "
          << static_cast<int64_t>(unit.candidates.size()) * 32
          << ", \"boundary_penalty\": "
          << static_cast<int64_t>(unit.inputBindings.size() +
                                  unit.outputBindings.size()) *
                 48
          << ", \"config_penalty\": "
          << static_cast<int64_t>(
                 preferredCandidate ? preferredCandidate->configFields.size()
                                    : 0) *
                 12
          << ", \"family_scarcity_penalty\": "
          << computeFamilyScarcityPenalty(techMapPlan, unit.familyIndex)
          << ", \"temporal_penalty\": "
          << computeSelectedUnitTemporalPenalty(unit)
          << ", \"candidate_penalty\": " << candidatePenalty
          << ", \"family_penalty\": " << familyPenalty
          << ", \"config_class_penalty\": " << configClassPenalty << "}"
          << ", \"demand_origin\": "
          << (unit.demandOrigin ? "true" : "false")
          << ", \"legacy_fallback_origin\": "
          << (unit.legacyFallbackOrigin ? "true" : "false")
          << ", \"mixed_origin\": " << (unit.mixedOrigin ? "true" : "false")
          << ", \"origin_kind\": \""
          << escapeJsonString(TechMapper::originKind(
                 unit.demandOrigin, unit.legacyFallbackOrigin,
                 unit.mixedOrigin))
          << "\""
          << ", \"conservative_fallback\": "
          << (unit.conservativeFallback ? "true" : "false");
      if (preferredCandidate) {
        const Node *preferredHwNode = adg.getNode(preferredCandidate->hwNodeId);
        out << ", \"preferred_hw_node\": " << preferredCandidate->hwNodeId
            << ", \"preferred_pe_name\": \""
            << escapeJsonString(preferredHwNode
                                    ? getNodeAttrStr(preferredHwNode, "pe_name")
                                    : llvm::StringRef())
            << "\""
            << ", \"preferred_support_class_id\": "
            << preferredCandidate->supportClassId
            << ", \"preferred_support_class_key\": \""
            << escapeJsonString(preferredSupportClass
                                    ? preferredSupportClass->key
                                    : llvm::StringRef())
            << "\""
            << ", \"preferred_support_class_kind\": \""
            << escapeJsonString(preferredSupportClass
                                    ? preferredSupportClass->kind
                                    : llvm::StringRef())
            << "\""
            << ", \"preferred_support_class_temporal\": "
            << (preferredSupportClass && preferredSupportClass->temporal
                    ? "true"
                    : "false")
            << ", \"preferred_support_class_enforce_hard_capacity\": "
            << (preferredSupportClass &&
                        preferredSupportClass->enforceHardCapacity
                    ? "true"
                    : "false")
            << ", \"preferred_support_class_capacity\": "
            << (preferredSupportClass ? preferredSupportClass->capacity : 0)
            << ", \"preferred_config_class_id\": "
            << preferredCandidate->configClassId
            << ", \"preferred_config_class_key\": \""
            << escapeJsonString(preferredConfigClass
                                    ? preferredConfigClass->key
                                    : llvm::StringRef())
            << "\""
            << ", \"preferred_config_class_reason\": \""
            << escapeJsonString(preferredConfigClass
                                    ? preferredConfigClass->reason
                                    : llvm::StringRef())
            << "\""
            << ", \"preferred_config_class_temporal\": "
            << (preferredConfigClass && preferredConfigClass->temporal
                    ? "true"
                    : "false")
            << ", \"preferred_temporal\": "
            << (preferredCandidate->temporal ? "true" : "false")
            << ", \"preferred_config_fields\": [";
        for (size_t fieldIdx = 0;
             fieldIdx < preferredCandidate->configFields.size();
             ++fieldIdx) {
          if (fieldIdx != 0)
            out << ", ";
          out << "\""
              << escapeJsonString(
                     summarizeConfigField(
                         preferredCandidate->configFields[fieldIdx]))
              << "\"";
        }
        out << "]";
      }
      out << "}";
    }
    out << "]";
  }
  if (techMapPlan && !TechMapper::allNodeTechInfos(*techMapPlan).empty()) {
    out << ",\n    \"node_tech_info\": [";
    llvm::ArrayRef<TechMapper::NodeTechInfo> nodeTechInfos =
        TechMapper::allNodeTechInfos(*techMapPlan);
    for (size_t idx = 0; idx < nodeTechInfos.size(); ++idx) {
      const auto &info = nodeTechInfos[idx];
      if (idx != 0)
        out << ", ";
      out << "{\"sw_node\": " << info.swNodeId
          << ", \"contracted_node\": ";
      if (info.contractedNodeId == INVALID_ID)
        out << "null";
      else
        out << info.contractedNodeId;
      out << ", \"selection_component_id\": ";
      writeOptionalUIntJson(out, info.selectionComponentId);
      out << ", \"selected_unit_index\": ";
      writeOptionalUIntJson(out, info.selectedUnitIndex);
      out << ", \"selected_candidate_id\": ";
      writeOptionalUIntJson(out, info.selectedCandidateId);
      out << ", \"candidate_count\": " << info.candidateCount
          << ", \"candidate_ids\": [";
      for (size_t candIdx = 0; candIdx < info.candidateIds.size(); ++candIdx) {
        if (candIdx != 0)
          out << ", ";
        out << info.candidateIds[candIdx];
      }
      out << "]"
          << ", \"support_class_count\": " << info.supportClassCount
          << ", \"support_class_ids\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        if (scIdx != 0)
          out << ", ";
        out << info.supportClassIds[scIdx];
      }
      out << "], \"support_class_keys\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        if (scIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(
                   lookupSupportClassKey(techMapPlan, info.supportClassIds[scIdx]))
            << "\"";
      }
      out << "], \"support_class_kinds\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        const auto *supportClass =
            TechMapper::findSupportClass(*techMapPlan, info.supportClassIds[scIdx]);
        if (scIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(supportClass ? llvm::StringRef(supportClass->kind)
                                             : llvm::StringRef())
            << "\"";
      }
      out << "], \"support_class_temporal\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        const auto *supportClass =
            TechMapper::findSupportClass(*techMapPlan, info.supportClassIds[scIdx]);
        if (scIdx != 0)
          out << ", ";
        out << (supportClass && supportClass->temporal ? "true" : "false");
      }
      out << "], \"support_class_enforce_hard_capacity\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        const auto *supportClass =
            TechMapper::findSupportClass(*techMapPlan, info.supportClassIds[scIdx]);
        if (scIdx != 0)
          out << ", ";
        out << (supportClass && supportClass->enforceHardCapacity ? "true"
                                                                  : "false");
      }
      out << "], \"support_class_capacities\": [";
      for (size_t scIdx = 0; scIdx < info.supportClassIds.size(); ++scIdx) {
        const auto *supportClass =
            TechMapper::findSupportClass(*techMapPlan, info.supportClassIds[scIdx]);
        if (scIdx != 0)
          out << ", ";
        out << (supportClass ? supportClass->capacity : 0);
      }
      out << "]"
          << ", \"config_class_count\": " << info.configClassCount
          << ", \"config_class_ids\": [";
      for (size_t ccIdx = 0; ccIdx < info.configClassIds.size(); ++ccIdx) {
        if (ccIdx != 0)
          out << ", ";
        out << info.configClassIds[ccIdx];
      }
      out << "], \"config_class_keys\": [";
      for (size_t ccIdx = 0; ccIdx < info.configClassIds.size(); ++ccIdx) {
        if (ccIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(
                   lookupConfigClassKey(techMapPlan, info.configClassIds[ccIdx]))
            << "\"";
      }
      out << "], \"config_class_reasons\": [";
      for (size_t ccIdx = 0; ccIdx < info.configClassIds.size(); ++ccIdx) {
        const auto *configClass =
            TechMapper::findConfigClass(*techMapPlan, info.configClassIds[ccIdx]);
        if (ccIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(configClass ? llvm::StringRef(configClass->reason)
                                            : llvm::StringRef())
            << "\"";
      }
      out << "], \"config_class_temporal\": [";
      for (size_t ccIdx = 0; ccIdx < info.configClassIds.size(); ++ccIdx) {
        const auto *configClass =
            TechMapper::findConfigClass(*techMapPlan, info.configClassIds[ccIdx]);
        if (ccIdx != 0)
          out << ", ";
        out << (configClass && configClass->temporal ? "true" : "false");
      }
      out << "]"
          << ", \"max_fusion_size\": " << info.maxFusionSize
          << ", \"selected\": " << (info.selected ? "true" : "false")
          << ", \"selected_as_fusion\": "
          << (info.selectedAsFusion ? "true" : "false")
          << ", \"selected_from_demand\": "
          << (info.selectedFromDemand ? "true" : "false")
          << ", \"selected_from_legacy_fallback\": "
          << (info.selectedFromLegacyFallback ? "true" : "false")
          << ", \"selected_from_mixed_origin\": "
          << (info.selectedFromMixedOrigin ? "true" : "false")
          << ", \"origin_kind\": \""
          << escapeJsonString(TechMapper::originKind(
                 info.selectedFromDemand, info.selectedFromLegacyFallback,
                 info.selectedFromMixedOrigin))
          << "\""
          << ", \"conservative_fallback\": "
          << (info.conservativeFallback ? "true" : "false")
          << ", \"status\": \"" << escapeJsonString(info.status) << "\"}";
    }
    out << "]";
  }
  if (techMapPlan &&
      !TechMapper::conservativeFallbackSwNodes(*techMapPlan).empty()) {
    llvm::ArrayRef<IdIndex> fallbackSwNodes =
        TechMapper::conservativeFallbackSwNodes(*techMapPlan);
    out << ",\n    \"conservative_fallback_sw_nodes\": [";
    for (size_t idx = 0; idx < fallbackSwNodes.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << fallbackSwNodes[idx];
    }
    out << "]";
  }
  if (techMapPlan &&
      !TechMapper::conservativeFallbackDFG(*techMapPlan).nodes.empty()) {
    const Graph &fallbackDFG = TechMapper::conservativeFallbackDFG(*techMapPlan);
    out << ",\n    \"conservative_fallback_plan\": {\"node_candidates\": [";
    bool firstFallbackNode = true;
    for (IdIndex swNodeId = 0;
         swNodeId < static_cast<IdIndex>(fallbackDFG.nodes.size());
         ++swNodeId) {
      const Node *fallbackNode = fallbackDFG.getNode(swNodeId);
      if (!fallbackNode || fallbackNode->kind != Node::OperationNode)
        continue;
      if (!firstFallbackNode)
        out << ", ";
      firstFallbackNode = false;
      out << "{\"sw_node\": " << swNodeId << ", \"sw_op\": \""
          << getNodeAttrStr(fallbackNode, "op_name") << "\"";
      out << ", \"candidate_hw_nodes\": [";
      if (const auto *hwNodes =
              TechMapper::findConservativeFallbackCandidates(*techMapPlan,
                                                             swNodeId)) {
        for (size_t idx = 0; idx < hwNodes->size(); ++idx) {
          if (idx != 0)
            out << ", ";
          out << (*hwNodes)[idx];
        }
      }
      out << "]";
      out << ", \"candidate_support_classes\": [";
      if (const auto *supportClasses =
              TechMapper::findConservativeFallbackCandidateSupportClasses(
                  *techMapPlan, swNodeId)) {
        for (size_t idx = 0; idx < supportClasses->size(); ++idx) {
          if (idx != 0)
            out << ", ";
          out << (*supportClasses)[idx];
        }
      }
      out << "], \"candidate_support_class_keys\": [";
      if (const auto *supportClasses =
              TechMapper::findConservativeFallbackCandidateSupportClasses(
                  *techMapPlan, swNodeId)) {
        for (size_t idx = 0; idx < supportClasses->size(); ++idx) {
          if (idx != 0)
            out << ", ";
          out << "\""
              << escapeJsonString(
                     lookupSupportClassKey(techMapPlan, (*supportClasses)[idx]))
              << "\"";
        }
      }
      out << "], \"candidate_support_class_kinds\": [";
      if (const auto *supportClasses =
              TechMapper::findConservativeFallbackCandidateSupportClasses(
                  *techMapPlan, swNodeId)) {
        for (size_t idx = 0; idx < supportClasses->size(); ++idx) {
          const auto *supportClass =
              TechMapper::findSupportClass(*techMapPlan, (*supportClasses)[idx]);
          if (idx != 0)
            out << ", ";
          out << "\""
              << escapeJsonString(supportClass
                                      ? llvm::StringRef(supportClass->kind)
                                      : llvm::StringRef())
              << "\"";
        }
      }
      out << "], \"candidate_support_class_temporal\": [";
      if (const auto *supportClasses =
              TechMapper::findConservativeFallbackCandidateSupportClasses(
                  *techMapPlan, swNodeId)) {
        for (size_t idx = 0; idx < supportClasses->size(); ++idx) {
          if (idx != 0)
            out << ", ";
          out << (TechMapper::isTemporalSupportClass(*techMapPlan,
                                                     (*supportClasses)[idx])
                      ? "true"
                      : "false");
        }
      }
      out << "], \"candidate_support_class_enforce_hard_capacity\": [";
      if (const auto *supportClasses =
              TechMapper::findConservativeFallbackCandidateSupportClasses(
                  *techMapPlan, swNodeId)) {
        for (size_t idx = 0; idx < supportClasses->size(); ++idx) {
          if (idx != 0)
            out << ", ";
          out << (TechMapper::supportClassEnforcesHardCapacity(
                      *techMapPlan, (*supportClasses)[idx])
                      ? "true"
                      : "false");
        }
      }
      out << "], \"candidate_support_class_capacities\": [";
      if (const auto *supportClasses =
              TechMapper::findConservativeFallbackCandidateSupportClasses(
                  *techMapPlan, swNodeId)) {
        for (size_t idx = 0; idx < supportClasses->size(); ++idx) {
          if (idx != 0)
            out << ", ";
          out << TechMapper::supportClassCapacity(*techMapPlan,
                                                  (*supportClasses)[idx]);
        }
      }
      out << "]";
      out << ", \"candidate_config_classes\": [";
      if (const auto *configClasses =
              TechMapper::findConservativeFallbackCandidateConfigClasses(
                  *techMapPlan, swNodeId)) {
        for (size_t idx = 0; idx < configClasses->size(); ++idx) {
          if (idx != 0)
            out << ", ";
          out << (*configClasses)[idx];
        }
      }
      out << "], \"candidate_config_class_keys\": [";
      if (const auto *configClasses =
              TechMapper::findConservativeFallbackCandidateConfigClasses(
                  *techMapPlan, swNodeId)) {
        for (size_t idx = 0; idx < configClasses->size(); ++idx) {
          if (idx != 0)
            out << ", ";
          out << "\""
              << escapeJsonString(
                     lookupConfigClassKey(techMapPlan, (*configClasses)[idx]))
              << "\"";
        }
      }
      out << "], \"candidate_config_class_reasons\": [";
      if (const auto *configClasses =
              TechMapper::findConservativeFallbackCandidateConfigClasses(
                  *techMapPlan, swNodeId)) {
        for (size_t idx = 0; idx < configClasses->size(); ++idx) {
          if (idx != 0)
            out << ", ";
          out << "\""
              << escapeJsonString(
                     lookupConfigClassReason(techMapPlan, (*configClasses)[idx]))
              << "\"";
        }
      }
      out << "], \"candidate_config_class_temporal\": [";
      if (const auto *configClasses =
              TechMapper::findConservativeFallbackCandidateConfigClasses(
                  *techMapPlan, swNodeId)) {
        for (size_t idx = 0; idx < configClasses->size(); ++idx) {
          if (idx != 0)
            out << ", ";
          out << (TechMapper::isTemporalConfigClass(*techMapPlan,
                                                    (*configClasses)[idx])
                      ? "true"
                      : "false");
        }
      }
      out << "]";
      out << ", \"candidates\": [";
      if (const auto *candidateDetails =
              TechMapper::findConservativeFallbackCandidateDetails(
                  *techMapPlan, swNodeId)) {
        for (size_t idx = 0; idx < candidateDetails->size(); ++idx) {
          const auto &candidate = (*candidateDetails)[idx];
          if (idx != 0)
            out << ", ";
          out << "{\"hw_node\": " << candidate.hwNodeId
              << ", \"support_class_id\": " << candidate.supportClassId
              << ", \"support_class_capacity\": "
              << TechMapper::supportClassCapacity(*techMapPlan,
                                                  candidate.supportClassId)
              << ", \"support_class_key\": \""
              << escapeJsonString(
                     lookupSupportClassKey(techMapPlan, candidate.supportClassId))
              << "\""
              << ", \"support_class_kind\": \""
              << escapeJsonString(
                     [&]() -> llvm::StringRef {
                       const auto *supportClass =
                           TechMapper::findSupportClass(*techMapPlan,
                                                        candidate.supportClassId);
                       return supportClass ? llvm::StringRef(supportClass->kind)
                                           : llvm::StringRef();
                     }())
              << "\""
              << ", \"support_class_temporal\": "
              << (TechMapper::isTemporalSupportClass(*techMapPlan,
                                                     candidate.supportClassId)
                      ? "true"
                      : "false")
              << ", \"support_class_enforce_hard_capacity\": "
              << (TechMapper::supportClassEnforcesHardCapacity(
                      *techMapPlan, candidate.supportClassId)
                      ? "true"
                      : "false")
              << ", \"config_class_id\": " << candidate.configClassId
              << ", \"config_class_key\": \""
              << escapeJsonString(
                     lookupConfigClassKey(techMapPlan, candidate.configClassId))
              << "\""
              << ", \"config_class_reason\": \""
              << escapeJsonString(lookupConfigClassReason(techMapPlan,
                                                          candidate.configClassId))
              << "\""
              << ", \"config_class_temporal\": "
              << (TechMapper::isTemporalConfigClass(*techMapPlan,
                                                    candidate.configClassId)
                      ? "true"
                      : "false")
              << ", \"temporal\": " << (candidate.temporal ? "true" : "false")
              << ", \"config_fields\": [";
          for (size_t fieldIdx = 0; fieldIdx < candidate.configFields.size();
               ++fieldIdx) {
            if (fieldIdx != 0)
              out << ", ";
            out << "\""
                << escapeJsonString(
                       summarizeConfigField(candidate.configFields[fieldIdx]))
                << "\"";
          }
          out << "]}";
        }
      }
      out << "]";
      if (const auto *preferred =
              TechMapper::findConservativeFallbackPreferredCandidate(
                  *techMapPlan, swNodeId)) {
        out << ", \"preferred_candidate\": {\"hw_node\": " << preferred->hwNodeId
            << ", \"support_class_id\": " << preferred->supportClassId
            << ", \"support_class_capacity\": "
            << TechMapper::supportClassCapacity(*techMapPlan,
                                                preferred->supportClassId)
            << ", \"support_class_key\": \""
            << escapeJsonString(
                   lookupSupportClassKey(techMapPlan, preferred->supportClassId))
            << "\""
            << ", \"support_class_kind\": \""
            << escapeJsonString(
                   [&]() -> llvm::StringRef {
                     const auto *supportClass =
                         TechMapper::findSupportClass(*techMapPlan,
                                                      preferred->supportClassId);
                     return supportClass ? llvm::StringRef(supportClass->kind)
                                         : llvm::StringRef();
                   }())
            << "\""
            << ", \"support_class_temporal\": "
            << (TechMapper::isTemporalSupportClass(*techMapPlan,
                                                   preferred->supportClassId)
                    ? "true"
                    : "false")
            << ", \"support_class_enforce_hard_capacity\": "
            << (TechMapper::supportClassEnforcesHardCapacity(
                    *techMapPlan, preferred->supportClassId)
                    ? "true"
                    : "false")
            << ", \"config_class_id\": " << preferred->configClassId
            << ", \"config_class_key\": \""
            << escapeJsonString(
                   lookupConfigClassKey(techMapPlan, preferred->configClassId))
            << "\""
            << ", \"config_class_reason\": \""
            << escapeJsonString(lookupConfigClassReason(techMapPlan,
                                                        preferred->configClassId))
            << "\""
            << ", \"config_class_temporal\": "
            << (TechMapper::isTemporalConfigClass(*techMapPlan,
                                                  preferred->configClassId)
                    ? "true"
                    : "false")
            << ", \"temporal\": " << (preferred->temporal ? "true" : "false")
            << ", \"config_fields\": [";
        for (size_t fieldIdx = 0; fieldIdx < preferred->configFields.size();
             ++fieldIdx) {
          if (fieldIdx != 0)
            out << ", ";
          out << "\""
              << escapeJsonString(
                     summarizeConfigField(preferred->configFields[fieldIdx]))
              << "\"";
        }
        out << "]}";
      }
      out << ", \"selection_component_id\": ";
      unsigned selectionComponentId = std::numeric_limits<unsigned>::max();
      if (const auto *nodeInfo =
              TechMapper::findNodeTechInfo(*techMapPlan, swNodeId))
        selectionComponentId = nodeInfo->selectionComponentId;
      writeOptionalUIntJson(out, selectionComponentId);
      out << "}";
    }
    out << "]}";
  }
  if (techMapPlan && !TechMapper::allConfigClasses(*techMapPlan).empty()) {
    out << ",\n    \"config_classes\": [";
    llvm::ArrayRef<TechMapper::ConfigClassInfo> configClasses =
        TechMapper::allConfigClasses(*techMapPlan);
    for (size_t idx = 0; idx < configClasses.size(); ++idx) {
      const auto &info = configClasses[idx];
      if (idx != 0)
        out << ", ";
      out << "{\"id\": " << info.id << ", \"key\": \""
          << escapeJsonString(info.key) << "\", \"reason\": \""
          << escapeJsonString(info.reason) << "\", \"temporal\": "
          << (info.temporal ? "true" : "false")
          << ", \"compatible_with\": [";
      llvm::ArrayRef<unsigned> compatibleConfigClasses =
          TechMapper::compatibleConfigClasses(*techMapPlan, info.id);
      for (size_t compatIdx = 0; compatIdx < compatibleConfigClasses.size();
           ++compatIdx) {
        if (compatIdx != 0)
          out << ", ";
        out << compatibleConfigClasses[compatIdx];
      }
      out << "], \"compatible_with_keys\": [";
      for (size_t compatIdx = 0; compatIdx < compatibleConfigClasses.size();
           ++compatIdx) {
        if (compatIdx != 0)
          out << ", ";
        out << "\""
            << escapeJsonString(lookupConfigClassKey(
                   techMapPlan, compatibleConfigClasses[compatIdx]))
            << "\"";
      }
      out << "]}";
    }
    out << "]";
  }
  if (techMapPlan && !TechMapper::temporalIncompatibilities(*techMapPlan).empty()) {
    out << ",\n    \"temporal_incompatibilities\": [";
    llvm::ArrayRef<TechMapper::TemporalIncompatibilityInfo> temporalIncompat =
        TechMapper::temporalIncompatibilities(*techMapPlan);
    for (size_t idx = 0; idx < temporalIncompat.size(); ++idx) {
      const auto &info = temporalIncompat[idx];
      if (idx != 0)
        out << ", ";
      out << "{\"lhs_config_class_id\": " << info.lhsConfigClassId
          << ", \"lhs_config_class_key\": \""
          << escapeJsonString(
                 lookupConfigClassKey(techMapPlan, info.lhsConfigClassId))
          << "\""
          << ", \"rhs_config_class_id\": " << info.rhsConfigClassId
          << ", \"rhs_config_class_key\": \""
          << escapeJsonString(
                 lookupConfigClassKey(techMapPlan, info.rhsConfigClassId))
          << "\""
          << ", \"reason\": \"" << escapeJsonString(info.reason) << "\"}";
    }
    out << "]";
  }
  if (techMapPlan &&
      !TechMapper::legacyOracleMissingSamples(*techMapPlan).empty()) {
    llvm::ArrayRef<TechMapper::LegacyOracleSampleInfo> legacySamples =
        TechMapper::legacyOracleMissingSamples(*techMapPlan);
    out << ",\n    \"legacy_oracle_missing_samples\": [";
    for (size_t idx = 0; idx < legacySamples.size(); ++idx) {
      const auto &info = legacySamples[idx];
      if (idx != 0)
        out << ", ";
      out << "{\"key\": \"" << escapeJsonString(info.key) << "\""
          << ", \"family_index\": ";
      writeOptionalUIntJson(out, info.familyIndex);
      out << ", \"family_signature\": \""
          << escapeJsonString(info.familySignature) << "\""
          << ", \"hw_node\": " << info.hwNodeId
          << ", \"pe_name\": \"" << escapeJsonString(info.peName) << "\""
          << ", \"hw_name\": \"" << escapeJsonString(info.hwName) << "\"}";
    }
    out << "]";
  }
  out << "\n  },\n";
  writeMapJsonTailSections(out, state, dfg, adg, flattener, edgeKinds,
                           fuConfigs, timingSummary, searchSummary);
  return true;
}

} // namespace loom

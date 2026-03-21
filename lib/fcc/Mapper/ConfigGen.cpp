// ConfigGen.cpp -- ConfigGen::writeMapJson and ConfigGen::writeMapText.

#include "ConfigGenInternal.h"

#include "fcc/Mapper/ConfigGen.h"
#include "fcc/Mapper/BridgeBinding.h"
#include "fcc/Mapper/Mapper.h"
#include "fcc/Mapper/OpCompat.h"

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

namespace fcc {

namespace {

std::string escapeJsonString(llvm::StringRef text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (char ch : text) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(ch);
      break;
    }
  }
  return escaped;
}

std::string summarizeConfigField(const FUConfigField &field) {
  std::string summary;
  llvm::raw_string_ostream os(summary);
  os << static_cast<unsigned>(field.kind) << ":" << field.opIndex << ":"
     << field.templateOpIndex << ":" << field.opName << ":" << field.bitWidth
     << ":" << field.value << ":" << field.sel << ":" << field.discard << ":"
     << field.disconnect;
  return summary;
}

bool getEffectiveFifoBypassed(const Node *hwNode, IdIndex hwId,
                              const MappingState &state) {
  bool bypassable = false;
  bool bypassed = false;
  if (!hwNode)
    return false;
  for (const auto &attr : hwNode->attributes) {
    if (attr.getName() == "bypassable") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        bypassable = boolAttr.getValue();
    } else if (attr.getName() == "bypassed") {
      if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
        bypassed = boolAttr.getValue();
    }
  }
  if (!bypassable)
    return false;
  if (hwId < state.hwNodeFifoBypassedOverride.size()) {
    int8_t overrideValue = state.hwNodeFifoBypassedOverride[hwId];
    if (overrideValue == 0)
      return false;
    if (overrideValue > 0)
      return true;
  }
  return bypassed;
}

void writeOptionalUIntJson(llvm::raw_ostream &out, unsigned value) {
  if (value == std::numeric_limits<unsigned>::max()) {
    out << "null";
    return;
  }
  out << value;
}

int64_t computeSelectedUnitTemporalPenalty(const TechMapper::Unit &unit) {
  const auto *preferredCandidate = TechMapper::findPreferredUnitCandidate(unit);
  if (!preferredCandidate)
    return 0;
  return preferredCandidate->temporal ? 64 : 0;
}

int64_t computeFamilyScarcityPenalty(const TechMapper::Plan *techMapPlan,
                                     unsigned familyIndex) {
  if (!techMapPlan)
    return 0;
  llvm::ArrayRef<TechMapper::FamilyTechInfo> familyTechInfos =
      TechMapper::allFamilyTechInfos(*techMapPlan);
  if (familyIndex >= familyTechInfos.size())
    return 0;
  unsigned hwSupportCount = familyTechInfos[familyIndex].hwSupportCount;
  if (hwSupportCount >= 4)
    return 0;
  return static_cast<int64_t>(4 - hwSupportCount) * 48;
}

void writeOptionalUIntText(llvm::raw_ostream &out, unsigned value) {
  if (value == std::numeric_limits<unsigned>::max()) {
    out << "-";
    return;
  }
  out << value;
}

llvm::StringRef
lookupSupportClassKey(const TechMapper::Plan *techMapPlan, unsigned id) {
  if (!techMapPlan)
    return {};
  if (const auto *info = TechMapper::findSupportClass(*techMapPlan, id))
    return info->key;
  return {};
}

llvm::StringRef
lookupConfigClassKey(const TechMapper::Plan *techMapPlan, unsigned id) {
  if (!techMapPlan)
    return {};
  if (const auto *info = TechMapper::findConfigClass(*techMapPlan, id))
    return info->key;
  return {};
}

llvm::StringRef
lookupConfigClassReason(const TechMapper::Plan *techMapPlan, unsigned id) {
  if (!techMapPlan)
    return {};
  if (const auto *info = TechMapper::findConfigClass(*techMapPlan, id))
    return info->reason;
  return {};
}

llvm::StringRef
lookupFamilySignature(const TechMapper::Plan *techMapPlan, unsigned id) {
  if (!techMapPlan)
    return {};
  if (const auto *info = TechMapper::findFamilyTechInfo(*techMapPlan, id))
    return info->signature;
  return {};
}

const TechMapper::CandidateSummaryInfo *
lookupCandidateSummary(const TechMapper::Plan *techMapPlan, unsigned id) {
  if (!techMapPlan)
    return nullptr;
  return TechMapper::findCandidateSummary(*techMapPlan, id);
}

void writeCandidateFeedbackRefJson(llvm::raw_ostream &out,
                                   const TechMapper::Plan *techMapPlan,
                                   unsigned id) {
  const auto *candidate = lookupCandidateSummary(techMapPlan, id);
  out << "{\"id\": " << id;
  if (candidate) {
    out << ", \"family_index\": ";
    writeOptionalUIntJson(out, candidate->familyIndex);
    out << ", \"family_signature\": \""
        << escapeJsonString(candidate->familySignature) << "\"";
    out << ", \"selection_component_id\": ";
    writeOptionalUIntJson(out, candidate->selectionComponentId);
    out << ", \"contracted_node\": ";
    if (candidate->contractedNodeId == INVALID_ID)
      out << "null";
    else
      out << candidate->contractedNodeId;
    out << ", \"support_class_id\": ";
    writeOptionalUIntJson(out, candidate->supportClassId);
    out << ", \"support_class_key\": \""
        << escapeJsonString(candidate->supportClassKey) << "\"";
    out << ", \"config_class_id\": ";
    writeOptionalUIntJson(out, candidate->configClassId);
    out << ", \"config_class_key\": \""
        << escapeJsonString(candidate->configClassKey) << "\"";
    out << ", \"config_class_reason\": \""
        << escapeJsonString(candidate->configClassReason) << "\"";
    out << ", \"status\": \"" << escapeJsonString(candidate->status) << "\"";
    out << ", \"origin_kind\": \""
        << escapeJsonString(TechMapper::originKind(
               candidate->demandOrigin, candidate->legacyFallbackOrigin,
               candidate->mixedOrigin))
        << "\"";
    out << ", \"sw_nodes\": [";
    for (size_t idx = 0; idx < candidate->swNodeIds.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << candidate->swNodeIds[idx];
    }
    out << "]";
  }
  out << "}";
}

void writeFamilyFeedbackRefJson(llvm::raw_ostream &out,
                                const TechMapper::Plan *techMapPlan,
                                unsigned id) {
  out << "{\"id\": " << id << ", \"signature\": \""
      << escapeJsonString(lookupFamilySignature(techMapPlan, id)) << "\"}";
}

void writeConfigClassFeedbackRefJson(llvm::raw_ostream &out,
                                     const TechMapper::Plan *techMapPlan,
                                     unsigned id) {
  out << "{\"id\": " << id << ", \"key\": \""
      << escapeJsonString(lookupConfigClassKey(techMapPlan, id)) << "\""
      << ", \"reason\": \""
      << escapeJsonString(lookupConfigClassReason(techMapPlan, id)) << "\"}";
}

void writeCandidatePenaltyJson(llvm::raw_ostream &out,
                               const TechMapper::Plan *techMapPlan,
                               const TechMapper::WeightedIdPenalty &penalty) {
  out << "{\"id\": " << penalty.id << ", \"penalty\": " << penalty.penalty;
  const auto *candidate = lookupCandidateSummary(techMapPlan, penalty.id);
  if (candidate) {
    out << ", \"family_index\": ";
    writeOptionalUIntJson(out, candidate->familyIndex);
    out << ", \"family_signature\": \""
        << escapeJsonString(candidate->familySignature) << "\"";
    out << ", \"selection_component_id\": ";
    writeOptionalUIntJson(out, candidate->selectionComponentId);
    out << ", \"contracted_node\": ";
    if (candidate->contractedNodeId == INVALID_ID)
      out << "null";
    else
      out << candidate->contractedNodeId;
    out << ", \"config_class_id\": ";
    writeOptionalUIntJson(out, candidate->configClassId);
    out << ", \"config_class_key\": \""
        << escapeJsonString(candidate->configClassKey) << "\"";
    out << ", \"status\": \"" << escapeJsonString(candidate->status) << "\"";
  }
  out << "}";
}

void writeFamilyPenaltyJson(llvm::raw_ostream &out,
                            const TechMapper::Plan *techMapPlan,
                            const TechMapper::WeightedIdPenalty &penalty) {
  out << "{\"id\": " << penalty.id << ", \"penalty\": " << penalty.penalty
      << ", \"signature\": \""
      << escapeJsonString(lookupFamilySignature(techMapPlan, penalty.id))
      << "\"}";
}

void writeConfigClassPenaltyJson(llvm::raw_ostream &out,
                                 const TechMapper::Plan *techMapPlan,
                                 const TechMapper::WeightedIdPenalty &penalty) {
  out << "{\"id\": " << penalty.id << ", \"penalty\": " << penalty.penalty
      << ", \"key\": \""
      << escapeJsonString(lookupConfigClassKey(techMapPlan, penalty.id)) << "\""
      << ", \"reason\": \""
      << escapeJsonString(lookupConfigClassReason(techMapPlan, penalty.id))
      << "\"}";
}

void writeCandidateFeedbackRefText(llvm::raw_ostream &out,
                                   const TechMapper::Plan *techMapPlan,
                                   unsigned id) {
  out << id;
  const auto *candidate = lookupCandidateSummary(techMapPlan, id);
  if (!candidate)
    return;
  out << "[family=";
  writeOptionalUIntText(out, candidate->familyIndex);
  if (!candidate->familySignature.empty())
    out << "/" << candidate->familySignature;
  out << ",component=";
  writeOptionalUIntText(out, candidate->selectionComponentId);
  out << ",cfg=";
  writeOptionalUIntText(out, candidate->configClassId);
  if (!candidate->configClassKey.empty())
    out << "/" << candidate->configClassKey;
  out << ",status=" << candidate->status << "]";
}

void writeFamilyFeedbackRefText(llvm::raw_ostream &out,
                                const TechMapper::Plan *techMapPlan,
                                unsigned id) {
  out << id;
  llvm::StringRef signature = lookupFamilySignature(techMapPlan, id);
  if (!signature.empty())
    out << "[" << signature << "]";
}

void writeConfigClassFeedbackRefText(llvm::raw_ostream &out,
                                     const TechMapper::Plan *techMapPlan,
                                     unsigned id) {
  out << id;
  llvm::StringRef key = lookupConfigClassKey(techMapPlan, id);
  llvm::StringRef reason = lookupConfigClassReason(techMapPlan, id);
  if (!key.empty() || !reason.empty()) {
    out << "[";
    if (!key.empty())
      out << key;
    if (!reason.empty()) {
      if (!key.empty())
        out << "/";
      out << reason;
    }
    out << "]";
  }
}

std::string sanitizeTechMapDiagnosticsForArtifact(llvm::StringRef diagnostics) {
  if (diagnostics.empty())
    return {};
  llvm::SmallVector<llvm::StringRef, 16> parts;
  diagnostics.split(parts, ", ");
  std::string sanitized;
  llvm::raw_string_ostream os(sanitized);
  bool first = true;
  for (llvm::StringRef part : parts) {
    if (part.starts_with("total_layer2_us=") ||
        part.starts_with("candidate_gen_us=") ||
        part.starts_with("selection_us=")) {
      continue;
    }
    if (!first)
      os << ", ";
    os << part;
    first = false;
  }
  return os.str();
}

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
  out << "  \"search\": {\n";
  if (!searchSummary) {
    out << "    \"available\": false\n";
  } else {
    out << "    \"available\": true";
    out << ",\n    \"placement_seed_lane_count\": "
        << searchSummary->placementSeedLaneCount;
    out << ",\n    \"successful_placement_seed_count\": "
        << searchSummary->successfulPlacementSeedCount;
    out << ",\n    \"routed_lane_count\": "
        << searchSummary->routedLaneCount;
    out << ",\n    \"local_repair_attempts\": "
        << searchSummary->localRepairAttempts;
    out << ",\n    \"local_repair_successes\": "
        << searchSummary->localRepairSuccesses;
    out << ",\n    \"route_aware_refinement_passes\": "
        << searchSummary->routeAwareRefinementPasses;
    out << ",\n    \"route_aware_checkpoint_rescore_passes\": "
        << searchSummary->routeAwareCheckpointRescorePasses;
    out << ",\n    \"route_aware_checkpoint_restore_count\": "
        << searchSummary->routeAwareCheckpointRestoreCount;
    out << ",\n    \"route_aware_neighborhood_attempts\": "
        << searchSummary->routeAwareNeighborhoodAttempts;
    out << ",\n    \"route_aware_neighborhood_accepted_moves\": "
        << searchSummary->routeAwareNeighborhoodAcceptedMoves;
    out << ",\n    \"route_aware_coarse_fallback_moves\": "
        << searchSummary->routeAwareCoarseFallbackMoves;
    out << ",\n    \"fifo_bufferization_accepted_toggles\": "
        << searchSummary->fifoBufferizationAcceptedToggles;
    out << ",\n    \"outer_joint_accepted_rounds\": "
        << searchSummary->outerJointAcceptedRounds << "\n";
  }
  out << "  },\n";
  out << "  \"timing\": {\n";
  if (!timingSummary) {
    out << "    \"available\": false\n";
  } else {
    out << "    \"available\": true";
    out << ",\n    \"estimated_critical_path_delay\": "
        << timingSummary->estimatedCriticalPathDelay;
    out << ",\n    \"estimated_clock_period\": "
        << timingSummary->estimatedClockPeriod;
    out << ",\n    \"estimated_initiation_interval\": "
        << timingSummary->estimatedInitiationInterval;
    out << ",\n    \"estimated_throughput_cost\": "
        << timingSummary->estimatedThroughputCost;
    out << ",\n    \"recurrence_pressure\": "
        << timingSummary->recurrencePressure;
    out << ",\n    \"critical_path_edges\": [";
    for (size_t idx = 0; idx < timingSummary->criticalPathEdges.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->criticalPathEdges[idx];
    }
    out << "]";
    out << ",\n    \"fifo_buffer_count\": "
        << timingSummary->fifoBufferCount;
    out << ",\n    \"forced_buffered_fifo_count\": "
        << timingSummary->forcedBufferedFifoCount;
    out << ",\n    \"forced_buffered_fifo_nodes\": [";
    for (size_t idx = 0; idx < timingSummary->forcedBufferedFifoNodes.size();
         ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->forcedBufferedFifoNodes[idx];
    }
    out << "],\n    \"forced_buffered_fifo_depths\": [";
    for (size_t idx = 0; idx < timingSummary->forcedBufferedFifoDepths.size();
         ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->forcedBufferedFifoDepths[idx];
    }
    out << "],\n    \"mapper_selected_buffered_fifo_count\": "
        << timingSummary->mapperSelectedBufferedFifoCount;
    out << ",\n    \"mapper_selected_buffered_fifo_nodes\": [";
    for (size_t idx = 0;
         idx < timingSummary->mapperSelectedBufferedFifoNodes.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->mapperSelectedBufferedFifoNodes[idx];
    }
    out << "],\n    \"mapper_selected_buffered_fifo_depths\": [";
    for (size_t idx = 0;
         idx < timingSummary->mapperSelectedBufferedFifoDepths.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->mapperSelectedBufferedFifoDepths[idx];
    }
    out << "],\n    \"bufferized_edges\": [";
    for (size_t idx = 0; idx < timingSummary->bufferizedEdges.size(); ++idx) {
      if (idx != 0)
        out << ", ";
      out << timingSummary->bufferizedEdges[idx];
    }
    out << "],\n    \"recurrence_cycles\": [";
    for (size_t idx = 0; idx < timingSummary->recurrenceCycles.size(); ++idx) {
      const auto &cycle = timingSummary->recurrenceCycles[idx];
      if (idx != 0)
        out << ", ";
      out << "{\"cycle_id\": " << cycle.cycleId
          << ", \"sw_nodes\": [";
      for (size_t nodeIdx = 0; nodeIdx < cycle.swNodes.size(); ++nodeIdx) {
        if (nodeIdx != 0)
          out << ", ";
        out << cycle.swNodes[nodeIdx];
      }
      out << "], \"sw_edges\": [";
      for (size_t edgeIdx = 0; edgeIdx < cycle.swEdges.size(); ++edgeIdx) {
        if (edgeIdx != 0)
          out << ", ";
        out << cycle.swEdges[edgeIdx];
      }
      out << "], \"recurrence_distance\": " << cycle.recurrenceDistance
          << ", \"sequential_latency_cycles\": "
          << cycle.sequentialLatencyCycles
          << ", \"fifo_stage_cut_contribution\": "
          << cycle.fifoStageCutContribution
          << ", \"max_interval_on_cycle\": "
          << cycle.maxIntervalOnCycle
          << ", \"estimated_cycle_ii\": "
          << cycle.estimatedCycleII
          << ", \"combinational_delay\": "
          << cycle.combinationalDelay << "}";
    }
    out << "]\n";
  }
  out << "  },\n";
  out << "  \"node_mappings\": [\n";

  llvm::DenseMap<IdIndex, unsigned> configClassBySwNode;
  llvm::DenseMap<IdIndex, unsigned> supportClassBySwNode;
  for (const auto &selection : fuConfigs) {
    for (IdIndex swNodeId : selection.swNodeIds) {
      configClassBySwNode[swNodeId] = selection.configClassId;
      supportClassBySwNode[swNodeId] = selection.supportClassId;
    }
  }

  bool first = true;
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode || swNode->kind != Node::OperationNode)
      continue;
    if (swId >= state.swNodeToHwNode.size() ||
        state.swNodeToHwNode[swId] == INVALID_ID)
      continue;

    if (!first)
      out << ",\n";
    first = false;

    IdIndex hwId = state.swNodeToHwNode[swId];
    const Node *hwNode = adg.getNode(hwId);

    out << "    {\"sw_node\": " << swId << ", \"hw_node\": " << hwId;
    out << ", \"sw_op\": \"" << getNodeAttrStr(swNode, "op_name") << "\"";
    if (auto it = supportClassBySwNode.find(swId); it != supportClassBySwNode.end())
      out << ", \"support_class_id\": " << it->second;
    if (auto it = configClassBySwNode.find(swId); it != configClassBySwNode.end())
      out << ", \"config_class_id\": " << it->second;
    if (hwNode) {
      out << ", \"hw_name\": \"" << getNodeAttrStr(hwNode, "op_name") << "\"";
      out << ", \"pe_name\": \"" << getNodeAttrStr(hwNode, "pe_name") << "\"";
    }
    out << "}";
  }

  out << "\n  ],\n";

  // Edge routing.
  out << "  \"edge_routings\": [\n";
  first = true;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *edge = dfg.getEdge(eid);
    if (!edge)
      continue;

    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"sw_edge\": " << eid;
    if (eid < edgeKinds.size() && edgeKinds[eid] == TechMappedEdgeKind::IntraFU) {
      IdIndex hwNodeId = INVALID_ID;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      if (srcPort && srcPort->parentNode != INVALID_ID &&
          srcPort->parentNode < state.swNodeToHwNode.size()) {
        hwNodeId = state.swNodeToHwNode[srcPort->parentNode];
      }
      out << ", \"kind\": \"intra_fu\"";
      if (hwNodeId != INVALID_ID)
        out << ", \"hw_node\": " << hwNodeId;
      out << ", \"path\": []}";
      continue;
    }
    if (eid < edgeKinds.size() &&
        edgeKinds[eid] == TechMappedEdgeKind::TemporalReg) {
      IdIndex hwNodeId = INVALID_ID;
      const Port *srcPort = dfg.getPort(edge->srcPort);
      if (srcPort && srcPort->parentNode != INVALID_ID &&
          srcPort->parentNode < state.swNodeToHwNode.size()) {
        hwNodeId = state.swNodeToHwNode[srcPort->parentNode];
      }
      out << ", \"kind\": \"temporal_reg\"";
      if (hwNodeId != INVALID_ID)
        out << ", \"hw_node\": " << hwNodeId;
      out << ", \"path\": []}";
      continue;
    }

    if (eid >= state.swEdgeToHwPaths.size() || state.swEdgeToHwPaths[eid].empty()) {
      out << ", \"kind\": \"unrouted\", \"path\": []}";
      continue;
    }

    auto exportPath = buildExportPathForEdge(eid, state, dfg, adg);
    out << ", \"kind\": \"routed\", \"path\": [";
    for (size_t i = 0; i < exportPath.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << exportPath[i];
    }
    out << "]}";
  }

  out << "\n  ],\n";

  // Port table: maps flat port IDs to viz-compatible component names.
  struct PortVizInfo {
    bool valid = false;
    std::string kind;
    std::string component;
    std::string pe;
    std::string fu;
    int index = -1;
    std::string dir;
  };

  std::vector<PortVizInfo> portInfo(adg.ports.size());
  out << "  \"port_table\": [\n";
  first = true;
  for (IdIndex pid = 0; pid < static_cast<IdIndex>(adg.ports.size()); ++pid) {
    const Port *p = adg.getPort(pid);
    if (!p)
      continue;
    const Node *pn = adg.getNode(p->parentNode);
    if (!pn)
      continue;

    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"id\": " << pid;
    portInfo[pid].valid = true;
    portInfo[pid].dir =
        (p->direction == Port::Input ? std::string("in") : std::string("out"));

    if (pn->kind == Node::ModuleInputNode) {
      int argIdx = getNodeAttrInt(pn, "arg_index");
      portInfo[pid].kind = "module_in";
      portInfo[pid].component = "module_in";
      portInfo[pid].index = argIdx;
      out << ", \"kind\": \"module_in\", \"index\": " << argIdx;
    } else if (pn->kind == Node::ModuleOutputNode) {
      int resIdx = getNodeAttrInt(pn, "result_index");
      portInfo[pid].kind = "module_out";
      portInfo[pid].component = "module_out";
      portInfo[pid].index = resIdx;
      out << ", \"kind\": \"module_out\", \"index\": " << resIdx;
    } else {
      llvm::StringRef resClass = getNodeAttrStr(pn, "resource_class");
      if (resClass == "routing") {
        llvm::StringRef opName = getNodeAttrStr(pn, "op_name");
        llvm::StringRef opKind = getNodeAttrStr(pn, "op_kind");
        bool isSwitchLike =
            pn->inputPorts.size() > 1 || pn->outputPorts.size() > 1;
        if (opKind == "temporal_sw")
          portInfo[pid].kind = "temporal_sw";
        else if (opKind == "spatial_sw")
          portInfo[pid].kind = "sw";
        else if (opKind == "add_tag" || opKind == "del_tag" ||
                 opKind == "map_tag")
          portInfo[pid].kind = opKind.str();
        else
          portInfo[pid].kind = isSwitchLike ? "sw" : "fifo";
        portInfo[pid].component = opName.str();
        out << ", \"kind\": \"" << portInfo[pid].kind << "\", \"name\": \""
            << opName << "\"";
      } else if (resClass == "memory") {
        llvm::StringRef opName = getNodeAttrStr(pn, "op_name");
        portInfo[pid].kind = "memory";
        portInfo[pid].component = opName.str();
        out << ", \"kind\": \"memory\", \"name\": \"" << opName << "\"";
      } else {
        llvm::StringRef peName = getNodeAttrStr(pn, "pe_name");
        llvm::StringRef opName = getNodeAttrStr(pn, "op_name");
        portInfo[pid].kind = "fu";
        portInfo[pid].component = peName.str();
        portInfo[pid].pe = peName.str();
        portInfo[pid].fu = opName.str();
        out << ", \"kind\": \"fu\", \"pe\": \"" << peName << "\", \"fu\": \""
            << opName << "\"";
      }
      // Port index within the node.
      int portIdx = -1;
      if (p->direction == Port::Input) {
        for (unsigned i = 0; i < pn->inputPorts.size(); ++i) {
          if (pn->inputPorts[i] == pid) {
            portIdx = static_cast<int>(i);
            break;
          }
        }
      } else {
        for (unsigned i = 0; i < pn->outputPorts.size(); ++i) {
          if (pn->outputPorts[i] == pid) {
            portIdx = static_cast<int>(i);
            break;
          }
        }
      }
      portInfo[pid].index = portIdx;
      out << ", \"index\": " << portIdx;
    }

    out << ", \"dir\": \""
        << (p->direction == Port::Input ? "in" : "out") << "\"}";
  }
  out << "\n  ],\n";

  out << "  \"switch_routes\": [\n";
  using SwitchRouteKey = std::tuple<std::string, int, int>;
  struct SwitchRouteEntry {
    std::string component;
    IdIndex inputPortId = INVALID_ID;
    IdIndex outputPortId = INVALID_ID;
    int inputPort = -1;
    int outputPort = -1;
    std::vector<IdIndex> swEdges;
  };
  std::map<SwitchRouteKey, SwitchRouteEntry> switchRoutes;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++eid) {
    auto hwPath = buildExportPathForEdge(eid, state, dfg, adg);
    if (hwPath.size() < 3)
      continue;
    for (size_t i = 1; i + 1 < hwPath.size(); i += 2) {
      IdIndex inPortId = hwPath[i];
      IdIndex outPortId = hwPath[i + 1];
      if (inPortId >= portInfo.size() || outPortId >= portInfo.size())
        continue;
      const auto &inInfo = portInfo[inPortId];
      const auto &outInfo = portInfo[outPortId];
      if (!inInfo.valid || !outInfo.valid)
        continue;
      bool inSwitch =
          (inInfo.kind == "sw" || inInfo.kind == "temporal_sw");
      bool outSwitch =
          (outInfo.kind == "sw" || outInfo.kind == "temporal_sw");
      if (!inSwitch || !outSwitch)
        continue;
      if (inInfo.component != outInfo.component)
        continue;

      auto key =
          std::make_tuple(inInfo.component, inInfo.index, outInfo.index);
      auto &entry = switchRoutes[key];
      if (entry.component.empty()) {
        entry.component = inInfo.component;
        entry.inputPortId = inPortId;
        entry.outputPortId = outPortId;
        entry.inputPort = inInfo.index;
        entry.outputPort = outInfo.index;
      }
      entry.swEdges.push_back(eid);
    }
  }

  first = true;
  for (const auto &it : switchRoutes) {
    const auto &entry = it.second;
    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"component\": \"" << entry.component << "\"";
    out << ", \"input_port_id\": " << entry.inputPortId;
    out << ", \"output_port_id\": " << entry.outputPortId;
    out << ", \"input_port\": " << entry.inputPort;
    out << ", \"output_port\": " << entry.outputPort;
    out << ", \"sw_edges\": [";
    for (size_t i = 0; i < entry.swEdges.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << entry.swEdges[i];
    }
    out << "]}";
  }

  out << "\n  ],\n";

  out << "  \"pe_routes\": [\n";
  using PERouteKey =
      std::tuple<std::string, std::string, std::string, std::string>;
  struct PERouteEntry {
    std::string peName;
    std::string direction;
    std::string pePortKey;
    std::string fuPortKey;
    std::vector<IdIndex> swEdges;
  };
  std::map<PERouteKey, PERouteEntry> peRoutes;
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(state.swEdgeToHwPaths.size());
       ++eid) {
    auto hwPath = buildExportPathForEdge(eid, state, dfg, adg);
    if (hwPath.size() < 2)
      continue;

    for (size_t i = 0; i + 1 < hwPath.size(); ++i) {
      IdIndex srcPortId = hwPath[i];
      IdIndex dstPortId = hwPath[i + 1];
      const Port *srcPort = adg.getPort(srcPortId);
      const Port *dstPort = adg.getPort(dstPortId);
      if (!srcPort || !dstPort)
        continue;

      const Edge *flatEdge = findEdgeByPorts(adg, srcPortId, dstPortId);
      if (!flatEdge)
        continue;

      if (dstPortId < portInfo.size()) {
        const auto &dstInfo = portInfo[dstPortId];
        const Node *dstNode =
            dstPort->parentNode != INVALID_ID ? adg.getNode(dstPort->parentNode)
                                              : nullptr;
        auto peInputIndex = getUIntEdgeAttr(flatEdge, "pe_input_index");
        int fuInputIndex =
            dstNode ? findNodeInputIndex(dstNode, dstPortId) : -1;
        if (dstInfo.valid && dstInfo.kind == "fu" && peInputIndex &&
            fuInputIndex >= 0) {
          std::string peName = dstInfo.pe;
          std::string hwName = dstInfo.fu;
          std::string pePortKey =
              peName + "_in_" + std::to_string(*peInputIndex);
          std::string fuPortKey =
              peName + "/" + hwName + "/in_" + std::to_string(fuInputIndex);
          auto key =
              std::make_tuple(peName, std::string("in"), pePortKey, fuPortKey);
          auto &entry = peRoutes[key];
          if (entry.peName.empty()) {
            entry.peName = peName;
            entry.direction = "in";
            entry.pePortKey = pePortKey;
            entry.fuPortKey = fuPortKey;
          }
          entry.swEdges.push_back(eid);
        }
      }

      if (srcPortId < portInfo.size()) {
        const auto &srcInfo = portInfo[srcPortId];
        const Node *srcNode =
            srcPort->parentNode != INVALID_ID ? adg.getNode(srcPort->parentNode)
                                              : nullptr;
        auto peOutputIndex = getUIntEdgeAttr(flatEdge, "pe_output_index");
        int fuOutputIndex =
            srcNode ? findNodeOutputIndex(srcNode, srcPortId) : -1;
        if (srcInfo.valid && srcInfo.kind == "fu" && peOutputIndex &&
            fuOutputIndex >= 0) {
          std::string peName = srcInfo.pe;
          std::string hwName = srcInfo.fu;
          std::string pePortKey =
              peName + "_out_" + std::to_string(*peOutputIndex);
          std::string fuPortKey =
              peName + "/" + hwName + "/out_" + std::to_string(fuOutputIndex);
          auto key =
              std::make_tuple(peName, std::string("out"), pePortKey, fuPortKey);
          auto &entry = peRoutes[key];
          if (entry.peName.empty()) {
            entry.peName = peName;
            entry.direction = "out";
            entry.pePortKey = pePortKey;
            entry.fuPortKey = fuPortKey;
          }
          entry.swEdges.push_back(eid);
        }
      }
    }
  }

  first = true;
  for (const auto &it : peRoutes) {
    const auto &entry = it.second;
    if (!first)
      out << ",\n";
    first = false;

    out << "    {\"pe_name\": \"" << entry.peName << "\"";
    out << ", \"direction\": \"" << entry.direction << "\"";
    out << ", \"pe_port_key\": \"" << entry.pePortKey << "\"";
    out << ", \"fu_port_key\": \"" << entry.fuPortKey << "\"";
    out << ", \"sw_edges\": [";
    for (size_t i = 0; i < entry.swEdges.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << entry.swEdges[i];
    }
    out << "]}";
  }
  out << "\n  ],\n";

  out << "  \"fu_configs\": [\n";
  first = true;
  for (const auto &selection : fuConfigs) {
    if (!first)
      out << ",\n";
    first = false;
    out << "    {\"hw_node\": " << selection.hwNodeId;
    out << ", \"hw_name\": \"" << selection.hwName << "\"";
    out << ", \"pe_name\": \"" << selection.peName << "\"";
    out << ", \"support_class_id\": " << selection.supportClassId;
    out << ", \"config_class_id\": " << selection.configClassId;
    out << ", \"sw_nodes\": [";
    for (size_t i = 0; i < selection.swNodeIds.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << selection.swNodeIds[i];
    }
    out << "], \"fields\": [";
    for (size_t i = 0; i < selection.fields.size(); ++i) {
      if (i > 0)
        out << ", ";
      const auto &field = selection.fields[i];
      out << "{\"op_index\": " << field.opIndex;
      out << ", \"op_name\": \"" << field.opName << "\"";
      out << ", \"kind\": \"" << configFieldKindName(field.kind) << "\"";
      out << ", \"bit_width\": " << field.bitWidth;
      out << ", \"value\": " << field.value;
      out << ", \"display\": \"" << formatConfigFieldValue(field) << "\"";
      if (field.kind == FUConfigFieldKind::Mux) {
        out << ", \"sel\": " << field.sel;
        out << ", \"discard\": " << (field.discard ? "true" : "false");
        out << ", \"disconnect\": "
            << (field.disconnect ? "true" : "false");
      }
      out << "}";
    }
    out << "]}";
  }
  out << "\n  ],\n";

  out << "  \"tag_configs\": [\n";
  first = true;
  for (const auto &slice : configSlices_) {
    if (slice.kind != "add_tag" && slice.kind != "map_tag")
      continue;
    const Node *hwNode =
        slice.hwNode == INVALID_ID ? nullptr : adg.getNode(slice.hwNode);
    if (!first)
      out << ",\n";
    first = false;
    out << "    {\"name\": \"" << slice.name << "\", \"kind\": \""
        << slice.kind << "\", \"complete\": "
        << (slice.complete ? "true" : "false");
    if (slice.kind == "add_tag") {
      out << ", \"tag\": " << getNodeAttrInt(hwNode, "tag", 0);
    } else {
      out << ", \"table_size\": " << getNodeAttrInt(hwNode, "table_size", 0);
      auto [inTagWidth, outTagWidth] = getMapTagTagWidths(hwNode, adg);
      out << ", \"input_tag_width\": " << inTagWidth;
      out << ", \"output_tag_width\": " << outTagWidth;
      out << ", \"table\": [";
      bool firstElem = true;
      for (const auto &entry : getMapTagTableEntries(hwNode)) {
        if (!firstElem)
          out << ", ";
        firstElem = false;
        out << "{\"valid\": " << (entry.valid ? "true" : "false")
            << ", \"src_tag\": " << entry.srcTag
            << ", \"dst_tag\": " << entry.dstTag << "}";
      }
      out << "]";
    }
    out << "}";
  }
  out << "\n  ],\n";

  out << "  \"fifo_configs\": [\n";
  first = true;
  for (const auto &slice : configSlices_) {
    if (slice.kind != "fifo")
      continue;
    const Node *hwNode =
        slice.hwNode == INVALID_ID ? nullptr : adg.getNode(slice.hwNode);
    bool bypassable = false;
    bool bypassed = getEffectiveFifoBypassed(hwNode, slice.hwNode, state);
    for (const auto &attr : hwNode->attributes) {
      if (attr.getName() == "bypassable") {
        if (auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attr.getValue()))
          bypassable = boolAttr.getValue();
      }
    }
    if (!first)
      out << ",\n";
    first = false;
    out << "    {\"name\": \"" << slice.name << "\", \"complete\": "
        << (slice.complete ? "true" : "false");
    out << ", \"bypassable\": " << (bypassable ? "true" : "false");
    out << ", \"bypassed\": " << (bypassed ? "true" : "false");
    out << ", \"depth\": "
        << std::max<int64_t>(0, getNodeAttrInt(hwNode, "depth", 0));
    out << "}";
  }
  out << "\n  ],\n";

  out << "  \"temporal_registers\": [\n";
  first = true;
  for (const auto &pe : flattener.getPEContainment()) {
    if (pe.peKind != "temporal_pe")
      continue;
    auto temporalPlan = buildTemporalConfigPlan(pe, state, dfg, adg, edgeKinds);
    for (const auto &binding : temporalPlan.registerBindings) {
      if (!first)
        out << ",\n";
      first = false;
      out << "    {\"pe_name\": \"" << binding.peName << "\"";
      out << ", \"sw_edge\": " << binding.swEdgeId;
      out << ", \"register_index\": " << binding.registerIndex;
      out << ", \"writer_sw_node\": " << binding.writerSwNode;
      out << ", \"reader_sw_node\": " << binding.readerSwNode;
      out << ", \"writer_hw_node\": " << binding.writerHwNode;
      out << ", \"reader_hw_node\": " << binding.readerHwNode;
      out << ", \"writer_output_index\": " << binding.writerOutputIndex;
      out << ", \"reader_input_index\": " << binding.readerInputIndex << "}";
    }
  }
  out << "\n  ],\n";

  out << "  \"memory_regions\": [\n";
  first = true;
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;

    if (!first)
      out << ",\n";
    first = false;

    auto regions = collectMemoryRegionsForNode(hwId, state, dfg, adg);
    auto addrTable = buildAddrOffsetTable(hwNode, hwId, state, dfg, adg);

    out << "    {\"hw_node\": " << hwId;
    out << ", \"hw_name\": \"" << getNodeAttrStr(hwNode, "op_name") << "\"";
    out << ", \"memory_kind\": \"" << getNodeAttrStr(hwNode, "op_kind")
        << "\"";
    out << ", \"num_region\": " << getNodeAttrInt(hwNode, "numRegion", 1);
    out << ", \"regions\": [";
    for (size_t i = 0; i < regions.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << "{\"region_index\": " << i;
      out << ", \"sw_node\": " << regions[i].swNode;
      out << ", \"memref_arg_index\": " << regions[i].memrefArgIndex;
      out << ", \"start_lane\": " << regions[i].startLane;
      out << ", \"end_lane\": " << regions[i].endLane;
      out << ", \"ld_count\": " << regions[i].ldCount;
      out << ", \"st_count\": " << regions[i].stCount;
      out << ", \"elem_size_log2\": " << regions[i].elemSizeLog2 << "}";
    }
    out << "], \"addr_offset_table\": [";
    for (size_t i = 0; i < addrTable.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << addrTable[i];
    }
    out << "]}";
  }

  out << "\n  ]\n";
  out << "}\n";

  return true;
}

// ===========================================================================
// ConfigGen::writeMapText
// ===========================================================================

bool ConfigGen::writeMapText(const MappingState &state, const Graph &dfg,
                             const Graph &adg, const ADGFlattener &flattener,
                             llvm::ArrayRef<TechMappedEdgeKind> edgeKinds,
                             const std::string &path,
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

  out << "=== Mapping Report ===\n\n";
  const std::string stableTechMapDiagnostics =
      sanitizeTechMapDiagnosticsForArtifact(techMapDiagnostics);

  out << "--- Techmap Summary ---\n";
  if (!techMapPlan && !techMapMetrics && techMapDiagnostics.empty()) {
    out << "(not available)\n";
  } else {
    if (techMapMetrics) {
      llvm::StringRef legacyOracleStatus = "disabled";
      if (techMapMetrics->legacyOracleEnabled) {
        legacyOracleStatus =
            techMapMetrics->legacyOracleMissingCount == 0
                ? "passed"
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
      out << "coverage_score: " << techMapMetrics->coverageScore << "\n";
      out << "selected_candidates: " << techMapMetrics->selectedCandidateCount
          << "\n";
      out << "candidate_statuses: overlap="
          << techMapMetrics->rejectedOverlapCandidateCount
          << ", temporal=" << techMapMetrics->rejectedTemporalCandidateCount
          << ", support="
          << techMapMetrics->rejectedSupportCapacityCandidateCount
          << ", spatial="
          << techMapMetrics->rejectedSpatialPoolCandidateCount
          << ", objective="
          << techMapMetrics->objectiveDroppedCandidateCount << "\n";
      out << "selection_components: "
          << techMapMetrics->selectionComponentCount << " (exact="
          << techMapMetrics->exactComponentCount << ", cpsat="
          << techMapMetrics->cpSatComponentCount << ", greedy="
          << techMapMetrics->greedyComponentCount << ")\n";
      out << "support_classes: " << techMapMetrics->supportClassCount
          << ", config_classes: " << techMapMetrics->configClassCount << "\n";
      out << "fallback_nodes: " << techMapMetrics->conservativeFallbackCount
          << " (no_candidate=" << techMapMetrics->fallbackNoCandidateCount
          << ", rejected=" << techMapMetrics->fallbackRejectedCount << ")\n";
      out << "selected_from_legacy_fallback: "
          << techMapMetrics->selectedLegacyFallbackCount << "\n";
      out << "selected_from_legacy_derived_support: "
          << techMapMetrics->selectedLegacyDerivedCount
          << "\n";
      out << "legacy_fallback_candidates: "
          << techMapMetrics->legacyFallbackCandidateCount << "\n";
      out << "legacy_contaminated_candidates: "
          << techMapMetrics->legacyContaminatedCandidateCount << "\n";
      out << "legacy_derived_sources: "
          << techMapMetrics->legacyDerivedSourceCount << "\n";
      out << "legacy_fallback_sources: "
          << techMapMetrics->legacyFallbackCount << "\n";
      out << "demand_driven_primary_plan: "
          << ((techMapPlan &&
               TechMapper::isDemandDrivenPrimaryPlan(*techMapPlan))
                  ? "true"
                  : "false")
          << "\n";
      out << "selected_uses_legacy_derived_support: "
          << (selectedUsesLegacyDerivedSupport ? "true" : "false") << "\n";
      out << "candidate_pool_uses_legacy_derived_support: "
          << (candidatePoolUsesLegacyDerivedSupport ? "true" : "false")
          << "\n";
      out << "source_pool_uses_legacy_derived_support: "
          << (sourcePoolUsesLegacyDerivedSupport ? "true" : "false") << "\n";
      out << "selected_from_mixed_origin: "
          << techMapMetrics->selectedMixedOriginCount << "\n";
      out << "layer2_handoff_status: " << layer2HandoffStatus << "\n";
      out << "layer2_handoff_blockers: ";
      if (layer2HandoffBlockers.empty()) {
        out << "-";
      } else {
        for (size_t idx = 0; idx < layer2HandoffBlockers.size(); ++idx) {
          if (idx != 0)
            out << ",";
          out << layer2HandoffBlockers[idx];
        }
      }
      out << "\n";
      out << "oracle: " << legacyOracleStatus << " (checks="
          << techMapMetrics->legacyOracleCheckCount
          << ", missing=" << techMapMetrics->legacyOracleMissingCount << ")\n";
      out << "feedback: reselections="
          << techMapMetrics->feedbackReselectionCount
          << ", filtered_candidates="
          << techMapMetrics->feedbackFilteredCandidateCount
          << ", penalty_terms=" << techMapMetrics->feedbackPenaltyCount
          << ", unknown_candidate_refs="
          << techMapMetrics->feedbackUnknownCandidateRefCount
          << ", unknown_family_refs="
          << techMapMetrics->feedbackUnknownFamilyRefCount
          << ", unknown_config_refs="
          << techMapMetrics->feedbackUnknownConfigClassRefCount
          << "\n";
      out << "timing_us: total=0, candidate_gen=0, selection=0\n";
    }
    if (techMapPlan) {
      llvm::ArrayRef<TechMapper::Unit> units = TechMapper::allUnits(*techMapPlan);
      out << "selected_units: " << units.size() << "\n";
      out << "candidate_table_entries: "
          << TechMapper::allCandidateSummaries(*techMapPlan).size() << "\n";
      if (TechMapper::hasAppliedFeedback(*techMapPlan)) {
        const auto &feedback = TechMapper::appliedFeedback(*techMapPlan);
        const auto &resolution = TechMapper::feedbackResolution(*techMapPlan);
        out << "feedback_request:";
        out << " banned_candidates=" << feedback.bannedCandidateIds.size();
        out << " banned_families=" << feedback.bannedFamilyIds.size();
        out << " banned_config_classes="
            << feedback.bannedConfigClassIds.size();
        out << " split_candidates=" << feedback.splitCandidateIds.size();
        out << " candidate_penalties=" << feedback.candidatePenalties.size();
        out << " family_penalties=" << feedback.familyPenalties.size();
        out << " config_class_penalties="
            << feedback.configClassPenalties.size();
        out << " unknown_candidate_refs="
            << TechMapper::feedbackUnknownCandidateRefCount(*techMapPlan);
        out << " unknown_family_refs="
            << TechMapper::feedbackUnknownFamilyRefCount(*techMapPlan);
        out << " unknown_config_refs="
            << TechMapper::feedbackUnknownConfigClassRefCount(*techMapPlan)
            << "\n";
        if (!feedback.bannedCandidateIds.empty()) {
          out << "  banned_candidate_ids=";
          for (size_t idx = 0; idx < feedback.bannedCandidateIds.size(); ++idx) {
            if (idx != 0)
              out << ",";
            writeCandidateFeedbackRefText(out, techMapPlan,
                                          feedback.bannedCandidateIds[idx]);
          }
          out << "\n";
        }
        if (!feedback.bannedFamilyIds.empty()) {
          out << "  banned_family_ids=";
          for (size_t idx = 0; idx < feedback.bannedFamilyIds.size(); ++idx) {
            if (idx != 0)
              out << ",";
            writeFamilyFeedbackRefText(out, techMapPlan,
                                       feedback.bannedFamilyIds[idx]);
          }
          out << "\n";
        }
        if (!feedback.bannedConfigClassIds.empty()) {
          out << "  banned_config_class_ids=";
          for (size_t idx = 0; idx < feedback.bannedConfigClassIds.size(); ++idx) {
            if (idx != 0)
              out << ",";
            writeConfigClassFeedbackRefText(out, techMapPlan,
                                            feedback.bannedConfigClassIds[idx]);
          }
          out << "\n";
        }
        if (!feedback.splitCandidateIds.empty()) {
          out << "  split_candidate_ids=";
          for (size_t idx = 0; idx < feedback.splitCandidateIds.size(); ++idx) {
            if (idx != 0)
              out << ",";
            writeCandidateFeedbackRefText(out, techMapPlan,
                                          feedback.splitCandidateIds[idx]);
          }
          out << "\n";
        }
        if (!feedback.candidatePenalties.empty()) {
          out << "  candidate_penalties=";
          for (size_t idx = 0; idx < feedback.candidatePenalties.size(); ++idx) {
            if (idx != 0)
              out << ",";
            writeCandidateFeedbackRefText(out, techMapPlan,
                                          feedback.candidatePenalties[idx].id);
            out << ":" << feedback.candidatePenalties[idx].penalty;
          }
          out << "\n";
        }
        if (!feedback.familyPenalties.empty()) {
          out << "  family_penalties=";
          for (size_t idx = 0; idx < feedback.familyPenalties.size(); ++idx) {
            if (idx != 0)
              out << ",";
            writeFamilyFeedbackRefText(out, techMapPlan,
                                       feedback.familyPenalties[idx].id);
            out << ":" << feedback.familyPenalties[idx].penalty;
          }
          out << "\n";
        }
        if (!feedback.configClassPenalties.empty()) {
          out << "  config_class_penalties=";
          for (size_t idx = 0; idx < feedback.configClassPenalties.size(); ++idx) {
            if (idx != 0)
              out << ",";
            writeConfigClassFeedbackRefText(
                out, techMapPlan, feedback.configClassPenalties[idx].id);
            out << ":" << feedback.configClassPenalties[idx].penalty;
          }
          out << "\n";
        }
        if (!resolution.unknownBannedCandidateIds.empty()) {
          out << "  unknown_banned_candidate_ids=";
          for (size_t idx = 0; idx < resolution.unknownBannedCandidateIds.size();
               ++idx) {
            if (idx != 0)
              out << ",";
            out << resolution.unknownBannedCandidateIds[idx];
          }
          out << "\n";
        }
        if (!resolution.unknownBannedFamilyIds.empty()) {
          out << "  unknown_banned_family_ids=";
          for (size_t idx = 0; idx < resolution.unknownBannedFamilyIds.size();
               ++idx) {
            if (idx != 0)
              out << ",";
            out << resolution.unknownBannedFamilyIds[idx];
          }
          out << "\n";
        }
        if (!resolution.unknownBannedConfigClassIds.empty()) {
          out << "  unknown_banned_config_class_ids=";
          for (size_t idx = 0; idx < resolution.unknownBannedConfigClassIds.size();
               ++idx) {
            if (idx != 0)
              out << ",";
            out << resolution.unknownBannedConfigClassIds[idx];
          }
          out << "\n";
        }
        if (!resolution.unknownSplitCandidateIds.empty()) {
          out << "  unknown_split_candidate_ids=";
          for (size_t idx = 0; idx < resolution.unknownSplitCandidateIds.size();
               ++idx) {
            if (idx != 0)
              out << ",";
            out << resolution.unknownSplitCandidateIds[idx];
          }
          out << "\n";
        }
        if (!resolution.unknownCandidatePenalties.empty()) {
          out << "  unknown_candidate_penalties=";
          for (size_t idx = 0; idx < resolution.unknownCandidatePenalties.size();
               ++idx) {
            if (idx != 0)
              out << ",";
            out << resolution.unknownCandidatePenalties[idx].id << ":"
                << resolution.unknownCandidatePenalties[idx].penalty;
          }
          out << "\n";
        }
        if (!resolution.unknownFamilyPenalties.empty()) {
          out << "  unknown_family_penalties=";
          for (size_t idx = 0; idx < resolution.unknownFamilyPenalties.size();
               ++idx) {
            if (idx != 0)
              out << ",";
            out << resolution.unknownFamilyPenalties[idx].id << ":"
                << resolution.unknownFamilyPenalties[idx].penalty;
          }
          out << "\n";
        }
        if (!resolution.unknownConfigClassPenalties.empty()) {
          out << "  unknown_config_class_penalties=";
          for (size_t idx = 0;
               idx < resolution.unknownConfigClassPenalties.size(); ++idx) {
            if (idx != 0)
              out << ",";
            out << resolution.unknownConfigClassPenalties[idx].id << ":"
                << resolution.unknownConfigClassPenalties[idx].penalty;
          }
          out << "\n";
        }
      }
      if (!TechMapper::allConfigClasses(*techMapPlan).empty()) {
        out << "config_class_details:\n";
        for (const auto &info : TechMapper::allConfigClasses(*techMapPlan)) {
          llvm::ArrayRef<unsigned> compatibleConfigClasses =
              TechMapper::compatibleConfigClasses(*techMapPlan, info.id);
          out << "  config_class[" << info.id << "]"
              << " key=" << info.key
              << " temporal=" << (info.temporal ? "true" : "false")
              << " reason=" << info.reason;
          if (!compatibleConfigClasses.empty()) {
            out << " compatible_with=";
            for (size_t idx = 0; idx < compatibleConfigClasses.size(); ++idx) {
              if (idx != 0)
                out << ",";
              unsigned compatId = compatibleConfigClasses[idx];
              out << compatId;
              llvm::StringRef compatKey =
                  lookupConfigClassKey(techMapPlan, compatId);
              if (!compatKey.empty())
                out << "[" << compatKey << "]";
            }
          }
          out << "\n";
        }
      }
      if (!TechMapper::allSupportClasses(*techMapPlan).empty()) {
        out << "support_class_details:\n";
        for (const auto &info : TechMapper::allSupportClasses(*techMapPlan)) {
          out << "  support_class[" << info.id << "]"
              << " key=" << info.key
              << " kind=" << info.kind
              << " temporal=" << (info.temporal ? "true" : "false")
              << " enforce_hard_capacity="
              << (info.enforceHardCapacity ? "true" : "false")
              << " capacity=" << info.capacity;
          if (!info.hwNodeIds.empty()) {
            out << " hw_nodes=";
            for (size_t idx = 0; idx < info.hwNodeIds.size(); ++idx) {
              if (idx != 0)
                out << ",";
              out << info.hwNodeIds[idx];
            }
          }
          if (!info.peNames.empty()) {
            out << " pe_names=";
            for (size_t idx = 0; idx < info.peNames.size(); ++idx) {
              if (idx != 0)
                out << ",";
              out << info.peNames[idx];
            }
          }
          out << "\n";
        }
      }
      if (!TechMapper::temporalIncompatibilities(*techMapPlan).empty()) {
        out << "temporal_incompatibilities:\n";
        for (const auto &info :
             TechMapper::temporalIncompatibilities(*techMapPlan)) {
          out << "  lhs=" << info.lhsConfigClassId;
          llvm::StringRef lhsKey =
              lookupConfigClassKey(techMapPlan, info.lhsConfigClassId);
          if (!lhsKey.empty())
            out << "[" << lhsKey << "]";
          out << " rhs=" << info.rhsConfigClassId;
          llvm::StringRef rhsKey =
              lookupConfigClassKey(techMapPlan, info.rhsConfigClassId);
          if (!rhsKey.empty())
            out << "[" << rhsKey << "]";
          out << " reason=" << info.reason << "\n";
        }
      }
      if (!TechMapper::allSelectionComponents(*techMapPlan).empty()) {
        out << "selection_component_details:\n";
        for (const auto &component :
             TechMapper::allSelectionComponents(*techMapPlan)) {
          out << "  component[" << component.id << "] solver="
              << component.solver
              << " candidates=" << component.candidateCount
              << " selected=" << component.selectedCount
              << " base_max_score=" << component.baseMaxCandidateScore
              << " max_score=" << component.maxCandidateScore
              << " base_selected_score_sum="
              << component.baseSelectedScoreSum
              << " selected_score_sum=" << component.selectedScoreSum
              << " sw_nodes=" << component.swNodeCount
              << " temporal="
              << (component.containsTemporalCandidate ? "true" : "false");
          if (!component.selectedCandidateIds.empty()) {
            out << " selected_candidate_ids=";
            for (size_t idx = 0; idx < component.selectedCandidateIds.size();
                 ++idx) {
              if (idx != 0)
                out << ",";
              out << component.selectedCandidateIds[idx];
            }
          }
          if (!component.filteredCandidateIds.empty()) {
            out << " filtered_candidate_ids=";
            for (size_t idx = 0; idx < component.filteredCandidateIds.size();
                 ++idx) {
              if (idx != 0)
                out << ",";
              out << component.filteredCandidateIds[idx];
            }
          }
          if (!component.selectedUnitIndices.empty()) {
            out << " selected_unit_indices=";
            for (size_t idx = 0; idx < component.selectedUnitIndices.size();
                 ++idx) {
              if (idx != 0)
                out << ",";
              out << component.selectedUnitIndices[idx];
            }
          }
          out << "\n";
        }
      }
      if (!TechMapper::allFallbackNodes(*techMapPlan).empty()) {
        out << "fallback_node_details:\n";
        for (const auto &fallback : TechMapper::allFallbackNodes(*techMapPlan)) {
          out << "  sw_node=" << fallback.swNodeId
              << " contracted_node=";
          if (fallback.contractedNodeId == INVALID_ID)
            out << "-";
          else
            out << fallback.contractedNodeId;
          out << " reason=" << fallback.reason
              << " component=";
          writeOptionalUIntText(out, fallback.selectionComponentId);
          out << " candidate_hw_nodes=" << fallback.candidateHwNodeCount
              << " support_classes=" << fallback.supportClassCount
              << " config_classes=" << fallback.configClassCount;
          if (!fallback.supportClassIds.empty()) {
            out << " support_class_ids=";
            for (size_t idx = 0; idx < fallback.supportClassIds.size(); ++idx) {
              if (idx != 0)
                out << ",";
              out << fallback.supportClassIds[idx];
            }
            out << " support_class_keys=";
            for (size_t idx = 0; idx < fallback.supportClassIds.size(); ++idx) {
              if (idx != 0)
                out << ",";
              out << lookupSupportClassKey(techMapPlan,
                                           fallback.supportClassIds[idx]);
            }
            out << " support_class_kinds=";
            for (size_t idx = 0; idx < fallback.supportClassIds.size(); ++idx) {
              const auto *supportClass =
                  TechMapper::findSupportClass(*techMapPlan,
                                               fallback.supportClassIds[idx]);
              if (idx != 0)
                out << ",";
              out << (supportClass ? supportClass->kind : "");
            }
            out << " support_class_temporal=";
            for (size_t idx = 0; idx < fallback.supportClassIds.size(); ++idx) {
              const auto *supportClass =
                  TechMapper::findSupportClass(*techMapPlan,
                                               fallback.supportClassIds[idx]);
              if (idx != 0)
                out << ",";
              out << (supportClass && supportClass->temporal ? "true"
                                                             : "false");
            }
            out << " support_class_enforce_hard_capacity=";
            for (size_t idx = 0; idx < fallback.supportClassIds.size(); ++idx) {
              const auto *supportClass =
                  TechMapper::findSupportClass(*techMapPlan,
                                               fallback.supportClassIds[idx]);
              if (idx != 0)
                out << ",";
              out << (supportClass && supportClass->enforceHardCapacity
                          ? "true"
                          : "false");
            }
            out << " support_class_capacities=";
            for (size_t idx = 0; idx < fallback.supportClassIds.size(); ++idx) {
              const auto *supportClass =
                  TechMapper::findSupportClass(*techMapPlan,
                                               fallback.supportClassIds[idx]);
              if (idx != 0)
                out << ",";
              out << (supportClass ? supportClass->capacity : 0);
            }
          }
          if (!fallback.configClassIds.empty()) {
            out << " config_class_ids=";
            for (size_t idx = 0; idx < fallback.configClassIds.size(); ++idx) {
              if (idx != 0)
                out << ",";
              out << fallback.configClassIds[idx];
            }
            out << " config_class_keys=";
            for (size_t idx = 0; idx < fallback.configClassIds.size(); ++idx) {
              if (idx != 0)
                out << ",";
              out << lookupConfigClassKey(techMapPlan,
                                          fallback.configClassIds[idx]);
            }
            out << " config_class_reasons=";
            for (size_t idx = 0; idx < fallback.configClassIds.size(); ++idx) {
              const auto *configClass =
                  TechMapper::findConfigClass(*techMapPlan,
                                              fallback.configClassIds[idx]);
              if (idx != 0)
                out << ",";
              out << (configClass ? configClass->reason : "");
            }
            out << " config_class_temporal=";
            for (size_t idx = 0; idx < fallback.configClassIds.size(); ++idx) {
              const auto *configClass =
                  TechMapper::findConfigClass(*techMapPlan,
                                              fallback.configClassIds[idx]);
              if (idx != 0)
                out << ",";
              out << (configClass && configClass->temporal ? "true"
                                                           : "false");
            }
          }
          if (!fallback.candidateIds.empty()) {
            out << " candidate_ids=";
            for (size_t idx = 0; idx < fallback.candidateIds.size(); ++idx) {
              if (idx != 0)
                out << ",";
              out << fallback.candidateIds[idx];
            }
          }
          out << "\n";
        }
      }
      if (!TechMapper::legacyOracleMissingSamples(*techMapPlan).empty()) {
        llvm::ArrayRef<TechMapper::LegacyOracleSampleInfo> legacySamples =
            TechMapper::legacyOracleMissingSamples(*techMapPlan);
        out << "legacy_oracle_missing_samples:\n";
        for (const auto &sample : legacySamples) {
          out << "  key=" << sample.key;
          if (sample.familyIndex != std::numeric_limits<unsigned>::max())
            out << " family_index=" << sample.familyIndex;
          if (!sample.familySignature.empty())
            out << " family_sig=" << sample.familySignature;
          if (sample.hwNodeId != INVALID_ID)
            out << " hw_node=" << sample.hwNodeId;
          if (!sample.peName.empty())
            out << " pe=" << sample.peName;
          if (!sample.hwName.empty())
            out << " hw_name=" << sample.hwName;
          out << "\n";
        }
      }
      for (size_t unitIdx = 0; unitIdx < units.size(); ++unitIdx) {
        const auto &unit = units[unitIdx];
        const auto *selectionComponent =
            TechMapper::findSelectionComponent(*techMapPlan, unit);
        const auto *familyInfo =
            TechMapper::findFamilyTechInfo(*techMapPlan, unit);
        const auto *configClassInfo =
            TechMapper::findSelectedUnitConfigClass(*techMapPlan, unit);
        const auto *selectedCandidateSummary =
            TechMapper::findSelectedCandidateSummary(*techMapPlan, unit);
        const auto *preferredCandidate =
            TechMapper::findPreferredUnitCandidate(unit);
        const TechMapper::SupportClassInfo *preferredSupportClass =
            TechMapper::findPreferredUnitSupportClass(*techMapPlan, unit);
        out << "  unit[" << unitIdx << "]";
        if (unit.contractedNodeId != INVALID_ID)
          out << " contracted_node=" << unit.contractedNodeId;
        out << " component=";
        writeOptionalUIntText(out, unit.selectionComponentId);
        if (selectionComponent)
          out << " solver=" << selectionComponent->solver;
        out << " family=" << unit.familyIndex;
        if (familyInfo)
          out << " sig=" << familyInfo->signature;
        out << " sw_nodes=";
        for (size_t swIdx = 0; swIdx < unit.swNodes.size(); ++swIdx) {
          if (swIdx != 0)
            out << ",";
          out << unit.swNodes[swIdx];
        }
        out << " config_class=" << unit.configClassId;
        if (configClassInfo)
          out << " config_key=" << configClassInfo->key
              << " config_reason=" << configClassInfo->reason
              << " config_temporal="
              << (configClassInfo->temporal ? "true" : "false");
        out << " preferred_candidate_index=" << unit.preferredCandidateIndex;
        int64_t baseSelectionScore = unit.selectionScore;
        int64_t candidatePenalty = 0;
        int64_t familyPenalty = 0;
        int64_t configClassPenalty = 0;
        int64_t familyScarcityPenalty =
            computeFamilyScarcityPenalty(techMapPlan, unit.familyIndex);
        if (selectedCandidateSummary) {
          baseSelectionScore = selectedCandidateSummary->baseSelectionScore;
          candidatePenalty = selectedCandidateSummary->candidatePenalty;
          familyPenalty = selectedCandidateSummary->familyPenalty;
          configClassPenalty = selectedCandidateSummary->configClassPenalty;
        }
        out << " candidate_id=" << unit.selectedCandidateId;
        out << " demand_origin=" << (unit.demandOrigin ? "true" : "false");
        out << " legacy_fallback="
            << (unit.legacyFallbackOrigin ? "true" : "false");
        out << " mixed_origin=" << (unit.mixedOrigin ? "true" : "false");
        out << " origin_kind="
            << TechMapper::originKind(unit.demandOrigin,
                                      unit.legacyFallbackOrigin,
                                      unit.mixedOrigin);
        out << " internal_edges=" << unit.internalEdges.size();
        out << " inputs=" << unit.inputBindings.size();
        out << " outputs=" << unit.outputBindings.size();
        out << " candidates=" << unit.candidates.size();
        if (!unit.candidates.empty()) {
          out << " candidate_hw_nodes=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            out << unit.candidates[candIdx].hwNodeId;
          }
          out << " candidate_pe_names=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            const Node *candidateHwNode =
                adg.getNode(unit.candidates[candIdx].hwNodeId);
            if (candIdx != 0)
              out << ",";
            out << (candidateHwNode
                        ? getNodeAttrStr(candidateHwNode, "pe_name")
                        : llvm::StringRef());
          }
          out << " candidate_temporal=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            out << (unit.candidates[candIdx].temporal ? "true" : "false");
          }
        }
        out << " base_score=" << baseSelectionScore;
        out << " family_scarcity_penalty=" << familyScarcityPenalty;
        out << " candidate_penalty=" << candidatePenalty;
        out << " family_penalty=" << familyPenalty;
        out << " config_penalty=" << configClassPenalty;
        out << " score=" << unit.selectionScore;
        if (preferredCandidate) {
          const Node *preferredHwNode = adg.getNode(preferredCandidate->hwNodeId);
          out << " preferred_hw=" << preferredCandidate->hwNodeId
              << " preferred_pe="
              << (preferredHwNode ? getNodeAttrStr(preferredHwNode, "pe_name")
                                  : llvm::StringRef())
              << " support_class=" << preferredCandidate->supportClassId;
          if (preferredSupportClass)
            out << " support_key=" << preferredSupportClass->key
                << " support_kind=" << preferredSupportClass->kind
                << " support_temporal="
                << (preferredSupportClass->temporal ? "true" : "false")
                << " support_enforce_hard_capacity="
                << (preferredSupportClass->enforceHardCapacity ? "true"
                                                               : "false")
                << " support_capacity=" << preferredSupportClass->capacity;
          const auto *preferredConfigClass =
              TechMapper::findPreferredUnitConfigClass(*techMapPlan, unit);
          if (preferredConfigClass)
            out << " preferred_config_class="
                << preferredCandidate->configClassId
                << " preferred_config_key=" << preferredConfigClass->key
                << " preferred_config_reason=" << preferredConfigClass->reason
                << " preferred_config_temporal="
                << (preferredConfigClass->temporal ? "true" : "false");
          out << " preferred_temporal="
              << (preferredCandidate->temporal ? "true" : "false");
          if (!preferredCandidate->configFields.empty()) {
            out << " preferred_config_fields=";
            for (size_t fieldIdx = 0;
                 fieldIdx < preferredCandidate->configFields.size();
                 ++fieldIdx) {
              if (fieldIdx != 0)
                out << ";";
              out << summarizeConfigField(
                  preferredCandidate->configFields[fieldIdx]);
            }
          }
        }
        if (!unit.candidates.empty()) {
          out << " candidate_support_classes=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            out << unit.candidates[candIdx].supportClassId;
          }
          out << " candidate_support_class_keys=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            const auto *supportClass =
                TechMapper::findSupportClass(*techMapPlan, unit.candidates[candIdx]);
            out << (supportClass ? supportClass->key : "");
          }
          out << " candidate_support_class_kinds=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            const auto *supportClass =
                TechMapper::findSupportClass(*techMapPlan, unit.candidates[candIdx]);
            out << (supportClass ? supportClass->kind : "");
          }
          out << " candidate_support_class_temporal=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            const auto *supportClass =
                TechMapper::findSupportClass(*techMapPlan, unit.candidates[candIdx]);
            out << (supportClass && supportClass->temporal ? "true" : "false");
          }
          out << " candidate_support_class_enforce_hard_capacity=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            const auto *supportClass =
                TechMapper::findSupportClass(*techMapPlan, unit.candidates[candIdx]);
            out << (supportClass && supportClass->enforceHardCapacity ? "true"
                                                                      : "false");
          }
          out << " candidate_support_class_capacities=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            const auto *supportClass =
                TechMapper::findSupportClass(*techMapPlan, unit.candidates[candIdx]);
            out << (supportClass ? supportClass->capacity : 0);
          }
          out << " candidate_config_classes=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            out << unit.candidates[candIdx].configClassId;
          }
          out << " candidate_config_class_keys=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            const auto *configClass =
                TechMapper::findConfigClass(*techMapPlan, unit.candidates[candIdx]);
            out << (configClass ? configClass->key : "");
          }
          out << " candidate_config_class_reasons=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            const auto *configClass =
                TechMapper::findConfigClass(*techMapPlan, unit.candidates[candIdx]);
            out << (configClass ? configClass->reason : "");
          }
          out << " candidate_config_class_temporal=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << ",";
            const auto *configClass =
                TechMapper::findConfigClass(*techMapPlan, unit.candidates[candIdx]);
            out << (configClass && configClass->temporal ? "true" : "false");
          }
          out << " candidate_config_fields=";
          for (size_t candIdx = 0; candIdx < unit.candidates.size(); ++candIdx) {
            if (candIdx != 0)
              out << "|";
            const auto &candidate = unit.candidates[candIdx];
            for (size_t fieldIdx = 0; fieldIdx < candidate.configFields.size();
                 ++fieldIdx) {
              if (fieldIdx != 0)
                out << ";";
              out << summarizeConfigField(candidate.configFields[fieldIdx]);
            }
          }
        }
        out << "\n";
      }
    }
    if (!stableTechMapDiagnostics.empty())
      out << "diagnostics: " << stableTechMapDiagnostics << "\n";
  }
  out << "\n";

  out << "--- Search Summary ---\n";
  if (!searchSummary) {
    out << "(not available)\n\n";
  } else {
    out << "placement_seed_lane_count: "
        << searchSummary->placementSeedLaneCount << "\n";
    out << "successful_placement_seed_count: "
        << searchSummary->successfulPlacementSeedCount << "\n";
    out << "routed_lane_count: " << searchSummary->routedLaneCount << "\n";
    out << "local_repair_attempts: " << searchSummary->localRepairAttempts
        << "\n";
    out << "local_repair_successes: " << searchSummary->localRepairSuccesses
        << "\n";
    out << "route_aware_refinement_passes: "
        << searchSummary->routeAwareRefinementPasses << "\n";
    out << "route_aware_checkpoint_rescore_passes: "
        << searchSummary->routeAwareCheckpointRescorePasses << "\n";
    out << "route_aware_checkpoint_restore_count: "
        << searchSummary->routeAwareCheckpointRestoreCount << "\n";
    out << "route_aware_neighborhood_attempts: "
        << searchSummary->routeAwareNeighborhoodAttempts << "\n";
    out << "route_aware_neighborhood_accepted_moves: "
        << searchSummary->routeAwareNeighborhoodAcceptedMoves << "\n";
    out << "route_aware_coarse_fallback_moves: "
        << searchSummary->routeAwareCoarseFallbackMoves << "\n";
    out << "fifo_bufferization_accepted_toggles: "
        << searchSummary->fifoBufferizationAcceptedToggles << "\n";
    out << "outer_joint_accepted_rounds: "
        << searchSummary->outerJointAcceptedRounds << "\n\n";
  }

  out << "--- Timing Summary ---\n";
  if (!timingSummary) {
    out << "(not available)\n\n";
  } else {
    out << "estimated_critical_path_delay: "
        << timingSummary->estimatedCriticalPathDelay << "\n";
    out << "estimated_clock_period: "
        << timingSummary->estimatedClockPeriod << "\n";
    out << "estimated_initiation_interval: "
        << timingSummary->estimatedInitiationInterval << "\n";
    out << "estimated_throughput_cost: "
        << timingSummary->estimatedThroughputCost << "\n";
    out << "recurrence_pressure: " << timingSummary->recurrencePressure
        << "\n";
    out << "critical_path_edges: ";
    if (timingSummary->criticalPathEdges.empty()) {
      out << "-";
    } else {
      for (size_t idx = 0; idx < timingSummary->criticalPathEdges.size();
           ++idx) {
        if (idx != 0)
          out << ",";
        out << timingSummary->criticalPathEdges[idx];
      }
    }
    out << "\n";
    out << "fifo_buffer_count: " << timingSummary->fifoBufferCount << "\n";
    out << "forced_buffered_fifo_count: "
        << timingSummary->forcedBufferedFifoCount << "\n";
    out << "forced_buffered_fifo_nodes: ";
    if (timingSummary->forcedBufferedFifoNodes.empty()) {
      out << "-";
    } else {
      for (size_t idx = 0; idx < timingSummary->forcedBufferedFifoNodes.size();
           ++idx) {
        if (idx != 0)
          out << ",";
        out << timingSummary->forcedBufferedFifoNodes[idx];
      }
    }
    out << "\n";
    out << "forced_buffered_fifo_depths: ";
    if (timingSummary->forcedBufferedFifoDepths.empty()) {
      out << "-";
    } else {
      for (size_t idx = 0; idx < timingSummary->forcedBufferedFifoDepths.size();
           ++idx) {
        if (idx != 0)
          out << ",";
        out << timingSummary->forcedBufferedFifoDepths[idx];
      }
    }
    out << "\n";
    out << "mapper_selected_buffered_fifo_count: "
        << timingSummary->mapperSelectedBufferedFifoCount << "\n";
    out << "mapper_selected_buffered_fifo_nodes: ";
    if (timingSummary->mapperSelectedBufferedFifoNodes.empty()) {
      out << "-";
    } else {
      for (size_t idx = 0;
           idx < timingSummary->mapperSelectedBufferedFifoNodes.size();
           ++idx) {
        if (idx != 0)
          out << ",";
        out << timingSummary->mapperSelectedBufferedFifoNodes[idx];
      }
    }
    out << "\n";
    out << "mapper_selected_buffered_fifo_depths: ";
    if (timingSummary->mapperSelectedBufferedFifoDepths.empty()) {
      out << "-";
    } else {
      for (size_t idx = 0;
           idx < timingSummary->mapperSelectedBufferedFifoDepths.size();
           ++idx) {
        if (idx != 0)
          out << ",";
        out << timingSummary->mapperSelectedBufferedFifoDepths[idx];
      }
    }
    out << "\n";
    out << "bufferized_edges: ";
    if (timingSummary->bufferizedEdges.empty()) {
      out << "-";
    } else {
      for (size_t idx = 0; idx < timingSummary->bufferizedEdges.size(); ++idx) {
        if (idx != 0)
          out << ",";
        out << timingSummary->bufferizedEdges[idx];
      }
    }
    out << "\n";
    if (!timingSummary->recurrenceCycles.empty()) {
      out << "recurrence_cycles:\n";
      for (const auto &cycle : timingSummary->recurrenceCycles) {
        out << "  cycle[" << cycle.cycleId << "]"
            << " recurrence_distance=" << cycle.recurrenceDistance
            << " sequential_latency_cycles=" << cycle.sequentialLatencyCycles
            << " fifo_stage_cut_contribution="
            << cycle.fifoStageCutContribution
            << " max_interval_on_cycle=" << cycle.maxIntervalOnCycle
            << " estimated_cycle_ii=" << cycle.estimatedCycleII
            << " combinational_delay=" << cycle.combinationalDelay
            << " sw_nodes=";
        for (size_t idx = 0; idx < cycle.swNodes.size(); ++idx) {
          if (idx != 0)
            out << ",";
          out << cycle.swNodes[idx];
        }
        out << " sw_edges=";
        for (size_t idx = 0; idx < cycle.swEdges.size(); ++idx) {
          if (idx != 0)
            out << ",";
          out << cycle.swEdges[idx];
        }
        out << "\n";
      }
    }
    out << "\n";
  }

  // Node placements.
  out << "--- Node Placements ---\n";
  for (IdIndex swId = 0; swId < static_cast<IdIndex>(dfg.nodes.size());
       ++swId) {
    const Node *swNode = dfg.getNode(swId);
    if (!swNode)
      continue;

    llvm::StringRef opName = getNodeAttrStr(swNode, "op_name");
    out << "DFG[" << swId << "] " << opName;

    if (swNode->kind == Node::ModuleInputNode) {
      out << " (input sentinel)";
    } else if (swNode->kind == Node::ModuleOutputNode) {
      out << " (output sentinel)";
    }

    if (swId < state.swNodeToHwNode.size() &&
        state.swNodeToHwNode[swId] != INVALID_ID) {
      IdIndex hwId = state.swNodeToHwNode[swId];
      const Node *hwNode = adg.getNode(hwId);
      out << " -> ADG[" << hwId << "]";
      if (hwNode) {
        out << " " << getNodeAttrStr(hwNode, "op_name");
        llvm::StringRef pe = getNodeAttrStr(hwNode, "pe_name");
        if (!pe.empty())
          out << " (PE: " << pe << ")";
      }
    } else {
      // Check if this is a memref sentinel (direct binding to extmemory,
      // not mapped to an ADG sentinel node).
      bool isMemrefSentinel = false;
      if (swNode->kind == Node::ModuleInputNode &&
          !swNode->outputPorts.empty()) {
        const Port *p = dfg.getPort(swNode->outputPorts[0]);
        if (p && mlir::isa<mlir::MemRefType>(p->type))
          isMemrefSentinel = true;
      }
      if (isMemrefSentinel)
        out << " -> MEMORY_INTERFACE_BINDING";
      else
        out << " -> UNMAPPED";
    }
    out << "\n";
  }

  // Edge routings.
  out << "\n--- Edge Routings ---\n";
  for (IdIndex eid = 0; eid < static_cast<IdIndex>(dfg.edges.size()); ++eid) {
    const Edge *edge = dfg.getEdge(eid);
    if (!edge)
      continue;

    const Port *srcPort = dfg.getPort(edge->srcPort);
    const Port *dstPort = dfg.getPort(edge->dstPort);

    out << "Edge[" << eid << "] ";
    if (srcPort && srcPort->parentNode != INVALID_ID) {
      out << "node " << srcPort->parentNode << " -> ";
    }
    if (dstPort && dstPort->parentNode != INVALID_ID) {
      out << "node " << dstPort->parentNode;
    }

    if (eid < edgeKinds.size() && edgeKinds[eid] == TechMappedEdgeKind::IntraFU) {
      IdIndex hwNodeId = INVALID_ID;
      if (srcPort && srcPort->parentNode != INVALID_ID &&
          srcPort->parentNode < state.swNodeToHwNode.size())
        hwNodeId = state.swNodeToHwNode[srcPort->parentNode];
      out << " : INTRA_FU";
      if (hwNodeId != INVALID_ID)
        out << " (hw_node " << hwNodeId << ")";
    } else if (eid < edgeKinds.size() &&
               edgeKinds[eid] == TechMappedEdgeKind::TemporalReg) {
      out << " : TEMPORAL_REG";
    } else if (eid < state.swEdgeToHwPaths.size() &&
               !state.swEdgeToHwPaths[eid].empty()) {
      auto path = buildExportPathForEdge(eid, state, dfg, adg);
      // Detect synthetic memref-binding path (same port for src and dst).
      if (path.size() == 2 && path[0] == path[1]) {
        out << " : MEMORY_INTERFACE_BINDING";
      } else {
        out << " : [";
        for (size_t i = 0; i < path.size(); ++i) {
          if (i > 0)
            out << ", ";
          out << path[i];
        }
        out << "]";
      }
    } else {
      out << " : UNROUTED";
    }
    out << "\n";
  }

  out << "\n--- Temporal Registers ---\n";
  bool emittedTemporalReg = false;
  for (const auto &pe : flattener.getPEContainment()) {
    if (pe.peKind != "temporal_pe")
      continue;
    auto temporalPlan = buildTemporalConfigPlan(pe, state, dfg, adg, edgeKinds);
    for (const auto &binding : temporalPlan.registerBindings) {
      emittedTemporalReg = true;
      out << pe.peName << " reg[" << binding.registerIndex << "] <- edge["
          << binding.swEdgeId << "] writer_sw=" << binding.writerSwNode
          << " reader_sw=" << binding.readerSwNode << " writer_out="
          << binding.writerOutputIndex << " reader_in="
          << binding.readerInputIndex << "\n";
    }
  }
  if (!emittedTemporalReg)
    out << "(none)\n";

  // PE utilization summary.
  out << "\n--- PE Utilization ---\n";
  auto &peContainment = flattener.getPEContainment();
  for (auto &pe : peContainment) {
    bool used = false;
    for (IdIndex fuId : pe.fuNodeIds) {
      if (fuId < state.hwNodeToSwNodes.size() &&
          !state.hwNodeToSwNodes[fuId].empty()) {
        used = true;
        out << pe.peName << " (" << pe.row << "," << pe.col << "): "
            << getNodeAttrStr(adg.getNode(fuId), "op_name");
        out << " <- [";
        for (size_t i = 0; i < state.hwNodeToSwNodes[fuId].size(); ++i) {
          if (i > 0)
            out << ", ";
          const Node *swNode = dfg.getNode(state.hwNodeToSwNodes[fuId][i]);
          if (swNode)
            out << getNodeAttrStr(swNode, "op_name");
          else
            out << state.hwNodeToSwNodes[fuId][i];
        }
        out << "]";
        out << "\n";
      }
    }
    if (!used) {
      out << pe.peName << " (" << pe.row << "," << pe.col << "): unused\n";
    }
  }

  out << "\n--- Memory Regions ---\n";
  for (IdIndex hwId = 0; hwId < static_cast<IdIndex>(adg.nodes.size());
       ++hwId) {
    const Node *hwNode = adg.getNode(hwId);
    if (!hwNode || getNodeAttrStr(hwNode, "resource_class") != "memory")
      continue;

    out << "ADG[" << hwId << "] " << getNodeAttrStr(hwNode, "op_name")
        << " (" << getNodeAttrStr(hwNode, "op_kind") << ")";
    out << " numRegion=" << getNodeAttrInt(hwNode, "numRegion", 1) << "\n";

    auto regions = collectMemoryRegionsForNode(hwId, state, dfg, adg);
    auto addrTable = buildAddrOffsetTable(hwNode, hwId, state, dfg, adg);
    for (size_t i = 0; i < regions.size(); ++i) {
      out << "  region[" << i << "] <- DFG[" << regions[i].swNode << "]";
      out << " memref_arg=" << regions[i].memrefArgIndex;
      out << " lane_range=[" << regions[i].startLane << ", "
          << regions[i].endLane << ")";
      out << " ld=" << regions[i].ldCount;
      out << " st=" << regions[i].stCount;
      out << " elem_size_log2=" << regions[i].elemSizeLog2 << "\n";
    }
    out << "  addr_offset_table = [";
    for (size_t i = 0; i < addrTable.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << addrTable[i];
    }
    out << "]\n";
  }

  return true;
}

} // namespace fcc

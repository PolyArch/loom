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
    out << "techmap_feedback_attempts: "
        << searchSummary->techMapFeedbackAttempts << "\n";
    out << "techmap_feedback_accepted_reconfigurations: "
        << searchSummary->techMapFeedbackAcceptedReconfigurations << "\n";
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


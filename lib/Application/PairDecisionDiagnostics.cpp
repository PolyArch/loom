#include "Application/BuildDiagnostics.h"

#include "Application/Build.h"
#include "Application/RuntimeManifest.h"
#include "BuildDiagnosticsInternal.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/InvocationDiagnosticLog.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"

namespace loom::application {
using diagnostics_detail::addOptionalRoot;
using diagnostics_detail::addOptionalUnsigned;
using diagnostics_detail::encodeObjectiveScalar;
using diagnostics_detail::encodePairDecision;
using diagnostics_detail::encodeQualityProvenance;
using diagnostics_detail::encodeRoot;
using diagnostics_detail::spelling;

namespace {

using diagnostics_detail::addOptionalRoot;
using diagnostics_detail::addOptionalUnsigned;
using diagnostics_detail::encodeObjectiveScalar;
using diagnostics_detail::encodePairDecision;
using diagnostics_detail::encodeQualityProvenance;
using diagnostics_detail::encodeRoot;
using diagnostics_detail::spelling;

llvm::StringRef spelling(ApplicationObjectiveDimension value) {
  switch (value) {
  case ApplicationObjectiveDimension::HostOnlyWork:
    return "host_only_work";
  case ApplicationObjectiveDimension::DfgCycles:
    return "dfg_cycles";
  case ApplicationObjectiveDimension::CgraCycles:
    return "cgra_cycles";
  case ApplicationObjectiveDimension::HostResidualWork:
    return "host_residual_work";
  case ApplicationObjectiveDimension::CutTransferWork:
    return "cut_transfer_work";
  case ApplicationObjectiveDimension::LaunchSynchronizationWork:
    return "launch_synchronization_work";
  case ApplicationObjectiveDimension::ResourceCoreCost:
    return "resource_core_cost";
  case ApplicationObjectiveDimension::MappingWork:
    return "mapping_work";
  case ApplicationObjectiveDimension::Area:
    return "area";
  case ApplicationObjectiveDimension::Power:
    return "power";
  case ApplicationObjectiveDimension::Energy:
    return "energy";
  }
  llvm_unreachable("unknown application objective dimension");
}

llvm::StringRef spelling(ApplicationObjectiveEvidence value) {
  switch (value) {
  case ApplicationObjectiveEvidence::Exact:
    return "exact";
  case ApplicationObjectiveEvidence::SoundBound:
    return "sound_bound";
  case ApplicationObjectiveEvidence::Analytic:
    return "analytic";
  case ApplicationObjectiveEvidence::Calibrated:
    return "calibrated";
  case ApplicationObjectiveEvidence::RuntimeMeasured:
    return "runtime_measured";
  case ApplicationObjectiveEvidence::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown application objective evidence");
}

llvm::json::Object
encodeObjectiveObservation(const ApplicationObjectiveObservation &observation) {
  llvm::json::Object result;
  result["dimension"] = spelling(observation.dimension);
  if (observation.value)
    result["value"] = *observation.value;
  else
    result["value"] = nullptr;
  result["evidence"] = spelling(observation.evidence);
  result["confidence_permille"] = observation.confidencePermille;
  result["out_of_distribution"] = observation.outOfDistribution;
  return result;
}

llvm::json::Object
encodePortfolioInput(const SelectedApplicationInput &selection,
                     ApplicationPortfolioExecutionBinding binding) {
  llvm::json::Object result;
  result["application_identity"] = selection.applicationIdentity;
  result["input_name"] = selection.input.name;
  result["source_kind"] = toString(selection.source.kind);
  result["source_root"] = selection.source.root;
  result["build_entry"] = selection.build.entry;
  result["language"] = toString(selection.build.language);
  llvm::json::Array sources;
  for (const std::string &source : selection.build.sources)
    sources.push_back(source);
  result["sources"] = std::move(sources);
  llvm::json::Array compilerOptions;
  for (const std::string &option : selection.build.compilerOptions)
    compilerOptions.push_back(option);
  result["compiler_options"] = std::move(compilerOptions);
  llvm::json::Array inputCompilerOptions;
  for (const std::string &option : selection.input.compilerOptions)
    inputCompilerOptions.push_back(option);
  result["input_compiler_options"] = std::move(inputCompilerOptions);
  llvm::json::Array linkOptions;
  for (const std::string &option : selection.build.linkOptions)
    linkOptions.push_back(option);
  result["link_options"] = std::move(linkOptions);
  llvm::json::Array operatorProtocolSymbols;
  for (const std::string &symbol : selection.build.operatorProtocolSymbols)
    operatorProtocolSymbols.push_back(symbol);
  result["operator_protocol_symbols"] = std::move(operatorProtocolSymbols);
  result["declared_workload"] = selection.input.workload;
  result["declared_runtime_input"] = selection.input.runtimeInput;
  result["declared_oracle"] =
      llvm::json::Object{{"kind", toString(selection.input.oracle.kind)},
                         {"entry", selection.input.oracle.entry}};
  llvm::json::Object declaredProfile{
      {"warmup_samples", selection.input.profile.warmupSamples},
      {"measured_samples", selection.input.profile.measuredSamples},
      {"total_samples", selection.input.profile.totalSamples()},
      {"oracle_coverage", toString(selection.input.profile.oracleCoverage)},
      {"deadline_milliseconds", selection.input.profile.deadlineMilliseconds}};
  if (selection.input.profile.maximumSimulatedTicks)
    declaredProfile["maximum_simulated_ticks"] =
        *selection.input.profile.maximumSimulatedTicks;
  result["declared_profile"] = std::move(declaredProfile);
  llvm::json::Array cachedInputs;
  for (const CachedInput &input : selection.cachedInputs)
    cachedInputs.push_back(
        llvm::json::Object{{"logical_name", input.logicalName},
                           {"path", input.path},
                           {"sha256", formatBlobDigestHex(input.digest)}});
  result["cached_inputs"] = std::move(cachedInputs);
  result["execution_binding"] = toString(binding);
  result["execution_binding_established"] =
      binding ==
      ApplicationPortfolioExecutionBinding::CanonicalSimulationAndOracle;
  return result;
}

} // namespace

namespace diagnostics_detail {

llvm::json::Object
encodePairDecision(const ApplicationPairDecisionRecord &decision) {
  const auto encodeQualityObservation = [](const auto &observation) {
    llvm::json::Array codes;
    for (std::uint64_t code : observation.objectiveCodes)
      codes.push_back(code);
    llvm::json::Object result{
        {"system_mapping", encodeRoot(observation.candidate)},
        {"objective_codes", std::move(codes)},
        {"provenance", encodeQualityProvenance(observation.provenance)},
        {"incomplete_reason",
         observation.incompleteReason
             ? llvm::json::Value(spelling(*observation.incompleteReason))
             : llvm::json::Value(nullptr)}};
    addOptionalRoot(result, "evidence", observation.evidence);
    return result;
  };
  const auto encodeHardwarePromotion = [](const auto &observation) {
    llvm::json::Array codes;
    for (std::uint64_t code : observation.objectiveCodes)
      codes.push_back(code);
    llvm::json::Object result{
        {"plan_ordinal", observation.planOrdinal},
        {"system", encodeRoot(observation.system)},
        {"objective_codes", std::move(codes)},
        {"provenance", encodeQualityProvenance(observation.provenance)},
        {"incomplete_reason",
         observation.incompleteReason
             ? llvm::json::Value(spelling(*observation.incompleteReason))
             : llvm::json::Value(nullptr)},
        {"promoted_to_exact_mapping", observation.promotedToExactMapping}};
    addOptionalRoot(result, "evidence", observation.evidence);
    return result;
  };
  llvm::json::Object result{{"schema", "loom.application_pair_decision"},
                            {"version", "1.1"}};
  if (decision.portfolioInput)
    result["portfolio_input"] = encodePortfolioInput(
        *decision.portfolioInput, decision.portfolioExecutionBinding);
  else
    result["portfolio_input"] = nullptr;
  if (decision.pairIdentity)
    result["pair_identity"] =
        formatComponentViewDigestHex(*decision.pairIdentity);
  else
    result["pair_identity"] = nullptr;
  if (decision.invocationRunKey)
    result["invocation_manifest_run_key"] =
        llvm::toHex(llvm::ArrayRef<std::uint8_t>(*decision.invocationRunKey),
                    /*LowerCase=*/true);
  else
    result["invocation_manifest_run_key"] = nullptr;
  if (decision.manifestJoinOwner)
    result["manifest_join_owner"] = *decision.manifestJoinOwner;
  else
    result["manifest_join_owner"] = nullptr;
  if (decision.manifestJoinContract)
    result["manifest_join_contract"] = *decision.manifestJoinContract;
  else
    result["manifest_join_contract"] = nullptr;
  result["manifest_join_owner_verified"] = decision.manifestJoinOwnerVerified;
  result["disposition"] = toString(decision.disposition);
  result["invocation_manifest_join_status"] =
      toString(decision.manifestJoinStatus);
  addOptionalRoot(result, "source_program", decision.sourceProgram);
  addOptionalRoot(result, "fabric", decision.fabric);
  addOptionalRoot(result, "workload", decision.workload);
  addOptionalRoot(result, "runtime_input", decision.runtimeInput);
  result["planning_record_count"] = decision.planningRecordCount;
  result["non_candidate_planning_record_count"] =
      decision.nonCandidatePlanningRecordCount;
  llvm::json::Array qualityLabels;
  for (const std::string &label : decision.qualityObjectiveDimensionLabels)
    qualityLabels.push_back(label);
  result["quality_objective_dimension_labels"] = std::move(qualityLabels);
  result["quality_disposition"] = spelling(decision.qualityDisposition);
  addOptionalRoot(result, "quality_incomplete_candidate",
                  decision.qualityIncompleteCandidate);
  llvm::json::Array qualityObservations;
  for (const dse::JointDesignQualityObservation &observation :
       decision.qualityObservations)
    qualityObservations.push_back(encodeQualityObservation(observation));
  result["quality_observations"] = std::move(qualityObservations);
  llvm::json::Array hardwarePromotionLabels;
  for (const std::string &label :
       decision.hardwarePromotionObjectiveDimensionLabels)
    hardwarePromotionLabels.push_back(label);
  result["hardware_promotion_objective_dimension_labels"] =
      std::move(hardwarePromotionLabels);
  llvm::json::Array hardwarePromotions;
  for (const dse::JointHardwarePromotionObservation &observation :
       decision.hardwarePromotionObservations)
    hardwarePromotions.push_back(encodeHardwarePromotion(observation));
  result["hardware_promotion_observations"] = std::move(hardwarePromotions);
  llvm::json::Array qualityInvocations;
  for (const ApplicationPairQualityInvocationRecord &invocation :
       decision.qualityInvocations) {
    llvm::json::Object encoded;
    encoded["plan_ordinal_base"] = invocation.planOrdinalBase;
    if (invocation.invocationRunKey)
      encoded["invocation_manifest_run_key"] = llvm::toHex(
          llvm::ArrayRef<std::uint8_t>(*invocation.invocationRunKey),
          /*LowerCase=*/true);
    else
      encoded["invocation_manifest_run_key"] = nullptr;
    encoded["quality_disposition"] = spelling(invocation.qualityDisposition);
    addOptionalRoot(encoded, "quality_incomplete_candidate",
                    invocation.qualityIncompleteCandidate);
    addOptionalUnsigned(encoded, "selected_plan_ordinal",
                        invocation.selectedPlanOrdinal);
    addOptionalRoot(encoded, "selected_system_mapping",
                    invocation.selectedMapping);
    llvm::json::Array invocationQualityLabels;
    for (const std::string &label : invocation.qualityObjectiveDimensionLabels)
      invocationQualityLabels.push_back(label);
    encoded["quality_objective_dimension_labels"] =
        std::move(invocationQualityLabels);
    llvm::json::Array invocationQualityObservations;
    for (const dse::JointDesignQualityObservation &observation :
         invocation.qualityObservations)
      invocationQualityObservations.push_back(
          encodeQualityObservation(observation));
    encoded["quality_observations"] = std::move(invocationQualityObservations);
    llvm::json::Array invocationHardwareLabels;
    for (const std::string &label :
         invocation.hardwarePromotionObjectiveDimensionLabels)
      invocationHardwareLabels.push_back(label);
    encoded["hardware_promotion_objective_dimension_labels"] =
        std::move(invocationHardwareLabels);
    llvm::json::Array invocationHardwareObservations;
    for (const dse::JointHardwarePromotionObservation &observation :
         invocation.hardwarePromotionObservations)
      invocationHardwareObservations.push_back(
          encodeHardwarePromotion(observation));
    encoded["hardware_promotion_observations"] =
        std::move(invocationHardwareObservations);
    qualityInvocations.push_back(std::move(encoded));
  }
  result["quality_invocations"] = std::move(qualityInvocations);
  result["host_only_baseline_complete"] = decision.hostOnlyBaselineComplete;
  result["final_application_qor_complete"] =
      decision.finalApplicationQorComplete;
  if (decision.detail)
    result["detail"] = *decision.detail;
  else
    result["detail"] = nullptr;
  llvm::json::Array baseline;
  for (const ApplicationObjectiveObservation &observation :
       decision.hostOnlyBaseline)
    baseline.push_back(encodeObjectiveObservation(observation));
  result["host_only_baseline"] = std::move(baseline);
  llvm::json::Array selectedObjective;
  for (const ApplicationObjectiveObservation &observation :
       decision.selectedObjective)
    selectedObjective.push_back(encodeObjectiveObservation(observation));
  result["selected_objective"] = std::move(selectedObjective);
  addOptionalRoot(result, "selected_system_mapping",
                  decision.selectedSystemMapping);
  if (decision.selectedScheduleHintDigest)
    result["selected_schedule_hint_digest"] =
        formatComponentViewDigestHex(*decision.selectedScheduleHintDigest);
  else
    result["selected_schedule_hint_digest"] = nullptr;
  llvm::json::Array candidates;
  for (const ApplicationPairCandidateRecord &candidate : decision.candidates) {
    llvm::json::Object encoded;
    if (candidate.candidateIdentity)
      encoded["candidate_identity"] =
          formatComponentViewDigestHex(*candidate.candidateIdentity);
    else
      encoded["candidate_identity"] = nullptr;
    addOptionalRoot(encoded, "structured_program", candidate.structuredProgram);
    addOptionalRoot(encoded, "canonical_dataflow", candidate.canonicalDataflow);
    if (candidate.planningProjectionIdentity)
      encoded["planning_projection_identity"] =
          formatComponentViewDigestHex(*candidate.planningProjectionIdentity);
    else
      encoded["planning_projection_identity"] = nullptr;
    if (candidate.materializedProjectionIdentity)
      encoded["materialized_projection_identity"] =
          formatComponentViewDigestHex(
              *candidate.materializedProjectionIdentity);
    else
      encoded["materialized_projection_identity"] = nullptr;
    encoded["planning_disposition"] =
        candidate.planningDisposition
            ? llvm::json::Value(dse::toString(*candidate.planningDisposition))
            : llvm::json::Value(nullptr);
    encoded["schedule_intent"] =
        candidate.scheduleIntent
            ? llvm::json::Value(dse::toString(*candidate.scheduleIntent))
            : llvm::json::Value(nullptr);
    encoded["planning_incomplete_reason"] =
        candidate.planningIncompleteReason
            ? llvm::json::Value(
                  dse::toString(*candidate.planningIncompleteReason))
            : llvm::json::Value(nullptr);
    encoded["verified_spectrum"] =
        candidate.verifiedSpectrum
            ? llvm::json::Value(dse::toString(*candidate.verifiedSpectrum))
            : llvm::json::Value(nullptr);
    encoded["planning_record_ordinal"] = candidate.planningRecordOrdinal;
    addOptionalUnsigned(encoded, "plan_ordinal", candidate.planOrdinal);
    encoded["entered_mapping"] = candidate.enteredMapping;
    encoded["selected"] = candidate.selected;
    llvm::json::Array objective;
    for (const ApplicationObjectiveObservation &observation :
         candidate.objective)
      objective.push_back(encodeObjectiveObservation(observation));
    encoded["objective"] = std::move(objective);
    llvm::json::Array mappingObservations;
    for (const ApplicationPairMappingObservation &observation :
         candidate.mappingObservations) {
      llvm::json::Object mapping;
      mapping["plan_ordinal"] = observation.planOrdinal;
      mapping["schedule_hint_digest"] =
          formatComponentViewDigestHex(observation.scheduleHintDigest);
      mapping["system"] = encodeRoot(observation.system);
      mapping["mapping_disposition"] = spelling(observation.mappingDisposition);
      mapping["runtime_disposition"] = spelling(observation.runtimeDisposition);
      mapping["incomplete_reason"] =
          observation.incompleteReason
              ? llvm::json::Value(dse::toString(*observation.incompleteReason))
              : llvm::json::Value(nullptr);
      mapping["verified_spectrum"] =
          observation.verifiedSpectrum
              ? llvm::json::Value(dse::toString(*observation.verifiedSpectrum))
              : llvm::json::Value(nullptr);
      mapping["resource_time_verification_incomplete_reason"] =
          observation.resourceTimeSpectrumIncompleteReason
              ? llvm::json::Value(
                    dse::resourceTimeSpectrumIncompleteReasonSpelling(
                        *observation.resourceTimeSpectrumIncompleteReason))
              : llvm::json::Value(nullptr);
      llvm::json::Array mappings;
      for (const ArtifactRootReference &reference : observation.systemMappings)
        mappings.push_back(encodeRoot(reference));
      mapping["system_mappings"] = std::move(mappings);
      addOptionalRoot(mapping, "hardware_mutation_repair_record",
                      observation.hardwareMutationRepairRecord);
      llvm::json::Array runtimeEvidence;
      for (const ArtifactRootReference &reference : observation.runtimeEvidence)
        runtimeEvidence.push_back(encodeRoot(reference));
      mapping["runtime_evidence"] = std::move(runtimeEvidence);
      llvm::json::Array oracleEvidence;
      for (const ArtifactRootReference &reference : observation.oracleEvidence)
        oracleEvidence.push_back(encodeRoot(reference));
      mapping["oracle_evidence"] = std::move(oracleEvidence);
      addOptionalUnsigned(mapping, "dfg_cycles", observation.dfgCycles);
      addOptionalUnsigned(mapping, "cgra_cycles", observation.cgraCycles);
      addOptionalUnsigned(mapping, "resource_core_cost",
                          observation.resourceCoreCost);
      addOptionalUnsigned(mapping, "predicted_makespan_picoseconds",
                          observation.predictedMakespanPicoseconds);
      mapping["predicted_support"] = dse::resourceTimeEstimateSupportSpelling(
          observation.predictedSupport);
      mapping["physical_model_support"] =
          dse::resourceTimeEstimateSupportSpelling(
              observation.physicalModelSupport);
      addOptionalUnsigned(mapping, "measured_makespan_picoseconds",
                          observation.measuredMakespanPicoseconds);
      addOptionalUnsigned(mapping, "prediction_error_ppm",
                          observation.predictionErrorPartsPerMillion);
      mappingObservations.push_back(std::move(mapping));
    }
    encoded["mapping_observations"] = std::move(mappingObservations);
    candidates.push_back(std::move(encoded));
  }
  result["candidates"] = std::move(candidates);
  const ApplicationFunnelExactComparison &comparison =
      decision.funnelExactComparison;
  llvm::json::Object funnelComparison{
      {"mapped_candidates", comparison.mappedCandidates},
      {"predicted_feasible_candidates", comparison.predictedFeasibleCandidates},
      {"verified_candidates", comparison.verifiedCandidates},
      {"measured_candidates", comparison.measuredCandidates},
      {"out_of_distribution_candidates",
       comparison.outOfDistributionCandidates},
      {"best_ranking_match",
       comparison.bestRankingMatch
           ? llvm::json::Value(*comparison.bestRankingMatch)
           : llvm::json::Value(nullptr)},
      {"analytic_clock_period_picoseconds",
       comparison.analyticClockPeriodPicoseconds != 0
           ? llvm::json::Value(comparison.analyticClockPeriodPicoseconds)
           : llvm::json::Value(nullptr)},
      {"prediction_error_candidates", comparison.predictionErrorCandidates}};
  addOptionalUnsigned(funnelComparison, "maximum_prediction_error_ppm",
                      comparison.maximumPredictionErrorPartsPerMillion);
  result["funnel_exact_comparison"] = std::move(funnelComparison);
  if (decision.selectedCandidateIdentity)
    result["selected_candidate_identity"] =
        formatComponentViewDigestHex(*decision.selectedCandidateIdentity);
  else
    result["selected_candidate_identity"] = nullptr;
  if (decision.selectedSystem)
    result["selected_system"] = encodeRoot(*decision.selectedSystem);
  else
    result["selected_system"] = nullptr;
  return result;
}

} // namespace diagnostics_detail

void emitApplicationPairDecisionDiagnostics(
    const ApplicationPairDecisionRecord &decision) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::DataflowLowering,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object payload;
        payload["schema"] = "loom.application_pair_disposition";
        payload["version"] = "1.0";
        payload["domain"] = "application_pair_decision";
        payload["pair_decision"] = encodePairDecision(decision);
        return llvm::json::Value(std::move(payload));
      });
}

llvm::json::Object projectApplicationPairDecisionJson(
    const ApplicationPairDecisionRecord &decision) {
  return encodePairDecision(decision);
}

void emitApplicationRuntimeManifestDiagnostics(
    const FinalizedApplicationRuntimeManifest &manifest) {
  const ApplicationRuntimeManifest &record = manifest.manifest();
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::Deployment,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object payload;
        payload["schema"] = "loom.application_runtime_manifest_binding";
        payload["version"] = "1.1";
        payload["domain"] = "application_runtime_manifest";
        payload["runtime_manifest"] = encodeRoot(manifest.reference());
        payload["pair_identity"] =
            formatComponentViewDigestHex(record.pairIdentity());
        payload["invocation_manifest_run_key"] =
            llvm::toHex(llvm::ArrayRef<std::uint8_t>(record.invocationRunKey()),
                        /*LowerCase=*/true);
        payload["disposition"] = toString(record.pairDisposition());
        payload["selected_candidate_identity"] =
            formatComponentViewDigestHex(record.selectedCandidateIdentity());
        payload["selected_plan_ordinal"] = record.selectedPlanOrdinal();
        payload["source_program"] = encodeRoot(record.sourceProgram());
        payload["fabric"] = encodeRoot(record.fabric());
        payload["workload"] = encodeRoot(record.workload());
        payload["runtime_input"] = encodeRoot(record.runtimeInput());
        payload["selected_system"] = encodeRoot(record.selectedSystem());
        payload["selected_mapping"] = encodeRoot(record.selectedMapping());
        payload["deployment"] = encodeRoot(record.deployment());
        payload["activation_workload"] =
            encodeRoot(record.activationWorkload());
        payload["activation_runtime_input"] =
            encodeRoot(record.activationRuntimeInput());
        addOptionalRoot(payload, "selected_hardware_mutation_repair_record",
                        record.selectedHardwareMutationRepairRecord());
        llvm::json::Array repairRecords;
        for (const ArtifactRootReference &reference :
             record.hardwareMutationRepairRecords())
          repairRecords.push_back(encodeRoot(reference));
        payload["hardware_mutation_repair_records"] = std::move(repairRecords);
        return llvm::json::Value(std::move(payload));
      });
}
} // namespace loom::application

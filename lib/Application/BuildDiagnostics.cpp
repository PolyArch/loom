#include "Application/BuildDiagnostics.h"

#include "Application/Build.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/InvocationDiagnosticLog.h"
#include "Common/MappingDebugLog.h"
#include "DSE/PreMappingEvidence.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace loom::application {
namespace {

llvm::StringRef spelling(ApplicationBuildOperation operation) {
  switch (operation) {
  case ApplicationBuildOperation::ProductTargetPreparation:
    return "product_target_preparation";
  case ApplicationBuildOperation::FinalLinkImport:
    return "final_link_import";
  case ApplicationBuildOperation::ApplicationPreparation:
    return "application_preparation";
  case ApplicationBuildOperation::MappingExecution:
    return "mapping_execution";
  case ApplicationBuildOperation::MappingImport:
    return "mapping_import";
  case ApplicationBuildOperation::ConfigurationAbiDerivation:
    return "configuration_abi_derivation";
  case ApplicationBuildOperation::HardwareBindingDerivation:
    return "hardware_binding_derivation";
  case ApplicationBuildOperation::CompilerTargetResolution:
    return "compiler_target_resolution";
  case ApplicationBuildOperation::HostProgramFinalization:
    return "host_program_finalization";
  case ApplicationBuildOperation::InstructionBinaryFinalization:
    return "instruction_binary_finalization";
  case ApplicationBuildOperation::DeclarativeDeploymentFinalization:
    return "declarative_deployment_finalization";
  case ApplicationBuildOperation::DeploymentConstruction:
    return "deployment_construction";
  case ApplicationBuildOperation::PackagePublication:
    return "package_publication";
  }
  llvm_unreachable("unknown application build operation");
}

llvm::StringRef spelling(dse::JointDesignAttemptDisposition value) {
  using Disposition = dse::JointDesignAttemptDisposition;
  switch (value) {
  case Disposition::Verified:
    return "verified";
  case Disposition::ProvenNoFeasibleCandidate:
    return "proven_no_feasible_candidate";
  case Disposition::Incomplete:
    return "incomplete";
  }
  llvm_unreachable("unknown joint-design attempt disposition");
}

llvm::StringRef spelling(dse::JointDesignQualityDisposition value) {
  using Disposition = dse::JointDesignQualityDisposition;
  switch (value) {
  case Disposition::NotRequested:
    return "not_requested";
  case Disposition::Complete:
    return "complete";
  case Disposition::Unsupported:
    return "unsupported";
  case Disposition::ProofNotEstablished:
    return "proof_not_established";
  case Disposition::ExecutionFailed:
    return "execution_failed";
  case Disposition::CancelledOrTimeout:
    return "cancelled_or_timeout";
  }
  llvm_unreachable("unknown joint-design quality disposition");
}

llvm::StringRef spelling(dse::JointDesignQualityIncompleteReason value) {
  using Reason = dse::JointDesignQualityIncompleteReason;
  switch (value) {
  case Reason::Unsupported:
    return "unsupported";
  case Reason::ProofNotEstablished:
    return "proof_not_established";
  case Reason::ExecutionFailed:
    return "execution_failed";
  case Reason::CancelledOrTimeout:
    return "cancelled_or_timeout";
  }
  llvm_unreachable("unknown joint-design quality incomplete reason");
}

llvm::StringRef spelling(ApplicationMappingRuntimeDisposition value) {
  using Disposition = ApplicationMappingRuntimeDisposition;
  switch (value) {
  case Disposition::NotRequested:
    return "not_requested";
  case Disposition::Completed:
    return "completed";
  case Disposition::Unsupported:
    return "unsupported";
  case Disposition::ProofNotEstablished:
    return "proof_not_established";
  case Disposition::ExecutionFailed:
    return "execution_failed";
  case Disposition::CancelledOrTimeout:
    return "cancelled_or_timeout";
  }
  llvm_unreachable("unknown application runtime disposition");
}

llvm::json::Value encodeObjectiveScalar(
    const ResolvedObjectiveScalar &value) {
  if (const auto *integer = std::get_if<ResolvedObjectiveInteger>(&value))
    return llvm::json::Object{{"kind", "integer"},
                              {"negative", integer->negative},
                              {"magnitude", integer->magnitude}};
  const auto &decimal = std::get<ResolvedObjectiveDecimal>(value);
  return llvm::json::Object{{"kind", "decimal"},
                            {"coefficient", decimal.coefficient},
                            {"base10_exponent", decimal.base10Exponent}};
}

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

void addOptionalUnsigned(llvm::json::Object &object, llvm::StringRef key,
                         std::optional<std::uint64_t> value);
void addOptionalRoot(llvm::json::Object &object, llvm::StringRef key,
    const std::optional<ArtifactRootReference> &value);
std::string encodeRoot(const ArtifactRootReference &reference);

llvm::json::Object encodeQualityProvenance(
    const dse::JointDesignQualityProvenance &provenance) {
  llvm::json::Array rawMeasures;
  for (const ResolvedObjectiveScalar &measure : provenance.rawMeasures)
    rawMeasures.push_back(encodeObjectiveScalar(measure));
  llvm::json::Array supportingEvidence;
  for (const ArtifactRootReference &reference : provenance.supportingEvidence)
    supportingEvidence.push_back(encodeRoot(reference));
  llvm::json::Array verificationEvidence;
  for (const ArtifactRootReference &reference : provenance.verificationEvidence)
    verificationEvidence.push_back(encodeRoot(reference));
  llvm::json::Object result{
      {"raw_measures", std::move(rawMeasures)},
      {"supporting_evidence", std::move(supportingEvidence)},
      {"verification_evidence", std::move(verificationEvidence)},
      {"resource_core_cost",
       provenance.resourceCoreCost
           ? llvm::json::Value(*provenance.resourceCoreCost)
           : llvm::json::Value(nullptr)}};
  if (provenance.spatialFifoFeedback)
    result["spatial_fifo_feedback"] = llvm::json::Object{
        {"disposition", dse::spatialFifoRuntimeFeedbackDispositionSpelling(
                            provenance.spatialFifoFeedback->disposition)},
        {"reason", dse::spatialFifoRuntimeFeedbackReasonSpelling(
                       provenance.spatialFifoFeedback->reason)},
        {"parent_mapping",
         encodeRoot(provenance.spatialFifoFeedback->parentMapping)},
        {"spatial_mapping",
         encodeRoot(provenance.spatialFifoFeedback->spatialMapping)}};
  else
    result["spatial_fifo_feedback"] = nullptr;
  if (provenance.spatialOperandQueueFeedback) {
    llvm::json::Object feedback{
        {"disposition",
         dse::spatialOperandQueueRuntimeFeedbackDispositionSpelling(
             provenance.spatialOperandQueueFeedback->disposition)},
        {"reason", dse::spatialOperandQueueRuntimeFeedbackReasonSpelling(
                       provenance.spatialOperandQueueFeedback->reason)}};
    addOptionalRoot(feedback, "parent_mapping",
                    provenance.spatialOperandQueueFeedback->parentMapping);
    result["spatial_operand_queue_feedback"] = std::move(feedback);
  } else {
    result["spatial_operand_queue_feedback"] = nullptr;
  }
  if (provenance.spatialTransportFeedback) {
    llvm::json::Object feedback{
        {"disposition", dse::spatialTransportRuntimeFeedbackDispositionSpelling(
                            provenance.spatialTransportFeedback->disposition)},
        {"reason", dse::spatialTransportRuntimeFeedbackReasonSpelling(
                       provenance.spatialTransportFeedback->reason)}};
    addOptionalRoot(feedback, "parent_mapping",
                    provenance.spatialTransportFeedback->parentMapping);
    result["spatial_transport_feedback"] = std::move(feedback);
  } else {
    result["spatial_transport_feedback"] = nullptr;
  }
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
  result["declared_profile"] = llvm::json::Object{
      {"warmup_samples", selection.input.profile.warmupSamples},
      {"measured_samples", selection.input.profile.measuredSamples},
      {"total_samples", selection.input.profile.totalSamples()},
      {"oracle_coverage", toString(selection.input.profile.oracleCoverage)},
      {"deadline_milliseconds", selection.input.profile.deadlineMilliseconds}};
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

llvm::json::Object encodePairDecision(
    const ApplicationPairDecisionRecord &decision) {
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
                            {"version", "1.0"}};
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
    encoded["quality_disposition"] =
        spelling(invocation.qualityDisposition);
    addOptionalRoot(encoded, "quality_incomplete_candidate",
                    invocation.qualityIncompleteCandidate);
    addOptionalUnsigned(encoded, "selected_plan_ordinal",
                        invocation.selectedPlanOrdinal);
    addOptionalRoot(encoded, "selected_system_mapping",
                    invocation.selectedMapping);
    llvm::json::Array invocationQualityLabels;
    for (const std::string &label :
         invocation.qualityObjectiveDimensionLabels)
      invocationQualityLabels.push_back(label);
    encoded["quality_objective_dimension_labels"] =
        std::move(invocationQualityLabels);
    llvm::json::Array invocationQualityObservations;
    for (const dse::JointDesignQualityObservation &observation :
         invocation.qualityObservations)
      invocationQualityObservations.push_back(
          encodeQualityObservation(observation));
    encoded["quality_observations"] =
        std::move(invocationQualityObservations);
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
      mappingObservations.push_back(std::move(mapping));
    }
    encoded["mapping_observations"] = std::move(mappingObservations);
    candidates.push_back(std::move(encoded));
  }
  result["candidates"] = std::move(candidates);
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

std::string encodeRoot(const ArtifactRootReference &reference) {
  return llvm::toHex(encodeArtifactRootReference(reference),
                     /*LowerCase=*/true);
}

std::string encodeStructuredRoot(const frontend::StructuredEntityRef &root) {
  return llvm::toHex(frontend::encodeStructuredEntityRef(root),
                     /*LowerCase=*/true);
}

template <typename Counter>
llvm::json::Object workCounter(const Counter &counter) {
  return llvm::json::Object{
      {"limit", counter.limit},
      {"planned", counter.planned},
      {"consumed", counter.consumed},
      {"reserved", counter.reserved},
      {"rejected", counter.rejected},
      {"cancelled", counter.cancelled},
      {"elapsed_nanoseconds", counter.elapsedNanoseconds}};
}

llvm::json::Object mappingProviderWork(
    const ApplicationMappingProviderWorkObservation &work) {
  return llvm::json::Object{
      {"tech_mapping_invocations", work.techMappingInvocations},
      {"spatial_pnr_invocations", work.spatialPnrInvocations},
      {"system_pnr_invocations", work.systemPnrInvocations},
      {"tech_mapping_dispatches", work.techMappingDispatches},
      {"spatial_pnr_dispatches", work.spatialPnrDispatches},
      {"system_pnr_dispatches", work.systemPnrDispatches},
      {"tech_mapping_journal_replays", work.techMappingJournalReplays},
      {"spatial_pnr_journal_replays", work.spatialPnrJournalReplays},
      {"system_pnr_journal_replays", work.systemPnrJournalReplays}};
}

llvm::json::Object
resourceTimeFunnelObject(const dse::ResourceTimeMappingFunnel &funnel) {
  const dse::ResourceTimeMappingFunnelAccounting &accounting =
      funnel.accounting;
  llvm::json::Object result{
      {"generated_candidates", accounting.generatedCandidates},
      {"screened_candidates", accounting.screenedCandidates},
      {"detailed_frontier_candidates", accounting.detailedFrontierCandidates},
      {"successive_halving_deferred_candidates",
       accounting.successiveHalvingDeferredCandidates},
      {"sound_gate_rejected_candidates",
       accounting.soundGateRejectedCandidates},
      {"estimated_candidates", accounting.estimatedCandidates},
      {"incomplete_candidates", accounting.incompleteCandidates},
      {"mapping_eligible_schedule_hints",
       accounting.mappingEligibleScheduleHints},
      {"screening_comparison_candidates",
       accounting.screeningComparisonCandidates},
      {"detailed_schedule_feasible_candidates",
       accounting.detailedScheduleFeasibleCandidates},
      {"screening_admissible_candidates",
       accounting.screeningAdmissibleCandidates},
      {"screening_detailed_feasible_intersection",
       accounting.screeningDetailedFeasibleIntersection},
      {"screening_detailed_best_rank_matches",
       accounting.screeningDetailedBestRankMatches},
      {"screening_out_of_domain_candidates",
       accounting.screeningOutOfDomainCandidates},
      {"maximum_screening_lower_bound_gap_picoseconds",
       accounting.maximumScreeningLowerBoundGapPicoseconds},
      {"mapping_finalists", accounting.mappingFinalists},
      {"functional_replay_candidates", accounting.functionalReplayCandidates},
      {"dataflow_projection_requests", accounting.dataflowProjectionRequests},
      {"dataflow_projection_cache_hits",
       accounting.dataflowProjectionCacheHits},
      {"dataflow_projection_cache_misses",
       accounting.dataflowProjectionCacheMisses},
      {"dataflow_projection_cache_capacity_bypasses",
       accounting.dataflowProjectionCacheCapacityBypasses},
      {"dataflow_projection_cache_entries",
       accounting.dataflowProjectionCacheEntries},
      {"dataflow_projection_cache_retained_bytes",
       accounting.dataflowProjectionCacheRetainedBytes},
      {"dataflow_projection_elapsed_nanoseconds",
       accounting.dataflowProjectionElapsedNanoseconds},
      {"dataflow_materialized_candidates",
       accounting.dataflowMaterializedCandidates},
      {"mapping_plan_candidates", accounting.mappingPlanCandidates},
      {"unsupported_before_mapping_candidates",
       accounting.unsupportedBeforeMappingCandidates},
      {"unsupported_before_mapping_schedule_hints",
       accounting.unsupportedBeforeMappingScheduleHints},
      {"application_promotion_accounting_complete",
       accounting.applicationPromotionAccountingComplete},
      {"mapping_plan_constructions_avoided_by_exact_memo",
       accounting.mappingPlanConstructionsAvoidedByExactMemo},
      {"mapping_calls_deferred_by_model",
       accounting.mappingCallsDeferredByModel},
      {"mapping_calls_withheld_by_incomplete",
       accounting.mappingCallsWithheldByIncomplete},
      {"exact_invocation_memo_hits", accounting.exactInvocationMemoHits},
      {"exact_invocation_memo_misses", accounting.exactInvocationMemoMisses},
      {"exact_invocation_memo_single_flight_waits",
       accounting.exactInvocationMemoSingleFlightWaits},
      {"exact_invocation_memo_coalesced_uncached_results",
       accounting.exactInvocationMemoCoalescedUncachedResults},
      {"exact_invocation_memo_cancelled_waits",
       accounting.exactInvocationMemoCancelledWaits},
      {"exact_invocation_memo_capacity_bypasses",
       accounting.exactInvocationMemoCapacityBypasses},
      {"exact_invocation_memo_entries", accounting.exactInvocationMemoEntries},
      {"exact_invocation_memo_retained_bytes",
       accounting.exactInvocationMemoRetainedBytes},
      {"frontier_work",
       llvm::json::Object{
           {"source_projections",
            workCounter(accounting.frontierAccounting.sourceProjections)},
           {"actions", workCounter(accounting.frontierAccounting.actions)},
           {"states", workCounter(accounting.frontierAccounting.states)},
           {"estimates", workCounter(accounting.frontierAccounting.estimates)},
           {"finalists", workCounter(accounting.frontierAccounting.finalists)},
           {"state_memo_hits", accounting.frontierAccounting.stateMemoHits},
           {"state_memo_misses", accounting.frontierAccounting.stateMemoMisses},
           {"state_memo_pareto_insertions",
            accounting.frontierAccounting.stateMemoParetoInsertions},
           {"state_memo_dominated_states",
            accounting.frontierAccounting.stateMemoDominatedStates},
           {"state_memo_hit_capacity_rejections",
            accounting.frontierAccounting.stateMemoHitCapacityRejections},
           {"state_memo_miss_capacity_rejections",
            accounting.frontierAccounting.stateMemoMissCapacityRejections},
           {"states_pruned_by_beam",
            accounting.frontierAccounting.statesPrunedByBeam},
           {"terminal_hints_generated",
            accounting.frontierAccounting.terminalHintsGenerated},
           {"terminal_hints_retained",
            accounting.frontierAccounting.terminalHintsRetained},
           {"terminal_hints_pruned",
            accounting.frontierAccounting.terminalHintsPruned},
           {"incremental_lower_bound_updates",
            accounting.frontierAccounting.incrementalLowerBoundUpdates},
           {"maximum_retained_bytes",
            accounting.frontierAccounting.maximumRetainedBytes}}},
      {"elapsed_nanoseconds", accounting.elapsedNanoseconds},
      {"truncated", funnel.truncated}};
  if (funnel.incompleteReason)
    result["incomplete_reason"] =
        dse::resourceTimeFrontierIncompleteReasonSpelling(
            *funnel.incompleteReason);
  else
    result["incomplete_reason"] = nullptr;
  return result;
}

void addOptionalUnsigned(llvm::json::Object &object, llvm::StringRef key,
                         std::optional<std::uint64_t> value) {
  if (value)
    object[key] = *value;
  else
    object[key] = nullptr;
}

void addOptionalRoot(llvm::json::Object &object, llvm::StringRef key,
                     const std::optional<ArtifactRootReference> &value) {
  if (value)
    object[key] = encodeRoot(*value);
  else
    object[key] = nullptr;
}

llvm::json::Object
candidateProjection(const dse::PreMappingCandidateProjection &projection) {
  llvm::json::Object object;
  object["identity"] =
      llvm::toHex(projection.identity.bytes(), /*LowerCase=*/true);
  object["owned_region_count"] = projection.ownedRegionCount;
  object["host_region_count"] = projection.hostRegionCount;
  object["internal_dependency_count"] = projection.internalDependencyCount;
  object["internal_known_bytes"] = projection.internalKnownBytes;
  object["internal_unknown_object_count"] =
      projection.internalUnknownObjectCount;
  object["cut_dependency_count"] = projection.cutDependencyCount;
  object["cut_known_bytes"] = projection.cutKnownBytes;
  object["cut_unknown_object_count"] = projection.cutUnknownObjectCount;
  object["unknown_internal_pair_count"] = projection.unknownInternalPairCount;
  object["unknown_cut_pair_count"] = projection.unknownCutPairCount;
  object["channel_opportunity_count"] = projection.channelOpportunityCount;
  object["maximum_producer_fanout"] = projection.maximumProducerFanout;
  object["owned_dynamic_activations"] = projection.ownedDynamicActivations;
  object["owned_dynamic_leaf_executions"] =
      projection.ownedDynamicLeafExecutions;
  object["host_dynamic_activations"] = projection.hostDynamicActivations;
  object["host_dynamic_leaf_executions"] = projection.hostDynamicLeafExecutions;
  addOptionalUnsigned(object, "estimated_cut_traffic_bytes",
                      projection.estimatedCutTrafficBytes);
  object["producer_rate_lower_bound"] = projection.producerRateLowerBound;
  object["consumer_rate_lower_bound"] = projection.consumerRateLowerBound;
  object["channel_depth_lower_bound"] = projection.channelDepthLowerBound;
  object["launch_synchronization_cost"] = projection.launchSynchronizationCost;
  object["parallelism_lower_bound"] = projection.parallelismLowerBound;
  object["topology_congestion_proxy"] = projection.topologyCongestionProxy;
  object["reconfiguration_live_state_bytes"] =
      projection.reconfigurationLiveStateBytes;
  object["reconfiguration_live_state_known"] =
      projection.reconfigurationLiveStateKnown;
  object["exact_gate"] = dse::toString(projection.exactGate);
  object["estimate_support"] = dse::toString(projection.estimateSupport);
  object["estimate_confidence"] = dse::toString(projection.estimateConfidence);
  return object;
}

void addCandidateInventorySummary(
    llvm::json::Object &payload,
    llvm::ArrayRef<dse::PreMappingCandidatePlanningRecord> inventory) {
  std::uint64_t temporalHint = 0;
  std::uint64_t spatialHint = 0;
  std::uint64_t unconstrainedHint = 0;
  std::uint64_t verifiedTemporal = 0;
  std::uint64_t verifiedSpatial = 0;
  std::uint64_t verifiedIntermediate = 0;
  std::uint64_t canonical = 0;
  std::uint64_t identities = 0;
  for (const dse::PreMappingCandidatePlanningRecord &record : inventory) {
    switch (record.scheduleIntent.value_or(
        dse::PreMappingScheduleIntent::Unconstrained)) {
    case dse::PreMappingScheduleIntent::TemporalReuse:
      ++temporalHint;
      break;
    case dse::PreMappingScheduleIntent::SpatialParallel:
      ++spatialHint;
      break;
    case dse::PreMappingScheduleIntent::Unconstrained:
      ++unconstrainedHint;
      break;
    }
    canonical += record.canonicalDataflow.has_value();
    identities += record.candidateIdentity.has_value();
    if (record.verifiedSpectrum == dse::PreMappingSpectrumClass::MaxTemporal)
      ++verifiedTemporal;
    else if (record.verifiedSpectrum ==
             dse::PreMappingSpectrumClass::MaxSpatial)
      ++verifiedSpatial;
    else if (record.verifiedSpectrum ==
             dse::PreMappingSpectrumClass::Intermediate)
      ++verifiedIntermediate;
  }
  payload["candidate_inventory_count"] = inventory.size();
  payload["candidate_canonical_dataflow_count"] = canonical;
  payload["candidate_identity_count"] = identities;
  payload["temporal_schedule_hint_count"] = temporalHint;
  payload["spatial_schedule_hint_count"] = spatialHint;
  payload["unconstrained_schedule_hint_count"] = unconstrainedHint;
  payload["verified_max_temporal_count"] = verifiedTemporal;
  payload["verified_max_spatial_count"] = verifiedSpatial;
  payload["verified_intermediate_count"] = verifiedIntermediate;
}

llvm::json::Object materializedProjection(
    const dse::PreMappingMaterializedProjection &projection) {
  llvm::json::Object object;
  object["identity"] =
      llvm::toHex(projection.identity.bytes(), /*LowerCase=*/true);
  object["root_thread_launch_count"] = projection.rootThreadLaunchCount;
  object["rooted_graph_launch_count"] = projection.rootedGraphLaunchCount;
  object["static_logical_domain_point_count"] =
      projection.staticLogicalDomainPointCount;
  object["unknown_logical_domain_count"] = projection.unknownLogicalDomainCount;
  object["available_acc_core_count"] = projection.availableAccCoreCount;
  addOptionalUnsigned(object, "minimum_execution_waves",
                      projection.minimumExecutionWaves);
  addOptionalUnsigned(object, "maximum_parallel_acc_core_count",
                      projection.maximumParallelAccCoreCount);
  object["actor_count"] = projection.actorCount;
  object["compute_actor_count"] = projection.computeActorCount;
  object["control_actor_count"] = projection.controlActorCount;
  object["memory_actor_count"] = projection.memoryActorCount;
  object["graph_edge_count"] = projection.graphEdgeCount;
  object["logical_memory_root_count"] = projection.logicalMemoryRootCount;
  object["stream_actor_count"] = projection.streamActorCount;
  object["system_transport_resource_count"] =
      projection.systemTransportResourceCount;
  object["system_transfer_pattern_count"] =
      projection.systemTransferPatternCount;
  object["temporal_logical_epoch_count"] =
      projection.temporalWitness.logicalEpochCount;
  object["temporal_acc_core_occupancy"] =
      projection.temporalWitness.accCoreOccupancy;
  object["temporal_launch_count"] = projection.temporalWitness.launchCount;
  object["temporal_synchronization_count"] =
      projection.temporalWitness.synchronizationCount;
  object["temporal_live_state_bytes"] =
      projection.temporalWitness.liveStateBytes;
  object["temporal_live_state_known"] =
      projection.temporalWitness.liveStateKnown;
  object["temporal_exact"] = projection.temporalWitness.exact;
  object["logical_domain_support"] =
      dse::toString(projection.logicalDomainSupport);
  return object;
}

} // namespace

void emitApplicationBuildOperationStatistics(
    const ApplicationBuildOperationStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::Deployment,
      InvocationDiagnosticEvent::ApplicationBuildStatistics, [&] {
        llvm::json::Object payload;
        payload["operation"] = spelling(statistics.operation);
        payload["duration_ns"] = statistics.durationNanoseconds;
        payload["deterministic_work"] = statistics.deterministicWork;
#if defined(__linux__)
        struct rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss >= 0)
          // Linux reports ru_maxrss in KiB. This is a process high-water
          // observation, not a per-operation allocation attribution.
          payload["peak_resident_bytes"] =
              static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
        return llvm::json::Value(std::move(payload));
      });
}

void emitApplicationMappingExecutionPolicyStatistics(
    const ApplicationMappingExecutionPolicyStatistics &statistics) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::SystemPnr,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object payload;
        payload["domain"] = "application_mapping_execution_policy";
        payload["requested_wall_time_limit_ms"] =
            statistics.requestedWallTimeLimitMilliseconds;
        payload["dispatch_not_after_unix_ns"] =
            statistics.dispatchNotAfterUnixNanoseconds;
        payload["observed_wall_time_ns"] =
            statistics.observedWallTimeNanoseconds;
        payload["deadline_observed"] = statistics.deadlineObserved;
        return llvm::json::Value(std::move(payload));
      });
}

void emitApplicationPlanningDiagnostics(
    const PreparedApplicationBuild &prepared) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::DataflowLowering,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object payload;
        payload["domain"] = "pre_mapping_frontier";
        payload["source_program"] =
            encodeRoot(prepared.preMappingSourceProgram);
        payload["fabric"] = encodeRoot(prepared.preMappingFabric);
        payload["workload"] = encodeRoot(prepared.preMappingWorkload);
        payload["runtime_input"] = encodeRoot(prepared.preMappingRuntimeInput);
        payload["frontier_policy_digest"] =
            llvm::toHex(prepared.preMappingFrontierPolicyDigest.bytes(),
                        /*LowerCase=*/true);
        payload["stopping_policy"] = dse::jointDesignStoppingPolicySpelling(
            prepared.preMappingFrontierPolicy.stoppingPolicy);
        payload["requested_planner_mode"] =
            dse::toString(prepared.preMappingRequestedPlannerMode);
        payload["resolved_planner_mode"] =
            dse::toString(prepared.preMappingResolvedPlannerMode);
        llvm::json::Array beamWidths;
        for (std::uint64_t width :
             prepared.preMappingFrontierPolicy.beamWidthByExpansionDepth)
          beamWidths.push_back(width);
        payload["beam_width_by_expansion_depth"] = std::move(beamWidths);
        payload["diversity_candidate_count"] =
            prepared.preMappingFrontierPolicy.diversityCandidateCount;
        payload["maximum_expansion_depth"] =
            prepared.preMappingFrontierPolicy.maximumExpansionDepth;
        payload["maximum_compositional_groups"] =
            prepared.preMappingFrontierPolicy.maximumCompositionalGroups;
        payload["spectrum_endpoint"] =
            dse::toString(prepared.preMappingFrontierPolicy.spectrumEndpoint);
        payload["eligible_coordinate_count"] =
            prepared.preMappingEligibleCoordinateCount;
        payload["coordinate_frontier_truncated"] =
            prepared.preMappingCoordinateFrontierTruncated;
        payload["pre_mapping_elapsed_semantics"] = "inclusive_nested";
        payload["evaluation_timing"] = dse::serializePreMappingEvaluationTiming(
            prepared.preMappingEvaluationTiming);
        payload["source_observation_work"] =
            workCounter(prepared.preMappingWorkAccounting.sourceObservations);
        payload["coordinate_work"] =
            workCounter(prepared.preMappingWorkAccounting.coordinates);
        payload["program_materialization_work"] = workCounter(
            prepared.preMappingWorkAccounting.programMaterializations);
        payload["analytic_evaluation_work"] =
            workCounter(prepared.preMappingWorkAccounting.analyticEvaluations);
        payload["functional_replay_work"] =
            workCounter(prepared.preMappingWorkAccounting.functionalReplays);
        payload["dataflow_promotion_work"] =
            workCounter(prepared.preMappingWorkAccounting.dataflowPromotions);
        payload["mapping_pair_work"] =
            workCounter(prepared.preMappingWorkAccounting.mappingPairs);
        payload["candidate_count"] = prepared.candidateInventory.size();
        payload["funnel"] = dse::serializePreMappingFunnelSummary(
            prepared.candidateInventory, {}, prepared.preMappingWorkAccounting,
            prepared.preMappingEvaluationTiming);
        addCandidateInventorySummary(payload, prepared.candidateInventory);
        payload["selected_software_count"] = prepared.software.size();
        addOptionalUnsigned(payload, "source_host_only_work",
                            prepared.preMappingSourceHostOnlyWork);
        payload["mapping_alternative_count"] =
            prepared.mappingAlternatives.size();
        payload["resource_time_funnel"] =
            resourceTimeFunnelObject(prepared.resourceTimeFunnel);
        llvm::json::Array resourceTimeEvaluations;
        for (const dse::ResourceTimeCandidateFunnelEvaluation &evaluation :
             prepared.resourceTimeFunnel.evaluations) {
          llvm::json::Object row;
          row["candidate_identity"] =
              formatComponentViewDigestHex(evaluation.candidateIdentity);
          row["disposition"] =
              dse::resourceTimeCandidateFunnelDispositionSpelling(
                  evaluation.disposition);
          row["screening_lower_bound_picoseconds"] =
              evaluation.screeningLowerBoundPicoseconds;
          row["screening_feature_score"] = evaluation.screeningFeatureScore;
          row["screening_support"] = dse::resourceTimeEstimateSupportSpelling(
                  evaluation.screeningSupport);
          row["screening_confidence"] =
              dse::resourceTimeEstimateConfidenceSpelling(
                  evaluation.screeningConfidence);
          row["detailed_frontier_evaluated"] =
              evaluation.detailedFrontierEvaluated;
          if (evaluation.concurrencyBounds) {
            row["minimum_peak_concurrent_regions"] =
                evaluation.concurrencyBounds->minimumPeakConcurrentRegions;
            row["maximum_peak_concurrent_regions"] =
                evaluation.concurrencyBounds->maximumPeakConcurrentRegions;
            row["concurrency_bound_support"] =
                dse::resourceTimeEstimateSupportSpelling(
                    evaluation.concurrencyBounds->support);
          }
          row["accelerated_region_count"] = evaluation.acceleratedRegionCount;
          row["accelerated_graph_count"] = evaluation.acceleratedGraphCount;
          row["accelerated_actor_count"] = evaluation.acceleratedActorCount;
          const auto regionEpochCount = [&]() -> std::uint64_t {
            const auto alternative = llvm::find_if(
                prepared.mappingAlternatives, [&](const auto &candidate) {
                  return candidate.candidateIdentity ==
                         evaluation.candidateIdentity;
                });
            if (alternative == prepared.mappingAlternatives.end())
              return 0;
            std::uint64_t maximum = 0;
            for (const auto &region : alternative->resourceTimeRegions)
              maximum = std::max(maximum, region.logicalEpochCount);
            return maximum;
          }();
          row["maximum_logical_epoch_count"] = regionEpochCount;
          row["input_preference_rank"] = evaluation.inputPreferenceRank;
          row["maximum_useful_resource_units"] =
              evaluation.maximumUsefulResourceUnits;
          const std::uint64_t retainedScheduleCount = llvm::count_if(
              prepared.resourceTimeFunnel.finalists, [&](const auto &finalist) {
                return finalist.candidateIdentity ==
                       evaluation.candidateIdentity;
              });
          row["retained_mapping_schedule_count"] = retainedScheduleCount;
          row["retained_for_mapping"] = retainedScheduleCount != 0;
          const auto alternative = llvm::find_if(
              prepared.mappingAlternatives, [&](const auto &candidate) {
                return candidate.candidateIdentity ==
                       evaluation.candidateIdentity;
              });
          row["exact_static_mapping_schedule_count"] =
              alternative == prepared.mappingAlternatives.end()
                  ? 0
                  : alternative->equivalentScheduleHintDigests.size();
          if (evaluation.bestHint) {
            row["estimated_makespan_picoseconds"] =
                evaluation.bestHint->estimatedMakespanPicoseconds;
            row["peak_concurrent_regions"] =
                evaluation.bestHint->peakConcurrentRegions;
            row["estimate_support"] = dse::resourceTimeEstimateSupportSpelling(
                evaluation.bestHint->support);
          }
          if (evaluation.incompleteReason)
            row["incomplete_reason"] =
                dse::resourceTimeFrontierIncompleteReasonSpelling(
                    *evaluation.incompleteReason);
          if (evaluation.infeasibleReason)
            row["infeasible_reason"] =
                dse::resourceTimeFrontierInfeasibleReasonSpelling(
                    *evaluation.infeasibleReason);
          resourceTimeEvaluations.push_back(std::move(row));
        }
        payload["resource_time_evaluations"] =
            std::move(resourceTimeEvaluations);
        payload["joint_software_frontier_limit"] =
            prepared.jointPolicy.maximumSoftwareFrontier();
        payload["joint_system_frontier_limit"] =
            prepared.jointPolicy.maximumSystemFrontier();
        payload["joint_pair_evaluation_limit"] =
            prepared.jointPolicy.maximumPairEvaluations();
        payload["joint_tech_mapping_limit_per_module"] =
            prepared.jointPolicy.maximumTechMappingsPerModule();
        payload["joint_spatial_mapping_limit_per_pair"] =
            prepared.jointPolicy.maximumSpatialMappingsPerPair();
        payload["profile_cache_hit_count"] =
            prepared.preMappingSharedEvaluationStatistics.profileCacheHits;
        payload["profile_cache_miss_count"] =
            prepared.preMappingSharedEvaluationStatistics.profileCacheMisses;
        payload["profile_single_flight_wait_count"] =
            prepared.preMappingSharedEvaluationStatistics
                .profileSingleFlightWaits;
        const auto &cache = prepared.preMappingEvaluationCacheStatistics;
        payload["evaluation_cache"] = llvm::json::Object{
            {"analytic_primes", cache.analyticPrimeCount},
            {"analytic_hits", cache.analyticHitCount},
            {"analytic_misses", cache.analyticMissCount},
            {"analytic_single_flight_waits",
             cache.analyticSingleFlightWaitCount},
            {"functional_primes", cache.functionalPrimeCount},
            {"functional_hits", cache.functionalHitCount},
            {"functional_misses", cache.functionalMissCount},
            {"functional_single_flight_waits",
             cache.functionalSingleFlightWaitCount},
            {"dataflow_functional_single_flight_waits",
             cache.dataflowFunctionalSingleFlightWaitCount},
            {"source_observation_primes", cache.sourceObservationPrimeCount},
            {"source_observation_hits", cache.sourceObservationHitCount},
            {"source_observation_misses", cache.sourceObservationMissCount},
            {"source_observation_single_flight_waits",
             cache.sourceObservationSingleFlightWaitCount},
            {"fabric_root_single_flight_waits",
             cache.fabricRootSingleFlightWaitCount},
            {"capacity_bypasses", cache.capacityBypassCount}};
        payload["retained_incompleteness_count"] =
            prepared.retainedPreMappingIncompleteness.size();
        payload["domain_complete"] =
            prepared.preMappingCompleteness.domainComplete;
        payload["budget_complete"] =
            prepared.preMappingCompleteness.budgetComplete;
        payload["provider_complete"] =
            prepared.preMappingCompleteness.providerComplete;
        payload["evidence_complete"] =
            prepared.preMappingCompleteness.evidenceComplete;
        payload["selection_complete"] =
            prepared.preMappingCompleteness.selectionComplete;
        payload["exact_complete"] =
            prepared.preMappingCompleteness.exactComplete();
        if (prepared.preMappingShadowRecall) {
          payload["shadow_recall_eligible_subsets"] =
              prepared.preMappingShadowRecall->eligibleSubsets;
          payload["shadow_recall_generated_subsets"] =
              prepared.preMappingShadowRecall->generatedSubsets;
          payload["shadow_recall_covered_subsets"] =
              prepared.preMappingShadowRecall->coveredSubsets;
          payload["shadow_recall"] = prepared.preMappingShadowRecall->recall();
        }
        return llvm::json::Value(std::move(payload));
      });

  for (auto indexed : llvm::enumerate(prepared.candidateInventory)) {
    emitInvocationDiagnostic(
        DiagnosticVerbosity::Detail,
        InvocationDiagnosticStage::DataflowLowering,
        InvocationDiagnosticEvent::Candidate, [&] {
          const dse::PreMappingCandidatePlanningRecord &record =
              indexed.value();
          llvm::json::Object payload;
          payload["domain"] = "pre_mapping_frontier";
          payload["planning_record_ordinal"] = indexed.index();
          payload["disposition"] = dse::toString(record.disposition);
          if (record.candidateIdentity)
            payload["candidate_identity"] =
                formatComponentViewDigestHex(*record.candidateIdentity);
          else
            payload["candidate_identity"] = nullptr;
          addOptionalRoot(payload, "structured_program",
                          record.structuredProgram);
          addOptionalRoot(payload, "canonical_dataflow",
                          record.canonicalDataflow);
          llvm::json::Array ownedRoots;
          for (const frontend::StructuredEntityRef &root :
               record.ownedProtocolRoots)
            ownedRoots.push_back(encodeStructuredRoot(root));
          payload["owned_protocol_roots"] = std::move(ownedRoots);
          llvm::json::Array seedKinds;
          for (dse::PreMappingSpectrumSeedKind kind : record.seedKinds)
            seedKinds.push_back(dse::toString(kind));
          payload["seed_kinds"] = std::move(seedKinds);
          if (record.scheduleIntent)
            payload["schedule_intent"] = dse::toString(*record.scheduleIntent);
          else
            payload["schedule_intent"] = nullptr;
          if (record.projection)
            payload["projection"] = candidateProjection(*record.projection);
          else
            payload["projection"] = nullptr;
          addOptionalUnsigned(payload, "estimated_runtime_ps",
                              record.estimatedRuntimePicoseconds);
          addOptionalUnsigned(payload, "preference_rank",
                              record.preferenceRank);
          if (record.incompleteReason)
            payload["incomplete_reason"] =
                dse::toString(*record.incompleteReason);
          else
            payload["incomplete_reason"] = nullptr;
          if (record.verifiedSpectrum)
            payload["verified_spectrum"] =
                dse::toString(*record.verifiedSpectrum);
          else
            payload["verified_spectrum"] = nullptr;
          if (record.temporalWitness) {
            payload["logical_domain_fact"] = llvm::json::Object{
                {"logical_epoch_count",
                 record.temporalWitness->logicalEpochCount},
                {"acc_core_occupancy",
                 record.temporalWitness->accCoreOccupancy},
                {"launch_count", record.temporalWitness->launchCount},
                {"synchronization_count",
                 record.temporalWitness->synchronizationCount},
                {"live_state_bytes", record.temporalWitness->liveStateBytes},
                {"live_state_known", record.temporalWitness->liveStateKnown},
                {"exact", record.temporalWitness->exact}};
          } else {
            payload["logical_domain_fact"] = nullptr;
          }
          if (record.materializedProjection)
            payload["materialized_projection"] =
                materializedProjection(*record.materializedProjection);
          else
            payload["materialized_projection"] = nullptr;
          return llvm::json::Value(std::move(payload));
        });
  }
}

void emitApplicationPreMappingIncompleteDiagnostics(
    const dse::IncompletePreMappingExploration &incomplete) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::DataflowLowering,
      InvocationDiagnosticEvent::Statistics, [&] {
        llvm::json::Object payload;
        payload["domain"] = "pre_mapping_incomplete";
        if (incomplete.planNodeOrdinal)
          payload["plan_node_ordinal"] = *incomplete.planNodeOrdinal;
        else
          payload["plan_node_ordinal"] = nullptr;
        payload["reason"] = dse::toString(incomplete.reason);
        payload["domain_complete"] = incomplete.completeness.domainComplete;
        payload["budget_complete"] = incomplete.completeness.budgetComplete;
        payload["provider_complete"] = incomplete.completeness.providerComplete;
        payload["evidence_complete"] = incomplete.completeness.evidenceComplete;
        payload["selection_complete"] =
            incomplete.completeness.selectionComplete;
        addOptionalUnsigned(payload, "source_host_only_work",
                            incomplete.sourceHostOnlyWork);
        payload["evaluation_timing"] = dse::serializePreMappingEvaluationTiming(
            incomplete.evaluationTiming);
        if (incomplete.checkpoint) {
          const dse::PreMappingCheckpoint &checkpoint = *incomplete.checkpoint;
          payload["checkpoint_boundary"] =
              static_cast<std::uint32_t>(checkpoint.boundary);
          payload["checkpoint_source_program"] =
              encodeRoot(checkpoint.sourceProgram);
          payload["checkpoint_fabric"] = encodeRoot(checkpoint.fabric);
          payload["checkpoint_workload"] = encodeRoot(checkpoint.workload);
          payload["checkpoint_runtime_input"] =
              encodeRoot(checkpoint.runtimeInput);
          payload["checkpoint_frontier_policy_digest"] = llvm::toHex(
              checkpoint.frontierPolicyDigest.bytes(), /*LowerCase=*/true);
          payload["checkpoint_eligible_coordinate_count"] =
              checkpoint.eligibleCoordinateCount;
          payload["checkpoint_coordinate_frontier_truncated"] =
              checkpoint.coordinateFrontierTruncated;
          payload["checkpoint_retained_candidate_count"] =
              checkpoint.retainedCandidates.size();
          addCandidateInventorySummary(payload, checkpoint.candidateInventory);
          payload["checkpoint_coordinate_work"] =
              workCounter(checkpoint.workAccounting.coordinates);
          payload["checkpoint_source_observation_work"] =
              workCounter(checkpoint.workAccounting.sourceObservations);
          payload["checkpoint_program_materializations"] =
              workCounter(checkpoint.workAccounting.programMaterializations);
          payload["checkpoint_analytic_evaluations"] =
              workCounter(checkpoint.workAccounting.analyticEvaluations);
          payload["checkpoint_functional_replays"] =
              workCounter(checkpoint.workAccounting.functionalReplays);
          payload["checkpoint_dataflow_promotions"] =
              workCounter(checkpoint.workAccounting.dataflowPromotions);
          payload["checkpoint_mapping_pairs"] =
              workCounter(checkpoint.workAccounting.mappingPairs);
          payload["checkpoint_funnel"] = dse::serializePreMappingFunnelSummary(
              checkpoint.candidateInventory, {}, checkpoint.workAccounting,
              incomplete.evaluationTiming);
          payload["checkpoint_domain_complete"] =
              checkpoint.completeness.domainComplete;
          payload["checkpoint_budget_complete"] =
              checkpoint.completeness.budgetComplete;
          payload["checkpoint_provider_complete"] =
              checkpoint.completeness.providerComplete;
          payload["checkpoint_evidence_complete"] =
              checkpoint.completeness.evidenceComplete;
          payload["checkpoint_selection_complete"] =
              checkpoint.completeness.selectionComplete;
        } else {
          payload["checkpoint_boundary"] = nullptr;
        }
        return llvm::json::Value(std::move(payload));
      });
}

void emitApplicationResourceTimeFunnelTerminalDiagnostics(
    const dse::ResourceTimeMappingFunnel &funnel, llvm::StringRef status) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "resource_time_application_funnel";
        fields["status"] = status;
        fields["resource_time_funnel"] = resourceTimeFunnelObject(funnel);
      });
}

void emitApplicationMappingDiagnostics(
    const ApplicationMappingExecution &execution) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::SystemPnr,
      InvocationDiagnosticEvent::Statistics, [&] {
        const dse::JointDesignExecutionSummary &summary =
            execution.execution.summary;
        llvm::json::Object payload;
        payload["schema"] = "loom.application_pair_evidence";
        payload["version"] = "1.0";
        payload["domain"] = "application_mapping_join";
        if (execution.provenance.pairDecision)
          payload["pair_decision"] =
              encodePairDecision(*execution.provenance.pairDecision);
        else
          payload["pair_decision"] = nullptr;
        payload["stopping_policy"] =
            dse::jointDesignStoppingPolicySpelling(summary.stoppingPolicy);
        payload["eligible_joint_pair_count"] = summary.eligibleJointPairCount;
        payload["analytic_evaluated_joint_pair_count"] =
            summary.analyticEvaluatedJointPairCount;
        payload["analytic_deferred_joint_pair_count"] =
            summary.analyticDeferredJointPairCount;
        payload["retained_joint_pair_count"] = summary.retainedJointPairCount;
        payload["joint_frontier_truncated"] = summary.jointFrontierTruncated;
        llvm::json::Array analyticPairs;
        for (const dse::JointPairAnalyticObservation &observation :
             summary.retainedJointPairAnalytics) {
          const dse::JointPairAnalyticProjection &projection =
              observation.projection;
          analyticPairs.push_back(llvm::json::Object{
              {"dataflow", encodeRoot(observation.dataflow)},
              {"system", encodeRoot(observation.system)},
              {"software_actor_count", projection.softwareActorCount},
              {"software_graph_count", projection.softwareGraphCount},
              {"software_memory_root_count",
               projection.softwareMemoryRootCount},
              {"system_acc_core_count", projection.systemAccCoreCount},
              {"system_transport_resource_count",
               projection.systemTransportResourceCount},
              {"minimum_execution_waves", projection.minimumExecutionWaves},
              {"estimated_work_units", projection.estimatedWorkUnits},
              {"confidence", dse::jointPairEstimateConfidenceSpelling(
                                 projection.confidence)}});
        }
        payload["retained_joint_pair_analytics"] = std::move(analyticPairs);
        payload["attempted_software_plans"] = summary.attemptedSoftwarePlans;
        payload["hardware_reopen_searches"] = summary.hardwareReopenSearches;
        payload["hardware_parent_promotions"] =
            summary.hardwareParentPromotions;
        payload["hardware_reopens_deferred_by_quality"] =
            summary.hardwareReopensDeferredByQuality;
        payload["hardware_reopens_withheld_without_exact_feedback"] =
            summary.hardwareReopensWithheldWithoutExactFeedback;
        payload["hardware_repair_work"] = llvm::json::Object{
            {"limit", summary.hardwareRepairProbeLimit},
            {"planned", summary.hardwareRepairProbesPlanned},
            {"reserved", summary.hardwareRepairProbesReserved},
            {"consumed", summary.hardwareRepairProbesConsumed},
            {"rejected", summary.hardwareRepairProbesRejected},
            {"cancelled", summary.hardwareRepairProbesCancelled}};
        payload["spatial_mapping_repair_work"] = llvm::json::Object{
            {"limit", summary.spatialMappingRepairCandidateLimit},
            {"planned", summary.spatialMappingRepairsPlanned},
            {"reserved", summary.spatialMappingRepairsReserved},
            {"consumed", summary.spatialMappingRepairsConsumed},
            {"rejected", summary.spatialMappingRepairsRejected},
            {"cancelled", summary.spatialMappingRepairsCancelled}};
        payload["tech_mapping_dispatch_count"] =
            summary.techMappingDispatchCount;
        payload["spatial_pnr_dispatch_count"] = summary.spatialPnrDispatchCount;
        payload["system_pnr_dispatch_count"] = summary.systemPnrDispatchCount;
        payload["tech_mapping_invocation_count"] =
            summary.techMappingInvocationCount;
        payload["spatial_pnr_invocation_count"] =
            summary.spatialPnrInvocationCount;
        payload["system_pnr_invocation_count"] =
            summary.systemPnrInvocationCount;
        payload["tech_mapping_journal_replay_count"] =
            summary.techMappingJournalReplayCount;
        payload["spatial_pnr_journal_replay_count"] =
            summary.spatialPnrJournalReplayCount;
        payload["system_pnr_journal_replay_count"] =
            summary.systemPnrJournalReplayCount;
        if (execution.provenance.resourceTimeFunnelAccounting) {
          const auto &resourceTime =
              *execution.provenance.resourceTimeFunnelAccounting;
          payload["resource_time_generated_candidates"] =
              resourceTime.generatedCandidates;
          payload["resource_time_screened_candidates"] =
              resourceTime.screenedCandidates;
          payload["resource_time_detailed_frontier_candidates"] =
              resourceTime.detailedFrontierCandidates;
          payload["resource_time_successive_halving_deferred_candidates"] =
              resourceTime.successiveHalvingDeferredCandidates;
          payload["resource_time_mapping_finalists"] =
              resourceTime.mappingFinalists;
          payload["resource_time_mapping_eligible_schedule_hints"] =
              resourceTime.mappingEligibleScheduleHints;
          payload["resource_time_screening_comparison_candidates"] =
              resourceTime.screeningComparisonCandidates;
          payload["resource_time_detailed_schedule_feasible_candidates"] =
              resourceTime.detailedScheduleFeasibleCandidates;
          payload["resource_time_screening_admissible_candidates"] =
              resourceTime.screeningAdmissibleCandidates;
          payload["resource_time_screening_detailed_feasible_intersection"] =
              resourceTime.screeningDetailedFeasibleIntersection;
          payload["resource_time_screening_detailed_best_rank_matches"] =
              resourceTime.screeningDetailedBestRankMatches;
          payload["resource_time_screening_out_of_domain_candidates"] =
              resourceTime.screeningOutOfDomainCandidates;
          payload
              ["resource_time_maximum_screening_lower_bound_gap_picoseconds"] =
                  resourceTime.maximumScreeningLowerBoundGapPicoseconds;
          payload["resource_time_functional_replay_candidates"] =
              resourceTime.functionalReplayCandidates;
          payload["resource_time_dataflow_projection_requests"] =
              resourceTime.dataflowProjectionRequests;
          payload["resource_time_dataflow_projection_cache_hits"] =
              resourceTime.dataflowProjectionCacheHits;
          payload["resource_time_dataflow_projection_cache_misses"] =
              resourceTime.dataflowProjectionCacheMisses;
          payload["resource_time_dataflow_projection_cache_capacity_bypasses"] =
              resourceTime.dataflowProjectionCacheCapacityBypasses;
          payload["resource_time_dataflow_projection_cache_entries"] =
              resourceTime.dataflowProjectionCacheEntries;
          payload["resource_time_dataflow_projection_cache_retained_bytes"] =
              resourceTime.dataflowProjectionCacheRetainedBytes;
          payload["resource_time_dataflow_projection_elapsed_nanoseconds"] =
              resourceTime.dataflowProjectionElapsedNanoseconds;
          payload["resource_time_dataflow_materialized_candidates"] =
              resourceTime.dataflowMaterializedCandidates;
          payload["resource_time_mapping_plan_candidates"] =
              resourceTime.mappingPlanCandidates;
          payload["resource_time_mapping_plan_constructions_avoided_by_exact_"
                  "memo"] =
              resourceTime.mappingPlanConstructionsAvoidedByExactMemo;
          payload["resource_time_unsupported_before_mapping_schedule_hints"] =
              resourceTime.unsupportedBeforeMappingScheduleHints;
          payload["resource_time_application_promotion_accounting_complete"] =
              resourceTime.applicationPromotionAccountingComplete;
          payload["resource_time_mapping_calls_deferred_by_model"] =
              resourceTime.mappingCallsDeferredByModel;
          payload["resource_time_mapping_calls_withheld_by_incomplete"] =
              resourceTime.mappingCallsWithheldByIncomplete;
          payload["resource_time_exact_invocation_memo_hits"] =
              resourceTime.exactInvocationMemoHits;
          payload["resource_time_exact_invocation_memo_misses"] =
              resourceTime.exactInvocationMemoMisses;
          payload["resource_time_exact_invocation_memo_single_flight_waits"] =
              resourceTime.exactInvocationMemoSingleFlightWaits;
          payload["resource_time_exact_invocation_memo_coalesced_uncached_"
                  "results"] =
              resourceTime.exactInvocationMemoCoalescedUncachedResults;
          payload["resource_time_exact_invocation_memo_cancelled_waits"] =
              resourceTime.exactInvocationMemoCancelledWaits;
          payload["resource_time_exact_invocation_memo_capacity_bypasses"] =
              resourceTime.exactInvocationMemoCapacityBypasses;
          payload["resource_time_exact_invocation_memo_entries"] =
              resourceTime.exactInvocationMemoEntries;
          payload["resource_time_exact_invocation_memo_retained_bytes"] =
              resourceTime.exactInvocationMemoRetainedBytes;
          payload["resource_time_frontier_work"] = llvm::json::Object{
              {"source_projections",
               workCounter(resourceTime.frontierAccounting.sourceProjections)},
              {"actions", workCounter(resourceTime.frontierAccounting.actions)},
              {"states", workCounter(resourceTime.frontierAccounting.states)},
              {"estimates",
               workCounter(resourceTime.frontierAccounting.estimates)},
              {"finalists",
               workCounter(resourceTime.frontierAccounting.finalists)},
              {"state_memo_hits",
               resourceTime.frontierAccounting.stateMemoHits},
              {"state_memo_misses",
               resourceTime.frontierAccounting.stateMemoMisses},
              {"state_memo_pareto_insertions",
               resourceTime.frontierAccounting.stateMemoParetoInsertions},
              {"state_memo_dominated_states",
               resourceTime.frontierAccounting.stateMemoDominatedStates},
              {"state_memo_hit_capacity_rejections",
               resourceTime.frontierAccounting.stateMemoHitCapacityRejections},
              {"state_memo_miss_capacity_rejections",
               resourceTime.frontierAccounting.stateMemoMissCapacityRejections},
              {"states_pruned_by_beam",
               resourceTime.frontierAccounting.statesPrunedByBeam},
              {"terminal_hints_generated",
               resourceTime.frontierAccounting.terminalHintsGenerated},
              {"terminal_hints_retained",
               resourceTime.frontierAccounting.terminalHintsRetained},
              {"terminal_hints_pruned",
               resourceTime.frontierAccounting.terminalHintsPruned},
              {"incremental_lower_bound_updates",
               resourceTime.frontierAccounting.incrementalLowerBoundUpdates},
              {"maximum_retained_bytes",
               resourceTime.frontierAccounting.maximumRetainedBytes}};
          payload["resource_time_funnel_truncated"] =
              execution.provenance.resourceTimeFunnelTruncated;
          payload["resource_time_actual_system_pnr_dispatch_count"] =
              summary.systemPnrDispatchCount;
          if (execution.provenance.resourceTimeFunnelIncompleteReason)
            payload["resource_time_funnel_incomplete_reason"] =
                dse::resourceTimeFrontierIncompleteReasonSpelling(
                    *execution.provenance.resourceTimeFunnelIncompleteReason);
          else
            payload["resource_time_funnel_incomplete_reason"] = nullptr;
        }
        payload["cold_reopen_wall_time_ns"] =
            summary.coldReopenWallTimeNanoseconds;
        payload["incremental_reopen_wall_time_ns"] =
            summary.incrementalReopenWallTimeNanoseconds;
        addOptionalUnsigned(payload, "time_to_first_feasible_wall_time_ns",
            summary.timeToFirstFeasibleWallTimeNanoseconds);
        addOptionalUnsigned(payload, "time_to_best_wall_time_ns",
                            summary.timeToBestWallTimeNanoseconds);
        payload["preserved_tech_mappings"] = summary.preservedTechMappings;
        payload["preserved_spatial_mappings"] =
            summary.preservedSpatialMappings;
        payload["repaired_tech_mappings"] = summary.repairedTechMappings;
        payload["repaired_spatial_mappings"] = summary.repairedSpatialMappings;
        payload["invalidated_tech_mappings"] = summary.invalidatedTechMappings;
        payload["invalidated_spatial_mappings"] =
            summary.invalidatedSpatialMappings;
        payload["mapping_rebase_work"] = llvm::json::Object{
            {"parent_tech_decisions", summary.parentTechDecisions},
            {"parent_spatial_decisions", summary.parentSpatialDecisions},
            {"preserved_tech_decisions", summary.preservedTechDecisions},
            {"preserved_spatial_decisions", summary.preservedSpatialDecisions},
            {"reopened_tech_decisions", summary.reopenedTechDecisions},
            {"reopened_spatial_decisions", summary.reopenedSpatialDecisions},
            {"repaired_tech_decisions", summary.repairedTechDecisions},
            {"repaired_spatial_decisions", summary.repairedSpatialDecisions},
            {"invalidation_root_count", summary.invalidationRootCount},
            {"invalidation_cone_decision_count",
             summary.invalidationConeDecisionCount},
            {"parent_route_node_count", summary.parentRouteNodeCount},
            {"preserved_route_node_count", summary.preservedRouteNodeCount},
            {"reopened_route_node_count", summary.reopenedRouteNodeCount},
            {"repaired_route_node_count", summary.repairedRouteNodeCount},
            {"parent_service_leg_count", summary.parentServiceLegCount},
            {"preserved_service_leg_count", summary.preservedServiceLegCount},
            {"reopened_service_leg_count", summary.reopenedServiceLegCount}};
        payload["system_mapping_rebase_work"] = llvm::json::Object{
            {"parent_thread_binding_count", summary.parentThreadBindingCount},
            {"preserved_thread_binding_count",
             summary.preservedThreadBindingCount},
            {"reopened_thread_binding_count",
             summary.reopenedThreadBindingCount},
            {"parent_graph_binding_count", summary.parentGraphBindingCount},
            {"preserved_graph_binding_count",
             summary.preservedGraphBindingCount},
            {"reopened_graph_binding_count", summary.reopenedGraphBindingCount},
            {"parent_resource_use_count", summary.parentResourceUseCount},
            {"preserved_resource_use_count", summary.preservedResourceUseCount},
            {"reopened_resource_use_count", summary.reopenedResourceUseCount},
            {"parent_service_realization_count",
             summary.parentServiceRealizationCount},
            {"preserved_service_realization_count",
             summary.preservedServiceRealizationCount},
            {"reopened_service_realization_count",
             summary.reopenedServiceRealizationCount}};
        llvm::json::Array incrementalTransitions;
        for (const auto indexed : llvm::enumerate(
                 execution.provenance.incrementalMappingObservations)) {
          const ApplicationIncrementalMappingObservation &observation =
              indexed.value();
          llvm::json::Object transition;
          transition["observation_ordinal"] = indexed.index();
          transition["parent_mapping"] = encodeRoot(observation.parentMapping);
          transition["child_system"] = encodeRoot(observation.childSystem);
          if (observation.childMapping)
            transition["child_mapping"] = encodeRoot(*observation.childMapping);
          else
            transition["child_mapping"] = nullptr;
          if (observation.coldMapping)
            transition["cold_mapping"] = encodeRoot(*observation.coldMapping);
          else
            transition["cold_mapping"] = nullptr;
          transition["parent_plan_ordinal"] = observation.parentPlanOrdinal;
          transition["child_plan_ordinal"] = observation.childPlanOrdinal;
          transition["parent_schedule_hint_digest"] =
              formatComponentViewDigestHex(
                  observation.parentScheduleHintDigest);
          transition["child_schedule_hint_digest"] =
              formatComponentViewDigestHex(observation.childScheduleHintDigest);
          transition["mapping_reuse_disposition"] =
              dse::jointMappingReuseDispositionSpelling(
                  observation.reuseDisposition);
          transition["reopened_root_count"] = observation.reopenedRoots.size();
          transition["preserved_tech_mappings"] =
              observation.preservedTechMappings;
          transition["preserved_spatial_mappings"] =
              observation.preservedSpatialMappings;
          transition["repaired_tech_mappings"] =
              observation.repairedTechMappings;
          transition["repaired_spatial_mappings"] =
              observation.repairedSpatialMappings;
          transition["preserved_system_bindings"] =
              observation.preservedSystemBindings;
          transition["reopened_system_bindings"] =
              observation.reopenedSystemBindings;
          transition["disposition"] = spelling(observation.disposition);
          if (observation.incompleteReason)
            transition["incomplete_reason"] =
                dse::toString(*observation.incompleteReason);
          else
            transition["incomplete_reason"] = nullptr;
          transition["cold_wall_time_ns"] = observation.coldWallTimeNanoseconds;
          transition["incremental_wall_time_ns"] =
              observation.incrementalWallTimeNanoseconds;
          transition["wall_time_ns"] = observation.wallTimeNanoseconds;
          transition["cold_verifier_retained_bytes"] =
              observation.coldVerifierRetainedBytes;
          transition["incremental_verifier_retained_bytes"] =
              observation.incrementalVerifierRetainedBytes;
          transition["cold_verifier_work"] = observation.coldVerifierWork;
          transition["incremental_verifier_work"] =
              observation.incrementalVerifierWork;
          transition["cold_provider_work"] =
              mappingProviderWork(observation.coldProviderWork);
          transition["incremental_provider_work"] =
              mappingProviderWork(observation.incrementalProviderWork);
          transition["cold_dfg_cycles"] =
              observation.coldDfgCycles
                  ? llvm::json::Value(*observation.coldDfgCycles)
                  : llvm::json::Value(nullptr);
          transition["cold_cgra_cycles"] =
              observation.coldCgraCycles
                  ? llvm::json::Value(*observation.coldCgraCycles)
                  : llvm::json::Value(nullptr);
          transition["incremental_dfg_cycles"] =
              observation.incrementalDfgCycles
                  ? llvm::json::Value(*observation.incrementalDfgCycles)
                  : llvm::json::Value(nullptr);
          transition["incremental_cgra_cycles"] =
              observation.incrementalCgraCycles
                  ? llvm::json::Value(*observation.incrementalCgraCycles)
                  : llvm::json::Value(nullptr);
          transition["verified"] = observation.verified;
          incrementalTransitions.push_back(std::move(transition));
        }
        payload["application_incremental_mapping_transitions"] =
            std::move(incrementalTransitions);
        if (execution.provenance.resourceTimeMappingPath) {
          const ApplicationResourceTimeMappingPath &path =
              *execution.provenance.resourceTimeMappingPath;
          llvm::json::Array observationOrdinals;
          for (const std::uint64_t ordinal : path.observationOrdinals)
            observationOrdinals.push_back(ordinal);
          payload["application_resource_time_mapping_path"] =
              llvm::json::Object{
                  {"schedule_owner_plan_ordinal",
                   path.scheduleOwnerPlanOrdinal},
                  {"schedule_hint_digest",
                   formatComponentViewDigestHex(path.scheduleHintDigest)},
                  {"observation_ordinals", std::move(observationOrdinals)}};
        } else {
          payload["application_resource_time_mapping_path"] = nullptr;
        }
        payload["verified_alternatives"] = summary.verifiedAlternatives;
        payload["quality_disposition"] = spelling(summary.qualityDisposition);
        payload["declared_work_exhausted"] = summary.declaredWorkExhausted;
        payload["quality_objective_dimension_count"] =
            summary.qualityObjectiveDimensionLabels.size();
        llvm::json::Array qualityObjectiveLabels;
        for (const std::string &label : summary.qualityObjectiveDimensionLabels)
          qualityObjectiveLabels.push_back(label);
        payload["quality_objective_dimension_labels"] =
            std::move(qualityObjectiveLabels);
        addOptionalRoot(payload, "source_program",
                        execution.provenance.sourceProgram);
        addOptionalRoot(payload, "fabric", execution.provenance.fabric);
        addOptionalRoot(payload, "workload", execution.provenance.workload);
        addOptionalRoot(payload, "runtime_input",
                        execution.provenance.runtimeInput);
        if (execution.provenance.frontierPolicyDigest)
          payload["frontier_policy_digest"] =
              llvm::toHex(execution.provenance.frontierPolicyDigest->bytes(),
                          /*LowerCase=*/true);
        else
          payload["frontier_policy_digest"] = nullptr;
        payload["requested_planner_mode"] =
            dse::toString(execution.provenance.requestedPlannerMode);
        payload["resolved_planner_mode"] =
            dse::toString(execution.provenance.resolvedPlannerMode);
        payload["domain_complete"] =
            execution.provenance.preMappingCompleteness.domainComplete;
        payload["budget_complete"] =
            execution.provenance.preMappingCompleteness.budgetComplete;
        payload["provider_complete"] =
            execution.provenance.preMappingCompleteness.providerComplete;
        payload["evidence_complete"] =
            execution.provenance.preMappingCompleteness.evidenceComplete;
        payload["selection_complete"] =
            execution.provenance.preMappingCompleteness.selectionComplete;
        addOptionalUnsigned(payload, "selected_plan_ordinal",
                            summary.selectedPlanOrdinal);
        addOptionalRoot(payload, "selected_mapping", summary.selectedMapping);
        payload["outcome_count"] = execution.candidateOutcomes.size();
        std::uint64_t joinedIdentityCount = 0;
        std::uint64_t mappingTemporalCount = 0;
        std::uint64_t mappingSpatialCount = 0;
        std::uint64_t mappingIntermediateCount = 0;
        std::uint64_t runtimeTemporalCount = 0;
        std::uint64_t runtimeSpatialCount = 0;
        std::uint64_t runtimeIntermediateCount = 0;
        for (const ApplicationMappingCandidateOutcome &outcome :
             execution.candidateOutcomes) {
          if (outcome.planningRecord &&
              outcome.planningRecord->candidateIdentity)
            ++joinedIdentityCount;
          if (!outcome.resourceTimeSpectrum)
            continue;
          if (const auto *verified =
                  std::get_if<dse::VerifiedResourceTimeSpectrum>(
                      &outcome.resourceTimeSpectrum->verification)) {
            for (const auto &scenario : verified->scenarios)
              if (scenario.spectrumClass ==
                  dse::PreMappingSpectrumClass::MaxTemporal)
                ++mappingTemporalCount;
              else if (scenario.spectrumClass ==
                       dse::PreMappingSpectrumClass::MaxSpatial)
                ++mappingSpatialCount;
              else
                ++mappingIntermediateCount;
            if (outcome.runtimeDisposition ==
                ApplicationMappingRuntimeDisposition::Completed) {
              for (const auto &scenario : verified->scenarios)
                if (scenario.spectrumClass ==
                    dse::PreMappingSpectrumClass::MaxTemporal)
                  ++runtimeTemporalCount;
                else if (scenario.spectrumClass ==
                         dse::PreMappingSpectrumClass::MaxSpatial)
                  ++runtimeSpatialCount;
                else
                  ++runtimeIntermediateCount;
            }
          }
        }
        payload["joined_candidate_identity_count"] = joinedIdentityCount;
        payload["joined_max_temporal_outcome_count"] = runtimeTemporalCount;
        payload["joined_max_spatial_outcome_count"] = runtimeSpatialCount;
        payload["joined_intermediate_outcome_count"] = runtimeIntermediateCount;
        payload["verified_mapping_max_temporal_count"] = mappingTemporalCount;
        payload["verified_mapping_max_spatial_count"] = mappingSpatialCount;
        payload["verified_mapping_intermediate_count"] =
            mappingIntermediateCount;
        llvm::json::Array qualityObservations;
        for (const dse::JointDesignQualityObservation &observation :
             summary.qualityObservations) {
          llvm::json::Object entry;
          entry["candidate"] = encodeRoot(observation.candidate);
          llvm::json::Array objective;
          for (std::uint64_t code : observation.objectiveCodes)
            objective.push_back(code);
          entry["objective_codes"] = std::move(objective);
          entry["provenance"] =
              encodeQualityProvenance(observation.provenance);
          if (observation.incompleteReason)
            entry["incomplete_reason"] =
                spelling(*observation.incompleteReason);
          else
            entry["incomplete_reason"] = nullptr;
          if (observation.evidence)
            entry["evidence"] = encodeRoot(*observation.evidence);
          else
            entry["evidence"] = nullptr;
          qualityObservations.push_back(std::move(entry));
        }
        payload["quality_observations"] = std::move(qualityObservations);
        llvm::json::Array hardwarePromotionObservations;
        for (const dse::JointHardwarePromotionObservation &observation :
             summary.hardwarePromotionObservations) {
          llvm::json::Object entry;
          entry["plan_ordinal"] = observation.planOrdinal;
          entry["system"] = encodeRoot(observation.system);
          llvm::json::Array objective;
          for (std::uint64_t code : observation.objectiveCodes)
            objective.push_back(code);
          entry["objective_codes"] = std::move(objective);
          entry["provenance"] =
              encodeQualityProvenance(observation.provenance);
          if (observation.incompleteReason)
            entry["incomplete_reason"] =
                spelling(*observation.incompleteReason);
          else
            entry["incomplete_reason"] = nullptr;
          if (observation.evidence)
            entry["evidence"] = encodeRoot(*observation.evidence);
          else
            entry["evidence"] = nullptr;
          entry["promoted_to_exact_mapping"] =
              observation.promotedToExactMapping;
          hardwarePromotionObservations.push_back(std::move(entry));
        }
        payload["hardware_promotion_observations"] =
            std::move(hardwarePromotionObservations);
        std::uint64_t completeQualityObservations = 0;
        std::uint64_t incompleteQualityObservations = 0;
        for (const dse::JointDesignQualityObservation &observation :
             summary.qualityObservations) {
          if (observation.incompleteReason)
            ++incompleteQualityObservations;
          else
            ++completeQualityObservations;
        }
        payload["quality_complete_observation_count"] =
            completeQualityObservations;
        payload["quality_incomplete_observation_count"] =
            incompleteQualityObservations;
        return llvm::json::Value(std::move(payload));
      });

  for (const ApplicationMappingCandidateOutcome &outcome :
       execution.candidateOutcomes) {
    emitInvocationDiagnostic(
        DiagnosticVerbosity::Summary, InvocationDiagnosticStage::SystemPnr,
        InvocationDiagnosticEvent::Candidate, [&] {
          llvm::json::Object payload;
          payload["domain"] = "application_mapping_join";
          payload["planning_record_ordinal"] =
              outcome.preMappingCandidateRecordOrdinal;
          if (outcome.planningRecord &&
              outcome.planningRecord->candidateIdentity)
            payload["candidate_identity"] = formatComponentViewDigestHex(
                *outcome.planningRecord->candidateIdentity);
          else
            payload["candidate_identity"] = nullptr;
          payload["plan_ordinal"] = outcome.planOrdinal;
          payload["resource_time_schedule_hint_digest"] =
              formatComponentViewDigestHex(
                  outcome.resourceTimeScheduleHintDigest);
          payload["dataflow"] = encodeRoot(outcome.dataflow);
          payload["system"] = encodeRoot(outcome.system);
          payload["disposition"] = spelling(outcome.disposition);
          payload["runtime_disposition"] = spelling(outcome.runtimeDisposition);
          addOptionalUnsigned(payload, "dfg_cycles", outcome.dfgCycles);
          addOptionalUnsigned(payload, "cgra_cycles", outcome.cgraCycles);
          addOptionalUnsigned(payload, "resource_core_cost",
                              outcome.resourceCoreCost);
          llvm::json::Array qualityObjective;
          for (std::uint64_t code : outcome.qualityObjectiveCodes)
            qualityObjective.push_back(code);
          payload["quality_objective_codes"] = std::move(qualityObjective);
          llvm::json::Array partitions;
          for (const pnr::SystemBindingPartitionIntent &partition :
               outcome.systemBindingPartitions) {
            llvm::json::Object encoded;
            encoded["dataflow"] =
                llvm::toHex(partition.root.artifact.bytes(), true);
            encoded["root"] = partition.root.entity.value();
            encoded["partition_count"] = partition.partitionCount;
            partitions.push_back(std::move(encoded));
          }
          payload["system_binding_partitions"] = std::move(partitions);
          if (outcome.resourceTimeSpectrum) {
            const auto &accounting = outcome.resourceTimeSpectrum->accounting;
            payload["resource_time_hint_candidates"] =
                accounting.hintCandidates;
            payload["resource_time_matching_mapping_checks"] =
                accounting.matchingMappingChecks;
            payload["resource_time_materialized_scenarios"] =
                accounting.materializedScenarios;
            payload["resource_time_unmatched_hints"] =
                accounting.unmatchedHints;
            payload["resource_time_transition_unsupported_hints"] =
                accounting.transitionUnsupportedHints;
            payload["resource_time_transition_proof_failures"] =
                accounting.transitionProofFailures;
            payload["resource_time_verified_scenarios"] =
                accounting.verifiedScenarios;
            payload["resource_time_independent_mapping_imports"] =
                accounting.independentlyImportedMappings;
            payload["resource_time_mapping_import_requests"] =
                accounting.mappingImportRequests;
            payload["resource_time_mapping_import_cache_hits"] =
                accounting.mappingImportCacheHits;
            payload["resource_time_mapping_import_cache_misses"] =
                accounting.mappingImportCacheMisses;
            payload["resource_time_mapping_import_retained_bytes"] =
                accounting.mappingImportRetainedBytes;
            payload["resource_time_mapping_progress_qualified"] =
                accounting.mappingProgressQualified;
            payload["resource_time_mapping_progress_proof_not_established"] =
                accounting.mappingProgressProofNotEstablished;
            payload["resource_time_verification_elapsed_nanoseconds"] =
                accounting.elapsedNanoseconds;
            if (const auto *incomplete =
                    std::get_if<dse::IncompleteResourceTimeSpectrum>(
                        &outcome.resourceTimeSpectrum->verification)) {
              payload["resource_time_verification_incomplete_reason"] =
                  dse::resourceTimeSpectrumIncompleteReasonSpelling(
                      incomplete->reason);
              payload["resource_time_verification_diagnostic"] =
                  incomplete->diagnostic;
            } else {
              payload["resource_time_verification_incomplete_reason"] = nullptr;
              payload["resource_time_verification_diagnostic"] = nullptr;
            }
          }
          addOptionalUnsigned(payload, "incomplete_node_ordinal",
                              outcome.incompleteNodeOrdinal);
          if (outcome.incompleteReason)
            payload["incomplete_reason"] =
                dse::toString(*outcome.incompleteReason);
          else
            payload["incomplete_reason"] = nullptr;
          if (outcome.planningRecord) {
            const dse::PreMappingCandidatePlanningRecord &record =
                *outcome.planningRecord;
            addOptionalRoot(payload, "candidate_structured_program",
                            record.structuredProgram);
            addOptionalRoot(payload, "candidate_canonical_dataflow",
                            record.canonicalDataflow);
            llvm::json::Array roots;
            for (const frontend::StructuredEntityRef &root :
                 record.ownedProtocolRoots)
              roots.push_back(encodeStructuredRoot(root));
            payload["candidate_owned_protocol_roots"] = std::move(roots);
            llvm::json::Array seeds;
            for (dse::PreMappingSpectrumSeedKind kind : record.seedKinds)
              seeds.push_back(dse::toString(kind));
            payload["candidate_seed_kinds"] = std::move(seeds);
            if (record.scheduleIntent)
              payload["candidate_schedule_intent"] =
                  dse::toString(*record.scheduleIntent);
            else
              payload["candidate_schedule_intent"] = nullptr;
            if (record.projection)
              payload["candidate_projection"] =
                  candidateProjection(*record.projection);
            else
              payload["candidate_projection"] = nullptr;
            addOptionalUnsigned(payload, "candidate_estimated_runtime_ps",
                                record.estimatedRuntimePicoseconds);
            if (record.materializedProjection)
              payload["candidate_materialized_projection"] =
                  materializedProjection(*record.materializedProjection);
            else
              payload["candidate_materialized_projection"] = nullptr;
            if (record.temporalWitness)
              payload["candidate_logical_domain_fact"] = llvm::json::Object{
                  {"logical_epoch_count",
                   record.temporalWitness->logicalEpochCount},
                  {"acc_core_occupancy",
                   record.temporalWitness->accCoreOccupancy},
                  {"launch_count", record.temporalWitness->launchCount},
                  {"synchronization_count",
                   record.temporalWitness->synchronizationCount},
                  {"live_state_bytes", record.temporalWitness->liveStateBytes},
                  {"live_state_known", record.temporalWitness->liveStateKnown},
                  {"exact", record.temporalWitness->exact}};
            else
              payload["candidate_logical_domain_fact"] = nullptr;
            if (record.verifiedSpectrum)
              payload["candidate_verified_spectrum"] =
                  dse::toString(*record.verifiedSpectrum);
            else
              payload["candidate_verified_spectrum"] = nullptr;
          } else {
            payload["candidate_structured_program"] = nullptr;
            payload["candidate_canonical_dataflow"] = nullptr;
            payload["candidate_projection"] = nullptr;
            payload["candidate_materialized_projection"] = nullptr;
            payload["candidate_logical_domain_fact"] = nullptr;
            payload["candidate_verified_spectrum"] = nullptr;
          }
          llvm::json::Array mappings;
          for (const ArtifactRootReference &mapping : outcome.systemMappings)
            mappings.push_back(encodeRoot(mapping));
          payload["system_mappings"] = std::move(mappings);
          llvm::json::Array runtimeEvidence;
          for (const ArtifactRootReference &evidence : outcome.runtimeEvidence)
            runtimeEvidence.push_back(encodeRoot(evidence));
          payload["runtime_evidence"] = std::move(runtimeEvidence);
          return llvm::json::Value(std::move(payload));
        });
  }
}

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

} // namespace loom::application

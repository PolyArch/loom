#include "Application/BuildDiagnostics.h"

#include "Application/Build.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/InvocationDiagnosticLog.h"
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
        struct rusage usage {
        };
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
        payload["mapping_alternative_count"] =
            prepared.mappingAlternatives.size();
        const auto &resourceTime = prepared.resourceTimeFunnel.accounting;
        payload["resource_time_funnel"] = llvm::json::Object{
            {"generated_candidates", resourceTime.generatedCandidates},
            {"screened_candidates", resourceTime.screenedCandidates},
            {"detailed_frontier_candidates",
             resourceTime.detailedFrontierCandidates},
            {"successive_halving_deferred_candidates",
             resourceTime.successiveHalvingDeferredCandidates},
            {"sound_gate_rejected_candidates",
             resourceTime.soundGateRejectedCandidates},
            {"estimated_candidates", resourceTime.estimatedCandidates},
            {"incomplete_candidates", resourceTime.incompleteCandidates},
            {"mapping_eligible_schedule_hints",
             resourceTime.mappingEligibleScheduleHints},
            {"analytic_shadow_compared_candidates",
             resourceTime.analyticShadowComparedCandidates},
            {"analytic_shadow_exact_feasible_candidates",
             resourceTime.analyticShadowExactFeasibleCandidates},
            {"analytic_shadow_admissible_candidates",
             resourceTime.analyticShadowAdmissibleCandidates},
            {"analytic_shadow_feasible_intersection",
             resourceTime.analyticShadowFeasibleIntersection},
            {"analytic_shadow_best_rank_matches",
             resourceTime.analyticShadowBestRankMatches},
            {"analytic_shadow_out_of_domain_candidates",
             resourceTime.analyticShadowOutOfDomainCandidates},
            {"analytic_shadow_maximum_lower_bound_gap_picoseconds",
             resourceTime.analyticShadowMaximumLowerBoundGapPicoseconds},
            {"mapping_finalists", resourceTime.mappingFinalists},
            {"functional_replay_candidates",
             resourceTime.functionalReplayCandidates},
            {"dataflow_projection_requests",
             resourceTime.dataflowProjectionRequests},
            {"dataflow_projection_cache_hits",
             resourceTime.dataflowProjectionCacheHits},
            {"dataflow_projection_cache_misses",
             resourceTime.dataflowProjectionCacheMisses},
            {"dataflow_projection_cache_capacity_bypasses",
             resourceTime.dataflowProjectionCacheCapacityBypasses},
            {"dataflow_projection_cache_entries",
             resourceTime.dataflowProjectionCacheEntries},
            {"dataflow_projection_cache_retained_bytes",
             resourceTime.dataflowProjectionCacheRetainedBytes},
            {"dataflow_projection_elapsed_nanoseconds",
             resourceTime.dataflowProjectionElapsedNanoseconds},
            {"dataflow_materialized_candidates",
             resourceTime.dataflowMaterializedCandidates},
            {"mapping_plan_candidates", resourceTime.mappingPlanCandidates},
            {"unsupported_before_mapping_candidates",
             resourceTime.unsupportedBeforeMappingCandidates},
            {"unsupported_before_mapping_schedule_hints",
             resourceTime.unsupportedBeforeMappingScheduleHints},
            {"application_promotion_accounting_complete",
             resourceTime.applicationPromotionAccountingComplete},
            {"mapping_calls_avoided_by_sound_gate",
             resourceTime.mappingCallsAvoidedBySoundGate},
            {"mapping_calls_deferred_by_model",
             resourceTime.mappingCallsDeferredByModel},
            {"mapping_calls_withheld_by_incomplete",
             resourceTime.mappingCallsWithheldByIncomplete},
            {"exact_invocation_memo_hits",
             resourceTime.exactInvocationMemoHits},
            {"exact_invocation_memo_misses",
             resourceTime.exactInvocationMemoMisses},
            {"exact_invocation_memo_single_flight_waits",
             resourceTime.exactInvocationMemoSingleFlightWaits},
            {"exact_invocation_memo_coalesced_uncached_results",
             resourceTime.exactInvocationMemoCoalescedUncachedResults},
            {"exact_invocation_memo_cancelled_waits",
             resourceTime.exactInvocationMemoCancelledWaits},
            {"exact_invocation_memo_capacity_bypasses",
             resourceTime.exactInvocationMemoCapacityBypasses},
            {"exact_invocation_memo_entries",
             resourceTime.exactInvocationMemoEntries},
            {"exact_invocation_memo_retained_bytes",
             resourceTime.exactInvocationMemoRetainedBytes},
            {"frontier_work", llvm::json::Object{
                                   {"source_projections",
                                    workCounter(resourceTime.frontierAccounting
                                                    .sourceProjections)},
                                   {"actions", workCounter(
                                                   resourceTime.frontierAccounting
                                                       .actions)},
                                   {"states", workCounter(
                                                   resourceTime.frontierAccounting
                                                       .states)},
                                   {"estimates", workCounter(
                                                     resourceTime
                                                         .frontierAccounting
                                                         .estimates)},
                                   {"finalists", workCounter(
                                                     resourceTime
                                                         .frontierAccounting
                                                         .finalists)},
                                   {"state_memo_hits",
                                    resourceTime.frontierAccounting
                                        .stateMemoHits},
                                   {"state_memo_misses",
                                    resourceTime.frontierAccounting
                                        .stateMemoMisses},
                                   {"state_memo_pareto_insertions",
                                    resourceTime.frontierAccounting
                                        .stateMemoParetoInsertions},
                                   {"state_memo_dominated_states",
                                    resourceTime.frontierAccounting
                                        .stateMemoDominatedStates},
                                   {"state_memo_hit_capacity_rejections",
                                    resourceTime.frontierAccounting
                                        .stateMemoHitCapacityRejections},
                                   {"state_memo_miss_capacity_rejections",
                                    resourceTime.frontierAccounting
                                        .stateMemoMissCapacityRejections},
                                   {"states_pruned_by_beam",
                                    resourceTime.frontierAccounting
                                        .statesPrunedByBeam},
                                   {"terminal_hints_generated",
                                    resourceTime.frontierAccounting
                                        .terminalHintsGenerated},
                                   {"terminal_hints_retained",
                                    resourceTime.frontierAccounting
                                        .terminalHintsRetained},
                                   {"terminal_hints_pruned",
                                    resourceTime.frontierAccounting
                                        .terminalHintsPruned},
                                   {"incremental_lower_bound_updates",
                                    resourceTime.frontierAccounting
                                        .incrementalLowerBoundUpdates},
                                   {"maximum_retained_bytes",
                                    resourceTime.frontierAccounting
                                        .maximumRetainedBytes}}},
            {"elapsed_nanoseconds", resourceTime.elapsedNanoseconds},
            {"truncated", prepared.resourceTimeFunnel.truncated}};
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
          row["screening_support"] =
              dse::resourceTimeEstimateSupportSpelling(
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
                return finalist.candidateIdentity == evaluation.candidateIdentity;
              });
          row["retained_mapping_schedule_count"] = retainedScheduleCount;
          row["retained_for_mapping"] = retainedScheduleCount != 0;
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

void emitApplicationMappingDiagnostics(
    const ApplicationMappingExecution &execution) {
  emitInvocationDiagnostic(
      DiagnosticVerbosity::Summary, InvocationDiagnosticStage::SystemPnr,
      InvocationDiagnosticEvent::Statistics, [&] {
        const dse::JointDesignExecutionSummary &summary =
            execution.execution.summary;
        llvm::json::Object payload;
        payload["domain"] = "application_mapping_join";
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
        payload["tech_mapping_dispatch_count"] =
            summary.techMappingDispatchCount;
        payload["spatial_pnr_dispatch_count"] = summary.spatialPnrDispatchCount;
        payload["system_pnr_dispatch_count"] = summary.systemPnrDispatchCount;
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
          payload["resource_time_analytic_shadow_compared_candidates"] =
              resourceTime.analyticShadowComparedCandidates;
          payload["resource_time_analytic_shadow_exact_feasible_candidates"] =
              resourceTime.analyticShadowExactFeasibleCandidates;
          payload["resource_time_analytic_shadow_admissible_candidates"] =
              resourceTime.analyticShadowAdmissibleCandidates;
          payload["resource_time_analytic_shadow_feasible_intersection"] =
              resourceTime.analyticShadowFeasibleIntersection;
          payload["resource_time_analytic_shadow_best_rank_matches"] =
              resourceTime.analyticShadowBestRankMatches;
          payload["resource_time_analytic_shadow_out_of_domain_candidates"] =
              resourceTime.analyticShadowOutOfDomainCandidates;
          payload["resource_time_analytic_shadow_maximum_lower_bound_gap_picoseconds"] =
              resourceTime.analyticShadowMaximumLowerBoundGapPicoseconds;
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
          payload["resource_time_unsupported_before_mapping_schedule_hints"] =
              resourceTime.unsupportedBeforeMappingScheduleHints;
          payload["resource_time_application_promotion_accounting_complete"] =
              resourceTime.applicationPromotionAccountingComplete;
          payload["resource_time_mapping_calls_avoided_by_sound_gate"] =
              resourceTime.mappingCallsAvoidedBySoundGate;
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
          payload["resource_time_exact_invocation_memo_coalesced_uncached_results"] =
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
        addOptionalUnsigned(
            payload, "time_to_first_feasible_wall_time_ns",
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
          if (observation.incompleteReason)
            entry["incomplete_reason"] =
                spelling(*observation.incompleteReason);
          else
            entry["incomplete_reason"] = nullptr;
          qualityObservations.push_back(std::move(entry));
        }
        payload["quality_observations"] = std::move(qualityObservations);
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
                        &outcome.resourceTimeSpectrum->verification))
              payload["resource_time_verification_incomplete_reason"] =
                  incomplete->diagnostic;
            else
              payload["resource_time_verification_incomplete_reason"] = nullptr;
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

} // namespace loom::application

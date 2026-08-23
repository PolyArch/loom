#include "DSE/PreMappingEvidence.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ComponentViewDigest.h"
#include "DSE/JointDesignPolicy.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

namespace loom::dse {
namespace {

std::string encodeRoot(const ArtifactRootReference &reference) {
  return llvm::toHex(encodeArtifactRootReference(reference),
                     /*LowerCase=*/true);
}

std::string encodeStructuredRoot(const frontend::StructuredEntityRef &root) {
  return llvm::toHex(frontend::encodeStructuredEntityRef(root),
                     /*LowerCase=*/true);
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
serializeTemporalWitness(const PreMappingTemporalWitness &witness) {
  return llvm::json::Object{
      {"logical_epoch_count", witness.logicalEpochCount},
      {"acc_core_occupancy", witness.accCoreOccupancy},
      {"launch_count", witness.launchCount},
      {"synchronization_count", witness.synchronizationCount},
      {"live_state_bytes", witness.liveStateBytes},
      {"live_state_known", witness.liveStateKnown},
      {"exact", witness.exact}};
}

llvm::json::Object
serializeCompleteness(const PreMappingSearchCompleteness &completeness) {
  return llvm::json::Object{
      {"domain_complete", completeness.domainComplete},
      {"budget_complete", completeness.budgetComplete},
      {"provider_complete", completeness.providerComplete},
      {"evidence_complete", completeness.evidenceComplete},
      {"selection_complete", completeness.selectionComplete},
      {"exact_complete", completeness.exactComplete()}};
}

llvm::json::Object
serializeFrontierPolicy(const PreMappingFrontierPolicy &policy) {
  llvm::json::Array beamWidths;
  for (std::uint64_t width : policy.beamWidthByExpansionDepth)
    beamWidths.push_back(width);
  return llvm::json::Object{
      {"maximum_coordinates_generated",
       policy.budget.maximumCoordinatesGenerated},
      {"maximum_source_observations", policy.budget.maximumSourceObservations},
      {"maximum_programs_materialized",
       policy.budget.maximumProgramsMaterialized},
      {"maximum_analytic_evaluations",
       policy.budget.maximumAnalyticEvaluations},
      {"maximum_functional_replays", policy.budget.maximumFunctionalReplays},
      {"maximum_dataflow_promotions", policy.budget.maximumDataflowPromotions},
      {"maximum_mapping_pairs", policy.budget.maximumMappingPairs},
      {"beam_width_by_expansion_depth", std::move(beamWidths)},
      {"diversity_candidate_count", policy.diversityCandidateCount},
      {"maximum_expansion_depth", policy.maximumExpansionDepth},
      {"maximum_compositional_groups", policy.maximumCompositionalGroups},
      {"spectrum_endpoint", toString(policy.spectrumEndpoint)},
      {"stopping_policy",
       jointDesignStoppingPolicySpelling(policy.stoppingPolicy)}};
}

} // namespace

llvm::json::Object
serializePreMappingWorkCounter(const PreMappingWorkCounter &counter) {
  return llvm::json::Object{
      {"limit", counter.limit},
      {"planned", counter.planned},
      {"reserved", counter.reserved},
      {"consumed", counter.consumed},
      {"rejected", counter.rejected},
      {"cancelled", counter.cancelled},
      {"elapsed_nanoseconds", counter.elapsedNanoseconds}};
}

llvm::json::Object
serializePreMappingWorkAccounting(const PreMappingWorkAccounting &accounting) {
  return llvm::json::Object{
      // WorkTimer scopes may nest (for example, functional replay can be
      // measured inside the enclosing analytic invocation). Consumers must
      // not sum these durations as if they were disjoint intervals.
      {"elapsed_semantics", "inclusive_nested"},
      {"source_observations",
       serializePreMappingWorkCounter(accounting.sourceObservations)},
      {"coordinates", serializePreMappingWorkCounter(accounting.coordinates)},
      {"program_materializations",
       serializePreMappingWorkCounter(accounting.programMaterializations)},
      {"analytic_evaluations",
       serializePreMappingWorkCounter(accounting.analyticEvaluations)},
      {"functional_replays",
       serializePreMappingWorkCounter(accounting.functionalReplays)},
      {"dataflow_promotions",
       serializePreMappingWorkCounter(accounting.dataflowPromotions)},
      {"mapping_pairs",
       serializePreMappingWorkCounter(accounting.mappingPairs)}};
}

llvm::json::Object serializePreMappingEvaluationTiming(
    const StructuredOwnershipEvaluationTiming &timing) {
  return llvm::json::Object{
      {"analytic_calls", timing.analyticCalls},
      {"analytic_elapsed_nanoseconds", timing.analyticElapsedNanoseconds},
      {"functional_replay_requests", timing.functionalReplayCalls},
      {"functional_replay_request_elapsed_nanoseconds",
       timing.functionalReplayElapsedNanoseconds}};
}

llvm::json::Object serializePreMappingFunnelSummary(
    llvm::ArrayRef<PreMappingCandidatePlanningRecord> inventory,
    llvm::ArrayRef<SelectedPreMappingCompilation> selected,
    const PreMappingWorkAccounting &accounting,
    const StructuredOwnershipEvaluationTiming &evaluationTiming) {
  std::uint64_t exactGateRejected = 0;
  std::uint64_t estimated = 0;
  std::uint64_t materialized = 0;
  std::uint64_t budgetDeferred = 0;
  std::uint64_t unsupported = 0;
  std::uint64_t unknown = 0;
  for (const PreMappingCandidatePlanningRecord &record : inventory) {
    exactGateRejected +=
        record.disposition ==
        PreMappingCandidatePlanningDisposition::ExactGateRejected;
    estimated += record.projection.has_value();
    materialized += record.materializedProjection.has_value();
    switch (record.disposition) {
    case PreMappingCandidatePlanningDisposition::CoordinateBudget:
    case PreMappingCandidatePlanningDisposition::ProgramMaterializationBudget:
    case PreMappingCandidatePlanningDisposition::AnalyticEvaluationBudget:
    case PreMappingCandidatePlanningDisposition::FunctionalReplayBudget:
    case PreMappingCandidatePlanningDisposition::DataflowPromotionBudget:
    case PreMappingCandidatePlanningDisposition::MappingPairBudget:
      ++budgetDeferred;
      break;
    case PreMappingCandidatePlanningDisposition::Unsupported:
      ++unsupported;
      break;
    case PreMappingCandidatePlanningDisposition::Unknown:
      ++unknown;
      break;
    case PreMappingCandidatePlanningDisposition::Retained:
    case PreMappingCandidatePlanningDisposition::HeuristicPruned:
    case PreMappingCandidatePlanningDisposition::ExactGateRejected:
    case PreMappingCandidatePlanningDisposition::CancelledOrTimeout:
      break;
    }
  }
  std::uint64_t replayed = 0;
  for (const SelectedPreMappingCompilation &candidate : selected)
    replayed += candidate.functionalReplay.has_value();
  return llvm::json::Object{
      {"planning_records_generated", inventory.size()},
      {"candidate_records_generated", inventory.size()},
      {"exact_gate_rejected", exactGateRejected},
      {"analytic_records_estimated", estimated},
      {"analytic_candidates_estimated", estimated},
      {"dataflow_records_materialized", materialized},
      {"dataflow_candidates_materialized", materialized},
      {"selected_candidates_with_functional_replay", replayed},
      {"functional_replay_requests",
       evaluationTiming.functionalReplayCalls},
      {"functional_replay_request_elapsed_nanoseconds",
       evaluationTiming.functionalReplayElapsedNanoseconds},
      {"budget_deferred_records", budgetDeferred},
      {"unsupported_records", unsupported},
      {"unknown_records", unknown},
      {"selected_records", selected.size()},
      {"mapping_units_admitted", accounting.mappingPairs.consumed},
      {"mapping_pair_slots_consumed", accounting.mappingPairs.consumed},
      {"mapping_units_reserved", accounting.mappingPairs.reserved},
      {"mapping_units_rejected", accounting.mappingPairs.rejected},
      {"mapping_units_cancelled", accounting.mappingPairs.cancelled}};
}

llvm::json::Object serializePreMappingCandidateProjection(
    const PreMappingCandidateProjection &projection) {
  llvm::json::Object object;
  object["identity"] = formatComponentViewDigestHex(projection.identity);
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
  object["exact_gate"] = toString(projection.exactGate);
  object["estimate_support"] = toString(projection.estimateSupport);
  object["estimate_confidence"] = toString(projection.estimateConfidence);
  return object;
}

llvm::json::Object serializePreMappingMaterializedProjection(
    const PreMappingMaterializedProjection &projection) {
  llvm::json::Object object;
  object["identity"] = formatComponentViewDigestHex(projection.identity);
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
  object["temporal_witness"] =
      serializeTemporalWitness(projection.temporalWitness);
  object["logical_domain_support"] = toString(projection.logicalDomainSupport);
  return object;
}

llvm::json::Object serializePreMappingCandidatePlanningRecord(
    const PreMappingCandidatePlanningRecord &record) {
  llvm::json::Object object;
  object["disposition"] = toString(record.disposition);
  if (record.candidateIdentity)
    object["candidate_identity"] =
        formatComponentViewDigestHex(*record.candidateIdentity);
  else
    object["candidate_identity"] = nullptr;
  addOptionalRoot(object, "structured_program", record.structuredProgram);
  addOptionalRoot(object, "canonical_dataflow", record.canonicalDataflow);
  llvm::json::Array ownedRoots;
  for (const frontend::StructuredEntityRef &root : record.ownedProtocolRoots)
    ownedRoots.push_back(encodeStructuredRoot(root));
  object["owned_protocol_roots"] = std::move(ownedRoots);
  llvm::json::Array seedKinds;
  for (PreMappingSpectrumSeedKind kind : record.seedKinds)
    seedKinds.push_back(toString(kind));
  object["seed_kinds"] = std::move(seedKinds);
  if (record.scheduleIntent)
    object["schedule_intent"] = toString(*record.scheduleIntent);
  else
    object["schedule_intent"] = nullptr;
  if (record.projection)
    object["projection"] =
        serializePreMappingCandidateProjection(*record.projection);
  else
    object["projection"] = nullptr;
  addOptionalUnsigned(object, "estimated_runtime_ps",
                      record.estimatedRuntimePicoseconds);
  addOptionalUnsigned(object, "preference_rank", record.preferenceRank);
  if (record.materializedProjection)
    object["materialized_projection"] =
        serializePreMappingMaterializedProjection(
            *record.materializedProjection);
  else
    object["materialized_projection"] = nullptr;
  if (record.temporalWitness)
    object["logical_domain_fact"] =
        serializeTemporalWitness(*record.temporalWitness);
  else
    object["logical_domain_fact"] = nullptr;
  if (record.verifiedSpectrum)
    object["verified_spectrum"] = toString(*record.verifiedSpectrum);
  else
    object["verified_spectrum"] = nullptr;
  if (record.incompleteReason)
    object["incomplete_reason"] = toString(*record.incompleteReason);
  else
    object["incomplete_reason"] = nullptr;
  return object;
}

llvm::json::Object serializePreMappingSelectionEvidence(
    const CompletedPreMappingSelection &selection) {
  llvm::json::Object object;
  object["schema"] = "loom.pre_mapping.evidence.2";
  object["status"] = "completed_selection";
  object["source_program"] = encodeRoot(selection.sourceProgram);
  object["fabric"] = encodeRoot(selection.fabric);
  object["workload"] = encodeRoot(selection.workload);
  object["runtime_input"] = encodeRoot(selection.runtimeInput);
  object["frontier_policy_digest"] =
      formatComponentViewDigestHex(selection.frontierPolicyDigest);
  object["requested_planner_mode"] = toString(selection.requestedPlannerMode);
  object["resolved_planner_mode"] = toString(selection.resolvedPlannerMode);
  object["frontier_policy"] = serializeFrontierPolicy(selection.frontierPolicy);
  object["eligible_coordinate_count"] = selection.eligibleCoordinateCount;
  object["coordinate_frontier_truncated"] =
      selection.coordinateFrontierTruncated;
  object["completeness"] = serializeCompleteness(selection.completeness);
  object["work_accounting"] =
      serializePreMappingWorkAccounting(selection.frontierAccounting);
  object["evaluation_timing"] =
      serializePreMappingEvaluationTiming(selection.evaluationTiming);
  object["funnel"] = serializePreMappingFunnelSummary(
      selection.candidateInventory, selection.selected,
      selection.frontierAccounting, selection.evaluationTiming);
  object["profile_cache_hits"] =
      selection.sharedEvaluationStatistics.profileCacheHits;
  object["profile_cache_misses"] =
      selection.sharedEvaluationStatistics.profileCacheMisses;
  object["profile_single_flight_waits"] =
      selection.sharedEvaluationStatistics.profileSingleFlightWaits;
  const auto &cache = selection.evaluationCacheStatistics;
  object["evaluation_cache"] = llvm::json::Object{
      {"analytic_primes", cache.analyticPrimeCount},
      {"analytic_hits", cache.analyticHitCount},
      {"analytic_misses", cache.analyticMissCount},
      {"analytic_single_flight_waits", cache.analyticSingleFlightWaitCount},
      {"functional_primes", cache.functionalPrimeCount},
      {"functional_hits", cache.functionalHitCount},
      {"functional_misses", cache.functionalMissCount},
      {"functional_single_flight_waits", cache.functionalSingleFlightWaitCount},
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
  if (selection.shadowRecall) {
    llvm::json::Array missing;
    for (const std::vector<std::size_t> &subset :
         selection.shadowRecall->missingSubsets) {
      llvm::json::Array encoded;
      for (std::size_t ordinal : subset)
        encoded.push_back(static_cast<std::uint64_t>(ordinal));
      missing.push_back(std::move(encoded));
    }
    object["shadow_recall"] = llvm::json::Object{
        {"eligible_subsets", selection.shadowRecall->eligibleSubsets},
        {"generated_subsets", selection.shadowRecall->generatedSubsets},
        {"covered_subsets", selection.shadowRecall->coveredSubsets},
        {"recall", selection.shadowRecall->recall()},
        {"missing_subsets", std::move(missing)}};
  } else {
    object["shadow_recall"] = nullptr;
  }
  llvm::json::Array inventory;
  for (auto indexed : llvm::enumerate(selection.candidateInventory)) {
    llvm::json::Object candidate =
        serializePreMappingCandidatePlanningRecord(indexed.value());
    candidate["planning_record_ordinal"] =
        static_cast<std::uint64_t>(indexed.index());
    inventory.push_back(std::move(candidate));
  }
  object["candidate_inventory"] = std::move(inventory);
  llvm::json::Array selected;
  for (const SelectedPreMappingCompilation &candidate : selection.selected) {
    llvm::json::Object selectedCandidate{
        {"preference_rank", candidate.preferenceRank}};
    if (candidate.planningRecordOrdinal) {
      selectedCandidate["planning_record_ordinal"] =
          static_cast<std::uint64_t>(*candidate.planningRecordOrdinal);
      if (*candidate.planningRecordOrdinal <
          selection.candidateInventory.size()) {
        const auto &record =
            selection.candidateInventory[*candidate.planningRecordOrdinal];
        if (record.candidateIdentity)
          selectedCandidate["candidate_identity"] =
              formatComponentViewDigestHex(*record.candidateIdentity);
        else
          selectedCandidate["candidate_identity"] = nullptr;
      }
    } else {
      selectedCandidate["planning_record_ordinal"] = nullptr;
    }
    selected.push_back(std::move(selectedCandidate));
  }
  object["selected_candidates"] = std::move(selected);
  return object;
}

llvm::json::Object serializePreMappingIncompleteEvidence(
    const IncompletePreMappingExploration &incomplete) {
  llvm::json::Object object;
  object["schema"] = "loom.pre_mapping.evidence.2";
  object["status"] = "incomplete";
  object["reason"] = toString(incomplete.reason);
  if (incomplete.planNodeOrdinal)
    object["plan_node_ordinal"] = *incomplete.planNodeOrdinal;
  else
    object["plan_node_ordinal"] = nullptr;
  object["completeness"] = serializeCompleteness(incomplete.completeness);
  object["evaluation_timing"] =
      serializePreMappingEvaluationTiming(incomplete.evaluationTiming);
  if (incomplete.checkpoint) {
    const PreMappingCheckpoint &checkpoint = *incomplete.checkpoint;
    llvm::json::Array retained;
    for (const ArtifactRootReference &candidate : checkpoint.retainedCandidates)
      retained.push_back(encodeRoot(candidate));
    llvm::json::Object checkpointObject{
        {"boundary", static_cast<std::uint64_t>(checkpoint.boundary)},
        {"reason", toString(checkpoint.reason)},
        {"source_program", encodeRoot(checkpoint.sourceProgram)},
        {"fabric", encodeRoot(checkpoint.fabric)},
        {"workload", encodeRoot(checkpoint.workload)},
        {"runtime_input", encodeRoot(checkpoint.runtimeInput)},
        {"frontier_policy_digest",
         formatComponentViewDigestHex(checkpoint.frontierPolicyDigest)},
        {"work_accounting",
         serializePreMappingWorkAccounting(checkpoint.workAccounting)},
        {"funnel", serializePreMappingFunnelSummary(
                       checkpoint.candidateInventory, {},
                       checkpoint.workAccounting, incomplete.evaluationTiming)},
        {"completeness", serializeCompleteness(checkpoint.completeness)},
        {"eligible_coordinate_count", checkpoint.eligibleCoordinateCount},
        {"coordinate_frontier_truncated",
         checkpoint.coordinateFrontierTruncated},
        {"retained_candidates", std::move(retained)}};
    llvm::json::Array inventory;
    for (auto indexed : llvm::enumerate(checkpoint.candidateInventory)) {
      llvm::json::Object candidate =
          serializePreMappingCandidatePlanningRecord(indexed.value());
      candidate["planning_record_ordinal"] =
          static_cast<std::uint64_t>(indexed.index());
      inventory.push_back(std::move(candidate));
    }
    checkpointObject["candidate_inventory"] = std::move(inventory);
    object["checkpoint"] = std::move(checkpointObject);
  } else {
    object["checkpoint"] = nullptr;
  }
  return object;
}

} // namespace loom::dse

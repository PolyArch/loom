#include "Application/BuildDiagnostics.h"
#include "ApplicationRuntimeValidationInternal.h"
#include "BuildInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/MappingDebugLog.h"
#include "DSE/JointHardwareReopen.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "PnR/PnrDerivedContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::application {

namespace build_detail {

/// A resource-time transition may reuse a SystemMapping only when the
/// Dataflow owner is unchanged and the typed partition allocation changes for
/// a finite set of roots. The returned roots are the complete application
/// invalidation seed; the System migration owner expands it transitively.
llvm::Expected<std::vector<dataflow::RootThreadLaunchRef>>
deriveApplicationPartitionDelta(const dse::JointDesignExplorationPlan &parent,
                                const dse::JointDesignExplorationPlan &child) {
  if (parent.pairOutputs.size() != 1 || child.pairOutputs.size() != 1)
    return invalid("application resource-time transition requires one pair");
  if (parent.pairOutputs.front().pair.software.dataflow.artifact !=
      child.pairOutputs.front().pair.software.dataflow.artifact)
    return invalid("application resource-time transition changes Dataflow");
  std::vector<dataflow::RootThreadLaunchRef> changed;
  for (const pnr::SystemBindingPartitionIntent &parentPartition :
       parent.systemBindingPartitions) {
    auto childPartition = llvm::find_if(
        child.systemBindingPartitions, [&](const auto &candidate) {
          return candidate.root == parentPartition.root;
        });
    if (childPartition == child.systemBindingPartitions.end() ||
        childPartition->partitionCount != parentPartition.partitionCount)
      changed.push_back(parentPartition.root);
  }
  for (const pnr::SystemBindingPartitionIntent &childPartition :
       child.systemBindingPartitions)
    if (llvm::none_of(parent.systemBindingPartitions,
                      [&](const auto &candidate) {
                        return candidate.root == childPartition.root;
                      }))
      changed.push_back(childPartition.root);
  llvm::sort(changed, [](const auto &lhs, const auto &rhs) {
    if (lhs.artifact != rhs.artifact)
      return lhs.artifact.bytes() < rhs.artifact.bytes();
    return lhs.entity.value() < rhs.entity.value();
  });
  changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
  return changed;
}

llvm::Expected<const dse::ResourceTimeScheduleHint *>
findResourceTimeScheduleHint(
    const dse::ResourceTimeCandidateFunnelEvaluation &evaluation,
    const ComponentViewDigest &digest) {
  const dse::ResourceTimeScheduleHint *result = nullptr;
  for (const dse::ResourceTimeScheduleHint &hint : evaluation.retainedHints) {
    auto candidate = dse::deriveResourceTimeScheduleHintDigest(hint);
    if (!candidate)
      return candidate.takeError();
    if (*candidate != digest)
      continue;
    if (result)
      return invalid("resource-time schedule provenance is not unique");
    result = &hint;
  }
  if (!result)
    return invalid("resource-time Mapping finalist lost its schedule "
                   "provenance");
  return result;
}

llvm::Expected<std::optional<dse::ResourceTimeSpectrumFunnelResult>>
verifyResourceTimeAlternative(
    const dse::ResourceTimeMappingFunnel &funnel,
    const PreparedApplicationMappingAlternative &alternative,
    llvm::ArrayRef<ArtifactRootReference> systemMappings,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const ComponentViewDigest &scheduleHintDigest,
    llvm::ArrayRef<dse::ResourceTimeMappingDeploymentEndpoint> endpoints,
    ExecutionControlView executionControl) {
  const auto evaluation =
      llvm::find_if(funnel.evaluations, [&](const auto &candidate) {
        return candidate.candidateIdentity == alternative.candidateIdentity;
      });
  if (evaluation == funnel.evaluations.end())
    return invalid("Mapping outcome has no resource-time evaluation");
  auto hint = findResourceTimeScheduleHint(*evaluation, scheduleHintDigest);
  if (!hint)
    return hint.takeError();
  auto verified = dse::verifyResourceTimeMappingFinalists(
      llvm::ArrayRef<dse::ResourceTimeScheduleHint>(*hint, 1),
      alternative.resourceTimeRegions, alternative.resourceTimeRegionBounds,
      systemMappings, artifacts, executionControl,
      evaluation->concurrencyBounds, &blobs, endpoints);
  if (!verified)
    return verified.takeError();
  return std::optional<dse::ResourceTimeSpectrumFunnelResult>(
      std::move(*verified));
}

} // namespace build_detail

using build_detail::ApplicationBuildOperationTimer;
using build_detail::classifyResourceTimeSelectionOutcome;
using build_detail::deriveApplicationPairDecision;
using build_detail::deriveApplicationPartitionDelta;
using build_detail::invalid;
using build_detail::requestedResourceTimeSpectrumClass;
using build_detail::verifyResourceTimeAlternative;

llvm::Expected<ApplicationMappingExecution>
executeApplicationMapping(const PreparedApplicationBuild &prepared,
                          ApplicationMappingExecutionRequest request,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::MappingExecution);
  if (prepared.mappingAlternatives.empty())
    return invalid("Mapping execution has no software alternative");
  pnr::PnrDerivedContextSession pnrDerivedContextSession;
  llvm::scope_exit emitPnrDerivedContextSession([&] {
    const pnr::PnrDerivedContextSessionStatistics statistics =
        pnrDerivedContextSession.statistics();
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "application_pnr_derived_context_session";
          fields["requests"] = statistics.requests;
          fields["cache_hits"] = statistics.cacheHits;
          fields["cache_misses"] = statistics.cacheMisses;
          fields["coalesced_waits"] = statistics.coalescedWaits;
          fields["revalidation_count"] = statistics.revalidationCount;
          fields["unique_constructions"] = statistics.uniqueConstructions;
          fields["uncached_constructions"] = statistics.uncachedConstructions;
          fields["construction_time_ns"] = statistics.constructionNanoseconds;
          fields["construction_time_saved_ns"] =
              statistics.constructionNanosecondsSaved;
          fields["deterministic_work"] = statistics.deterministicWork;
          fields["retained_bytes"] = statistics.retainedBytes;
          fields["retained_bytes_reused"] = statistics.retainedBytesReused;
          fields["entry_count"] = statistics.entryCount;
          fields["entry_limit"] = statistics.entryLimit;
        });
  });
  std::vector<ArtifactRootReference> evidence = prepared.satisfiedEvidence;
  evidence.insert(evidence.end(), request.preexistingEvidence.begin(),
                  request.preexistingEvidence.end());
  std::vector<ArtifactRootReference> qualitySemanticInputs;
  if (request.boundedQuality)
    qualitySemanticInputs = request.boundedQuality->semanticInputs;
  llvm::sort(qualitySemanticInputs, artifactRootReferenceLess);
  qualitySemanticInputs.erase(
      std::unique(qualitySemanticInputs.begin(), qualitySemanticInputs.end()),
      qualitySemanticInputs.end());
  std::vector<const dse::JointDesignExplorationPlan *> plans;
  plans.reserve(prepared.mappingAlternatives.size());
  for (const PreparedApplicationMappingAlternative &alternative :
       prepared.mappingAlternatives)
    plans.push_back(&alternative.plan);
  std::vector<ApplicationMappingCandidateOutcome> outcomes;
  std::vector<dse::JointDesignAttemptRecord> attempts;
  std::vector<ApplicationPairQualityInvocationRecord> qualityInvocations;
  std::uint64_t attemptedSoftwarePlans = 0;
  std::uint64_t hardwareReopenSearches = 0;
  std::uint64_t hardwareParentPromotions = 0;
  std::uint64_t hardwareReopensDeferredByQuality = 0;
  std::uint64_t hardwareReopensWithheldWithoutExactFeedback = 0;
  std::uint64_t hardwareRepairProbeLimit = 0;
  std::uint64_t hardwareRepairProbesPlanned = 0;
  std::uint64_t hardwareRepairProbesReserved = 0;
  std::uint64_t hardwareRepairProbesConsumed = 0;
  std::uint64_t hardwareRepairProbesRejected = 0;
  std::uint64_t hardwareRepairProbesCancelled = 0;
  std::uint64_t spatialMappingRepairCandidateLimit = 0;
  std::uint64_t spatialMappingRepairsPlanned = 0;
  std::uint64_t spatialMappingRepairsReserved = 0;
  std::uint64_t spatialMappingRepairsConsumed = 0;
  std::uint64_t spatialMappingRepairsRejected = 0;
  std::uint64_t spatialMappingRepairsCancelled = 0;
  std::uint64_t parentTechDecisions = 0;
  std::uint64_t parentSpatialDecisions = 0;
  std::uint64_t preservedTechDecisions = 0;
  std::uint64_t preservedSpatialDecisions = 0;
  std::uint64_t reopenedTechDecisions = 0;
  std::uint64_t reopenedSpatialDecisions = 0;
  std::uint64_t repairedTechDecisions = 0;
  std::uint64_t repairedSpatialDecisions = 0;
  std::uint64_t invalidationRootCount = 0;
  std::uint64_t invalidationConeDecisionCount = 0;
  std::uint64_t parentRouteNodeCount = 0;
  std::uint64_t preservedRouteNodeCount = 0;
  std::uint64_t reopenedRouteNodeCount = 0;
  std::uint64_t repairedRouteNodeCount = 0;
  std::uint64_t parentServiceLegCount = 0;
  std::uint64_t preservedServiceLegCount = 0;
  std::uint64_t reopenedServiceLegCount = 0;
  std::uint64_t verifiedAlternatives = 0;
  std::uint64_t techMappingDispatches = 0;
  std::uint64_t spatialPnrDispatches = 0;
  std::uint64_t systemPnrDispatches = 0;
  std::vector<ApplicationIncrementalMappingObservation>
      incrementalMappingObservations;
  const auto requestedSpectrumClass = requestedResourceTimeSpectrumClass(
      prepared.resourceTimePolicy.spectrumEndpoint);
  const auto outcomeMatchesRequestedSpectrum =
      [&](std::uint64_t planOrdinal, const ArtifactRootReference &mapping) {
        return llvm::any_of(outcomes, [&](const auto &outcome) {
          if (outcome.planOrdinal != planOrdinal ||
              !llvm::is_contained(outcome.systemMappings, mapping))
            return false;
          return !classifyResourceTimeSelectionOutcome(
              outcome.resourceTimeSpectrum, requestedSpectrumClass);
        });
      };
  const std::size_t mappingImportEntryLimit =
      prepared.mappingAlternatives.size() >
              std::numeric_limits<std::size_t>::max() / 16
          ? std::numeric_limits<std::size_t>::max()
          : std::max<std::size_t>(1, prepared.mappingAlternatives.size() * 16);
  // Keep one invocation-local immutable import session across all finalist
  // joins. Nested Spectrum verification reuses it but still runs its own
  // active-set and endpoint verifier.
  mapping::SystemMappingImportSession mappingImportSession(
      artifacts, mappingImportEntryLimit);

  const auto appendOutcomes = [&](const dse::JointDesignExecution &execution,
                                  std::size_t planOrdinalBase) -> llvm::Error {
    for (const dse::JointDesignAttemptRecord &attempt :
         execution.summary.attempts) {
      if (attempt.planOrdinal >
          std::numeric_limits<std::uint64_t>::max() - planOrdinalBase)
        return invalid("joint Mapping plan ordinal overflowed");
      const std::uint64_t planOrdinal = attempt.planOrdinal + planOrdinalBase;
      if (planOrdinal >= prepared.mappingAlternatives.size())
        return invalid("joint Mapping outcome has a foreign plan ordinal");
      const PreparedApplicationMappingAlternative &alternative =
          prepared.mappingAlternatives[planOrdinal];
      for (const ArtifactRootReference &mappingReference :
           attempt.systemMappings) {
        auto mapping =
            mapping::importSystemMapping(mappingReference, artifacts);
        if (!mapping)
          return mapping.takeError();
        if (mapping->view().dataflowIdentity() !=
                alternative.dataflow.artifact ||
            mapping->view().fabricIdentity() != attempt.system.artifact)
          return invalid("joint Mapping outcome disagrees with its exact "
                         "software/System owners");
      }
      if (alternative.preMappingCandidateRecordOrdinal >=
          prepared.candidateInventory.size())
        return invalid("Mapping outcome has a foreign planning-record ordinal");
      std::optional<dse::ResourceTimeSpectrumFunnelResult>
          emptyScheduleSpectrum;
      for (const ComponentViewDigest &scheduleHintDigest :
           alternative.equivalentScheduleHintDigests) {
        std::optional<dse::ResourceTimeSpectrumFunnelResult>
            resourceTimeSpectrum;
        if (!attempt.systemMappings.empty()) {
          auto verified = verifyResourceTimeAlternative(
              prepared.resourceTimeFunnel, alternative, attempt.systemMappings,
              artifacts, blobs, scheduleHintDigest, {},
              request.executionControl);
          if (!verified)
            return verified.takeError();
          resourceTimeSpectrum = std::move(*verified);
        }
        outcomes.push_back(ApplicationMappingCandidateOutcome{
            alternative.preMappingCandidateRecordOrdinal,
            planOrdinal,
            scheduleHintDigest,
            alternative.dataflow,
            attempt.system,
            attempt.disposition,
            attempt.incompleteNodeOrdinal,
            attempt.incompleteReason,
            attempt.systemMappings,
            prepared.candidateInventory[alternative
                                            .preMappingCandidateRecordOrdinal],
            alternative.plan.systemBindingPartitions,
            ApplicationMappingRuntimeDisposition::NotRequested,
            {},
            {},
            std::move(resourceTimeSpectrum),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            {}});
      }
      if (alternative.equivalentScheduleHintDigests.empty())
        outcomes.push_back(ApplicationMappingCandidateOutcome{
            alternative.preMappingCandidateRecordOrdinal,
            planOrdinal,
            alternative.resourceTimeScheduleHintDigest,
            alternative.dataflow,
            attempt.system,
            attempt.disposition,
            attempt.incompleteNodeOrdinal,
            attempt.incompleteReason,
            attempt.systemMappings,
            prepared.candidateInventory[alternative
                                            .preMappingCandidateRecordOrdinal],
            alternative.plan.systemBindingPartitions,
            ApplicationMappingRuntimeDisposition::NotRequested,
            {},
            {},
            std::move(emptyScheduleSpectrum),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            {}});
      dse::JointDesignAttemptRecord adjusted = attempt;
      adjusted.planOrdinal = planOrdinal;
      attempts.push_back(std::move(adjusted));
    }
    return llvm::Error::success();
  };

  const auto executeTail =
      [&](std::size_t firstPlan) -> llvm::Expected<dse::JointDesignExecution> {
    llvm::ArrayRef<const dse::JointDesignExplorationPlan *> tail(plans);
    tail = tail.drop_front(firstPlan);
    std::string journalRoot = request.journalRoot;
    if (firstPlan != 0) {
      llvm::SmallString<256> childJournal(journalRoot);
      llvm::sys::path::append(childJournal,
                              "runtime-qualified-" + std::to_string(firstPlan));
      journalRoot = childJournal.str().str();
    }
    std::uint64_t maximumUsefulAccCoreCount = 0;
    for (std::size_t ordinal = firstPlan;
         ordinal != prepared.mappingAlternatives.size(); ++ordinal)
      for (const auto &bound :
           prepared.mappingAlternatives[ordinal].resourceTimeRegionBounds)
        maximumUsefulAccCoreCount = std::max(maximumUsefulAccCoreCount,
                                             bound.maximumUsefulResourceUnits);
    std::optional<dse::JointBoundedQualityPolicy> boundedQuality =
        request.boundedQuality;
    if (boundedQuality && firstPlan != 0) {
      dse::JointDesignQualityAcquirer acquire = boundedQuality->acquire;
      boundedQuality->acquire = [acquire = std::move(acquire), firstPlan](
                                    const dse::JointDesignExecution &execution,
                                    std::uint64_t planOrdinal)
          -> llvm::Expected<dse::JointDesignQualityAcquisition> {
        if (planOrdinal > std::numeric_limits<std::uint64_t>::max() - firstPlan)
          return invalid("bounded-quality plan ordinal overflowed");
        return acquire(execution, planOrdinal + firstPlan);
      };
      if (boundedQuality->hardwarePromotion) {
        dse::JointHardwarePromotionQualityAcquirer promote =
            boundedQuality->hardwarePromotion->acquire;
        boundedQuality->hardwarePromotion->acquire =
            [promote = std::move(promote),
             firstPlan](const dse::JointDesignExplorationPlan &plan,
                        std::uint64_t planOrdinal)
            -> llvm::Expected<dse::JointDesignQualityAcquisition> {
          if (planOrdinal >
              std::numeric_limits<std::uint64_t>::max() - firstPlan)
            return invalid("hardware-promotion plan ordinal overflowed");
          return promote(plan, planOrdinal + firstPlan);
        };
      }
    }
    dse::JointHardwareReopenRequest reopenRequest{
        request.producer,
        std::move(journalRoot),
        evidence,
        prepared.preMappingFrontierPolicy.stoppingPolicy,
        std::move(boundedQuality),
        maximumUsefulAccCoreCount == 0
            ? std::nullopt
            : std::optional<std::uint64_t>(maximumUsefulAccCoreCount),
        request.siteCapacity,
        request.executionPolicy};
    reopenRequest.spectrumEndpoint =
        prepared.resourceTimePolicy.spectrumEndpoint;
    reopenRequest.hardwareExplorationScope = request.hardwareExplorationScope;
    reopenRequest.invocationSemanticInputs = qualitySemanticInputs;
    return dse::executeJointDesignWithHardwareReopen(
        tail, prepared.jointPolicy, std::move(reopenRequest), artifacts, blobs);
  };

  const auto makeRepairRequest =
      [&](std::string journalRoot) -> dse::JointHardwareReopenRequest {
    dse::JointHardwareReopenRequest repairRequest{
        request.producer,
        std::move(journalRoot),
        evidence,
        dse::JointDesignStoppingPolicy::FirstVerified,
        std::nullopt,
        std::nullopt,
        request.siteCapacity,
        request.executionPolicy};
    repairRequest.invocationSemanticInputs = qualitySemanticInputs;
    return repairRequest;
  };

  std::optional<dse::JointDesignExecution> selectedExecution;
  std::size_t firstPlan = 0;
  while (firstPlan < plans.size()) {
    auto execution = executeTail(firstPlan);
    if (!execution)
      return execution.takeError();
    qualityInvocations.push_back(ApplicationPairQualityInvocationRecord{
        static_cast<std::uint64_t>(firstPlan),
        execution->summary.invocationRunKey,
        execution->summary.qualityDisposition,
        execution->summary.qualityIncompleteCandidate,
        execution->summary.qualityObjectiveDimensionLabels,
        execution->summary.qualityObservations,
        execution->summary.hardwarePromotionObjectiveDimensionLabels,
        execution->summary.hardwarePromotionObservations,
        execution->summary.selectedPlanOrdinal,
        execution->summary.selectedMapping});
    for (dse::JointHardwarePromotionObservation &observation :
         execution->summary.hardwarePromotionObservations) {
      if (observation.planOrdinal >
          std::numeric_limits<std::uint64_t>::max() - firstPlan)
        return invalid("hardware-promotion observation ordinal overflowed");
      observation.planOrdinal += firstPlan;
    }
    if (llvm::Error error = appendOutcomes(*execution, firstPlan))
      return std::move(error);
    attemptedSoftwarePlans += execution->summary.attemptedSoftwarePlans;
    hardwareReopenSearches += execution->summary.hardwareReopenSearches;
    hardwareParentPromotions += execution->summary.hardwareParentPromotions;
    hardwareReopensDeferredByQuality +=
        execution->summary.hardwareReopensDeferredByQuality;
    hardwareReopensWithheldWithoutExactFeedback +=
        execution->summary.hardwareReopensWithheldWithoutExactFeedback;
    hardwareRepairProbeLimit += execution->summary.hardwareRepairProbeLimit;
    hardwareRepairProbesPlanned +=
        execution->summary.hardwareRepairProbesPlanned;
    hardwareRepairProbesReserved +=
        execution->summary.hardwareRepairProbesReserved;
    hardwareRepairProbesConsumed +=
        execution->summary.hardwareRepairProbesConsumed;
    hardwareRepairProbesRejected +=
        execution->summary.hardwareRepairProbesRejected;
    hardwareRepairProbesCancelled +=
        execution->summary.hardwareRepairProbesCancelled;
    spatialMappingRepairCandidateLimit +=
        execution->summary.spatialMappingRepairCandidateLimit;
    spatialMappingRepairsPlanned +=
        execution->summary.spatialMappingRepairsPlanned;
    spatialMappingRepairsReserved +=
        execution->summary.spatialMappingRepairsReserved;
    spatialMappingRepairsConsumed +=
        execution->summary.spatialMappingRepairsConsumed;
    spatialMappingRepairsRejected +=
        execution->summary.spatialMappingRepairsRejected;
    spatialMappingRepairsCancelled +=
        execution->summary.spatialMappingRepairsCancelled;
    parentTechDecisions += execution->summary.parentTechDecisions;
    parentSpatialDecisions += execution->summary.parentSpatialDecisions;
    preservedTechDecisions += execution->summary.preservedTechDecisions;
    preservedSpatialDecisions += execution->summary.preservedSpatialDecisions;
    reopenedTechDecisions += execution->summary.reopenedTechDecisions;
    reopenedSpatialDecisions += execution->summary.reopenedSpatialDecisions;
    repairedTechDecisions += execution->summary.repairedTechDecisions;
    repairedSpatialDecisions += execution->summary.repairedSpatialDecisions;
    invalidationRootCount += execution->summary.invalidationRootCount;
    invalidationConeDecisionCount +=
        execution->summary.invalidationConeDecisionCount;
    parentRouteNodeCount += execution->summary.parentRouteNodeCount;
    preservedRouteNodeCount += execution->summary.preservedRouteNodeCount;
    reopenedRouteNodeCount += execution->summary.reopenedRouteNodeCount;
    repairedRouteNodeCount += execution->summary.repairedRouteNodeCount;
    parentServiceLegCount += execution->summary.parentServiceLegCount;
    preservedServiceLegCount += execution->summary.preservedServiceLegCount;
    reopenedServiceLegCount += execution->summary.reopenedServiceLegCount;
    verifiedAlternatives += execution->summary.verifiedAlternatives;
    techMappingDispatches += execution->summary.techMappingDispatchCount;
    spatialPnrDispatches += execution->summary.spatialPnrDispatchCount;
    systemPnrDispatches += execution->summary.systemPnrDispatchCount;

    if (!execution->summary.selectedPlanOrdinal ||
        !execution->summary.selectedMapping) {
      selectedExecution.emplace(std::move(*execution));
      break;
    }
    if (*execution->summary.selectedPlanOrdinal >
        std::numeric_limits<std::uint64_t>::max() - firstPlan)
      return invalid("selected Mapping plan ordinal overflowed");
    const std::uint64_t selectedPlanOrdinal =
        *execution->summary.selectedPlanOrdinal + firstPlan;
    if (selectedPlanOrdinal >= prepared.mappingAlternatives.size())
      return invalid("selected Mapping has a foreign plan ordinal");
    auto runtime = detail::validateApplicationMappingRuntime(
        prepared, prepared.mappingAlternatives[selectedPlanOrdinal], *execution,
        request.executionPolicy, artifacts, blobs);
    if (!runtime)
      return runtime.takeError();
    bool joined = false;
    for (ApplicationMappingCandidateOutcome &outcome : outcomes) {
      if (outcome.planOrdinal != selectedPlanOrdinal ||
          !llvm::is_contained(outcome.systemMappings,
                              *execution->summary.selectedMapping))
        continue;
      outcome.runtimeDisposition = runtime->disposition;
      outcome.runtimeEvidence = runtime->evidence;
      outcome.oracleEvidence = runtime->oracleEvidence;
      outcome.dfgCycles = runtime->dfgCycles;
      outcome.cgraCycles = runtime->cgraCycles;
      outcome.resourceCoreCost = prepared.preMappingFabricAccCoreCount;
      joined = true;
    }
    if (!joined)
      return invalid("runtime validation has no exact Mapping attempt join");
    // An explicit spectrum endpoint constrains selection, not just reporting.
    // Keep a verified non-endpoint result as evidence, but continue through
    // the already bounded finalist frontier until a real SystemMapping proves
    // the requested class.
    if (runtime->disposition ==
            ApplicationMappingRuntimeDisposition::Completed &&
        !outcomeMatchesRequestedSpectrum(selectedPlanOrdinal,
                                         *execution->summary.selectedMapping)) {
      execution->summary.selectedPlanOrdinal.reset();
      execution->summary.selectedMapping.reset();
      selectedExecution.emplace(std::move(*execution));
      firstPlan = static_cast<std::size_t>(selectedPlanOrdinal) + 1;
      continue;
    }
    auto consumeRepairedExecutions =
        [&](auto &repaired) -> llvm::Expected<bool> {
      for (std::size_t childOrdinal = 0;
           childOrdinal != repaired->executions.size(); ++childOrdinal) {
        dse::JointDesignExecution &childExecution =
            repaired->executions[childOrdinal];
        if (childOrdinal >= repaired->childSystems.size())
          return invalid("hardware repair lost its child System");
        techMappingDispatches +=
            childExecution.summary.techMappingDispatchCount;
        spatialPnrDispatches += childExecution.summary.spatialPnrDispatchCount;
        systemPnrDispatches += childExecution.summary.systemPnrDispatchCount;
        std::vector<ArtifactRootReference> childMappings;
        for (const dse::JointMappedPair &pair : childExecution.mappedPairs)
          childMappings.insert(childMappings.end(), pair.systemMappings.begin(),
                               pair.systemMappings.end());
        llvm::sort(childMappings, artifactRootReferenceLess);
        childMappings.erase(
            std::unique(childMappings.begin(), childMappings.end()),
            childMappings.end());
        attempts.push_back({selectedPlanOrdinal,
                            repaired->childSystems[childOrdinal],
                            childMappings.empty()
                                ? dse::JointDesignAttemptDisposition::Incomplete
                                : dse::JointDesignAttemptDisposition::Verified,
                            std::nullopt,
                            childMappings.empty()
                                ? std::optional<dse::DsePlanIncompleteReason>(
                                      dse::CandidateGeneratorIncompleteReason::
                                          ProofNotEstablished)
                                : std::nullopt,
                            childMappings});
        if (childMappings.empty() || !childExecution.summary.selectedMapping)
          continue;
        auto childRuntime = detail::validateApplicationMappingRuntime(
            prepared, prepared.mappingAlternatives[selectedPlanOrdinal],
            childExecution, request.executionPolicy, artifacts, blobs);
        if (!childRuntime)
          return childRuntime.takeError();
        auto childSpectrum = verifyResourceTimeAlternative(
            prepared.resourceTimeFunnel,
            prepared.mappingAlternatives[selectedPlanOrdinal], childMappings,
            artifacts, blobs,
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .resourceTimeScheduleHintDigest,
            {}, request.executionControl);
        if (!childSpectrum)
          return childSpectrum.takeError();
        outcomes.push_back(ApplicationMappingCandidateOutcome{
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .preMappingCandidateRecordOrdinal,
            selectedPlanOrdinal,
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .resourceTimeScheduleHintDigest,
            prepared.mappingAlternatives[selectedPlanOrdinal].dataflow,
            repaired->childSystems[childOrdinal],
            dse::JointDesignAttemptDisposition::Verified,
            std::nullopt,
            std::nullopt,
            childMappings,
            prepared.candidateInventory
                [prepared.mappingAlternatives[selectedPlanOrdinal]
                     .preMappingCandidateRecordOrdinal],
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .plan.systemBindingPartitions,
            childRuntime->disposition,
            childRuntime->evidence,
            {},
            std::move(*childSpectrum),
            childRuntime->dfgCycles,
            childRuntime->cgraCycles,
            std::nullopt,
            childRuntime->oracleEvidence});
        if (childRuntime->disposition ==
                ApplicationMappingRuntimeDisposition::Completed &&
            outcomeMatchesRequestedSpectrum(
                selectedPlanOrdinal,
                childExecution.summary.selectedMapping.value())) {
          childExecution.summary.selectedPlanOrdinal = selectedPlanOrdinal;
          selectedExecution.emplace(std::move(childExecution));
          return true;
        }
      }
      return false;
    };
    if (runtime->disposition !=
            ApplicationMappingRuntimeDisposition::Completed &&
        runtime->spatialTransportFeedback &&
        runtime->spatialTransportFeedback->disposition ==
            dse::SpatialTransportRuntimeFeedbackDisposition::Exact) {
      llvm::SmallString<256> feedbackJournal(request.journalRoot);
      llvm::sys::path::append(feedbackJournal,
                              "transport-runtime-feedback-" +
                                  std::to_string(selectedPlanOrdinal));
      auto repaired = dse::executeSpatialTransportRuntimeRepair(
          prepared.mappingAlternatives[selectedPlanOrdinal].plan, *execution,
          prepared.jointPolicy, *runtime->spatialTransportFeedback,
          makeRepairRequest(feedbackJournal.str().str()), artifacts, blobs);
      if (!repaired)
        return repaired.takeError();
      spatialMappingRepairCandidateLimit += repaired->candidateLimit;
      spatialMappingRepairsPlanned += repaired->candidatesPlanned;
      spatialMappingRepairsReserved += repaired->candidatesReserved;
      spatialMappingRepairsConsumed += repaired->candidatesConsumed;
      spatialMappingRepairsRejected += repaired->candidatesRejected;
      spatialMappingRepairsCancelled += repaired->candidatesCancelled;
      auto selected = consumeRepairedExecutions(repaired);
      if (!selected)
        return selected.takeError();
      if (*selected)
        break;
    }
    if (runtime->disposition !=
            ApplicationMappingRuntimeDisposition::Completed &&
        runtime->spatialFifoFeedback &&
        runtime->spatialFifoFeedback->disposition ==
            dse::SpatialFifoRuntimeFeedbackDisposition::Exact) {
      llvm::SmallString<256> feedbackJournal(request.journalRoot);
      llvm::sys::path::append(feedbackJournal,
                              "fifo-runtime-feedback-" +
                                  std::to_string(selectedPlanOrdinal));
      auto repaired = dse::executeSpatialFifoHardwareFeedbackReopen(
          prepared.mappingAlternatives[selectedPlanOrdinal].plan, *execution,
          prepared.jointPolicy, *runtime->spatialFifoFeedback,
          makeRepairRequest(feedbackJournal.str().str()), artifacts, blobs);
      if (!repaired)
        return repaired.takeError();
      hardwareRepairProbeLimit += repaired->candidateLimit;
      hardwareRepairProbesPlanned += repaired->candidatesPlanned;
      hardwareRepairProbesReserved += repaired->candidatesReserved;
      hardwareRepairProbesConsumed += repaired->candidatesConsumed;
      hardwareRepairProbesRejected += repaired->candidatesRejected;
      hardwareRepairProbesCancelled += repaired->candidatesCancelled;
      for (std::size_t childOrdinal = 0;
           childOrdinal != repaired->executions.size(); ++childOrdinal) {
        dse::JointDesignExecution &childExecution =
            repaired->executions[childOrdinal];
        if (childOrdinal >= repaired->childSystems.size())
          return invalid("FIFO hardware repair lost its child System");
        techMappingDispatches +=
            childExecution.summary.techMappingDispatchCount;
        spatialPnrDispatches += childExecution.summary.spatialPnrDispatchCount;
        systemPnrDispatches += childExecution.summary.systemPnrDispatchCount;
        std::vector<ArtifactRootReference> childMappings;
        for (const dse::JointMappedPair &pair : childExecution.mappedPairs)
          childMappings.insert(childMappings.end(), pair.systemMappings.begin(),
                               pair.systemMappings.end());
        llvm::sort(childMappings, artifactRootReferenceLess);
        childMappings.erase(
            std::unique(childMappings.begin(), childMappings.end()),
            childMappings.end());
        attempts.push_back({selectedPlanOrdinal,
                            repaired->childSystems[childOrdinal],
                            childMappings.empty()
                                ? dse::JointDesignAttemptDisposition::Incomplete
                                : dse::JointDesignAttemptDisposition::Verified,
                            std::nullopt,
                            childMappings.empty()
                                ? std::optional<dse::DsePlanIncompleteReason>(
                                      dse::CandidateGeneratorIncompleteReason::
                                          ProofNotEstablished)
                                : std::nullopt,
                            childMappings});
        if (childMappings.empty() || !childExecution.summary.selectedMapping)
          continue;
        auto childRuntime = detail::validateApplicationMappingRuntime(
            prepared, prepared.mappingAlternatives[selectedPlanOrdinal],
            childExecution, request.executionPolicy, artifacts, blobs);
        if (!childRuntime)
          return childRuntime.takeError();
        auto childSpectrum = verifyResourceTimeAlternative(
            prepared.resourceTimeFunnel,
            prepared.mappingAlternatives[selectedPlanOrdinal], childMappings,
            artifacts, blobs,
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .resourceTimeScheduleHintDigest,
            {}, request.executionControl);
        if (!childSpectrum)
          return childSpectrum.takeError();
        outcomes.push_back(ApplicationMappingCandidateOutcome{
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .preMappingCandidateRecordOrdinal,
            selectedPlanOrdinal,
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .resourceTimeScheduleHintDigest,
            prepared.mappingAlternatives[selectedPlanOrdinal].dataflow,
            repaired->childSystems[childOrdinal],
            dse::JointDesignAttemptDisposition::Verified,
            std::nullopt,
            std::nullopt,
            childMappings,
            prepared.candidateInventory
                [prepared.mappingAlternatives[selectedPlanOrdinal]
                     .preMappingCandidateRecordOrdinal],
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .plan.systemBindingPartitions,
            childRuntime->disposition,
            childRuntime->evidence,
            {},
            std::move(*childSpectrum),
            childRuntime->dfgCycles,
            childRuntime->cgraCycles,
            std::nullopt,
            childRuntime->oracleEvidence});
        if (childRuntime->disposition !=
            ApplicationMappingRuntimeDisposition::Completed)
          continue;
        if (!outcomeMatchesRequestedSpectrum(
                selectedPlanOrdinal,
                childExecution.summary.selectedMapping.value()))
          continue;
        childExecution.summary.selectedPlanOrdinal = selectedPlanOrdinal;
        selectedExecution.emplace(std::move(childExecution));
        break;
      }
      if (selectedExecution && selectedExecution->summary.selectedPlanOrdinal)
        break;
    }
    if (runtime->disposition !=
            ApplicationMappingRuntimeDisposition::Completed &&
        request.hardwareExplorationScope ==
            dse::JointHardwareExplorationScope::BoundedHardwareReopen &&
        runtime->spatialOperandQueueFeedback &&
        runtime->spatialOperandQueueFeedback->disposition ==
            dse::SpatialOperandQueueRuntimeFeedbackDisposition::Exact) {
      llvm::SmallString<256> feedbackJournal(request.journalRoot);
      llvm::sys::path::append(feedbackJournal,
                              "operand-buffer-runtime-feedback-" +
                                  std::to_string(selectedPlanOrdinal));
      auto repaired = dse::executeSpatialOperandBufferHardwareFeedbackReopen(
          prepared.mappingAlternatives[selectedPlanOrdinal].plan, *execution,
          prepared.jointPolicy, *runtime->spatialOperandQueueFeedback,
          makeRepairRequest(feedbackJournal.str().str()), artifacts, blobs);
      if (!repaired)
        return repaired.takeError();
      hardwareRepairProbeLimit += repaired->candidateLimit;
      hardwareRepairProbesPlanned += repaired->candidatesPlanned;
      hardwareRepairProbesReserved += repaired->candidatesReserved;
      hardwareRepairProbesConsumed += repaired->candidatesConsumed;
      hardwareRepairProbesRejected += repaired->candidatesRejected;
      hardwareRepairProbesCancelled += repaired->candidatesCancelled;
      auto selected = consumeRepairedExecutions(repaired);
      if (!selected)
        return selected.takeError();
      if (*selected)
        break;
    }
    if (runtime->disposition ==
        ApplicationMappingRuntimeDisposition::Completed) {
      // Exercise every already-retained resource-time state for this exact
      // Dataflow through the application owner while the parent Mapping is
      // live. The frontier has already supplied the finite bound; this loop
      // does not enumerate a new candidate domain. Walking both directions
      // matters when the selected quality winner is the last retained plan.
      // Each child remains subject to the ordinary System verifier and is
      // recorded even when it cannot close.
      for (std::size_t childOrdinal = 0;
           childOrdinal != prepared.mappingAlternatives.size();
           ++childOrdinal) {
        if (childOrdinal == selectedPlanOrdinal)
          continue;
        const PreparedApplicationMappingAlternative &childAlternative =
            prepared.mappingAlternatives[childOrdinal];
        if (childAlternative.dataflow.artifact !=
            prepared.mappingAlternatives[selectedPlanOrdinal].dataflow.artifact)
          continue;
        auto reopenedRoots = deriveApplicationPartitionDelta(
            prepared.mappingAlternatives[selectedPlanOrdinal].plan,
            childAlternative.plan);
        if (!reopenedRoots)
          return reopenedRoots.takeError();
        if (reopenedRoots->empty())
          continue;
        llvm::SmallString<256> adjacentJournal(request.journalRoot);
        llvm::sys::path::append(adjacentJournal,
                                "application-resource-time-adjacent-" +
                                    std::to_string(selectedPlanOrdinal) + "-" +
                                    std::to_string(childOrdinal));
        dse::JointHardwareReopenRequest adjacentRequest =
            makeRepairRequest(adjacentJournal.str().str());
        adjacentRequest.spectrumEndpoint =
            prepared.resourceTimePolicy.spectrumEndpoint;
        auto adjacent = dse::executeResourceTimeAdjacentMappingRepair(
            prepared.mappingAlternatives[selectedPlanOrdinal].plan, *execution,
            prepared.jointPolicy, childAlternative.plan.systemBindingPartitions,
            *reopenedRoots, std::move(adjacentRequest), artifacts, blobs);
        if (!adjacent)
          return adjacent.takeError();
        const dse::JointDesignExecutionSummary &childSummary =
            adjacent->execution.summary;
        techMappingDispatches += childSummary.techMappingDispatchCount;
        spatialPnrDispatches += childSummary.spatialPnrDispatchCount;
        systemPnrDispatches += childSummary.systemPnrDispatchCount;
        std::vector<ArtifactRootReference> childMappings;
        for (const dse::JointMappedPair &pair : adjacent->execution.mappedPairs)
          childMappings.insert(childMappings.end(), pair.systemMappings.begin(),
                               pair.systemMappings.end());
        llvm::sort(childMappings, artifactRootReferenceLess);
        childMappings.erase(
            std::unique(childMappings.begin(), childMappings.end()),
            childMappings.end());
        const bool childHasMapping =
            childSummary.selectedMapping &&
            llvm::is_contained(childMappings, *childSummary.selectedMapping);
        dse::JointDesignAttemptDisposition childDisposition =
            childHasMapping ? dse::JointDesignAttemptDisposition::Verified
                            : dse::JointDesignAttemptDisposition::Incomplete;
        std::optional<dse::DsePlanIncompleteReason> childIncompleteReason;
        if (!childHasMapping) {
          for (const dse::JointDesignAttemptRecord &attempt :
               childSummary.attempts) {
            childDisposition = attempt.disposition;
            childIncompleteReason = attempt.incompleteReason;
            break;
          }
          if (childDisposition ==
                  dse::JointDesignAttemptDisposition::Incomplete &&
              !childIncompleteReason)
            childIncompleteReason =
                dse::CandidateGeneratorIncompleteReason::ProofNotEstablished;
        }
        ApplicationIncrementalMappingObservation observation{
            *execution->summary.selectedMapping,
            childAlternative.plan.pairOutputs.front().pair.system,
            childSummary.selectedMapping,
            adjacent->coldMapping,
            static_cast<std::uint64_t>(selectedPlanOrdinal),
            static_cast<std::uint64_t>(childOrdinal),
            prepared.mappingAlternatives[selectedPlanOrdinal]
                .resourceTimeScheduleHintDigest,
            childAlternative.resourceTimeScheduleHintDigest,
            *reopenedRoots,
            adjacent->reuseDisposition,
            childSummary.preservedTechMappings,
            childSummary.preservedSpatialMappings,
            childSummary.repairedTechMappings,
            childSummary.repairedSpatialMappings,
            childSummary.preservedThreadBindingCount +
                childSummary.preservedGraphBindingCount,
            childSummary.reopenedThreadBindingCount +
                childSummary.reopenedGraphBindingCount,
            childDisposition,
            childIncompleteReason,
            adjacent->coldExecution.summary.executionWallTimeNanoseconds,
            childSummary.incrementalReopenWallTimeNanoseconds,
            adjacent->coldExecution.summary.executionWallTimeNanoseconds +
                childSummary.incrementalReopenWallTimeNanoseconds,
            adjacent->coldVerification.retainedBytes,
            adjacent->incrementalVerification.retainedBytes,
            adjacent->coldVerification.deterministicWork,
            adjacent->incrementalVerification.deterministicWork,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            false};
        if (childHasMapping) {
          adjacent->execution.summary.selectedPlanOrdinal = childOrdinal;
          auto childRuntime = detail::validateApplicationMappingRuntime(
              prepared, childAlternative, adjacent->execution,
              request.executionPolicy, artifacts, blobs);
          if (!childRuntime)
            return childRuntime.takeError();
          auto childSpectrum = verifyResourceTimeAlternative(
              prepared.resourceTimeFunnel, childAlternative, childMappings,
              artifacts, blobs, childAlternative.resourceTimeScheduleHintDigest,
              {}, request.executionControl);
          if (!childSpectrum)
            return childSpectrum.takeError();
          bool coldVerified = false;
          if (adjacent->coldMapping) {
            adjacent->coldExecution.summary.selectedPlanOrdinal = childOrdinal;
            auto coldRuntime = detail::validateApplicationMappingRuntime(
                prepared, childAlternative, adjacent->coldExecution,
                request.executionPolicy, artifacts, blobs);
            if (!coldRuntime)
              return coldRuntime.takeError();
            std::vector<ArtifactRootReference> coldMappings;
            for (const dse::JointMappedPair &pair :
                 adjacent->coldExecution.mappedPairs)
              coldMappings.insert(coldMappings.end(),
                                  pair.systemMappings.begin(),
                                  pair.systemMappings.end());
            llvm::sort(coldMappings, artifactRootReferenceLess);
            coldMappings.erase(
                std::unique(coldMappings.begin(), coldMappings.end()),
                coldMappings.end());
            auto coldSpectrum = verifyResourceTimeAlternative(
                prepared.resourceTimeFunnel, childAlternative, coldMappings,
                artifacts, blobs,
                childAlternative.resourceTimeScheduleHintDigest, {},
                request.executionControl);
            if (!coldSpectrum)
              return coldSpectrum.takeError();
            observation.coldDfgCycles = coldRuntime->dfgCycles;
            observation.coldCgraCycles = coldRuntime->cgraCycles;
            coldVerified =
                coldRuntime->disposition ==
                    ApplicationMappingRuntimeDisposition::Completed &&
                coldSpectrum->has_value() &&
                std::holds_alternative<dse::VerifiedResourceTimeSpectrum>(
                    (*coldSpectrum)->verification);
          }
          observation.incrementalDfgCycles = childRuntime->dfgCycles;
          observation.incrementalCgraCycles = childRuntime->cgraCycles;
          observation.verified =
              coldVerified &&
              childRuntime->disposition ==
                  ApplicationMappingRuntimeDisposition::Completed &&
              childSpectrum->has_value() &&
              std::holds_alternative<dse::VerifiedResourceTimeSpectrum>(
                  (*childSpectrum)->verification);
          outcomes.push_back(ApplicationMappingCandidateOutcome{
              childAlternative.preMappingCandidateRecordOrdinal,
              childOrdinal,
              childAlternative.resourceTimeScheduleHintDigest,
              childAlternative.dataflow,
              childAlternative.plan.pairOutputs.front().pair.system,
              childDisposition,
              std::nullopt,
              std::nullopt,
              childMappings,
              prepared.candidateInventory
                  [childAlternative.preMappingCandidateRecordOrdinal],
              childAlternative.plan.systemBindingPartitions,
              childRuntime->disposition,
              childRuntime->evidence,
              {},
              std::move(*childSpectrum),
              childRuntime->dfgCycles,
              childRuntime->cgraCycles,
              std::nullopt,
              childRuntime->oracleEvidence});
        } else {
          outcomes.push_back(ApplicationMappingCandidateOutcome{
              childAlternative.preMappingCandidateRecordOrdinal,
              childOrdinal,
              childAlternative.resourceTimeScheduleHintDigest,
              childAlternative.dataflow,
              childAlternative.plan.pairOutputs.front().pair.system,
              childDisposition,
              std::nullopt,
              childIncompleteReason,
              {},
              prepared.candidateInventory
                  [childAlternative.preMappingCandidateRecordOrdinal],
              childAlternative.plan.systemBindingPartitions,
              ApplicationMappingRuntimeDisposition::NotRequested,
              {},
              {},
              std::nullopt,
              std::nullopt,
              std::nullopt,
              std::nullopt,
              {}});
        }
        incrementalMappingObservations.push_back(std::move(observation));
        mapping_debug::emit(
            mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
            mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
              fields["operation"] = "application_resource_time_incremental";
              fields["parent_plan_ordinal"] = selectedPlanOrdinal;
              fields["child_plan_ordinal"] = childOrdinal;
              fields["reopened_root_count"] = reopenedRoots->size();
              fields["mapping_reuse_disposition"] =
                  dse::jointMappingReuseDispositionSpelling(
                      adjacent->reuseDisposition);
              fields["preserved_tech_mappings"] =
                  childSummary.preservedTechMappings;
              fields["preserved_spatial_mappings"] =
                  childSummary.preservedSpatialMappings;
              fields["repaired_tech_mappings"] =
                  childSummary.repairedTechMappings;
              fields["repaired_spatial_mappings"] =
                  childSummary.repairedSpatialMappings;
              fields["wall_time_ns"] =
                  childSummary.incrementalReopenWallTimeNanoseconds;
            });
      }
      execution->summary.selectedPlanOrdinal = selectedPlanOrdinal;
      selectedExecution.emplace(std::move(*execution));
      break;
    }

    execution->summary.selectedPlanOrdinal.reset();
    execution->summary.selectedMapping.reset();
    selectedExecution.emplace(std::move(*execution));
    if (runtime->disposition ==
        ApplicationMappingRuntimeDisposition::CancelledOrTimeout)
      break;
    firstPlan = static_cast<std::size_t>(selectedPlanOrdinal) + 1;
    // A verified Mapping is not an application result.  Runtime validation
    // may reject the selected QoR winner (for example, a functional replay
    // mismatch or an execution timeout).  Continue through the remaining
    // bounded software frontier for every stopping policy; otherwise
    // BoundedQuality would silently turn one failed application-level check
    // into a terminal Mapping failure.
  }
  if (!selectedExecution)
    return invalid("joint Mapping execution produced no bounded outcome");
  const auto qualityRuntimeDisposition =
      [](const std::optional<dse::JointDesignQualityIncompleteReason> &reason) {
        if (!reason)
          return ApplicationMappingRuntimeDisposition::Completed;
        switch (*reason) {
        case dse::JointDesignQualityIncompleteReason::Unsupported:
          return ApplicationMappingRuntimeDisposition::Unsupported;
        case dse::JointDesignQualityIncompleteReason::ProofNotEstablished:
          return ApplicationMappingRuntimeDisposition::ProofNotEstablished;
        case dse::JointDesignQualityIncompleteReason::ExecutionFailed:
          return ApplicationMappingRuntimeDisposition::ExecutionFailed;
        case dse::JointDesignQualityIncompleteReason::CancelledOrTimeout:
          return ApplicationMappingRuntimeDisposition::CancelledOrTimeout;
        }
        llvm_unreachable("unknown application quality disposition");
      };
  for (ApplicationMappingCandidateOutcome &outcome : outcomes) {
    const dse::JointDesignQualityObservation *projected = nullptr;
    std::size_t matchingObservationCount = 0;
    for (const dse::JointDesignQualityObservation &observation :
         selectedExecution->summary.qualityObservations) {
      if (!llvm::is_contained(outcome.systemMappings, observation.candidate))
        continue;
      ++matchingObservationCount;
      if (selectedExecution->summary.selectedPlanOrdinal ==
              outcome.planOrdinal &&
          selectedExecution->summary.selectedMapping == observation.candidate)
        projected = &observation;
      else if (!projected)
        projected = &observation;
    }
    if (!projected ||
        (matchingObservationCount != 1 &&
         !(selectedExecution->summary.selectedPlanOrdinal ==
               outcome.planOrdinal &&
           selectedExecution->summary.selectedMapping ==
               projected->candidate)))
      continue;
    outcome.qualityObjectiveCodes = projected->objectiveCodes;
    if (outcome.runtimeDisposition ==
        ApplicationMappingRuntimeDisposition::NotRequested)
      outcome.runtimeDisposition =
          qualityRuntimeDisposition(projected->incompleteReason);
  }
  selectedExecution->summary.attemptedSoftwarePlans = attemptedSoftwarePlans;
  selectedExecution->summary.hardwareReopenSearches = hardwareReopenSearches;
  selectedExecution->summary.hardwareParentPromotions =
      hardwareParentPromotions;
  selectedExecution->summary.hardwareReopensDeferredByQuality =
      hardwareReopensDeferredByQuality;
  selectedExecution->summary.hardwareReopensWithheldWithoutExactFeedback =
      hardwareReopensWithheldWithoutExactFeedback;
  selectedExecution->summary.hardwareRepairProbeLimit =
      hardwareRepairProbeLimit;
  selectedExecution->summary.hardwareRepairProbesPlanned =
      hardwareRepairProbesPlanned;
  selectedExecution->summary.hardwareRepairProbesReserved =
      hardwareRepairProbesReserved;
  selectedExecution->summary.hardwareRepairProbesConsumed =
      hardwareRepairProbesConsumed;
  selectedExecution->summary.hardwareRepairProbesRejected =
      hardwareRepairProbesRejected;
  selectedExecution->summary.hardwareRepairProbesCancelled =
      hardwareRepairProbesCancelled;
  selectedExecution->summary.spatialMappingRepairCandidateLimit =
      spatialMappingRepairCandidateLimit;
  selectedExecution->summary.spatialMappingRepairsPlanned =
      spatialMappingRepairsPlanned;
  selectedExecution->summary.spatialMappingRepairsReserved =
      spatialMappingRepairsReserved;
  selectedExecution->summary.spatialMappingRepairsConsumed =
      spatialMappingRepairsConsumed;
  selectedExecution->summary.spatialMappingRepairsRejected =
      spatialMappingRepairsRejected;
  selectedExecution->summary.spatialMappingRepairsCancelled =
      spatialMappingRepairsCancelled;
  selectedExecution->summary.parentTechDecisions = parentTechDecisions;
  selectedExecution->summary.parentSpatialDecisions = parentSpatialDecisions;
  selectedExecution->summary.preservedTechDecisions = preservedTechDecisions;
  selectedExecution->summary.preservedSpatialDecisions =
      preservedSpatialDecisions;
  selectedExecution->summary.reopenedTechDecisions = reopenedTechDecisions;
  selectedExecution->summary.reopenedSpatialDecisions =
      reopenedSpatialDecisions;
  selectedExecution->summary.repairedTechDecisions = repairedTechDecisions;
  selectedExecution->summary.repairedSpatialDecisions =
      repairedSpatialDecisions;
  selectedExecution->summary.invalidationRootCount = invalidationRootCount;
  selectedExecution->summary.invalidationConeDecisionCount =
      invalidationConeDecisionCount;
  selectedExecution->summary.parentRouteNodeCount = parentRouteNodeCount;
  selectedExecution->summary.preservedRouteNodeCount = preservedRouteNodeCount;
  selectedExecution->summary.reopenedRouteNodeCount = reopenedRouteNodeCount;
  selectedExecution->summary.repairedRouteNodeCount = repairedRouteNodeCount;
  selectedExecution->summary.parentServiceLegCount = parentServiceLegCount;
  selectedExecution->summary.preservedServiceLegCount =
      preservedServiceLegCount;
  selectedExecution->summary.reopenedServiceLegCount = reopenedServiceLegCount;
  selectedExecution->summary.verifiedAlternatives = verifiedAlternatives;
  selectedExecution->summary.techMappingDispatchCount = techMappingDispatches;
  selectedExecution->summary.spatialPnrDispatchCount = spatialPnrDispatches;
  selectedExecution->summary.systemPnrDispatchCount = systemPnrDispatches;
  selectedExecution->summary.attempts = std::move(attempts);
  ApplicationMappingProvenance provenance;
  provenance.sourceProgram = prepared.preMappingSourceProgram;
  provenance.fabric = prepared.preMappingFabric;
  provenance.workload = prepared.preMappingWorkload;
  provenance.runtimeInput = prepared.preMappingRuntimeInput;
  provenance.frontierPolicyDigest = prepared.preMappingFrontierPolicyDigest;
  provenance.resourceTimeFunnelAccounting =
      prepared.resourceTimeFunnel.accounting;
  provenance.resourceTimeFunnelTruncated =
      prepared.resourceTimeFunnel.truncated;
  provenance.resourceTimeFunnelIncompleteReason =
      prepared.resourceTimeFunnel.incompleteReason;
  provenance.preMappingCompleteness = prepared.preMappingCompleteness;
  provenance.requestedPlannerMode = prepared.preMappingRequestedPlannerMode;
  provenance.resolvedPlannerMode = prepared.preMappingResolvedPlannerMode;
  provenance.incrementalMappingObservations =
      std::move(incrementalMappingObservations);
  provenance.pairDecision = deriveApplicationPairDecision(
      prepared, outcomes, selectedExecution->summary, qualityInvocations);
  ApplicationMappingExecution result{std::move(*selectedExecution),
                                     std::move(outcomes),
                                     std::move(provenance)};
  emitApplicationMappingDiagnostics(result);
  return result;
}

} // namespace loom::application

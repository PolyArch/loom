#include "Application/BuildDiagnostics.h"
#include "ApplicationRuntimeValidationInternal.h"
#include "BuildInternal.h"
#include "QualityInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/MappingDebugLog.h"
#include "DSE/JointHardwareReopen.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "PnR/PnrDerivedContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <chrono>
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
  // Mapping execution strict-imports the same Fabric roots once per
  // alternative, per repair and per verification pass. One session scoped to
  // the execution turns those repeats into cache hits; ReuseEnclosing keeps an
  // outer session, such as an Application package, as the single owner.
  fabric::FabricArtifactImportSession fabricImportSession;
  llvm::scope_exit emitFabricImportStatistics([&] {
    fabric::emitFabricArtifactImportSessionStatistics(
        fabric::FabricArtifactImportVerificationDomain::SourceInvocation,
        InvocationDiagnosticStage::SystemPnr, fabricImportSession.statistics());
  });
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
  std::vector<ArtifactRootReference> invocationSemanticInputs;
  if (request.boundedQuality)
    invocationSemanticInputs = request.boundedQuality->semanticInputs;
  invocationSemanticInputs.insert(
      invocationSemanticInputs.end(),
      {prepared.preMappingSourceProgram, prepared.preMappingFabric,
       prepared.preMappingWorkload, prepared.preMappingRuntimeInput});
  for (const PreparedApplicationSoftware &software : prepared.software)
    for (const sim::SourceBackedDfgReplayCaseReference &replay :
         software.replayCases)
      invocationSemanticInputs.insert(invocationSemanticInputs.end(),
                                      {replay.workload, replay.runtimeInput});
  llvm::sort(invocationSemanticInputs, artifactRootReferenceLess);
  invocationSemanticInputs.erase(std::unique(invocationSemanticInputs.begin(),
                                             invocationSemanticInputs.end()),
                                 invocationSemanticInputs.end());
  std::vector<const dse::JointDesignExplorationPlan *> plans;
  plans.reserve(prepared.mappingAlternatives.size());
  for (const PreparedApplicationMappingAlternative &alternative :
       prepared.mappingAlternatives)
    plans.push_back(&alternative.plan);
  std::vector<ApplicationMappingCandidateOutcome> outcomes;
  std::vector<ArtifactRootReference> hardwareMutationRepairRecords;
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
  std::uint64_t sharedHardwareRepairProbesReserved = 0;
  std::uint64_t spatialMappingRepairCandidateLimit = 0;
  std::uint64_t spatialMappingRepairsPlanned = 0;
  std::uint64_t spatialMappingRepairsReserved = 0;
  std::uint64_t spatialMappingRepairsConsumed = 0;
  std::uint64_t spatialMappingRepairsRejected = 0;
  std::uint64_t spatialMappingRepairsCancelled = 0;
  const auto addSaturated = [](std::uint64_t &total, std::uint64_t value) {
    total = value > std::numeric_limits<std::uint64_t>::max() - total
                ? std::numeric_limits<std::uint64_t>::max()
                : total + value;
  };
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
  std::uint64_t techMappingInvocations = 0;
  std::uint64_t spatialPnrInvocations = 0;
  std::uint64_t systemPnrInvocations = 0;
  std::uint64_t techMappingDispatches = 0;
  std::uint64_t spatialPnrDispatches = 0;
  std::uint64_t systemPnrDispatches = 0;
  std::uint64_t techMappingJournalReplays = 0;
  std::uint64_t spatialPnrJournalReplays = 0;
  std::uint64_t systemPnrJournalReplays = 0;
  const auto providerWork =
      [](const dse::JointDesignExecutionSummary &summary) {
        return ApplicationMappingProviderWorkObservation{
            summary.techMappingInvocationCount,
            summary.spatialPnrInvocationCount,
            summary.systemPnrInvocationCount,
            summary.techMappingDispatchCount,
            summary.spatialPnrDispatchCount,
            summary.systemPnrDispatchCount,
            summary.techMappingJournalReplayCount,
            summary.spatialPnrJournalReplayCount,
            summary.systemPnrJournalReplayCount};
      };
  const auto accumulateProviderWork =
      [&](const dse::JointDesignExecutionSummary &summary) {
        techMappingInvocations += summary.techMappingInvocationCount;
        spatialPnrInvocations += summary.spatialPnrInvocationCount;
        systemPnrInvocations += summary.systemPnrInvocationCount;
        techMappingDispatches += summary.techMappingDispatchCount;
        spatialPnrDispatches += summary.spatialPnrDispatchCount;
        systemPnrDispatches += summary.systemPnrDispatchCount;
        techMappingJournalReplays += summary.techMappingJournalReplayCount;
        spatialPnrJournalReplays += summary.spatialPnrJournalReplayCount;
        systemPnrJournalReplays += summary.systemPnrJournalReplayCount;
      };
  std::vector<ApplicationIncrementalMappingObservation>
      incrementalMappingObservations;
  std::optional<ApplicationResourceTimeMappingPath> resourceTimeMappingPath;
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

  const auto alternativeScheduleHintDigests =
      [](const PreparedApplicationMappingAlternative &alternative) {
        std::vector<ComponentViewDigest> result =
            alternative.equivalentScheduleHintDigests;
        if (!llvm::is_contained(result,
                                alternative.resourceTimeScheduleHintDigest))
          result.push_back(alternative.resourceTimeScheduleHintDigest);
        return result;
      };

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
      for (const ComponentViewDigest &scheduleHintDigest :
           alternativeScheduleHintDigests(alternative)) {
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
            {},
            std::nullopt,
            std::nullopt});
      }
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
        detail::rebaseApplicationBoundedQualityPolicy(request.boundedQuality,
                                                      firstPlan);
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
    reopenRequest.invocationSemanticInputs = invocationSemanticInputs;
    return dse::executeJointDesignWithHardwareReopen(
        tail, prepared.jointPolicy, std::move(reopenRequest), artifacts, blobs);
  };

  const auto remainingSharedHardwareRepairProbes = [&]() -> std::uint64_t {
    if (!request.boundedQuality)
      return std::numeric_limits<std::uint64_t>::max();
    if (sharedHardwareRepairProbesReserved >=
        request.boundedQuality->maximumHardwareRepairProbes)
      return 0;
    return request.boundedQuality->maximumHardwareRepairProbes -
           sharedHardwareRepairProbesReserved;
  };
  const auto reserveSharedHardwareRepairProbes =
      [&](std::uint64_t reserved) -> llvm::Error {
    if (!request.boundedQuality)
      return llvm::Error::success();
    if (reserved > remainingSharedHardwareRepairProbes())
      return invalid("runtime repair exceeded its shared hardware probe "
                     "limit");
    sharedHardwareRepairProbesReserved += reserved;
    return llvm::Error::success();
  };
  const auto makeRepairRequest =
      [&](std::string journalRoot,
          std::uint64_t planOrdinal) -> dse::JointHardwareReopenRequest {
    std::optional<dse::JointBoundedQualityPolicy> boundedQuality =
        detail::rebaseApplicationBoundedQualityPolicy(request.boundedQuality,
                                                      planOrdinal);
    dse::JointHardwareReopenRequest repairRequest{
        request.producer,
        std::move(journalRoot),
        evidence,
        boundedQuality ? dse::JointDesignStoppingPolicy::BoundedQuality
                       : dse::JointDesignStoppingPolicy::FirstVerified,
        std::move(boundedQuality),
        std::nullopt,
        request.siteCapacity,
        request.executionPolicy};
    repairRequest.hardwareExplorationScope = request.hardwareExplorationScope;
    if (request.mappingRepairCandidateLimit)
      repairRequest.maximumMappingRepairCandidates =
          *request.mappingRepairCandidateLimit;
    repairRequest.invocationSemanticInputs = invocationSemanticInputs;
    return repairRequest;
  };

  const auto resolveRuntime =
      [&](const PreparedApplicationMappingAlternative &alternative,
          const dse::JointDesignExecution &execution,
          const ArtifactRootReference &mapping)
      -> llvm::Expected<detail::ApplicationRuntimeValidation> {
    if (request.boundedQuality &&
        request.boundedQuality->provenanceDomain ==
            dse::JointDesignQualityProvenanceDomain::ApplicationRuntime)
      return detail::projectApplicationQualityRuntime(
          execution, mapping, *request.boundedQuality, artifacts);
    return detail::validateApplicationMappingRuntime(
        prepared, alternative, execution, request.executionPolicy, artifacts,
        blobs);
  };

  std::optional<dse::JointDesignExecution> selectedExecution;
  std::size_t firstPlan = 0;
  while (firstPlan < plans.size()) {
    const auto mappingStart = std::chrono::steady_clock::now();
    auto execution = executeTail(firstPlan);
    if (!execution)
      return execution.takeError();
    if (llvm::Error error = detail::recordApplicationQualityInvocation(
            *execution, firstPlan, qualityInvocations))
      return std::move(error);
    if (llvm::Error error = appendOutcomes(*execution, firstPlan))
      return std::move(error);
    attemptedSoftwarePlans += execution->summary.attemptedSoftwarePlans;
    hardwareReopenSearches += execution->summary.hardwareReopenSearches;
    hardwareParentPromotions += execution->summary.hardwareParentPromotions;
    hardwareReopensDeferredByQuality +=
        execution->summary.hardwareReopensDeferredByQuality;
    hardwareReopensWithheldWithoutExactFeedback +=
        execution->summary.hardwareReopensWithheldWithoutExactFeedback;
    addSaturated(hardwareRepairProbeLimit,
                 execution->summary.hardwareRepairProbeLimit);
    addSaturated(hardwareRepairProbesPlanned,
                 execution->summary.hardwareRepairProbesPlanned);
    addSaturated(hardwareRepairProbesReserved,
                 execution->summary.hardwareRepairProbesReserved);
    addSaturated(hardwareRepairProbesConsumed,
                 execution->summary.hardwareRepairProbesConsumed);
    addSaturated(hardwareRepairProbesRejected,
                 execution->summary.hardwareRepairProbesRejected);
    addSaturated(hardwareRepairProbesCancelled,
                 execution->summary.hardwareRepairProbesCancelled);
    addSaturated(spatialMappingRepairCandidateLimit,
                 execution->summary.spatialMappingRepairCandidateLimit);
    addSaturated(spatialMappingRepairsPlanned,
                 execution->summary.spatialMappingRepairsPlanned);
    addSaturated(spatialMappingRepairsReserved,
                 execution->summary.spatialMappingRepairsReserved);
    addSaturated(spatialMappingRepairsConsumed,
                 execution->summary.spatialMappingRepairsConsumed);
    addSaturated(spatialMappingRepairsRejected,
                 execution->summary.spatialMappingRepairsRejected);
    addSaturated(spatialMappingRepairsCancelled,
                 execution->summary.spatialMappingRepairsCancelled);
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
    accumulateProviderWork(execution->summary);

    std::optional<std::uint64_t> mappingPlanOrdinal =
        execution->summary.selectedPlanOrdinal;
    std::optional<ArtifactRootReference> runtimeMapping =
        execution->summary.selectedMapping;
    if (mappingPlanOrdinal.has_value() != runtimeMapping.has_value())
      return invalid("joint Mapping selection has an incomplete exact join");
    if (!runtimeMapping && request.boundedQuality &&
        execution->summary.qualityIncompleteCandidate) {
      runtimeMapping = execution->summary.qualityIncompleteCandidate;
      for (const dse::JointDesignAttemptRecord &attempt :
           execution->summary.attempts) {
        if (!llvm::is_contained(attempt.systemMappings, *runtimeMapping))
          continue;
        if (mappingPlanOrdinal && *mappingPlanOrdinal != attempt.planOrdinal)
          return invalid("quality-incomplete Mapping has conflicting plan "
                         "owners");
        mappingPlanOrdinal = attempt.planOrdinal;
      }
      if (!mappingPlanOrdinal)
        return invalid("quality-incomplete Mapping has no exact plan owner");
    }
    if (!mappingPlanOrdinal || !runtimeMapping) {
      selectedExecution.emplace(std::move(*execution));
      break;
    }
    if (*mappingPlanOrdinal >
        std::numeric_limits<std::uint64_t>::max() - firstPlan)
      return invalid("selected Mapping plan ordinal overflowed");
    const std::uint64_t selectedPlanOrdinal = *mappingPlanOrdinal + firstPlan;
    if (selectedPlanOrdinal >= prepared.mappingAlternatives.size())
      return invalid("selected Mapping has a foreign plan ordinal");
    auto runtime =
        resolveRuntime(prepared.mappingAlternatives[selectedPlanOrdinal],
                       *execution, *runtimeMapping);
    if (!runtime)
      return runtime.takeError();
    // The parent's own Mapping and runtime validation wall time, measured
    // from the joint execution start through the replay, is the cost estimate
    // of one hardware child; the runtime witness repair reserves it per
    // admitted child before Mapping repair may dispatch.
    const std::uint64_t parentCostNanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - mappingStart)
            .count());
    bool joined = false;
    for (ApplicationMappingCandidateOutcome &outcome : outcomes) {
      if (outcome.planOrdinal != selectedPlanOrdinal ||
          !llvm::is_contained(outcome.systemMappings, *runtimeMapping))
        continue;
      outcome.runtimeDisposition = runtime->disposition;
      outcome.runtimeEvidence = runtime->evidence;
      outcome.runtimeMemoryContractRefusal = runtime->cgraMemoryContractRefusal;
      outcome.oracleEvidence = runtime->oracleEvidence;
      outcome.dfgCycles = runtime->dfgCycles;
      outcome.cgraCycles = runtime->cgraCycles;
      outcome.resourceCoreCost = runtime->resourceCoreCost;
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
        execution->summary.selectedMapping &&
        !outcomeMatchesRequestedSpectrum(selectedPlanOrdinal,
                                         *runtimeMapping)) {
      execution->summary.selectedPlanOrdinal.reset();
      execution->summary.selectedMapping.reset();
      selectedExecution.emplace(std::move(*execution));
      firstPlan = static_cast<std::size_t>(selectedPlanOrdinal) + 1;
      continue;
    }
    std::vector<ArtifactRootReference> repairSystems;
    std::vector<std::optional<ArtifactRootReference>> repairRecords;
    std::vector<dse::JointDesignExecution> repairExecutions;
    if (runtime->disposition !=
        ApplicationMappingRuntimeDisposition::Completed) {
      // One witness set, two separately budgeted repair families: Mapping
      // repair spends its own candidates inside the window that remains after
      // the hardware reservation; hardware children spend only the shared
      // probe ledger of this invocation.
      llvm::SmallString<256> witnessJournal(request.journalRoot);
      llvm::sys::path::append(witnessJournal,
                              "runtime-witness-repair-" +
                                  std::to_string(selectedPlanOrdinal));
      std::optional<std::uint64_t> remainingHardwareProbes;
      if (request.boundedQuality)
        remainingHardwareProbes = remainingSharedHardwareRepairProbes();
      auto repaired = dse::executeJointRuntimeWitnessRepair(
          prepared.mappingAlternatives[selectedPlanOrdinal].plan, *execution,
          prepared.jointPolicy,
          {runtime->spatialTransportFeedback, runtime->spatialFifoFeedback,
           runtime->spatialOperandQueueFeedback},
          parentCostNanoseconds, remainingHardwareProbes,
          makeRepairRequest(witnessJournal.str().str(), selectedPlanOrdinal),
          artifacts, blobs);
      if (!repaired)
        return repaired.takeError();
      const dse::JointRepairWorkLedger &mappingRepair =
          repaired->mappingRepairLedger;
      addSaturated(spatialMappingRepairCandidateLimit,
                   mappingRepair.candidateLimit);
      addSaturated(spatialMappingRepairsPlanned, mappingRepair.planned);
      addSaturated(spatialMappingRepairsReserved, mappingRepair.reserved);
      addSaturated(spatialMappingRepairsConsumed, mappingRepair.consumed);
      addSaturated(spatialMappingRepairsRejected, mappingRepair.rejected);
      addSaturated(spatialMappingRepairsCancelled, mappingRepair.cancelled);
      const dse::JointRepairWorkLedger &hardwareReopen =
          repaired->hardwareReopenLedger;
      addSaturated(hardwareRepairProbeLimit, hardwareReopen.candidateLimit);
      addSaturated(hardwareRepairProbesPlanned, hardwareReopen.planned);
      addSaturated(hardwareRepairProbesReserved, hardwareReopen.reserved);
      addSaturated(hardwareRepairProbesConsumed, hardwareReopen.consumed);
      addSaturated(hardwareRepairProbesRejected, hardwareReopen.rejected);
      addSaturated(hardwareRepairProbesCancelled, hardwareReopen.cancelled);
      if (llvm::Error error =
              reserveSharedHardwareRepairProbes(hardwareReopen.reserved))
        return std::move(error);
      repairSystems = std::move(repaired->childSystems);
      repairRecords = std::move(repaired->hardwareMutationRepairRecords);
      repairExecutions = std::move(repaired->executions);
      if (repairSystems.size() != repairRecords.size() ||
          repairSystems.size() != repairExecutions.size())
        return invalid("runtime witness repair lost aligned child lineage");
      for (const auto &record : repairRecords)
        if (record &&
            !llvm::is_contained(hardwareMutationRepairRecords, *record))
          hardwareMutationRepairRecords.push_back(*record);
    }
    if (!repairExecutions.empty()) {
      auto qualityChoice = detail::chooseApplicationRepairByQuality(
          repairExecutions, request.boundedQuality, artifacts);
      if (!qualityChoice)
        return qualityChoice.takeError();
      const auto *qualitySelection =
          std::get_if<dse::JointRepairQualitySelection>(&*qualityChoice);
      const auto *qualityIncomplete =
          std::get_if<dse::JointRepairQualityIncomplete>(&*qualityChoice);
      std::vector<std::vector<ArtifactRootReference>> repairMappings;
      repairMappings.reserve(repairExecutions.size());
      const auto appendRepairOutcome =
          [&](std::size_t childOrdinal,
              const std::vector<ArtifactRootReference> &childMappings,
              const detail::ApplicationRuntimeValidation &childRuntime)
          -> llvm::Error {
        const PreparedApplicationMappingAlternative &alternative =
            prepared.mappingAlternatives[selectedPlanOrdinal];
        for (const ComponentViewDigest &scheduleHintDigest :
             alternativeScheduleHintDigests(alternative)) {
          auto childSpectrum = verifyResourceTimeAlternative(
              prepared.resourceTimeFunnel, alternative, childMappings,
              artifacts, blobs, scheduleHintDigest, {},
              request.executionControl);
          if (!childSpectrum)
            return childSpectrum.takeError();
          outcomes.push_back(ApplicationMappingCandidateOutcome{
              alternative.preMappingCandidateRecordOrdinal,
              selectedPlanOrdinal,
              scheduleHintDigest,
              alternative.dataflow,
              repairSystems[childOrdinal],
              dse::JointDesignAttemptDisposition::Verified,
              std::nullopt,
              std::nullopt,
              childMappings,
              prepared.candidateInventory
                  [alternative.preMappingCandidateRecordOrdinal],
              alternative.plan.systemBindingPartitions,
              childRuntime.disposition,
              childRuntime.evidence,
              {},
              std::move(*childSpectrum),
              childRuntime.dfgCycles,
              childRuntime.cgraCycles,
              childRuntime.resourceCoreCost,
              childRuntime.oracleEvidence,
              childRuntime.cgraMemoryContractRefusal,
              repairRecords[childOrdinal]});
        }
        return llvm::Error::success();
      };
      for (std::size_t childOrdinal = 0;
           childOrdinal != repairExecutions.size(); ++childOrdinal) {
        dse::JointDesignExecution &childExecution =
            repairExecutions[childOrdinal];
        if (llvm::Error error = detail::recordApplicationQualityInvocation(
                childExecution, selectedPlanOrdinal, qualityInvocations))
          return std::move(error);
        accumulateProviderWork(childExecution.summary);
        std::vector<ArtifactRootReference> childMappings;
        for (const dse::JointMappedPair &pair : childExecution.mappedPairs)
          childMappings.insert(childMappings.end(), pair.systemMappings.begin(),
                               pair.systemMappings.end());
        llvm::sort(childMappings, artifactRootReferenceLess);
        childMappings.erase(
            std::unique(childMappings.begin(), childMappings.end()),
            childMappings.end());
        repairMappings.push_back(childMappings);
        attempts.push_back({selectedPlanOrdinal, repairSystems[childOrdinal],
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
        if (selectedExecution || qualityIncomplete ||
            (qualitySelection &&
             qualitySelection->executionOrdinal != childOrdinal) ||
            childMappings.empty() || !childExecution.summary.selectedMapping)
          continue;
        auto childRuntime = resolveRuntime(
            prepared.mappingAlternatives[selectedPlanOrdinal], childExecution,
            *childExecution.summary.selectedMapping);
        if (!childRuntime)
          return childRuntime.takeError();
        if (llvm::Error error =
                appendRepairOutcome(childOrdinal, childMappings, *childRuntime))
          return std::move(error);
        if (childRuntime->disposition ==
                ApplicationMappingRuntimeDisposition::Completed &&
            outcomeMatchesRequestedSpectrum(
                selectedPlanOrdinal,
                childExecution.summary.selectedMapping.value())) {
          childExecution.summary.selectedPlanOrdinal = selectedPlanOrdinal;
          selectedExecution.emplace(std::move(childExecution));
        }
      }
      if (selectedExecution && selectedExecution->summary.selectedPlanOrdinal)
        break;
      if (qualityIncomplete) {
        if (qualityIncomplete->executionOrdinal >= repairExecutions.size())
          return invalid("repair quality incomplete result has a foreign "
                         "execution owner");
        dse::JointDesignExecution &owner =
            repairExecutions[qualityIncomplete->executionOrdinal];
        if (owner.summary.selectedMapping ||
            owner.summary.qualityIncompleteCandidate !=
                qualityIncomplete->incomplete.candidate)
          return invalid("repair quality incomplete result lost its exact "
                         "summary");
        if (qualityIncomplete->incomplete.candidate) {
          const std::vector<ArtifactRootReference> &ownerMappings =
              repairMappings[qualityIncomplete->executionOrdinal];
          if (!llvm::is_contained(ownerMappings,
                                  *qualityIncomplete->incomplete.candidate))
            return invalid("repair quality incomplete candidate is outside "
                           "its execution owner");
          detail::ApplicationRuntimeValidation projected;
          projected.disposition =
              ApplicationMappingRuntimeDisposition::NotRequested;
          if (request.boundedQuality &&
              request.boundedQuality->provenanceDomain ==
                  dse::JointDesignQualityProvenanceDomain::ApplicationRuntime) {
            auto runtime = detail::projectApplicationQualityRuntime(
                owner, *qualityIncomplete->incomplete.candidate,
                *request.boundedQuality, artifacts);
            if (!runtime)
              return runtime.takeError();
            projected = std::move(*runtime);
          }
          if (llvm::Error error =
                  appendRepairOutcome(qualityIncomplete->executionOrdinal,
                                      ownerMappings, projected))
            return std::move(error);
        }
        selectedExecution.emplace(std::move(owner));
        break;
      }
    }
    if (runtime->disposition ==
            ApplicationMappingRuntimeDisposition::Completed &&
        execution->summary.selectedMapping) {
      // Exercise every already-retained resource-time state for this exact
      // Dataflow through the application owner while one verified parent
      // Mapping is live. Each retained plan is attempted at most once. A
      // child becomes the next parent only after its independent cold and
      // incremental checks, runtime validation, and Spectrum verification all
      // close, so the resulting path is a finite chain of actual Mapping
      // executions rather than a relabeled selected-parent star.
      std::vector<std::size_t> transitionPlanOrdinals;
      transitionPlanOrdinals.reserve(prepared.mappingAlternatives.size() - 1);
      for (std::size_t offset = 1;
           offset != prepared.mappingAlternatives.size(); ++offset)
        transitionPlanOrdinals.push_back(
            (static_cast<std::size_t>(selectedPlanOrdinal) + offset) %
            prepared.mappingAlternatives.size());
      std::optional<dse::JointDesignExecution> chainedParentExecution;
      const PreparedApplicationMappingAlternative &scheduleOwner =
          prepared.mappingAlternatives[selectedPlanOrdinal];
      std::optional<dse::JointDesignExplorationPlan> chainedParentPlan;
      const dse::JointDesignExplorationPlan *parentPlan = &scheduleOwner.plan;
      const dse::JointDesignExecution *parentExecution = &*execution;
      std::size_t parentPlanOrdinal = selectedPlanOrdinal;
      ApplicationResourceTimeMappingPath candidatePath{
          selectedPlanOrdinal,
          scheduleOwner.resourceTimeScheduleHintDigest,
          {}};
      std::vector<ArtifactRootReference> pathMappings = {
          *execution->summary.selectedMapping};
      for (const std::size_t childOrdinal : transitionPlanOrdinals) {
        const PreparedApplicationMappingAlternative &childAlternative =
            prepared.mappingAlternatives[childOrdinal];
        if (childAlternative.candidateIdentity !=
                scheduleOwner.candidateIdentity ||
            childAlternative.dataflow != scheduleOwner.dataflow ||
            childAlternative.plan.pairOutputs.size() != 1 ||
            scheduleOwner.plan.pairOutputs.size() != 1 ||
            childAlternative.plan.pairOutputs.front().pair.system !=
                scheduleOwner.plan.pairOutputs.front().pair.system)
          continue;
        auto reopenedRoots =
            deriveApplicationPartitionDelta(*parentPlan, childAlternative.plan);
        if (!reopenedRoots)
          return reopenedRoots.takeError();
        if (reopenedRoots->empty())
          continue;
        llvm::SmallString<256> adjacentJournal(request.journalRoot);
        llvm::sys::path::append(adjacentJournal,
                                "application-resource-time-adjacent-" +
                                    std::to_string(parentPlanOrdinal) + "-" +
                                    std::to_string(childOrdinal));
        dse::JointHardwareReopenRequest adjacentRequest =
            makeRepairRequest(adjacentJournal.str().str(), childOrdinal);
        adjacentRequest.spectrumEndpoint =
            prepared.resourceTimePolicy.spectrumEndpoint;
        auto adjacent = dse::executeResourceTimeAdjacentMappingRepair(
            *parentPlan, *parentExecution, prepared.jointPolicy,
            childAlternative.plan.systemBindingPartitions, *reopenedRoots,
            std::move(adjacentRequest), artifacts, blobs);
        if (!adjacent)
          return adjacent.takeError();
        if (llvm::Error error = detail::recordApplicationQualityInvocation(
                adjacent->coldExecution, childOrdinal, qualityInvocations))
          return std::move(error);
        if (llvm::Error error = detail::recordApplicationQualityInvocation(
                adjacent->execution, childOrdinal, qualityInvocations))
          return std::move(error);
        const dse::JointDesignExecutionSummary &childSummary =
            adjacent->execution.summary;
        accumulateProviderWork(adjacent->coldExecution.summary);
        accumulateProviderWork(childSummary);
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
            *parentExecution->summary.selectedMapping,
            childAlternative.plan.pairOutputs.front().pair.system,
            childSummary.selectedMapping,
            adjacent->coldMapping,
            static_cast<std::uint64_t>(parentPlanOrdinal),
            static_cast<std::uint64_t>(childOrdinal),
            prepared.mappingAlternatives[parentPlanOrdinal]
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
            providerWork(adjacent->coldExecution.summary),
            providerWork(childSummary),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            false};
        if (childHasMapping) {
          adjacent->execution.summary.selectedPlanOrdinal = childOrdinal;
          auto childRuntime =
              resolveRuntime(childAlternative, adjacent->execution,
                             *adjacent->execution.summary.selectedMapping);
          if (!childRuntime)
            return childRuntime.takeError();
          bool incrementalSpectrumVerified = false;
          for (const ComponentViewDigest &scheduleHintDigest :
               alternativeScheduleHintDigests(childAlternative)) {
            auto childSpectrum = verifyResourceTimeAlternative(
                prepared.resourceTimeFunnel, childAlternative, childMappings,
                artifacts, blobs, scheduleHintDigest, {},
                request.executionControl);
            if (!childSpectrum)
              return childSpectrum.takeError();
            if (scheduleHintDigest ==
                childAlternative.resourceTimeScheduleHintDigest)
              incrementalSpectrumVerified =
                  childSpectrum->has_value() &&
                  std::holds_alternative<dse::VerifiedResourceTimeSpectrum>(
                      (*childSpectrum)->verification);
            outcomes.push_back(ApplicationMappingCandidateOutcome{
                childAlternative.preMappingCandidateRecordOrdinal,
                childOrdinal,
                scheduleHintDigest,
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
                childRuntime->oracleEvidence,
                childRuntime->cgraMemoryContractRefusal,
                std::nullopt});
          }
          bool coldVerified = false;
          if (adjacent->coldMapping) {
            adjacent->coldExecution.summary.selectedPlanOrdinal = childOrdinal;
            auto coldRuntime = resolveRuntime(
                childAlternative, adjacent->coldExecution,
                *adjacent->coldExecution.summary.selectedMapping);
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
              incrementalSpectrumVerified;
        } else {
          for (const ComponentViewDigest &scheduleHintDigest :
               alternativeScheduleHintDigests(childAlternative))
            outcomes.push_back(ApplicationMappingCandidateOutcome{
                childAlternative.preMappingCandidateRecordOrdinal,
                childOrdinal,
                scheduleHintDigest,
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
                {},
                std::nullopt,
                std::nullopt});
        }
        const std::uint64_t observationOrdinal =
            incrementalMappingObservations.size();
        const bool promoteChild =
            observation.verified && observation.childMapping &&
            *observation.childMapping != observation.parentMapping &&
            !llvm::is_contained(pathMappings, *observation.childMapping);
        incrementalMappingObservations.push_back(std::move(observation));
        mapping_debug::emit(
            mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
            mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
              fields["operation"] = "application_resource_time_incremental";
              fields["parent_plan_ordinal"] = parentPlanOrdinal;
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
        if (promoteChild) {
          candidatePath.observationOrdinals.push_back(observationOrdinal);
          pathMappings.push_back(*observation.childMapping);
          chainedParentPlan.emplace(std::move(adjacent->plan));
          chainedParentExecution.emplace(std::move(adjacent->execution));
          parentPlan = &*chainedParentPlan;
          parentExecution = &*chainedParentExecution;
          parentPlanOrdinal = childOrdinal;
        }
      }
      if (!candidatePath.observationOrdinals.empty())
        resourceTimeMappingPath.emplace(std::move(candidatePath));
      execution->summary.selectedPlanOrdinal = selectedPlanOrdinal;
      selectedExecution.emplace(std::move(*execution));
      break;
    }

    execution->summary.selectedPlanOrdinal.reset();
    execution->summary.selectedMapping.reset();
    selectedExecution.emplace(std::move(*execution));
    if (request.boundedQuality ||
        runtime->disposition ==
            ApplicationMappingRuntimeDisposition::CancelledOrTimeout)
      break;
    firstPlan = static_cast<std::size_t>(selectedPlanOrdinal) + 1;
    // A verified Mapping is not an application result.  Runtime validation
    // may reject the selected QoR winner (for example, a functional replay
    // mismatch or an execution timeout).  Continue through the remaining
    // FirstVerified frontier. BoundedQuality already assessed the complete
    // bounded frontier and retains its typed incomplete observation.
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
           selectedExecution->summary.selectedMapping == projected->candidate)))
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
  selectedExecution->summary.techMappingInvocationCount =
      techMappingInvocations;
  selectedExecution->summary.spatialPnrInvocationCount = spatialPnrInvocations;
  selectedExecution->summary.systemPnrInvocationCount = systemPnrInvocations;
  selectedExecution->summary.techMappingDispatchCount = techMappingDispatches;
  selectedExecution->summary.spatialPnrDispatchCount = spatialPnrDispatches;
  selectedExecution->summary.systemPnrDispatchCount = systemPnrDispatches;
  selectedExecution->summary.techMappingJournalReplayCount =
      techMappingJournalReplays;
  selectedExecution->summary.spatialPnrJournalReplayCount =
      spatialPnrJournalReplays;
  selectedExecution->summary.systemPnrJournalReplayCount =
      systemPnrJournalReplays;
  selectedExecution->summary.attempts = std::move(attempts);
  if (!selectedExecution->summary.selectedMapping)
    selectedExecution->summary.declaredWorkExhausted |=
        firstPlan >= plans.size();
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
  llvm::sort(hardwareMutationRepairRecords, artifactRootReferenceLess);
  hardwareMutationRepairRecords.erase(
      std::unique(hardwareMutationRepairRecords.begin(),
                  hardwareMutationRepairRecords.end()),
      hardwareMutationRepairRecords.end());
  provenance.hardwareMutationRepairRecords =
      std::move(hardwareMutationRepairRecords);
  provenance.incrementalMappingObservations =
      std::move(incrementalMappingObservations);
  provenance.resourceTimeMappingPath = std::move(resourceTimeMappingPath);
  provenance.pairDecision = deriveApplicationPairDecision(
      prepared, outcomes, *selectedExecution, qualityInvocations);
  ApplicationMappingExecution result{std::move(*selectedExecution),
                                     std::move(outcomes),
                                     std::move(provenance)};
  emitApplicationMappingDiagnostics(result);
  return result;
}

} // namespace loom::application

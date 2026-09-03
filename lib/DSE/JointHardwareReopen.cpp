#include "DSE/JointHardwareReopen.h"

#include "JointHardwareReopenInternal.h"

#include "JointHardwareReopenExecution.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/MappingDebugLog.h"
#include "DSE/ExecutionJournal.h"
#include "DSE/HardwareDecision.h"
#include "DSE/ProductionOwners.h"
#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "Evaluation/Evidence.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "PnR/PnrDerivedContext.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse {

using namespace joint_reopen_detail;

llvm::Expected<std::vector<SpatialMicroarchitectureDecisionDomain>>
deriveSpatialCapacityHardwareReopenDomains(
    const pnr::SpatialFifoCapacityShortfall &feedback) {
  if (feedback.logicalNets.empty() || feedback.routeAnchors.empty())
    return invalid("static FIFO capacity feedback is incomplete or outside "
                   "the hardware depth domain");
  auto domain = deriveFifoCapacityDepthDomain(
      feedback.owner, feedback.selectedCapacity, feedback.minimumLegalCapacity);
  if (!domain)
    return domain.takeError();
  return std::vector<SpatialMicroarchitectureDecisionDomain>{
      std::move(*domain)};
}

llvm::Expected<JointDesignExecution> executeJointDesignWithHardwareReopen(
    llvm::ArrayRef<const JointDesignExplorationPlan *> plans,
    const JointDesignPolicy &policy, JointHardwareReopenRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (llvm::Error error = registerProductionDseOwners())
    return std::move(error);
  if (request.journalRoot.empty())
    return invalid("hardware reopen requires a journal root");
  if (plans.empty())
    return invalid("hardware reopen requires at least one Mapping plan");
  auto scheduler = SiteScheduler::create(std::move(request.siteCapacity));
  if (!scheduler)
    return scheduler.takeError();
  loom::pnr::PnrDerivedContextSession derivedContextSession;
  struct FailedSoftwareAttempt final {
    std::uint64_t planOrdinal = 0;
    const JointDesignExplorationPlan *plan = nullptr;
    JointSoftwareCoverage coverage;
    JointDesignExecution execution;
    /// Exact Tech Hall pressure is ranking provenance for bounded hardware
    /// parent promotion. It never changes the typed feedback disposition or
    /// proves a child Mapping legal.
    std::uint64_t techHallDeficit = 0;
  };
  struct VerifiedAlternative final {
    std::uint64_t planOrdinal = 0;
    JointDesignExecution execution;
  };
  std::vector<FailedSoftwareAttempt> failedSoftwareAttempts;
  failedSoftwareAttempts.reserve(plans.size());
  std::vector<VerifiedAlternative> verifiedAlternatives;
  verifiedAlternatives.reserve(plans.size());
  std::optional<JointDesignExecution> firstIncomplete;
  std::optional<JointDesignExecution> lastNoFeasible;
  std::vector<JointDesignInvocationManifestReference> encounteredInvocations;
  std::uint64_t attemptedSoftwarePlans = 0;
  std::uint64_t hardwareReopenSearches = 0;
  std::uint64_t hardwareParentPromotions = 0;
  std::uint64_t hardwareReopensDeferredByQuality = 0;
  std::uint64_t hardwareReopensWithheldWithoutExactFeedback = 0;
  dse::JointDesignExecutionSummary accounting;
  std::uint64_t verifiedMappingCount = 0;
  const auto executionStart = std::chrono::steady_clock::now();
  std::optional<std::uint64_t> timeToFirstFeasible;
  bool boundedQualitySearchIncomplete = false;
  bool deadlineObserved = dispatchDeadlineReached(request.executionPolicy);
  const auto saturatingAdd = [](std::uint64_t &target, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - target)
      target = std::numeric_limits<std::uint64_t>::max();
    else
      target += value;
  };
  std::vector<JointDesignAttemptRecord> attemptRecords;
  std::vector<JointDesignQualityObservation> qualityObservations;
  std::vector<JointHardwarePromotionObservation> hardwarePromotionObservations;
  if (request.stoppingPolicy == JointDesignStoppingPolicy::BoundedQuality) {
    if (!request.boundedQuality || !request.boundedQuality->objectiveProgram ||
        !request.boundedQuality->acquire ||
        request.boundedQuality->maximumHardwareSpectrumParents == 0 ||
        request.boundedQuality->maximumHardwareRepairProbes == 0)
      return invalid("bounded-quality stopping requires one complete QoR "
                     "acquisition policy");
    const auto &quality = *request.boundedQuality;
    if (quality.objectiveDimensionLabels.size() !=
        quality.objectiveProgram->dimensionCount())
      return invalid("bounded-quality objective labels do not match its "
                     "objective dimension count");
    for (const std::string &label : quality.objectiveDimensionLabels) {
      if (label.empty() ||
          llvm::count(quality.objectiveDimensionLabels, label) != 1)
        return invalid("bounded-quality objective labels must be non-empty "
                       "and unique");
    }
    if (quality.hardwarePromotion) {
      const auto &promotion = *quality.hardwarePromotion;
      if (!promotion.objectiveProgram || !promotion.acquire ||
          promotion.totalOrdering >=
              promotion.objectiveProgram->totalOrderingCount() ||
          promotion.objectiveDimensionLabels.size() !=
              promotion.objectiveProgram->dimensionCount())
        return invalid("bounded-quality hardware promotion is incomplete");
      for (const std::string &label : promotion.objectiveDimensionLabels)
        if (label.empty() ||
            llvm::count(promotion.objectiveDimensionLabels, label) != 1)
          return invalid("bounded-quality hardware-promotion labels must be "
                         "non-empty and unique");
    }
  } else if (request.boundedQuality) {
    return invalid("FirstVerified stopping cannot carry a bounded-quality "
                   "policy");
  }
  struct HardwarePromotionAssessment final {
    std::optional<CandidateObjectiveVector> objective;
    std::optional<IncompleteJointDesignQuality> incomplete;
  };
  std::map<std::uint64_t, HardwarePromotionAssessment>
      hardwarePromotionAssessments;
  const auto validateQualityEvidence =
      [&](const std::optional<ArtifactRootReference> &evidence) -> llvm::Error {
    if (!evidence)
      return llvm::Error::success();
    if (evidence->schemaIdentity !=
            evaluation::EvaluationEvidence::artifactSchema.identity ||
        evidence->schemaVersion !=
            evaluation::EvaluationEvidence::artifactSchema.version)
      return invalid("quality acquisition returned a foreign Evidence root");
    auto stored = artifacts.get(*evidence);
    if (!stored)
      return stored.takeError();
    return llvm::Error::success();
  };
  const auto validateQualityEvidenceSet =
      [&](llvm::ArrayRef<ArtifactRootReference> evidence) -> llvm::Error {
    for (const ArtifactRootReference &reference : evidence)
      if (llvm::Error error = validateQualityEvidence(reference))
        return error;
    return llvm::Error::success();
  };
  const auto validateQualityProvenance =
      [&](const ArtifactRootReference &candidate,
          const std::optional<ArtifactRootReference> &evidence,
          llvm::ArrayRef<ArtifactRootReference> supportingEvidence,
          llvm::ArrayRef<ArtifactRootReference> verificationEvidence,
          const JointDesignQualityProvenance &provenance) -> llvm::Error {
    if (llvm::Error error = validateQualityEvidence(evidence))
      return error;
    if (llvm::Error error = validateQualityEvidenceSet(supportingEvidence))
      return error;
    if (llvm::Error error = validateQualityEvidenceSet(verificationEvidence))
      return error;
    if (provenance.calibratedModelSupport !=
            JointDesignCalibratedModelSupport::NotEvaluated &&
        !evidence)
      return invalid("calibrated quality provenance has no primary "
                     "Evaluation Evidence");
    for (const ArtifactRootReference &verification : verificationEvidence)
      if (!llvm::is_contained(supportingEvidence, verification))
        return invalid("quality verification Evidence is outside its "
                       "supporting Evidence");
    if (provenance.spatialFifoFeedback &&
        provenance.spatialFifoFeedback->parentMapping != candidate)
      return invalid("quality FIFO feedback names a foreign Mapping");
    if (provenance.spatialOperandQueueFeedback &&
        provenance.spatialOperandQueueFeedback->parentMapping &&
        *provenance.spatialOperandQueueFeedback->parentMapping != candidate)
      return invalid("quality operand feedback names a foreign Mapping");
    if (provenance.spatialTransportFeedback &&
        provenance.spatialTransportFeedback->parentMapping &&
        *provenance.spatialTransportFeedback->parentMapping != candidate)
      return invalid("quality transport feedback names a foreign Mapping");
    return llvm::Error::success();
  };
  const auto acquireHardwarePromotion =
      [&](const JointDesignExplorationPlan &plan, std::uint64_t planOrdinal)
      -> llvm::Expected<const CandidateObjectiveVector *> {
    if (!request.boundedQuality || !request.boundedQuality->hardwarePromotion)
      return static_cast<const CandidateObjectiveVector *>(nullptr);
    if (plan.frontier.systemFrontier.size() != 1)
      return invalid("hardware promotion plan has no exact System");
    const ArtifactRootReference &system = plan.frontier.systemFrontier.front();
    auto [position, inserted] = hardwarePromotionAssessments.try_emplace(
        planOrdinal, HardwarePromotionAssessment{});
    HardwarePromotionAssessment &assessment = position->second;
    if (!inserted)
      return assessment.objective ? &*assessment.objective : nullptr;
    auto acquired =
        request.boundedQuality->hardwarePromotion->acquire(plan, planOrdinal);
    if (!acquired)
      return acquired.takeError();
    if (auto *incomplete =
            std::get_if<IncompleteJointDesignQuality>(&*acquired)) {
      if (incomplete->candidate && *incomplete->candidate != system)
        return invalid("hardware promotion incomplete result names a foreign "
                       "System");
      if (!incomplete->candidate)
        incomplete->candidate = system;
      if (llvm::Error error = validateQualityProvenance(
              system, incomplete->evidence,
              incomplete->provenance.supportingEvidence,
              incomplete->provenance.verificationEvidence,
              incomplete->provenance))
        return std::move(error);
      if (incomplete->provenance.calibratedModelSupport ==
              JointDesignCalibratedModelSupport::OutOfDomain &&
          incomplete->reason != JointDesignQualityIncompleteReason::Unsupported)
        return invalid("out-of-domain hardware-promotion quality has a "
                       "foreign incomplete disposition");
      assessment.incomplete = std::move(*incomplete);
      hardwarePromotionObservations.push_back(
          {planOrdinal,
           system,
           {},
           assessment.incomplete->reason,
           assessment.incomplete->evidence,
           false,
           assessment.incomplete->provenance});
      boundedQualitySearchIncomplete = true;
      return static_cast<const CandidateObjectiveVector *>(nullptr);
    }
    auto objectives = std::get<std::vector<JointDesignQualityCandidate>>(
        std::move(*acquired));
    if (objectives.size() != 1 ||
        objectives.front().objective.candidate != system)
      return invalid("hardware promotion acquisition must return exactly one "
                     "objective for its System");
    if (llvm::Error error = validateQualityProvenance(
            system, objectives.front().evidence,
            objectives.front().provenance.supportingEvidence,
            objectives.front().provenance.verificationEvidence,
            objectives.front().provenance))
      return std::move(error);
    if (objectives.front().provenance.calibratedModelSupport ==
        JointDesignCalibratedModelSupport::OutOfDomain)
      return invalid("complete hardware-promotion quality claimed an "
                     "out-of-domain calibrated model");
    if (llvm::Error error = validateJointDesignQualityObjective(
            *request.boundedQuality->hardwarePromotion->objectiveProgram,
            objectives.front().provenance,
            objectives.front().objective.objective.codes()))
      return std::move(error);
    hardwarePromotionObservations.push_back(
        {planOrdinal, system,
         std::vector<std::uint64_t>(
             objectives.front().objective.objective.codes().begin(),
             objectives.front().objective.objective.codes().end()),
         std::nullopt, objectives.front().evidence, false,
         objectives.front().provenance});
    assessment.objective = std::move(objectives.front().objective);
    return &*assessment.objective;
  };
  const auto markHardwarePromotion =
      [&](std::uint64_t planOrdinal) -> llvm::Expected<ArtifactRootReference> {
    auto observation = llvm::find_if(
        hardwarePromotionObservations, [&](const auto &candidate) {
          return candidate.planOrdinal == planOrdinal;
        });
    if (observation == hardwarePromotionObservations.end() ||
        observation->incompleteReason || observation->objectiveCodes.empty())
      return invalid("hardware promotion has no completed quality owner");
    if (observation->promotedToExactMapping)
      return invalid("hardware quality owner promoted one parent twice");
    observation->promotedToExactMapping = true;
    return observation->system;
  };
  const auto finish =
      [&](JointDesignExecution execution,
          std::optional<std::uint64_t> selectedPlanOrdinal,
          std::optional<ArtifactRootReference> selectedMapping,
          JointDesignQualityDisposition qualityDisposition,
          std::optional<ArtifactRootReference> qualityIncompleteCandidate,
          bool declaredWorkExhausted) -> llvm::Expected<JointDesignExecution> {
    if (request.boundedQuality && request.boundedQuality->hardwarePromotion &&
        llvm::count_if(hardwarePromotionObservations,
                       [](const auto &observation) {
                         return observation.promotedToExactMapping;
                       }) != hardwareParentPromotions)
      return invalid("hardware-promotion observations disagree with the exact "
                     "promotion count");
    if (llvm::Error error = attachJointDesignSupportingInvocationManifests(
            execution, encounteredInvocations))
      return std::move(error);
    if (request.stoppingPolicy == JointDesignStoppingPolicy::BoundedQuality) {
      for (const VerifiedAlternative &alternative : verifiedAlternatives)
        mergeMappedPairs(execution, alternative.execution);
    }
    if (accounting.hardwareRepairProbesReserved >=
        accounting.hardwareRepairProbesConsumed) {
      const std::uint64_t accounted = accounting.hardwareRepairProbesConsumed +
                                      accounting.hardwareRepairProbesRejected +
                                      accounting.hardwareRepairProbesCancelled;
      if (accounted < accounting.hardwareRepairProbesReserved) {
        const std::uint64_t remainder =
            accounting.hardwareRepairProbesReserved - accounted;
        if (deadlineObserved ||
            dispatchDeadlineReached(request.executionPolicy))
          accounting.hardwareRepairProbesCancelled += remainder;
        else
          accounting.hardwareRepairProbesRejected += remainder;
      }
    }
    JointDesignExecutionSummary summary;
    summary.stoppingPolicy = request.stoppingPolicy;
    if (!plans.empty() && plans.front()) {
      const BoundedJointFrontier &frontier = plans.front()->frontier;
      summary.eligibleJointPairCount = frontier.eligiblePairCount;
      summary.analyticEvaluatedJointPairCount =
          frontier.analyticEvaluatedPairCount;
      summary.analyticDeferredJointPairCount =
          frontier.analyticDeferredPairCount;
      summary.retainedJointPairCount = frontier.pairs.size();
      summary.jointFrontierTruncated = frontier.truncated;
      summary.retainedJointPairAnalytics.reserve(frontier.pairs.size());
      for (std::size_t index = 0; index != frontier.pairs.size(); ++index)
        summary.retainedJointPairAnalytics.push_back(
            {frontier.pairs[index].software.dataflow,
             frontier.pairs[index].system, frontier.pairProjections[index]});
    }
    summary.attemptedSoftwarePlans = attemptedSoftwarePlans;
    summary.hardwareReopenSearches = hardwareReopenSearches;
    summary.hardwareParentPromotions = hardwareParentPromotions;
    summary.hardwareReopensDeferredByQuality = hardwareReopensDeferredByQuality;
    summary.hardwareReopensWithheldWithoutExactFeedback =
        hardwareReopensWithheldWithoutExactFeedback;
    summary.hardwareRepairProbeLimit = accounting.hardwareRepairProbeLimit;
    summary.hardwareRepairProbesPlanned =
        accounting.hardwareRepairProbesPlanned;
    summary.hardwareRepairProbesReserved =
        accounting.hardwareRepairProbesReserved;
    summary.hardwareRepairProbesConsumed =
        accounting.hardwareRepairProbesConsumed;
    summary.hardwareRepairProbesRejected =
        accounting.hardwareRepairProbesRejected;
    summary.hardwareRepairProbesCancelled =
        accounting.hardwareRepairProbesCancelled;
    summary.techMappingInvocationCount = accounting.techMappingInvocationCount;
    summary.spatialPnrInvocationCount = accounting.spatialPnrInvocationCount;
    summary.systemPnrInvocationCount = accounting.systemPnrInvocationCount;
    summary.techMappingDispatchCount = accounting.techMappingDispatchCount;
    summary.spatialPnrDispatchCount = accounting.spatialPnrDispatchCount;
    summary.systemPnrDispatchCount = accounting.systemPnrDispatchCount;
    summary.techMappingJournalReplayCount =
        accounting.techMappingJournalReplayCount;
    summary.spatialPnrJournalReplayCount =
        accounting.spatialPnrJournalReplayCount;
    summary.systemPnrJournalReplayCount =
        accounting.systemPnrJournalReplayCount;
    summary.coldReopenWallTimeNanoseconds =
        accounting.coldReopenWallTimeNanoseconds;
    summary.incrementalReopenWallTimeNanoseconds =
        accounting.incrementalReopenWallTimeNanoseconds;
    summary.timeToFirstFeasibleWallTimeNanoseconds = timeToFirstFeasible;
    summary.timeToBestWallTimeNanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - executionStart)
            .count());
    summary.preservedTechMappings = accounting.preservedTechMappings;
    summary.preservedSpatialMappings = accounting.preservedSpatialMappings;
    summary.repairedTechMappings = accounting.repairedTechMappings;
    summary.repairedSpatialMappings = accounting.repairedSpatialMappings;
    summary.invalidatedTechMappings = accounting.invalidatedTechMappings;
    summary.invalidatedSpatialMappings = accounting.invalidatedSpatialMappings;
    summary.parentTechDecisions = accounting.parentTechDecisions;
    summary.parentSpatialDecisions = accounting.parentSpatialDecisions;
    summary.preservedTechDecisions = accounting.preservedTechDecisions;
    summary.preservedSpatialDecisions = accounting.preservedSpatialDecisions;
    summary.reopenedTechDecisions = accounting.reopenedTechDecisions;
    summary.reopenedSpatialDecisions = accounting.reopenedSpatialDecisions;
    summary.repairedTechDecisions = accounting.repairedTechDecisions;
    summary.repairedSpatialDecisions = accounting.repairedSpatialDecisions;
    summary.invalidationRootCount = accounting.invalidationRootCount;
    summary.invalidationConeDecisionCount =
        accounting.invalidationConeDecisionCount;
    summary.parentRouteNodeCount = accounting.parentRouteNodeCount;
    summary.preservedRouteNodeCount = accounting.preservedRouteNodeCount;
    summary.reopenedRouteNodeCount = accounting.reopenedRouteNodeCount;
    summary.repairedRouteNodeCount = accounting.repairedRouteNodeCount;
    summary.parentServiceLegCount = accounting.parentServiceLegCount;
    summary.preservedServiceLegCount = accounting.preservedServiceLegCount;
    summary.reopenedServiceLegCount = accounting.reopenedServiceLegCount;
    summary.parentThreadBindingCount = accounting.parentThreadBindingCount;
    summary.preservedThreadBindingCount =
        accounting.preservedThreadBindingCount;
    summary.reopenedThreadBindingCount = accounting.reopenedThreadBindingCount;
    summary.parentGraphBindingCount = accounting.parentGraphBindingCount;
    summary.preservedGraphBindingCount = accounting.preservedGraphBindingCount;
    summary.reopenedGraphBindingCount = accounting.reopenedGraphBindingCount;
    summary.parentResourceUseCount = accounting.parentResourceUseCount;
    summary.preservedResourceUseCount = accounting.preservedResourceUseCount;
    summary.reopenedResourceUseCount = accounting.reopenedResourceUseCount;
    summary.parentServiceRealizationCount =
        accounting.parentServiceRealizationCount;
    summary.preservedServiceRealizationCount =
        accounting.preservedServiceRealizationCount;
    summary.reopenedServiceRealizationCount =
        accounting.reopenedServiceRealizationCount;
    summary.verifiedAlternatives = verifiedMappingCount;
    summary.selectedPlanOrdinal = selectedPlanOrdinal;
    summary.selectedMapping = std::move(selectedMapping);
    summary.qualityDisposition = qualityDisposition;
    summary.qualityIncompleteCandidate = std::move(qualityIncompleteCandidate);
    if (request.boundedQuality)
      summary.qualityObjectiveDimensionLabels =
          request.boundedQuality->objectiveDimensionLabels;
    summary.qualityObservations = qualityObservations;
    if (request.boundedQuality && request.boundedQuality->hardwarePromotion)
      summary.hardwarePromotionObjectiveDimensionLabels =
          request.boundedQuality->hardwarePromotion->objectiveDimensionLabels;
    summary.hardwarePromotionObservations = hardwarePromotionObservations;
    llvm::sort(summary.hardwarePromotionObservations,
               [](const auto &lhs, const auto &rhs) {
                 return lhs.planOrdinal < rhs.planOrdinal;
               });
    summary.declaredWorkExhausted = declaredWorkExhausted;
    summary.attempts = attemptRecords;
    execution.summary = std::move(summary);
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "joint_design_stopping";
          fields["policy"] =
              jointDesignStoppingPolicySpelling(request.stoppingPolicy);
          fields["attempted_software_plans"] = attemptedSoftwarePlans;
          fields["hardware_reopen_searches"] = hardwareReopenSearches;
          fields["hardware_parent_promotions"] = hardwareParentPromotions;
          fields["hardware_reopens_deferred_by_quality"] =
              hardwareReopensDeferredByQuality;
          fields["hardware_reopens_withheld_without_exact_feedback"] =
              hardwareReopensWithheldWithoutExactFeedback;
          fields["hardware_repair_probe_limit"] =
              accounting.hardwareRepairProbeLimit;
          fields["hardware_repair_probes_planned"] =
              accounting.hardwareRepairProbesPlanned;
          fields["hardware_repair_probes_reserved"] =
              accounting.hardwareRepairProbesReserved;
          fields["hardware_repair_probes_consumed"] =
              accounting.hardwareRepairProbesConsumed;
          fields["hardware_repair_probes_rejected"] =
              accounting.hardwareRepairProbesRejected;
          fields["hardware_repair_probes_cancelled"] =
              accounting.hardwareRepairProbesCancelled;
          fields["tech_mapping_invocation_count"] =
              accounting.techMappingInvocationCount;
          fields["spatial_pnr_invocation_count"] =
              accounting.spatialPnrInvocationCount;
          fields["system_pnr_invocation_count"] =
              accounting.systemPnrInvocationCount;
          fields["tech_mapping_dispatch_count"] =
              accounting.techMappingDispatchCount;
          fields["spatial_pnr_dispatch_count"] =
              accounting.spatialPnrDispatchCount;
          fields["system_pnr_dispatch_count"] =
              accounting.systemPnrDispatchCount;
          fields["tech_mapping_journal_replay_count"] =
              accounting.techMappingJournalReplayCount;
          fields["spatial_pnr_journal_replay_count"] =
              accounting.spatialPnrJournalReplayCount;
          fields["system_pnr_journal_replay_count"] =
              accounting.systemPnrJournalReplayCount;
          fields["cold_reopen_wall_time_ns"] =
              accounting.coldReopenWallTimeNanoseconds;
          fields["incremental_reopen_wall_time_ns"] =
              accounting.incrementalReopenWallTimeNanoseconds;
          fields["preserved_tech_mappings"] = accounting.preservedTechMappings;
          fields["preserved_spatial_mappings"] =
              accounting.preservedSpatialMappings;
          fields["repaired_tech_mappings"] = accounting.repairedTechMappings;
          fields["repaired_spatial_mappings"] =
              accounting.repairedSpatialMappings;
          fields["invalidated_tech_mappings"] =
              accounting.invalidatedTechMappings;
          fields["invalidated_spatial_mappings"] =
              accounting.invalidatedSpatialMappings;
          fields["parent_tech_decisions"] = accounting.parentTechDecisions;
          fields["parent_spatial_decisions"] =
              accounting.parentSpatialDecisions;
          fields["preserved_tech_decisions"] =
              accounting.preservedTechDecisions;
          fields["preserved_spatial_decisions"] =
              accounting.preservedSpatialDecisions;
          fields["reopened_tech_decisions"] = accounting.reopenedTechDecisions;
          fields["reopened_spatial_decisions"] =
              accounting.reopenedSpatialDecisions;
          fields["repaired_tech_decisions"] = accounting.repairedTechDecisions;
          fields["repaired_spatial_decisions"] =
              accounting.repairedSpatialDecisions;
          fields["invalidation_root_count"] = accounting.invalidationRootCount;
          fields["invalidation_cone_decision_count"] =
              accounting.invalidationConeDecisionCount;
          fields["parent_route_node_count"] = accounting.parentRouteNodeCount;
          fields["preserved_route_node_count"] =
              accounting.preservedRouteNodeCount;
          fields["reopened_route_node_count"] =
              accounting.reopenedRouteNodeCount;
          fields["repaired_route_node_count"] =
              accounting.repairedRouteNodeCount;
          fields["parent_service_leg_count"] = accounting.parentServiceLegCount;
          fields["preserved_service_leg_count"] =
              accounting.preservedServiceLegCount;
          fields["reopened_service_leg_count"] =
              accounting.reopenedServiceLegCount;
          fields["verified_alternatives"] =
              execution.summary.verifiedAlternatives;
          fields["declared_work_exhausted"] = declaredWorkExhausted;
          if (selectedPlanOrdinal)
            fields["selected_plan_ordinal"] = *selectedPlanOrdinal;
          if (execution.summary.selectedMapping)
            fields["selected_mapping"] = formatArtifactIdentityHex(
                execution.summary.selectedMapping->artifact);
          fields["quality_disposition"] =
              static_cast<std::uint64_t>(qualityDisposition);
          fields["quality_objective_dimension_count"] =
              execution.summary.qualityObjectiveDimensionLabels.size();
        });
    return execution;
  };
  for (auto indexed : llvm::enumerate(plans)) {
    // The first plan execution owns the typed cancellation checkpoint. Even
    // when the absolute deadline has already elapsed, enter that boundary
    // once so PlanExecutor can publish Incomplete instead of leaving this
    // controller with no terminal outcome. Never admit a sibling afterward.
    if (attemptedSoftwarePlans != 0 &&
        dispatchDeadlineReached(request.executionPolicy)) {
      deadlineObserved = true;
      boundedQualitySearchIncomplete = true;
      break;
    }
    const JointDesignExplorationPlan *planPointer = indexed.value();
    if (!planPointer)
      return invalid("hardware reopen plan pointer is null");
    const JointDesignExplorationPlan &plan = *planPointer;
    ++attemptedSoftwarePlans;
    std::optional<PlanExecutionPolicy> planExecutionPolicy;
    if (request.stoppingPolicy == JointDesignStoppingPolicy::BoundedQuality) {
      const std::uint64_t remainingPlans = plans.size() - indexed.index();
      auto fair =
          fairBoundedQualityPlanPolicy(request.executionPolicy, remainingPlans);
      if (!fair)
        return fair.takeError();
      planExecutionPolicy.emplace(std::move(*fair));
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = "bounded_quality_plan_slice";
            fields["plan_ordinal"] = indexed.index();
            fields["remaining_plan_count"] = remainingPlans;
            if (planExecutionPolicy->dispatchNotAfterUnixNanoseconds())
              fields["dispatch_not_after_unix_ns"] =
                  *planExecutionPolicy->dispatchNotAfterUnixNanoseconds();
          });
    }
    auto initial = executeJointPlan(
        plan, request.evidence, request, *scheduler, artifacts, blobs,
        planExecutionPolicy ? &*planExecutionPolicy : nullptr);
    if (!initial)
      return initial.takeError();
    if (llvm::Error error = retainJointDesignExecutionInvocations(
            encounteredInvocations, *initial))
      return std::move(error);
    // The initial parent execution is outside tryHardwareFeedbackReopen, so
    // carry its invocation-local accounting into the stopping summary here.
    // Reopen attempts are accounted at their dispatch boundary below.
    saturatingAdd(accounting.techMappingInvocationCount,
                  initial->summary.techMappingInvocationCount);
    saturatingAdd(accounting.spatialPnrInvocationCount,
                  initial->summary.spatialPnrInvocationCount);
    saturatingAdd(accounting.systemPnrInvocationCount,
                  initial->summary.systemPnrInvocationCount);
    saturatingAdd(accounting.techMappingDispatchCount,
                  initial->summary.techMappingDispatchCount);
    saturatingAdd(accounting.spatialPnrDispatchCount,
                  initial->summary.spatialPnrDispatchCount);
    saturatingAdd(accounting.systemPnrDispatchCount,
                  initial->summary.systemPnrDispatchCount);
    saturatingAdd(accounting.techMappingJournalReplayCount,
                  initial->summary.techMappingJournalReplayCount);
    saturatingAdd(accounting.spatialPnrJournalReplayCount,
                  initial->summary.spatialPnrJournalReplayCount);
    saturatingAdd(accounting.systemPnrJournalReplayCount,
                  initial->summary.systemPnrJournalReplayCount);
    saturatingAdd(accounting.coldReopenWallTimeNanoseconds,
                  initial->summary.executionWallTimeNanoseconds);
    saturatingAdd(accounting.incrementalReopenWallTimeNanoseconds,
                  initial->summary.incrementalReopenWallTimeNanoseconds);
    saturatingAdd(accounting.preservedTechMappings,
                  initial->summary.preservedTechMappings);
    saturatingAdd(accounting.preservedSpatialMappings,
                  initial->summary.preservedSpatialMappings);
    saturatingAdd(accounting.repairedTechMappings,
                  initial->summary.repairedTechMappings);
    saturatingAdd(accounting.repairedSpatialMappings,
                  initial->summary.repairedSpatialMappings);
    saturatingAdd(accounting.invalidatedTechMappings,
                  initial->summary.invalidatedTechMappings);
    saturatingAdd(accounting.invalidatedSpatialMappings,
                  initial->summary.invalidatedSpatialMappings);
    saturatingAdd(accounting.parentTechDecisions,
                  initial->summary.parentTechDecisions);
    saturatingAdd(accounting.parentSpatialDecisions,
                  initial->summary.parentSpatialDecisions);
    saturatingAdd(accounting.preservedTechDecisions,
                  initial->summary.preservedTechDecisions);
    saturatingAdd(accounting.preservedSpatialDecisions,
                  initial->summary.preservedSpatialDecisions);
    saturatingAdd(accounting.reopenedTechDecisions,
                  initial->summary.reopenedTechDecisions);
    saturatingAdd(accounting.reopenedSpatialDecisions,
                  initial->summary.reopenedSpatialDecisions);
    saturatingAdd(accounting.repairedTechDecisions,
                  initial->summary.repairedTechDecisions);
    saturatingAdd(accounting.repairedSpatialDecisions,
                  initial->summary.repairedSpatialDecisions);
    saturatingAdd(accounting.invalidationRootCount,
                  initial->summary.invalidationRootCount);
    saturatingAdd(accounting.invalidationConeDecisionCount,
                  initial->summary.invalidationConeDecisionCount);
    saturatingAdd(accounting.parentRouteNodeCount,
                  initial->summary.parentRouteNodeCount);
    saturatingAdd(accounting.preservedRouteNodeCount,
                  initial->summary.preservedRouteNodeCount);
    saturatingAdd(accounting.reopenedRouteNodeCount,
                  initial->summary.reopenedRouteNodeCount);
    saturatingAdd(accounting.repairedRouteNodeCount,
                  initial->summary.repairedRouteNodeCount);
    saturatingAdd(accounting.parentServiceLegCount,
                  initial->summary.parentServiceLegCount);
    saturatingAdd(accounting.preservedServiceLegCount,
                  initial->summary.preservedServiceLegCount);
    saturatingAdd(accounting.reopenedServiceLegCount,
                  initial->summary.reopenedServiceLegCount);
    saturatingAdd(accounting.parentThreadBindingCount,
                  initial->summary.parentThreadBindingCount);
    saturatingAdd(accounting.preservedThreadBindingCount,
                  initial->summary.preservedThreadBindingCount);
    saturatingAdd(accounting.reopenedThreadBindingCount,
                  initial->summary.reopenedThreadBindingCount);
    saturatingAdd(accounting.parentGraphBindingCount,
                  initial->summary.parentGraphBindingCount);
    saturatingAdd(accounting.preservedGraphBindingCount,
                  initial->summary.preservedGraphBindingCount);
    saturatingAdd(accounting.reopenedGraphBindingCount,
                  initial->summary.reopenedGraphBindingCount);
    saturatingAdd(accounting.parentResourceUseCount,
                  initial->summary.parentResourceUseCount);
    saturatingAdd(accounting.preservedResourceUseCount,
                  initial->summary.preservedResourceUseCount);
    saturatingAdd(accounting.reopenedResourceUseCount,
                  initial->summary.reopenedResourceUseCount);
    saturatingAdd(accounting.parentServiceRealizationCount,
                  initial->summary.parentServiceRealizationCount);
    saturatingAdd(accounting.preservedServiceRealizationCount,
                  initial->summary.preservedServiceRealizationCount);
    saturatingAdd(accounting.reopenedServiceRealizationCount,
                  initial->summary.reopenedServiceRealizationCount);
    if (plan.frontier.systemFrontier.size() != 1)
      return invalid("application Mapping alternative has no exact System");
    if (llvm::Error error =
            recordJointAttempt(attemptRecords, indexed.index(),
                               plan.frontier.systemFrontier.front(), *initial))
      return std::move(error);
    if (mappingCount(*initial) != 0) {
      verifiedMappingCount += mappingCount(*initial);
      if (!timeToFirstFeasible)
        timeToFirstFeasible = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - executionStart)
                .count());
      if (request.stoppingPolicy == JointDesignStoppingPolicy::FirstVerified) {
        const auto selectedMapping = firstMapping(*initial);
        return finish(std::move(*initial), indexed.index(), selectedMapping,
                      JointDesignQualityDisposition::NotRequested, std::nullopt,
                      false);
      }
      if (const auto *incomplete =
              std::get_if<IncompleteDsePlanExecution>(&initial->planExecution))
        boundedQualitySearchIncomplete |= incomplete->executionStopped();
      verifiedAlternatives.push_back(
          {static_cast<std::uint64_t>(indexed.index()), std::move(*initial)});
      if (dispatchDeadlineReached(request.executionPolicy)) {
        deadlineObserved = true;
        boundedQualitySearchIncomplete = true;
        break;
      }
      continue;
    }
    if (const auto *incomplete =
            std::get_if<IncompleteDsePlanExecution>(&initial->planExecution);
        incomplete && incomplete->executionStopped()) {
      if (request.hardwareExplorationScope ==
          JointHardwareExplorationScope::FixedSystemFrontier) {
        if (!firstIncomplete)
          firstIncomplete = std::move(*initial);
        if (dispatchDeadlineReached(request.executionPolicy)) {
          deadlineObserved = true;
          boundedQualitySearchIncomplete = true;
          break;
        }
        continue;
      }
      // An incomplete parent never proves that its siblings are infeasible,
      // but an exact owner feedback payload retained by that parent can still
      // justify one bounded hardware repair. Keep the parent typed incomplete
      // while admitting only the actionable feedback path; absent feedback
      // remains the ordinary first-incomplete witness.
      auto tech = selectTechHardwareFeedback(*initial, artifacts);
      if (!tech)
        return tech.takeError();
      auto spatial = selectSpatialHardwareFeedback(*initial, artifacts);
      if (!spatial)
        return spatial.takeError();
      auto system = selectSystemHardwareFeedback(*initial, artifacts);
      if (!system)
        return system.takeError();
      if (request.spectrumEndpoint != PreMappingSpectrumEndpoint::Automatic &&
          (*tech || *spatial || *system)) {
        auto coverage = projectJointSoftwareCoverage(plan, artifacts);
        if (!coverage)
          return coverage.takeError();
        mapping_debug::emit(
            mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
            mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
              fields["operation"] = "incomplete_parent_hardware_feedback";
              fields["plan_ordinal"] = indexed.index();
              fields["tech_feedback"] = static_cast<bool>(*tech);
              fields["spatial_feedback"] = static_cast<bool>(*spatial);
              fields["system_feedback"] = static_cast<bool>(*system);
              fields["parent_disposition"] = "incomplete";
            });
        failedSoftwareAttempts.push_back(
            {static_cast<std::uint64_t>(indexed.index()), planPointer,
             std::move(*coverage), std::move(*initial), 0});
      } else if (!firstIncomplete) {
        firstIncomplete = std::move(*initial);
      }
      if (dispatchDeadlineReached(request.executionPolicy)) {
        deadlineObserved = true;
        boundedQualitySearchIncomplete = true;
        break;
      }
      continue;
    }
    if (request.hardwareExplorationScope ==
        JointHardwareExplorationScope::FixedSystemFrontier) {
      lastNoFeasible = std::move(*initial);
      continue;
    }
    auto coverage = projectJointSoftwareCoverage(plan, artifacts);
    if (!coverage)
      return coverage.takeError();
    failedSoftwareAttempts.push_back(
        {static_cast<std::uint64_t>(indexed.index()), planPointer,
         std::move(*coverage), std::move(*initial), 0});
  }
  // Hardware feedback is consumed only after every bounded software/System
  // pair has been tried on the parent System. This preserves the declared
  // software frontier order and prevents repairable early failures from
  // hiding a later parent-hardware solution.
  std::vector<FailedSoftwareAttempt *> hardwareFeedbackFrontier;
  if (request.hardwareExplorationScope ==
          JointHardwareExplorationScope::BoundedHardwareReopen &&
      request.stoppingPolicy != JointDesignStoppingPolicy::BoundedQuality) {
    for (FailedSoftwareAttempt &attempt : failedSoftwareAttempts)
      hardwareFeedbackFrontier.push_back(&attempt);
  } else if (request.hardwareExplorationScope ==
             JointHardwareExplorationScope::BoundedHardwareReopen) {
    for (FailedSoftwareAttempt &attempt : failedSoftwareAttempts) {
      auto tech = selectTechHardwareFeedback(attempt.execution, artifacts);
      if (!tech)
        return tech.takeError();
      auto spatial =
          selectSpatialHardwareFeedback(attempt.execution, artifacts);
      if (!spatial)
        return spatial.takeError();
      auto system = selectSystemHardwareFeedback(attempt.execution, artifacts);
      if (!system)
        return system.takeError();
      attempt.techHallDeficit = *tech ? (*tech)->feedback.deficit() : 0;
      if (*tech || *spatial || *system)
        hardwareFeedbackFrontier.push_back(&attempt);
    }
    llvm::sort(hardwareFeedbackFrontier, [&](const FailedSoftwareAttempt *lhs,
                                             const FailedSoftwareAttempt *rhs) {
      if (request.spectrumEndpoint != PreMappingSpectrumEndpoint::Automatic &&
          lhs->techHallDeficit != rhs->techHallDeficit)
        return lhs->techHallDeficit > rhs->techHallDeficit;
      if (lhs->coverage.acceleratedRootCount !=
          rhs->coverage.acceleratedRootCount)
        return lhs->coverage.acceleratedRootCount >
               rhs->coverage.acceleratedRootCount;
      if (lhs->coverage.graphCount != rhs->coverage.graphCount)
        return lhs->coverage.graphCount > rhs->coverage.graphCount;
      if (lhs->coverage.actorCount != rhs->coverage.actorCount)
        return lhs->coverage.actorCount > rhs->coverage.actorCount;
      return lhs->planOrdinal < rhs->planOrdinal;
    });
    const std::size_t actionableFeedbackCount = hardwareFeedbackFrontier.size();
    if (request.boundedQuality->hardwarePromotion) {
      std::vector<FailedSoftwareAttempt *> ranked;
      ranked.reserve(hardwareFeedbackFrontier.size());
      const auto &promotion = *request.boundedQuality->hardwarePromotion;
      for (FailedSoftwareAttempt *candidate : hardwareFeedbackFrontier) {
        auto candidateObjective =
            acquireHardwarePromotion(*candidate->plan, candidate->planOrdinal);
        if (!candidateObjective)
          return candidateObjective.takeError();
        if (!*candidateObjective)
          continue;
        auto insertion = ranked.begin();
        for (; insertion != ranked.end(); ++insertion) {
          auto existingObjective = acquireHardwarePromotion(
              *(*insertion)->plan, (*insertion)->planOrdinal);
          if (!existingObjective)
            return existingObjective.takeError();
          if (!*existingObjective)
            return invalid("ranked hardware promotion lost its objective");
          auto comparison = promotion.objectiveProgram->compareTotalOrdering(
              (*candidateObjective)->objective,
              encodeArtifactRootReference(
                  candidate->plan->frontier.systemFrontier.front()),
              (*existingObjective)->objective,
              encodeArtifactRootReference(
                  (*insertion)->plan->frontier.systemFrontier.front()),
              promotion.totalOrdering);
          if (!comparison)
            return comparison.takeError();
          if (*comparison < 0)
            break;
        }
        ranked.insert(insertion, candidate);
      }
      hardwareFeedbackFrontier = std::move(ranked);
    }
    const std::size_t limit = static_cast<std::size_t>(std::min<std::uint64_t>(
        request.boundedQuality->maximumHardwareSpectrumParents,
        hardwareFeedbackFrontier.size()));
    hardwareFeedbackFrontier.resize(limit);
    hardwareReopensDeferredByQuality =
        actionableFeedbackCount - hardwareFeedbackFrontier.size();
    hardwareReopensWithheldWithoutExactFeedback =
        failedSoftwareAttempts.size() - actionableFeedbackCount;
  }
  for (auto indexedAttempt : llvm::enumerate(hardwareFeedbackFrontier)) {
    FailedSoftwareAttempt &attempt = *indexedAttempt.value();
    if (dispatchDeadlineReached(request.executionPolicy)) {
      deadlineObserved = true;
      boundedQualitySearchIncomplete = true;
      break;
    }
    std::optional<PlanExecutionPolicy> feedbackExecutionPolicy;
    std::optional<ArtifactRootReference> promotedParentSystem;
    if (request.stoppingPolicy == JointDesignStoppingPolicy::BoundedQuality) {
      auto fair = fairBoundedQualityPlanPolicy(request.executionPolicy,
                                               hardwareFeedbackFrontier.size() -
                                                   indexedAttempt.index());
      if (!fair)
        return fair.takeError();
      feedbackExecutionPolicy.emplace(std::move(*fair));
      if (request.boundedQuality->hardwarePromotion) {
        auto promoted = markHardwarePromotion(attempt.planOrdinal);
        if (!promoted)
          return promoted.takeError();
        if (attempt.plan->frontier.systemFrontier.size() != 1 ||
            *promoted != attempt.plan->frontier.systemFrontier.front())
          return invalid("hardware promotion quality owner names a foreign "
                         "parent System");
        promotedParentSystem = std::move(*promoted);
      }
      ++hardwareParentPromotions;
    }
    ++hardwareReopenSearches;
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "hardware_feedback_promotion";
          fields["plan_ordinal"] = attempt.planOrdinal;
          fields["tech_hall_deficit"] = attempt.techHallDeficit;
          fields["accelerated_root_count"] =
              attempt.coverage.acceleratedRootCount;
          fields["graph_count"] = attempt.coverage.graphCount;
          fields["actor_count"] = attempt.coverage.actorCount;
        });
    std::optional<JointDesignExecution> lastReopenedFailure;
    auto reopened = tryHardwareFeedbackReopen(
        policy, *attempt.plan, attempt.execution, lastReopenedFailure,
        attempt.planOrdinal, attemptRecords, accounting, encounteredInvocations,
        request.evidence, request, *scheduler, artifacts, blobs,
        promotedParentSystem,
        feedbackExecutionPolicy ? &*feedbackExecutionPolicy : nullptr);
    if (!reopened)
      return reopened.takeError();
    if (*reopened) {
      if (llvm::Error error = retainJointDesignExecutionInvocations(
              encounteredInvocations, **reopened))
        return std::move(error);
      if (mappingCount(**reopened) == 0) {
        if (std::holds_alternative<IncompleteDsePlanExecution>(
                (*reopened)->planExecution)) {
          if (!firstIncomplete)
            firstIncomplete = std::move(**reopened);
          continue;
        }
        return finish(std::move(**reopened), std::nullopt, std::nullopt,
                      JointDesignQualityDisposition::NotRequested, std::nullopt,
                      false);
      }
      verifiedMappingCount += mappingCount(**reopened);
      if (!timeToFirstFeasible)
        timeToFirstFeasible = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - executionStart)
                .count());
      if (request.stoppingPolicy == JointDesignStoppingPolicy::FirstVerified) {
        const auto selectedMapping = firstMapping(**reopened);
        return finish(
            std::move(**reopened), attempt.planOrdinal, selectedMapping,
            JointDesignQualityDisposition::NotRequested, std::nullopt, false);
      }
      verifiedAlternatives.push_back(
          {attempt.planOrdinal, std::move(**reopened)});
      if (dispatchDeadlineReached(request.executionPolicy)) {
        deadlineObserved = true;
        boundedQualitySearchIncomplete = true;
        break;
      }
      continue;
    }
    JointDesignExecution &failed =
        lastReopenedFailure ? *lastReopenedFailure : attempt.execution;
    if (llvm::Error error = retainJointDesignExecutionInvocations(
            encounteredInvocations, failed))
      return std::move(error);
    if (std::holds_alternative<IncompleteDsePlanExecution>(
            failed.planExecution)) {
      if (!firstIncomplete)
        firstIncomplete = std::move(failed);
    } else {
      lastNoFeasible = std::move(failed);
    }
  }

  // Hardware expansion is the next expensive rung after the complete base
  // software frontier. Exact failed-candidate feedback consumes the shared
  // parent budget first in semantic coverage order. Any remaining budget may
  // expand verified parents in analytic order. Both paths reserve a terminal
  // share for application QoR and retain their original typed outcome.
  if (request.hardwareExplorationScope ==
          JointHardwareExplorationScope::BoundedHardwareReopen &&
      request.stoppingPolicy == JointDesignStoppingPolicy::BoundedQuality &&
      !verifiedAlternatives.empty()) {
    const std::size_t baseAlternativeCount = verifiedAlternatives.size();
    std::vector<std::size_t> hardwareParentOrder;
    hardwareParentOrder.reserve(baseAlternativeCount);
    if (request.boundedQuality->hardwarePromotion) {
      const auto &promotion = *request.boundedQuality->hardwarePromotion;
      for (std::size_t candidateIndex = 0;
           candidateIndex != baseAlternativeCount; ++candidateIndex) {
        VerifiedAlternative &candidate = verifiedAlternatives[candidateIndex];
        if (candidate.planOrdinal >= plans.size() ||
            !plans[candidate.planOrdinal])
          return invalid("bounded-quality hardware parent lost its plan");
        auto candidateObjective = acquireHardwarePromotion(
            *plans[candidate.planOrdinal], candidate.planOrdinal);
        if (!candidateObjective)
          return candidateObjective.takeError();
        if (!*candidateObjective)
          continue;
        auto insertion = hardwareParentOrder.begin();
        for (; insertion != hardwareParentOrder.end(); ++insertion) {
          VerifiedAlternative &existing = verifiedAlternatives[*insertion];
          auto existingObjective = acquireHardwarePromotion(
              *plans[existing.planOrdinal], existing.planOrdinal);
          if (!existingObjective)
            return existingObjective.takeError();
          if (!*existingObjective)
            return invalid("ranked hardware parent lost its objective");
          auto comparison = promotion.objectiveProgram->compareTotalOrdering(
              (*candidateObjective)->objective,
              encodeArtifactRootReference(
                  plans[candidate.planOrdinal]
                      ->frontier.systemFrontier.front()),
              (*existingObjective)->objective,
              encodeArtifactRootReference(
                  plans[existing.planOrdinal]->frontier.systemFrontier.front()),
              promotion.totalOrdering);
          if (!comparison)
            return comparison.takeError();
          if (*comparison < 0)
            break;
        }
        hardwareParentOrder.insert(insertion, candidateIndex);
      }
    } else {
      hardwareParentOrder.resize(baseAlternativeCount);
      std::iota(hardwareParentOrder.begin(), hardwareParentOrder.end(), 0);
    }
    const std::uint64_t remainingParentBudget =
        request.boundedQuality->maximumHardwareSpectrumParents >
                hardwareParentPromotions
            ? request.boundedQuality->maximumHardwareSpectrumParents -
                  hardwareParentPromotions
            : 0;
    const std::uint64_t parentLimit = std::min<std::uint64_t>(
        remainingParentBudget, hardwareParentOrder.size());
    saturatingAdd(hardwareReopensDeferredByQuality,
                  baseAlternativeCount - parentLimit);
    for (std::uint64_t parentOrdinal = 0; parentOrdinal != parentLimit;
         ++parentOrdinal) {
      if (dispatchDeadlineReached(request.executionPolicy)) {
        deadlineObserved = true;
        boundedQualitySearchIncomplete = true;
        break;
      }
      VerifiedAlternative &parent =
          verifiedAlternatives[hardwareParentOrder[parentOrdinal]];
      if (parent.planOrdinal >= plans.size() || !plans[parent.planOrdinal])
        return invalid("bounded-quality hardware parent lost its plan");
      const std::uint64_t parentPlanOrdinal = parent.planOrdinal;
      std::optional<ArtifactRootReference> promotedParentSystem;
      if (request.boundedQuality->hardwarePromotion) {
        auto promoted = markHardwarePromotion(parentPlanOrdinal);
        if (!promoted)
          return promoted.takeError();
        if (plans[parentPlanOrdinal]->frontier.systemFrontier.size() != 1 ||
            *promoted !=
                plans[parentPlanOrdinal]->frontier.systemFrontier.front())
          return invalid("hardware promotion quality owner names a foreign "
                         "parent System");
        promotedParentSystem = std::move(*promoted);
      }
      ++hardwareParentPromotions;
      auto spectrumPolicy = fairBoundedQualityPlanPolicy(
          request.executionPolicy, parentLimit - parentOrdinal);
      if (!spectrumPolicy)
        return spectrumPolicy.takeError();
      auto spectrum = exploreFinalizedMappingHardwareSpectrum(
          policy, *plans[parentPlanOrdinal], parent.execution, request.evidence,
          request, *scheduler, artifacts, blobs, &*spectrumPolicy);
      if (!spectrum)
        return spectrum.takeError();
      for (const JointDesignInvocationManifestReference &invocation :
           spectrum->invocations)
        if (llvm::Error error = retainJointDesignInvocationManifest(
                encounteredInvocations, invocation))
          return std::move(error);
      hardwareReopenSearches += spectrum->attemptedSystems;
      boundedQualitySearchIncomplete |= spectrum->incomplete;
      for (JointDesignExecution &execution : spectrum->verified) {
        if (llvm::Error error = recordJointAttempt(
                attemptRecords, parentPlanOrdinal,
                plans[parentPlanOrdinal]->frontier.systemFrontier.front(),
                execution, promotedParentSystem))
          return std::move(error);
        verifiedMappingCount += mappingCount(execution);
        saturatingAdd(accounting.techMappingInvocationCount,
                      execution.summary.techMappingInvocationCount);
        saturatingAdd(accounting.spatialPnrInvocationCount,
                      execution.summary.spatialPnrInvocationCount);
        saturatingAdd(accounting.systemPnrInvocationCount,
                      execution.summary.systemPnrInvocationCount);
        saturatingAdd(accounting.techMappingDispatchCount,
                      execution.summary.techMappingDispatchCount);
        saturatingAdd(accounting.spatialPnrDispatchCount,
                      execution.summary.spatialPnrDispatchCount);
        saturatingAdd(accounting.systemPnrDispatchCount,
                      execution.summary.systemPnrDispatchCount);
        saturatingAdd(accounting.techMappingJournalReplayCount,
                      execution.summary.techMappingJournalReplayCount);
        saturatingAdd(accounting.spatialPnrJournalReplayCount,
                      execution.summary.spatialPnrJournalReplayCount);
        saturatingAdd(accounting.systemPnrJournalReplayCount,
                      execution.summary.systemPnrJournalReplayCount);
        verifiedAlternatives.push_back(
            {parentPlanOrdinal, std::move(execution)});
      }
    }
  }
  if (!verifiedAlternatives.empty()) {
    const JointBoundedQualityPolicy &quality = *request.boundedQuality;
    std::vector<ArtifactRootReference> candidates;
    std::vector<CandidateObjectiveVector> objectives;
    std::map<ArtifactRootReference, std::size_t,
             decltype(&artifactRootReferenceLess)>
        objectiveIndices(&artifactRootReferenceLess);
    std::optional<IncompleteJointDesignQuality> firstQualityIncomplete;
    for (VerifiedAlternative &alternative : verifiedAlternatives) {
      std::vector<ArtifactRootReference> alternativeMappings =
          mappingRoots(alternative.execution);
      // The application QoR owner evaluates one concrete SystemMapping at a
      // time.  The temporary selectedMapping field is invocation evidence,
      // not candidate identity; restoring it after acquisition keeps the
      // outer stopping summary authoritative.
      std::vector<CandidateObjectiveVector> acquiredObjectives;
      acquiredObjectives.reserve(alternativeMappings.size());
      for (const ArtifactRootReference &mapping : alternativeMappings) {
        // A deadline is a cooperative cancellation boundary. Preserve an
        // observation for every already-materialized Mapping without starting
        // another application replay after the deadline.
        if (deadlineObserved ||
            dispatchDeadlineReached(request.executionPolicy)) {
          deadlineObserved = true;
          boundedQualitySearchIncomplete = true;
          JointDesignQualityProvenance provenance;
          if (quality.provenanceDomain ==
              JointDesignQualityProvenanceDomain::ApplicationRuntime) {
            auto resourceCoreCost = deriveApplicationRuntimeResourceCoreCost(
                alternative.execution, mapping, artifacts);
            if (!resourceCoreCost)
              return resourceCoreCost.takeError();
            provenance.resourceCoreCost = *resourceCoreCost;
          }
          if (llvm::Error error = validateJointDesignQualityProvenanceDomain(
                  quality, provenance, false))
            return std::move(error);
          qualityObservations.push_back(
              {mapping,
               {},
               JointDesignQualityIncompleteReason::CancelledOrTimeout,
               std::nullopt,
               provenance});
          if (!firstQualityIncomplete)
            firstQualityIncomplete = IncompleteJointDesignQuality{
                JointDesignQualityIncompleteReason::CancelledOrTimeout, mapping,
                std::nullopt, std::move(provenance)};
          continue;
        }
        alternative.execution.summary.selectedMapping = mapping;
        auto acquired =
            quality.acquire(alternative.execution, alternative.planOrdinal);
        if (!acquired)
          return acquired.takeError();
        if (const auto *incomplete =
                std::get_if<IncompleteJointDesignQuality>(&*acquired)) {
          if (incomplete->candidate && incomplete->candidate != mapping)
            return invalid("bounded-quality incomplete acquisition named a "
                           "foreign SystemMapping");
          if (llvm::Error error = validateQualityProvenance(
                  mapping, incomplete->evidence,
                  incomplete->provenance.supportingEvidence,
                  incomplete->provenance.verificationEvidence,
                  incomplete->provenance))
            return std::move(error);
          if (llvm::Error error = validateJointDesignQualityProvenanceDomain(
                  quality, incomplete->provenance, false))
            return std::move(error);
          qualityObservations.push_back({mapping,
                                         {},
                                         incomplete->reason,
                                         incomplete->evidence,
                                         incomplete->provenance});
          if (!firstQualityIncomplete)
            firstQualityIncomplete = IncompleteJointDesignQuality{
                incomplete->reason, mapping, incomplete->evidence,
                incomplete->provenance};
          alternative.execution.summary.selectedMapping.reset();
          continue;
        }
        std::vector<JointDesignQualityCandidate> one =
            std::get<std::vector<JointDesignQualityCandidate>>(
                std::move(*acquired));
        if (one.size() != 1 || one.front().objective.candidate != mapping)
          return invalid("bounded-quality acquisition must return exactly one "
                         "objective for the selected SystemMapping");
        if (llvm::Error error = validateQualityProvenance(
                mapping, one.front().evidence,
                one.front().provenance.supportingEvidence,
                one.front().provenance.verificationEvidence,
                one.front().provenance))
          return std::move(error);
        if (llvm::Error error = validateJointDesignQualityProvenanceDomain(
                quality, one.front().provenance, true))
          return std::move(error);
        if (llvm::Error error = validateJointDesignQualityObjective(
                *quality.objectiveProgram, one.front().provenance,
                one.front().objective.objective.codes()))
          return std::move(error);
        qualityObservations.push_back(
            {mapping,
             std::vector<std::uint64_t>(
                 one.front().objective.objective.codes().begin(),
                 one.front().objective.objective.codes().end()),
             std::nullopt, one.front().evidence, one.front().provenance});
        acquiredObjectives.push_back(std::move(one.front().objective));
      }
      alternative.execution.summary.selectedMapping.reset();
      for (CandidateObjectiveVector &objective : acquiredObjectives) {
        auto [position, inserted] =
            objectiveIndices.emplace(objective.candidate, objectives.size());
        if (!inserted) {
          if (objectives[position->second].objective.codes() !=
              objective.objective.codes())
            return invalid("bounded-quality acquisition assigned conflicting "
                           "objectives to one SystemMapping");
          continue;
        }
        candidates.push_back(objective.candidate);
        objectives.push_back(std::move(objective));
      }
    }
    llvm::sort(qualityObservations,
               [](const JointDesignQualityObservation &lhs,
                  const JointDesignQualityObservation &rhs) {
                 return artifactRootReferenceLess(lhs.candidate, rhs.candidate);
               });
    for (std::size_t index = 1; index < qualityObservations.size(); ++index) {
      if (qualityObservations[index - 1].candidate !=
          qualityObservations[index].candidate)
        continue;
      if (qualityObservations[index - 1].objectiveCodes !=
              qualityObservations[index].objectiveCodes ||
          qualityObservations[index - 1].incompleteReason !=
              qualityObservations[index].incompleteReason ||
          qualityObservations[index - 1].evidence !=
              qualityObservations[index].evidence ||
          qualityObservations[index - 1].provenance !=
              qualityObservations[index].provenance)
        return invalid("bounded-quality acquisition assigned conflicting "
                       "observations to one SystemMapping");
    }
    qualityObservations.erase(
        std::unique(qualityObservations.begin(), qualityObservations.end(),
                    [](const JointDesignQualityObservation &lhs,
                       const JointDesignQualityObservation &rhs) {
                      return lhs.candidate == rhs.candidate;
                    }),
        qualityObservations.end());
    const auto executionOwner = [&](const ArtifactRootReference &candidate)
        -> llvm::Expected<std::size_t> {
      for (std::size_t ordinal = 0; ordinal != verifiedAlternatives.size();
           ++ordinal)
        if (llvm::is_contained(
                mappingRoots(verifiedAlternatives[ordinal].execution),
                candidate))
          return ordinal;
      return invalid("bounded-quality candidate has no verified execution "
                     "owner");
    };
    if (objectives.empty()) {
      if (!firstQualityIncomplete)
        return invalid("bounded-quality acquisition produced no objectives");
      auto fallback = firstMapping(verifiedAlternatives.front().execution);
      if (!firstQualityIncomplete->candidate && !fallback)
        return invalid("bounded-quality incomplete result has no candidate");
      const ArtifactRootReference &candidate =
          firstQualityIncomplete->candidate ? *firstQualityIncomplete->candidate
                                            : *fallback;
      auto owner = executionOwner(candidate);
      if (!owner)
        return owner.takeError();
      return finish(
          std::move(verifiedAlternatives[*owner].execution), std::nullopt,
          std::nullopt,
          jointDesignQualityDisposition(firstQualityIncomplete->reason),
          candidate, !deadlineObserved);
    }
    if (firstQualityIncomplete || boundedQualitySearchIncomplete ||
        deadlineObserved) {
      std::optional<ArtifactRootReference> candidate =
          firstQualityIncomplete ? firstQualityIncomplete->candidate
                                 : std::nullopt;
      if (!candidate && !candidates.empty())
        candidate = candidates.front();
      if (!candidate)
        candidate = firstMapping(verifiedAlternatives.front().execution);
      if (!candidate)
        return invalid("bounded-quality incomplete result has no candidate");
      auto owner = executionOwner(*candidate);
      if (!owner)
        return owner.takeError();
      return finish(
          std::move(verifiedAlternatives[*owner].execution), std::nullopt,
          std::nullopt,
          firstQualityIncomplete
              ? jointDesignQualityDisposition(firstQualityIncomplete->reason)
              : JointDesignQualityDisposition::ProofNotEstablished,
          *candidate, false);
    }
    llvm::sort(candidates, artifactRootReferenceLess);
    auto candidateSet =
        CandidateSet::get(mapping::mappingArtifactSchema, candidates);
    if (!candidateSet)
      return candidateSet.takeError();
    auto pareto =
        applyCandidateSelection(*candidateSet, candidates, objectives,
                                ParetoSelection{quality.paretoDimensions},
                                quality.objectiveProgram.get());
    if (!pareto)
      return pareto.takeError();
    auto selected =
        applyCandidateSelection(*candidateSet, *pareto, objectives,
                                TopKSelection{quality.finalTotalOrdering, 1},
                                quality.objectiveProgram.get());
    if (!selected)
      return selected.takeError();
    if (selected->size() != 1)
      return invalid("bounded-quality selection did not produce one winner");
    for (VerifiedAlternative &alternative : verifiedAlternatives) {
      const std::vector<ArtifactRootReference> roots =
          mappingRoots(alternative.execution);
      if (llvm::is_contained(roots, selected->front()))
        return finish(std::move(alternative.execution), alternative.planOrdinal,
                      selected->front(),
                      JointDesignQualityDisposition::Complete, std::nullopt,
                      true);
    }
    return invalid("bounded-quality winner has no verified execution owner");
  }
  if (firstIncomplete)
    return finish(std::move(*firstIncomplete), std::nullopt, std::nullopt,
                  JointDesignQualityDisposition::NotRequested, std::nullopt,
                  !deadlineObserved);
  if (!lastNoFeasible)
    return invalid("hardware reopen produced no terminal execution");
  return finish(std::move(*lastNoFeasible), std::nullopt, std::nullopt,
                JointDesignQualityDisposition::NotRequested, std::nullopt,
                !deadlineObserved);
}

} // namespace loom::dse

#include "JointHardwareReopenInternal.h"

#include "JointHardwareReopenExecution.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/MappingDebugLog.h"
#include "DSE/JointMappingMigration.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/STLExtras.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse::joint_reopen_detail {
namespace {

/// Shares of the remaining parent slice when a chain reserves a retreat. A
/// Hall deficit only reports that a cover was not admitted, so its closure is
/// the speculative supply and takes one share. The retreat consumes a
/// shortfall a Mapping actually reached transport to report, so it is the
/// evidenced repair and keeps the rest.
constexpr std::uint64_t hallClosureShareDivisor = 3;

} // namespace

llvm::Expected<std::optional<dse::JointDesignExecution>>
tryHardwareFeedbackReopen(
    const JointDesignPolicy &policy, const JointDesignExplorationPlan &plan,
    const dse::JointDesignExecution &failedExecution,
    std::optional<dse::JointDesignExecution> &lastFailedExecution,
    std::uint64_t planOrdinal,
    std::vector<dse::JointDesignAttemptRecord> &attemptRecords,
    dse::JointDesignExecutionSummary &accounting,
    std::vector<JointDesignInvocationManifestReference> &encounteredInvocations,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    std::optional<ArtifactRootReference> hardwarePromotionParentSystem,
    const PlanExecutionPolicy *executionPolicy) {
  const auto saturatingAdd = [](std::uint64_t &target, std::uint64_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max() - target)
      target = std::numeric_limits<std::uint64_t>::max();
    else
      target += value;
  };
  if (policy.maximumSystemFrontier() <= 1)
    return std::optional<dse::JointDesignExecution>{};
  const PlanExecutionPolicy &effectiveExecutionPolicy =
      executionPolicy ? *executionPolicy : request.executionPolicy;
  if (plan.frontier.systemFrontier.size() != 1 ||
      plan.frontier.softwareFrontier.size() != 1)
    return invalid("application hardware reopen requires one exact pair");
  const dse::JointSoftwareScope &software =
      plan.frontier.softwareFrontier.front();
  auto reopenPolicy = dse::JointDesignPolicy::get(
      1, 1, 1, policy.maximumTechMappingsPerModule(),
      policy.maximumSpatialMappingsPerPair());
  if (!reopenPolicy)
    return reopenPolicy.takeError();

  ResolvedConfig currentConfig = plan.resolvedConfig;
  currentConfig.dse.planNodes.clear();
  const std::uint64_t parentContexts =
      currentConfig.hardwareTarget.parameters.temporalResidentContexts;
  const std::uint64_t parentGateways =
      currentConfig.hardwareTarget.parameters.gatewayCount;
  const std::uint64_t parentAccCores =
      currentConfig.hardwareTarget.parameters.accCoreCount;
  const dse::JointDesignExecution *currentFailure = &failedExecution;
  const dse::JointDesignExplorationPlan *currentPlan = &plan;
  const bool parentHasNoMappingFrontier = mappingCount(failedExecution) == 0;
  std::optional<dse::JointDesignExecution> latestFailed;
  std::optional<dse::JointDesignExplorationPlan> latestFailedPlan;
  std::optional<std::vector<ArtifactRootReference>> reusableSpatialMappings;
  std::vector<JointDesignInvocationManifestReference> supportingInvocations;
  const auto retainObservedInvocation =
      [&](const JointDesignInvocationManifestReference &reference)
      -> llvm::Error {
    if (llvm::Error error = retainJointDesignInvocationManifest(
            supportingInvocations, reference))
      return error;
    return retainJointDesignInvocationManifest(encounteredInvocations,
                                               reference);
  };
  const auto retainObservedExecution =
      [&](const JointDesignExecution &value) -> llvm::Error {
    if (value.invocationManifest())
      if (llvm::Error error =
              retainObservedInvocation(*value.invocationManifest()))
        return error;
    for (const JointDesignInvocationManifestReference &reference :
         value.supportingInvocationManifests())
      if (llvm::Error error = retainObservedInvocation(reference))
        return error;
    return llvm::Error::success();
  };
  const auto attachSupportingInvocations =
      [&](JointDesignExecution &value) -> llvm::Error {
    return attachJointDesignSupportingInvocationManifests(
        value, supportingInvocations);
  };
  if (llvm::Error error = retainObservedExecution(failedExecution))
    return error;
  // The relation half of the continuation proof is owned by the Hall feedback
  // owner and shared with the qualification search; the chain adds only which
  // supply it offered, because a changed supply kind is progress it has not
  // measured yet.
  struct HallProgressObservation final {
    TechMappingComputeContextHallProgress progress;
    std::optional<TechMappingComputeContextGrowthDirection> direction;
  };
  std::optional<HallProgressObservation> previousHallProgress;
  // Compute-context supply evidence for this chain. The structurally local,
  // atomic instruction-store closure is the first supply. A probe on it that
  // publishes no Mapping withdraws the preference, which makes the growth
  // owner offer the Spatial FU occurrence supply instead. That supply reopens
  // every Mapping layer and rebuilds the Module, so the chain admits exactly
  // one such probe: repeating it would multiply the invocation's mapping cost.
  bool preferTemporalInstructionStore = true;
  bool spatialFuGrowthProbeConsumed = false;
  // Exactly one probe of a chain reserves a retreat share. Reserving again on
  // every later probe would shrink the window geometrically and spend the
  // parent slice on probes too small to finish.
  bool retreatShareReserved = false;
  const std::uint64_t candidateLimit =
      request.stoppingPolicy == JointDesignStoppingPolicy::BoundedQuality &&
              request.boundedQuality
          ? request.boundedQuality->maximumHardwareRepairProbes
          : policy.maximumSystemFrontier() - 1;
  saturatingAdd(accounting.hardwareRepairProbeLimit, candidateLimit);
  for (std::uint64_t candidateOrdinal = 0; candidateOrdinal != candidateLimit;
       ++candidateOrdinal) {
    if (dispatchDeadlineReached(effectiveExecutionPolicy))
      break;
    auto feedback = selectMappingHardwareFeedback(*currentFailure, artifacts);
    if (!feedback)
      return feedback.takeError();
    if (!*feedback)
      break;
    const auto *techObservation =
        std::get_if<TechHardwareFeedbackObservation>(&**feedback);
    const auto *systemObservation =
        std::get_if<SystemHardwareFeedbackObservation>(&**feedback);
    // The growth owner chooses the supply direction from the exact observed
    // relation, so the direction is derived before the funnel decides whether
    // this observation repeats the previous one.
    llvm::Expected<std::optional<HardwareRecipeGrowth>> growth =
        (request.spectrumEndpoint != PreMappingSpectrumEndpoint::Automatic &&
         parentHasNoMappingFrontier && candidateOrdinal == 0 &&
         techObservation && techObservation->feedback.deficit() > 1)
            ? deriveUniformTechHardwareRecipeGrowth(currentConfig,
                                                    *techObservation, artifacts)
            : deriveHardwareRecipeGrowth(currentConfig, **feedback, artifacts,
                                         preferTemporalInstructionStore);
    if (!growth)
      return growth.takeError();
    if (!*growth) {
      ++accounting.hardwareRepairProbesRejected;
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
          mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "hardware_repair_funnel";
            fields["closure_status"] = "unsupported";
            fields["reason"] = "observed_feedback_admits_no_growth";
            fields["candidate_ordinal"] = candidateOrdinal;
            fields["diagnostic"] =
                "the observed compute-context relation has neither a Temporal "
                "context supply nor a closing Spatial FU occurrence";
          });
      break;
    }
    const bool spatialFuGrowthProbe =
        (*growth)->computeContextGrowthDirection ==
        TechMappingComputeContextGrowthDirection::SpatialFuOccurrence;
    if (spatialFuGrowthProbe) {
      spatialFuGrowthProbeConsumed = true;
      preferTemporalInstructionStore = true;
    }
    // A Temporal instruction-store closure child can cover its graphs at Tech
    // level and still exhaust route closure, and its own Spatial or System
    // feedback is then the next typed alternative. A closure probe that
    // consumed the whole parent slice leaves that alternative untried, so it
    // runs under a share of the remaining window and the retreat probe keeps
    // the rest. The invocation deadline is unchanged; only this probe's local
    // slice moves.
    //
    // The Spatial FU occurrence direction holds no share. It is taken only
    // when one decision closes the complete relation, and the owner offers it
    // either after instruction-store growth already failed or because the
    // relation admits no Temporal supply at all. In both cases there is no
    // cheaper alternative left to hold budget for, so starving it would only
    // lose the one supply the relation has.
    const bool temporalClosureProbe =
        (*growth)->computeContextGrowthDirection ==
        TechMappingComputeContextGrowthDirection::TemporalInstructionStore;
    const bool reserveRetreatShare = temporalClosureProbe &&
                                     !retreatShareReserved &&
                                     candidateOrdinal + 1 != candidateLimit;
    if (reserveRetreatShare)
      retreatShareReserved = true;
    auto probeExecutionPolicy =
        reserveRetreatShare
            ? fairRemainingPlanPolicy(effectiveExecutionPolicy,
                                      hallClosureShareDivisor, 0)
            : llvm::Expected<PlanExecutionPolicy>(effectiveExecutionPolicy);
    if (!probeExecutionPolicy)
      return probeExecutionPolicy.takeError();
    // A probe stopped by its own reserved share has not exhausted the parent
    // slice, so the chain retreats to the next feedback alternative instead of
    // ending with that incomplete execution.
    const auto reservedShareExpired = [&]() {
      return reserveRetreatShare &&
             !dispatchDeadlineReached(effectiveExecutionPolicy);
    };
    // A probe that reached ordinary Mapping and published no SystemMapping is
    // the evidence that more Temporal residency does not close this relation.
    const auto withdrawTemporalInstructionStorePreference = [&]() {
      if (spatialFuGrowthProbe || spatialFuGrowthProbeConsumed ||
          !preferTemporalInstructionStore)
        return;
      preferTemporalInstructionStore = false;
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = "compute_context_supply_preference_withdrawn";
            fields["candidate_ordinal"] = candidateOrdinal;
            fields["withdrawn_direction"] =
                techMappingComputeContextGrowthDirectionSpelling(
                    TechMappingComputeContextGrowthDirection::
                        TemporalInstructionStore);
            fields["diagnostic"] =
                "instruction-store growth published no Mapping; the Spatial FU "
                "occurrence supply becomes the next typed alternative";
          });
    };
    if (techObservation) {
      const HallProgressObservation currentHallProgress{
          observeTechMappingComputeContextHallProgress(
              techObservation->feedback),
          (*growth)->computeContextGrowthDirection};
      // Equal demand and context growth under an unchanged deficit means the
      // previous probe bought nothing. That is only a funnel boundary while
      // the owner keeps offering the same kind of supply: a changed growth
      // direction offers the relation a structurally different supply the
      // funnel has not measured yet.
      if (previousHallProgress &&
          currentHallProgress.direction == previousHallProgress->direction &&
          techMappingComputeContextHallGrowthStagnates(
              previousHallProgress->progress, currentHallProgress.progress)) {
        mapping_debug::emit(
            mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
            mapping_debug::Event::MappingFailure,
            [&](llvm::json::Object &fields) {
              fields["failure_scope"] = "hardware_repair_funnel";
              fields["closure_status"] = "proof_not_established";
              fields["reason"] = "hall_repair_stagnation";
              fields["diagnostic"] =
                  "typed context growth increased observed demand and "
                  "context supply equally; no alternate repair owner is "
                  "admitted";
              fields["previous_hall_demand"] =
                  previousHallProgress->progress.demand;
              fields["previous_hall_contexts"] =
                  previousHallProgress->progress.contexts;
              fields["current_hall_demand"] =
                  currentHallProgress.progress.demand;
              fields["current_hall_contexts"] =
                  currentHallProgress.progress.contexts;
              fields["hall_deficit"] = currentHallProgress.progress.deficit;
              if (currentHallProgress.direction)
                fields["compute_context_growth_direction"] =
                    techMappingComputeContextGrowthDirectionSpelling(
                        *currentHallProgress.direction);
            });
        break;
      }
      previousHallProgress = currentHallProgress;
    } else {
      previousHallProgress.reset();
    }
    ++accounting.hardwareRepairProbesPlanned;
    ++accounting.hardwareRepairProbesReserved;
    const bool accCoreOnlyGrowth = (*growth)->addedAccCores != 0 &&
                                   (*growth)->addedContexts == 0 &&
                                   (*growth)->addedGateways == 0;
    const bool typedModuleGrowth = techObservation != nullptr;
    using RefusableMaterialization =
        std::optional<HardwareRecipeMaterializationOutcome>;
    auto materialization = [&]() -> llvm::Expected<RefusableMaterialization> {
      if (!accCoreOnlyGrowth && !typedModuleGrowth) {
        auto recipe =
            materializeHardwareRecipeGrowth(std::move(**growth), evidence,
                                            request, scheduler, artifacts,
                                            blobs);
        if (!recipe)
          return recipe.takeError();
        return RefusableMaterialization(std::move(*recipe));
      }
      if (accCoreOnlyGrowth) {
        auto core = materializeTypedAccCoreGrowth(std::move(**growth),
                                                  artifacts, blobs);
        if (!core)
          return core.takeError();
        return RefusableMaterialization(
            HardwareRecipeMaterializationOutcome{std::move(*core)});
      }
      auto candidate = materializeTypedModuleSystemGrowth(
          std::move(**growth), currentPlan->frontier.systemFrontier.front(),
          artifacts, blobs);
      if (!candidate)
        return candidate.takeError();
      if (!*candidate)
        return RefusableMaterialization();
      return RefusableMaterialization(
          HardwareRecipeMaterializationOutcome{std::move(**candidate)});
    }();
    if (!materialization)
      return materialization.takeError();
    if (!*materialization) {
      ++accounting.hardwareRepairProbesRejected;
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "hardware_repair_funnel";
            fields["closure_status"] = "unsupported";
            fields["reason"] = "typed_growth_decision_published_no_child";
            fields["candidate_ordinal"] = candidateOrdinal;
            fields["diagnostic"] =
                "the ADG Builder refused the typed growth decision; no child "
                "Module was published";
            if (spatialFuGrowthProbe)
              fields["retreat_direction"] =
                  techMappingComputeContextGrowthDirectionSpelling(
                      TechMappingComputeContextGrowthDirection::
                          TemporalInstructionStore);
          });
      // A refused Spatial FU occurrence decision costs no Mapping work, so the
      // chain retreats to the atomic instruction-store closure on the same
      // parent instead of ending with an empty child.
      if (spatialFuGrowthProbe)
        continue;
      break;
    }
    if (const auto *incomplete =
            std::get_if<IncompleteHardwareRecipeMaterialization>(
                &**materialization)) {
      if (llvm::Error error =
              retainObservedInvocation(incomplete->constructionInvocation))
        return error;
      break;
    }
    auto *system =
        &std::get<MaterializedHardwareCandidate>(**materialization);
    if (system->constructionInvocation)
      if (llvm::Error error =
              retainObservedInvocation(*system->constructionInvocation))
        return error;
    auto timing = normalizedTimingProfiles(system->reference, artifacts);
    if (!timing)
      return timing.takeError();
    std::optional<JointMappingRebaseResult> rebased;
    const auto mappingReuseStart = std::chrono::steady_clock::now();
    if (!accCoreOnlyGrowth) {
      if (mappingCount(*currentFailure) == 0) {
        rebased = JointMappingRebaseResult{
            {},
            {},
            {{JointMappingRebaseFailureReason::MissingParentFrontier,
              std::nullopt, "parent execution has no finalized Mapping"}},
            JointMappingReuseDisposition::ColdFallback};
      } else {
        auto projected = rebaseJointMappingFrontier(
            *currentPlan, *currentFailure, system->reference,
            system->moduleCorrespondences,
            system->mappingImpact ? llvm::ArrayRef<HardwareImpactProjection>(
                                        *system->mappingImpact)
                                  : llvm::ArrayRef<HardwareImpactProjection>(),
            artifacts);
        if (!projected)
          return projected.takeError();
        rebased = std::move(*projected);
      }
      saturatingAdd(
          accounting.incrementalReopenWallTimeNanoseconds,
          static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - mappingReuseStart)
                  .count()));
      saturatingAdd(accounting.preservedTechMappings,
                    rebased->accounting.preservedTechMappings);
      saturatingAdd(accounting.preservedSpatialMappings,
                    rebased->accounting.preservedSpatialMappings);
      saturatingAdd(accounting.repairedTechMappings,
                    rebased->accounting.repairedTechMappings);
      saturatingAdd(accounting.repairedSpatialMappings,
                    rebased->accounting.repairedSpatialMappings);
      saturatingAdd(accounting.invalidatedTechMappings,
                    rebased->accounting.invalidatedTechMappings);
      saturatingAdd(accounting.invalidatedSpatialMappings,
                    rebased->accounting.invalidatedSpatialMappings);
      saturatingAdd(accounting.parentTechDecisions,
                    rebased->accounting.parentTechDecisions);
      saturatingAdd(accounting.parentSpatialDecisions,
                    rebased->accounting.parentSpatialDecisions);
      saturatingAdd(accounting.preservedTechDecisions,
                    rebased->accounting.preservedTechDecisions);
      saturatingAdd(accounting.preservedSpatialDecisions,
                    rebased->accounting.preservedSpatialDecisions);
      saturatingAdd(accounting.reopenedTechDecisions,
                    rebased->accounting.reopenedTechDecisions);
      saturatingAdd(accounting.reopenedSpatialDecisions,
                    rebased->accounting.reopenedSpatialDecisions);
      saturatingAdd(accounting.repairedTechDecisions,
                    rebased->accounting.repairedTechDecisions);
      saturatingAdd(accounting.repairedSpatialDecisions,
                    rebased->accounting.repairedSpatialDecisions);
      saturatingAdd(accounting.invalidationRootCount,
                    rebased->accounting.invalidationRootCount);
      saturatingAdd(accounting.invalidationConeDecisionCount,
                    rebased->accounting.invalidationConeDecisionCount);
      saturatingAdd(accounting.parentRouteNodeCount,
                    rebased->accounting.parentRouteNodeCount);
      saturatingAdd(accounting.preservedRouteNodeCount,
                    rebased->accounting.preservedRouteNodeCount);
      saturatingAdd(accounting.reopenedRouteNodeCount,
                    rebased->accounting.reopenedRouteNodeCount);
      saturatingAdd(accounting.repairedRouteNodeCount,
                    rebased->accounting.repairedRouteNodeCount);
      saturatingAdd(accounting.parentServiceLegCount,
                    rebased->accounting.parentServiceLegCount);
      saturatingAdd(accounting.preservedServiceLegCount,
                    rebased->accounting.preservedServiceLegCount);
      saturatingAdd(accounting.reopenedServiceLegCount,
                    rebased->accounting.reopenedServiceLegCount);
    }
    const JointDesignMappingSeed *mappingSeed =
        rebased && (!rebased->seed.techMappings.empty() ||
                    !rebased->seed.spatialMappings.empty())
            ? &rebased->seed
            : nullptr;
    const auto planBuildStart = std::chrono::steady_clock::now();
    auto reopenPlanResult = dse::buildJointDesignExplorationPlan(
        {{software.workloads}, {system->reference}}, *timing, *reopenPolicy,
        system->config, artifacts, mappingSeed,
        currentPlan->systemBindingPartitions);
    const std::uint64_t planBuildNanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - planBuildStart)
            .count());
    if (!reopenPlanResult) {
      if (mappingSeed)
        saturatingAdd(accounting.incrementalReopenWallTimeNanoseconds,
                      planBuildNanoseconds);
      else
        saturatingAdd(accounting.coldReopenWallTimeNanoseconds,
                      planBuildNanoseconds);
      return reopenPlanResult.takeError();
    }
    std::optional<JointDesignExplorationPlan> reopenPlan(
        std::move(*reopenPlanResult));
    if (mappingSeed)
      saturatingAdd(accounting.incrementalReopenWallTimeNanoseconds,
                    planBuildNanoseconds);
    else
      saturatingAdd(accounting.coldReopenWallTimeNanoseconds,
                    planBuildNanoseconds);

    if (typedModuleGrowth) {
      const auto gateStart = std::chrono::steady_clock::now();
      auto gate = executeTechGate(*reopenPlan, evidence, request, scheduler,
                                  artifacts, blobs, *probeExecutionPolicy);
      const std::uint64_t gateNanoseconds = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - gateStart)
              .count());
      if (!gate)
        return gate.takeError();
      if (llvm::Error error = retainObservedExecution(gate->execution))
        return error;
      saturatingAdd(accounting.techMappingInvocationCount,
                    gate->execution.summary.techMappingInvocationCount);
      saturatingAdd(accounting.techMappingDispatchCount,
                    gate->execution.summary.techMappingDispatchCount);
      saturatingAdd(accounting.techMappingJournalReplayCount,
                    gate->execution.summary.techMappingJournalReplayCount);
      if (mappingSeed)
        saturatingAdd(accounting.incrementalReopenWallTimeNanoseconds,
                      gateNanoseconds);
      else
        saturatingAdd(accounting.coldReopenWallTimeNanoseconds,
                      gateNanoseconds);
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = "hardware_reopen_tech_gate";
            fields["candidate_ordinal"] = candidateOrdinal;
            fields["tech_mapping_count"] = gate->techMappings.size();
            fields["covers_required_graphs"] = gate->coversRequiredGraphs;
            fields["downstream_mapping_dispatched"] =
                gate->coversRequiredGraphs;
            fields["wall_time_ns"] = gateNanoseconds;
          });
      if (!gate->coversRequiredGraphs) {
        if (llvm::Error error = recordJointAttempt(
                attemptRecords, planOrdinal, system->reference, gate->execution,
                hardwarePromotionParentSystem))
          return std::move(error);
        ++accounting.hardwareRepairProbesConsumed;
        if (const auto *incomplete = std::get_if<IncompleteDsePlanExecution>(
                &gate->execution.planExecution);
            incomplete && incomplete->executionStopped() &&
            !reservedShareExpired()) {
          if (llvm::Error error = attachSupportingInvocations(gate->execution))
            return std::move(error);
          return std::optional<dse::JointDesignExecution>{
              std::move(gate->execution)};
        }
        // A Spatial FU occurrence child whose TechMapping covers nothing is
        // evidence about that supply, not about the relation. Retreat to the
        // atomic instruction-store closure on the same parent.
        if (spatialFuGrowthProbe) {
          mapping_debug::emit(
              mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
              mapping_debug::Event::MappingFailure,
              [&](llvm::json::Object &fields) {
                fields["failure_scope"] = "hardware_repair_funnel";
                fields["closure_status"] = "proof_not_established";
                fields["reason"] = "spatial_fu_growth_child_covers_no_graph";
                fields["candidate_ordinal"] = candidateOrdinal;
                fields["retreat_direction"] =
                    techMappingComputeContextGrowthDirectionSpelling(
                        TechMappingComputeContextGrowthDirection::
                            TemporalInstructionStore);
              });
          continue;
        }
        currentConfig = system->config;
        latestFailed = std::move(gate->execution);
        latestFailedPlan = std::move(*reopenPlan);
        currentFailure = &*latestFailed;
        currentPlan = &*latestFailedPlan;
        break;
      }

      JointDesignMappingSeed gateSeed;
      if (rebased)
        gateSeed = rebased->seed;
      std::vector<ArtifactRootReference> gateTechCandidates =
          gateSeed.techMappings;
      gateTechCandidates.insert(gateTechCandidates.end(),
                                gate->techMappings.begin(),
                                gate->techMappings.end());
      auto boundedTechMappings = boundTechMappingFrontierForRepair(
          gateTechCandidates, policy.maximumTechMappingsPerModule(), artifacts);
      if (!boundedTechMappings) {
        ++accounting.hardwareRepairProbesRejected;
        mapping_debug::emit(
            mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
            mapping_debug::Event::MappingFailure,
            [&](llvm::json::Object &fields) {
              fields["failure_scope"] = "hardware_repair_funnel";
              fields["closure_status"] = "unsupported";
              fields["reason"] = "tech_frontier_bound_cannot_preserve_coverage";
              fields["diagnostic"] =
                  llvm::toString(boundedTechMappings.takeError());
            });
        // The Tech gate is a real DSE occurrence even when its bounded
        // frontier cannot preserve graph coverage. Keep it as the terminal
        // typed failure so callers retain its manifest and ancestry instead
        // of silently falling back to the original parent attempt.
        currentConfig = system->config;
        latestFailed = std::move(gate->execution);
        latestFailedPlan = std::move(*reopenPlan);
        currentFailure = &*latestFailed;
        currentPlan = &*latestFailedPlan;
        break;
      }
      gateSeed.techMappings = std::move(*boundedTechMappings);
      canonicalizeRoots(gateSeed.techMappings);
      canonicalizeRoots(gateSeed.spatialMappings);
      const auto gatedPlanStart = std::chrono::steady_clock::now();
      auto gatedPlanResult = dse::buildJointDesignExplorationPlan(
          {{software.workloads}, {system->reference}}, *timing, *reopenPolicy,
          system->config, artifacts, &gateSeed,
          currentPlan->systemBindingPartitions);
      const std::uint64_t gatedPlanNanoseconds = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - gatedPlanStart)
              .count());
      if (!gatedPlanResult)
        return gatedPlanResult.takeError();
      if (mappingSeed)
        saturatingAdd(accounting.incrementalReopenWallTimeNanoseconds,
                      gatedPlanNanoseconds);
      else
        saturatingAdd(accounting.coldReopenWallTimeNanoseconds,
                      gatedPlanNanoseconds);
      reopenPlan = std::move(*gatedPlanResult);
    }
    if (accCoreOnlyGrowth) {
      if (!reusableSpatialMappings) {
        auto resolved =
            resolveJointSpatialMappingFrontier(*currentPlan, *currentFailure);
        if (!resolved)
          return resolved.takeError();
        reusableSpatialMappings = std::move(*resolved);
      }
      if (llvm::Error error = bindImmutableSpatialMappingFrontier(
              *reopenPlan, *reusableSpatialMappings, artifacts))
        return std::move(error);
      if (!systemObservation || !system->executionBindingCorrespondence)
        return invalid("typed AddAccCore reopen lost its Mapping checkpoint or "
                       "parent-to-child correspondence");
      auto migrationContext = deriveSystemMappingMigrationContext(*reopenPlan);
      if (!migrationContext)
        return migrationContext.takeError();
      auto migrationSeed = pnr::finalizeSystemMappingCheckpointMigrationSeed(
          systemObservation->feedback.executionBindingCheckpoint(),
          *system->executionBindingCorrespondence, *migrationContext,
          systemObservation->feedback.witnessAccCore(), artifacts);
      if (!migrationSeed)
        return migrationSeed.takeError();
      if (llvm::Error error = bindCheckpointSystemMappingMigrationSeed(
              *reopenPlan, migrationSeed->reference(), artifacts))
        return std::move(error);
    } else {
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = mappingSeed ? "rebase_mapping_frontier"
                                              : "mapping_rebase_cold_fallback";
            fields["typed_impact_projection_present"] =
                system->mappingImpact.has_value();
            if (system->mappingImpact) {
              fields["typed_impact_locality"] =
                  static_cast<std::uint64_t>(system->mappingImpact->locality);
              fields["typed_impact_tech_kind"] =
                  static_cast<std::uint64_t>(system->mappingImpact->tech.kind);
              fields["typed_impact_spatial_kind"] = static_cast<std::uint64_t>(
                  system->mappingImpact->spatial.kind);
            }
            fields["seed_source"] = mappingSeed ? "rebased_mapping" : "cold";
            fields["mapping_reuse_disposition"] =
                jointMappingReuseDispositionSpelling(rebased->disposition);
            if (system->mappingImpact) {
              fields["hardware_mutation_family"] =
                  hardwareMutationFamilySpelling(system->mappingImpact->family);
              fields["hardware_mutation_locality"] =
                  hardwareMutationLocalitySpelling(
                      system->mappingImpact->locality);
              fields["hardware_tech_impact"] =
                  hardwareMappingImpactKindSpelling(
                      system->mappingImpact->tech.kind);
              fields["hardware_spatial_impact"] =
                  hardwareMappingImpactKindSpelling(
                      system->mappingImpact->spatial.kind);
              fields["hardware_system_impact"] =
                  hardwareMappingImpactKindSpelling(
                      system->mappingImpact->system.kind);
            } else {
              // Generic recipe growth is an admitted hardware child, but it
              // has no typed parent correspondence. Keep that fact explicit;
              // the downstream cold verifier remains the legality owner.
              fields["hardware_mutation_family"] = "unprojected_recipe_growth";
              fields["hardware_mutation_locality"] = "global_reopen";
              fields["hardware_tech_impact"] = "unknown";
              fields["hardware_spatial_impact"] = "unknown";
              fields["hardware_system_impact"] = "unknown";
            }
            fields["parent_tech_mappings"] =
                rebased->accounting.parentTechMappings;
            fields["parent_spatial_mappings"] =
                rebased->accounting.parentSpatialMappings;
            fields["preserved_tech_mappings"] =
                rebased->accounting.preservedTechMappings;
            fields["preserved_spatial_mappings"] =
                rebased->accounting.preservedSpatialMappings;
            fields["repaired_tech_mappings"] =
                rebased->accounting.repairedTechMappings;
            fields["repaired_spatial_mappings"] =
                rebased->accounting.repairedSpatialMappings;
            fields["invalidated_tech_mappings"] =
                rebased->accounting.invalidatedTechMappings;
            fields["invalidated_spatial_mappings"] =
                rebased->accounting.invalidatedSpatialMappings;
          });
      for (const JointMappingRebaseFailure &failure : rebased->failures) {
        mapping_debug::emit(
            mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
            mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
              fields["operation"] = "mapping_rebase_fallback";
              fields["fallback_reason"] =
                  jointMappingRebaseFailureReasonSpelling(failure.reason);
              fields["diagnostic"] = failure.diagnostic;
              if (failure.parent)
                fields["parent_mapping"] =
                    formatArtifactIdentityHex(failure.parent->artifact);
            });
      }
      reusableSpatialMappings.reset();
    }
    const auto pnrStart = std::chrono::steady_clock::now();
    auto execution =
        executeJointPlan(*reopenPlan, evidence, request, scheduler, artifacts,
                         blobs, &*probeExecutionPolicy);
    const std::uint64_t pnrNanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - pnrStart)
            .count());
    if (mappingSeed)
      saturatingAdd(accounting.incrementalReopenWallTimeNanoseconds,
                    pnrNanoseconds);
    else
      saturatingAdd(accounting.coldReopenWallTimeNanoseconds, pnrNanoseconds);
    if (!execution)
      return execution.takeError();
    if (llvm::Error error = retainObservedExecution(*execution))
      return error;
    ++accounting.hardwareRepairProbesConsumed;
    saturatingAdd(accounting.techMappingInvocationCount,
                  execution->summary.techMappingInvocationCount);
    saturatingAdd(accounting.spatialPnrInvocationCount,
                  execution->summary.spatialPnrInvocationCount);
    saturatingAdd(accounting.systemPnrInvocationCount,
                  execution->summary.systemPnrInvocationCount);
    saturatingAdd(accounting.techMappingDispatchCount,
                  execution->summary.techMappingDispatchCount);
    saturatingAdd(accounting.spatialPnrDispatchCount,
                  execution->summary.spatialPnrDispatchCount);
    saturatingAdd(accounting.systemPnrDispatchCount,
                  execution->summary.systemPnrDispatchCount);
    saturatingAdd(accounting.techMappingJournalReplayCount,
                  execution->summary.techMappingJournalReplayCount);
    saturatingAdd(accounting.spatialPnrJournalReplayCount,
                  execution->summary.spatialPnrJournalReplayCount);
    saturatingAdd(accounting.systemPnrJournalReplayCount,
                  execution->summary.systemPnrJournalReplayCount);
    saturatingAdd(accounting.parentThreadBindingCount,
                  execution->summary.parentThreadBindingCount);
    saturatingAdd(accounting.preservedThreadBindingCount,
                  execution->summary.preservedThreadBindingCount);
    saturatingAdd(accounting.reopenedThreadBindingCount,
                  execution->summary.reopenedThreadBindingCount);
    saturatingAdd(accounting.parentGraphBindingCount,
                  execution->summary.parentGraphBindingCount);
    saturatingAdd(accounting.preservedGraphBindingCount,
                  execution->summary.preservedGraphBindingCount);
    saturatingAdd(accounting.reopenedGraphBindingCount,
                  execution->summary.reopenedGraphBindingCount);
    saturatingAdd(accounting.parentResourceUseCount,
                  execution->summary.parentResourceUseCount);
    saturatingAdd(accounting.preservedResourceUseCount,
                  execution->summary.preservedResourceUseCount);
    saturatingAdd(accounting.reopenedResourceUseCount,
                  execution->summary.reopenedResourceUseCount);
    saturatingAdd(accounting.parentServiceRealizationCount,
                  execution->summary.parentServiceRealizationCount);
    saturatingAdd(accounting.preservedServiceRealizationCount,
                  execution->summary.preservedServiceRealizationCount);
    saturatingAdd(accounting.reopenedServiceRealizationCount,
                  execution->summary.reopenedServiceRealizationCount);
    if (llvm::Error error =
            recordJointAttempt(attemptRecords, planOrdinal, system->reference,
                               *execution, hardwarePromotionParentSystem))
      return std::move(error);
    const std::size_t systemMappingCount = mappingCount(*execution);
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "hardware_reopen_mapping_attempt";
          fields["candidate_ordinal"] = candidateOrdinal;
          fields["resized_instruction_store_count"] =
              system->resizedInstructionStoreCount;
          fields["maximum_instruction_store_capacity"] =
              system->maximumInstructionStoreCapacity;
          fields["added_temporal_contexts"] =
              system->resultingContexts - parentContexts;
          fields["temporal_resident_contexts"] = system->resultingContexts;
          fields["added_gateways"] = system->resultingGateways - parentGateways;
          fields["gateway_count"] = system->resultingGateways;
          fields["added_acc_cores"] =
              system->resultingAccCores - parentAccCores;
          fields["acc_core_count"] = system->resultingAccCores;
          if (system->computeContextGrowthDirection)
            fields["compute_context_growth_direction"] =
                techMappingComputeContextGrowthDirectionSpelling(
                    *system->computeContextGrowthDirection);
          fields["added_spatial_fu_occurrences"] =
              system->addedSpatialFuOccurrences;
          fields["added_spatial_fu_contexts"] = system->addedSpatialFuContexts;
          fields["spatial_fu_context_supply_bound"] =
              system->spatialFuContextSupplyBound;
          fields["spatial_fu_unclosed_deficit"] =
              system->spatialFuUnclosedDeficit;
          fields["system"] =
              formatArtifactIdentityHex(system->reference.artifact);
          fields["parent_system"] = formatArtifactIdentityHex(
              currentPlan->frontier.systemFrontier.front().artifact);
          fields["system_mapping_count"] = systemMappingCount;
        });
    if (systemMappingCount != 0) {
      if (llvm::Error error = attachSupportingInvocations(*execution))
        return std::move(error);
      return std::optional<dse::JointDesignExecution>{std::move(*execution)};
    }
    if (const auto *incomplete =
            std::get_if<IncompleteDsePlanExecution>(&execution->planExecution);
        incomplete && incomplete->executionStopped()) {
      if (!reservedShareExpired()) {
        if (llvm::Error error = attachSupportingInvocations(*execution))
          return std::move(error);
        return std::optional<dse::JointDesignExecution>{std::move(*execution)};
      }
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = "hardware_repair_probe_share_expired";
            fields["candidate_ordinal"] = candidateOrdinal;
            fields["diagnostic"] =
                "the Hall closure probe consumed its reserved share; the "
                "chain retreats to the child's own feedback alternative";
          });
    }
    withdrawTemporalInstructionStorePreference();

    currentConfig = std::move(system->config);
    latestFailed = std::move(*execution);
    latestFailedPlan = std::move(*reopenPlan);
    currentFailure = &*latestFailed;
    currentPlan = &*latestFailedPlan;
  }
  if (latestFailed) {
    if (llvm::Error error = attachSupportingInvocations(*latestFailed))
      return std::move(error);
    lastFailedExecution = std::move(*latestFailed);
  }
  return std::optional<dse::JointDesignExecution>{};
}

} // namespace loom::dse::joint_reopen_detail

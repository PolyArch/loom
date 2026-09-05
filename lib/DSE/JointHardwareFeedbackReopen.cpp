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
  struct HallProgressObservation final {
    std::uint64_t deficit = 0;
    std::uint64_t demand = 0;
    std::uint64_t contexts = 0;
  };
  std::optional<HallProgressObservation> previousHallProgress;
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
    auto techObservation =
        selectTechHardwareFeedback(*currentFailure, artifacts);
    if (!techObservation)
      return techObservation.takeError();
    auto spatialObservation =
        selectSpatialHardwareFeedback(*currentFailure, artifacts);
    if (!spatialObservation)
      return spatialObservation.takeError();
    auto systemObservation =
        selectSystemHardwareFeedback(*currentFailure, artifacts);
    if (!systemObservation)
      return systemObservation.takeError();
    if (!*techObservation && !*spatialObservation && !*systemObservation)
      break;
    if (*techObservation) {
      const HallProgressObservation currentHallProgress{
          (*techObservation)->feedback.deficit(),
          (*techObservation)->feedback.hallDemandCount(),
          (*techObservation)->feedback.hallContextValueCount()};
      if (previousHallProgress &&
          currentHallProgress.deficit == previousHallProgress->deficit &&
          currentHallProgress.demand > previousHallProgress->demand &&
          currentHallProgress.contexts > previousHallProgress->contexts &&
          currentHallProgress.demand - previousHallProgress->demand ==
              currentHallProgress.contexts - previousHallProgress->contexts) {
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
              fields["previous_hall_demand"] = previousHallProgress->demand;
              fields["previous_hall_contexts"] = previousHallProgress->contexts;
              fields["current_hall_demand"] = currentHallProgress.demand;
              fields["current_hall_contexts"] = currentHallProgress.contexts;
              fields["hall_deficit"] = currentHallProgress.deficit;
            });
        break;
      }
      previousHallProgress = currentHallProgress;
    }
    ++accounting.hardwareRepairProbesPlanned;
    ++accounting.hardwareRepairProbesReserved;

    llvm::Expected<HardwareRecipeGrowth> growth =
        (request.spectrumEndpoint != PreMappingSpectrumEndpoint::Automatic &&
         parentHasNoMappingFrontier && candidateOrdinal == 0 &&
         techObservation && *techObservation &&
         (*techObservation)->feedback.deficit() > 1)
            ? deriveUniformTechHardwareRecipeGrowth(
                  currentConfig, **techObservation, artifacts)
            : deriveHardwareRecipeGrowth(currentConfig, *techObservation,
                                         *spatialObservation,
                                         *systemObservation, artifacts);
    if (!growth)
      return growth.takeError();
    const bool accCoreOnlyGrowth = growth->addedAccCores != 0 &&
                                   growth->addedContexts == 0 &&
                                   growth->addedGateways == 0;
    const bool typedModuleGrowth =
        *techObservation && !*spatialObservation && !*systemObservation;
    llvm::Expected<MaterializedHardwareCandidate> system =
        accCoreOnlyGrowth ? materializeTypedAccCoreGrowth(std::move(*growth),
                                                          artifacts, blobs)
        : typedModuleGrowth
            ? materializeTypedModuleSystemGrowth(
                  std::move(*growth),
                  currentPlan->frontier.systemFrontier.front(), artifacts,
                  blobs)
            : materializeHardwareRecipeGrowth(std::move(*growth), evidence,
                                              request, scheduler, artifacts,
                                              blobs);
    if (!system)
      return system.takeError();
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
                                  artifacts, blobs, effectiveExecutionPolicy);
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
            incomplete && incomplete->executionStopped()) {
          if (llvm::Error error = attachSupportingInvocations(gate->execution))
            return std::move(error);
          return std::optional<dse::JointDesignExecution>{
              std::move(gate->execution)};
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
      if (!*systemObservation || !system->executionBindingCorrespondence)
        return invalid("typed AddAccCore reopen lost its Mapping checkpoint or "
                       "parent-to-child correspondence");
      auto migrationContext = deriveSystemMappingMigrationContext(*reopenPlan);
      if (!migrationContext)
        return migrationContext.takeError();
      auto migrationSeed = pnr::finalizeSystemMappingCheckpointMigrationSeed(
          (*systemObservation)->feedback.executionBindingCheckpoint(),
          *system->executionBindingCorrespondence, *migrationContext,
          (*systemObservation)->feedback.witnessAccCore(), artifacts);
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
                         blobs, &effectiveExecutionPolicy);
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
          fields["system"] =
              formatArtifactIdentityHex(system->reference.artifact);
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
      if (llvm::Error error = attachSupportingInvocations(*execution))
        return std::move(error);
      return std::optional<dse::JointDesignExecution>{std::move(*execution)};
    }

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

#include "DSE/JointHardwareReopen.h"

#include "JointHardwareReopenInternal.h"

#include "JointHardwareReopenExecution.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/MappingDebugLog.h"
#include "DSE/ExecutionJournal.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "DSE/HardwareDecision.h"
#include "DSE/JointMappingMigration.h"
#include "DSE/ProductionOwners.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "DSE/SystemCompositionCandidateGenerator.h"
#include "DSE/TechMappingHardwareFeedback.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Evidence.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SpatialMappingHardwareDemand.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingHardwareDemand.h"
#include "Mapping/Tech/TechMappingHardwareDemand.h"
#include "PnR/PnrDerivedContext.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::dse {

using namespace joint_reopen_detail;

llvm::Expected<std::vector<SpatialMicroarchitectureDecisionDomain>>
deriveSpatialCapacityHardwareReopenDomains(
    const pnr::SpatialFifoCapacitySuggestion &feedback) {
  if (feedback.logicalNets.empty() || feedback.routeAnchors.empty())
    return invalid("static FIFO capacity feedback is incomplete or outside "
                   "the hardware depth domain");
  auto domain = deriveFifoCapacityDepthDomain(
      feedback.owner, feedback.selectedCapacity, feedback.sufficientCapacity);
  if (!domain)
    return domain.takeError();
  return std::vector<SpatialMicroarchitectureDecisionDomain>{
      std::move(*domain)};
}

namespace {

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
    const PlanExecutionPolicy *executionPolicy = nullptr) {
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
  bool currentFailureIsTechGate = false;
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
      if (currentFailureIsTechGate) {
        rebased = JointMappingRebaseResult{
            {},
            {},
            {{JointMappingRebaseFailureReason::MissingParentFrontier,
              std::nullopt, "parent execution stopped at the Tech gate"}},
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
        if (llvm::Error error =
                recordJointAttempt(attemptRecords, planOrdinal,
                                   system->reference, gate->execution))
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
        currentFailureIsTechGate = true;
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
        currentFailureIsTechGate = true;
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
    if (llvm::Error error = recordJointAttempt(attemptRecords, planOrdinal,
                                               system->reference, *execution))
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
    currentFailureIsTechGate = false;
  }
  if (latestFailed) {
    if (llvm::Error error = attachSupportingInvocations(*latestFailed))
      return std::move(error);
    lastFailedExecution = std::move(*latestFailed);
  }
  return std::optional<dse::JointDesignExecution>{};
}

} // namespace

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
  const auto markHardwarePromotion = [&](std::uint64_t planOrdinal) {
    auto observation = llvm::find_if(
        hardwarePromotionObservations, [&](const auto &candidate) {
          return candidate.planOrdinal == planOrdinal;
        });
    if (observation != hardwarePromotionObservations.end())
      observation->promotedToExactMapping = true;
  };
  const auto finish =
      [&](JointDesignExecution execution,
          std::optional<std::uint64_t> selectedPlanOrdinal,
          std::optional<ArtifactRootReference> selectedMapping,
          JointDesignQualityDisposition qualityDisposition,
          std::optional<ArtifactRootReference> qualityIncompleteCandidate,
          bool declaredWorkExhausted) -> llvm::Expected<JointDesignExecution> {
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
    if (request.stoppingPolicy == JointDesignStoppingPolicy::BoundedQuality) {
      auto fair = fairBoundedQualityPlanPolicy(request.executionPolicy,
                                               hardwareFeedbackFrontier.size() -
                                                   indexedAttempt.index());
      if (!fair)
        return fair.takeError();
      feedbackExecutionPolicy.emplace(std::move(*fair));
      ++hardwareParentPromotions;
      markHardwarePromotion(attempt.planOrdinal);
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
      ++hardwareParentPromotions;
      markHardwarePromotion(parentPlanOrdinal);
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
                execution))
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

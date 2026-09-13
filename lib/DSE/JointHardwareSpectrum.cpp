//===- JointHardwareSpectrum.cpp - growth for a verified parent ----------===//
//
// The hardware spectrum of a parent whose Mapping already verified. Nothing
// here answers a refused relation: the cover closed, the routes closed, and
// the only evidence left is what the candidate measured. Two axes read that
// evidence. More accelerator cores spend the same Module on more of the
// program; one composed capability spends fewer realizations on the same
// actors, which is what a deployment short of nothing but its own routed
// latency is actually missing.
//
//===----------------------------------------------------------------------===//

#include "JointHardwareReopenInternal.h"

#include "JointHardwareReopenExecution.h"

#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "DSE/CompositeFuSupply.h"
#include "DSE/JointMappingMigration.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "PnR/System/SystemMappingMigration.h"

#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::dse::joint_reopen_detail {
namespace {

/// `value * factor` without wrapping. A saturated product only makes the
/// comparison it feeds more conservative, so no shortfall an overflow invented
/// is ever claimed.
std::uint64_t saturatingProduct(std::uint64_t value, std::uint64_t factor) {
  if (factor != 0 && value > std::numeric_limits<std::uint64_t>::max() / factor)
    return std::numeric_limits<std::uint64_t>::max();
  return value * factor;
}

/// Whether the measured mapped replay trails its own dataflow oracle by the
/// declared factor. An oracle of zero cycles measured nothing to trail.
bool observesLatencyShortfall(
    const FinalizedMappingLatencyObservation &latency) {
  if (latency.dataflowCycles == 0)
    return false;
  return saturatingProduct(latency.mappedCycles,
                           compositeFuLatencyShortfallDenominator) >=
         saturatingProduct(latency.dataflowCycles,
                           compositeFuLatencyShortfallNumerator);
}

bool observesControlDensity(const CanonicalDataflowActorCensus &census) {
  if (census.totalActors() == 0)
    return false;
  return saturatingProduct(census.controlActors,
                           compositeFuControlShareDenominator) >=
         saturatingProduct(census.totalActors(),
                           compositeFuControlShareNumerator);
}

} // namespace

llvm::Expected<FinalizedMappingHardwareSpectrum>
exploreFinalizedMappingHardwareSpectrum(
    const JointDesignPolicy &policy, const JointDesignExplorationPlan &plan,
    const JointDesignExecution &parentExecution,
    const std::optional<FinalizedMappingLatencyObservation> &latency,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const PlanExecutionPolicy *executionPolicy) {
  FinalizedMappingHardwareSpectrum result;
  if (policy.maximumSystemFrontier() <= plan.frontier.systemFrontier.size())
    return result;
  if (plan.pairOutputs.size() != 1 ||
      plan.frontier.softwareFrontier.size() != 1 ||
      plan.frontier.systemFrontier.size() != 1)
    return invalid("finalized Mapping spectrum requires one exact pair");
  auto reusableSpatialMappings =
      resolveJointSpatialMappingFrontier(plan, parentExecution);
  if (!reusableSpatialMappings)
    return reusableSpatialMappings.takeError();
  auto parentModules = projectJointDesignTargetModules(
      plan.frontier.systemFrontier.front(), artifacts);
  if (!parentModules)
    return parentModules.takeError();
  if (parentModules->empty())
    return invalid("finalized Mapping spectrum has no target Module");
  std::vector<ArtifactRootReference> targetModules = std::move(*parentModules);
  auto reopenPolicy =
      JointDesignPolicy::get(1, 1, 1, policy.maximumTechMappingsPerModule(),
                             policy.maximumSpatialMappingsPerPair());
  if (!reopenPolicy)
    return reopenPolicy.takeError();

  ArtifactRootReference currentSystem = plan.frontier.systemFrontier.front();
  ResolvedConfig currentConfig = plan.resolvedConfig;
  currentConfig.dse.planNodes.clear();
  std::optional<ArtifactRootReference> parentMapping =
      firstMapping(parentExecution);
  const JointSoftwareScope &software = plan.frontier.softwareFrontier.front();
  std::uint64_t remaining =
      policy.maximumSystemFrontier() - plan.frontier.systemFrontier.size();
  const PlanExecutionPolicy &effectiveExecutionPolicy =
      executionPolicy ? *executionPolicy : request.executionPolicy;
  // An AddAccCore child keeps its parent's imported Modules, so the parent's
  // Spatial realizations remain this chain's own. A composed capability
  // publishes a different Module and that reuse ends with it.
  bool spatialFrontierRetained = true;

  // The composed-capability axis runs at most once and before the core axis,
  // because a Module whose FU inventory is about to change makes every child
  // built from the old inventory a child of a different parent. The decision
  // is the parent's own measurement plus the census of the Dataflow it
  // deployed; the miner and the canonical capability derivation stay the only
  // owners of what the template is.
  if (remaining != 0 && !dispatchDeadlineReached(effectiveExecutionPolicy)) {
    // The measured half is free and the structural half costs one Dataflow
    // import, so a parent whose replay keeps up with its oracle never pays for
    // the census.
    const bool shortfall = latency && observesLatencyShortfall(*latency);
    std::optional<CanonicalDataflowActorCensus> census;
    if (shortfall) {
      auto counted = censusCanonicalDataflowActors(software.dataflow, artifacts);
      if (!counted)
        return counted.takeError();
      census = *counted;
    }
    const bool controlDense = census && observesControlDensity(*census);
    MinedCompositeFuSupplyOutcome supply;
    if (controlDense) {
      auto proposed = proposeMinedCompositeFuSupply(
          software.dataflow, census->controlActors,
          currentConfig.hardwareTarget.parameters, artifacts);
      if (!proposed)
        return proposed.takeError();
      supply = std::move(*proposed);
    }
    const std::optional<MinedCompositeFuProposal> &proposal = supply.proposal;
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "measured_latency_composed_supply";
          fields["dataflow_cycles"] = latency ? latency->dataflowCycles : 0;
          fields["mapped_cycles"] = latency ? latency->mappedCycles : 0;
          fields["compute_actors"] =
              census ? llvm::json::Value(census->computeActors)
                     : llvm::json::Value(nullptr);
          fields["control_actors"] =
              census ? llvm::json::Value(census->controlActors)
                     : llvm::json::Value(nullptr);
          fields["memory_actors"] =
              census ? llvm::json::Value(census->memoryActors)
                     : llvm::json::Value(nullptr);
          fields["latency_shortfall_observed"] = shortfall;
          fields["control_density_observed"] = controlDense;
          if (controlDense)
            describeMinedCompositeFuSupply(fields, supply);
          fields["diagnostic"] =
              !latency ? "this parent has no measured window to read"
              : !shortfall
                  ? "the mapped replay already keeps up with its dataflow "
                    "oracle"
              : !controlDense
                  ? "the deployed graph does not spend its actors on the "
                    "token plane"
              : !proposal ? "the deployed software mines no admissible "
                            "composite shape"
                          : "one composed capability answers the measured "
                            "per-iteration actor count";
        });
    if (proposal) {
      HardwareRecipeGrowth growth;
      growth.config = currentConfig;
      growth.computeContextGrowthDirection = dse::
          TechMappingComputeContextGrowthDirection::MinedCompositeFuTemplate;
      growth.minedCompositeFus = proposal->selection;
      growth.minedDataflow = software.dataflow;
      growth.minedActorsPerRealization = proposal->actorsPerRealization;
      growth.minedSearchBounded = supply.bounded;
      growth.addedSpatialFuOccurrences = proposal->occurrences;
      growth.resultingContexts =
          currentConfig.hardwareTarget.parameters.temporalResidentContexts;
      growth.resultingGateways =
          currentConfig.hardwareTarget.parameters.gatewayCount;
      growth.resultingAccCores =
          currentConfig.hardwareTarget.parameters.accCoreCount;
      auto materialized = materializeHardwareRecipeGrowth(
          std::move(growth), evidence, request, scheduler, artifacts, blobs);
      if (!materialized)
        return materialized.takeError();
      if (const auto *incomplete =
              std::get_if<IncompleteHardwareRecipeMaterialization>(
                  &*materialized)) {
        if (llvm::Error error = retainJointDesignInvocationManifest(
                result.invocations, incomplete->constructionInvocation))
          return std::move(error);
        result.incomplete = true;
      } else {
        auto &child = std::get<MaterializedHardwareCandidate>(*materialized);
        if (child.constructionInvocation)
          if (llvm::Error error = retainJointDesignInvocationManifest(
                  result.invocations, *child.constructionInvocation))
            return std::move(error);
        auto timing = normalizedTimingProfiles(child.reference, artifacts);
        if (!timing)
          return timing.takeError();
        // A changed FU inventory invalidates every realization the parent
        // holds, so this child is mapped cold: no Spatial frontier and no
        // System migration seed may cross it.
        auto childPlan = buildJointDesignExplorationPlan(
            {{software.workloads}, {child.reference}}, *timing, *reopenPolicy,
            child.config, artifacts, nullptr, plan.systemBindingPartitions);
        if (!childPlan)
          return childPlan.takeError();
        auto execution =
            executeJointPlan(*childPlan, evidence, request, scheduler,
                             artifacts, blobs, &effectiveExecutionPolicy);
        if (!execution)
          return execution.takeError();
        if (llvm::Error error = retainJointDesignExecutionInvocations(
                result.invocations, *execution))
          return error;
        ++result.attemptedSystems;
        --remaining;
        const std::size_t count = mappingCount(*execution);
        mapping_debug::emit(
            mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
            mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
              fields["operation"] = "bounded_quality_hardware_spectrum";
              fields["candidate_ordinal"] = 0;
              fields["composite_fu_occurrences"] = proposal->occurrences;
              fields["acc_core_count"] = child.resultingAccCores;
              fields["system"] =
                  formatArtifactIdentityHex(child.reference.artifact);
              fields["system_mapping_count"] = count;
              fields["seed_source"] = "cold";
            });
        if (count != 0) {
          auto composedModules =
              projectJointDesignTargetModules(child.reference, artifacts);
          if (!composedModules)
            return composedModules.takeError();
          if (composedModules->empty())
            return invalid("composed capability published no target Module");
          targetModules = std::move(*composedModules);
          currentSystem = child.reference;
          currentConfig = child.config;
          spatialFrontierRetained = false;
          parentMapping = firstMapping(*execution);
        } else if (std::holds_alternative<IncompleteDsePlanExecution>(
                       execution->planExecution)) {
          // The composed child published nothing, so the chain continues from
          // the parent it never left and keeps the parent's own seed.
          result.incomplete = true;
        }
        result.attempts.push_back({child.reference, std::move(*execution)});
      }
    }
  }

  for (std::uint64_t ordinal = 0; ordinal != remaining; ++ordinal) {
    if (dispatchDeadlineReached(effectiveExecutionPolicy)) {
      result.incomplete = true;
      break;
    }
    auto currentRoot = fabric::importEntireFabricRoot(currentSystem, artifacts);
    if (!currentRoot)
      return currentRoot.takeError();
    auto currentView = fabric::requireSystemRoot(currentRoot->view());
    if (!currentView)
      return currentView.takeError();
    if (request.maximumUsefulAccCoreCount &&
        currentView->artifact().accCoreOccurrences().size() >=
            *request.maximumUsefulAccCoreCount)
      break;
    if (currentView->artifact().accCoreOccurrences().size() ==
        std::numeric_limits<std::uint32_t>::max())
      return invalid("finalized Mapping spectrum exceeds u32 AccCores");
    const ArtifactRootReference targetModule =
        targetModules[ordinal % targetModules.size()];
    HardwareRecipeGrowth growth;
    growth.config = currentConfig;
    growth.accCoreParent = currentSystem;
    growth.accCoreTargetModule = targetModule;
    growth.addedAccCores = 1;
    growth.resultingAccCores =
        currentView->artifact().accCoreOccurrences().size() + 1;
    growth.config.hardwareTarget.parameters.accCoreCount =
        static_cast<std::uint32_t>(growth.resultingAccCores);
    auto child =
        materializeTypedAccCoreGrowth(std::move(growth), artifacts, blobs);
    if (!child)
      return child.takeError();
    auto timing = normalizedTimingProfiles(child->reference, artifacts);
    if (!timing)
      return timing.takeError();
    auto childPlan = buildJointDesignExplorationPlan(
        {{software.workloads}, {child->reference}}, *timing, *reopenPolicy,
        child->config, artifacts, nullptr, plan.systemBindingPartitions);
    if (!childPlan)
      return childPlan.takeError();
    if (spatialFrontierRetained)
      if (llvm::Error error = bindImmutableSpatialMappingFrontier(
              *childPlan, *reusableSpatialMappings, artifacts))
        return std::move(error);
    if (parentMapping) {
      if (!child->executionBindingCorrespondence)
        return invalid("typed AddAccCore child lost its correspondence");
      auto migrationContext = deriveSystemMappingMigrationContext(*childPlan);
      if (!migrationContext)
        return migrationContext.takeError();
      auto migration = pnr::finalizeSystemMappingMigrationSeed(
          *parentMapping, *child->executionBindingCorrespondence,
          *migrationContext, artifacts);
      if (!migration)
        return migration.takeError();
      if (llvm::Error error = bindFinalizedSystemMappingMigrationSeed(
              *childPlan, migration->reference(), artifacts))
        return std::move(error);
    }
    auto execution =
        executeJointPlan(*childPlan, evidence, request, scheduler, artifacts,
                         blobs, &effectiveExecutionPolicy);
    if (!execution)
      return execution.takeError();
    if (llvm::Error error = retainJointDesignExecutionInvocations(
            result.invocations, *execution))
      return error;
    ++result.attemptedSystems;
    const std::size_t count = mappingCount(*execution);
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "bounded_quality_hardware_spectrum";
          fields["candidate_ordinal"] = ordinal;
          fields["acc_core_count"] = child->resultingAccCores;
          fields["system"] =
              formatArtifactIdentityHex(child->reference.artifact);
          fields["system_mapping_count"] = count;
          fields["seed_source"] = parentMapping ? "finalized_mapping" : "cold";
        });
    if (count != 0) {
      parentMapping = firstMapping(*execution);
    } else {
      parentMapping.reset();
      if (std::holds_alternative<IncompleteDsePlanExecution>(
              execution->planExecution))
        result.incomplete = true;
    }
    result.attempts.push_back({child->reference, std::move(*execution)});
    currentSystem = child->reference;
    currentConfig = std::move(child->config);
  }
  return result;
}

} // namespace loom::dse::joint_reopen_detail

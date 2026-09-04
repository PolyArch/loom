#include "JointHardwareReopenInternal.h"

#include "JointHardwareReopenExecution.h"

#include "Common/ArtifactText.h"
#include "Common/MappingDebugLog.h"
#include "DSE/JointMappingMigration.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "PnR/System/SystemMappingMigration.h"

#include <limits>
#include <utility>

namespace loom::dse::joint_reopen_detail {

llvm::Expected<FinalizedMappingHardwareSpectrum>
exploreFinalizedMappingHardwareSpectrum(
    const JointDesignPolicy &policy, const JointDesignExplorationPlan &plan,
    const JointDesignExecution &parentExecution,
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
  auto targetModules = projectJointDesignTargetModules(
      plan.frontier.systemFrontier.front(), artifacts);
  if (!targetModules)
    return targetModules.takeError();
  if (targetModules->empty())
    return invalid("finalized Mapping spectrum has no target Module");
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
  const std::uint64_t remaining =
      policy.maximumSystemFrontier() - plan.frontier.systemFrontier.size();
  const PlanExecutionPolicy &effectiveExecutionPolicy =
      executionPolicy ? *executionPolicy : request.executionPolicy;
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
        (*targetModules)[ordinal % targetModules->size()];
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
      result.verified.push_back(std::move(*execution));
    } else {
      parentMapping.reset();
      if (std::holds_alternative<IncompleteDsePlanExecution>(
              execution->planExecution))
        result.incomplete = true;
    }
    currentSystem = child->reference;
    currentConfig = std::move(child->config);
  }
  return result;
}

} // namespace loom::dse::joint_reopen_detail

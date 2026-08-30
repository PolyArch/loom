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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::dse {

using namespace joint_reopen_detail;

namespace {

llvm::Expected<mapping::SystemMappingImportSessionStatistics>
independentlyVerifyChildMappings(llvm::ArrayRef<ArtifactRootReference> mappings,
                                 const ArtifactRootReference &dataflow,
                                 const ArtifactRootReference &childSystem,
                                 const ArtifactStore &artifacts);

} // namespace

llvm::Expected<JointResourceTimeAdjacentRepair>
executeResourceTimeAdjacentMappingRepair(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy,
    llvm::ArrayRef<pnr::SystemBindingPartitionIntent> childPartitions,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    JointHardwareReopenRequest request, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  if (llvm::Error error = registerProductionDseOwners())
    return std::move(error);
  // The cold and preserve-first plans import the same System and Module roots,
  // so one session across both keeps the repeated strict imports as cache hits.
  ::loom::fabric::FabricArtifactImportSession fabricImportSession;
  llvm::scope_exit emitFabricImportStatistics([&] {
    ::loom::fabric::emitFabricArtifactImportSessionStatistics(
        ::loom::fabric::FabricArtifactImportVerificationDomain::
            SourceInvocation,
        ::loom::InvocationDiagnosticStage::SystemPnr,
        fabricImportSession.statistics());
  });
  if (request.journalRoot.empty())
    return invalid("resource-time repair requires a journal root");
  if (reopenedRoots.empty())
    return invalid("resource-time repair has no typed invalidation root");
  if (parentPlan.pairOutputs.size() != 1)
    return invalid("resource-time repair requires one exact parent pair");
  std::optional<ArtifactRootReference> parentMapping;
  if (parentExecution.summary.selectedMapping)
    parentMapping = *parentExecution.summary.selectedMapping;
  else {
    auto available = firstMapping(parentExecution);
    if (!available)
      return invalid("resource-time repair parent has no verified Mapping");
    parentMapping = *available;
  }

  const JointDesignPair &parentPair = parentPlan.pairOutputs.front().pair;
  const ArtifactRootReference system = parentPair.system;
  auto timing = normalizedTimingProfiles(system, artifacts);
  if (!timing)
    return timing.takeError();
  auto repairPolicy =
      JointDesignPolicy::get(1, 1, 1, policy.maximumTechMappingsPerModule(),
                             policy.maximumSpatialMappingsPerPair());
  if (!repairPolicy)
    return repairPolicy.takeError();
  const JointSoftwareScope &software = parentPair.software;
  ResolvedConfig childConfig = parentPlan.resolvedConfig;
  childConfig.dse.planNodes.clear();
  childConfig.dse.systemPnr.search.completionGoal =
      ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  auto childPlan = buildJointDesignExplorationPlan(
      {{software.workloads}, {system}}, *timing, *repairPolicy, childConfig,
      artifacts, nullptr, childPartitions);
  if (!childPlan)
    return childPlan.takeError();
  JointDesignExplorationPlan coldPlan = *childPlan;

  auto spatialMappings =
      resolveJointSpatialMappingFrontier(parentPlan, parentExecution);
  if (!spatialMappings)
    return spatialMappings.takeError();
  if (llvm::Error error = bindImmutableSpatialMappingFrontier(
          *childPlan, *spatialMappings, artifacts))
    return std::move(error);
  auto correspondence =
      pnr::SystemExecutionBindingCorrespondence::getIdentity(system, artifacts);
  if (!correspondence)
    return correspondence.takeError();
  auto context = deriveSystemMappingMigrationContext(*childPlan);
  if (!context)
    return context.takeError();
  auto seed = pnr::finalizeSystemMappingMigrationSeed(
      *parentMapping, *correspondence, *context, reopenedRoots, artifacts);
  if (!seed)
    return seed.takeError();
  if (llvm::Error error = bindFinalizedSystemMappingMigrationSeed(
          *childPlan, seed->reference(), artifacts))
    return std::move(error);

  JointHardwareReopenRequest coldRequest = request;
  llvm::SmallString<256> coldJournal(coldRequest.journalRoot);
  llvm::sys::path::append(coldJournal, "cold");
  coldRequest.journalRoot = coldJournal.str().str();
  JointHardwareReopenRequest incrementalRequest = request;
  llvm::SmallString<256> incrementalJournal(incrementalRequest.journalRoot);
  llvm::sys::path::append(incrementalJournal, "incremental");
  incrementalRequest.journalRoot = incrementalJournal.str().str();
  const auto executeIndependent = [&](const JointDesignExplorationPlan &plan,
                                      const JointHardwareReopenRequest &run)
      -> llvm::Expected<JointDesignExecution> {
    loom::pnr::PnrDerivedContextSession derivedContextSession;
    return executeJointRepairPlan(plan, *repairPolicy, run, artifacts, blobs);
  };
  auto coldExecution = executeIndependent(coldPlan, coldRequest);
  if (!coldExecution)
    return coldExecution.takeError();
  coldExecution->summary.coldReopenWallTimeNanoseconds =
      coldExecution->summary.executionWallTimeNanoseconds;

  auto execution = executeIndependent(*childPlan, incrementalRequest);
  if (!execution)
    return execution.takeError();
  execution->summary.incrementalReopenWallTimeNanoseconds =
      execution->summary.executionWallTimeNanoseconds;
  const std::optional<ArtifactRootReference> coldMapping =
      coldExecution->summary.selectedMapping;
  const std::optional<ArtifactRootReference> incrementalMapping =
      execution->summary.selectedMapping;
  for (const auto *reference : {&coldMapping, &incrementalMapping}) {
    if (!*reference)
      continue;
    auto imported = mapping::importSystemMapping(**reference, artifacts);
    if (!imported)
      return imported.takeError();
    if (imported->view().dataflowIdentity() != software.dataflow.artifact ||
        imported->view().fabricIdentity() != system.artifact)
      return invalid("paired resource-time Mapping has foreign owners");
  }
  const std::vector<ArtifactRootReference> coldMappings =
      mappingRoots(*coldExecution);
  const std::vector<ArtifactRootReference> incrementalMappings =
      mappingRoots(*execution);
  auto coldVerification = independentlyVerifyChildMappings(
      coldMappings, software.dataflow, system, artifacts);
  if (!coldVerification)
    return coldVerification.takeError();
  auto incrementalVerification = independentlyVerifyChildMappings(
      incrementalMappings, software.dataflow, system, artifacts);
  if (!incrementalVerification)
    return incrementalVerification.takeError();
  std::set<ArtifactIdentity::Storage> preservedTech;
  for (const ArtifactRootReference &reference : *spatialMappings) {
    auto mapping = mapping::importSpatialMapping(reference, artifacts);
    if (!mapping)
      return mapping.takeError();
    preservedTech.insert(mapping->view().techMappingIdentity().bytes());
  }
  execution->summary.preservedSpatialMappings = spatialMappings->size();
  execution->summary.preservedTechMappings = preservedTech.size();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "resource_time_adjacent_mapping_repair";
        fields["reopened_root_count"] = reopenedRoots.size();
        fields["preserved_tech_mappings"] = preservedTech.size();
        fields["preserved_spatial_mappings"] = spatialMappings->size();
        fields["tech_mapping_dispatch_count"] =
            execution->summary.techMappingDispatchCount;
        fields["spatial_pnr_dispatch_count"] =
            execution->summary.spatialPnrDispatchCount;
        fields["system_pnr_dispatch_count"] =
            execution->summary.systemPnrDispatchCount;
        fields["cold_wall_time_ns"] =
            coldExecution->summary.executionWallTimeNanoseconds;
        fields["incremental_wall_time_ns"] =
            execution->summary.executionWallTimeNanoseconds;
        fields["cold_verifier_retained_bytes"] =
            coldVerification->retainedBytes;
        fields["incremental_verifier_retained_bytes"] =
            incrementalVerification->retainedBytes;
        fields["cold_verifier_work"] = coldVerification->deterministicWork;
        fields["incremental_verifier_work"] =
            incrementalVerification->deterministicWork;
      });
  JointMappingReuseDisposition reuseDisposition =
      JointMappingReuseDisposition::ColdFallback;
  if (execution->summary.preservedTechMappings != 0 ||
      execution->summary.preservedSpatialMappings != 0) {
    reuseDisposition = execution->summary.repairedTechMappings != 0 ||
                               execution->summary.repairedSpatialMappings != 0
                           ? JointMappingReuseDisposition::LocalRepair
                           : JointMappingReuseDisposition::Preserved;
  }
  return JointResourceTimeAdjacentRepair{*parentMapping,
                                         seed->reference(),
                                         std::move(*childPlan),
                                         coldMapping,
                                         incrementalMapping,
                                         std::move(*coldExecution),
                                         std::move(*execution),
                                         reuseDisposition,
                                         std::move(*coldVerification),
                                         std::move(*incrementalVerification)};
}

namespace {

void applyMappingRebaseAccounting(
    JointDesignExecutionSummary &summary,
    const JointMappingRebaseAccounting &accounting,
    JointMappingReuseDisposition disposition) {
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
  summary.preservedThreadBindingCount = accounting.preservedThreadBindingCount;
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
  if (disposition == JointMappingReuseDisposition::ColdFallback)
    summary.coldReopenWallTimeNanoseconds =
        summary.executionWallTimeNanoseconds;
  else
    summary.incrementalReopenWallTimeNanoseconds =
        summary.executionWallTimeNanoseconds;
}

llvm::Error
accountSystemColdFallback(JointMappingRebaseAccounting &accounting) {
  const std::array preserved = {
      accounting.preservedThreadBindingCount,
      accounting.preservedGraphBindingCount,
      accounting.preservedResourceUseCount,
      accounting.preservedServiceRealizationCount,
      accounting.preservedServiceLegCount,
  };
  for (const std::uint64_t count : preserved) {
    auto total = llvm::checkedAddUnsigned(
        accounting.invalidationConeDecisionCount, count);
    if (!total)
      return invalid("System cold-fallback accounting overflows");
    accounting.invalidationConeDecisionCount = *total;
  }
  accounting.preservedThreadBindingCount = 0;
  accounting.reopenedThreadBindingCount = accounting.parentThreadBindingCount;
  accounting.preservedGraphBindingCount = 0;
  accounting.reopenedGraphBindingCount = accounting.parentGraphBindingCount;
  accounting.preservedResourceUseCount = 0;
  accounting.reopenedResourceUseCount = accounting.parentResourceUseCount;
  accounting.preservedServiceRealizationCount = 0;
  accounting.reopenedServiceRealizationCount =
      accounting.parentServiceRealizationCount;
  accounting.preservedServiceLegCount = 0;
  accounting.reopenedServiceLegCount = accounting.parentServiceLegCount;
  return validateJointMappingRebaseAccounting(accounting);
}

llvm::Expected<JointDesignExecution>
executeIndependentMutationPlan(const JointDesignExplorationPlan &plan,
                               const JointDesignPolicy &policy,
                               const JointHardwareReopenRequest &request,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs) {
  loom::pnr::PnrDerivedContextSession derivedContextSession;
  return executeJointRepairPlan(plan, policy, request, artifacts, blobs);
}

llvm::Expected<mapping::SystemMappingImportSessionStatistics>
independentlyVerifyChildMappings(llvm::ArrayRef<ArtifactRootReference> mappings,
                                 const ArtifactRootReference &dataflow,
                                 const ArtifactRootReference &childSystem,
                                 const ArtifactStore &artifacts) {
  mapping::SystemMappingImportSession importSession(
      artifacts, std::max<std::size_t>(1, mappings.size()),
      mapping::SystemMappingImportSessionMode::New);
  for (const ArtifactRootReference &reference : mappings) {
    auto imported = mapping::importSystemMapping(reference, artifacts);
    if (!imported)
      return imported.takeError();
    if (imported->view().dataflowIdentity() != dataflow.artifact ||
        imported->view().fabricIdentity() != childSystem.artifact)
      return invalid("independently verified Mapping has foreign owners");
  }
  const mapping::SystemMappingImportSessionStatistics statistics =
      importSession.statistics();
  mapping::emitSystemMappingImportSessionStatistics(
      mapping::SystemMappingImportVerificationDomain::IndependentReplay,
      statistics);
  return statistics;
}

struct ReplacementModuleLineageProjection final {
  std::vector<pnr::SystemModuleCorrespondence> modules;
  bool oneToOne = true;
};

llvm::Expected<std::optional<ReplacementModuleLineageProjection>>
deriveReplacementModuleLineage(
    const ArtifactRootReference &parentSystem,
    const ArtifactRootReference &childSystem,
    const SystemCompositionCandidateDecision &decision,
    const ArtifactStore &artifacts) {
  if (!std::holds_alternative<ReplaceSpatialAttachment>(decision.decision))
    return std::nullopt;
  auto parentRoot = fabric::importEntireFabricRoot(parentSystem, artifacts);
  if (!parentRoot)
    return parentRoot.takeError();
  auto childRoot = fabric::importEntireFabricRoot(childSystem, artifacts);
  if (!childRoot)
    return childRoot.takeError();
  auto parentView = fabric::requireSystemRoot(parentRoot->view());
  if (!parentView)
    return parentView.takeError();
  auto childView = fabric::requireSystemRoot(childRoot->view());
  if (!childView)
    return childView.takeError();
  const auto moduleForCore =
      [](const fabric::FinalizedFabricRoot &root,
         const fabric::FabricSystemRootView &view,
         fabric::AccCoreOccurrenceRef core,
         llvm::StringRef role) -> llvm::Expected<ArtifactRootReference> {
    const auto target = view.spatialCoreTarget(core);
    if (!target ||
        target->dependencyOrdinal >= root.directDependencies().size())
      return invalid(role + " AccCore has no exact Module target");
    return root.directDependencies()[target->dependencyOrdinal].root;
  };

  ReplacementModuleLineageProjection result;
  for (fabric::AccCoreOccurrenceRef parentCore :
       parentView->artifact().accCoreOccurrences()) {
    const auto entity =
        llvm::find_if(decision.entities, [&](const auto &entry) {
          return entry.source.kind ==
                     fabric::FabricEntityKind::AccCoreOccurrence &&
                 entry.source.id == parentCore.id();
        });
    if (entity == decision.entities.end())
      return invalid("replacement lineage omits a preserved AccCore");
    if (entity->target.kind != fabric::FabricEntityKind::AccCoreOccurrence)
      return invalid("replacement lineage changes an AccCore entity kind");
    const fabric::AccCoreOccurrenceRef childCore(entity->target.id);
    auto parentModule =
        moduleForCore(*parentRoot, *parentView, parentCore, "parent");
    if (!parentModule)
      return parentModule.takeError();
    auto childModule =
        moduleForCore(*childRoot, *childView, childCore, "child");
    if (!childModule)
      return childModule.takeError();
    bool repeated = false;
    for (const pnr::SystemModuleCorrespondence &existing : result.modules) {
      if (existing.parent == *parentModule && existing.child == *childModule) {
        repeated = true;
        continue;
      }
      if ((existing.parent == *parentModule &&
           existing.child != *childModule) ||
          (existing.child == *childModule && existing.parent != *parentModule))
        result.oneToOne = false;
    }
    if (!repeated)
      result.modules.push_back({*parentModule, *childModule});
  }
  llvm::sort(result.modules, [](const auto &lhs, const auto &rhs) {
    if (lhs.parent != rhs.parent)
      return artifactRootReferenceLess(lhs.parent, rhs.parent);
    return artifactRootReferenceLess(lhs.child, rhs.child);
  });
  return std::optional<ReplacementModuleLineageProjection>(std::move(result));
}

struct TypedModuleHardwareRepair final {
  JointDesignExecution execution;
  JointMappingReuseDisposition disposition =
      JointMappingReuseDisposition::ColdFallback;
};

llvm::Expected<TypedModuleHardwareRepair> executeTypedModuleHardwareReopen(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy, const ArtifactRootReference &parentMapping,
    MaterializedHardwareCandidate child, JointHardwareReopenRequest request,
    llvm::StringRef operation, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  if (!child.mappingImpact)
    return invalid("typed Module hardware child has no Mapping impact");
  if (!child.executionBindingCorrespondence)
    return invalid("typed Module hardware child has no System correspondence");
  if (!llvm::equal(child.executionBindingCorrespondence->modules(),
                   child.moduleCorrespondences))
    return invalid("typed Module hardware child has divergent Module lineage");
  auto repair = executeJointHardwareMutationRepair(
      parentPlan, parentExecution, policy, parentMapping,
      JointHardwareMutationChild{
          child.reference,
          std::move(child.config),
          std::move(child.executionBindingCorrespondence),
          {std::move(*child.mappingImpact)}},
      std::move(request), artifacts, blobs);
  if (!repair)
    return repair.takeError();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = operation;
        fields["cold_mapping_count"] = repair->coldMappings.size();
        fields["incremental_mapping_count"] =
            repair->incrementalMappings.size();
        fields["liveness"] = "requires_child_cgra_replay";
        fields["reconfiguration_support"] = "unsupported";
      });
  const JointMappingReuseDisposition disposition = repair->rebase.disposition;
  return TypedModuleHardwareRepair{std::move(repair->incrementalExecution),
                                   disposition};
}

} // namespace

llvm::StringRef jointSystemMappingReuseDispositionSpelling(
    JointSystemMappingReuseDisposition disposition) {
  switch (disposition) {
  case JointSystemMappingReuseDisposition::Preserved:
    return "preserved";
  case JointSystemMappingReuseDisposition::Reopened:
    return "reopened";
  case JointSystemMappingReuseDisposition::ColdFallback:
    return "cold_fallback";
  }
  llvm_unreachable("unknown System Mapping reuse disposition");
}

llvm::Expected<JointHardwareMutationChild>
materializeJointModuleHardwareMutation(
    ResolvedConfig config, const ArtifactRootReference &parentSystem,
    const ArtifactRootReference &parentModule,
    SpatialMicroarchitectureDecisionDomain decision,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (llvm::Error error = registerProductionDseOwners())
    return std::move(error);
  HardwareRecipeGrowth growth;
  growth.config = std::move(config);
  growth.config.dse.planNodes.clear();
  growth.techModule = parentModule;
  growth.moduleDecision = std::move(decision);
  auto materialized = materializeTypedModuleSystemGrowth(
      std::move(growth), parentSystem, artifacts, blobs);
  if (!materialized)
    return materialized.takeError();
  if (!materialized->mappingImpact ||
      !materialized->executionBindingCorrespondence)
    return invalid("Module mutation child lost typed lineage");
  if (!llvm::equal(materialized->executionBindingCorrespondence->modules(),
                   materialized->moduleCorrespondences))
    return invalid("Module mutation child has divergent Module lineage");
  return JointHardwareMutationChild{
      materialized->reference,
      std::move(materialized->config),
      std::move(materialized->executionBindingCorrespondence),
      {std::move(*materialized->mappingImpact)}};
}

llvm::Expected<JointHardwareMutationChild>
materializeJointSystemHardwareMutation(
    ResolvedConfig config, const ArtifactRootReference &parentSystem,
    SystemCompositionDecisionDomain decision,
    llvm::ArrayRef<ArtifactRootReference> admissibleModules,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (llvm::Error error = registerProductionDseOwners())
    return std::move(error);
  auto rewrite = resolveSystemCompositionRewriteConfig(
      llvm::ArrayRef<SystemCompositionDecisionDomain>(decision), 1);
  if (!rewrite)
    return rewrite.takeError();
  auto inputs = bindSystemCompositionCandidateGeneratorInputs(
      llvm::ArrayRef<ArtifactRootReference>(parentSystem), admissibleModules);
  if (!inputs)
    return inputs.takeError();
  auto binding = resolveSystemCompositionCandidateGeneratorBinding(*rewrite);
  if (!binding)
    return binding.takeError();
  auto generated =
      invokeCandidateGenerator(*inputs, *binding, artifacts, blobs);
  if (!generated)
    return generated.takeError();
  const auto *completed =
      std::get_if<CompletedCandidateGeneratorResult>(&generated->outcome);
  if (!completed || completed->outputBindings.size() != 1 ||
      completed->outputBindings.front().artifacts.size() != 1 ||
      completed->lineageEdges.size() != 1)
    return invalid("System hardware mutation did not publish one child");
  const ArtifactRootReference childSystem =
      completed->outputBindings.front().artifacts.front();
  const CandidateGeneratorLineageEdge &lineage =
      completed->lineageEdges.front();
  if (lineage.kind != CandidateGeneratorLineageEdgeKind::CandidateDecision ||
      lineage.output != childSystem ||
      lineage.parents != std::vector<ArtifactRootReference>{parentSystem})
    return invalid("System hardware mutation lost its exact parent lineage");
  auto adopted = adoptSystemCompositionDecision(lineage.ownerPayload);
  if (!adopted)
    return adopted.takeError();
  if (adopted->parent != parentSystem)
    return invalid("System hardware mutation changed its parent owner");

  std::vector<pnr::SystemModuleCorrespondence> modules;
  bool functional = true;
  auto replacement = deriveReplacementModuleLineage(parentSystem, childSystem,
                                                    *adopted, artifacts);
  if (!replacement)
    return replacement.takeError();
  if (*replacement) {
    modules = std::move((*replacement)->modules);
    functional = (*replacement)->oneToOne;
  } else {
    auto parentModules =
        projectJointDesignTargetModules(parentSystem, artifacts);
    if (!parentModules)
      return parentModules.takeError();
    auto childModules = projectJointDesignTargetModules(childSystem, artifacts);
    if (!childModules)
      return childModules.takeError();
    for (const ArtifactRootReference &module : *parentModules)
      if (llvm::is_contained(*childModules, module))
        modules.push_back({module, module});
  }
  std::optional<pnr::SystemExecutionBindingCorrespondence> correspondence;
  if (functional) {
    auto exact = pnr::SystemExecutionBindingCorrespondence::get(
        parentSystem, childSystem, adopted->entities, adopted->transferPatterns,
        std::move(modules), artifacts);
    if (!exact)
      return exact.takeError();
    correspondence = std::move(*exact);
  }
  HardwareImpactProjection impact =
      projectHardwareImpact(*adopted, childSystem);
  config.dse.planNodes.clear();
  return JointHardwareMutationChild{childSystem,
                                    std::move(config),
                                    std::move(correspondence),
                                    {std::move(impact)}};
}

llvm::Expected<JointHardwareMutationChild>
composeJointHardwareMutationChildren(JointHardwareMutationChild first,
                                     JointHardwareMutationChild second,
                                     const ArtifactStore &artifacts) {
  if (!first.executionBindingCorrespondence ||
      !second.executionBindingCorrespondence)
    return invalid("combined hardware child lacks System correspondence");
  if (first.system != second.executionBindingCorrespondence->parentSystem())
    return invalid("combined hardware child lineage is not consecutive");
  auto correspondence = pnr::composeSystemExecutionBindingCorrespondence(
      *first.executionBindingCorrespondence,
      *second.executionBindingCorrespondence, artifacts);
  if (!correspondence)
    return correspondence.takeError();
  first.impacts.insert(first.impacts.end(),
                       std::make_move_iterator(second.impacts.begin()),
                       std::make_move_iterator(second.impacts.end()));
  return JointHardwareMutationChild{second.system, std::move(second.config),
                                    std::move(*correspondence),
                                    std::move(first.impacts)};
}

llvm::Expected<JointHardwareMutationRepair> executeJointHardwareMutationRepair(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy, const ArtifactRootReference &parentMapping,
    JointHardwareMutationChild child, JointHardwareReopenRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (llvm::Error error = registerProductionDseOwners())
    return std::move(error);
  // One repair imports the same parent and child Fabric roots across plan
  // construction, freeze and verification, and each strict import recomputes
  // the canonical labeling and the stored-domain validation. Scope one import
  // session to the repair so those repeats become cache hits. ReuseEnclosing
  // keeps an outer session as the single owner when one already exists.
  ::loom::fabric::FabricArtifactImportSession fabricImportSession;
  llvm::scope_exit emitFabricImportStatistics([&] {
    ::loom::fabric::emitFabricArtifactImportSessionStatistics(
        ::loom::fabric::FabricArtifactImportVerificationDomain::
            SourceInvocation,
        ::loom::InvocationDiagnosticStage::SystemPnr,
        fabricImportSession.statistics());
  });
  if (parentPlan.pairOutputs.size() != 1)
    return invalid("hardware mutation repair requires one exact parent pair");
  if (request.journalRoot.empty())
    return invalid("hardware mutation repair requires a journal root");
  if (child.impacts.empty())
    return invalid("hardware mutation repair has no typed impact");
  if (child.executionBindingCorrespondence &&
      (child.executionBindingCorrespondence->parentSystem() !=
           parentPlan.pairOutputs.front().pair.system ||
       child.executionBindingCorrespondence->childSystem() != child.system))
    return invalid(
        "hardware mutation System correspondence has foreign owners");
  if (llvm::any_of(child.impacts,
                   [](const HardwareImpactProjection &impact) {
                     return !impact.child;
                   }) ||
      *child.impacts.back().child != child.system)
    return invalid(
        "hardware mutation lineage does not end at its child System");
  const JointDesignPair &parentPair = parentPlan.pairOutputs.front().pair;
  auto importedParentMapping =
      mapping::importSystemMapping(parentMapping, artifacts);
  if (!importedParentMapping)
    return importedParentMapping.takeError();
  if (importedParentMapping->view().fabricIdentity() !=
          parentPair.system.artifact ||
      importedParentMapping->view().dataflowIdentity() !=
          parentPair.software.dataflow.artifact)
    return invalid("hardware mutation parent Mapping has foreign owners");
  if (!llvm::is_contained(mappingRoots(parentExecution), parentMapping))
    return invalid("hardware mutation parent Mapping is not in its execution");

  auto rebased = rebaseJointMappingFrontier(
      parentPlan, parentExecution, child.system,
      child.executionBindingCorrespondence
          ? child.executionBindingCorrespondence->modules()
          : llvm::ArrayRef<pnr::SystemModuleCorrespondence>(),
      child.impacts, artifacts, parentMapping);
  if (!rebased)
    return rebased.takeError();
  auto timing = normalizedTimingProfiles(child.system, artifacts);
  if (!timing)
    return timing.takeError();
  auto repairPolicy =
      JointDesignPolicy::get(1, 1, 1, policy.maximumTechMappingsPerModule(),
                             policy.maximumSpatialMappingsPerPair());
  if (!repairPolicy)
    return repairPolicy.takeError();
  child.config.dse.planNodes.clear();
  child.config.dse.systemPnr.search.completionGoal =
      ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  auto coldPlan = buildJointDesignExplorationPlan(
      {{parentPair.software.workloads}, {child.system}}, *timing, *repairPolicy,
      child.config, artifacts, nullptr, parentPlan.systemBindingPartitions);
  if (!coldPlan)
    return coldPlan.takeError();
  const JointDesignMappingSeed *mappingSeed =
      rebased->seed.techMappings.empty() &&
              rebased->seed.spatialMappings.empty()
          ? nullptr
          : &rebased->seed;
  auto incrementalPlan = buildJointDesignExplorationPlan(
      {{parentPair.software.workloads}, {child.system}}, *timing, *repairPolicy,
      child.config, artifacts, mappingSeed, parentPlan.systemBindingPartitions);
  if (!incrementalPlan)
    return incrementalPlan.takeError();
  const bool systemImpactReopened =
      llvm::any_of(child.impacts, [](const HardwareImpactProjection &impact) {
        return impact.system.kind == HardwareMappingImpactKind::Reopen;
      });
  const bool systemImpactRequiresColdFallback =
      child.impacts.size() != 1 ||
      llvm::any_of(child.impacts, [](const HardwareImpactProjection &impact) {
        return impact.system.kind == HardwareMappingImpactKind::Reopen &&
               (!impact.system.transportRoots.empty() ||
                !impact.system.routeRoots.empty() ||
                !impact.system.serviceRoots.empty() ||
                !impact.system.memoryServiceRoots.empty() ||
                !impact.system.memoryRoots.empty());
      });
  const std::vector<::dataflow::RootThreadLaunchRef> systemReopenRoots =
      projectJointSystemReopenRoots(importedParentMapping->view(),
                                    child.impacts);
  JointSystemMappingReuseDisposition systemDisposition =
      systemImpactRequiresColdFallback
          ? JointSystemMappingReuseDisposition::ColdFallback
      : systemImpactReopened && !systemReopenRoots.empty()
          ? JointSystemMappingReuseDisposition::Reopened
          : JointSystemMappingReuseDisposition::Preserved;
  const bool systemInputsNeedGeneration =
      !incrementalPlan->pairOutputs.front().spatialMappings.empty();
  if (rebased->disposition == JointMappingReuseDisposition::ColdFallback ||
      systemInputsNeedGeneration) {
    systemDisposition = JointSystemMappingReuseDisposition::ColdFallback;
    if (llvm::Error error = accountSystemColdFallback(rebased->accounting))
      return std::move(error);
  }
  if (rebased->disposition != JointMappingReuseDisposition::ColdFallback &&
      systemDisposition != JointSystemMappingReuseDisposition::ColdFallback) {
    if (!child.executionBindingCorrespondence)
      return invalid("incremental System repair has no exact correspondence");
    auto context = deriveSystemMappingMigrationContext(*incrementalPlan);
    if (!context)
      return context.takeError();
    auto seed = pnr::finalizeSystemMappingMigrationSeed(
        parentMapping, *child.executionBindingCorrespondence, *context,
        systemReopenRoots, artifacts);
    if (!seed)
      return seed.takeError();
    if (llvm::Error error = bindFinalizedSystemMappingMigrationSeed(
            *incrementalPlan, seed->reference(), artifacts))
      return std::move(error);
  }

  const bool coldComparisonBaseline = request.coldComparisonBaseline;
  JointHardwareReopenRequest coldRequest = request;
  llvm::SmallString<256> coldJournal(coldRequest.journalRoot);
  llvm::sys::path::append(coldJournal, "cold");
  coldRequest.journalRoot = coldJournal.str().str();
  JointHardwareReopenRequest incrementalRequest = std::move(request);
  llvm::SmallString<256> incrementalJournal(incrementalRequest.journalRoot);
  llvm::sys::path::append(incrementalJournal, "incremental");
  incrementalRequest.journalRoot = incrementalJournal.str().str();
  // The cold plan is an independent comparison oracle, never the repaired
  // Mapping. Executing it unconditionally doubled every hardware repair, and
  // when the rebase preserved nothing the preserve-first plan is itself
  // unseeded, so the identical plan ran twice.
  std::optional<JointDesignExecution> coldExecution;
  if (coldComparisonBaseline) {
    auto executed = executeIndependentMutationPlan(*coldPlan, policy,
                                                   coldRequest, artifacts,
                                                   blobs);
    if (!executed)
      return executed.takeError();
    coldExecution = std::move(*executed);
    coldExecution->summary.coldReopenWallTimeNanoseconds =
        coldExecution->summary.executionWallTimeNanoseconds;
  }
  auto incrementalExecution = executeIndependentMutationPlan(
      *incrementalPlan, policy, incrementalRequest, artifacts, blobs);
  if (!incrementalExecution)
    return incrementalExecution.takeError();
  applyMappingRebaseAccounting(incrementalExecution->summary,
                               rebased->accounting, rebased->disposition);

  std::vector<ArtifactRootReference> coldMappings =
      coldExecution ? mappingRoots(*coldExecution)
                    : std::vector<ArtifactRootReference>();
  std::vector<ArtifactRootReference> incrementalMappings =
      mappingRoots(*incrementalExecution);
  if (coldExecution)
    coldExecution->summary.verifiedAlternatives = coldMappings.size();
  incrementalExecution->summary.verifiedAlternatives =
      incrementalMappings.size();
  auto coldVerification = independentlyVerifyChildMappings(
      coldMappings, parentPair.software.dataflow, child.system, artifacts);
  if (!coldVerification)
    return coldVerification.takeError();
  auto incrementalVerification = independentlyVerifyChildMappings(
      incrementalMappings, parentPair.software.dataflow, child.system,
      artifacts);
  if (!incrementalVerification)
    return incrementalVerification.takeError();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "joint_hardware_mutation_repair";
        llvm::json::Array families;
        for (const HardwareImpactProjection &impact : child.impacts)
          families.push_back(hardwareMutationFamilySpelling(impact.family));
        fields["families"] = std::move(families);
        fields["parent_mapping"] =
            formatArtifactIdentityHex(parentMapping.artifact);
        fields["child_system"] =
            formatArtifactIdentityHex(child.system.artifact);
        fields["mapping_reuse_disposition"] =
            jointMappingReuseDispositionSpelling(rebased->disposition);
        fields["system_mapping_reuse_disposition"] =
            jointSystemMappingReuseDispositionSpelling(systemDisposition);
        fields["rebase_failure_count"] = rebased->failures.size();
        fields["cold_comparison_baseline"] = coldComparisonBaseline;
        fields["cold_mapping_count"] = coldMappings.size();
        fields["incremental_mapping_count"] = incrementalMappings.size();
        if (coldExecution)
          fields["cold_wall_time_ns"] =
              coldExecution->summary.executionWallTimeNanoseconds;
        fields["incremental_wall_time_ns"] =
            incrementalExecution->summary.executionWallTimeNanoseconds;
        fields["cold_verifier_retained_bytes"] =
            coldVerification->retainedBytes;
        fields["incremental_verifier_retained_bytes"] =
            incrementalVerification->retainedBytes;
        fields["cold_verifier_work"] = coldVerification->deterministicWork;
        fields["incremental_verifier_work"] =
            incrementalVerification->deterministicWork;
      });
  return JointHardwareMutationRepair{parentMapping,
                                     std::move(child),
                                     std::move(*rebased),
                                     systemDisposition,
                                     std::move(*coldPlan),
                                     std::move(*incrementalPlan),
                                     std::move(coldMappings),
                                     std::move(incrementalMappings),
                                     std::move(coldExecution),
                                     std::move(*incrementalExecution),
                                     std::move(*coldVerification),
                                     std::move(*incrementalVerification)};
}

llvm::Expected<JointSpatialFifoHardwareRepair>
executeSpatialFifoHardwareFeedbackReopen(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy, const SpatialFifoRuntimeFeedback &feedback,
    JointHardwareReopenRequest request, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  JointSpatialFifoHardwareRepair result{feedback, {}, {}, {}, false};
  if (feedback.disposition != SpatialFifoRuntimeFeedbackDisposition::Exact)
    return result;
  if (!feedback.fifo ||
      (!feedback.minimumCandidateDepth &&
       !feedback.candidateQueueDiscipline))
    return invalid("exact FIFO feedback has no physical candidate");
  if (parentPlan.pairOutputs.size() != 1 || request.journalRoot.empty())
    return invalid("FIFO hardware repair requires one exact parent pair");
  const JointDesignPair &parentPair = parentPlan.pairOutputs.front().pair;
  auto parentMapping =
      mapping::importSystemMapping(feedback.parentMapping, artifacts);
  if (!parentMapping)
    return parentMapping.takeError();
  if (parentMapping->view().fabricIdentity() != parentPair.system.artifact)
    return invalid("FIFO feedback parent Mapping names another System");
  auto parentSpatial =
      mapping::importSpatialMapping(feedback.spatialMapping, artifacts);
  if (!parentSpatial)
    return parentSpatial.takeError();
  ArtifactRootReference parentModule{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version,
      parentSpatial->view().fabricIdentity()};
  auto module = fabric::importEntireFabricRoot(parentModule, artifacts);
  if (!module)
    return module.takeError();
  const auto currentDiscipline =
      module->view().fifoQueueDiscipline(*feedback.fifo);
  if (!currentDiscipline ||
      (feedback.currentQueueDiscipline &&
       *feedback.currentQueueDiscipline != *currentDiscipline))
    return invalid("FIFO feedback discipline disagrees with its Module");

  std::vector<SpatialMicroarchitectureDecision> decisions;
  if (feedback.candidateQueueDiscipline) {
    if (*feedback.candidateQueueDiscipline == *currentDiscipline)
      return invalid("FIFO feedback discipline candidate is a no-op");
    decisions.push_back(ChangeFifoQueueDiscipline{
        *feedback.fifo, *feedback.candidateQueueDiscipline});
  }
  if (feedback.minimumCandidateDepth)
    decisions.push_back(
        ResizeFifo{*feedback.fifo, *feedback.minimumCandidateDepth});
  result.candidateLimit = decisions.size();
  if (request.boundedQuality)
    result.candidateLimit = std::min<std::uint64_t>(
        result.candidateLimit,
        request.boundedQuality->maximumHardwareRepairProbes);
  for (std::size_t ordinal = 0; ordinal != result.candidateLimit; ++ordinal) {
    if (dispatchDeadlineReached(request.executionPolicy)) {
      const std::uint64_t remaining = result.candidateLimit - ordinal;
      result.candidatesPlanned += remaining;
      result.candidatesReserved += remaining;
      result.candidatesCancelled += remaining;
      break;
    }
    ++result.candidatesPlanned;
    ++result.candidatesReserved;
    HardwareRecipeGrowth growth;
    growth.config = parentPlan.resolvedConfig;
    growth.config.dse.planNodes.clear();
    growth.techModule = parentModule;
    if (const auto *discipline =
            std::get_if<ChangeFifoQueueDiscipline>(&decisions[ordinal]))
      growth.fifoDisciplineChange = *discipline;
    else
      growth.fifoResize = std::get<ResizeFifo>(decisions[ordinal]);
    auto child = materializeTypedModuleSystemGrowth(
        std::move(growth), parentPair.system, artifacts, blobs);
    if (!child)
      return child.takeError();
    const ArtifactRootReference childReference = child->reference;
    JointHardwareReopenRequest childRequest = request;
    llvm::SmallString<256> childJournal(request.journalRoot);
    llvm::sys::path::append(childJournal,
                            "fifo-runtime-feedback-" +
                                std::to_string(ordinal));
    childRequest.journalRoot = childJournal.str().str();
    auto repaired = executeTypedModuleHardwareReopen(
        parentPlan, parentExecution, policy, feedback.parentMapping,
        std::move(*child), std::move(childRequest),
        "spatial_fifo_hardware_repair", artifacts, blobs);
    if (!repaired)
      return repaired.takeError();
    ++result.candidatesConsumed;
    result.childSystems.push_back(childReference);
    result.reuseDispositions.push_back(repaired->disposition);
    result.executions.push_back(std::move(repaired->execution));
  }
  const std::uint64_t settled = result.candidatesConsumed +
                                result.candidatesRejected +
                                result.candidatesCancelled;
  if (settled > result.candidatesReserved)
    return invalid("FIFO hardware repair candidate ledger overflowed");
  result.candidatesRejected += result.candidatesReserved - settled;
  if (result.candidatesPlanned != result.candidatesReserved)
    return invalid("FIFO hardware repair candidate ledger is not reserved");
  result.bypassAlternativeUnsupported = feedback.bypassCapable;
  return result;
}

llvm::Expected<JointSpatialOperandBufferHardwareRepair>
executeSpatialOperandBufferHardwareFeedbackReopen(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy,
    const SpatialOperandQueueRuntimeFeedback &feedback,
    JointHardwareReopenRequest request, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  JointSpatialOperandBufferHardwareRepair result{feedback, {}, {}, {}};
  if (feedback.disposition !=
          SpatialOperandQueueRuntimeFeedbackDisposition::Exact ||
      !feedback.parentMapping || !feedback.owners || !feedback.repairTarget)
    return result;
  if (parentPlan.pairOutputs.size() != 1 || request.journalRoot.empty())
    return invalid(
        "operand-buffer hardware repair requires one exact parent pair");
  if (request.boundedQuality &&
      request.boundedQuality->maximumHardwareRepairProbes == 0)
    return invalid("bounded operand-buffer repair requires a positive probe "
                   "limit");
  auto parentMapping =
      mapping::importSystemMapping(*feedback.parentMapping, artifacts);
  if (!parentMapping)
    return parentMapping.takeError();
  if (parentMapping->view().fabricIdentity() !=
      parentPlan.pairOutputs.front().pair.system.artifact)
    return invalid("operand-buffer feedback parent Mapping names another "
                   "System");
  if (parentMapping->view().dataflowIdentity() !=
          feedback.owners->dataflow.artifact ||
      !llvm::is_contained(
          parentMapping->view().executionBindings().spatialMappingImports(),
          feedback.owners->spatialMapping))
    return invalid("operand-buffer feedback is not attached to its parent "
                   "SystemMapping");
  auto feedbackSpatial =
      mapping::importSpatialMapping(feedback.owners->spatialMapping, artifacts);
  if (!feedbackSpatial)
    return feedbackSpatial.takeError();
  if (feedbackSpatial->view().dataflowIdentity() !=
          feedback.owners->dataflow.artifact ||
      feedbackSpatial->view().fabricIdentity() !=
          feedback.owners->fabric.artifact ||
      feedbackSpatial->view().techMappingIdentity() !=
          feedback.owners->techMapping.artifact)
    return invalid("operand-buffer feedback owner identities disagree");
  auto module =
      fabric::importEntireFabricRoot(feedback.owners->fabric, artifacts);
  if (!module)
    return module.takeError();
  if (module->view().rootKind() != fabric::FabricRootKind::Module)
    return invalid("operand-buffer feedback names a non-Module Fabric root");
  const SpatialOperandBufferRepairTarget &target = *feedback.repairTarget;
  const auto currentMode = module->view().peOperandBufferMode(target.pe);
  if (!currentMode || *currentMode != target.currentMode ||
      module->view().peOperandBufferSize(target.pe) !=
          target.currentEntriesPerAllocationUnit)
    return invalid("operand-buffer feedback target disagrees with its Module");
  if (target.candidateEntriesPerAllocationUnit <=
      target.currentEntriesPerAllocationUnit)
    return invalid("operand-buffer feedback does not request growth");
  std::optional<::fabric::OperandBufferMode> expectedSeparatedMode;
  if (target.currentMode == ::fabric::OperandBufferMode::AllFuShare)
    expectedSeparatedMode = ::fabric::OperandBufferMode::PerInputPort;
  else if (target.currentMode == ::fabric::OperandBufferMode::PerInputPort)
    expectedSeparatedMode = ::fabric::OperandBufferMode::PerInstruction;
  if (target.separatedMode != expectedSeparatedMode)
    return invalid("operand-buffer mode feedback is not the next separated "
                   "mode");
  std::vector<SpatialMicroarchitectureDecision> decisions;
  if (target.separatedMode)
    decisions.push_back(
        ChangeTemporalOperandBufferMode{target.pe, *target.separatedMode});
  decisions.push_back(ResizeTemporalOperandBuffer{
      target.pe, target.candidateEntriesPerAllocationUnit});
  result.candidateLimit = decisions.size();
  if (request.boundedQuality)
    result.candidateLimit = std::min<std::uint64_t>(
        result.candidateLimit,
        request.boundedQuality->maximumHardwareRepairProbes);
  for (std::size_t ordinal = 0; ordinal != result.candidateLimit; ++ordinal) {
    if (dispatchDeadlineReached(request.executionPolicy)) {
      const std::uint64_t remaining = result.candidateLimit - ordinal;
      result.candidatesPlanned += remaining;
      result.candidatesReserved += remaining;
      result.candidatesCancelled += remaining;
      break;
    }
    ++result.candidatesPlanned;
    ++result.candidatesReserved;
    HardwareRecipeGrowth growth;
    growth.config = parentPlan.resolvedConfig;
    growth.config.dse.planNodes.clear();
    growth.techModule = feedback.owners->fabric;
    if (const auto *mode =
            std::get_if<ChangeTemporalOperandBufferMode>(&decisions[ordinal]))
      growth.operandBufferModeChange = *mode;
    else
      growth.operandBufferResize =
          std::get<ResizeTemporalOperandBuffer>(decisions[ordinal]);
    auto child = materializeTypedModuleSystemGrowth(
        std::move(growth), parentPlan.pairOutputs.front().pair.system,
        artifacts, blobs);
    if (!child)
      return child.takeError();
    const ArtifactRootReference childReference = child->reference;
    JointHardwareReopenRequest childRequest = request;
    llvm::SmallString<256> childJournal(request.journalRoot);
    llvm::sys::path::append(childJournal, "operand-buffer-runtime-feedback-" +
                                              std::to_string(ordinal));
    childRequest.journalRoot = childJournal.str().str();
    auto repaired = executeTypedModuleHardwareReopen(
        parentPlan, parentExecution, policy, *feedback.parentMapping,
        std::move(*child), std::move(childRequest),
        "spatial_operand_buffer_hardware_repair", artifacts, blobs);
    if (!repaired)
      return repaired.takeError();
    ++result.candidatesConsumed;
    if (llvm::none_of(result.childSystems, [&](const auto &existing) {
          return existing == childReference;
        })) {
      result.childSystems.push_back(childReference);
      result.reuseDispositions.push_back(repaired->disposition);
      result.executions.push_back(std::move(repaired->execution));
    }
  }
  const std::uint64_t settled = result.candidatesConsumed +
                                result.candidatesRejected +
                                result.candidatesCancelled;
  if (settled > result.candidatesReserved)
    return invalid(
        "operand-buffer hardware repair candidate ledger overflowed");
  result.candidatesRejected += result.candidatesReserved - settled;
  if (result.candidatesPlanned != result.candidatesReserved)
    return invalid("operand-buffer hardware repair candidate ledger is not "
                   "reserved");
  return result;
}

} // namespace loom::dse

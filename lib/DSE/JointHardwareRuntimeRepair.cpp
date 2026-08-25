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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::dse {

using namespace joint_reopen_detail;

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
    auto scheduler = SiteScheduler::create(run.siteCapacity);
    if (!scheduler)
      return scheduler.takeError();
    loom::pnr::PnrDerivedContextSession derivedContextSession;
    return executeJointPlan(plan, run.evidence, run, *scheduler, artifacts,
                            blobs);
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
      firstMapping(*coldExecution);
  const std::optional<ArtifactRootReference> incrementalMapping =
      firstMapping(*execution);
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
  coldExecution->summary.selectedMapping = coldMapping;
  execution->summary.selectedMapping = incrementalMapping;

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
  return JointResourceTimeAdjacentRepair{
      *parentMapping,        seed->reference(),  std::move(*childPlan),
      coldMapping,           incrementalMapping, std::move(*coldExecution),
      std::move(*execution), reuseDisposition};
}

namespace {

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
  if (parentPlan.pairOutputs.size() != 1)
    return invalid("typed Module hardware repair requires one exact pair");
  if (!child.mappingImpact)
    return invalid("typed Module hardware child has no Mapping impact");
  const JointDesignPair &parentPair = parentPlan.pairOutputs.front().pair;
  auto importedParentMapping =
      mapping::importSystemMapping(parentMapping, artifacts);
  if (!importedParentMapping)
    return importedParentMapping.takeError();
  if (importedParentMapping->view().fabricIdentity() !=
      parentPair.system.artifact)
    return invalid("typed Module repair parent Mapping names another System");

  auto rebased = rebaseJointMappingFrontier(
      parentPlan, parentExecution, child.reference, child.moduleCorrespondences,
      &*child.mappingImpact, artifacts);
  if (!rebased)
    return rebased.takeError();
  const JointDesignMappingSeed *mappingSeed =
      rebased->seed.techMappings.empty() &&
              rebased->seed.spatialMappings.empty()
          ? nullptr
          : &rebased->seed;
  auto timing = normalizedTimingProfiles(child.reference, artifacts);
  if (!timing)
    return timing.takeError();
  auto repairPolicy =
      JointDesignPolicy::get(1, 1, 1, policy.maximumTechMappingsPerModule(),
                             policy.maximumSpatialMappingsPerPair());
  if (!repairPolicy)
    return repairPolicy.takeError();
  ResolvedConfig childConfig = child.config;
  childConfig.dse.planNodes.clear();
  childConfig.dse.systemPnr.search.completionGoal =
      ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  auto childPlan = buildJointDesignExplorationPlan(
      {{parentPair.software.workloads}, {child.reference}}, *timing,
      *repairPolicy, childConfig, artifacts, mappingSeed,
      parentPlan.systemBindingPartitions);
  if (!childPlan)
    return childPlan.takeError();
  if (rebased->disposition == JointMappingReuseDisposition::Preserved &&
      child.executionBindingCorrespondence) {
    auto context = deriveSystemMappingMigrationContext(*childPlan);
    if (!context)
      return context.takeError();
    auto seed = pnr::finalizeSystemMappingMigrationSeed(
        parentMapping, *child.executionBindingCorrespondence, *context,
        artifacts);
    if (!seed)
      return seed.takeError();
    if (llvm::Error error = bindFinalizedSystemMappingMigrationSeed(
            *childPlan, seed->reference(), artifacts))
      return std::move(error);
  }

  auto scheduler = SiteScheduler::create(std::move(request.siteCapacity));
  if (!scheduler)
    return scheduler.takeError();
  loom::pnr::PnrDerivedContextSession derivedContextSession;
  const auto begin = std::chrono::steady_clock::now();
  auto execution = executeJointPlan(*childPlan, request.evidence, request,
                                    *scheduler, artifacts, blobs);
  if (!execution)
    return execution.takeError();
  const std::uint64_t elapsedNanoseconds = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
  execution->summary.preservedTechMappings =
      rebased->accounting.preservedTechMappings;
  execution->summary.preservedSpatialMappings =
      rebased->accounting.preservedSpatialMappings;
  execution->summary.repairedTechMappings =
      rebased->accounting.repairedTechMappings;
  execution->summary.repairedSpatialMappings =
      rebased->accounting.repairedSpatialMappings;
  execution->summary.invalidatedTechMappings =
      rebased->accounting.invalidatedTechMappings;
  execution->summary.invalidatedSpatialMappings =
      rebased->accounting.invalidatedSpatialMappings;
  execution->summary.parentTechDecisions =
      rebased->accounting.parentTechDecisions;
  execution->summary.parentSpatialDecisions =
      rebased->accounting.parentSpatialDecisions;
  execution->summary.preservedTechDecisions =
      rebased->accounting.preservedTechDecisions;
  execution->summary.preservedSpatialDecisions =
      rebased->accounting.preservedSpatialDecisions;
  execution->summary.reopenedTechDecisions =
      rebased->accounting.reopenedTechDecisions;
  execution->summary.reopenedSpatialDecisions =
      rebased->accounting.reopenedSpatialDecisions;
  execution->summary.repairedTechDecisions =
      rebased->accounting.repairedTechDecisions;
  execution->summary.repairedSpatialDecisions =
      rebased->accounting.repairedSpatialDecisions;
  execution->summary.invalidationRootCount =
      rebased->accounting.invalidationRootCount;
  execution->summary.invalidationConeDecisionCount =
      rebased->accounting.invalidationConeDecisionCount;
  execution->summary.parentRouteNodeCount =
      rebased->accounting.parentRouteNodeCount;
  execution->summary.preservedRouteNodeCount =
      rebased->accounting.preservedRouteNodeCount;
  execution->summary.reopenedRouteNodeCount =
      rebased->accounting.reopenedRouteNodeCount;
  execution->summary.repairedRouteNodeCount =
      rebased->accounting.repairedRouteNodeCount;
  execution->summary.parentServiceLegCount =
      rebased->accounting.parentServiceLegCount;
  execution->summary.preservedServiceLegCount =
      rebased->accounting.preservedServiceLegCount;
  execution->summary.reopenedServiceLegCount =
      rebased->accounting.reopenedServiceLegCount;
  execution->summary.parentThreadBindingCount =
      rebased->accounting.parentThreadBindingCount;
  execution->summary.preservedThreadBindingCount =
      rebased->accounting.preservedThreadBindingCount;
  execution->summary.reopenedThreadBindingCount =
      rebased->accounting.reopenedThreadBindingCount;
  execution->summary.parentGraphBindingCount =
      rebased->accounting.parentGraphBindingCount;
  execution->summary.preservedGraphBindingCount =
      rebased->accounting.preservedGraphBindingCount;
  execution->summary.reopenedGraphBindingCount =
      rebased->accounting.reopenedGraphBindingCount;
  execution->summary.parentResourceUseCount =
      rebased->accounting.parentResourceUseCount;
  execution->summary.preservedResourceUseCount =
      rebased->accounting.preservedResourceUseCount;
  execution->summary.reopenedResourceUseCount =
      rebased->accounting.reopenedResourceUseCount;
  execution->summary.parentServiceRealizationCount =
      rebased->accounting.parentServiceRealizationCount;
  execution->summary.preservedServiceRealizationCount =
      rebased->accounting.preservedServiceRealizationCount;
  execution->summary.reopenedServiceRealizationCount =
      rebased->accounting.reopenedServiceRealizationCount;
  if (rebased->disposition == JointMappingReuseDisposition::ColdFallback)
    execution->summary.coldReopenWallTimeNanoseconds =
        execution->summary.executionWallTimeNanoseconds;
  else
    execution->summary.incrementalReopenWallTimeNanoseconds =
        execution->summary.executionWallTimeNanoseconds;
  if (auto selected = firstMapping(*execution)) {
    execution->summary.selectedMapping = *selected;
    execution->summary.selectedPlanOrdinal = 0;
  }
  execution->summary.verifiedAlternatives = mappingCount(*execution);
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = operation;
        fields["parent_mapping"] =
            formatArtifactIdentityHex(parentMapping.artifact);
        fields["child_system"] =
            formatArtifactIdentityHex(child.reference.artifact);
        fields["mapping_reuse_disposition"] =
            jointMappingReuseDispositionSpelling(rebased->disposition);
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
        fields["parent_thread_binding_count"] =
            rebased->accounting.parentThreadBindingCount;
        fields["preserved_thread_binding_count"] =
            rebased->accounting.preservedThreadBindingCount;
        fields["reopened_thread_binding_count"] =
            rebased->accounting.reopenedThreadBindingCount;
        fields["parent_graph_binding_count"] =
            rebased->accounting.parentGraphBindingCount;
        fields["preserved_graph_binding_count"] =
            rebased->accounting.preservedGraphBindingCount;
        fields["reopened_graph_binding_count"] =
            rebased->accounting.reopenedGraphBindingCount;
        fields["parent_resource_use_count"] =
            rebased->accounting.parentResourceUseCount;
        fields["preserved_resource_use_count"] =
            rebased->accounting.preservedResourceUseCount;
        fields["reopened_resource_use_count"] =
            rebased->accounting.reopenedResourceUseCount;
        fields["parent_service_realization_count"] =
            rebased->accounting.parentServiceRealizationCount;
        fields["preserved_service_realization_count"] =
            rebased->accounting.preservedServiceRealizationCount;
        fields["reopened_service_realization_count"] =
            rebased->accounting.reopenedServiceRealizationCount;
        fields["rebase_failure_count"] = rebased->failures.size();
        fields["system_mapping_count"] = mappingCount(*execution);
        fields["wall_time_ns"] = elapsedNanoseconds;
        fields["liveness"] = "requires_child_cgra_replay";
        fields["reconfiguration_support"] = "unsupported";
      });
  return TypedModuleHardwareRepair{std::move(*execution), rebased->disposition};
}

} // namespace

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
  if (!feedback.fifo || !feedback.minimumCandidateDepth)
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
  result.candidateLimit = 1;
  if (dispatchDeadlineReached(request.executionPolicy)) {
    result.candidatesPlanned = 1;
    result.candidatesReserved = 1;
    result.candidatesCancelled = 1;
    return result;
  }
  result.candidatesPlanned = 1;
  result.candidatesReserved = 1;
  ArtifactRootReference parentModule{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version,
      parentSpatial->view().fabricIdentity()};

  HardwareRecipeGrowth growth;
  growth.config = parentPlan.resolvedConfig;
  growth.config.dse.planNodes.clear();
  growth.techModule = parentModule;
  growth.fifoResize =
      ResizeFifo{*feedback.fifo, *feedback.minimumCandidateDepth};
  auto child = materializeTypedModuleSystemGrowth(
      std::move(growth), parentPair.system, artifacts, blobs);
  if (!child)
    return child.takeError();
  const ArtifactRootReference childReference = child->reference;
  auto repaired = executeTypedModuleHardwareReopen(
      parentPlan, parentExecution, policy, feedback.parentMapping,
      std::move(*child), std::move(request), "spatial_fifo_hardware_repair",
      artifacts, blobs);
  if (!repaired)
    return repaired.takeError();
  result.candidatesConsumed = 1;
  if (result.candidatesConsumed + result.candidatesRejected !=
      result.candidatesReserved)
    return invalid("FIFO hardware repair candidate ledger is not closed");
  result.childSystems.push_back(childReference);
  result.reuseDispositions.push_back(repaired->disposition);
  result.executions.push_back(std::move(repaired->execution));
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
  for (std::size_t ordinal = 0; ordinal != decisions.size(); ++ordinal) {
    if (dispatchDeadlineReached(request.executionPolicy)) {
      const std::uint64_t remaining = decisions.size() - ordinal;
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

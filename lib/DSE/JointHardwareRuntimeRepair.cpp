#include "DSE/JointHardwareReopen.h"

#include "JointHardwareReopenInternal.h"

#include "JointHardwareReopenExecution.h"
#include "ResourceTimeAdjacentMappingSelection.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/MappingDebugLog.h"
#include "DSE/ExecutionJournal.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "DSE/HardwareDecision.h"
#include "DSE/HardwareMutationRepairRecord.h"
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

void saturatingAdd(std::uint64_t &target, std::uint64_t value);

struct ReopenedRootLowerMappingPlan final {
  JointDesignExplorationPlan plan;
  std::vector<PlanOutputRef> spatialMappings;
};

llvm::Expected<ReopenedRootLowerMappingPlan> buildReopenedRootLowerMappingPlan(
    const JointDesignExplorationPlan &completePlan,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    const ArtifactStore &artifacts) {
  constexpr std::size_t constraintInputOrdinal = 1;
  if (completePlan.pairOutputs.size() != 1 || reopenedRoots.empty())
    return invalid("reopened-root lower Mapping requires one exact pair and "
                   "a nonempty root set");
  const JointDesignPlanPair &completePair = completePlan.pairOutputs.front();
  const std::uint64_t systemNodeOrdinal =
      completePair.systemMappings.producerNodeOrdinal;
  if (systemNodeOrdinal >= completePlan.resolvedConfig.dse.planNodes.size() ||
      systemNodeOrdinal + 1 !=
          completePlan.resolvedConfig.dse.planNodes.size())
    return invalid("reopened-root lower Mapping requires a terminal System "
                   "provider");
  const auto *systemNode = std::get_if<GeneratePlanNodeDefinition>(
      &completePlan.resolvedConfig.dse.planNodes[systemNodeOrdinal]);
  if (!systemNode ||
      systemNode->descriptor !=
          applicationSystemPnrCandidateGeneratorDescriptor().reference())
    return invalid("reopened-root lower Mapping has no canonical System "
                   "provider");
  if (completePair.techMappings.empty() ||
      completePair.spatialMappings.empty())
    return invalid("reopened-root lower Mapping has no generated frontier");

  auto dataflowArtifact = ::dataflow::importCanonicalDataflow(
      completePair.pair.software.dataflow, artifacts);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  auto systemArtifact =
      fabric::importEntireFabricRoot(completePair.pair.system, artifacts);
  if (!systemArtifact)
    return systemArtifact.takeError();
  auto system = fabric::requireSystemRoot(systemArtifact->view());
  if (!system)
    return system.takeError();
  auto constraints = mapping::finalizeEmptySystemMappingConstraintSet(
      *dataflow, *system, reopenedRoots, artifacts);
  if (!constraints)
    return constraints.takeError();

  JointDesignExplorationPlan lowerPlan = completePlan;
  JointDesignPlanPair &lowerPair = lowerPlan.pairOutputs.front();
  for (const PlanOutputRef output : lowerPair.techMappings) {
    if (output.outputSlotOrdinal != 0 ||
        output.producerNodeOrdinal >= systemNodeOrdinal)
      return invalid("reopened-root TechMapping output names a foreign node");
    auto *techNode = std::get_if<GeneratePlanNodeDefinition>(
        &lowerPlan.resolvedConfig.dse.planNodes[output.producerNodeOrdinal]);
    if (!techNode ||
        techNode->descriptor !=
            applicationGraphTechMappingCandidateGeneratorDescriptor()
                .reference() ||
        techNode->inputBindings.size() <= constraintInputOrdinal)
      return invalid("reopened-root lower Mapping has a noncanonical Tech "
                     "provider");
    techNode->inputBindings[constraintInputOrdinal] =
        ExactPlanArtifacts{{constraints->reference()}};
  }
  lowerPlan.resolvedConfig.dse.planNodes.pop_back();
  std::vector<PlanOutputRef> spatialMappings = lowerPair.spatialMappings;
  lowerPlan.pairOutputs.clear();
  auto admitted = projectResolvedDseConfigView(lowerPlan.resolvedConfig);
  if (!admitted)
    return admitted.takeError();
  return ReopenedRootLowerMappingPlan{std::move(lowerPlan),
                                      std::move(spatialMappings)};
}

llvm::Expected<std::vector<ArtifactRootReference>>
resolveReopenedRootLowerMappingFrontier(
    const ReopenedRootLowerMappingPlan &lowerPlan,
    const JointDesignExecution &execution) {
  const auto *completed =
      std::get_if<CompletedDsePlanExecution>(&execution.planExecution);
  if (!completed)
    return invalid("reopened-root lower Mapping did not materialize its "
                   "complete frontier");
  std::vector<ArtifactRootReference> mappings;
  for (const PlanOutputRef output : lowerPlan.spatialMappings) {
    if (!completed->hasOutput(output))
      return invalid("reopened-root lower Mapping omits a SpatialMapping "
                     "output");
    const llvm::ArrayRef<ArtifactRootReference> available =
        completed->resolve(output);
    mappings.insert(mappings.end(), available.begin(), available.end());
  }
  canonicalizeRoots(mappings);
  if (mappings.empty())
    return invalid("reopened-root lower Mapping frontier is empty");
  return mappings;
}

llvm::Expected<std::vector<ArtifactRootReference>>
materializeHybridSpatialMappingFrontier(
    const JointDesignExplorationPlan &completePlan,
    llvm::ArrayRef<ArtifactRootReference> reopenedSpatialMappings,
    const pnr::SystemMappingMigrationConePartition &cone,
    const ArtifactStore &artifacts) {
  constexpr std::size_t spatialMappingInputOrdinal = 1;
  if (completePlan.pairOutputs.size() != 1)
    return invalid("hybrid SpatialMapping frontier requires one exact pair");
  const JointDesignPlanPair &pair = completePlan.pairOutputs.front();
  if (pair.systemMappings.producerNodeOrdinal >=
      completePlan.resolvedConfig.dse.planNodes.size())
    return invalid("hybrid SpatialMapping frontier names a foreign System "
                   "node");
  const auto *systemNode = std::get_if<GeneratePlanNodeDefinition>(
      &completePlan.resolvedConfig.dse
           .planNodes[pair.systemMappings.producerNodeOrdinal]);
  if (!systemNode ||
      systemNode->descriptor !=
          applicationSystemPnrCandidateGeneratorDescriptor().reference() ||
      systemNode->inputBindings.size() <= spatialMappingInputOrdinal)
    return invalid("hybrid SpatialMapping frontier has no canonical System "
                   "provider");
  const auto *join = std::get_if<BoundedPlanOutputJoin>(
      &systemNode->inputBindings[spatialMappingInputOrdinal]);
  if (!join)
    return invalid("hybrid SpatialMapping frontier has no bounded lower "
                   "join");

  std::vector<ArtifactRootReference> preserved(
      cone.preservedSpatialMappings.begin(),
      cone.preservedSpatialMappings.end());
  canonicalizeRoots(preserved);
  if (preserved.size() != cone.preservedSpatialMappings.size())
    return invalid("hybrid SpatialMapping frontier repeats a preserved root");
  if (preserved.size() > join->maximumArtifacts)
    return invalid("preserved SpatialMapping cone exceeds the System join "
                   "bound");
  std::vector<ArtifactRootReference> reopened(reopenedSpatialMappings.begin(),
                                               reopenedSpatialMappings.end());
  canonicalizeRoots(reopened);
  std::vector<ArtifactRootReference> result = preserved;
  for (const ArtifactRootReference &mappingReference : reopened) {
    if (result.size() >= join->maximumArtifacts)
      break;
    if (llvm::is_contained(result, mappingReference))
      continue;
    auto spatial =
        mapping::importSpatialMapping(mappingReference, artifacts);
    if (!spatial)
      return spatial.takeError();
    auto tech = mapping::importTechMapping(
        {mapping::mappingArtifactSchema.identity.str(),
         mapping::mappingArtifactSchema.version,
         spatial->view().techMappingIdentity()},
        artifacts);
    if (!tech)
      return tech.takeError();
    if (!cone.admitsReplacementGraphs(tech->view().covers()))
      continue;
    result.push_back(mappingReference);
  }
  canonicalizeRoots(result);
  if (result.empty())
    return invalid("hybrid SpatialMapping frontier is empty");
  return result;
}

llvm::Expected<PlanExecutionPolicy> systemOnlyExecutionPolicy(
    const PlanExecutionPolicy &base, std::uint64_t originalSystemNodeOrdinal,
    std::uint64_t consumedDispatches) {
  std::vector<WorkUnitResourceBinding> systemBindings;
  for (const WorkUnitResourceBinding &binding : base.resourceBindings()) {
    if (binding.key.planNodeOrdinal() != originalSystemNodeOrdinal)
      continue;
    auto key = WorkUnitKey::get(0, binding.key.descriptor(),
                                binding.key.stableOrdinal());
    if (!key)
      return key.takeError();
    systemBindings.push_back({std::move(*key), binding.claim});
  }
  llvm::sort(systemBindings,
             [](const WorkUnitResourceBinding &lhs,
                const WorkUnitResourceBinding &rhs) {
               return lhs.key < rhs.key;
             });
  std::optional<std::uint64_t> remainingDispatches;
  if (base.maximumDispatches())
    remainingDispatches =
        consumedDispatches >= *base.maximumDispatches()
            ? 0
            : *base.maximumDispatches() - consumedDispatches;
  return PlanExecutionPolicy::get(
      base.workerCount(), base.inProcessClaim(), base.externalSite(),
      systemBindings, remainingDispatches,
      base.dispatchNotAfterUnixNanoseconds());
}

void mergeLowerMappingExecutionAccounting(
    JointDesignExecutionSummary &systemSummary,
    const JointDesignExecutionSummary &lowerSummary) {
  saturatingAdd(systemSummary.techMappingInvocationCount,
                lowerSummary.techMappingInvocationCount);
  saturatingAdd(systemSummary.spatialPnrInvocationCount,
                lowerSummary.spatialPnrInvocationCount);
  saturatingAdd(systemSummary.systemPnrInvocationCount,
                lowerSummary.systemPnrInvocationCount);
  saturatingAdd(systemSummary.techMappingDispatchCount,
                lowerSummary.techMappingDispatchCount);
  saturatingAdd(systemSummary.spatialPnrDispatchCount,
                lowerSummary.spatialPnrDispatchCount);
  saturatingAdd(systemSummary.systemPnrDispatchCount,
                lowerSummary.systemPnrDispatchCount);
  saturatingAdd(systemSummary.techMappingJournalReplayCount,
                lowerSummary.techMappingJournalReplayCount);
  saturatingAdd(systemSummary.spatialPnrJournalReplayCount,
                lowerSummary.spatialPnrJournalReplayCount);
  saturatingAdd(systemSummary.systemPnrJournalReplayCount,
                lowerSummary.systemPnrJournalReplayCount);
  saturatingAdd(systemSummary.executionWallTimeNanoseconds,
                lowerSummary.executionWallTimeNanoseconds);
  if (systemSummary.timeToFirstFeasibleWallTimeNanoseconds)
    saturatingAdd(*systemSummary.timeToFirstFeasibleWallTimeNanoseconds,
                  lowerSummary.executionWallTimeNanoseconds);
  if (systemSummary.timeToBestWallTimeNanoseconds)
    saturatingAdd(*systemSummary.timeToBestWallTimeNanoseconds,
                  lowerSummary.executionWallTimeNanoseconds);
}

} // namespace

std::vector<::dataflow::RootThreadLaunchRef> deriveSystemPartitionDelta(
    llvm::ArrayRef<pnr::SystemBindingPartitionIntent> parent,
    llvm::ArrayRef<pnr::SystemBindingPartitionIntent> child) {
  std::vector<::dataflow::RootThreadLaunchRef> changed;
  for (const pnr::SystemBindingPartitionIntent &parentPartition : parent) {
    const auto childPartition =
        llvm::find_if(child, [&](const auto &candidate) {
          return candidate.root == parentPartition.root;
        });
    if (childPartition == child.end() ||
        childPartition->partitionCount != parentPartition.partitionCount)
      changed.push_back(parentPartition.root);
  }
  for (const pnr::SystemBindingPartitionIntent &childPartition : child)
    if (llvm::none_of(parent, [&](const auto &candidate) {
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

llvm::Expected<JointResourceTimeAdjacentRepair>
executeResourceTimeAdjacentMappingRepair(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy,
    llvm::ArrayRef<pnr::SystemBindingPartitionIntent> childPartitions,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> reopenedRoots,
    JointResourceTimeMappingVerifier mappingVerifier,
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
  const std::vector<::dataflow::RootThreadLaunchRef> partitionDelta =
      deriveSystemPartitionDelta(parentPlan.systemBindingPartitions,
                                 childPartitions);
  std::vector<::dataflow::RootThreadLaunchRef> canonicalReopenedRoots(
      reopenedRoots.begin(), reopenedRoots.end());
  llvm::sort(canonicalReopenedRoots, [](const auto &lhs, const auto &rhs) {
    if (lhs.artifact != rhs.artifact)
      return lhs.artifact.bytes() < rhs.artifact.bytes();
    return lhs.entity.value() < rhs.entity.value();
  });
  if (std::adjacent_find(canonicalReopenedRoots.begin(),
                         canonicalReopenedRoots.end()) !=
          canonicalReopenedRoots.end() ||
      canonicalReopenedRoots != partitionDelta)
    return invalid("resource-time repair roots differ from the exact "
                   "System partition delta");
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
      request.stoppingPolicy == JointDesignStoppingPolicy::FirstVerified
          ? ResolvedPnrCompletionGoal::ExhaustConfiguredWork
          : ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  auto childPlan = buildJointDesignExplorationPlan(
      {{software.workloads}, {system}}, *timing, *repairPolicy, childConfig,
      artifacts, nullptr, childPartitions);
  if (!childPlan)
    return childPlan.takeError();
  JointDesignExplorationPlan coldPlan = *childPlan;
  auto importedParentMapping =
      mapping::importSystemMapping(*parentMapping, artifacts);
  if (!importedParentMapping)
    return importedParentMapping.takeError();
  auto parentCone = pnr::projectSystemMappingMigrationConePartition(
      importedParentMapping->view(), reopenedRoots, artifacts);
  if (!parentCone)
    return parentCone.takeError();
  auto lowerPlan = buildReopenedRootLowerMappingPlan(
      *childPlan, parentCone->reopenedRoots, artifacts);
  if (!lowerPlan)
    return lowerPlan.takeError();
  const std::uint64_t originalSystemNodeOrdinal =
      childPlan->pairOutputs.front().systemMappings.producerNodeOrdinal;

  JointHardwareReopenRequest coldRequest = request;
  llvm::SmallString<256> coldJournal(coldRequest.journalRoot);
  llvm::sys::path::append(coldJournal, "cold");
  coldRequest.journalRoot = coldJournal.str().str();
  JointHardwareReopenRequest lowerRequest = request;
  llvm::SmallString<256> lowerJournal(lowerRequest.journalRoot);
  llvm::sys::path::append(lowerJournal, "incremental", "lower");
  lowerRequest.journalRoot = lowerJournal.str().str();
  JointHardwareReopenRequest systemRequest = request;
  llvm::SmallString<256> systemJournal(systemRequest.journalRoot);
  llvm::sys::path::append(systemJournal, "incremental", "system");
  systemRequest.journalRoot = systemJournal.str().str();
  const auto executeIndependent = [&](const JointDesignExplorationPlan &plan,
                                      const JointHardwareReopenRequest &run)
      -> llvm::Expected<JointDesignExecution> {
    loom::pnr::PnrDerivedContextSession derivedContextSession;
    return executeJointRepairPlan(plan, *repairPolicy, run, artifacts, blobs);
  };
  auto coldExecution = executeIndependent(coldPlan, coldRequest);
  if (!coldExecution)
    return coldExecution.takeError();
  std::optional<ResourceTimeSpectrumFunnelResult> coldSelectionSpectrum;
  std::vector<ArtifactRootReference> coldEligibleMappings;
  std::vector<DsePlanIncompleteReason> coldExecutionIncompleteReasons;
  if (request.stoppingPolicy == JointDesignStoppingPolicy::FirstVerified) {
    auto selected = selectResourceTimePartitionMapping(
        *coldExecution, software.dataflow, system, childPartitions,
        reopenedRoots, nullptr, {}, request.spectrumEndpoint,
        JointResourceTimeMappingRepairSide::Cold, mappingVerifier, artifacts);
    if (!selected)
      return selected.takeError();
    coldSelectionSpectrum = std::move(selected->spectrum);
    coldEligibleMappings = std::move(selected->eligibleMappings);
    coldExecutionIncompleteReasons =
        std::move(selected->executionIncompleteReasons);
  }
  coldExecution->summary.coldReopenWallTimeNanoseconds =
      coldExecution->summary.executionWallTimeNanoseconds;

  auto lowerScheduler = SiteScheduler::create(lowerRequest.siteCapacity);
  if (!lowerScheduler)
    return lowerScheduler.takeError();
  llvm::Expected<JointDesignExecution> lowerExecution = [&]() {
    loom::pnr::PnrDerivedContextSession derivedContextSession;
    return executeJointPlan(lowerPlan->plan, lowerRequest.evidence,
                            lowerRequest,
                            *lowerScheduler, artifacts, blobs);
  }();
  if (!lowerExecution)
    return lowerExecution.takeError();
  std::vector<DsePlanIncompleteReason> incrementalPrerequisiteReasons;
  if (const auto *incomplete = std::get_if<IncompleteDsePlanExecution>(
          &lowerExecution->planExecution))
    incrementalPrerequisiteReasons.push_back(incomplete->reason());
  lowerExecution->summary.incrementalReopenWallTimeNanoseconds =
      lowerExecution->summary.executionWallTimeNanoseconds;
  if (!incrementalPrerequisiteReasons.empty()) {
    const std::vector<ArtifactRootReference> coldMappings =
        mappingRoots(*coldExecution);
    auto coldVerification = independentlyVerifyChildMappings(
        coldMappings, software.dataflow, system, artifacts);
    if (!coldVerification)
      return coldVerification.takeError();
    auto incrementalVerification = independentlyVerifyChildMappings(
        {}, software.dataflow, system, artifacts);
    if (!incrementalVerification)
      return incrementalVerification.takeError();
    return JointResourceTimeAdjacentRepair{
        *parentMapping,
        std::nullopt,
        std::move(*childPlan),
        coldExecution->summary.selectedMapping,
        std::nullopt,
        std::move(coldSelectionSpectrum),
        std::nullopt,
        std::move(coldEligibleMappings),
        {},
        std::move(coldExecutionIncompleteReasons),
        std::move(incrementalPrerequisiteReasons),
        std::move(*coldExecution),
        std::move(*lowerExecution),
        std::nullopt,
        JointMappingReuseDisposition::ColdFallback,
        std::move(*coldVerification),
        std::move(*incrementalVerification)};
  }
  auto reopenedSpatialMappings =
      resolveReopenedRootLowerMappingFrontier(*lowerPlan, *lowerExecution);
  if (!reopenedSpatialMappings)
    return reopenedSpatialMappings.takeError();
  auto hybridSpatialMappings = materializeHybridSpatialMappingFrontier(
      *childPlan, *reopenedSpatialMappings, *parentCone, artifacts);
  if (!hybridSpatialMappings)
    return hybridSpatialMappings.takeError();
  if (llvm::Error error = bindImmutableSpatialMappingFrontier(
          *childPlan, *hybridSpatialMappings, artifacts))
    return std::move(error);

  auto correspondence =
      pnr::SystemExecutionBindingCorrespondence::getIdentity(system, artifacts);
  if (!correspondence)
    return correspondence.takeError();
  auto context = deriveSystemMappingMigrationContext(*childPlan);
  if (!context)
    return context.takeError();
  auto seed = pnr::finalizeSystemMappingMigrationSeed(
      *parentMapping, *correspondence, *context, parentCone->reopenedRoots,
      artifacts);
  if (!seed)
    return seed.takeError();
  if (llvm::Error error = bindFinalizedSystemMappingMigrationSeed(
          *childPlan, seed->reference(), artifacts))
    return std::move(error);

  std::uint64_t lowerDispatches =
      lowerExecution->summary.techMappingDispatchCount;
  saturatingAdd(lowerDispatches,
                lowerExecution->summary.spatialPnrDispatchCount);
  saturatingAdd(lowerDispatches,
                lowerExecution->summary.systemPnrDispatchCount);
  auto systemPolicy = systemOnlyExecutionPolicy(
      request.executionPolicy, originalSystemNodeOrdinal, lowerDispatches);
  if (!systemPolicy)
    return systemPolicy.takeError();
  systemRequest.executionPolicy = std::move(*systemPolicy);
  auto execution = executeIndependent(*childPlan, systemRequest);
  if (!execution)
    return execution.takeError();
  std::optional<ResourceTimeSpectrumFunnelResult> incrementalSelectionSpectrum;
  std::vector<ArtifactRootReference> incrementalEligibleMappings;
  std::vector<DsePlanIncompleteReason> incrementalExecutionIncompleteReasons;
  if (request.stoppingPolicy == JointDesignStoppingPolicy::FirstVerified) {
    auto selected = selectResourceTimePartitionMapping(
        *execution, software.dataflow, system, childPartitions, reopenedRoots,
        &importedParentMapping->view(), incrementalPrerequisiteReasons,
        request.spectrumEndpoint,
        JointResourceTimeMappingRepairSide::Incremental, mappingVerifier,
        artifacts);
    if (!selected)
      return selected.takeError();
    incrementalSelectionSpectrum = std::move(selected->spectrum);
    incrementalEligibleMappings = std::move(selected->eligibleMappings);
    incrementalExecutionIncompleteReasons =
        std::move(selected->executionIncompleteReasons);
  }
  std::vector<JointDesignInvocationManifestReference> lowerInvocations;
  if (llvm::Error error = retainJointDesignExecutionInvocations(
          lowerInvocations, *lowerExecution))
    return std::move(error);
  if (llvm::Error error = attachJointDesignSupportingInvocationManifests(
          *execution, lowerInvocations))
    return std::move(error);
  mergeLowerMappingExecutionAccounting(execution->summary,
                                       lowerExecution->summary);
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

  execution->summary.preservedSpatialMappings =
      parentCone->preservedSpatialMappings.size();
  execution->summary.preservedTechMappings =
      parentCone->preservedTechMappings.size();
  execution->summary.invalidatedSpatialMappings =
      parentCone->reopenedSpatialMappings.size();
  execution->summary.invalidatedTechMappings =
      parentCone->reopenedTechMappings.size();
  execution->summary.invalidationRootCount = parentCone->reopenedRoots.size();
  execution->summary.parentThreadBindingCount =
      parentCone->preservedThreadBindings + parentCone->reopenedThreadBindings;
  execution->summary.preservedThreadBindingCount =
      parentCone->preservedThreadBindings;
  execution->summary.reopenedThreadBindingCount =
      parentCone->reopenedThreadBindings;
  execution->summary.parentGraphBindingCount =
      parentCone->preservedGraphBindings + parentCone->reopenedGraphBindings;
  execution->summary.preservedGraphBindingCount =
      parentCone->preservedGraphBindings;
  execution->summary.reopenedGraphBindingCount =
      parentCone->reopenedGraphBindings;
  if (incrementalMapping) {
    auto importedChildMapping =
        mapping::importSystemMapping(*incrementalMapping, artifacts);
    if (!importedChildMapping)
      return importedChildMapping.takeError();
    auto preserved = pnr::preservesSystemMappingMigrationCone(
        importedParentMapping->view(), importedChildMapping->view(),
        parentCone->reopenedRoots, artifacts);
    if (!preserved)
      return preserved.takeError();
    if (!*preserved)
      return invalid("resource-time System repair changed an exact "
                     "cone-external System selection");
    auto childCone = pnr::projectSystemMappingMigrationConePartition(
        importedChildMapping->view(), parentCone->reopenedRoots, artifacts);
    if (!childCone)
      return childCone.takeError();
    execution->summary.repairedSpatialMappings =
        childCone->reopenedSpatialMappings.size();
    execution->summary.repairedTechMappings =
        childCone->reopenedTechMappings.size();
  }
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "resource_time_adjacent_mapping_repair";
        fields["reopened_root_count"] = parentCone->reopenedRoots.size();
        fields["preserved_tech_mappings"] =
            execution->summary.preservedTechMappings;
        fields["preserved_spatial_mappings"] =
            execution->summary.preservedSpatialMappings;
        fields["repaired_tech_mappings"] =
            execution->summary.repairedTechMappings;
        fields["repaired_spatial_mappings"] =
            execution->summary.repairedSpatialMappings;
        fields["preserved_system_bindings"] =
            parentCone->preservedSystemBindings();
        fields["reopened_system_bindings"] =
            parentCone->reopenedSystemBindings();
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
  return JointResourceTimeAdjacentRepair{
      *parentMapping,
      seed->reference(),
      std::move(*childPlan),
      coldMapping,
      incrementalMapping,
      std::move(coldSelectionSpectrum),
      std::move(incrementalSelectionSpectrum),
      std::move(coldEligibleMappings),
      std::move(incrementalEligibleMappings),
      std::move(coldExecutionIncompleteReasons),
      std::move(incrementalExecutionIncompleteReasons),
      std::move(*coldExecution),
      std::move(*lowerExecution),
      std::optional<JointDesignExecution>(std::move(*execution)),
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

llvm::Expected<JointDesignExecution> executeIndependentMutationPlan(
    const JointDesignExplorationPlan &plan, const JointDesignPolicy &policy,
    const JointHardwareReopenRequest &request, const ArtifactStore &artifacts,
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
  ArtifactRootReference record;
  JointDesignExecution execution;
  JointMappingReuseDisposition disposition =
      JointMappingReuseDisposition::ColdFallback;
};

/// One hardware alternative is an ordered sequence of exact Module rewrites
/// materialized into a single child System.
using SpatialFifoHardwareAlternative =
    std::vector<SpatialMicroarchitectureDecisionDomain>;

/// The finite hardware alternatives one exact FIFO witness admits: the
/// discipline change on every witnessed target as one child, then the depth
/// probe. The reopen and the runtime witness scheduler derive their candidate
/// counts from this one owner.
std::vector<SpatialFifoHardwareAlternative>
spatialFifoHardwareAlternatives(const SpatialFifoRuntimeFeedback &feedback) {
  std::vector<SpatialFifoHardwareAlternative> alternatives;
  if (!feedback.fifo)
    return alternatives;
  if (feedback.candidateQueueDiscipline &&
      !feedback.disciplineTargets.empty()) {
    SpatialFifoHardwareAlternative sequence;
    for (const auto &target : feedback.disciplineTargets)
      sequence.push_back(ChangeFifoQueueDisciplineDomain{
          target, {*feedback.candidateQueueDiscipline}});
    alternatives.push_back(std::move(sequence));
  }
  if (feedback.minimumCandidateDepth)
    alternatives.push_back(
        {ResizeFifoDomain{*feedback.fifo, {*feedback.minimumCandidateDepth}}});
  return alternatives;
}

/// The finite hardware candidate set one exact operand-queue witness admits:
/// the next separated mode when one exists, then the depth probe.
std::vector<SpatialMicroarchitectureDecision>
spatialOperandBufferHardwareDecisions(
    const SpatialOperandBufferRepairTarget &target) {
  std::vector<SpatialMicroarchitectureDecision> decisions;
  if (target.separatedMode)
    decisions.push_back(
        ChangeTemporalOperandBufferMode{target.pe, *target.separatedMode});
  decisions.push_back(ResizeTemporalOperandBuffer{
      target.pe, target.candidateEntriesPerAllocationUnit});
  return decisions;
}

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
          {std::move(*child.mappingImpact)},
          std::move(child.decisionLineage)},
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
  return TypedModuleHardwareRepair{
      repair->record, std::move(repair->incrementalExecution), disposition};
}

} // namespace

static llvm::Expected<JointHardwareMutationChild>
materializeJointModuleGrowthChild(HardwareRecipeGrowth growth,
                                  const ArtifactRootReference &parentSystem,
                                  const ArtifactStore &artifacts,
                                  const BlobStore &blobs) {
  if (llvm::Error error = registerProductionDseOwners())
    return std::move(error);
  growth.config.dse.planNodes.clear();
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
      {std::move(*materialized->mappingImpact)},
      std::move(materialized->decisionLineage)};
}

llvm::Expected<JointHardwareMutationChild>
materializeJointModuleHardwareMutation(
    ResolvedConfig config, const ArtifactRootReference &parentSystem,
    const ArtifactRootReference &parentModule,
    SpatialMicroarchitectureDecisionDomain decision,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  HardwareRecipeGrowth growth;
  growth.config = std::move(config);
  growth.techModule = parentModule;
  growth.moduleDecision = std::move(decision);
  return materializeJointModuleGrowthChild(std::move(growth), parentSystem,
                                           artifacts, blobs);
}

llvm::Expected<JointHardwareMutationChild>
materializeJointModuleHardwareMutation(
    ResolvedConfig config, const ArtifactRootReference &parentSystem,
    const ArtifactRootReference &parentModule,
    SpatialTopologyDecisionDomain decision, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  HardwareRecipeGrowth growth;
  growth.config = std::move(config);
  growth.techModule = parentModule;
  growth.topologyDecision = std::move(decision);
  return materializeJointModuleGrowthChild(std::move(growth), parentSystem,
                                           artifacts, blobs);
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
                                    {std::move(impact)},
                                    {{systemCompositionCandidateGeneratorKind,
                                      lineage.output, lineage.parents,
                                      lineage.ownerPayload}}};
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
  first.decisionLineage.insert(
      first.decisionLineage.end(),
      std::make_move_iterator(second.decisionLineage.begin()),
      std::make_move_iterator(second.decisionLineage.end()));
  return JointHardwareMutationChild{second.system, std::move(second.config),
                                    std::move(*correspondence),
                                    std::move(first.impacts),
                                    std::move(first.decisionLineage)};
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
    auto executed = executeIndependentMutationPlan(
        *coldPlan, policy, coldRequest, artifacts, blobs);
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
  // The record is content-addressed from every other field, so its
  // reference is assigned after publication; the parent Mapping reference
  // only occupies the slot until then.
  JointHardwareMutationRepair repair{parentMapping,
                                     parentMapping,
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
  auto record = publishHardwareMutationRepairRecord(repair, artifacts);
  if (!record)
    return record.takeError();
  repair.record = record->reference();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "joint_hardware_mutation_repair";
        fields["record"] = formatArtifactIdentityHex(repair.record.artifact);
        llvm::json::Array families;
        for (const HardwareImpactProjection &impact : repair.child.impacts)
          families.push_back(hardwareMutationFamilySpelling(impact.family));
        fields["families"] = std::move(families);
        fields["parent_mapping"] =
            formatArtifactIdentityHex(parentMapping.artifact);
        fields["child_system"] =
            formatArtifactIdentityHex(repair.child.system.artifact);
        fields["mapping_reuse_disposition"] =
            jointMappingReuseDispositionSpelling(repair.rebase.disposition);
        fields["system_mapping_reuse_disposition"] =
            jointSystemMappingReuseDispositionSpelling(systemDisposition);
        fields["rebase_failure_count"] = repair.rebase.failures.size();
        fields["cold_comparison_baseline"] = coldComparisonBaseline;
        fields["cold_mapping_count"] = repair.coldMappings.size();
        fields["incremental_mapping_count"] = repair.incrementalMappings.size();
        if (repair.coldExecution)
          fields["cold_wall_time_ns"] =
              repair.coldExecution->summary.executionWallTimeNanoseconds;
        fields["incremental_wall_time_ns"] =
            repair.incrementalExecution.summary.executionWallTimeNanoseconds;
        fields["cold_verifier_retained_bytes"] =
            repair.coldVerification.retainedBytes;
        fields["incremental_verifier_retained_bytes"] =
            repair.incrementalVerification.retainedBytes;
        fields["cold_verifier_work"] =
            repair.coldVerification.deterministicWork;
        fields["incremental_verifier_work"] =
            repair.incrementalVerification.deterministicWork;
      });
  return llvm::Expected<JointHardwareMutationRepair>(std::move(repair));
}

llvm::Expected<JointSpatialFifoHardwareRepair>
executeSpatialFifoHardwareFeedbackReopen(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy, const SpatialFifoRuntimeFeedback &feedback,
    JointHardwareReopenRequest request, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  JointSpatialFifoHardwareRepair result{feedback, {}, {}, {}, {}, false};
  if (feedback.disposition != SpatialFifoRuntimeFeedbackDisposition::Exact)
    return result;
  if (!feedback.fifo ||
      (!feedback.minimumCandidateDepth && !feedback.candidateQueueDiscipline))
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
  if (feedback.candidateQueueDiscipline &&
      *feedback.candidateQueueDiscipline == *currentDiscipline)
    return invalid("FIFO feedback discipline candidate is a no-op");

  const std::vector<SpatialFifoHardwareAlternative> alternatives =
      spatialFifoHardwareAlternatives(feedback);
  result.candidateLimit = alternatives.size();
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
    // One alternative materializes as one child: every rewrite of the
    // sequence is an exact Module mutation of the previous child, and the
    // lineages compose so the child keeps the complete typed impact.
    std::optional<JointHardwareMutationChild> child;
    ArtifactRootReference currentSystem = parentPair.system;
    ArtifactRootReference currentModule = parentModule;
    for (const SpatialMicroarchitectureDecisionDomain &domain :
         alternatives[ordinal]) {
      auto step = materializeJointModuleHardwareMutation(
          parentPlan.resolvedConfig, currentSystem, currentModule, domain,
          artifacts, blobs);
      if (!step)
        return step.takeError();
      currentSystem = step->system;
      auto modules = projectJointDesignTargetModules(currentSystem, artifacts);
      if (!modules)
        return modules.takeError();
      if (modules->size() != 1)
        return invalid("FIFO hardware child has more than one target Module");
      currentModule = modules->front();
      if (!child) {
        child = std::move(*step);
        continue;
      }
      auto composed = composeJointHardwareMutationChildren(
          std::move(*child), std::move(*step), artifacts);
      if (!composed)
        return composed.takeError();
      child = std::move(*composed);
    }
    if (!child)
      return invalid("FIFO hardware alternative has no Module rewrite");
    const ArtifactRootReference childReference = child->system;
    const std::size_t rewriteCount = alternatives[ordinal].size();
    JointHardwareReopenRequest childRequest = request;
    llvm::SmallString<256> childJournal(request.journalRoot);
    llvm::sys::path::append(childJournal,
                            "fifo-runtime-feedback-" + std::to_string(ordinal));
    childRequest.journalRoot = childJournal.str().str();
    auto repaired = executeJointHardwareMutationRepair(
        parentPlan, parentExecution, policy, feedback.parentMapping,
        std::move(*child), std::move(childRequest), artifacts, blobs);
    if (!repaired)
      return repaired.takeError();
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "spatial_fifo_hardware_repair";
          fields["alternative"] = ordinal;
          fields["module_rewrite_count"] = rewriteCount;
          fields["child_system"] =
              formatArtifactIdentityHex(childReference.artifact);
          fields["mapping_reuse_disposition"] =
              jointMappingReuseDispositionSpelling(
                  repaired->rebase.disposition);
          fields["system_mapping_reuse_disposition"] =
              jointSystemMappingReuseDispositionSpelling(
                  repaired->systemDisposition);
          fields["incremental_mapping_count"] =
              repaired->incrementalMappings.size();
          fields["incremental_wall_time_ns"] =
              repaired->incrementalExecution.summary
                  .executionWallTimeNanoseconds;
          fields["liveness"] = "requires_child_cgra_replay";
        });
    ++result.candidatesConsumed;
    result.childSystems.push_back(childReference);
    result.repairRecords.push_back(repaired->record);
    result.reuseDispositions.push_back(repaired->rebase.disposition);
    result.executions.push_back(std::move(repaired->incrementalExecution));
  }
  const std::uint64_t settled = result.candidatesConsumed +
                                result.candidatesRejected +
                                result.candidatesCancelled;
  if (settled > result.candidatesReserved)
    return invalid("FIFO hardware repair candidate ledger overflowed");
  result.candidatesRejected += result.candidatesReserved - settled;
  if (result.candidatesPlanned != result.candidatesReserved)
    return invalid("FIFO hardware repair candidate ledger is not reserved");
  if (result.childSystems.size() != result.repairRecords.size() ||
      result.childSystems.size() != result.executions.size() ||
      result.childSystems.size() != result.reuseDispositions.size())
    return invalid("FIFO hardware repair lost durable child lineage");
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
  JointSpatialOperandBufferHardwareRepair result{feedback, {}, {}, {}, {}};
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
  const std::vector<SpatialMicroarchitectureDecision> decisions =
      spatialOperandBufferHardwareDecisions(target);
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
      result.repairRecords.push_back(repaired->record);
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
  if (result.childSystems.size() != result.repairRecords.size() ||
      result.childSystems.size() != result.executions.size() ||
      result.childSystems.size() != result.reuseDispositions.size())
    return invalid("operand-buffer hardware repair lost durable child lineage");
  return result;
}

llvm::Expected<PlanExecutionPolicy>
reserveDispatchWindow(const PlanExecutionPolicy &policy,
                      std::uint64_t reservedNanoseconds) {
  const std::optional<std::uint64_t> deadline =
      policy.dispatchNotAfterUnixNanoseconds();
  if (!deadline || reservedNanoseconds == 0)
    return policy;
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const std::uint64_t nowNanoseconds =
      now.count() < 0
          ? 0
          : static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now)
                    .count());
  const std::uint64_t reservedDeadline = std::max(
      nowNanoseconds,
      *deadline > reservedNanoseconds ? *deadline - reservedNanoseconds : 0);
  if (reservedDeadline >= *deadline)
    return policy;
  return PlanExecutionPolicy::get(
      policy.workerCount(), policy.inProcessClaim(), policy.externalSite(),
      policy.resourceBindings(), policy.maximumDispatches(), reservedDeadline);
}

namespace {

void saturatingAdd(std::uint64_t &target, std::uint64_t value) {
  target = value > std::numeric_limits<std::uint64_t>::max() - target
               ? std::numeric_limits<std::uint64_t>::max()
               : target + value;
}

template <typename Repair>
void foldRepairLedger(JointRepairWorkLedger &ledger, const Repair &repair) {
  saturatingAdd(ledger.candidateLimit, repair.candidateLimit);
  saturatingAdd(ledger.planned, repair.candidatesPlanned);
  saturatingAdd(ledger.reserved, repair.candidatesReserved);
  saturatingAdd(ledger.consumed, repair.candidatesConsumed);
  saturatingAdd(ledger.rejected, repair.candidatesRejected);
  saturatingAdd(ledger.cancelled, repair.candidatesCancelled);
}

llvm::Error appendRepairChildren(JointRuntimeWitnessRepair &result,
                                 JointSpatialTransportMappingRepair &repair) {
  if (repair.childSystems.size() != repair.executions.size())
    return invalid("runtime witness repair lost its child System domain");
  result.childSystems.insert(result.childSystems.end(),
                             repair.childSystems.begin(),
                             repair.childSystems.end());
  result.hardwareMutationRepairRecords.insert(
      result.hardwareMutationRepairRecords.end(), repair.childSystems.size(),
      std::nullopt);
  for (JointDesignExecution &child : repair.executions)
    result.executions.push_back(std::move(child));
  repair.executions.clear();
  return llvm::Error::success();
}

template <typename Repair>
llvm::Error appendRepairChildren(JointRuntimeWitnessRepair &result,
                                 Repair &repair) {
  if (repair.childSystems.size() != repair.executions.size() ||
      repair.childSystems.size() != repair.repairRecords.size())
    return invalid("runtime witness repair lost durable hardware lineage");
  result.childSystems.insert(result.childSystems.end(),
                             repair.childSystems.begin(),
                             repair.childSystems.end());
  for (const ArtifactRootReference &record : repair.repairRecords)
    result.hardwareMutationRepairRecords.push_back(record);
  for (JointDesignExecution &child : repair.executions)
    result.executions.push_back(std::move(child));
  repair.executions.clear();
  return llvm::Error::success();
}

JointHardwareReopenRequest familyRequest(const JointHardwareReopenRequest &base,
                                         llvm::StringRef family) {
  JointHardwareReopenRequest request = base;
  llvm::SmallString<256> journal(base.journalRoot);
  llvm::sys::path::append(journal, family);
  request.journalRoot = journal.str().str();
  return request;
}

} // namespace

llvm::Expected<JointRuntimeWitnessRepair> executeJointRuntimeWitnessRepair(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy, const JointRuntimeWitnessSet &witnesses,
    std::uint64_t parentCostNanoseconds,
    std::optional<std::uint64_t> remainingHardwareRepairProbes,
    JointHardwareReopenRequest request, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  if (request.journalRoot.empty())
    return invalid("runtime witness repair requires a journal root");
  JointRuntimeWitnessRepair result;
  const bool mappingRepairAdmitted =
      witnesses.transport &&
      witnesses.transport->disposition ==
          SpatialTransportRuntimeFeedbackDisposition::Exact;
  // A fixed System frontier permits Mapping repair on the immutable parent
  // but never materializes a child System; an exhausted shared probe ledger
  // admits none either.
  const bool hardwareAdmitted =
      request.hardwareExplorationScope ==
          JointHardwareExplorationScope::BoundedHardwareReopen &&
      (!remainingHardwareRepairProbes || *remainingHardwareRepairProbes != 0);
  const bool fifoAdmitted = hardwareAdmitted && witnesses.fifo &&
                            witnesses.fifo->disposition ==
                                SpatialFifoRuntimeFeedbackDisposition::Exact;
  const bool operandAdmitted =
      hardwareAdmitted && witnesses.operandQueue &&
      witnesses.operandQueue->disposition ==
          SpatialOperandQueueRuntimeFeedbackDisposition::Exact &&
      witnesses.operandQueue->repairTarget.has_value();

  std::uint64_t hardwareChildCount = 0;
  if (fifoAdmitted)
    saturatingAdd(hardwareChildCount,
                  spatialFifoHardwareAlternatives(*witnesses.fifo).size());
  if (operandAdmitted)
    saturatingAdd(hardwareChildCount, spatialOperandBufferHardwareDecisions(
                                          *witnesses.operandQueue->repairTarget)
                                          .size());
  if (remainingHardwareRepairProbes)
    hardwareChildCount =
        std::min(hardwareChildCount, *remainingHardwareRepairProbes);
  if (hardwareChildCount != 0 && parentCostNanoseconds != 0)
    result.hardwareReopenReservedNanoseconds =
        parentCostNanoseconds >
                std::numeric_limits<std::uint64_t>::max() / hardwareChildCount
            ? std::numeric_limits<std::uint64_t>::max()
            : parentCostNanoseconds * hardwareChildCount;

  std::uint64_t mappingRepairWallNanoseconds = 0;
  std::uint64_t hardwareReopenWallNanoseconds = 0;
  const auto elapsedNanoseconds =
      [](std::chrono::steady_clock::time_point start) -> std::uint64_t {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  };
  if (mappingRepairAdmitted) {
    const auto mappingRepairStart = std::chrono::steady_clock::now();
    JointHardwareReopenRequest mappingRequest =
        familyRequest(request, "mapping-repair");
    auto window = reserveDispatchWindow(
        request.executionPolicy, result.hardwareReopenReservedNanoseconds);
    if (!window)
      return window.takeError();
    mappingRequest.executionPolicy = std::move(*window);
    auto repaired = executeSpatialTransportRuntimeRepair(
        parentPlan, parentExecution, policy, *witnesses.transport,
        std::move(mappingRequest), artifacts, blobs);
    if (!repaired)
      return repaired.takeError();
    foldRepairLedger(result.mappingRepairLedger, *repaired);
    if (llvm::Error error = appendRepairChildren(result, *repaired))
      return error;
    result.mappingRepair = std::move(*repaired);
    mappingRepairWallNanoseconds = elapsedNanoseconds(mappingRepairStart);
  }
  const auto hardwareReopenStart = std::chrono::steady_clock::now();
  // A first-verified invocation returns the first verified Mapping: once the
  // Mapping repair has retired a child on the immutable parent System, no
  // hardware child can be selected ahead of it, so none is materialized.
  // Bounded quality keeps every admitted family so the children compete.
  const bool mappingRepairRetired =
      result.mappingRepair && result.mappingRepair->cegar &&
      result.mappingRepair->cegar->termination ==
          SpatialTransportCegarTermination::Retired &&
      !result.executions.empty();
  const bool hardwareWithheldByMappingRepair =
      mappingRepairRetired &&
      request.stoppingPolicy == JointDesignStoppingPolicy::FirstVerified;

  std::optional<std::uint64_t> remainingProbes = remainingHardwareRepairProbes;
  const auto hardwareRequest =
      [&](llvm::StringRef family) -> JointHardwareReopenRequest {
    JointHardwareReopenRequest child = familyRequest(request, family);
    if (child.boundedQuality && remainingProbes)
      child.boundedQuality->maximumHardwareRepairProbes = std::min(
          child.boundedQuality->maximumHardwareRepairProbes, *remainingProbes);
    return child;
  };
  const auto chargeProbes = [&](std::uint64_t reserved) {
    if (remainingProbes)
      *remainingProbes =
          reserved >= *remainingProbes ? 0 : *remainingProbes - reserved;
  };
  if (fifoAdmitted && !hardwareWithheldByMappingRepair &&
      (!remainingProbes || *remainingProbes != 0)) {
    auto repaired = executeSpatialFifoHardwareFeedbackReopen(
        parentPlan, parentExecution, policy, *witnesses.fifo,
        hardwareRequest("fifo-reopen"), artifacts, blobs);
    if (!repaired)
      return repaired.takeError();
    foldRepairLedger(result.hardwareReopenLedger, *repaired);
    chargeProbes(repaired->candidatesReserved);
    if (llvm::Error error = appendRepairChildren(result, *repaired))
      return error;
    result.fifoReopen = std::move(*repaired);
  }
  if (operandAdmitted && !hardwareWithheldByMappingRepair &&
      (!remainingProbes || *remainingProbes != 0)) {
    auto repaired = executeSpatialOperandBufferHardwareFeedbackReopen(
        parentPlan, parentExecution, policy, *witnesses.operandQueue,
        hardwareRequest("operand-buffer-reopen"), artifacts, blobs);
    if (!repaired)
      return repaired.takeError();
    foldRepairLedger(result.hardwareReopenLedger, *repaired);
    chargeProbes(repaired->candidatesReserved);
    if (llvm::Error error = appendRepairChildren(result, *repaired))
      return error;
    result.operandBufferReopen = std::move(*repaired);
  }
  hardwareReopenWallNanoseconds = elapsedNanoseconds(hardwareReopenStart);
  if (result.childSystems.size() != result.executions.size() ||
      result.childSystems.size() != result.hardwareMutationRepairRecords.size())
    return invalid("runtime witness repair lost aligned child lineage");
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        const auto ledger = [](const JointRepairWorkLedger &value) {
          return llvm::json::Object{{"candidate_limit", value.candidateLimit},
                                    {"planned", value.planned},
                                    {"reserved", value.reserved},
                                    {"consumed", value.consumed},
                                    {"rejected", value.rejected},
                                    {"cancelled", value.cancelled}};
        };
        fields["operation"] = "joint_runtime_witness_repair";
        fields["mapping_repair_admitted"] = mappingRepairAdmitted;
        fields["hardware_reopen_admitted"] = hardwareAdmitted;
        fields["hardware_reopen_withheld_by_retired_mapping_repair"] =
            hardwareWithheldByMappingRepair;
        fields["hardware_child_count"] = hardwareChildCount;
        fields["parent_cost_ns"] = parentCostNanoseconds;
        fields["hardware_reopen_reserved_ns"] =
            result.hardwareReopenReservedNanoseconds;
        fields["mapping_repair"] = ledger(result.mappingRepairLedger);
        fields["mapping_repair_wall_ns"] = mappingRepairWallNanoseconds;
        fields["hardware_reopen"] = ledger(result.hardwareReopenLedger);
        fields["hardware_reopen_wall_ns"] = hardwareReopenWallNanoseconds;
        fields["child_system_count"] = result.childSystems.size();
        fields["hardware_mutation_repair_record_count"] = llvm::count_if(
            result.hardwareMutationRepairRecords,
            [](const auto &record) { return record.has_value(); });
      });
  return result;
}

} // namespace loom::dse

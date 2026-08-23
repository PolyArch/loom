#include "DSE/JointHardwareReopen.h"

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
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"
#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "DSE/SystemCompositionCandidateGenerator.h"
#include "DSE/TechMappingHardwareFeedback.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SpatialMappingHardwareDemand.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
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
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "joint_hardware_reopen_invalid: " + message);
}

struct JointSoftwareCoverage final {
  std::uint64_t acceleratedRootCount = 0;
  std::uint64_t graphCount = 0;
  std::uint64_t actorCount = 0;
};

llvm::Expected<JointSoftwareCoverage> projectJointSoftwareCoverage(
    const JointDesignExplorationPlan &plan, const ArtifactStore &artifacts) {
  if (plan.frontier.softwareFrontier.size() != 1)
    return invalid("software coverage requires one exact Dataflow scope");
  auto imported = ::dataflow::importCanonicalDataflow(
      plan.frontier.softwareFrontier.front().dataflow, artifacts);
  if (!imported)
    return imported.takeError();
  auto view = imported->view();
  if (!view)
    return view.takeError();
  return JointSoftwareCoverage{
      static_cast<std::uint64_t>(view->rootThreadLaunches().size()),
      static_cast<std::uint64_t>(view->graphs().size()),
      static_cast<std::uint64_t>(view->actors().size())};
}

// PlanExecutor propagates this deadline into each provider, but the joint
// controller also owns software-frontier and hardware-reopen loops.  Those
// loops must observe the same absolute deadline before admitting another
// child; otherwise a bounded invocation can overrun after its last provider
// has already reported cancellation.
bool dispatchDeadlineReached(const PlanExecutionPolicy &policy) {
  const auto deadline = policy.dispatchNotAfterUnixNanoseconds();
  if (!deadline)
    return false;
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  if (now.count() < 0)
    return false;
  return static_cast<std::uint64_t>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(now)
                 .count()) >= *deadline;
}

llvm::Expected<PlanExecutionPolicy> fairBoundedQualityPlanPolicy(
    const PlanExecutionPolicy &base, std::uint64_t remainingPlanCount) {
  if (remainingPlanCount == 0)
    return invalid("bounded-quality plan slice has no remaining plan");
  const auto globalDeadline = base.dispatchNotAfterUnixNanoseconds();
  if (!globalDeadline)
    return base;
  const auto elapsed = std::chrono::system_clock::now().time_since_epoch();
  const auto signedNow =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  if (signedNow <= 0)
    return invalid("system clock cannot derive a bounded-quality plan slice");
  const std::uint64_t now = static_cast<std::uint64_t>(signedNow);
  if (now >= *globalDeadline)
    return PlanExecutionPolicy::get(
        base.workerCount(), base.inProcessClaim(), base.externalSite(),
        base.resourceBindings(), base.maximumDispatches(), *globalDeadline);
  const std::uint64_t remaining = *globalDeadline - now;
  // Reserve one equal share for terminal application QoR acquisition. Each
  // untried Mapping plan receives a fair share of the rest, so one difficult
  // finalist cannot consume the entire invocation deadline before its
  // siblings are dispatched. The global deadline remains the hard ceiling.
  const std::uint64_t divisor =
      remainingPlanCount == std::numeric_limits<std::uint64_t>::max()
          ? remainingPlanCount
          : remainingPlanCount + 1;
  const std::uint64_t slice = std::max<std::uint64_t>(1, remaining / divisor);
  const std::uint64_t localDeadline =
      slice > *globalDeadline - now ? *globalDeadline : now + slice;
  return PlanExecutionPolicy::get(
      base.workerCount(), base.inProcessClaim(), base.externalSite(),
      base.resourceBindings(), base.maximumDispatches(), localDeadline);
}

struct TechHardwareFeedbackObservation final {
  ArtifactRootReference module;
  mapping::TechMappingComputeContextHallDeficit feedback;
};

struct SpatialHardwareFeedbackObservation final {
  mapping::SpatialGraphBoundaryEndpointHallDeficit feedback;
};

struct SystemHardwareFeedbackObservation final {
  mapping::SystemAccCoreCapacityPressure feedback;
};

struct HardwareRecipeGrowth final {
  ResolvedConfig config;
  std::optional<ArtifactRootReference> accCoreParent;
  std::optional<ArtifactRootReference> accCoreTargetModule;
  std::optional<ArtifactRootReference> techModule;
  std::vector<ResizeInstructionStore> instructionStoreResizes;
  std::optional<ResizeFifo> fifoResize;
  std::optional<ChangeFifoBypassCapability> fifoBypassChange;
  std::uint64_t resizedInstructionStoreCount = 0;
  std::uint64_t maximumInstructionStoreCapacity = 0;
  std::uint64_t addedContexts = 0;
  std::uint64_t resultingContexts = 0;
  std::uint64_t addedGateways = 0;
  std::uint64_t resultingGateways = 0;
  std::uint64_t addedAccCores = 0;
  std::uint64_t resultingAccCores = 0;
};

struct MaterializedHardwareCandidate final {
  ArtifactRootReference reference;
  ResolvedConfig config;
  std::optional<pnr::SystemExecutionBindingCorrespondence>
      executionBindingCorrespondence;
  std::vector<pnr::SystemModuleCorrespondence> moduleCorrespondences;
  std::optional<HardwareImpactProjection> mappingImpact;
  std::uint64_t resizedInstructionStoreCount = 0;
  std::uint64_t maximumInstructionStoreCapacity = 0;
  std::uint64_t addedContexts = 0;
  std::uint64_t resultingContexts = 0;
  std::uint64_t addedGateways = 0;
  std::uint64_t resultingGateways = 0;
  std::uint64_t addedAccCores = 0;
  std::uint64_t resultingAccCores = 0;
};

struct FinalizedMappingHardwareSpectrum final {
  std::vector<JointDesignExecution> verified;
  std::uint64_t attemptedSystems = 0;
  bool incomplete = false;
};

const dse::CompletedDsePlanExecution &
availableExecution(const dse::DsePlanExecutionResult &execution) {
  if (const auto *completed =
          std::get_if<dse::CompletedDsePlanExecution>(&execution))
    return *completed;
  return std::get<dse::IncompleteDsePlanExecution>(execution)
      .availableExecution();
}

std::size_t mappingCount(const dse::JointDesignExecution &execution) {
  std::size_t count = 0;
  for (const dse::JointMappedPair &pair : execution.mappedPairs)
    count += pair.systemMappings.size();
  return count;
}

void canonicalizeRoots(std::vector<ArtifactRootReference> &roots) {
  llvm::sort(roots, artifactRootReferenceLess);
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
}

std::vector<ArtifactRootReference>
mappingRoots(const dse::JointDesignExecution &execution) {
  std::vector<ArtifactRootReference> roots;
  roots.reserve(mappingCount(execution));
  for (const dse::JointMappedPair &pair : execution.mappedPairs)
    roots.insert(roots.end(), pair.systemMappings.begin(),
                 pair.systemMappings.end());
  canonicalizeRoots(roots);
  return roots;
}

std::optional<ArtifactRootReference>
firstMapping(const dse::JointDesignExecution &execution) {
  for (const dse::JointMappedPair &pair : execution.mappedPairs)
    if (!pair.systemMappings.empty())
      return pair.systemMappings.front();
  return std::nullopt;
}

llvm::Error
recordJointAttempt(std::vector<dse::JointDesignAttemptRecord> &records,
                   std::uint64_t planOrdinal,
                   const ArtifactRootReference &fallbackSystem,
                   const dse::JointDesignExecution &execution) {
  ArtifactRootReference system = fallbackSystem;
  for (const dse::JointMappedPair &pair : execution.mappedPairs) {
    if (system != fallbackSystem && pair.pair.system != system)
      return invalid("one joint attempt produced multiple System owners");
    system = pair.pair.system;
  }
  std::vector<ArtifactRootReference> mappings = mappingRoots(execution);
  dse::JointDesignAttemptDisposition disposition =
      dse::JointDesignAttemptDisposition::ProvenNoFeasibleCandidate;
  std::optional<std::uint64_t> incompleteNodeOrdinal;
  std::optional<dse::DsePlanIncompleteReason> incompleteReason;
  if (!mappings.empty())
    disposition = dse::JointDesignAttemptDisposition::Verified;
  else if (const auto *incomplete =
               std::get_if<dse::IncompleteDsePlanExecution>(
                   &execution.planExecution)) {
    disposition = dse::JointDesignAttemptDisposition::Incomplete;
    incompleteNodeOrdinal = incomplete->nodeOrdinal();
    incompleteReason = incomplete->reason();
  }
  records.push_back({planOrdinal, system, disposition, incompleteNodeOrdinal,
                     std::move(incompleteReason), std::move(mappings)});
  return llvm::Error::success();
}

llvm::Error bindImmutableSpatialMappingFrontier(
    JointDesignExplorationPlan &plan,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &artifacts) {
  constexpr std::size_t spatialMappingInputOrdinal = 1;
  if (plan.pairOutputs.size() != 1 || spatialMappings.empty())
    return invalid("Spatial frontier reuse requires one nonempty exact pair");
  JointDesignPlanPair &pair = plan.pairOutputs.front();
  if (pair.systemMappings.producerNodeOrdinal >=
      plan.resolvedConfig.dse.planNodes.size())
    return invalid("System Mapping output names a foreign plan node");
  auto *systemNode = std::get_if<GeneratePlanNodeDefinition>(
      &plan.resolvedConfig.dse
           .planNodes[pair.systemMappings.producerNodeOrdinal]);
  if (!systemNode ||
      systemNode->descriptor !=
          applicationSystemPnrCandidateGeneratorDescriptor().reference() ||
      systemNode->inputBindings.size() <= spatialMappingInputOrdinal)
    return invalid("joint plan has no canonical application System provider");

  auto targetModules =
      projectJointDesignTargetModules(pair.pair.system, artifacts);
  if (!targetModules)
    return targetModules.takeError();
  std::vector<ArtifactRootReference> canonicalMappings(spatialMappings.begin(),
                                                       spatialMappings.end());
  const std::size_t originalCount = canonicalMappings.size();
  canonicalizeRoots(canonicalMappings);
  if (canonicalMappings.size() != originalCount)
    return invalid("reused Spatial frontier contains a duplicate root");
  const auto *join = std::get_if<BoundedPlanOutputJoin>(
      &systemNode->inputBindings[spatialMappingInputOrdinal]);
  if (!join || canonicalMappings.size() > join->maximumArtifacts)
    return invalid("reused Spatial frontier exceeds its exact plan bound");
  for (const ArtifactRootReference &reference : canonicalMappings) {
    auto mapping = mapping::importSpatialMapping(reference, artifacts);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() !=
        pair.pair.software.dataflow.artifact)
      return invalid("reused SpatialMapping has a foreign Dataflow owner");
    if (!llvm::any_of(*targetModules, [&](const auto &module) {
          return module.artifact == mapping->view().fabricIdentity();
        }))
      return invalid("reused SpatialMapping has a foreign Module owner");
  }

  GeneratePlanNodeDefinition retainedSystemNode = *systemNode;
  retainedSystemNode.inputBindings[spatialMappingInputOrdinal] =
      ExactPlanArtifacts{canonicalMappings};
  plan.resolvedConfig.dse.planNodes = {std::move(retainedSystemNode)};
  pair.techMappings.clear();
  pair.spatialMappings.clear();
  pair.immutableTechMappings.clear();
  pair.immutableSpatialMappings = canonicalMappings;
  pair.systemMappings = PlanOutputRef{0, 0};
  auto admitted = projectResolvedDseConfigView(plan.resolvedConfig);
  if (!admitted)
    return admitted.takeError();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "reuse_immutable_spatial_frontier";
        fields["spatial_mapping_count"] = canonicalMappings.size();
      });
  return llvm::Error::success();
}

llvm::Error bindCheckpointSystemMappingMigrationSeed(
    JointDesignExplorationPlan &plan,
    const ArtifactRootReference &migrationSeed,
    const ArtifactStore &artifacts) {
  constexpr std::size_t migrationSeedInputOrdinal = 5;
  if (migrationSeed.schemaIdentity !=
          pnr::systemMappingCheckpointMigrationSeedArtifactSchema.identity ||
      migrationSeed.schemaVersion !=
          pnr::systemMappingCheckpointMigrationSeedArtifactSchema.version)
    return invalid("System migration seed has a foreign schema");
  auto imported =
      pnr::importSystemMappingCheckpointMigrationSeed(migrationSeed, artifacts);
  if (!imported)
    return imported.takeError();
  if (plan.pairOutputs.size() != 1)
    return invalid("System migration requires one exact Mapping pair");
  JointDesignPlanPair &pair = plan.pairOutputs.front();
  if (pair.systemMappings.producerNodeOrdinal >=
      plan.resolvedConfig.dse.planNodes.size())
    return invalid("System migration output names a foreign plan node");
  auto *systemNode = std::get_if<GeneratePlanNodeDefinition>(
      &plan.resolvedConfig.dse
           .planNodes[pair.systemMappings.producerNodeOrdinal]);
  if (!systemNode ||
      systemNode->descriptor !=
          applicationSystemPnrCandidateGeneratorDescriptor().reference() ||
      systemNode->inputBindings.size() <= migrationSeedInputOrdinal)
    return invalid("joint plan has no canonical migration-seed input");
  systemNode->inputBindings[migrationSeedInputOrdinal] =
      ExactPlanArtifacts{{migrationSeed}};
  auto admitted = projectResolvedDseConfigView(plan.resolvedConfig);
  if (!admitted)
    return admitted.takeError();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "bind_system_mapping_migration_seed";
        fields["checkpoint"] = formatArtifactIdentityHex(
            imported->checkpoint().reference().artifact);
        fields["migration_seed"] =
            formatArtifactIdentityHex(migrationSeed.artifact);
        fields["preserved_acc_core_correspondences"] =
            imported->correspondence().accCores().size();
      });
  return llvm::Error::success();
}

llvm::Error bindFinalizedSystemMappingMigrationSeed(
    JointDesignExplorationPlan &plan,
    const ArtifactRootReference &migrationSeed,
    const ArtifactStore &artifacts) {
  constexpr std::size_t migrationSeedInputOrdinal = 6;
  if (migrationSeed.schemaIdentity !=
          pnr::systemMappingFinalizedMigrationSeedArtifactSchema.identity ||
      migrationSeed.schemaVersion !=
          pnr::systemMappingFinalizedMigrationSeedArtifactSchema.version)
    return invalid("finalized System migration seed has a foreign schema");
  auto imported =
      pnr::importSystemMappingMigrationSeed(migrationSeed, artifacts);
  if (!imported)
    return imported.takeError();
  if (plan.pairOutputs.size() != 1)
    return invalid("finalized System migration requires one exact pair");
  JointDesignPlanPair &pair = plan.pairOutputs.front();
  if (imported->correspondence().childSystem() != pair.pair.system ||
      imported->parentMapping().view().dataflowIdentity() !=
          pair.pair.software.dataflow.artifact)
    return invalid("finalized System migration seed has foreign owners");
  if (pair.systemMappings.producerNodeOrdinal >=
      plan.resolvedConfig.dse.planNodes.size())
    return invalid("finalized System migration output names a foreign node");
  auto *systemNode = std::get_if<GeneratePlanNodeDefinition>(
      &plan.resolvedConfig.dse
           .planNodes[pair.systemMappings.producerNodeOrdinal]);
  if (!systemNode ||
      systemNode->descriptor !=
          applicationSystemPnrCandidateGeneratorDescriptor().reference() ||
      systemNode->inputBindings.size() <= migrationSeedInputOrdinal)
    return invalid("joint plan has no finalized migration-seed input");
  systemNode->inputBindings[migrationSeedInputOrdinal] =
      ExactPlanArtifacts{{migrationSeed}};
  auto admitted = projectResolvedDseConfigView(plan.resolvedConfig);
  if (!admitted)
    return admitted.takeError();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "bind_finalized_system_mapping_migration_seed";
        fields["parent_mapping"] = formatArtifactIdentityHex(
            imported->parentMapping().reference().artifact);
        fields["migration_seed"] =
            formatArtifactIdentityHex(migrationSeed.artifact);
        fields["preserved_acc_core_correspondences"] =
            imported->correspondence().accCores().size();
      });
  return llvm::Error::success();
}

llvm::Expected<pnr::SystemMappingMigrationContext>
deriveSystemMappingMigrationContext(const JointDesignExplorationPlan &plan) {
  constexpr std::size_t spatialMappingInputOrdinal = 1;
  constexpr std::size_t constraintsInputOrdinal = 4;
  if (plan.pairOutputs.size() != 1)
    return invalid("System migration context requires one exact pair");
  const JointDesignPlanPair &pair = plan.pairOutputs.front();
  if (pair.systemMappings.producerNodeOrdinal >=
      plan.resolvedConfig.dse.planNodes.size())
    return invalid("System migration context names a foreign plan node");
  const auto *systemNode = std::get_if<GeneratePlanNodeDefinition>(
      &plan.resolvedConfig.dse
           .planNodes[pair.systemMappings.producerNodeOrdinal]);
  if (!systemNode ||
      systemNode->descriptor !=
          applicationSystemPnrCandidateGeneratorDescriptor().reference() ||
      systemNode->inputBindings.size() <= constraintsInputOrdinal)
    return invalid("System migration context has no canonical provider");
  const auto *mappings = std::get_if<ExactPlanArtifacts>(
      &systemNode->inputBindings[spatialMappingInputOrdinal]);
  const auto *constraints = std::get_if<ExactPlanArtifacts>(
      &systemNode->inputBindings[constraintsInputOrdinal]);
  if (!mappings || mappings->artifacts.empty() || !constraints ||
      constraints->artifacts.size() != 1)
    return invalid("System migration context is not fully materialized");
  return pnr::SystemMappingMigrationContext::get(constraints->artifacts.front(),
                                                 mappings->artifacts,
                                                 systemNode->configDigest);
}

llvm::Expected<dse::DsePlanExecutionResult> executeResolvedGeneratePlan(
    const ResolvedConfig &config,
    std::vector<ArtifactRootReference> semanticInputs,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const PlanExecutionPolicy *executionPolicy = nullptr) {
  canonicalizeRoots(semanticInputs);
  auto publishedConfig = artifacts.put(ResolvedConfig::artifactSchema,
                                       canonicalResolvedConfigBytes(config));
  if (!publishedConfig)
    return publishedConfig.takeError();
  if (*publishedConfig != resolvedConfigIdentity(config))
    return invalid("ResolvedConfig publication changed its identity");
  auto closure = dse::DseRunClosure::get(request.producer, semanticInputs,
                                         config, evidence, artifacts);
  if (!closure)
    return closure.takeError();
  auto configView = dse::projectResolvedDseConfigView(config);
  if (!configView)
    return configView.takeError();
  llvm::SmallString<256> journalPath(request.journalRoot);
  llvm::sys::path::append(journalPath,
                          llvm::toHex(closure->runKey().bytes(), true));
  if (std::error_code error = llvm::sys::fs::create_directories(journalPath))
    return invalid("cannot create hardware-reopen journal: " + error.message());
  auto journal = dse::openExecutionJournal(journalPath, *closure, *configView);
  if (!journal)
    return journal.takeError();
  return dse::resumeDsePlan(*configView, *closure, *journal, scheduler,
                            executionPolicy ? *executionPolicy
                                            : request.executionPolicy,
                            artifacts, blobs);
}

struct TechGateExecution final {
  JointDesignExecution execution;
  std::vector<ArtifactRootReference> techMappings;
  bool coversRequiredGraphs = false;
};

llvm::Expected<TechGateExecution> executeTechGate(
    const JointDesignExplorationPlan &plan,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const PlanExecutionPolicy &executionPolicy) {
  if (plan.frontier.softwareFrontier.size() != 1 ||
      plan.pairOutputs.size() != 1)
    return invalid("Tech gate requires one exact software/System pair");
  ResolvedConfig config = plan.resolvedConfig;
  config.dse.planNodes.clear();
  for (const DsePlanNodeDefinition &node : plan.resolvedConfig.dse.planNodes) {
    const auto *generate = std::get_if<GeneratePlanNodeDefinition>(&node);
    if (generate &&
        generate->descriptor ==
            applicationGraphTechMappingCandidateGeneratorDescriptor()
                .reference())
      config.dse.planNodes.push_back(*generate);
  }

  std::vector<ArtifactRootReference> retained =
      plan.pairOutputs.front().immutableTechMappings;
  JointDesignExecutionSummary summary;
  auto planExecution = executeResolvedGeneratePlan(
      config, projectJointDesignSemanticInputs(plan), evidence, request,
      scheduler, artifacts, blobs, &executionPolicy);
  if (!planExecution)
    return planExecution.takeError();
  const CompletedDsePlanExecution &available =
      availableExecution(*planExecution);
  for (std::size_t node = 0; node != config.dse.planNodes.size(); ++node) {
    if (available.generateInvocationWasDispatched(node))
      ++summary.techMappingDispatchCount;
    const PlanOutputRef output{node, 0};
    if (available.hasOutput(output)) {
      const auto roots = available.resolve(output);
      retained.insert(retained.end(), roots.begin(), roots.end());
    }
  }
  canonicalizeRoots(retained);

  const ArtifactRootReference dataflowReference =
      plan.frontier.softwareFrontier.front().dataflow;
  auto dataflowArtifact =
      ::dataflow::importCanonicalDataflow(dataflowReference, artifacts);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflow = dataflowArtifact->view();
  if (!dataflow)
    return dataflow.takeError();
  std::vector<::dataflow::GraphRef> requiredGraphs;
  for (const DsePlanNodeDefinition &node : plan.resolvedConfig.dse.planNodes) {
    const auto *generate = std::get_if<GeneratePlanNodeDefinition>(&node);
    if (!generate ||
        generate->descriptor !=
            applicationGraphTechMappingCandidateGeneratorDescriptor()
                .reference())
      continue;
    if (generate->inputBindings.size() < 2)
      return invalid("Tech gate node has no System constraint input");
    const auto *exact =
        std::get_if<ExactPlanArtifacts>(&generate->inputBindings[1]);
    if (!exact || exact->artifacts.size() != 1)
      return invalid("Tech gate node has a non-exact constraint input");
    auto constraints = mapping::importSystemMappingConstraintSet(
        exact->artifacts.front(), artifacts);
    if (!constraints)
      return constraints.takeError();
    for (const auto &root : constraints->view().rootThreadLaunches()) {
      llvm::Error graphError = llvm::Error::success();
      dataflow->forEachRootedGraphLaunch(
          [&](::dataflow::RootedGraphLaunchRef launch) {
            if (graphError || launch.rootThreadLaunch != root)
              return;
            auto graph = dataflow->resolve(launch);
            if (graph)
              requiredGraphs.push_back(*graph);
            else
              graphError = graph.takeError();
          });
      if (graphError)
        return std::move(graphError);
    }
  }
  const auto graphLess = [](const ::dataflow::GraphRef &lhs,
                            const ::dataflow::GraphRef &rhs) {
    if (lhs.artifact != rhs.artifact)
      return lhs.artifact.bytes() < rhs.artifact.bytes();
    return lhs.entity.value() < rhs.entity.value();
  };
  llvm::sort(requiredGraphs, graphLess);
  requiredGraphs.erase(
      std::unique(requiredGraphs.begin(), requiredGraphs.end()),
      requiredGraphs.end());
  std::vector<::dataflow::GraphRef> coveredGraphs;
  for (const ArtifactRootReference &reference : retained) {
    auto tech = mapping::importTechMapping(reference, artifacts);
    if (!tech)
      return tech.takeError();
    coveredGraphs.insert(coveredGraphs.end(), tech->view().covers().begin(),
                         tech->view().covers().end());
  }
  llvm::sort(coveredGraphs, graphLess);
  coveredGraphs.erase(std::unique(coveredGraphs.begin(), coveredGraphs.end()),
                      coveredGraphs.end());
  const bool coversRequiredGraphs =
      llvm::all_of(requiredGraphs, [&](const auto &required) {
        return llvm::binary_search(coveredGraphs, required, graphLess);
      });
  return TechGateExecution{
      JointDesignExecution{std::move(*planExecution), {}, std::move(summary)},
      std::move(retained), coversRequiredGraphs};
}

llvm::Expected<dse::JointDesignExecution>
executeJointPlan(const dse::JointDesignExplorationPlan &plan,
                 llvm::ArrayRef<ArtifactRootReference> evidence,
                 const JointHardwareReopenRequest &request,
                 dse::SiteScheduler &scheduler, const ArtifactStore &artifacts,
                 const BlobStore &blobs,
                 const PlanExecutionPolicy *executionPolicy = nullptr) {
  const ResolvedConfig &config = plan.resolvedConfig;
  auto publishedConfig = artifacts.put(ResolvedConfig::artifactSchema,
                                       canonicalResolvedConfigBytes(config));
  if (!publishedConfig)
    return publishedConfig.takeError();
  if (*publishedConfig != resolvedConfigIdentity(config))
    return invalid("ResolvedConfig publication changed its identity");
  std::vector<ArtifactRootReference> semanticInputs =
      dse::projectJointDesignSemanticInputs(plan);
  auto closure = dse::DseRunClosure::get(request.producer, semanticInputs,
                                         config, evidence, artifacts);
  if (!closure)
    return closure.takeError();
  auto configView = dse::projectResolvedDseConfigView(config);
  if (!configView)
    return configView.takeError();
  llvm::SmallString<256> journalPath(request.journalRoot);
  llvm::sys::path::append(journalPath,
                          llvm::toHex(closure->runKey().bytes(), true));
  if (std::error_code error = llvm::sys::fs::create_directories(journalPath))
    return invalid("cannot create Mapping alternative journal: " +
                   error.message());
  auto journal = dse::openExecutionJournal(journalPath, *closure, *configView);
  if (!journal)
    return journal.takeError();
  return dse::executeJointDesignExploration(plan, *closure, *journal, scheduler,
                                            executionPolicy ? *executionPolicy
                                                            : request.executionPolicy,
                                            artifacts,
                                            blobs);
}

llvm::Expected<std::optional<TechHardwareFeedbackObservation>>
selectTechHardwareFeedback(const dse::JointDesignExecution &execution,
                           const ArtifactStore &artifacts) {
  const dse::CompletedDsePlanExecution &available =
      availableExecution(execution.planExecution);
  std::optional<TechHardwareFeedbackObservation> selected;
  for (const dse::GenerateInvocationFeedback &feedback :
       available.generateFeedback()) {
    const auto invocation = llvm::find_if(
        available.generateInvocations(), [&](const auto &candidate) {
          return candidate.planNodeOrdinal == feedback.planNodeOrdinal;
        });
    if (invocation == available.generateInvocations().end())
      return invalid("Generate feedback has no invocation owner");
    const dse::CandidateGeneratorDescriptor *descriptor =
        invocation->generatorBinding.descriptorRef().descriptor();
    if (!descriptor || !descriptor->ownerFeedbackPayload ||
        descriptor->ownerFeedbackPayload->schemaDescriptorBytes !=
            mapping::techMappingComputeContextHallFeedbackSchemaBytes())
      continue;

    std::optional<ArtifactRootReference> moduleReference;
    std::optional<fabric::FinalizedFabricRoot> module;
    for (const dse::CandidateGeneratorInputBinding &binding :
         invocation->inputBindings)
      for (const ArtifactRootReference &input : binding.artifacts) {
        if (input.schemaIdentity != fabric::fabricArtifactSchema.identity ||
            input.schemaVersion != fabric::fabricArtifactSchema.version)
          continue;
        auto imported = fabric::importEntireFabricRoot(input, artifacts);
        if (!imported)
          return imported.takeError();
        if (imported->view().rootKind() != fabric::FabricRootKind::Module)
          continue;
        if (moduleReference)
          return invalid("TechMapping feedback names multiple Module inputs");
        moduleReference = input;
        module = std::move(*imported);
      }
    if (!moduleReference || !module)
      return invalid("TechMapping feedback has no exact Module input");
    auto adopted = mapping::adoptTechMappingComputeContextHallFeedback(
        feedback.canonicalPayload, module->view());
    if (!adopted)
      return adopted.takeError();
    TechHardwareFeedbackObservation candidate{*moduleReference,
                                              std::move(*adopted)};
    mapping_debug::emit(
        mapping_debug::Level::Decision, mapping_debug::Stage::TechMapping,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "hardware_feedback_hall_projection";
          fields["hall_deficit"] = candidate.feedback.deficit();
          fields["hall_demand_count"] = candidate.feedback.hallDemandCount();
          fields["hall_context_value_count"] =
              candidate.feedback.hallContextValueCount();
          llvm::json::Array groups;
          for (const auto &group : candidate.feedback.groups()) {
            std::map<std::uint64_t, std::uint64_t> peCounts;
            for (const auto context : group.compatibleContexts)
              ++peCounts[context.pe.id()];
            llvm::json::Object encoded;
            encoded["demand_count"] = group.demandCount;
            llvm::json::Array peValues;
            for (const auto &[pe, count] : peCounts)
              peValues.push_back(llvm::json::Object{{"pe", pe},
                                                     {"context_count", count}});
            encoded["compatible_pes"] = std::move(peValues);
            groups.push_back(std::move(encoded));
          }
          fields["groups"] = std::move(groups);
        });
    if (!selected ||
        candidate.feedback.deficit() > selected->feedback.deficit() ||
        (candidate.feedback.deficit() == selected->feedback.deficit() &&
         artifactRootReferenceLess(candidate.module, selected->module)))
      selected = std::move(candidate);
  }
  return selected;
}

llvm::Expected<std::optional<SpatialHardwareFeedbackObservation>>
selectSpatialHardwareFeedback(const dse::JointDesignExecution &execution,
                              const ArtifactStore &artifacts) {
  const dse::CompletedDsePlanExecution &available =
      availableExecution(execution.planExecution);
  std::optional<SpatialHardwareFeedbackObservation> selected;
  for (const dse::GenerateInvocationFeedback &feedback :
       available.generateFeedback()) {
    const auto invocation = llvm::find_if(
        available.generateInvocations(), [&](const auto &candidate) {
          return candidate.planNodeOrdinal == feedback.planNodeOrdinal;
        });
    if (invocation == available.generateInvocations().end())
      return invalid("Generate feedback has no invocation owner");
    const dse::CandidateGeneratorDescriptor *descriptor =
        invocation->generatorBinding.descriptorRef().descriptor();
    if (!descriptor || !descriptor->ownerFeedbackPayload ||
        descriptor->ownerFeedbackPayload->schemaDescriptorBytes !=
            mapping::spatialGraphBoundaryEndpointHallFeedbackSchemaBytes())
      continue;

    std::optional<ArtifactRootReference> moduleReference;
    std::vector<ArtifactRootReference> techMappings;
    for (const dse::CandidateGeneratorInputBinding &binding :
         invocation->inputBindings)
      for (const ArtifactRootReference &input : binding.artifacts) {
        if (input.schemaIdentity == fabric::fabricArtifactSchema.identity &&
            input.schemaVersion == fabric::fabricArtifactSchema.version) {
          auto imported = fabric::importEntireFabricRoot(input, artifacts);
          if (!imported)
            return imported.takeError();
          if (imported->view().rootKind() != fabric::FabricRootKind::Module)
            continue;
          if (moduleReference)
            return invalid(
                "SpatialMapping feedback names multiple Module inputs");
          moduleReference = input;
          continue;
        }
        if (input.schemaIdentity == mapping::mappingArtifactSchema.identity &&
            input.schemaVersion == mapping::mappingArtifactSchema.version)
          techMappings.push_back(input);
      }
    if (!moduleReference || techMappings.empty())
      return invalid("SpatialMapping feedback lacks its exact Mapping inputs");
    canonicalizeRoots(techMappings);
    auto adopted = mapping::adoptSpatialGraphBoundaryEndpointHallFeedback(
        feedback.canonicalPayload, *moduleReference, techMappings, artifacts);
    if (!adopted)
      return adopted.takeError();
    SpatialHardwareFeedbackObservation candidate{std::move(*adopted)};
    if (!selected ||
        candidate.feedback.requiredBoundaryPairs() >
            selected->feedback.requiredBoundaryPairs() ||
        (candidate.feedback.requiredBoundaryPairs() ==
             selected->feedback.requiredBoundaryPairs() &&
         mapping::encodeSpatialGraphBoundaryEndpointHallFeedback(
             candidate.feedback) <
             mapping::encodeSpatialGraphBoundaryEndpointHallFeedback(
                 selected->feedback)))
      selected = std::move(candidate);
  }
  return selected;
}

llvm::Expected<std::optional<SystemHardwareFeedbackObservation>>
selectSystemHardwareFeedback(const dse::JointDesignExecution &execution,
                             const ArtifactStore &artifacts) {
  const dse::CompletedDsePlanExecution &available =
      availableExecution(execution.planExecution);
  std::optional<SystemHardwareFeedbackObservation> selected;
  for (const dse::GenerateInvocationFeedback &feedback :
       available.generateFeedback()) {
    const auto invocation = llvm::find_if(
        available.generateInvocations(), [&](const auto &candidate) {
          return candidate.planNodeOrdinal == feedback.planNodeOrdinal;
        });
    if (invocation == available.generateInvocations().end())
      return invalid("Generate feedback has no invocation owner");
    const dse::CandidateGeneratorDescriptor *descriptor =
        invocation->generatorBinding.descriptorRef().descriptor();
    if (!descriptor || !descriptor->ownerFeedbackPayload ||
        descriptor->ownerFeedbackPayload->schemaDescriptorBytes !=
            mapping::systemAccCoreCapacityPressureSchemaBytes())
      continue;

    std::optional<ArtifactRootReference> system;
    std::optional<ArtifactRootReference> dataflow;
    std::vector<ArtifactRootReference> spatialMappings;
    for (const dse::CandidateGeneratorInputBinding &binding :
         invocation->inputBindings)
      for (const ArtifactRootReference &input : binding.artifacts) {
        if (input.schemaIdentity == fabric::fabricArtifactSchema.identity &&
            input.schemaVersion == fabric::fabricArtifactSchema.version) {
          auto imported = fabric::importEntireFabricRoot(input, artifacts);
          if (!imported)
            return imported.takeError();
          if (imported->view().rootKind() != fabric::FabricRootKind::System)
            continue;
          if (system)
            return invalid(
                "SystemMapping feedback names multiple System inputs");
          system = input;
          continue;
        }
        if (input.schemaIdentity == mapping::mappingArtifactSchema.identity &&
            input.schemaVersion == mapping::mappingArtifactSchema.version)
          spatialMappings.push_back(input);
        if (input.schemaIdentity ==
                ::dataflow::canonicalDataflowSchema.identity &&
            input.schemaVersion ==
                ::dataflow::canonicalDataflowSchema.version) {
          if (dataflow)
            return invalid(
                "SystemMapping feedback names multiple Dataflow inputs");
          dataflow = input;
        }
      }
    if (!system || !dataflow || spatialMappings.empty())
      return invalid("SystemMapping feedback lacks its exact Mapping inputs");
    auto adopted = mapping::adoptSystemAccCoreCapacityPressure(
        feedback.canonicalPayload, *system, *dataflow, spatialMappings,
        artifacts);
    if (!adopted)
      return adopted.takeError();
    SystemHardwareFeedbackObservation candidate{std::move(*adopted)};
    const std::uint64_t candidateOveruse = candidate.feedback.witnessUsage() -
                                           candidate.feedback.witnessCapacity();
    const std::uint64_t selectedOveruse =
        selected ? selected->feedback.witnessUsage() -
                       selected->feedback.witnessCapacity()
                 : 0;
    if (!selected || candidateOveruse > selectedOveruse ||
        (candidateOveruse == selectedOveruse &&
         mapping::encodeSystemAccCoreCapacityPressure(candidate.feedback) <
             mapping::encodeSystemAccCoreCapacityPressure(selected->feedback)))
      selected = std::move(candidate);
  }
  return selected;
}

llvm::Expected<HardwareRecipeGrowth> deriveHardwareRecipeGrowth(
    const ResolvedConfig &baseConfig,
    const std::optional<TechHardwareFeedbackObservation> &techObservation,
    const std::optional<SpatialHardwareFeedbackObservation> &spatialObservation,
    const std::optional<SystemHardwareFeedbackObservation> &systemObservation,
    const ArtifactStore &artifacts) {
  HardwareRecipeGrowth growth;
  growth.config = baseConfig;
  growth.resultingContexts =
      baseConfig.hardwareTarget.parameters.temporalResidentContexts;
  growth.resultingGateways = baseConfig.hardwareTarget.parameters.gatewayCount;
  growth.resultingAccCores = baseConfig.hardwareTarget.parameters.accCoreCount;

  if (techObservation) {
    auto module =
        fabric::importEntireFabricRoot(techObservation->module, artifacts);
    if (!module)
      return module.takeError();
    auto plan = dse::projectTechMappingComputeContextJointGrowthPlan(
        techObservation->feedback, module->view());
    if (!plan)
      return plan.takeError();
    for (const dse::ResizeInstructionStore &decision : plan->decisions) {
      const std::uint64_t currentCapacity =
          module->view().peResidentContextCount(decision.target);
      if (decision.instructionCapacity <= currentCapacity)
        return invalid("joint Module growth contains a non-growth decision");
      growth.maximumInstructionStoreCapacity =
          std::max(growth.maximumInstructionStoreCapacity,
                   static_cast<std::uint64_t>(decision.instructionCapacity));
    }
    growth.techModule = techObservation->module;
    growth.instructionStoreResizes = plan->decisions;
    growth.resizedInstructionStoreCount = plan->decisions.size();
  }

  if (spatialObservation) {
    const std::uint64_t addedGateways =
        spatialObservation->feedback.requiredBoundaryPairs();
    if (addedGateways == 0 ||
        addedGateways > std::numeric_limits<std::uint32_t>::max() -
                            growth.resultingGateways)
      return invalid("graph-boundary Hall feedback has no representable "
                     "gateway growth");
    growth.addedGateways = addedGateways;
    growth.resultingGateways += addedGateways;
    growth.config.hardwareTarget.parameters.gatewayCount =
        static_cast<std::uint32_t>(growth.resultingGateways);
  }

  if (systemObservation) {
    if (systemObservation->feedback.compatibleAccCoreCount() !=
        growth.resultingAccCores)
      return invalid("uniform recipe cannot represent heterogeneous AccCore "
                     "capacity growth");
    if (growth.resultingAccCores == std::numeric_limits<std::uint32_t>::max())
      return invalid("AccCore capacity growth exceeds the builtin recipe");
    growth.addedAccCores = systemObservation->feedback.additionalAccCoreCount();
    growth.resultingAccCores += growth.addedAccCores;
    growth.config.hardwareTarget.parameters.accCoreCount =
        static_cast<std::uint32_t>(growth.resultingAccCores);
    growth.accCoreParent = systemObservation->feedback.system();
    growth.accCoreTargetModule = systemObservation->feedback.targetModule();
  }

  if (growth.instructionStoreResizes.empty() && growth.addedContexts == 0 &&
      growth.addedGateways == 0 && growth.addedAccCores == 0)
    return invalid("Mapping feedback requests no builtin recipe growth");
  growth.config.dse.planNodes.clear();
  return growth;
}

llvm::Expected<MaterializedHardwareCandidate> materializeHardwareRecipeGrowth(
    HardwareRecipeGrowth growth, llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::TechMapping,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "mapping_feedback_recipe_growth";
        fields["resized_instruction_store_count"] =
            growth.resizedInstructionStoreCount;
        fields["maximum_instruction_store_capacity"] =
            growth.maximumInstructionStoreCapacity;
        fields["added_temporal_contexts"] = growth.addedContexts;
        fields["temporal_resident_contexts"] = growth.resultingContexts;
        fields["added_gateways"] = growth.addedGateways;
        fields["gateway_count"] = growth.resultingGateways;
        fields["added_acc_cores"] = growth.addedAccCores;
        fields["acc_core_count"] = growth.resultingAccCores;
      });

  auto templateConfig =
      dse::projectResolvedFabricTemplateConfigView(growth.config);
  if (!templateConfig)
    return templateConfig.takeError();
  auto binding =
      dse::resolveFabricTemplateCandidateGeneratorBinding(*templateConfig);
  if (!binding)
    return binding.takeError();
  growth.config.dse.planNodes = {dse::GeneratePlanNodeDefinition{
      binding->descriptorRef(),
      {},
      templateConfig->canonicalViewBytes().vec(),
      templateConfig->digest()}};
  auto execution = executeResolvedGeneratePlan(
      growth.config, {}, evidence, request, scheduler, artifacts, blobs);
  if (!execution)
    return execution.takeError();
  const dse::CompletedDsePlanExecution &available =
      availableExecution(*execution);
  const auto outputs = available.resolve({0, 0});
  if (outputs.size() != 1 || available.generateInvocations().size() != 1)
    return invalid("uniform recipe growth did not publish one System");
  auto system = fabric::importEntireFabricRoot(outputs.front(), artifacts);
  if (!system)
    return system.takeError();
  if (system->view().rootKind() != fabric::FabricRootKind::System)
    return invalid("uniform recipe growth published a non-System root");
  growth.config.dse.planNodes.clear();
  return MaterializedHardwareCandidate{outputs.front(),
                                       std::move(growth.config),
                                       std::nullopt,
                                       {},
                                       std::nullopt,
                                       growth.resizedInstructionStoreCount,
                                       growth.maximumInstructionStoreCapacity,
                                       growth.addedContexts,
                                       growth.resultingContexts,
                                       growth.addedGateways,
                                       growth.resultingGateways,
                                       growth.addedAccCores,
                                       growth.resultingAccCores};
}

llvm::Expected<ArtifactRootReference>
accCoreTargetModule(const fabric::FinalizedFabricRoot &system,
                    const fabric::FabricSystemRootView &view,
                    fabric::AccCoreOccurrenceRef core) {
  const auto target = view.spatialCoreTarget(core);
  if (!target ||
      target->dependencyOrdinal >= system.directDependencies().size())
    return invalid("AccCore has no exact imported Module target");
  const fabric::FabricDirectDependency &dependency =
      system.directDependencies()[target->dependencyOrdinal];
  if (dependency.role != fabric::FabricDependencyRole::ImportedModule)
    return invalid("AccCore target is not an ImportedModule dependency");
  return dependency.root;
}

struct MaterializedModuleGrowth final {
  ArtifactRootReference reference;
  std::vector<pnr::SystemModuleCorrespondence> correspondence;
  std::optional<HardwareImpactProjection> impact;
};

llvm::Expected<MaterializedModuleGrowth>
materializeTypedModuleGrowth(const HardwareRecipeGrowth &growth,
                             const ArtifactStore &artifacts,
                             const BlobStore &blobs) {
  const unsigned decisionKinds =
      !growth.instructionStoreResizes.empty() + growth.fifoResize.has_value() +
      growth.fifoBypassChange.has_value();
  if (!growth.techModule || decisionKinds != 1 ||
      growth.addedContexts != 0 ||
      growth.addedGateways != 0 || growth.addedAccCores != 0)
    return invalid("typed Module growth received a mixed or empty change");

  std::vector<SpatialMicroarchitectureDecisionDomain> domains;
  if (!growth.instructionStoreResizes.empty())
    domains.push_back(
        ResizeInstructionStoresDomain{growth.instructionStoreResizes});
  else if (growth.fifoResize)
    domains.push_back(ResizeFifoDomain{growth.fifoResize->target,
                                       {growth.fifoResize->depth}});
  else
    domains.push_back(ChangeFifoBypassCapabilityDomain{
        growth.fifoBypassChange->target,
        {growth.fifoBypassChange->bypassable}});
  auto config = resolveSpatialMicroarchitectureRewriteConfig(domains, 1);
  if (!config)
    return config.takeError();
  auto inputs = bindSpatialMicroarchitectureCandidateGeneratorInputs(
      {*growth.techModule});
  if (!inputs)
    return inputs.takeError();
  auto binding =
      resolveSpatialMicroarchitectureCandidateGeneratorBinding(*config);
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
    return invalid("typed Module growth did not publish one exact child");
  const ArtifactRootReference childReference =
      completed->outputBindings.front().artifacts.front();
  const CandidateGeneratorLineageEdge &lineage =
      completed->lineageEdges.front();
  if (lineage.kind != CandidateGeneratorLineageEdgeKind::CandidateDecision ||
      lineage.output != childReference ||
      lineage.parents != std::vector<ArtifactRootReference>{*growth.techModule})
    return invalid("typed Module growth lost its exact parent lineage");
  auto decision = adoptSpatialMicroarchitectureDecision(lineage.ownerPayload);
  if (!decision)
    return decision.takeError();
  if (decision->parent != *growth.techModule)
    return invalid("typed Module growth changed its parent owner");
  if (!growth.instructionStoreResizes.empty()) {
    const auto *resizes =
        std::get_if<ResizeInstructionStores>(&decision->decision);
    if (!resizes ||
        resizes->stores.size() != growth.instructionStoreResizes.size() ||
        !llvm::equal(resizes->stores, growth.instructionStoreResizes,
                     [](const ResizeInstructionStore &lhs,
                        const ResizeInstructionStore &rhs) {
                       return lhs.target == rhs.target &&
                              lhs.instructionCapacity ==
                                  rhs.instructionCapacity;
                     }))
      return invalid("typed Module growth changed its instruction resize");
  } else if (growth.fifoResize) {
    const auto *resize = std::get_if<ResizeFifo>(&decision->decision);
    if (!resize || resize->target != growth.fifoResize->target ||
        resize->depth != growth.fifoResize->depth)
      return invalid("typed Module growth changed its FIFO resize");
  } else {
    const auto *change =
        std::get_if<ChangeFifoBypassCapability>(&decision->decision);
    if (!change || change->target != growth.fifoBypassChange->target ||
        change->bypassable != growth.fifoBypassChange->bypassable)
      return invalid("typed Module growth changed its FIFO bypass decision");
  }
  auto impact = projectHardwareImpact(*decision, childReference);
  if (!impact.child ||
      (growth.instructionStoreResizes.empty() &&
       impact.family != HardwareMutationFamily::SpatialFifo))
    return invalid("typed Module growth has an incompatible impact family");
  if (!growth.instructionStoreResizes.empty() &&
      (impact.tech.kind != HardwareMappingImpactKind::Rebase ||
       impact.tech.realizationRoots.empty()))
    return invalid("instruction-store growth has no typed Tech impact");
  auto child = fabric::importEntireFabricRoot(childReference, artifacts);
  if (!child)
    return child.takeError();
  if (child->view().rootKind() != fabric::FabricRootKind::Module)
    return invalid("typed Module growth published a non-Module child");
  return MaterializedModuleGrowth{childReference,
                                  {{*growth.techModule, childReference}},
                                  std::move(impact)};
}

using SystemEntityCorrespondence = fabric::FabricSystemEntityCorrespondence;
using SystemTransferPatternCorrespondence =
    fabric::FabricSystemTransferPatternCorrespondence;

const SystemEntityCorrespondence *findEntityCorrespondence(
    llvm::ArrayRef<SystemEntityCorrespondence> correspondence,
    const fabric::FabricSystemEntityReference &source) {
  return llvm::find_if(correspondence, [&](const auto &entry) {
    return entry.source == source;
  });
}

const SystemTransferPatternCorrespondence *findTransferPatternCorrespondence(
    llvm::ArrayRef<SystemTransferPatternCorrespondence> correspondence,
    const fabric::FabricTransferPatternRef &source) {
  return llvm::find_if(correspondence, [&](const auto &entry) {
    return entry.source == source;
  });
}

llvm::Error composeSystemCorrespondence(
    std::vector<SystemEntityCorrespondence> &composedEntities,
    std::vector<SystemTransferPatternCorrespondence> &composedPatterns,
    llvm::ArrayRef<SystemEntityCorrespondence> nextEntities,
    llvm::ArrayRef<SystemTransferPatternCorrespondence> nextPatterns) {
  for (SystemEntityCorrespondence &entry : composedEntities) {
    const auto *next = findEntityCorrespondence(nextEntities, entry.target);
    if (!next)
      return invalid("System child lineage omits a preserved entity");
    entry.target = next->target;
  }
  for (SystemTransferPatternCorrespondence &entry : composedPatterns) {
    const auto *next =
        findTransferPatternCorrespondence(nextPatterns, entry.target);
    if (!next)
      return invalid("System child lineage omits a preserved transfer pattern");
    entry.target = next->target;
  }
  return llvm::Error::success();
}

llvm::Error remapCurrentAccCores(
    std::vector<fabric::AccCoreOccurrenceRef> &cores,
    llvm::ArrayRef<SystemEntityCorrespondence> correspondence) {
  for (fabric::AccCoreOccurrenceRef &core : cores) {
    const auto *mapped = findEntityCorrespondence(
        correspondence,
        fabric::FabricSystemEntityReference{
            fabric::FabricEntityKind::AccCoreOccurrence, core.id()});
    if (!mapped ||
        mapped->target.kind != fabric::FabricEntityKind::AccCoreOccurrence)
      return invalid("System child lineage lost a preserved AccCore");
    core = fabric::AccCoreOccurrenceRef(mapped->target.id);
  }
  return llvm::Error::success();
}

llvm::Expected<MaterializedHardwareCandidate>
materializeTypedModuleSystemGrowth(HardwareRecipeGrowth growth,
                                   const ArtifactRootReference &parentSystem,
                                   const ArtifactStore &artifacts,
                                   const BlobStore &blobs) {
  if (!growth.techModule)
    return invalid("typed Module System growth has no parent Module");
  auto parent = fabric::importEntireFabricRoot(parentSystem, artifacts);
  if (!parent)
    return parent.takeError();
  auto parentView = fabric::requireSystemRoot(parent->view());
  if (!parentView)
    return parentView.takeError();
  auto module = materializeTypedModuleGrowth(growth, artifacts, blobs);
  if (!module)
    return module.takeError();
  auto parentModules = projectJointDesignTargetModules(parentSystem, artifacts);
  if (!parentModules)
    return parentModules.takeError();
  for (const ArtifactRootReference &parentModule : *parentModules)
    if (parentModule != *growth.techModule)
      module->correspondence.push_back({parentModule, parentModule});
  llvm::sort(module->correspondence, [](const auto &lhs, const auto &rhs) {
    return artifactRootReferenceLess(lhs.parent, rhs.parent);
  });

  std::vector<fabric::AccCoreOccurrenceRef> currentCores;
  for (fabric::AccCoreOccurrenceRef core :
       parentView->artifact().accCoreOccurrences()) {
    auto target = accCoreTargetModule(*parent, *parentView, core);
    if (!target)
      return target.takeError();
    if (*target == *growth.techModule)
      currentCores.push_back(core);
  }
  if (currentCores.empty())
    return invalid("typed Module growth has no parent System attachment");

  ArtifactRootReference currentSystem = parentSystem;
  std::vector<SystemEntityCorrespondence> composedEntities;
  std::vector<SystemTransferPatternCorrespondence> composedPatterns;
  for (std::size_t index = 0; index != currentCores.size(); ++index) {
    auto config = resolveSystemCompositionRewriteConfig(
        {ReplaceSpatialAttachmentDomain{currentCores[index],
                                        {module->reference}}},
        1);
    if (!config)
      return config.takeError();
    auto inputs = bindSystemCompositionCandidateGeneratorInputs(
        {currentSystem}, {module->reference});
    if (!inputs)
      return inputs.takeError();
    auto binding = resolveSystemCompositionCandidateGeneratorBinding(*config);
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
      return invalid(
          "typed System Module replacement did not publish one child");
    const ArtifactRootReference childReference =
        completed->outputBindings.front().artifacts.front();
    const CandidateGeneratorLineageEdge &lineage =
        completed->lineageEdges.front();
    if (lineage.kind != CandidateGeneratorLineageEdgeKind::CandidateDecision ||
        lineage.output != childReference ||
        lineage.parents != std::vector<ArtifactRootReference>{currentSystem})
      return invalid("typed System replacement lost its exact parent lineage");
    auto decision = adoptSystemCompositionDecision(lineage.ownerPayload);
    if (!decision)
      return decision.takeError();
    const auto *replacement =
        std::get_if<ReplaceSpatialAttachment>(&decision->decision);
    if (decision->parent != currentSystem || !replacement ||
        replacement->target != currentCores[index] ||
        replacement->module != module->reference)
      return invalid("typed System replacement changed its exact decision");
    auto impact = projectHardwareImpact(*decision, childReference);
    if (!impact.child ||
        impact.system.kind != HardwareMappingImpactKind::Reopen ||
        impact.system.executionRoots.empty())
      return invalid("System Module replacement has no typed impact");
    if (index == 0) {
      composedEntities = decision->entities;
      composedPatterns = decision->transferPatterns;
    } else if (llvm::Error error = composeSystemCorrespondence(
                   composedEntities, composedPatterns, decision->entities,
                   decision->transferPatterns))
      return std::move(error);
    if (llvm::Error error =
            remapCurrentAccCores(currentCores, decision->entities))
      return std::move(error);
    currentSystem = childReference;
  }

  auto correspondence = pnr::SystemExecutionBindingCorrespondence::get(
      parentSystem, currentSystem, std::move(composedEntities),
      std::move(composedPatterns), module->correspondence, artifacts);
  if (!correspondence)
    return correspondence.takeError();
  std::optional<HardwareImpactProjection> mappingImpact =
      std::move(module->impact);
  if (mappingImpact) {
    mappingImpact->child = currentSystem;
  }
  growth.config.dse.planNodes.clear();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] =
            growth.instructionStoreResizes.empty()
                ? "typed_spatial_fifo_growth"
                : "typed_resize_instruction_stores_growth";
        fields["resized_instruction_store_count"] =
            growth.resizedInstructionStoreCount;
        fields["maximum_instruction_store_capacity"] =
            growth.maximumInstructionStoreCapacity;
        fields["replaced_acc_cores"] = currentCores.size();
        fields["preserved_acc_core_correspondences"] =
            correspondence->accCores().size();
      });
  return MaterializedHardwareCandidate{currentSystem,
                                       std::move(growth.config),
                                       std::move(*correspondence),
                                       std::move(module->correspondence),
                                       std::move(mappingImpact),
                                       growth.resizedInstructionStoreCount,
                                       growth.maximumInstructionStoreCapacity,
                                       growth.addedContexts,
                                       growth.resultingContexts,
                                       growth.addedGateways,
                                       growth.resultingGateways,
                                       growth.addedAccCores,
                                       growth.resultingAccCores};
}

llvm::Expected<MaterializedHardwareCandidate>
materializeTypedAccCoreGrowth(HardwareRecipeGrowth growth,
                              const ArtifactStore &artifacts,
                              const BlobStore &blobs) {
  if (growth.addedContexts != 0 || growth.addedGateways != 0 ||
      growth.addedAccCores != 1 || !growth.accCoreParent ||
      !growth.accCoreTargetModule)
    return invalid("typed AddAccCore materialization received a mixed change");
  auto parent =
      fabric::importEntireFabricRoot(*growth.accCoreParent, artifacts);
  if (!parent)
    return parent.takeError();
  auto parentView = fabric::requireSystemRoot(parent->view());
  if (!parentView)
    return parentView.takeError();
  std::optional<fabric::AccCoreOccurrenceRef> prototype;
  for (fabric::AccCoreOccurrenceRef core :
       parentView->artifact().accCoreOccurrences()) {
    auto target = accCoreTargetModule(*parent, *parentView, core);
    if (!target)
      return target.takeError();
    if (*target == *growth.accCoreTargetModule) {
      prototype = core;
      break;
    }
  }
  if (!prototype)
    return invalid("typed AddAccCore growth has no compatible prototype");

  std::vector<SystemCompositionDecisionDomain> domains = {
      AddAccCoreDomain{*prototype, {*growth.accCoreTargetModule}}};
  auto config = resolveSystemCompositionRewriteConfig(domains, 1);
  if (!config)
    return config.takeError();
  auto inputs = bindSystemCompositionCandidateGeneratorInputs(
      {*growth.accCoreParent}, {*growth.accCoreTargetModule});
  if (!inputs)
    return inputs.takeError();
  auto binding = resolveSystemCompositionCandidateGeneratorBinding(*config);
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
    return invalid("typed AddAccCore growth did not publish one exact child");
  const ArtifactRootReference childReference =
      completed->outputBindings.front().artifacts.front();
  const CandidateGeneratorLineageEdge &lineage =
      completed->lineageEdges.front();
  if (lineage.kind != CandidateGeneratorLineageEdgeKind::CandidateDecision ||
      lineage.output != childReference ||
      lineage.parents !=
          std::vector<ArtifactRootReference>{*growth.accCoreParent})
    return invalid("typed AddAccCore child lost its CandidateDecision lineage");
  auto decision = adoptSystemCompositionDecision(lineage.ownerPayload);
  if (!decision)
    return decision.takeError();
  const auto *added = std::get_if<AddAccCore>(&decision->decision);
  if (decision->parent != *growth.accCoreParent || !added ||
      added->prototype != *prototype ||
      added->module != *growth.accCoreTargetModule)
    return invalid("typed AddAccCore lineage changed its exact decision");
  auto child = fabric::importEntireFabricRoot(childReference, artifacts);
  if (!child)
    return child.takeError();
  auto modules =
      projectJointDesignTargetModules(parent->reference(), artifacts);
  if (!modules)
    return modules.takeError();
  std::vector<pnr::SystemModuleCorrespondence> moduleCorrespondence;
  moduleCorrespondence.reserve(modules->size());
  for (const ArtifactRootReference &module : *modules)
    moduleCorrespondence.push_back({module, module});
  auto correspondence = pnr::SystemExecutionBindingCorrespondence::get(
      parent->reference(), child->reference(), decision->entities,
      decision->transferPatterns, std::move(moduleCorrespondence), artifacts);
  if (!correspondence)
    return correspondence.takeError();
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "typed_add_acc_core_growth";
        fields["preserved_acc_core_correspondences"] =
            correspondence->accCores().size();
        fields["added_acc_cores"] = growth.addedAccCores;
      });
  growth.config.dse.planNodes.clear();
  return MaterializedHardwareCandidate{childReference,
                                       std::move(growth.config),
                                       std::move(*correspondence),
                                       {},
                                       std::nullopt,
                                       growth.resizedInstructionStoreCount,
                                       growth.maximumInstructionStoreCapacity,
                                       growth.addedContexts,
                                       growth.resultingContexts,
                                       growth.addedGateways,
                                       growth.resultingGateways,
                                       growth.addedAccCores,
                                       growth.resultingAccCores};
}

llvm::Expected<std::vector<ArtifactRootReference>>
normalizedTimingProfiles(const ArtifactRootReference &system,
                         const ArtifactStore &artifacts) {
  auto modules = dse::projectJointDesignTargetModules(system, artifacts);
  if (!modules)
    return modules.takeError();
  std::vector<ArtifactRootReference> profiles;
  profiles.reserve(modules->size());
  for (const ArtifactRootReference &moduleReference : *modules) {
    auto module = fabric::importEntireFabricRoot(moduleReference, artifacts);
    if (!module)
      return module.takeError();
    auto timing =
        fabric::projectNormalizedFabricPhysicalTimingProfile(module->view());
    if (!timing)
      return timing.takeError();
    auto reference =
        fabric::publishFabricPhysicalTimingProfile(*timing, artifacts);
    if (!reference)
      return reference.takeError();
    profiles.push_back(std::move(*reference));
  }
  return profiles;
}

llvm::Expected<FinalizedMappingHardwareSpectrum>
exploreFinalizedMappingHardwareSpectrum(
    const JointDesignPolicy &policy, const JointDesignExplorationPlan &plan,
    const JointDesignExecution &parentExecution,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const PlanExecutionPolicy *executionPolicy = nullptr) {
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
    auto execution = executeJointPlan(*childPlan, evidence, request, scheduler,
                                      artifacts, blobs,
                                      &effectiveExecutionPolicy);
    if (!execution)
      return execution.takeError();
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

llvm::Expected<std::optional<dse::JointDesignExecution>>
tryHardwareFeedbackReopen(
    const JointDesignPolicy &policy, const JointDesignExplorationPlan &plan,
    const dse::JointDesignExecution &failedExecution,
    std::optional<dse::JointDesignExecution> &lastFailedExecution,
    std::uint64_t planOrdinal,
    std::vector<dse::JointDesignAttemptRecord> &attemptRecords,
    dse::JointDesignExecutionSummary &accounting,
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
  bool currentFailureIsTechGate = false;
  std::optional<dse::JointDesignExecution> latestFailed;
  std::optional<dse::JointDesignExplorationPlan> latestFailedPlan;
  std::optional<std::vector<ArtifactRootReference>> reusableSpatialMappings;
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
              fields["previous_hall_contexts"] =
                  previousHallProgress->contexts;
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

    auto growth = deriveHardwareRecipeGrowth(currentConfig, *techObservation,
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
              std::nullopt,
              "parent execution stopped at the Tech gate"}},
            JointMappingReuseDisposition::ColdFallback};
      } else {
        auto projected = rebaseJointMappingFrontier(
            *currentPlan, *currentFailure, system->reference,
            system->moduleCorrespondences,
            system->mappingImpact ? &*system->mappingImpact : nullptr,
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
    auto reopenPlan = dse::buildJointDesignExplorationPlan(
        {{software.workloads}, {system->reference}}, *timing, *reopenPolicy,
        system->config, artifacts, mappingSeed,
        currentPlan->systemBindingPartitions);
    const std::uint64_t planBuildNanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - planBuildStart)
            .count());
    if (!reopenPlan) {
      if (mappingSeed)
        saturatingAdd(accounting.incrementalReopenWallTimeNanoseconds,
                      planBuildNanoseconds);
      else
        saturatingAdd(accounting.coldReopenWallTimeNanoseconds,
                      planBuildNanoseconds);
      return reopenPlan.takeError();
    }
    if (mappingSeed)
      saturatingAdd(accounting.incrementalReopenWallTimeNanoseconds,
                    planBuildNanoseconds);
    else
      saturatingAdd(accounting.coldReopenWallTimeNanoseconds,
                    planBuildNanoseconds);

    if (typedModuleGrowth) {
      const auto gateStart = std::chrono::steady_clock::now();
      auto gate = executeTechGate(*reopenPlan, evidence, request, scheduler,
                                  artifacts, blobs,
                                  effectiveExecutionPolicy);
      const std::uint64_t gateNanoseconds = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - gateStart)
              .count());
      if (!gate)
        return gate.takeError();
      saturatingAdd(accounting.techMappingDispatchCount,
                    gate->execution.summary.techMappingDispatchCount);
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
                attemptRecords, planOrdinal, system->reference,
                gate->execution))
          return std::move(error);
        ++accounting.hardwareRepairProbesConsumed;
        if (const auto *incomplete = std::get_if<IncompleteDsePlanExecution>(
                &gate->execution.planExecution);
            incomplete && incomplete->executionStopped())
          return std::optional<dse::JointDesignExecution>{
              std::move(gate->execution)};
        currentConfig = system->config;
        latestFailed = std::move(gate->execution);
        latestFailedPlan = std::move(*reopenPlan);
        currentFailure = &*latestFailed;
        currentPlan = &*latestFailedPlan;
        currentFailureIsTechGate = true;
        continue;
      }

      JointDesignMappingSeed gateSeed;
      if (rebased)
        gateSeed = rebased->seed;
      gateSeed.techMappings.insert(gateSeed.techMappings.end(),
                                   gate->techMappings.begin(),
                                   gate->techMappings.end());
      canonicalizeRoots(gateSeed.techMappings);
      canonicalizeRoots(gateSeed.spatialMappings);
      const auto gatedPlanStart = std::chrono::steady_clock::now();
      auto gatedPlan = dse::buildJointDesignExplorationPlan(
          {{software.workloads}, {system->reference}}, *timing, *reopenPolicy,
          system->config, artifacts, &gateSeed,
          currentPlan->systemBindingPartitions);
      const std::uint64_t gatedPlanNanoseconds = static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - gatedPlanStart)
              .count());
      if (!gatedPlan)
        return gatedPlan.takeError();
      if (mappingSeed)
        saturatingAdd(accounting.incrementalReopenWallTimeNanoseconds,
                      gatedPlanNanoseconds);
      else
        saturatingAdd(accounting.coldReopenWallTimeNanoseconds,
                      gatedPlanNanoseconds);
      reopenPlan = std::move(gatedPlan);
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
    auto execution = executeJointPlan(*reopenPlan, evidence, request, scheduler,
                                      artifacts, blobs,
                                      &effectiveExecutionPolicy);
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
    ++accounting.hardwareRepairProbesConsumed;
    saturatingAdd(accounting.techMappingDispatchCount,
                  execution->summary.techMappingDispatchCount);
    saturatingAdd(accounting.spatialPnrDispatchCount,
                  execution->summary.spatialPnrDispatchCount);
    saturatingAdd(accounting.systemPnrDispatchCount,
                  execution->summary.systemPnrDispatchCount);
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
    if (systemMappingCount != 0)
      return std::optional<dse::JointDesignExecution>{std::move(*execution)};
    if (const auto *incomplete =
            std::get_if<IncompleteDsePlanExecution>(&execution->planExecution);
        incomplete && incomplete->executionStopped())
      return std::optional<dse::JointDesignExecution>{std::move(*execution)};

    currentConfig = std::move(system->config);
    latestFailed = std::move(*execution);
    latestFailedPlan = std::move(*reopenPlan);
    currentFailure = &*latestFailed;
    currentPlan = &*latestFailedPlan;
    currentFailureIsTechGate = false;
  }
  if (latestFailed)
    lastFailedExecution = std::move(*latestFailed);
  return std::optional<dse::JointDesignExecution>{};
}

} // namespace

llvm::StringRef spatialFifoRuntimeFeedbackDispositionSpelling(
    SpatialFifoRuntimeFeedbackDisposition disposition) {
  switch (disposition) {
  case SpatialFifoRuntimeFeedbackDisposition::Exact:
    return "exact";
  case SpatialFifoRuntimeFeedbackDisposition::ProofNotEstablished:
    return "proof_not_established";
  case SpatialFifoRuntimeFeedbackDisposition::Unsupported:
    return "unsupported";
  }
  llvm_unreachable("unknown Spatial FIFO runtime feedback disposition");
}

llvm::StringRef spatialFifoRuntimeFeedbackReasonSpelling(
    SpatialFifoRuntimeFeedbackReason reason) {
  switch (reason) {
  case SpatialFifoRuntimeFeedbackReason::ExactFullFifoCycle:
    return "exact_full_fifo_cycle";
  case SpatialFifoRuntimeFeedbackReason::MissingWaitCycle:
    return "missing_wait_cycle";
  case SpatialFifoRuntimeFeedbackReason::MissingCanonicalFifo:
    return "missing_canonical_fifo";
  case SpatialFifoRuntimeFeedbackReason::AmbiguousFifo:
    return "ambiguous_fifo";
  case SpatialFifoRuntimeFeedbackReason::StorageNotFull:
    return "storage_not_full";
  case SpatialFifoRuntimeFeedbackReason::MissingCausalReleaseContext:
    return "missing_causal_release_context";
  }
  llvm_unreachable("unknown Spatial FIFO runtime feedback reason");
}

llvm::Expected<SpatialFifoRuntimeFeedback> deriveSpatialFifoRuntimeFeedback(
    const ArtifactRootReference &parentMapping,
    const ArtifactRootReference &spatialMapping,
    const sim::CgraClosedWaitSetDiagnostic &closedWait,
    const ArtifactStore &artifacts) {
  auto parent = mapping::importSystemMapping(parentMapping, artifacts);
  if (!parent)
    return parent.takeError();
  if (!llvm::is_contained(
          parent->view().executionBindings().spatialMappingImports(),
          spatialMapping))
    return invalid("FIFO runtime feedback names a foreign SpatialMapping");
  auto spatial = mapping::importSpatialMapping(spatialMapping, artifacts);
  if (!spatial)
    return spatial.takeError();
  ArtifactRootReference moduleReference{
      fabric::fabricArtifactSchema.identity.str(),
      fabric::fabricArtifactSchema.version, spatial->view().fabricIdentity()};
  auto module = fabric::importEntireFabricRoot(moduleReference, artifacts);
  if (!module)
    return module.takeError();
  if (module->view().rootKind() != fabric::FabricRootKind::Module)
    return invalid("FIFO runtime feedback does not bind a Module");

  SpatialFifoRuntimeFeedback feedback{
      parentMapping,
      spatialMapping,
      SpatialFifoRuntimeFeedbackDisposition::ProofNotEstablished,
      SpatialFifoRuntimeFeedbackReason::MissingWaitCycle,
      std::nullopt,
      0,
      0,
      std::nullopt,
      false,
      closedWait.transferWaitCycle.size(),
      closedWait.actorWaitCycle.size(),
      std::nullopt,
      std::nullopt,
      std::nullopt};

  const auto inTransferCycle = [&](const auto &transfer) {
    return llvm::any_of(closedWait.transferWaitCycle, [&](const auto &edge) {
      return (edge.waitingBindingOrdinal == transfer.bindingOrdinal &&
              edge.waitingOccurrenceOrdinal == transfer.occurrenceOrdinal) ||
             (edge.blockingBindingOrdinal == transfer.bindingOrdinal &&
              edge.blockingOccurrenceOrdinal == transfer.occurrenceOrdinal);
    });
  };
  const auto inActorCycle = [&](const auto &transfer) {
    return llvm::any_of(closedWait.actorWaitCycle, [&](const auto &edge) {
      return edge.waitingActorOrdinal == transfer.producerActorOrdinal ||
             edge.blockingActorOrdinal == transfer.producerActorOrdinal ||
             edge.waitingActorOrdinal == transfer.blockingActorOrdinal ||
             edge.blockingActorOrdinal == transfer.blockingActorOrdinal;
    });
  };
  const bool hasCycle = !closedWait.transferWaitCycle.empty() ||
                        !closedWait.actorWaitCycle.empty();
  std::vector<const sim::CgraClosedWaitSetDiagnostic::Transfer *> fifoWaits;
  for (const auto &transfer : closedWait.transfers) {
    if (!transfer.blocked || !transfer.blockingFifoOccurrence)
      continue;
    if (hasCycle && !inTransferCycle(transfer) && !inActorCycle(transfer))
      continue;
    fifoWaits.push_back(&transfer);
  }
  if (fifoWaits.empty()) {
    feedback.disposition =
        SpatialFifoRuntimeFeedbackDisposition::Unsupported;
    feedback.reason = SpatialFifoRuntimeFeedbackReason::MissingCanonicalFifo;
    return feedback;
  }
  llvm::sort(fifoWaits, [](const auto *lhs, const auto *rhs) {
    return fabric::canonicalFabricBytes(*lhs->blockingFifoOccurrence) <
           fabric::canonicalFabricBytes(*rhs->blockingFifoOccurrence);
  });
  const auto fifo = *fifoWaits.front()->blockingFifoOccurrence;
  if (llvm::any_of(fifoWaits, [&](const auto *transfer) {
        return *transfer->blockingFifoOccurrence != fifo;
      })) {
    feedback.reason = SpatialFifoRuntimeFeedbackReason::AmbiguousFifo;
    return feedback;
  }
  if (llvm::Error error = fabric::validateFabricRef(module->view(), fifo))
    return std::move(error);
  if (!mapping::spatialMappingUsesFifoOccurrence(spatial->view(), fifo))
    return invalid("FIFO runtime feedback names an unselected occurrence");
  feedback.fifo = fifo;
  feedback.occupancy = fifoWaits.front()->blockingStorageOccupancy;
  feedback.capacity = fifoWaits.front()->blockingStorageCapacity;
  for (const auto *transfer : fifoWaits)
    if (transfer->blockingStorageOccupancy != feedback.occupancy ||
        transfer->blockingStorageCapacity != feedback.capacity) {
      feedback.reason = SpatialFifoRuntimeFeedbackReason::AmbiguousFifo;
      return feedback;
    }
  for (const auto &traversal : module->view().admittedTraversals()) {
    const auto *candidate =
        std::get_if<fabric::FabricFifoTraversalPayload>(&traversal.payload);
    feedback.bypassCapable |=
        candidate && candidate->owner == fifo &&
        candidate->mode == fabric::FabricFifoTraversalMode::Bypass;
  }
  if (!hasCycle)
    return feedback;
  if (feedback.capacity == 0 || feedback.occupancy != feedback.capacity) {
    feedback.reason = SpatialFifoRuntimeFeedbackReason::StorageNotFull;
    return feedback;
  }

  for (const auto &action : closedWait.physicalActions) {
    if (!action.semanticActorOrdinal || !action.granted ||
        !action.requiresCausalRelease || !action.intrinsicReleaseReached ||
        action.causalReleaseReached)
      continue;
    const auto firing = llvm::find_if(
        closedWait.actorFirings, [&](const auto &candidate) {
          return candidate.semanticActorOrdinal ==
                     *action.semanticActorOrdinal &&
                 candidate.occurrenceOrdinal == action.occurrenceOrdinal &&
                 candidate.physicalComplete &&
                 !candidate.causalReleaseSatisfied;
        });
    if (firing == closedWait.actorFirings.end())
      continue;
    const bool ownsWait = llvm::any_of(fifoWaits, [&](const auto *transfer) {
      return transfer->producerActorOrdinal == *action.semanticActorOrdinal ||
             transfer->blockingActorOrdinal == *action.semanticActorOrdinal;
    });
    if (!ownsWait)
      continue;
    if (feedback.causalActorOrdinal &&
        (*feedback.causalActorOrdinal != *action.semanticActorOrdinal ||
         *feedback.causalActionOrdinal != action.actionOrdinal ||
         *feedback.causalOccurrenceOrdinal != action.occurrenceOrdinal)) {
      feedback.reason = SpatialFifoRuntimeFeedbackReason::
          MissingCausalReleaseContext;
      return feedback;
    }
    feedback.causalActorOrdinal = *action.semanticActorOrdinal;
    feedback.causalActionOrdinal = action.actionOrdinal;
    feedback.causalOccurrenceOrdinal = action.occurrenceOrdinal;
  }
  if (!feedback.causalActorOrdinal) {
    feedback.reason =
        SpatialFifoRuntimeFeedbackReason::MissingCausalReleaseContext;
    return feedback;
  }
  if (feedback.capacity == std::numeric_limits<std::uint32_t>::max())
    return invalid("FIFO runtime feedback depth overflows u32");
  feedback.minimumCandidateDepth = feedback.capacity + 1;
  feedback.disposition = SpatialFifoRuntimeFeedbackDisposition::Exact;
  feedback.reason = SpatialFifoRuntimeFeedbackReason::ExactFullFifoCycle;
  return feedback;
}

void emitSpatialFifoRuntimeFeedback(
    const SpatialFifoRuntimeFeedback &feedback) {
  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "spatial_fifo_runtime_feedback";
        fields["parent_mapping"] =
            formatArtifactIdentityHex(feedback.parentMapping.artifact);
        fields["spatial_mapping"] =
            formatArtifactIdentityHex(feedback.spatialMapping.artifact);
        fields["disposition"] =
            spatialFifoRuntimeFeedbackDispositionSpelling(
                feedback.disposition);
        fields["reason"] =
            spatialFifoRuntimeFeedbackReasonSpelling(feedback.reason);
        fields["occupancy"] = feedback.occupancy;
        fields["capacity"] = feedback.capacity;
        fields["bypass_capable"] = feedback.bypassCapable;
        fields["transfer_cycle_edge_count"] =
            feedback.transferCycleEdgeCount;
        fields["actor_cycle_edge_count"] = feedback.actorCycleEdgeCount;
        if (feedback.fifo)
          fields["fifo"] = llvm::toHex(
              fabric::canonicalFabricBytes(*feedback.fifo), true);
        else
          fields["fifo"] = nullptr;
        if (feedback.minimumCandidateDepth)
          fields["minimum_candidate_depth"] =
              *feedback.minimumCandidateDepth;
        else
          fields["minimum_candidate_depth"] = nullptr;
        if (feedback.causalActorOrdinal)
          fields["causal_actor"] = *feedback.causalActorOrdinal;
        else
          fields["causal_actor"] = nullptr;
        if (feedback.causalActionOrdinal)
          fields["causal_action"] = *feedback.causalActionOrdinal;
        else
          fields["causal_action"] = nullptr;
        if (feedback.causalOccurrenceOrdinal)
          fields["causal_occurrence"] = *feedback.causalOccurrenceOrdinal;
        else
          fields["causal_occurrence"] = nullptr;
        fields["hardware_child_count"] = 0;
      });
}

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
  auto repairPolicy = JointDesignPolicy::get(
      1, 1, 1, policy.maximumTechMappingsPerModule(),
      policy.maximumSpatialMappingsPerPair());
  if (!repairPolicy)
    return repairPolicy.takeError();
  const JointSoftwareScope &software = parentPair.software;
  ResolvedConfig childConfig = parentPlan.resolvedConfig;
  childConfig.dse.planNodes.clear();
  childConfig.dse.systemPnr.search.completionGoal =
      ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  auto childPlan = buildJointDesignExplorationPlan(
      {{software.workloads}, {system}}, *timing, *repairPolicy,
      childConfig, artifacts, nullptr, childPartitions);
  if (!childPlan)
    return childPlan.takeError();

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
  if (elapsedNanoseconds >
      std::numeric_limits<std::uint64_t>::max() -
          execution->summary.incrementalReopenWallTimeNanoseconds)
    execution->summary.incrementalReopenWallTimeNanoseconds =
        std::numeric_limits<std::uint64_t>::max();
  else
    execution->summary.incrementalReopenWallTimeNanoseconds +=
        elapsedNanoseconds;

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
        fields["wall_time_ns"] = elapsedNanoseconds;
      });
  return JointResourceTimeAdjacentRepair{
      *parentMapping, seed->reference(), std::move(*childPlan),
      std::move(*execution)};
}

llvm::Expected<JointSpatialFifoHardwareRepair>
executeSpatialFifoHardwareFeedbackReopen(
    const JointDesignExplorationPlan &parentPlan,
    const JointDesignExecution &parentExecution,
    const JointDesignPolicy &policy,
    const SpatialFifoRuntimeFeedback &feedback,
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
  auto parentMapping = mapping::importSystemMapping(feedback.parentMapping,
                                                    artifacts);
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

  HardwareRecipeGrowth growth;
  growth.config = parentPlan.resolvedConfig;
  growth.config.dse.planNodes.clear();
  growth.techModule = parentModule;
  growth.fifoResize = ResizeFifo{*feedback.fifo,
                                 *feedback.minimumCandidateDepth};
  auto child = materializeTypedModuleSystemGrowth(
      std::move(growth), parentPair.system, artifacts, blobs);
  if (!child)
    return child.takeError();
  if (!child->mappingImpact)
    return invalid("FIFO hardware child has no typed Mapping impact");
  auto rebased = rebaseJointMappingFrontier(
      parentPlan, parentExecution, child->reference,
      child->moduleCorrespondences, &*child->mappingImpact, artifacts);
  if (!rebased)
    return rebased.takeError();

  const JointDesignMappingSeed *mappingSeed =
      rebased->seed.techMappings.empty() &&
              rebased->seed.spatialMappings.empty()
          ? nullptr
          : &rebased->seed;
  auto timing = normalizedTimingProfiles(child->reference, artifacts);
  if (!timing)
    return timing.takeError();
  auto repairPolicy = JointDesignPolicy::get(
      1, 1, 1, policy.maximumTechMappingsPerModule(),
      policy.maximumSpatialMappingsPerPair());
  if (!repairPolicy)
    return repairPolicy.takeError();
  ResolvedConfig childConfig = child->config;
  childConfig.dse.planNodes.clear();
  childConfig.dse.systemPnr.search.completionGoal =
      ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  auto childPlan = buildJointDesignExplorationPlan(
      {{parentPair.software.workloads}, {child->reference}}, *timing,
      *repairPolicy, childConfig, artifacts, mappingSeed,
      parentPlan.systemBindingPartitions);
  if (!childPlan)
    return childPlan.takeError();
  if (rebased->disposition == JointMappingReuseDisposition::Preserved &&
      child->executionBindingCorrespondence) {
    auto context = deriveSystemMappingMigrationContext(*childPlan);
    if (!context)
      return context.takeError();
    auto seed = pnr::finalizeSystemMappingMigrationSeed(
        feedback.parentMapping, *child->executionBindingCorrespondence,
        *context, artifacts);
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
  if (rebased->disposition == JointMappingReuseDisposition::ColdFallback)
    execution->summary.coldReopenWallTimeNanoseconds = elapsedNanoseconds;
  else
    execution->summary.incrementalReopenWallTimeNanoseconds =
        elapsedNanoseconds;
  if (auto selected = firstMapping(*execution)) {
    execution->summary.selectedMapping = *selected;
    execution->summary.selectedPlanOrdinal = 0;
  }
  execution->summary.verifiedAlternatives = mappingCount(*execution);

  mapping_debug::emit(
      mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
      mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
        fields["operation"] = "spatial_fifo_hardware_repair";
        fields["parent_mapping"] =
            formatArtifactIdentityHex(feedback.parentMapping.artifact);
        fields["parent_system"] =
            formatArtifactIdentityHex(parentPair.system.artifact);
        fields["child_system"] =
            formatArtifactIdentityHex(child->reference.artifact);
        fields["candidate_depth"] = *feedback.minimumCandidateDepth;
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
        fields["rebase_failure_count"] = rebased->failures.size();
        llvm::json::Array failures;
        for (const JointMappingRebaseFailure &failure : rebased->failures) {
          llvm::json::Object encoded;
          encoded["reason"] =
              jointMappingRebaseFailureReasonSpelling(failure.reason);
          encoded["diagnostic"] = failure.diagnostic;
          if (failure.parent)
            encoded["parent"] =
                formatArtifactIdentityHex(failure.parent->artifact);
          else
            encoded["parent"] = nullptr;
          failures.push_back(std::move(encoded));
        }
        fields["rebase_failures"] = std::move(failures);
        fields["system_mapping_count"] = mappingCount(*execution);
        fields["wall_time_ns"] = elapsedNanoseconds;
        fields["liveness"] = "requires_child_cgra_replay";
        fields["ii_support"] = "unsupported";
        fields["throughput_support"] = "unsupported";
        fields["latency_support"] = "unsupported";
        fields["timing_fmax_support"] = "unsupported";
        fields["area_support"] = "unsupported";
        fields["power_energy_support"] = "unsupported";
        fields["reconfiguration_support"] = "unsupported";
        fields["bypass_alternative"] =
            feedback.bypassCapable ? "unsupported" : "not_applicable";
      });
  result.childSystems.push_back(child->reference);
  result.reuseDispositions.push_back(rebased->disposition);
  result.executions.push_back(std::move(*execution));
  result.bypassAlternativeUnsupported = feedback.bypassCapable;
  return result;
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
  } else if (request.boundedQuality) {
    return invalid("FirstVerified stopping cannot carry a bounded-quality "
                   "policy");
  }
  const auto finish = [&](JointDesignExecution execution,
                          std::optional<std::uint64_t> selectedPlanOrdinal,
                          std::optional<ArtifactRootReference> selectedMapping,
                          JointDesignQualityDisposition qualityDisposition,
                          std::optional<ArtifactRootReference>
                              qualityIncompleteCandidate,
                          bool declaredWorkExhausted) {
    if (accounting.hardwareRepairProbesReserved >=
        accounting.hardwareRepairProbesConsumed) {
      const std::uint64_t accounted =
          accounting.hardwareRepairProbesConsumed +
          accounting.hardwareRepairProbesRejected +
          accounting.hardwareRepairProbesCancelled;
      if (accounted < accounting.hardwareRepairProbesReserved) {
        const std::uint64_t remainder =
            accounting.hardwareRepairProbesReserved - accounted;
        if (deadlineObserved || dispatchDeadlineReached(request.executionPolicy))
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
    summary.hardwareReopensDeferredByQuality =
        hardwareReopensDeferredByQuality;
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
    summary.techMappingDispatchCount = accounting.techMappingDispatchCount;
    summary.spatialPnrDispatchCount = accounting.spatialPnrDispatchCount;
    summary.systemPnrDispatchCount = accounting.systemPnrDispatchCount;
    summary.coldReopenWallTimeNanoseconds =
        accounting.coldReopenWallTimeNanoseconds;
    summary.incrementalReopenWallTimeNanoseconds =
        accounting.incrementalReopenWallTimeNanoseconds;
    summary.timeToFirstFeasibleWallTimeNanoseconds = timeToFirstFeasible;
    summary.timeToBestWallTimeNanoseconds =
        static_cast<std::uint64_t>(
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
    summary.verifiedAlternatives = verifiedMappingCount;
    summary.selectedPlanOrdinal = selectedPlanOrdinal;
    summary.selectedMapping = std::move(selectedMapping);
    summary.qualityDisposition = qualityDisposition;
    summary.qualityIncompleteCandidate = std::move(qualityIncompleteCandidate);
    if (request.boundedQuality)
      summary.qualityObjectiveDimensionLabels =
          request.boundedQuality->objectiveDimensionLabels;
    summary.qualityObservations = qualityObservations;
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
          fields["hardware_parent_promotions"] =
              hardwareParentPromotions;
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
          fields["tech_mapping_dispatch_count"] =
              accounting.techMappingDispatchCount;
          fields["spatial_pnr_dispatch_count"] =
              accounting.spatialPnrDispatchCount;
          fields["system_pnr_dispatch_count"] =
              accounting.systemPnrDispatchCount;
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
          fields["reopened_tech_decisions"] =
              accounting.reopenedTechDecisions;
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
          fields["parent_service_leg_count"] =
              accounting.parentServiceLegCount;
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
      auto fair = fairBoundedQualityPlanPolicy(request.executionPolicy,
                                               remainingPlans);
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
    // The initial parent execution is outside tryHardwareFeedbackReopen, so
    // carry its invocation-local accounting into the stopping summary here.
    // Reopen attempts are accounted at their dispatch boundary below.
    saturatingAdd(accounting.techMappingDispatchCount,
                  initial->summary.techMappingDispatchCount);
    saturatingAdd(accounting.spatialPnrDispatchCount,
                  initial->summary.spatialPnrDispatchCount);
    saturatingAdd(accounting.systemPnrDispatchCount,
                  initial->summary.systemPnrDispatchCount);
    saturatingAdd(accounting.coldReopenWallTimeNanoseconds,
                  initial->summary.coldReopenWallTimeNanoseconds);
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
      // A bounded/incomplete software candidate does not prove that its
      // siblings are infeasible. Preserve the first witness and continue
      // through the declared software frontier; only an external caller
      // may decide to stop the whole joint invocation.
      if (!firstIncomplete)
        firstIncomplete = std::move(*initial);
      if (dispatchDeadlineReached(request.executionPolicy)) {
        deadlineObserved = true;
        boundedQualitySearchIncomplete = true;
        break;
      }
      continue;
    }
    auto coverage = projectJointSoftwareCoverage(plan, artifacts);
    if (!coverage)
      return coverage.takeError();
    failedSoftwareAttempts.push_back(
        {static_cast<std::uint64_t>(indexed.index()), planPointer,
         std::move(*coverage), std::move(*initial)});
  }
  // Hardware feedback is consumed only after every bounded software/System
  // pair has been tried on the parent System. This preserves the declared
  // software frontier order and prevents repairable early failures from
  // hiding a later parent-hardware solution.
  std::vector<FailedSoftwareAttempt *> hardwareFeedbackFrontier;
  if (request.stoppingPolicy != JointDesignStoppingPolicy::BoundedQuality ||
      verifiedAlternatives.empty()) {
    for (FailedSoftwareAttempt &attempt : failedSoftwareAttempts)
      hardwareFeedbackFrontier.push_back(&attempt);
  } else {
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
      if (*tech || *spatial || *system)
        hardwareFeedbackFrontier.push_back(&attempt);
    }
    llvm::sort(hardwareFeedbackFrontier,
               [](const FailedSoftwareAttempt *lhs,
                  const FailedSoftwareAttempt *rhs) {
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
    const std::size_t actionableFeedbackCount =
        hardwareFeedbackFrontier.size();
    const std::size_t limit = static_cast<std::size_t>(
        std::min<std::uint64_t>(
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
      auto fair = fairBoundedQualityPlanPolicy(
          request.executionPolicy,
          hardwareFeedbackFrontier.size() - indexedAttempt.index());
      if (!fair)
        return fair.takeError();
      feedbackExecutionPolicy.emplace(std::move(*fair));
      ++hardwareParentPromotions;
    }
    ++hardwareReopenSearches;
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::SystemPnr,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "hardware_feedback_promotion";
          fields["plan_ordinal"] = attempt.planOrdinal;
          fields["accelerated_root_count"] =
              attempt.coverage.acceleratedRootCount;
          fields["graph_count"] = attempt.coverage.graphCount;
          fields["actor_count"] = attempt.coverage.actorCount;
        });
    std::optional<JointDesignExecution> lastReopenedFailure;
    auto reopened = tryHardwareFeedbackReopen(
        policy, *attempt.plan, attempt.execution, lastReopenedFailure,
        attempt.planOrdinal, attemptRecords, accounting, request.evidence,
        request, *scheduler, artifacts, blobs,
        feedbackExecutionPolicy ? &*feedbackExecutionPolicy : nullptr);
    if (!reopened)
      return reopened.takeError();
    if (*reopened) {
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
  if (request.stoppingPolicy == JointDesignStoppingPolicy::BoundedQuality &&
      !verifiedAlternatives.empty()) {
    const std::size_t baseAlternativeCount = verifiedAlternatives.size();
    const std::uint64_t remainingParentBudget =
        request.boundedQuality->maximumHardwareSpectrumParents >
                hardwareParentPromotions
            ? request.boundedQuality->maximumHardwareSpectrumParents -
                  hardwareParentPromotions
            : 0;
    const std::uint64_t parentLimit =
        std::min<std::uint64_t>(
            remainingParentBudget,
            baseAlternativeCount);
    for (std::uint64_t parentOrdinal = 0; parentOrdinal != parentLimit;
         ++parentOrdinal) {
      if (dispatchDeadlineReached(request.executionPolicy)) {
        deadlineObserved = true;
        boundedQualitySearchIncomplete = true;
        break;
      }
      VerifiedAlternative &parent = verifiedAlternatives[parentOrdinal];
      if (parent.planOrdinal >= plans.size() || !plans[parent.planOrdinal])
        return invalid("bounded-quality hardware parent lost its plan");
      const std::uint64_t parentPlanOrdinal = parent.planOrdinal;
      ++hardwareParentPromotions;
      auto spectrumPolicy = fairBoundedQualityPlanPolicy(
          request.executionPolicy, parentLimit - parentOrdinal);
      if (!spectrumPolicy)
        return spectrumPolicy.takeError();
      auto spectrum = exploreFinalizedMappingHardwareSpectrum(
          policy, *plans[parentPlanOrdinal], parent.execution,
          request.evidence, request, *scheduler, artifacts, blobs,
          &*spectrumPolicy);
      if (!spectrum)
        return spectrum.takeError();
      hardwareReopenSearches += spectrum->attemptedSystems;
      boundedQualitySearchIncomplete |= spectrum->incomplete;
      for (JointDesignExecution &execution : spectrum->verified) {
        if (llvm::Error error = recordJointAttempt(
                attemptRecords, parentPlanOrdinal,
                plans[parentPlanOrdinal]->frontier.systemFrontier.front(),
                execution))
          return std::move(error);
        verifiedMappingCount += mappingCount(execution);
        saturatingAdd(accounting.techMappingDispatchCount,
                      execution.summary.techMappingDispatchCount);
        saturatingAdd(accounting.spatialPnrDispatchCount,
                      execution.summary.spatialPnrDispatchCount);
        saturatingAdd(accounting.systemPnrDispatchCount,
                      execution.summary.systemPnrDispatchCount);
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
    const auto qualityDisposition =
        [](JointDesignQualityIncompleteReason reason) {
          switch (reason) {
          case JointDesignQualityIncompleteReason::Unsupported:
            return JointDesignQualityDisposition::Unsupported;
          case JointDesignQualityIncompleteReason::ProofNotEstablished:
            return JointDesignQualityDisposition::ProofNotEstablished;
          case JointDesignQualityIncompleteReason::ExecutionFailed:
            return JointDesignQualityDisposition::ExecutionFailed;
          case JointDesignQualityIncompleteReason::CancelledOrTimeout:
            return JointDesignQualityDisposition::CancelledOrTimeout;
          }
          llvm_unreachable("unknown bounded-quality incomplete reason");
        };
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
          qualityObservations.push_back(
              {mapping,
               {},
               JointDesignQualityIncompleteReason::CancelledOrTimeout});
          if (!firstQualityIncomplete)
            firstQualityIncomplete = IncompleteJointDesignQuality{
                JointDesignQualityIncompleteReason::CancelledOrTimeout,
                mapping};
          continue;
        }
        alternative.execution.summary.selectedMapping = mapping;
        auto acquired =
            quality.acquire(alternative.execution, alternative.planOrdinal);
        if (!acquired)
          return acquired.takeError();
        if (const auto *incomplete =
                std::get_if<IncompleteJointDesignQuality>(&*acquired)) {
          qualityObservations.push_back({mapping, {}, incomplete->reason});
          if (!firstQualityIncomplete)
            firstQualityIncomplete = *incomplete;
          alternative.execution.summary.selectedMapping.reset();
          continue;
        }
        std::vector<CandidateObjectiveVector> one =
            std::get<std::vector<CandidateObjectiveVector>>(
                std::move(*acquired));
        if (one.size() != 1 || one.front().candidate != mapping)
          return invalid("bounded-quality acquisition must return exactly one "
                         "objective for the selected SystemMapping");
        qualityObservations.push_back(
            {mapping,
             std::vector<std::uint64_t>(one.front().objective.codes().begin(),
                                        one.front().objective.codes().end()),
             std::nullopt});
        acquiredObjectives.push_back(std::move(one.front()));
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
              qualityObservations[index].incompleteReason)
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
    if (objectives.empty()) {
      if (!firstQualityIncomplete)
        return invalid("bounded-quality acquisition produced no objectives");
      auto fallback = firstMapping(verifiedAlternatives.front().execution);
      if (!firstQualityIncomplete->candidate && !fallback)
        return invalid("bounded-quality incomplete result has no candidate");
      const ArtifactRootReference candidate =
          firstQualityIncomplete->candidate.value_or(*fallback);
      return finish(std::move(verifiedAlternatives.front().execution),
                    std::nullopt, std::nullopt,
                    qualityDisposition(firstQualityIncomplete->reason),
                    candidate, !deadlineObserved);
    }
    if (firstQualityIncomplete || boundedQualitySearchIncomplete ||
        deadlineObserved) {
      const ArtifactRootReference candidate =
          firstQualityIncomplete
              ? firstQualityIncomplete->candidate.value_or(
                    candidates.empty()
                        ? firstMapping(verifiedAlternatives.front().execution)
                              .value()
                        : candidates.front())
              : firstMapping(verifiedAlternatives.front().execution).value();
      return finish(std::move(verifiedAlternatives.front().execution),
                    std::nullopt, std::nullopt,
                    JointDesignQualityDisposition::ProofNotEstablished,
                    candidate, false);
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

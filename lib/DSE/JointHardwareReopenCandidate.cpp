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

namespace loom::dse::joint_reopen_detail {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "joint_hardware_reopen_invalid: " + message);
}

llvm::Expected<JointSoftwareCoverage>
projectJointSoftwareCoverage(const JointDesignExplorationPlan &plan,
                             const ArtifactStore &artifacts) {
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

llvm::Expected<PlanExecutionPolicy>
fairBoundedQualityPlanPolicy(const PlanExecutionPolicy &base,
                             std::uint64_t remainingPlanCount) {
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
  return PlanExecutionPolicy::get(base.workerCount(), base.inProcessClaim(),
                                  base.externalSite(), base.resourceBindings(),
                                  base.maximumDispatches(), localDeadline);
}

static const dse::CompletedDsePlanExecution &
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

llvm::Expected<std::vector<ArtifactRootReference>>
boundTechMappingFrontierForRepair(
    llvm::ArrayRef<ArtifactRootReference> candidates, std::uint64_t limit,
    const ArtifactStore &artifacts) {
  if (limit == 0)
    return invalid("Tech repair frontier has a zero mapping bound");
  std::vector<ArtifactRootReference> canonical(candidates.begin(),
                                               candidates.end());
  canonicalizeRoots(canonical);
  std::map<std::string, ArtifactRootReference> representativeByGraph;
  for (const ArtifactRootReference &reference : canonical) {
    auto mapping = mapping::importTechMapping(reference, artifacts);
    if (!mapping)
      return mapping.takeError();
    for (const ::dataflow::GraphRef graph : mapping->view().covers()) {
      std::vector<std::uint8_t> key(graph.artifact.bytes().begin(),
                                    graph.artifact.bytes().end());
      for (int shift = 56; shift >= 0; shift -= 8)
        key.push_back(static_cast<std::uint8_t>(graph.entity.value() >> shift));
      const std::string spelling(reinterpret_cast<const char *>(key.data()),
                                 key.size());
      representativeByGraph.emplace(spelling, reference);
    }
  }
  if (representativeByGraph.size() > limit)
    return invalid("Tech repair frontier bound cannot preserve graph coverage");
  std::vector<ArtifactRootReference> selected;
  selected.reserve(static_cast<std::size_t>(limit));
  for (const auto &[graph, reference] : representativeByGraph) {
    (void)graph;
    if (!llvm::is_contained(selected, reference))
      selected.push_back(reference);
  }
  for (const ArtifactRootReference &reference : canonical) {
    if (selected.size() >= limit)
      break;
    if (!llvm::is_contained(selected, reference))
      selected.push_back(reference);
  }
  canonicalizeRoots(selected);
  return selected;
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

static llvm::Expected<dse::DsePlanExecutionResult> executeResolvedGeneratePlan(
    const ResolvedConfig &config,
    std::vector<ArtifactRootReference> semanticInputs,
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const JointHardwareReopenRequest &request, dse::SiteScheduler &scheduler,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const PlanExecutionPolicy *executionPolicy = nullptr) {
  semanticInputs.insert(semanticInputs.end(),
                        request.invocationSemanticInputs.begin(),
                        request.invocationSemanticInputs.end());
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

llvm::Expected<TechGateExecution>
executeTechGate(const JointDesignExplorationPlan &plan,
                llvm::ArrayRef<ArtifactRootReference> evidence,
                const JointHardwareReopenRequest &request,
                SiteScheduler &scheduler, const ArtifactStore &artifacts,
                const BlobStore &blobs,
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
  for (auto indexed : llvm::enumerate(available.generateInvocations())) {
    ++summary.techMappingInvocationCount;
    if (available.generateInvocationWasDispatched(indexed.index()))
      ++summary.techMappingDispatchCount;
    else
      ++summary.techMappingJournalReplayCount;
  }
  for (std::size_t node = 0; node != config.dse.planNodes.size(); ++node) {
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
              peValues.push_back(
                  llvm::json::Object{{"pe", pe}, {"context_count", count}});
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

/// A parent without a reusable Tech/Spatial frontier can expose an exact Hall
/// deficit before it has any child Mapping. In that case the minimal
/// compatible-PE closure is a useful lower bound but not always a robust PnR
/// seed. Admit one bounded uniform Temporal-PE capacity alternative so the
/// hardware owner can measure that tradeoff without enumerating a powerset of
/// PE subsets.
llvm::Expected<HardwareRecipeGrowth> deriveUniformTechHardwareRecipeGrowth(
    const ResolvedConfig &baseConfig,
    const TechHardwareFeedbackObservation &observation,
    const ArtifactStore &artifacts) {
  if (observation.feedback.deficit() <= 1)
    return invalid("uniform Tech growth requires a deficit greater than one");
  auto module = fabric::importEntireFabricRoot(observation.module, artifacts);
  if (!module)
    return module.takeError();
  if (module->view().rootKind() != fabric::FabricRootKind::Module)
    return invalid("uniform Tech growth target is not a Module");

  HardwareRecipeGrowth growth;
  growth.config = baseConfig;
  growth.techModule = observation.module;
  const std::uint64_t baseContexts =
      baseConfig.hardwareTarget.parameters.temporalResidentContexts;
  if (observation.feedback.deficit() >
      std::numeric_limits<std::uint32_t>::max() - baseContexts)
    return invalid("uniform Tech growth exceeds the global context ABI");
  growth.resultingContexts = baseContexts + observation.feedback.deficit();
  growth.config.hardwareTarget.parameters.temporalResidentContexts =
      static_cast<std::uint32_t>(growth.resultingContexts);
  growth.resultingGateways = baseConfig.hardwareTarget.parameters.gatewayCount;
  growth.resultingAccCores = baseConfig.hardwareTarget.parameters.accCoreCount;
  for (const fabric::FabricPeOccurrenceRef pe :
       module->view().peOccurrences()) {
    auto schedule = module->view().peSchedule(pe);
    if (!schedule)
      return invalid("uniform Tech growth target PE has no schedule");
    if (*schedule != ::fabric::Schedule::Temporal)
      continue;
    const std::uint64_t current = module->view().peResidentContextCount(pe);
    if (current == 0 || observation.feedback.deficit() >
                            std::numeric_limits<std::uint32_t>::max() - current)
      return invalid("uniform Tech growth exceeds the Temporal PE context ABI");
    const std::uint32_t target =
        static_cast<std::uint32_t>(current + observation.feedback.deficit());
    growth.instructionStoreResizes.push_back({pe, target});
    growth.maximumInstructionStoreCapacity =
        std::max(growth.maximumInstructionStoreCapacity,
                 static_cast<std::uint64_t>(target));
  }
  if (growth.instructionStoreResizes.empty())
    return invalid("uniform Tech growth has no Temporal PE");
  growth.resizedInstructionStoreCount = growth.instructionStoreResizes.size();
  growth.uniformContextGrowth = true;
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
        fields["uniform_context_growth"] = growth.uniformContextGrowth;
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

namespace {

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
  const unsigned decisionKinds = !growth.instructionStoreResizes.empty() +
                                 growth.fifoResize.has_value() +
                                 growth.fifoBypassChange.has_value() +
                                 growth.operandBufferModeChange.has_value() +
                                 growth.operandBufferResize.has_value() +
                                 growth.moduleDecision.has_value();
  if (!growth.techModule || decisionKinds != 1 || growth.addedContexts != 0 ||
      growth.addedGateways != 0 || growth.addedAccCores != 0)
    return invalid("typed Module growth received a mixed or empty change");

  std::vector<SpatialMicroarchitectureDecisionDomain> domains;
  if (growth.moduleDecision)
    domains.push_back(*growth.moduleDecision);
  else if (!growth.instructionStoreResizes.empty())
    domains.push_back(
        ResizeInstructionStoresDomain{growth.instructionStoreResizes});
  else if (growth.fifoResize)
    domains.push_back(ResizeFifoDomain{growth.fifoResize->target,
                                       {growth.fifoResize->depth}});
  else if (growth.fifoBypassChange)
    domains.push_back(ChangeFifoBypassCapabilityDomain{
        growth.fifoBypassChange->target,
        {growth.fifoBypassChange->bypassable}});
  else if (growth.operandBufferModeChange)
    domains.push_back(ChangeTemporalOperandBufferModeDomain{
        growth.operandBufferModeChange->target,
        {growth.operandBufferModeChange->mode}});
  else
    domains.push_back(ResizeTemporalOperandBufferDomain{
        growth.operandBufferResize->target,
        {growth.operandBufferResize->entriesPerAllocationUnit}});
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
  if (growth.moduleDecision) {
    auto expanded = expandSpatialMicroarchitectureDecisionDomains(
        llvm::ArrayRef<SpatialMicroarchitectureDecisionDomain>(
            *growth.moduleDecision));
    if (!expanded)
      return expanded.takeError();
    if (expanded->size() != 1 ||
        expanded->front().index() != decision->decision.index())
      return invalid("typed Module growth changed its decision domain");
  } else if (!growth.instructionStoreResizes.empty()) {
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
  } else if (growth.fifoBypassChange) {
    const auto *change =
        std::get_if<ChangeFifoBypassCapability>(&decision->decision);
    if (!change || change->target != growth.fifoBypassChange->target ||
        change->bypassable != growth.fifoBypassChange->bypassable)
      return invalid("typed Module growth changed its FIFO bypass decision");
  } else if (growth.operandBufferModeChange) {
    const auto *change =
        std::get_if<ChangeTemporalOperandBufferMode>(&decision->decision);
    if (!change || change->target != growth.operandBufferModeChange->target ||
        change->mode != growth.operandBufferModeChange->mode)
      return invalid(
          "typed Module growth changed its operand-buffer mode decision");
  } else {
    const auto *resize =
        std::get_if<ResizeTemporalOperandBuffer>(&decision->decision);
    if (!resize || resize->target != growth.operandBufferResize->target ||
        resize->entriesPerAllocationUnit !=
            growth.operandBufferResize->entriesPerAllocationUnit)
      return invalid(
          "typed Module growth changed its operand-buffer resize decision");
  }
  auto impact = projectHardwareImpact(*decision, childReference);
  if (!impact.child ||
      (!growth.moduleDecision && growth.instructionStoreResizes.empty() &&
       !growth.operandBufferModeChange && !growth.operandBufferResize &&
       impact.family != HardwareMutationFamily::SpatialFifo))
    return invalid("typed Module growth has an incompatible impact family");
  if (!growth.moduleDecision && !growth.instructionStoreResizes.empty() &&
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

} // namespace

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
            growth.moduleDecision ? "typed_module_hardware_mutation"
            : !growth.instructionStoreResizes.empty()
                ? "typed_resize_instruction_stores_growth"
            : growth.operandBufferModeChange || growth.operandBufferResize
                ? "typed_temporal_operand_buffer_growth"
                : "typed_spatial_fifo_growth";
        fields["resized_instruction_store_count"] =
            growth.resizedInstructionStoreCount;
        fields["maximum_instruction_store_capacity"] =
            growth.maximumInstructionStoreCapacity;
        fields["uniform_context_growth"] = growth.uniformContextGrowth;
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

} // namespace loom::dse::joint_reopen_detail

#include "DSE/JointDesignExploration.h"
#include "DSE/HardwareDecision.h"
#include "DSE/JointHardwareReopen.h"
#include "DSE/JointMappingMigration.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/ResolvedConfigView.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Frontend/IR/LoomOps.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "PnR/System/SystemMappingMigration.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "joint design exploration anchor failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory("loom-joint-design", path_))
      fail("cannot create test directory: " + error.message());
  }
  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }
  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry
      .insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
              mlir::DLTIDialect, mlir::func::FuncDialect, loom::LoomDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context,
                                                  std::int32_t constant) {
  const std::string source = R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker domain(#dataflow.thread_domain<dense>)(
      %value: i32) ctrl (%ctrl: none) iv (%i: index) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%value) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %value = arith.constant )mlir" +
                             std::to_string(constant) + R"mlir( : i32
    %extent = arith.constant 4 : index
    %thread = dataflow.thread.launch @worker(%value) grid(%extent)
        : (i32) -> !dataflow.thread_token
    return
  }
}

)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    fail("cannot parse Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

loom::ArtifactRootReference
publishApplicationWorkload(const dataflow::CanonicalDataflowArtifact &artifact,
                           const loom::ArtifactStore &store) {
  auto view = take(artifact.view());
  if (view.rootThreadLaunches().size() != 1 ||
      view.staticGraphLaunches().size() != 1)
    fail("application fixture does not have one rooted graph launch");
  dataflow::RootedGraphLaunchRef launch{view.rootThreadLaunches().front().ref,
                                        view.staticGraphLaunches().front().ref};
  loom::sim::SpatialSimulationWorkload draft{launch};
  draft.denseCoordinates = {0};
  auto shapes =
      take(loom::sim::projectSpatialSimulationBoundaryShapes(view, launch));
  draft.valueInputPlan.assign(shapes.valueInputs.size(),
                              loom::sim::RuntimeValueInput{});
  auto workload = take(loom::sim::finalizeSimulationWorkload(draft, view));
  return take(loom::sim::publishSimulationWorkload(workload, store));
}

std::string key(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

bool everyCoreIsUsed(const loom::ArtifactRootReference &systemReference,
                     llvm::ArrayRef<loom::ArtifactRootReference> mappings,
                     const loom::ArtifactStore &store) {
  auto systemArtifact =
      take(loom::fabric::importEntireFabricRoot(systemReference, store));
  auto system = take(loom::fabric::requireSystemRoot(systemArtifact.view()));
  std::set<std::string> used;
  for (const loom::ArtifactRootReference &reference : mappings) {
    auto mapping = take(loom::mapping::importSystemMapping(reference, store));
    loom::ArtifactRootReference dataflowReference{
        dataflow::canonicalDataflowSchema.identity.str(),
        dataflow::canonicalDataflowSchema.version,
        mapping.view().dataflowIdentity()};
    auto dataflowArtifact =
        take(dataflow::importCanonicalDataflow(dataflowReference, store));
    auto dataflowView = take(dataflowArtifact.view());
    auto projection = take(loom::mapping::projectSystemExecutionContexts(
        dataflowView, mapping.view().executionBindings()));
    for (const auto &domain : projection.instructionDomains)
      used.insert(
          key(loom::fabric::canonicalFabricBytes(domain.context.accCore)));
  }
  return llvm::all_of(
      system.artifact().accCoreOccurrences(),
      [&](loom::fabric::AccCoreOccurrenceRef core) {
        return used.count(key(loom::fabric::canonicalFabricBytes(core))) != 0;
      });
}

void exerciseJointExploration(bool runFifoHardwareRepair) {
  TemporaryDirectory temporary;
  llvm::SmallString<128> blobPath(temporary.path());
  llvm::sys::path::append(blobPath, "blobs");
  if (std::error_code error = llvm::sys::fs::create_directories(blobPath))
    fail("cannot create BlobStore directory: " + error.message());
  loom::ArtifactStore store(temporary.path());
  loom::BlobStore blobs(blobPath);
  mlir::MLIRContext context = makeContext();

  auto first = buildDataflow(context, 7);
  auto second = buildDataflow(context, 11);
  take(dataflow::publishCanonicalDataflow(first, store));
  take(dataflow::publishCanonicalDataflow(second, store));
  const loom::ArtifactRootReference firstWorkload =
      publishApplicationWorkload(first, store);
  const loom::ArtifactRootReference secondWorkload =
      publishApplicationWorkload(second, store);
  auto small = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Small));
  auto alternate = take(loom::adg::buildBuiltinTarget(
      store, loom::adg::BuiltinTargetPreset::Coverage));
  if (small.roots().size() != 1 || alternate.roots().size() != 1)
    fail("builtin fixture did not publish one complete System");
  const loom::ArtifactRootReference system = small.roots().front().reference();
  const loom::ArtifactRootReference alternateSystem =
      alternate.roots().front().reference();
  auto systemArtifact =
      take(loom::fabric::importEntireFabricRoot(system, store));
  auto systemView =
      take(loom::fabric::requireSystemRoot(systemArtifact.view()));
  auto timingProfiles = take(
      loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(systemView));
  std::vector<loom::ArtifactRootReference> timingProfileRoots;
  for (const auto &profile : timingProfiles)
    timingProfileRoots.push_back(
        take(loom::fabric::publishFabricPhysicalTimingProfile(profile, store)));

  const loom::dse::JointDesignPolicy policy =
      take(loom::dse::JointDesignPolicy::get(2, 1, 1, 2, 32));
  loom::ResolvedConfig config = loom::defaultResolvedConfig();
  config.dse.techMapping.candidatePublicationLimit = 4;
  auto plan = take(loom::dse::buildJointDesignExplorationPlan(
      {{{firstWorkload}, {secondWorkload}}, {system}}, timingProfileRoots,
      policy, config, store));
  if (plan.frontier.eligiblePairCount != 2 || !plan.frontier.truncated ||
      plan.frontier.pairs.size() != 1 || plan.pairOutputs.size() != 1)
    fail("bounded pair frontier did not declare deterministic truncation");
  if (plan.frontier.analyticEvaluatedPairCount != 2 ||
      plan.frontier.analyticDeferredPairCount != 1 ||
      plan.frontier.pairProjections.size() != 1 ||
      plan.frontier.pairProjections.front().softwareActorCount == 0 ||
      plan.frontier.pairProjections.front().systemAccCoreCount == 0)
    fail("analytic pair funnel lost bounded ranking evidence");
  if (plan.pairOutputs.front().techMappings.empty() ||
      plan.pairOutputs.front().spatialMappings.empty())
    fail("joint Mapping plan lost an intermediate result projection");
  const auto &systemNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
      plan.resolvedConfig.dse.planNodes
          [plan.pairOutputs.front().systemMappings.producerNodeOrdinal]);
  const auto &join =
      std::get<loom::dse::BoundedPlanOutputJoin>(systemNode.inputBindings[1]);
  if (join.outputs.empty() || join.maximumArtifacts != 32)
    fail("joint Mapping plan lost its explicit SpatialMapping bound");
  for (const loom::dse::PlanOutputRef &spatialOutput : join.outputs) {
    const auto &spatialNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
        plan.resolvedConfig.dse.planNodes[spatialOutput.producerNodeOrdinal]);
    const auto &techJoin =
        std::get<loom::dse::BoundedPlanOutputJoin>(
            spatialNode.inputBindings.front());
    if (techJoin.outputs.size() != 1 || techJoin.maximumArtifacts != 2)
      fail("joint Mapping plan lost its TechMapping admission bound");
    const auto &techOutput = techJoin.outputs.front();
    const auto &techNode = std::get<loom::dse::GeneratePlanNodeDefinition>(
        plan.resolvedConfig.dse.planNodes[techOutput.producerNodeOrdinal]);
    if (techNode.descriptor !=
        loom::dse::applicationGraphTechMappingCandidateGeneratorDescriptor()
            .reference())
      fail("joint Mapping plan used a whole-program TechMapping cover");
  }

  auto view =
      take(loom::dse::projectResolvedDseConfigView(plan.resolvedConfig));
  auto execution = take(loom::dse::executeDsePlan(view, store, blobs));
  const loom::dse::CompletedDsePlanExecution *completed =
      std::get_if<loom::dse::CompletedDsePlanExecution>(&execution);
  if (!completed) {
    const auto &incomplete =
        std::get<loom::dse::IncompleteDsePlanExecution>(execution);
    const auto *reason =
        std::get_if<loom::dse::CandidateGeneratorIncompleteReason>(
            &incomplete.reason());
    if (!reason ||
        *reason != loom::dse::CandidateGeneratorIncompleteReason::
                       SemanticLimitReached ||
        incomplete.executionStopped())
      fail("joint Mapping plan changed retained frontier semantics: " +
           loom::dse::toString(incomplete.reason()));
    completed = &incomplete.availableExecution();
  }
  const std::vector<loom::ArtifactRootReference> mappings =
      completed->resolve(plan.pairOutputs.front().systemMappings).vec();
  if (mappings.empty())
    fail("joint Mapping plan produced no complete SystemMapping");
  for (const loom::ArtifactRootReference &reference : mappings) {
    auto mapping = take(loom::mapping::importSystemMapping(reference, store));
    if (mapping.view().dataflowIdentity() !=
            plan.frontier.pairs.front().software.dataflow.artifact ||
        mapping.view().fabricIdentity() != system.artifact)
      fail("joint Mapping output lost its exact pair owners");
  }

  auto mappedDataflow = take(dataflow::importCanonicalDataflow(
      plan.frontier.pairs.front().software.dataflow, store));
  auto mappedDataflowView = take(mappedDataflow.view());
  std::vector<dataflow::RootThreadLaunchRef> mappedRoots;
  for (const auto &root : mappedDataflowView.rootThreadLaunches())
    mappedRoots.push_back(root.ref);
  if (mappedRoots.size() != 1 ||
      systemView.artifact().accCoreOccurrences().size() < 2)
    fail("adjacent resource-time repair fixture lacks one root and two cores");
  loom::dse::JointDesignExecution parentExecution{
      std::move(execution),
      {{plan.frontier.pairs.front(), mappings}},
      {}};
  parentExecution.summary.selectedMapping = mappings.front();
  parentExecution.summary.selectedPlanOrdinal = 0;
  parentExecution.summary.verifiedAlternatives = mappings.size();
  const auto targetModules =
      take(loom::dse::projectJointDesignTargetModules(system, store));
  std::vector<loom::pnr::SystemModuleCorrespondence>
      identityModuleCorrespondence;
  for (const auto &module : targetModules)
    identityModuleCorrespondence.push_back({module, module});
  loom::dse::HardwareImpactProjection systemOnlyImpact{
      system, system, {}, {}, {}, {}};
  systemOnlyImpact.family =
      loom::dse::HardwareMutationFamily::SystemTransport;
  systemOnlyImpact.locality = loom::dse::HardwareMutationLocality::LocalCone;
  systemOnlyImpact.system.kind =
      loom::dse::HardwareMappingImpactKind::Reopen;
  if (!systemView.transportResources().empty())
    systemOnlyImpact.system.transportRoots.push_back(
        systemView.transportResources().front());
  const auto preservedFrontier =
      take(loom::dse::rebaseJointMappingFrontier(
          plan, parentExecution, system, identityModuleCorrespondence,
          &systemOnlyImpact, store));
  if (preservedFrontier.disposition !=
          loom::dse::JointMappingReuseDisposition::Preserved ||
      preservedFrontier.seed.techMappings.empty() ||
      preservedFrontier.seed.spatialMappings.empty() ||
      preservedFrontier.accounting.invalidatedTechMappings != 0 ||
      preservedFrontier.accounting.invalidatedSpatialMappings != 0)
    fail("System-only impact did not preserve lower Mapping layers");

  auto targetModule =
      take(loom::fabric::importEntireFabricRoot(targetModules.front(), store));
  if (targetModule.view().fifoOccurrences().empty())
    fail("FIFO feedback fixture has no physical FIFO");
  auto feedbackParentMapping =
      take(loom::mapping::importSystemMapping(mappings.front(), store));
  std::optional<loom::ArtifactRootReference> feedbackSpatialMapping;
  std::optional<loom::fabric::FabricFifoOccurrenceRef> feedbackFifo;
  for (const auto &reference : feedbackParentMapping.view()
                                   .executionBindings()
                                   .spatialMappingImports()) {
    auto spatial = take(loom::mapping::importSpatialMapping(reference, store));
    if (spatial.view().fabricIdentity() != targetModule.view().identity())
      continue;
    for (const auto fifo : targetModule.view().fifoOccurrences())
      if (loom::mapping::spatialMappingUsesFifoOccurrence(spatial.view(),
                                                          fifo)) {
        feedbackSpatialMapping = reference;
        feedbackFifo = fifo;
        break;
      }
    if (feedbackSpatialMapping)
      break;
  }
  if (!feedbackSpatialMapping || !feedbackFifo)
    fail("FIFO feedback fixture has no selected physical FIFO");
  loom::sim::CgraClosedWaitSetDiagnostic exactFifoWait;
  exactFifoWait.pendingActorFirings = 1;
  exactFifoWait.pendingTransfers = 1;
  exactFifoWait.pendingPhysicalActions = 1;
  exactFifoWait.actorFirings.push_back({0, 0, 0, 1, 0, true, false});
  loom::sim::CgraClosedWaitSetDiagnostic::Transfer blockedTransfer;
  blockedTransfer.bindingOrdinal = 0;
  blockedTransfer.occurrenceOrdinal = 0;
  blockedTransfer.producerActorOrdinal = 0;
  blockedTransfer.blocked = true;
  blockedTransfer.blockingActorOrdinal = 0;
  blockedTransfer.blockingFifoOccurrence = *feedbackFifo;
  blockedTransfer.blockingStorageOccupancy = 1;
  blockedTransfer.blockingStorageCapacity = 1;
  exactFifoWait.transfers.push_back(std::move(blockedTransfer));
  exactFifoWait.physicalActions.push_back(
      {0, 0, 0, 0, true, true, true, true, false});
  exactFifoWait.transferWaitCycle.push_back({0, 0, 0, 0, 0});
  const auto exactFifoFeedback =
      take(loom::dse::deriveSpatialFifoRuntimeFeedback(
          mappings.front(), *feedbackSpatialMapping, exactFifoWait, store));
  if (exactFifoFeedback.disposition !=
          loom::dse::SpatialFifoRuntimeFeedbackDisposition::Exact ||
      exactFifoFeedback.minimumCandidateDepth != 2 ||
      exactFifoFeedback.occupancy != 1 || exactFifoFeedback.capacity != 1)
    fail("exact FIFO wait did not admit the minimal hardware candidate");
  if (runFifoHardwareRepair) {
    llvm::SmallString<128> fifoJournal(temporary.path());
    llvm::sys::path::append(fifoJournal, "fifo-hardware-feedback");
    const auto fifoHardwareRepair =
        take(loom::dse::executeSpatialFifoHardwareFeedbackReopen(
            plan, parentExecution, policy, exactFifoFeedback,
            {take(loom::dse::DseProducerSemanticBuildIdentity::get(
                 "loom.test.spatial_fifo_feedback.v1")),
             fifoJournal.str().str(),
             {},
             loom::dse::JointDesignStoppingPolicy::FirstVerified,
             std::nullopt,
             std::nullopt,
             take(loom::dse::SiteCapacity::get(2, 0, 0)),
             take(loom::dse::PlanExecutionPolicy::get(
                 2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
            store, blobs));
    if (fifoHardwareRepair.childSystems.size() != 1 ||
        fifoHardwareRepair.executions.size() != 1 ||
        fifoHardwareRepair.reuseDispositions.size() != 1 ||
        fifoHardwareRepair.childSystems.front() == system)
      fail("exact FIFO feedback did not materialize one typed System child");
    std::vector<loom::ArtifactRootReference> fifoChildMappings;
    for (const auto &pair : fifoHardwareRepair.executions.front().mappedPairs)
      fifoChildMappings.insert(fifoChildMappings.end(),
                               pair.systemMappings.begin(),
                               pair.systemMappings.end());
    if (fifoChildMappings.empty())
      fail("exact FIFO hardware child produced no verified SystemMapping");
    const auto &repairSummary = fifoHardwareRepair.executions.front().summary;
    if (repairSummary.parentSpatialDecisions == 0 ||
        repairSummary.repairedTechDecisions == 0 ||
        repairSummary.parentRouteNodeCount == 0 ||
        (fifoHardwareRepair.reuseDispositions.front() ==
             loom::dse::JointMappingReuseDisposition::ColdFallback
             ? (repairSummary.repairedSpatialDecisions != 0 ||
                repairSummary.coldReopenWallTimeNanoseconds == 0 ||
                repairSummary.reopenedSpatialDecisions == 0)
             : repairSummary.repairedSpatialDecisions == 0))
      fail("FIFO hardware repair did not expose decision and route-cone "
           "accounting");
    for (const auto &reference : fifoChildMappings) {
      auto childMapping =
          take(loom::mapping::importSystemMapping(reference, store));
      if (childMapping.view().fabricIdentity() !=
          fifoHardwareRepair.childSystems.front().artifact)
        fail("FIFO hardware repair Mapping names the parent System");
    }
  }
  auto incompleteFifoWait = exactFifoWait;
  incompleteFifoWait.transferWaitCycle.clear();
  const auto incompleteFifoFeedback =
      take(loom::dse::deriveSpatialFifoRuntimeFeedback(
          mappings.front(), *feedbackSpatialMapping, incompleteFifoWait,
          store));
  if (incompleteFifoFeedback.disposition !=
          loom::dse::SpatialFifoRuntimeFeedbackDisposition::
              ProofNotEstablished ||
      incompleteFifoFeedback.reason !=
          loom::dse::SpatialFifoRuntimeFeedbackReason::MissingWaitCycle ||
      incompleteFifoFeedback.minimumCandidateDepth)
    fail("probe-incomplete FIFO wait synthesized a hardware child");
  if (runFifoHardwareRepair)
    return;
  if (targetModule.view().fuOccurrences().empty())
    fail("mapping-reuse fixture has no FU impact root");
  const auto moduleRoot =
      take(loom::fabric::FabricModulePhysicalOwnerRef::create(
          targetModule.view().fuOccurrences().front()));
  loom::dse::HardwareImpactProjection localSpatialImpact{
      targetModules.front(), system, {}, {}, {}, {}};
  localSpatialImpact.family = loom::dse::HardwareMutationFamily::SpatialFifo;
  localSpatialImpact.locality =
      loom::dse::HardwareMutationLocality::LocalCone;
  localSpatialImpact.tech.kind =
      loom::dse::HardwareMappingImpactKind::Rebase;
  localSpatialImpact.spatial.kind =
      loom::dse::HardwareMappingImpactKind::Reopen;
  localSpatialImpact.spatial.placementRoots.push_back(moduleRoot);
  const auto localRepairFrontier =
      take(loom::dse::rebaseJointMappingFrontier(
          plan, parentExecution, system, identityModuleCorrespondence,
          &localSpatialImpact, store));
  if (localRepairFrontier.disposition !=
          loom::dse::JointMappingReuseDisposition::LocalRepair ||
      localRepairFrontier.seed.techMappings.empty() ||
      !localRepairFrontier.seed.spatialMappings.empty() ||
      localRepairFrontier.accounting.invalidatedSpatialMappings == 0)
    fail("typed local Spatial impact did not isolate layer repair");

  auto globalImpact = localSpatialImpact;
  globalImpact.family = loom::dse::HardwareMutationFamily::FuCapability;
  globalImpact.locality =
      loom::dse::HardwareMutationLocality::GlobalReopen;
  globalImpact.tech.kind = loom::dse::HardwareMappingImpactKind::Reopen;
  globalImpact.tech.realizationRoots.push_back(moduleRoot);
  const auto coldFallbackFrontier =
      take(loom::dse::rebaseJointMappingFrontier(
          plan, parentExecution, system, identityModuleCorrespondence,
          &globalImpact, store));
  if (coldFallbackFrontier.disposition !=
          loom::dse::JointMappingReuseDisposition::ColdFallback ||
      !coldFallbackFrontier.seed.techMappings.empty() ||
      !coldFallbackFrontier.seed.spatialMappings.empty())
    fail("typed global impact did not preserve a cold fallback");
  llvm::SmallString<128> adjacentJournal(temporary.path());
  llvm::sys::path::append(adjacentJournal, "adjacent-resource-time");
  const std::array adjacentPartitions = {
      loom::pnr::SystemBindingPartitionIntent{mappedRoots.front(), 2}};
  const std::array adjacentRoots = {mappedRoots.front()};
  const auto adjacentRepair =
      take(loom::dse::executeResourceTimeAdjacentMappingRepair(
          plan, parentExecution, policy, adjacentPartitions, adjacentRoots,
          {take(loom::dse::DseProducerSemanticBuildIdentity::get(
               "loom.test.resource_time_adjacent.v1")),
           adjacentJournal.str().str(),
           {},
           loom::dse::JointDesignStoppingPolicy::FirstVerified,
           std::nullopt,
           std::nullopt,
           take(loom::dse::SiteCapacity::get(2, 0, 0)),
           take(loom::dse::PlanExecutionPolicy::get(
               2, take(loom::dse::SiteResourceClaim::get(1, 0, 0))))},
          store, blobs));
  const auto adjacentSeed = take(loom::pnr::importSystemMappingMigrationSeed(
      adjacentRepair.migrationSeed, store));
  if (adjacentSeed.reopenedRoots() !=
          llvm::ArrayRef<dataflow::RootThreadLaunchRef>(adjacentRoots) ||
      adjacentRepair.execution.summary.techMappingDispatchCount != 0 ||
      adjacentRepair.execution.summary.spatialPnrDispatchCount != 0 ||
      adjacentRepair.execution.summary.systemPnrDispatchCount != 1 ||
      adjacentRepair.execution.summary.preservedTechMappings == 0 ||
      adjacentRepair.execution.summary.preservedSpatialMappings == 0)
    fail("adjacent resource-time finalist did not use preserve-first repair");
  std::vector<loom::ArtifactRootReference> adjacentMappings;
  for (const auto &pair : adjacentRepair.execution.mappedPairs)
    adjacentMappings.insert(adjacentMappings.end(), pair.systemMappings.begin(),
                            pair.systemMappings.end());
  llvm::sort(adjacentMappings, loom::artifactRootReferenceLess);
  adjacentMappings.erase(
      std::unique(adjacentMappings.begin(), adjacentMappings.end()),
      adjacentMappings.end());
  if (adjacentMappings.empty() ||
      llvm::is_contained(adjacentMappings, mappings.front()))
    fail("adjacent resource-time repair did not publish a distinct Mapping");
  auto adjacentMapping =
      take(loom::mapping::importSystemMapping(adjacentMappings.front(), store));
  if (adjacentMapping.view().dataflowIdentity() !=
          plan.frontier.pairs.front().software.dataflow.artifact ||
      adjacentMapping.view().fabricIdentity() != system.artifact)
    fail("adjacent resource-time repair changed immutable owners");

  const std::vector<loom::ArtifactRootReference> systems = {system,
                                                            alternateSystem};
  const std::vector<loom::dse::JointMemberPromotion> memberPromotions = {
      {plan.frontier.pairs.front().software.dataflow,
       loom::dse::CompletedSelection{mappings, {}}}};
  auto selected = take(loom::dse::selectJointDesignSystems(
      systems, memberPromotions, {}, loom::dse::AllPassingSelection{}, nullptr,
      store));
  const bool covered = everyCoreIsUsed(system, mappings, store);
  bool sawMissingAlternate = false;
  bool sawUnusedPrimary = false;
  std::vector<loom::dse::JointSystemGateOutcome> *outcomes = nullptr;
  if (auto *completedSelection =
          std::get_if<loom::dse::JointDesignSelection>(&selected)) {
    outcomes = &completedSelection->systemOutcomes;
    if (!covered ||
        completedSelection->selectedSystems !=
            std::vector<loom::ArtifactRootReference>{system} ||
        completedSelection->acceptedMappings != mappings)
      fail("aggregate selection bypassed member-local System gates");
  } else {
    auto &noFeasible =
        std::get<loom::dse::JointDesignNoFeasibleSystem>(selected);
    outcomes = &noFeasible.systemOutcomes;
    if (covered)
      fail("fully covered System was rejected before aggregate selection");
  }
  for (const loom::dse::JointSystemGateOutcome &outcome : *outcomes) {
    if (const auto *missing =
            std::get_if<loom::dse::JointSystemMissingMember>(&outcome))
      sawMissingAlternate |= missing->system == alternateSystem;
    if (const auto *unused =
            std::get_if<loom::dse::JointSystemUnusedAccCore>(&outcome))
      sawUnusedPrimary |= unused->system == system;
  }
  if (!sawMissingAlternate || sawUnusedPrimary == covered)
    fail("typed System dispositions lost missing-member or AccCore coverage");

  auto oversized = loom::dse::buildBoundedJointFrontier(
      {{{firstWorkload}, {secondWorkload}}, {system}},
      take(loom::dse::JointDesignPolicy::get(1, 1, 1, 1, 1)), store);
  if (oversized)
    fail("joint frontier accepted a software set beyond its resolved bound");
  const std::string oversizedMessage = llvm::toString(oversized.takeError());
  if (!llvm::StringRef(oversizedMessage).contains("exceeds"))
    fail("frontier-bound rejection lost its diagnostic");
}

} // namespace

int main(int argc, char **argv) {
  if (argc > 2 ||
      (argc == 2 && llvm::StringRef(argv[1]) != "fifo-feedback"))
    fail("expected no workflow or fifo-feedback");
  exerciseJointExploration(argc == 2);
  return 0;
}

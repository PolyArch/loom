#include "JointDesignExplorationFixture.h"

#include "ADG/Builtin.h"
#include "Config/ResolvedConfig.h"
#include "DSE/HardwareMutationRepairRecord.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/JointHardwareReopen.h"
#include "DSE/ResolvedConfigView.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace loom::dse::joint_test {
namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "joint FIFO discipline feedback failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

} // namespace

void exerciseFifoDisciplineHardwareFeedback(
    const ArtifactRootReference &workload, llvm::StringRef temporaryPath,
    const ArtifactStore &store, const BlobStore &blobs) {
  auto config = defaultResolvedConfig();
  config.hardwareTarget = {adg::builtinSmallTarget.templateIdentity.str(),
                           {adg::builtinSmallTarget.schemaMajor,
                            adg::builtinSmallTarget.schemaMinor},
                           adg::builtinSmallTarget.scale};
  auto &scale = config.hardwareTarget.parameters;
  scale.accCoreCount = 2;
  scale.meshDimension = 2;
  scale.spatialMeshLanesPerDirection = 1;
  scale.temporalMeshLanesPerDirection = 1;
  scale.spatialPeCount = 1;
  scale.temporalPeCount = 1;
  // Only the temporal PE implements the workload's sync. Mapping must cross
  // the tagged network, independently of candidate ordering or route costs.
  scale.spatialFuOccurrences = {1, 0, 0, 0, 0, 0, 0, 0};
  scale.temporalFuOccurrences = {0, 0, 0, 0, 1, 0, 0, 0};
  config.dse.techMapping.candidatePublicationLimit = 1;
  const auto design = take(adg::buildBuiltinTarget(store, scale));
  if (design.roots().size() != 1)
    fail("temporal target did not publish one System");
  const auto system = design.roots().front().reference();
  const auto imported = take(fabric::importEntireFabricRoot(system, store));
  const auto systemView = take(fabric::requireSystemRoot(imported.view()));
  std::vector<ArtifactRootReference> timingRoots;
  for (const auto &profile :
       take(fabric::projectNormalizedSystemPhysicalTimingProfiles(systemView)))
    timingRoots.push_back(
        take(fabric::publishFabricPhysicalTimingProfile(profile, store)));
  const auto policy = take(JointDesignPolicy::get(1, 1, 1, 1, 8));
  const auto plan = take(buildJointDesignExplorationPlan(
      {{{workload}}, {system}}, timingRoots, policy, config, store));
  auto execution = take(executeDsePlan(
      take(projectResolvedDseConfigView(plan.resolvedConfig)), store, blobs));
  const auto *completed = std::get_if<CompletedDsePlanExecution>(&execution);
  if (!completed) {
    const auto &incomplete = std::get<IncompleteDsePlanExecution>(execution);
    const auto *reason =
        std::get_if<CandidateGeneratorIncompleteReason>(&incomplete.reason());
    if (!reason ||
        *reason != CandidateGeneratorIncompleteReason::SemanticLimitReached ||
        incomplete.executionStopped())
      fail("temporal mapping did not complete: " +
           toString(incomplete.reason()));
    completed = &incomplete.availableExecution();
  }
  const auto mappings =
      completed->resolve(plan.pairOutputs.front().systemMappings).vec();
  if (mappings.empty())
    fail("temporal target produced no verified SystemMapping");
  JointDesignExecution parent{
      std::move(execution), {{plan.frontier.pairs.front(), mappings}}, {}};
  parent.summary.selectedMapping = mappings.front();
  parent.summary.selectedPlanOrdinal = 0;
  parent.summary.verifiedAlternatives = mappings.size();

  const auto parentMapping =
      take(mapping::importSystemMapping(mappings.front(), store));
  const auto modules = take(projectJointDesignTargetModules(system, store));
  if (modules.size() != 1)
    fail("temporal target has more than one Module");
  const auto module =
      take(fabric::importEntireFabricRoot(modules.front(), store));
  std::optional<ArtifactRootReference> spatialMapping;
  std::vector<fabric::FabricFifoOccurrenceRef> taggedFifos;
  std::uint32_t tagWidth = 0;
  for (const auto &reference :
       parentMapping.view().executionBindings().spatialMappingImports()) {
    const auto spatial = take(mapping::importSpatialMapping(reference, store));
    for (const auto fifo : module.view().fifoOccurrences()) {
      if (!mapping::spatialMappingUsesFifoOccurrence(spatial.view(), fifo))
        continue;
      const auto port = module.view().transportEndpointDataPath(
          {fabric::FabricTransportEndpointOwnerRef::of(fifo), 0});
      if (port && port->kind == ::fabric::DataPathKind::BitsTag &&
          module.view().fifoQueueDiscipline(fifo) ==
              ::fabric::FifoQueueDiscipline::StrictFifo) {
        spatialMapping = reference;
        taggedFifos.push_back(fifo);
        tagWidth = port->tagWidthBits;
      }
    }
    if (!taggedFifos.empty())
      break;
  }
  if (taggedFifos.size() < 2 || !spatialMapping || tagWidth == 0)
    fail("temporal mapping selected fewer than two tagged StrictFifos");

  const auto exerciseTargets = [&](const std::vector<
                                       fabric::FabricFifoOccurrenceRef>
                                       &targets,
                                   llvm::StringRef journalName) {
    using ClosedWait = sim::CgraClosedWaitSetDiagnostic;
    ClosedWait wait;
    for (const auto fifo : targets) {
      ClosedWait::WaitEdge order;
      order.from =
          ClosedWait::WaitOwnerKey{ClosedWait::WaitActorFiringKey{0, 0}};
      order.to = ClosedWait::WaitOwnerKey{ClosedWait::WaitStorageQueueKey{
          ClosedWait::WaitStorageDomain::TraversalStorage, 0,
          ClosedWait::WaitQueueClass::global()}};
      order.kind = ClosedWait::WaitEdgeKind::StorageOrder;
      order.fifoOccurrence = fifo;
      order.awaitedTagValue = llvm::APInt(tagWidth, 0);
      order.headTagValue = llvm::APInt(tagWidth, 1);
      ClosedWait::WaitEdge consumer;
      consumer.from = order.to;
      consumer.to = order.from;
      consumer.kind = ClosedWait::WaitEdgeKind::StorageConsumer;
      wait.waitCertificate.push_back(order);
      wait.waitCertificate.push_back(consumer);
    }
    const auto feedback = take(deriveSpatialFifoRuntimeFeedback(
        mappings.front(), *spatialMapping, wait, store));
    if (feedback.disposition != SpatialFifoRuntimeFeedbackDisposition::Exact ||
        feedback.reason !=
            SpatialFifoRuntimeFeedbackReason::ExactCrossTagGlobalHolCycle ||
        feedback.currentQueueDiscipline !=
            ::fabric::FifoQueueDiscipline::StrictFifo ||
        feedback.candidateQueueDiscipline !=
            ::fabric::FifoQueueDiscipline::PerTagVirtualChannel ||
        feedback.disciplineTargets != targets || feedback.minimumCandidateDepth)
      fail("cross-tag global HOL did not admit the VC hardware candidate");

    llvm::SmallString<128> journal(temporaryPath);
    llvm::sys::path::append(journal, journalName);
    const auto repair = take(executeSpatialFifoHardwareFeedbackReopen(
        plan, parent, policy, feedback,
        {take(DseProducerSemanticBuildIdentity::get(
             "loom.test.spatial_fifo_recipe_feedback.v1")),
         journal.str().str(),
         {},
         JointDesignStoppingPolicy::FirstVerified,
         std::nullopt,
         std::nullopt,
         take(SiteCapacity::get(2, 0, 0)),
         take(PlanExecutionPolicy::get(2,
                                       take(SiteResourceClaim::get(1, 0, 0))))},
        store, blobs));
    if (repair.childSystems.size() != 1 || repair.repairRecords.size() != 1 ||
        repair.executions.size() != 1 ||
        repair.reuseDispositions !=
            std::vector<JointMappingReuseDisposition>{
                targets.size() == 1
                    ? JointMappingReuseDisposition::LocalRepair
                    : JointMappingReuseDisposition::ColdFallback})
      fail("FIFO discipline feedback lost its typed repair disposition");
    const auto record = take(importHardwareMutationRepairRecord(
        repair.repairRecords.front(), store));
    if (record.record().parentSystem != system ||
        record.record().childSystem != repair.childSystems.front() ||
        record.record().impacts.size() != targets.size() ||
        !llvm::all_of(
            record.record().impacts,
            [](const auto &impact) {
              return impact.family == HardwareMutationFamily::SpatialFifo &&
                     impact.locality == HardwareMutationLocality::LocalCone &&
                     impact.spatial.kind == HardwareMappingImpactKind::Reopen;
            }) ||
        record.record().incremental.mappings.empty())
      fail("FIFO discipline repair record lost its reopened Spatial cone");

    const auto childModules = take(
        projectJointDesignTargetModules(repair.childSystems.front(), store));
    if (childModules.size() != 1)
      fail("FIFO discipline child lost its Module");
    const auto child =
        take(fabric::importEntireFabricRoot(childModules.front(), store));
    std::size_t virtualChannels = 0;
    for (const auto fifo : child.view().fifoOccurrences())
      virtualChannels += child.view().fifoQueueDiscipline(fifo) ==
                         ::fabric::FifoQueueDiscipline::PerTagVirtualChannel;
    if (virtualChannels != targets.size())
      fail("FIFO discipline repair changed the wrong number of occurrences");
    for (const auto target : targets) {
      const auto parentFifos = module.view().fifoOccurrences();
      fabric::FabricModuleEntityReference entity{
          fabric::FabricEntityKind::FabricFifoOccurrence, target.id(),
          static_cast<std::uint64_t>(
              llvm::find(parentFifos, target) - parentFifos.begin())};
      for (const auto &impact : record.record().impacts) {
        const auto mapped =
            llvm::find_if(impact.moduleEntities, [&](const auto &entry) {
              return entry.source == entity;
            });
        if (mapped == impact.moduleEntities.end())
          fail("FIFO discipline record lost its target correspondence");
        entity = mapped->target;
      }
      if (module.view().fifoQueueDiscipline(target) !=
              ::fabric::FifoQueueDiscipline::StrictFifo ||
          child.view().fifoQueueDiscipline(
              fabric::FabricFifoOccurrenceRef(entity.id)) !=
              ::fabric::FifoQueueDiscipline::PerTagVirtualChannel)
        fail("FIFO discipline repair changed a different physical occurrence");
    }
  };
  exerciseTargets({taggedFifos.front()}, "fifo-discipline-feedback");

  // Select a second witnessed FIFO whose identity really moves after the
  // first rewrite. Reusing its old ordinal must change the wrong occurrence
  // or fail, rather than accidentally passing on stable canonical labels.
  const auto firstChild = take(materializeJointModuleHardwareMutation(
      config, system, modules.front(),
      SpatialMicroarchitectureDecisionDomain{ChangeFifoQueueDisciplineDomain{
          taggedFifos.front(),
          {::fabric::FifoQueueDiscipline::PerTagVirtualChannel}}},
      store, blobs));
  const auto &entities = firstChild.impacts.front().moduleEntities;
  const auto renamed =
      llvm::find_if(llvm::drop_begin(taggedFifos), [&](const auto fifo) {
        return llvm::any_of(entities, [&](const auto &entry) {
          return entry.source.kind ==
                     fabric::FabricEntityKind::FabricFifoOccurrence &&
                 entry.source.id == fifo.id() &&
                 entry.target.id != entry.source.id;
        });
      });
  if (renamed == taggedFifos.end())
    fail("FIFO discipline fixture did not relabel a remaining target");
  exerciseTargets({taggedFifos.front(), *renamed},
                  "fifo-discipline-composed-feedback");
}

} // namespace loom::dse::joint_test

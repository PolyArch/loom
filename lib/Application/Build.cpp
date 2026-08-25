#include "Application/Build.h"
#include "Application/BuildDiagnostics.h"
#include "BuildInternal.h"
#include "ExecutionGlue.h"

#include "Common/ArtifactLocalReference.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/MappingDebugLog.h"
#include "DSE/ProductionOwners.h"
#include "DSE/ResolvedConfigView.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Runtime/Gem5DispatchABI.h"
#include "Simulator/SpatialInvocation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <variant>

namespace loom::application {
namespace build_detail {

std::uint64_t elapsedNanoseconds(MonotonicClock::time_point begin) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             MonotonicClock::now() - begin)
      .count();
}

void emitElapsed(ApplicationBuildOperation operation,
                 MonotonicClock::time_point begin,
                 std::uint64_t deterministicWork) {
  emitApplicationBuildOperationStatistics(
      {operation, elapsedNanoseconds(begin), deterministicWork});
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_build_invalid: " + message);
}

/// Derive the same semantic DSE closure that owns Mapping manifests.  The
/// pre-Mapping planner may stop before a PlanExecutor occurrence exists, but
/// its pair-level decision still needs an exact, reproducible run-key join.
llvm::Expected<std::optional<std::array<std::uint8_t, 32>>>
derivePreMappingInvocationRunKey(
    const std::optional<ArtifactRootReference> &sourceProgram,
    const std::optional<ArtifactRootReference> &fabric,
    const std::optional<ArtifactRootReference> &workload,
    const std::optional<ArtifactRootReference> &runtimeInput,
    const std::optional<ArtifactRootReference> &edaPredictionModelWeight,
    const ResolvedConfig &config, const ArtifactStore &artifacts) {
  if (!sourceProgram || !fabric || !workload || !runtimeInput)
    return std::optional<std::array<std::uint8_t, 32>>{};
  auto producer = dse::DseProducerSemanticBuildIdentity::get(
      applicationBuildProducerIdentity);
  if (!producer)
    return producer.takeError();
  std::vector<ArtifactRootReference> inputs = {*sourceProgram, *fabric,
                                               *workload, *runtimeInput};
  if (edaPredictionModelWeight)
    inputs.push_back(*edaPredictionModelWeight);
  for (const ArtifactRootReference &input : inputs) {
    auto stored = artifacts.get(input);
    if (!stored)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "pre_mapping_input_unavailable: schema='" + input.schemaIdentity +
              "' version=" + llvm::Twine(input.schemaVersion.major) + "." +
              llvm::Twine(input.schemaVersion.minor) +
              "' identity=" + formatArtifactIdentityHex(input.artifact) + ": " +
              llvm::toString(stored.takeError()));
  }
  auto closure = dse::DseRunClosure::get(std::move(*producer), inputs, config,
                                         {}, artifacts);
  if (!closure)
    return closure.takeError();
  return std::optional<std::array<std::uint8_t, 32>>{closure->runKey().bytes()};
}

struct SourceSimulationInputs final {
  sim::CanonicalSimulationWorkload workload;
  sim::CanonicalSimulationRuntimeInput runtimeInput;
};

llvm::Expected<std::optional<std::vector<pnr::SystemBindingPartitionIntent>>>
deriveSystemBindingPartitionIntent(const dse::ResourceTimeScheduleHint &hint) {
  std::map<std::uint64_t, pnr::SystemBindingPartitionIntent> byRoot;
  for (const dse::ResourceTimeHintState &state : hint.states)
    for (const dse::ResourceTimeHintAllocation &allocation : state.active) {
      if (allocation.resourceUnits.size() != 1 ||
          allocation.resourceUnits.front() == 0)
        return invalid("resource-time allocation has no scalar System "
                       "partition count");
      auto [position, inserted] = byRoot.try_emplace(
          allocation.region.entity.value(),
          pnr::SystemBindingPartitionIntent{allocation.region,
                                            allocation.resourceUnits.front()});
      if (!inserted) {
        if (position->second.root != allocation.region)
          return invalid("resource-time partition intent crosses Dataflow "
                         "owners");
        if (position->second.partitionCount != allocation.resourceUnits.front())
          return std::nullopt;
      }
    }
  if (byRoot.empty())
    return invalid("resource-time schedule has no System partition intent");
  std::vector<pnr::SystemBindingPartitionIntent> result;
  result.reserve(byRoot.size());
  for (auto &[ordinal, partition] : byRoot) {
    (void)ordinal;
    result.push_back(std::move(partition));
  }
  return std::optional<std::vector<pnr::SystemBindingPartitionIntent>>(
      std::move(result));
}

llvm::Expected<SourceSimulationInputs>
makeSourceSimulationInputs(const frontend::StructuredProgramCandidate &program,
                           ApplicationSourceInvocation invocation) {
  if (invocation.entrySymbol.empty())
    return invalid("source invocation requires an ABI entry symbol");
  auto entries = frontend::resolveDefinedLlvmCallables(
      program, {llvm::StringRef(invocation.entrySymbol)});
  if (!entries)
    return entries.takeError();
  if (entries->size() != 1)
    return invalid("source invocation entry does not resolve uniquely");

  sim::StructuredProgramSimulationWorkload workloadDraft{entries->front()};
  workloadDraft.argumentPlan = std::move(invocation.argumentPlan);
  workloadDraft.observableContract.returnValue = invocation.observeReturnValue;
  for (const ApplicationPointerMemoryObservable &observable :
       invocation.memoryObservables) {
    workloadDraft.observableContract.memories.push_back(
        {sim::EntryPointerArgumentTarget{observable.argumentOrdinal},
         observable.form});
  }

  auto view = program.view();
  if (!view)
    return view.takeError();
  auto workload = sim::finalizeSimulationWorkload(workloadDraft, *view);
  if (!workload)
    return workload.takeError();

  sim::StructuredProgramSimulationRuntimeInputDraft runtimeDraft{
      workload->identity()};
  runtimeDraft.runtimeValues = std::move(invocation.runtimeValues);
  runtimeDraft.memoryObjects = std::move(invocation.memoryObjects);
  runtimeDraft.pointerBindings = std::move(invocation.pointerBindings);
  auto runtimeInput =
      sim::finalizeSimulationRuntimeInput(runtimeDraft, *workload, *view);
  if (!runtimeInput)
    return runtimeInput.takeError();
  return SourceSimulationInputs{std::move(*workload), std::move(*runtimeInput)};
}

llvm::Expected<std::variant<std::vector<ArtifactRootReference>,
                            UnsupportedApplicationBuild>>
publishApplicationWorkloads(
    const frontend::PublishedPreMappingCompilation &published,
    const dataflow::CanonicalDataflowArtifact &canonical,
    llvm::StringRef entrySymbol, const ArtifactStore &artifacts) {
  auto view = canonical.view();
  if (!view)
    return view.takeError();
  auto roots =
      view->projectRootThreadLaunchesReachableFromAbiEntry(entrySymbol);
  if (!roots)
    return roots.takeError();

  std::vector<ArtifactRootReference> workloads;
  for (dataflow::RootThreadLaunchRef root : *roots) {
    auto invocationPaths =
        view->projectRootThreadInvocationPathsFromAbiEntry(entrySymbol, root);
    if (!invocationPaths)
      return invocationPaths.takeError();
    if (llvm::any_of(*invocationPaths,
                     [](const auto &path) { return path.calls.empty(); }))
      return std::variant<std::vector<ArtifactRootReference>,
                          UnsupportedApplicationBuild>{
          UnsupportedApplicationBuild{
              ApplicationBuildUnsupportedKind::DirectInvocationBoundary,
              published.canonicalDataflow, root}};
    llvm::Error workloadError = llvm::Error::success();
    bool unsupportedCoordinates = false;
    view->forEachRootedGraphLaunch([&](dataflow::RootedGraphLaunchRef launch) {
      if (workloadError || unsupportedCoordinates ||
          launch.rootThreadLaunch != root)
        return;
      auto coordinates = view->enumerateStaticDenseCoordinates(
          launch, runtime::gem5MaximumDynamicSpatialInvocations, entrySymbol);
      if (!coordinates) {
        workloadError = coordinates.takeError();
        return;
      }
      if (!*coordinates) {
        unsupportedCoordinates = true;
        return;
      }
      auto shapes = sim::projectSpatialSimulationBoundaryShapes(*view, launch);
      if (!shapes) {
        workloadError = shapes.takeError();
        return;
      }
      auto writableRoots =
          sim::projectSpatialInvocationWritableMemoryRoots(*view, launch);
      if (!writableRoots) {
        workloadError = writableRoots.takeError();
        return;
      }
      for (const std::vector<std::uint64_t> &point : **coordinates) {
        sim::SpatialSimulationWorkload workloadDraft{launch};
        workloadDraft.denseCoordinates = point;
        workloadDraft.valueInputPlan.assign(shapes->valueInputs.size(),
                                            sim::RuntimeValueInput{});
        workloadDraft.observableContract.valueResults.resize(
            shapes->valueResults.size());
        std::iota(workloadDraft.observableContract.valueResults.begin(),
                  workloadDraft.observableContract.valueResults.end(), 0);
        for (dataflow::LogicalMemoryRootRef memory : *writableRoots)
          workloadDraft.observableContract.memories.push_back(
              {dataflow::LogicalMemoryRootOrViewRef{memory},
               sim::MemoryObservationForm::DiffFromRuntimeInput});
        auto workload = sim::finalizeSimulationWorkload(workloadDraft, *view);
        if (!workload) {
          workloadError = workload.takeError();
          return;
        }
        auto reference = sim::publishSimulationWorkload(*workload, artifacts);
        if (!reference) {
          workloadError = reference.takeError();
          return;
        }
        workloads.push_back(std::move(*reference));
      }
    });
    if (workloadError)
      return std::move(workloadError);
    if (unsupportedCoordinates)
      return std::variant<std::vector<ArtifactRootReference>,
                          UnsupportedApplicationBuild>{
          UnsupportedApplicationBuild{
              ApplicationBuildUnsupportedKind::RootCoordinates,
              published.canonicalDataflow, root}};
  }
  llvm::sort(workloads, artifactRootReferenceLess);
  workloads.erase(std::unique(workloads.begin(), workloads.end()),
                  workloads.end());
  if (workloads.empty())
    return invalid("source entry reaches no Spatial workload");
  return std::variant<std::vector<ArtifactRootReference>,
                      UnsupportedApplicationBuild>{std::move(workloads)};
}

} // namespace build_detail
using build_detail::ApplicationBuildOperationTimer;
using build_detail::derivePreMappingInvocationRunKey;
using build_detail::deriveSystemBindingPartitionIntent;
using build_detail::elapsedNanoseconds;
using build_detail::findResourceTimeScheduleHint;
using build_detail::invalid;
using build_detail::makePreAdmissionFailurePairDecision;
using build_detail::makePreparationPairDecision;
using build_detail::makeSourceSimulationInputs;
using build_detail::mapIncompleteReasonToPairDisposition;
using build_detail::MonotonicClock;
using build_detail::publishApplicationWorkloads;

namespace {
llvm::Expected<ApplicationBuildPreparationOutcome> prepareApplicationBuildImpl(
    const llvm::Module &finalLinkedModule, ApplicationBuildRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::ApplicationPreparation);
  if (llvm::Error error = dse::registerProductionDseOwners())
    return std::move(error);
  std::optional<ArtifactRootReference> edaPredictionModelWeight;
  if (request.edaPredictionModelWeight) {
    auto imported = evaluation::models::importEdaPredictionModelWeight(
        *request.edaPredictionModelWeight, artifacts, blobs);
    if (!imported)
      return imported.takeError();
    edaPredictionModelWeight = imported->reference();
  } else if (!request.fpaOperatingConditions.empty()) {
    return invalid("FPA operating conditions require a frozen model weight");
  }
  auto system = fabric::importEntireFabricRoot(request.system, artifacts);
  if (!system)
    return system.takeError();
  auto systemView = fabric::requireSystemRoot(system->view());
  if (!systemView)
    return systemView.takeError();

  auto source = frontend::raiseLlvmModuleToStructured(
      llvm::CloneModule(finalLinkedModule), *system,
      request.compilationOptions.raising);
  if (!source)
    return source.takeError();
  if (!request.operatorProtocolSymbols.empty()) {
    if (!request.preMappingOptions.ownership.protocolCallableRoots.empty())
      return invalid("operator protocol has two competing declarations");
    llvm::SmallVector<llvm::StringRef> symbols;
    symbols.reserve(request.operatorProtocolSymbols.size());
    for (const std::string &symbol : request.operatorProtocolSymbols)
      symbols.push_back(symbol);
    auto roots = frontend::resolveDefinedLlvmCallables(
        source->structuredProgram, symbols);
    if (!roots)
      return roots.takeError();
    request.preMappingOptions.ownership.protocolCallableRoots =
        std::move(*roots);
  }
  auto sourceInputs = makeSourceSimulationInputs(source->structuredProgram,
                                                 request.sourceInvocation);
  if (!sourceInputs)
    return sourceInputs.takeError();

  auto preMapping = dse::exploreStructuredCompilationToPreMapping(
      std::move(*source), sourceInputs->workload, sourceInputs->runtimeInput,
      *system, request.resolvedConfig, request.preMappingOptions, artifacts,
      blobs);
  if (!preMapping)
    return preMapping.takeError();
  const auto preMappingRunKeyFor = [&](const auto &value)
      -> llvm::Expected<std::optional<std::array<std::uint8_t, 32>>> {
    return derivePreMappingInvocationRunKey(
        value.sourceProgram, value.fabric, value.workload, value.runtimeInput,
        edaPredictionModelWeight, request.resolvedConfig, artifacts);
  };
  if (auto *incomplete =
          std::get_if<dse::IncompletePreMappingExploration>(&*preMapping)) {
    emitApplicationPreMappingIncompleteDiagnostics(*incomplete);
    if (incomplete->checkpoint) {
      const dse::PreMappingCheckpoint &checkpoint = *incomplete->checkpoint;
      auto invocationRunKey = derivePreMappingInvocationRunKey(
          checkpoint.sourceProgram, checkpoint.fabric, checkpoint.workload,
          checkpoint.runtimeInput, edaPredictionModelWeight,
          request.resolvedConfig, artifacts);
      if (!invocationRunKey)
        return invocationRunKey.takeError();
      auto decision = makePreparationPairDecision(
          checkpoint.sourceProgram, checkpoint.fabric, checkpoint.workload,
          checkpoint.runtimeInput, checkpoint.candidateInventory,
          mapIncompleteReasonToPairDisposition(incomplete->reason),
          dse::toString(incomplete->reason), incomplete->sourceHostOnlyWork,
          *invocationRunKey, false, request.portfolioInput);
      emitApplicationPairDecisionDiagnostics(decision);
    } else {
      auto decision = makePreparationPairDecision(
          std::nullopt, std::nullopt, std::nullopt, std::nullopt, {},
          mapIncompleteReasonToPairDisposition(incomplete->reason),
          dse::toString(incomplete->reason), incomplete->sourceHostOnlyWork,
          std::nullopt, true, request.portfolioInput);
      emitApplicationPairDecisionDiagnostics(decision);
    }
    return ApplicationBuildPreparationOutcome{std::move(*incomplete)};
  }
  if (auto *noFeasible =
          std::get_if<dse::CompletedPreMappingNoFeasibleCandidate>(
              &*preMapping)) {
    auto invocationRunKey = preMappingRunKeyFor(*noFeasible);
    if (!invocationRunKey)
      return invocationRunKey.takeError();
    auto decision = makePreparationPairDecision(
        noFeasible->sourceProgram, noFeasible->fabric, noFeasible->workload,
        noFeasible->runtimeInput, noFeasible->candidateInventory,
        noFeasible->completeness.exactComplete()
            ? ApplicationPairDecisionDisposition::NoPromisingCandidate
            : ApplicationPairDecisionDisposition::BudgetExhausted,
        noFeasible->completeness.exactComplete()
            ? "bounded front-end retained no candidate"
            : "front-end terminated before a complete candidate-domain proof",
        noFeasible->sourceHostOnlyWork, *invocationRunKey, false,
        request.portfolioInput);
    emitApplicationPairDecisionDiagnostics(decision);
    return ApplicationBuildPreparationOutcome{std::move(*noFeasible)};
  }

  auto completed =
      std::get<dse::CompletedPreMappingSelection>(std::move(*preMapping));
  auto completedInvocationRunKey = derivePreMappingInvocationRunKey(
      completed.sourceProgram, completed.fabric, completed.workload,
      completed.runtimeInput, edaPredictionModelWeight, request.resolvedConfig,
      artifacts);
  if (!completedInvocationRunKey)
    return completedInvocationRunKey.takeError();
  if (completed.selected.empty())
    return invalid("completed pre-Mapping selection is empty");
  for (std::size_t index = 0; index != completed.selected.size(); ++index)
    if (completed.selected[index].preferenceRank != index)
      return invalid("pre-Mapping software preference ranks are not dense");
  if (completed.selected.size() > request.jointPolicy.maximumSoftwareFrontier())
    return invalid("pre-Mapping software frontier exceeds its joint bound");
  if (completed.selected.size() > request.jointPolicy.maximumPairEvaluations())
    return invalid("pre-Mapping alternatives exceed the pair-evaluation "
                   "bound");

  struct PendingResourceTimeCandidate final {
    dse::SelectedPreMappingCompilation compilation;
    std::size_t planningRecordOrdinal = 0;
    ComponentViewDigest candidateIdentity;
    std::shared_ptr<const dse::ResourceTimeDataflowProjection> projection;
    std::uint64_t inputPreferenceRank = 0;
  };

  // The projection-only pass is deliberately limited to the already
  // materialized Canonical Dataflow view and the provider-owned resource-time
  // projection. Workload publication and joint-plan construction belong after
  // the bounded resource-time funnel so rejected estimates cannot trigger
  // Mapping work.
  std::vector<PendingResourceTimeCandidate> pendingCandidates;
  std::vector<dse::ResourceTimeMappingCandidateInput> resourceTimeInputs;
  pendingCandidates.reserve(completed.selected.size());
  resourceTimeInputs.reserve(completed.selected.size());
  std::uint64_t resourceTimeProjectionRequests = 0;
  std::uint64_t resourceTimeProjectionElapsedNanoseconds = 0;
  std::uint64_t resourceTimeProjectionCacheHits = 0;
  std::uint64_t resourceTimeProjectionCacheMisses = 0;
  std::uint64_t resourceTimeProjectionCacheCapacityBypasses = 0;
  std::uint64_t resourceTimeProjectionCacheRetainedBytes = 0;
  std::map<std::string,
           std::pair<std::shared_ptr<const dse::ResourceTimeDataflowProjection>,
                     std::uint64_t>>
      resourceTimeProjectionCache;
  auto resourceTimeModelSnapshot =
      dse::resourceTimeAnalyticModelSnapshotDigest();
  if (!resourceTimeModelSnapshot)
    return resourceTimeModelSnapshot.takeError();
  auto resourceTimeConfig =
      dse::projectResolvedDseConfigView(request.resolvedConfig);
  if (!resourceTimeConfig)
    return resourceTimeConfig.takeError();
  auto alternativePolicy = dse::JointDesignPolicy::get(
      1, 1, 1, request.jointPolicy.maximumTechMappingsPerModule(),
      request.jointPolicy.maximumSpatialMappingsPerPair());
  if (!alternativePolicy)
    return alternativePolicy.takeError();
  for (dse::SelectedPreMappingCompilation &selected : completed.selected) {
    if (!selected.planningRecordOrdinal ||
        *selected.planningRecordOrdinal >= completed.candidateInventory.size())
      return invalid("selected software has no exact planning record");
    const std::size_t planningRecordOrdinal = *selected.planningRecordOrdinal;
    const dse::PreMappingCandidatePlanningRecord &planningRecord =
        completed.candidateInventory[planningRecordOrdinal];
    if (!planningRecord.structuredProgram)
      return invalid("selected software has no Structured lineage root");
    auto candidateIdentity = dse::computePreMappingCandidateIdentity(
        planningRecord, completed.sourceProgram, completed.fabric,
        completed.workload, completed.runtimeInput,
        completed.frontierPolicyDigest);
    if (!candidateIdentity)
      return candidateIdentity.takeError();
    if (!planningRecord.candidateIdentity ||
        *planningRecord.candidateIdentity != *candidateIdentity)
      return invalid("pre-Mapping candidate identity failed its application "
                     "join validation");
    if (!planningRecord.canonicalDataflow ||
        planningRecord.canonicalDataflow->artifact !=
            selected.compilation.canonicalDataflow.identity())
      return invalid("selected software and planning Dataflow disagree");
    auto dataflowView = selected.compilation.canonicalDataflow.view();
    if (!dataflowView)
      return dataflowView.takeError();
    const ArtifactRootReference dataflow{
        dataflow::canonicalDataflowSchema.identity.str(),
        dataflow::canonicalDataflowSchema.version,
        selected.compilation.canonicalDataflow.identity()};
    dse::ResourceTimeInvocationKey invocation{
        *planningRecord.structuredProgram,
        dataflow,
        request.system,
        completed.workload,
        completed.runtimeInput,
        resourceTimeConfig->digest(),
        *resourceTimeModelSnapshot,
        request.sourceInvocation.entrySymbol,
        planningRecord.estimatedRuntimePicoseconds};
    auto projectionKey = dse::deriveResourceTimeProjectionCacheKey(invocation);
    if (!projectionKey)
      return projectionKey.takeError();
    const std::string projectionKeySpelling =
        formatComponentViewDigestHex(*projectionKey);
    ++resourceTimeProjectionRequests;
    std::shared_ptr<const dse::ResourceTimeDataflowProjection>
        resourceTimeProjection;
    auto cachedProjection =
        resourceTimeProjectionCache.find(projectionKeySpelling);
    if (cachedProjection != resourceTimeProjectionCache.end()) {
      resourceTimeProjection = cachedProjection->second.first;
      ++resourceTimeProjectionCacheHits;
    } else {
      ++resourceTimeProjectionCacheMisses;
      const MonotonicClock::time_point projectionBegin = MonotonicClock::now();
      auto computedProjection = dse::projectResourceTimeDataflow(
          *dataflowView, *systemView, request.sourceInvocation.entrySymbol,
          planningRecord.estimatedRuntimePicoseconds);
      const std::uint64_t projectionElapsed =
          elapsedNanoseconds(projectionBegin);
      resourceTimeProjectionElapsedNanoseconds =
          projectionElapsed > std::numeric_limits<std::uint64_t>::max() -
                                  resourceTimeProjectionElapsedNanoseconds
              ? std::numeric_limits<std::uint64_t>::max()
              : resourceTimeProjectionElapsedNanoseconds + projectionElapsed;
      if (!computedProjection)
        return computedProjection.takeError();
      resourceTimeProjection =
          std::make_shared<const dse::ResourceTimeDataflowProjection>(
              std::move(*computedProjection));
      const std::uint64_t retainedBytes =
          dse::resourceTimeProjectionRetainedBytes(*resourceTimeProjection);
      const bool fitsEntryLimit =
          resourceTimeProjectionCache.size() <
          request.resourceTimePolicy.maximumInvocationMemoEntries;
      const std::uint64_t availableBytes =
          request.resourceTimePolicy.maximumInvocationMemoBytes >=
                  resourceTimeProjectionCacheRetainedBytes
              ? request.resourceTimePolicy.maximumInvocationMemoBytes -
                    resourceTimeProjectionCacheRetainedBytes
              : 0;
      if (fitsEntryLimit && retainedBytes <= availableBytes) {
        resourceTimeProjectionCache.emplace(
            projectionKeySpelling,
            std::make_pair(resourceTimeProjection, retainedBytes));
        resourceTimeProjectionCacheRetainedBytes += retainedBytes;
      } else {
        ++resourceTimeProjectionCacheCapacityBypasses;
      }
    }
    if (request.resourceTimePolicy.availableResourceUnits.empty())
      request.resourceTimePolicy.availableResourceUnits =
          resourceTimeProjection->availableResourceUnits;
    else if (request.resourceTimePolicy.availableResourceUnits !=
             resourceTimeProjection->availableResourceUnits)
      return invalid("resource-time policy capacity disagrees with the exact "
                     "System projection");
    const auto maximumResourceBound =
        llvm::max_element(resourceTimeProjection->regionBounds,
                          [](const auto &lhs, const auto &rhs) {
                            return lhs.maximumUsefulResourceUnits <
                                   rhs.maximumUsefulResourceUnits;
                          });
    if (maximumResourceBound == resourceTimeProjection->regionBounds.end())
      return invalid("resource-time projection has no region bound");
    const std::uint64_t maximumUsefulResourceUnits =
        maximumResourceBound->maximumUsefulResourceUnits;
    const std::uint64_t inputPreferenceRank = selected.preferenceRank;
    pendingCandidates.push_back(
        {std::move(selected), planningRecordOrdinal, *candidateIdentity,
         std::move(resourceTimeProjection), inputPreferenceRank});
    const PendingResourceTimeCandidate &pending = pendingCandidates.back();
    resourceTimeInputs.push_back(
        {*candidateIdentity, pending.inputPreferenceRank,
         planningRecord.ownedProtocolRoots.size(),
         pending.projection->acceleratedGraphCount,
         pending.projection->acceleratedActorCount, maximumUsefulResourceUnits,
         std::move(invocation), pending.projection->resourceClasses,
         pending.projection->regions});
  }
  auto resourceTimeFunnel = dse::selectResourceTimeMappingFinalists(
      resourceTimeInputs, request.resourceTimePolicy,
      request.preMappingOptions.executionControl);
  if (!resourceTimeFunnel)
    return resourceTimeFunnel.takeError();
  resourceTimeFunnel->accounting.dataflowProjectionRequests =
      resourceTimeProjectionRequests;
  resourceTimeFunnel->accounting.dataflowProjectionCacheHits =
      resourceTimeProjectionCacheHits;
  resourceTimeFunnel->accounting.dataflowProjectionCacheMisses =
      resourceTimeProjectionCacheMisses;
  resourceTimeFunnel->accounting.dataflowProjectionCacheCapacityBypasses =
      resourceTimeProjectionCacheCapacityBypasses;
  resourceTimeFunnel->accounting.dataflowProjectionCacheEntries =
      resourceTimeProjectionCache.size();
  resourceTimeFunnel->accounting.dataflowProjectionCacheRetainedBytes =
      resourceTimeProjectionCacheRetainedBytes;
  resourceTimeFunnel->accounting.dataflowProjectionElapsedNanoseconds =
      resourceTimeProjectionElapsedNanoseconds;
  if (llvm::Error error = dse::validateResourceTimeMappingFunnelAccounting(
          resourceTimeFunnel->accounting))
    return std::move(error);
  const auto emitResourceTimeFunnelTerminal = [&](llvm::StringRef status) {
    const auto &accounting = resourceTimeFunnel->accounting;
    const auto counterObject = [](const dse::ResourceTimeWorkCounter &counter) {
      return llvm::json::Object{
          {"limit", counter.limit},
          {"planned", counter.planned},
          {"reserved", counter.reserved},
          {"consumed", counter.consumed},
          {"rejected", counter.rejected},
          {"cancelled", counter.cancelled},
          {"elapsed_nanoseconds", counter.elapsedNanoseconds}};
    };
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "resource_time_application_funnel";
          fields["status"] = status;
          llvm::json::Object frontierWork{
              {"source_projections",
               counterObject(accounting.frontierAccounting.sourceProjections)},
              {"actions", counterObject(accounting.frontierAccounting.actions)},
              {"states", counterObject(accounting.frontierAccounting.states)},
              {"estimates",
               counterObject(accounting.frontierAccounting.estimates)},
              {"finalists",
               counterObject(accounting.frontierAccounting.finalists)},
              {"state_memo_hits", accounting.frontierAccounting.stateMemoHits},
              {"state_memo_misses",
               accounting.frontierAccounting.stateMemoMisses},
              {"state_memo_pareto_insertions",
               accounting.frontierAccounting.stateMemoParetoInsertions},
              {"state_memo_dominated_states",
               accounting.frontierAccounting.stateMemoDominatedStates},
              {"state_memo_hit_capacity_rejections",
               accounting.frontierAccounting.stateMemoHitCapacityRejections},
              {"state_memo_miss_capacity_rejections",
               accounting.frontierAccounting.stateMemoMissCapacityRejections},
              {"states_pruned_by_beam",
               accounting.frontierAccounting.statesPrunedByBeam},
              {"terminal_hints_generated",
               accounting.frontierAccounting.terminalHintsGenerated},
              {"terminal_hints_retained",
               accounting.frontierAccounting.terminalHintsRetained},
              {"terminal_hints_pruned",
               accounting.frontierAccounting.terminalHintsPruned},
              {"incremental_lower_bound_updates",
               accounting.frontierAccounting.incrementalLowerBoundUpdates},
              {"maximum_retained_bytes",
               accounting.frontierAccounting.maximumRetainedBytes}};
          llvm::json::Object funnel{
              {"generated_candidates", accounting.generatedCandidates},
              {"screened_candidates", accounting.screenedCandidates},
              {"detailed_frontier_candidates",
               accounting.detailedFrontierCandidates},
              {"successive_halving_deferred_candidates",
               accounting.successiveHalvingDeferredCandidates},
              {"sound_gate_rejected_candidates",
               accounting.soundGateRejectedCandidates},
              {"estimated_candidates", accounting.estimatedCandidates},
              {"incomplete_candidates", accounting.incompleteCandidates},
              {"mapping_eligible_schedule_hints",
               accounting.mappingEligibleScheduleHints},
              {"screening_comparison_candidates",
               accounting.screeningComparisonCandidates},
              {"detailed_schedule_feasible_candidates",
               accounting.detailedScheduleFeasibleCandidates},
              {"screening_admissible_candidates",
               accounting.screeningAdmissibleCandidates},
              {"screening_detailed_feasible_intersection",
               accounting.screeningDetailedFeasibleIntersection},
              {"screening_detailed_best_rank_matches",
               accounting.screeningDetailedBestRankMatches},
              {"screening_out_of_domain_candidates",
               accounting.screeningOutOfDomainCandidates},
              {"maximum_screening_lower_bound_gap_picoseconds",
               accounting.maximumScreeningLowerBoundGapPicoseconds},
              {"mapping_finalists", accounting.mappingFinalists},
              {"dataflow_projection_requests",
               accounting.dataflowProjectionRequests},
              {"dataflow_projection_cache_hits",
               accounting.dataflowProjectionCacheHits},
              {"dataflow_projection_cache_misses",
               accounting.dataflowProjectionCacheMisses},
              {"dataflow_projection_cache_capacity_bypasses",
               accounting.dataflowProjectionCacheCapacityBypasses},
              {"dataflow_projection_cache_entries",
               accounting.dataflowProjectionCacheEntries},
              {"dataflow_projection_cache_retained_bytes",
               accounting.dataflowProjectionCacheRetainedBytes},
              {"dataflow_projection_elapsed_nanoseconds",
               accounting.dataflowProjectionElapsedNanoseconds},
              {"dataflow_materialized_candidates",
               accounting.dataflowMaterializedCandidates},
              {"mapping_plan_candidates", accounting.mappingPlanCandidates},
              {"unsupported_before_mapping_candidates",
               accounting.unsupportedBeforeMappingCandidates},
              {"unsupported_before_mapping_schedule_hints",
               accounting.unsupportedBeforeMappingScheduleHints},
              {"application_promotion_accounting_complete",
               accounting.applicationPromotionAccountingComplete},
              {"mapping_calls_deferred_by_model",
               accounting.mappingCallsDeferredByModel},
              {"mapping_plan_constructions_avoided_by_exact_memo",
               accounting.mappingPlanConstructionsAvoidedByExactMemo},
              {"mapping_calls_withheld_by_incomplete",
               accounting.mappingCallsWithheldByIncomplete},
              {"exact_invocation_memo_hits",
               accounting.exactInvocationMemoHits},
              {"exact_invocation_memo_misses",
               accounting.exactInvocationMemoMisses},
              {"exact_invocation_memo_single_flight_waits",
               accounting.exactInvocationMemoSingleFlightWaits},
              {"exact_invocation_memo_coalesced_uncached_results",
               accounting.exactInvocationMemoCoalescedUncachedResults},
              {"exact_invocation_memo_cancelled_waits",
               accounting.exactInvocationMemoCancelledWaits},
              {"exact_invocation_memo_capacity_bypasses",
               accounting.exactInvocationMemoCapacityBypasses},
              {"exact_invocation_memo_entries",
               accounting.exactInvocationMemoEntries},
              {"exact_invocation_memo_retained_bytes",
               accounting.exactInvocationMemoRetainedBytes},
              {"frontier_work", std::move(frontierWork)},
              {"elapsed_nanoseconds", accounting.elapsedNanoseconds},
              {"truncated", resourceTimeFunnel->truncated}};
          if (resourceTimeFunnel->incompleteReason)
            funnel["incomplete_reason"] =
                dse::resourceTimeFrontierIncompleteReasonSpelling(
                    *resourceTimeFunnel->incompleteReason);
          fields["resource_time_funnel"] = std::move(funnel);
        });
  };
  if (resourceTimeFunnel->incompleteReason ==
      dse::ResourceTimeFrontierIncompleteReason::CancelledOrTimeout)
    emitResourceTimeFunnelTerminal("cancelled_or_timeout");
  if (resourceTimeFunnel->incompleteReason ==
      dse::ResourceTimeFrontierIncompleteReason::CancelledOrTimeout) {
    auto decision = makePreparationPairDecision(
        completed.sourceProgram, completed.fabric, completed.workload,
        completed.runtimeInput, completed.candidateInventory,
        ApplicationPairDecisionDisposition::CancelledOrTimeout,
        "resource-time funnel cancelled or timed out",
        completed.sourceHostOnlyWork, *completedInvocationRunKey, false,
        request.portfolioInput);
    emitApplicationPairDecisionDiagnostics(decision);
    return ApplicationBuildPreparationOutcome{
        IncompleteApplicationResourceTimePlanning{
            *resourceTimeFunnel->incompleteReason,
            std::move(*resourceTimeFunnel),
            std::move(completed.candidateInventory), completed.sourceProgram,
            completed.fabric, completed.workload, completed.runtimeInput,
            completed.frontierPolicyDigest, completed.sourceHostOnlyWork}};
  }
  if (resourceTimeFunnel->finalists.empty())
    emitResourceTimeFunnelTerminal(resourceTimeFunnel->incompleteReason
                                       ? "incomplete"
                                       : "no_mapping_finalist");
  if (resourceTimeFunnel->finalists.empty() &&
      resourceTimeFunnel->incompleteReason) {
    const auto disposition =
        *resourceTimeFunnel->incompleteReason ==
                dse::ResourceTimeFrontierIncompleteReason::Unsupported
            ? ApplicationPairDecisionDisposition::UnsupportedSemantic
            : ApplicationPairDecisionDisposition::BudgetExhausted;
    auto decision = makePreparationPairDecision(
        completed.sourceProgram, completed.fabric, completed.workload,
        completed.runtimeInput, completed.candidateInventory, disposition,
        dse::resourceTimeFrontierIncompleteReasonSpelling(
            *resourceTimeFunnel->incompleteReason),
        completed.sourceHostOnlyWork, *completedInvocationRunKey, false,
        request.portfolioInput);
    emitApplicationPairDecisionDiagnostics(decision);
    return ApplicationBuildPreparationOutcome{
        IncompleteApplicationResourceTimePlanning{
            *resourceTimeFunnel->incompleteReason,
            std::move(*resourceTimeFunnel),
            std::move(completed.candidateInventory), completed.sourceProgram,
            completed.fabric, completed.workload, completed.runtimeInput,
            completed.frontierPolicyDigest, completed.sourceHostOnlyWork}};
  }
  if (resourceTimeFunnel->finalists.empty()) {
    const bool completeNoPromisingProof =
        completed.completeness.exactComplete() &&
        !resourceTimeFunnel->truncated;
    auto decision = makePreparationPairDecision(
        completed.sourceProgram, completed.fabric, completed.workload,
        completed.runtimeInput, completed.candidateInventory,
        completeNoPromisingProof
            ? ApplicationPairDecisionDisposition::NoPromisingCandidate
            : ApplicationPairDecisionDisposition::BudgetExhausted,
        completeNoPromisingProof
            ? "resource-time funnel retained no Mapping finalist"
            : "resource-time funnel ended without a complete candidate proof",
        completed.sourceHostOnlyWork, *completedInvocationRunKey, false,
        request.portfolioInput);
    emitApplicationPairDecisionDiagnostics(decision);
    return ApplicationBuildPreparationOutcome{
        dse::CompletedPreMappingNoFeasibleCandidate{
            std::move(completed.satisfiedEvidence),
            std::move(completed.planGenerateInvocations),
            completed.sourceProgram, completed.fabric, completed.workload,
            completed.runtimeInput, std::move(completed.candidateInventory),
            completed.completeness, completed.frontierPolicyDigest,
            completed.sourceHostOnlyWork}};
  }

  std::vector<PreparedApplicationSoftware> preparedSoftware;
  std::vector<PreparedApplicationMappingAlternative> mappingAlternatives;
  std::optional<UnsupportedApplicationBuild> firstUnsupported;
  preparedSoftware.reserve(resourceTimeFunnel->finalists.size());
  mappingAlternatives.reserve(resourceTimeFunnel->finalists.size());
  std::vector<ComponentViewDigest> promotedIdentities;
  promotedIdentities.reserve(resourceTimeFunnel->finalists.size());
  std::map<std::string, std::size_t> softwareByCandidate;
  std::set<std::string> unsupportedCandidates;
  std::map<std::string, std::uint64_t> finalistCountByCandidate;
  for (const dse::ResourceTimeMappingFinalist &finalist :
       resourceTimeFunnel->finalists)
    ++finalistCountByCandidate[formatComponentViewDigestHex(
        finalist.candidateIdentity)];
  for (const dse::ResourceTimeMappingFinalist &finalist :
       resourceTimeFunnel->finalists) {
    const ComponentViewDigest &identity = finalist.candidateIdentity;
    const std::string identitySpelling = formatComponentViewDigestHex(identity);
    if (unsupportedCandidates.count(identitySpelling) != 0)
      continue;
    auto pending = llvm::find_if(
        pendingCandidates, [&](const PendingResourceTimeCandidate &candidate) {
          return candidate.candidateIdentity == identity;
        });
    if (pending == pendingCandidates.end())
      return invalid("resource-time finalist has no application candidate");
    const auto evaluation = llvm::find_if(
        resourceTimeFunnel->evaluations, [&](const auto &candidate) {
          return candidate.candidateIdentity == identity;
        });
    if (evaluation == resourceTimeFunnel->evaluations.end())
      return invalid("resource-time finalist has no funnel evaluation");
    auto scheduleHint =
        findResourceTimeScheduleHint(*evaluation, finalist.scheduleHintDigest);
    if (!scheduleHint)
      return scheduleHint.takeError();
    auto partitions = deriveSystemBindingPartitionIntent(**scheduleHint);
    if (!partitions)
      return partitions.takeError();
    if (!*partitions) {
      ++resourceTimeFunnel->accounting.unsupportedBeforeMappingScheduleHints;
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
          mapping_debug::Event::MappingFailure,
          [&](llvm::json::Object &fields) {
            fields["failure_scope"] = "application_resource_time_preflight";
            fields["operation"] = "resource_time_mapping_transition";
            fields["disposition"] = "unsupported";
            fields["candidate_identity"] = identitySpelling;
            fields["schedule_hint_digest"] =
                formatComponentViewDigestHex(finalist.scheduleHintDigest);
          });
      continue;
    }

    std::size_t softwareOrdinal = 0;
    auto existingSoftware = softwareByCandidate.find(identitySpelling);
    if (existingSoftware == softwareByCandidate.end()) {
      auto published = frontend::publishPreMappingCompilation(
          pending->compilation.compilation, artifacts);
      if (!published)
        return published.takeError();
      auto workloads = publishApplicationWorkloads(
          *published, pending->compilation.compilation.canonicalDataflow,
          request.sourceInvocation.entrySymbol, artifacts);
      if (!workloads)
        return workloads.takeError();
      if (auto *unsupported =
              std::get_if<UnsupportedApplicationBuild>(&*workloads)) {
        if (!firstUnsupported)
          firstUnsupported = std::move(*unsupported);
        auto &record =
            completed.candidateInventory[pending->planningRecordOrdinal];
        record.disposition =
            dse::PreMappingCandidatePlanningDisposition::Unsupported;
        ++resourceTimeFunnel->accounting.unsupportedBeforeMappingCandidates;
        resourceTimeFunnel->accounting.unsupportedBeforeMappingScheduleHints +=
            finalistCountByCandidate[identitySpelling];
        unsupportedCandidates.insert(identitySpelling);
        continue;
      }
      auto roots =
          std::get<std::vector<ArtifactRootReference>>(std::move(*workloads));
      ++resourceTimeFunnel->accounting.dataflowMaterializedCandidates;
      if (pending->compilation.functionalReplay)
        ++resourceTimeFunnel->accounting.functionalReplayCandidates;
      // Deployment reconstructs this exact invocation plan again. Validate it
      // before any Tech/Spatial/System provider is dispatched so a candidate
      // with an inexact dynamic capture becomes a typed unsupported finalist,
      // rather than a late deployment failure after expensive Mapping work.
      auto invocationDataflow =
          pending->compilation.compilation.canonicalDataflow.view();
      if (!invocationDataflow)
        return invocationDataflow.takeError();
      auto invocationPreflight = detail::deriveApplicationSpatialInvocationPlan(
          *invocationDataflow, request.sourceInvocation.entrySymbol);
      if (!invocationPreflight) {
        const std::string diagnostic =
            llvm::toString(invocationPreflight.takeError());
        mapping_debug::emit(mapping_debug::Level::Summary,
                            mapping_debug::Stage::DataflowLowering,
                            mapping_debug::Event::MappingFailure,
                            [&](llvm::json::Object &fields) {
                              fields["failure_scope"] =
                                  "application_resource_time_preflight";
                              fields["operation"] =
                                  "resource_time_application_preflight";
                              fields["disposition"] = "unsupported";
                              fields["diagnostic"] = diagnostic;
                              fields["candidate_identity"] = identitySpelling;
                            });
        if (!firstUnsupported)
          firstUnsupported = UnsupportedApplicationBuild{
              ApplicationBuildUnsupportedKind::DynamicInvocationBoundary,
              published->canonicalDataflow,
              pending->projection->regions.front().region};
        auto &record =
            completed.candidateInventory[pending->planningRecordOrdinal];
        record.disposition =
            dse::PreMappingCandidatePlanningDisposition::Unsupported;
        ++resourceTimeFunnel->accounting.unsupportedBeforeMappingCandidates;
        resourceTimeFunnel->accounting.unsupportedBeforeMappingScheduleHints +=
            finalistCountByCandidate[identitySpelling];
        unsupportedCandidates.insert(identitySpelling);
        continue;
      }
      std::vector<sim::SourceBackedDfgReplayCaseReference> replayCases;
      if (pending->compilation.functionalReplay)
        replayCases = pending->compilation.functionalReplay->replayCases;
      const std::uint64_t firstRank = mappingAlternatives.size();
      preparedSoftware.push_back({firstRank, pending->planningRecordOrdinal,
                                  identity, std::move(*published),
                                  std::move(roots), std::move(replayCases)});
      softwareOrdinal = preparedSoftware.size() - 1;
      softwareByCandidate.emplace(identitySpelling, softwareOrdinal);
      promotedIdentities.push_back(identity);
      auto &record =
          completed.candidateInventory[pending->planningRecordOrdinal];
      record.disposition =
          dse::PreMappingCandidatePlanningDisposition::Retained;
      record.preferenceRank = firstRank;
    } else {
      softwareOrdinal = existingSoftware->second;
    }

    const PreparedApplicationSoftware &software =
        preparedSoftware[softwareOrdinal];
    // A schedule hint can differ in event timing while producing the exact
    // same static System partition intent. Such a hint is not a new Mapping
    // input: retain its provenance and verify it against the one real result.
    // Resource-time transitions with a changed partition remain separate;
    // this exact reuse must never be mistaken for remapping.
    const auto exactAlternative = llvm::find_if(
        mappingAlternatives,
        [&](const PreparedApplicationMappingAlternative &alternative) {
          return alternative.candidateIdentity == identity &&
                 alternative.dataflow.artifact ==
                     software.compilation.canonicalDataflow.artifact &&
                 alternative.plan.systemBindingPartitions == **partitions;
        });
    if (exactAlternative != mappingAlternatives.end()) {
      exactAlternative->equivalentScheduleHintDigests.push_back(
          finalist.scheduleHintDigest);
      ++resourceTimeFunnel->accounting
            .mappingPlanConstructionsAvoidedByExactMemo;
      mapping_debug::emit(
          mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
          mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
            fields["operation"] = "resource_time_exact_mapping_reuse";
            fields["candidate_identity"] = identitySpelling;
            fields["schedule_hint_digest"] =
                formatComponentViewDigestHex(finalist.scheduleHintDigest);
            fields["mapping_plan_rank"] = exactAlternative->preferenceRank;
          });
      continue;
    }
    auto mappingPlan = dse::buildJointDesignExplorationPlan(
        {{software.workloads}, {request.system}},
        request.physicalTimingProfiles, *alternativePolicy,
        request.resolvedConfig, artifacts, nullptr, **partitions);
    if (!mappingPlan)
      return mappingPlan.takeError();
    ++resourceTimeFunnel->accounting.mappingPlanCandidates;
    const std::uint64_t rank = mappingAlternatives.size();
    mappingAlternatives.push_back({rank,
                                   pending->planningRecordOrdinal,
                                   identity,
                                   finalist.scheduleHintDigest,
                                   {finalist.scheduleHintDigest},
                                   software.compilation.canonicalDataflow,
                                   pending->projection->regions,
                                   pending->projection->regionBounds,
                                   std::move(*mappingPlan)});
  }
  resourceTimeFunnel->accounting.applicationPromotionAccountingComplete = true;
  if (llvm::Error error = dse::validateResourceTimeMappingFunnelAccounting(
          resourceTimeFunnel->accounting))
    return std::move(error);
  if (mappingAlternatives.empty()) {
    emitResourceTimeFunnelTerminal("all_finalists_rejected_before_mapping");
    if (firstUnsupported) {
      auto decision = makePreparationPairDecision(
          completed.sourceProgram, completed.fabric, completed.workload,
          completed.runtimeInput, completed.candidateInventory,
          ApplicationPairDecisionDisposition::UnsupportedSemantic,
          "all retained finalists were rejected at the application boundary",
          completed.sourceHostOnlyWork, *completedInvocationRunKey, false,
          request.portfolioInput);
      emitApplicationPairDecisionDiagnostics(decision);
      return ApplicationBuildPreparationOutcome{std::move(*firstUnsupported)};
    }
    if (resourceTimeFunnel->accounting.unsupportedBeforeMappingScheduleHints !=
        0) {
      auto decision = makePreparationPairDecision(
          completed.sourceProgram, completed.fabric, completed.workload,
          completed.runtimeInput, completed.candidateInventory,
          ApplicationPairDecisionDisposition::UnsupportedSemantic,
          "resource-time finalists were unsupported before Mapping",
          completed.sourceHostOnlyWork, *completedInvocationRunKey, false,
          request.portfolioInput);
      emitApplicationPairDecisionDiagnostics(decision);
      return ApplicationBuildPreparationOutcome{
          IncompleteApplicationResourceTimePlanning{
              dse::ResourceTimeFrontierIncompleteReason::Unsupported,
              std::move(*resourceTimeFunnel),
              std::move(completed.candidateInventory), completed.sourceProgram,
              completed.fabric, completed.workload, completed.runtimeInput,
              completed.frontierPolicyDigest, completed.sourceHostOnlyWork}};
    }
    const bool completeNoPromisingProof =
        completed.completeness.exactComplete() &&
        !resourceTimeFunnel->truncated && !resourceTimeFunnel->incompleteReason;
    auto decision = makePreparationPairDecision(
        completed.sourceProgram, completed.fabric, completed.workload,
        completed.runtimeInput, completed.candidateInventory,
        completeNoPromisingProof
            ? ApplicationPairDecisionDisposition::NoPromisingCandidate
            : ApplicationPairDecisionDisposition::BudgetExhausted,
        completeNoPromisingProof
            ? "bounded resource-time funnel retained no Mapping finalist"
            : "resource-time funnel did not close its bounded candidate domain",
        completed.sourceHostOnlyWork, *completedInvocationRunKey, false,
        request.portfolioInput);
    emitApplicationPairDecisionDiagnostics(decision);
    return ApplicationBuildPreparationOutcome{
        dse::CompletedPreMappingNoFeasibleCandidate{
            std::move(completed.satisfiedEvidence),
            std::move(completed.planGenerateInvocations),
            completed.sourceProgram, completed.fabric, completed.workload,
            completed.runtimeInput, std::move(completed.candidateInventory),
            completed.completeness, completed.frontierPolicyDigest,
            completed.sourceHostOnlyWork}};
  }
  for (dse::PreMappingCandidatePlanningRecord &record :
       completed.candidateInventory) {
    if (!record.candidateIdentity ||
        llvm::is_contained(promotedIdentities, *record.candidateIdentity))
      continue;
    if (record.disposition ==
        dse::PreMappingCandidatePlanningDisposition::Retained) {
      record.disposition =
          dse::PreMappingCandidatePlanningDisposition::HeuristicPruned;
      record.preferenceRank.reset();
    }
  }
  PreparedApplicationBuild prepared{
      std::move(request.sourceInvocation),
      request.jointPolicy,
      std::move(preparedSoftware),
      std::move(completed.satisfiedEvidence),
      std::move(completed.planGenerateInvocations),
      std::move(completed.protocolDependencyProjection),
      std::move(completed.candidateInventory),
      completed.frontierPolicy,
      completed.eligibleCoordinateCount,
      completed.coordinateFrontierTruncated,
      std::move(completed.frontierAccounting),
      completed.evaluationTiming,
      completed.sharedEvaluationStatistics,
      completed.evaluationCacheStatistics,
      std::move(completed.retainedPlanIncompleteness),
      std::move(mappingAlternatives),
      request.resourceTimePolicy,
      std::move(*resourceTimeFunnel),
      completed.requestedPlannerMode,
      completed.resolvedPlannerMode,
      completed.completeness,
      std::move(completed.shadowRecall),
      completed.sourceHostOnlyWork,
      completed.sourceProgram,
      completed.fabric,
      completed.workload,
      completed.runtimeInput,
      completed.frontierPolicyDigest,
      systemView->artifact().accCoreOccurrences().size(),
      std::nullopt,
      std::move(request.portfolioInput),
      std::move(edaPredictionModelWeight),
      std::move(request.fpaOperatingConditions)};
  prepared.preMappingInvocationRunKey = *completedInvocationRunKey;
  emitApplicationPlanningDiagnostics(prepared);
  return ApplicationBuildPreparationOutcome{std::move(prepared)};
}

} // namespace

llvm::Expected<ApplicationBuildPreparationOutcome> prepareApplicationBuild(
    const llvm::Module &finalLinkedModule, ApplicationBuildRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  // Before source/workload/runtime roots exist there is no sound
  // InvocationManifest run-key to publish.  Close that narrow boundary with
  // an explicit owner-level status instead of emitting a bare Missing join.
  const ArtifactRootReference requestedSystem = request.system;
  std::optional<SelectedApplicationInput> portfolioInput =
      request.portfolioInput;
  const ExecutionControlView executionControl =
      request.preMappingOptions.executionControl;
  auto outcome = prepareApplicationBuildImpl(
      finalLinkedModule, std::move(request), artifacts, blobs);
  if (outcome)
    return outcome;

  llvm::Error error = outcome.takeError();
  const std::string diagnostic = llvm::toString(std::move(error));
  const ApplicationPairDecisionDisposition disposition =
      executionControl.stopRequested()
          ? ApplicationPairDecisionDisposition::CancelledOrTimeout
      : llvm::StringRef(diagnostic).contains("unsupported")
          ? ApplicationPairDecisionDisposition::UnsupportedSemantic
          : ApplicationPairDecisionDisposition::ImplementationFailure;
  ApplicationPairDecisionRecord decision = makePreAdmissionFailurePairDecision(
      std::move(portfolioInput), requestedSystem, disposition, diagnostic);
  emitApplicationPairDecisionDiagnostics(decision);
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument), diagnostic);
}
} // namespace loom::application

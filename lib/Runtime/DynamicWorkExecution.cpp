#include "Runtime/DynamicWorkExecution.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Mapping/Artifact/SystemMappingClosureProjection.h"
#include "Simulator/SimulationArtifacts.h"
#include "Simulator/SpatialObservationComparison.h"

#include "llvm/ADT/STLExtras.h"

#include <map>
#include <numeric>
#include <tuple>

namespace loom::runtime {

namespace {

llvm::Expected<sim::RetirementEffect>
retireDynamicWorkAssignment(sim::DynamicWorkScheduler &scheduler,
                            sim::DynamicWorkAssignment &&assignment,
                            DynamicWorkExecutionAction action) {
  if (action == DynamicWorkExecutionAction::Complete)
    return scheduler.complete(std::move(assignment));

  auto request = scheduler.requestCancellation(assignment.id());
  if (!request)
    return request.takeError();
  if (request->kind() != sim::DynamicWorkCancellationKind::RequestedActive)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dynamic_work_execution_invalid: active cancellation was not "
        "delivered");
  return scheduler.cancel(std::move(assignment));
}

llvm::Error verifyDynamicWorkJoin(const sim::DynamicWorkScheduler &scheduler,
                                  sim::RetirementEffect effect) {
  if (effect == sim::RetirementEffect::DomainCompleted &&
      scheduler.completed() && scheduler.activeCount() == 0)
    return llvm::Error::success();
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "dynamic_work_execution_invalid: responsibility join observed an "
      "incomplete responsibility domain");
}

llvm::Error cancelDynamicWorkDomain(
    sim::DynamicWorkScheduler &scheduler,
    sim::DynamicWorkAssignment &&activeAssignment) {
  auto active = retireDynamicWorkAssignment(
      scheduler, std::move(activeAssignment),
      DynamicWorkExecutionAction::RequestCancellation);
  if (!active)
    return active.takeError();
  sim::RetirementEffect finalEffect = *active;
  auto queued = scheduler.cancelQueued();
  if (!queued)
    return queued.takeError();
  if (*queued)
    finalEffect = **queued;
  return verifyDynamicWorkJoin(scheduler, finalEffect);
}

llvm::StringRef spelling(sim::SpatialExecutionSessionState state) {
  switch (state) {
  case sim::SpatialExecutionSessionState::Runnable:
    return "runnable";
  case sim::SpatialExecutionSessionState::Retired:
    return "retired";
  case sim::SpatialExecutionSessionState::Halted:
    return "halted";
  case sim::SpatialExecutionSessionState::StoppedByLimit:
    return "stopped_by_limit";
  case sim::SpatialExecutionSessionState::Failed:
    return "failed";
  }
  llvm_unreachable("unknown Spatial execution state");
}

llvm::Error unsupported(DynamicWorkExecutionUnsupportedReason reason,
                        const llvm::Twine &message) {
  return llvm::make_error<DynamicWorkExecutionUnsupported>(reason,
                                                           message.str());
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

std::vector<std::uint64_t> workItemPath(sim::WorkItemId item) {
  std::vector<std::uint64_t> path;
  while (true) {
    path.push_back(item.ordinal());
    auto parent = item.parent();
    if (!parent)
      break;
    item = std::move(*parent);
  }
  std::reverse(path.begin(), path.end());
  return path;
}

bool equivalentReplay(llvm::ArrayRef<sim::DynamicWorkScheduleAction> lhs,
                      llvm::ArrayRef<sim::DynamicWorkScheduleAction> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (std::size_t index = 0; index != lhs.size(); ++index)
    if (lhs[index].kind != rhs[index].kind ||
        lhs[index].sourceWorker != rhs[index].sourceWorker ||
        lhs[index].targetWorker != rhs[index].targetWorker ||
        workItemPath(lhs[index].item) != workItemPath(rhs[index].item))
      return false;
  return true;
}

bool equivalentDispatch(const DynamicWorkExecutionResult &lhs,
                        const DynamicWorkExecutionResult &rhs) {
  return lhs.dispatchOccurrence != rhs.dispatchOccurrence &&
         lhs.joinEffect == rhs.joinEffect && lhs.cancelled == rhs.cancelled &&
         lhs.processedItemCount == rhs.processedItemCount &&
         lhs.publishedChildCount == rhs.publishedChildCount &&
         lhs.completedItemCount == rhs.completedItemCount &&
         lhs.cancelledItemCount == rhs.cancelledItemCount &&
         equivalentReplay(lhs.replay, rhs.replay);
}

bool equivalentServicePlans(
    llvm::ArrayRef<DynamicWorkSelectedServicePlan> lhs,
    llvm::ArrayRef<DynamicWorkSelectedServicePlan> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (std::size_t index = 0; index != lhs.size(); ++index)
    if (!(lhs[index].obligation == rhs[index].obligation) ||
        !(lhs[index].selection == rhs[index].selection) ||
        lhs[index].planOrdinal != rhs[index].planOrdinal)
      return false;
  return true;
}

bool equivalentCounters(const sim::CgraSimulationCounters &lhs,
                        const sim::CgraSimulationCounters &rhs) {
  return std::tie(
             lhs.eventFrameCount, lhs.actorCommitCount,
             lhs.actorRetirementCount, lhs.tokenPublicationCount,
             lhs.memoryLinearizationCount, lhs.physicalRequestCount,
             lhs.physicalGrantCount, lhs.physicalRetirementCount,
             lhs.emptyEventFrameCount, lhs.computeSourceFrameCount,
             lhs.memorySourceFrameCount, lhs.transportSourceFrameCount,
             lhs.physicalSourceFrameCount, lhs.maximumReferenceCycleNumerator,
             lhs.maximumEventDelta, lhs.physicalGrantWaitCycleSum,
             lhs.physicalGrantWaitCycleMax, lhs.physicalActionLifetimeCycleSum,
             lhs.physicalActionLifetimeCycleMax,
             lhs.physicalGrantedLifetimeCycleSum,
             lhs.physicalGrantedLifetimeCycleMax,
             lhs.physicalGrantSameCycleCount, lhs.physicalGrantDelayedCount,
             lhs.nonIntegralTimingObservationCount) ==
         std::tie(
             rhs.eventFrameCount, rhs.actorCommitCount,
             rhs.actorRetirementCount, rhs.tokenPublicationCount,
             rhs.memoryLinearizationCount, rhs.physicalRequestCount,
             rhs.physicalGrantCount, rhs.physicalRetirementCount,
             rhs.emptyEventFrameCount, rhs.computeSourceFrameCount,
             rhs.memorySourceFrameCount, rhs.transportSourceFrameCount,
             rhs.physicalSourceFrameCount, rhs.maximumReferenceCycleNumerator,
             rhs.maximumEventDelta, rhs.physicalGrantWaitCycleSum,
             rhs.physicalGrantWaitCycleMax, rhs.physicalActionLifetimeCycleSum,
             rhs.physicalActionLifetimeCycleMax,
             rhs.physicalGrantedLifetimeCycleSum,
             rhs.physicalGrantedLifetimeCycleMax,
             rhs.physicalGrantSameCycleCount, rhs.physicalGrantDelayedCount,
             rhs.nonIntegralTimingObservationCount);
}

bool equivalentCompletedExecution(const DynamicWorkCgraExecutionResult &lhs,
                                  const DynamicWorkCgraExecutionResult &rhs) {
  return equivalentDispatch(lhs.dispatch, rhs.dispatch) &&
         lhs.instructionContext == rhs.instructionContext &&
         lhs.spatialContext.spatialMapping ==
             rhs.spatialContext.spatialMapping &&
         lhs.spatialContext.context == rhs.spatialContext.context &&
         equivalentServicePlans(lhs.servicePlans, rhs.servicePlans) &&
         sim::haveExactlyEqualSpatialFunctionalObservations(
             lhs.execution.observations, rhs.execution.observations) &&
         equivalentCounters(lhs.execution.counters, rhs.execution.counters) &&
         sim::compareSpatialEventCoordinates(
             lhs.execution.progress.launchAccepted,
             rhs.execution.progress.launchAccepted) == 0 &&
         lhs.execution.progress.graphRetirementVisible.has_value() ==
             rhs.execution.progress.graphRetirementVisible.has_value() &&
         (!lhs.execution.progress.graphRetirementVisible ||
          sim::compareSpatialEventCoordinates(
              *lhs.execution.progress.graphRetirementVisible,
              *rhs.execution.progress.graphRetirementVisible) == 0) &&
         sim::compareSpatialEventCoordinates(
             lhs.execution.progress.terminalObserved,
             rhs.execution.progress.terminalObserved) == 0;
}

llvm::Expected<std::vector<DynamicWorkSelectedServicePlan>>
selectDynamicWorkServicePlans(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::SystemMappingClosureProjection &closure,
    dataflow::RootThreadLaunchRef root,
    const mapping::InstructionExecutionContextKey &instruction,
    const std::optional<mapping::SelectedSystemSpatialContext> &spatial,
    dataflow::DynamicWorkStableItemKey stableItem) {
  auto obligations = mapping::projectSystemServiceObligations(dataflow, {root});
  if (!obligations)
    return obligations.takeError();
  std::map<std::string, const mapping::SystemServiceObligationProjection *>
      relevant;
  for (const auto &obligation : *obligations) {
    auto key = mapping::encodeSystemServiceObligationKey(dataflow.identity(),
                                                         obligation.key);
    if (!key)
      return key.takeError();
    relevant.emplace(byteKey(*key), &obligation);
  }

  std::vector<DynamicWorkSelectedServicePlan> selected;
  const mapping::ExecutionContextKey instructionContext(instruction);
  for (const auto &realization : closure.serviceRealizations) {
    auto key = mapping::encodeSystemServiceObligationKey(dataflow.identity(),
                                                         realization.key);
    if (!key)
      return key.takeError();
    const auto found = relevant.find(byteKey(*key));
    if (found == relevant.end())
      continue;
    for (const auto &selection : realization.selections) {
      if (!mapping::systemServiceSelectionAnchorBelongsTo(selection.key.anchor,
                                                          *found->second))
        continue;
      const bool selectsInstruction =
          selection.key.context == instructionContext;
      const bool selectsSpatial =
          spatial && selection.key.context ==
                         mapping::ExecutionContextKey(spatial->context);
      if (!selectsInstruction && !selectsSpatial)
        continue;
      auto plan = mapping::selectSystemDynamicWorkServicePlanOrdinal(
          realization, selection.key.anchor, selection.key.context, stableItem);
      if (!plan)
        return plan.takeError();
      selected.push_back({realization.key, selection.key, *plan});
    }
  }
  return selected;
}

} // namespace

char DynamicWorkExecutionUnsupported::ID = 0;
char DynamicWorkCgraExecutionIncomplete::ID = 0;
char DynamicWorkExecutionIncomplete::ID = 0;

void DynamicWorkExecutionUnsupported::log(llvm::raw_ostream &stream) const {
  stream << "dynamic_work_execution_unsupported: " << message_;
}

std::error_code DynamicWorkExecutionUnsupported::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

void DynamicWorkCgraExecutionIncomplete::log(llvm::raw_ostream &stream) const {
  stream << "dynamic_work_cgra_execution_incomplete: selected graph is "
         << spelling(outcome_.state);
}

std::error_code DynamicWorkCgraExecutionIncomplete::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

void DynamicWorkExecutionIncomplete::log(llvm::raw_ostream &stream) const {
  switch (reason_) {
  case DynamicWorkExecutionIncompleteReason::QueueCapacity:
    stream << "dynamic_work_execution_incomplete: child publication exceeds "
              "the admitted queue capacity";
    return;
  }
  llvm_unreachable("unknown DynamicWork execution incomplete reason");
}

std::error_code DynamicWorkExecutionIncomplete::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<sim::ThreadDispatchOccurrenceId>
DynamicWorkExecutionSession::allocateDispatchOccurrence() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (nextDispatchOccurrence_ == 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dynamic_work_execution_invalid: dispatch occurrence space is "
        "exhausted");
  return sim::ThreadDispatchOccurrenceId(nextDispatchOccurrence_++);
}

llvm::Expected<DynamicWorkExecutionResult>
DynamicWorkExecutionSession::executeRoot(
    const dataflow::CanonicalDataflowProgramView &dataflow,
    const mapping::FinalizedSystemMapping &systemMapping,
    dataflow::RootThreadLaunchRef root, DynamicWorkExecutionRequest request,
    DynamicWorkSelectedBodyExecutor executor) {
  if (!executor)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dynamic_work_execution_invalid: execution body is absent");
  auto dynamic = dataflow.projectDynamicWork(root);
  if (!dynamic)
    return dynamic.takeError();
  if (dynamic->stableItemKeys.size() != 1)
    return llvm::make_error<DynamicWorkExecutionUnsupported>(
        DynamicWorkExecutionUnsupportedReason::StableItemDomainUnavailable,
        "DynamicWork stable-item domain is not the admitted singleton");
  if (request.rootPayload.size() != dynamic->payloadByteWidth)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dynamic_work_execution_invalid: root payload byte width differs "
        "from canonical Dataflow");
  if (systemMapping.view().dataflowIdentity() != dataflow.identity())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dynamic_work_execution_invalid: SystemMapping names a foreign "
        "Dataflow owner");

  const auto &contexts = systemMapping.verifiedClosure().executionContexts;
  auto instruction = selectSystemDynamicWorkInstructionExecutionContext(
      contexts, root, dynamic->stableItemKeys.front());
  if (!instruction)
    return instruction.takeError();
  std::optional<mapping::SelectedSystemSpatialContext> spatial;
  if (!dynamic->directGraphLaunches.empty()) {
    auto selected = selectSystemDynamicWorkSpatialExecutionContext(
        contexts, dynamic->directGraphLaunches.front(),
        dynamic->stableItemKeys.front());
    if (!selected)
      return selected.takeError();
    spatial = std::move(*selected);
  }
  auto servicePlans = selectDynamicWorkServicePlans(
      dataflow, systemMapping.verifiedClosure(), root, *instruction, spatial,
      dynamic->stableItemKeys.front());
  if (!servicePlans)
    return servicePlans.takeError();

  auto dispatchOccurrence = allocateDispatchOccurrence();
  if (!dispatchOccurrence)
    return dispatchOccurrence.takeError();
  auto scheduler = sim::DynamicWorkScheduler::create(
      *dispatchOccurrence, request.workerCount, request.queueCapacityPerWorker,
      request.rootPayload);
  if (!scheduler)
    return scheduler.takeError();

  DynamicWorkExecutionResult result;
  result.dispatchOccurrence = *dispatchOccurrence;
  std::uint32_t nextWorker = 0;
  while (!(*scheduler)->completed()) {
    std::optional<sim::DynamicWorkAssignment> assignment;
    for (std::uint32_t offset = 0; offset < request.workerCount; ++offset) {
      const std::uint32_t worker = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(nextWorker) + offset) %
          request.workerCount);
      auto acquired = (*scheduler)->acquire(worker);
      if (!acquired)
        return acquired.takeError();
      if (!*acquired)
        continue;
      assignment.emplace(std::move(**acquired));
      nextWorker = (worker + 1) % request.workerCount;
      break;
    }
    if (!assignment)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "dynamic_work_execution_invalid: live responsibility domain has "
          "no schedulable item");
    if (assignment->payload().size() != dynamic->payloadByteWidth) {
      llvm::Error error = llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "dynamic_work_execution_invalid: child payload byte width differs "
          "from canonical Dataflow");
      return llvm::joinErrors(
          std::move(error),
          cancelDynamicWorkDomain(**scheduler, std::move(*assignment)));
    }
    auto stableItem = sim::projectDynamicWorkStableItemKey(assignment->id());
    if (!stableItem) {
      llvm::Error error = stableItem.takeError();
      return llvm::joinErrors(
          std::move(error),
          cancelDynamicWorkDomain(**scheduler, std::move(*assignment)));
    }
    if (!(*stableItem == dynamic->stableItemKeys.front())) {
      llvm::Error error = llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "dynamic_work_execution_invalid: scheduler item class differs "
          "from Dataflow projection");
      return llvm::joinErrors(
          std::move(error),
          cancelDynamicWorkDomain(**scheduler, std::move(*assignment)));
    }

    DynamicWorkExecutionAssignment executionAssignment{
        assignment->id(), assignment->workerOrdinal(), assignment->payload(),
        *instruction, spatial, *servicePlans};
    auto itemResult = executor(executionAssignment);
    ++result.processedItemCount;
    if (!itemResult) {
      llvm::Error bodyError = itemResult.takeError();
      return llvm::joinErrors(
          std::move(bodyError),
          cancelDynamicWorkDomain(**scheduler, std::move(*assignment)));
    }
    if (itemResult->action ==
            DynamicWorkExecutionAction::RequestCancellation &&
        !itemResult->childPayloads.empty()) {
      llvm::Error error = llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "dynamic_work_execution_invalid: a cancelled item cannot publish "
          "children");
      return llvm::joinErrors(
          std::move(error),
          cancelDynamicWorkDomain(**scheduler, std::move(*assignment)));
    }
    for (const auto &payload : itemResult->childPayloads) {
      if (payload.size() == dynamic->payloadByteWidth)
        continue;
      llvm::Error error = llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "dynamic_work_execution_invalid: child payload byte width differs "
          "from canonical Dataflow");
      return llvm::joinErrors(
          std::move(error),
          cancelDynamicWorkDomain(**scheduler, std::move(*assignment)));
    }
    auto published = (*scheduler)->publishChildren(
        *assignment, itemResult->childPayloads);
    if (!published) {
      llvm::Error error = published.takeError();
      return llvm::joinErrors(
          std::move(error),
          cancelDynamicWorkDomain(**scheduler, std::move(*assignment)));
    }
    if (published->kind == sim::DynamicWorkPublishKind::WouldBlock) {
      llvm::Error error = llvm::make_error<DynamicWorkExecutionIncomplete>(
          DynamicWorkExecutionIncompleteReason::QueueCapacity,
          assignment->id());
      return llvm::joinErrors(
          std::move(error),
          cancelDynamicWorkDomain(**scheduler, std::move(*assignment)));
    }
    if (published->kind != sim::DynamicWorkPublishKind::Published) {
      llvm::Error error = llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "dynamic_work_execution_invalid: child publication observed an "
          "unexpected cancellation request");
      return llvm::joinErrors(
          std::move(error),
          cancelDynamicWorkDomain(**scheduler, std::move(*assignment)));
    }
    result.publishedChildCount += published->children.size();

    const bool cancelled = itemResult->action ==
                           DynamicWorkExecutionAction::RequestCancellation;
    auto retirement = retireDynamicWorkAssignment(
        **scheduler, std::move(*assignment), itemResult->action);
    if (!retirement)
      return retirement.takeError();
    result.joinEffect = *retirement;
    result.cancelled |= cancelled;
    if (cancelled)
      ++result.cancelledItemCount;
    else
      ++result.completedItemCount;
  }
  if (llvm::Error error = verifyDynamicWorkJoin(**scheduler, result.joinEffect))
    return std::move(error);
  result.replay = (*scheduler)->replay();
  return result;
}

llvm::Expected<DynamicWorkCgraExecutionResult>
DynamicWorkExecutionSession::executeRootCgra(
    const dataflow::CanonicalDataflowArtifact &dataflowArtifact,
    const mapping::FinalizedSystemMapping &systemMapping,
    dataflow::RootThreadLaunchRef root, DynamicWorkCgraExecutionRequest request,
    const ::loom::ArtifactStore &artifacts) {
  if (request.maxEventFrames == 0)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dynamic_work_execution_invalid: CGRA event-frame budget is zero");
  auto view = dataflowArtifact.view();
  if (!view)
    return view.takeError();
  auto dynamic = view->projectDynamicWork(root);
  if (!dynamic)
    return dynamic.takeError();
  if (dynamic->directGraphLaunches.size() != 1)
    return unsupported(
        DynamicWorkExecutionUnsupportedReason::SelectedGraphUnavailable,
        "the direct-CGRA profile requires exactly one selected graph");

  auto resolvedRoot = view->resolve(root);
  if (!resolvedRoot)
    return resolvedRoot.takeError();
  auto thread = llvm::dyn_cast<dataflow::ThreadOp>(resolvedRoot->callee);
  const auto graphRef = dynamic->directGraphLaunches.front();
  auto resolvedGraph = view->resolve(graphRef.staticGraphLaunch);
  if (!resolvedGraph)
    return resolvedGraph.takeError();
  auto graphLaunch = llvm::dyn_cast<dataflow::GraphLaunchOp>(resolvedGraph->op);
  if (!thread || !graphLaunch)
    return unsupported(
        DynamicWorkExecutionUnsupportedReason::ThreadBodyUnavailable,
        "the direct-CGRA profile cannot resolve its thread and graph body");

  for (mlir::Operation &operation : thread.getBody().front())
    if (!llvm::isa<dataflow::GraphLaunchOp, dataflow::ThreadYieldOp>(operation))
      return unsupported(
          DynamicWorkExecutionUnsupportedReason::ThreadBodyUnavailable,
          "the direct-CGRA profile admits only one graph launch and thread "
          "yield");
  const mlir::Value workItem =
      thread.getBody().front().getArgument(dynamic->workItemArgumentOrdinal);
  if (graphLaunch.getValueInputs().size() != 1 ||
      graphLaunch.getValueInputs().front() != workItem ||
      !graphLaunch.getStreamInputs().empty() ||
      !graphLaunch.getMemoryInputs().empty() ||
      !graphLaunch.getStreamOutputs().empty() ||
      !graphLaunch.getMemoryResults().empty())
    return unsupported(
        DynamicWorkExecutionUnsupportedReason::GraphBoundaryUnavailable,
        "the direct-CGRA profile requires one forwarded value input and no "
        "stream or memory boundary");
  auto payloadType = llvm::dyn_cast<mlir::IntegerType>(dynamic->workItemType);
  if (!payloadType || !payloadType.isSignless() ||
      payloadType.getWidth() == 0 || (payloadType.getWidth() % 8) != 0)
    return unsupported(
        DynamicWorkExecutionUnsupportedReason::ScalarPayloadUnavailable,
        "the direct-CGRA profile requires a byte-addressable signless integer "
        "work item");

  std::optional<mapping::InstructionExecutionContextKey> instruction;
  std::optional<mapping::SelectedSystemSpatialContext> spatial;
  std::vector<DynamicWorkSelectedServicePlan> servicePlans;
  std::optional<sim::RetiredCgraSimulation> retired;
  const std::uint64_t maxEventFrames = request.maxEventFrames;
  auto dispatch = executeRoot(
      *view, systemMapping, root, std::move(request.dispatch),
      [&](const DynamicWorkExecutionAssignment &assignment)
          -> llvm::Expected<DynamicWorkItemExecution> {
        if (!assignment.spatialContext)
          return unsupported(
              DynamicWorkExecutionUnsupportedReason::SelectedGraphUnavailable,
              "the direct-CGRA profile has no selected Spatial context");
        instruction = assignment.instructionContext;
        spatial = *assignment.spatialContext;
        servicePlans = assignment.servicePlans;

        llvm::APInt payload(payloadType.getWidth(), 0);
        for (const auto indexed : llvm::enumerate(assignment.payload))
          payload.insertBits(llvm::APInt(8, indexed.value()),
                             indexed.index() * 8);
        sim::CanonicalValueSequence value{
            1, {sim::SemanticLane::defined(std::move(payload))}};
        sim::SpatialSimulationWorkload workloadDraft{graphRef};
        workloadDraft.valueInputPlan = {sim::RuntimeValueInput{}};
        workloadDraft.observableContract.valueResults.resize(
            graphLaunch.getValueResults().size());
        std::iota(workloadDraft.observableContract.valueResults.begin(),
                  workloadDraft.observableContract.valueResults.end(), 0);
        auto workload = sim::finalizeSimulationWorkload(workloadDraft, *view);
        if (!workload)
          return workload.takeError();
        sim::SpatialSimulationRuntimeInputDraft runtimeDraft{
            workload->identity()};
        runtimeDraft.runtimeValues = {{0, std::move(value)}};
        auto runtimeInput =
            sim::finalizeSimulationRuntimeInput(runtimeDraft, *workload, *view);
        if (!runtimeInput)
          return runtimeInput.takeError();

        const ArtifactRootReference dataflowReference{
            dataflow::canonicalDataflowSchema.identity.str(),
            dataflow::canonicalDataflowSchema.version,
            dataflowArtifact.identity()};
        auto selectedSpatialMapping =
            mapping::importSpatialMapping(spatial->spatialMapping, artifacts);
        if (!selectedSpatialMapping)
          return selectedSpatialMapping.takeError();
        const ArtifactRootReference fabricReference{
            fabric::fabricArtifactSchema.identity.str(),
            fabric::fabricArtifactSchema.version,
            selectedSpatialMapping->view().fabricIdentity()};
        auto prepared =
            sim::prepareCgraExecution(dataflowReference, fabricReference,
                                      spatial->spatialMapping, artifacts);
        if (!prepared)
          return prepared.takeError();
        auto outcome = sim::simulateCgraWorkload(*prepared, *workload,
                                                 *runtimeInput, maxEventFrames);
        if (!outcome)
          return outcome.takeError();
        if (outcome->state != sim::SpatialExecutionSessionState::Retired ||
            !outcome->retired)
          return llvm::make_error<DynamicWorkCgraExecutionIncomplete>(
              std::move(*outcome));
        retired = std::move(*outcome->retired);
        return DynamicWorkItemExecution{};
      });
  if (!dispatch)
    return dispatch.takeError();
  if (!instruction || !spatial || !retired)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dynamic_work_execution_invalid: retired CGRA execution omitted its "
        "selected context");
  return DynamicWorkCgraExecutionResult{std::move(*dispatch), *instruction,
                                        *spatial, std::move(servicePlans),
                                        std::move(*retired)};
}

llvm::Expected<DynamicWorkCgraReplayResult>
DynamicWorkExecutionSession::executeRootCgraReplay(
    const dataflow::CanonicalDataflowArtifact &dataflowArtifact,
    const mapping::FinalizedSystemMapping &systemMapping,
    dataflow::RootThreadLaunchRef root, DynamicWorkCgraExecutionRequest request,
    const ::loom::ArtifactStore &artifacts) {
  const DynamicWorkCgraExecutionRequest replayRequest = request;
  const DynamicWorkExecutionRequest cancellationRequest = request.dispatch;

  auto completed = executeRootCgra(dataflowArtifact, systemMapping, root,
                                   std::move(request), artifacts);
  if (!completed)
    return completed.takeError();
  auto completedReplay = executeRootCgra(dataflowArtifact, systemMapping, root,
                                         replayRequest, artifacts);
  if (!completedReplay)
    return completedReplay.takeError();
  if (!equivalentCompletedExecution(*completed, *completedReplay))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dynamic_work_execution_invalid: completed replay changed its "
        "persistent selection, schedule, or retired CGRA observation");

  auto view = dataflowArtifact.view();
  if (!view)
    return view.takeError();

  std::optional<mapping::InstructionExecutionContextKey> cancelledInstruction;
  std::optional<mapping::SelectedSystemSpatialContext> cancelledSpatial;
  std::vector<DynamicWorkSelectedServicePlan> cancelledServicePlans;
  auto cancel = [&](const DynamicWorkExecutionAssignment &assignment)
      -> llvm::Expected<DynamicWorkItemExecution> {
    cancelledInstruction = assignment.instructionContext;
    cancelledSpatial = assignment.spatialContext;
    cancelledServicePlans = assignment.servicePlans;
    return DynamicWorkItemExecution{
        DynamicWorkExecutionAction::RequestCancellation, {}};
  };
  auto cancelled =
      executeRoot(*view, systemMapping, root, cancellationRequest, cancel);
  if (!cancelled)
    return cancelled.takeError();
  if (!cancelledInstruction || !cancelledSpatial)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dynamic_work_execution_invalid: cancellation replay omitted its "
        "selected context");

  std::optional<mapping::InstructionExecutionContextKey> replayedInstruction;
  std::optional<mapping::SelectedSystemSpatialContext> replayedSpatial;
  std::vector<DynamicWorkSelectedServicePlan> replayedServicePlans;
  auto cancelReplay = [&](const DynamicWorkExecutionAssignment &assignment)
      -> llvm::Expected<DynamicWorkItemExecution> {
    replayedInstruction = assignment.instructionContext;
    replayedSpatial = assignment.spatialContext;
    replayedServicePlans = assignment.servicePlans;
    return DynamicWorkItemExecution{
        DynamicWorkExecutionAction::RequestCancellation, {}};
  };
  auto cancelledReplay = executeRoot(*view, systemMapping, root,
                                     cancellationRequest, cancelReplay);
  if (!cancelledReplay)
    return cancelledReplay.takeError();
  if (!replayedInstruction || !replayedSpatial ||
      !(*cancelledInstruction == *replayedInstruction) ||
      cancelledSpatial->spatialMapping != replayedSpatial->spatialMapping ||
      !(cancelledSpatial->context == replayedSpatial->context) ||
      !(*cancelledInstruction == completed->instructionContext) ||
      cancelledSpatial->spatialMapping !=
          completed->spatialContext.spatialMapping ||
      !(cancelledSpatial->context == completed->spatialContext.context) ||
      !equivalentServicePlans(cancelledServicePlans, replayedServicePlans) ||
      !equivalentServicePlans(cancelledServicePlans, completed->servicePlans) ||
      !equivalentDispatch(*cancelled, *cancelledReplay))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "dynamic_work_execution_invalid: cancellation replay changed its "
        "persistent selection or schedule");

  return DynamicWorkCgraReplayResult{
      std::move(*completed), std::move(*completedReplay), std::move(*cancelled),
      std::move(*cancelledReplay)};
}

} // namespace loom::runtime

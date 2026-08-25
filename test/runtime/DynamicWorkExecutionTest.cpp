#include "DeploymentTestSupport.h"
#include "RootCompleteSpatialPnrTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Tech/TechMappingConfig.h"
#include "PnR/PnrConfig.h"
#include "Runtime/DynamicWorkExecution.h"
#include "Simulator/CGRASimulator.h"
#include "Simulator/DFGSimulator.h"
#include "Simulator/SimulationArtifacts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    loom::deployment::test::fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, mlir::arith::ArithDialect,
                  mlir::DLTIDialect, mlir::func::FuncDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact
buildDynamicDataflow(llvm::StringRef test, mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @sync(%start: none, %value: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
  dataflow.thread private @worker
      domain(#dataflow.thread_domain<dynamic_work, work_item_arg = 0>)(
      %work: i32) ctrl (%ctrl: none) {
    %result, %done = dataflow.graph.launch @sync deps(%ctrl)
        values(%work) stream_inputs() memories() stream_outputs()
        : (none, i32) -> (i32, none)
    dataflow.thread.yield %done : none
  }
  func.func private @host() {
    %work = arith.constant 7 : i32
    %completion = dataflow.thread.launch @worker(%work)
        : (i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  loom::deployment::test::require(test, static_cast<bool>(module),
                                  "cannot parse DynamicWork Dataflow");
  return take(test, dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildCapturedDynamicDataflow(llvm::StringRef test, mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.thread private @worker
      domain(#dataflow.thread_domain<dynamic_work, work_item_arg = 0>)(
      %work: i32, %capture: i32) ctrl (%ctrl: none) {
    %sum = arith.addi %work, %capture : i32
    dataflow.thread.yield
  }
  func.func private @host() {
    %work = arith.constant 7 : i32
    %capture = arith.constant 3 : i32
    %completion = dataflow.thread.launch @worker(%work, %capture)
        : (i32, i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  loom::deployment::test::require(test, static_cast<bool>(module),
                                  "cannot parse captured DynamicWork Dataflow");
  return take(test, dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildGraphlessDynamicDataflow(llvm::StringRef test,
                              mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.thread private @worker
      domain(#dataflow.thread_domain<dynamic_work, work_item_arg = 0>)(
      %work: i32) ctrl (%ctrl: none) {
    dataflow.thread.yield
  }
  func.func private @host() {
    %work = arith.constant 7 : i32
    %completion = dataflow.thread.launch @worker(%work)
        : (i32) -> !dataflow.thread_token
    return
  }
}
)mlir",
                                                        &context);
  loom::deployment::test::require(
      test, static_cast<bool>(module),
      "cannot parse graphless DynamicWork Dataflow");
  return take(test, dataflow::finalizeCanonicalDataflow(*module));
}

loom::sim::CanonicalValueSequence scalarI32(std::uint32_t value) {
  return {1, {loom::sim::SemanticLane::defined(llvm::APInt(32, value))}};
}

std::uint64_t
observedI32(llvm::StringRef test,
            const loom::sim::SpatialFunctionalObservations &observations) {
  loom::deployment::test::require(
      test, observations.valueResults.size() == 1,
      "execution did not publish one selected value result");
  const auto *published = std::get_if<loom::sim::PublishedValueResult>(
      &observations.valueResults.front());
  loom::deployment::test::require(
      test,
      published && published->value.tokenCount == 1 &&
          published->value.lanes.size() == 1 &&
          published->value.lanes.front().state ==
              loom::sim::SemanticState::Defined &&
          !published->value.lanes.front().pointerTarget &&
          published->value.lanes.front().bits.getBitWidth() == 32,
      "execution result is not one defined i32 token");
  return published->value.lanes.front().bits.getZExtValue();
}

void typedDynamicWorkGatesRemainDistinct(mlir::MLIRContext &context) {
  const llvm::StringRef test = __func__;
  auto captured = buildCapturedDynamicDataflow(test, context);
  auto view = take(test, captured.view());
  loom::deployment::test::require(
      test, view.rootThreadLaunches().size() == 1,
      "captured fixture did not retain one root launch");
  auto projection =
      view.projectDynamicWork(view.rootThreadLaunches().front().ref);
  loom::deployment::test::require(
      test, !projection,
      "captured DynamicWork entered the root-only execution profile");
  bool captureGate = false;
  llvm::handleAllErrors(
      projection.takeError(),
      [&](const dataflow::DynamicWorkProjectionUnsupported &error) {
        captureGate = error.reason() ==
                      dataflow::DynamicWorkProjectionUnsupportedReason::
                          LaunchCapturesUnavailable;
      },
      [&](const llvm::ErrorInfoBase &error) {
        loom::deployment::test::fail(test, error.message());
      });
  loom::deployment::test::require(
      test, captureGate,
      "capture rejection lost its DynamicWork projection reason");

  const auto root =
      loom::sim::WorkItemId::root(loom::sim::ThreadDispatchOccurrenceId(17));
  auto stable = loom::sim::projectDynamicWorkStableItemKey(
      loom::sim::WorkItemId::child(root, 0));
  loom::deployment::test::require(
      test, static_cast<bool>(stable),
      "child WorkItemId lost the domain-wide stable execution class");
}

const std::vector<loom::ArtifactRootReference> &
usableOutputs(llvm::StringRef test,
              const loom::dse::CandidateGeneratorProviderResult &outcome) {
  if (const auto *completed =
          std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
              &outcome.outcome)) {
    loom::deployment::test::require(
        test, completed->outputBindings.size() == 1,
        "candidate generator returned an unexpected output shape");
    return completed->outputBindings.front().artifacts;
  }
  const auto *incomplete =
      std::get_if<loom::dse::IncompleteCandidateGeneratorResult>(
          &outcome.outcome);
  loom::deployment::test::require(
      test,
      incomplete &&
          incomplete->reason == loom::dse::CandidateGeneratorIncompleteReason::
                                    SemanticLimitReached &&
          incomplete->retainedOutputBindings.size() == 1,
      "candidate generator did not retain a verified output prefix");
  return incomplete->retainedOutputBindings.front().artifacts;
}

loom::ArtifactRootReference generateTechMapping(
    llvm::StringRef test, const loom::ArtifactRootReference &dataflow,
    const loom::ArtifactRootReference &fabric, loom::ArtifactStore &artifacts,
    const loom::BlobStore &blobs) {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.dse.techMapping.candidatePublicationLimit = 1;
  auto config =
      take(test, loom::mapping::projectResolvedTechMappingConfigView(resolved));
  auto inputs =
      take(test, loom::dse::bindRootCompleteTechMappingCandidateGeneratorInputs(
                     {dataflow}, fabric));
  auto binding = take(
      test, loom::dse::resolveRootCompleteTechMappingCandidateGeneratorBinding(
                config));
  auto outcome = take(test, loom::dse::invokeCandidateGenerator(
                                inputs, binding, artifacts, blobs));
  const auto &outputs = usableOutputs(test, outcome);
  loom::deployment::test::require(
      test, outputs.size() == 1,
      "Tech Mapping did not produce one verified candidate");
  return outputs.front();
}

loom::ArtifactRootReference generateSpatialMapping(
    llvm::StringRef test, const loom::ArtifactRootReference &techMapping,
    const loom::fabric::FinalizedFabricRoot &fabric,
    loom::ArtifactStore &artifacts, const loom::BlobStore &blobs) {
  const auto timing =
      take(test, loom::fabric::projectNormalizedFabricPhysicalTimingProfile(
                     fabric.view()));
  const auto timingReference =
      take(test,
           loom::fabric::publishFabricPhysicalTimingProfile(timing, artifacts));

  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  auto &search = resolved.dse.spatialPnr.search;
  search.initializer.seedAttemptCount = 1;
  search.routing.negotiationIterationLimit = 8;
  search.actionProposal = {0, 1, 0};
  search.annealing.calibrationProposalCount = 1;
  search.annealing.fallbackTemperature = 1;
  search.annealing.minimumTemperature = 1;
  search.annealing.coolingRatio = {1, 2};
  search.annealing.proposalsPerLevelBase = 1;
  search.annealing.proposalsPerMovableDecision = 0;
  search.exactRepair = {loom::ResolvedPnrExactRepairKind::Disabled, 0, 0};
  auto config =
      take(test, loom::pnr::projectResolvedSpatialPnrConfigView(resolved));
  auto inputs =
      take(test, loom::dse::bindRootCompleteSpatialPnrCandidateGeneratorInputs(
                     {techMapping}, fabric.reference(), timingReference));
  auto binding = take(
      test, loom::dse::resolveRootCompleteSpatialPnrCandidateGeneratorBinding(
                config));
  auto outcome = take(test, loom::dse::invokeCandidateGenerator(
                                inputs, binding, artifacts, blobs));
  const auto &outputs = usableOutputs(test, outcome);
  loom::deployment::test::require(
      test, outputs.size() == 1,
      "Spatial PnR did not produce one verified candidate");
  return outputs.front();
}

void requireReplay(
    llvm::StringRef test,
    llvm::ArrayRef<loom::sim::DynamicWorkScheduleAction> replay,
    llvm::ArrayRef<loom::sim::DynamicWorkScheduleActionKind> expected,
    loom::sim::WorkItemId item) {
  loom::deployment::test::require(test, replay.size() == expected.size(),
                                  "runtime replay length changed");
  for (std::size_t ordinal = 0; ordinal != replay.size(); ++ordinal)
    loom::deployment::test::require(
        test,
        replay[ordinal].kind == expected[ordinal] &&
            replay[ordinal].item == item,
        "runtime replay lost its stable item or transition order");
}

void dynamicWorkTraversesMappingAndJoins() {
  const llvm::StringRef test = __func__;
  loom::deployment::test::TemporaryTree tree(test);
  loom::ArtifactStore artifacts(tree.path("artifacts"));
  loom::BlobStore blobs(tree.path("blobs"));
  mlir::MLIRContext context = makeContext();

  auto dataflow = buildDynamicDataflow(test, context);
  const auto dataflowReference =
      take(test, dataflow::publishCanonicalDataflow(dataflow, artifacts));
  auto dataflowView = take(test, dataflow.view());
  loom::deployment::test::require(
      test, dataflowView.rootThreadLaunches().size() == 1,
      "DynamicWork fixture did not retain one root launch");
  const dataflow::RootThreadLaunchRef root =
      dataflowView.rootThreadLaunches().front().ref;
  auto dynamic = take(test, dataflowView.projectDynamicWork(root));
  loom::deployment::test::require(
      test,
      dynamic.directGraphLaunches.size() == 1 &&
          dynamic.stableItemKeys.size() == 1 && dynamic.payloadByteWidth == 4,
      "Dataflow projection changed the admitted DynamicWork domain");
  const std::array<std::uint8_t, 4> expectedRootKey{0, 0, 0, 0};
  const auto encodedRootKey =
      dataflow::encodeDynamicWorkStableItemKey(dynamic.stableItemKeys.front());
  loom::deployment::test::require(
      test,
      llvm::ArrayRef<std::uint8_t>(encodedRootKey) ==
          llvm::ArrayRef<std::uint8_t>(expectedRootKey.data(),
                                       expectedRootKey.size()),
      "Dataflow root stable-key encoding changed");

  auto spatialCore = loom::test::buildSpatialCore(artifacts);
  const auto techMapping = generateTechMapping(
      test, dataflowReference, spatialCore.reference(), artifacts, blobs);
  const auto spatialMapping =
      generateSpatialMapping(test, techMapping, spatialCore, artifacts, blobs);
  const std::array<mlir::Type, 2> messagePayloads{
      mlir::NoneType::get(&context), mlir::IntegerType::get(&context, 32)};
  auto system = loom::deployment::test::buildMappedSpatialSystem(
      test, spatialCore, messagePayloads, artifacts, false);
  auto systemMapping = loom::deployment::test::buildMappedSystemMapping(
      test, dataflow, system, {spatialMapping}, artifacts);

  auto graphless = buildGraphlessDynamicDataflow(test, context);
  auto graphlessView = take(test, graphless.view());
  loom::runtime::DynamicWorkCgraExecutionRequest graphlessRequest;
  graphlessRequest.dispatch.workerCount = 1;
  graphlessRequest.dispatch.queueCapacityPerWorker = 1;
  graphlessRequest.dispatch.rootPayload = {7, 0, 0, 0};
  graphlessRequest.maxEventFrames = 1000;
  loom::runtime::DynamicWorkExecutionSession graphlessSession;
  auto graphlessExecution = graphlessSession.executeRootCgra(
      graphless, systemMapping, graphlessView.rootThreadLaunches().front().ref,
      std::move(graphlessRequest), artifacts);
  loom::deployment::test::require(
      test, !graphlessExecution,
      "graphless DynamicWork entered the direct-CGRA profile");
  bool selectedGraphGate = false;
  llvm::handleAllErrors(
      graphlessExecution.takeError(),
      [&](const loom::runtime::DynamicWorkExecutionUnsupported &error) {
        selectedGraphGate =
            error.reason() ==
            loom::runtime::DynamicWorkExecutionUnsupportedReason::
                SelectedGraphUnavailable;
      },
      [&](const llvm::ErrorInfoBase &error) {
        loom::deployment::test::fail(test, error.message());
      });
  loom::deployment::test::require(
      test, selectedGraphGate,
      "graphless rejection lost its direct-CGRA capability reason");

  loom::runtime::DynamicWorkCgraExecutionRequest limitedRequest;
  limitedRequest.dispatch.workerCount = 1;
  limitedRequest.dispatch.queueCapacityPerWorker = 1;
  limitedRequest.dispatch.rootPayload = {7, 0, 0, 0};
  limitedRequest.maxEventFrames = 1;
  loom::runtime::DynamicWorkExecutionSession limitedSession;
  auto limitedExecution = limitedSession.executeRootCgra(
      dataflow, systemMapping, root, std::move(limitedRequest), artifacts);
  loom::deployment::test::require(
      test, !limitedExecution,
      "bounded CGRA execution retired beyond its event-frame budget");
  bool incompleteOutcome = false;
  llvm::handleAllErrors(
      limitedExecution.takeError(),
      [&](const loom::runtime::DynamicWorkCgraExecutionIncomplete &error) {
        incompleteOutcome =
            error.outcome().state ==
            loom::sim::SpatialExecutionSessionState::StoppedByLimit;
      },
      [&](const llvm::ErrorInfoBase &error) {
        loom::deployment::test::fail(test, error.message());
      });
  loom::deployment::test::require(
      test, incompleteOutcome,
      "event-frame exhaustion lost its typed incomplete outcome");

  loom::sim::SpatialSimulationWorkload workloadDraft{
      dynamic.directGraphLaunches.front()};
  workloadDraft.valueInputPlan = {loom::sim::RuntimeValueInput{}};
  workloadDraft.observableContract.valueResults = {0};
  auto workload = take(
      test, loom::sim::finalizeSimulationWorkload(workloadDraft, dataflowView));
  loom::sim::SpatialSimulationRuntimeInputDraft runtimeDraft{
      workload.identity()};
  runtimeDraft.runtimeValues = {{0, scalarI32(7)}};
  auto runtimeInput = take(test, loom::sim::finalizeSimulationRuntimeInput(
                                     runtimeDraft, workload, dataflowView));
  auto dfgReference = take(test, loom::sim::simulateRetiredDfgWorkload(
                                     dataflow, workload, runtimeInput, 1000));
  loom::deployment::test::require(
      test, observedI32(test, dfgReference.observations) == 7,
      "DFG reference changed the DynamicWork item value");
  const auto &execution = systemMapping.view().executionBindings();
  const auto threadBindings = execution.threadBindings();
  const auto graphBindings = execution.graphBindings();
  loom::deployment::test::require(
      test,
      threadBindings.size() == 1 && graphBindings.size() == 1 &&
          threadBindings.front().relationKind ==
              ::mapping::SystemBindingRelationKind::StableKeyLookup &&
          graphBindings.front().relationKind ==
              ::mapping::SystemBindingRelationKind::StableKeyLookup &&
          threadBindings.front().stableKeyEntries.size() == 1 &&
          graphBindings.front().stableKeyEntries.size() == 1,
      "System Mapping did not materialize exact stable-key relations");

  std::optional<loom::mapping::InstructionExecutionContextKey>
      completedInstruction;
  std::optional<loom::mapping::SelectedSystemSpatialContext> completedSpatial;
  loom::runtime::DynamicWorkExecutionSession executionSession;
  loom::runtime::DynamicWorkCgraExecutionRequest completeRequest;
  completeRequest.dispatch.workerCount = 2;
  completeRequest.dispatch.queueCapacityPerWorker = 2;
  completeRequest.dispatch.rootPayload = {7, 0, 0, 0};
  completeRequest.maxEventFrames = 1000;
  auto complete = take(test, executionSession.executeRootCgra(
                                 dataflow, systemMapping, root,
                                 std::move(completeRequest), artifacts));
  completedInstruction = complete.instructionContext;
  completedSpatial = complete.spatialContext;
  loom::deployment::test::require(
      test,
      complete.dispatch.dispatchOccurrence ==
              loom::sim::ThreadDispatchOccurrenceId(1) &&
          complete.dispatch.joinEffect ==
              loom::sim::RetirementEffect::DomainCompleted &&
          !complete.dispatch.cancelled &&
          complete.dispatch.processedItemCount == 1 &&
          complete.dispatch.publishedChildCount == 0 &&
          complete.dispatch.completedItemCount == 1 &&
          complete.dispatch.cancelledItemCount == 0 &&
          complete.spatialContext.spatialMapping == spatialMapping &&
          complete.spatialContext.context.accCore ==
              complete.instructionContext.accCore &&
          !complete.servicePlans.empty() &&
          observedI32(test, complete.execution.observations) ==
              observedI32(test, dfgReference.observations),
      "completed execution did not join its selected System contexts");
  const std::array completeKinds{
      loom::sim::DynamicWorkScheduleActionKind::AdmitRoot,
      loom::sim::DynamicWorkScheduleActionKind::AcquireLocal,
      loom::sim::DynamicWorkScheduleActionKind::Complete};
  requireReplay(
      test, complete.dispatch.replay, completeKinds,
      loom::sim::WorkItemId::root(loom::sim::ThreadDispatchOccurrenceId(1)));

  loom::runtime::DynamicWorkExecutionRequest treeRequest;
  treeRequest.workerCount = 2;
  treeRequest.queueCapacityPerWorker = 2;
  treeRequest.rootPayload = {2, 0, 0, 0};
  std::vector<loom::sim::WorkItemId> visitedItems;
  std::vector<std::uint32_t> visitedWorkers;
  auto workTree = take(
      test,
      executionSession.executeRoot(
          dataflowView, systemMapping, root, std::move(treeRequest),
          [&](const loom::runtime::DynamicWorkExecutionAssignment &assignment)
              -> llvm::Expected<loom::runtime::DynamicWorkItemExecution> {
            visitedItems.push_back(assignment.item);
            visitedWorkers.push_back(assignment.workerOrdinal);
            loom::deployment::test::require(
                test,
                assignment.instructionContext == *completedInstruction &&
                    assignment.spatialContext &&
                    assignment.spatialContext->context ==
                        completedSpatial->context &&
                    !assignment.servicePlans.empty(),
                "a child item changed the persistent System selection");
            if (!assignment.item.isRoot())
              return loom::runtime::DynamicWorkItemExecution{};
            return loom::runtime::DynamicWorkItemExecution{
                loom::runtime::DynamicWorkExecutionAction::Complete,
                {{1, 0, 0, 0}, {2, 0, 0, 0}}};
          }));
  const auto treeRoot =
      loom::sim::WorkItemId::root(loom::sim::ThreadDispatchOccurrenceId(2));
  const auto firstChild = loom::sim::WorkItemId::child(treeRoot, 0);
  const auto secondChild = loom::sim::WorkItemId::child(treeRoot, 1);
  loom::deployment::test::require(
      test,
      workTree.dispatchOccurrence ==
              loom::sim::ThreadDispatchOccurrenceId(2) &&
          workTree.joinEffect ==
              loom::sim::RetirementEffect::DomainCompleted &&
          !workTree.cancelled && workTree.processedItemCount == 3 &&
          workTree.publishedChildCount == 2 &&
          workTree.completedItemCount == 3 &&
          workTree.cancelledItemCount == 0 &&
          visitedItems ==
              std::vector<loom::sim::WorkItemId>{treeRoot, firstChild,
                                                 secondChild} &&
          visitedWorkers == std::vector<std::uint32_t>{0, 1, 0},
      "bounded execution did not steal and join the complete work tree");
  const std::array treeKinds{
      loom::sim::DynamicWorkScheduleActionKind::AdmitRoot,
      loom::sim::DynamicWorkScheduleActionKind::AcquireLocal,
      loom::sim::DynamicWorkScheduleActionKind::PublishChild,
      loom::sim::DynamicWorkScheduleActionKind::PublishChild,
      loom::sim::DynamicWorkScheduleActionKind::Complete,
      loom::sim::DynamicWorkScheduleActionKind::Steal,
      loom::sim::DynamicWorkScheduleActionKind::Complete,
      loom::sim::DynamicWorkScheduleActionKind::AcquireLocal,
      loom::sim::DynamicWorkScheduleActionKind::Complete};
  loom::deployment::test::require(
      test, workTree.replay.size() == treeKinds.size(),
      "work-tree replay has an unexpected transition count");
  for (std::size_t index = 0; index < treeKinds.size(); ++index)
    loom::deployment::test::require(
        test, workTree.replay[index].kind == treeKinds[index],
        "work-tree replay changed deterministic transition order");

  loom::runtime::DynamicWorkExecutionRequest blockedRequest;
  blockedRequest.workerCount = 2;
  blockedRequest.queueCapacityPerWorker = 1;
  blockedRequest.rootPayload = {2, 0, 0, 0};
  auto blocked = executionSession.executeRoot(
      dataflowView, systemMapping, root, std::move(blockedRequest),
      [](const loom::runtime::DynamicWorkExecutionAssignment &)
          -> llvm::Expected<loom::runtime::DynamicWorkItemExecution> {
        return loom::runtime::DynamicWorkItemExecution{
            loom::runtime::DynamicWorkExecutionAction::Complete,
            {{1, 0, 0, 0}, {2, 0, 0, 0}}};
      });
  loom::deployment::test::require(
      test, !blocked,
      "an over-capacity child batch entered the responsibility domain");
  bool capacityGate = false;
  llvm::handleAllErrors(
      blocked.takeError(),
      [&](const loom::runtime::DynamicWorkExecutionIncomplete &error) {
        capacityGate =
            error.reason() ==
                loom::runtime::DynamicWorkExecutionIncompleteReason::
                    QueueCapacity &&
            error.item() == loom::sim::WorkItemId::root(
                                loom::sim::ThreadDispatchOccurrenceId(3));
      },
      [&](const llvm::ErrorInfoBase &error) {
        loom::deployment::test::fail(test, error.message());
      });
  loom::deployment::test::require(
      test, capacityGate,
      "queue backpressure lost its typed incomplete item witness");

  loom::runtime::DynamicWorkExecutionRequest cancelRequest;
  cancelRequest.workerCount = 2;
  cancelRequest.queueCapacityPerWorker = 2;
  cancelRequest.rootPayload = {9, 0, 0, 0};
  auto cancelled = take(
      test,
      executionSession.executeRoot(
          dataflowView, systemMapping, root, std::move(cancelRequest),
          [&](const loom::runtime::DynamicWorkExecutionAssignment &assignment) {
            loom::deployment::test::require(
                test,
                assignment.item ==
                        loom::sim::WorkItemId::root(
                            loom::sim::ThreadDispatchOccurrenceId(4)) &&
                    assignment.instructionContext == *completedInstruction &&
                    assignment.spatialContext &&
                    assignment.spatialContext->context ==
                        completedSpatial->context &&
                    !assignment.servicePlans.empty(),
                "dispatch-local identity changed persistent selection");
            return loom::runtime::DynamicWorkItemExecution{
                loom::runtime::DynamicWorkExecutionAction::
                    RequestCancellation,
                {}};
          }));
  loom::deployment::test::require(
      test,
      cancelled.dispatchOccurrence ==
              loom::sim::ThreadDispatchOccurrenceId(4) &&
          cancelled.joinEffect ==
              loom::sim::RetirementEffect::DomainCompleted &&
          cancelled.cancelled && cancelled.processedItemCount == 1 &&
          cancelled.publishedChildCount == 0 &&
          cancelled.completedItemCount == 0 &&
          cancelled.cancelledItemCount == 1,
      "cancelled execution did not join its responsibility domain");
  const std::array cancelKinds{
      loom::sim::DynamicWorkScheduleActionKind::AdmitRoot,
      loom::sim::DynamicWorkScheduleActionKind::AcquireLocal,
      loom::sim::DynamicWorkScheduleActionKind::RequestCancellation,
      loom::sim::DynamicWorkScheduleActionKind::CancelActive};
  requireReplay(
      test, cancelled.replay, cancelKinds,
      loom::sim::WorkItemId::root(loom::sim::ThreadDispatchOccurrenceId(4)));
}

} // namespace

int main() {
  mlir::MLIRContext context = makeContext();
  typedDynamicWorkGatesRemainDistinct(context);
  dynamicWorkTraversesMappingAndJoins();
  return 0;
}

#include "DeploymentTestSupport.h"

#include "Application/ResourceTimeExecution.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"
#include "DSE/ResourceTimeSpectrum.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Mapping/Artifact/MappingProgressAnalysis.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "PnR/System/SystemMappingMigration.h"
#include "Runtime/DeploymentLoader.h"
#include "Runtime/InProcessPlatform.h"
#include "Runtime/ResourceTimeTransitionSelection.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

using namespace loom;
using namespace loom::deployment;

namespace {

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    deployment::test::fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef marker) {
  if (value)
    deployment::test::fail(test, "accepted invalid Deployment");
  const std::string message = llvm::toString(value.takeError());
  deployment::test::require(test, llvm::StringRef(message).contains(marker),
                            message);
}

void expectSelectionError(
    llvm::StringRef test, llvm::Error error,
    runtime::ResourceTimeSelectionErrorReason expectedReason) {
  if (!error)
    deployment::test::fail(test, "accepted invalid resource-time selection");
  std::optional<runtime::ResourceTimeSelectionErrorReason> observedReason;
  error = llvm::handleErrors(
      std::move(error),
      [&](const runtime::ResourceTimeSelectionError &error) -> llvm::Error {
        observedReason = error.reason();
        return llvm::Error::success();
      });
  if (error)
    deployment::test::fail(test, llvm::toString(std::move(error)));
  deployment::test::require(
      test, observedReason && *observedReason == expectedReason,
      "resource-time selector returned another typed rejection reason");
}

template <typename T>
void expectSelectionError(
    llvm::StringRef test, llvm::Expected<T> value,
    runtime::ResourceTimeSelectionErrorReason expectedReason) {
  if (value)
    deployment::test::fail(test, "accepted invalid resource-time selection");
  expectSelectionError(test, value.takeError(), expectedReason);
}

template <typename T>
runtime::RuntimeActivationReplacementErrorReason
expectActivationReplacementError(llvm::StringRef test,
                                 llvm::Expected<T> value) {
  if (value)
    deployment::test::fail(test, "invalid activation replacement unexpectedly "
                                 "succeeded");
  std::optional<runtime::RuntimeActivationReplacementErrorReason> reason;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const runtime::RuntimeActivationReplacementError &replacement) {
        reason = replacement.reason();
      },
      [&](const llvm::ErrorInfoBase &other) {
        deployment::test::fail(test, other.message());
      });
  if (!reason)
    deployment::test::fail(test,
                           "activation replacement had no typed diagnostic");
  return *reason;
}

std::vector<ArtifactIdentity> implementationIdentities(
    llvm::StringRef test, const deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  const auto bindings = deployment.deployment().hardwareBindings();
  std::vector<std::optional<ArtifactIdentity>> indexed(bindings.size());
  for (const deployment::DeploymentHardwareBinding &binding : bindings) {
    const auto runtimeBinding =
        take(test, runtime::importRuntimePlatformBinding(
                       binding.runtimePlatformBinding, artifacts, blobs));
    const auto *reported = std::get_if<runtime::HardwareReportedIdentity>(
        &runtimeBinding.binding().identityVerification());
    deployment::test::require(
        test, reported != nullptr,
        "resource-time activation fixture has no reported identity");
    bool found = false;
    for (std::size_t ordinal = 0; ordinal != bindings.size(); ++ordinal) {
      if (!(reported->implementationIdentityEndpoint ==
            runtime::inProcessRuntimeEndpoint(
                runtime::RuntimeEndpointClass::Identity, ordinal)))
        continue;
      deployment::test::require(test, !indexed[ordinal],
                                "runtime identity endpoint is duplicated");
      indexed[ordinal] = binding.hardwareImplementation.artifact;
      found = true;
      break;
    }
    deployment::test::require(test, found,
                              "runtime identity endpoint is outside fixture");
  }
  std::vector<ArtifactIdentity> result;
  result.reserve(indexed.size());
  for (const auto &identity : indexed) {
    deployment::test::require(test, identity.has_value(),
                              "runtime identity coverage is incomplete");
    result.push_back(*identity);
  }
  return result;
}

void exactClosureRoundTripsAndRejectsStaleChild() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  auto imported = importDeployment(deployment.reference(), artifacts, blobs);
  if (!imported)
    deployment::test::fail(test, llvm::toString(imported.takeError()));
  deployment::test::require(test,
                            imported->reference() == deployment.reference(),
                            "Deployment identity changed during strict import");
  deployment::test::require(
      test, !imported->deployment().spatialLaunchImage().has_value(),
      "empty SpatialMapping closure produced a SpatialLaunchImage");

  std::vector<std::uint8_t> stale(deployment.canonicalBytes().bytes().begin(),
                                  deployment.canonicalBytes().bytes().end());
  const llvm::StringRef bytes(reinterpret_cast<const char *>(stale.data()),
                              stale.size());
  const std::size_t admission =
      bytes.find("\"schema\":\"loom.admission_image\"");
  deployment::test::require(test, admission != llvm::StringRef::npos,
                            "Deployment fixture has no AdmissionImage");
  constexpr llvm::StringLiteral marker = "\"capacity\":";
  const std::size_t capacity = bytes.find(marker, admission);
  deployment::test::require(test, capacity != llvm::StringRef::npos,
                            "AdmissionImage fixture has no capacity cell");
  const std::size_t digit = capacity + marker.size();
  deployment::test::require(
      test, digit < stale.size() && stale[digit] >= '0' && stale[digit] <= '9',
      "AdmissionImage capacity is not an integer");
  stale[digit] = stale[digit] == '9' ? '8' : stale[digit] + 1;
  auto identity =
      artifacts.put(deploymentSchema, CanonicalSemanticBytes(std::move(stale)));
  if (!identity)
    deployment::test::fail(test, llvm::toString(identity.takeError()));
  expectError(test,
              importDeployment({deploymentSchema.identity.str(),
                                deploymentSchema.version, *identity},
                               artifacts, blobs),
              "stale derived runtime images");
}

void finalLinkedProgramMustMatchHostTarget() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  expectError(test,
              deployment::test::tryBuildMinimalDeployment(
                  test, artifacts, blobs, tree, "x86_64-unknown-linux-gnu"),
              "final linked module is incompatible with the host target");
}

void resourceTimeTransitionRequiresExactDeploymentClosure() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const FinalizedDeployment parent =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const FinalizedDeployment child =
      deployment::test::buildRetargetedMinimalDeployment(test, artifacts, blobs,
                                                         tree);
  deployment::test::require(test,
                            parent.deployment().systemMapping() !=
                                child.deployment().systemMapping(),
                            "resource-time fixture has one SystemMapping");
  deployment::test::require(test, parent.reference() != child.reference(),
                            "resource-time fixture has one Deployment state");

  const auto parentMapping =
      take(test, loom::mapping::importSystemMapping(
                     parent.deployment().systemMapping(), artifacts));
  const auto childMapping =
      take(test, loom::mapping::importSystemMapping(
                     child.deployment().systemMapping(), artifacts));
  deployment::test::require(
      test,
      parentMapping.view().dataflowIdentity() ==
              childMapping.view().dataflowIdentity() &&
          parentMapping.view().fabricIdentity() ==
              childMapping.view().fabricIdentity(),
      "resource-time endpoints changed Dataflow or Fabric");
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      parentMapping.view().dataflowIdentity()};
  auto dataflowArtifact = take(
      test, dataflow::importCanonicalDataflow(dataflowReference, artifacts));
  auto dataflow = take(test, dataflowArtifact.view());
  deployment::test::require(test, !dataflow.rootThreadLaunches().empty(),
                            "resource-time fixture has no execution root");
  deployment::test::require(test, dataflow.rootThreadLaunches().size() == 2,
                            "resource-time fixture needs two execution roots");
  const dataflow::RootThreadLaunchRef firstRoot =
      dataflow.rootThreadLaunches().front().ref;
  const dataflow::RootThreadLaunchRef secondRoot =
      dataflow.rootThreadLaunches()[1].ref;
  const std::array<dataflow::EventFamilyKey, 4> boundaryEvents = {
      dataflow::rootThreadStartEventFamily(firstRoot),
      dataflow::rootThreadCompletionEventFamily(firstRoot),
      dataflow::rootThreadStartEventFamily(secondRoot),
      dataflow::rootThreadCompletionEventFamily(secondRoot)};
  const auto causality =
      take(test,
           loom::mapping::freezeMappingProgressModel(dataflow, boundaryEvents));
  const bool firstPrecedesSecond = take(
      test, loom::mapping::mappingEventPrecedes(
                causality, dataflow::rootThreadCompletionEventFamily(firstRoot),
                dataflow::rootThreadStartEventFamily(secondRoot)));
  const bool secondPrecedesFirst =
      take(test,
           loom::mapping::mappingEventPrecedes(
               causality, dataflow::rootThreadCompletionEventFamily(secondRoot),
               dataflow::rootThreadStartEventFamily(firstRoot)));
  deployment::test::require(test, !firstPrecedesSecond && !secondPrecedesFirst,
                            "resource-time fixture roots are not independent");
  const dataflow::RootThreadLaunchRef precedingRoot = firstRoot;
  const dataflow::RootThreadLaunchRef root = secondRoot;
  const auto contexts =
      take(test, loom::mapping::projectSystemExecutionContexts(
                     dataflow, parentMapping.view().executionBindings()));
  const auto resourcesFor = [&](const auto &selectedContexts,
                                dataflow::RootThreadLaunchRef selected) {
    std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef> result;
    const auto appendCore = [&](loom::fabric::AccCoreOccurrenceRef core) {
      result.push_back(
          take(test, loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
                         loom::fabric::FabricInventoryOwnerRef::of(core))));
    };
    for (const auto &domain : selectedContexts.instructionDomains)
      if (domain.root == selected)
        appendCore(domain.context.accCore);
    for (const auto &domain : selectedContexts.spatialDomains)
      if (domain.graph.rootThreadLaunch == selected)
        appendCore(domain.context.accCore);
    return result;
  };
  const auto resources = resourcesFor(contexts, root);
  const auto precedingResources = resourcesFor(contexts, precedingRoot);
  deployment::test::require(test, !resources.empty(),
                            "resource-time fixture root has no AccCore");
  deployment::test::require(test, !precedingResources.empty(),
                            "resource-time fixture prior root has no AccCore");
  const auto startRoot = [&](auto &selector,
                             dataflow::RootThreadLaunchRef startedRoot) {
    if (llvm::Error error = selector.startRoot(startedRoot))
      deployment::test::fail(test, llvm::toString(std::move(error)));
  };

  pnr::ResourceTimeTransition draft{
      dataflow::rootThreadCompletionEventFamily(root),
      pnr::ResourceTimeSafePointReference{
          dataflowReference, pnr::ResourceTimeSafePointKind::Completion},
      {parentMapping.reference(), parent.reference()},
      {childMapping.reference(), child.reference()},
      {{root, resources}},
      {},
      {precedingRoot},
      {},
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      pnr::ResourceTimeTransitionStatus::ProofNotEstablished};
  const pnr::ResourceTimeTransition transition =
      take(test, pnr::finalizeResourceTimeTransition(draft, artifacts, blobs));
  deployment::test::require(
      test, transition.status == pnr::ResourceTimeTransitionStatus::Verified,
      "resource-time finalizer did not publish a verified edge");
  if (llvm::Error error = pnr::verifyResourceTimeTransitionClosure(
          transition, artifacts, blobs))
    deployment::test::fail(test, llvm::toString(std::move(error)));
  const pnr::ResourceTimeTransitionGraph transitionGraph{
      transition.parent, {transition.parent, transition.child}, {transition}};
  if (llvm::Error error = pnr::verifyResourceTimeTransitionGraph(
          transitionGraph, artifacts, blobs))
    deployment::test::fail(
        test, (llvm::Twine("finite transition graph failed closure: ") +
               llvm::toString(std::move(error)))
                  .str());

  const auto childContexts =
      take(test, loom::mapping::projectSystemExecutionContexts(
                     dataflow, childMapping.view().executionBindings()));
  auto regressingDraft = draft;
  regressingDraft.trigger =
      dataflow::rootThreadCompletionEventFamily(precedingRoot);
  regressingDraft.parent = transition.child;
  regressingDraft.child = transition.parent;
  regressingDraft.beforeActive = {
      {precedingRoot, resourcesFor(childContexts, precedingRoot)}};
  regressingDraft.completedBefore.clear();
  const auto regressingTransition =
      take(test, pnr::finalizeResourceTimeTransition(std::move(regressingDraft),
                                                     artifacts, blobs));
  const pnr::ResourceTimeTransitionGraph regressingGraph{
      transition.parent,
      {transition.parent, transition.child},
      {transition, regressingTransition}};
  llvm::Error regressingError =
      pnr::verifyResourceTimeTransitionGraph(regressingGraph, artifacts, blobs);
  if (!regressingError)
    deployment::test::fail(
        test, "transition graph accepted a regressing completion frontier");
  const std::string regressingMessage =
      llvm::toString(std::move(regressingError));
  deployment::test::require(
      test, llvm::StringRef(regressingMessage).contains("unrealizable"),
      regressingMessage);

  auto duplicateEdgeGraph = transitionGraph;
  duplicateEdgeGraph.transitions.push_back(transition);
  llvm::Error duplicateEdgeError =
      pnr::validateResourceTimeTransitionGraph(duplicateEdgeGraph);
  deployment::test::require(test, static_cast<bool>(duplicateEdgeError),
                            "transition graph accepted a duplicate edge");
  llvm::consumeError(std::move(duplicateEdgeError));

  auto wrongEntry = runtime::ResourceTimeTransitionSelectionSession::create(
      transitionGraph, child, artifacts, blobs);
  expectSelectionError(
      test, std::move(wrongEntry),
      runtime::ResourceTimeSelectionErrorReason::EntryDeploymentMismatch);

  auto premature =
      take(test, runtime::ResourceTimeTransitionSelectionSession::create(
                     transitionGraph, parent, artifacts, blobs));
  expectSelectionError(
      test, premature.completeRoot(precedingRoot, std::nullopt),
      runtime::ResourceTimeSelectionErrorReason::CompletionBeforeStart);
  startRoot(premature, root);
  expectSelectionError(
      test, premature.startRoot(root),
      runtime::ResourceTimeSelectionErrorReason::DuplicateStart);
  expectSelectionError(
      test, premature.completeRoot(root, transition.child),
      runtime::ResourceTimeSelectionErrorReason::TransitionUnavailable);
  expectSelectionError(
      test, premature.joinMappedRoots(),
      runtime::ResourceTimeSelectionErrorReason::IncompleteMappedRootJoin);
  deployment::test::require(test,
                            premature.currentEndpoint() == transition.parent &&
                                premature.completedRoots().empty() &&
                                premature.activeRoots() == std::vector{root} &&
                                premature.replay().size() == 1,
                            "rejected transition mutated selector state");

  const auto implementations =
      implementationIdentities(test, parent, artifacts, blobs);
  auto provider = take(test, runtime::createInProcessRuntimeProvider(
                                 {{implementations, std::nullopt, {}}}));
  auto loaded = take(
      test, runtime::loadDeployment(parent, {provider, 0}, artifacts, blobs));
  startRoot(premature, precedingRoot);
  (void)take(test, premature.completeRootAndActivate(precedingRoot,
                                                     std::nullopt, loaded));
  const auto unpreparedReason = expectActivationReplacementError(
      test, premature.completeRootAndActivate(root, transition.child, loaded));
  deployment::test::require(
      test,
      unpreparedReason == runtime::RuntimeActivationReplacementErrorReason::
                              PreparationFailed &&
          premature.currentEndpoint() == transition.parent &&
          premature.completedRoots().size() == 1 &&
          premature.completedRoots().front() == precedingRoot &&
          provider->activeDeployment(0) == parent.reference() &&
          provider->statistics().activationPreparationCount == 0,
      "unprepared activation did not fail before changing provider state");

  auto selector = take(
      test, runtime::ResourceTimeTransitionSelectionSession::createPrepared(
                transitionGraph, loaded, artifacts, blobs));
  deployment::test::require(
      test,
      provider->statistics().activationPreparationCount == 1 &&
          provider->statistics().activationReplacementCount == 0,
      "resource-time child endpoint was not prepared before execution");
  startRoot(selector, precedingRoot);
  auto stayed = take(test, selector.completeRootAndActivate(
                               precedingRoot, std::nullopt, loaded));
  deployment::test::require(
      test,
      !stayed && selector.currentEndpoint() == transition.parent &&
          selector.completedRoots().size() == 1 &&
          selector.completedRoots().front() == precedingRoot &&
          loaded.deployment().reference() == parent.reference(),
      "explicit stay changed the Mapping endpoint or completion frontier");

  auto foreignProvider = take(test, runtime::createInProcessRuntimeProvider(
                                        {{implementations, std::nullopt, {}}}));
  auto foreignLoaded =
      take(test, runtime::loadDeployment(parent, {foreignProvider, 0},
                                         artifacts, blobs));
  startRoot(selector, root);
  const auto foreignReason = expectActivationReplacementError(
      test,
      selector.completeRootAndActivate(root, transition.child, foreignLoaded));
  deployment::test::require(
      test,
      foreignReason == runtime::RuntimeActivationReplacementErrorReason::
                           PreparationFailed &&
          selector.currentEndpoint() == transition.parent &&
          selector.completedRoots().size() == 1 &&
          selector.completedRoots().front() == precedingRoot &&
          loaded.deployment().reference() == parent.reference() &&
          foreignLoaded.deployment().reference() == parent.reference() &&
          provider->activeDeployment(0) == parent.reference() &&
          foreignProvider->activeDeployment(0) == parent.reference() &&
          provider->statistics().activationReplacementCount == 0 &&
          foreignProvider->statistics().activationReplacementCount == 0,
      "prepared selector crossed its exact loaded Deployment association");

  auto selected = take(
      test, selector.completeRootAndActivate(root, transition.child, loaded));
  deployment::test::require(
      test,
      selected && selected->parent == transition.parent &&
          selected->child == transition.child &&
          selector.currentEndpoint() == transition.child &&
          loaded.deployment().reference() == child.reference() &&
          provider->activeDeployment(0) == child.reference() &&
          provider->statistics().activationPreparationCount == 1 &&
          provider->statistics().activationReplacementCount == 1 &&
          provider->statistics().activationCount == 1 &&
          provider->statistics().executableRegistrationCount == 1 &&
          provider->statistics().resetCount == 1,
      "selector did not atomically activate the exact preverified edge");
  if (llvm::Error error = selector.joinMappedRoots())
    deployment::test::fail(test, llvm::toString(std::move(error)));
  deployment::test::require(
      test, selector.mappedRootsJoined() && selector.replay().size() == 5,
      "selector did not join the complete mapped-root inventory");

  auto replayed = take(
      test, runtime::ResourceTimeTransitionSelectionSession::replay(
                transitionGraph, parent, artifacts, blobs, selector.replay()));
  deployment::test::require(
      test,
      replayed.mappedRootsJoined() &&
          replayed.currentEndpoint() == transition.child &&
          replayed.completedRoots() == selector.completedRoots(),
      "selector replay diverged from the accepted transition sequence");

  auto eventProvider = take(test, runtime::createInProcessRuntimeProvider(
                                      {{implementations, std::nullopt, {}}}));
  auto eventLoaded =
      take(test, runtime::loadDeployment(parent, {eventProvider, 0}, artifacts,
                                         blobs));
  auto eventSession =
      take(test,
           application::ApplicationResourceTimeExecutionSession::createPrepared(
               transitionGraph, eventLoaded, artifacts, blobs));
  const auto precedingStart =
      take(test,
           eventSession.apply(
               {dataflow::rootThreadStartEventFamily(precedingRoot), 1, {1, 0}},
               eventLoaded));
  const auto noEdge = take(
      test,
      eventSession.apply(
          {dataflow::rootThreadCompletionEventFamily(precedingRoot), 1, {2, 0}},
          eventLoaded));
  const auto selectedStart =
      take(test, eventSession.apply(
                     {dataflow::rootThreadStartEventFamily(root), 2, {3, 0}},
                     eventLoaded));
  const auto selectedEvent = take(
      test, eventSession.apply(
                {dataflow::rootThreadCompletionEventFamily(root), 2, {4, 0}},
                eventLoaded));
  deployment::test::require(
      test,
      precedingStart.outcome ==
              application::ApplicationResourceTimeEventOutcome::RootStarted &&
          noEdge.outcome == application::ApplicationResourceTimeEventOutcome::
                                NoLegalTransition &&
          !noEdge.transition && noEdge.parent == transition.parent &&
          noEdge.current == transition.parent && noEdge.activeRoots.empty() &&
          noEdge.completedRoots == std::vector{precedingRoot} &&
          selectedStart.outcome ==
              application::ApplicationResourceTimeEventOutcome::RootStarted &&
          selectedEvent.outcome ==
              application::ApplicationResourceTimeEventOutcome::SelectedChild &&
          selectedEvent.transition &&
          selectedEvent.transition->parent == transition.parent &&
          selectedEvent.transition->child == transition.child &&
          selectedEvent.transition->beforeActive.size() == 1 &&
          selectedEvent.transition->beforeActive.front().region == root &&
          selectedEvent.transition->beforeActive.front().resources ==
              resources &&
          selectedEvent.transition->afterActive.empty() &&
          selectedEvent.transition->logicalMemories.empty() &&
          selectedEvent.transition->reprogrammingTimePicoseconds == 0 &&
          selectedEvent.transition->migrationTimePicoseconds == 0 &&
          selectedEvent.current == transition.child &&
          selectedEvent.activeRoots.empty() &&
          selectedEvent.completedRoots.size() == 2 &&
          llvm::is_contained(selectedEvent.completedRoots, precedingRoot) &&
          llvm::is_contained(selectedEvent.completedRoots, root) &&
          !eventSession.joined() && eventSession.events().size() == 4 &&
          eventLoaded.deployment().reference() == child.reference() &&
          eventProvider->activeDeployment(0) == child.reference() &&
          eventProvider->statistics().activationPreparationCount == 1 &&
          eventProvider->statistics().activationReplacementCount == 1,
      "Application event session lost its typed stay or selected child");
  if (llvm::Error error = eventSession.joinMappedRoots())
    deployment::test::fail(test, llvm::toString(std::move(error)));
  deployment::test::require(test, eventSession.joined(),
                            "Application event session did not join roots");

  auto firstCycleDraft = draft;
  firstCycleDraft.trigger =
      dataflow::rootThreadCompletionEventFamily(precedingRoot);
  firstCycleDraft.beforeActive = {{precedingRoot, precedingResources}};
  firstCycleDraft.afterActive = {{root, resourcesFor(childContexts, root)}};
  firstCycleDraft.completedBefore.clear();
  const auto firstCycle =
      take(test, pnr::finalizeResourceTimeTransition(std::move(firstCycleDraft),
                                                     artifacts, blobs));
  auto secondCycleDraft = draft;
  secondCycleDraft.parent = transition.child;
  secondCycleDraft.child = transition.parent;
  secondCycleDraft.beforeActive = {{root, resourcesFor(childContexts, root)}};
  const auto secondCycle =
      take(test, pnr::finalizeResourceTimeTransition(
                     std::move(secondCycleDraft), artifacts, blobs));
  const pnr::ResourceTimeTransitionGraph cycleGraph{
      transition.parent,
      {transition.parent, transition.child},
      {firstCycle, secondCycle}};
  if (llvm::Error error =
          pnr::verifyResourceTimeTransitionGraph(cycleGraph, artifacts, blobs))
    deployment::test::fail(test, llvm::toString(std::move(error)));
  auto cycleProvider = take(test, runtime::createInProcessRuntimeProvider(
                                      {{implementations, std::nullopt, {}}}));
  {
    auto cycleLoaded =
        take(test, runtime::loadDeployment(parent, {cycleProvider, 0},
                                           artifacts, blobs));
    auto cycleSelector = take(
        test, runtime::ResourceTimeTransitionSelectionSession::createPrepared(
                  cycleGraph, cycleLoaded, artifacts, blobs));
    startRoot(cycleSelector, precedingRoot);
    (void)take(test, cycleSelector.completeRootAndActivate(
                         precedingRoot, firstCycle.child, cycleLoaded));
    expectSelectionError(
        test,
        cycleSelector.completeRootAndActivate(root, secondCycle.child,
                                              cycleLoaded),
        runtime::ResourceTimeSelectionErrorReason::CompletionBeforeStart);
    startRoot(cycleSelector, root);
    (void)take(test, cycleSelector.completeRootAndActivate(
                         root, secondCycle.child, cycleLoaded));
    if (llvm::Error error = cycleSelector.joinMappedRoots())
      deployment::test::fail(test, llvm::toString(std::move(error)));
    deployment::test::require(
        test,
        cycleSelector.currentEndpoint() == transition.parent &&
            cycleLoaded.deployment().reference() == parent.reference() &&
            cycleProvider->activeDeployment(0) == parent.reference() &&
            cycleProvider->statistics().activationPreparationCount == 2 &&
            cycleProvider->statistics().activationReplacementCount == 2,
        "prepared endpoint handles did not support a verified endpoint "
        "revisit");
  }
  deployment::test::require(
      test,
      cycleProvider->preparedActivationCount(0) == 0 &&
          cycleProvider->statistics().activationDiscardCount == 2 &&
          cycleProvider->statistics().resetCount == 2 &&
          cycleProvider->statistics().leaseReleaseCount == 1,
      "loaded Deployment teardown did not discard prepared activations");

  runtime::InProcessRuntimeFailurePlan discardFailure;
  discardFailure.activationDiscardFailures = 1;
  auto discardProvider = take(
      test, runtime::createInProcessRuntimeProvider(
                {{implementations, std::nullopt, std::move(discardFailure)}}));
  {
    auto discardLoaded =
        take(test, runtime::loadDeployment(parent, {discardProvider, 0},
                                           artifacts, blobs));
    auto discardSelector = take(
        test, runtime::ResourceTimeTransitionSelectionSession::createPrepared(
                  transitionGraph, discardLoaded, artifacts, blobs));
    deployment::test::require(
        test,
        discardProvider->preparedActivationCount(0) == 1 &&
            discardSelector.currentEndpoint() == transition.parent,
        "discard fallback fixture did not retain one prepared activation");
  }
  deployment::test::require(
      test,
      discardProvider->preparedActivationCount(0) == 0 &&
          discardProvider->statistics().activationDiscardCount == 0 &&
          discardProvider->statistics().resetCount == 2 &&
          discardProvider->statistics().leaseReleaseCount == 1 &&
          !discardProvider->isQuarantined(0),
      "reset did not recover from prepared activation discard failure");

  auto cancelled =
      take(test, runtime::ResourceTimeTransitionSelectionSession::create(
                     transitionGraph, parent, artifacts, blobs));
  if (llvm::Error error = cancelled.cancel())
    deployment::test::fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = cancelled.cancel())
    deployment::test::fail(test, llvm::toString(std::move(error)));
  deployment::test::require(
      test, cancelled.cancelled() && cancelled.replay().size() == 1,
      "selector cancellation was not idempotent");
  expectSelectionError(
      test, cancelled.completeRoot(precedingRoot, std::nullopt),
      runtime::ResourceTimeSelectionErrorReason::InvalidLifecycle);
  expectSelectionError(
      test, cancelled.startRoot(precedingRoot),
      runtime::ResourceTimeSelectionErrorReason::InvalidLifecycle);
  auto cancelledReplay = take(
      test, runtime::ResourceTimeTransitionSelectionSession::replay(
                transitionGraph, parent, artifacts, blobs, cancelled.replay()));
  deployment::test::require(test, cancelledReplay.cancelled(),
                            "selector cancellation replay was not terminal");

  runtime::InProcessRuntimeFailurePlan preparationFailure;
  preparationFailure.activationPreparationOrdinal = 1;
  auto preparationProvider = take(test, runtime::createInProcessRuntimeProvider(
                                            {{implementations, std::nullopt,
                                              std::move(preparationFailure)}}));
  auto preparationLoaded =
      take(test, runtime::loadDeployment(parent, {preparationProvider, 0},
                                         artifacts, blobs));
  const auto preparationReason = expectActivationReplacementError(
      test, runtime::ResourceTimeTransitionSelectionSession::createPrepared(
                cycleGraph, preparationLoaded, artifacts, blobs));
  deployment::test::require(
      test,
      preparationReason == runtime::RuntimeActivationReplacementErrorReason::
                               PreparationFailed &&
          preparationLoaded.deployment().reference() == parent.reference() &&
          preparationProvider->activeDeployment(0) == parent.reference() &&
          preparationProvider->preparedActivationCount(0) == 0 &&
          preparationProvider->statistics().activationPreparationCount == 1 &&
          preparationProvider->statistics().activationDiscardCount == 1 &&
          preparationProvider->statistics().activationReplacementCount == 0,
      "partial activation preparation did not discard its prepared prefix");
  auto retriedPreparation = take(
      test, runtime::ResourceTimeTransitionSelectionSession::createPrepared(
                cycleGraph, preparationLoaded, artifacts, blobs));
  deployment::test::require(
      test,
      preparationProvider->preparedActivationCount(0) == 2 &&
          preparationProvider->statistics().activationPreparationCount == 3 &&
          preparationProvider->statistics().activationDiscardCount == 1 &&
          retriedPreparation.currentEndpoint() == transition.parent,
      "fully cleaned activation preparation was not retryable");

  runtime::InProcessRuntimeFailurePlan retainedPreparationFailure;
  retainedPreparationFailure.activationPreparationOrdinal = 1;
  retainedPreparationFailure.activationDiscardFailures = 1;
  auto retainedPreparationProvider =
      take(test, runtime::createInProcessRuntimeProvider(
                     {{implementations, std::nullopt,
                       std::move(retainedPreparationFailure)}}));
  {
    auto retainedPreparationLoaded = take(
        test, runtime::loadDeployment(parent, {retainedPreparationProvider, 0},
                                      artifacts, blobs));
    const auto retainedReason = expectActivationReplacementError(
        test, runtime::ResourceTimeTransitionSelectionSession::createPrepared(
                  cycleGraph, retainedPreparationLoaded, artifacts, blobs));
    deployment::test::require(
        test,
        retainedReason == runtime::RuntimeActivationReplacementErrorReason::
                              PreparationFailed &&
            retainedPreparationProvider->preparedActivationCount(0) == 1 &&
            retainedPreparationProvider->statistics()
                    .activationPreparationCount == 1 &&
            retainedPreparationProvider->statistics().activationDiscardCount ==
                0,
        "failed preparation did not retain an undiscarded handle");
    const auto lockedReason = expectActivationReplacementError(
        test, runtime::ResourceTimeTransitionSelectionSession::createPrepared(
                  cycleGraph, retainedPreparationLoaded, artifacts, blobs));
    deployment::test::require(
        test,
        lockedReason == runtime::RuntimeActivationReplacementErrorReason::
                            PreparationFailed &&
            retainedPreparationProvider->preparedActivationCount(0) == 1 &&
            retainedPreparationProvider->statistics()
                    .activationPreparationCount == 1,
        "retained preparation state did not reject another attempt");
  }
  deployment::test::require(
      test,
      retainedPreparationProvider->preparedActivationCount(0) == 0 &&
          retainedPreparationProvider->statistics().resetCount == 2 &&
          retainedPreparationProvider->statistics().leaseReleaseCount == 1 &&
          !retainedPreparationProvider->isQuarantined(0),
      "reset did not clear retained preparation state");

  runtime::InProcessRuntimeFailurePlan replacementFailure;
  replacementFailure.activationReplacementFailures = 1;
  auto failingProvider = take(test, runtime::createInProcessRuntimeProvider(
                                        {{implementations, std::nullopt,
                                          std::move(replacementFailure)}}));
  auto failingLoaded =
      take(test, runtime::loadDeployment(parent, {failingProvider, 0},
                                         artifacts, blobs));
  auto failingSelector = take(
      test, runtime::ResourceTimeTransitionSelectionSession::createPrepared(
                transitionGraph, failingLoaded, artifacts, blobs));
  startRoot(failingSelector, precedingRoot);
  (void)take(test, failingSelector.completeRootAndActivate(
                       precedingRoot, std::nullopt, failingLoaded));
  startRoot(failingSelector, root);
  const auto replacementReason = expectActivationReplacementError(
      test, failingSelector.completeRootAndActivate(root, transition.child,
                                                    failingLoaded));
  deployment::test::require(
      test,
      replacementReason == runtime::RuntimeActivationReplacementErrorReason::
                               ActivationFailed &&
          failingSelector.currentEndpoint() == transition.parent &&
          failingSelector.completedRoots().size() == 1 &&
          failingSelector.completedRoots().front() == precedingRoot &&
          failingSelector.activeRoots() == std::vector{root} &&
          failingSelector.replay().size() == 3 &&
          failingLoaded.deployment().reference() == parent.reference() &&
          failingProvider->activeDeployment(0) == parent.reference() &&
          failingProvider->statistics().activationPreparationCount == 1 &&
          failingProvider->statistics().activationReplacementCount == 0,
      "failed provider activation mutated selector or Deployment state");
  auto retried = take(test, failingSelector.completeRootAndActivate(
                                root, transition.child, failingLoaded));
  deployment::test::require(
      test,
      retried && failingSelector.currentEndpoint() == transition.child &&
          failingSelector.completedRoots().size() == 2 &&
          failingSelector.replay().size() == 4 &&
          failingLoaded.deployment().reference() == child.reference() &&
          failingProvider->activeDeployment(0) == child.reference() &&
          failingProvider->statistics().activationPreparationCount == 1 &&
          failingProvider->statistics().activationReplacementCount == 1,
      "prepared activation handle was not reusable after provider rejection");

  dse::ResourceTimeScheduleHint applicationHint;
  applicationHint.actions = {
      {dse::ResourceTimeActionKind::AdmitRegion,
       precedingRoot,
       0,
       0,
       0,
       {},
       {},
       {}},
      {dse::ResourceTimeActionKind::AdvanceEvent,
       std::nullopt,
       std::nullopt,
       0,
       10,
       {precedingRoot},
       {},
       {}},
      {dse::ResourceTimeActionKind::AdmitRegion, root, 0, 10, 10, {}, {}, {}},
      {dse::ResourceTimeActionKind::AdvanceEvent,
       std::nullopt,
       std::nullopt,
       10,
       20,
       {root},
       {},
       {}}};
  applicationHint.states = {
      {0, {}, {precedingRoot, root}, {}, 20},
      {0,
       {{precedingRoot, 0, {precedingResources.size()}, 10}},
       {root},
       {},
       20},
      {10, {}, {root}, {precedingRoot}, 20},
      {10, {{root, 0, {resources.size()}, 20}}, {}, {precedingRoot}, 20},
      {20, {}, {}, {precedingRoot, root}, 20}};
  applicationHint.estimatedMakespanPicoseconds = 20;
  applicationHint.optimisticMakespanLowerBoundPicoseconds = 20;
  applicationHint.peakConcurrentRegions = 1;
  applicationHint.totalAllocatedResourceTime =
      10 * (resources.size() + precedingResources.size());
  applicationHint.support = dse::ResourceTimeEstimateSupport::Exact;
  const std::vector<dse::ResourceTimeRegionFeature> applicationRegions = {
      {precedingRoot,
       {},
       {{{precedingResources.size()},
         10,
         std::nullopt,
         std::nullopt,
         0,
         0,
         0,
         dse::ResourceTimeEstimateSupport::Exact}},
       0,
       false,
       {}},
      {root,
       {},
       {{{resources.size()},
         10,
         std::nullopt,
         std::nullopt,
         0,
         0,
         0,
         dse::ResourceTimeEstimateSupport::Exact}},
       0,
       false,
       {}}};
  const std::vector<dse::ResourceTimeRegionResourceBound> applicationBounds = {
      {precedingRoot, precedingResources.size(),
       dse::ResourceTimeEstimateSupport::Exact, precedingResources.size(),
       dse::ResourceTimeEstimateSupport::Exact},
      {root, resources.size(), dse::ResourceTimeEstimateSupport::Exact,
       resources.size(), dse::ResourceTimeEstimateSupport::Exact}};
  const std::array applicationMappings = {parentMapping.reference(),
                                          childMapping.reference()};
  const std::array applicationEndpoints = {
      dse::ResourceTimeMappingDeploymentEndpoint{parentMapping.reference(),
                                                 parent.reference()},
      dse::ResourceTimeMappingDeploymentEndpoint{childMapping.reference(),
                                                 child.reference()}};
  const auto applicationFunnel =
      take(test, dse::verifyResourceTimeMappingFinalists(
                     {applicationHint}, applicationRegions, applicationBounds,
                     applicationMappings, artifacts, {},
                     dse::ResourceTimeConcurrencyBounds{
                         1, 1, dse::ResourceTimeEstimateSupport::Exact},
                     &blobs, applicationEndpoints));
  const auto *verifiedApplication =
      std::get_if<dse::VerifiedResourceTimeSpectrum>(
          &applicationFunnel.verification);
  if (!verifiedApplication) {
    const auto *incompleteApplication =
        std::get_if<dse::IncompleteResourceTimeSpectrum>(
            &applicationFunnel.verification);
    const std::string diagnostic =
        incompleteApplication
            ? (llvm::Twine("application materializer remained incomplete: ") +
               incompleteApplication->diagnostic)
                  .str()
            : "application materializer was not verified";
    deployment::test::fail(test, diagnostic);
  }
  deployment::test::require(
      test,
      verifiedApplication->scenarios.size() == 1 &&
          verifiedApplication->scenarios.front().systemMappings.size() == 2 &&
          verifiedApplication->scenarios.front().transitionGraph &&
          verifiedApplication->scenarios.front()
                  .transitionGraph->transitions.size() == 1 &&
          verifiedApplication->scenarios.front()
                  .transitions.transitions.front()
                  .parent.deployment == parent.reference() &&
          verifiedApplication->scenarios.front()
                  .transitions.transitions.front()
                  .child.deployment == child.reference(),
      "application materializer lost its verified Mapping/Deployment edge");

  const std::array childMappingOnly = {childMapping.reference()};
  const auto childFunnel =
      take(test, dse::verifyResourceTimeMappingFinalists(
                     {applicationHint}, applicationRegions, applicationBounds,
                     childMappingOnly, artifacts, {},
                     dse::ResourceTimeConcurrencyBounds{
                         1, 1, dse::ResourceTimeEstimateSupport::Exact},
                     &blobs));
  const auto *verifiedChild =
      std::get_if<dse::VerifiedResourceTimeSpectrum>(&childFunnel.verification);
  deployment::test::require(
      test,
      verifiedChild &&
          llvm::any_of(verifiedChild->scenarios.front().states,
                       [](const auto &state) { return !state.active.empty(); }),
      "child Mapping has no independently verified active-work schedule");

  const std::array parentEndpointOnly = {
      dse::ResourceTimeMappingDeploymentEndpoint{parentMapping.reference(),
                                                 parent.reference()}};
  expectError(test,
              dse::verifyResourceTimeMappingFinalists(
                  {applicationHint}, applicationRegions, applicationBounds,
                  applicationMappings, artifacts, {},
                  dse::ResourceTimeConcurrencyBounds{
                      1, 1, dse::ResourceTimeEstimateSupport::Exact},
                  &blobs, parentEndpointOnly),
              "must cover every Mapping finalist exactly once");

  const std::array mismatchedEndpoints = {
      dse::ResourceTimeMappingDeploymentEndpoint{parentMapping.reference(),
                                                 child.reference()},
      dse::ResourceTimeMappingDeploymentEndpoint{childMapping.reference(),
                                                 parent.reference()}};
  expectError(test,
              dse::verifyResourceTimeMappingFinalists(
                  {applicationHint}, applicationRegions, applicationBounds,
                  applicationMappings, artifacts, {},
                  dse::ResourceTimeConcurrencyBounds{
                      1, 1, dse::ResourceTimeEstimateSupport::Exact},
                  &blobs, mismatchedEndpoints),
              "does not select its paired SystemMapping");

  constexpr llvm::StringLiteral mutationOwner =
      "loom.test.resource_time_delta_mutation";
  constexpr llvm::StringLiteral mutationValue = "mutated";
  const ComponentViewDigest mutation = take(
      test, computeComponentViewDigest(
                {reinterpret_cast<const std::uint8_t *>(mutationOwner.data()),
                 mutationOwner.size()},
                {reinterpret_cast<const std::uint8_t *>(mutationValue.data()),
                 mutationValue.size()}));
  const auto requireClosureFailure = [&](pnr::ResourceTimeTransition candidate,
                                         llvm::StringRef marker) {
    llvm::Error error =
        pnr::verifyResourceTimeTransitionClosure(candidate, artifacts, blobs);
    deployment::test::require(test, static_cast<bool>(error),
                              "transition mutation passed closure");
    const std::string message = llvm::toString(std::move(error));
    deployment::test::require(
        test, llvm::StringRef(message).contains(marker),
        (llvm::Twine("expected '") + marker + "': " + message).str());
  };
  const auto requireFinalizationFailure =
      [&](pnr::ResourceTimeTransition candidate, llvm::StringRef marker) {
        auto finalized = pnr::finalizeResourceTimeTransition(
            std::move(candidate), artifacts, blobs);
        deployment::test::require(test, !finalized,
                                  "transition mutation was finalized");
        const std::string message = llvm::toString(finalized.takeError());
        deployment::test::require(
            test, llvm::StringRef(message).contains(marker),
            (llvm::Twine("expected '") + marker + "': " + message).str());
      };
  auto authoredDelta = draft;
  authoredDelta.resourceDeltaDigest = mutation;
  requireFinalizationFailure(std::move(authoredDelta),
                             "authored delta digests");
  auto authoredStatus = draft;
  authoredStatus.status = pnr::ResourceTimeTransitionStatus::Verified;
  requireFinalizationFailure(std::move(authoredStatus),
                             "authored terminal status");
  auto authoredCost = draft;
  authoredCost.migrationTimePicoseconds = 0;
  requireFinalizationFailure(std::move(authoredCost),
                             "authored cost components");
  auto authoredReprogrammingCost = draft;
  authoredReprogrammingCost.reprogrammingTimePicoseconds = 0;
  requireFinalizationFailure(std::move(authoredReprogrammingCost),
                             "authored cost components");
  auto missingSafePoint = draft;
  missingSafePoint.safePoint.reset();
  requireFinalizationFailure(std::move(missingSafePoint),
                             "completion safe point");
  auto wrongSafePointOwner = draft;
  wrongSafePointOwner.safePoint->artifact = parent.reference();
  requireFinalizationFailure(std::move(wrongSafePointOwner),
                             "must be owned by Canonical Dataflow");
  auto survivingRegion = draft;
  survivingRegion.afterActive = survivingRegion.beforeActive;
  requireFinalizationFailure(std::move(survivingRegion), "remains active");
  auto authoredCorrespondence = draft;
  authoredCorrespondence.logicalMemories.push_back(
      {dataflow::LogicalMemoryRootRef{dataflowReference.artifact,
                                      dataflow::LogicalMemoryRootId(0)},
       *transition.resourceDeltaDigest,
       *transition.resourceDeltaDigest,
       pnr::ResourceTimeLiveStateMigration::RetainedInPlace,
       0});
  requireFinalizationFailure(std::move(authoredCorrespondence),
                             "authored live-state correspondence");
  const FinalizedDeployment multiRootParent =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const FinalizedDeployment multiRootChild =
      deployment::test::buildRetargetedMinimalDeployment(test, artifacts, blobs,
                                                         tree);
  const auto multiRootParentMapping =
      take(test, loom::mapping::importSystemMapping(
                     multiRootParent.deployment().systemMapping(), artifacts));
  const ArtifactRootReference multiRootDataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      multiRootParentMapping.view().dataflowIdentity()};
  auto multiRootDataflowArtifact =
      take(test, dataflow::importCanonicalDataflow(multiRootDataflowReference,
                                                   artifacts));
  auto multiRootDataflow = take(test, multiRootDataflowArtifact.view());
  const dataflow::RootThreadLaunchRef omittedRoot =
      multiRootDataflow.rootThreadLaunches().front().ref;
  const auto multiRootContexts =
      take(test, loom::mapping::projectSystemExecutionContexts(
                     multiRootDataflow,
                     multiRootParentMapping.view().executionBindings()));
  std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef>
      omittedRootResources;
  const auto appendOmittedRootCore =
      [&](loom::fabric::AccCoreOccurrenceRef core) {
        omittedRootResources.push_back(
            take(test, loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
                           loom::fabric::FabricInventoryOwnerRef::of(core))));
      };
  for (const auto &domain : multiRootContexts.instructionDomains)
    if (domain.root == omittedRoot)
      appendOmittedRootCore(domain.context.accCore);
  for (const auto &domain : multiRootContexts.spatialDomains)
    if (domain.graph.rootThreadLaunch == omittedRoot)
      appendOmittedRootCore(domain.context.accCore);
  auto nonterminalCompletion = draft;
  nonterminalCompletion.trigger =
      dataflow::rootThreadCompletionEventFamily(omittedRoot);
  nonterminalCompletion.safePoint = pnr::ResourceTimeSafePointReference{
      multiRootDataflowReference, pnr::ResourceTimeSafePointKind::Completion};
  nonterminalCompletion.parent = {multiRootParentMapping.reference(),
                                  multiRootParent.reference()};
  nonterminalCompletion.child = {multiRootChild.deployment().systemMapping(),
                                 multiRootChild.reference()};
  nonterminalCompletion.beforeActive = {{omittedRoot, omittedRootResources}};
  nonterminalCompletion.completedBefore.clear();
  const auto nonterminalTransition =
      take(test, pnr::finalizeResourceTimeTransition(
                     std::move(nonterminalCompletion), artifacts, blobs));
  if (llvm::Error error = pnr::verifyResourceTimeTransitionClosure(
          nonterminalTransition, artifacts, blobs))
    deployment::test::fail(
        test, (llvm::Twine(
                   "nonterminal completion edge failed independent closure: ") +
               llvm::toString(std::move(error)))
                  .str());

  auto repeatedCompletion = draft;
  repeatedCompletion.completedBefore.push_back(precedingRoot);
  requireFinalizationFailure(std::move(repeatedCompletion), "duplicate region");

  const FinalizedDeployment changedProgramming =
      deployment::test::buildRetargetedSharedProgrammingEndpointDeployment(
          test, artifacts, blobs, tree);
  deployment::test::require(
      test,
      changedProgramming.deployment().systemMapping() ==
          childMapping.reference(),
      "changed-programming fixture selected another child Mapping");
  auto changedProgrammingDraft = draft;
  changedProgrammingDraft.child.deployment = changedProgramming.reference();
  const auto changedProgrammingTransition =
      take(test, pnr::finalizeResourceTimeTransition(
                     std::move(changedProgrammingDraft), artifacts, blobs));
  deployment::test::require(
      test,
      changedProgrammingTransition.reprogrammingTimePicoseconds.value_or(0) !=
          0,
      "changed configuration words have no provider-derived cost");

  auto wrongTrigger = transition;
  wrongTrigger.trigger = dataflow::rootThreadStartEventFamily(root);
  requireClosureFailure(std::move(wrongTrigger), "completion safe point");
  auto wrongResourceDelta = transition;
  wrongResourceDelta.resourceDeltaDigest = mutation;
  requireClosureFailure(std::move(wrongResourceDelta), "resource delta");
  auto wrongConfigurationDelta = transition;
  wrongConfigurationDelta.configurationDeltaDigest = mutation;
  requireClosureFailure(std::move(wrongConfigurationDelta),
                        "configuration delta");
  auto wrongRouteDelta = transition;
  wrongRouteDelta.routeDeltaDigest = mutation;
  requireClosureFailure(std::move(wrongRouteDelta), "route delta");
  auto wrongReprogrammingCost = transition;
  wrongReprogrammingCost.reprogrammingTimePicoseconds = 1;
  requireClosureFailure(std::move(wrongReprogrammingCost),
                        "changed-word projection");
  auto wrongMigrationCost = transition;
  wrongMigrationCost.migrationTimePicoseconds = 1;
  requireClosureFailure(std::move(wrongMigrationCost),
                        "migration time disagrees");
  auto wrongParentDeployment = transition;
  wrongParentDeployment.parent.deployment = child.reference();
  requireClosureFailure(std::move(wrongParentDeployment),
                        "parent Deployment does not select");
  auto wrongParentMapping = transition;
  wrongParentMapping.parent.mapping = childMapping.reference();
  requireClosureFailure(std::move(wrongParentMapping),
                        "parent Deployment does not select");
  auto wrongChildDeployment = transition;
  wrongChildDeployment.child.deployment = parent.reference();
  requireClosureFailure(std::move(wrongChildDeployment),
                        "child Deployment does not select");
  auto wrongStatus = transition;
  wrongStatus.status = pnr::ResourceTimeTransitionStatus::ProofNotEstablished;
  requireClosureFailure(std::move(wrongStatus), "requires a verified edge");
  auto wrongBeforeRegion = transition;
  wrongBeforeRegion.beforeActive.front().region = precedingRoot;
  wrongBeforeRegion.completedBefore = {root};
  requireClosureFailure(std::move(wrongBeforeRegion),
                        "completion safe point is not the completion event");

  auto explicitSafePoint = transition;
  explicitSafePoint.safePoint->kind = pnr::ResourceTimeSafePointKind::Explicit;
  explicitSafePoint.safePoint->artifact = parent.reference();
  llvm::Error explicitSafePointError = pnr::verifyResourceTimeTransitionClosure(
      explicitSafePoint, artifacts, blobs);
  deployment::test::require(
      test, static_cast<bool>(explicitSafePointError),
      "transition accepted an unproven explicit safe point");
  const std::string explicitSafePointMessage =
      llvm::toString(std::move(explicitSafePointError));
  deployment::test::require(test,
                            llvm::StringRef(explicitSafePointMessage)
                                .contains("typed compiler proof importer"),
                            explicitSafePointMessage);

  auto wrongEndpoint = transition;
  wrongEndpoint.child.mapping = ArtifactRootReference{
      loom::mapping::mappingArtifactSchema.identity.str(),
      loom::mapping::mappingArtifactSchema.version, child.reference().artifact};
  llvm::Error wrongEndpointError =
      pnr::verifyResourceTimeTransitionClosure(wrongEndpoint, artifacts, blobs);
  deployment::test::require(test, static_cast<bool>(wrongEndpointError),
                            "transition accepted a mismatched Deployment");
  const std::string wrongEndpointMessage =
      llvm::toString(std::move(wrongEndpointError));
  deployment::test::require(test,
                            llvm::StringRef(wrongEndpointMessage)
                                .contains("child Deployment does not select"),
                            wrongEndpointMessage);

  auto wrongAllocation = transition;
  wrongAllocation.beforeActive.front().resources.front() =
      take(test, loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
                     loom::fabric::FabricInventoryOwnerRef::of(
                         loom::fabric::HostCoreOccurrenceRef(4096))));
  llvm::Error wrongAllocationError = pnr::verifyResourceTimeTransitionClosure(
      wrongAllocation, artifacts, blobs);
  deployment::test::require(test, static_cast<bool>(wrongAllocationError),
                            "transition accepted a foreign allocation");
  llvm::consumeError(std::move(wrongAllocationError));
}

} // namespace

int main() {
  exactClosureRoundTripsAndRejectsStaleChild();
  finalLinkedProgramMustMatchHostTarget();
  resourceTimeTransitionRequiresExactDeploymentClosure();
  return 0;
}

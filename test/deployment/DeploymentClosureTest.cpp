#include "DeploymentTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"
#include "DSE/ResourceTimeSpectrum.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>
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
  const dataflow::RootThreadLaunchRef root =
      dataflow.rootThreadLaunches().front().ref;
  const dataflow::RootThreadLaunchRef precedingRoot =
      dataflow.rootThreadLaunches()[1].ref;
  const auto contexts =
      take(test, loom::mapping::projectSystemExecutionContexts(
                     dataflow, parentMapping.view().executionBindings()));
  const auto resourcesFor = [&](dataflow::RootThreadLaunchRef selected) {
    std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef> result;
    const auto appendCore = [&](loom::fabric::AccCoreOccurrenceRef core) {
      result.push_back(
          take(test, loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
                         loom::fabric::FabricInventoryOwnerRef::of(core))));
    };
    for (const auto &domain : contexts.instructionDomains)
      if (domain.root == selected)
        appendCore(domain.context.accCore);
    for (const auto &domain : contexts.spatialDomains)
      if (domain.graph.rootThreadLaunch == selected)
        appendCore(domain.context.accCore);
    return result;
  };
  const auto resources = resourcesFor(root);
  const auto precedingResources = resourcesFor(precedingRoot);
  deployment::test::require(test, !resources.empty(),
                            "resource-time fixture root has no AccCore");
  deployment::test::require(test, !precedingResources.empty(),
                            "resource-time fixture prior root has no AccCore");

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
      {},
      std::nullopt,
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
                     &blobs, applicationEndpoints,
                     dse::ResourceTimeMappingTransitionCandidate{
                         parentMapping.reference(), childMapping.reference()}));
  const auto *verifiedApplication =
      std::get_if<dse::VerifiedResourceTimeSpectrum>(
          &applicationFunnel.verification);
  deployment::test::require(
      test,
      verifiedApplication && verifiedApplication->scenarios.size() == 1 &&
          verifiedApplication->scenarios.front().systemMappings.size() == 2 &&
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
  const auto incompleteApplicationFunnel =
      take(test, dse::verifyResourceTimeMappingFinalists(
                     {applicationHint}, applicationRegions, applicationBounds,
                     applicationMappings, artifacts, {},
                     dse::ResourceTimeConcurrencyBounds{
                         1, 1, dse::ResourceTimeEstimateSupport::Exact},
                     &blobs, parentEndpointOnly,
                     dse::ResourceTimeMappingTransitionCandidate{
                         parentMapping.reference(), childMapping.reference()}));
  const auto *incompleteApplication =
      std::get_if<dse::IncompleteResourceTimeSpectrum>(
          &incompleteApplicationFunnel.verification);
  deployment::test::require(
      test,
      incompleteApplication &&
          incompleteApplication->reason ==
              dse::ResourceTimeSpectrumIncompleteReason::ProofNotEstablished,
      "missing child Deployment did not remain typed proof-not-established");

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
  requireFinalizationFailure(std::move(survivingRegion),
                             "not globally quiescent");
  auto liveWork = draft;
  liveWork.beforeLiveWork = {parent.reference()};
  requireFinalizationFailure(std::move(liveWork), "unproved live or token");
  auto afterLiveWork = draft;
  afterLiveWork.afterLiveWork = {child.reference()};
  requireFinalizationFailure(std::move(afterLiveWork),
                             "unproved live or token");
  auto tokenCorrespondence = draft;
  tokenCorrespondence.tokenLiveStateCorrespondence = dataflowReference;
  requireFinalizationFailure(std::move(tokenCorrespondence),
                             "unproved live or token");

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
  auto omittedActiveRoot = draft;
  omittedActiveRoot.trigger =
      dataflow::rootThreadCompletionEventFamily(omittedRoot);
  omittedActiveRoot.safePoint = pnr::ResourceTimeSafePointReference{
      multiRootDataflowReference, pnr::ResourceTimeSafePointKind::Completion};
  omittedActiveRoot.parent = {multiRootParentMapping.reference(),
                              multiRootParent.reference()};
  omittedActiveRoot.child = {multiRootChild.deployment().systemMapping(),
                             multiRootChild.reference()};
  omittedActiveRoot.beforeActive = {{omittedRoot, omittedRootResources}};
  omittedActiveRoot.completedBefore.clear();
  requireFinalizationFailure(std::move(omittedActiveRoot),
                             "canonical root launch inventory");

  auto repeatedCompletion = draft;
  repeatedCompletion.completedBefore.push_back(precedingRoot);
  requireFinalizationFailure(std::move(repeatedCompletion),
                             "canonical root launch inventory");

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
  requireFinalizationFailure(std::move(changedProgrammingDraft),
                             "reprogramming-time owner");

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
                        "nonzero reprogramming time");
  auto wrongMigrationCost = transition;
  wrongMigrationCost.migrationTimePicoseconds = 1;
  requireClosureFailure(std::move(wrongMigrationCost),
                        "live-state migration work");
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

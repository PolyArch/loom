#include "DeploymentTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowEventDerivation.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefs.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"
#include "PnR/System/SystemMappingMigration.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

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
      deployment::test::buildTrustedIdentityDeployment(test, artifacts, blobs,
                                                       tree);
  deployment::test::require(test,
                            parent.deployment().systemMapping() ==
                                child.deployment().systemMapping(),
                            "resource-time fixture changed its SystemMapping");
  deployment::test::require(test, parent.reference() != child.reference(),
                            "resource-time fixture has one Deployment state");

  const auto mapping =
      take(test, loom::mapping::importSystemMapping(
                     parent.deployment().systemMapping(), artifacts));
  const ArtifactRootReference dataflowReference{
      dataflow::canonicalDataflowSchema.identity.str(),
      dataflow::canonicalDataflowSchema.version,
      mapping.view().dataflowIdentity()};
  auto dataflowArtifact = take(
      test, dataflow::importCanonicalDataflow(dataflowReference, artifacts));
  auto dataflow = take(test, dataflowArtifact.view());
  deployment::test::require(test, !dataflow.rootThreadLaunches().empty(),
                            "resource-time fixture has no execution root");
  const dataflow::RootThreadLaunchRef root =
      dataflow.rootThreadLaunches().front().ref;
  const auto contexts =
      take(test, loom::mapping::projectSystemExecutionContexts(
                     dataflow, mapping.view().executionBindings()));
  std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef> resources;
  const auto appendCore = [&](loom::fabric::AccCoreOccurrenceRef core) {
    resources.push_back(
        take(test, loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
                       loom::fabric::FabricInventoryOwnerRef::of(core))));
  };
  for (const auto &domain : contexts.instructionDomains)
    if (domain.root == root)
      appendCore(domain.context.accCore);
  for (const auto &domain : contexts.spatialDomains)
    if (domain.graph.rootThreadLaunch == root)
      appendCore(domain.context.accCore);
  deployment::test::require(test, !resources.empty(),
                            "resource-time fixture root has no AccCore");

  constexpr llvm::StringLiteral deltaOwner =
      "loom.test.resource_time_deployment_delta";
  constexpr llvm::StringLiteral deltaValue = "configuration_and_route";
  const ComponentViewDigest delta =
      take(test, computeComponentViewDigest(
                     {reinterpret_cast<const std::uint8_t *>(deltaOwner.data()),
                      deltaOwner.size()},
                     {reinterpret_cast<const std::uint8_t *>(deltaValue.data()),
                      deltaValue.size()}));
  pnr::ResourceTimeTransition transition{
      dataflow::rootThreadCompletionEventFamily(root),
      pnr::ResourceTimeSafePointReference{
          dataflowReference, pnr::ResourceTimeSafePointKind::Completion},
      {mapping.reference(), parent.reference()},
      {mapping.reference(), child.reference()},
      {{root, resources}},
      {},
      {},
      {},
      std::nullopt,
      delta,
      delta,
      delta,
      1,
      pnr::ResourceTimeTransitionStatus::Verified};
  if (llvm::Error error = pnr::verifyResourceTimeTransitionClosure(
          transition, artifacts, blobs))
    deployment::test::fail(test, llvm::toString(std::move(error)));

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

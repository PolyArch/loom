#include "DeploymentTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace loom;
using namespace loom::deployment;

namespace {

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

} // namespace

int main() {
  exactClosureRoundTripsAndRejectsStaleChild();
  finalLinkedProgramMustMatchHostTarget();
  return 0;
}

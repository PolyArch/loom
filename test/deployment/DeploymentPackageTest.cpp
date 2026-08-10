#include "DeploymentTestSupport.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Deployment/Package.h"

#include "llvm/Support/MemoryBuffer.h"

#include <filesystem>
#include <string>

using namespace loom;
using namespace loom::deployment;

namespace {

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    deployment::test::fail(test, llvm::toString(std::move(error)));
}

void packageReplaysFromEmptyStoresAndPublishesOnce() {
  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree(test);
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const FinalizedDeployment deployment =
      deployment::test::buildMinimalDeployment(test, artifacts, blobs, tree);
  const std::string package = tree.path("package");
  requireSuccess(
      test, publishDeploymentPackage(deployment, package, artifacts, blobs));
  deployment::test::require(test, std::filesystem::is_directory(package),
                            "publisher did not create the package directory");

  auto root = llvm::MemoryBuffer::getFile(package + "/root", false, false);
  if (!root)
    deployment::test::fail(test, root.getError().message());
  deployment::test::require(
      test,
      (*root)->getBuffer() ==
          formatArtifactIdentityHex(deployment.reference().artifact),
      "package root does not contain the exact Deployment identity");
  ArtifactStore packagedArtifacts(package + "/objects");
  BlobStore packagedBlobs(package + "/blobs");
  auto imported = importDeployment(deployment.reference(), packagedArtifacts,
                                   packagedBlobs);
  if (!imported)
    deployment::test::fail(test, llvm::toString(imported.takeError()));
  deployment::test::require(
      test,
      imported->canonicalBytes().bytes() == deployment.canonicalBytes().bytes(),
      "empty-store package import changed Deployment bytes");

  llvm::Error duplicate =
      publishDeploymentPackage(deployment, package, artifacts, blobs);
  if (!duplicate)
    deployment::test::fail(test, "publisher replaced an existing package");
  const std::string message = llvm::toString(std::move(duplicate));
  deployment::test::require(test, llvm::StringRef(message).contains("exists"),
                            message);
  for (const auto &entry : std::filesystem::directory_iterator(tree.path("")))
    deployment::test::require(test,
                              !llvm::StringRef(entry.path().filename().string())
                                   .starts_with(".package.loom-package-"),
                              "publisher retained a staging directory");
}

} // namespace

int main() {
  packageReplaysFromEmptyStoresAndPublishesOnce();
  return 0;
}

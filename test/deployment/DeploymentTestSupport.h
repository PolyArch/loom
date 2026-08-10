#ifndef LOOM_TEST_DEPLOYMENT_DEPLOYMENTTESTSUPPORT_H
#define LOOM_TEST_DEPLOYMENT_DEPLOYMENTTESTSUPPORT_H

#include "Deployment/Deployment.h"

#include "llvm/ADT/StringRef.h"

#include <string>

namespace loom {
class ArtifactStore;
class BlobStore;
namespace runtime {
struct RuntimeProviderDescriptor;
}
} // namespace loom

namespace loom::deployment::test {

class TemporaryTree final {
public:
  explicit TemporaryTree(llvm::StringRef label);
  ~TemporaryTree();

  TemporaryTree(const TemporaryTree &) = delete;
  TemporaryTree &operator=(const TemporaryTree &) = delete;

  std::string path(llvm::StringRef leaf) const;

private:
  std::string root_;
};

[[noreturn]] void fail(llvm::StringRef test, const std::string &message);
void require(llvm::StringRef test, bool condition, llvm::StringRef message);

FinalizedDeployment buildMinimalDeployment(llvm::StringRef test,
                                           ArtifactStore &artifacts,
                                           BlobStore &blobs,
                                           const TemporaryTree &tree);

FinalizedDeployment buildTrustedIdentityDeployment(llvm::StringRef test,
                                                   ArtifactStore &artifacts,
                                                   BlobStore &blobs,
                                                   const TemporaryTree &tree);

FinalizedDeployment buildSharedProgrammingEndpointDeployment(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree);

FinalizedDeployment buildRuntimeProviderDeployment(
    llvm::StringRef test, ArtifactStore &artifacts, BlobStore &blobs,
    const TemporaryTree &tree,
    const runtime::RuntimeProviderDescriptor &provider);

llvm::Expected<FinalizedDeployment>
tryBuildMinimalDeployment(llvm::StringRef test, ArtifactStore &artifacts,
                          BlobStore &blobs, const TemporaryTree &tree,
                          llvm::StringRef finalLinkedTriple);

} // namespace loom::deployment::test

#endif // LOOM_TEST_DEPLOYMENT_DEPLOYMENTTESTSUPPORT_H

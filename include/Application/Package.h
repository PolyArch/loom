#ifndef LOOM_APPLICATION_PACKAGE_H
#define LOOM_APPLICATION_PACKAGE_H

#include "Application/RuntimeManifest.h"
#include "Deployment/Deployment.h"

#include <cstddef>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::application {

/// Entry limit of the invocation-local Artifact import cache that spans one
/// product deployment tail (construction, manifest, package closure) and one
/// staged package import. It bounds retained root-level views, not replay
/// pairs, so a small multiple of the distinct root kinds suffices.
inline constexpr std::size_t productDeploymentImportCacheEntries = 256;

struct ApplicationPackageClosure final {
  std::vector<ArtifactRootReference> artifacts;
  std::vector<BlobDigest> blobs;
};

class ImportedApplicationPackage final {
public:
  const FinalizedApplicationRuntimeManifest &manifest() const {
    return manifest_;
  }
  const deployment::FinalizedDeployment &deployment() const {
    return deployment_;
  }

private:
  ImportedApplicationPackage(FinalizedApplicationRuntimeManifest manifest,
                             deployment::FinalizedDeployment deployment)
      : manifest_(std::move(manifest)), deployment_(std::move(deployment)) {}

  FinalizedApplicationRuntimeManifest manifest_;
  deployment::FinalizedDeployment deployment_;

  friend llvm::Expected<ImportedApplicationPackage>
  importApplicationPackage(llvm::StringRef);
};

/// Publishes the exact Deployment closure plus the Application runtime
/// manifest and every dependency it names. The output remains a flat package
/// so existing execution workspaces can import it without a second store.
llvm::Error
publishApplicationPackage(const FinalizedApplicationRuntimeManifest &manifest,
                          const deployment::FinalizedDeployment &deployment,
                          llvm::StringRef outputPath,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs);

/// Derives the exact object and blob closure owned by an Application package.
/// Consumers may use this projection for strict dependency resolution, but it
/// does not acquire a second identity or authorize additional package roots.
llvm::Expected<ApplicationPackageClosure> deriveApplicationPackageClosure(
    const FinalizedApplicationRuntimeManifest &manifest,
    const deployment::FinalizedDeployment &entryDeployment,
    const ArtifactStore &artifacts, const BlobStore &blobs);

/// Strictly imports an Application package from its embedded stores, replays
/// the manifest and every endpoint Deployment, and rejects missing or
/// unreferenced top-level, object, and blob entries.
llvm::Expected<ImportedApplicationPackage>
importApplicationPackage(llvm::StringRef packagePath);

} // namespace loom::application

#endif // LOOM_APPLICATION_PACKAGE_H

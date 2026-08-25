#ifndef LOOM_APPLICATION_PACKAGE_H
#define LOOM_APPLICATION_PACKAGE_H

#include "Application/RuntimeManifest.h"
#include "Deployment/Deployment.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <utility>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::application {

struct ApplicationDeploymentArtifacts;

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
publishApplicationPackage(const ApplicationDeploymentArtifacts &application,
                          llvm::StringRef outputPath,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs);

/// Strictly imports an Application package from its embedded stores, replays
/// the manifest and every endpoint Deployment, and rejects missing or
/// unreferenced top-level, object, and blob entries.
llvm::Expected<ImportedApplicationPackage>
importApplicationPackage(llvm::StringRef packagePath);

} // namespace loom::application

#endif // LOOM_APPLICATION_PACKAGE_H

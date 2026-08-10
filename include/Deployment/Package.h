#ifndef LOOM_DEPLOYMENT_PACKAGE_H
#define LOOM_DEPLOYMENT_PACKAGE_H

#include "Deployment/Deployment.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::deployment {

/// Publishes the exact Deployment execution closure as a content-addressed
/// directory. The output path is an invocation binding and never contributes
/// to Artifact identity. An existing destination is rejected.
llvm::Error publishDeploymentPackage(const FinalizedDeployment &deployment,
                                     llvm::StringRef outputPath,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs);

} // namespace loom::deployment

#endif // LOOM_DEPLOYMENT_PACKAGE_H

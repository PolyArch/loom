#ifndef LOOM_DEPLOYMENT_PACKAGE_H
#define LOOM_DEPLOYMENT_PACKAGE_H

#include "Deployment/Deployment.h"

#include "Common/BlobDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::deployment {

enum class DeploymentPackageOperation : std::uint8_t {
  SourceClosure,
  StagingWrite,
  IndependentRootImport,
  IndependentClosure,
  StagingEntryValidation,
  AtomicPublish,
};

struct DeploymentPackageOperationStatistics final {
  DeploymentPackageOperation operation;
  std::uint64_t durationNanoseconds = 0;
  std::uint64_t artifactCount = 0;
  std::uint64_t blobCount = 0;
  std::uint64_t fabricImportCacheHits = 0;
  std::uint64_t fabricImportCacheMisses = 0;
  std::uint64_t fabricImportConstructionNanoseconds = 0;
  std::uint64_t fabricImportDeterministicWork = 0;
  std::uint64_t fabricImportRetainedPayloadBytes = 0;
};

void emitDeploymentPackageOperationStatistics(
    const DeploymentPackageOperationStatistics &statistics);

/// The exact content-addressed object closure required to import one
/// Deployment from an otherwise empty ArtifactStore and BlobStore.
class DeploymentPackageClosure final {
public:
  llvm::ArrayRef<ArtifactRootReference> artifacts() const {
    return artifacts_;
  }
  llvm::ArrayRef<BlobDigest> blobs() const { return blobs_; }

private:
  DeploymentPackageClosure(std::vector<ArtifactRootReference> artifacts,
                           std::vector<BlobDigest> blobs)
      : artifacts_(std::move(artifacts)), blobs_(std::move(blobs)) {}

  std::vector<ArtifactRootReference> artifacts_;
  std::vector<BlobDigest> blobs_;

  friend llvm::Expected<DeploymentPackageClosure>
  deriveDeploymentPackageClosure(const FinalizedDeployment &,
                                 const ArtifactStore &, const BlobStore &);
};

/// Strictly derives the package closure from an already imported Deployment.
/// The result is a projection only: every member remains owned by its
/// content-addressed store and no package-path identity is introduced.
llvm::Expected<DeploymentPackageClosure>
deriveDeploymentPackageClosure(const FinalizedDeployment &deployment,
                               const ArtifactStore &artifacts,
                               const BlobStore &blobs);

/// Publishes the exact Deployment execution closure as a content-addressed
/// directory. The output path is an invocation binding and never contributes
/// to Artifact identity. An existing destination is rejected.
llvm::Error publishDeploymentPackage(const FinalizedDeployment &deployment,
                                     llvm::StringRef outputPath,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs);

} // namespace loom::deployment

#endif // LOOM_DEPLOYMENT_PACKAGE_H

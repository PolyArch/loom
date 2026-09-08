#ifndef LOOM_DEPLOYMENT_DEPLOYMENTSPATIALLAUNCHSELECTION_H
#define LOOM_DEPLOYMENT_DEPLOYMENTSPATIALLAUNCHSELECTION_H

#include "Deployment/Deployment.h"
#include "Mapping/Artifact/SystemMappingExecutionProjection.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::deployment {

/// Deployment-owned projection of the exact mapped Spatial launch selected by
/// one concrete workload point. It is transient and introduces no new root.
struct DeploymentSpatialLaunchSelection final {
  ArtifactRootReference hardwareImplementation;
  ArtifactRootReference dataflow;
  ArtifactRootReference spatialMapping;
  mapping::SpatialExecutionContextKey context;
  std::vector<ArtifactRootReference> configurationImages;
};

/// Immutable launch domain derived by strictly importing one Deployment's
/// dependencies. Coordinates select from this projection without repeating
/// hardware and configuration validation. It is local to its owner's import
/// invocation and introduces no persistent state or cross-store cache.
class DeploymentSpatialLaunchProjection final {
public:
  static llvm::Expected<DeploymentSpatialLaunchProjection>
  get(const FinalizedDeployment &deployment, const ArtifactStore &artifacts,
      const BlobStore &blobs);

  const mapping::SystemExecutionContextProjection &contexts() const {
    return contexts_;
  }

  llvm::Expected<DeploymentSpatialLaunchSelection>
  select(dataflow::RootedGraphLaunchRef graph,
         llvm::ArrayRef<std::uint64_t> denseCoordinates) const;

private:
  struct SpatialCoreBinding final {
    fabric::SpatialCoreOccurrenceRef subject;
    ArtifactRootReference implementation;
    std::vector<ArtifactRootReference> configurationImages;
  };

  DeploymentSpatialLaunchProjection(
      ArtifactRootReference dataflow,
      mapping::SystemExecutionContextProjection contexts,
      std::vector<SpatialCoreBinding> bindings)
      : dataflow_(std::move(dataflow)), contexts_(std::move(contexts)),
        bindings_(std::move(bindings)) {}

  ArtifactRootReference dataflow_;
  mapping::SystemExecutionContextProjection contexts_;
  std::vector<SpatialCoreBinding> bindings_;
};

llvm::Expected<DeploymentSpatialLaunchSelection>
resolveDeploymentSpatialLaunchSelection(
    const FinalizedDeployment &deployment, dataflow::RootedGraphLaunchRef graph,
    llvm::ArrayRef<std::uint64_t> denseCoordinates,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::deployment

#endif // LOOM_DEPLOYMENT_DEPLOYMENTSPATIALLAUNCHSELECTION_H

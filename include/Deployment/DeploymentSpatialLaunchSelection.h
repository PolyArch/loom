#ifndef LOOM_DEPLOYMENT_DEPLOYMENTSPATIALLAUNCHSELECTION_H
#define LOOM_DEPLOYMENT_DEPLOYMENTSPATIALLAUNCHSELECTION_H

#include "Deployment/Deployment.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom {
class ArtifactStore;
}

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

llvm::Expected<DeploymentSpatialLaunchSelection>
resolveDeploymentSpatialLaunchSelection(
    const FinalizedDeployment &deployment, dataflow::RootedGraphLaunchRef graph,
    llvm::ArrayRef<std::uint64_t> denseCoordinates,
    const ArtifactStore &artifacts);

} // namespace loom::deployment

#endif // LOOM_DEPLOYMENT_DEPLOYMENTSPATIALLAUNCHSELECTION_H

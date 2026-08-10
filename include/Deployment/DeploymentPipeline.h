#ifndef LOOM_DEPLOYMENT_DEPLOYMENTPIPELINE_H
#define LOOM_DEPLOYMENT_DEPLOYMENTPIPELINE_H

#include "Deployment/Deployment.h"

#include "llvm/Support/Error.h"

#include <vector>

namespace llvm {
class Module;
}

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::deployment {

/// Exact selections made before Deployment finalization. Static logical-memory
/// images are deliberately absent: the pipeline derives them from the selected
/// SystemMapping and the final linked LLVM module.
struct DeploymentPipelineInputs final {
  ArtifactRootReference systemMapping;
  HostProgramLeaf hostProgram;
  std::vector<ArtifactRootReference> instructionCoreBinaries;
  std::vector<DeploymentHardwareBinding> hardwareBindings;
};

llvm::Expected<FinalizedDeployment> buildDeploymentFromLinkedProgram(
    DeploymentPipelineInputs inputs, const llvm::Module &finalLinkedModule,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::deployment

#endif // LOOM_DEPLOYMENT_DEPLOYMENTPIPELINE_H

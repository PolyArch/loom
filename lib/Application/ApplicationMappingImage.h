#ifndef LOOM_APPLICATION_APPLICATIONMAPPINGIMAGE_H
#define LOOM_APPLICATION_APPLICATIONMAPPINGIMAGE_H

#include "Application/Build.h"
#include "Hardware/Configuration/ConfigurationABI.h"

namespace loom::application::detail {

struct ImportedApplicationMapping;

/// Executable artifacts of one verified Mapping, independent of any final
/// winner or activation decision. Deployment owns its binaries and bindings.
struct ApplicationMappingImage final {
  hardware::FinalizedConfigurationABI configurationAbi;
  deployment::FinalizedDeployment deployment;
};

llvm::Expected<ApplicationMappingImage> buildApplicationMappingImage(
    const PreparedApplicationBuild &prepared,
    const ImportedApplicationMapping &imported,
    const llvm::Module &finalLinkedModule, ApplicationDeploymentRequest request,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::application::detail

#endif // LOOM_APPLICATION_APPLICATIONMAPPINGIMAGE_H

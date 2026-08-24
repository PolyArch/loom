#ifndef LOOM_APPLICATION_PRODUCTVISUALIZATION_H
#define LOOM_APPLICATION_PRODUCTVISUALIZATION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::fabric {
class FinalizedFabricRoot;
}

namespace loom::application {

struct ApplicationDeploymentArtifacts;
struct ApplicationMappingExecution;
struct PreparedApplicationBuild;

/// Writes a removable visualization directory from exact artifacts produced
/// by one completed product invocation. No projected byte participates in an
/// Artifact identity or Mapping decision.
llvm::Error exportProductVisualization(
    llvm::StringRef destination, const fabric::FinalizedFabricRoot &system,
    const PreparedApplicationBuild &prepared,
    const ApplicationMappingExecution &mapping,
    const ApplicationDeploymentArtifacts &deployment,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::application

#endif // LOOM_APPLICATION_PRODUCTVISUALIZATION_H

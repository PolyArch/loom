#ifndef LOOM_APPLICATION_ACTIVATIONINPUTS_H
#define LOOM_APPLICATION_ACTIVATIONINPUTS_H

#include "Common/Artifact.h"
#include "Deployment/Deployment.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom::application {

/// Exact Deployment-owned System inputs derived from the canonical source.
struct ApplicationActivationInputs final {
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
};

llvm::Expected<ApplicationActivationInputs>
materializeApplicationActivationInputs(
    const ArtifactRootReference &sourceProgram,
    const ArtifactRootReference &sourceWorkload,
    const ArtifactRootReference &sourceRuntimeInput,
    const deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts,
    std::optional<std::uint64_t> maximumSimulatedTicks = std::nullopt);

} // namespace loom::application

#endif // LOOM_APPLICATION_ACTIVATIONINPUTS_H

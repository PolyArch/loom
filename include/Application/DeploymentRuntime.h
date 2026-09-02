#ifndef LOOM_APPLICATION_DEPLOYMENTRUNTIME_H
#define LOOM_APPLICATION_DEPLOYMENTRUNTIME_H

#include "Application/ResourceTimeExecution.h"
#include "Runtime/Gem5RootEventControl.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::application {

struct ApplicationDeploymentArtifacts;
class ApplicationRuntimeManifest;
class FinalizedApplicationRuntimeManifest;

/// One exact Deployment-owned System invocation pair derived from the
/// Application's canonical source invocation. The source invocation remains
/// authoritative; these roots are removable execution materializations.
struct ApplicationActivationInputs final {
  ArtifactRootReference workload;
  ArtifactRootReference runtimeInput;
};

/// The exact invocation pair for one preverified resource-time endpoint.
struct ApplicationEndpointActivationInputs final {
  pnr::ResourceTimeTransitionEndpointReference endpoint;
  ApplicationActivationInputs inputs;
};

/// Invocation-local runtime state for one exact Application build result.
/// The loaded Deployment remains the execution owner. The optional selector
/// exists only when the compiler produced a finite resource-time graph.
class LoadedApplicationDeployment final {
public:
  LoadedApplicationDeployment(LoadedApplicationDeployment &&) noexcept =
      default;
  LoadedApplicationDeployment &
  operator=(LoadedApplicationDeployment &&) noexcept = default;

  LoadedApplicationDeployment(const LoadedApplicationDeployment &) = delete;
  LoadedApplicationDeployment &
  operator=(const LoadedApplicationDeployment &) = delete;

  runtime::LoadedDeployment &loadedDeployment() { return loaded_; }
  const runtime::LoadedDeployment &loadedDeployment() const { return loaded_; }

  ApplicationResourceTimeExecutionSession *resourceTimeExecution() {
    return resourceTime_ ? &*resourceTime_ : nullptr;
  }
  const ApplicationResourceTimeExecutionSession *resourceTimeExecution() const {
    return resourceTime_ ? &*resourceTime_ : nullptr;
  }
  llvm::ArrayRef<ApplicationEndpointActivationInputs>
  endpointActivationInputs() const {
    return endpointInputs_;
  }

  llvm::Expected<ApplicationResourceTimeExecutionEvent> applyResourceTimeEvent(
      const sim::SystemRootLifecycleObservation &observation);

  /// Synchronous gem5 drive of one root event: applies the event through the
  /// prepared selector and loaded Deployment, then projects the accepted
  /// outcome onto the device decision addressed by `endpoints`. A root start
  /// continues, a typed stay keeps the active endpoint, and a selected child
  /// activates its endpoint ordinal.
  llvm::Expected<runtime::Gem5RootEventDecision> driveGem5RootEvent(
      const sim::SystemRootLifecycleObservation &observation,
      const runtime::Gem5RootEventEndpointTable &endpoints);

  llvm::Expected<FinalizedApplicationResourceTimeExecutionTrace>
  publishResourceTimeExecutionTrace(const ArtifactStore &artifacts,
                                    const BlobStore &blobs) const;

private:
  LoadedApplicationDeployment(
      runtime::LoadedDeployment loaded, ArtifactRootReference runtimeManifest,
      std::optional<ApplicationResourceTimeExecutionSession> resourceTime,
      std::vector<ApplicationEndpointActivationInputs> endpointInputs)
      : loaded_(std::move(loaded)),
        runtimeManifest_(std::move(runtimeManifest)),
        resourceTime_(std::move(resourceTime)),
        endpointInputs_(std::move(endpointInputs)) {}

  runtime::LoadedDeployment loaded_;
  ArtifactRootReference runtimeManifest_;
  std::optional<ApplicationResourceTimeExecutionSession> resourceTime_;
  std::vector<ApplicationEndpointActivationInputs> endpointInputs_;

  friend llvm::Expected<LoadedApplicationDeployment>
  loadApplicationDeployment(const ApplicationDeploymentArtifacts &,
                            runtime::RuntimeProviderSelection,
                            const ArtifactStore &, const BlobStore &);
  friend llvm::Expected<LoadedApplicationDeployment>
  loadApplicationDeployment(const FinalizedApplicationRuntimeManifest &,
                            const deployment::FinalizedDeployment &,
                            runtime::RuntimeProviderSelection,
                            const ArtifactStore &, const BlobStore &);
};

/// Loads one retained Application build result and binds its compiler-owned
/// resource-time graph, when present, to prepared provider activations.
/// Runtime does not derive a graph, create Mapping work, or define package
/// persistence here.
llvm::Expected<LoadedApplicationDeployment>
loadApplicationDeployment(const ApplicationDeploymentArtifacts &application,
                          runtime::RuntimeProviderSelection selection,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs);

/// Package-facing overload. It consumes the same persisted manifest and entry
/// Deployment as the build-result overload and creates no replacement graph.
llvm::Expected<LoadedApplicationDeployment>
loadApplicationDeployment(const FinalizedApplicationRuntimeManifest &manifest,
                          const deployment::FinalizedDeployment &deployment,
                          runtime::RuntimeProviderSelection selection,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs);

/// Mechanically projects one canonical Structured Program invocation onto an
/// exact Deployment and publishes the resulting System invocation roots.
llvm::Expected<ApplicationActivationInputs>
materializeApplicationActivationInputs(
    const ArtifactRootReference &sourceProgram,
    const ArtifactRootReference &sourceWorkload,
    const ArtifactRootReference &sourceRuntimeInput,
    const deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts,
    std::optional<std::uint64_t> maximumSimulatedTicks = std::nullopt);

/// Materializes an independently importable System invocation pair for every
/// endpoint in the manifest's preverified graph. The entry pair must reproduce
/// the manifest's existing activation roots exactly.
llvm::Expected<std::vector<ApplicationEndpointActivationInputs>>
materializeApplicationEndpointActivationInputs(
    const ApplicationRuntimeManifest &manifest, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::application

#endif // LOOM_APPLICATION_DEPLOYMENTRUNTIME_H

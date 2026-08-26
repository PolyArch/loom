#ifndef LOOM_APPLICATION_DEPLOYMENTRUNTIME_H
#define LOOM_APPLICATION_DEPLOYMENTRUNTIME_H

#include "Runtime/DeploymentLoader.h"
#include "Runtime/ResourceTimeTransitionSelection.h"

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

  runtime::ResourceTimeTransitionSelectionSession *resourceTimeSelection() {
    return resourceTime_ ? &*resourceTime_ : nullptr;
  }
  const runtime::ResourceTimeTransitionSelectionSession *
  resourceTimeSelection() const {
    return resourceTime_ ? &*resourceTime_ : nullptr;
  }
  llvm::ArrayRef<ApplicationEndpointActivationInputs>
  endpointActivationInputs() const {
    return endpointInputs_;
  }

private:
  LoadedApplicationDeployment(
      runtime::LoadedDeployment loaded,
      std::optional<runtime::ResourceTimeTransitionSelectionSession>
          resourceTime,
      std::vector<ApplicationEndpointActivationInputs> endpointInputs)
      : loaded_(std::move(loaded)), resourceTime_(std::move(resourceTime)),
        endpointInputs_(std::move(endpointInputs)) {}

  runtime::LoadedDeployment loaded_;
  std::optional<runtime::ResourceTimeTransitionSelectionSession> resourceTime_;
  std::vector<ApplicationEndpointActivationInputs> endpointInputs_;

  friend llvm::Expected<LoadedApplicationDeployment>
  loadApplicationDeployment(const ApplicationDeploymentArtifacts &,
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

/// Mechanically projects one canonical Structured Program invocation onto an
/// exact Deployment and publishes the resulting System invocation roots.
llvm::Expected<ApplicationActivationInputs>
materializeApplicationActivationInputs(
    const ArtifactRootReference &sourceProgram,
    const ArtifactRootReference &sourceWorkload,
    const ArtifactRootReference &sourceRuntimeInput,
    const deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts);

/// Materializes an independently importable System invocation pair for every
/// endpoint in the manifest's preverified graph. The entry pair must reproduce
/// the manifest's existing activation roots exactly.
llvm::Expected<std::vector<ApplicationEndpointActivationInputs>>
materializeApplicationEndpointActivationInputs(
    const ApplicationRuntimeManifest &manifest, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::application

#endif // LOOM_APPLICATION_DEPLOYMENTRUNTIME_H

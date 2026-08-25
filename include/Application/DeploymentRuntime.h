#ifndef LOOM_APPLICATION_DEPLOYMENTRUNTIME_H
#define LOOM_APPLICATION_DEPLOYMENTRUNTIME_H

#include "Runtime/DeploymentLoader.h"
#include "Runtime/ResourceTimeTransitionSelection.h"

#include "llvm/Support/Error.h"

#include <optional>
#include <utility>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::application {

struct ApplicationDeploymentArtifacts;

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

private:
  LoadedApplicationDeployment(
      runtime::LoadedDeployment loaded,
      std::optional<runtime::ResourceTimeTransitionSelectionSession>
          resourceTime)
      : loaded_(std::move(loaded)), resourceTime_(std::move(resourceTime)) {}

  runtime::LoadedDeployment loaded_;
  std::optional<runtime::ResourceTimeTransitionSelectionSession> resourceTime_;

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

} // namespace loom::application

#endif // LOOM_APPLICATION_DEPLOYMENTRUNTIME_H

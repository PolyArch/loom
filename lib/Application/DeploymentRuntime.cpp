#include "Application/DeploymentRuntime.h"

#include "Application/Build.h"

namespace loom::application {

llvm::Expected<LoadedApplicationDeployment>
loadApplicationDeployment(const ApplicationDeploymentArtifacts &application,
                          runtime::RuntimeProviderSelection selection,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  auto loaded = runtime::loadDeployment(application.deployment,
                                        std::move(selection), artifacts, blobs);
  if (!loaded)
    return loaded.takeError();

  std::optional<runtime::ResourceTimeTransitionSelectionSession> resourceTime;
  if (application.resourceTimeTransitionGraph) {
    auto prepared =
        runtime::ResourceTimeTransitionSelectionSession::createPrepared(
            *application.resourceTimeTransitionGraph, *loaded, artifacts,
            blobs);
    if (!prepared)
      return prepared.takeError();
    resourceTime.emplace(std::move(*prepared));
  }

  return LoadedApplicationDeployment(std::move(*loaded),
                                     std::move(resourceTime));
}

} // namespace loom::application

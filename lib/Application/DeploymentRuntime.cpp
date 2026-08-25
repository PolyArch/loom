#include "Application/DeploymentRuntime.h"

#include "Application/Build.h"

namespace loom::application {

llvm::Expected<LoadedApplicationDeployment>
loadApplicationDeployment(const ApplicationDeploymentArtifacts &application,
                          runtime::RuntimeProviderSelection selection,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  auto importedManifest = importApplicationRuntimeManifest(
      application.runtimeManifest.reference(), artifacts, blobs);
  if (!importedManifest)
    return importedManifest.takeError();
  if (importedManifest->manifest().deployment() !=
      application.deployment.reference())
    return llvm::make_error<ApplicationRuntimeManifestError>(
        ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
        "Application runtime manifest names a foreign entry Deployment");

  auto loaded = runtime::loadDeployment(application.deployment,
                                        std::move(selection), artifacts, blobs);
  if (!loaded)
    return loaded.takeError();

  std::optional<runtime::ResourceTimeTransitionSelectionSession> resourceTime;
  if (importedManifest->manifest().transitionGraph()) {
    auto prepared =
        runtime::ResourceTimeTransitionSelectionSession::createPrepared(
            *importedManifest->manifest().transitionGraph(), *loaded, artifacts,
            blobs);
    if (!prepared)
      return prepared.takeError();
    resourceTime.emplace(std::move(*prepared));
  }

  return LoadedApplicationDeployment(std::move(*loaded),
                                     std::move(resourceTime));
}

} // namespace loom::application

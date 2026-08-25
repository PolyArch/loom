#include "Application/DeploymentRuntime.h"

#include "Application/Build.h"

namespace loom::application {

llvm::Expected<LoadedApplicationDeployment>
loadApplicationDeployment(const ApplicationDeploymentArtifacts &application,
                          runtime::RuntimeProviderSelection selection,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  const std::optional<pnr::ResourceTimeTransitionGraph> *transitionGraph =
      &application.resourceTimeTransitionGraph;
  std::optional<FinalizedApplicationRuntimeManifest> importedManifest;
  if (application.runtimeManifest) {
    auto imported = importApplicationRuntimeManifest(
        application.runtimeManifest->reference(), artifacts, blobs);
    if (!imported)
      return imported.takeError();
    if (imported->manifest().deployment() != application.deployment.reference())
      return llvm::make_error<ApplicationRuntimeManifestError>(
          ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
          "Application runtime manifest names a foreign entry Deployment");
    importedManifest.emplace(std::move(*imported));
    transitionGraph = &importedManifest->manifest().transitionGraph();
  }

  auto loaded = runtime::loadDeployment(application.deployment,
                                        std::move(selection), artifacts, blobs);
  if (!loaded)
    return loaded.takeError();

  std::optional<runtime::ResourceTimeTransitionSelectionSession> resourceTime;
  if (*transitionGraph) {
    auto prepared =
        runtime::ResourceTimeTransitionSelectionSession::createPrepared(
            **transitionGraph, *loaded, artifacts, blobs);
    if (!prepared)
      return prepared.takeError();
    resourceTime.emplace(std::move(*prepared));
  }

  return LoadedApplicationDeployment(std::move(*loaded),
                                     std::move(resourceTime));
}

} // namespace loom::application

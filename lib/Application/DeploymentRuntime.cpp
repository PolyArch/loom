#include "Application/DeploymentRuntime.h"

#include "Application/Build.h"
#include "Application/RuntimeManifest.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <iterator>
#include <system_error>
#include <utility>

namespace loom::application {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_deployment_runtime_invalid: " + message);
}

} // namespace

llvm::Expected<std::vector<ApplicationEndpointActivationInputs>>
materializeApplicationEndpointActivationInputs(
    const ApplicationRuntimeManifest &manifest, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  std::optional<std::uint64_t> maximumSimulatedTicks;
  auto activationInputs = sim::importSystemSimulationInputs(
      manifest.activationWorkload(), manifest.activationRuntimeInput(),
      artifacts, blobs);
  if (!activationInputs)
    return activationInputs.takeError();
  if (const auto *runtime = activationInputs->runtimeInput.system())
    maximumSimulatedTicks = runtime->maximumSimulatedTicks;
  std::vector<pnr::ResourceTimeTransitionEndpointReference> endpoints;
  if (manifest.transitionGraph()) {
    if (llvm::Error error = pnr::verifyResourceTimeTransitionGraph(
            *manifest.transitionGraph(), artifacts, blobs))
      return std::move(error);
    endpoints = manifest.transitionGraph()->endpoints;
  } else {
    endpoints.push_back({manifest.selectedMapping(), manifest.deployment()});
  }

  std::vector<ApplicationEndpointActivationInputs> materialized;
  materialized.reserve(endpoints.size());
  for (const pnr::ResourceTimeTransitionEndpointReference &endpoint :
       endpoints) {
    if (!endpoint.deployment)
      return invalid("resource-time endpoint has no Deployment");
    auto endpointDeployment =
        deployment::importDeployment(*endpoint.deployment, artifacts, blobs);
    if (!endpointDeployment)
      return endpointDeployment.takeError();
    auto inputs = materializeApplicationActivationInputs(
        manifest.sourceProgram(), manifest.workload(), manifest.runtimeInput(),
        *endpointDeployment, artifacts, maximumSimulatedTicks);
    if (!inputs)
      return inputs.takeError();
    auto imported = sim::importSystemSimulationInputs(
        inputs->workload, inputs->runtimeInput, artifacts, blobs);
    if (!imported)
      return imported.takeError();
    if (imported->deployment.reference() != *endpoint.deployment)
      return invalid("materialized activation names another Deployment");
    materialized.push_back({endpoint, std::move(*inputs)});
  }

  const auto entry = llvm::find_if(materialized, [&](const auto &candidate) {
    return candidate.endpoint.mapping == manifest.selectedMapping() &&
           candidate.endpoint.deployment == manifest.deployment();
  });
  if (entry == materialized.end() ||
      entry->inputs.workload != manifest.activationWorkload() ||
      entry->inputs.runtimeInput != manifest.activationRuntimeInput())
    return invalid("entry activation does not reproduce the manifest roots");
  return materialized;
}

llvm::Expected<ApplicationResourceTimeExecutionEvent>
LoadedApplicationDeployment::applyResourceTimeEvent(
    const sim::SystemRootLifecycleObservation &observation) {
  if (!resourceTime_)
    return llvm::make_error<ApplicationResourceTimeExecutionError>(
        ApplicationResourceTimeExecutionErrorReason::TransitionGraphUnavailable,
        "Application Deployment has no resource-time transition graph");
  return resourceTime_->apply(observation, loaded_);
}

llvm::Expected<runtime::Gem5RootEventDecision>
LoadedApplicationDeployment::driveGem5RootEvent(
    const sim::SystemRootLifecycleObservation &observation,
    const runtime::Gem5RootEventEndpointTable &endpoints) {
  auto event = applyResourceTimeEvent(observation);
  if (!event)
    return event.takeError();
  if (!event->current.deployment)
    return invalid("resource-time endpoint has no Deployment");
  const auto endpoint =
      llvm::find(endpoints.deployments, *event->current.deployment);
  if (endpoint == endpoints.deployments.end())
    return invalid("resource-time endpoint is outside the gem5 endpoint table");
  const auto ordinal = static_cast<std::uint64_t>(
      std::distance(endpoints.deployments.begin(), endpoint));
  switch (event->outcome) {
  case ApplicationResourceTimeEventOutcome::RootStarted:
    return runtime::Gem5RootEventDecision{
        runtime::Gem5RootEventControlDecision::Continue, ordinal};
  case ApplicationResourceTimeEventOutcome::NoLegalTransition:
    return runtime::Gem5RootEventDecision{
        runtime::Gem5RootEventControlDecision::Stay, ordinal};
  case ApplicationResourceTimeEventOutcome::SelectedChild:
    return runtime::Gem5RootEventDecision{
        runtime::Gem5RootEventControlDecision::ActivateEndpoint, ordinal};
  }
  llvm_unreachable("unknown Application resource-time event outcome");
}

llvm::Expected<FinalizedApplicationResourceTimeExecutionTrace>
LoadedApplicationDeployment::publishResourceTimeExecutionTrace(
    const ArtifactStore &artifacts, const BlobStore &blobs) const {
  if (!resourceTime_)
    return llvm::make_error<ApplicationResourceTimeExecutionError>(
        ApplicationResourceTimeExecutionErrorReason::TransitionGraphUnavailable,
        "Application Deployment has no resource-time transition graph");
  auto manifest =
      importApplicationRuntimeManifest(runtimeManifest_, artifacts, blobs);
  if (!manifest)
    return manifest.takeError();
  return application::publishApplicationResourceTimeExecutionTrace(
      *manifest, *resourceTime_, artifacts, blobs);
}

llvm::Expected<LoadedApplicationDeployment>
loadApplicationDeployment(const ApplicationDeploymentArtifacts &application,
                          runtime::RuntimeProviderSelection selection,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  return loadApplicationDeployment(application.runtimeManifest,
                                   application.deployment, std::move(selection),
                                   artifacts, blobs);
}

llvm::Expected<LoadedApplicationDeployment>
loadApplicationDeployment(const FinalizedApplicationRuntimeManifest &manifest,
                          const deployment::FinalizedDeployment &deployment,
                          runtime::RuntimeProviderSelection selection,
                          const ArtifactStore &artifacts,
                          const BlobStore &blobs) {
  auto importedManifest =
      importApplicationRuntimeManifest(manifest.reference(), artifacts, blobs);
  if (!importedManifest)
    return importedManifest.takeError();
  if (importedManifest->manifest().deployment() != deployment.reference())
    return llvm::make_error<ApplicationRuntimeManifestError>(
        ApplicationRuntimeManifestErrorReason::DeploymentMismatch,
        "Application runtime manifest names a foreign entry Deployment");

  auto endpointInputs = materializeApplicationEndpointActivationInputs(
      importedManifest->manifest(), artifacts, blobs);
  if (!endpointInputs)
    return endpointInputs.takeError();

  auto loaded = runtime::loadDeployment(deployment, std::move(selection),
                                        artifacts, blobs);
  if (!loaded)
    return loaded.takeError();

  std::optional<ApplicationResourceTimeExecutionSession> resourceTime;
  if (importedManifest->manifest().transitionGraph()) {
    auto prepared = ApplicationResourceTimeExecutionSession::createPrepared(
        *importedManifest->manifest().transitionGraph(), *loaded, artifacts,
        blobs);
    if (!prepared)
      return prepared.takeError();
    resourceTime.emplace(std::move(*prepared));
  }

  return LoadedApplicationDeployment(std::move(*loaded), manifest.reference(),
                                     std::move(resourceTime),
                                     std::move(*endpointInputs));
}

} // namespace loom::application

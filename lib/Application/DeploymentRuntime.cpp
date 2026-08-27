#include "Application/DeploymentRuntime.h"

#include "Application/Build.h"
#include "Application/RuntimeManifest.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <system_error>
#include <utility>

namespace loom::application {
namespace {

constexpr std::uint64_t kApplicationHostProgramEntryOrdinal = 0;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_deployment_runtime_invalid: " + message);
}

} // namespace

llvm::Expected<ApplicationActivationInputs>
materializeApplicationActivationInputs(
    const ArtifactRootReference &sourceProgram,
    const ArtifactRootReference &sourceWorkload,
    const ArtifactRootReference &sourceRuntimeInput,
    const deployment::FinalizedDeployment &deployment,
    const ArtifactStore &artifacts,
    std::optional<std::uint64_t> maximumSimulatedTicks) {
  auto source = sim::importStructuredProgramSimulationInputs(
      sourceWorkload, sourceRuntimeInput, artifacts);
  if (!source)
    return source.takeError();
  if (source->structuredProgram.identity() != sourceProgram.artifact)
    return invalid("source invocation names another Structured Program");
  const sim::StructuredProgramSimulationWorkload *sourceWorkloadView =
      source->workload.structuredProgram();
  const sim::StructuredProgramSimulationRuntimeInput *sourceRuntimeView =
      source->runtimeInput.structuredProgram();
  if (!sourceWorkloadView || !sourceRuntimeView)
    return invalid("source invocation is not a Structured Program");
  if (!sourceWorkloadView->observableContract.memories.empty() ||
      !sourceRuntimeView->memoryObjects.empty() ||
      !sourceRuntimeView->pointerBindings.empty())
    return invalid("Deployment activation memory ingress is unsupported");

  sim::SystemSimulationWorkload activation{
      {deployment.reference().artifact, kApplicationHostProgramEntryOrdinal}};
  activation.valueInputPlan.reserve(sourceWorkloadView->argumentPlan.size());
  for (const sim::StructuredProgramArgumentSource &argument :
       sourceWorkloadView->argumentPlan) {
    if (const auto *fixed =
            std::get_if<sim::CanonicalValueSequence>(&argument)) {
      activation.valueInputPlan.push_back(*fixed);
      continue;
    }
    if (std::holds_alternative<sim::StructuredRuntimeValueInput>(argument)) {
      activation.valueInputPlan.push_back(sim::RuntimeValueInput{});
      continue;
    }
    return invalid("Deployment activation pointer ingress is unsupported");
  }
  if (sourceWorkloadView->observableContract.returnValue)
    activation.observableContract.valueResults.push_back(0);
  auto workload =
      sim::finalizeSimulationWorkload(activation, deployment, artifacts);
  if (!workload)
    return workload.takeError();

  sim::SystemSimulationRuntimeInputDraft runtime{workload->identity()};
  runtime.maximumSimulatedTicks = maximumSimulatedTicks;
  runtime.runtimeEntryValues.reserve(sourceRuntimeView->runtimeValues.size());
  for (const sim::StructuredRuntimeValueEntry &value :
       sourceRuntimeView->runtimeValues)
    runtime.runtimeEntryValues.push_back({value.argumentOrdinal, value.value});
  auto runtimeInput = sim::finalizeSimulationRuntimeInput(
      runtime, *workload, deployment, artifacts);
  if (!runtimeInput)
    return runtimeInput.takeError();
  auto workloadReference = sim::publishSimulationWorkload(*workload, artifacts);
  if (!workloadReference)
    return workloadReference.takeError();
  auto runtimeReference =
      sim::publishSimulationRuntimeInput(*runtimeInput, artifacts);
  if (!runtimeReference)
    return runtimeReference.takeError();
  return ApplicationActivationInputs{std::move(*workloadReference),
                                     std::move(*runtimeReference)};
}

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

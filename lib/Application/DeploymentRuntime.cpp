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

  const deployment::DeploymentProgramEntryRef entryRef{
      deployment.reference().artifact, kApplicationHostProgramEntryOrdinal};
  auto programEntry =
      deployment::resolveDeploymentProgramEntry(deployment, entryRef);
  if (!programEntry)
    return programEntry.takeError();

  sim::SystemSimulationWorkload activation{entryRef};
  activation.valueInputPlan.reserve((*programEntry)->valueArgumentTypes.size());
  std::vector<std::optional<std::uint64_t>> valueOrdinals(
      sourceWorkloadView->argumentPlan.size());
  std::vector<std::optional<std::uint64_t>> interfaceOrdinals(
      sourceWorkloadView->argumentPlan.size());
  std::uint64_t valueOrdinal = 0;
  std::uint64_t interfaceOrdinal = 0;
  for (const auto indexed :
       llvm::enumerate(sourceWorkloadView->argumentPlan)) {
    const sim::StructuredProgramArgumentSource &argument = indexed.value();
    if (std::holds_alternative<sim::StructuredRuntimeMemoryInput>(argument)) {
      if (interfaceOrdinal >=
          (*programEntry)->externalInterfaceOrdinals.size())
        return invalid("source pointer arguments exceed Deployment memory "
                       "interfaces");
      const std::uint64_t selected =
          (*programEntry)->externalInterfaceOrdinals[interfaceOrdinal++];
      const deployment::DeploymentExternalInterfaceRef reference{
          deployment.reference().artifact, selected};
      auto interface = deployment::resolveDeploymentExternalInterface(
          deployment, reference);
      if (!interface)
        return interface.takeError();
      if ((*interface)->kind !=
          deployment::HostExternalInterfaceKind::Memory)
        return invalid("source pointer argument selects a non-memory "
                       "Deployment interface");
      interfaceOrdinals[indexed.index()] = selected;
      continue;
    }
    valueOrdinals[indexed.index()] = valueOrdinal++;
    if (const auto *fixed =
            std::get_if<sim::CanonicalValueSequence>(&argument)) {
      activation.valueInputPlan.push_back(*fixed);
      continue;
    }
    if (std::holds_alternative<sim::StructuredRuntimeValueInput>(argument)) {
      activation.valueInputPlan.push_back(sim::RuntimeValueInput{});
      continue;
    }
    return invalid("source invocation has an unknown argument source");
  }
  if (valueOrdinal != (*programEntry)->valueArgumentTypes.size() ||
      interfaceOrdinal != (*programEntry)->externalInterfaceOrdinals.size())
    return invalid("source invocation differs from the Deployment entry ABI");
  if (sourceWorkloadView->observableContract.returnValue)
    activation.observableContract.valueResults.push_back(0);
  for (const sim::StructuredProgramMemoryObservable &observable :
       sourceWorkloadView->observableContract.memories) {
    const auto *argument =
        std::get_if<sim::EntryPointerArgumentTarget>(&observable.target);
    if (!argument || argument->argumentOrdinal >= interfaceOrdinals.size() ||
        !interfaceOrdinals[argument->argumentOrdinal])
      return invalid("source memory observable is not an entry pointer");
    activation.observableContract.memories.push_back(
        {{deployment.reference().artifact,
          *interfaceOrdinals[argument->argumentOrdinal]},
         observable.form});
  }
  auto workload =
      sim::finalizeSimulationWorkload(activation, deployment, artifacts);
  if (!workload)
    return workload.takeError();

  sim::SystemSimulationRuntimeInputDraft runtime{workload->identity()};
  runtime.maximumSimulatedTicks = maximumSimulatedTicks;
  runtime.runtimeEntryValues.reserve(sourceRuntimeView->runtimeValues.size());
  for (const sim::StructuredRuntimeValueEntry &value :
       sourceRuntimeView->runtimeValues) {
    if (value.argumentOrdinal >= valueOrdinals.size() ||
        !valueOrdinals[value.argumentOrdinal])
      return invalid("source runtime value does not select a value argument");
    runtime.runtimeEntryValues.push_back(
        {*valueOrdinals[value.argumentOrdinal], value.value});
  }
  runtime.memoryObjects = sourceRuntimeView->memoryObjects;
  runtime.memoryInterfaceBindings.reserve(
      sourceRuntimeView->pointerBindings.size());
  for (const sim::StructuredPointerBindingEntry &binding :
       sourceRuntimeView->pointerBindings) {
    if (binding.argumentOrdinal >= interfaceOrdinals.size() ||
        !interfaceOrdinals[binding.argumentOrdinal])
      return invalid("source pointer binding does not select a memory "
                     "interface");
    runtime.memoryInterfaceBindings.push_back(
        {{deployment.reference().artifact,
          *interfaceOrdinals[binding.argumentOrdinal]},
         binding.binding.objectOrdinal, binding.binding.byteOffset});
  }
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

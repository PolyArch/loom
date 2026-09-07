#include "Application/ActivationInputs.h"

#include "Common/ArtifactStore.h"
#include "Simulator/SimulationArtifacts.h"
#include "llvm/ADT/STLExtras.h"

#include <system_error>

namespace loom::application {
namespace {
constexpr std::uint64_t kApplicationHostProgramEntryOrdinal = 0;
llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                 "application_activation_inputs_invalid: " + message);
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


} // namespace loom::application

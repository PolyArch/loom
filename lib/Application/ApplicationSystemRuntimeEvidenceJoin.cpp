#include "ApplicationSystemRuntimeEvidence.h"

#include "Deployment/Deployment.h"
#include "Evaluation/ProductionRegistry.h"
#include "Runtime/Gem5SystemExecution.h"

#include <system_error>

namespace loom::application::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_system_evidence_invalid: " + message);
}

} // namespace

llvm::Expected<std::uint64_t> resolveApplicationSystemRuntimeEvidenceJoin(
    llvm::ArrayRef<ArtifactRootReference> evidence,
    const ApplicationSystemRuntimeEvidenceContext &context,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (evidence.size() != 2)
    return invalid("System quality requires exactly one host/candidate pair");
  struct Run final {
    deployment::FinalizedDeployment deployment;
    ImportedApplicationSystemRun imported;
    std::optional<std::uint64_t> maximumTicks;
  };
  std::optional<Run> host;
  std::optional<Run> candidate;
  auto model = evaluation::builtinEvaluationModelDescriptorRef(
      evaluation::BuiltinEvaluationModel::Gem5SystemCgra);
  if (!model)
    return model.takeError();
  for (const auto &root : evidence) {
    auto projection = evaluation::importEvaluationEvidenceDependencyProjection(
        root, artifacts);
    if (!projection)
      return projection.takeError();
    if (projection->outcomeKind != evaluation::EvidenceOutcomeKind::Completed ||
        projection->outputBindings.size() != 1 ||
        projection->outputBindings.front().artifacts.size() != 1)
      return invalid(
          "System quality Evidence has no unique completed execution");
    auto references = evaluation::importEvaluationRequestArtifactReferences(
        projection->request, artifacts);
    if (!references)
      return references.takeError();
    std::optional<ArtifactRootReference> binding;
    std::optional<ArtifactRootReference> workload;
    std::optional<ArtifactRootReference> input;
    for (const auto &reference : *references) {
      std::optional<ArtifactRootReference> *slot = nullptr;
      if (reference.schemaIdentity ==
          runtime::gem5SimulationBindingSchema.identity)
        slot = &binding;
      else if (reference.schemaIdentity ==
               sim::simulationWorkloadSchema.identity)
        slot = &workload;
      else if (reference.schemaIdentity ==
               sim::simulationRuntimeInputSchema.identity)
        slot = &input;
      if (!slot)
        continue;
      if (*slot)
        return invalid("System Request repeats a binding or input root");
      *slot = reference;
    }
    if (!binding || !workload || !input)
      return invalid("System Request omits its binding or input pair");
    auto inputs =
        sim::importSystemSimulationInputs(*workload, *input, artifacts, blobs);
    if (!inputs)
      return inputs.takeError();
    const auto &deployment = inputs->deployment;
    const auto maximumTicks =
        inputs->runtimeInput.system()->maximumSimulatedTicks;
    auto expected = materializeApplicationActivationInputs(
        context.sourceProgram, context.sourceWorkload,
        context.sourceRuntimeInput, deployment, artifacts, maximumTicks);
    if (!expected)
      return expected.takeError();
    if (expected->workload != *workload || expected->runtimeInput != *input)
      return invalid(
          "System quality does not execute the exact source invocation");
    const bool isHost = deployment.deployment().hostOnly() != nullptr;
    if (!isHost &&
        (!deployment.deployment().systemMapping() ||
         *deployment.deployment().systemMapping() != context.mapping))
      return invalid("System quality execution names a foreign Mapping");
    auto importedBinding =
        runtime::importGem5SimulationBinding(*binding, artifacts);
    if (!importedBinding)
      return importedBinding.takeError();
    auto resolution = runtime::buildGem5SystemCaseResolution(
        deployment, *importedBinding, *workload, *input, artifacts, blobs);
    if (!resolution)
      return resolution.takeError();
    auto imported = importApplicationSystemRun(
        {projection->outputBindings.front().artifacts.front(), root,
         std::nullopt},
        deployment.reference(), *workload, *input, *resolution, artifacts,
        blobs);
    if (!imported)
      return imported.takeError();
    if (imported->request.modelBinding().descriptorRef() != *model)
      return invalid("System quality requires the native CGRA model");
    auto &destination = isHost ? host : candidate;
    if (destination)
      return invalid("System quality repeats a pair member");
    destination.emplace(
        Run{std::move(inputs->deployment), std::move(*imported), maximumTicks});
  }
  if (!host || !candidate)
    return invalid("System quality omits a host or candidate member");
  auto candidateFabric = deployment::deploymentFabric(
      candidate->deployment.deployment(), artifacts);
  if (!candidateFabric)
    return candidateFabric.takeError();
  if (host->maximumTicks != candidate->maximumTicks ||
      host->deployment.deployment().hostOnly()->fabric != *candidateFabric ||
      host->deployment.deployment().hostProgram().compilerTargetBinding() !=
          candidate->deployment.deployment()
              .hostProgram()
              .compilerTargetBinding())
    return invalid("System quality changes the host target or work limit");
  auto compared =
      compareApplicationSystemRuns(host->imported, candidate->imported);
  if (!compared)
    return compared.takeError();
  if (!*compared || !candidate->imported.measurement.computationInterval)
    return invalid("System quality has no passing measured computation pair");
  return candidate->imported.measurement.computationInterval->elapsedTicks();
}

} // namespace loom::application::detail

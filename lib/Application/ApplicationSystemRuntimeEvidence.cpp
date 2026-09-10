#include "ApplicationSystemRuntimeEvidence.h"

#include "Evaluation/ProductionRegistry.h"
#include "Runtime/Gem5BuiltinModels.h"
#include "Simulator/SpatialObservationComparison.h"

#include <limits>
#include <system_error>

namespace loom::application::detail {
namespace {
llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application System runtime: " + message);
}
} // namespace

llvm::Expected<ImportedApplicationSystemRun>
importApplicationSystemRun(const ApplicationSystemRunEvidence &roots,
                           const ArtifactRootReference &expectedDeployment,
                           const ArtifactRootReference &expectedWorkload,
                           const ArtifactRootReference &expectedInput,
                           const evaluation::CaseArtifactResolution &resolution,
                           const ArtifactStore &artifacts,
                           const BlobStore &blobs) {
  auto execution = sim::importSimulationExecution(roots.execution, resolution,
                                                  artifacts, blobs);
  if (!execution)
    return execution.takeError();
  const auto *system = execution->system();
  if (!system ||
      !std::holds_alternative<sim::RetiredExecution>(execution->terminal()))
    return invalid("pair member did not retire a complete System invocation");
  auto request = evaluation::importEvaluationRequest(
      execution->request(), resolution, artifacts, blobs);
  if (!request)
    return request.takeError();
  if (!request->workload() || *request->workload() != expectedWorkload ||
      !request->runtimeInput() || *request->runtimeInput() != expectedInput)
    return invalid("pair member does not execute its exact input pair");
  const ArtifactRootReference *binding = nullptr;
  const ArtifactRootReference *deployment = nullptr;
  for (const auto &role : request->subjectBindings().roleBindings())
    for (const auto &subject : role.subjects) {
      if (subject.schemaIdentity ==
          runtime::gem5SimulationBindingSchema.identity)
        binding = &subject;
      if (subject.schemaIdentity == deployment::deploymentSchema.identity)
        deployment = &subject;
    }
  if (!binding || !deployment || *deployment != expectedDeployment)
    return invalid(
        "pair member does not bind its exact Deployment and gem5 machine");
  auto utilization = sim::projectSystemMemoryUtilization(*execution, resolution,
                                                         artifacts, blobs);
  if (!utilization)
    return utilization.takeError();
  if (!*utilization)
    return invalid(
        "pair member omitted native shared-memory service occupancy");
  const auto elapsed =
      system->progressObservations.programExitVisible->gem5Tick -
      system->progressObservations.programEntryAccepted.gem5Tick;
  auto evidence = evaluation::importEvaluationEvidence(
      roots.evidence, resolution, artifacts, blobs);
  if (!evidence)
    return evidence.takeError();
  const auto *completed =
      std::get_if<evaluation::CompletedEvidence>(&evidence->outcome());
  if (!completed || evidence->requestRef() != execution->request() ||
      evidence->outputBindings().size() != 1 ||
      llvm::ArrayRef<ArtifactRootReference>(
          evidence->outputBindings().front().artifacts) !=
          llvm::ArrayRef<ArtifactRootReference>{roots.execution})
    return invalid(
        "execution Evidence is not joined to this exact pair member");
  if (request->metricRequests().size() != 1 ||
      completed->metricResults.size() != 1 ||
      request->metricRequests().front().query().metric !=
          evaluation::MetricKind::Runtime ||
      completed->metricResults.front().uncertainty !=
          evaluation::UncertaintyKind::ExactWithinModel)
    return invalid(
        "pair member omitted its exact complete-System Runtime observation");
  if (elapsed >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return invalid("pair runtime exceeds the metric decimal domain");
  auto expectedRuntime = evaluation::DecimalValue::get(
      static_cast<std::int64_t>(elapsed), runtime::gem5TickSecondsExponent);
  if (!expectedRuntime)
    return expectedRuntime.takeError();
  const auto *point = std::get_if<evaluation::PointObservation>(
      &completed->metricResults.front().observation);
  if (!point || point->value != evaluation::MetricValue{*expectedRuntime})
    return invalid(
        "Runtime observation disagrees with the full program tick window");

  const ArtifactRootReference exactBinding = *binding;
  ApplicationSystemRunMeasurement measurement{roots,
                                              evidence->requestRef(),
                                              std::nullopt,
                                              elapsed,
                                              *system->memoryActivity,
                                              **utilization,
                                              system->computationInterval};
  return ImportedApplicationSystemRun{std::move(*execution),
                                      std::move(*request), exactBinding,
                                      std::move(measurement)};
}

llvm::Expected<bool>
compareApplicationSystemRuns(const ImportedApplicationSystemRun &host,
                             const ImportedApplicationSystemRun &candidate) {
  if (host.gem5Binding != candidate.gem5Binding)
    return invalid("pair members use different exact gem5 machines");
  const auto &left = host.request;
  const auto &right = candidate.request;
  const auto &leftModel = left.modelBinding();
  const auto &rightModel = right.modelBinding();
  if (leftModel.descriptorRef() != rightModel.descriptorRef() ||
      leftModel.inputBindings() != rightModel.inputBindings() ||
      leftModel.resolvedModelConfig().digest() !=
          rightModel.resolvedModelConfig().digest() ||
      left.baseConditions() != right.baseConditions() ||
      left.metricRequests() != right.metricRequests() ||
      left.replicateIndex() != right.replicateIndex())
    return invalid(
        "pair members use different complete-System observation conditions");
  if (!host.execution.system()->progressObservations.rootLifecycle.empty())
    return invalid("host-only pair member launched accelerator work");
  if (host.measurement.computationInterval.has_value() !=
      candidate.measurement.computationInterval.has_value())
    return invalid("pair members disagree on the source computation boundary");
  return sim::haveExactlyEqualSystemFunctionalObservations(
      host.execution.system()->functionalObservations,
      candidate.execution.system()->functionalObservations);
}

} // namespace loom::application::detail

#include "Application/SystemQor.h"

#include "Application/ProductOracleEvaluation.h"
#include "Common/ArtifactText.h"
#include "Evaluation/ProductionRegistry.h"
#include "Runtime/Gem5BuiltinModels.h"
#include "Simulator/SystemActivity.h"
#include "Simulator/SpatialObservationComparison.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/JSON.h"

#include <limits>
#include <system_error>

namespace loom::application {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::make_error_code(std::errc::invalid_argument),
                                 "application System QoR: " + message);
}

struct ImportedRun final {
  sim::CanonicalSimulationExecution execution;
  evaluation::EvaluationRequest request;
  ArtifactRootReference gem5Binding;
  ApplicationSystemRunMeasurement measurement;
};

llvm::Expected<ImportedRun> importRun(
    const FinalizedApplicationRuntimeManifest &manifest,
    const ApplicationSystemRunEvidence &roots,
    const ArtifactRootReference &expectedDeployment,
    const ArtifactRootReference &expectedWorkload,
    const ArtifactRootReference &expectedInput,
    const evaluation::CaseArtifactResolution &resolution,
    const ResolvedConfig &config, const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto execution = sim::importSimulationExecution(roots.execution, resolution, artifacts, blobs);
  if (!execution)
    return execution.takeError();
  const auto *system = execution->system();
  if (!system || !std::holds_alternative<sim::RetiredExecution>(execution->terminal()))
    return invalid("pair member did not retire a complete System invocation");
  auto request = evaluation::importEvaluationRequest(execution->request(), resolution, artifacts, blobs);
  if (!request)
    return request.takeError();
  if (!request->workload() || *request->workload() != expectedWorkload ||
      !request->runtimeInput() || *request->runtimeInput() != expectedInput)
    return invalid("pair member does not execute its exact manifest input pair");
  const ArtifactRootReference *binding = nullptr;
  const ArtifactRootReference *deployment = nullptr;
  for (const auto &role : request->subjectBindings().roleBindings())
    for (const auto &subject : role.subjects) {
      if (subject.schemaIdentity == runtime::gem5SimulationBindingSchema.identity)
        binding = &subject;
      if (subject.schemaIdentity == deployment::deploymentSchema.identity)
        deployment = &subject;
    }
  if (!binding || !deployment || *deployment != expectedDeployment)
    return invalid("pair member does not bind its exact Deployment and gem5 machine");
  auto utilization = sim::projectSystemMemoryUtilization(*execution, resolution, artifacts, blobs);
  if (!utilization)
    return utilization.takeError();
  if (!*utilization)
    return invalid("pair member omitted native shared-memory service occupancy");
  const auto elapsed = system->progressObservations.programExitVisible->gem5Tick -
                       system->progressObservations.programEntryAccepted.gem5Tick;
  auto evidence = evaluation::importEvaluationEvidence(roots.evidence, resolution, artifacts, blobs);
  if (!evidence)
    return evidence.takeError();
  const auto *completed = std::get_if<evaluation::CompletedEvidence>(&evidence->outcome());
  if (!completed || evidence->requestRef() != execution->request() ||
      evidence->outputBindings().size() != 1 ||
      llvm::ArrayRef<ArtifactRootReference>(evidence->outputBindings().front().artifacts) != llvm::ArrayRef<ArtifactRootReference>{roots.execution})
    return invalid("execution Evidence is not joined to this exact pair member");
  if (request->metricRequests().size() != 1 || completed->metricResults.size() != 1 ||
      request->metricRequests().front().query().metric != evaluation::MetricKind::Runtime ||
      completed->metricResults.front().uncertainty != evaluation::UncertaintyKind::ExactWithinModel)
    return invalid("pair member omitted its exact complete-System Runtime observation");
  if (elapsed > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    return invalid("pair runtime exceeds the metric decimal domain");
  auto expectedRuntime = evaluation::DecimalValue::get(static_cast<std::int64_t>(elapsed),
                                                       runtime::gem5TickSecondsExponent);
  if (!expectedRuntime)
    return expectedRuntime.takeError();
  const auto *point = std::get_if<evaluation::PointObservation>(&completed->metricResults.front().observation);
  if (!point || point->value != evaluation::MetricValue{*expectedRuntime})
    return invalid("Runtime observation disagrees with the full program tick window");

  std::optional<ArtifactRootReference> oracleRequest;
  if (manifest.manifest().productOracle()) {
    if (!roots.productOracleEvidence)
      return invalid("product pair member has no independent source oracle Evidence");
    auto prepared = prepareProductOracleEvaluation(manifest, roots.execution, resolution,
                                                    config, artifacts, blobs);
    if (!prepared)
      return prepared.takeError();
    auto oracle = evaluation::importEvaluationEvidence(*roots.productOracleEvidence,
                                                        prepared->resolution, artifacts, blobs);
    if (!oracle)
      return oracle.takeError();
    const auto expected = evaluation::evaluationRequestReference(prepared->request);
    const auto *result = std::get_if<evaluation::CompletedEvidence>(&oracle->outcome());
    if (oracle->requestRef() != expected || !result || result->findingResults.size() != 1 ||
        !std::holds_alternative<evaluation::AbsentFinding>(result->findingResults.front().result))
      return invalid("product pair member lacks the exact passing source oracle");
    oracleRequest = expected;
  } else if (roots.productOracleEvidence) {
    return invalid("non-product pair member carries an unowned product oracle");
  }
  const ArtifactRootReference exactBinding = *binding;
  ApplicationSystemRunMeasurement measurement{
      roots, evidence->requestRef(), std::move(oracleRequest), elapsed,
      *system->memoryActivity, **utilization};
  return ImportedRun{std::move(*execution), std::move(*request), exactBinding,
                     std::move(measurement)};
}

void writeRoot(llvm::json::OStream &json, llvm::StringRef name,
               const ArtifactRootReference &reference) {
  json.attributeObject(name, [&] { writeArtifactRootReferenceJsonFields(json, reference); });
}

void writeRatio(llvm::json::OStream &json, llvm::StringRef name, evaluation::ExactRatio value) {
  json.attributeObject(name, [&] {
    json.attribute("numerator", value.numerator());
    json.attribute("denominator", value.denominator());
  });
}

void writeRun(llvm::json::OStream &json, const ApplicationSystemRunMeasurement &run) {
  writeRoot(json, "request", run.request);
  writeRoot(json, "evidence", run.roots.evidence);
  writeRoot(json, "execution", run.roots.execution);
  if (run.productOracleRequest) {
    writeRoot(json, "product_oracle_request", *run.productOracleRequest);
    writeRoot(json, "product_oracle_evidence", *run.roots.productOracleEvidence);
  }
  json.attribute("elapsed_ticks", run.elapsedTicks);
  json.attributeObject("shared_memory", [&] {
    json.attribute("occupied_ticks", run.memoryActivity.occupiedTicks);
    writeRatio(json, "utilization", run.memoryUtilization);
  });
}

} // namespace

ApplicationSystemQorStatus ApplicationSystemQor::status() const {
  constexpr unsigned width = 128;
  const auto used = llvm::APInt(width, candidate_.memoryActivity.occupiedTicks) *
                    applicationMinimumMemoryUtilizationDenominator;
  const auto target = llvm::APInt(width, candidate_.elapsedTicks) *
                      applicationMinimumMemoryUtilizationNumerator;
  return candidate_.elapsedTicks < host_.elapsedTicks && used.ugt(target)
             ? ApplicationSystemQorStatus::Qualified : ApplicationSystemQorStatus::NotQualified;
}

llvm::Expected<ApplicationSystemQor> qualifyApplicationSystemQor(
    const FinalizedApplicationRuntimeManifest &manifest,
    const ApplicationSystemRunEvidence &hostOnly,
    const evaluation::CaseArtifactResolution &hostResolution,
    const ApplicationSystemRunEvidence &candidate,
    const evaluation::CaseArtifactResolution &candidateResolution,
    const ResolvedConfig &config, const ArtifactStore &artifacts, const BlobStore &blobs) {
  const auto &runtime = manifest.manifest();
  const auto &baseline = runtime.hostOnlyBaseline();
  auto host = importRun(manifest, hostOnly, baseline.deployment, baseline.inputs.workload,
                        baseline.inputs.runtimeInput, hostResolution, config, artifacts, blobs);
  if (!host)
    return host.takeError();
  auto accelerated = importRun(manifest, candidate, runtime.deployment(), runtime.activationWorkload(),
                               runtime.activationRuntimeInput(), candidateResolution,
                               config, artifacts, blobs);
  if (!accelerated)
    return accelerated.takeError();
  if (host->gem5Binding != accelerated->gem5Binding)
    return invalid("pair members use different exact gem5 machines");
  const auto &left = host->request;
  const auto &right = accelerated->request;
  const auto &leftModel = left.modelBinding();
  const auto &rightModel = right.modelBinding();
  if (leftModel.descriptorRef() != rightModel.descriptorRef() ||
      leftModel.inputBindings() != rightModel.inputBindings() ||
      leftModel.resolvedModelConfig().digest() != rightModel.resolvedModelConfig().digest() ||
      left.baseConditions() != right.baseConditions() ||
      left.metricRequests() != right.metricRequests() ||
      left.replicateIndex() != right.replicateIndex())
    return invalid("pair members use different complete-System observation conditions");
  if (!host->execution.system()->progressObservations.rootLifecycle.empty())
    return invalid("host-only pair member launched accelerator work");
  if (!sim::haveExactlyEqualSystemFunctionalObservations(
          host->execution.system()->functionalObservations,
          accelerated->execution.system()->functionalObservations))
    return invalid("complete System pair functional observations differ");
  auto speedup = evaluation::ExactRatio::get(host->measurement.elapsedTicks,
                                            accelerated->measurement.elapsedTicks);
  if (!speedup)
    return speedup.takeError();
  return ApplicationSystemQor(manifest.reference(), host->gem5Binding,
                              std::move(host->measurement), std::move(accelerated->measurement),
                              *speedup);
}

void writeApplicationSystemQorJsonFields(llvm::json::OStream &json,
                                       const ApplicationSystemQor &qor) {
  json.attribute("schema", applicationSystemQorProjectionSchema);
  json.attribute("version", applicationSystemQorProjectionVersion);
  writeRoot(json, "application_runtime_manifest", qor.runtimeManifest());
  writeRoot(json, "gem5_binding", qor.gem5Binding());
  json.attributeObject("host_only", [&] { writeRun(json, qor.hostOnly()); });
  json.attributeObject("candidate", [&] { writeRun(json, qor.candidate()); });
  writeRatio(json, "speedup", qor.speedup());
  json.attribute("status", qor.status() == ApplicationSystemQorStatus::Qualified
                               ? "qualified" : "not_qualified");
  json.attributeObject("target", [&] {
    json.attribute("strict_speedup", true);
    json.attributeObject("minimum_memory_utilization_exclusive", [&] {
      json.attribute("numerator", applicationMinimumMemoryUtilizationNumerator);
      json.attribute("denominator", applicationMinimumMemoryUtilizationDenominator);
    });
  });
}

} // namespace loom::application
